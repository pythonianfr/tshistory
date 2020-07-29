import zlib

import pandas as pd

from sqlhelp import select

from tshistory.util import (
    binary_pack,
    binary_unpack,
    patch,
    numpy_serialize,
    numpy_deserialize,
)


class Snapshot:
    """Here's what's happening when we create a series with 3 insertions
    in a scenario representative of real world situations.

    We will insert initially:

    2019-1-1   1.0
    2019-1-2   2.0
    2019-1-3   3.0

    Then we do a pure append:

    2019-1-4   4.0
    2019-1-5   5.0
    2019-1-6   6.0

    Finally we insert with an overlap over the previous insert:

    2019-1-5   9.0   # previous 5.0 becomes 9.0
    2019-1-7   7.0
    2019-1-8   8.0

    Now let's look at the logical organisation: we have two tables
    `Revision` (which tracks all successive versions of a series) and
    `Snapshot` (which actually stores the series data using a tree
    structure).

    Series values   | Insertion table | Snapshot/storage table
                    +-----------------+------------------------
                    | id | snapshot   | id | parent | chunk
                    +----+------------+----+--------+----------
    1,2,3           | 1  | 1          | 1  | null   | 1,2,3
    1,2,3,4,5,6     | 2  | 2          | 2  | 1      | 4,5,6
    1,2,3,4,9,6,7,8 | 3  | 3          | 3  | 1      | 4,9,6,7,8

    Each version creates a chunk with the new data points, plus data
    from any existing chunk that contains points that are modified
    by the new version.

    We explain in practice what happens with the three successive
    insertions.

    So the first insertion trivially creates an initial chunk with the
    given data points. This initial series version only contains the
    points in this chunk.

    The second insertion creates a new chunk with the new data
    points. But also it is linked to the first chunk (its `parent`).
    At version 2 the series data is distributed amongst the two
    chunks.

    The third insertion creates a third chunk. The chunk contains the
    new data points, and because the inserted series overlaps with the
    existing values (at timestamp 2019-1-5) the new chunk contains a
    modified copy of the second chunk. It is also linked to the
    *first* chunk (its `parent`).

    The collection of chunks of the snapshot table form a tree through
    the parent relationship.

    To rebuild a series at a given version, we must concatenate the
    data points of its chunk and the successive parents.

    For instance, to get the series at version 2 we do as follow:

    * get the snapshot id associated to the the revision id 2 (will be
      also 2)

    * collect the chunk associated with id 2

    * since we have a `parent` at 1 also collect the chunk associated
      with id 1

    * since we don't have a parent at id 1 we stop, and return the
      concatenated chunks

    """
    __slots__ = ('cn', 'name', 'tsh', 'tablename')
    _max_bucket_size = 250

    def __init__(self, cn, tsh, name):
        self.cn = cn
        self.tsh = tsh
        self.name = name
        self.tablename = self.tsh._series_to_tablename(cn, name)

    # optimized/asymmetric de/serialisation

    @property
    def isstr(self):
        return self.tsh.metadata(self.cn, self.name)['value_type'] == 'object'

    def _serialize(self, ts):
        if ts is None:
            return None

        index, values = numpy_serialize(ts, self.isstr)
        return zlib.compress(binary_pack(index, values))

    def _ensure_tz_consistency(self, ts):
        """Return timeserie with tz aware index or not depending on metadata
        tzaware.
        """
        assert ts.name is not None
        metadata = self.tsh.metadata(self.cn, ts.name)
        if metadata and metadata.get('tzaware', False):
            return ts.tz_localize('UTC')
        return ts

    def _chunks_to_ts(self, chunks):
        chunks = (
            binary_unpack(zlib.decompress(chunk))
            for chunk in chunks
        )
        indexchunks, valueschunks = list(zip(*chunks))

        meta = self.tsh.metadata(self.cn, self.name)
        bseparator = b'\0' if meta['value_type'] == 'object' else b''

        index, values = numpy_deserialize(
            b''.join(indexchunks),
            bseparator.join(valueschunks),
            meta
        )

        assert len(values) == len(index)
        serie = pd.Series(values, index=index)
        assert serie.index.is_monotonic_increasing
        serie.name = self.name

        return self._ensure_tz_consistency(serie)

    # /serialisation

    def buckets(self, ts):
        if len(ts) < self._max_bucket_size:
            return [ts]

        buckets = []
        for start in range(0, len(ts),
                           self._max_bucket_size):
            buckets.append(ts[start:start + self._max_bucket_size])
        return buckets

    def insert_buckets(self, parent, ts):
        for bucket in self.buckets(ts):
            start = bucket.index.min()
            end = bucket.index.max()
            sql = (f'insert into "{self.tsh.namespace}.snapshot"."{self.tablename}" '
                   '(cstart, cend, parent, chunk) '
                   'values (%s, %s, %s, %s)'
                   'returning id')
            parent = self.cn.execute(
                sql,
                start,
                end,
                parent,
                self._serialize(bucket)
            ).scalar()

        return parent

    def create(self, initial_ts):
        return self.insert_buckets(None, initial_ts)

    def update(self, series_diff):
        # get last chunkhead for cset
        tablename = self.tsh._series_to_tablename(self.cn, self.name)
        headsql = ('select snapshot '
                   f'from "{self.tsh.namespace}.revision"."{tablename}" '
                   'order by id desc limit 1')
        head = self.cn.execute(headsql).scalar()

        # get raw chunks matching the limits
        diffstart = series_diff.index.min()
        rawchunks = self.rawchunks(head, diffstart)
        cid, parent, _ = rawchunks[0]
        oldsnapshot = self._chunks_to_ts(row[2] for row in rawchunks)

        if diffstart > oldsnapshot.index.max():
            # append: let't not rewrite anything
            newsnapshot = series_diff
            parent = cid
        else:
            # we got a point override, need to patch
            newsnapshot = patch(oldsnapshot, series_diff)

        return self.insert_buckets(parent, newsnapshot)

    rawsql = """
        with recursive allchunks as (
            select chunks.id as cid,
                   chunks.parent as parent,
                   chunks.chunk as chunk
            from "{namespace}"."{table}" as chunks
            where chunks.id in ({heads})
          union
            select chunks.id as cid,
                   chunks.parent as parent,
                   chunks.chunk as chunk
            from "{namespace}"."{table}" as chunks
            join allchunks on chunks.id = allchunks.parent
            {where}
        )
        select cid, parent, chunk from allchunks
    """

    def rawchunks(self, head, from_value_date=None):
        where = ''
        if from_value_date:
            where = 'where chunks.cend >= %(start)s '

        sql = self.rawsql.format(
            namespace=f'{self.tsh.namespace}.snapshot',
            table=self.tablename,
            heads=','.join([str(head)]),
            where=where
        )
        res = self.cn.execute(sql, start=from_value_date)
        chunks = list(res.fetchall())
        chunks.reverse()
        return chunks

    def chunk(self, head, from_value_date=None, to_value_date=None):
        snapdata = self._chunks_to_ts(
            raw[2] for raw in self.rawchunks(head, from_value_date)
        )
        try:
            return snapdata.loc[from_value_date:to_value_date]
        except TypeError as err:
            raise ValueError(
                f'from/to: {from_value_date}/{to_value_date}, '
                f'index type: {snapdata.index.dtype} '
                f'(from "{err}")'
            )

    def last(self, from_value_date=None, to_value_date=None):
        return self.find(from_value_date=from_value_date,
                         to_value_date=to_value_date)[1]

    def last_id(self, from_value_date=None, to_value_date=None):
        return self.find(from_value_date=from_value_date,
                         to_value_date=to_value_date)[0]

    def cset_heads_query(self, csetfilter=(), order='desc'):
        tablename = self.tsh._series_to_tablename(self.cn, self.name)
        q = select(
            'id', 'snapshot'
        ).table(
            f'"{self.tsh.namespace}.revision"."{tablename}"'
        )

        if csetfilter:
            for filtercb in csetfilter:
                filtercb(q)

        q.order('id', order)
        return q

    def find(self, csetfilter=(),
             from_value_date=None, to_value_date=None):

        q = self.cset_heads_query(csetfilter)
        q.limit(1)

        try:
            csid, cid = q.do(self.cn).fetchone()
        except TypeError:
            # this happens *only* because of the from/to restriction
            return None, None

        chunk = self.chunk(cid, from_value_date, to_value_date)
        return csid, chunk

    def allchunks(self, heads, from_value_date=None):
        where = ''
        if from_value_date:
            where = 'where chunks.cend >= %(start)s '

        sql = self.rawsql.format(
            namespace=f'{self.tsh.namespace}.snapshot',
            table=self.tablename,
            heads=','.join(str(head) for head in heads),
            where=where
        )
        res = self.cn.execute(sql, start=from_value_date)
        chunks = {cid: (parent, rawchunk)
                  for cid, parent, rawchunk in res.fetchall()}
        return chunks

    def findall(self, revs, from_value_date, to_value_date):
        # there might be a None in first position because
        # of the diff mode
        csets = [rev for rev, _ in revs if rev is not None]
        # csid -> heads

        q = self.cset_heads_query(
            (
                lambda q: q.where('id >= %(mincset)s', mincset=min(csets)),
                lambda q: q.where('id <= %(maxcset)s', maxcset=max(csets))
            ),
            order='asc'
        )

        cset_snap_map = {
            row.id: row.snapshot
            for row in q.do(self.cn).fetchall()
        }
        rawchunks = self.allchunks(
            sorted(cset_snap_map.values()),
            from_value_date
        )

        series = []
        for cset, idate in revs:
            if cset is None:
                # first occurrence, because of diff mode
                assert idate is None
                series.append((None, None))
                continue
            chunks = []
            head = cset_snap_map[cset]
            while True:
                parent, chunk = rawchunks.get(head, (None, None))
                if chunk is None:
                    break
                chunks.append(chunk)
                head = parent
            series.append(
                (idate, self._chunks_to_ts(reversed(chunks)).loc[
                    from_value_date:to_value_date
                ])
            )
        return series

    def garbage(self):
        """ inefficient but simple garbage list builder
        garbage chunks are created on strip operations
        """
        tablename = self.tsh._series_to_tablename(self.cn, self.name)
        reachablesql = f"""
        with recursive heads as (
            select snapshot from "{self.tsh.namespace}.revision"."{tablename}"
          ),
          allchunks as (
            select chunks.id as cid,
                   chunks.parent as parent
            from "{self.tsh.namespace}.snapshot"."{self.tablename}" as chunks
            where chunks.id in (select * from heads)
          union
            select chunks.id as cid,
                   chunks.parent as parent
            from "{self.tsh.namespace}.snapshot"."{self.tablename}" as chunks
            join allchunks on chunks.id = allchunks.parent
        )
        select cid from allchunks
        """

        reachable_chunks = {
            rev for rev, in self.cn.execute(reachablesql)
        }
        allsql = f'select id from "{self.tsh.namespace}.snapshot"."{self.tablename}" '
        allchuks = {
            rev for rev, in self.cn.execute(allsql).fetchall()
        }

        return allchuks - reachable_chunks

    def reclaim(self):
        todelete = ','.join(str(id) for id in self.garbage())
        sql = (f'delete from "{self.tsh.namespace}.snapshot"."{self.tablename}" '
               f'where id in ({todelete})')
        self.cn.execute(sql)
