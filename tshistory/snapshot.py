import zlib
from array import array
import struct
from pathlib import Path

import pandas as pd
import numpy as np

from sqlhelp import sqlfile, select

from tshistory.util import SeriesServices


SCHEMA = Path(__file__).parent / 'snapshot.sql'


class Snapshot(SeriesServices):
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
    `Revision` (which track all successive versions of a series) and
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
    __slots__ = ('cn', 'name', 'tsh')
    _max_bucket_size = 250

    def __init__(self, cn, tsh, seriename):
        self.cn = cn
        self.tsh = tsh
        self.seriename = seriename
        self.name = self.tsh._serie_to_tablename(cn, seriename)

    @property
    def table(self):
        return sqlfile(SCHEMA,
                       namespace=self.tsh.namespace,
                       tablename=self.name)

    # optimized/asymmetric de/serialisation

    @property
    def isstr(self):
        return self.tsh.metadata(self.cn, self.seriename)['value_type'] == 'object'

    def _serialize(self, ts):
        if ts is None:
            return None
        # use `view` as a workarround for "cannot include dtype 'M' in a buffer"
        indexes = np.ascontiguousarray(ts.index.values).view(np.uint8).data.tobytes()
        # will be a `long` or 4 octets structure
        indexes_size = struct.pack('!L', len(indexes))

        if self.isstr:
            # string separatd by 0 and nones/nans represented as 3 (ETX)
            END, ETX = b'\0'.decode(), b'\3'.decode()
            # first, safety belt
            for s in ts.values:
                if not pd.isnull(s):
                    assert END not in s and ETX not in s
            values = b'\0'.join(b'\3' if pd.isnull(v) else v.encode('utf-8')
                                for v in ts.values)
        else:
            values = ts.values.data.tobytes()
        return zlib.compress(indexes_size + indexes + values)

    def _ensure_tz_consistency(self, ts):
        """Return timeserie with tz aware index or not depending on metadata
        tzaware.
        """
        assert ts.name is not None
        metadata = self.tsh.metadata(self.cn, ts.name)
        if metadata and metadata.get('tzaware', False):
            return ts.tz_localize('UTC')
        return ts

    def _decodechunk(self, bytestring):
        bytestring = zlib.decompress(bytestring)
        [indexes_size] = struct.unpack('!L', bytestring[:4])
        values_offset = indexes_size + 4
        return bytestring[4:values_offset], bytestring[values_offset:]

    def _chunks_to_ts(self, chunks):
        chunks = (self._decodechunk(chunk) for chunk in chunks)
        indexchunks, valueschunks = list(zip(*chunks))
        metadata = self.tsh.metadata(self.cn, self.seriename)

        # array is a workaround for an obscure bug with pandas.isin
        index = np.frombuffer(
            array('d', b''.join(indexchunks)),
            metadata['index_dtype']
        )

        if self.isstr:
            values = [v.decode('utf-8') if v != b'\3' else None
                      for bvalues in valueschunks
                      for v in bvalues.split(b'\0')]
        else:
            values = np.frombuffer(
                b''.join(valueschunks),
                metadata['value_dtype']
            )

        assert len(values) == len(index)
        serie = pd.Series(values, index=index)
        assert serie.index.is_monotonic_increasing
        serie.name = self.seriename

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
            sql = (f'insert into "{self.tsh.namespace}.snapshot"."{self.name}" '
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
        self.cn.execute(self.table)
        return self.insert_buckets(None, initial_ts)

    def update(self, diff):
        # get last chunkhead for cset
        tablename = self.tsh._serie_to_tablename(self.cn, self.seriename)
        headsql = ('select snapshot '
                   f'from "{self.tsh.namespace}.timeserie"."{tablename}" '
                   'order by id desc limit 1')
        head = self.cn.execute(headsql).scalar()

        # get raw chunks matching the limits
        diffstart = diff.index.min()
        rawchunks = self.rawchunks(head, diffstart)
        cid, parent, _ = rawchunks[0]
        oldsnapshot = self._chunks_to_ts(row[2] for row in rawchunks)

        if diffstart > oldsnapshot.index.max():
            # append: let't not rewrite anything
            newsnapshot = diff
            parent = cid
        else:
            # we got a point override, need to patch
            newsnapshot = self.patch(oldsnapshot, diff)

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
            table=self.name,
            heads=','.join([str(head)]),
            where=where
        )
        res = self.cn.execute(sql, start=from_value_date)
        chunks = [(cid, parent, rawchunk)
                  for cid, parent, rawchunk in res.fetchall()]
        chunks.reverse()
        return chunks

    def chunk(self, head, from_value_date=None, to_value_date=None):
        snapdata = self._chunks_to_ts(
            raw[2] for raw in self.rawchunks(head, from_value_date)
        )
        return snapdata.loc[from_value_date:to_value_date]

    def last(self, from_value_date=None, to_value_date=None):
        return self.find(from_value_date=from_value_date,
                         to_value_date=to_value_date)[1]

    def last_id(self, from_value_date=None, to_value_date=None):
        return self.find(from_value_date=from_value_date,
                         to_value_date=to_value_date)[0]

    def cset_heads_query(self, csetfilter=(), order='desc'):
        tablename = self.tsh._serie_to_tablename(self.cn, self.seriename)
        q = select(
            'ts.cset',  'ts.snapshot'
        ).table(
            f'"{self.tsh.namespace}.timeserie"."{tablename}" as ts',
        ).join(
            f'"{self.tsh.namespace}".changeset as cset on cset.id = ts.cset'
        )

        if csetfilter:
            q.where('ts.cset <= cset.id')
            for filtercb in csetfilter:
                filtercb(q)

        q.order('ts.id', order)
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
            table=self.name,
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
                lambda q: q.where('cset.id >= %(mincset)s', mincset=min(csets)),
                lambda q: q.where('cset.id <= %(maxcset)s', maxcset=max(csets))
            ),
            order='asc'
        )

        cset_snap_map = {
            row.cset: row.snapshot
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
                    ]
                )
            )
        return series

    def garbage(self):
        """ inefficient but simple garbage list builder
        garbage chunks are created on strip operations
        """
        tablename = self.tsh._serie_to_tablename(self.cn, self.seriename)
        reachablesql = f"""
        with recursive heads as (
            select snapshot from "{self.tsh.namespace}.timeserie"."{tablename}"
          ),
          allchunks as (
            select chunks.id as cid,
                   chunks.parent as parent
            from "{self.tsh.namespace}.snapshot"."{self.name}" as chunks
            where chunks.id in (select * from heads)
          union
            select chunks.id as cid,
                   chunks.parent as parent
            from "{self.tsh.namespace}.snapshot"."{self.name}" as chunks
            join allchunks on chunks.id = allchunks.parent
        )
        select cid from allchunks
        """

        reachable_chunks = {
            rev for rev, in self.cn.execute(reachablesql)
        }
        allsql = f'select id from "{self.tsh.namespace}.snapshot"."{self.name}" '
        allchuks = {
            rev for rev, in self.cn.execute(allsql).fetchall()
        }

        return allchuks - reachable_chunks

    def reclaim(self):
        todelete = ','.join(str(id) for id in self.garbage())
        sql = (f'delete from "{self.tsh.namespace}.snapshot"."{self.name}" '
               f'where id in ({todelete})')
        self.cn.execute(sql)
