import os

import pandas as pd
import pytz
from sqlhelp import select

from tshistory.util import (
    empty_series,
    patch
)
from tshistory.codecs import iohelper


class Postgres:
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

    Now let's look at the logical organisation: we have two relations
    `Revision` (which tracks all successive versions of a series) and
    `Tree` (which actually stores the series data using a tree
    structure).

    Series values   | Revisions table | Tree of chunks table
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
    _max_bucket_size = 150

    def __init__(self, cn, tsh, name):
        self.cn = cn
        self.tsh = tsh
        self.name = name
        self.tablename = self.tsh._series_to_tablename(cn, name)

    @property
    def isstr(self):
        return self.tsh.internal_metadata(
            self.cn, self.name
        )['value_type'] == 'object'

    def buckets(self, ts):
        if len(ts) < self._max_bucket_size:
            return [ts]

        buckets = []
        for start in range(0, len(ts),
                           self._max_bucket_size):
            buckets.append(ts[start:start + self._max_bucket_size])
        return buckets

    def insert_buckets(self, parent, ts):
        isstr = self.isstr
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
                iohelper.serialize_ts(bucket, isstr)
            ).scalar()

        return parent

    def create(self, initial_ts):
        return self.insert_buckets(None, initial_ts)

    def update(self, series_diff):
        meta = self.tsh.internal_metadata(self.cn, self.name)
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
        oldsnapshot = iohelper.chunks_to_ts(
            meta,
            (row[2] for row in rawchunks)
        )

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
        meta = self.tsh.internal_metadata(self.cn, self.name)
        snapdata = iohelper.chunks_to_ts(
            meta,
            (raw[2] for raw in self.rawchunks(head, from_value_date))
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


class FS1:
    _rev_size = 32
    _node_size = 26
    _max_bucket_size = 150
    __slots__ = 'imeta', 'tz', 'root'

    def __init__(self, cn, tsh, name, path=None):
        self.imeta = tsh.internal_metadata(cn, name)
        self.tz = pytz.utc if self.imeta['tzaware'] else None
        path = path or tsh._path(cn, name)
        self.root = tsh.root / path

    @property
    def tzaware(self):
        return self.imeta['tzaware']

    @property
    def revs(self):
        return self.root / 'revs'

    @property
    def tree(self):
        return self.root / 'tree'

    @property
    def chunks(self):
        return self.root / 'chunks'

    @property
    def revs_size(self):
        return os.stat(self.revs).st_size

    @property
    def revs_entries(self):
        return self.revs_size // self._rev_size

    def revs_range(self, fromdate=None, todate=None, limit=None):
        if limit == 0:
            return []

        tz = pytz.utc if self.imeta['tzaware'] else None
        with open(self.revs, 'rb') as frevs:
            # long prologue to determine the boundaries
            index = None
            if fromdate is not None:
                index, startrev = self.find_rev(tz, fromdate)
                if index is None:
                    # could not find anything fromdate is out of range
                    # if it is in the future, we can't do much
                    if fromdate > self.last_rev(tz).revdate:
                        return []
            if fromdate is None or index is None:
                # we then can assume we start from the beginning
                index = 0
                frevs.seek(0)
                startrev = iohelper.unpack_rev(tz, frevs.read(self._rev_size))

            if todate is not None:
                toindex, endrev = self.find_rev(tz, todate)
                if toindex is None:
                    # could not find anything: todate is out of range
                    # if it is in the past, we can't do much
                    if todate < self.first_rev(tz).revdate:
                        return []

            count = 1
            revs = [(index, startrev)]
            frevs.seek(self._rev_size * (index + 1))

            while True:
                if limit and count >= limit:
                    break
                brev = frevs.read(self._rev_size)
                if brev == b'':
                    break
                rev = iohelper.unpack_rev(tz, brev)
                if todate is not None and todate < rev.revdate:
                    break
                count += 1
                index += 1
                revs.append((index, rev))

            return revs

    def last_rev(self, tz):
        with open(self.revs, 'rb') as frevs:
            frevs.seek(self.revs_size - self._rev_size)  # end of penultimate rev
            return iohelper.unpack_rev(tz, frevs.read(self._rev_size))

    def first_rev(self, tz):
        with open(self.revs, 'rb') as frevs:
            frevs.seek(0)
            return iohelper.unpack_rev(tz, frevs.read(self._rev_size))

    @property
    def tree_size(self):
        return os.stat(self.tree).st_size

    @property
    def tree_entries(self):
        return self.tree_size // self._node_size

    @property
    def chunks_size(self):
        return os.stat(self.chunks).st_size

    def node_at(self, tz, node_index):
        with open(self.tree, 'rb') as ftree:
            ftree.seek((node_index - 1) * self._node_size)
            bnode = ftree.read(self._node_size)

        return iohelper.unpack_node(tz, bnode)

    @property
    def nodes(self):
        nodes = []
        with open(self.tree, 'rb') as ftree:
            while True:
                bnode = ftree.read(self._node_size)
                if not len(bnode):
                    return nodes
                nodes.append(
                    iohelper.unpack_node(self.tz, bnode)
                )

    def chunk_at(self, start, size):
        with open(self.chunks, 'rb') as fchunks:
            fchunks.seek(start)
            return fchunks.read(size)

    def buckets(self, ts):
        if len(ts) < self._max_bucket_size:
            return [ts]

        buckets = []
        for start in range(0, len(ts), self._max_bucket_size):
            buckets.append(ts[start:start + self._max_bucket_size])
        return buckets

    def initial_update(self, ts, revdate, metaid):
        # I/O prologue
        self.root.mkdir()
        open(self.root / 'revs', mode='x')
        open(self.root / 'tree', mode='x')
        open(self.root / 'chunks', mode='x')

        # the actual update
        buckets = self.buckets(ts)

        parent = 0  # no parent
        address = 0  # initial chunk
        isstr = ts.values.dtype.name == 'object'
        for idx, bucket in enumerate(buckets, start=1):
            packed = iohelper.serialize_ts(bucket, isstr)
            with open(self.chunks, 'ab') as fchunks:
                fchunks.write(packed)

            bnode = iohelper.pack_node(
                bucket.index[0],
                bucket.index[-1],
                parent,
                address,
                len(packed)
            )
            parent = idx
            address = address + len(packed)
            with open(self.tree, 'ab') as ftree:
                ftree.write(bnode)

        brev = iohelper.pack_rev(
            revdate or pd.Timestamp.utcnow(),
            ts.index[0],
            ts.index[-1],
            idx,
            metaid
        )

        with open(self.revs, 'ab') as frevs:
            frevs.write(brev)

    def find_rev(self, tz, revdate):
        with open(self.revs, 'rb') as frevs:
            start = 0
            end = self.revs_entries - 1

            frevs.seek(start)
            startrev = iohelper.unpack_rev(tz, frevs.read(self._rev_size))
            frevs.seek(self.revs_size - self._rev_size)
            endrev = iohelper.unpack_rev(tz, frevs.read(self._rev_size))

            if revdate < startrev.revdate:
                return None, None
            if revdate == startrev.revdate:
                return start, startrev
            if revdate == endrev.revdate:
                return end, endrev
            if revdate > endrev.revdate:
                return None, None

            # now, let's bisect between these points to find the best
            # candidate
            while end - start > 1:
                middle = (start + end) >> 1

                # seek + read
                frevs.seek(middle * self._rev_size)
                rev = iohelper.unpack_rev(tz, frevs.read(self._rev_size))

                if revdate >= rev.revdate:
                    start = middle
                else:
                    end = middle

            frevs.seek(start * self._rev_size)
            rev = iohelper.unpack_rev(tz, frevs.read(self._rev_size))
            return start, rev

    def last(self, from_value_date=None, to_value_date=None):
        tz = pytz.utc if self.imeta['tzaware'] else None
        return self.get(self.last_rev(tz).revdate, from_value_date, to_value_date)

    def get(self, revdate, from_value_date=None, to_value_date=None):
        tz = pytz.utc if self.imeta['tzaware'] else None
        _, rev = self.find_rev(tz, revdate)
        if rev is None:
            rev = self.last_rev(tz)
            if revdate < rev.revdate:
                # that was in the past
                # for the future, we will provide the last rev
                return empty_series(self.imeta['tzaware'])
        node = self.node_at(tz, rev.index)

        chunks = []
        # walk the tree downwards
        while True:
            if not (to_value_date and to_value_date < node.start):
                # we can skip collecting irrelevant chunks
                chunks.append(
                    self.chunk_at(node.address, node.size)
                )
            parent = node.parent
            if not parent:
                break
            if from_value_date and from_value_date >= node.start:
                break
            node = self.node_at(tz, parent)

        if not chunks:
            return empty_series(self.imeta['tzaware'])
        chunks.reverse()
        return iohelper.chunks_to_ts(self.imeta, chunks)[from_value_date:to_value_date]

    def find_node_index_matching(self, tz, nodeindex, mindate):
        node = self.node_at(tz, nodeindex)
        if mindate < node.start:
            if not node.parent:
                return 0  # no base node
            return self.find_node_index_matching(tz, node.parent, mindate)

        assert nodeindex >= 1
        return nodeindex  # we found the base node

    def update(self, ts, revdate, diffstart, diffend, metaid):
        """We will build a new node, whith a parent node.

        The parent may be immediate or older (at worst there is no parent)

        A node is a 18 bytes record. They are addressed from their
        index in the rev.

        """
        # fetch the latest node, which will be our parent
        tz = pytz.utc if self.imeta['tzaware'] else None
        rev = self.last_rev(tz)
        nodeindex = self.find_node_index_matching(tz, rev.index, ts.index.min())
        if nodeindex == 0:
            # O means we didn't find any !
            # we're adding new points in the past
            # That's fine but we will merge
            nodeindex = 1
        node = self.node_at(tz, nodeindex)
        base = iohelper.chunks_to_ts(
            self.imeta,
            [self.chunk_at(node.address, node.size)]
        )

        if base.index.max() >= ts.index.min():
            # there is an overlap, we need to patch our series with it
            ts = patch(base, ts)
            # this node with which we just merge cannot be our parent
            # so we take its parent
            nodeindex = node.parent

        # we can now have our tree node
        address = self.chunks_size
        for index, bucket in enumerate(self.buckets(ts), start=nodeindex):
            packed = iohelper.serialize_ts(
                bucket,
                ts.values.dtype.name == 'object'
            )

            newbnode = iohelper.pack_node(
                bucket.index.min(),
                bucket.index.max(),
                index,  # index of the parent node
                address,
                len(packed)
            )
            address = address + len(packed)

            # build and write the chunk
            # do this *after* the previous step
            # to have the correct chunks size
            with open(self.chunks, 'ab') as fchunks:
                fchunks.write(packed)

            # write it and get the index
            with open(self.tree, 'ab') as ftree:
                ftree.write(newbnode)

        # let's create the rev
        newbrev = iohelper.pack_rev(
            revdate or pd.Timestamp.utcnow(),
            diffstart,
            diffend,
            self.tree_entries,
            metaid
        )
        with open(self.revs, 'ab') as frevs:
            frevs.write(newbrev)

    def replace(self, ts, revdate, diffstart, diffend, metaid):
        isstr = ts.values.dtype.name == 'object'
        for index, bucket in enumerate(self.buckets(ts)):
            # index always starts at Zero, because we replace everything
            packed = iohelper.serialize_ts(bucket, isstr)

            newbnode = iohelper.pack_node(
                bucket.index.min(),
                bucket.index.max(),
                index,
                self.chunks_size,
                len(packed)
            )

            with open(self.chunks, 'ab') as fchunks:
                fchunks.write(packed)

            with open(self.tree, 'ab') as ftree:
                ftree.write(newbnode)

        # let's create the rev
        newbrev = iohelper.pack_rev(
            revdate or pd.Timestamp.utcnow(),
            diffstart,
            diffend,
            self.tree_entries,
            metaid
        )
        with open(self.revs, 'ab') as frevs:
            frevs.write(newbrev)

    def strip(self, revdate):
        index, rev = self.find_rev(None, revdate)
        # we will remove all that's above, and not touch
        # the nodes nor the chunks
        # a garbage collector maye be useful in the future ...
        with open(self.revs, 'r+b') as frevs:
            # truncates needs the 'r+b' mode to not mangle the file contents
            frevs.seek(0)
            frevs.truncate(index * self._rev_size)
