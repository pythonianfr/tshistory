import os

import pandas as pd
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
    _node_size = 18
    _max_bucket_size = 150

    def __init__(self, root, name):
        self.name = name
        self.root = root / name
        self.revs = self.root / 'revs'
        self.tree = self.root / 'tree'
        self.chunks = self.root / 'chunks'

    def initialize(self):
        self.root.mkdir()
        open(self.revs, mode='x')
        open(self.tree, mode='x')
        open(self.chunks, mode='x')

    @property
    def revs_size(self):
        return os.stat(self.revs).st_size

    @property
    def revs_entries(self):
        return self.revs_size // self._rev_size

    @property
    def last_rev(self):
        with open(self.revs, 'rb') as frevs:
            frevs.seek(self.revs_size - self._rev_size)  # end of penultimate rev
            brev = frevs.read(self._rev_size)
            return iohelper.unpack_rev(brev)

    def revisions(self):
        revs = []
        with open(self.revs, 'rb') as frevs:
            while True:
                brev = frevs.read(self._rev_size)
                if not brev:
                    break
                revs.append(iohelper.unpack_rev(brev))
        return revs

    @property
    def tree_size(self):
        return os.stat(self.tree).st_size

    @property
    def tree_entries(self):
        return self.tree_size // self._node_size

    @property
    def chunks_size(self):
        return os.stat(self.chunks).st_size

    def node_at(self, node_index):
        with open(self.tree, 'rb') as ftree:
            ftree.seek((node_index - 1) * self._node_size)
            bnode = ftree.read(self._node_size)

        return iohelper.unpack_node(bnode)

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

    def initial_update(self, ts, revdate, authorid, metaid):
        buckets = self.buckets(ts)

        parent = 0  # no parent
        address = 0  # initial chunk
        for idx, bucket in enumerate(buckets, start=1):
            packed = iohelper.serialize_ts(bucket, False)
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
            address = len(packed)
            with open(self.tree, 'ab') as ftree:
                ftree.write(bnode)

        brev = iohelper.pack_rev(
            revdate or pd.Timestamp.utcnow(),
            ts.index[0],
            ts.index[-1],
            ts.index[0],
            ts.index[-1],
            idx,
            authorid,
            metaid
        )

        with open(self.revs, 'ab') as frevs:
            frevs.write(brev)

    def last(self, imeta, from_value_date=None):
        node = self.node_at(self.last_rev.index)
        chunks = []
        # walk the tree downwards
        while True:
            chunks.append(
                self.chunk_at(node.address, node.size)
            )
            parent = node.parent
            if not parent:
                break
            if from_value_date and from_value_date >= node.start:
                break
            node = self.node_at(parent)

        if not chunks:
            return empty_series(imeta['tzaware'])
        chunks.reverse()
        return iohelper.chunks_to_ts(imeta, chunks)[from_value_date:]

    def find_node_index_matching(self, nodeindex, mindate):
        node = self.node_at(nodeindex)
        if mindate < node.start:
            return self.find_node_index_matching(node.parent, mindate)
        return nodeindex, True  # we found the base node

    def update(self, ts, imeta, revdate, start, end, diffstart, diffend, authorid, metaid):
        """We will build a new node, whith a parent node.

        The parent may be immediate or older (at worst there is no parent)

        A node is a 18 bytes record. They are addressed from their
        index in the rev.

        """
        # fetch the latest node, which will be our parent
        rev = self.last_rev
        nodeindex, patchme = self.find_node_index_matching(rev.index, ts.index.min())

        if patchme:
            # we found a parent node, and we need to patch our series with it
            node = self.node_at(nodeindex)
            chunk = self.chunk_at(node.address, node.size)
            base = iohelper.chunks_to_ts(imeta, [chunk])
            ts = patch(base, ts)
            # this node with which we just merge cannot be our parent
            # so we take its parent
            nodeindex = node.parent

        # we can now have our tree node
        for index, bucket in enumerate(self.buckets(ts), start=nodeindex):
            packed = iohelper.serialize_ts(bucket, False)

            newbnode = iohelper.pack_node(
                bucket.index.min(),
                bucket.index.max(),
                index,  # index of the parent node
                self.chunks_size,
                len(packed)
            )

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
            start,
            end,
            diffstart,
            diffend,
            self.tree_entries,
            authorid,
            metaid
        )
        with open(self.revs, 'ab') as frevs:
            frevs.write(newbrev)
