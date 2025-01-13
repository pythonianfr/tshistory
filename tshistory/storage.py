import io
import os

import zstandard as zstd
import pandas as pd
import pytz
from sqlhelp import select

from tshistory.util import (
    empty_series,
    patch
)
from tshistory.codecs import (
    iohelper,
    node,
    rev
)


class base:
    _max_bucket_size = 150

    def buckets(self, ts):
        if len(ts) < self._max_bucket_size:
            return [ts]

        buckets = []
        for start in range(0, len(ts),
                           self._max_bucket_size):
            buckets.append(ts[start:start + self._max_bucket_size])
        return buckets


class Postgres(base):
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

    def insert_buckets(self, parent, ts):
        isstr = self.isstr
        for bucket in self.buckets(ts):
            start = bucket.index[0]
            end = bucket.index[-1]
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
        diffstart = series_diff.index[0]
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


class irange:
    __slots__ = 'start', 'end'

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __contains__(self, index):
        return self.start <= index <= self.end

    def __repr__(self):
        return f'<{self.start}:{self.end-self.start}:{self.end}>'


class pager:
    __slots__ = 'source', 'range', 'page', 'offset'
    _size = 2 ** 18

    def __init__(self, source):
        self.source = source
        self.page = None
        self.range = None
        self.offset = 0

    def _fault(self, start, end):
        # We want to read up to 256kb in a page knowing we need to
        # prepare for reads coming with lower addresses !
        # So what we are reading now should contain the start/end
        # interval at the very end.
        newstart = max(0, end - self._size)
        self.source.seek(newstart)
        readsize = min(self._size, end)
        self.page = self.source.read(readsize)
        self.range = irange(newstart, len(self.page) + newstart)
        self.offset = newstart
        assert len(self.page) == self.range.end - self.range.start

    def get(self, start, end):
        if not self.page:
            self._fault(start, end)

        if start not in self.range or end not in self.range:
            self._fault(start, end)

        return self.page[start-self.offset:end-self.offset]


class FS1(base):
    __slots__ = 'imeta', 'tz', 'root', 'cache'

    def __init__(self, cn, tsh, name, path=None):
        self.imeta = tsh.internal_metadata(cn, name)
        self.tz = pytz.utc if self.imeta['tzaware'] else None
        path = path or tsh._path(cn, name)
        self.root = tsh.root / path
        self.cache = {
            'rbfiles': {},
            'nodes': {},
            'blocks': None
        }

    def rbfile(self, path):
        """maintain a cache of files opened in binary read-only mode

        This will greatly help small accessor methods like .node_at
        and .chunk_at
        """
        cached = self.cache['rbfiles'].get(path)
        if cached:
            return cached
        self.cache['rbfiles'][path] = io.FileIO(path, 'rb')
        return self.rbfile(path)

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
        return self.revs_size // rev._size

    def revs_range(self, fromdate=None, todate=None, limit=None):
        if limit == 0:
            return []

        with open(self.revs, 'rb') as frevs:
            # long prologue to determine the boundaries
            index = None
            if fromdate is not None:
                index, startrev = self.find_rev(fromdate)
                if index is None:
                    # could not find anything fromdate is out of range
                    # if it is in the future, we can't do much
                    if fromdate > self.last_rev.revdate:
                        return []
            if fromdate is None or index is None:
                # we then can assume we start from the beginning
                index = 0
                frevs.seek(0)
                startrev = rev.unpack(self.tz, frevs.read(rev._size))

            if todate is not None:
                toindex, endrev = self.find_rev(todate)
                if toindex is None:
                    # could not find anything: todate is out of range
                    # if it is in the past, we can't do much
                    if todate < self.first_rev.revdate:
                        return []

            count = 1
            revs = [(index, startrev)]
            frevs.seek(rev._size * (index + 1))

            while True:
                if limit and count >= limit:
                    break
                brev = frevs.read(rev._size)
                if brev == b'':
                    break
                irev = rev.unpack(self.tz, brev)
                if todate is not None and todate < irev.revdate:
                    break
                count += 1
                index += 1
                revs.append((index, irev))

            return revs

    @property
    def last_rev(self):
        with open(self.revs, 'rb') as frevs:
            frevs.seek(self.revs_size - rev._size)  # end of penultimate rev
            return rev.unpack(self.tz, frevs.read(rev._size))

    @property
    def first_rev(self):
        with open(self.revs, 'rb') as frevs:
            frevs.seek(0)
            return rev.unpack(self.tz, frevs.read(rev._size))

    @property
    def tree_size(self):
        return os.stat(self.tree).st_size

    @property
    def tree_entries(self):
        return self.tree_size // node._size

    @property
    def chunks_size(self):
        return os.stat(self.chunks).st_size

    def chunk_at(self, start, size):
        blocks = self.cache['blocks']
        if blocks is None:
            # first visit, let's setup the pager
            self.cache['blocks'] = blocks = pager(self.rbfile(self.chunks))
        return blocks.get(start, start + size)

    def node_at(self, node_index):
        c = self.cache['nodes']
        n = c.get(node_index)
        if n:
            return n
        ftree = self.rbfile(self.tree)
        # We read not just 1 node but up to 64 if possible
        # and cache them.
        # Also we do this backwards from the asked index
        # since nodes are followed through the parent chain
        # up to the root.
        index = node_index
        delta = min(64, index - 1)
        node_index -= delta
        ftree.seek((node_index - 1) * node._size)
        bnodes = ftree.read(node._size * (delta + 1))
        for idx, n in enumerate(node.unpack_many(self.tz, bnodes)):
            c[node_index + idx] = n

        return self.node_at(index)

    def nodes(self, fromindex=1):
        nodes = []
        fromindex -= 1
        with open(self.tree, 'rb') as ftree:
            ftree.seek(node._size * fromindex)

            while True:
                bnode = ftree.read(node._size)
                if not len(bnode):
                    return nodes

                nodes.append(
                    node.unpack(self.tz, bnode)
                )

    def find_rev(self, revdate):
        with open(self.revs, 'rb') as frevs:
            start = 0
            end = self.revs_entries - 1

            frevs.seek(start)
            startrev = rev.unpack(self.tz, frevs.read(rev._size))
            frevs.seek(self.revs_size - rev._size)
            endrev = rev.unpack(self.tz, frevs.read(rev._size))

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
                frevs.seek(middle * rev._size)
                irev = rev.unpack(self.tz, frevs.read(rev._size))

                if revdate >= irev.revdate:
                    start = middle
                else:
                    end = middle

            frevs.seek(start * rev._size)
            return start, rev.unpack(self.tz, frevs.read(rev._size))

    def last(self, from_value_date=None, to_value_date=None):
        return self.get(self.last_rev.revdate, from_value_date, to_value_date)

    def get(self, revdate, from_value_date=None, to_value_date=None):
        _, rev = self.find_rev(revdate)
        if rev is None:
            rev = self.last_rev
            if revdate < rev.revdate:
                # that was in the past
                # for the future, we will provide the last rev
                return empty_series(self.imeta['tzaware'])

        chunks = []
        # walk the tree downwards
        for _, n in self.find_nodes_matching(rev.index, from_value_date, to_value_date):
            chunks.append(
                self.chunk_at(n.address, n.size)
            )

        if not chunks:
            return empty_series(self.imeta['tzaware'])
        chunks.reverse()
        return iohelper.chunks_to_ts(
            self.imeta, chunks, compressor=zstd
        )[from_value_date:to_value_date]

    def find_nodes_matching(self, nodeindex, mindate=None, maxdate=None):
        """return nodes (and their index) from a given index, walking
        down the parent chain until the end or a given date
        """
        node = self.node_at(nodeindex)
        yield (nodeindex, node)
        while True:
            if not node.parent:
                return
            nodeindex = node.parent
            node = self.node_at(nodeindex)
            if mindate and node.end < mindate:
                return
            if maxdate and node.start > maxdate:
                continue
            yield (nodeindex, node)

    def series_from_nodes(self, nodes):
        chunks = []
        for n in nodes:
            chunks.append(
                self.chunk_at(n.address, n.size)
            )

        chunks.reverse()
        if not chunks:
            return empty_series(self.imeta['tzaware'])
        return iohelper.chunks_to_ts(self.imeta, chunks, compressor=zstd)

    def initial_update(self, ts, revdate, metaid):
        # I/O prologue
        self.root.mkdir()
        open(self.root / 'revs', mode='x')
        open(self.root / 'tree', mode='x')
        open(self.root / 'chunks', mode='x')

        parent = 0  # no parent
        address = 0  # initial chunk
        self._update(parent, address, ts, revdate, ts.index[0], ts.index[-1], metaid)

    def update(self, ts, revdate, diffstart, diffend, metaid):
        """We will build a new node, whith a parent node.

        The parent may be immediate or older (at worst there is no parent)

        A node is a 18 bytes record. They are addressed from their
        index in the rev.

        """
        nodes = list(
            self.find_nodes_matching(self.last_rev.index, ts.index[0])
        )
        assert len(nodes)
        firstnode = nodes[-1][1]
        parentindex = nodes[-1][0]

        if nodes[0][1].end >= ts.index[0]:
            # there is an overlap, we need to patch our series with it
            base = self.series_from_nodes(
                [node for _, node in nodes]
            )
            ts = patch(base, ts)
            # this base node with which we just merged cannot be our
            # parent so we take its parent
            parentindex = firstnode.parent

        self._update(parentindex, self.chunks_size, ts, revdate, diffstart, diffend, metaid)

    def replace(self, ts, revdate, metaid):
        self._update(0, self.chunks_size, ts, revdate, ts.index[0], ts.index[-1], metaid)

    def _update(self, parent, address, ts, revdate, diffstart, diffend, metaid):
        # we can now have our tree node
        isstr = ts.values.dtype.name == 'object'
        for idx, bucket in enumerate(self.buckets(ts)):
            packed = iohelper.serialize_ts(bucket, isstr, compressor=zstd)
            newbnode = node.pack(
                bucket.index[0],
                bucket.index[-1],
                parent if not idx else self.tree_entries,  # index of the parent node
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
        newbrev = rev.pack(
            revdate or pd.Timestamp.utcnow(),
            diffstart,
            diffend,
            self.tree_entries,
            metaid
        )
        with open(self.revs, 'ab') as frevs:
            frevs.write(newbrev)

    def strip(self, revdate):
        index, rev = self.find_rev(revdate)
        # we will remove all that's above, and not touch
        # the nodes nor the chunks
        # a garbage collector maye be useful in the future ...
        with open(self.revs, 'r+b') as frevs:
            # truncates needs the 'r+b' mode to not mangle the file contents
            frevs.seek(0)
            frevs.truncate(index * rev._size)
