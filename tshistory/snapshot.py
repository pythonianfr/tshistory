import pandas as pd
import zlib

from sqlalchemy import Table, Column, Integer, ForeignKey
from sqlalchemy.sql.expression import select, desc
from sqlalchemy.dialects.postgresql import BYTEA, TIMESTAMP

from tshistory.util import (
    fromjson,
    subset,
    SeriesServices,
    tojson
)


TABLES = {}


class Snapshot(SeriesServices):
    __slots__ = ('cn', 'name', 'tsh')
    _max_bucket_size = 250
    _min_bucket_size = 10

    def __init__(self, cn, tsh, seriename):
        self.cn = cn
        self.tsh = tsh
        self.name = seriename

    @property
    def namespace(self):
        return '{}.snapshot'.format(self.tsh.namespace)

    @property
    def table(self):
        tablename = '{}.{}'.format(self.namespace, self.name)
        table = TABLES.get(tablename)
        if table is None:
            TABLES[tablename] = table = Table(
                self.name, self.tsh.schema.meta,
                Column('id', Integer, primary_key=True),
                Column('start', TIMESTAMP(timezone=True), index=True),
                Column('end', TIMESTAMP(timezone=True), index=True),
                Column('chunk', BYTEA),
                Column('parent', Integer,
                       ForeignKey('{}.{}.id'.format(
                           self.namespace,
                           self.name)),
                       index=True),
                schema=self.namespace,
                extend_existing=True
            )
        return table

    # optimized/asymmetric de/serialisation

    def _serialize(self, ts):
        if ts is None:
            return None
        return zlib.compress(tojson(ts, self._precision).encode('utf-8')[1:-1])

    def _deserialize(self, bytestring):
        return zlib.decompress(bytestring)

    def _chunks_to_ts(self, chunks):
        body = b'{' + b','.join(self._deserialize(chunk) for chunk in chunks) + b'}'
        return self.tsh._ensure_tz_consistency(
            self.cn,
            fromjson(body.decode('utf-8'), self.name)
        )

    # /serialisation

    def split(self, ts):
        if len(ts) < self._max_bucket_size:
            return [ts]

        buckets = []
        for start in range(0, len(ts),
                           self._max_bucket_size):
            buckets.append(ts[start:start + self._max_bucket_size])
        return buckets

    def insert_buckets(self, parent, buckets):
        for bucket in buckets:
            start = bucket.index.min()
            end = bucket.index.max()
            sql = self.table.insert().values(
                start=start,
                end=end,
                parent=parent,
                chunk=self._serialize(bucket)
            )
            parent = self.cn.execute(sql).inserted_primary_key[0]

        return parent

    def create(self, initial_ts):
        self.table.create(self.cn)
        buckets = self.split(initial_ts)
        return self.insert_buckets(None, buckets)

    def update(self, diff):
        # get last chunkhead for cset
        tstable = self.tsh._get_ts_table(self.cn, self.name)
        headsql = select(
            [tstable.c.snapshot]
        ).order_by(desc(tstable.c.id)
        ).limit(1)
        head = self.cn.execute(headsql).scalar()

        # get raw chunks matching the limits
        diffstart = diff.index.min()
        rawchunks = self.rawchunks(head, diffstart)
        cid, parent, _ = rawchunks[0]
        oldsnapshot = self._chunks_to_ts(row[2] for row in rawchunks)

        if (len(oldsnapshot) >= self._min_bucket_size and
            diffstart > oldsnapshot.index.max()):
            # append: let't not rewrite anything
            newsnapshot = diff
            parent = cid
        else:
            # we got a point override, need to patch
            newsnapshot = self.patch(oldsnapshot, diff)
        buckets = self.split(newsnapshot)

        return self.insert_buckets(parent, buckets)

    def rawchunks(self, head, from_value_date=None):
        where = ''
        if from_value_date:
            where = 'where chunks.end >= %(start)s '

        sql = """
        with recursive allchunks as (
            select chunks.id as cid,
                   chunks.parent as parent,
                   chunks.chunk as chunk
            from "{namespace}"."{table}" as chunks
            where chunks.id = {head}
          union
            select chunks.id as cid,
                   chunks.parent as parent,
                   chunks.chunk as chunk
            from "{namespace}"."{table}" as chunks
            join allchunks on chunks.id = allchunks.parent
            {where}
        )
        select cid, parent, chunk from allchunks
        """.format(namespace=self.namespace,
                   table=self.name,
                   head=head,
                   where=where)
        res = self.cn.execute(sql, start=from_value_date)
        chunks = [(cid, parent, rawchunk)
                  for cid, parent, rawchunk in res.fetchall()]
        chunks.reverse()
        return chunks

    def chunk(self, head, from_value_date=None, to_value_date=None):
        snapdata = self._chunks_to_ts(
            raw[2] for raw in self.rawchunks(head, from_value_date)
        )
        return subset(snapdata,
            from_value_date, to_value_date
        )

    @property
    def first(self):
        return self.find(qfilter=[lambda _, table: table.c.id == 1])[1]

    def last(self, from_value_date=None, to_value_date=None):
        return self.find(from_value_date=from_value_date,
                         to_value_date=to_value_date)[1]

    def find(self, qfilter=(),
             from_value_date=None, to_value_date=None):
        cset = self.tsh.schema.changeset
        table = self.tsh._get_ts_table(self.cn, self.name)
        sql = select([table.c.cset, table.c.snapshot]
        ).order_by(desc(table.c.id)
        ).limit(1
        ).select_from(table.join(cset))

        if qfilter:
            sql = sql.where(table.c.cset <= cset.c.id)
            for filtercb in qfilter:
                sql = sql.where(filtercb(cset, table))

        try:
            csid, cid = self.cn.execute(sql).fetchone()
        except TypeError:
            # this happens *only* because of the from/to restriction
            return None, None

        chunk = self.chunk(cid, from_value_date, to_value_date)
        return csid, chunk
