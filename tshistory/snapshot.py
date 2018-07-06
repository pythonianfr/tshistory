import pandas as pd
import zlib

from sqlalchemy import Table, Column, Integer, ForeignKey
from sqlalchemy.sql.expression import select, asc, desc
from sqlalchemy.dialects.postgresql import BYTEA, TIMESTAMP

from tshistory.util import (
    fromjson,
    subset,
    SeriesServices,
    tojson
)


def patch_sqlalchemy():
    # PATCH for sqlalchemy https://bitbucket.org/zzzeek/sqlalchemy/issues/4289/automatic-index-name-hash-collision
    from sqlalchemy.sql import compiler
    elements = compiler.elements
    util = compiler.util

    def _prepared_index_name(self, index, include_schema=False):
        if index.table is not None:
            effective_schema = self.preparer.schema_for_object(index.table)
        else:
            effective_schema = None
        if include_schema and effective_schema:
            schema_name = self.preparer.quote_schema(effective_schema)
        else:
            schema_name = None

        ident = index.name
        if isinstance(ident, elements._truncated_label):
            max_ = self.dialect.max_index_name_length or \
                self.dialect.max_identifier_length
            if len(ident) > max_:
                # PATCH
                idhash = util.md5_hex(ident)
                ident = ident[0:max_ - len(idhash)] + "_" + idhash
                # /PATCH
        else:
            self.dialect.validate_identifier(ident)

        index_name = self.preparer.quote(ident)

        if schema_name:
            index_name = schema_name + "." + index_name
        return index_name

    compiler.DDLCompiler._prepared_index_name = _prepared_index_name

patch_sqlalchemy()
# /patch

TABLES = {}


class Snapshot(SeriesServices):
    __slots__ = ('cn', 'name', 'tsh')
    _max_bucket_size = 250

    def __init__(self, cn, tsh, seriename):
        self.cn = cn
        self.tsh = tsh
        self.seriename = seriename
        self.name = self.tsh._serie_to_tablename(cn, seriename)

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
                keep_existing=True
            )
        return table

    # optimized/asymmetric de/serialisation

    def _serialize(self, ts):
        if ts is None:
            return None
        return zlib.compress(tojson(ts, self._precision).encode('utf-8')[1:-1])

    def _deserialize(self, bytestring):
        return zlib.decompress(bytestring)

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
        body = b'{' + b','.join(self._deserialize(chunk) for chunk in chunks) + b'}'
        return self._ensure_tz_consistency(
            fromjson(body.decode('utf-8'), self.seriename)
        )

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
        return self.insert_buckets(None, initial_ts)

    def update(self, diff):
        # get last chunkhead for cset
        tstable = self.tsh._get_ts_table(self.cn, self.seriename)
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

        if diffstart > oldsnapshot.index.max():
            # append: let't not rewrite anything
            newsnapshot = diff
            parent = cid
        else:
            # we got a point override, need to patch
            newsnapshot = self.patch(oldsnapshot, diff)

        return self.insert_buckets(parent, newsnapshot)

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
        return self.find(seriefilter=[lambda serie: serie.c.id == 1])[1]

    def last(self, from_value_date=None, to_value_date=None):
        return self.find(from_value_date=from_value_date,
                         to_value_date=to_value_date)[1]

    def last_id(self, from_value_date=None, to_value_date=None):
        return self.find(from_value_date=from_value_date,
                         to_value_date=to_value_date)[0]

    def cset_heads_query(self, csetfilter=(), seriefilter=(), order=desc):
        cset = self.tsh.schema.changeset
        serie = self.tsh._get_ts_table(self.cn, self.seriename)
        sql = select([serie.c.cset, serie.c.snapshot]
        ).order_by(order(serie.c.id)
        ).select_from(serie.join(cset))

        if csetfilter:
            sql = sql.where(serie.c.cset <= cset.c.id)
            for filtercb in csetfilter:
                sql = sql.where(filtercb(cset))

        for tablecb in seriefilter:
            sql = sql.where(tablefilter(serie))

        return sql

    def find(self, csetfilter=(), seriefilter=(),
             from_value_date=None, to_value_date=None):

        sql = self.cset_heads_query(csetfilter, seriefilter)
        sql = sql.limit(1)

        try:
            csid, cid = self.cn.execute(sql).fetchone()
        except TypeError:
            # this happens *only* because of the from/to restriction
            return None, None

        chunk = self.chunk(cid, from_value_date, to_value_date)
        return csid, chunk

    def allchunks(self, heads, from_value_date=None):
        where = ''
        if from_value_date:
            where = 'where chunks.end >= %(start)s '

        sql = """
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
        """.format(namespace=self.namespace,
                   table=self.name,
                   heads=','.join(str(head) for head in heads),
                   where=where)
        res = self.cn.execute(sql, start=from_value_date)
        chunks = {cid: (parent, rawchunk)
                  for cid, parent, rawchunk in res.fetchall()}
        return chunks

    def findall(self, revs, from_value_date, to_value_date):
        csets = [rev for rev, _ in revs if rev is not None]
        # csid -> heads

        sql = self.cset_heads_query((lambda cset: cset.c.id >= min(csets),
                                     lambda cset: cset.c.id <= max(csets)),
                                     order=asc)

        cset_snap_map = {
            row.cset: row.snapshot
            for row in self.cn.execute(sql).fetchall()
        }
        rawchunks = self.allchunks(
            sorted(cset_snap_map.values()),
            from_value_date
        )

        series = []
        for cset, idate in revs:
            if cset is None:
                series.append((idate, None))
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
                (idate, subset(self._chunks_to_ts(chunks),
                               from_value_date,
                               to_value_date)
                )
            )
        return series
