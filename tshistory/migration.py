import struct
import zlib

import numpy as np
import pandas as pd

from sqlalchemy import Table, Column, Integer, ForeignKey, Index, bindparam
from sqlalchemy.sql.elements import NONE_NAME
from sqlalchemy.sql.expression import select, asc, desc
from sqlalchemy.dialects.postgresql import BYTEA, TIMESTAMP

from tshistory.util import fromjson
from tshistory.snapshot import Snapshot


TABLES = {}

class SnapshotMigrator:
    __slots__ = ('cn', 'name', 'seriename', 'tsh')

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
                Column('start', TIMESTAMP(timezone=True)),
                Column('end', TIMESTAMP(timezone=True)),
                Column('chunk', BYTEA),
                Column('parent', Integer,
                       ForeignKey('{}.{}.id'.format(
                           self.namespace,
                           self.name))),
                Index(NONE_NAME, 'start'),
                Index(NONE_NAME, 'end'),
                Index(NONE_NAME, 'parent'),
                schema=self.namespace,
                keep_existing=True
            )
        return table

    # new serializer
    def _serialize(self, ts):
        if ts is None:
            return None
        # use `view` as a workarround for "cannot include dtype 'M' in a buffer"
        indexes = ts.index.view(np.uint8).data.tobytes()
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

    # old

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

    def _chunk_to_ts(self, chunk):
        body = b'{' + self._deserialize(chunk) + b'}'
        return self._ensure_tz_consistency(
            fromjson(body.decode('utf-8'), self.seriename)
        )

    # /serialisation

    allsql = """
        select chunk.id, chunk.chunk
        from "{namespace}"."{table}" as chunk
    """

    def allrawchunks(self):
        sql = self.allsql.format(
            namespace=self.namespace,
            table=self.name
        )
        return self.cn.execute(sql)

    @property
    def isstr(self):
        return self.tsh.metadata(self.cn, self.seriename)['value_type'] == 'object'

    def migratechunks(self):
        table = self.table
        chunks = self.allrawchunks()

        for idx, (cid, chunk) in enumerate(chunks):
            ts = self._chunk_to_ts(chunk)
            sql = table.update().values(
                chunk=self._serialize(ts)
            ).where(
                table.c.id == cid
            )
            self.cn.execute(sql)

        self.tsh.update_metadata(
            self.cn, self.seriename,
            {
                'index_dtype': ts.index.dtype.str,
                'value_dtype': ts.dtypes.str if not self.isstr else '|O'
            }
        )

        print('chunks for serie {}: {}'.format(self.seriename, idx + 1))

    rawdatessql = """
        with recursive allchunks as (
            select chunks.id as cid,
                   chunks.parent as parent,
                   chunks.start as start,
                   chunks."end" as "end"
            from "{namespace}"."{table}" as chunks
            where chunks.id in ({heads})
          union
            select chunks.id as cid,
                   chunks.parent as parent,
                   chunks.start as start,
                   chunks."end" as "end"
            from "{namespace}"."{table}" as chunks
            join allchunks on chunks.id = allchunks.parent
        )
        select cid, parent, start, "end" from allchunks
    """

    def alldates(self, heads):
        sql = self.rawdatessql.format(
            namespace=self.namespace,
            table=self.name,
            heads=','.join(str(head) for head in heads)
        )
        res = self.cn.execute(sql)
        dates = {cid: (parent, start, end)
                  for cid, parent, start, end in res.fetchall()}
        return dates

    def findall_startend(self, csets):
        snap = Snapshot(self.cn, self.tsh, self.seriename)
        sql = snap.cset_heads_query((lambda cset: cset.c.id >= min(csets),
                                     lambda cset: cset.c.id <= max(csets)),
                                     order=asc)

        cset_snap_map = {
            row.cset: row.snapshot
            for row in self.cn.execute(sql).fetchall()
        }
        alldates = self.alldates(
            sorted(cset_snap_map.values())
        )

        series = []
        for cset in csets:
            head = cset_snap_map[cset]
            start, end = None, None
            while True:
                parent, cstart, cend = alldates.get(head, (None, None, None))
                if cstart is None:
                    break
                start = min(start or cstart, cstart)
                end = max(end or cend, cend)
                head = parent
            series.append((cset, start, end))
        return series

    def migrateseries(self):
        sql = 'select id, cset from "{ns}.timeserie"."{name}"'.format(
            ns=self.tsh.namespace,
            name=self.name
        )
        revs = [row for row in self.cn.execute(sql)]
        sql = ('alter table "{}.timeserie"."{}" '
               'add column "start" timestamp, '
               'add column "end" timestamp '
        ).format(self.tsh.namespace, self.name)
        self.cn.execute(sql)
        sql = 'create index on "{}.timeserie"."{}" (start)'.format(self.tsh.namespace, self.name)
        self.cn.execute(sql)
        sql = 'create index on "{}.timeserie"."{}" ("end")'.format(self.tsh.namespace, self.name)
        self.cn.execute(sql)

        fromto = self.findall_startend([row.cset for row in revs])
        cset_id_map = {
            row.cset: row.id
            for row in revs
        }
        table = self.tsh._get_ts_table(self.cn, self.seriename)
        sql = table.update().where(
            table.c.id == bindparam('id_')
        ).values({
            'start': bindparam('start'),
            'end': bindparam('end')
        })
        self.cn.execute(sql, [
            {'id_': cset_id_map[cset], 'start': start, 'end': end}
            for cset, start, end in fromto
        ])
        print('versions for serie {}: {}'.format(self.seriename, len(fromto)))


    def fix_start_end(self):
        sql = 'select id, cset from "{ns}.timeserie"."{name}"'.format(
            ns=self.tsh.namespace,
            name=self.name
        )
        revs = [row for row in self.cn.execute(sql)]
        fromto = self.findall_startend([row.cset for row in revs])
        cset_id_map = {
            row.cset: row.id
            for row in revs
        }
        table = self.tsh._get_ts_table(self.cn, self.seriename)
        sql = table.update().where(
            table.c.id == bindparam('id_')
        ).values({
            'start': bindparam('start'),
            'end': bindparam('end')
        })
        self.cn.execute(sql, [
            {'id_': cset_id_map[cset], 'start': start, 'end': end}
            for cset, start, end in fromto
        ])
        print('versions for serie {}: {}'.format(self.seriename, len(fromto)))
