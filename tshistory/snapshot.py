import pandas as pd

from sqlalchemy import Table, Column, Integer, ForeignKey
from sqlalchemy.sql.expression import select, desc, func
from sqlalchemy.dialects.postgresql import BYTEA

from tshistory.util import (
    subset,
    SeriesServices,
)


class Snapshot(SeriesServices):
    __slots__ = ('cn', 'name', 'tsh')
    _interval = 10

    def __init__(self, cn, tsh, seriename):
        self.cn = cn
        self.tsh = tsh
        self.name = seriename

    @property
    def table(self):
        return Table(
            self.name, self.tsh.schema.meta,
            Column('id', Integer, primary_key=True),
            Column('cset', Integer,
                   ForeignKey('{}.changeset.id'.format(self.tsh.namespace)),
                   index=True, nullable=False),
            Column('chunk', BYTEA),
            schema='{}.snapshot'.format(self.tsh.namespace),
            extend_existing=True
        )

    def create(self, csid, initial_ts):
        self.table.create(self.cn)
        sql = self.table.insert().values(
            cset=csid,
            chunk=self._serialize(initial_ts)
        )
        self.cn.execute(sql)

    def update(self, csid, diff):
        # note the current tip id for later
        table = self.table
        sql = select([func.max(table.c.id)])
        tipid = self.cn.execute(sql).scalar()

        snapshot = self.last
        newsnapshot = self.patch(snapshot, diff)
        sql = table.insert().values(
            cset=csid,
            chunk=self._serialize(newsnapshot)
        )
        self.cn.execute(sql)

        if tipid > 1 and tipid % self._interval:
            self.cn.execute(table.delete().where(table.c.id == tipid))

    @property
    def first(self):
        return self.find(qfilter=[lambda _, table: table.c.id == 1])[1]

    @property
    def last(self):
        return self.find()[1]

    def find(self, qfilter=(),
             from_value_date=None, to_value_date=None):
        cset = self.tsh.schema.changeset
        table = self.table
        sql = select([table.c.cset, table.c.chunk]
        ).order_by(desc(table.c.id)
        ).limit(1
        ).select_from(table.join(cset))

        if qfilter:
            sql = sql.where(table.c.cset <= cset.c.id)
            for filtercb in qfilter:
                sql = sql.where(filtercb(cset, table))

        try:
            csid, snapdata = self.cn.execute(sql).fetchone()
            snapdata = subset(self._deserialize(snapdata, self.name),
                              from_value_date, to_value_date)
            snapdata = self.tsh._ensure_tz_consistency(self.cn, snapdata)
        except TypeError:
            # this happens *only* because of the from/to restriction
            return None, None
        return csid, snapdata

    def build_upto(self, qfilter=(),
                   from_value_date=None, to_value_date=None):
        csid, snapshot = self.find(qfilter,
                                   from_value_date=from_value_date,
                                   to_value_date=to_value_date)

        if csid is None:
            return

        cset = self.tsh.schema.changeset
        # beware the potential cartesian product
        # between table & cset if there is no qfilter
        table = self.tsh._get_ts_table(self.cn, self.name)
        sql = select([table.c.id,
                      table.c.diff,
                      cset.c.insertion_date]
        ).order_by(table.c.id
        ).where(table.c.cset > csid)

        if qfilter:
            sql = sql.where(table.c.cset == cset.c.id)
            for filtercb in qfilter:
                sql = sql.where(filtercb(cset, table))

        alldiffs = self.cn.execute(sql).fetchall()
        if not len(alldiffs):
            return snapshot

        # initial ts
        ts = self._deserialize(alldiffs[0].diff, self.name)
        ts = self.tsh._ensure_tz_consistency(self.cn, ts)
        for row in alldiffs[1:]:
            diff = subset(self._deserialize(row.diff, self.name),
                          from_value_date, to_value_date)
            diff = self.tsh._ensure_tz_consistency(self.cn, diff)
            ts = self.patch(ts, diff)
        ts = self.patch(snapshot, ts)
        assert ts.index.dtype.name == 'datetime64[ns]' or len(ts) == 0
        return ts

    def strip_at(self, csid):
        table = self.table
        self.cn.execute(table.delete().where(table.c.cset >= csid))
        if self.cn.execute(select([table.c.id]).where(table.c.cset == csid)).scalar():
            return

        # rebuild the top-level chunk
        snap = self.build_upto(
            qfilter=[lambda cset, _t: cset.c.id < csid]
        )
        sql = table.update().where(
            table.c.cset == csid
        ).values(
            chunk=self._serialize(snap)
        )
        self.cn.execute(sql)
