from datetime import datetime
from contextlib import contextmanager
import logging

import pandas as pd

from sqlalchemy import Table, Column, Integer, ForeignKey
from sqlalchemy.sql.expression import select, func, desc
from sqlalchemy.dialects.postgresql import BYTEA

from tshistory.schema import tsschema
from tshistory.util import (
    inject_in_index,
    mindate,
    maxdate,
    num2float,
    subset,
    SeriesServices,
    tzaware_serie
)
from tshistory.snapshot import Snapshot

L = logging.getLogger('tshistory.tsio')
TABLES = {}


class TimeSerie(SeriesServices):
    _csid = None
    namespace = 'tsh'
    schema = None

    def __init__(self, namespace='tsh'):
        self.namespace = namespace
        self.schema = tsschema(namespace)
        self.schema.define()
        self.metadatacache = {}

    # API : changeset, insert, get, delete
    @contextmanager
    def newchangeset(self, cn, author, _insertion_date=None):
        """A context manager to allow insertion of several series within the
        same changeset identifier

        This allows to group changes to several series, hence
        producing a macro-change.

        _insertion_date is *only* provided for migration purposes and
        not part of the API.
        """
        assert self._csid is None
        self._csid = self._newchangeset(cn, author, _insertion_date)
        self._author = author
        yield
        del self._csid
        del self._author

    def insert(self, cn, newts, name, author=None, _insertion_date=None):
        """Create a new revision of a given time series

        newts: pandas.Series with date index

        name: str unique identifier of the serie

        author: str free-form author name (mandatory, unless provided
        to the newchangeset context manager).

        """
        assert self._csid or author, 'author is mandatory'
        if self._csid and author:
            L.info('author r{} will not be used when in a changeset'.format(author))
            author = None
        assert isinstance(newts, pd.Series)
        assert not newts.index.duplicated().any()

        newts = num2float(newts)

        if not len(newts):
            return

        assert ('<M8[ns]' == newts.index.dtype or
                'datetime' in str(newts.index.dtype) or
                isinstance(newts.index, pd.MultiIndex))

        newts.name = name
        table = self._get_ts_table(cn, name)

        if isinstance(newts.index, pd.MultiIndex):
            # we impose an order to survive rountrips
            newts = newts.reorder_levels(sorted(newts.index.names))

        if table is None:
            return self._create(cn, newts, name, author, _insertion_date)

        return self._update(cn, table, newts, name, author, _insertion_date)

    def get(self, cn, name, revision_date=None,
            from_value_date=None, to_value_date=None):
        """Compute and return the serie of a given name

        revision_date: datetime filter to get previous versions of the
        serie

        """
        table = self._get_ts_table(cn, name)
        if table is None:
            return

        qfilter = []
        if revision_date:
            qfilter.append(lambda cset, _: cset.c.insertion_date <= revision_date)
        snap = Snapshot(cn, self, name)
        current = snap.build_upto(qfilter,
                                  from_value_date=from_value_date,
                                  to_value_date=to_value_date)

        if current is not None:
            current.name = name
            current = current[~current.isnull()]
        return current

    def metadata(self, cn, tsname):
        """Return metadata dict of timeserie."""
        if tsname in self.metadatacache:
            return self.metadatacache[tsname]
        reg = self.schema.registry
        sql = select([reg.c.metadata]).where(
            reg.c.name == tsname
        )
        meta = cn.execute(sql).scalar()
        self.metadatacache[tsname] = meta
        return meta

    def get_group(self, cn, name, revision_date=None):
        csid = self._latest_csid_for(cn, name)

        group = {}
        for seriename in self._changeset_series(cn, csid):
            serie = self.get(cn, seriename, revision_date)
            if serie is not None:
                group[seriename] = serie
        return group

    def get_history(self, cn, name,
                    from_insertion_date=None,
                    to_insertion_date=None,
                    from_value_date=None,
                    to_value_date=None,
                    diffmode=False):
        table = self._get_ts_table(cn, name)
        if table is None:
            return

        # compute diffs above the snapshot
        cset = self.schema.changeset
        diffsql = select([cset.c.id, cset.c.insertion_date, table.c.diff]
        ).order_by(cset.c.id
        ).where(table.c.cset == cset.c.id)

        if from_insertion_date:
            diffsql = diffsql.where(cset.c.insertion_date >= from_insertion_date)
        if to_insertion_date:
            diffsql = diffsql.where(cset.c.insertion_date <= to_insertion_date)

        diffs = cn.execute(diffsql).fetchall()
        if not diffs:
            # it's fine to ask for an insertion date range
            # where noting did happen, but you get nothing
            return

        if diffmode:
            snapshot = Snapshot(cn, self, name)
            series = []
            for csid, revdate, diff in diffs:
                if diff is None:  # we must fetch the initial snapshot
                    serie = subset(snapshot.first, from_value_date, to_value_date)
                else:
                    serie = subset(self._deserialize(diff, name), from_value_date, to_value_date)
                    serie = self._ensure_tz_consistency(cn, serie)
                inject_in_index(serie, revdate)
                series.append(serie)
            series = pd.concat(series)
            series.name = name
            return series

        csid, revdate, diff_ = diffs[0]
        snap = Snapshot(cn, self, name)
        snapshot = snap.build_upto([lambda cset, _: cset.c.id <= csid],
                                   from_value_date, to_value_date)

        series = [(revdate, subset(snapshot, from_value_date, to_value_date))]
        for csid_, revdate, diff in diffs[1:]:
            diff = subset(self._deserialize(diff, table.name),
                          from_value_date, to_value_date)
            diff = self._ensure_tz_consistency(cn, diff)

            serie = self.patch(series[-1][1], diff)
            series.append((revdate, serie))

        for revdate, serie in series:
            inject_in_index(serie, revdate)

        serie = pd.concat([serie for revdate_, serie in series])
        serie.name = name
        return serie

    def exists(self, cn, name):
        return self._get_ts_table(cn, name) is not None

    def latest_insertion_date(self, cn, name):
        cset = self.schema.changeset
        tstable = self._get_ts_table(cn, name)
        sql = select([func.max(cset.c.insertion_date)]
        ).where(tstable.c.cset == cset.c.id)
        return cn.execute(sql).scalar()

    def changeset_at(self, cn, seriename, revdate, mode='strict'):
        assert mode in ('strict', 'before', 'after')
        cset = self.schema.changeset
        table = self._table_definition_for(seriename)
        sql = select([table.c.cset]).where(
            table.c.cset == cset.c.id
        )
        if mode == 'strict':
            sql = sql.where(cset.c.insertion_date == revdate)
        elif mode == 'before':
            sql = sql.where(cset.c.insertion_date <= revdate)
        else:
            sql = sql.where(cset.c.insertion_date >= revdate)
        return cn.execute(sql).scalar()

    def strip(self, cn, seriename, csid):
        logs = self.log(cn, fromrev=csid, names=(seriename,))
        assert logs

        # put stripping info in the metadata
        cset = self.schema.changeset
        cset_serie = self.schema.changeset_series
        for log in logs:
            # update changeset.metadata
            metadata = cn.execute(
                select([cset.c.metadata]).where(cset.c.id == log['rev'])
            ).scalar() or {}
            metadata['tshistory.info'] = 'got stripped from {}'.format(csid)
            sql = cset.update().where(cset.c.id == log['rev']
            ).values(metadata=metadata)
            cn.execute(sql)
            # delete changset_serie item
            sql = cset_serie.delete().where(
                cset_serie.c.cset == log['rev']
            ).where(
                cset_serie.c.serie == seriename
            )
            cn.execute(sql)

        # wipe the diffs
        table = self._table_definition_for(seriename)
        cn.execute(table.delete().where(table.c.cset >= csid))

    def info(self, cn):
        """Gather global statistics on the current tshistory repository
        """
        sql = 'select count(*) from {}.registry'.format(self.namespace)
        stats = {'series count': cn.execute(sql).scalar()}
        sql = 'select max(id) from {}.changeset'.format(self.namespace)
        stats['changeset count'] = cn.execute(sql).scalar()
        sql = 'select distinct name from {}.registry order by name'.format(self.namespace)
        stats['serie names'] = [row for row, in cn.execute(sql).fetchall()]
        return stats

    def log(self, cn, limit=0, diff=False, names=None, authors=None,
            stripped=False,
            fromrev=None, torev=None,
            fromdate=None, todate=None):
        """Build a structure showing the history of all the series in the db,
        per changeset, in chronological order.
        """
        log = []
        cset, cset_series, reg = (
            self.schema.changeset,
            self.schema.changeset_series,
            self.schema.registry
        )

        sql = select([cset.c.id, cset.c.author, cset.c.insertion_date, cset.c.metadata]
        ).distinct().order_by(desc(cset.c.id))

        if limit:
            sql = sql.limit(limit)

        if names:
            sql = sql.where(reg.c.name.in_(names))

        if authors:
            sql = sql.where(cset.c.author.in_(authors))

        if fromrev:
            sql = sql.where(cset.c.id >= fromrev)

        if torev:
            sql = sql.where(cset.c.id <= torev)

        if fromdate:
            sql = sql.where(cset.c.insertion_date >= fromdate)

        if todate:
            sql = sql.where(cset.c.insertion_date <= todate)

        if stripped:
            # outerjoin to show dead things
            sql = sql.select_from(cset.outerjoin(cset_series))
        else:
            sql = sql.where(cset.c.id == cset_series.c.cset
            ).where(cset_series.c.serie == reg.c.name)

        rset = cn.execute(sql)
        for csetid, author, revdate, meta in rset.fetchall():
            log.append({'rev': csetid, 'author': author,
                        'date': pd.Timestamp(revdate, tz='utc'),
                        'meta': meta or {},
                        'names': self._changeset_series(cn, csetid)})

        if diff:
            for rev in log:
                rev['diff'] = {name: self.diff_at(cn, rev['rev'], name)
                               for name in rev['names']}

        log.sort(key=lambda rev: rev['rev'])
        return log

    # /API
    # Helpers

    # creation / update

    def _create(self, cn, newts, name, author, insertion_date=None):
        # initial insertion
        newts = newts[~newts.isnull()]
        if len(newts) == 0:
            return None
        snapshot = Snapshot(cn, self, name)
        csid = self._csid or self._newchangeset(cn, author, insertion_date)
        head = snapshot.create(newts)
        value = {
            'cset': csid,
            'snapshot': head
        }
        table = self._make_ts_table(cn, name, newts)
        cn.execute(table.insert().values(value))
        self._finalize_insertion(cn, csid, name)
        L.info('first insertion of %s (size=%s) by %s',
               name, len(newts), author or self._author)
        return newts

    def _update(self, cn, table, newts, name, author, insertion_date=None):
        self._validate(cn, newts, name)
        snapshot = Snapshot(cn, self, name)
        diff = self.diff(snapshot.last(mindate(newts), maxdate(newts)), newts)
        if not len(diff):
            L.info('no difference in %s by %s (for ts of size %s)',
                   name, author or self._author, len(newts))
            return

        csid = self._csid or self._newchangeset(cn, author, insertion_date)
        head = snapshot.update(diff)
        value = {
            'cset': csid,
            'diff': self._serialize(diff),
            'snapshot': head
        }
        cn.execute(table.insert().values(value))
        self._finalize_insertion(cn, csid, name)

        L.info('inserted diff (size=%s) for ts %s by %s',
               len(diff), name, author or self._author)
        return diff

    # ts serialisation

    def _ensure_tz_consistency(self, cn, ts):
        """Return timeserie with tz aware index or not depending on metadata
        tzaware.
        """
        assert ts.name is not None
        metadata = self.metadata(cn, ts.name)
        if metadata and metadata.get('tzaware', False):
            if isinstance(ts.index, pd.MultiIndex):
                for i in range(len(ts.index.levels)):
                    ts.index = ts.index.set_levels(
                        ts.index.levels[i].tz_localize('UTC'),
                        level=i)
                return ts
            return ts.tz_localize('UTC')
        return ts

    # serie table handling

    def _ts_table_name(self, seriename):
        # namespace.seriename
        return '{}.timeserie.{}'.format(self.namespace, seriename)

    def _table_definition_for(self, seriename):
        tablename = self._ts_table_name(seriename)
        table = TABLES.get(tablename)
        if table is None:
            TABLES[tablename] = table = Table(
                seriename, self.schema.meta,
                Column('id', Integer, primary_key=True),
                Column('cset', Integer,
                       ForeignKey('{}.changeset.id'.format(self.namespace)),
                       index=True, nullable=False),
                Column('diff', BYTEA),
                Column('snapshot', Integer,
                       ForeignKey('{}.snapshot.{}.id'.format(
                           self.namespace,
                           seriename)),
                       index=True),
                schema='{}.timeserie'.format(self.namespace),
                extend_existing=True
            )
        return table

    def _make_ts_table(self, cn, name, ts):
        tablename = self._ts_table_name(name)
        table = self._table_definition_for(name)
        table.create(cn)
        index = ts.index
        inames = [name for name in index.names if name]
        sql = self.schema.registry.insert().values(
            name=name,
            table_name=tablename,
            metadata={
                'tzaware': tzaware_serie(ts),
                'index_type': index.dtype.name,
                'index_names': inames,
                'value_type': ts.dtypes.name
            },
        )
        cn.execute(sql)
        return table

    def _get_ts_table(self, cn, name):
        reg = self.schema.registry
        tablename = self._ts_table_name(name)
        sql = reg.select().where(reg.c.table_name == tablename)
        tid = cn.execute(sql).scalar()
        if tid:
            return self._table_definition_for(name)

    # changeset handling

    def _newchangeset(self, cn, author, _insertion_date=None):
        table = self.schema.changeset
        if _insertion_date is not None:
            assert _insertion_date.tzinfo is not None
        idate = pd.Timestamp(_insertion_date or datetime.utcnow(), tz='UTC')
        sql = table.insert().values(
            author=author,
            insertion_date=idate)
        return cn.execute(sql).inserted_primary_key[0]

    def _latest_csid_for(self, cn, name):
        table = self._get_ts_table(cn, name)
        sql = select([func.max(table.c.cset)])
        return cn.execute(sql).scalar()

    def _changeset_series(self, cn, csid):
        cset_serie = self.schema.changeset_series
        sql = select([cset_serie.c.serie]
        ).where(cset_serie.c.cset == csid)

        return [seriename for seriename, in cn.execute(sql).fetchall()]

    # insertion handling

    def _validate(self, cn, ts, name):
        if ts.isnull().all():
            # ts erasure
            return
        tstype = ts.dtype
        meta = self.metadata(cn, name)
        if tstype != meta['value_type']:
            m = 'Type error when inserting {}, new type is {}, type in base is {}'.format(
                name, tstype, meta['value_type'])
            raise Exception(m)
        if ts.index.dtype.name != meta['index_type']:
            raise Exception('Incompatible index types')
        inames = [name for name in ts.index.names if name]
        if inames != meta['index_names']:
            raise Exception('Incompatible multi indexes: {} vs {}'.format(
                meta['index_names'], inames)
            )

    def _finalize_insertion(self, cn, csid, name):
        table = self.schema.changeset_series
        sql = table.insert().values(
            cset=csid,
            serie=name
        )
        cn.execute(sql)

    def diff_at(self, cn, csetid, name):
        table = self._get_ts_table(cn, name)
        cset = self.schema.changeset

        def filtercset(sql):
            return sql.where(table.c.cset == cset.c.id
            ).where(cset.c.id == csetid)

        sql = filtercset(select([table.c.id]))
        tsid = cn.execute(sql).scalar()

        if tsid == 1:
            return Snapshot(cn, self, name).first

        sql = filtercset(select([table.c.diff]))
        ts = self._deserialize(cn.execute(sql).scalar(), name)
        return self._ensure_tz_consistency(cn, ts)
