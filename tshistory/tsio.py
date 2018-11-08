from datetime import datetime
import logging

import pandas as pd

from sqlalchemy import Table, Column, Integer, ForeignKey, Index
from sqlalchemy.sql.elements import NONE_NAME
from sqlalchemy.engine.base import Engine
from sqlalchemy.sql.expression import select, func, desc
from sqlalchemy.dialects.postgresql import TIMESTAMP

from tshistory.schema import tsschema
from tshistory.util import (
    closed_overlaps,
    num2float,
    subset,
    SeriesServices,
    start_end,
    tzaware_serie
)
from tshistory.snapshot import Snapshot, TABLES as SNAPTABLES

L = logging.getLogger('tshistory.tsio')
TABLES = {}


class TimeSerie(SeriesServices):
    namespace = 'tsh'
    schema = None
    metadatacache = None
    metakeys = {
        'tzaware',
        'index_type',
        'index_names',
        'index_dtype',
        'value_dtype',
        'value_type'
    }
    registry_map = None
    serie_tablename = None
    create_lock_id = None

    def __init__(self, namespace='tsh'):
        self.namespace = namespace
        self.schema = tsschema(namespace)
        self.schema.define()
        self.metadatacache = {}
        self.registry_map = {}
        self.serie_tablename = {}
        self.create_lock_id = sum(ord(c) for c in namespace)

    def _check_tx(self, cn):
        # safety belt to make sure important api points are tx-safe
        if isinstance(cn, Engine) or not cn.in_transaction():
            if not getattr(self, '_testing', False):
                raise TypeError('You must use a transaction object')

    def insert(self, cn, newts, seriename, author,
               metadata=None,
               _insertion_date=None):
        """Create a new revision of a given time series

        newts: pandas.Series with date index
        seriename: str unique identifier of the serie
        author: str free-form author name
        metadata: optional dict for changeset metadata
        """
        self._check_tx(cn)
        assert isinstance(newts, pd.Series), 'Not a pd.Series'
        assert isinstance(seriename, str), 'Name not a string'
        assert isinstance(author, str), 'Author not a string'
        assert metadata is None or isinstance(metadata, dict), 'Bad format for metadata'
        assert (_insertion_date is None or
                isinstance(_insertion_date, datetime)), 'Bad format for insertion date'
        assert not newts.index.duplicated().any(), 'There are some duplicates in the index'

        assert newts.index.notna().all(), 'The index contains NaT entries'
        if not newts.index.is_monotonic_increasing:
            newts = newts.sort_index()

        newts = num2float(newts)

        if not len(newts):
            return

        assert ('<M8[ns]' == newts.index.dtype or
                'datetime' in str(newts.index.dtype) and not
                isinstance(newts.index, pd.MultiIndex))

        newts.name = seriename
        table = self._get_ts_table(cn, seriename)

        if table is None:
            return self._create(cn, newts, seriename, author,
                                metadata, _insertion_date)

        return self._update(cn, table, newts, seriename, author,
                            metadata, _insertion_date)

    def get(self, cn, seriename, revision_date=None,
            from_value_date=None, to_value_date=None,
            _keep_nans=False):
        """Compute and return the serie of a given name

        revision_date: datetime filter to get previous versions of the
        serie

        """
        table = self._get_ts_table(cn, seriename)
        if table is None:
            return

        csetfilter = []
        if revision_date:
            csetfilter.append(lambda cset: cset.c.insertion_date <= revision_date)
        snap = Snapshot(cn, self, seriename)
        _, current = snap.find(csetfilter=csetfilter,
                               from_value_date=from_value_date,
                               to_value_date=to_value_date)

        if current is not None and not _keep_nans:
            current.name = seriename
            current = current[~current.isnull()]
        return current

    def metadata(self, cn, seriename):
        """Return metadata dict of timeserie."""
        if seriename in self.metadatacache:
            return self.metadatacache[seriename]
        reg = self.schema.registry
        sql = select([reg.c.metadata]).where(
            reg.c.seriename == seriename
        )
        meta = cn.execute(sql).scalar()
        self.metadatacache[seriename] = meta
        return meta

    def update_metadata(self, cn, seriename, metadata):
        self._check_tx(cn)
        assert isinstance(metadata, dict)
        assert not set(metadata.keys()) & self.metakeys
        meta = self.metadata(cn, seriename)
        # remove al but internal stuff
        newmeta = {key: meta[key] for key in self.metakeys}
        newmeta.update(metadata)
        reg = self.schema.registry
        sql = reg.update().where(
            reg.c.seriename == seriename
        ).values(metadata=newmeta)
        self.metadatacache.pop(seriename)
        cn.execute(sql)

    def changeset_metadata(self, cn, csid):
        sql = 'select metadata from "{ns}".changeset where id = {id}'.format(
            ns=self.namespace,
            id=csid
        )
        return cn.execute(sql).scalar()

    def get_history(self, cn, seriename,
                    from_insertion_date=None,
                    to_insertion_date=None,
                    from_value_date=None,
                    to_value_date=None,
                    deltabefore=None,
                    deltaafter=None,
                    diffmode=False):
        table = self._get_ts_table(cn, seriename)
        if table is None:
            return

        cset = self.schema.changeset
        revsql = select(
            [cset.c.id, cset.c.insertion_date]
        ).order_by(
            cset.c.id
        ).where(
            table.c.cset == cset.c.id
        )

        if from_insertion_date:
            revsql = revsql.where(cset.c.insertion_date >= from_insertion_date)
        if to_insertion_date:
            revsql = revsql.where(cset.c.insertion_date <= to_insertion_date)

        if from_value_date or to_value_date:
            revsql = revsql.where(
                closed_overlaps(from_value_date, to_value_date)
            )

        revs = cn.execute(
            revsql,
            {'fromdate': from_value_date, 'todate': to_value_date}
        ).fetchall()
        if not revs:
            return {}

        if diffmode:
            # compute the previous serie value
            first_csid = revs[0][0]
            previous_csid = self._previous_cset(cn, seriename, first_csid)
            revs.insert(0, (previous_csid, None))

        snapshot = Snapshot(cn, self, seriename)
        series = []
        if (deltabefore, deltaafter) != (None, None):
            for csid, idate in revs:
                from_date = None
                to_date = None
                if deltabefore is not None:
                    from_date = idate - deltabefore
                if deltaafter is not None:
                    to_date = idate + deltaafter
                series.append((
                    idate,
                    snapshot.find(csetfilter=[lambda cset: cset.c.id == csid],
                                  from_value_date=from_date,
                                  to_value_date=to_date)[1]
                ))
        else:
            series = snapshot.findall(revs,
                                      from_value_date,
                                      to_value_date
            )

        if diffmode:
            diffs = []
            for (_revdate_a, serie_a), (revdate_b, serie_b) in zip(series, series[1:]):
                if serie_a is None:
                    # when we scan the entirety of the history: there exists no "previous" serie
                    # we therefore consider the first serie as a diff to the "null" serie
                    diffs.append((revdate_b, serie_b))
                else:
                    diffs.append((revdate_b, self.diff(serie_a, serie_b)))
            series = diffs
        else:
            series = [
                (idate, ts.dropna())
                 for idate, ts in series
            ]

        return {
            idate: serie
            for idate, serie in series
        }

    def get_delta(self, cn, seriename, delta,
                  from_value_date=None,
                  to_value_date=None):
        """ compute a series whose value dates are bounded to be
        `delta` time after the insertion dates and where we
        keep the most recent ones
        """
        histo = self.get_history(
            cn, seriename, deltabefore=-delta,
            from_value_date=from_value_date,
            to_value_date=to_value_date
        )
        if histo is None:
            return None

        vimap = {}
        vvmap = {}
        for idate, series in histo.items():
            for vdate, value in series.iteritems():
                if vdate not in vimap or vimap[vdate] < idate:
                    vimap[vdate] = idate
                    vvmap[vdate] = value

        ts = subset(pd.Series(vvmap).sort_index(), from_value_date, to_value_date)
        ts.name = seriename
        return ts

    def exists(self, cn, seriename):
        return self._get_ts_table(cn, seriename) is not None

    def latest_insertion_date(self, cn, seriename):
        cset = self.schema.changeset
        tstable = self._get_ts_table(cn, seriename)
        sql = select([func.max(cset.c.insertion_date)]
        ).where(tstable.c.cset == cset.c.id)
        return cn.execute(sql).scalar()

    def last_id(self, cn, seriename):
        snapshot = Snapshot(cn, self, seriename)
        return snapshot.last_id()

    def changeset_at(self, cn, seriename, revdate, mode='strict'):
        assert mode in ('strict', 'before', 'after')
        cset = self.schema.changeset
        table = self._table_definition_for(cn, seriename)
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

    def delete(self, cn, seriename):
        self._check_tx(cn)
        assert not isinstance(cn, Engine), 'use a transaction object'
        assert self.exists(cn, seriename)
        # changeset will keep ghost entries
        # we cleanup changeset series, then registry
        # then we drop the two remaining tables
        # cn *must* be a transaction scope
        rid, tablename = cn.execute(
            'select id, table_name from "{}".registry '
            'where seriename = %(seriename)s'.format(self.namespace),
            seriename=seriename
        ).fetchone()
        # drop series tables
        cn.execute(
            'drop table "{}.timeserie"."{}" cascade'.format(self.namespace, tablename)
        )
        cn.execute(
            'drop table "{}.snapshot"."{}" cascade'.format(self.namespace, tablename)
        )
        # cleanup changesets table
        cn.execute('with csets as ('
                   ' select cset from "{ns}".changeset_series '
                   ' where serie = %(rid)s'
                   ') '
                   'delete from "{ns}".changeset as cset using csets '
                   'where cset.id = csets.cset'.format(ns=self.namespace),
                   rid=rid
        )
        cn.execute('delete from "{}".registry '
                   'where id = %(rid)s'.format(self.namespace),
                   rid=rid)
        # -> this will transitively cleanup state changeset_series entries
        self._resetcaches()

    def strip(self, cn, seriename, csid):
        self._check_tx(cn)
        logs = self.log(cn, fromrev=csid, names=(seriename,))
        assert logs

        # put stripping info in the metadata
        cset = self.schema.changeset
        cset_serie = self.schema.changeset_series
        for log in logs:
            # update changeset.metadata
            metadata = self.changeset_metadata(cn, log['rev']) or {}
            metadata['tshistory.info'] = 'got stripped from {}'.format(csid)
            sql = cset.update().where(cset.c.id == log['rev']
            ).values(metadata=metadata)
            cn.execute(sql)
            # delete changset_serie item
            sql = cset_serie.delete().where(
                cset_serie.c.cset == log['rev']
            ).where(
                cset_serie.c.serie == self._name_to_regid(cn, seriename)
            )
            cn.execute(sql)

        # wipe the diffs
        table = self._table_definition_for(cn, seriename)
        cn.execute(table.delete().where(table.c.cset >= csid))

    def info(self, cn):
        """Gather global statistics on the current tshistory repository
        """
        sql = 'select count(*) from "{}".registry'.format(self.namespace)
        stats = {'series count': cn.execute(sql).scalar()}
        sql = 'select max(id) from "{}".changeset'.format(self.namespace)
        stats['changeset count'] = cn.execute(sql).scalar()
        sql = 'select distinct seriename from "{}".registry order by seriename'.format(
            self.namespace
        )
        stats['serie names'] = [row for row, in cn.execute(sql).fetchall()]
        return stats

    def log(self, cn, limit=0, names=None, authors=None,
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
            sql = sql.where(reg.c.seriename.in_(names))
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
            ).where(cset_series.c.serie == reg.c.id)

        rset = cn.execute(sql)
        for csetid, author, revdate, meta in rset.fetchall():
            log.append({'rev': csetid, 'author': author,
                        'date': pd.Timestamp(revdate, tz='utc'),
                        'meta': meta or {},
                        'names': self._changeset_series(cn, csetid)})

        log.sort(key=lambda rev: rev['rev'])
        return log

    def interval(self, cn, seriename, notz=False):
        tablename = self._serie_to_tablename(cn, seriename)
        if tablename is None:
            raise ValueError(f'no such serie: {seriename}')
        sql = (f'select start, "end" '
               f'from "{self.namespace}.timeserie"."{tablename}" '
               f'order by cset desc limit 1')
        res = cn.execute(sql).fetchone()
        start, end = res.start, res.end
        if self.metadata(cn, seriename).get('tzaware') and not notz:
            start, end = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
        return pd.Interval(left=start, right=end, closed='both')

    # /API
    # Helpers

    # creation / update

    def _create(self, cn, newts, seriename, author,
                metadata=None, insertion_date=None):
        start, end = start_end(newts, notz=False)
        if start is None:
            assert end is None
            # this is just full of nans
            return None
        # chop off unwanted nans
        newts = newts.loc[start:end]
        if len(newts) == 0:
            return None

        # at creation time we take an exclusive lock to avoid
        # a deadlock on created tables against the changeset-series fk
        cn.execute(
            'select pg_advisory_xact_lock({})'.format(self.create_lock_id)
        )
        self._register_serie(cn, seriename, newts)
        snapshot = Snapshot(cn, self, seriename)
        csid = self._newchangeset(cn, author, insertion_date, metadata)
        head = snapshot.create(newts)
        start, end = start_end(newts)
        value = {
            'cset': csid,
            'snapshot': head,
            'start': start,
            'end': end
        }
        table = self._make_ts_table(cn, seriename)
        cn.execute(table.insert().values(value))
        self._finalize_insertion(cn, csid, seriename)
        L.info('first insertion of %s (size=%s) by %s',
               seriename, len(newts), author)
        return newts

    def _update(self, cn, table, newts, seriename, author,
                metadata=None, insertion_date=None):
        self._validate(cn, newts, seriename)
        snapshot = Snapshot(cn, self, seriename)
        diff = self.diff(snapshot.last(newts.index.min(),
                                       newts.index.max()),
                         newts)
        if not len(diff):
            L.info('no difference in %s by %s (for ts of size %s)',
                   seriename, author, len(newts))
            return

        # compute series start/end stamps
        tsstart, tsend = start_end(newts)
        ival = self.interval(cn, seriename, notz=True)
        start = min(tsstart or ival.left, ival.left)
        end = max(tsend or ival.right, ival.right)

        if pd.isnull(diff[0]) or pd.isnull(diff[-1]):
            # we *might* be shrinking, let's look at the full series
            # and yes, shrinkers have a slow path
            last = snapshot.last()
            patched = self.patch(last, diff).dropna()
            if not len(patched):
                raise ValueError('complete erasure of a series is forbidden')
            if pd.isnull(diff[0]):
                start = patched.index[0]
            if pd.isnull(diff[-1]):
                end = patched.index[-1]

        csid = self._newchangeset(cn, author, insertion_date, metadata)
        head = snapshot.update(diff)
        value = {
            'cset': csid,
            'snapshot': head,
            'start': start,
            'end': end
        }
        cn.execute(table.insert().values(value))
        self._finalize_insertion(cn, csid, seriename)

        L.info('inserted diff (size=%s) for ts %s by %s',
               len(diff), seriename, author)
        return diff

    # serie table handling

    def _serie_to_tablename(self, cn, seriename):
        tablename = self.serie_tablename.get(seriename)
        if tablename is not None:
            return tablename

        reg = self.schema.registry
        sql = select([reg.c.table_name]).where(reg.c.seriename == seriename)
        tablename = cn.execute(sql).scalar()
        if tablename is None:
            # creation time
            return
        self.serie_tablename[seriename] = tablename
        return tablename

    def _table_definition_for(self, cn, seriename):
        tablename = self._serie_to_tablename(cn, seriename)
        if tablename is None:
            # creation time
            tablename = self._make_tablename(seriename)
        fq_tablename = '{}.timeserie.{}'.format(self.namespace, tablename)
        table = TABLES.get(fq_tablename)
        if table is None:
            TABLES[fq_tablename] = table = Table(
                tablename, self.schema.meta,
                Column('id', Integer, primary_key=True),
                Column('cset', Integer,
                       ForeignKey('{}.changeset.id'.format(self.namespace)),
                       nullable=False),
                Column('start', TIMESTAMP, nullable=False),
                Column('end', TIMESTAMP, nullable=False),
                Column('snapshot', Integer,
                       ForeignKey('{}.snapshot.{}.id'.format(
                           self.namespace,
                           tablename))),
                Index(NONE_NAME, 'cset'),
                Index(NONE_NAME, 'snapshot'),
                Index(NONE_NAME, 'start'),
                Index(NONE_NAME, 'end'),
                schema='{}.timeserie'.format(self.namespace),
                keep_existing=True
            )
        return table

    def _make_ts_table(self, cn, seriename):
        table = self._table_definition_for(cn, seriename)
        table.create(cn)
        return table

    def _register_serie(self, cn, seriename, ts):
        index = ts.index
        inames = [name for name in index.names if name]
        sql = self.schema.registry.insert().values(
            seriename=seriename,
            table_name=self._make_tablename(seriename),
            metadata={
                'tzaware': tzaware_serie(ts),
                'index_type': index.dtype.name,
                'index_names': inames,
                'index_dtype': index.dtype.str,
                'value_dtype': ts.dtypes.str,
                'value_type': ts.dtypes.name
            }
        )
        regid = cn.execute(sql).inserted_primary_key[0]
        self.registry_map[seriename] = regid

    def _get_ts_table(self, cn, seriename):
        tablename = self._serie_to_tablename(cn, seriename)
        if tablename:
            return self._table_definition_for(cn, tablename)

    # changeset handling

    def _newchangeset(self, cn, author, insertion_date=None, metadata=None):
        table = self.schema.changeset
        if insertion_date is not None:
            assert insertion_date.tzinfo is not None
        idate = pd.Timestamp(insertion_date or datetime.utcnow(), tz='UTC')
        sql = table.insert().values(
            author=author,
            metadata=metadata,
            insertion_date=idate)
        return cn.execute(sql).inserted_primary_key[0]

    def _changeset_series(self, cn, csid):
        cset_serie = self.schema.changeset_series
        reg = self.schema.registry
        sql = select(
            [reg.c.seriename]
        ).where(cset_serie.c.cset == csid
        ).where(cset_serie.c.serie == reg.c.id)

        return [
            row.seriename
            for row in cn.execute(sql).fetchall()
        ]

    def _previous_cset(self, cn, seriename, csid):
        tablename = self._serie_to_tablename(cn, seriename)
        sql = ('select cset from "{}.timeserie"."{}" '
               'where cset < %(csid)s '
               'order by cset desc limit 1').format(self.namespace, tablename)
        return cn.execute(sql, csid=csid).scalar()

    # insertion handling

    def _validate(self, cn, ts, seriename):
        if ts.isnull().all():
            # ts erasure
            return
        tstype = ts.dtype
        meta = self.metadata(cn, seriename)
        if tstype != meta['value_type']:
            m = 'Type error when inserting {}, new type is {}, type in base is {}'.format(
                seriename, tstype, meta['value_type'])
            raise Exception(m)
        if ts.index.dtype.name != meta['index_type']:
            raise Exception('Incompatible index types')

    def _name_to_regid(self, cn, seriename):
        regid = self.registry_map.get(seriename)
        if regid is not None:
            return regid

        registry = self.schema.registry
        sql = select([registry.c.id]).where(registry.c.seriename == seriename)
        regid = self.registry_map[seriename] = cn.execute(sql).scalar()
        return regid

    def _finalize_insertion(self, cn, csid, seriename):
        table = self.schema.changeset_series
        sql = table.insert().values(
            cset=csid,
            serie=self._name_to_regid(cn, seriename)
        )
        cn.execute(sql)

    def _resetcaches(self):
        TABLES.clear()
        SNAPTABLES.clear()
        self.metadatacache.clear()
        self.registry_map.clear()
        self.serie_tablename.clear()
