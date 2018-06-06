from datetime import datetime
from contextlib import contextmanager
import logging
import hashlib

import pandas as pd

from sqlalchemy import Table, Column, Integer, ForeignKey
from sqlalchemy.sql.expression import select, func, desc
from sqlalchemy.dialects.postgresql import BYTEA

from tshistory.schema import tsschema
from tshistory.util import (
    inject_in_index,
    num2float,
    subset,
    SeriesServices,
    tzaware_serie
)
from tshistory.snapshot import Snapshot

L = logging.getLogger('tshistory.tsio')
TABLES = {}


class TimeSerie(SeriesServices):
    namespace = 'tsh'
    schema = None

    def __init__(self, namespace='tsh'):
        self.namespace = namespace
        self.schema = tsschema(namespace)
        self.schema.define()
        self.metadatacache = {}

    def insert(self, cn, newts, name, author,
               metadata=None,
               _insertion_date=None):
        """Create a new revision of a given time series

        newts: pandas.Series with date index
        name: str unique identifier of the serie
        author: str free-form author name
        metadata: optional dict for changeset metadata
        """
        assert isinstance(newts, pd.Series)
        assert isinstance(name, str)
        assert isinstance(author, str)
        assert metadata is None or isinstance(metadata, dict)
        assert _insertion_date is None or isinstance(_insertion_date, datetime)
        assert not newts.index.duplicated().any()

        newts = num2float(newts)

        if not len(newts):
            return

        assert ('<M8[ns]' == newts.index.dtype or
                'datetime' in str(newts.index.dtype) and not
                isinstance(newts.index, pd.MultiIndex))

        newts.name = name
        table = self._get_ts_table(cn, name)

        if table is None:
            return self._create(cn, newts, name, author,
                                metadata, _insertion_date)

        return self._update(cn, table, newts, name, author,
                            metadata, _insertion_date)

    def get(self, cn, name, revision_date=None,
            from_value_date=None, to_value_date=None,
            _keep_nans=False):
        """Compute and return the serie of a given name

        revision_date: datetime filter to get previous versions of the
        serie

        """
        table = self._get_ts_table(cn, name)
        if table is None:
            return

        csetfilter = []
        if revision_date:
            csetfilter.append(lambda cset: cset.c.insertion_date <= revision_date)
        snap = Snapshot(cn, self, name)
        _, current = snap.find(csetfilter=csetfilter,
                               from_value_date=from_value_date,
                               to_value_date=to_value_date)

        if current is not None and not _keep_nans:
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

    def update_metadata(self, cn, tsname, metadata, internal=False):
        assert isinstance(metadata, dict)
        meta = self.metadata(cn, tsname)
        if not internal:
            assert set(meta.keys()).intersection(metadata.keys()) == set()
        meta.update(metadata)
        reg = self.schema.registry
        sql = reg.update().where(
            reg.c.name == tsname
        ).values(metadata=metadata)
        cn.execute(sql)

    def changeset_metadata(self, cn, csid):
        cset = self.schema.changeset
        sql = 'select metadata from "{ns}".changeset where id = {id}'.format(
            ns=self.namespace,
            id=csid
        )
        return cn.execute(sql).scalar()

    def get_history(self, cn, name,
                    from_insertion_date=None,
                    to_insertion_date=None,
                    from_value_date=None,
                    to_value_date=None,
                    deltabefore=None,
                    deltaafter=None):
        table = self._get_ts_table(cn, name)
        if table is None:
            return

        if deltabefore is not None or deltaafter is not None:
            assert from_value_date is None
            assert to_value_date is None

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

        revs = cn.execute(revsql).fetchall()
        if not revs:
            return

        snapshot = Snapshot(cn, self, name)
        series = []
        for csid, idate in revs:
            if (deltabefore, deltaafter) != (None, None):
                from_value_date = None
                to_value_date = None
                if deltabefore is not None:
                    from_value_date = idate - deltabefore
                if deltaafter is not None:
                    to_value_date = idate + deltaafter
            series.append((
                idate,
                snapshot.find(csetfilter=[lambda cset: cset.c.id == csid],
                              from_value_date=from_value_date,
                              to_value_date=to_value_date)[1]
            ))

        for revdate, serie in series:
            inject_in_index(serie, revdate)

        serie = pd.concat([serie for revdate_, serie in series])
        serie.name = name
        return serie

    def get_delta(self, cn, name, delta):
        histo = self.get_history(
            cn, name, deltabefore=-delta
        )

        df = histo.reset_index()
        # df_date is a dataframe with two columns: value_date and insertion_date
        df_date = df.loc[:, ['insertion_date', 'value_date']]

        # now in selected_dates each value_date has only one occurence
        # which is the last inserted
        selected_dates = df_date.groupby('value_date').max().reset_index()

        ts = df[name]
        # ts is built from the df returned from get_history
        # ts index is now a simple index of tuples (insert_date, value_date)
        ts.index = ((row.insertion_date, row.value_date)
                    for row in df.itertuples())
        # in ts, there ie still all the couple value_date * insertion_date
        # We now used the selected_dates to select in ts only
        # the couple (value_date, insertion_date)
        # which corresponds to the last insertion_date
        ts_select = ts[[(row[2], row[1])
                        for row in selected_dates.itertuples()]]

        # ts_select has still a simple index of tuples (value_date, insertion_date)
        new_index = (elt[1] for elt in ts_select.index)

        # we only keep the value_date information from the index
        ts_select.index = new_index
        return ts_select


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
            metadata = self.changeset_metadata(cn, log['rev']) or {}
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

        log.sort(key=lambda rev: rev['rev'])
        return log

    # /API
    # Helpers

    # creation / update

    def _create(self, cn, newts, name, author,
                metadata=None, insertion_date=None):
        # initial insertion
        if len(newts) == 0:
            return None
        snapshot = Snapshot(cn, self, name)
        csid = self._newchangeset(cn, author, insertion_date, metadata)
        head = snapshot.create(newts)
        value = {
            'cset': csid,
            'snapshot': head
        }
        table = self._make_ts_table(cn, name, newts)
        cn.execute(table.insert().values(value))
        self._finalize_insertion(cn, csid, name)
        L.info('first insertion of %s (size=%s) by %s',
               name, len(newts), author)
        return newts

    def _update(self, cn, table, newts, name, author,
                metadata=None, insertion_date=None):
        self._validate(cn, newts, name)
        snapshot = Snapshot(cn, self, name)
        diff = self.diff(snapshot.last(newts.index.min(),
                                       newts.index.max()),
                         newts)
        if not len(diff):
            L.info('no difference in %s by %s (for ts of size %s)',
                   name, author, len(newts))
            return

        csid = self._newchangeset(cn, author, insertion_date, metadata)
        head = snapshot.update(diff)
        value = {
            'cset': csid,
            'snapshot': head
        }
        cn.execute(table.insert().values(value))
        self._finalize_insertion(cn, csid, name)

        L.info('inserted diff (size=%s) for ts %s by %s',
               len(diff), name, author)
        return diff

    # ts serialisation

    def _ensure_tz_consistency(self, cn, ts):
        """Return timeserie with tz aware index or not depending on metadata
        tzaware.
        """
        assert ts.name is not None
        metadata = self.metadata(cn, ts.name)
        if metadata and metadata.get('tzaware', False):
            return ts.tz_localize('UTC')
        return ts

    # serie table handling

    def _ts_table_name(self, seriename):
        seriename = self._tablename(seriename)
        return '{}.timeserie.{}'.format(self.namespace, seriename)

    def _table_definition_for(self, seriename):
        tablename = self._ts_table_name(seriename)
        seriename = self._tablename(seriename)
        table = TABLES.get(tablename)
        if table is None:
            TABLES[tablename] = table = Table(
                seriename, self.schema.meta,
                Column('id', Integer, primary_key=True),
                Column('cset', Integer,
                       ForeignKey('{}.changeset.id'.format(self.namespace)),
                       index=True, nullable=False),
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

    def _finalize_insertion(self, cn, csid, name):
        table = self.schema.changeset_series
        sql = table.insert().values(
            cset=csid,
            serie=name
        )
        cn.execute(sql)
