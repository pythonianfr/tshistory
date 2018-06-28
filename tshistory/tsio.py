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
    metadatacache = None
    registry_map = None
    serie_tablename = None

    def __init__(self, namespace='tsh'):
        self.namespace = namespace
        self.schema = tsschema(namespace)
        self.schema.define()
        self.metadatacache = {}
        self.registry_map = {}
        self.serie_tablename = {}

    def insert(self, cn, newts, seriename, author,
               metadata=None,
               _insertion_date=None):
        """Create a new revision of a given time series

        newts: pandas.Series with date index
        seriename: str unique identifier of the serie
        author: str free-form author name
        metadata: optional dict for changeset metadata
        """
        assert isinstance(newts, pd.Series)
        assert isinstance(seriename, str)
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

    def update_metadata(self, cn, seriename, metadata, internal=False):
        assert isinstance(metadata, dict)
        meta = self.metadata(cn, seriename)
        if not internal:
            assert set(meta.keys()).intersection(metadata.keys()) == set()
        meta.update(metadata)
        reg = self.schema.registry
        sql = reg.update().where(
            reg.c.seriename == seriename
        ).values(metadata=metadata)
        cn.execute(sql)

    def changeset_metadata(self, cn, csid):
        cset = self.schema.changeset
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
                    deltaafter=None):
        table = self._get_ts_table(cn, seriename)
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

        snapshot = Snapshot(cn, self, seriename)
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
        serie.name = seriename
        return serie

    def get_delta(self, cn, seriename, delta,
                  from_value_date=None,
                  to_value_date=None):

        histo = self.get_history(
            cn, seriename, deltabefore=-delta
        )

        df = histo.reset_index()

        # df_date is a dataframe with two columns: value_date and insertion_date
        df_date = df.loc[:, ['insertion_date', 'value_date']]

        # now in selected_dates each value_date has only one occurence
        # which is the last inserted
        selected_dates = df_date.groupby('value_date').max().reset_index()

        ts = df[seriename]
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
        return subset(ts_select, from_value_date, to_value_date)


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
                cset_serie.c.serie == self._name_to_regid(cn, seriename)
            )
            cn.execute(sql)

        # wipe the diffs
        table = self._table_definition_for(cn, seriename)
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

    # /API
    # Helpers

    # creation / update

    def _create(self, cn, newts, seriename, author,
                metadata=None, insertion_date=None):
        # initial insertion
        if len(newts) == 0:
            return None
        self._register_serie(cn, seriename, newts)
        snapshot = Snapshot(cn, self, seriename)
        csid = self._newchangeset(cn, author, insertion_date, metadata)
        head = snapshot.create(newts)
        value = {
            'cset': csid,
            'snapshot': head
        }
        table = self._make_ts_table(cn, seriename, newts)
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

        csid = self._newchangeset(cn, author, insertion_date, metadata)
        head = snapshot.update(diff)
        value = {
            'cset': csid,
            'snapshot': head
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
                       index=True, nullable=False),
                Column('snapshot', Integer,
                       ForeignKey('{}.snapshot.{}.id'.format(
                           self.namespace,
                           tablename)),
                       index=True),
                schema='{}.timeserie'.format(self.namespace),
                extend_existing=True
            )
        return table

    def _make_ts_table(self, cn, seriename, ts):
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
                'value_type': ts.dtypes.name
            },
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

    # don't use this

    def resetcaches(self):
        self.metadatacache.clear()
        self.registry_map.clear()
        self.serie_tablename.clear()
