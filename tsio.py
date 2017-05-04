from datetime import datetime
from contextlib import contextmanager
import logging

import pandas as pd
import numpy as np

from sqlalchemy import Table, Column, Integer, ForeignKey
from sqlalchemy.sql.expression import select, func, desc
from sqlalchemy.dialects.postgresql import JSONB

from tshistory import schema


def setuplogging():
    logger = logging.getLogger('tshistory.tsio')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    return logger

L = setuplogging()


def tojson(ts):
    if ts is None:
        return None

    if not isinstance(ts.index, pd.MultiIndex):
        return ts.to_json(date_format='iso')

    # multi index case
    return ts.to_frame().reset_index().to_json(date_format='iso')


def fromjson(jsonb, tsname):
    result = pd.read_json(jsonb, typ='series', dtype=False)
    if isinstance(result.index, pd.DatetimeIndex):
        return result

    # multi index case
    columns = result.index.values.tolist()
    columns.remove(tsname)
    result = pd.read_json(jsonb, typ='frame',
                          convert_dates=columns)
    result.set_index(sorted(columns), inplace=True)

    return result.iloc[:,0] # get a Series object


class TimeSerie(object):
    _csid = None
    _snapshot_interval = 10
    _precision = 1e-14

    # API : changeset, insert, get, delete
    @contextmanager
    def newchangeset(self, cnx, author, _insertion_date=None):
        """A context manager to allow insertion of several series within the
        same changeset identifier

        This allows to group changes to several series, hence
        producing a macro-change.

        It is possible to strip a changeset using
        `.delete_last_changeset_for`.

        _insertion_date is *only* provided for migration purposes and
        not part of the API.
        """
        assert self._csid is None
        self._csid = self._newchangeset(cnx, author, _insertion_date)
        yield
        del self._csid

    def insert(self, cnx, newts, name, author=None,
               extra_scalars={}):
        """Create a new revision of a given time series

        newts: pandas.Series with date index

        name: str unique identifier of the serie

        author: str free-form author name (mandatory, unless provided
        to the newchangeset context manager).

        """
        assert self._csid or author, 'author is mandatory'
        if self._csid and author:
            L.info('author will not be used when in a changeset')
        assert isinstance(newts, pd.Series)
        assert not newts.index.duplicated().any()

        if str(newts.dtype).startswith('int'):
            newts = newts.astype('float64')

        if not len(newts):
            return

        newts.name = name
        table = self._get_ts_table(cnx, name)

        if isinstance(newts.index, pd.MultiIndex):
            # we impose an order to survive rountrips
            newts = newts.reorder_levels(sorted(newts.index.names))

        if table is None:
            # initial insertion
            if newts.isnull().all():
                return None
            newts = newts[~newts.isnull()]
            table = self._make_ts_table(cnx, name)
            csid = self._csid or self._newchangeset(cnx, author)
            value = {
                'csid': csid,
                'snapshot': tojson(newts),
            }
            # callback for extenders
            self._complete_insertion_value(value, extra_scalars)
            cnx.execute(table.insert().values(value))
            self._finalize_insertion(cnx, csid, name)
            L.info('First insertion of %s by %s', name, author)
            return newts

        diff, newsnapshot = self._compute_diff_and_newsnapshot(
            cnx, table, newts, **extra_scalars
        )
        if diff is None:
            L.info('No difference in %s by %s', name, author)
            return

        tip_id = self._get_tip_id(cnx, table)
        csid = self._csid or self._newchangeset(cnx, author)
        value = {
            'csid': csid,
            'diff': tojson(diff),
            'snapshot': tojson(newsnapshot),
            'parent': tip_id,
        }
        # callback for extenders
        self._complete_insertion_value(value, extra_scalars)
        cnx.execute(table.insert().values(value))
        self._finalize_insertion(cnx, csid, name)

        if tip_id > 1 and tip_id % self._snapshot_interval:
            self._purge_snapshot_at(cnx, table, tip_id)
        L.info('Inserted diff for ts %s by %s', name, author)
        return diff

    def get(self, cnx, name, revision_date=None):
        """Compute and return the serie of a given name

        revision_date: datetime filter to get previous versions of the
        serie

        """
        table = self._get_ts_table(cnx, name)
        if table is None:
            return

        qfilter = []
        if revision_date:
            qfilter.append(lambda cset, _: cset.c.insertion_date <= revision_date)
        current = self._build_snapshot_upto(cnx, table, qfilter)

        if current is not None:
            current.name = name
        return current

    def get_group(self, cnx, name, revision_date=None):
        csid = self._latest_csid_for(cnx, name)

        group = {}
        for seriename in self._changeset_series(cnx, csid):
            serie = self.get(cnx, seriename, revision_date)
            if serie is not None:
                group[seriename] = serie
        return group

    def latest_insertion_date(self, cnx, name):
        cset = schema.changeset
        tstable = self._get_ts_table(cnx, name)
        sql = select([func.max(cset.c.insertion_date)]
        ).where(tstable.c.csid == cset.c.id)
        return cnx.execute(sql).scalar()

    # /API
    # Helpers

    # serie table handling

    def _ts_table_name(self, seriename):
        # namespace.seriename
        return 'timeserie.%s' % seriename

    def _table_definition_for(self, seriename):
        return Table(
            seriename, schema.meta,
            Column('id', Integer, primary_key=True),
            Column('csid', Integer, ForeignKey('changeset.id'),
                   nullable=False),
            # constraint: there is either .diff or .snapshot
            Column('diff', JSONB(none_as_null=True)),
            Column('snapshot', JSONB(none_as_null=True)),
            Column('parent',
                   Integer,
                   ForeignKey('timeserie.%s.id' % seriename,
                              ondelete='cascade'),
                   nullable=True,
                   unique=True,
                   index=True),
            schema='timeserie',
            extend_existing=True
        )

    def _make_ts_table(self, cnx, name):
        tablename = self._ts_table_name(name)
        table = self._table_definition_for(name)
        table.create(cnx)
        sql = schema.registry.insert().values(
            name=name,
            table_name=tablename)
        cnx.execute(sql)
        return table

    def _get_ts_table(self, cnx, name):
        reg = schema.registry
        tablename = self._ts_table_name(name)
        sql = reg.select().where(reg.c.table_name == tablename)
        tid = cnx.execute(sql).scalar()
        if tid:
            return self._table_definition_for(name)

    # changeset handling

    def _newchangeset(self, cnx, author, _insertion_date=None):
        table = schema.changeset
        sql = table.insert().values(
            author=author,
            insertion_date=_insertion_date or datetime.now())
        return cnx.execute(sql).inserted_primary_key[0]

    def _latest_csid_for(self, cnx, name):
        table = self._get_ts_table(cnx, name)
        sql = select([func.max(table.c.csid)])
        return cnx.execute(sql).scalar()

    def _changeset_series(self, cnx, csid):
        cset_serie = schema.changeset_series
        sql = select([cset_serie.c.serie]
        ).where(cset_serie.c.csid == csid)

        return [seriename for seriename, in cnx.execute(sql).fetchall()]

    # insertion handling

    def _get_tip_id(self, cnx, table):
        sql = select([func.max(table.c.id)])
        return cnx.execute(sql).scalar()

    def _complete_insertion_value(self, value, extra_scalars):
        pass

    def _finalize_insertion(self, cnx, csid, name):
        table = schema.changeset_series
        sql = table.insert().values(
            csid=csid,
            serie=name
        )
        cnx.execute(sql)

    # snapshot handling

    def _purge_snapshot_at(self, cnx, table, diffid):
        cnx.execute(
            table.update(
            ).where(table.c.id == diffid
            ).values(snapshot=None)
        )

    def _compute_diff_and_newsnapshot(self, cnx, table, newts, **extra_scalars):
        snapshot = self._build_snapshot_upto(cnx, table)
        diff = self._compute_diff(snapshot, newts)

        if len(diff) == 0:
            return None, None

        # full state computation & insertion
        newsnapshot = self._apply_diff(snapshot, diff)
        return diff, newsnapshot

    def _find_snapshot(self, cnx, table, qfilter=(), column='snapshot'):
        cset = schema.changeset
        sql = select([table.c.id, table.c[column]]
        ).order_by(desc(table.c.id)
        ).limit(1
        ).where(table.c[column] != None)

        if qfilter:
            sql = sql.where(table.c.csid == cset.c.id)
            for filtercb in qfilter:
                sql = sql.where(filtercb(cset, table))

        try:
            snapid, snapdata = cnx.execute(sql).fetchone()
        except TypeError:
            return None, None
        return snapid, fromjson(snapdata, table.name)

    def _build_snapshot_upto(self, cnx, table, qfilter=()):
        snapid, snapshot = self._find_snapshot(cnx, table, qfilter)
        if snapid is None:
            return None

        cset = schema.changeset
        sql = select([table.c.id,
                      table.c.diff,
                      table.c.parent,
                      cset.c.insertion_date]
        ).order_by(table.c.id
        ).where(table.c.id > snapid)

        if qfilter:
            sql = sql.where(table.c.csid == cset.c.id)
            for filtercb in qfilter:
                sql = sql.where(filtercb(cset, table))

        alldiffs = pd.read_sql(sql, cnx)

        if len(alldiffs) == 0:
            return snapshot

        # initial ts
        ts = snapshot
        for _, row in alldiffs.iterrows():
            diff = fromjson(row['diff'], table.name)
            ts = self._apply_diff(ts, diff)
        assert ts.index.dtype.name == 'datetime64[ns]' or len(ts) == 0
        return ts

    # diff handling

    def _compute_diff(self, fromts, tots):
        """Compute the difference between fromts and tots
        (like in tots - fromts).

        """
        if fromts is None or not len(fromts):
            return tots

        mask_overlap = tots.index.isin(fromts.index)
        fromts_overlap = fromts[tots.index[mask_overlap]]
        tots_overlap = tots[mask_overlap]

        if fromts.dtype == 'float64':
            mask_equal = np.isclose(fromts_overlap, tots_overlap,
                                    atol=self._precision)
        else:
            mask_equal = fromts_overlap == tots_overlap

        diff_overlap = tots[mask_overlap][~mask_equal]
        diff_new = tots[~mask_overlap]
        return pd.concat([diff_overlap, diff_new])

    def _apply_diff(self, base_ts, new_ts):
        """Produce a new ts using base_ts as a base and taking any
        intersecting and new values from new_ts.

        """
        if base_ts is None:
            return new_ts
        if new_ts is None:
            return base_ts
        result_ts = pd.Series([0.0], index=base_ts.index.union(new_ts.index))
        result_ts[base_ts.index] = base_ts
        result_ts[new_ts.index] = new_ts
        result_ts = result_ts[~result_ts.isnull()]
        result_ts.sort_index(inplace=True)
        result_ts.name = base_ts.name
        return result_ts
