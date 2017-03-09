from datetime import datetime
from contextlib import contextmanager
import logging

import pandas as pd
import numpy as np

from sqlalchemy import Table, Column, Integer, ForeignKey
from sqlalchemy.sql.expression import select, desc, func, text
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
    return ts.to_json(orient='split', date_format='iso')


def fromjson(jsonb):
    return pd.read_json(jsonb, orient='split',
                        typ='series', dtype=False)


class TimeSerie(object):
    _csid = None
    _snapshot_interval = 10
    _precision = 1e-14

    # API : changeset, insert, get, delete
    @contextmanager
    def newchangeset(self, cnx, author):
        """A context manager to allow insertion of several series within the
        same changeset identifier

        This allows to group changes to several series, hence
        producing a macro-change.

        It is possible to strip a changeset using
        `.delete_last_changeset_for`.

        """
        assert self._csid is None
        self._csid = self._newchangeset(cnx, author)
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

        if newts.dtype == 'int64':
            newts = newts.astype('float64')
        newts = newts[~newts.isnull()]  # wipe the the NaNs
        if not len(newts):
            return

        newts.name = name
        table = self._get_ts_table(cnx, name)

        if table is None:
            # initial insertion
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
            L.info('Fisrt insertion of %s by %s', name, author)
            return

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
        L.info('Insertion differential of %s by %s', name, author)

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
        current = self._build_snapshot_upto(cnx, table, *qfilter)

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

    def delete_last_changeset_for(self, cnx, name, **kw):
        """Delete the most recent changeset associated with a serie.

        Not only will the named serie be stripped but all other series
        particiapting in the changeset (if any) will also be stripped.

        """
        csid = self._latest_csid_for(cnx, name)
        if not csid:
            return False

        tables = [self._get_ts_table(cnx, seriename)
                  for seriename in self._changeset_series(cnx, csid)]
        for table in tables:
            sql = table.delete().where(
                table.c.csid == csid
            )
            cnx.execute(sql)
        return True

    # /API
    # Helpers

    # serie table handling

    def _ts_table_name(self, name):
        return 'ts_%s' % name

    def _table_definition_for(self, tablename):
        return Table(
            tablename, schema.meta,
            Column('id', Integer, primary_key=True),
            Column('csid', Integer, ForeignKey('ts_changeset.id'),
                   nullable=False),
            # constraint: there is either .diff or .snapshot
            Column('diff', JSONB),
            Column('snapshot', JSONB),
            Column('parent',
                   Integer,
                   ForeignKey('%s.id' % tablename, ondelete='cascade'),
                   nullable=True,
                   unique=True,
                   index=True),
        )

    def _make_ts_table(self, cnx, name):
        tablename = self._ts_table_name(name)
        table = self._table_definition_for(tablename)
        table.create(cnx)
        sql = schema.ts_registry.insert().values(
            name=name,
            table_name=tablename)
        cnx.execute(sql)
        return table

    def _get_ts_table(self, cnx, name):
        reg = schema.ts_registry
        sql = reg.select().where(reg.c.name == name)
        tid = cnx.execute(sql).scalar()
        if tid:
            return Table(self._ts_table_name(name), schema.meta,
                         autoload=True, autoload_with=cnx.engine)

    # changeset handling

    def _newchangeset(self, cnx, author):
        table = schema.ts_changeset
        sql = table.insert().values(
            author=author,
            insertion_date=datetime.now())
        return cnx.execute(sql).inserted_primary_key[0]

    def _latest_csid_for(self, cnx, name):
        table = self._get_ts_table(cnx, name)
        sql = select([table.c.csid]).order_by(desc(table.c.id)).limit(1)
        return cnx.execute(sql).scalar()

    def _changeset_series(self, cnx, csid):
        cset_serie = schema.ts_changeset_series
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
        table = schema.ts_changeset_series
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

    def _find_snapshot(self, cnx, table, *qfilter):
        cset = schema.ts_changeset
        sql = select([func.max(table.c.id), table.c.snapshot]
        ).group_by(table.c.id, table.c.snapshot
        ).where(table.c.csid == cset.c.id
        ).where(text("not snapshot @> 'null'")) # jsonb weirdness

        if qfilter:
            for filtercb in qfilter:
                sql = sql.where(filtercb(cset, table))

        try:
            snapid, snapdata = cnx.execute(sql).fetchone()
        except TypeError:
            return None, None
        return snapid, fromjson(snapdata)

    def _build_snapshot_upto(self, cnx, table, *qfilter):
        snapid, snapshot = self._find_snapshot(cnx, table, *qfilter)
        if snapid is None:
            return None

        cset = schema.ts_changeset
        sql = select([table.c.id,
                      table.c.diff,
                      table.c.parent,
                      cset.c.insertion_date]
        ).order_by(table.c.id
        ).where(table.c.csid == cset.c.id
        ).where(table.c.id > snapid)

        for filtercb in qfilter:
            sql = sql.where(filtercb(cset, table))

        alldiffs = pd.read_sql(sql, cnx)

        if len(alldiffs) == 0:
            return snapshot

        # initial ts
        ts = snapshot
        for _, row in alldiffs.iterrows():
            diff = fromjson(row['diff'])
            ts = self._apply_diff(ts, diff)
        assert ts.index.dtype.name == 'datetime64[ns]' or len(ts) == 0
        return ts

    # diff handling

    def _compute_diff(self, fromts, tots):
        """Compute the difference between fromts and tots
        (like in tots - fromts).

        Deletions are not handled. New lines in tots and lines that
        changed in tots relatively to tots will appear in the diff.

        """
        if fromts is None:
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
        result_ts.sort_index(inplace=True)
        return result_ts
