from datetime import datetime
from contextlib import contextmanager

import pandas as pd
import numpy as np

from sqlalchemy import Table, Column, Integer, String, ForeignKey
from sqlalchemy.sql.expression import select, desc, func
from sqlalchemy.dialects.postgresql import JSONB

from tshistory import schema

PRECISION = 1e-14


def tojson(ts):
    if ts is None:
        return None
    return ts.to_json(orient='split', date_format='iso')


def fromjson(jsonb):
    return pd.read_json(jsonb, orient='split',
                        typ='series', dtype=False)


class TimeSerie(object):
    _csid = None

    # API : changeset, insert, get, delete
    @contextmanager
    def newchangeset(self, cnx, author):
        assert self._csid is None
        self._csid = self._newchangeset(cnx, author)
        yield
        del self._csid

    def insert(self, engine, newts, name, author=None,
               extra_scalars={}):
        """Create a new revision of a given time series
        ts: pandas.Series with date index and float values
        name: str unique identifier of the serie
        author: str free-form author name
        """
        assert self._csid or author, 'author is mandatory'
        if self._csid and author:
            print('author will not be used when in a changeset')
        assert isinstance(newts, pd.Series)

        newts = newts[~newts.isnull()]  # wipe the the NaNs
        if len(newts):
            assert newts.index.dtype.name == 'datetime64[ns]'
        else:
            return

        newts = newts.astype('float64')
        newts.name = name

        with engine.connect() as cnx:
            table = self._get_ts_table(cnx, name)

            if table is None:
                # initial insertion
                table = self._make_ts_table(cnx, name)
                jsonts = tojson(newts)
                value = {
                    'csid': self._csid or self._newchangeset(cnx, author),
                    'data': jsonts,
                    'snapshot': jsonts,
                }
                # callback for extenders
                self._complete_insertion_value(value, extra_scalars)

                cnx.execute(table.insert().values(value))
                print('Fisrt insertion of %s by %s' % (name, author))
                return

            diff, newsnapshot = self._compute_diff_and_newsnapshot(
                cnx, table, newts, **extra_scalars
            )
            if diff is None:
                print('No difference in %s by %s' % (name, author))
                return

            tip_id = self._get_tip_id(cnx, table)
            value = {
                'csid': self._csid or self._newchangeset(cnx, author),
                'data': tojson(diff),
                'snapshot': tojson(newsnapshot),
                'parent': tip_id,
            }
            # callback for extenders
            self._complete_insertion_value(value, extra_scalars)
            cnx.execute(table.insert().values(value))

            cnx.execute(
                table.update(
                ).where(table.c.id == tip_id
                ).values(snapshot=None)
            )
            print('Insertion differential of %s by %s' % (name, author))

    def get(self, cnx, name, revision_date=None):
        """Compute the top-most timeseries of a given name
        with manual overrides applied
        """
        table = self._get_ts_table(cnx, name)
        if table is None:
            return

        if revision_date is None:
            current = self._read_latest_snapshot(cnx, table)
        else:
            current = self._build_snapshot_upto(
                cnx, table, lambda cset, _: cset.c.insertion_date <= revision_date
            )

        if current is not None:
            current.name = name
        return current

    def delete_last_diff(self, engine, name, **kw):
        with engine.connect() as cnx:
            table = self._get_ts_table(cnx, name)
            sql = select([table.c.id,
                          table.c.parent]
            ).order_by(desc(table.c.id)
            ).limit(1)

            diff_id, parent_id = cnx.execute(sql).fetchone()
            if not diff_id:
                return False

            sql = table.delete().where(
                table.c.id == diff_id
            )
            cnx.execute(sql)

            # apply on flat
            current = self._build_snapshot_upto(cnx, table)
            parent_id = self._get_tip_id(cnx, table)

            update_snapshot_sql = table.update(
            ).where(table.c.id == parent_id
            ).values(snapshot=tojson(current))

            cnx.execute(update_snapshot_sql)
            return True

    # /API
    # Helpers

    def _ts_table_name(self, name):
        return 'ts_%s' % name

    def _table_definition_for(self, tablename):
        return Table(
            tablename, schema.meta,
            Column('id', Integer, primary_key=True),
            Column('csid', Integer, ForeignKey('ts_changeset.id'),
                   nullable=False),
            Column('data', JSONB, nullable=False),
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

    def _newchangeset(self, cnx, author):
        table = schema.ts_changeset
        sql = table.insert().values(
            author=author,
            insertion_date=datetime.now())
        return cnx.execute(sql).inserted_primary_key[0]

    def _get_tip_id(self, cnx, table):
        sql = select([func.max(table.c.id)])
        return cnx.execute(sql).scalar()

    def _complete_insertion_value(self, value, extra_scalars):
        pass

    def _compute_diff_and_newsnapshot(self, cnx, table, newts, **extra_scalars):
        # NOTE: this depends on the snapshot being always maintained
        #       at the top-level
        snapshot = self._read_latest_snapshot(cnx, table)
        # this is the diff between our computed parent
        diff = self._compute_diff(snapshot, newts)

        if len(diff) == 0:
            return None, None

        # full state computation & insertion
        newsnapshot = self._apply_diff(snapshot, diff)
        return diff, newsnapshot

    def _read_latest_snapshot(self, cnx, table):
        sql = select([table.c.snapshot]
        ).order_by(desc(table.c.id)
        ).limit(1)

        snapjson = cnx.execute(sql).scalar()
        if snapjson is None:
            return
        return fromjson(snapjson)

    def _compute_diff(self, ts1, ts2):
        if ts1 is None:
            return ts2
        mask_overlap = ts2.index.isin(ts1.index)
        ts_bef_overlap = ts1[ts2.index[mask_overlap]]
        ts_overlap = ts2[mask_overlap]
        mask_equal = np.isclose(ts_bef_overlap, ts_overlap, atol=PRECISION)
        ts_diff_overlap = ts2[mask_overlap][~mask_equal]
        ts_diff_new = ts2[~mask_overlap]
        ts_result = pd.concat([ts_diff_overlap, ts_diff_new])
        return ts_result

    def _apply_diff(self, base_ts, new_ts):
        """Produce a new ts using base_ts as a base and
        taking any intersecting and new values from new_ts
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

    def _build_snapshot_upto(self, cnx, table, *qfilter):
        cset = schema.ts_changeset
        sql = select([table.c.id,
                      table.c.data,
                      table.c.parent,
                      cset.c.insertion_date]
        ).order_by(table.c.id
        ).where(table.c.csid == cset.c.id)

        for filtercb in qfilter:
            sql = sql.where(filtercb(cset, table))

        alldiffs = pd.read_sql(sql, cnx)

        if len(alldiffs) == 0:
            return None

        # initial ts
        ts = fromjson(alldiffs.iloc[0]['data'])
        for _, row in alldiffs[1:].iterrows():
            diff = fromjson(row['data'])
            ts = self._apply_diff(ts, diff)
        assert ts.index.dtype.name == 'datetime64[ns]' or len(ts) == 0
        return ts
