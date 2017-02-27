from datetime import datetime

import pandas as pd
import numpy as np

from sqlalchemy.sql.expression import select, desc

from tshistory.schema import make_ts_table, get_ts_table


PRECISION = 1e-14


def tojson(ts):
    if ts is None:
        return None
    return ts.to_json(orient='split', date_format='iso')


def fromjson(jsonb):
    return pd.read_json(jsonb, orient='split',
                        typ='series', dtype=False)



class TimeSerie(object):

    # API : insert, get

    def insert(self, engine, newts, name, author):
        """Create a new revision of a given time series
        ts: pandas.Series with date index and float values
        name: str unique identifier of the serie
        author: str free-form author name
        """
        assert isinstance(newts, pd.Series)
        newts = newts[~newts.isnull()]  # wipe the the NaNs

        if len(newts):
            assert newts.index.dtype.name == 'datetime64[ns]'
        else:
            return

        newts = newts.astype('float64')
        newts.name = name

        with engine.connect() as cnx:
            table = get_ts_table(cnx, name)

            if table is None:
                # initial insertion
                table = make_ts_table(cnx, name)
                jsonts = tojson(newts)
                value = {
                    'data': jsonts,
                    'snapshot': jsonts,
                    'insertion_date': datetime.now(),
                    'author': author
                }
                cnx.execute(table.insert().values(value))
                print('Fisrt insertion of %s by %s' % (name, author))
                return

            # NOTE: this depends on the snapshot being always maintained
            #       at the top-level
            snapshot, tip_id = self._read_latest_snapshot(cnx, table)
            # this is the diff between our computed parent
            diff = self._compute_diff(snapshot, newts)

            if len(diff) == 0:
                print('No difference in %s by %s' % (name, author))
                return

            assert tip_id is not None
            # full state computation & insertion
            newsnapshot = self._apply_diff(snapshot, diff)
            value = {
                'data': tojson(diff),
                'snapshot': tojson(newsnapshot),
                'insertion_date': datetime.now(),
                'author': author,
                'parent': tip_id,
            }
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
        table = get_ts_table(cnx, name)
        if table is None:
            return

        if revision_date is None:
            current, _ = self._read_latest_snapshot(cnx, table)
        else:
            current, _ = self._build_snapshot_upto(cnx, table, revision_date)

        if current is not None:
            current.name = name
        return current

    def delete_last_diff(self, engine, name):
        with engine.connect() as cnx:
            table = get_ts_table(cnx, name)
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
            current, parent_id = self._build_snapshot_upto(cnx, table)

            update_snapshot_sql = table.update(
            ).where(table.c.id == parent_id
            ).values(snapshot=tojson(current))

            cnx.execute(update_snapshot_sql)
            return True

    # /API
    # Helpers

    def _read_latest_snapshot(self, cnx, table):
        sql = select([table.c.id,
                      table.c.snapshot]
        ).order_by(desc(table.c.id)
        ).limit(1)

        df = pd.read_sql(sql, cnx)
        if len(df) == 0:
            return None, None

        assert len(df) == 1

        diff_id = df['id'].iloc[0]
        snapshot = fromjson(df['snapshot'].iloc[0])

        return snapshot, int(diff_id)

    def _compute_diff(self, ts1, ts2):
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

    def _build_snapshot_upto(self, cnx, table, revision_date=None):
        sql = select([table.c.id,
                      table.c.data,
                      table.c.parent,
                      table.c.insertion_date]
        ).order_by(table.c.id)

        alldiffs = pd.read_sql(sql, cnx)

        if revision_date:
            alldiffs = alldiffs[alldiffs['insertion_date'] <= revision_date]

        if len(alldiffs) == 0:
            return None, None

        base = alldiffs.loc[alldiffs.loc[:, 'parent'].isnull()]
        # initial ts and its id
        ts = fromjson(base['data'].iloc[0])
        parent_id = base['id'].iloc[0]  # actually the root

        if len(alldiffs) == 1:
            assert ts.index.dtype.name == 'datetime64[ns]' or len(ts) == 0
            return ts, parent_id

        while True:
            child_row = alldiffs.loc[alldiffs.loc[:, 'parent'] == parent_id, :]
            child_ts = fromjson(child_row['data'].iloc[0])
            parent_id = child_row['id'].iloc[0]
            ts = self._apply_diff(ts, child_ts)
            if parent_id not in alldiffs['parent'].tolist():
                return ts, int(parent_id)
