from datetime import datetime

import pandas as pd
import numpy as np

from sqlalchemy.sql.expression import select, desc

from tshistory.schema import ts_revlog


PRECISION = 1e-14


def insert_ts(engine, newts, name, author):
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

    newrev_sql = ts_revlog.insert()
    with engine.connect() as cnx:
        snapshot, tip_id = _get_snapshot(cnx, name)

        if snapshot is None:
            # initial insertion
            jsonts = tojson(newts)
            value = {
                'data': jsonts,
                'snapshot': jsonts,
                'insertion_date': datetime.now(),
                'author': author,
                'name': name,
            }
            cnx.execute(newrev_sql.values(value))
            print('Fisrt insertion of %s by %s' % (name, author))
            return

        # this is the diff between our computed parent
        diff = compute_diff(snapshot, newts)

        if len(diff) == 0:
            print('No difference in %s by %s' % (name, author))
            return

        assert tip_id is not None
        # full state computation & insertion
        newsnapshot = apply_diff(snapshot, diff)
        value = {
            'data': tojson(diff),
            'snapshot': tojson(newsnapshot),
            'insertion_date': datetime.now(),
            'author': author,
            'name': name,
            'parent': tip_id,
        }
        cnx.execute(newrev_sql.values(value))

        cnx.execute(
            ts_revlog.update(
            ).where(ts_revlog.c.id == tip_id
            ).values(snapshot=None)
        )
        print('Insertion differential of %s by %s' % (name, author))


def get_ts(engine, name, revision_date=None):
    """Compute the top-most timeseries of a given name
    with manual overrides applied
    """
    if revision_date is None:
        current, _ = _get_snapshot(engine, name)
    else:
        current, _ = apply_diffs_upto(engine, name, revision_date)

    if current is not None:
        current.name = name
    return current

def tojson(ts):
    if ts is None:
        return None
    return ts.to_json(orient='split', date_format='iso')


def fromjson(jsonb):
    return pd.read_json(jsonb, orient='split',
                        typ='series', dtype=False)


def _get_snapshot(engine, name):
    sql = select([ts_revlog.c.id,
                  ts_revlog.c.snapshot]
    ).order_by(desc(ts_revlog.c.id)
    ).limit(1
    ).where(ts_revlog.c.name == name)

    df = pd.read_sql(sql, engine)
    if len(df) == 0:
        return None, None

    assert len(df) == 1

    diff_id = df['id'].iloc[0]
    snapshot = fromjson(df['snapshot'].iloc[0])

    return snapshot, int(diff_id)


def compute_diff(ts1, ts2):
    mask_overlap = ts2.index.isin(ts1.index)
    ts_bef_overlap = ts1[ts2.index[mask_overlap]]
    ts_overlap = ts2[mask_overlap]
    mask_equal = np.isclose(ts_bef_overlap, ts_overlap, atol=PRECISION)
    ts_diff_overlap = ts2[mask_overlap][~mask_equal]
    ts_diff_new = ts2[~mask_overlap]
    ts_result = pd.concat([ts_diff_overlap, ts_diff_new])
    return ts_result


def apply_diff(base_ts, new_ts):
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


def apply_diffs_upto(engine, name, revision_date=None):
    sql = select([ts_revlog.c.id,
                  ts_revlog.c.data,
                  ts_revlog.c.parent,
                  ts_revlog.c.insertion_date]
    ).order_by(ts_revlog.c.id
    ).where(ts_revlog.c.name == name)

    alldiffs = pd.read_sql(sql, engine)

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
        ts = apply_diff(ts, child_ts)
        if parent_id not in alldiffs['parent'].tolist():
            return ts, int(parent_id)


def delete_last_diff(engine, name):
    with engine.connect() as cnx:
        sql = select([ts_revlog.c.id,
                      ts_revlog.c.parent]
        ).order_by(desc(ts_revlog.c.id)
        ).limit(1
        ).where(ts_revlog.c.name == name)

        diff_id, parent_id = cnx.execute(sql).fetchone()
        if not diff_id:
            return False

        sql = ts_revlog.delete().where(
            ts_revlog.c.id == diff_id
        )
        cnx.execute(sql)

        # apply on flat
        current, parent_id = apply_diffs_upto(cnx, name)

        update_snapshot_sql = ts_revlog.update(
        ).where(ts_revlog.c.id == parent_id
        ).values(snapshot=tojson(current))

        cnx.execute(update_snapshot_sql)
        return True

