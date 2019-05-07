from datetime import timedelta
from time import time
import calendar
from pprint import pprint

import pytest
import pandas as pd
import numpy as np

from tshistory.snapshot import Snapshot
from tshistory.testutil import (
    assert_df,
    genserie,
    utcdt,
    tempattr
)


@pytest.mark.perf
def test_hourly_forecast(engine, tracker, ptsh):
    tsh = ptsh

    # build a sin curve, like e.g. solar input
    base = [np.sin(x) for x in np.linspace(0, 3.14, 20)]
    base.insert(0, 0)
    base.insert(0, 0)
    base.append(0)
    base.append(0)
    base = np.array(base * 3) * 10
    # hourly 3 days
    forecasts = pd.date_range(start=utcdt(2013, 1, 1),
                              freq='H',
                              periods=24 * 365 * 5)
    perturbations = [1 + np.random.binomial(10, .5) / 100.
                     for _ in range(5000)]

    def create(name, bsize, limit=None):
        with tempattr(Snapshot, '_max_bucket_size', bsize):
            for idx, idate in enumerate(forecasts):
                dr = pd.date_range(start=idate, freq='H', periods=48)
                perturbation = perturbations[idx]
                localbase = base * perturbation
                serie = pd.Series(localbase[idate.hour:idate.hour + 48],
                                  index=dr)
                with engine.begin() as cn:
                    diff = tsh.insert(cn, serie, name, 'test',
                                      _insertion_date=idate)
                if limit and idx > limit:
                    break

    sizes = (25,)#(25, 50, 100, 200, 300)
    for bsize in sizes:
        t0 = time()
        name = 'fcast_{}'.format(bsize)
        create(name, bsize, limit=500)
        t1 = time() - t0
        sql = 'select parent, chunk from "{}.snapshot".{} order by id'.format(
            tsh.namespace,
            name
        )
        out = engine.execute(sql).fetchall()
        ssize = sum([len(c) for _, c in out])
        noparentcount = len([x for x, _ in out if x is None])
        print('max_bucket_size, snap_size, noparent, time : ',
              bsize, ssize, noparentcount, t1)

    engine.execute('drop table if exists fcast_sql')
    def sqlcreate(limit=None):
        for idx, idate in enumerate(forecasts):
            dr = pd.date_range(start=idate, freq='H', periods=48)
            perturbation = perturbations[idx]
            localbase = base * perturbation
            serie = pd.DataFrame(
                localbase[idate.hour:idate.hour + 48],
                index=dr
            ).reset_index().rename(columns={
                'index': 'value_date',
                0: 'value'
            })
            serie['insertion'] = idate
            serie = serie[['insertion', 'value_date', 'value']]
            serie['value_date'] = serie['value_date'].apply(lambda x: pd.to_datetime(str(x)))
            serie['insertion'] = serie['insertion'].apply(lambda x: pd.to_datetime(str(x)))
            serie.to_sql('fcast_sql', engine, if_exists='append', index=False)
            if limit and idx > limit:
                break

    t0 = time()
    sqlcreate(500)
    print('sql insert', time() - t0)
    query = '''
            WITH tmp as (
                SELECT value_date, max(insertion) as insertion
                FROM fcast_sql
                GROUP BY value_date
            )
            SELECT main.value_date as "value_date",
                   main.value as value
            FROM fcast_sql as main
                JOIN tmp ON (tmp.value_date = main.value_date
                             AND tmp.insertion = main.insertion)
        '''

    t0 = time()
    sqlts = pd.read_sql(query, engine).set_index('value_date').squeeze()
    print('SQL GET', time() - t0)
    assert len(sqlts) == 549
    t0 = time()
    tshts = tsh.get(engine, 'fcast_25')
    print('TSH GET', time() - t0)
    assert len(tshts) == 549
    t0 = time()
    with engine.begin() as cn:
        hist = tsh.get_history(cn, 'fcast_25')
    assert len(hist) == 502
    print('TSH HIST', time() - t0)
    t0 = time()
    with engine.begin() as cn:
        d1 = tsh.get_delta(cn, 'fcast_25', timedelta(days=1))
    print('DELTA all value dates', time() - t0)
    assert len(d1) == 525
    t0 = time()
    with engine.begin() as cn:
        d2 = tsh.get_delta(cn, 'fcast_25', timedelta(days=1),
                           from_value_date=utcdt(2013, 1, 22),
                           to_value_date=utcdt(2013, 1, 23))
    print('DELTA 1 day ', time() - t0)
    assert d2.index.min() == pd.Timestamp('2013-01-22 00:00:00+0000', tz='UTC')
    assert d2.index.max() == pd.Timestamp('2013-01-23 00:00:00+0000', tz='UTC')


@pytest.mark.perf
def test_bigdata(engine, tracker, ptsh):
    tsh = ptsh

    def create_data():
        # 4 years of sub-hourly points
        for year in range(2015, 2020):
            date = utcdt(year, 1, 1)
            serie = genserie(date, '10Min', 6 * 24 * 365)
            with engine.begin() as cn:
                tsh.insert(cn, serie, 'big', 'aurelien.campeas@pythonian.fr',
                           _insertion_date=date)

    t0 = time()
    create_data()
    t1 = time() - t0
    tshclass = tsh.__class__.__name__

    tracker.append({'test': 'bigdata_insert',
                    'class': tshclass,
                    'time': t1})

    t0 = time()
    with engine.begin() as cn:
        tsh.get_history(cn, 'big')
    t1 = time() - t0
    tracker.append({'test': 'bigdata_history_all',
                    'class': tshclass,
                    'time': t1})

    t0 = time()
    for year in (2015, 2017, 2019):
        for month in (1, 5, 9, 12):
            date = utcdt(year, month, 1)
            with engine.begin() as cn:
                tsh.get_history(cn, 'big',
                                from_insertion_date=date,
                                to_insertion_date=date + timedelta(days=31))
    t1 = time() - t0
    tracker.append({'test': 'bigdata_history_chunks',
                    'class': tshclass,
                    'time': t1})


@pytest.mark.perf
def test_lots_of_diffs(engine, tracker, ptsh):
    tsh = ptsh

    def create_data():
        # one insert per day for 4 months
        for month in range(1, 4):
            days = calendar.monthrange(2017, month)[1]
            for day in range(1, days + 1):
                date = utcdt(2017, month, day)
                serie = genserie(date, '10Min', 6 * 24)
                with engine.begin() as cn:
                    tsh.insert(cn, serie, 'manydiffs',
                               'aurelien.campeas@pythonian.fr',
                               _insertion_date=date.replace(year=2018)
                    )

    t0 = time()
    create_data()
    t1 = time() - t0
    tshclass = tsh.__class__.__name__

    tracker.append({'test': 'manydiffs_insert',
                    'class': tshclass,
                    'time': t1})

    t0 = time()
    with engine.begin() as cn:
        tsh.get_history(cn, 'manydiffs')
    t1 = time() - t0
    tracker.append({'test': 'manydiffs_history_all',
                    'class': tshclass,
                    'time': t1})

    t0 = time()
    for month in range(1, 3):
        for day in range(1, 5):
            date = utcdt(2018, month, day)
            with engine.begin() as cn:
                ts = tsh.get_history(cn, 'manydiffs',
                                     from_insertion_date=date,
                                     to_insertion_date=date + timedelta(days=31))
            assert ts is not None
    t1 = time() - t0
    tracker.append({'test': 'manydiffs_history_chunks',
                    'class': tshclass,
                    'time': t1})
