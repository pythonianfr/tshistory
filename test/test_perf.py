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
        with tempattr(Snapshot, '_bucket_size', bsize):
            for idx, idate in enumerate(forecasts):
                dr = pd.date_range(start=idate, freq='H', periods=48)
                perturbation = perturbations[idx]
                localbase = base * perturbation
                serie = pd.Series(localbase[idate.hour:idate.hour + 48],
                                  index=dr)
                diff = tsh.insert(engine, serie, name, 'test',
                                  _insertion_date=idate)
                if limit and idx > limit:
                    break

    sizes = (100,)#(25, 50, 100, 200, 300)
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
        print('bucket_size, snap_size, noparent, time : ',
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
    t0 = time()
    tshts = tsh.get(engine, 'fcast_100')
    print('TSH GET', time() - t0)


@pytest.mark.perf
def test_bigdata(engine, tracker, ptsh):
    tsh = ptsh

    def create_data():
        # 4 years of sub-hourly points
        for year in range(2015, 2020):
            date = utcdt(year, 1, 1)
            serie = genserie(date, '10Min', 6 * 24 * 365)
            tsh.insert(engine, serie, 'big', 'aurelien.campeas@pythonian.fr',
                       _insertion_date=date)

    t0 = time()
    create_data()
    t1 = time() - t0
    tshclass = tsh.__class__.__name__

    with engine.connect() as cn:
        cn.execute('set search_path to "{}.timeserie"'.format(tsh.namespace))
        df = pd.read_sql('select id, diff from big order by id', cn)

    df['diff'] = df['diff'].apply(lambda x: 0 if x is None else len(x))

    size = df['diff'].sum()
    tracker.append({'test': 'bigdata_insert',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': size})

    t0 = time()
    tsh.get_history(engine, 'big')
    t1 = time() - t0
    tracker.append({'test': 'bigdata_history_all',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': None})

    t0 = time()
    for year in (2015, 2017, 2019):
        for month in (1, 5, 9, 12):
            date = utcdt(year, month, 1)
            tsh.get_history(engine, 'big',
                            from_insertion_date=date,
                            to_insertion_date=date + timedelta(days=31))
    t1 = time() - t0
    tracker.append({'test': 'bigdata_history_chunks',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': None})


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
                with engine.connect() as cn:
                    tsh.insert(cn, serie, 'manydiffs',
                               'aurelien.campeas@pythonian.fr',
                               _insertion_date=date.replace(year=2018)
                    )

    t0 = time()
    create_data()
    t1 = time() - t0
    tshclass = tsh.__class__.__name__

    with engine.connect() as cn:
        cn.execute('set search_path to "{}.timeserie"'.format(tsh.namespace))
        df = pd.read_sql("select id, diff from manydiffs order by id ",
                         cn)
    df['diff'] = df['diff'].apply(lambda x: 0 if x is None else len(x))

    size = df['diff'].sum()
    tracker.append({'test': 'manydiffs_insert',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': size})

    t0 = time()
    tsh.get_history(engine, 'manydiffs')
    t1 = time() - t0
    tracker.append({'test': 'manydiffs_history_all',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': None})

    t0 = time()
    for month in range(1, 3):
        for day in range(1, 5):
            date = utcdt(2018, month, day)
            ts = tsh.get_history(engine, 'manydiffs',
                                 from_insertion_date=date,
                                 to_insertion_date=date + timedelta(days=31))
            assert ts is not None
    t1 = time() - t0
    tracker.append({'test': 'manydiffs_history_chunks',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': None})

    t0 = time()
    for month in range(1, 3):
        for day in range(1, 5):
            date = utcdt(2018, month, day)
            ts = tsh.get_history(engine, 'manydiffs',
                                 from_insertion_date=date,
                                 to_insertion_date=date + timedelta(days=31),
                                 from_value_date=date + timedelta(days=10),
                                 to_value_date=date + timedelta(days=20))
            assert ts is not None
    t1 = time() - t0
    tracker.append({'test': 'manydiffs_history_chunks_valuedate',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': None})
