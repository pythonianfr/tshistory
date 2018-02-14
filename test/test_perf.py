from datetime import datetime
from time import time

import pytest
import pandas as pd

from tshistory.testutil import genserie


@pytest.mark.perf
def _test_bigdata(engine, tracker, ptsh):
    tsh = ptsh

    def create_data():
        for year in range(2015, 2020):
            date = datetime(year, 1, 1)
            serie = genserie(date, '10Min', 6 * 24 * 365)
            with tsh.newchangeset(engine, 'aurelien.campeas@pythonian.fr',
                                  _insertion_date=date):
                tsh.insert(engine, serie, 'big')

    t0 = time()
    create_data()
    t1 = time() - t0
    tshclass = tsh.__class__.__name__

    with engine.connect() as cn:
        cn.execute('set search_path to "{}.timeserie"'.format(tsh.namespace))
        df = pd.read_sql('select id, diff, snapshot from big order by id', cn)

    for attr in ('diff', 'snapshot'):
        df[attr] = df[attr].apply(lambda x: 0 if x is None else len(x))

    size = df[['diff', 'snapshot']].sum().to_dict()
    tracker.append({'test': 'bigdata_insert',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': size['diff'],
                    'snapsize': size['snapshot']})

    t0 = time()
    tsh.get_history(engine, 'big')
    t1 = time() - t0
    tracker.append({'test': 'bigdata_history_all',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': None,
                    'snapsize': None})

    t0 = time()
    for year in (2015, 2017, 2019):
        for month in (1, 5, 9, 12):
            date = datetime(year, month, 1)
            tsh.get_history(engine, 'big',
                            from_insertion_date=date,
                            to_insertion_date=date + timedelta(days=31))
    t1 = time() - t0
    tracker.append({'test': 'bigdata_history_chunks',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': None,
                    'snapsize': None})


@pytest.mark.perf
def _test_lots_of_diffs(engine, tracker, ptsh):
    tsh = ptsh

    def create_data():
        # one insert per day for 4 months
        for month in range(1, 4):
            days = calendar.monthrange(2017, month)[1]
            for day in range(1, days + 1):
                date = datetime(2017, month, day)
                serie = genserie(date, '10Min', 6 * 24)
                with engine.connect() as cn:
                    with tsh.newchangeset(cn, 'aurelien.campeas@pythonian.fr',
                                          _insertion_date=date.replace(year=2018)):
                        tsh.insert(cn, serie, 'manydiffs')

    t0 = time()
    create_data()
    t1 = time() - t0
    tshclass = tsh.__class__.__name__

    with engine.connect() as cn:
        cn.execute('set search_path to "{}.timeserie"'.format(tsh.namespace))
        df = pd.read_sql("select id, diff, snapshot from manydiffs order by id ",
                         cn)
    for attr in ('diff', 'snapshot'):
        df[attr] = df[attr].apply(lambda x: 0 if x is None else len(x))

    size = df[['diff', 'snapshot']].sum().to_dict()
    tracker.append({'test': 'manydiffs_insert',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': size['diff'],
                    'snapsize': size['snapshot']})

    t0 = time()
    tsh.get_history(engine, 'manydiffs')
    t1 = time() - t0
    tracker.append({'test': 'manydiffs_history_all',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': None,
                    'snapsize': None})

    t0 = time()
    for month in range(1, 3):
        for day in range(1, 5):
            date = datetime(2018, month, day)
            ts = tsh.get_history(engine, 'manydiffs',
                                 from_insertion_date=date,
                                 to_insertion_date=date + timedelta(days=31))
            assert ts is not None
    t1 = time() - t0
    tracker.append({'test': 'manydiffs_history_chunks',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': None,
                    'snapsize': None})

    t0 = time()
    for month in range(1, 3):
        for day in range(1, 5):
            date = datetime(2018, month, day)
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
                    'diffsize': None,
                    'snapsize': None})
