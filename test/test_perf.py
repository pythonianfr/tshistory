import os
from pathlib import Path
import shutil
from time import time

import pandas as pd
import pytest
from pytest_sa_pg import db as dbsetup
from sqlalchemy import create_engine

from tshistory import schema, tsio
from tshistory.testutil import tempconfig


DATADIR = Path(__file__).parent.parent / 'test' / 'data'
DBURI = 'postgresql://localhost:5433/postgres'
SIZES = {}


def get_dir_size(path):
    size = {}
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            size[f] = os.path.getsize(fp) / 1024
    return sum(size.values())


@pytest.fixture(scope='session')
def db(request):
    shutil.rmtree(DATADIR / 'pgdb', ignore_errors=True)
    dbsetup.setup_local_pg_cluster(
        request, DATADIR, 5433, {
        'timezone': 'UTC',
        'log_timezone': 'UTC'
        }
    )
    SIZES['init'] = get_dir_size(DATADIR / 'pgdb')
    print('initial empty postgres db size: ', SIZES['init'])


def show_sizes():
    # size computation
    SIZES['pg'] = get_dir_size(DATADIR / 'pgdb')
    print('filled postgres db size: ', SIZES['pg'])

    SIZES['fs1'] = get_dir_size(DATADIR / 'perf')
    print('filled postgres fs1 size: ', SIZES['fs1'])

    for k in SIZES:
        print(f'{k}:', SIZES[k])
    total = SIZES['pg'] + SIZES['fs1']
    print('delta: ', (total - SIZES['init']))



@pytest.fixture(scope='session')
def engine(db):
    return create_engine(DBURI)


@pytest.fixture(params=['a', 'b'],
                scope='session')
def tsh(request, engine):
    namespace = 'perf'
    schema.tsschema(namespace).create(engine, reset=True)

    datapath = DATADIR/namespace
    shutil.rmtree(datapath, ignore_errors=True)
    if not datapath.exists():
        datapath.mkdir()

    conf = (
        f'[dburi]\n'
        f'test = {DBURI}\n'
        f'[storage]\n'
        f'test = filesystem1\n'
        f'test.path = {datapath}'
    )

    if request.param == 'b':
        with tempconfig(conf.encode()):
            yield tsio.timeseriesfs1(namespace, None, uri=DBURI)
    else:
        yield tsio.timeseries(namespace)


@pytest.mark.perf
def test_big_update(engine, tsh):
    # One update with a big series (600k points)
    name = 'test_one_update_many_points'
    tsh.delete(engine, name)
    # we start with a small initial update
    # because we want to test the ganeral update path
    # (not initial)
    ts = pd.Series(
        [1.1, 2.2, 3.3],
        index=pd.date_range(
            pd.Timestamp('2025-1-1', tz='utc'),
            freq='h',
            periods=3
        )
    )
    tsh.update(
        engine,
        ts,
        name,
        'Babar'
    )

    # now, with a lot of points
    # 600.000 takes roughly 1s for the pg insert on my machine
    ts = pd.Series(
        [1, 2, 3] * 200000,
        index=pd.date_range(
            pd.Timestamp('2025-1-1', tz='utc'),
            freq='h',
            periods=600000
        )
    )

    t0 = time()
    tsh.update(
        engine,
        ts,
        name,
        'Babar'
    )

    print(f'{tsh}.update ran in {time() - t0} seconds.')

    t0 = time()
    ts = tsh.get(engine, name)
    assert len(ts) == 600000
    print(f'{tsh}.get ran in {time() - t0} seconds.')

    show_sizes()


@pytest.mark.perf
def test_meteo_versions(engine, tsh):
    # Simulate a meteo series: 4 updates a day with each update
    # an hourly solar series (contains a lot of zeroes) over 2 weeks
    ts = pd.read_csv(DATADIR / 'solar.fcst', index_col=0, header=None, parse_dates=True)

    t0 = time()
    i = 0
    for d in range(24):
        for h in (0, 6, 12, 18):
            i += 1
            tsh.update(
                engine,
                ts[1] * ((i % 2) * .1),  # a small perturbation to make sure we have points
                'solar-fcst',
                'Babar',
                insertion_date=pd.Timestamp(f'2025-1-{6+d} {h}:00:00', tz='utc')
            )
    print(f'{tsh}.update ran in {time() - t0} seconds.')

    t0 = time()
    tsh.history(engine, 'solar-fcst')
    print(f'{tsh}.history ran in {time() - t0} seconds.')

    show_sizes()
