import os
from pathlib import Path
import shutil
from time import time

import pandas as pd
import pytest
from sqlhelp.testutil import setup_local_pg_cluster
from sqlhelp.pgapi import pgdb as create_engine

from tshistory import schema, tsio
from tshistory.testutil import gengroup, tempconfig


DATADIR = Path(__file__).parent.parent / 'test' / 'data'
DBURI = 'postgresql://localhost:5433/postgres'

ISIZE = None
SIZES = pd.DataFrame()
TIMES = pd.DataFrame()


def get_dir_size(path):
    size = {}
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            size[f] = os.path.getsize(fp) / 1024
    return sum(size.values())


@pytest.fixture
def db(request):
    shutil.rmtree(DATADIR / 'pgdb', ignore_errors=True)
    setup_local_pg_cluster(
        request, DATADIR, 5433, {
        'timezone': 'UTC',
        'log_timezone': 'UTC'
        }
    )
    global ISIZE
    ISIZE = get_dir_size(DATADIR / 'pgdb')


def driver(tsh):
    if tsh.__class__.__name__ == 'timeseries':
        return 'pg'
    return 'fs1'


def record_sizes(testname, tsh, op):
    # size computation
    pgsize = get_dir_size(DATADIR / 'pgdb') - ISIZE
    fssize = get_dir_size(DATADIR / 'fs1' / 'perf')
    total = pgsize + fssize

    driv = driver(tsh)
    sizes = pd.DataFrame.from_records(
        [
            (testname, driv, op, pgsize, fssize, total),
        ],
        columns=(
            'test      ',
            'driver    ',
            'operation ',
            'pgsize    ',
            'fssize    ',
            'total     '
        )
    )
    global SIZES
    SIZES = pd.concat([SIZES, sizes], ignore_index=True)


def record_times(testname, tsh, op, time):
    driv = driver(tsh)
    times = pd.DataFrame.from_records(
        [
            (testname, driv, op, "{:2.4f}".format(time))
        ],
        columns=(
            'test      ',
            'driver    ',
            'operation ',
            'time      '
        )
    )
    global TIMES
    TIMES = pd.concat([TIMES, times], ignore_index=True)


def write_stats():
    import tabulate
    tabulate.PRESERVE_WHITESPACE = True
    args = {'tablefmt': 'mixed_outline'}
    (DATADIR / 'sizes').write_text(SIZES.to_markdown(**args) + '\n')
    (DATADIR / 'times').write_text(TIMES.to_markdown(**args) + '\n')


@pytest.fixture
def engine(db):
    return create_engine(DBURI)


@pytest.fixture(params=['a', 'b'])
def tsh(request, engine):
    namespace = 'perf'
    schema.tsschema(namespace).create(engine)

    datapath = DATADIR / 'fs1' / namespace
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

    write_stats()


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

    record_sizes('bigupdate', tsh, 'update1')
    record_times('bigupdate', tsh, 'update1', time() - t0)

    t0 = time()
    ts = tsh.get(engine, name)
    assert len(ts) == 600000

    record_times('bigupdate', tsh, 'get1', time() - t0)

    # edit 1 point at the beginning, exhibiting the worst case scenario
    ts[0] = 42
    t0 = time()
    tsh.update(
        engine,
        ts,
        name,
        'Babar'
    )

    record_sizes('bigupdate', tsh, 'update2')
    record_times('bigupdate', tsh, 'get2', time() - t0)

    t0 = time()
    ts = tsh.get(engine, name)
    assert len(ts) == 600000
    record_times('bigupdate', tsh, 'get3', time() - t0)


@pytest.mark.perf
def _test_parquet_feather():
    t0 = time()
    ts = pd.Series(
        [1, 2, 3] * 200000,
        index=pd.date_range(
            pd.Timestamp('2025-1-1', tz='utc'),
            freq='h',
            periods=600000
        )
    )

    t0 = time()
    ts.to_frame().to_parquet('ts.parquet')
    print(f'parquet.update ran in {time() - t0} seconds.')

    t0 = time()
    pd.read_parquet('ts.parquet')
    print(f'parquet.get ran in {time() - t0} seconds.')

    t0 = time()
    ts.to_frame().to_feather('ts.feather')
    print(f'feather.update ran in {time() - t0} seconds.')

    t0 = time()
    pd.read_feather('ts.feather')
    print(f'feather.get ran in {time() - t0} seconds.')



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
    record_times('meteo', tsh, 'update', time() - t0)
    record_sizes('meteo', tsh, 'update')

    t0 = time()
    tsh.history(engine, 'solar-fcst')
    record_times('meteo', tsh, 'history', time() - t0)
    record_sizes('meteo', tsh, 'history')


@pytest.mark.perf
def test_big_group(engine, tsh):
    df = gengroup(
        n_scenarios=99,
        from_date=pd.Timestamp('2025-1-1', tz='utc'),
        length=9999,
        freq='h',
        seed=2
    )
    tsh.group_replace(
        engine,
        df,
        'group-perf',
        'Babar'
    )
    t1 = time()
    df = tsh.group_get(
        engine,
        'group-perf'
    )
    assert len(df) == 9999
    print('GROUP GET', time() - t1)
    t1 = time()
    df = tsh.group_get(
        engine,
        'group-perf'
    )
    assert len(df) == 9999
    print('GROUP GET', time() - t1)
