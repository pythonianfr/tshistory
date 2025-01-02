from pathlib import Path
import shutil
from time import time

import pandas as pd
import pytest

from tshistory import schema, tsio
from tshistory.testutil import tempconfig


DATADIR = Path(__file__).parent.parent / 'test' / 'data'
DBURI = 'postgresql://localhost:5433/postgres'


@pytest.fixture(params=['a', 'b'],
                scope='session')
def tsh(request, engine):
    namespace = 'tsh'
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
