from pathlib import Path
import logging

from sqlalchemy import create_engine, MetaData
import pandas as pd

import pytest
from pytest_sa_pg import db

from tshistory import schema, tsio
from tshistory.snapshot import Snapshot


DATADIR = Path(__file__).parent / 'test' / 'data'
DBURI = 'postgresql://localhost:5433/postgres'

schema.L.addHandler(logging.StreamHandler())
tsio.L.addHandler(logging.StreamHandler())


@pytest.fixture(scope='session')
def engine(request):
    db.setup_local_pg_cluster(request, DATADIR, 5433, {
        'timezone': 'UTC',
        'log_timezone': 'UTC'}
    )
    return create_engine(DBURI)


@pytest.fixture(params=['tsh', 'zzz'],
                scope='session')
def tsh(request, engine):
    namespace = request.param
    schema.delete_schema(engine, namespace)
    schema.init(engine, MetaData(), namespace)

    if namespace == 'zzz':
        Snapshot._max_bucket_size = 5
    tsh = tsio.TimeSerie(namespace)
    yield tsh

    # build a ts using the logs from another
    log = tsh.log(engine, diff=True)
    allnames = set()
    for rev in log:
        for name, ts in rev['diff'].items():
            if 'big' in name:
                continue
            allnames.add(name)
            tsh.insert(engine, ts, 'new_' + name,
                       rev['author'], _insertion_date=rev['date'])

    # NOTE: the name set varies depending on the amount of tests
    # so we don't capture that exact set for regression purpposes
    # We only want to prove the manipulated series can be reconstructed
    # using the logger.
    for name in allnames:
        assert (tsh.get(engine, name) == tsh.get(engine, 'new_' + name)).all()

    schema.reset(engine, namespace)


@pytest.fixture(scope='session')
def ptsh(engine):
    schema.reset(engine)
    schema.init(engine, MetaData())
    return tsio.TimeSerie()


OUT = []

@pytest.fixture(scope='session')
def tracker():
    yield OUT
    print(pd.DataFrame(OUT))
