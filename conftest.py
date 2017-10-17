from time import time
from pathlib import Path
import logging

from sqlalchemy import create_engine, MetaData

import pytest

from pytest_sa_pg.fixture import db

from tshistory import schema, tsio


DATADIR = Path(__file__).parent / 'test' / 'data'
DBURI = 'postgresql://localhost:5433/postgres'

schema.L.addHandler(logging.StreamHandler())
tsio.L.addHandler(logging.StreamHandler())


@pytest.fixture(scope='session')
def engine(request):
    port = 5433
    db.setup_local_pg_cluster(request, DATADIR, port)
    uri = 'postgresql://localhost:{}/postgres'.format(port)
    e = create_engine(uri)
    yield e


@pytest.fixture(params=['tsh', 'zzz'],
                scope='session')
def tsh(request, engine):
    namespace = request.param
    schema.reset(engine, namespace)
    schema.init(engine, MetaData(), namespace)
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
            with tsh.newchangeset(engine, rev['author'],
                                  _insertion_date=rev['date']):
                tsh.insert(engine, ts, 'new_' + name)

    # NOTE: the name set varies depending on the amount of tests
    # so we don't capture that exact set for regression purpposes
    # We only want to prove the manipulated series can be reconstructed
    # using the logger.
    for name in allnames:
        assert (tsh.get(engine, name) == tsh.get(engine, 'new_' + name)).all()

    schema.reset(engine, namespace)


OUT = []

@pytest.fixture
def tracker(scope='session'):
    import pandas as pd
    yield OUT
    print(pd.DataFrame(OUT))
