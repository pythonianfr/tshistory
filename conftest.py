from pathlib import Path
import logging

from sqlalchemy import MetaData
import pandas as pd

import pytest
from pytest_sa_pg.fixture import engine_fixture

from tshistory import schema, tsio


DATADIR = Path(__file__).parent / 'test' / 'data'
DBURI = 'postgresql://localhost:5433/postgres'

schema.L.addHandler(logging.StreamHandler())
tsio.L.addHandler(logging.StreamHandler())


engine = engine_fixture(schema.meta, DATADIR, 5433,
                        # callback does nothing because we do the job
                        # in the other fixtures
                        lambda e, m: None
)


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
