from pathlib import Path

from sqlalchemy import create_engine

import pytest

from pytest_sa_pg.fixture import db

from tshistory import schema, tsio


DATADIR = Path(__file__).parent / 'test' / 'data'
DBURI = 'postgresql://localhost:5433/postgres'

tshclass = tsio.TimeSerie


@pytest.fixture(scope='session')
def engine(request):
    port = 5433
    db.setup_local_pg_cluster(request, DATADIR, port)
    uri = 'postgresql://localhost:{}/postgres'.format(port)
    engine = create_engine(uri)

    schema.reset(engine)
    schema.init(engine)

    e = create_engine(uri)
    yield e

    # build a ts using the logs from another
    tsh = tshclass()
    log = tsh.log(engine, diff=True)
    allnames = set()
    for rev in log:
        for name, ts in rev['diff'].items():
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


@pytest.fixture()
def tsh():
    return tsio.TimeSerie()
