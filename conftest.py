from pathlib import Path

from sqlalchemy import create_engine

import pytest

from pytest_sa_pg.fixture import db

from tshistory import schema, tsio


DATADIR = Path(__file__).parent / 'test' / 'data'
DBURI = 'postgresql://localhost:5433/postgres'


@pytest.fixture(scope='session')
def engine(request):
    port = 5433
    db.setup_local_pg_cluster(request, DATADIR, port)
    uri = 'postgresql://localhost:{}/postgres'.format(port)
    engine = create_engine(uri)

    metadata = schema.meta
    # explicitly cleanup the ts tables
    if schema.registry.exists(engine):
        engine.execute('drop schema timeserie cascade')
    metadata.drop_all(engine)
    # /cleanup

    schema.init(engine)
    e = create_engine(uri)
    yield e

    # build a ts using the logs from another
    tsh = tsio.TimeSerie()
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
