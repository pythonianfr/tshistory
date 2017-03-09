from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.schema import CreateSchema

import pytest

from pytest_sa_pg.fixture import db

from tshistory import schema


DATADIR = Path(__file__).parent / 'test' / 'data'
DBURI = 'postgresql://localhost:5433/postgres'


@pytest.fixture(scope='session')
def engine(request):
    port = 5433
    db.setup_local_pg_cluster(request, DATADIR, port)
    uri = 'postgresql://localhost:{}/postgres'.format(port)
    engine = create_engine(uri)

    # explicitly cleanup the ts tables
    if schema.registry.exists(engine):
        engine.execute('drop schema timeserie cascade')
    engine.execute(CreateSchema('timeserie'))
    # /cleanup

    metadata = schema.meta
    metadata.drop_all(engine)
    metadata.create_all(engine)
    return create_engine(uri)
