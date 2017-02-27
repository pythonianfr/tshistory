from pathlib import Path

from sqlalchemy import create_engine, select, Table, MetaData
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
    reg = schema.ts_registry
    if reg.exists(engine):
        meta = MetaData()
        for tname, in engine.execute(select([reg.c.table_name])):
            table = Table(tname, meta)
            with engine.connect() as cnx:
                table.drop(cnx)
    # /cleanup

    metadata = schema.meta
    metadata.drop_all(engine)
    metadata.create_all(engine)
    return create_engine(uri)
