from pathlib import Path

from pytest_sa_pg.fixture import engine_fixture

from tshistory import schema


DATADIR = Path(__file__).parent / 'test' / 'data'
DBURI = 'postgresql://localhost:5433/postgres'


engine = engine_fixture(schema.meta, DATADIR, 5433)

