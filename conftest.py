from pathlib import Path
import logging

from sqlalchemy import create_engine, MetaData
import pandas as pd

import pytest
from pytest_sa_pg import db
from click.testing import CliRunner

from tshistory import schema, tsio, command
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


@pytest.fixture
def cli():
    def runner(*args):
        return CliRunner().invoke(command.tsh, [str(a) for a in args])
    return runner
