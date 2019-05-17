from pathlib import Path

from sqlalchemy import create_engine
import pandas as pd

import pytest
from pytest_sa_pg import db
from click.testing import CliRunner

from tshistory import schema, tsio, cli as command
from tshistory.snapshot import Snapshot


DATADIR = Path(__file__).parent / 'test' / 'data'
DBURI = 'postgresql://localhost:5433/postgres'


@pytest.fixture(scope='session')
def engine(request):
    db.setup_local_pg_cluster(request, DATADIR, 5433, {
        'timezone': 'UTC',
        'log_timezone': 'UTC'}
    )
    return create_engine(DBURI)


@pytest.fixture(scope='session')
def datadir():
    return DATADIR


@pytest.fixture(params=['tsh', 'z-z'],
                scope='session')
def tsh(request, engine):
    namespace = request.param
    sch = schema.tsschema(namespace)
    sch.create(engine)

    if namespace == 'z-z':
        Snapshot._max_bucket_size = 5
    yield tsio.timeseries(namespace)


@pytest.fixture(scope='session')
def ptsh(engine):
    sch = schema.tsschema()
    sch.create(engine)
    return tsio.timeseries()


OUT = []

@pytest.fixture(scope='session')
def tracker():
    yield OUT
    print(pd.DataFrame(OUT))


@pytest.fixture
def cli():
    def runner(*args, **kw):
        args = [str(a) for a in args]
        for k, v in kw.items():
            if isinstance(v, bool):
                if v:
                    args.append(f'--{k}')
            else:
                args.append(f'--{k}')
                args.append(str(v))
        return CliRunner().invoke(command.tsh, args)
    return runner
