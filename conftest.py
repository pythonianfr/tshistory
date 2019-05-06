from pathlib import Path
import logging

from sqlalchemy import create_engine
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


@pytest.fixture(scope='session')
def datadir():
    return DATADIR


@pytest.fixture(params=['tsh', 'zzz'],
                scope='session')
def tsh(request, engine):
    namespace = request.param
    sch = schema.tsschema(namespace)
    sch.destroy(engine)
    schema.init_schemas(engine, namespace)

    if namespace == 'zzz':
        Snapshot._max_bucket_size = 5
    yield tsio.TimeSerie(namespace)


@pytest.fixture(scope='session')
def ptsh(engine):
    sch = schema.tsschema()
    sch.destroy(engine)
    schema.register_schema(sch)
    schema.init_schemas(engine)
    return tsio.TimeSerie()


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
