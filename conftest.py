from pathlib import Path

from sqlalchemy import create_engine
import pandas as pd

import pytest
from pytest_sa_pg import db as dbsetup
from click.testing import CliRunner

from tshistory import (
    api as tsh_api,
    cli as command,
    schema,
    tsio
)
from tshistory.snapshot import Snapshot

try:
    from tshistory_formula import schema as fschema
except ImportError:
    fschema = None


DATADIR = Path(__file__).parent / 'test' / 'data'
DBURI = 'postgresql://localhost:5433/postgres'


@pytest.fixture(scope='session')
def db(request):
    dbsetup.setup_local_pg_cluster(
        request, DATADIR, 5433, {
        'timezone': 'UTC',
        'log_timezone': 'UTC'
        }
    )


@pytest.fixture(scope='session')
def engine(db):
    return create_engine(DBURI)


@pytest.fixture(scope='session')
def api(engine):
    if fschema:
        sch = fschema.formula_schema('test-api')
        sch.create(engine)
        sch = schema.tsschema('test-api-upstream')
        sch.create(engine)
    else:
        sch = schema.tsschema('test-api')
        sch.create(engine)
    return tsh_api.timeseries(
        DBURI,
        namespace='test-api'
    )


# multi-source

@pytest.fixture(scope='session')
def mapi(engine):
    sch = schema.tsschema('test-mapi')
    sch.create(engine)
    sch = schema.tsschema('test-mapi-2')
    sch.create(engine)
    o = tsh_api.multisourcetimeseries(
        DBURI, namespace='test-mapi'
    )
    o.addsource(
        DBURI, 'test-mapi-2'
    )
    return o


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
