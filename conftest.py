from pathlib import Path

from sqlalchemy import create_engine
import pandas as pd
import webtest

import pytest
import responses
from pytest_sa_pg import db as dbsetup
from click.testing import CliRunner

from tshistory import (
    api as tsh_api,
    cli as command,
    schema,
    tsio
)
from tshistory.http import (
    app,
    client as http_client
)
from tshistory.storage import Postgres
from tshistory.testutil import (
    make_tsx,
    with_tester,
    WebTester
)


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


# api fixtures
# multi-source

@pytest.fixture(scope='session')
def mapi(engine):
    schema.tsschema('ns-test-mapi').create(engine, reset=True)
    fschema.formula_schema('ns-test-mapi').create(engine)
    schema.tsschema('ns-test-mapi-2').create(engine, reset=True)
    fschema.formula_schema('ns-test-mapi-2').create(engine)

    return tsh_api.timeseries(
        DBURI,
        namespace='ns-test-mapi',
        handler=tsio.timeseries,
        sources=[
            (DBURI, 'ns-test-mapi-2')
        ]
    )


@pytest.fixture(scope='session')
def datadir():
    return DATADIR


@pytest.fixture(params=['tsh', 'z-z'],
                scope='session')
def tsh(request, engine):
    namespace = request.param
    schema.tsschema(namespace).create(engine, reset=True)

    if namespace == 'z-z':
        Postgres._max_bucket_size = 5
    yield tsio.timeseries(namespace)


@pytest.fixture(scope='session')
def ptsh(engine):
    schema.tsschema().create(engine, reset=True)
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

# http server

class NoRaiseWebTester(webtest.TestApp):

    def _check_status(self, status, res):
        try:
            super()._check_status(status, res)
        except:
            print('ERRORS', res.errors)
            # raise <- default behaviour on 4xx is silly


@pytest.fixture(scope='session')
def http(engine):
    schema.tsschema().create(engine, reset=True)
    schema.tsschema(ns='other').create(engine, reset=True)

    wsgi = app.make_app(
        tsh_api.timeseries(
            str(engine.url),
            handler=tsio.timeseries,
            namespace='tsh',
            sources=[(DBURI, 'other')]
        )
    )
    yield NoRaiseWebTester(wsgi)


# http client

@pytest.fixture(scope='session')
def client(engine):
    schema.tsschema().create(engine, reset=True)
    schema.tsschema('tsh-upstream').create(engine, reset=True)
    schema.tsschema('other').create(engine, reset=True)

    uri = 'http://perdu.com'

    from tshistory_formula.http import formula_httpapi
    wsgitester = WebTester(
        app.make_app(
            tsh_api.timeseries(
                str(engine.url),
                handler=tsio.timeseries,
                sources=[(DBURI, 'other')]
            ),
            formula_httpapi
        )
    )
    with responses.RequestsMock(assert_all_requests_are_fired=False) as resp:
        with_tester(uri, resp, wsgitester)
        yield http_client.Client(uri)


# federation api (direct + http)

def _initschema(engine):
    schema.tsschema().create(engine, reset=True)
    fschema.formula_schema().create(engine)


from tshistory_formula.http import formula_httpapi, FormulaClient
from tshistory_formula.tsio import timeseries as formula_timeseries

tsx = make_tsx(
    'http://test-uri',
    _initschema,
    formula_timeseries,
    formula_httpapi,
    FormulaClient
)


# formula test
URI2 = 'http://test-uri2'

@pytest.fixture(scope='session')
def mapihttp(engine):
    from tshistory_formula import tsio
    schema.tsschema('ns-test-local').create(engine, reset=True)
    fschema.formula_schema('ns-test-local').create(engine)
    schema.tsschema('ns-test-remote').create(engine, reset=True)
    fschema.formula_schema('ns-test-remote').create(engine)

    wsgitester = WebTester(
        app.make_app(
            tsh_api.timeseries(
                DBURI,
                namespace='ns-test-remote',
                handler=formula_timeseries
            ),
            formula_httpapi
        )
    )
    with responses.RequestsMock(assert_all_requests_are_fired=False) as resp:
        with_tester(URI2, resp, wsgitester)

        yield tsh_api.timeseries(
            DBURI,
            namespace='ns-test-local',
            handler=formula_timeseries,
            sources=[
                (URI2, 'ns-test-remote'),
                ('http://unavailable', 'ns-test-unavailable-remote')
            ]
        )
