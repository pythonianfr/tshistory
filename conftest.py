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
    client as http_client,
    server as http_server
)
from tshistory.storage import Postgres
from tshistory.testutil import (
    make_tsx,
    with_http_bridge,
    WebTester
)


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
    schema.tsschema('ns-test-mapi-2').create(engine, reset=True)

    return tsh_api.timeseries(
        DBURI,
        namespace='ns-test-mapi',
        handler=tsio.timeseries,
        sources={'remote': (DBURI, 'ns-test-mapi-2')}
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
def cleanup(engine, tsh):
    with engine.begin() as cn:
        for name in tsh.list_series(engine):
            tsh.delete(cn, name)


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
            sources={'other': (DBURI, 'other')}
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

    wsgitester = WebTester(
        app.make_app(
            tsh_api.timeseries(
                str(engine.url),
                handler=tsio.timeseries,
                sources={'other': (DBURI, 'other')}
            ),
            http_server.httpapi
        )
    )
    with responses.RequestsMock(assert_all_requests_are_fired=False) as resp:
        with_http_bridge(uri, resp, wsgitester)
        yield http_client.httpclient(uri)


# federation api (direct + http)

def _initschema(engine):
    schema.tsschema().create(engine, reset=True)


tsx = make_tsx(
    'http://test-uri',
    _initschema,
    tsio.timeseries,
    http_server.httpapi,
    http_client.httpclient
)
