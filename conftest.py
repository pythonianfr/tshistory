import io
from pathlib import Path
from functools import partial

from sqlalchemy import create_engine
import pandas as pd

import pytest
import webtest
import responses
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


# api fixtures

@pytest.fixture(scope='session')
def pgapi(engine):
    schema.tsschema('tsh').create(engine)
    fschema.formula_schema('tsh').create(engine)
    schema.tsschema('tsh-upstream').create(engine)
    return tsh_api.timeseries(str(engine.url), 'tsh')

# multi-source

@pytest.fixture(scope='session')
def mapi(engine):
    schema.tsschema('ns-test-mapi').create(engine)
    fschema.formula_schema('ns-test-mapi').create(engine)
    schema.tsschema('ns-test-mapi-2').create(engine)
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


# http api

class WebTester(webtest.TestApp):

    def _check_status(self, status, res):
        try:
            super(WebTester, self)._check_status(status, res)
        except:
            print(res.errors)
            # raise <- default behaviour on 4xx is silly

    def _gen_request(self, method, url, params,
                     headers=None,
                     extra_environ=None,
                     status=None,
                     upload_files=None,
                     expect_errors=False,
                     content_type=None):
        """
        Do a generic request.
        PATCH: *bypass* all transformation as params comes
               straight from a prepared (python-requests) request.
        """
        environ = self._make_environ(extra_environ)

        environ['REQUEST_METHOD'] = str(method)
        url = str(url)
        url = self._remove_fragment(url)
        req = self.RequestClass.blank(url, environ)

        req.environ['wsgi.input'] = io.BytesIO(params.encode('utf-8'))
        req.content_length = len(params)
        if headers:
            req.headers.update(headers)
        return self.do_request(req, status=status,
                               expect_errors=expect_errors)


def read_request_bridge(client, request):
    resp = client.get(request.url,
                      params=request.body,
                      headers=request.headers)
    return (resp.status_code, resp.headers, resp.body)


def write_request_bridge(method):
    def bridge(request):
        resp = method(request.url,
                      params=request.body,
                      headers=request.headers)
        return (resp.status_code, resp.headers, resp.body)
    return bridge


URI = 'http://test-uri'

def with_tester(uri, resp, wsgitester):
    resp.add_callback(
        responses.GET, uri + '/series/state',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.GET, uri + '/series/staircase',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.GET, uri + '/series/history',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.GET, uri + '/series/catalog',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.PATCH, uri + '/series/state',
        callback=write_request_bridge(wsgitester.patch)
    )

    resp.add_callback(
        responses.GET, uri + '/series/metadata',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.PUT, uri + '/series/metadata',
        callback=write_request_bridge(wsgitester.put)
    )

    resp.add_callback(
        responses.GET, uri + '/series/insertion_dates',
        callback=partial(read_request_bridge, wsgitester)
    )


@pytest.fixture(scope='session')
def httpapi(engine):
    from tshistory_rest import app
    wsgitester = WebTester(
        app.make_app(
            tsh_api.timeseries(
                str(engine.url),
                sources=[(DBURI, 'ns-test-mapi-2')]
            )
        )
    )
    with responses.RequestsMock(assert_all_requests_are_fired=False) as resp:
        with_tester(URI, resp, wsgitester)
        # yields local-tsh + http ns-test-mapi-2
        yield tsh_api.timeseries(URI, 'tsh')


# formula test
URI2 = 'http://test-uri2'

@pytest.fixture(scope='session')
def mapihttp(engine):
    from tshistory_rest import app
    from tshistory_formula import tsio
    schema.tsschema('ns-test-local').create(engine)
    fschema.formula_schema('ns-test-local').create(engine)
    schema.tsschema('ns-test-remote').create(engine)
    fschema.formula_schema('ns-test-remote').create(engine)
    wsgitester = WebTester(
        app.make_app(
            tsh_api.timeseries(
                DBURI,
                namespace='ns-test-remote',
                handler=tsio.timeseries
            )
        )
    )
    with responses.RequestsMock(assert_all_requests_are_fired=False) as resp:
        with_tester(URI2, resp, wsgitester)

        yield tsh_api.timeseries(
            DBURI,
            namespace='ns-test-local',
            handler=tsio.timeseries,
            sources=[
                (URI2, 'ns-test-remote'),
                ('http://unavailable', 'ns-test-unavailable-remote')
            ]
        )
