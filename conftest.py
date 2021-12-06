import io
from pathlib import Path

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
from tshistory.testutil import with_tester

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

        if isinstance(params, str):
            params = params.encode('utf-8')
        req.environ['wsgi.input'] = io.BytesIO(params)
        req.content_length = len(params)
        if headers:
            req.headers.update(headers)
        return self.do_request(req, status=status,
                               expect_errors=expect_errors)


URI = 'http://test-uri'


@pytest.fixture(params=['pg', 'http'])
def tsx(request, engine):
    schema.tsschema().create(engine)
    fschema.formula_schema().create(engine)

    from tshistory_formula import tsio

    if request.param == 'pg':

        yield tsh_api.timeseries(
            str(engine.url),
            handler=tsio.timeseries
        )

    else:

        from tshistory_rest import app
        from tshistory_formula.http import formula_httpapi
        wsgitester = WebTester(
            app.make_app(
                tsh_api.timeseries(
                    str(engine.url),
                    handler=tsio.timeseries
                ),
                formula_httpapi
            )
        )
        with responses.RequestsMock(assert_all_requests_are_fired=False) as resp:
            with_tester(URI, resp, wsgitester)
            yield tsh_api.timeseries(URI, 'tsh', handler=tsio.timeseries)


# formula test
URI2 = 'http://test-uri2'

@pytest.fixture(scope='session')
def mapihttp(engine):
    from tshistory_rest import app
    from tshistory_formula import tsio
    from tshistory_formula.http import formula_httpapi
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
            ),
            formula_httpapi
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
