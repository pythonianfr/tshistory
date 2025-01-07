from pathlib import Path
import shutil

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
    app as appmaker,
    client as http_client,
    server as http_server
)
from tshistory.http.util import nosecurity
from tshistory.storage import (
    FS1,
    Postgres
)
from tshistory.testutil import (
    make_tsx,
    tempconfig,
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

    config = (
        f'[dburi]\n'
        f'test = {str(engine.url)}\n'
    ).encode()
    with tempconfig(config):
        yield tsh_api.timeseries(
            DBURI,
            namespace='ns-test-mapi',
            handler=tsio.timeseries,
            sources={'remote': (DBURI, 'ns-test-mapi-2')}
        )


@pytest.fixture(scope='session')
def datadir():
    return DATADIR


@pytest.fixture(params=[('pg', 'tsh'), ('pg', 'z-z'), ('fs1', 'tsh')],
                scope='session')
def tsh(request, engine):
    driver, namespace = request.param
    schema.tsschema(namespace).create(engine, reset=True)

    if driver == 'fs1':
        datapath = DATADIR/namespace
        shutil.rmtree(datapath, ignore_errors=True)
        if not datapath.exists():
            datapath.mkdir()

        conf = (
            f'[dburi]\n'
            f'test = {DBURI}\n'
            f'[storage]\n'
            f'test = filesystem1\n'
            f'test.path = {datapath}'
        )
        with tempconfig(conf.encode()):
            yield tsio.timeseriesfs1(namespace, None, uri=DBURI)

    else:
        if namespace == 'z-z':
            Postgres._max_bucket_size = 5
            FS1._max_bucket_size = 5

            yield tsio.timeseries(namespace)

            Postgres._max_bucket_size = 150
            FS1._max_bucket_size = 150

        else:
            yield tsio.timeseries(namespace)


@pytest.fixture(scope='session')
def tsp(engine):
    schema.tsschema('tsh').create(engine, reset=True)
    yield tsio.timeseries('tsh')


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


@pytest.fixture()
def http(engine):
    schema.tsschema().create(engine, reset=True)
    schema.tsschema(ns='other').create(engine, reset=True)

    config = (
        f'[dburi]\n'
        f'test = {str(engine.url)}\n'
    ).encode()
    with tempconfig(config):
        tsa = tsh_api.timeseries(
            str(engine.url),
            handler=tsio.timeseries,
            namespace='tsh',
        sources={'other': (DBURI, 'other')}
    )

    # do a cleanup
    for ts in tsa.find('(by.everything)'):
        tsa.delete(str(ts))

    wsgi = nosecurity(
        appmaker.make_app(tsa)
    )
    yield NoRaiseWebTester(wsgi)


# http client

@pytest.fixture(scope='session')
def client(engine):
    schema.tsschema().create(engine, reset=True)
    schema.tsschema('tsh-upstream').create(engine, reset=True)
    schema.tsschema('other').create(engine, reset=True)

    uri = 'http://perdu.com'

    config = (
        f'[dburi]\n'
        f'test = {str(engine.url)}\n'
    ).encode()
    with tempconfig(config):
        wsgitester = WebTester(
            nosecurity(
                appmaker.make_app(
                    tsh_api.timeseries(
                        str(engine.url),
                        handler=tsio.timeseries,
                        sources={'other': (DBURI, 'other')}
                    ),
                    http_server.httpapi
                )
            )
        )
        with responses.RequestsMock(assert_all_requests_are_fired=False) as resp:
            with_http_bridge(uri, resp, wsgitester)
            yield http_client.httpclient(uri)


# federation api (direct + http)

def _initschema(engine):
    schema.tsschema().create(engine, reset=True)
    schema.tsschema('remote').create(engine, reset=True)


tsx = make_tsx(
    'http://test-uri',
    _initschema,
    tsio.timeseries,
    http_server.httpapi,
    http_client.httpclient,
    sources={'remote': (DBURI, 'remote')}
)


# for the alternative storage, we need a humble beginning


@pytest.fixture(scope='session')
def tsf(engine):
    ns = 'fsns'
    schema.tsschema(ns).create(engine, reset=True)
    datapath = DATADIR/ns
    shutil.rmtree(datapath, ignore_errors=True)
    if not datapath.exists():
        datapath.mkdir()

    conf = (
        f'[dburi]\n'
        f'test = {DBURI}\n'
        f'[storage]\n'
        f'test = filesystem1\n'
        f'test.path = {datapath}'
    )

    with tempconfig(conf.encode()):
        yield tsio.timeseriesfs1(ns, None, uri=DBURI)
