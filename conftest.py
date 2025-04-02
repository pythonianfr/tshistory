from pathlib import Path
import shutil

from sqlhelp.pgapi import pgdb
from sqlhelp.testutil import setup_local_pg_cluster
import pandas as pd
import webtest

from dbcache import api as storeapi
import pytest
import responses
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
DBURI = 'postgresql://localhost:5434/postgres'


@pytest.fixture(scope='session')
def db(request):
    setup_local_pg_cluster(
        request, DATADIR, 5434, {
        'timezone': 'UTC',
        'log_timezone': 'UTC'
        }
    )


@pytest.fixture(scope='session')
def engine(db):
    return pgdb(DBURI)


# api fixtures
# multi-source

@pytest.fixture(scope='session')
def mapi(engine):
    schema.tsschema('ns-test-mapi').create(engine)
    schema.tsschema('ns-test-mapi-2').create(engine)

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
    schema.tsschema(namespace).create(engine)
    kvstore = storeapi.kvstore(  # noqa
        DBURI,
        namespace=f'{namespace}-kvstore'
    )

    if driver == 'fs1':
        datapath = DATADIR / 'fs1' / namespace
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
            yield tsio.timeseriesfs1(namespace, _kvstore=kvstore, uri=DBURI)

    else:
        if namespace == 'z-z':
            Postgres._max_bucket_size = 5
            FS1._max_bucket_size = 5

            yield tsio.timeseries(namespace, _kvstore=kvstore)

            Postgres._max_bucket_size = 150
            FS1._max_bucket_size = 150

        else:
            yield tsio.timeseries(namespace, _kvstore=kvstore)


@pytest.fixture(scope='session')
def tsp(engine):
    schema.tsschema('tsh').create(engine)
    yield tsio.timeseries('tsh')


@pytest.fixture(scope='session')
def pure(engine):
    schema.tsschema('pure').create(engine)
    yield tsio.timeseries('pure')


@pytest.fixture(scope='session')
def cleanup(engine, tsh):
    with engine.begin() as cn:
        for name in tsh.list_series(cn):
            tsh.delete(cn, name)


@pytest.fixture(scope='session')
def ptsh(engine):
    schema.tsschema().create(engine)
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
def tsa(engine):
    schema.tsschema().create(engine)
    schema.tsschema(ns='other').create(engine)

    config = (
        f'[dburi]\n'
        f'test = {str(engine.url)}\n'
        f'[sources]\n'
        f'test.source1=http://source1.com\n'
        f'test.source2=http://source2.com\n'
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

        yield tsa


@pytest.fixture()
def http(tsa):
    wsgi = nosecurity(
        appmaker.make_app(tsa)
    )
    yield NoRaiseWebTester(wsgi)


# http client

@pytest.fixture(scope='session')
def client(engine):
    schema.tsschema().create(engine)
    schema.tsschema('tsh-upstream').create(engine) # XXX
    schema.tsschema('other').create(engine)

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
    schema.tsschema().create(engine)
    schema.tsschema('remote').create(engine)


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
    schema.tsschema(ns).create(engine)
    datapath = DATADIR / 'fs1' / ns
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
