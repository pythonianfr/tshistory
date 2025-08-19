from pathlib import Path

from dbcache import (
    api as kvapi,
    schema as kvschema
)
from sqlhelp import sqlfile

from tshistory import __version__


BASE = Path(__file__).parent / 'schema.sql'
SERIES = Path(__file__).parent / 'registry.sql'
GROUP = Path(__file__).parent / 'group.sql'


class tsschema(object):
    namespace = 'tsh'

    def __init__(self, ns='tsh'):
        self.namespace = ns

    def create(self, engine, base=True, groups=True, stores=True):
        self._create_series(engine, self.namespace)
        if stores:
            self._create_kvstore(engine, self.namespace)
        if base:
            self._create_base(engine)
        if groups:
            self._create_groups(engine)

    def _create_base(self, engine):
        with engine.begin() as cn:
            cn.execute(sqlfile(BASE, ns=self.namespace), _binary=False)

    def _create_series(self, engine, namespace):
        with engine.begin() as cn:
            cn.execute(f'drop schema if exists "{namespace}" cascade')
            cn.execute(f'drop schema if exists "{namespace}.revision" cascade')
            cn.execute(f'drop schema if exists "{namespace}.snapshot" cascade')
            cn.execute(f'create schema "{namespace}"')
            cn.execute(f'create schema "{namespace}.revision"')
            cn.execute(f'create schema "{namespace}.snapshot"')
            cn.execute(sqlfile(SERIES, ns=namespace), _binary=False)

    def _create_groups(self, engine):
        # dedicated time series store for the groups
        with engine.begin() as cn:
            cn.execute(f'drop schema if exists "{self.namespace}.group" cascade')
            cn.execute(f'create schema "{self.namespace}.group"')
        self._create_series(engine, f'{self.namespace}.group')
        # group registry & mapping
        with engine.begin() as cn:
            cn.execute(sqlfile(GROUP, ns=self.namespace), _binary=False)

    def _create_kvstore(self, engine, namespace):
        ns = f'{namespace}-kvstore'
        kvschema.init(engine, ns=ns)
        kvstore = kvapi.kvstore(str(engine.url), namespace=ns)
        kvstore.set('tshistory-version', __version__)
