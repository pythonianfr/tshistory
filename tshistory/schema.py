from pathlib import Path

from sqlhelp import sqlfile


SERIES = Path(__file__).parent / 'schema.sql'
GROUP = Path(__file__).parent / 'group.sql'


class tsschema(object):
    namespace = 'tsh'

    def __init__(self, ns='tsh'):
        self.namespace = ns

    def create(self, engine):
        self._create_series(engine, self.namespace)
        self._create_group(engine)

    def _create_series(self, engine, namespace):
        with engine.begin() as cn:
            # cleanup
            cn.execute(f'drop schema if exists "{namespace}" cascade')
            cn.execute(f'drop schema if exists "{namespace}.revision" cascade')
            cn.execute(f'drop schema if exists "{namespace}.snapshot" cascade')
            cn.execute(f'create schema "{namespace}"')
            cn.execute(f'create schema "{namespace}.revision"')
            cn.execute(f'create schema "{namespace}.snapshot"')
            # creation
            cn.execute(sqlfile(SERIES, ns=namespace))

    def _create_group(self, engine):
        # cleanup
        with engine.begin() as cn:
            cn.execute(f'drop schema if exists "{self.namespace}.group" cascade')

        # creation
        # dedicated time series store for the groups
        self._create_series(engine, f'{self.namespace}.group')

        # group registry & mapping
        with engine.begin() as cn:
            cn.execute(sqlfile(GROUP, ns=self.namespace))
