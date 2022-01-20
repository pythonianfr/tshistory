from pathlib import Path

from sqlhelp import sqlfile


SERIES = Path(__file__).parent / 'schema.sql'
GROUP = Path(__file__).parent / 'group.sql'


class tsschema(object):
    namespace = 'tsh'

    def __init__(self, ns='tsh'):
        self.namespace = ns

    def create(self, engine, reset=False):
        self._create_series(engine, self.namespace)
        self._create_group(engine, reset=reset)

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

    def _create_group(self, engine, reset=False):
        # cleanup
        exists = engine.execute(
            'select 1 from information_schema.schemata where schema_name = %(name)s',
            name=f'{self.namespace}.group'
        ).scalar()
        if exists and not reset:
            print(f'The "{self.namespace}.group" namespace already exists.')
            return

        if reset:
            # undo the contents of group.sql
            with engine.begin() as cn:
                cn.execute(f'drop table if exists "{self.namespace}".group_registry')
                cn.execute(f'drop table if exists "{self.namespace}".groupmap')

        with engine.begin() as cn:
            cn.execute(f'drop schema if exists "{self.namespace}.group" cascade')

        # creation
        # dedicated time series store for the groups
        self._create_series(engine, f'{self.namespace}.group')

        # group registry & mapping
        with engine.begin() as cn:
            cn.execute(sqlfile(GROUP, ns=self.namespace))
