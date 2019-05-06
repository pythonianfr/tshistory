import logging
from threading import Lock
from pathlib import Path

from tshistory.util import unilist, sqlfile

L = logging.getLogger('tshistory.schema')
CREATEFILE = Path(__file__).parent / 'schema.sql'

# schemas registry
_SCHLOCK = Lock()
_SCHEMA_HANDLERS = unilist()


def register_schema(schema):
    for meth in ('define', 'exists', 'create', 'destroy'):
        getattr(schema, meth)
    with _SCHLOCK:
        if schema not in _SCHEMA_HANDLERS:
            _SCHEMA_HANDLERS.append(schema)


def init_schemas(engine, namespace='tsh'):
    for schema in _SCHEMA_HANDLERS:
        schema.define()
        schema.create(engine)


def reset_schemas(engine):
    for schema in reversed(_SCHEMA_HANDLERS):
        schema.destroy(engine)


def _delete_schema(engine, ns):
    with engine.begin() as cn:
        for subns in ('timeserie', 'snapshot'):
            cn.execute(
                'drop schema if exists "{}.{}" cascade'.format(ns, subns)
            )
        cn.execute('drop schema if exists "{}" cascade'.format(ns))


class tsschema(object):
    namespace = 'tsh'
    registry = None
    changeset = None
    changeset_series = None
    SCHEMAS = {}

    def __new__(cls, namespace='tsh'):
        # singleton-per-namespace handling
        with _SCHLOCK:
            if namespace in cls.SCHEMAS:
                return cls.SCHEMAS[namespace]
        return super(tsschema, cls).__new__(cls)

    def __init__(self, namespace='tsh'):
        self.namespace = namespace
        register_schema(self)

    def define(self):
        with _SCHLOCK:
            if self.namespace in self.SCHEMAS:
                return
        L.info('build schema %s', self.namespace)
        with _SCHLOCK:
            self.SCHEMAS[self.namespace] = self

    def exists(self, engine):
        return engine.execute(
            'select exists('
            '  select schema_name '
            '  from information_schema.schemata '
            '  where schema_name = %(name)s'
            ')',
            name=self.namespace
        ).scalar()

    def create(self, engine):
        L.info('create schema %s %s', self.namespace, self.exists(engine))
        if self.exists(engine):
            if self.namespace != 'tsh':
                L.warning('cannot create already existing namespace %s',
                          self.namespace)
            return
        engine.execute(f'create schema if not exists "{self.namespace}"')
        engine.execute(f'create schema if not exists "{self.namespace}.timeserie"')
        engine.execute(f'create schema if not exists "{self.namespace}.snapshot"')
        engine.execute(sqlfile(CREATEFILE, ns=self.namespace))

    def destroy(self, engine):
        L.info('destroy schema %s', self.namespace)
        _delete_schema(engine, self.namespace)
        self.SCHEMAS.pop(self.namespace, None)
