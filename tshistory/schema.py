import logging
from threading import Lock

from sqlalchemy import (Table, Column, Integer, String, MetaData, TIMESTAMP,
                        ForeignKey, UniqueConstraint)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.schema import CreateSchema

from tshistory.util import unilist

L = logging.getLogger('tshistory.schema')

# schemas registry
_SCHLOCK = Lock()
_SCHEMA_HANDLERS = unilist()


def register_schema(schema):
    for meth in ('define', 'exists', 'create', 'destroy'):
        getattr(schema, meth)
    with _SCHLOCK:
        if schema not in _SCHEMA_HANDLERS:
            _SCHEMA_HANDLERS.append(schema)


def init_schemas(engine, meta, namespace='tsh'):
    for schema in _SCHEMA_HANDLERS:
        schema.define(meta)
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
    meta = None
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

    def define(self, meta=MetaData()):
        with _SCHLOCK:
            if self.namespace in self.SCHEMAS:
                return
        L.info('build schema %s', self.namespace)
        self.meta = meta
        registry = Table(
            'registry', meta,
            Column('id', Integer, primary_key=True),
            Column('seriename', String, index=True, nullable=False, unique=True),
            Column('table_name', String, index=True,
                   nullable=False, unique=True),
            Column('metadata', JSONB(none_as_null=True)),
            schema=self.namespace,
            keep_existing=True
        )

        changeset = Table(
            'changeset', meta,
            Column('id', Integer, primary_key=True),
            Column('author', String, index=True, nullable=False),
            Column('insertion_date', TIMESTAMP(timezone=True), index=True, nullable=False),
            Column('metadata', JSONB(none_as_null=True)),
            schema=self.namespace,
            keep_existing=True
        )

        changeset_series = Table(
            'changeset_series', meta,
            Column('cset', Integer,
                   ForeignKey('{}.changeset.id'.format(self.namespace), ondelete='set null'),
                   index=True, nullable=True),
            Column('serie', Integer,
                   ForeignKey('{}.registry.id'.format(self.namespace), ondelete='cascade'),
                   index=True, nullable=False),
            UniqueConstraint(
                'cset', 'serie',
                name='{}_changeset_series_unique'.format(self.namespace)),
            schema=self.namespace,
            keep_existing=True
        )

        self.registry = registry
        self.changeset = changeset
        self.changeset_series = changeset_series
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
        engine.execute(CreateSchema(self.namespace))
        engine.execute(CreateSchema('{}.timeserie'.format(self.namespace)))
        engine.execute(CreateSchema('{}.snapshot'.format(self.namespace)))
        self.registry.create(engine)
        self.changeset.create(engine)
        self.changeset_series.create(engine)

    def destroy(self, engine):
        L.info('destroy schema %s', self.namespace)
        _delete_schema(engine, self.namespace)
        self.SCHEMAS.pop(self.namespace, None)
