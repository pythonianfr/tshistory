import logging
from threading import Lock

from sqlalchemy import (Table, Column, Integer, String, MetaData, TIMESTAMP,
                        ForeignKey, UniqueConstraint)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.schema import CreateSchema


L = logging.getLogger('tshistory.schema')

# schemas registry

SCHLOCK = Lock()
SCHEMAS = {}
meta = MetaData()

def delete_schema(engine, ns):
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

    def __new__(cls, namespace='tsh'):
        with SCHLOCK:
            if namespace in SCHEMAS:
                return SCHEMAS[namespace]
        return super(tsschema, cls).__new__(cls)

    def __init__(self, namespace='tsh'):
        self.namespace = namespace

    def define(self, meta=MetaData()):
        with SCHLOCK:
            if self.namespace in SCHEMAS:
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
        with SCHLOCK:
            SCHEMAS[self.namespace] = self

    def exists(self, engine):
        return engine.execute('select exists(select schema_name '
                              'from information_schema.schemata '
                              'where schema_name = %(name)s)',
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
        delete_schema(engine, self.namespace)
        del self.meta
        del self.registry
        del self.changeset
        del self.changeset_series


def init(engine, meta, namespace='tsh'):
    schem = tsschema(namespace)
    schem.define(meta)
    schem.create(engine)


def reset(engine, namespace='tsh'):
    SCHEMAS.pop(namespace, None)
    delete_schema(engine, namespace)
