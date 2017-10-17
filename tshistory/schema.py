import logging

from sqlalchemy import (Table, Column, Integer, String, MetaData, DateTime,
                        ForeignKey, PrimaryKeyConstraint)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.schema import CreateSchema


L = logging.getLogger('tshistory.schema')
meta = MetaData()

# schemas registry
SCHEMAS = {}


class tsschema(object):
    namespace = 'tsh'
    meta = None
    registry = None
    changeset = None
    changeset_series = None

    def __init__(self, namespace='tsh'):
        self.namespace = namespace

    def build(self, meta):
        L.info('build schema %s', self.namespace)
        self.meta = meta
        registry = Table(
            'registry', meta,
            Column('id', Integer, primary_key=True),
            Column('name', String, index=True, nullable=False, unique=True),
            Column('table_name', String, index=True, nullable=False, unique=True),
            Column('metadata', JSONB(none_as_null=True)),
            schema=self.namespace
        )

        changeset = Table(
            'changeset', meta,
            Column('id', Integer, primary_key=True),
            Column('author', String, index=True, nullable=False),
            Column('insertion_date', DateTime, index=True, nullable=False),
            Column('metadata', JSONB(none_as_null=True)),
            schema=self.namespace
        )

        changeset_series = Table(
            'changeset_series', meta,
            Column('csid', Integer,
                   ForeignKey('{}.changeset.id'.format(self.namespace)),
                   index=True, nullable=False),
            Column('serie', String,
                   ForeignKey('{}.registry.name'.format(self.namespace)),
                   index=True, nullable=False),
            PrimaryKeyConstraint(
                'csid', 'serie',
                name='{}_changeset_series_pk'.format(self.namespace)),
            schema=self.namespace
        )

        self.registry = registry
        self.changeset = changeset
        self.changeset_series = changeset_series
        SCHEMAS[self.namespace] = self

    def exists(self, engine):
        return engine.execute('select exists(select schema_name '
                              'from information_schema.schemata '
                              'where schema_name = %(name)s)',
                              name=self.namespace
        ).scalar()

    def create(self, engine):
        L.info('create schema %s %s', self.namespace, self.exists(engine))
        engine.execute(CreateSchema(self.namespace))
        engine.execute(CreateSchema('{}.timeserie'.format(self.namespace)))
        self.registry.create(engine)
        self.changeset.create(engine)
        self.changeset_series.create(engine)

    def destroy(self, engine):
        L.info('destroy schema %s', self.namespace)
        engine.execute('drop schema if exists "{}.timeserie" cascade'.format(self.namespace))
        engine.execute('drop schema if exists {} cascade'.format(self.namespace))
        del self.meta
        del self.registry
        del self.changeset
        del self.changeset_series


# create and register default db structure
tsschema().build(meta)


def init(engine, meta, namespace='tsh'):
    schem = tsschema(namespace)
    if schem.exists(engine):
        L.warning('cannot create already existing namespace %s', namespace)
        return
    schem.build(meta)
    schem.create(engine)


def reset(engine, namespace='tsh'):
    if namespace not in SCHEMAS:
        L.warning('unknown ns %s cannot be reset', namespace)
        return
    schem = SCHEMAS.pop(namespace)
    schem.destroy(engine)
