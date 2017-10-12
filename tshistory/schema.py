from sqlalchemy import (Table, Column, Integer, String, MetaData, DateTime,
                        ForeignKey, PrimaryKeyConstraint)
from sqlalchemy.dialects.postgresql import JSONB


meta = MetaData()

def make_schema(meta, namespace='tsh'):

    registry = Table(
        'registry', meta,
        Column('id', Integer, primary_key=True),
        Column('name', String, index=True, nullable=False, unique=True),
        Column('table_name', String, index=True, nullable=False, unique=True),
        Column('metadata', JSONB(none_as_null=True)),
        schema=namespace
    )


    changeset = Table(
        'changeset', meta,
        Column('id', Integer, primary_key=True),
        Column('author', String, index=True, nullable=False),
        Column('insertion_date', DateTime, index=True, nullable=False),
        Column('metadata', JSONB(none_as_null=True)),
        schema=namespace
    )

    changeset_series = Table(
        'changeset_series', meta,
        Column('csid', Integer, ForeignKey('{}.changeset.id'.format(namespace)),
               index=True, nullable=False),
        Column('serie', String, ForeignKey('{}.registry.name'.format(namespace)),
               index=True, nullable=False),
        PrimaryKeyConstraint('csid', 'serie',
                             name='{}_changeset_series_pk'.format(namespace)),
        schema=namespace
    )

    return registry, changeset, changeset_series


# default schema
registry, changeset, changeset_series = make_schema(meta)


def init(engine, namespace='tsh'):
    from sqlalchemy.schema import CreateSchema
    engine.execute(CreateSchema(namespace))
    engine.execute(CreateSchema('{}.timeserie'.format(namespace)))
    registry.create(engine)
    changeset.create(engine)
    changeset_series.create(engine)


def reset(engine, namespace='tsh'):
    engine.execute('drop schema if exists "{}.timeserie" cascade'.format(namespace))
    engine.execute('drop schema if exists {} cascade'.format(namespace))
