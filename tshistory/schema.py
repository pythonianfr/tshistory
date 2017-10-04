from sqlalchemy import (Table, Column, Integer, String, MetaData, DateTime,
                        ForeignKey, PrimaryKeyConstraint)
from sqlalchemy.dialects.postgresql import JSONB


meta = MetaData()

registry = Table(
    'registry', meta,
    Column('id', Integer, primary_key=True),
    Column('name', String, index=True, nullable=False, unique=True),
    Column('table_name', String, index=True, nullable=False, unique=True),
    Column('metadata', JSONB(none_as_null=True))
)


changeset = Table(
    'changeset', meta,
    Column('id', Integer, primary_key=True),
    Column('author', String, index=True, nullable=False),
    Column('insertion_date', DateTime, index=True, nullable=False),
    Column('metadata', JSONB(none_as_null=True))
)

changeset_series = Table(
    'changeset_series', meta,
    Column('csid', Integer, ForeignKey('changeset.id'),
           index=True, nullable=False),
    Column('serie', String, ForeignKey('registry.name'),
           index=True, nullable=False),
    PrimaryKeyConstraint('csid', 'serie', name='changeset_series_pk'),
)


def init(engine):
    from sqlalchemy.schema import CreateSchema
    engine.execute(CreateSchema('timeserie'))
    registry.create(engine)
    changeset.create(engine)
    changeset_series.create(engine)


def reset(engine):
    # explicitly cleanup the ts tables
    if registry.exists(engine):
        engine.execute('drop schema if exists timeserie cascade')
        changeset_series.drop(engine)
        registry.drop(engine)
        changeset.drop(engine)
