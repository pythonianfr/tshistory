from sqlalchemy import (Table, Column, Integer, String, MetaData, DateTime,
                        ForeignKey, PrimaryKeyConstraint)


meta = MetaData()

registry = Table(
    'registry', meta,
    Column('id', Integer, primary_key=True),
    Column('name', String, index=True, nullable=False, unique=True),
    Column('table_name', String, index=True, nullable=False, unique=True),
    schema='timeserie'
)


changeset = Table(
    'changeset', meta,
    Column('id', Integer, primary_key=True),
    Column('author', String, index=True, nullable=False),
    Column('insertion_date', DateTime, index=True, nullable=False),
    schema='timeserie'
)

changeset_series = Table(
    'changeset_series', meta,
    Column('csid', Integer, ForeignKey('timeserie.changeset.id'), nullable=False),
    Column('serie', String, ForeignKey('timeserie.registry.name'), nullable=False),
    PrimaryKeyConstraint('csid', 'serie', name='timeserie.changeset_series_pk'),
    schema='timeserie'
)
