from sqlalchemy import (Table, Column, Integer, String, MetaData, DateTime,
                        ForeignKey, PrimaryKeyConstraint)


meta = MetaData()

ts_registry = Table(
    'ts_registry', meta,
    Column('id', Integer, primary_key=True),
    Column('name', String, index=True, nullable=False, unique=True),
    Column('table_name', String, index=True, nullable=False, unique=True)
)


ts_changeset = Table(
    'ts_changeset', meta,
    Column('id', Integer, primary_key=True),
    Column('author', String, index=True, nullable=False),
    Column('insertion_date', DateTime, index=True, nullable=False)
)

ts_changeset_series = Table(
    'ts_changeset_series', meta,
    Column('csid', Integer, ForeignKey('ts_changeset.id'), nullable=False),
    Column('serie', String, ForeignKey('ts_registry.name'), nullable=False),
    PrimaryKeyConstraint('csid', 'serie', name='ts_changeset_series_pk')
)
