from sqlalchemy import Table, Column, Integer, String, MetaData, DateTime


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
