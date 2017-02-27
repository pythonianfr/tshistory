from sqlalchemy import Table, Column, Integer, String, MetaData


meta = MetaData()

ts_registry = Table(
    'ts_registry', meta,
    Column('id', Integer, primary_key=True),
    Column('name', String, index=True, nullable=False, unique=True),
    Column('table_name', String, index=True, nullable=False, unique=True)
)
