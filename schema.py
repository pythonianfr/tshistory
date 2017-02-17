from sqlalchemy import (Table, Column, Integer, String, DateTime,
                        MetaData, ForeignKey, UniqueConstraint)
from sqlalchemy.dialects.postgresql import JSONB

meta = MetaData()


ts_revlog = Table(
    'ts_revlog', meta,
    Column('id', Integer, primary_key=True),
    Column('name', String, index=True, nullable=False),
    Column('author', String, index=True, nullable=False),
    Column('insertion_date', DateTime, index=True, nullable=False),
    Column('data', JSONB, nullable=False),
    Column('snapshot', JSONB),
    Column('parent',
           Integer,
           ForeignKey('ts_revlog.id', ondelete='cascade'),
           nullable=True,
           index=True),
    UniqueConstraint('name', 'parent')
)
