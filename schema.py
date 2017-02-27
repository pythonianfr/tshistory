from sqlalchemy import (Table, Column, Integer, String, DateTime,
                        MetaData, ForeignKey, UniqueConstraint,
                        exc)
from sqlalchemy.dialects.postgresql import JSONB

meta = MetaData()


ts_registry = Table(
    'ts_registry', meta,
    Column('id', Integer, primary_key=True),
    Column('name', String, index=True, nullable=False, unique=True),
    Column('table_name', String, index=True, nullable=False, unique=True)
)


def ts_table_name(name):
    return 'ts_%s' % name


def make_ts_table(cnx, name):
    tablename = ts_table_name(name)
    table = Table(
        tablename, meta,
        Column('id', Integer, primary_key=True),
        Column('author', String, index=True, nullable=False),
        Column('insertion_date', DateTime, index=True, nullable=False),
        Column('data', JSONB, nullable=False),
        Column('snapshot', JSONB),
        Column('parent',
               Integer,
               ForeignKey('%s.id' % tablename, ondelete='cascade'),
               nullable=True,
               unique=True,
               index=True),
    )
    table.create(cnx)
    sql = ts_registry.insert().values(name=name,
                                      table_name=tablename)
    cnx.execute(sql)
    return table


def get_ts_table(cnx, name):
    sql = ts_registry.select().where(ts_registry.c.name==name)
    tid = cnx.execute(sql).scalar()
    if tid:
        return Table(ts_table_name(name), meta,
                     autoload=True, autoload_with=cnx.engine)
