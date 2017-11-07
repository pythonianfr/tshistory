from __future__ import print_function

from pkg_resources import iter_entry_points
import logging

from dateutil import parser
import json
from pathlib2 import Path

import click
from click_plugins import with_plugins
from sqlalchemy import create_engine
from dateutil.parser import parse as temporal
import pandas as pd

from tshistory.tsio import TimeSerie, fromjson
from tshistory.db import dump as dump_db, restore as restore_db
from tshistory.schema import init as init_schema, meta


TSH = TimeSerie()


# Override points
# * for the log command

REVFMT = """
revision: {rev}
author:   {author}
date:     {date}
""".strip()


def format_rev(rev):
    fmt = REVFMT + '\n'
    if rev.get('diff'):
        fmt += 'series: {names}\n\n'
        lines = []
        for ts in rev['diff'].values():
            lines.append(ts.to_string())
        fmt += '\n'.join(lines)
    else:
        fmt += 'series:   {names}'

    return fmt.format(**rev)


# * for the restore command

def read_and_insert(cn, tsh, json_cset):
    rev_data = json.loads(json_cset)
    date = parser.parse(rev_data['date'])
    author = rev_data['author']
    with tsh.newchangeset(cn, author, _insertion_date=date):
        for name in rev_data['names']:
            ts = fromjson(rev_data['diff'][name], name)
            ts.name = name
            tsh.insert(cn, ts, name)


def additional_dumping(dburi, dump_path):
    return


def additional_restoring(path_dump, dburi):
    return


@with_plugins(iter_entry_points('tshistory.subcommands'))
@click.group()
def tsh():
    pass


@tsh.command()
@click.argument('db-uri')
@click.argument('seriename')
@click.option('--json', is_flag=True, default=False)
def get(db_uri, seriename, json):
    """show a serie in its current state """
    engine = create_engine(db_uri)

    ts = TSH.get(engine, seriename)
    if json:
        print(ts.to_json())
    else:
        with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
            print(ts)


@tsh.command()
@click.argument('db-uri')
@click.argument('seriename')
@click.option('--json', is_flag=True, default=False)
@click.option('--from-insertion-date', type=temporal)
@click.option('--to-insertion-date', type=temporal)
@click.option('--from-value-date', type=temporal)
@click.option('--to-value-date', type=temporal)
@click.option('--diff/--no-diff', is_flag=True, default=True)
def history(db_uri, seriename,
            from_insertion_date, to_insertion_date,
            from_value_date, to_value_date,
            diff, json):
    """show a serie full history """
    engine = create_engine(db_uri)

    ts = TSH.get_history(engine, seriename,
                         from_insertion_date, to_insertion_date,
                         from_value_date, to_value_date,
                         diffmode=diff)
    if json:
        print(ts.to_json())
    else:
        with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
            print(ts)


@tsh.command()
@click.argument('db-uri')
@click.option('--limit', '-l', default=None)
@click.option('--show-diff', is_flag=True, default=False)
@click.option('--serie', '-s', multiple=True)
@click.option('--from-rev')
@click.option('--to-rev')
@click.option('--from-insertion-date', type=temporal)
@click.option('--to-insertion-date', type=temporal)
def log(db_uri, limit, show_diff, serie, from_rev, to_rev,
        from_insertion_date, to_insertion_date):
    """show revision history of entire repository or series"""
    engine = create_engine(db_uri)

    for rev in TSH.log(engine, limit=limit, diff=show_diff, names=serie,
                       fromrev=from_rev, torev=to_rev,
                       fromdate=from_insertion_date, todate=to_insertion_date):
        rev['names'] = ','.join(rev['names'])
        print(format_rev(rev))
        print()


INFOFMT = """
changeset count: {changeset count}
series count:    {series count}
series names:    {serie names}
""".strip()


@tsh.command()
@click.argument('db-uri')
def info(db_uri):
    """show global statistics of the repository"""
    engine = create_engine(db_uri)

    info = TSH.info(engine)
    info['serie names'] = ', '.join(info['serie names'])
    print(INFOFMT.format(**info))


@tsh.command()
@click.argument('db-uri')
@click.argument('dump-path')
def dump(db_uri, dump_path):
    """dump all time series revisions in a zip file"""
    dump_path = Path(dump_path)
    out_path = dump_db(db_uri, dump_path, TSH, additional_dumping)
    print('db dump avaible at %s' % out_path)


def verbose_logs():
    logger = logging.getLogger('tshistory.tsio')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)


@tsh.command()
@click.argument('out-path')
@click.argument('db-uri')
def restore(out_path, db_uri):
    """restore zip file in a freshly initialized database (see init_db command)"""
    verbose_logs()
    restore_db(out_path, db_uri, TSH, read_and_insert, additional_restoring)


@tsh.command(name='init-db')
@click.argument('db-uri')
def init_db(db_uri):
    """initialize an new db."""
    init_schema(create_engine(db_uri), meta)


if __name__ == '__main__':
    tsh()
