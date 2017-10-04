from pkg_resources import iter_entry_points

import click
from click_plugins import with_plugins
from sqlalchemy import create_engine
from dateutil.parser import parse as temporal
import pandas as pd

from tshistory.tsio import TimeSerie


TSH = TimeSerie()


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


@with_plugins(iter_entry_points('tshistory.subcommands'))
@click.group(invoke_without_command=True)
@click.pass_context
def tsh(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_usage())


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


if __name__ == '__main__':
    tsh()
