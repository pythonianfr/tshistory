import os, signal
from pkg_resources import iter_entry_points
from time import time
import random
from pathlib import Path
from collections import defaultdict

from dateutil import parser
from json import dumps

import click
from sqlalchemy import create_engine
from dateutil.parser import parse as temporal
import pandas as pd

from tshistory.tsio import timeseries
from tshistory.util import (
    delete_series,
    find_dburi,
    fromjson
)

import tshistory.schema


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
        fmt += 'series: {name}\n\n'
        lines = []
        for ts in rev['diff'].values():
            lines.append(ts.to_string())
        fmt += '\n'.join(lines)
    else:
        fmt += 'series:   {name}'

    return fmt.format(**rev)


@click.group()
def tsh():
    pass


@tsh.command()
@click.argument('db-uri')
@click.argument('seriename')
@click.option('--json', is_flag=True, default=False)
@click.option('--namespace', default='tsh')
def get(db_uri, seriename, json, namespace='tsh'):
    """show a serie in its current state """
    engine = create_engine(find_dburi(db_uri))
    tsh = timeseries(namespace)

    ts = tsh.get(engine, seriename)
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
@click.option('--namespace', default='tsh')
def history(db_uri, seriename,
            from_insertion_date, to_insertion_date,
            from_value_date, to_value_date,
            diff, json,
            namespace='tsh'):
    """show a serie full history """
    engine = create_engine(find_dburi(db_uri))

    tsh = timeseries(namespace)
    with engine.begin() as cn:
        hist = tsh.history(
            cn, seriename,
            from_insertion_date, to_insertion_date,
            from_value_date, to_value_date,
            diffmode=diff
        )
    if json:
        out = {
            str(idate): {
                str(vdate): val
                for vdate, val in ts.to_dict().items()
            }
            for idate, ts in hist.items()
        }
        print(dumps(out))
    else:
        for idate in hist:
            print(hist[idate])


@tsh.command()
@click.argument('db-uri')
@click.option('--limit', '-l', default=None)
@click.option('--serie', '-s', multiple=True)
@click.option('--from-rev')
@click.option('--to-rev')
@click.option('--from-insertion-date', type=temporal)
@click.option('--to-insertion-date', type=temporal)
@click.option('--namespace', default='tsh')
def log(db_uri, limit, serie, from_rev, to_rev,
        from_insertion_date, to_insertion_date,
        namespace='tsh'):
    """show revision history of entire repository or series"""
    engine = create_engine(find_dburi(db_uri))
    tsh = timeseries(namespace)

    for rev in tsh.log(engine, limit=limit, names=serie,
                       fromrev=from_rev, torev=to_rev,
                       fromdate=from_insertion_date, todate=to_insertion_date):
        # rev['name'] = ','.join(rev['names'])
        print(format_rev(rev))
        print()


INFOFMT = """
changeset count: {changeset count}
series count:    {series count}
series names:    {serie names}
""".strip()


@tsh.command()
@click.argument('db-uri')
@click.option('--namespace', default='tsh')
def info(db_uri, namespace='tsh'):
    """show global statistics of the repository"""
    engine = create_engine(find_dburi(db_uri))

    info = timeseries(namespace).info(engine)
    info['serie names'] = ', '.join(info['serie names'])
    print(INFOFMT.format(**info))


# series maintenance

@tsh.command()
@click.argument('db-uri')
@click.argument('mapfile', type=click.Path(exists=True))
@click.option('--namespace', default='tsh')
def rename(db_uri, mapfile, namespace='tsh'):
    """rename series by providing a map file (csv format)

    map file header must be `old,new`
    """
    seriesmap = {
        p.old: p.new
        for p in pd.read_csv(mapfile).itertuples()
    }
    engine = create_engine(find_dburi(db_uri))
    tsh = timeseries(namespace)
    for old, new in seriesmap.items():
        with engine.begin() as cn:
            print('rename', old, '->', new)
            tsh.rename(cn, old, new)


@tsh.command()
@click.argument('db-uri')
@click.option('--series')
@click.option('--deletefile', type=click.Path(exists=True))
@click.option('--namespace', default='tsh')
def delete(db_uri, series=None, deletefile=None, namespace='tsh'):
    """delete series by providing a one-column file (csv format)

    file header must be `name`
    """
    if not (series or deletefile):
        print('You must provide a series name _or_ a csv file path')
        return

    if deletefile:
        series = [
            p.name
            for p in pd.read_csv(deletefile).itertuples()
        ]
    else:
        series = [series]

    engine = create_engine(find_dburi(db_uri))
    delete_series(engine, series, namespace)


# db maintenance

@tsh.command(name='init-db')
@click.argument('db-uri')
@click.option('--reset', is_flag=True, default=False)
@click.option('--namespace', default='tsh')
def init_db(db_uri, reset=False, namespace='tsh'):
    """initialize an new db."""
    engine = create_engine(find_dburi(db_uri))
    schem = tshistory.schema.tsschema(namespace)
    schem.create(engine)


@tsh.command(name='check')
@click.argument('db-uri')
@click.option('--series', default=None, help='series name to check')
@click.option('--namespace', default='tsh')
def check(db_uri, series=None, namespace='tsh'):
    "coherence checks of the db"
    e = create_engine(find_dburi(db_uri))
    if series is None:
        sql = 'select seriename from "{}".registry order by seriename'.format(namespace)
        series = [row.seriename for row in e.execute(sql)]
    else:
        series = [series]

    tsh = timeseries(namespace)
    for idx, s in enumerate(series):
        t0 = time()
        with e.begin() as cn:
            hist = tsh.history(cn, s)
        start, end = None, None
        mon = True
        for ts in hist.values():
            cmin = ts.index.min()
            cmax = ts.index.max()
            start = min(start or cmin, cmin)
            end = max(end or cmax, cmax)
            mon = ts.index.is_monotonic_increasing
        ival = tsh.interval(e, s)
        if ival.left != start:
            print('  start:', s, f'{ival.left} != {start}')
        if ival.right != end:
            print('  end:', s, f'{ival.right} != {end}')
        monmsg = '' if mon else 'non-monotonic'
        print(idx, s, 'inserts={}, read-time={} {}'.format(
            len(hist), time() - t0, monmsg)
        )


@tsh.command(name='shell')
@click.argument('db-uri')
@click.option('--namespace', default='tsh')
def shell(db_uri, namespace='tsh'):
    e = create_engine(find_dburi(db_uri))

    tsh = timeseries(namespace)
    import pdb; pdb.set_trace()


@tsh.command(name='migrate-0.6-to-0.7')
@click.argument('db-uri')
@click.option('--namespace', default='tsh')
def migrate_dot_6_to_dot_7(db_uri, namespace='tsh'):
    e = create_engine(find_dburi(db_uri))
    tsh = timeseries(namespace)
    with e.begin() as cn:
        # drop not null
        print(f'{namespace}: alter changeset_series table')
        cn.execute(f'alter table "{namespace}".changeset_series '
                   'alter column serie drop not null')
        # alter foreign key on delete: delete -> set null
        cn.execute(f'alter table "{namespace}".changeset_series '
                   'drop constraint "changeset_series_serie_fkey"')
        cn.execute(f'alter table "{namespace}".changeset_series '
                   'add constraint "changeset_series_serie_fkey" '
                   f'foreign key (serie) references "{namespace}".registry (id) '
                   'on delete set null')

    from tqdm import tqdm
    from tshistory.snapshot import Snapshot
    series = tsh.list_series(e)
    print('migrate series and snapshot start/end columns')
    bar = tqdm(range(len(series)))
    with e.begin() as cn:
        for name in series:
            table = tsh._serie_to_tablename(cn, name)
            cn.execute(
                f'alter table "{namespace}.timeserie"."{table}" '
                'rename column start to tsstart'
            )
            cn.execute(
                f'alter table "{namespace}.timeserie"."{table}" '
                'rename column "end" to tsend'
            )
            cn.execute(
                f'alter table "{namespace}.snapshot"."{table}" '
                'rename column start to cstart'
            )
            cn.execute(
                f'alter table "{namespace}.snapshot"."{table}" '
                'rename column "end" to cend'
            )
            bar.update()

    print()
    print('reclaim unreachable chunks left behind by strip')
    bar = tqdm(range(len(series)))
    for name in series:
        snap = Snapshot(e, tsh, name)
        garb = snap.garbage()
        if garb:
            print(f'{name} garbage = {len(garb)}')
            snap.reclaim()
        bar.update()


def register_plugin_subcommands():
    errors = defaultdict(set)
    for ep in iter_entry_points('tshistory.subcommands'):
        try:
            cmd = ep.load()
        except Exception as e:
            errors[str(e)].add(ep.module_name)
            continue
        tsh.add_command(cmd)

    if errors:
        from colorama import init, Fore, Style
        for error, eplist in errors.items():
            print(Fore.YELLOW +
                  f'impossible to add subcommands from {",".join(eplist)}')
            print(Fore.RED +
                  f'cause: {error}')
        print(Fore.RESET + '')


register_plugin_subcommands()
