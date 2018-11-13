import os, signal
from pkg_resources import iter_entry_points
from time import time
import random
from pathlib import Path

from dateutil import parser
import json

import click
from sqlalchemy import create_engine, MetaData
from dateutil.parser import parse as temporal
import pandas as pd

from tshistory.tsio import TimeSerie
from tshistory.util import (
    delete_series,
    find_dburi,
    fromjson,
    rename_series
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
        fmt += 'series: {names}\n\n'
        lines = []
        for ts in rev['diff'].values():
            lines.append(ts.to_string())
        fmt += '\n'.join(lines)
    else:
        fmt += 'series:   {names}'

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
    tsh = TimeSerie(namespace)

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

    tsh = TimeSerie(namespace)
    ts = tsh.get_history(engine, seriename,
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
@click.option('--namespace', default='tsh')
def log(db_uri, limit, show_diff, serie, from_rev, to_rev,
        from_insertion_date, to_insertion_date,
        namespace='tsh'):
    """show revision history of entire repository or series"""
    engine = create_engine(find_dburi(db_uri))
    tsh = TimeSerie(namespace)

    for rev in tsh.log(engine, limit=limit, diff=show_diff, names=serie,
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
@click.option('--namespace', default='tsh')
def info(db_uri, namespace='tsh'):
    """show global statistics of the repository"""
    engine = create_engine(find_dburi(db_uri))

    info = TimeSerie(namespace).info(engine)
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
    rename_series(engine, seriesmap, namespace)


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
    schem.define()

    if reset:
        assert schem.exists(engine)
        schem.destroy(engine)

        # needed because of del self.meta & return in define() :
        schem.meta = MetaData()

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

    tsh = TimeSerie(namespace)
    for idx, s in enumerate(series):
        t0 = time()
        hist = tsh.get_history(e, s)
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

    tsh = TimeSerie(namespace)
    import pdb; pdb.set_trace()


# repair

@tsh.command(name='repair-start-end')
@click.argument('db-uri')
@click.option('--namespace', default='tsh')
@click.option('--processes', default=1)
@click.option('--series', default=None)
def repair_start_end(db_uri, namespace='tsh', processes=1, series=None):
    from tshistory.migration import SnapshotMigrator
    if series:
        series = [series]
    else:
        engine = create_engine(find_dburi(db_uri))
        sql = 'select seriename from "{}".registry order by seriename'.format(namespace)
        series = [row.seriename for row in engine.execute(sql)]
        engine.dispose()

    def _migrate(seriename):
        e = create_engine(find_dburi(db_uri), pool_size=1)
        tsh = TimeSerie(namespace)
        with e.begin() as cn:
            m = SnapshotMigrator(cn, tsh, seriename)
            m.fix_start_end()
        e.dispose()

    def migrate(seriename):
        try:
            _migrate(seriename)
        except Exception:
            import traceback as tb
            tb.print_exc()
            print(seriename, 'FAIL')

    def run(proc, series):
        seriescount = len(series)
        pid = os.getpid()
        for idx, ts in enumerate(series, 1):
            print('migrate {} proc: {} [{}/{}]'.format(ts, proc, idx, seriescount))
            migrate(ts)

    if processes == 1:
        run(0, series)
        return

    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    # try to distribute the payload randomly as in practice it is definitely
    # *not* evenly dsitributed along the lexical order ...
    random.shuffle(series)
    chunks = list(chunks(series, len(series) // processes))
    print('running with {} processes'.format(len(chunks)))

    pids = []
    for idx, chunk in enumerate(chunks):
        pid = os.fork()
        if not pid:
            # please the eyes
            chunk.sort()
            run(idx, chunk)
            return
        pids.append(pid)

    try:
        for pid in pids:
            print('waiting for', pid)
            os.waitpid(pid, 0)
    except KeyboardInterrupt:
        for pid in pids:
            print('kill', pid)
            os.kill(pid, signal.SIGINT)

# migration

@tsh.command(name='migrate-0.3-to-0.4')
@click.argument('db-uri')
@click.option('--namespace', default='tsh')
@click.option('--processes', default=1)
@click.option('--tryserie', default=None)
def migrate_zerodotthree_to_zerodotfour(db_uri, namespace='tsh', processes=1, tryserie=None):
    """ in-place migration for going from 0.3 to 0.4

    Will populate the start/end fields on series tables and
    update the chunk representation on snapshots tables.
    """
    from tshistory.migration import SnapshotMigrator
    if tryserie:
        series = [tryserie]
    else:
        engine = create_engine(find_dburi(db_uri))
        sql = 'select seriename from "{}".registry order by seriename'.format(namespace)
        series = [row.seriename for row in engine.execute(sql)]
        engine.dispose()

    def _migrate(seriename):
        e = create_engine(find_dburi(db_uri), pool_size=1)
        tsh = TimeSerie(namespace)
        with e.begin() as cn:
            m = SnapshotMigrator(cn, tsh, seriename)
            m.migratechunks()
        with e.begin() as cn:
            m = SnapshotMigrator(cn, tsh, seriename)
            m.migrateseries()
        e.dispose()

    def migrate(seriename):
        try:
            _migrate(seriename)
        except Exception:
            import traceback as tb
            tb.print_exc()
            print(seriename, 'FAIL')

    def run(proc, series):
        seriescount = len(series)
        pid = os.getpid()
        for idx, ts in enumerate(series, 1):
            print('migrate {} proc: {} [{}/{}]'.format(ts, proc, idx, seriescount))
            migrate(ts)

    if processes == 1:
        run(series)
        return

    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    # try to distribute the payload randomly as in practice it is definitely
    # *not* evenly dsitributed along the lexical order ...
    random.shuffle(series)
    chunks = list(chunks(series, len(series) // processes))
    print('running with {} processes'.format(len(chunks)))

    pids = []
    for idx, chunk in enumerate(chunks):
        pid = os.fork()
        if not pid:
            # please the eyes
            chunk.sort()
            run(idx, chunk)
            return
        pids.append(pid)

    try:
        for pid in pids:
            print('waiting for', pid)
            os.waitpid(pid, 0)
    except KeyboardInterrupt:
        for pid in pids:
            print('kill', pid)
            os.kill(pid, signal.SIGINT)


for ep in iter_entry_points('tshistory.subcommands'):
    tsh.add_command(ep.load())


if __name__ == '__main__':
    tsh()
