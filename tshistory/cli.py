from time import time
from collections import defaultdict
from json import dumps

from pkg_resources import iter_entry_points
import click
from sqlalchemy import create_engine
from dateutil.parser import parse as temporal
import pandas as pd

from tshistory.tsio import timeseries as tshclass
from tshistory.api import timeseries
from tshistory.util import (
    find_dburi,
    get_cfg_path
)
from tshistory.schema import tsschema


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
        lines = []
        for ts in rev['diff'].values():
            lines.append(ts.to_string())
        fmt += '\n'.join(lines)

    return fmt.format(**rev)


@click.group()
def tsh():
    pass


@tsh.command()
@click.argument('db-uri')
@click.argument('series')
@click.option('--json', is_flag=True, default=False)
@click.option('--namespace', default='tsh')
def get(db_uri, series, json, namespace='tsh'):
    """show a series in its current state """
    tsa = timeseries(
        find_dburi(db_uri),
        namespace,
        handlerclass=tshclass
    )

    ts = tsa.get(series)
    if json:
        print(ts.to_json())
    else:
        with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
            print(ts)


@tsh.command()
@click.argument('db-uri')
@click.argument('series')
@click.option('--json', is_flag=True, default=False)
@click.option('--from-insertion-date', type=temporal)
@click.option('--to-insertion-date', type=temporal)
@click.option('--from-value-date', type=temporal)
@click.option('--to-value-date', type=temporal)
@click.option('--diff/--no-diff', is_flag=True, default=True)
@click.option('--namespace', default='tsh')
def history(db_uri, series,
            from_insertion_date, to_insertion_date,
            from_value_date, to_value_date,
            diff, json,
            namespace='tsh'):
    """show a series full history """
    tsa = timeseries(
        find_dburi(db_uri),
        namespace,
        handler=tshclass
    )
    hist = tsa.history(
        series,
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
@click.argument('series')
@click.option('--limit', '-l', default=None)
@click.option('--from-insertion-date', type=temporal)
@click.option('--to-insertion-date', type=temporal)
@click.option('--namespace', default='tsh')
def log(db_uri, limit, series,
        from_insertion_date=None, to_insertion_date=None,
        namespace='tsh'):
    """show revision history of entire repository or series"""
    tsa = timeseries(
        find_dburi(db_uri),
        namespace,
        handler=tshclass
    )

    for rev in tsa.log(
        series,
        limit=limit,
        fromdate=from_insertion_date,
        todate=to_insertion_date):

        print(format_rev(rev))
        print()


INFOFMT = """
series count:    {series count}
series names:    {serie names}
""".strip()


@tsh.command()
@click.argument('db-uri')
@click.option('--namespace', default='tsh')
def info(db_uri, namespace='tsh'):
    """show global statistics of the repository"""
    tsa = timeseries(
        find_dburi(db_uri),
        namespace,
        handler=tshclass
    )
    info = tsa.tsh.info(tsa.engine)
    info['serie names'] = ', '.join(info['serie names'])
    print(INFOFMT.format(**info))


# technical


@tsh.command()
def configpath():
    print(get_cfg_path())


# migration

@tsh.command(name='fix-primary-groups-metadata')
@click.argument('db-uri')
@click.option('--deletebroken', default=False, is_flag=True)
@click.option('--namespace', default='tsh')
def fix_groups_metadata(db_uri, deletebroken=False, namespace='tsh'):
    engine = create_engine(find_dburi(db_uri))
    tsh = tshclass()

    for name, kind in tsh.list_groups(engine).items():
        if kind != 'primary':
            continue

        if deletebroken:
            try:
                tsh.group_get(engine, name)
            except:
                print('Deleting broken group', name)
                tsh.group_delete(engine, name)
                continue

        with engine.begin() as cn:
            tsmeta = cn.execute(
                'select tsr.metadata '
                f'from "{namespace}".group_registry as gr, '
                f'     "{namespace}".groupmap as gm,'
                f'     "{namespace}.group".registry as tsr '
                'where gr.name = %(name)s and '
                '      gr.id = gm.groupid and '
                '      gm.seriesid = tsr.id '
                'limit 1',
                name=name
            ).scalar()

            grmeta = tsh.group_metadata(engine, name) or {}
            #import pdb;pdb.set_trace()
            grmeta.update(tsmeta)
            cn.execute(
                f'update "{namespace}".group_registry '
                'set metadata = %(metadata)s '
                f'where name = %(name)s',
                metadata=dumps(grmeta),
                name=name
            )
        print(f'updated `{name}` with {grmeta}')


@tsh.command(name='migrate-to-groups')
@click.argument('db-uri')
@click.option('--reset', is_flag=True, default=False)
@click.option('--namespace', default='tsh')
def migrate_to_groups(db_uri, reset=False, namespace='tsh'):
    engine = create_engine(find_dburi(db_uri))
    sch = tsschema(namespace)
    sch._create_group(engine, reset)


# db maintenance

@tsh.command(name='init-db')
@click.argument('db-uri')
@click.option('--namespace', default='tsh')
def init_db(db_uri, namespace='tsh'):
    """initialize an new db."""
    engine = create_engine(find_dburi(db_uri))
    schem = tsschema(namespace)
    schem.create(engine)


@tsh.command(name='check')
@click.argument('db-uri')
@click.option('--series', default=None, help='series name to check')
@click.option('--namespace', default='tsh')
def check(db_uri, series=None, namespace='tsh'):
    "coherence checks of the db"
    tsa = timeseries(
        find_dburi(db_uri),
        namespace,
        handler=tshclass
    )
    if series is None:
        series = tsa.tsh.list_series(tsa.engine)
    else:
        series = [series]

    for idx, s in enumerate(series):
        t0 = time()
        hist = tsa.history(s)
        start, end = None, None
        mon = True
        for ts in hist.values():
            cmin = ts.index.min()
            cmax = ts.index.max()
            start = min(start or cmin, cmin)
            end = max(end or cmax, cmax)
            mon = ts.index.is_monotonic_increasing
        ival = tsa.interval(s)
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
    tsa = timeseries(
        find_dburi(db_uri),
        namespace,
        handler=tshclass
    )
    import pdb; pdb.set_trace()


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
        from colorama import Fore
        for error, eplist in errors.items():
            print(Fore.YELLOW +
                  f'impossible to add subcommands from {",".join(eplist)}')
            print(Fore.RED +
                  f'cause: {error}')
        print(Fore.RESET + '')


register_plugin_subcommands()
