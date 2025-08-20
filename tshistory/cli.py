from collections import defaultdict
from importlib.metadata import entry_points

import click
from sqlhelp.pgapi import pgdb as create_engine

from dbcache import api as storeapi
from tshistory.api import timeseries
from tshistory.config import configuration
from tshistory.util import run_migrations

from tshistory.schema import tsschema


@click.group()
def tsh():
    pass


@tsh.command()
def configpath():
    path = configuration.path().resolve()  # pytype: disable=attribute-error
    if path is None:
        print('No config file was found!')
        return
    print(path)


# db maintenance

@tsh.command(name='init-db')
@click.argument('db-uri')
@click.option('--namespace', default='tsh')
def init_db(db_uri, namespace='tsh'):
    """initialize an new db."""
    engine = create_engine(configuration().find_dburi(db_uri))
    schem = tsschema(namespace)
    schem.create(engine)


@tsh.command(name='migrate')
@click.argument('db-uri')
@click.option('--interactive/--no-interactive', is_flag=True, default=True)
@click.option('--initial')
@click.option('--force')
@click.option('--last', is_flag=True, default=False)
@click.option('--namespace', default='tsh')
def migrate(db_uri, interactive=True, initial=None, force=None, last=False, namespace='tsh'):
    uri = configuration().find_dburi(db_uri)
    run_migrations(uri, interactive, initial, force, last, namespace)


@tsh.command(name='dbversions')
@click.argument('db-uri')
@click.option('--namespace', default='tsh')
def dbversions(db_uri, namespace='tsh'):
    uri = configuration().find_dburi(db_uri)
    store = storeapi.kvstore(
        uri,
        namespace=f'{namespace}-kvstore'
    )
    for k, v in sorted(store.all().items()):
        print(f'{k} -> {v}')


@tsh.command(name='shell')
@click.argument('db-uri')
@click.option('--namespace', default='tsh')
def shell(db_uri, namespace='tsh'):
    uri = configuration().find_dburi(db_uri)
    tsa = timeseries(  # noqa
        uri,
        namespace
    )
    store = storeapi.kvstore(  # noqa
        uri,
        namespace=f'{namespace}-kvstore'
    )
    import pdb; pdb.set_trace()


@tsh.command(name='diagnose-indexes')
@click.argument('db-uri')
@click.option('--namespace', default='tsh')
@click.option('--fix', is_flag=True, help='Fix the issues (rename and drop duplicates)')
def diagnose_indexes(db_uri, namespace='tsh', fix=False):
    """Diagnose and optionally fix index issues (duplicates, missing, wrong names)"""
    from tshistory import dbdiag
    from tshistory.sqlparser import (
        parse_indexes,
        TSHISTORY_SQLFILES
    )

    uri = configuration().find_dburi(db_uri)
    engine = create_engine(uri)

    # Always show diagnosis first
    report = dbdiag.diagnose_indexes(engine, namespace)
    print(report)

    if not fix:
        return

    print("\nApplying fixes...")

    # Get expected indexes for this namespace
    expected_indexes = parse_indexes(TSHISTORY_SQLFILES, namespace)

    dbdiag.fix_indexes(engine, namespace, expected_indexes)
    print("Fix operations completed.")

    # Show diagnosis again to confirm fix
    print("\nFinal state:")
    report = dbdiag.diagnose_indexes(engine, namespace)
    print(report)


def register_plugin_subcommands():
    errors = defaultdict(set)
    entrypoints = list(entry_points().select(group='tshistory.subcommands'))
    entrypoints.sort(key=lambda ep: 'pro' in ep.module)
    for ep in entrypoints:
        try:
            cmd = ep.load()
        except Exception as e:
            errors[str(e)].add(ep.module)
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
