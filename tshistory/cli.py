from collections import defaultdict

from pkg_resources import iter_entry_points
import click
from sqlalchemy import create_engine

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
    print(configuration.path())


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
