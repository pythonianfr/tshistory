from collections import defaultdict

from pkg_resources import iter_entry_points
import click
from sqlalchemy import create_engine

from tshistory.tsio import timeseries as tshclass
from tshistory.api import timeseries
from tshistory.util import (
    find_dburi,
    get_cfg_path,
    objects
)
from tshistory.schema import tsschema


@click.group()
def tsh():
    pass


@tsh.command()
def configpath():
    print(get_cfg_path())


# migration

@tsh.command(name='fix-primary-groups-metadata')
@click.argument('db-uri')
@click.option('--deletebroken', default=False, is_flag=True)
@click.option('--namespace', default='tsh')
def fix_groups_metadata_(db_uri, deletebroken=False, namespace='tsh'):
    engine = create_engine(find_dburi(db_uri))

    from tshistory.migrate import fix_groups_metadata
    fix_groups_metadata(
        engine, namespace, interactive=True, deletebroken=deletebroken
    )


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


@tsh.command(name='migrate')
@click.argument('db-uri')
@click.option('--interactive', is_flag=True, default=True)
@click.option('--initial')
@click.option('--namespace', default='tsh')
def migrate(db_uri, interactive=True, initial=None, namespace='tsh'):
    uri = find_dburi(db_uri)
    # call the plugins
    for migrator in sorted(objects('migrator'), key=lambda x: x._order):
        migrator(
            uri, namespace, interactive=True, start=initial
        ).run_migrations()


@tsh.command(name='shell')
@click.argument('db-uri')
@click.option('--namespace', default='tsh')
def shell(db_uri, namespace='tsh'):
    tsa = timeseries(  # noqa: F841
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
