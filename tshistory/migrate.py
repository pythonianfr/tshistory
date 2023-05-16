from json import dumps

from dbcache import (
    api as dbapi,
    schema as dbschema
)

from tshistory.tsio import timeseries as tshclass
from tshistory.util import (
    read_versions,
    NoVersion
)


MIGRATE = entry_points(group='tshistory.migrate.run_migrations')


def yesno(msg):
    out = input(msg)
    return out in 'yY'


def run_migrations(engine, namespace, interactive=False):
    # determine versions
    storens = f'{namespace}-kvstore'
    stored_version = None
    try:
        stored_version, known_version = read_versions(str(engine.url), namespace)
    except NoVersion:
        # bootstrap: we're in a stage where this was never installed
        if interactive:
            if not yesno('Initialize the versions ? [y/n] '):
                return
        dbschema.init(engine, ns=storens)

    if stored_version is None:
        # first time
        from tshistory import __version__ as known_version
        initial_migration(engine, namespace, interactive)
        store = dbapi.kvstore(str(engine.url), namespace=storens)
        store.set('tshistory-version', known_version)


    # call the plugins
    for migrator_plugin in MIGRATE:
        migrator = migrator_plugin.load()
        migrator(engine, namespace, interactive)


def initial_migration(engine, namespace, interactive):
    migrate_metadata(engine, namespace, interactive)
    fix_user_metadata(engine, namespace, interactive)
    migrate_to_baskets(engine, namespace, interactive)


def migrate_metadata(engine, namespace, interactive):
    ns = namespace

    with engine.begin() as cn:

        # check initial condition
        unmigrated = cn.execute(
            "select exists (select 1 "
            "from information_schema.columns "
            f"where table_schema='{ns}' and "
            "        table_name='registry' and "
            "        column_name='tablename'"
            ")"
        ).scalar()
        # add internal_metadata, add gin indexes
        # rename seriesname -> name
        # split internal / user metadata
        # drop tablename
        cn.execute(
            f'alter table "{ns}".registry '
            f'add column if not exists "internal_metadata" jsonb'
        )
        cn.execute(
            f'create index if not exists idx_metadata '
            f'on "{ns}".registry using gin (metadata)'
        )
        cn.execute(
            f'create index if not exists idx_internal_metadata '
            f'on "{ns}".registry using gin (internal_metadata)'
        )
        if unmigrated:
            cn.execute(
                f'alter table "{ns}".registry rename column seriesname to name'
            )

        # collect all series metadata and split internal / user
        if unmigrated:
            print('migrating data')
            allmetas = {}
            metakeys = tshclass.metakeys | {'supervision_status'}

            for name, tablename, imeta in cn.execute(
                    f'select name, tablename, metadata from "{ns}".registry'):
                umeta = {}
                for k in list(imeta):
                    if k not in metakeys:
                        umeta[k] = imeta.pop(k)
                imeta['tablename'] = tablename
                allmetas[name] = (imeta, umeta)

            # store them
            for name, (imeta, umeta) in allmetas.items():
                cn.execute(
                    f'update "{ns}".registry '
                    'set (internal_metadata, metadata) = '
                    '    (%(imeta)s, %(umeta)s) '
                    'where name=%(name)s',
                    name=name,
                    imeta=dumps(imeta),
                    umeta=dumps(umeta)
                )

        cn.execute(
            f'alter table "{ns}".registry drop column if exists "tablename"'
        )


def fix_user_metadata(engine, namespace, interactive):
    ns = namespace

    with engine.begin() as cn:
        names = [
            name for name, in cn.execute(
                f'select name from "{ns}".registry '
                'where metadata is null'
            ).fetchall()
        ]
        for name in names:
            cn.execute(
                f'update "{ns}".registry '
                'set metadata = %(meta)s '
                'where name = %(name)s',
                name=name,
                meta=dumps({})
            )


def migrate_to_baskets(engine, namespace, interactive):
    sql = f"""
    create table if not exists "{namespace}".basket (
      id serial primary key,
      name text not null,
      "query" text not null,
      unique(name)
    );
    """
    with engine.begin() as cn:
        cn.execute(sql)
