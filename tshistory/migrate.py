from json import dumps

from sqlalchemy import (
    create_engine,
    exc
)

from version_parser import Version as _Version
from dbcache import (
    api as dbapi,
    schema as dbschema
)

from tshistory import __version__
from tshistory.tsio import timeseries as tshclass
from tshistory.util import (
    read_versions,
    NoVersion
)


VERSIONS = {}


class Version(_Version):

    def __init__(self, vstring, *a, **k):
        super().__init__(vstring, *a, **k)
        self.raw_version = vstring

    def __hash__(self):
        return hash(self.raw_version)


def version(numversion):
    def decorate(func):
        VERSIONS[Version(numversion)] = func
        return func

    return decorate


def yesno(msg):
    out = input(msg)
    return out in 'yY'


class Migrator:
    _order = 0
    _package = 'tshistory'
    _known_version = __version__
    __slots__ = 'uri', 'namespace', 'interactive', 'start'

    def __init__(self, uri, namespace, interactive=False, start=None):
        self.uri = uri
        self.namespace = namespace
        self.interactive = interactive
        self.start = start

    @property
    def engine(self):
        return create_engine(self.uri)

    def run_migrations(self):
        print(f'Running migrations for {self._package}.')
        # determine versions
        storens = f'{self.namespace}-kvstore'
        stored_version = None
        version_string = f'{self._package}-version'
        store = dbapi.kvstore(self.uri, namespace=storens)
        try:
            stored_version = store.get(version_string)
        except exc.ProgrammingError:
            # bootstrap: we're in a stage where this was never installed
            if self.interactive:
                if not yesno('Initialize the versions ? [y/n] '):
                    return
            dbschema.init(self.engine, ns=storens)

        if stored_version is None or self.start == '0.0.0':
            # first time
            print(f'initial migration to {self._known_version} for {self._package}')
            self.initial_migration()
            store.set(version_string, self._known_version)

        to_migrate = list(VERSIONS)
        # filter from _known
        if self._known_version is not None or self.start is not None:
            known = Version(self._known_version or self.start)
            to_migrate = [
                ver for ver in to_migrate
                if ver > known
            ]
        for version in to_migrate:
            VERSIONS[version](self.uri, self.namespace, self.interactive)

    def initial_migration(self):
        engine = self.engine
        migrate_metadata(engine, self.namespace, self.interactive)
        fix_user_metadata(engine, self.namespace, self.interactive)
        migrate_to_baskets(engine, self.namespace, self.interactive)
        gns = f'{self.namespace}.group'
        migrate_metadata(engine, gns, self.interactive)
        fix_user_metadata(engine, gns, self.interactive)
        migrate_to_baskets(engine, gns, self.interactive)


def migrate_metadata(engine, namespace, interactive):
    ns = namespace

    print(f'migrate metadata for {ns}')
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

    print(f'fix user metadata for {ns}')
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
    print(f'migrate to baskets for {namespace}')

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
