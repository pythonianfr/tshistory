from json import dumps
import os

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
from tshistory import util


VERSIONS = {}


class Version(_Version):

    def __init__(self, package, vstring, *a, **k):
        super().__init__(vstring, *a, **k)
        self.package = package
        self.raw_version = vstring

    def __hash__(self):
        return hash((self.package, self.raw_version))

    def __repr__(self):
        return f'Version({self.package}:{self.raw_version})'


def version(package, numversion):
    def decorate(func):
        VERSIONS[Version(package, numversion)] = func
        return func

    return decorate


def yesno(msg):
    out = input(msg)
    return out in 'yY'


class Migrator:
    _order = 0
    _package = 'tshistory'
    _package_version = __version__
    # __slots__ = 'uri', 'namespace', 'interactive', 'start', 'force'

    def __init__(self, uri, namespace, interactive=False, start=None, force=None):
        self.uri = uri
        self.namespace = namespace
        self.interactive = interactive
        self.start = start
        # "package:version"
        self.force = force.split(':') if force else (None, None)

    @property
    def engine(self):
        return create_engine(self.uri)

    @property
    def storens(self):
        return f'{self.namespace}-kvstore'

    @property
    def store(self):
        return dbapi.kvstore(self.uri, namespace=self.storens)

    @property
    def versionkey(self):
        return f'{self._package}-version'

    @property
    def initialversion(self):
        stored_version = Version(self._package, '0.0.0')
        try:
            stored_version = Version(
                self._package,
                self.store.get(self.versionkey)
            )
        except (exc.ProgrammingError, ValueError):
            # bootstrap: we're in a stage where this was never installed
            # yes, this is a bit aggressive for a propery, but that
            # happens only once ...
            if self.interactive:
                if not yesno('Initialize the versions ? [y/n] '):
                    return
            dbschema.init(self.engine, ns=self.storens)

        if self.start:
            return Version(self._package, self.start)
        return stored_version

    @property
    def finalversion(self):
        # prepare version migration forcing
        forced_package, forced_version = self.force
        if forced_package != self._package:
            # we are not concerned
            forced_version = None
        return Version(self._package, forced_version or self._package_version)

    def run_migrations(self):
        print(f'Running migrations for {self._package}.')
        # determine from where we start (stored version or provided
        # initial)
        start = self.initialversion

        if start.raw_version == '0.0.0':
            # first time
            print(f'Initial migration to {self._package_version}')
            self.initial_migration()

        to_migrate = list(VERSIONS)

        end = self.finalversion
        print(f'Versions: from {start} to {end}')

        # build migration plan (from stored_version to package_version
        # or forced_version)
        to_migrate = [
            ver for ver in to_migrate
            if start < ver <= end
            and ver.package == self._package
        ]

        if not to_migrate:
            print(f'Nothing to migrate for `{self._package}`, skipping.')
        else:
            print(
                f'Migration plan for `{self._package}`: {[v.raw_version for v in to_migrate]}'
            )
            if self.interactive:
                if not yesno('Execute this migration plan ? [y/n] '):
                    return

            for version in to_migrate:
                VERSIONS[version](self.engine, self.namespace, self.interactive)

        self.store.set(self.versionkey, self._package_version)

    def initial_migration(self):
        engine = self.engine
        migrate_metadata(engine, self.namespace, self.interactive)
        fix_user_metadata(engine, self.namespace, self.interactive)
        migrate_to_baskets(engine, self.namespace, self.interactive)
        fix_groups_metadata(self.engine, self.namespace, self.interactive)
        migrate_groups_metadata(engine, self.namespace, self.interactive)

        gns = f'{self.namespace}.group'
        migrate_metadata(engine, gns, self.interactive)
        fix_user_metadata(engine, gns, self.interactive)
        migrate_to_baskets(engine, gns, self.interactive)


@version('tshistory', '0.20.0')
def migrate_series_versions(engine, namespace, interactive):
    migrate_add_diffstart_diffend(engine, namespace, interactive)
    migrate_add_diffstart_diffend(engine, f'{namespace}.group', interactive)


def populatedata(pid, cn, tsh, namespace, name, tablename):
    diffsb = []
    delete = []
    for csid, idate, diff in util.diffs(cn, tsh, name, tablename, None, None):
        if len(diff):
            diffsb.append(
                {
                    'csid': csid,
                    'diffstart': diff.index[0],
                    'diffend': diff.index[-1]
                }
            )
        else:
            delete.append(
                {
                    'csid': csid,
                    'idate': idate
                }
            )

    if diffsb:
        sql = (
            f'update "{namespace}.revision"."{tablename}" '
            f'set diffstart=%(diffstart)s, '
            f'    diffend=%(diffend)s '
            f'where id=%(csid)s'
        )
        cn.execute(
            sql, diffsb
        )

    if delete:
        print(f'{pid}: revs to delete:', ','.join(x['idate'].isoformat()
                                                    for x in delete))
        sql = (
            f'delete from "{namespace}.revision"."{tablename}" '
            f'where id = %(csid)s'
        )
        cn.execute(sql, [{'csid': x['csid']} for x in delete])


def migrate_add_diffstart_diffend(engine, namespace, interactive, onlydata=False, cpus=1):
    import signal
    import sys
    import multiprocessing
    from sqlalchemy import create_engine

    if onlydata:
        print(f'data migration for columns `diffstart` and `diffend` to {namespace}.revision')
    else:
        print(f'add columns `diffstart` and `diffend` to {namespace}.revision')

    migdata = True
    if interactive:
        migdata = not yesno('Defer data migration to the "migrate_diffs" task ? [y/n] ')

    def migrated(cn, tablename):
        sql = (
            f"select exists (select 1 "
            f" from information_schema.columns "
            f" where table_schema='{namespace}.revision' and "
            f" table_name='{tablename}' and "
            f" column_name='diffstart'"
            f")"
        )
        migrated = cn.execute(sql).scalar()
        return migrated

    def addattributes(cn, tablename):
        cn.execute(
            f'alter table "{namespace}.revision"."{tablename}" '
            f'add column diffstart timestamptz'
        )
        cn.execute(
            f'create index if not exists "rev.{tablename}.idx_diffstart" '
            f'on "{namespace}.revision"."{tablename}" (diffstart)'
        )

        cn.execute(
            f'alter table "{namespace}.revision"."{tablename}" '
            f'add column diffend timestamptz'
        )
        cn.execute(
            f'create index if not exists "rev.{tablename}.idx_diffend" '
            f'on "{namespace}.revision"."{tablename}" (diffend)'
        )
        return True

    def finalizeattributes(cn, tablename):
        # drop the not null constraints for tsstart/tsend
        cn.execute(
            f'alter table "{namespace}.revision"."{tablename}" '
            f'alter column tsstart drop not null'
        )
        cn.execute(
            f'alter table "{namespace}.revision"."{tablename}" '
            f'alter column tsend drop not null'
        )

    def listchunks(alist, n):
        import numpy as np
        return list(
            np.array_split(np.array(alist), n)
        )

    # main
    tsh = tshclass(namespace)
    with engine.begin() as cn:
        cn.cache = {'series_tablename': {}}
        allnames = {
            name: tsh._series_to_tablename(cn, name)
            for name in tsh.list_series(engine).keys()
        }
        names = [
            name
            for name in allnames
            if allnames[name] is not None
            and (onlydata or not migrated(cn, allnames[name]))
        ]

    print(f'{len(names)} series to migrate.')
    cpus = cpus if onlydata else 1 if sys.platform == 'win32' else int(multiprocessing.cpu_count() / 2)
    chunked = listchunks(names, int(cpus))

    print(f'Starting with {cpus} processes.')

    def migrate(url, names):
        pid = os.getpid()
        engine = create_engine(url)
        for name in names:
            with engine.begin() as cn:
                cn.cache = {'series_tablename': {}}
                tablename = tsh._series_to_tablename(cn, name)
                print(f'{pid}: migrating `{name}` (table: {tablename})')
                if not migdata:
                    print(f'{pid}: no data migration')
                if tablename is None:
                    continue
                if not onlydata:
                    addattributes(cn, tablename)
                if migdata or onlydata:
                    populatedata(pid, cn, tsh, namespace, name, tablename)
                if not onlydata:
                    finalizeattributes(cn, tablename)

    if cpus == 1:
        migrate(str(engine.url), names)
    else:
        pids = []
        for idx, names in enumerate(chunked):
            pid = os.fork()
            if not pid:
                names.sort()
                migrate(str(engine.url), names)
                # rewrite this stuff to be compatible with a task
                sys.exit(0)

            pids.append(pid)

        try:
            for pid in pids:
                print('waiting for', pid)
                os.waitpid(pid, 0)
        except KeyboardInterrupt:
            for pid in pids:
                print('kill', pid)
                os.kill(pid, signal.SIGINT)

    if migdata:
        print(
            'Do not forget to schedule the "migrate_diffs" task to complete the migration.'
        )


def migrate_seriesdata_diffstart_diffend(engine, namespace, name):
    tsh = tshclass(namespace)
    pid = os.getpid()

    with engine.begin() as cn:
        cn.cache = {'series_tablename': {}}
        tablename = tsh._series_to_tablename(cn, name)
        if tablename:
            print(f'{pid}: migrating `{name}` (table: {tablename})')
            populatedata(pid, cn, tsh, namespace, name, tablename)


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


def migrate_groups_metadata(engine, namespace, interactive):
    ns = namespace

    print(f'migrate group metadata for {ns}')
    with engine.begin() as cn:

        # check initial condition
        unmigrated = cn.execute(
            "select not exists (select 1 "
            "  from information_schema.columns "
            f" where table_schema='{ns}' and "
            "        table_name='group_registry' and "
            "        column_name='internal_metadata'"
            ")"
        ).scalar()
        if not unmigrated:
            print('already migrated')
            return

        # add internal_metadata, add gin indexes
        cn.execute(
            f'alter table "{ns}".group_registry '
            f'add column if not exists "internal_metadata" jsonb'
        )
        cn.execute(
            f'create index if not exists idx_group_metadata '
            f'on "{ns}".group_registry using gin (metadata)'
        )
        cn.execute(
            f'create index if not exists idx_group_internal_metadata '
            f'on "{ns}".group_registry using gin (internal_metadata)'
        )

        # collect all groups metadata and split internal / user
        allmetas = {}
        metakeys = tshclass.metakeys | {'supervision_status'}

        for name, imeta in cn.execute(
                f'select name, metadata from "{ns}".group_registry'):
            umeta = {}
            for k in list(imeta):
                if k not in metakeys:
                    umeta[k] = imeta.pop(k)
            allmetas[name] = (imeta, umeta)

        # store them
        for name, (imeta, umeta) in allmetas.items():
            cn.execute(
                f'update "{ns}".group_registry '
                'set (internal_metadata, metadata) = '
                '    (%(imeta)s, %(umeta)s) '
                'where name=%(name)s',
                name=name,
                imeta=dumps(imeta),
                umeta=dumps(umeta)
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


def fix_groups_metadata(engine, namespace, interactive, deletebroken=False):
    tsh = tshclass(namespace)
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
            if tsmeta is None:
                continue

            grmeta = tsh.group_metadata(engine, name) or {}
            grmeta.update(tsmeta)
            cn.execute(
                f'update "{namespace}".group_registry '
                'set metadata = %(metadata)s '
                f'where name = %(name)s',
                metadata=dumps(grmeta),
                name=name
            )
        print(f'updated `{name}` with {grmeta}')


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
