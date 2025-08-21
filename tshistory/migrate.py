from json import dumps
import os

from sqlhelp.pgapi import pgdb

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
    # __slots__ = 'uri', 'namespace', 'interactive', 'start', 'force', 'last'

    def __init__(self, uri, namespace, interactive=False, start=None, force=None, last=False):
        self.uri = uri
        self.namespace = namespace
        self.interactive = interactive
        self.start = start
        # "package:version"
        self.force = force.split(':') if force else (None, None)
        self.last = last

    @property
    def engine(self):
        return pgdb(self.uri)

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
        except Exception:
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
        print(f'Running migrations for {self._package} (ns={self.namespace}).')
        # determine from where we start (stored version or provided
        # initial)
        if self.last:
            to_migrate = [
                ver
                for ver in VERSIONS
                if ver.package == self._package
            ]
            to_migrate.sort(key=lambda ver: ver.get_number())
            if to_migrate:
                to_migrate = [to_migrate[-1]]
        else:
            start = self.initialversion

            if start.raw_version == '0.0.0':
                # first time
                print(f'Initial migration to {self._package_version}')
                self.initial_migration()

            to_migrate = list(VERSIONS)

            end = self.finalversion

            to_migrate = [
                ver for ver in to_migrate
                if start < ver <= end
                and ver.package == self._package
            ]
            to_migrate.sort(key=lambda ver: ver.get_number())

        # build migration plan (from stored_version to package_version
        # or forced_version)

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


@version('tshistory', '0.22.0')
def migrate_022(engine, namespace, interactive):
    do_migrate_tree(engine, namespace, interactive)
    do_make_ltree_unique(engine, namespace, interactive)
    do_migrate_old_metadata(engine, namespace, interactive)
    do_migrate_revision_metadata(engine, namespace, interactive)
    do_enforce_series_metadata_integrity(engine, namespace, interactive)
    do_enforce_groups_metadata_integrity(engine, namespace, interactive)
    do_enforce_series_metadata_integrity(engine, f'{namespace}.group', interactive)
    do_migrate_basket_kinds(engine, namespace, interactive)
    do_cleanup_kvstore(engine, f'{namespace}.group', interactive)

    # Fix indexes with explicit expected indexes for tshistory
    from tshistory.sqlparser import (
        parse_indexes,
        TSHISTORY_SQLFILES,
        TSHISTORY_PATH
    )

    tshistory_indexes = parse_indexes(TSHISTORY_SQLFILES, namespace)
    do_fix_indexes(engine, namespace, interactive, tshistory_indexes)

    # Also fix indexes for .group namespace
    group_indexes = parse_indexes(
        [TSHISTORY_PATH / 'registry.sql'],
        f'{namespace}.group'
    )
    do_fix_indexes(engine, f'{namespace}.group', interactive, group_indexes)


def create_revision_metadata_for_ns(engine, ns):
    """Create revision_metadata table for a namespace"""
    print(f'create revision_metadata table for {ns}')
    with engine.begin() as cn:
        cn.execute(
            f'create table if not exists "{ns}".revision_metadata ('
            f'  id serial primary key,'
            f'  series integer not null references "{ns}".registry(id) on delete cascade,'
            f'  author text not null,'
            f'  metadata jsonb'
            f')'
        )

        cn.execute(
            f'create index if not exists "{ns}_revision_metadata_series_idx" '
            f'on "{ns}".revision_metadata(series)'
        )


def do_migrate_revision_metadata(engine, namespace, interactive):
    """Create revision_metadata table for commit history support"""
    # Create for main namespace
    create_revision_metadata_for_ns(engine, namespace)

    # Create for .group namespace
    create_revision_metadata_for_ns(engine, f'{namespace}.group')


def do_migrate_basket_kinds(engine, namespace, interactive):
    """Add kind column to basket table for group support"""
    ns = namespace

    with engine.begin() as cn:
        type_exists = cn.execute(
            f"""
            select exists (
                select 1 from pg_type t
                join pg_namespace n on t.typnamespace = n.oid
                where n.nspname = '{ns}'
                and t.typname = 'kinds'
            )
            """
        ).scalar()

        if not type_exists:
            cn.execute(
                f'create type "{ns}".kinds as enum (\'Series\', \'Group\')'
            )

        cn.execute(
            f'alter table "{ns}".basket add column if not exists kind "{ns}".kinds '
            f'not null default \'Series\''
        )
        cn.execute(
            f'alter table "{ns}".basket drop constraint if exists basket_name_key'
        )

        constraint_exists = cn.execute(
            f"""
            select exists (
                select 1 from pg_constraint c
                join pg_namespace n on n.oid = c.connamespace
                join pg_class r on r.oid = c.conrelid
                where n.nspname = '{ns}'
                and r.relname = 'basket'
                and c.conname = 'basket_name_kind_key'
            )
            """
        ).scalar()

        if not constraint_exists:
            cn.execute(
                f'alter table "{ns}".basket add constraint basket_name_kind_key '
                f'unique(name, kind)'
            )

        cn.execute(
            f'create index if not exists "{ns}_basket_kind_idx" on "{ns}".basket (kind)'
        )


def do_migrate_tree(engine, namespace, interactive):
    ns = namespace
    with engine.begin() as cn:
        cn.execute(f"""
create extension if not exists ltree;

create table if not exists "{ns}".tree (
  id serial primary key,
  path ltree
);

create index if not exists tree_path_idx on "{ns}".tree using gist (path);


create table if not exists "{ns}".tree_series_map (
  seriesid integer unique references "{ns}".registry (id) on delete cascade,
  treeid integer references "{ns}".tree (id) on delete cascade
);

create index if not exists tree_series_map_idx on "{ns}".tree_series_map (treeid);
""", _binary=False)


def do_make_ltree_unique(engine, namespace, interactive):
    ns = namespace
    with engine.begin() as cn:
        cn.execute(
            f'alter table "{ns}".tree drop constraint if exists unique_tree;'
            f'alter table "{ns}".tree drop constraint if exists tree_path_key;'
            f'alter table "{ns}".tree add unique (path);',
            _binary=False
        )


def do_migrate_old_metadata(engine, namespace, interactive):
    ns = namespace
    with engine.begin() as cn:
        cn.execute(f"""
create table if not exists "{ns}".ts_oldmeta (
  moment timestamptz unique not null default now(),
  seriesid integer not null references "{ns}".registry (id) on delete cascade,
  userid text default 'no-user',
  metadata jsonb not null
);

create index if not exists "{ns}_ts_oldmeta_moment_idx" on "{ns}".ts_oldmeta (moment);
create index if not exists "{ns}_ts_oldmeta_seriesid_idx" on "{ns}".ts_oldmeta (seriesid);

create table if not exists "{ns}".gr_oldmeta (
  moment timestamptz unique not null default now(),
  groupid integer not null references "{ns}".group_registry (id) on delete cascade,
  userid text default 'no-user',
  metadata jsonb not null
);

create index if not exists "{ns}_gr_oldmeta_moment_idx" on "{ns}".gr_oldmeta (moment);
create index if not exists "{ns}_gr_oldmeta_groupid_idx" on "{ns}".gr_oldmeta (groupid);
""", _binary=False)


def do_enforce_series_metadata_integrity(engine, namespace, interactive):
    print(f'enforce series metadata integrity for {namespace}')

    with engine.begin() as cn:
        # Data migration: ensure all NULL metadata values are set to empty objects
        cn.execute(
            f'update "{namespace}".registry '
            f'set metadata = %s '
            f'where metadata is null',
            '{}'
        )

        cn.execute(
            f'update "{namespace}".registry '
            f'set internal_metadata = %s '
            f'where internal_metadata is null',
            '{}'
        )

        # Schema migration: add constraints and defaults to match schema files
        # registry: add default and NOT NULL constraints
        cn.execute(
            f'alter table "{namespace}".registry '
            f'alter column metadata set default \'{{}}\' ::jsonb'
        )

        # Only add NOT NULL if column is currently nullable
        if cn.execute(
            "select is_nullable from information_schema.columns "
            f"where table_schema = '{namespace}' and table_name = 'registry' "
            "and column_name = 'metadata'"
        ).scalar() == 'YES':
            cn.execute(
                f'alter table "{namespace}".registry '
                f'alter column metadata set not null'
            )

        if cn.execute(
            "select is_nullable from information_schema.columns "
            f"where table_schema = '{namespace}' and table_name = 'registry' "
            "and column_name = 'internal_metadata'"
        ).scalar() == 'YES':
            cn.execute(
                f'alter table "{namespace}".registry '
                f'alter column internal_metadata set not null'
            )


def do_enforce_groups_metadata_integrity(engine, namespace, interactive):
    print(f'enforce groups metadata integrity for {namespace}')

    with engine.begin() as cn:
        # Data migration: ensure all NULL metadata values are set to empty objects
        cn.execute(
            f'update "{namespace}".group_registry '
            f'set metadata = %s '
            f'where metadata is null',
            '{}'
        )

        cn.execute(
            f'update "{namespace}".group_registry '
            f'set internal_metadata = %s '
            f'where internal_metadata is null',
            '{}'
        )

        # Schema migration: add constraints and defaults to match schema files
        # group_registry: add default and NOT NULL constraints
        cn.execute(
            f'alter table "{namespace}".group_registry '
            f'alter column metadata set default \'{{}}\' ::jsonb'
        )

        if cn.execute(
            "select is_nullable from information_schema.columns "
            f"where table_schema = '{namespace}' and table_name = 'group_registry' "
            "and column_name = 'metadata'"
        ).scalar() == 'YES':
            cn.execute(
                f'alter table "{namespace}".group_registry '
                f'alter column metadata set not null'
            )

        if cn.execute(
            "select is_nullable from information_schema.columns "
            f"where table_schema = '{namespace}' and table_name = 'group_registry' "
            "and column_name = 'internal_metadata'"
        ).scalar() == 'YES':
            cn.execute(
                f'alter table "{namespace}".group_registry '
                f'alter column internal_metadata set not null'
            )


def do_cleanup_kvstore(engine, namespace, interactive):
    kvstore_ns = f'{namespace}-kvstore'

    if interactive:
        if not yesno(f'Drop unnecessary kvstore schema "{kvstore_ns}" ? [y/n] '):
            return

    print(f'dropping unnecessary kvstore schema "{kvstore_ns}"')

    with engine.begin() as cn:
        cn.execute(f'drop schema if exists "{kvstore_ns}" cascade')


def do_fix_indexes(engine, namespace, interactive, indexes):
    from tshistory import dbdiag

    print(f'fix indexes to match naming convention for {namespace}')

    # Get current state
    actual = dbdiag.get_actual_indexes(engine, namespace)
    issues = dbdiag.find_issues(indexes, actual)

    # Only proceed if there are issues
    if not issues['wrong_name'] and not issues['missing'] and not issues['duplicates']:
        print('  All indexes are already correct')
        return

    # Report what we'll fix
    if issues['wrong_name']:
        print(f"  Will rename {len(issues['wrong_name'])} indexes")
    if issues['missing']:
        print(f"  Will create {len(issues['missing'])} missing indexes")
    if issues['duplicates']:
        print(f"  Will drop {len(issues['duplicates'])} duplicate index sets")

    # Use dbdiag to fix all index issues
    dbdiag.fix_indexes(engine, namespace, indexes)
    print('  Index operations completed')


@version('tshistory', '0.21.0')
def do_migrate_intervals(engine, namespace, interactive):
    migrate_intervals(engine, namespace, interactive)
    migrate_intervals(engine, f'{namespace}.group', interactive)

    # cleanup the schema
    with engine.begin() as cn:
        cn.execute(
            f'drop table if exists "{namespace}.group".basket'
        )


def migrate_intervals(engine, namespace, interactive):
    tsh = tshclass(namespace)
    with engine.begin() as cn:
        tables = {
            name: tsh._series_to_tablename(cn, name)
            for name in tsh.list_series(engine).keys()
        }

    for name, tablename in tables.items():
        if tablename is None:
            continue  # not a primary

        with engine.begin() as cn:
            imeta = tsh.internal_metadata(engine, name)
            if 'left' in imeta:
                # already migrated
                continue

            start, end = cn.execute(
                f'select tsstart, tsend from "{namespace}.revision"."{tablename}" '
                f'order by id desc limit 1'
            ).fetchone()
            start = start.isoformat() if start else None
            end = end.isoformat() if end else None
            tsh.update_internal_metadata(
                cn, name, {'left': start, 'right': end}
            )

            cn.execute(
                f'alter table "{namespace}.revision"."{tablename}" '
                f'drop column tsstart, drop column tsend'
            )


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
        engine = pgdb(url)
        for name in names:
            with engine.begin() as cn:
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

    if not migdata:
        print(
            'Do not forget to schedule the "migrate_diffs" task to complete the migration.'
        )


def migrate_seriesdata_diffstart_diffend(engine, namespace, name):
    tsh = tshclass(namespace)
    pid = os.getpid()

    with engine.begin() as cn:
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
            f'create index if not exists "{ns}_registry_metadata_idx" '
            f'on "{ns}".registry using gin (metadata)'
        )
        cn.execute(
            f'create index if not exists "{ns}_registry_internal_metadata_idx" '
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
            f'create index if not exists "{ns}_group_registry_metadata_idx" '
            f'on "{ns}".group_registry using gin (metadata)'
        )
        cn.execute(
            f'create index if not exists "{ns}_group_registry_internal_metadata_idx" '
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
