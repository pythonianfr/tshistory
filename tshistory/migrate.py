from dbcache import (
    api as dbapi,
    schema as dbschema
)

from tshistory.util import (
    read_versions,
    NoVersion
)


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
