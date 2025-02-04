from tshistory.migrate import version, VERSIONS, Migrator


def _prepare(engine):
    # tsh only to have a proper db setup
    run = []

    # We must play with versions much higher than the current package
    # version.
    # when running this test, we have:
    # stored_version == package_version == __version__

    @version('tshistory', '99.2.3')
    def migrate_c(engine, namespace, interactive):
        run.append(('tsh-99.2.3', namespace, interactive))

    @version('tshistory', '99.1.4')
    def migrate_b(engine, namespace, interactive):
        run.append(('tsh-99.1.4', namespace, interactive))

    @version('tshistory', '99.1.2')
    def migrate_a(engine, namespace, interactive):
        run.append(('tsh-99.1.2', namespace, interactive))

    @version('foo', '0.1.0')
    def migrate_d(engine, namespace, interactive):
        run.append(('foo-0.1.0', namespace, interactive))

    @version('foo', '0.1.1')
    def migrate_e(engine, namespace, interactive):
        run.append(('foo-0.1.1', namespace, interactive))

    # >= because we also have the real migrations there
    assert len(VERSIONS) >= 5

    return run


def test_migrate(engine, tsh):
    run = _prepare(engine)

    mig = Migrator(str(engine.url), 'tsh', False)
    mig._package_version = '99.1.5'
    mig.store.set(mig.versionkey, '1.0.0')

    mig.run_migrations()
    assert run == [
        ('tsh-99.1.2', 'tsh', False),
        ('tsh-99.1.4', 'tsh', False)
    ]


def test_migrate_force(engine, tsh):
    run = _prepare(engine)

    mig = Migrator(str(engine.url), 'tsh', False, force='tshistory:99.2.3')
    mig.run_migrations()
    assert run == [('tsh-99.2.3', 'tsh', False)]


def test_migrate_last(engine, tsh):
    run = _prepare(engine)

    # test 'last'

    mig = Migrator(str(engine.url), 'tsh', False, last=True)
    mig._package_version = '99.1.5'
    mig.store.set(mig.versionkey, '1.0.0')

    mig.run_migrations()
    assert run == [
        ('tsh-99.2.3', 'tsh', False),
    ]

    mig = Migrator(str(engine.url), 'tsh', False, last=True)
    mig._package_version = '0.1.0'
    mig._package = 'foo'
    mig.store.set(mig.versionkey, '0.1.0')
    mig.run_migrations()
    assert run == [
        ('tsh-99.2.3', 'tsh', False),
        ('foo-0.1.1', 'tsh', False)
    ]
