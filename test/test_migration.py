from tshistory.migrate import version, VERSIONS, Migrator


def test_migrate(engine, tsh):
    # tsh only to have a proper db setup
    run = []

    # We must play with versions much higher than the current package
    # version.
    # when running this test, we have:
    # stored_version == package_version == __version__

    @version('tshistory', '99.1.2')
    def migrate_foo(engine, namespace, interactive):
        run.append(('foo', namespace, interactive))


    @version('tshistory', '99.1.4')
    def migrate_bar(engine, namespace, interactive):
        run.append(('bar', namespace, interactive))


    @version('tshistory', '99.2.3')
    def migrate_quux(engine, namespace, interactive):
        run.append(('quux', namespace, interactive))

    # >= because we also have the real migrations there
    assert len(VERSIONS) >= 3

    mig = Migrator(str(engine.url), 'tsh', False)
    mig._package_version = '99.1.5'
    mig.store.set(mig.versionkey, '1.0.0')

    mig.run_migrations()
    assert run == [
        ('foo', 'tsh', False),
        ('bar', 'tsh', False),
    ]

    run[:] = []
    mig = Migrator(str(engine.url), 'tsh', False, force='tshistory:99.2.3')
    mig.run_migrations()
    assert run == [('quux', 'tsh', False)]
