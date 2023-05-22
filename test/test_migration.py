from tshistory.migrate import version, VERSIONS, Migrator


def test_migrate(engine):
    run = []

    @version('1.1.2')
    def migrate_foo(engine, namespace, interactive):
        run.append(('foo', namespace, interactive))


    @version('1.1.4')
    def migrate_bar(engine, namespace, interactive):
        run.append(('bar', namespace, interactive))


    @version('1.2.3')
    def migrate_quux(engine, namespace, interactive):
        run.append(('quux', namespace, interactive))


    assert len(VERSIONS) == 3

    mig = Migrator(engine, 'tsh', True)
    mig.run_migrations()
    assert run == [('foo', 'tsh', True), ('bar', 'tsh', True), ('quux', 'tsh', True)]

    run[:] = []
    Migrator._known_version = '1.1.4'
    mig.run_migrations()
    assert run == [('quux', 'tsh', True)]
