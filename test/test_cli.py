from datetime import datetime

from tshistory.tsio import TimeSerie
from tshistory.testutil import genserie, utcdt


def test_info(engine, cli, tsh):
    serie = genserie(datetime(2020, 1, 1), 'D', 3)
    tsh.insert(engine, serie, 'someseries', 'Babar')

    r = cli('info', engine.url,
            namespace=tsh.namespace)
    out = r.output.strip().split('\n')

    assert len(out) == 3
    assert out[0].startswith('changeset count:')
    assert out[1].startswith('series count:')
    assert out[2].startswith('series names:')


def test_log(engine, cli, tsh):
    serie = genserie(datetime(2020, 1, 1), 'D', 3)
    tsh.insert(engine, serie, 'log_me', 'Babar')
    serie = genserie(datetime(2020, 1, 2), 'D', 3)
    tsh.insert(engine, serie, 'log_me', 'Babar')

    r = cli('log', engine.url,
            serie='log_me',
            namespace=tsh.namespace)

    assert r.output.count('revision:') == 2


def test_history(engine, cli, tsh):
    serie = genserie(datetime(2020, 1, 1), 'D', 3)
    tsh.insert(engine, serie, 'some_history', 'Babar',
               _insertion_date=utcdt(2019, 1, 1))
    serie = genserie(datetime(2020, 1, 2), 'D', 3)
    tsh.insert(engine, serie, 'some_history', 'Babar',
               _insertion_date=utcdt(2019, 1, 2))

    r = cli('history', engine.url,
            'some_history',
            json=True)

    assert r.output == (
        '{"2019-01-01 00:00:00+00:00": {"2020-01-01 00:00:00": 0.0, "2020-01-02 00:00:00": 1.0, "2020-01-03 00:00:00": 2.0}, "2019-01-02 00:00:00+00:00": {"2020-01-02 00:00:00": 0.0, "2020-01-03 00:00:00": 1.0, "2020-01-04 00:00:00": 2.0}}\n'
    )


def test_check(engine, cli, tsh):
    serie = genserie(datetime(2020, 1, 1), 'D', 3)
    tsh.insert(engine, serie, 'things', 'Babar')
    tsh.insert(engine, serie, 'other_things', 'Babar')

    r = cli('check', engine.url,
            namespace=tsh.namespace)
    assert 'things inserts=1, read-time=' in r.output
    assert 'other_things inserts=1, read-time=' in r.output


def test_rename(engine, cli, datadir, tsh):
    serie = genserie(datetime(2020, 1, 1), 'D', 3)

    tsh.insert(engine, serie, 'afoo', 'Babar')
    tsh.insert(engine, serie, 'abar', 'Babar')
    tsh.insert(engine, serie, 'aquux', 'Babar')

    r = cli('rename', engine.url, datadir / 'rename.csv',
            namespace=tsh.namespace)

    tsh._resetcaches()

    assert tsh.get(engine, 'afoo') is None
    assert tsh.get(engine, 'abar') is None

    for name in ('aquux', 'anew-foo', 'anew-bar'):
        assert tsh.get(engine, name) is not None


def test_delete(engine, cli, datadir, tsh):
    serie = genserie(datetime(2020, 1, 1), 'D', 3)

    tsh.insert(engine, serie, 'bfoo', 'Babar')
    tsh.insert(engine, serie, 'bbar', 'Babar')
    tsh.insert(engine, serie, 'bquux', 'Babar')

    r = cli('delete', engine.url,
            deletefile=datadir / 'delete.csv',
            namespace=tsh.namespace)

    tsh = TimeSerie(tsh.namespace)
    tsh._testing = True
    assert not tsh.exists(engine, 'bfoo')
    assert tsh.get(engine, 'bfoo') is None
    assert tsh.get(engine, 'bbar') is None
    assert tsh.get(engine, 'bquux') is not None

    tsh.insert(engine, serie, 'bbq', 'Babar')

    tsh = TimeSerie(tsh.namespace)
    tsh._testing = True
    r = cli('delete', engine.url,
            series='bbq',
            namespace=tsh.namespace)

    assert not tsh.exists(engine, 'bbq')
