from datetime import datetime

from tshistory.tsio import TimeSerie
from tshistory.testutil import genserie


def test_info(engine, cli, tsh):
    serie = genserie(datetime(2020, 1, 1), 'D', 3)
    tsh.insert(engine, serie, 'someseries', 'Babar')

    r = cli('info', engine.url)
    out = r.output.strip().split('\n')

    assert len(out) == 3
    assert out[0].startswith('changeset count:')
    assert out[1].startswith('series count:')
    assert out[2].startswith('series names:')


def test_rename(engine, cli, datadir, tsh):
    serie = genserie(datetime(2020, 1, 1), 'D', 3)

    tsh.insert(engine, serie, 'afoo', 'Babar')
    tsh.insert(engine, serie, 'abar', 'Babar')
    tsh.insert(engine, serie, 'aquux', 'Babar')

    r = cli('rename', engine.url, datadir / 'rename.csv',
            namespace=tsh.namespace)

    tsh = TimeSerie(tsh.namespace)
    assert tsh.get(engine, 'afoo') is None
    assert tsh.get(engine, 'abar') is None

    for name in ('aquux', 'anew-foo', 'anew-bar'):
        assert tsh.get(engine, name) is not None


def test_delete(engine, cli, datadir, tsh):
    serie = genserie(datetime(2020, 1, 1), 'D', 3)

    tsh.insert(engine, serie, 'bfoo', 'Babar')
    tsh.insert(engine, serie, 'bbar', 'Babar')
    tsh.insert(engine, serie, 'bquux', 'Babar')

    r = cli('delete', engine.url, datadir / 'delete.csv',
            namespace=tsh.namespace)

    tsh = TimeSerie(tsh.namespace)
    assert tsh.get(engine, 'bfoo') is None
    assert tsh.get(engine, 'bbar') is None
    assert tsh.get(engine, 'bquux') is not None
