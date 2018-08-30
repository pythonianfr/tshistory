from datetime import datetime

from tshistory.tsio import TimeSerie
from tshistory.testutil import genserie


def test_info(engine, cli):
    r = cli('info', engine.url)
    out = r.output.strip().split('\n')

    assert len(out) == 3
    assert out[0].startswith('changeset count:')
    assert out[1].startswith('series count:')
    assert out[2].startswith('series names:')


def test_rename(engine, cli, datadir):
    serie = genserie(datetime(2020, 1, 1), 'D', 3)

    tsh = TimeSerie()
    tsh.insert(engine, serie, 'afoo', 'Babar')
    tsh.insert(engine, serie, 'abar', 'Babar')
    tsh.insert(engine, serie, 'aquux', 'Babar')

    r = cli('rename', engine.url, datadir / 'rename.csv')

    tsh = TimeSerie()
    assert tsh.get(engine, 'afoo') is None
    assert tsh.get(engine, 'abar') is None

    for name in ('aquux', 'anew-foo', 'anew-bar'):
        assert tsh.get(engine, name) is not None
