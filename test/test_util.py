from datetime import datetime

import pytest
import pandas as pd

from tshistory.util import (
    bisect_search,
    fromjson,
    nary_pack,
    nary_unpack,
    pack_history,
    unpack_history
)
from tshistory.testutil import (
    genserie,
    utcdt
)


def test_json():
    series = pd.Series(
        [1., 2., 3.],
        index=pd.date_range(datetime(2020, 1, 1), freq='H', periods=3)
    )
    jsonseries = series.to_json(date_format='iso')
    assert jsonseries == (
        '{"2020-01-01T00:00:00.000Z":1.0,'
        '"2020-01-01T01:00:00.000Z":2.0,'
        '"2020-01-01T02:00:00.000Z":3.0}'
    )

    series2 = pd.read_json(jsonseries, typ='series', dtype=False)
    if pd.__version__.startswith('0.24'):
        assert not getattr(series2.index.dtype, 'tz', False)
        assert series.equals(series2)
    elif pd.__version__.startswith('0.25'):
        assert series2.index.dtype.tz.zone == 'UTC'
        assert not series.equals(series2)

    series2 = fromjson(jsonseries, 'foo', tzaware=False)
    if pd.__version__.startswith('0.24'):
        assert not getattr(series2.index.dtype, 'tz', False)
        assert series.equals(series2)
    elif pd.__version__.startswith('0.25'):
        assert not getattr(series2.index.dtype, 'tz', False)
        assert series.equals(series2)


def test_bisect():
    values = [-4, -2, 1, 7]
    assert bisect_search(values, -5) == -1
    assert bisect_search(values, -4) == 0
    assert bisect_search(values, -3) == 0
    assert bisect_search(values, 0) == 1
    assert bisect_search(values, 1) == 2
    assert bisect_search(values, 3) == 2
    assert bisect_search(values, 7) == 3
    assert bisect_search(values, 8) == 4


def test_pack_unpack():
    chunks = (
        b'a',
        b'bb',
        b'ccc',
        b'dd',
        b'e'
    )
    packed = nary_pack(*chunks)
    assert len(packed) == 33
    unpacked = nary_unpack(packed)
    assert chunks == unpacked

    chunks = (
        b'aurelien',
        b'campeas',
        b'develops',
        b'tshistory'
    )
    packed = nary_pack(*chunks)
    assert packed == (
        b'\x00\x00\x00\x04\x00\x00\x00\x08\x00\x00\x00\x07\x00'
        b'\x00\x00\x08\x00\x00\x00\t'
        b'aureliencampeasdevelopstshistory'
    )
    unpacked = nary_unpack(packed)
    assert chunks == unpacked

    chunks = tuple()
    packed = nary_pack(*chunks)
    unpacked = nary_unpack(packed)
    assert chunks == unpacked


def test_pack_unpack_history(tsh, engine):
    for numserie in (1, 2, 3):
        with engine.begin() as cn:
            tsh.update(cn, genserie(datetime(2021, 1, 1), 'D', numserie),
                       'small-hist-naive',
                       'aurelien.campeas@pythonian.fr',
                       insertion_date=utcdt(2021, 2, numserie))

    hist = tsh.history(engine, 'small-hist-naive')
    meta = tsh.metadata(engine, 'small-hist-naive')
    packed = pack_history(meta, hist)
    meta2, hist2 = unpack_history(packed)
    assert meta2 == meta
    for idate, series in hist.items():
        assert hist[idate].equals(series)

    for numserie in (1, 2, 3):
        with engine.begin() as cn:
            tsh.update(cn, genserie(utcdt(2021, 1, 1), 'D', numserie),
                       'small-hist-tzaware',
                       'aurelien.campeas@pythonian.fr',
                       insertion_date=utcdt(2021, 2, numserie))

    hist = tsh.history(engine, 'small-hist-tzaware')
    meta = tsh.metadata(engine, 'small-hist-tzaware')
    packed = pack_history(meta, hist)
    meta2, hist2 = unpack_history(packed)
    assert meta2 == meta
    for idate, series in hist.items():
        assert hist[idate].equals(series)


def test_in_tx(tsh, engine):
    assert tsh.type(engine, 'foo') == 'primary'

    with pytest.raises(TypeError) as err:
        tsh.update(engine.connect(), 0, 0, 0)
    assert err.value.args[0] == 'You must use a transaction object'

    ts = genserie(datetime(2017, 10, 28, 23),
                  'H', 4, tz='UTC')
    with engine.begin() as cn:
        tsh.update(cn, ts, 'test_tx', 'Babar')
        # that old api is still there
        tsh.insert(cn, ts, 'test_tx', 'Babar')
