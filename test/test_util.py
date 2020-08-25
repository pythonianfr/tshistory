from datetime import datetime

import pytest
import pandas as pd
import numpy as np

from tshistory.util import (
    bisect_search,
    diff,
    fromjson,
    nary_pack,
    nary_unpack,
    pack_history,
    pack_many_series,
    pack_series,
    patch,
    patchmany,
    series_metadata,
    unpack_history,
    unpack_many_series,
    unpack_series
)
from tshistory.testutil import (
    assert_df,
    genserie,
    utcdt
)


def test_patch():
    s1 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(datetime(2020, 1, 1), freq='H', periods=4)
    )
    s2 = pd.Series(
        [12., 13., np.nan, 15.],
        index=pd.date_range(datetime(2020, 1, 1, 1), freq='H', periods=4)
    )
    p = patch(s1, s2)
    assert_df("""
2020-01-01 00:00:00     1.0
2020-01-01 01:00:00    12.0
2020-01-01 02:00:00    13.0
2020-01-01 03:00:00     NaN
2020-01-01 04:00:00    15.0
""", p)

    s3 = pd.Series([], dtype=s1.dtype)
    p = patch(s3, s3)
    assert len(p) == 0


def test_patch_tzaware():
    s1 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(utcdt(2020, 6, 23, 22), freq='H', periods=4)
    )
    s2 = pd.Series(
        [3.1, 4., 5.],
        index=pd.date_range(utcdt(2020, 6, 24), freq='H', periods=3)
    )
    p = patch(s1, s2)
    assert_df("""
2020-06-23 22:00:00+00:00    1.0
2020-06-23 23:00:00+00:00    2.0
2020-06-24 00:00:00+00:00    3.1
2020-06-24 01:00:00+00:00    4.0
2020-06-24 02:00:00+00:00    5.0
""", p)

    assert s1.index.dtype.name == 'datetime64[ns, UTC]'
    assert s2.index.dtype.name == 'datetime64[ns, UTC]'
    assert p.index.dtype.name == 'datetime64[ns, UTC]'

    p2 = patchmany([s1, s2])
    assert_df("""
2020-06-23 22:00:00+00:00    1.0
2020-06-23 23:00:00+00:00    2.0
2020-06-24 00:00:00+00:00    3.1
2020-06-24 01:00:00+00:00    4.0
2020-06-24 02:00:00+00:00    5.0
""", p2)

    assert p2.index.dtype.name == 'datetime64[ns, UTC]'


def test_patch_one_empty():
    s1 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(datetime(2020, 1, 1), freq='H', periods=4)
    )
    s2 = pd.Series(dtype='float64')
    p = patch(s1, s2)
    assert_df("""
2020-01-01 00:00:00    1.0
2020-01-01 01:00:00    2.0
2020-01-01 02:00:00    3.0
2020-01-01 03:00:00    4.0
""", p)


def test_patch_empty_one():
    s1 = pd.Series(dtype='float64')
    s2 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(datetime(2020, 1, 1), freq='H', periods=4)
    )
    p = patch(s1, s2)
    assert_df("""
2020-01-01 00:00:00    1.0
2020-01-01 01:00:00    2.0
2020-01-01 02:00:00    3.0
2020-01-01 03:00:00    4.0
""", p)


def test_float_patchmany():
    s1 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(datetime(2020, 1, 1), freq='H', periods=4)
    )
    s2 = pd.Series(
        [12., 13., np.nan, 15.],
        index=pd.date_range(datetime(2020, 1, 1, 1), freq='H', periods=4)
    )
    s3 = pd.Series(
        [0., 1., 2., 13., ],
        index=pd.date_range(datetime(2019, 12, 31, 23), freq='H', periods=4)
    )
    p = patchmany([s1, s2, s3])
    assert_df("""
2019-12-31 23:00:00     0.0
2020-01-01 00:00:00     1.0
2020-01-01 01:00:00     2.0
2020-01-01 02:00:00    13.0
2020-01-01 03:00:00     NaN
2020-01-01 04:00:00    15.0
""", p)

    s4 = pd.Series([], dtype=s1.dtype)
    p = patchmany([s4, s4, s4])
    assert len(p) == 0

    p = patchmany([s4, s1, s4])
    assert_df("""
2020-01-01 00:00:00    1.0
2020-01-01 01:00:00    2.0
2020-01-01 02:00:00    3.0
2020-01-01 03:00:00    4.0
""", p)


def test_string_patchmany():
    s1 = pd.Series(
        ['a', 'b', 'c', 'd'],
        index=pd.date_range(datetime(2020, 1, 1), freq='H', periods=4)
    )
    s2 = pd.Series(
        ['bb', 'cc', None, 'ee'],
        index=pd.date_range(datetime(2020, 1, 1, 1), freq='H', periods=4)
    )
    s3 = pd.Series(
        ['Z', 'a', 'b', 'cc'],
        index=pd.date_range(datetime(2019, 12, 31, 23), freq='H', periods=4)
    )
    p = patchmany([s1, s2, s3])
    assert_df("""
2019-12-31 23:00:00       Z
2020-01-01 00:00:00       a
2020-01-01 01:00:00       b
2020-01-01 02:00:00      cc
2020-01-01 03:00:00    None
2020-01-01 04:00:00      ee
""", p)


def test_diff():
    s1 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(datetime(2020, 1, 1), freq='H', periods=4)
    )
    s2 = pd.Series(
        [12., 13., np.nan, 15.],
        index=pd.date_range(datetime(2020, 1, 1, 1), freq='H', periods=4)
    )
    s2[datetime(2019, 12, 31, 23)] = -1

    ds1s2 = diff(s1, s2)
    # tail ends come as new items
    # middle elements as updates
    assert_df("""
2019-12-31 23:00:00    -1.0
2020-01-01 01:00:00    12.0
2020-01-01 02:00:00    13.0
2020-01-01 03:00:00     NaN
2020-01-01 04:00:00    15.0
""", ds1s2)

    ds2s1 = diff(s2, s1)
    # only updates there
    assert_df("""
2020-01-01 00:00:00    1.0
2020-01-01 01:00:00    2.0
2020-01-01 02:00:00    3.0
2020-01-01 03:00:00    4.0
""", ds2s1)


def test_diff_duplicated():
    # with a duplicated row (left)
    s1 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(datetime(2020, 1, 1), freq='H', periods=4)
    )
    dupe = pd.Series([5.], index=[datetime(2020, 1, 1, 3)])
    s1 = s1.append(dupe)
    s2 = pd.Series(
        [1., 2., 42., 4., .5],
        index=pd.date_range(datetime(2020, 1, 1), freq='H', periods=5)
    )
    with pytest.raises(ValueError):
        diff(s1, s2)


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


def test_pack_unpack_series():
    series1 = pd.Series(
        [1., 2., 3.],
        pd.date_range(utcdt(2021, 1, 1), freq='D', periods=3)
    )
    meta = series_metadata(series1)
    unpacked = unpack_series(
        'foo',
        pack_series(
            meta, series1
        )
    )
    assert_df("""
2021-01-01 00:00:00+00:00    1.0
2021-01-02 00:00:00+00:00    2.0
2021-01-03 00:00:00+00:00    3.0
""", unpacked)

    series2 = pd.Series(
        ['a', 'b', 'c'],
        pd.date_range(utcdt(2021, 1, 1), freq='D', periods=3)
    )
    meta = series_metadata(series1)
    assert_df("""
2021-01-01 00:00:00+00:00    1.0
2021-01-02 00:00:00+00:00    2.0
2021-01-03 00:00:00+00:00    3.0
""", unpack_series(
        'foo', pack_series(
            meta, series1
        )
    ))

    empty = pd.Series(dtype='float64')
    meta = series_metadata(series1)
    packed = pack_series(
        meta, empty
    )
    unpacked = unpack_series(
        'foo', packed
    )
    assert len(unpacked) == 0
    assert meta == {
        'index_dtype': '|M8[ns]',
        'index_type': 'datetime64[ns, UTC]',
        'tzaware': True,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }


def test_pack_unpack_many_series():
    s1 = pd.Series(
        [1., 2., 3.],
        index=pd.date_range(utcdt(2020, 1, 1), freq='H', periods=3)
    )
    s2 = pd.Series(
        [1.1, 2.2, 3.3],
        index=pd.date_range(datetime(2020, 1, 2), freq='H', periods=3)
    )
    meta1 = {
        'tzaware': True,
        'index_type': 'datetime64[ns, UTC]',
        'value_type': 'float64',
        'index_dtype': '|M8[ns]',
        'value_dtype': '<f8'
    }
    meta2 = {
        'index_dtype': '<M8[ns]',
        'index_type': 'datetime64[ns]',
        'tzaware': False,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }
    packed = pack_many_series(
        [
            (meta1, s1),
            (meta2, s2)
        ]
    )
    unpacked = unpack_many_series(packed)
    assert unpacked[0].equals(s1)
    assert unpacked[1].equals(s2)


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
        assert hist2[idate].equals(series)

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
        assert hist2[idate].equals(series)


def test_in_tx(tsh, engine):
    assert tsh.type(engine, 'foo') == 'primary'

    with pytest.raises(TypeError) as err:
        tsh.update(engine.connect(), 0, 0, 0)
    assert err.value.args[0] == 'You must use a transaction object'

    ts = genserie(datetime(2017, 10, 28, 23),
                  'H', 4, tz='UTC')
    with engine.begin() as cn:
        tsh.update(cn, ts, 'test_tx', 'Babar')


def test_timeseries_repr(tsh):
    assert repr(tsh) == f'tsio.timeseries({tsh.namespace},othersources=None)'
