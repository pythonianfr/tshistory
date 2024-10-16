from datetime import datetime
import io

import pytest
import pandas as pd
import numpy as np
from psyl.lisp import (
    parse,
    serialize
)

from tshistory import search, tsio
from tshistory.util import (
    bisect_search,
    diff,
    fromjson,
    infer_freq,
    make_url,
    objects,
    patch,
    patchmany,
    safe_urlparse,
    unflatten,
)
from tshistory.testutil import (
    assert_df,
    genserie,
    tables,
    utcdt
)


def test_make_url():
    uri = (
        'postgresql://becurvemanager:5ewjI}kxI:&8[(<dO~}Lk*g1WT?8a"{0'
        '@host.docker.internal:5432/becurvemanager'
    )
    u = make_url(uri)
    assert str(u) == (
        'postgresql://becurvemanager:***@host.docker.internal:5432/becurvemanager'
    )


def test_safe_urlparse():
    uri = (
        'postgresql://becurvemanager:5ewjI}kxI:&8[(<dO~}Lk*g1WT?8a"{0'
        '@host.docker.internal:5432/becurvemanager'
    )
    u = safe_urlparse(uri)
    assert u.host == 'host.docker.internal'
    assert u.port == 5432
    assert u.password == '5ewjI}kxI:&8[(<dO~}Lk*g1WT?8a"{0'
    assert u.username == 'becurvemanager'
    assert u.database == 'becurvemanager'


def test_objects():
    objs = objects('migrator')
    assert len(objs) > 0  # tshistory provides the base one

    objs = objects('tshclass')
    assert len(objs) > 0  # tshistory provides the base one


def test_unflatten():
    d = {
        'a': 42,
        'b.c': 'hello',
        'b.d': 'world'
    }
    assert unflatten(d) == {
        'a': 42,
        'b': {
            'c': 'hello',
            'd': 'world'
        }
    }


def test_unflatten2():
    d  = {
        'a.uri': 'http://series.com/api',
        'a.pkce.clientid': '123zogzog1',
        'a.pkce.scope': 'all.the.series'
    }
    u = unflatten(d)
    assert u == {
        'a':  {
            'pkce.clientid': '123zogzog1',
            'pkce.scope': 'all.the.series',
            'uri': 'http://series.com/api'
        }
    }


def test_patch():
    s1 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=4)
    )
    s2 = pd.Series(
        [12., 13., np.nan, 15.],
        index=pd.date_range(datetime(2020, 1, 1, 1), freq='h', periods=4)
    )
    p = patch(s1, s2)
    assert_df("""
2020-01-01 00:00:00     1.0
2020-01-01 01:00:00    12.0
2020-01-01 02:00:00    13.0
2020-01-01 03:00:00     NaN
2020-01-01 04:00:00    15.0
""", p)

    p2 = patchmany((s1, s2))
    assert_df("""
2020-01-01 00:00:00     1.0
2020-01-01 01:00:00    12.0
2020-01-01 02:00:00    13.0
2020-01-01 03:00:00     NaN
2020-01-01 04:00:00    15.0
""", p2)

    s3 = pd.Series([], dtype=s1.dtype)
    p = patch(s3, s3)
    assert len(p) == 0

    p3 = patchmany([p2])
    assert_df("""
2020-01-01 00:00:00     1.0
2020-01-01 01:00:00    12.0
2020-01-01 02:00:00    13.0
2020-01-01 03:00:00     NaN
2020-01-01 04:00:00    15.0
""", p3)


def test_patch_tzaware():
    s1 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(utcdt(2020, 6, 23, 22), freq='h', periods=4)
    )
    s2 = pd.Series(
        [3.1, 4., 5.],
        index=pd.date_range(utcdt(2020, 6, 24), freq='h', periods=3)
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
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=4)
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
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=4)
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
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=4)
    )
    s2 = pd.Series(
        [12., 13., np.nan, 15.],
        index=pd.date_range(datetime(2020, 1, 1, 1), freq='h', periods=4)
    )
    s3 = pd.Series(
        [0., 1., 2., 13., ],
        index=pd.date_range(datetime(2019, 12, 31, 23), freq='h', periods=4)
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
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=4)
    )
    s2 = pd.Series(
        ['bb', 'cc', None, 'ee'],
        index=pd.date_range(datetime(2020, 1, 1, 1), freq='h', periods=4)
    )
    s3 = pd.Series(
        ['Z', 'a', 'b', 'cc'],
        index=pd.date_range(datetime(2019, 12, 31, 23), freq='h', periods=4)
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
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=4)
    )
    s2 = pd.Series(
        [12., 13., np.nan, 15.],
        index=pd.date_range(datetime(2020, 1, 1, 1), freq='h', periods=4)
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


def test_diff_nan_pure():
    s1 = pd.Series(
        [1, np.nan, 3],
        index=pd.date_range(datetime(2024, 1, 1), freq='h', periods=3)
    )
    s2 = pd.Series(
        [1, np.nan, 3, 4],
        index=pd.date_range(datetime(2024, 1, 1), freq='h', periods=4)
    )
    d = diff(s1, s2)
    # looks good
    assert_df("""
2024-01-01 03:00:00    4.0
""", d)

    s3 = pd.Series(
        [1, np.nan, np.nan, 4],
        index=pd.date_range(datetime(2024, 1, 1), freq='h', periods=4)
    )
    d = diff(s2, s3)
    # still good
    assert_df("""
2024-01-01 02:00:00   NaN
""", d)


def test_nan_mixed():
    n1 = pd.Series(
        [np.nan],
        index=pd.date_range(datetime(2024, 1, 1), freq='h', periods=1)
    )
    n2 = pd.Series(
        [np.nan, np.nan, 3],
        index=pd.date_range(datetime(2024, 1, 1), freq='h', periods=3)
    )
    d = diff(n1, n2)
    assert_df("""
2024-01-01 01:00:00    NaN
2024-01-01 02:00:00    3.0
""", d)

    n1 = pd.Series(
        [np.nan, 2, 3],
        index=pd.date_range(datetime(2024, 1, 1), freq='h', periods=3)
    )
    n1 = n1.drop(n1.index[1])  # erase the '2' point
    n2 = pd.Series(
        [np.nan, np.nan, 3],
        index=pd.date_range(datetime(2024, 1, 1), freq='h', periods=3)
    )
    d = diff(n1, n2)
    assert_df("""
2024-01-01 01:00:00   NaN
""", d)


def test_diff_duplicated():
    # with a duplicated row (left)
    s1 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=4)
    )
    dupe = pd.Series([5.], index=[datetime(2020, 1, 1, 3)])
    s1 = pd.concat([s1, dupe])
    s2 = pd.Series(
        [1., 2., 42., 4., .5],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=5)
    )
    with pytest.raises(ValueError):
        diff(s1, s2)


def test_infer_freq():
    s1 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=4)
    )
    d, q = infer_freq(s1)
    assert d == pd.Timedelta(hours=1)
    assert q == 1

    s2 = pd.Series(
        [1., 2., 3., None, 5.],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=5)
    )
    d, q = infer_freq(s2.dropna())
    assert d == pd.Timedelta(hours=1)
    assert q == 0.6666666666666666

    s3 = pd.Series(
        [1.],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=1)
    )
    with pytest.raises(AssertionError):
        infer_freq(s3)


def test_json():
    series = pd.Series(
        [1., 2., 3.],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=3)
    )
    jsonseries = series.to_json(date_format='iso')
    assert jsonseries == (
        '{"2020-01-01T00:00:00.000":1.0,'
        '"2020-01-01T01:00:00.000":2.0,'
        '"2020-01-01T02:00:00.000":3.0}'
    )

    series2 = pd.read_json(io.StringIO(jsonseries), typ='series', dtype=False)
    assert not getattr(series2.index.dtype, 'tz', False)
    assert series.equals(series2)

    series2 = fromjson(jsonseries, 'foo', tzaware=False)
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


def test_tables(engine, pure):
    with engine.begin() as cn:
        assert tables(cn) == [
            ('pure', 'basket'),
            ('pure', 'group_registry'),
            ('pure', 'groupmap'),
            ('pure', 'registry'),
            ('pure', 'revision_metadata'),
            ('pure-kvstore', 'kvstore'),
            ('pure-kvstore', 'things'),
            ('pure-kvstore', 'version'),
            ('pure-kvstore', 'vkvstore'),
            ('pure.group', 'registry'),
            ('pure.group', 'revision_metadata'),
            ('pure.group-kvstore', 'kvstore'),
            ('pure.group-kvstore', 'things'),
            ('pure.group-kvstore', 'version'),
            ('pure.group-kvstore', 'vkvstore')
        ]


def test_in_tx(tsh, engine):
    assert tsh.type(engine, 'foo') == 'primary'

    ts = genserie(datetime(2017, 10, 28, 23),
                  'h', 4, tz='UTC')
    with engine.begin() as cn:
        tsh.update(cn, ts, 'test_tx', 'Babar')


def test_timeseries_repr(tsh):
    if isinstance(tsh, tsio.timeseries):
        assert repr(tsh) == f'tsio.timeseries({tsh.namespace},othersources=None)'


def _serialize_roundtrip(searchobj):
    return search.query.fromexpr(searchobj.expr()).expr() == searchobj.expr()


def test_search():
    s0 = search.tzaware()
    assert s0.expr() == '(by.tzaware)'
    assert _serialize_roundtrip(s0)

    s1 = search.byname('foo bar')
    assert s1.expr() == '(by.name "foo bar")'
    assert _serialize_roundtrip(s1)

    s2 = search.or_(s0, s1)
    assert s2.expr() == '(by.or (by.tzaware) (by.name "foo bar"))'
    assert _serialize_roundtrip(s2)

    s3 = search.and_(s0, s1)
    assert s3.expr() == '(by.and (by.tzaware) (by.name "foo bar"))'
    assert _serialize_roundtrip(s3)

    s4 = search.not_(s3)
    assert s4.expr() == '(by.not (by.and (by.tzaware) (by.name "foo bar")))'
    assert _serialize_roundtrip(s4)

    s5 = search.bymetakey('key')
    assert s5.expr() == '(by.metakey "key")'
    assert _serialize_roundtrip(s5)

    s6 = search.bymetaitem('key', 'value')
    assert s6.expr() == '(by.metaitem "key" "value")'
    assert _serialize_roundtrip(s6)

    s7 = search.bymetaitem('key', 42)
    assert s7.expr() == '(by.metaitem "key" 42)'
    assert _serialize_roundtrip(s7)

    s8 = search.lt('key', 42)
    assert s8.expr() == '(< "key" 42)'
    assert _serialize_roundtrip(s8)

    s9 = search.lte('key', 42)
    assert s9.expr() == '(<= "key" 42)'
    assert _serialize_roundtrip(s9)

    s10 = search.gt('key', 42)
    assert s10.expr() == '(> "key" 42)'
    assert _serialize_roundtrip(s10)

    s11 = search.gte('key', 42)
    assert s11.expr() == '(>= "key" 42)'
    assert _serialize_roundtrip(s11)

    s12 = search.eq('key', 42)
    assert s12.expr() == '(= "key" 42)'
    assert _serialize_roundtrip(s12)

    s13 = search.eq('key', "Hello")
    assert s13.expr() == '(= "key" "Hello")'
    assert _serialize_roundtrip(s13)

    s14 = search.bysource('remote')
    assert s14.expr() == '(by.source "remote")'
    assert _serialize_roundtrip(s14)

    s15 = search.byinternalmetaitem('key', 42)
    assert s15.expr() == '(by.internal-metaitem "key" 42)'
    assert _serialize_roundtrip(s15)


def test_search_types():
    types = {}
    for lispname, kname in search._OPMAP.items():
        if not getattr(search, kname, False):
            continue
        types[lispname] = search.query.klassbyname(kname).__sig__()

    assert types == {
        '<': {'key': 'str', 'return': 'query', 'value': 'Union[str, Number, bool]'},
        '<=': {'key': 'str', 'return': 'query', 'value': 'Union[str, Number, bool]'},
        '=': {'key': 'str', 'return': 'query', 'value': 'Union[str, Number, bool]'},
        '>': {'key': 'str', 'return': 'query', 'value': 'Union[str, Number, bool]'},
        '>=': {'key': 'str', 'return': 'query', 'value': 'Union[str, Number, bool]'},
        'by.and': {'items': 'Packed[query]', 'return': 'query'},
        'by.everything': {'return': 'query'},
        'by.internal-metaitem': {'key': 'str',
                                 'return': 'query',
                                 'value': 'Union[str, Number, bool]'},
        'by.metaitem': {'key': 'str',
                        'return': 'query',
                        'value': 'Union[str, Number, bool]'},
        'by.metakey': {'key': 'MetaKey', 'return': 'query'},
        'by.name': {'query': 'str', 'return': 'query'},
        'by.not': {'item': 'query', 'return': 'query'},
        'by.or': {'items': 'Packed[query]', 'return': 'query'},
        'by.source': {'source': 'Source', 'return': 'query'},
        'by.tzaware': {'return': 'query'}
    }


def test_prune_bysource():
    """Notion of by.source filter.

    Query without it: executed as is eveywhere.

    by.source "source" -> remove all by.source <source> that do not
    match "source"

    """
    assert search.prunebysource(
        'local',
        parse('(by.source "remote")')
    ) is None

    assert serialize(
        search.prunebysource(
            'local',
            parse(
                '(by.or '
                '  (by.name "foo")'
                '  (by.source "remote"))'
            )
        )
    ) == '(by.name "foo")'

    assert search.prunebysource(
        'local',
        parse(
            '(by.and '
            '  (by.name "foo")'
            '  (by.source "remote"))'
        )
    ) is None

    assert serialize(
        search.prunebysource(
            'remote',
            parse(
                '(by.or '
                '  (by.name "foo")'
                '  (by.source "local")'
                '  (by.and '
                '    (by.name "bar")'
                '    (by.source "remote")))'
            )
        )
    ) == '(by.or (by.name "foo") (by.and (by.name "bar") (by.source "remote")))'

    assert serialize(
        search.prunebysource(
            'remote',
            parse(
                '(by.or '
                '  (by.name "foo")'
                '  (by.source "remote")'
                '  (by.and '
                '    (by.name "bar")'
                '    (by.source "local")))'
            )
        )
    ) == '(by.or (by.name "foo") (by.source "remote"))'

    q = search.prunebysource(
        'local',
        parse(
            '(by.or '
            '  (by.not (by.and (by.name "basket.fed") (by.source "local")))'
            '  (by.source "remote"))'
        )
    )
    assert q == ['by.not', ['by.and', ['by.name', 'basket.fed'], ['by.source', 'local']]]
    q2 = search.removebysource(q)
    assert q2 == ['by.not', ['by.name', 'basket.fed']]
