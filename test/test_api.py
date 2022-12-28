from datetime import timedelta, datetime as dt
import io
import pytest

import pandas as pd
import numpy as np

from tshistory.api import timeseries
from tshistory.testutil import (
    assert_df,
    assert_hist,
    gengroup,
    genserie,
    hist_from_csv,
    ts_from_csv,
    utcdt
)


def formula_class():
    try:
        from tshistory_formula.tsio import timeseries
    except ImportError:
        return
    return timeseries


def supervision_class():
    try:
        from tshistory_supervision.tsio import timeseries
    except ImportError:
        return
    return timeseries


@pytest.mark.skipif(
    not formula_class(),
    reason='need formula plugin to be available'
)
def test_base_universal_api(tsx):
    for name in ('api-test',):
        tsx.delete(name)

    series = pd.Series(
        [1, 2, 3],
        index=pd.date_range(
            utcdt(2020, 1, 1), periods=3, freq='D'
        )
    )

    tsx.update(
        'api-test',
        series,
        'Babar',
        insertion_date=utcdt(2019, 1, 1),
        metadata={'about': 'test'}
    )

    out = tsx.get('api-test')
    assert_df("""
2020-01-01 00:00:00+00:00    1.0
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
""", out)

    series[utcdt(2020, 1, 4)] = 4
    tsx.update(
        'api-test',
        series,
        'Babar',
        insertion_date=utcdt(2019, 1, 2)
    )
    out = tsx.get(
        'api-test',
        from_value_date=utcdt(2020, 1, 2),
        to_value_date=utcdt(2020, 1, 3)
    )
    assert_df("""
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
""", out)

    series[utcdt(2019, 12, 31)] = 0
    tsx.replace(
        'api-test',
        series,
        'Babar',
        insertion_date=utcdt(2019, 1, 3)
    )

    out = tsx.get('api-test')
    assert_df("""
2019-12-31 00:00:00+00:00    0.0
2020-01-01 00:00:00+00:00    1.0
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
2020-01-04 00:00:00+00:00    4.0
""", out)

    assert tsx.type('api-test') == 'primary'
    assert tsx.interval('api-test') == pd.Interval(
        pd.Timestamp('2019-12-31', tz='UTC'),
        pd.Timestamp('2020-01-04', tz='UTC'),
        closed='both'
    )

    out = tsx.get(
        'api-test',
        revision_date=utcdt(2019, 1, 1)
    )
    assert_df("""
2020-01-01 00:00:00+00:00    1.0
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
""", out)

    hist = tsx.history(
        'api-test'
    )
    assert_hist("""
insertion_date             value_date               
2019-01-01 00:00:00+00:00  2020-01-01 00:00:00+00:00    1.0
                           2020-01-02 00:00:00+00:00    2.0
                           2020-01-03 00:00:00+00:00    3.0
2019-01-02 00:00:00+00:00  2020-01-01 00:00:00+00:00    1.0
                           2020-01-02 00:00:00+00:00    2.0
                           2020-01-03 00:00:00+00:00    3.0
                           2020-01-04 00:00:00+00:00    4.0
2019-01-03 00:00:00+00:00  2019-12-31 00:00:00+00:00    0.0
                           2020-01-01 00:00:00+00:00    1.0
                           2020-01-02 00:00:00+00:00    2.0
                           2020-01-03 00:00:00+00:00    3.0
                           2020-01-04 00:00:00+00:00    4.0
""", hist)

    empty_hist = tsx.history(
        'api-test',
        from_insertion_date=pd.Timestamp('2020-1-1', tz='UTC')
    )
    assert empty_hist == {}

    hist = tsx.history(
        'api-test',
        diffmode=True
    )
    assert_hist("""
insertion_date             value_date               
2019-01-01 00:00:00+00:00  2020-01-01 00:00:00+00:00    1.0
                           2020-01-02 00:00:00+00:00    2.0
                           2020-01-03 00:00:00+00:00    3.0
2019-01-02 00:00:00+00:00  2020-01-04 00:00:00+00:00    4.0
2019-01-03 00:00:00+00:00  2019-12-31 00:00:00+00:00    0.0
""", hist)

    assert tsx.exists('api-test')
    assert not tsx.exists('i-dont-exist')

    idates = tsx.insertion_dates('api-test')
    assert idates == [
        pd.Timestamp('2019-01-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2019-01-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2019-01-03 00:00:00+0000', tz='UTC')
    ]

    idates = tsx.insertion_dates(
        'api-test',
        from_value_date=pd.Timestamp('2020-1-4', tz='UTC'),
    )

    assert idates == [
        pd.Timestamp('2019-01-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2019-01-03 00:00:00+0000', tz='UTC')
    ]

    meta = tsx.internal_metadata('api-test')
    assert meta == {
        'tzaware': True,
        'tablename': 'api-test',
        'index_type': 'datetime64[ns, UTC]',
        'value_type': 'float64',
        'index_dtype': '|M8[ns]',
        'value_dtype': '<f8'
    }
    meta = tsx.metadata('api-test')
    assert meta == {}

    tsx.update_metadata('api-test', {
        'desc': 'a metadata test'
    })
    meta = tsx.metadata('api-test')
    assert meta == {
        'desc': 'a metadata test'
    }

    assert tsx.type('api-test') == 'primary'

    st = tsx.staircase('api-test', delta=timedelta(days=366))
    assert_df("""
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
2020-01-04 00:00:00+00:00    4.0
""", st)


def test_block_staircase(tsx):
    hist = hist_from_csv(io.StringIO("""
datetime,               2020-01-01 08:00+0, 2020-01-02 08:00+0, 2020-01-03 08:00+0
2020-01-03 00:00+00:00, 1.0,                10.0,               100.0
2020-01-03 04:00+00:00, 2.0,                20.0,               200.0
2020-01-03 08:00+00:00, 3.0,                30.0,               300.0
2020-01-03 16:00+00:00, 4.0,                40.0,               400.0
2020-01-04 00:00+00:00, 5.0,                50.0,               500.0
2020-01-04 04:00+00:00, 6.0,                60.0,               600.0
2020-01-04 08:00+00:00, 7.0,                70.0,               700.0
2020-01-04 16:00+00:00, 8.0,                80.0,               800.0
"""))
    for idate, ts in hist.items():
        tsx.update('test_b_staircase', ts, author='test', insertion_date=idate)

    computed_ts = tsx.block_staircase(
        'test_b_staircase',
        from_value_date=pd.Timestamp('2020-01-03', tz='utc'),
        to_value_date=pd.Timestamp('2020-01-05', tz='utc'),
        revision_freq={'days': 1},
        revision_time={'hour': 10},
        revision_tz='UTC',
        maturity_offset={'hours': 24},
        maturity_time={'hour': 4},
    )
    expected_ts = ts_from_csv(io.StringIO("""
datetime,               value
2020-01-03 00:00+00:00, 1.0
2020-01-03 04:00+00:00, 20.0
2020-01-03 08:00+00:00, 30.0
2020-01-03 16:00+00:00, 40.0
2020-01-04 00:00+00:00, 50.0
2020-01-04 04:00+00:00, 600.0
2020-01-04 08:00+00:00, 700.0
2020-01-04 16:00+00:00, 800.0
"""))
    pd.testing.assert_series_equal(computed_ts, expected_ts, check_names=False)


@pytest.mark.skipif(
    not formula_class() or not supervision_class(),
    reason='need formula and supervision plugins to be available'
)
def test_autotrophic_idates(mapihttp):
    from tshistory_formula import api
    from tshistory_formula.registry import func, finder

    @func('autotrophic')
    def custom() -> pd.Series:
        return pd.Series(
            [1, 2, 3],
            pd.date_range(utcdt(2020, 1, 1), periods=1, freq='D')
        )

    @finder('autotrophic')
    def custom(cn, tsh, tree):
        return {
            'I HAVE A NAME FOR DISPLAY PURPOSES': tree
        }

    mapihttp.register_formula(
        'autotrophic-idates',
        '(autotrophic)'
    )

    idates = mapihttp.insertion_dates('autotrophic-idates')
    assert idates == []


@pytest.mark.skipif(
    not formula_class(),
    reason='need formula plugin to be available'
)
def test_log(tsx):
    for name in ('log-me',):
        tsx.delete(name)

    series = genserie(utcdt(2020, 1, 1), 'D', 3, initval=[1])
    tsx.update(
        'log-me',
        series,
        'Babar',
        {'foo': 'A', 'bar': 42},
        insertion_date=utcdt(2020, 1, 1)
    )

    log = tsx.log('log-me')
    assert log == [{
        'rev': 1,
        'author': 'Babar',
        'date': pd.Timestamp('2020-1-1', tz='UTC'),
        'meta': {'foo': 'A', 'bar': 42}
    }]

    series[1] = 42
    tsx.update(
        'log-me',
        series,
        'Babar',
        insertion_date=utcdt(2020, 1, 2)
    )
    log = tsx.log('log-me', limit=1)
    assert len(log) == 1


@pytest.mark.skipif(
    not formula_class(),
    reason='need formula plugin to be available'
)
def test_multisource(mapi):
    for methname in ('get', 'update', 'replace', 'exists', 'type',
                     'history', 'staircase',
                     'catalog', 'interval',
                     'metadata', 'update_metadata',
                     'rename', 'delete'
    ):
        assert getattr(mapi, methname, False), methname


    def create(uri, ns, name):
        api = timeseries(uri, ns)
        series = pd.Series(
            [1, 2, 3],
            index=pd.date_range(
                utcdt(2020, 1, 1), periods=3, freq='D'
            )
        )

        api.update(
            name,
            series,
            'Babar',
            insertion_date=utcdt(2019, 1, 1),
            metadata={'about': 'test'}
        )
        out = api.get(name)
        assert_df("""
2020-01-01 00:00:00+00:00    1.0
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
""", out)

        series[utcdt(2020, 1, 4)] = 4
        api.update(
            name,
            series,
            'Babar',
            insertion_date=utcdt(2019, 1, 2)
        )
        out = api.get(
            name,
            from_value_date=utcdt(2020, 1, 2),
            to_value_date=utcdt(2020, 1, 3)
        )
        assert_df("""
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
""", out)

    create(mapi.uri, mapi.namespace, 'api-1')
    create(mapi.uri, 'ns-test-mapi-2', 'api-2')

    assert not mapi.exists('i-dont-exist')
    assert mapi.exists('api-1')
    assert mapi.exists('api-2')

    series = pd.Series(
        [10, 20, 30],
        index=pd.date_range(
            utcdt(2020, 1, 1), periods=3, freq='D'
        )
    )
    mapi.update('api-1', series, 'auc')

    with pytest.raises(ValueError) as err:
        mapi.update('api-2', series, 'auc')
    assert err.value.args[0].startswith('not allowed to update')

    mapi.replace('api-1', series, 'auc')

    with pytest.raises(ValueError) as err:
        mapi.replace('api-2', series, 'auc')
    assert err.value.args[0].startswith('not allowed to replace')


    api = timeseries(mapi.uri, mapi.namespace)
    catalog = api.catalog()
    catalog2 = mapi.catalog()
    assert catalog == {
        ('postgres@ns-test-mapi', 'ns-test-mapi'): [('api-1', 'primary')]
    }
    assert catalog2 == {
        ('postgres@ns-test-mapi', 'ns-test-mapi'): [('api-1', 'primary')],
        ('postgres@ns-test-mapi-2', 'ns-test-mapi-2'): [('api-2', 'primary')]
    }
    catalog3 = mapi.catalog(allsources=False)
    assert catalog3 == {
        ('postgres@ns-test-mapi', 'ns-test-mapi'): [('api-1', 'primary')]
    }

    mapi.update_metadata('api-1', {'descr': 'for the mapi test'})
    with pytest.raises(ValueError) as err:
        mapi.update_metadata('api-2', {'descr': 'for the mapi test'})
    assert err.value.args[0].startswith('not allowed to update metadata')
    assert mapi.internal_metadata('api-2') == {
        'index_dtype': '|M8[ns]',
        'index_type': 'datetime64[ns, UTC]',
        'tablename': 'api-2',
        'tzaware': True,
        'value_dtype': '<f8',
        'value_type': 'float64',
        'supervision_status': 'unsupervised'
    }
    assert mapi.metadata('api-2', all=True) == {
        'index_dtype': '|M8[ns]',
        'index_type': 'datetime64[ns, UTC]',
        'supervision_status': 'unsupervised',
        'tablename': 'api-2',
        'tzaware': True,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }

    mapi.rename('api-1', 'renamed-api-1')
    assert not mapi.exists('api-1')
    with pytest.raises(ValueError) as err:
        mapi.rename('api-2', 'renamed-api-2')
    assert err.value.args[0].startswith('not allowed to rename')

    mapi.delete('renamed-api-1')
    with pytest.raises(ValueError) as err:
        mapi.delete('api-2')
    assert err.value.args[0].startswith('not allowed to delete')

    assert not mapi.exists('renamed-api-1')

    # local shadowing of api-3
    create(mapi.uri, mapi.namespace, 'api-3')
    create(mapi.uri, 'ns-test-mapi-2', 'api-3')
    # update local version metadata
    mapi.update_metadata('api-3', {'foo': 'bar'})
    # delete local version
    mapi.delete('api-3')
    # remote version still exists
    assert mapi.exists('api-3')
    with pytest.raises(ValueError):
        mapi.update_metadata('api-3', {'foo': 'bar'})
    with pytest.raises(ValueError):
        mapi.delete('api-3')
    cat = mapi.catalog()
    assert cat == {
        ('postgres@ns-test-mapi-2', 'ns-test-mapi-2'): [
            ('api-2', 'primary'),
            ('api-3', 'primary')]
    }


@pytest.mark.skipif(
    not formula_class(),
    reason='need formula plugin to be available'
)
def test_http_api():
    tsh = timeseries('https://my.fancy.timeseries.store')
    for methname in ('get', 'update', 'replace', 'exists', 'type',
                     'history', 'staircase',
                     'catalog', 'interval',
                     'metadata', 'update_metadata',
                     'rename', 'delete'
    ):
        assert getattr(tsh, methname, False), methname


@pytest.mark.skipif(
    not formula_class(),
    reason='need formula plugin to be available'
)
def test_local_formula_remote_series(mapihttp, engine):
    from tshistory_formula.tsio import timeseries as pgseries

    mapi = mapihttp
    assert repr(mapi) == (
        'timeseries(uri=postgresql://localhost:5433/postgres,'
        'ns=ns-test-local,'
        'sources=[source(uri=http://test-uri2,ns=ns-test-remote), '
        'source(uri=http://unavailable,ns=ns-test-unavailable-remote)])'
    )

    assert len(mapi.othersources.sources) == 2
    assert mapi.namespace == 'ns-test-local'
    assert mapi.uri == 'postgresql://localhost:5433/postgres'
    assert mapi.othersources.sources[0].namespace == 'ns-test-remote'
    assert mapi.othersources.sources[0].uri == 'http://test-uri2'

    series = pd.Series(
        [1, 2, 3],
        index=pd.date_range(pd.Timestamp('2020-1-1'), periods=3, freq='H'),
    )
    mapi.update('local-series', series, 'Babar')

    rtsh = pgseries('ns-test-remote')
    rtsh.update(
        engine,
        series,
        'remote-series',
        'Celeste',
        insertion_date=pd.Timestamp('2020-1-1', tz='UTC')
    )

    cat = mapi.catalog(allsources=True)
    s1 = cat[('postgres@ns-test-local', 'ns-test-local')]

    assert ('local-series', 'primary') in s1

    s2 = list(cat[('postgres@ns-test-remote', 'ns-test-remote')])
    assert ['remote-series', 'primary'] in s2

    mapi.register_formula(
        'test-localformula-remoteseries',
        '(+ 1 (series "remote-series"))'
    )

    ts = mapi.get('test-localformula-remoteseries')
    assert_df("""
2020-01-01 00:00:00    2.0
2020-01-01 01:00:00    3.0
2020-01-01 02:00:00    4.0
""", ts)

    hist = mapi.history('test-localformula-remoteseries')
    assert_hist("""
insertion_date             value_date         
2020-01-01 00:00:00+00:00  2020-01-01 00:00:00    2.0
                           2020-01-01 01:00:00    3.0
                           2020-01-01 02:00:00    4.0
""", hist)

    ival = mapi.interval('remote-series')
    assert (ival.left, ival.right) == (
        pd.Timestamp('2020-01-01 00:00:00'),
        pd.Timestamp('2020-01-01 02:00:00')
    )

    with pytest.raises(ValueError) as err:
        mapi.interval('test-localformula-remoteseries')

    assert err.value.args[0] == 'no interval for series: test-localformula-remoteseries'

    meta = mapi.internal_metadata('remote-series')
    assert meta == {
        'index_dtype': '<M8[ns]',
        'index_type': 'datetime64[ns]',
        'tablename': 'remote-series',
        'tzaware': False,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }
    meta = mapi.internal_metadata('test-localformula-remoteseries')
    assert meta == {
        'index_dtype': '<M8[ns]',
        'index_type': 'datetime64[ns]',
        'tzaware': False,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }


@pytest.mark.skipif(
    not formula_class(),
    reason='need formula plugin to be available'
)
def test_formula_remote_autotrophic(mapihttp, engine):
    tsa = mapihttp

    from tshistory_formula.registry import func, metadata
    from tshistory_formula.tsio import timeseries as pgseries

    @func('customseries')
    def customseries() -> pd.Series:
        return pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.date_range(dt(2019, 1, 1), periods=3, freq='D')
        )

    @metadata('customseries')
    def metadata(cn, tsh, tree):
        return {
            tree[0]: {
                'tzaware': True,
                'index_type': 'datetime64[ns, UTC]',
                'value_type': 'float64',
                'index_dtype': '|M8[ns]',
                'value_dtype': '<f8'
            }
        }

    rtsh = pgseries('ns-test-remote')
    with engine.begin() as cn:
        rtsh.register_formula(
            cn,
            'autotrophic',
            '(customseries)',
        )

    tsa.register_formula(
        'remote-series',
        '(series "autotrophic")'
    )

    # bw compat
    assert tsa.metadata('remote-series', True) == {
        'index_dtype': '|M8[ns]',
        'index_type': 'datetime64[ns, UTC]',
        'tzaware': True,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }

    assert tsa.metadata('remote-series') == {}

    assert tsa.internal_metadata('remote-series') == {
        'index_dtype': '|M8[ns]',
        'index_type': 'datetime64[ns, UTC]',
        'tzaware': True,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }

    assert_df("""
2019-01-01 00:00:00+00:00    1.0
2019-01-02 00:00:00+00:00    2.0
2019-01-03 00:00:00+00:00    3.0
""", tsa.get('remote-series'))

    idates = tsa.insertion_dates('remote-series')
    # autotrophic series lack an `insertion_dates` protocol
    assert idates == []

    assert tsa.type('remote-series') == 'formula'
    assert tsa.othersources.sources[0].tsa.type('autotrophic') == 'formula'
    assert tsa.type('autotrophic') == 'formula'


def test_strip(tsx):
    for name in ('stripme',):
        tsx.delete(name)

    for i in range(3):
        ts = pd.Series(
            np.array([1, 2, 3]) + i,
            pd.date_range(utcdt(2021, 1, 1), freq='D', periods=3)
        )
        tsx.update(
            'stripme',
            ts,
            'Babar',
            insertion_date=utcdt(2021, 1, 1+i)
        )

    revs = tsx.insertion_dates('stripme')
    assert revs == [
        pd.Timestamp('2021-01-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2021-01-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2021-01-03 00:00:00+0000', tz='UTC')
    ]

    # in the future: a noop
    tsx.strip('stripme', utcdt(2021, 1, 31))
    revs = tsx.insertion_dates('stripme')
    assert len(revs) == 3

    # remove two
    tsx.strip('stripme', utcdt(2021, 1, 2))
    revs = tsx.insertion_dates('stripme')
    assert revs == [
        pd.Timestamp('2021-01-01 00:00:00+0000', tz='UTC'),
    ]

    # wipe all
    tsx.strip('stripme', utcdt(2021, 1, 1))
    revs = tsx.insertion_dates('stripme')
    assert revs == []

    ts = pd.Series(
        [1, 2, 3],
        pd.date_range(utcdt(2021, 1, 1), freq='D', periods=3)
    )
    # now this is interesting ... tsh.interval wants at least a revision
    with pytest.raises(Exception):
        tsx.update(
            'stripme',
            ts,
            'Babar',
            insertion_date=utcdt(2021, 1, 10)
        )


def test_conflicting_update(mapi):
    # behaviour when a series exists locally and remotely
    mapi.update(
        'here-and-there',
        pd.Series(
            [1, 2, 3],
            index=pd.date_range(utcdt(2020, 1, 1), periods=3, freq='D')
        ),
        'Babar'
    )
    # create a series with the same name in the other source
    remotesource = mapi.othersources.sources[0]
    remote = timeseries(
        remotesource.uri,
        namespace=remotesource.namespace
    )
    remote.update(
        'here-and-there',
        pd.Series(
            [1, 2, 3],
            index=pd.date_range(utcdt(2020, 1, 1), periods=3, freq='D')
        ),
        'Babar'
    )

    mapi.update(
        'here-and-there',
        pd.Series(
            [1, 2, 3, 4],
            index=pd.date_range(utcdt(2020, 1, 1), periods=4, freq='D')
        ),
        'Babar'
    )

    mapi.replace(
        'here-and-there',
        pd.Series(
            [1, 2, 3, 4],
            index=pd.date_range(utcdt(2020, 1, 1), periods=4, freq='D')
        ),
        'Babar'
    )

    # all allowed :)


# groups

def test_primary_group(tsx):
    for name in ('first_group_api',):
        tsx.group_delete(name)

    df = gengroup(
        n_scenarios=3,
        from_date=dt(2021, 1, 1),
        length=5,
        freq='D',
        seed=2
    )

    colnames = ['a', 'b', 'c']
    df.columns = colnames

    assert_df("""
            a  b  c
2021-01-01  2  3  4
2021-01-02  3  4  5
2021-01-03  4  5  6
2021-01-04  5  6  7
2021-01-05  6  7  8
    """, df)

    # first insert
    tsx.group_replace(
        'first_group_api',
        df,
        author='Babar',
        insertion_date=utcdt(2021, 1, 1)
    )
    assert tsx.group_type('first_group_api') == 'primary'
    assert tsx.group_exists('first_group_api')

    meta = tsx.group_metadata('first_group_api')
    assert meta == {}
    tsx.update_group_metadata('first_group_api', {'name': 'babar'})

    meta = tsx.group_metadata('first_group_api')
    assert meta == {'name': 'babar'}

    meta = tsx.group_metadata('first_group_api', all=True)
    meta.pop('tablename')
    assert meta == {
        'index_dtype': '<M8[ns]',
        'index_type': 'datetime64[ns]',
        'name': 'babar',
        'tzaware': False,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }

    assert tsx.group_metadata('no-such-group', all=True) is None
    assert tsx.group_metadata('no-such-group') is None

    df = tsx.group_get('first_group_api')
    assert_df("""
              a    b    c
2021-01-01  2.0  3.0  4.0
2021-01-02  3.0  4.0  5.0
2021-01-03  4.0  5.0  6.0
2021-01-04  5.0  6.0  7.0
2021-01-05  6.0  7.0  8.0
    """, df)

    # update
    df = gengroup(
        n_scenarios=3,
        from_date=dt(2021, 1, 2),
        length=5,
        freq='D',
        seed=-1
    )
    df.columns = colnames

    assert_df("""
            a  b  c
2021-01-02 -1  0  1
2021-01-03  0  1  2
2021-01-04  1  2  3
2021-01-05  2  3  4
2021-01-06  3  4  5
    """, df)

    tsx.group_replace('first_group_api', df, author='Babar')
    df = tsx.group_get('first_group_api')
    assert_df("""
              a    b    c
2021-01-02 -1.0  0.0  1.0
2021-01-03  0.0  1.0  2.0
2021-01-04  1.0  2.0  3.0
2021-01-05  2.0  3.0  4.0
2021-01-06  3.0  4.0  5.0
    """, df)

    # the update did work
    # let's load the previous version  (inserted in 2021-01-01)
    df = tsx.group_get(
        'first_group_api',
        revision_date=utcdt(2021, 1, 2)
    )
    assert_df("""
              a    b    c
2021-01-01  2.0  3.0  4.0
2021-01-02  3.0  4.0  5.0
2021-01-03  4.0  5.0  6.0
2021-01-04  5.0  6.0  7.0
2021-01-05  6.0  7.0  8.0
""", df)

    tsx.group_rename('first_group_api', 'new_name_api')
    assert not tsx.group_exists('first_group_api')
    assert tsx.group_exists('new_name_api')
    df2 = tsx.group_get(
        'new_name_api',
        revision_date=utcdt(2021, 1, 2)
    )

    assert df2.equals(df)


def test_group_errors(tsx):
    df = gengroup(
        n_scenarios=3,
        from_date=dt(2021, 1, 1),
        length=5,
        freq='D',
        seed=2
    )

    df.columns = ['a', 'b', 'c']
    tsx.group_replace(
        'group_error',
        df,
        author='Babar',
        insertion_date=utcdt(2021, 1, 1)
    )

    assert_df("""
            a  b  c
2021-01-01  2  3  4
2021-01-02  3  4  5
2021-01-03  4  5  6
2021-01-04  5  6  7
2021-01-05  6  7  8
    """, df)

    df2 = df[['a', 'b', 'c', 'a']]
    df2.columns = ['a', 'b', 'c', 'd']

    with pytest.raises(Exception) as excinfo:
        tsx.group_replace(
            'group_error',
            df['a'],
            author='Celeste'
        )
    assert str(excinfo.value) == (
        'group `group_error` must be updated with a dataframe'
    )

    with pytest.raises(Exception) as excinfo:
        tsx.group_replace(
            'group_error',
            df[['a', 'b']],
            author='Celeste'
        )
    assert str(excinfo.value) == (
        'group update error for `group_error`: `c` columns are missing'
    )

    with pytest.raises(Exception) as excinfo:
        tsx.group_replace(
            'group_error',
            df[['a', 'b', 'c', 'a']],
            author='Celeste'
        )
    assert 'duplicated' in str(excinfo.value) or 'redundant' in str(excinfo.value)

    with pytest.raises(Exception) as excinfo:
        tsx.group_replace(
            'group_error',
            df2,
            author='Celeste'
        )
    assert str(excinfo.value) == (
        'group update error for `group_error`: `d` columns are in excess'
    )

    # when dataframes columns are indexed with integer

    df = gengroup(
        n_scenarios=3,
        from_date=dt(2021, 1, 1),
        length=5,
        freq='D',
        seed=2
    )

    assert [0, 1, 2] == df.columns.to_list()
    tsx.group_replace('group_with_int_api', df, 'test')
    tsx.group_replace('group_with_int_api', df, 'test')
    df = tsx.group_get('group_with_int_api')

    # the integers are coerced into strings
    assert ['0', '1', '2'] == df.columns.to_list()


def test_group_catalog(tsx):
    # cleanup
    cat = list(tsx.group_catalog().values())
    if cat:
        for name, _ in cat[0]:
            tsx.group_delete(name)

    df = gengroup(
        n_scenarios=4,
        from_date=dt(2021, 1, 1),
        length=4,
        freq='D',
        seed=4
    )

    tsx.group_replace(
        'list-me',
        df,
        author='Babar',
        insertion_date=utcdt(2021, 1, 1)
    )

    lgroups = tsx.group_catalog()

    assert list(lgroups.values())[0] == [
        ('list-me', 'primary')
    ]

    tsx.group_delete('list-me')

    # the group disapeared
    assert len(list(tsx.group_catalog().values())) == 0


def test_more_group_errors(tsx):
    tsx.delete('toto')
    tsx.delete('tata')

    df = gengroup(
        n_scenarios=3,
        from_date=dt(2015, 1, 1),
        length=5,
        freq='D',
        seed=2
    )

    tsx.group_replace('conflict-name-primary', df, 'toto')
    tsx.register_group_formula('conflict-name-formula', '(group "toto")')

    # toto does not exist: for homogneity purposes an error should be raised

    # bad bindings
    with pytest.raises(Exception) as excinfo:
        tsx.register_formula_bindings(
            'conflict-name-binding',
            'toto',
            pd.DataFrame()
        )
    assert str(excinfo.value) == 'bindings must have `series` `groups` and `family` columns'

    with pytest.raises(Exception) as excinfo:
        tsx.register_formula_bindings(
            'conflict-name-binding',
            'toto',
            pd.DataFrame([['a', 'b', 'c']], columns=('series', 'group', 'family'))
        )
    assert str(excinfo.value) == '`toto` is not a formula'

    tsx.register_formula(
        'toto',
        '(constant 42.5 (date "1900-1-1") (date "2039-12-31") "D" (date "1900-1-1"))'
    )

    with pytest.raises(Exception) as excinfo:
        tsx.register_formula_bindings(
            'conflict-name-binding',
            'toto',
            pd.DataFrame([['a', 'b', 'c']], columns=('series', 'group', 'family'))
        )
    assert str(excinfo.value) == 'Group `b` does not exist.'

    for name in 'abc':
        tsx.group_replace(
            name,
            df,
            'Babar'
        )
        tsx.update(
            name,
            pd.Series(
                [1, 2, 3],
                pd.date_range(pd.Timestamp('2021-1-1'), freq='D', periods=3)
            ),
            'Babar'
        )

    tsx.register_formula_bindings(
        'conflict-name-binding',
        'toto',
        pd.DataFrame([['a', 'b', 'c']], columns=('series', 'group', 'family'))
    )

    with pytest.raises(Exception) as excinfo:
        tsx.group_replace('conflict-name-formula', df, 'toto')
    assert str(excinfo.value) == (
        'cannot group-replace `conflict-name-formula`: this name has type `formula`'
    )

    with pytest.raises(Exception) as excinfo:
        tsx.group_replace('conflict-name-binding', df, 'toto')
    assert str(excinfo.value) == (
        'cannot group-replace `conflict-name-binding`: this name has type `bound`'
    )

    with pytest.raises(Exception) as excinfo:
        tsx.register_group_formula('conflict-name-primary', '(group "toto")')
    assert str(excinfo.value) == (
        'cannot register formula `conflict-name-primary`: already a `primary`'
    )

    with pytest.raises(Exception) as excinfo:
        tsx.register_group_formula('conflict-name-binding', '(group "toto")')
    assert str(excinfo.value) == (
        'cannot register formula `conflict-name-binding`: already a `bound`'
    )

    with pytest.raises(Exception) as excinfo:
        tsx.register_formula_bindings(
            'conflict-name-primary',
            'toto',
            pd.DataFrame([['a', 'b', 'c']], columns=('series', 'group', 'family'))
        )
    assert str(excinfo.value) == (
        'cannot bind `conflict-name-primary`: already a primary'
    )
    with pytest.raises(Exception) as excinfo:
        tsx.register_formula_bindings(
            'conflict-name-formula',
            'toto',
            pd.DataFrame([['a', 'b', 'c']], columns=('series', 'group', 'family'))
        )
    assert str(excinfo.value) == (
        'cannot bind `conflict-name-formula`: already a formula'
    )

    # overrides
    tsx.register_group_formula('conflict-name-formula', '(group "tata")')

    # goes smoothly
    tsx.register_formula(
        'tata',
        '(constant 42.5 (date "1900-1-1") (date "2039-12-31") "D" (date "1900-1-1"))'
    )
    tsx.register_formula_bindings(
        'conflict-name-binding',
        'tata',
        pd.DataFrame([['a', 'b', 'c']], columns=('series', 'group', 'family'))
    )


def test_history_group(tsx):
    for idx, idate in enumerate(
            pd.date_range(
                start=utcdt(2022, 1, 1),
                end=utcdt(2022, 1, 5),
                freq='D'
            )
    ):
        df = gengroup(
            n_scenarios=3,
            from_date=idate.date(),  # tz-naive because daily
            length=3,
            freq='D',
            seed=10 * idx
        )
        tsx.group_replace('history_group', df, 'test', insertion_date=idate)

    hist = tsx.group_history(
        'history_group',
        from_value_date=dt(2022, 1, 3),
        to_value_date=dt(2022, 1, 6),
        from_insertion_date=utcdt(2022, 1, 2),
        to_insertion_date=utcdt(2022, 1, 4),
)

    assert_hist("""
                                         0     1     2
insertion_date            value_date                  
2022-01-02 00:00:00+00:00 2022-01-03  11.0  12.0  13.0
                          2022-01-04  12.0  13.0  14.0
2022-01-03 00:00:00+00:00 2022-01-03  20.0  21.0  22.0
                          2022-01-04  21.0  22.0  23.0
                          2022-01-05  22.0  23.0  24.0
2022-01-04 00:00:00+00:00 2022-01-04  30.0  31.0  32.0
                          2022-01-05  31.0  32.0  33.0
                          2022-01-06  32.0  33.0  34.0
    """, hist)

    idates = tsx.group_insertion_dates('history_group')
    assert idates == [
        pd.Timestamp('2022-01-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2022-01-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2022-01-03 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2022-01-04 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2022-01-05 00:00:00+0000', tz='UTC'),
    ]

    idates = tsx.group_insertion_dates(
        'history_group',
        from_insertion_date=utcdt(2022, 1, 2),
        to_insertion_date=utcdt(2022, 1, 3),
    )
    assert idates == [
        pd.Timestamp('2022-01-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2022-01-03 00:00:00+0000', tz='UTC'),
    ]

    # group does not exist
    assert tsx.group_insertion_dates('no_such_group') is None
