from datetime import timedelta
import pandas as pd
import pytest

from tshistory.api import timeseries

from tshistory.testutil import (
    assert_df,
    assert_hist,
    genserie,
    utcdt
)


def test_base_universal_api(pgapi, httpapi):
    series = pd.Series(
        [1, 2, 3],
        index=pd.date_range(
            utcdt(2020, 1, 1), periods=3, freq='D'
        )
    )

    pgapi.update(
        'api-test',
        series,
        'Babar',
        insertion_date=utcdt(2019, 1, 1),
        metadata={'about': 'test'}
    )
    out = httpapi.get('api-test')
    assert_df("""
2020-01-01 00:00:00+00:00    1.0
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
""", out)

    series[utcdt(2020, 1, 4)] = 4
    pgapi.update(
        'api-test',
        series,
        'Babar',
        insertion_date=utcdt(2019, 1, 2)
    )
    out = pgapi.get(
        'api-test',
        from_value_date=utcdt(2020, 1, 2),
        to_value_date=utcdt(2020, 1, 3)
    )
    assert_df("""
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
""", out)

    series[utcdt(2019, 12, 31)] = 0
    pgapi.replace(
        'api-test',
        series,
        'Babar',
        insertion_date=utcdt(2019, 1, 3)
    )

    out = httpapi.get('api-test')
    assert_df("""
2019-12-31 00:00:00+00:00    0.0
2020-01-01 00:00:00+00:00    1.0
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
2020-01-04 00:00:00+00:00    4.0
""", out)

    assert httpapi.type('api-test') == pgapi.type('api-test')
    assert httpapi.interval('api-test') == pgapi.interval('api-test')

    out = httpapi.get(
        'api-test',
        revision_date=utcdt(2019, 1, 1)
    )
    assert_df("""
2020-01-01 00:00:00+00:00    1.0
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
""", out)

    hist = httpapi.history(
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

    hist = pgapi.history(
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

    assert pgapi.exists('api-test')
    assert not pgapi.exists('i-dont-exist')
    assert httpapi.exists('api-test')
    assert not httpapi.exists('i-dont-exist')

    ival = pgapi.interval('api-test')
    assert ival.left == pd.Timestamp('2019-12-31 00:00:00+0000', tz='UTC')
    assert ival.right == pd.Timestamp('2020-01-04 00:00:00+0000', tz='UTC')

    meta = pgapi.metadata('api-test', all=True)
    assert meta == {
        'tzaware': True,
        'index_type': 'datetime64[ns, UTC]',
        'value_type': 'float64',
        'index_dtype': '|M8[ns]',
        'value_dtype': '<f8'
    }
    meta = httpapi.metadata('api-test')
    assert meta == {}

    pgapi.update_metadata('api-test', {
        'desc': 'a metadata test'
    })
    meta = pgapi.metadata('api-test')
    assert meta == {
        'desc': 'a metadata test'
    }

    assert pgapi.type('api-test') == 'primary'

    st = pgapi.staircase('api-test', delta=timedelta(days=366))
    assert_df("""
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
2020-01-04 00:00:00+00:00    4.0
""", st)

    pgapi.rename('api-test', 'api-test2')
    assert pgapi.exists('api-test2')
    assert not pgapi.exists('api-test')
    assert httpapi.exists('api-test2')
    assert not httpapi.exists('api-test')

    pgapi.delete('api-test2')
    assert not pgapi.exists('api-test')
    assert not httpapi.exists('api-test')
    pgapi.delete('api-test2')


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
    not formula_class() or not supervision_class(),
    reason='need formula and supervision plugins to be available'
)
def test_alternative_handler(pgapi):
    api = pgapi
    sapi = timeseries(api.uri, api.namespace, formula_class())
    sapi.update(
        'test-features',
        genserie(utcdt(2020, 1, 1), 'D', 3),
        'Babar',
    )
    sapi.tsh.register_formula(
        sapi.engine,
        'test-formula',
        '(+ 1 (series "test-features"))'
    )
    tsa = sapi.get('test-features')
    assert_df("""
2020-01-01 00:00:00+00:00    0.0
2020-01-02 00:00:00+00:00    1.0
2020-01-03 00:00:00+00:00    2.0
""", tsa)

    tsb = sapi.get('test-formula')
    assert_df("""
2020-01-01 00:00:00+00:00    1.0
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
""", tsb)

    class supervision_and_formula(supervision_class(),
                                  formula_class()):
        pass

    sapi = timeseries(api.uri, api.namespace, supervision_and_formula)
    tsa = sapi.get('test-features')
    assert_df("""
2020-01-01 00:00:00+00:00    0.0
2020-01-02 00:00:00+00:00    1.0
2020-01-03 00:00:00+00:00    2.0
""", tsa)

    tsb = sapi.get('test-formula')
    assert_df("""
2020-01-01 00:00:00+00:00    1.0
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
""", tsb)

    sapi.update(
        'test-features',
        genserie(utcdt(2020, 1, 2), 'D', 3),
        'Babar',
        manual=True
    )

    tsb = sapi.get('test-formula')
    assert_df("""
2020-01-01 00:00:00+00:00    1.0
2020-01-02 00:00:00+00:00    1.0
2020-01-03 00:00:00+00:00    2.0
2020-01-04 00:00:00+00:00    3.0
""", tsb)


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
        ('db://localhost:5433/postgres', 'ns-test-mapi'): [('api-1', 'primary')]
    }
    assert catalog2 == {
        ('db://localhost:5433/postgres', 'ns-test-mapi'): [('api-1', 'primary')],
        ('db://localhost:5433/postgres', 'ns-test-mapi-2'): [('api-2', 'primary')]
    }
    catalog3 = mapi.catalog(allsources=False)
    assert catalog3 == {
        ('db://localhost:5433/postgres', 'ns-test-mapi'): [('api-1', 'primary')]
    }

    mapi.update_metadata('api-1', {'descr': 'for the mapi test'})
    with pytest.raises(ValueError) as err:
        mapi.update_metadata('api-2', {'descr': 'for the mapi test'})
    assert err.value.args[0].startswith('not allowed to update metadata')
    assert mapi.metadata('api-2', all=True) == {
        'index_dtype': '|M8[ns]',
        'index_type': 'datetime64[ns, UTC]',
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
    assert dict(cat) == {
        ('db://localhost:5433/postgres', 'ns-test-local'): [
            ('local-series', 'primary')
        ],
        ('db://localhost:5433/postgres', 'ns-test-remote'): [
            ['remote-series', 'primary']
        ]
    }
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

    meta = mapi.metadata('remote-series', all=True)
    assert meta == {
        'index_dtype': '<M8[ns]',
        'index_type': 'datetime64[ns]',
        'tzaware': False,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }
    meta = mapi.metadata('test-localformula-remoteseries', all=True)
    assert meta == {
        'index_dtype': '<M8[ns]',
        'index_type': 'datetime64[ns]',
        'tzaware': False,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }
