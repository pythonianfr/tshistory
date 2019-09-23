from datetime import timedelta
import pandas as pd
import pytest

from tshistory.api import timeseries, multisourcetimeseries

from tshistory.testutil import (
    assert_df,
    assert_hist,
    utcdt
)


def test_bare_get(api):
    series = pd.Series(
        [1, 2, 3],
        index=pd.date_range(
            utcdt(2020, 1, 1), periods=3, freq='D'
        )
    )

    api.update(
        series, 'api-test', 'Babar',
        insertion_date=utcdt(2019, 1, 1),
        metadata={'about': 'test'}
    )
    out = api.get('api-test')
    assert_df("""
2020-01-01 00:00:00+00:00    1.0
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
""", out)

    series[utcdt(2020, 1, 4)] = 4
    api.update(
        series, 'api-test', 'Babar',
        insertion_date=utcdt(2019, 1, 2)
    )
    out = api.get(
        'api-test',
        from_value_date=utcdt(2020, 1, 2),
        to_value_date=utcdt(2020, 1, 3)
    )
    assert_df("""
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
""", out)

    series[utcdt(2019, 12, 31)] = 0
    api.replace(
        series, 'api-test', 'Babar',
        insertion_date=utcdt(2019, 1, 3)
    )
    out = api.get('api-test')
    assert_df("""
2019-12-31 00:00:00+00:00    0.0
2020-01-01 00:00:00+00:00    1.0
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
2020-01-04 00:00:00+00:00    4.0
""", out)

    out = api.get(
        'api-test',
        revision_date=utcdt(2019, 1, 1)
    )
    assert_df("""
2020-01-01 00:00:00+00:00    1.0
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
""", out)

    hist = api.history(
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

    hist = api.history(
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

    assert api.exists('api-test')
    assert not api.exists('i-dont-exist')

    ival = api.interval('api-test')
    assert ival.left == pd.Timestamp('2019-12-31 00:00:00+0000', tz='UTC')
    assert ival.right == pd.Timestamp('2020-01-04 00:00:00+0000', tz='UTC')

    meta = api.metadata('api-test', all=True)
    assert meta == {
        'tzaware': True,
        'index_type': 'datetime64[ns, UTC]',
        'value_type': 'float64',
        'index_dtype': '|M8[ns]',
        'value_dtype': '<f8'
    }
    meta = api.metadata('api-test')
    assert meta == {}

    api.update_metadata('api-test', {
        'desc': 'a metadata test'
    })
    meta = api.metadata('api-test')
    assert meta == {
        'desc': 'a metadata test'
    }

    assert api.type('api-test') == 'primary'

    st = api.staircase('api-test', delta=timedelta(days=366))
    assert_df("""
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
2020-01-04 00:00:00+00:00    4.0
""", st)

    api.rename('api-test', 'api-test2')
    assert api.exists('api-test2')
    assert not api.exists('api-test')

    api.delete('api-test2')
    assert not api.exists('api-test')
    api.delete('api-test2')


def test_multisource(mapi):

    def create(uri, ns, name):
        api = timeseries(uri, ns)
        series = pd.Series(
            [1, 2, 3],
            index=pd.date_range(
                utcdt(2020, 1, 1), periods=3, freq='D'
            )
        )

        api.update(
            series, name, 'Babar',
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
            series, name, 'Babar',
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
    create(mapi.uri, 'test-api-2', 'api-2')

    assert not mapi.exists('i-dont-exist')
    assert mapi.exists('api-1')
    assert mapi.exists('api-2')

    series = pd.Series(
        [10, 20, 30],
        index=pd.date_range(
            utcdt(2020, 1, 1), periods=3, freq='D'
        )
    )
    mapi.update(series, 'api-1', 'auc')

    with pytest.raises(ValueError) as err:
        mapi.update(series, 'api-2', 'auc')
    assert err.value.args[0].startswith('not allowed to update')

    mapi.replace(series, 'api-1', 'auc')

    with pytest.raises(ValueError) as err:
        mapi.replace(series, 'api-2', 'auc')
    assert err.value.args[0].startswith('not allowed to replace')


    api = timeseries(mapi.uri, mapi.namespace)
    catalog = api.catalog()
    catalog2 = mapi.catalog()
    assert catalog == catalog2
    assert catalog == {'api-1': 'primary'}
