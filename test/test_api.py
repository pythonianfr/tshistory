import pandas as pd

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
