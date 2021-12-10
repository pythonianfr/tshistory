import pandas as pd
import pytest
import datetime as dt
import pytz

from tshistory import tsio
from tshistory.testutil import (
    assert_df,
    assert_hist,
    gengroup,
    genserie,
    utcdt
)


def test_naive(client):
    series_in = genserie(pd.Timestamp('2018-1-1'), 'H', 3)
    client.update('test-naive', series_in, 'Babar',
                  insertion_date=utcdt(2019, 1, 1))

    # now let's get it back
    ts = client.get('test-naive')
    assert_df("""
2018-01-01 00:00:00    0.0
2018-01-01 01:00:00    1.0
2018-01-01 02:00:00    2.0
""", ts)
    assert not getattr(ts.index.dtype, 'tz', False)

    idates = client.insertion_dates('test-naive')
    assert idates == [
        pd.Timestamp('2019-01-01 00:00:00+0000', tz='UTC'),
    ]



def test_error(client):
    series_in = genserie(pd.Timestamp('2018-1-1'), 'H', 3)
    client.update(
        'test-error',
        series_in,
        'Babar'
    )

    v2 = pd.Series(
        ['a', 'b', 'c'],
        pd.date_range(pd.Timestamp('2018-1-1'), freq='D', periods=3)
    )
    with pytest.raises(Exception) as err:
        client.update(
            'test-error',
            v2,
            'Babar'
        )
    assert 'new type is object' in str(err.value)
    client.delete('test-error')


def test_base(client):
    assert repr(client) == "tshistory-http-client(uri='http://perdu.com')"

    ts = client.get('no-such-series')
    assert ts is None

    assert not client.exists('no-such-series')

    meta = client.metadata('no-such-series')
    assert meta == {
        'message': '`no-such-series` does not exists'
    }

    series_in = genserie(utcdt(2018, 1, 1), 'H', 3)
    diff = client.update(
        'test1', series_in, 'Babar',
        insertion_date=utcdt(2019, 1, 1)
    )
    assert_df("""
2018-01-01 00:00:00+00:00    0.0
2018-01-01 01:00:00+00:00    1.0
2018-01-01 02:00:00+00:00    2.0
""", diff)
    assert client.exists('test1')

    diff = client.update(
        'test1', series_in, 'Babar',
        insertion_date=utcdt(2019, 1, 1)
    )
    assert len(diff) == 0

    # now let's get it back
    ts = client.get('test1')
    assert_df("""
2018-01-01 00:00:00+00:00    0.0
2018-01-01 01:00:00+00:00    1.0
2018-01-01 02:00:00+00:00    2.0
""", ts)

    ts = client.get(
        'test1',
        from_value_date=utcdt(2018, 1, 1, 2)
    )
    assert_df("""
2018-01-01 02:00:00+00:00    2.0
""", ts)

    ts = client.get(
        'test1',
        to_value_date=utcdt(2018, 1, 1, 0)
    )
    assert_df("""
2018-01-01 00:00:00+00:00    0.0
""", ts)

    # out of range
    ts = client.get(
        'test1',
        from_value_date=utcdt(2020, 1, 1, 2),
        to_value_date=utcdt(2020, 1, 1, 2)
    )
    assert len(ts) == 0
    assert ts.name == 'test1'

    meta = client.metadata('test1', all=True)
    assert meta == {
        'tzaware': True,
        'index_type': 'datetime64[ns, UTC]',
        'value_type': 'float64',
        'index_dtype': '|M8[ns]',
        'value_dtype': '<f8'
    }

    # update
    client.update_metadata('test1', {'desc': 'banana spot price'})

    meta = client.metadata('test1', all=False)
    assert meta == {
        'desc': 'banana spot price',
    }

    # check the insertion_date
    series_in = genserie(utcdt(2018, 1, 2), 'H', 3)
    client.update('test1', series_in, 'Babar',
                  metadata={'event': 'hello'},
                  insertion_date=utcdt(2019, 1, 2))

    v1 = client.get('test1', revision_date=utcdt(2019, 1, 1))
    assert_df("""
2018-01-01 00:00:00+00:00    0.0
2018-01-01 01:00:00+00:00    1.0
2018-01-01 02:00:00+00:00    2.0
""", v1)

    d1, d2 = client.insertion_dates('test1')
    assert d1 == utcdt(2019, 1, 1)
    assert d2 > d1

    client.update('test2', series_in, 'Babar')
    series = client.catalog()
    assert ['test1', 'primary'] in series[('db://localhost:5433/postgres', 'tsh')]
    assert ['test2', 'primary'] in series[('db://localhost:5433/postgres', 'tsh')]

    client.replace('test2', genserie(utcdt(2020, 1, 1), 'D', 3), 'Babar')
    series = client.get('test2')
    assert_df("""
2020-01-01 00:00:00+00:00    0.0
2020-01-02 00:00:00+00:00    1.0
2020-01-03 00:00:00+00:00    2.0
""", series)

    type = client.type('test2')
    assert type == 'primary'

    ival = client.interval('test2')
    assert ival.left == pd.Timestamp('2020-01-01 00:00:00+0000', tz='UTC')
    assert ival.right == pd.Timestamp('2020-01-03 00:00:00+0000', tz='UTC')

    client.rename('test2', 'test3')
    assert not client.exists('test2')
    assert client.exists('test3')

    client.delete('test3')
    assert not client.exists('test3')

    idates = client.insertion_dates('test1')
    assert idates == [
        pd.Timestamp('2019-01-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2019-01-02 00:00:00+0000', tz='UTC')
    ]
    idates = client.insertion_dates(
        'test1',
        from_insertion_date=utcdt(2019, 1, 2)
    )
    assert idates == [
        pd.Timestamp('2019-01-02 00:00:00+0000', tz='UTC')
    ]
    idates = client.insertion_dates(
        'test1',
        to_insertion_date=utcdt(2019, 1, 1)
    )
    assert idates == [
        pd.Timestamp('2019-01-01 00:00:00+0000', tz='UTC')
    ]


def test_dates(client):
    series_in = genserie(utcdt(2020, 10, 25), 'H', 24)
    diff = client.update(
        'test_dates', series_in, 'Babar',
        insertion_date=utcdt(2020, 10, 1)
    )

    ts = client.get(
        'test_dates',
        revision_date=utcdt(2020, 10, 1),
        from_value_date=pd.Timestamp('2020-10-25', tz='Europe/Paris'),
        to_value_date=pd.Timestamp('2020-10-25 04:00:00+01', tz='Europe/Paris')
    ).tz_convert('Europe/Paris')

    assert_df("""
2020-10-25 02:00:00+02:00    0.0
2020-10-25 02:00:00+01:00    1.0
2020-10-25 03:00:00+01:00    2.0
2020-10-25 04:00:00+01:00    3.0
""", ts)

    ts = client.get(
        'test_dates',
        revision_date=utcdt(2020, 10, 1),
        from_value_date=pytz.UTC.localize(dt.datetime(2020, 10, 25)),
        to_value_date=dt.datetime(2020, 10, 25, 3) # will be interpreted as utc
    )
    assert_df("""
2020-10-25 00:00:00+00:00    0.0
2020-10-25 01:00:00+00:00    1.0
2020-10-25 02:00:00+00:00    2.0
2020-10-25 03:00:00+00:00    3.0
""", ts)


def test_staircase_history(client):
    # each days we insert 7 data points
    for idx, idate in enumerate(pd.date_range(start=utcdt(2015, 1, 1),
                                              end=utcdt(2015, 1, 4),
                                              freq='D')):
        series = genserie(start=idate, freq='H', repeat=7)
        client.update(
            'staircase',
            series, 'Babar',
            insertion_date=idate
        )

    series = client.staircase(
        'staircase',
        pd.Timedelta(hours=3),
        from_value_date=utcdt(2015, 1, 1, 4),
        to_value_date=utcdt(2015, 1, 2, 5)
    )
    assert series.name == 'staircase'

    with pytest.raises(Exception) as err:
        client.staircase(
            'staircase',
            'NOPE',
            from_value_date=utcdt(2015, 1, 1, 4),
            to_value_date=utcdt(2015, 1, 2, 5)
        )
    assert str(err.value) == (
        'Bad Query: {"errors": {"delta": "time delta in iso 8601 duration unit abbreviation w/o a number"}, '
        '"message": "Input payload validation failed"}\n'
    )

    assert_df("""
2015-01-01 04:00:00+00:00    4.0
2015-01-01 05:00:00+00:00    5.0
2015-01-01 06:00:00+00:00    6.0
2015-01-02 03:00:00+00:00    3.0
2015-01-02 04:00:00+00:00    4.0
2015-01-02 05:00:00+00:00    5.0
""", series)

    hist = client.history('staircase')
    assert len(hist) == 4
    hist = client.history(
        'staircase',
        from_insertion_date=utcdt(2015, 1, 2),
        to_insertion_date=utcdt(2015, 1, 3)
    )
    assert len(hist) == 2
    hist = client.history(
        'staircase',
        from_value_date=utcdt(2015, 1, 1, 3),
        to_value_date=utcdt(2015, 1, 2, 1)
    )

    assert all(
        series.name == 'staircase'
        for series in hist.values()
    )

    assert_hist("""
insertion_date             value_date               
2015-01-01 00:00:00+00:00  2015-01-01 03:00:00+00:00    3.0
                           2015-01-01 04:00:00+00:00    4.0
                           2015-01-01 05:00:00+00:00    5.0
                           2015-01-01 06:00:00+00:00    6.0
2015-01-02 00:00:00+00:00  2015-01-01 03:00:00+00:00    3.0
                           2015-01-01 04:00:00+00:00    4.0
                           2015-01-01 05:00:00+00:00    5.0
                           2015-01-01 06:00:00+00:00    6.0
                           2015-01-02 00:00:00+00:00    0.0
                           2015-01-02 01:00:00+00:00    1.0
""", hist)


def test_staircase_history_naive(client):
    # each days we insert 7 data points
    from datetime import datetime
    for idx, idate in enumerate(pd.date_range(start=utcdt(2015, 1, 1),
                                              end=utcdt(2015, 1, 4),
                                              freq='D')):
        series = genserie(
            start=idate.tz_convert(None),
            freq='H',
            repeat=7
        )
        client.update(
            'staircase-naive',
            series, 'Babar',
            insertion_date=idate
        )

    series = client.staircase(
        'staircase-naive',
        pd.Timedelta(hours=3),
        from_value_date=datetime(2015, 1, 1, 4),
        to_value_date=datetime(2015, 1, 2, 5)
    )
    assert series.name == 'staircase-naive'

    assert_df("""
2015-01-01 04:00:00    4.0
2015-01-01 05:00:00    5.0
2015-01-01 06:00:00    6.0
2015-01-02 03:00:00    3.0
2015-01-02 04:00:00    4.0
2015-01-02 05:00:00    5.0
""", series)

    # series = client.staircase(
    #     'staircase-naive',
    #     pd.Timedelta(hours=3),
    #     from_value_date=datetime(2015, 1, 1, 4),
    #     to_value_date=datetime(2015, 1, 2, 5)
    # )

    hist = client.history('staircase-naive')
    assert len(hist) == 4
    hist = client.history(
        'staircase-naive',
        from_insertion_date=datetime(2015, 1, 2),
        to_insertion_date=datetime(2015, 1, 3)
    )
    assert len(hist) == 2
    hist = client.history(
        'staircase-naive',
        from_value_date=datetime(2015, 1, 1, 3),
        to_value_date=datetime(2015, 1, 2, 1)
    )

    assert all(
        series.name == 'staircase-naive'
        for series in hist.values()
    )

    assert_hist("""
insertion_date             value_date         
2015-01-01 00:00:00+00:00  2015-01-01 03:00:00    3.0
                           2015-01-01 04:00:00    4.0
                           2015-01-01 05:00:00    5.0
                           2015-01-01 06:00:00    6.0
2015-01-02 00:00:00+00:00  2015-01-01 03:00:00    3.0
                           2015-01-01 04:00:00    4.0
                           2015-01-01 05:00:00    5.0
                           2015-01-01 06:00:00    6.0
                           2015-01-02 00:00:00    0.0
                           2015-01-02 01:00:00    1.0
""", hist)


def test_log_strip(client):
    series = genserie(utcdt(2020, 1, 1), 'D', 5)
    for d in range(5):
        res = client.update(
            'test-log',
            series,
            'Babar',
            metadata={'comment': f'day {d+1}'},
            insertion_date=utcdt(2020, 1, d + 1)
        )
        series[d] = 42

    v = client.get('test-log', revision_date=dt.datetime(2020, 1, 3))
    assert_df("""
2020-01-01 00:00:00+00:00    42.0
2020-01-02 00:00:00+00:00    42.0
2020-01-03 00:00:00+00:00     2.0
2020-01-04 00:00:00+00:00     3.0
2020-01-05 00:00:00+00:00     4.0
""", v)

    v = client.get('test-log', revision_date=dt.datetime(1900, 1, 1))
    assert len(v) == 0

    log = client.log('test-log')
    assert log == [
        {'author': 'Babar',
         'date': pd.Timestamp('2020-01-01T00:00:00+00:00'),
         'meta': {'comment': 'day 1'},
         'rev': 1},
        {'author': 'Babar',
         'date': pd.Timestamp('2020-01-02T00:00:00+00:00'),
         'meta': {'comment': 'day 2'},
         'rev': 2},
        {'author': 'Babar',
         'date': pd.Timestamp('2020-01-03T00:00:00+00:00'),
         'meta': {'comment': 'day 3'},
         'rev': 3},
        {'author': 'Babar',
         'date': pd.Timestamp('2020-01-04T00:00:00+00:00'),
         'meta': {'comment': 'day 4'},
         'rev': 4},
        {'author': 'Babar',
         'date': pd.Timestamp('2020-01-05T00:00:00+00:00'),
         'meta': {'comment': 'day 5'},
         'rev': 5}
    ]

    log = client.log('test-log', limit=2)
    assert log == [
        {'author': 'Babar',
         'date': pd.Timestamp('2020-01-04T00:00:00+00:00'),
         'meta': {'comment': 'day 4'},
         'rev': 4},
        {'author': 'Babar',
         'date': pd.Timestamp('2020-01-05T00:00:00+00:00'),
         'meta': {'comment': 'day 5'},
         'rev': 5}
    ]

    log = client.log(
        'test-log',
        fromdate=utcdt(2020, 1, 2),
        todate=utcdt(2020, 1, 3)
    )
    assert log == [
        {'author': 'Babar',
         'date': pd.Timestamp('2020-01-02T00:00:00+00:00'),
         'meta': {'comment': 'day 2'},
         'rev': 2},
        {'author': 'Babar',
         'date': pd.Timestamp('2020-01-03T00:00:00+00:00'),
         'meta': {'comment': 'day 3'},
         'rev': 3}
    ]

    client.strip('test-log', utcdt(2020, 1, 3))
    log = client.log(
        'test-log',
        fromdate=utcdt(2020, 1, 2),
        todate=utcdt(2020, 1, 3)
    )
    assert log == [{
        'author': 'Babar',
        'date': pd.Timestamp('2020-01-02T00:00:00+00:00'),
        'meta': {'comment': 'day 2'},
        'rev': 2
    }]


def test_multisources(client, engine):
    series = genserie(utcdt(2020, 1, 1), 'D', 3)
    tsh = tsio.timeseries('other')

    tsh.update(engine, series, 'test-other', 'Babar')

    client.update('test-mainsource', series, 'Babar')
    with pytest.raises(ValueError) as err:
        client.update('test-other', series, 'Babar')
    assert err.value.args[0] == 'not allowed to update to a secondary source'
    with pytest.raises(ValueError) as err:
        client.replace('test-other', series, 'Babar')
    assert err.value.args[0] == 'not allowed to replace to a secondary source'

    cat = client.catalog()
    assert cat == {
        ('db://localhost:5433/postgres', 'other'): [
            ['test-other', 'primary']
        ],
        ('db://localhost:5433/postgres', 'tsh'): [
            ['test-naive', 'primary'],
            ['test1', 'primary'],
            ['test_dates', 'primary'],
            ['staircase', 'primary'],
            ['staircase-naive', 'primary'],
            ['test-log', 'primary'],
            ['test-mainsource', 'primary'],
        ]
    }
    cat = client.catalog(allsources=False)
    assert ('db://localhost:5433/postgres', 'tsh') in cat
    assert ('db://localhost:5433/postgres', 'other') not in cat


# groups

def test_stored_groups(client):
    df = gengroup(
        n_scenarios=3,
        from_date=dt.datetime(2021, 1, 1),
        length=5,
        freq='D',
        seed=2.
    )
    df.columns = ['a', 'b', 'c']

    client.group_replace('test_group', df, 'Babar')

    df2 = client.group_get('test_group')
    assert df2.equals(df)

    assert client.group_exists('test_group')
    assert client.group_type('test_group') == 'primary'
    cat = list(client.group_catalog().values())[0]
    assert ('test_group', 'primary') in cat
    assert client.group_metadata('test_group') == {}

    client.update_group_metadata('test_group', {'foo': 'bar'})
    assert client.group_metadata('test_group') == {'foo': 'bar'}

    client.group_delete('test_group')
    assert not client.group_exists('test_group')


