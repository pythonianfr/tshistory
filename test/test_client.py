import io
import pandas as pd
from pathlib import Path
import pytest
import datetime as dt
import pytz

from tshistory import tsio
from tshistory.testutil import (
    assert_df,
    assert_hist,
    gengroup,
    genserie,
    gen_value_ranges,
    hist_from_csv,
    ts_from_csv,
    utcdt
)

DATADIR = Path(__file__).parent / 'data'


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


def test_block_staircase_no_series(client):
    assert client.block_staircase(
        name='no-such-series',
        from_value_date=pd.Timestamp('2021-10-29', tz='Europe/Brussels'),
        to_value_date=pd.Timestamp('2021-10-30', tz='Europe/Brussels'),
    ) is None


def test_block_staircase_empty_series(client):
    insert_date = pd.Timestamp('2021-10-15', tz='Europe/Brussels')
    value_start_date = insert_date + pd.Timedelta(1, 'D')
    ts = genserie(start=value_start_date, freq='H', repeat=24)
    client.update(
        'staircase-missed-insertion', ts, 'test', insertion_date=insert_date
    )

    # some data should be retrieved with 1-day-ahead revision
    ts = client.block_staircase(
        name='staircase-missed-insertion',
        from_value_date=value_start_date,
        to_value_date=value_start_date + pd.Timedelta(1, 'D'),
        revision_freq={'days': 1},
        revision_time={'hour': 9},
        revision_tz='Europe/Brussels',
        maturity_offset={'days': 1},
    )
    assert not ts.empty

    # not data should be retrieved outside value range
    ts = client.block_staircase(
        name='staircase-missed-insertion',
        from_value_date=value_start_date + pd.Timedelta(2, 'D'),
        to_value_date=value_start_date + pd.Timedelta(3, 'D'),
        revision_freq={'days': 1},
        revision_time={'hour': 9},
        revision_tz='Europe/Brussels',
        maturity_offset={'days': 1},
    )
    assert ts.empty

    # not data should be retrieved outside revision range
    ts = client.block_staircase(
        name='staircase-missed-insertion',
        from_value_date=value_start_date,
        to_value_date=value_start_date + pd.Timedelta(1, 'D'),
        revision_freq={'days': 1},
        revision_time={'hour': 9},
        revision_tz='Europe/Brussels',
        maturity_offset={'days': 3},
    )
    assert ts.empty


@pytest.mark.parametrize(
    ['ts_name', 'source_ts_is_tz_aware', 'revision_tz', 'expected_output_tz'],
    [
        ('tz_test_1', False, 'utc', None),
        ('tz_test_2', False, 'CET', None),
        ('tz_test_3', True, 'utc', 'utc'),
        ('tz_test_4', True, 'CET', 'CET'),
    ]
)
def test_block_staircase_output_timezone(
    client, ts_name, source_ts_is_tz_aware, revision_tz, expected_output_tz
):
    insert_date = pd.Timestamp('2021-10-15', tz='utc')
    value_start_date = insert_date + pd.Timedelta(1, 'D')
    ts = genserie(start=value_start_date, freq='H', repeat=24)
    ts = ts if source_ts_is_tz_aware else ts.tz_localize(None)
    client.update(ts_name, ts, 'test', insertion_date=insert_date)
    sc_ts = client.block_staircase(
        ts_name,
        from_value_date=value_start_date,
        to_value_date=value_start_date + pd.Timedelta(1, 'D'),
        revision_freq={'days': 1},
        revision_time={'hour': 9},
        maturity_offset={'days': 1},
        revision_tz=revision_tz,
    )
    if expected_output_tz:
        expected_output_tz = pytz.timezone(expected_output_tz)
    assert sc_ts.index.tz == expected_output_tz


def test_block_staircase_arg_errors(client):
    start_date = pd.Timestamp('2021-10-15', tz='Europe/Brussels')
    ts = genserie(start=start_date, freq='H', repeat=24)
    client.update(
        'block-staircase-arg-error', ts, 'test', insertion_date=start_date
    )

    with pytest.raises(Exception) as exc:
        _ = client.block_staircase(
            name='block-staircase-arg-error',
            from_value_date=start_date,
            to_value_date=start_date + pd.Timedelta(1, 'D'),
            revision_freq='WRONG_ARG_TYPE',
        )
    assert 'Expected shift offset `revision_freq` as dict' in str(exc)

    with pytest.raises(Exception) as exc:
        _ = client.block_staircase(
            name='block-staircase-arg-error',
            from_value_date=start_date,
            to_value_date=start_date + pd.Timedelta(1, 'D'),
            revision_time={'WRONG_KEY_NAME': 4},
        )
    assert 'revision_time' in str(exc)
    assert 'WRONG_KEY_NAME' in str(exc)


def test_block_staircase_revision_errors(client):
    """Test errors returned by block_staircase wit wrong arguments"""
    start_date = pd.Timestamp('2021-10-15', tz='Europe/Brussels')
    ts = genserie(start=start_date, freq='H', repeat=24)
    client.update(
        'block-staircase-rev-error', ts, 'test', insertion_date=start_date
    )

    # revisions with null frequency
    with pytest.raises(Exception) as exc:
        _ = client.block_staircase(
            name='block-staircase-rev-error',
            from_value_date=start_date,
            to_value_date=start_date + pd.Timedelta(1, 'D'),
            revision_freq={'days': 0},
            revision_time={'hour': 9},
            revision_tz='Europe/Brussels',
        )
    assert 'non-increasing block start dates' in str(exc)

    # revisions with identical block starts fixed on 1st of month
    with pytest.raises(Exception) as exc:
        _ = client.block_staircase(
            name='block-staircase-rev-error',
            from_value_date=start_date,
            to_value_date=start_date + pd.Timedelta(1, 'D'),
            revision_freq={'days': 1},
            revision_time={'hour': 9},
            revision_tz='Europe/Brussels',
            maturity_time={'day': 1},
            maturity_offset={'days': 0},
        )
    assert 'non-increasing block start dates' in str(exc)


def run_block_staircase_value_test(
    client, ts_name, hist_csv, staircase_csv, sc_kwargs, value_date_lag="1D"
):
    # Load history on db
    for idate, ts in hist_from_csv(hist_csv).items():
        client.update(ts_name, ts, "test", insertion_date=idate)

    # Expected output of block_staircase function
    sc_ts = ts_from_csv(staircase_csv)
    if sc_ts.index.tzinfo: # align expected output tz with revision_tz if tz-aware
        sc_ts = sc_ts.tz_convert(sc_kwargs.get("revision_tz") or "utc")
    sc_idx = sc_ts.index

    # Compute staircase and check output values on different value ranges
    value_ranges = gen_value_ranges(sc_idx[0], sc_idx[-1], lag=value_date_lag)
    for from_v_date, to_v_date in value_ranges:
        computed_ts = client.block_staircase(
            name=ts_name,
            from_value_date=from_v_date,
            to_value_date=to_v_date,
            **sc_kwargs,
        )
        expected_ts = sc_ts[
            (sc_idx >= (from_v_date or sc_idx[0])) &
            (sc_idx <= (to_v_date or sc_idx[-1]))
        ]
        pd.testing.assert_series_equal(
            computed_ts, expected_ts, check_freq=False, check_names=False
        )


def test_block_staircase_basic_daily(client):
    hist = io.StringIO("""
datetime,   2020-01-01 08:00+0, 2020-01-01 16:00+0, 2020-01-02 08:00+0, 2020-01-02 16:00+0, 2020-01-03 08:00+0
2020-01-01, 1.0,                -1.0,               NA,                 NA,                 NA
2020-01-02, 2.0,                -2.0,               6.0,                -6.0,               NA
2020-01-03, 3.0,                -3.0,               7.0,                -7.0,               11.0
2020-01-04, 4.0,                -4.0,               8.0,                -8.0,               12.0
2020-01-05, 5.0,                -5.0,               9.0,                -9.0,               13.0
2020-01-06, NA,                 NA,                 10.0,               -10.0,              14.0
2020-01-07, NA,                 NA,                 NA,                 NA,                 15.0
2020-01-08, NA,                 NA,                 NA,                 NA,                 NA
""")
    sc_kwargs = dict(
        revision_freq={"hours": 24},
        revision_time={"hour": 9},
        revision_tz="UTC",
        maturity_offset={"days": 3},
        maturity_time={"hour": 0},
    )
    expected_sc = io.StringIO("""
datetime,   value
2020-01-04, 4.0
2020-01-05, 9.0
2020-01-06, 14.0
2020-01-07, 15.0
""")
    run_block_staircase_value_test(
        client, "basic_sc_daily", hist, expected_sc, sc_kwargs
    )


def test_block_staircase_basic_hourly(client):
    hist = io.StringIO("""
datetime,               2020-01-01 08:00+0, 2020-01-02 08:00+0, 2020-01-03 08:00+0
2020-01-03 00:00+00:00, 1.0,                10.0,               100.0
2020-01-03 04:00+00:00, 2.0,                20.0,               200.0
2020-01-03 08:00+00:00, 3.0,                30.0,               300.0
2020-01-03 16:00+00:00, 4.0,                40.0,               400.0
2020-01-04 00:00+00:00, 5.0,                50.0,               500.0
2020-01-04 04:00+00:00, 6.0,                60.0,               600.0
2020-01-04 08:00+00:00, 7.0,                70.0,               700.0
2020-01-04 16:00+00:00, 8.0,                80.0,               800.0
""")
    sc_kwargs = dict(
        revision_freq={"days": 1},
        revision_time={"hour": 10},
        revision_tz="UTC",
        maturity_offset={"days": 1},
        maturity_time={"hour": 4},
    )
    expected_sc = io.StringIO("""
datetime,               value
2020-01-03 00:00+00:00, 1.0
2020-01-03 04:00+00:00, 20.0
2020-01-03 08:00+00:00, 30.0
2020-01-03 16:00+00:00, 40.0
2020-01-04 00:00+00:00, 50.0
2020-01-04 04:00+00:00, 600.0
2020-01-04 08:00+00:00, 700.0
2020-01-04 16:00+00:00, 800.0
""")
    run_block_staircase_value_test(
        client, "basic_sc_hourly", hist, expected_sc, sc_kwargs
    )


@pytest.mark.parametrize(["hist_file_name", "sc_file_name"], [
    ("hourly_no_dst_hist.csv", "hourly_no_dst_sc_da_9am.csv"),
    ("hourly_dst_1_hist.csv", "hourly_dst_1_sc_da_9am.csv"),
    ("hourly_dst_2_hist.csv", "hourly_dst_2_sc_da_9am.csv"),
])
def test_block_staircase_hourly_day_ahead(client, hist_file_name, sc_file_name):
    """Day-ahead staircase with 9am revision, daily frequency and value hours 0-23"""
    sc_kwargs = dict(
        revision_freq={"days": 1},
        revision_time={"hour": 9},
        revision_tz="Europe/Brussels",
        maturity_offset={"days": 1},
        maturity_time={"hour": 0},
    )
    hist_csv = DATADIR / "staircase" / hist_file_name
    sc_csv = DATADIR / "staircase" / sc_file_name
    run_block_staircase_value_test(
        client, sc_file_name, hist_csv, sc_csv, sc_kwargs, value_date_lag="36h"
    )


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
    # cleanup the db for the catalog assertion below to hold
    cat = client.catalog()
    for source in cat:
        for name, _ in cat[source]:
            client.delete(name)

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


