from datetime import timedelta, datetime as dt
import io
import pytest

import pandas as pd
import numpy as np

from tshistory.api import timeseries
from tshistory import tsio
from tshistory.testutil import (
    assert_df,
    assert_hist,
    gengroup,
    genserie,
    hist_from_csv,
    ts_from_csv,
    utcdt
)
from tshistory.util import (
    replicate_series,
    replicate_basket,
    threadpool,
)


def test_guard_insert(tsx):
    ts = pd.Series(
        [1, 2, 3],
        index=pd.date_range(
            utcdt(2020, 1, 1), periods=3, freq='D'
        )
    )
    with pytest.raises(AssertionError):
        # exception varies depending on nature of tsx
        tsx.update(
            ts,
            'nope',
            'Babar'
        )
    with pytest.raises(AssertionError):
        # exception varies depending on nature of tsx
        tsx.replace(
            ts,
            'nope',
            'Babar'
        )


def test_sources(tsx):
    assert tsx.sources() == ['remote']


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
        'value_dtype': '<f8',
        'left': '2019-12-31T00:00:00',
        'right': '2020-01-04T00:00:00',
    }
    meta = tsx.metadata('api-test')
    assert meta == {}

    out = tsx.update_metadata('api-test', {
        'desc': 'a metadata test'
    })
    assert out is None

    meta = tsx.metadata('api-test')
    assert meta == {
        'desc': 'a metadata test'
    }

    tsx.update_metadata('api-test', {
        'asc': 'a new entry'
    })
    meta = tsx.metadata('api-test')
    assert meta == {
        'asc': 'a new entry',
        'desc': 'a metadata test'
    }

    tsx.replace_metadata('api-test', {
        'replace all': 'a metadata test'
    })
    meta = tsx.metadata('api-test')
    assert meta == {
        'replace all': 'a metadata test'
    }

    assert tsx.type('api-test') == 'primary'

    st = tsx.staircase('api-test', delta=timedelta(days=366))
    assert_df("""
2020-01-02 00:00:00+00:00    2.0
2020-01-03 00:00:00+00:00    3.0
2020-01-04 00:00:00+00:00    4.0
""", st)

    tsx.rename('nope', 'nada')

    tsx.update('api-test2', series, 'Babar')

    with pytest.raises(ValueError):
        tsx.rename('api-test2', 'api-test')


def test_get_with_inferred_freq(tsx):
    ts = pd.Series(
        [0, 1, np.nan, np.nan, 6, 7, 8, 9],
        index=pd.date_range(
            utcdt(2024, 1, 1), periods=8, freq='h'
        )
    )

    tsx.update(
        'with-inferred-freq',
        ts,
        'Babar',
        insertion_date=utcdt(2024, 1, 1)
    )

    ts = tsx.get('with-inferred-freq', inferred_freq=True)
    assert_df("""
2024-01-01 00:00:00+00:00    0.0
2024-01-01 01:00:00+00:00    1.0
2024-01-01 02:00:00+00:00    NaN
2024-01-01 03:00:00+00:00    NaN
2024-01-01 04:00:00+00:00    6.0
2024-01-01 05:00:00+00:00    7.0
2024-01-01 06:00:00+00:00    8.0
2024-01-01 07:00:00+00:00    9.0
""", ts)

    ts = tsx.get(
        'with-inferred-freq',
        inferred_freq=True,
        from_value_date=pd.Timestamp('2024-1-1T02:00:00', tz='UTC')
    )
    assert_df("""
2024-01-01 02:00:00+00:00    NaN
2024-01-01 03:00:00+00:00    NaN
2024-01-01 04:00:00+00:00    6.0
2024-01-01 05:00:00+00:00    7.0
2024-01-01 06:00:00+00:00    8.0
2024-01-01 07:00:00+00:00    9.0
""", ts)


def test_inferred_freq_irregular(tsx):
    index_0 = pd.date_range(
        utcdt(2024, 1, 1), periods=3, freq='D'
    )
    # same freq, with an offset of 6h
    index_1 = pd.date_range(
        utcdt(2024, 1, 4, 6), periods=3, freq='D'
    )
    ts = pd.Series(
        [0, 1, 2, 3, 4, 5],
        index = index_0.append(index_1)
    )
    tsx.update('irregular_freq', ts, 'test')

    ts = tsx.get('irregular_freq')
    assert_df("""
2024-01-01 00:00:00+00:00    0.0
2024-01-02 00:00:00+00:00    1.0
2024-01-03 00:00:00+00:00    2.0
2024-01-04 06:00:00+00:00    3.0
2024-01-05 06:00:00+00:00    4.0
2024-01-06 06:00:00+00:00    5.0
""", ts)

    # when called with the option 'inferred_freq'
    # new index are created (good)
    # and the old ones are also kept (also good)
    ts = tsx.get('irregular_freq', inferred_freq=True)
    assert_df("""
2024-01-01 00:00:00+00:00    0.0
2024-01-02 00:00:00+00:00    1.0
2024-01-03 00:00:00+00:00    2.0
2024-01-04 00:00:00+00:00    NaN
2024-01-04 06:00:00+00:00    3.0
2024-01-05 00:00:00+00:00    NaN
2024-01-05 06:00:00+00:00    4.0
2024-01-06 00:00:00+00:00    NaN
2024-01-06 06:00:00+00:00    5.0
""", ts)


def test_with_inferred_freq_remote(tsx, engine):
    ts = pd.Series(
        [1, 2, 3, np.nan, 5],
        pd.date_range(utcdt(2023, 1, 1), freq='D', periods=5)
    )

    remoteapi = timeseries(
        str(engine.url), 'remote', handler=tsio.timeseries, sources={}
    )
    remoteapi.update(
        'remote.inferred-freq',
        ts,
        'Celeste'
    )

    ts = tsx.get('remote.inferred-freq', inferred_freq=True)
    assert_df("""
2023-01-01 00:00:00+00:00    1.0
2023-01-02 00:00:00+00:00    2.0
2023-01-03 00:00:00+00:00    3.0
2023-01-04 00:00:00+00:00    NaN
2023-01-05 00:00:00+00:00    5.0
""", ts)

    remoteapi.delete('remote.inferred-freq')


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


def test_block_staircase_remote(tsx, engine):
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

    remoteapi = timeseries(
        str(engine.url), 'remote', handler=tsio.timeseries, sources={}
    )
    for idate, ts in hist.items():
        remoteapi.update('remote_staircase', ts, author='test', insertion_date=idate)

    computed_ts = tsx.block_staircase(
        'remote_staircase',
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

    remoteapi.delete('remote_staircase')


def test_inferred_freq(tsx):
    tsx.delete('infer_freq')

    ts = pd.Series(
        [1, 2, 3],
        index=pd.date_range(
            pd.Timestamp('2023-1-1'),
            freq='D',
            periods=3
        )
    )
    tsx.update(
        'infer_freq',
        ts,
        'Babar',
        insertion_date=pd.Timestamp('2023-5-1', tz='utc')
    )

    assert tsx.inferred_freq('infer_freq') == (
        pd.Timedelta(days=1),
        1
    )
    assert tsx.inferred_freq(
        'infer_freq',
        from_value_date=pd.Timestamp('2023-1-3')
    ) is None
    assert tsx.inferred_freq('no-such-series') is None

    # frequence change
    ts = pd.Series(
        [1, 2, 3, 4, 5],
        index=pd.date_range(
            pd.Timestamp('2023-1-3T01:00:00'),
            freq='h',
            periods=5
        )
    )
    tsx.update(
        'infer_freq',
        ts,
        'Babar',
        insertion_date=pd.Timestamp('2023-5-2', tz='utc')
    )

    assert tsx.inferred_freq('infer_freq') == (
        pd.Timedelta(hours=1),
        0.7142857142857143
    )
    freq = tsx.inferred_freq(
        'infer_freq',
        revision_date=pd.Timestamp('2023-5-1')
    )
    assert freq == (
        pd.Timedelta(days=1),
        1
    )


def test_erasure(tsx):
    tsx.delete('erasure')

    ts = pd.Series(
        [np.nan, np.nan],
        index=pd.date_range(
            utcdt(2024, 1, 1),
            freq='h',
            periods=2
        )
    )

    tsx.update('erasure', ts, 'Babar')
    assert not tsx.insertion_dates('erasure')

    tsx.update('erasure', ts, 'Babar', keepnans=True)
    assert len(tsx.insertion_dates('erasure')) == 1

    ival = tsx.interval('erasure')
    assert ival is None

    empty = tsx.get('erasure', keepnans=True)
    assert_df("""
2024-01-01 00:00:00+00:00   NaN
2024-01-01 01:00:00+00:00   NaN
""", empty)

    meta = tsx.internal_metadata('erasure')
    assert meta['value_type'] == 'float64'


def test_rewrite_on_na_tzaware(tsx):

    # audit tools
    if tsx.uri.startswith('http'):
        return
    tsh = tsx.tsh
    e = tsx.engine

    # creation
    name = 'rewrite-on-na'
    ts = genserie(utcdt(2025, 1, 1), 'd', 3)
    ts.index = ts.index.tz_convert('Europe/Paris')
    tsx.update(name, ts, 'arnaud')

    assert len(tsx.get(name)) == 3

    assert (str(tsh.interval(e, name)) ==
            '[2025-01-01 00:00:00+00:00, 2025-01-03 00:00:00+00:00]')
    assert (str(tsh.interval(e, name, notz=True)) ==
            '[2025-01-01 00:00:00, 2025-01-03 00:00:00]')
    # so far, so good

    # erasure of second point
    ts.iloc[1] = np.nan
    tsx.update(name, ts, 'still-arnaud', keepnans=True)

    assert len(tsx.get(name)) == 2
    assert len(tsx.get(name, keepnans=True)) == 3

    assert str(tsh.interval(e, name)) == '[2025-01-01 00:00:00+00:00, 2025-01-03 00:00:00+00:00]'
    assert str(tsh.interval(e, name, notz=True)) == '[2025-01-01 00:00:00, 2025-01-03 00:00:00]'

    # rewrite
    ts.iloc[1] = 3.14
    tsx.update(name, ts, 'still-arnaud', keepnans=True)


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

    series.iloc[1] = 42
    tsx.update(
        'log-me',
        series,
        'Babar',
        insertion_date=utcdt(2020, 1, 2)
    )
    log = tsx.log('log-me', limit=1)
    assert len(log) == 1


def test_oldmeta(tsx):
    ts = pd.Series(
        [1, 2, 3],
        index=pd.date_range(utcdt(2025, 1, 1), freq='d', periods=3)
    )
    tsx.update(
        'oldmeta',
        ts,
        'Babar'
    )
    assert tsx.metadata('oldmeta') == {}

    tsx.replace_metadata(
        'oldmeta',
        {
            'foo': 'bar',
            'quux': 42
        }
    )
    # noop
    tsx.replace_metadata(
        'oldmeta',
        {
            'foo': 'bar',
            'quux': 42
        }
    )
    tsx.replace_metadata(
        'oldmeta',
        {
            'foo': 'baz',
            'quux': 42
        }
    )
    tsx.update_metadata(
        'oldmeta',
        {
            'quux': 43
        }
    )
    assert tsx.metadata('oldmeta') == {'foo': 'baz', 'quux': 43}
    # noop
    tsx.update_metadata(
        'oldmeta',
        {
            'quux': 43,
        }
    )

    old = tsx.old_metadata('oldmeta')
    assert [it[1] for it in old] == [
        {'foo': 'baz', 'quux': 42},
        {'foo': 'bar', 'quux': 42},
        {}
    ]
    assert old[0][2] == 'no-user'


def test_oldmeta_remote(engine, tsx):
    tsr = timeseries(str(engine.url), 'remote', sources={})

    ts = pd.Series(
        [1, 2, 3],
        index=pd.date_range(utcdt(2025, 1, 1), freq='d', periods=3)
    )
    tsr.update(
        'oldmeta',
        ts,
        'Babar'
    )
    assert tsx.metadata('oldmeta') == {}

    tsr.replace_metadata(
        'oldmeta',
        {
            'foo': 'bar',
            'quux': 42
        }
    )
    # noop
    tsr.replace_metadata(
        'oldmeta',
        {
            'foo': 'bar',
            'quux': 42
        }
    )
    tsr.replace_metadata(
        'oldmeta',
        {
            'foo': 'baz',
            'quux': 42
        }
    )
    tsr.update_metadata(
        'oldmeta',
        {
            'quux': 43
        }
    )
    assert tsr.metadata('oldmeta') == {'foo': 'baz', 'quux': 43}
    # noop
    tsr.update_metadata(
        'oldemeta',
        {
            'quux': 43,
            'foo': 'bar'
        }
    )

    old = tsx.old_metadata('oldmeta')
    assert [it[1] for it in old] == [
        {'foo': 'baz', 'quux': 42},
        {'foo': 'bar', 'quux': 42},
        {}
    ]


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


def test_conflicting_update(tsx, engine):
    # behaviour when a series exists locally and remotely
    tsx.update(
        'here-and-there',
        pd.Series(
            [1, 2, 3],
            index=pd.date_range(utcdt(2020, 1, 1), periods=3, freq='D')
        ),
        'Babar'
    )
    # create a series with the same name in the other source
    remoteapi = timeseries(
        str(engine.url), 'remote', handler=tsio.timeseries, sources={}
    )
    remoteapi.update(
        'here-and-there',
        pd.Series(
            [1, 2, 3],
            index=pd.date_range(utcdt(2020, 1, 1), periods=3, freq='D')
        ),
        'Babar'
    )

    tsx.update(
        'here-and-there',
        pd.Series(
            [1, 2, 3, 4],
            index=pd.date_range(utcdt(2020, 1, 1), periods=4, freq='D')
        ),
        'Babar'
    )

    tsx.replace(
        'here-and-there',
        pd.Series(
            [1, 2, 3, 4],
            index=pd.date_range(utcdt(2020, 1, 1), periods=4, freq='D')
        ),
        'Babar'
    )

    # all allowed :)


def test_find(tsx):
    ts = pd.Series(
        [1, 2, 3],
        pd.date_range(utcdt(2023, 1, 1), freq='D', periods=3)
    )
    tsx.update(
        'Find.me.1',
        ts,
        'Babar'
    )
    tsx.update(
        'find.me.2',
        ts,
        'Celeste'
    )

    assert tsx.source('Find.me.1') == 'local'

    # by name
    r = tsx.find('(by.name "nop")')
    assert r == []

    r = tsx.find('(by.name "find.me.1")')
    assert r == ['Find.me.1']

    assert r[0].kind == 'primary'

    r = tsx.find('(by.name ".me.")')
    assert len(r) == 2

    r = tsx.find('(by.name "find 1")')
    assert r == ['Find.me.1']

    tsx.replace_metadata(
        'Find.me.1',
        {
            'foo': 42
        }
    )
    tsx.replace_metadata(
        'find.me.2',
        {
            'bar': 'Hello',
            'foo': 43
        }
    )

    # by metadata key
    r = tsx.find('(by.metakey "foo")')
    assert r == ['Find.me.1', 'find.me.2']

    r = tsx.find('(by.metakey "nope")')
    assert r == []

    r = tsx.find('(by.metakey "bar")')
    assert r == ['find.me.2']

    # by metadata items

    r = tsx.find('(by.metaitem "foo" 43)')
    assert r == ['find.me.2']

    r = tsx.find('(by.metaitem "foo" 42)')
    assert r == ['Find.me.1']

    r = tsx.find('(by.metaitem "bar" "Hello")')
    assert r == ['find.me.2']

    # tzaware
    ts = pd.Series(
        [1, 2, 3],
        pd.date_range(dt(2023, 1, 1), freq='D', periods=3)
    )
    tsx.update(
        'find.me.tznaive',
        ts,
        'Babar'
    )
    tsx.replace_metadata(
        'find.me.tznaive',
        {
            'foo': 43
        }
    )

    r = tsx.find('(by.tzaware)')
    assert 'Find.me.1' in r and 'find.me.2' in r

    # and combination
    r = tsx.find(
        '(by.and '
        '  (by.metaitem "foo" 43) '
        '  (by.metaitem "bar" "Hello"))'
    )
    assert r == ['find.me.2']

    # negation
    r = tsx.find(
        '(by.not (by.tzaware))'
    )
    assert 'find.me.tznaive' in r and 'Find.me.1' not in r and 'find.me.2' not in r

    r = tsx.find(
        '(by.and '
        '  (by.metaitem "foo" 43)'
        '  (by.not (by.tzaware)))'
    )
    assert r == ['find.me.tznaive']

    r = tsx.find(
        '(by.and '
        '  (by.not (by.metaitem "foo" 43))'
        '  (by.tzaware))'
    )
    assert r == ['Find.me.1']

    # or

    r = tsx.find(
        '(by.or '
        '  (= "foo" 43)'
        '  (= "foo" 42))'
    )
    assert r == ['Find.me.1', 'find.me.2', 'find.me.tznaive']

    r = tsx.find(
        '(by.or '
        '  (by.metaitem "foo" 43)'
        '  (by.metaitem "foo" 42))'
    )
    assert r == ['Find.me.1', 'find.me.2', 'find.me.tznaive']

    r = tsx.find(
        '(by.and '
        '  (by.or '
        '     (by.metakey "bar")'
        '     (by.metaitem "foo" 42))'
        '  (by.tzaware))'
    )
    assert r == ['Find.me.1', 'find.me.2']

    ts = r[0]
    assert ts == 'Find.me.1'
    assert ts.imeta is None
    assert ts.meta is None
    assert ts.source == 'local'
    assert ts.kind == 'primary'

    r = tsx.find('(by.everything)', limit=1)
    assert len(r) == 1

    r = tsx.find('(by.metaitem "bar" "Hello")', meta=True)
    assert r == ['find.me.2']

    ts = r[0]
    assert ts == 'find.me.2'
    assert ts.imeta == {
        'tzaware': True,
        'tablename': 'find.me.2',
        'index_type': 'datetime64[ns, UTC]',
        'value_type': 'float64',
        'index_dtype':
        '|M8[ns]', 'value_dtype': '<f8',
        'left': '2023-01-01T00:00:00',
        'right': '2023-01-03T00:00:00'
    }
    assert ts.meta == {
        'bar': 'Hello',
        'foo': 43
    }
    assert ts.source == 'local'

    r = tsx.find('(by.internal-metaitem "tablename" "Find.me.1")')
    assert r == ['Find.me.1']


def test_find_noparam(tsx):
    for name in tsx.find():
        tsx.delete(name)

    ts = pd.Series(
        [1, 2, 3],
        pd.date_range(utcdt(2023, 1, 1), freq='d', periods=3)
    )
    tsx.update(
        'find.me-all',
        ts,
        'Babar'
    )

    assert len(tsx.find()) == 1


def test_find_two_metaitems(tsx):
    ts = pd.Series(
        [1, 2, 3],
        pd.date_range(utcdt(2023, 1, 1), freq='d', periods=3)
    )
    tsx.update(
        'formetaitem.1',
        ts,
        'Babar'
    )
    tsx.update(
        'formetaitem.2',
        ts,
        'Celeste'
    )
    tsx.replace_metadata(
        'formetaitem.1',
        {
            'foo': 42,
            'outages': 'entsoe',
            'country': 'FR'
        }
    )
    tsx.replace_metadata(
        'formetaitem.2',
        {
            'bar': 'Hello',
            'outages': 'rte',
            'country': 'FR',
            'foo': 43
        }
    )

    r = tsx.find(
        '(by.and '
        '  (by.metaitem "outages" "entsoe") '
        '  (by.metaitem "country" "FR"))'
    )
    assert r == ['formetaitem.1']


def test_find_and_byname(tsx):
    ts = pd.Series(
        [1, 2, 3],
        pd.date_range(pd.Timestamp('2024-1-1'), freq='D', periods=3)
    )
    tsx.update(
        'beginning.end',
        ts,
        'Babar'
    )

    r = tsx.find(
        '(by.and (by.name "begin") (by.name "end"))'
    )
    assert r == ['beginning.end']


def test_basket(tsx):
    ts = pd.Series(
        [1, 2, 3],
        pd.date_range(utcdt(2023, 1, 1), freq='D', periods=3)
    )
    tsx.update(
        'basket.1',
        ts,
        'Babar'
    )
    tsx.update(
        'basket.2',
        ts,
        'Celeste'
    )

    tsx.register_basket(
        'b1',
        '(by.name "t.1")'
    )
    assert tsx.list_baskets() == ['b1']
    # make sure upsert works
    tsx.register_basket(
        'b1',
        '(by.name "t.1")'
    )
    assert tsx.list_baskets() == ['b1']

    tsx.register_basket(
        'b2',
        '(by.name "basket.")'
    )
    assert tsx.list_baskets() == ['b1', 'b2']

    assert tsx.basket('b1') == ['basket.1']
    assert tsx.basket_definition('b1') == '(by.name "t.1")'
    assert tsx.basket('b2') == ['basket.1', 'basket.2']

    assert tsx.basket('b2', limit=1) == ['basket.1']

    tsx.delete_basket('b1')
    assert tsx.list_baskets() == ['b2']


def test_no_basket(tsx):
    assert tsx.basket('<nope>') == []


def test_group_basket(tsx):
    # test group support in basket operations through API layer

    # create some test groups
    group1 = gengroup(
        n_scenarios=2,
        from_date=dt(2021, 1, 1),
        length=3,
        freq='d',
        seed=1
    )
    group1.columns = ['test_group_basket_a', 'test_group_basket_b']

    group2 = gengroup(
        n_scenarios=2,
        from_date=dt(2021, 1, 1),
        length=3,
        freq='d',
        seed=2
    )
    group2.columns = ['test_group_basket_c', 'test_group_basket_d']

    tsx.group_replace(
        'test_group_basket_group1',
        group1,
        'test_author'
    )
    tsx.group_replace(
        'test_group_basket_group2',
        group2,
        'test_author'
    )

    # register group baskets
    tsx.register_basket(
        'test_group_basket_1',
        '(by.name "test_group_basket_group")',
        group=True
    )
    tsx.register_basket(
        'test_group_basket_2',
        '(by.name "test_group_basket_group1")',
        group=True
    )

    # test basket listing
    group_baskets = tsx.list_baskets(group=True)
    assert 'test_group_basket_1' in group_baskets
    assert 'test_group_basket_2' in group_baskets

    # test basket content retrieval
    basket_1_content = tsx.basket('test_group_basket_1', group=True)
    assert 'test_group_basket_group1' in basket_1_content
    assert 'test_group_basket_group2' in basket_1_content

    basket_2_content = tsx.basket('test_group_basket_2', group=True)
    assert basket_2_content == ['test_group_basket_group1']

    # test basket definition retrieval
    assert tsx.basket_definition(
        'test_group_basket_1',
        group=True
    ) == '(by.name "test_group_basket_group")'
    assert tsx.basket_definition(
        'test_group_basket_2',
        group=True
    ) == '(by.name "test_group_basket_group1")'

    # test basket deletion
    tsx.delete_basket('test_group_basket_1', group=True)
    remaining_group_baskets = tsx.list_baskets(group=True)
    assert 'test_group_basket_1' not in remaining_group_baskets
    assert 'test_group_basket_2' in remaining_group_baskets

    # cleanup
    tsx.delete_basket('test_group_basket_2', group=True)
    tsx.group_delete('test_group_basket_group1')
    tsx.group_delete('test_group_basket_group2')


def test_federated_basket(tsx, engine):
    ts = pd.Series(
        [1, 2, 3],
        pd.date_range(utcdt(2023, 1, 1), freq='D', periods=3)
    )
    tsx.update(
        'local.basket.fed',
        ts,
        'Babar'
    )

    remoteapi = timeseries(
        str(engine.url), 'remote', handler=tsio.timeseries, sources={}
    )
    remoteapi.update(
        'remote.basket.fed',
        ts,
        'Celeste'
    )

    tsx.register_basket(
        'federated.basket',
        '(by.name "basket.fed")'
    )

    b = tsx.basket('federated.basket')
    assert b == [
        'local.basket.fed',
        'remote.basket.fed'
    ]

    r = b[1]
    assert r.source == 'remote'

    b = tsx.basket('federated.basket', sources=['local'])
    assert b == [
        'local.basket.fed',
    ]
    assert b[0].meta is None

    b = tsx.basket('federated.basket', meta=True)
    assert b[0].meta == {}

    # {'local': {'primary_groups': 0, 'primary_series': 1},
    #  'remote': {'primary_groups': 0, 'primary_series': 1}}
    infos = tsx.info()
    assert 'local' in infos
    assert 'remote' in infos


def test_federated_find(tsx, engine):
    # cleanup
    cat = tsx.catalog()
    if cat:
        for name, _ in list(cat.values())[0]:
            tsx.delete(name)

    ts = pd.Series(
        [1, 2, 3],
        pd.date_range(utcdt(2023, 1, 1), freq='D', periods=3)
    )
    tsx.update(
        'local.basket.fed',
        ts,
        'Babar'
    )

    remoteapi = timeseries(
        str(engine.url), 'remote', handler=tsio.timeseries, sources={}
    )
    # cleanup
    cat = remoteapi.catalog()
    if cat:
        for name, _ in list(cat.values())[0]:
            remoteapi.delete(name)

    remoteapi.update(
        'remote.basket.fed',
        ts,
        'Celeste'
    )

    names = tsx.find('(by.name "basket.fed")')
    assert names == [
        'local.basket.fed',
        'remote.basket.fed'
    ]

    # some top-level bysource
    names = tsx.find('(by.everything)', sources=['remote'])
    assert names == ['remote.basket.fed']
    assert names[0].source == 'remote'

    names = tsx.find('(by.everything)', sources=['local'])
    assert names == ['local.basket.fed']
    assert names[0].source == 'local'

    names = tsx.find('(by.everything)', sources=['local', 'remote'])
    assert names == ['local.basket.fed', 'remote.basket.fed']
    assert names[0].source == 'local'
    assert names[1].source == 'remote'


def test_federated_find_homonyms(tsx, engine):
    # cleanup
    cat = tsx.catalog()
    if cat:
        for name, _ in list(cat.values())[0]:
            tsx.delete(name)

    ts = pd.Series(
        [1, 2, 3],
        pd.date_range(utcdt(2023, 1, 1), freq='D', periods=3)
    )
    tsx.update(
        'find.homonym.2sources',
        ts,
        'Babar'
    )

    remoteapi = timeseries(
        str(engine.url), 'remote', handler=tsio.timeseries, sources={}
    )
    # cleanup
    cat = remoteapi.catalog()
    if cat:
        for name, _ in list(cat.values())[0]:
            remoteapi.delete(name)

    remoteapi.update(
        'find.homonym.2sources',
        ts,
        'Babar'
    )

    names = tsx.find('(by.name "homonym")')
    assert names[0].source == 'local'
    assert names[1].source == 'remote'


def test_replicate_series(tsx):
    insertion_dates = pd.date_range(
        start=pd.Timestamp('2023-07-01', tz='utc'),
        end=pd.Timestamp('2023-07-03', tz='utc'),
        freq='D'
    )
    for idate in insertion_dates:
        ts = pd.Series(
            [1, 2],
            index = pd.date_range(start=idate.date(), periods=2, freq='h')
        )
        tsx.update(
            'original.series.from.tsx',
            ts,
            'sensei',
            insertion_date=idate
        )

    tsx.update_metadata('original.series.from.tsx', {'metadata1': 'value1'})

    replicate_series(
        tsx,
        tsx,
        'original.series.from.tsx',
        'replicated.series.from.tsx'
    )

    hist = tsx.history(
        'replicated.series.from.tsx'
    )
    assert_hist("""
insertion_date             value_date         
2023-07-01 00:00:00+00:00  2023-07-01 00:00:00    1.0
                           2023-07-01 01:00:00    2.0
2023-07-02 00:00:00+00:00  2023-07-01 00:00:00    1.0
                           2023-07-01 01:00:00    2.0
                           2023-07-02 00:00:00    1.0
                           2023-07-02 01:00:00    2.0
2023-07-03 00:00:00+00:00  2023-07-01 00:00:00    1.0
                           2023-07-01 01:00:00    2.0
                           2023-07-02 00:00:00    1.0
                           2023-07-02 01:00:00    2.0
                           2023-07-03 00:00:00    1.0
                           2023-07-03 01:00:00    2.0
""", hist)

    metadata = tsx.metadata('replicated.series.from.tsx')
    assert metadata == {'metadata1': 'value1'}

    insertion_dates = pd.date_range(
        start=pd.Timestamp('2023-07-01', tz='utc'),
        end=pd.Timestamp('2023-07-03', tz='utc'),
        freq='h'
    )
    for idate in insertion_dates:
        ts = pd.Series(
            [1, 2],
            index = pd.date_range(start=idate.date(), periods=2, freq='h')
        )
        tsx.update(
            'original.hourly.series.from.tsx',
            ts,
            'sensei',
            insertion_date=idate
        )

    replicate_series(
        tsx,
        tsx,
        'original.hourly.series.from.tsx',
        'replicated.hourly.series.from.tsx',
        insertion_freq_offset='D'
    )

    hist = tsx.history(
        'replicated.hourly.series.from.tsx'
    )
    assert_hist("""
insertion_date             value_date         
2023-07-01 00:00:00+00:00  2023-07-01 00:00:00    1.0
                           2023-07-01 01:00:00    2.0
2023-07-02 00:00:00+00:00  2023-07-01 00:00:00    1.0
                           2023-07-01 01:00:00    2.0
                           2023-07-02 00:00:00    1.0
                           2023-07-02 01:00:00    2.0
2023-07-03 00:00:00+00:00  2023-07-01 00:00:00    1.0
                           2023-07-01 01:00:00    2.0
                           2023-07-02 00:00:00    1.0
                           2023-07-02 01:00:00    2.0
                           2023-07-03 00:00:00    1.0
                           2023-07-03 01:00:00    2.0
""", hist)

    replicate_series(
        tsx,
        tsx,
        'original.hourly.series.from.tsx',
        'one.version.hourly.series.from.tsx',
        insertion_freq_offset='D',
        from_insertion_date=pd.Timestamp('2023-07-03', tz='utc')
    )

    hist = tsx.history(
        'one.version.hourly.series.from.tsx'
    )
    assert_hist("""
insertion_date             value_date         
2023-07-03 00:00:00+00:00  2023-07-01 00:00:00    1.0
                           2023-07-01 01:00:00    2.0
                           2023-07-02 00:00:00    1.0
                           2023-07-02 01:00:00    2.0
                           2023-07-03 00:00:00    1.0
                           2023-07-03 01:00:00    2.0
""", hist)


def test_replicate_from_basket(tsx):
    ts1 = genserie(utcdt(2025, 1, 1), 'd', 1)
    ts2 = genserie(utcdt(2025, 1, 1), 'd', 2)

    tsx.update('series-replicate-basket-1', ts1, 'test')
    tsx.update('series-replicate-basket-2', ts2, 'test')

    tsx.register_basket(
        'basket-to-push',
        '(by.name "series-replicate-basket")'
    )

    replicate_basket(
        tsx,
        tsx,
        'basket-to-push',
        prefix='replicate.',
        suffix='.suffix',
    )
    assert tsx.exists('replicate.series-replicate-basket-1.suffix')
    assert tsx.exists('replicate.series-replicate-basket-2.suffix')


def test_rename(tsx):
    ts = pd.Series(
        [1, 2, 3],
        index=pd.date_range(
            utcdt(2023, 1, 1),
            freq='d',
            periods=3
        )
    )
    tsx.update('rename-me', ts, 'Babar')
    tsx.rename('rename-me', 'me-renamed')
    assert tsx.exists('me-renamed')
    assert not tsx.exists('rename-me')

    tsx.rename('me-renamed', 'renamed-again', propagate=False)
    assert tsx.exists('renamed-again')
    assert not tsx.exists('me-renamed')


def test_str_series(tsx):
    ts = pd.Series(
        ['a', 'b', 'c'],
        index=pd.date_range(
            utcdt(2023, 1, 1),
            freq='D',
            periods=3
        )
    )

    tsx.update(
        'crashme-str',
        ts,
        'Babar'
    )
    # the bug in numpy_deserialize is gone
    tsx.update(
        'crashme-str',
        ts,
        'Babar'
    )


def test_insertion_dates_tzaware(tsx):
    for name in ('historical-series-tzaware',):
        tsx.delete(name)

    for i in range(10):
        ts = pd.Series(
            np.array([1, 2, 3]) + i*2,
            pd.date_range(utcdt(2024, 4, 1+i), freq='D', periods=3)
        )
        tsx.update(
            'historical-series-tzaware',
            ts,
            'Babar',
            insertion_date=utcdt(2024, 4, 1+i)
        )

    revs = tsx.insertion_dates('historical-series-tzaware')
    assert revs == [
        pd.Timestamp('2024-04-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-03 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-04 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-05 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-06 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-07 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-08 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-09 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-10 00:00:00+0000', tz='UTC')
    ]

    revs = tsx.insertion_dates(
        'historical-series-tzaware',
        from_value_date=pd.Timestamp("2024-04-04"),
        to_value_date=pd.Timestamp("2024-04-05")
    )
    assert revs == [
        pd.Timestamp('2024-04-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-03 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-04 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-05 00:00:00+0000', tz='UTC'),
    ]

    revs = tsx.insertion_dates(
        'historical-series-tzaware',
        from_value_date=pd.Timestamp("2024-04-04"),
        to_value_date=pd.Timestamp("2024-04-05"),
        limit=2
    )
    assert revs == [
        pd.Timestamp('2024-04-04 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-05 00:00:00+0000', tz='UTC'),
    ]

    revs = tsx.history(
        'historical-series-tzaware',
        from_value_date=pd.Timestamp("2024-04-04"),
        to_value_date=pd.Timestamp("2024-04-05")
    ).keys()
    assert list(revs) == [
        pd.Timestamp('2024-04-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-03 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-04 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-05 00:00:00+0000', tz='UTC')
    ]


def test_insertion_dates_tznaive(tsx):
    for name in ('historical-series-naive',):
        tsx.delete(name)

    for i in range(10):
        ts = pd.Series(
            np.array([1, 2, 3]) + i*2,
            pd.date_range(pd.Timestamp(f'2024-4-{1+i}'), freq='D', periods=3)
        )
        tsx.update(
            'historical-series-naive',
            ts,
            'Babar',
            insertion_date=utcdt(2024, 4, 1+i)
        )

    revs = tsx.insertion_dates('historical-series-naive')
    assert revs == [
        pd.Timestamp('2024-04-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-03 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-04 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-05 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-06 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-07 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-08 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-09 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-10 00:00:00+0000', tz='UTC')
    ]

    revs = tsx.insertion_dates(
        'historical-series-naive',
        from_value_date=pd.Timestamp("2024-04-04"),
        to_value_date=pd.Timestamp("2024-04-05")
    )
    assert revs == [
        pd.Timestamp('2024-04-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-03 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-04 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-05 00:00:00+0000', tz='UTC'),
    ]

    revs = tsx.history(
        'historical-series-naive',
        from_value_date=pd.Timestamp("2024-04-04"),
        to_value_date=pd.Timestamp("2024-04-05")
    ).keys()
    assert list(revs) == [
        pd.Timestamp('2024-04-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-03 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-04 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-05 00:00:00+0000', tz='UTC')
    ]


# tree stuff

def test_tree_api(tsx, engine):
    tsx.set_tree_attribute(None)
    assert tsx.tree_attribute() is None
    tsx.set_tree_attribute('tree')
    assert tsx.tree_attribute() == 'tree'

    ts = pd.Series(
        [1, 2, 3],
        index=pd.date_range(utcdt(2020, 1, 1), freq='d', periods=3)
    )

    for name in (
            'UE.Italy',
            'UE.France'
    ):
        sname = name.lower()
        tsx.update(
            sname,
            ts,
            'Babar'
        )
        tsx.update_metadata(sname, {'tree': name})

    assert tsx.path_series('UE.France') == ['ue.france']
    assert tsx.path_series('UE.Italy') == ['ue.italy']
    assert tsx.path_series('UE') == []

    assert tsx.series_path('ue.france') == 'UE.France'
    assert tsx.series_path('ue.italy') == 'UE.Italy'

    assert tsx.tree() == ['UE.Italy', 'UE.France']

    assert tsx.find('(by.without-path)') == []
    assert tsx.find('(by.at-path "UE")') == ['ue.france', 'ue.italy']
    assert tsx.find('(by.at-path "UE" #:childen #f)') == []
    assert tsx.find('(by.at-path "UE.France")') == ['ue.france']

    tsx.delete_path('UE.Italy')
    assert tsx.tree() == ['UE.France']

    assert tsx.path_series('UE.France') == ['ue.france']
    assert tsx.path_series('UE.Italy') == []

    assert tsx.series_path('ue.france') == 'UE.France'
    assert tsx.series_path('ue.italy') is None

    assert tsx.find('(by.without-path)') == ['ue.italy']
    assert tsx.find('(by.at-path "UE" #:children #t)') == ['ue.france']

    tsx.update_metadata('ue.france', {'tree': "a.name"})
    assert tsx.series_path('ue.france') == 'a.name'
    assert tsx.path_series('a.name') == ['ue.france']
    assert tsx.tree() == ['UE.France', 'a.name']

    tsx.rename_path('a.name', 'UE.RepubliqueFrancaise')
    assert tsx.tree() == ['UE.France', 'UE.RepubliqueFrancaise']
    assert tsx.path_series('a.name') == []
    assert tsx.path_series('UE.RepubliqueFrancaise') == ['ue.france']


def test_tree_parallel(tsx, tsh, engine):
    tsx.set_tree_attribute('tree')
    ts = genserie(dt(2025, 1, 1), 'd', 10)
    path = 'my.new.folder'
    names = [
        ( 'series.a.tree', path ),
        ( 'series.b.tree', path ),
        ( 'series.c.tree', path ),
        ( 'series.d.tree', path ),
        ( 'series.e.tree', path ),
        ( 'series.f.tree', path ),
    ]
    for sn in names:
        tsx.update(sn[0], ts, 'test')
    errors = []

    def put_in_folder(sn, path):
        try:
            tsx.update_metadata(sn, {'tree': path})
        except Exception as e:
            errors.append(e)

    pool = threadpool(7)
    pool(put_in_folder, names)
    assert tsx.metadata('series.a.tree') == {'tree': 'my.new.folder'}
    assert not len(errors)


def test_tree_roundtrip(tsx):
    # setup tree-attribute
    tsx.set_tree_attribute('folders')

    # Create empty node
    # Nothing yet

    # insert series and put in tree
    ts = pd.Series(
        [1, 2, 3],
        index=pd.date_range(utcdt(2020, 1, 1), freq='d', periods=3)
    )
    name = 'series-folder-0'
    tsx.update( name, ts, 'test')
    tsx.update_metadata(name, {'folders': 'a'})

    name = 'series-folder-1'
    tsx.update( name, ts, 'test')
    tsx.update_metadata(name, {'folders': 'a.b'})

    name = 'series-folder-2'
    tsx.update( name, ts, 'test')
    tsx.update_metadata(name, {'folders': 'a.b.c'})

    assert tsx.tree() == ['a', 'a.b', 'a.b.c']

    # deletion
    # terminal node

    tsx.delete_path('a.b.c')
    assert tsx.tree() == ['a', 'a.b']
    assert tsx.metadata('series-folder-2') == {'folders': 'a.b.c'}
    # i.e. incoherent state

    # restore previous state
    tsx.update_metadata('series-folder-2', {'folders': 'a.b'})
    tsx.update_metadata('series-folder-2', {'folders': 'a.b.c'})
    assert tsx.tree() == ['a', 'a.b', 'a.b.c']

    # intermediary node
    tsx.delete_path('a.b')
    assert tsx.tree() == ['a', 'a.b.c']
    assert tsx.metadata('series-folder-1') == {'folders': 'a.b'}
    assert tsx.metadata('series-folder-2') == {'folders': 'a.b.c'}

    # restore previous state
    tsx.update_metadata('series-folder-1', {'folders': 'a'})
    tsx.update_metadata('series-folder-1', {'folders': 'a.b'})
    assert tsx.tree() == ['a', 'a.b.c', 'a.b']
    # NB: the path are given in another order

    # rename
    # terminal node
    tsx.rename_path('a.b.c', 'a.b.x')
    assert tsx.tree() == ['a', 'a.b', 'a.b.x']
    assert tsx.metadata('series-folder-2') == {'folders': 'a.b.c'}
    # incoherent state

    # restore previous state
    tsx.rename_path('a.b.x', 'a.b.c')
    assert tsx.tree() == ['a', 'a.b', 'a.b.c']

    # intermediary node
    tsx.rename_path('a.b.c', 'a.x.c')
    assert tsx.tree() == ['a', 'a.b', 'a.x.c']
    assert tsx.metadata('series-folder-1') == {'folders': 'a.b'}
    assert tsx.metadata('series-folder-2') == {'folders': 'a.b.c'}
    # incoherent state


# groups

def test_remote_group(engine, tsx):
    tsr = timeseries(str(engine.url), 'remote', sources={})
    df = gengroup(
        n_scenarios=3,
        from_date=dt(2021, 1, 1),
        length=5,
        freq='D',
        seed=2
    )
    tsr.group_replace(
        'remote-group', df, 'Babar', insertion_date=pd.Timestamp('2024-1-1', tz='UTC')
    )

    assert tsx.group_exists('remote-group')
    assert tsx.group_type('remote-group') == 'primary'
    assert tsx.group_insertion_dates('remote-group') == [
        pd.Timestamp('2024-01-01 00:00:00+0000', tz='UTC')
    ]
    assert tsx.group_insertion_dates(
        'remote-group',
        from_insertion_date=pd.Timestamp('2023-1-1', tz='UTC'),
        to_insertion_date=pd.Timestamp('2023-1-2', tz='UTC'),
    ) == []

    meta = tsx.group_metadata('remote-group')
    assert meta == {}
    imeta = tsx.group_internal_metadata('remote-group')
    assert imeta == {
        'index_dtype': '<M8[ns]',
        'index_type': 'datetime64[ns]',
        'tzaware': False,
        'value_dtype': '<f8',
        'value_type': 'float64',
        'left': '2021-01-01T00:00:00', 'right': '2021-01-05T00:00:00'
    }

    h = tsx.group_history('remote-group')
    assert_hist("""
                                        0    1    2
insertion_date            value_date               
2024-01-01 00:00:00+00:00 2021-01-01  2.0  3.0  4.0
                          2021-01-02  3.0  4.0  5.0
                          2021-01-03  4.0  5.0  6.0
                          2021-01-04  5.0  6.0  7.0
                          2021-01-05  6.0  7.0  8.0
""", h)

    cat = tsx.group_catalog()
    assert cat == {('postgres@remote', 'remote'): [('remote-group', 'primary')]}

    gr = tsx.group_get('remote-group')
    assert_df("""
              0    1    2
2021-01-01  2.0  3.0  4.0
2021-01-02  3.0  4.0  5.0
2021-01-03  4.0  5.0  6.0
2021-01-04  5.0  6.0  7.0
2021-01-05  6.0  7.0  8.0
""", gr)

    gr = tsx.group_get(
        'remote-group',
        from_value_date=pd.Timestamp('2021-01-02'),
        to_value_date=pd.Timestamp('2021-01-04')
    )
    assert_df("""
              0    1    2
2021-01-02  3.0  4.0  5.0
2021-01-03  4.0  5.0  6.0
2021-01-04  5.0  6.0  7.0
""", gr)

    tsr.group_delete('remote-group')


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

    meta = tsx.group_internal_metadata('first_group_api')
    assert meta == {
        'index_dtype': '<M8[ns]',
        'index_type': 'datetime64[ns]',
        'tzaware': False,
        'value_dtype': '<f8',
        'value_type': 'float64',
        'left': '2021-01-01T00:00:00',
        'right': '2021-01-05T00:00:00'
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


def test_group_update(tsx):
    df = gengroup(
        n_scenarios=3,
        from_date=utcdt(2025, 1, 1),
        length=3,
        freq='h',
        seed=1
    )
    tsx.group_update(
        'group-update',
        df,
        'Babar',
        insertion_date=pd.Timestamp('2025-1-1', tz='utc')
    )

    dfo = tsx.group_get('group-update')
    assert_df("""
                             0    1    2
2025-01-01 00:00:00+00:00  1.0  2.0  3.0
2025-01-01 01:00:00+00:00  2.0  3.0  4.0
2025-01-01 02:00:00+00:00  3.0  4.0  5.0
""", dfo)

    df = df * 2
    df.index = df.index.shift(1, 'h')
    tsx.group_update(
        'group-update',
        df,
        'Babar',
        insertion_date=pd.Timestamp('2025-1-2', tz='utc')
    )

    dfo = tsx.group_get('group-update')
    assert_df("""
                             0    1     2
2025-01-01 00:00:00+00:00  1.0  2.0   3.0
2025-01-01 01:00:00+00:00  2.0  4.0   6.0
2025-01-01 02:00:00+00:00  4.0  6.0   8.0
2025-01-01 03:00:00+00:00  6.0  8.0  10.0
""", dfo)

    assert tsx.group_insertion_dates('group-update') == [
        pd.Timestamp('2025-01-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2025-01-02 00:00:00+0000', tz='UTC')
    ]

    tsx.group_delete('group-update')


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


def test_group_metadata(tsx):
    df = gengroup(
        n_scenarios=4,
        from_date=pd.Timestamp('2025-1-1'),
        length=4,
        freq='d',
        seed=4
    )

    tsx.group_replace(
        'for-metadata',
        df,
        author='Babar',
        insertion_date=pd.Timestamp('2025-01-01', tz='UTC')
    )

    m = tsx.group_metadata('for-metadata')
    assert m == {}

    im = tsx.group_internal_metadata('for-metadata')
    im.pop('path', None); im.pop('tablename', None)
    assert im == {
        'index_dtype': '<M8[ns]',
        'index_type': 'datetime64[ns]',
        'left': '2025-01-01T00:00:00',
        'right': '2025-01-04T00:00:00',
        'tzaware': False,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }

    tsx.replace_group_metadata(
        'for-metadata',
        {'foo': 42}
    )

    m = tsx.group_metadata('for-metadata')
    assert m == {'foo': 42}

    tsx.update_group_metadata(
        'for-metadata',
        {'name': 'Celeste'}
    )

    m = tsx.group_metadata('for-metadata')
    assert m == {
        'foo': 42,
        'name': 'Celeste'
    }

    tsx.replace_group_metadata(
        'for-metadata',
        {'bar': 42}
    )

    m = tsx.group_metadata('for-metadata')
    assert m == {
        'bar': 42,
    }


def test_group_oldmeta(tsx):
    df = gengroup(
        n_scenarios=2,
        from_date=dt(2025, 1, 1),
        length=2,
        freq='d',
        seed=1
    )
    tsx.group_update(
        'group-oldmeta',
        df,
        'Babar'
    )
    assert tsx.group_metadata('group-oldmeta') == {}

    tsx.replace_group_metadata(
        'group-oldmeta',
        {
            'foo': 'bar',
            'quux': 42
        }
    )
    # noop
    tsx.replace_group_metadata(
        'group-oldmeta',
        {
            'foo': 'bar',
            'quux': 42
        }
    )
    tsx.replace_group_metadata(
        'group-oldmeta',
        {
            'foo': 'baz',
            'quux': 42
        }
    )
    tsx.update_group_metadata(
        'group-oldmeta',
        {
            'quux': 43
        }
    )
    assert tsx.group_metadata('group-oldmeta') == {'foo': 'baz', 'quux': 43}
    # noop
    tsx.update_group_metadata(
        'group-oldmeta',
        {
            'quux': 43,
        }
    )

    old = tsx.group_old_metadata('group-oldmeta')
    assert [it[1] for it in old] == [
        {'foo': 'baz', 'quux': 42},
        {'foo': 'bar', 'quux': 42},
        {}
    ]
    assert old[0][2] == 'no-user'


def test_group_oldmeta_remote(engine, tsx):
    tsr = timeseries(str(engine.url), 'remote', sources={})

    df = gengroup(
        n_scenarios=2,
        from_date=dt(2025, 1, 1),
        length=2,
        freq='d',
        seed=1
    )
    tsr.group_update(
        'group-oldmeta',
        df,
        'Babar'
    )
    assert tsx.group_metadata('group-oldmeta') == {}

    tsr.replace_group_metadata(
        'group-oldmeta',
        {
            'foo': 'bar',
            'quux': 42
        }
    )
    # noop
    tsr.replace_group_metadata(
        'group-oldmeta',
        {
            'foo': 'bar',
            'quux': 42
        }
    )
    tsr.replace_group_metadata(
        'group-oldmeta',
        {
            'foo': 'baz',
            'quux': 42
        }
    )
    tsr.update_group_metadata(
        'group-oldmeta',
        {
            'quux': 43
        }
    )
    assert tsx.group_metadata('group-oldmeta') == {'foo': 'baz', 'quux': 43}
    # noop
    tsr.update_group_metadata(
        'gruop-oldmeta',
        {
            'quux': 43,
        }
    )

    old = tsx.group_old_metadata('group-oldmeta')
    assert [it[1] for it in old] == [
        {'foo': 'baz', 'quux': 42},
        {'foo': 'bar', 'quux': 42},
        {}
    ]


def test_group_log(tsx):
    df = gengroup(
        n_scenarios=4,
        from_date=pd.Timestamp('2025-1-1'),
        length=4,
        freq='d',
        seed=4
    )

    tsx.group_replace(
        'for-log',
        df,
        author='Babar',
        insertion_date=pd.Timestamp('2025-01-01', tz='UTC')
    )

    log = tsx.group_log('for-log')
    assert log == [
        {
            'author': 'Babar',
            'date': pd.Timestamp('2025-01-01 00:00:00+0000', tz='UTC'),
            'meta': {},
            'rev': 1
        }
    ]

    df = gengroup(
        n_scenarios=4,
        from_date=pd.Timestamp('2025-1-2'),
        length=4,
        freq='d',
        seed=4
    )
    tsx.group_replace(
        'for-log',
        df,
        author='Celeste',
        insertion_date=pd.Timestamp('2025-01-02', tz='UTC')
    )

    log = tsx.group_log('for-log')
    assert log == [
        {
            'author': 'Babar',
            'date': pd.Timestamp('2025-01-01 00:00:00+0000', tz='UTC'),
            'meta': {},
            'rev': 1
        },
        {
            'author': 'Celeste',
            'date': pd.Timestamp('2025-01-02 00:00:00+0000', tz='UTC'),
            'meta': {},
            'rev': 2
        }
    ]

    log = tsx.group_log('for-log', limit=1)
    assert len(log) == 1

    log = tsx.group_log('for-log', fromdate=pd.Timestamp('2025-1-2', tz='utc'))
    assert log == [
        {
            'author': 'Celeste',
            'date': pd.Timestamp('2025-01-02 00:00:00+0000', tz='UTC'),
            'meta': {},
            'rev': 2
        }
    ]

    log = tsx.group_log('for-log', todate=pd.Timestamp('2025-1-1', tz='utc'))
    assert log == [
        {
            'author': 'Babar',
            'date': pd.Timestamp('2025-01-01 00:00:00+0000', tz='UTC'),
            'meta': {},
            'rev': 1
        },
    ]


def test_group_find(tsx):
    df = gengroup(
        n_scenarios=3,
        from_date=utcdt(2025, 1, 1),
        length=5,
        freq='d',
        seed=2
    )
    tsx.group_replace(
        'gr.find.me.1',
        df,
        'Babar'
    )
    tsx.group_replace(
        'gr.find.me.2',
        df,
        'Celeste'
    )

    assert tsx.group_source('gr.find.me.1') == 'local'

    # by name
    r = tsx.group_find('(by.name "nop")')
    assert r == []

    r = tsx.group_find('(by.name "find.me.1")')
    assert r == ['gr.find.me.1']

    assert r[0].kind == 'primary'

    r = tsx.group_find('(by.name ".me.")')
    assert len(r) == 2

    r = tsx.group_find('(by.name "find 1")')
    assert r == ['gr.find.me.1']

    tsx.replace_group_metadata(
        'gr.find.me.1',
        {
            'foo': 42
        }
    )
    tsx.replace_group_metadata(
        'gr.find.me.2',
        {
            'bar': 'Hello',
            'foo': 43
        }
    )

    # by metadata key
    r = tsx.group_find('(by.metakey "foo")')
    assert r == ['gr.find.me.1', 'gr.find.me.2']

    r = tsx.group_find('(by.metakey "nope")')
    assert r == []

    r = tsx.group_find('(by.metakey "bar")')
    assert r == ['gr.find.me.2']

    # by metadata items

    r = tsx.group_find('(by.metaitem "foo" 43)')
    assert r == ['gr.find.me.2']

    r = tsx.group_find('(by.metaitem "foo" 42)')
    assert r == ['gr.find.me.1']

    r = tsx.group_find('(by.metaitem "bar" "Hello")')
    assert r == ['gr.find.me.2']

    # tzaware
    df = gengroup(
        n_scenarios=3,
        from_date=dt(2025, 1, 1),
        length=5,
        freq='d',
        seed=2
    )
    tsx.group_replace(
        'gr.find.me.tznaive',
        df,
        'Babar'
    )
    tsx.replace_group_metadata(
        'gr.find.me.tznaive',
        {
            'foo': 43
        }
    )

    r = tsx.group_find('(by.tzaware)')
    assert 'gr.find.me.1' in r and 'gr.find.me.2' in r

    # and combination
    r = tsx.group_find(
        '(by.and '
        '  (by.metaitem "foo" 43) '
        '  (by.metaitem "bar" "Hello"))'
    )
    assert r == ['gr.find.me.2']

    # negation
    r = tsx.group_find(
        '(by.not (by.tzaware))'
    )
    assert 'gr.find.me.tznaive' in r and 'gr.find.me.1' not in r and 'gr.find.me.2' not in r

    r = tsx.group_find(
        '(by.and '
        '  (by.metaitem "foo" 43)'
        '  (by.not (by.tzaware)))'
    )
    assert r == ['gr.find.me.tznaive']

    r = tsx.group_find(
        '(by.and '
        '  (by.not (by.metaitem "foo" 43))'
        '  (by.tzaware))'
    )
    assert r == ['gr.find.me.1']

    # or

    r = tsx.group_find(
        '(by.or '
        '  (= "foo" 43)'
        '  (= "foo" 42))'
    )
    assert r == ['gr.find.me.1', 'gr.find.me.2', 'gr.find.me.tznaive']

    r = tsx.group_find(
        '(by.or '
        '  (by.metaitem "foo" 43)'
        '  (by.metaitem "foo" 42))'
    )
    assert r == ['gr.find.me.1', 'gr.find.me.2', 'gr.find.me.tznaive']

    r = tsx.group_find(
        '(by.and '
        '  (by.or '
        '     (by.metakey "bar")'
        '     (by.metaitem "foo" 42))'
        '  (by.tzaware))'
    )
    assert r == ['gr.find.me.1', 'gr.find.me.2']

    gr = r[0]
    assert gr == 'gr.find.me.1'
    assert gr.imeta is None
    assert gr.meta is None
    assert gr.source == 'local'
    assert gr.kind == 'primary'

    r = tsx.group_find('(by.everything)', limit=1)
    assert len(r) == 1

    r = tsx.group_find('(by.metaitem "bar" "Hello")', meta=True)
    assert r == ['gr.find.me.2']

    gr = r[0]
    assert gr == 'gr.find.me.2'
    gr.imeta.pop('tablename', None)
    assert gr.imeta == {
        'index_dtype': '|M8[ns]',
        'index_type': 'datetime64[ns, UTC]',
        'left': '2025-01-01T00:00:00+00:00',
        'right': '2025-01-05T00:00:00+00:00',
        'tzaware': True,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }

    assert gr.meta == {
        'bar': 'Hello',
        'foo': 43
    }
    assert gr.source == 'local'


def test_federated_group_find(tsx, engine):
    # cleanup
    cat = tsx.group_catalog()
    if cat:
        for name, _ in list(cat.values())[0]:
            tsx.group_delete(name)

    df = gengroup(
        n_scenarios=3,
        from_date=utcdt(2025, 1, 1),
        length=5,
        freq='d',
        seed=2
    )
    tsx.group_update(
        'local.basket.fed',
        df,
        'Babar'
    )

    remoteapi = timeseries(
        str(engine.url), 'remote', handler=tsio.timeseries, sources={}
    )
    # cleanup
    cat = remoteapi.group_catalog()
    if cat:
        for name, _ in list(cat.values())[0]:
            remoteapi.group_delete(name)

    remoteapi.group_update(
        'remote.basket.fed',
        df,
        'Celeste'
    )

    names = tsx.group_find('(by.name "basket.fed")')
    assert names == [
        'local.basket.fed',
        'remote.basket.fed'
    ]

    names = tsx.group_find('(by.everything)', sources=['remote'])
    assert names == ['remote.basket.fed']
    assert names[0].source == 'remote'

    names = tsx.group_find('(by.everything)', sources=['local'])
    assert names == ['local.basket.fed']
    assert names[0].source == 'local'

    names = tsx.group_find('(by.everything)', sources=['local', 'remote'])
    assert names == ['local.basket.fed', 'remote.basket.fed']
    assert names[0].source == 'local'
    assert names[1].source == 'remote'
