# coding: utf-8
from pathlib import Path
from datetime import datetime
from dateutil import parser

import pandas as pd
import numpy as np
import pytest

from tshistory.tsio import TimeSerie

DATADIR = Path(__file__).parent / 'data'


def assert_group_equals(g1, g2):
    for (n1, s1), (n2, s2) in zip(sorted(g1.items()),
                                  sorted(g2.items())):
        assert n1 == n2
        assert s1.equals(s2)


def assert_df(expected, df):
    assert expected.strip() == df.to_string().strip()


def genserie(start, freq, repeat, initval=None, tz=None, name=None):
    if initval is None:
        values = range(repeat)
    else:
        values = initval * repeat
    return pd.Series(values,
                     name=name,
                     index=pd.date_range(start=start,
                                         freq=freq,
                                         periods=repeat,
                                         tz=tz))

def test_changeset(engine):
    # instantiate one time serie handler object
    tso = TimeSerie()

    index = pd.date_range(start=datetime(2017, 1, 1), freq='D', periods=3)
    data = [1., 2., 3.]

    with engine.connect() as cnx:
        with tso.newchangeset(cnx, 'babar'):
            tso.insert(cnx, pd.Series(data, index=index), 'ts_values')
            tso.insert(cnx, pd.Series(['a', 'b', 'c'], index=index), 'ts_othervalues')

    g = tso.get_group(engine, 'ts_values')
    g2 = tso.get_group(engine, 'ts_othervalues')
    assert_group_equals(g, g2)

    with pytest.raises(AssertionError):
        tso.insert(engine, pd.Series([2,3,4], index=index), 'ts_values')

    with engine.connect() as cnx:
        data.append(data.pop(0))
        with tso.newchangeset(cnx, 'celeste'):
            tso.insert(cnx, pd.Series(data, index=index), 'ts_values')
            # below should be a noop
            tso.insert(cnx, pd.Series(['a', 'b', 'c'], index=index), 'ts_othervalues')

    g = tso.get_group(engine, 'ts_values')
    assert ['ts_values'] == list(g.keys())

    assert_df("""
2017-01-01    2.0
2017-01-02    3.0
2017-01-03    1.0
""", tso.get(engine, 'ts_values'))

    assert_df("""
2017-01-01    a
2017-01-02    b
2017-01-03    c
""", tso.get(engine, 'ts_othervalues'))


def test_tstamp_roundtrip(engine):
    tso = TimeSerie()
    ts = genserie(datetime(2017, 10, 28, 23),
                  'H', 4, tz='UTC')
    ts.index = ts.index.tz_convert('Europe/Paris')

    assert_df("""
2017-10-29 01:00:00+02:00    0
2017-10-29 02:00:00+02:00    1
2017-10-29 02:00:00+01:00    2
2017-10-29 03:00:00+01:00    3
Freq: H
    """, ts)

    tso.insert(engine, ts, 'tztest', 'Babar')
    back = tso.get(engine, 'tztest')

    # though un localized we understand it's been normalized to utc
    assert_df("""
2017-10-28 23:00:00    0.0
2017-10-29 00:00:00    1.0
2017-10-29 01:00:00    2.0
2017-10-29 02:00:00    3.0
""", back)

    back.index = back.index.tz_localize('UTC')
    assert (ts.index == back.index).all()


def test_differential(engine):
    # instantiate one time serie handler object
    tso = TimeSerie()

    ts_begin = genserie(datetime(2010, 1, 1), 'D', 10)
    tso.insert(engine, ts_begin, 'ts_test', 'test')

    assert_df("""
2010-01-01    0.0
2010-01-02    1.0
2010-01-03    2.0
2010-01-04    3.0
2010-01-05    4.0
2010-01-06    5.0
2010-01-07    6.0
2010-01-08    7.0
2010-01-09    8.0
2010-01-10    9.0
""", tso.get(engine, 'ts_test'))

    # we should detect the emission of a message
    tso.insert(engine, ts_begin, 'ts_test', 'babar')

    assert_df("""
2010-01-01    0.0
2010-01-02    1.0
2010-01-03    2.0
2010-01-04    3.0
2010-01-05    4.0
2010-01-06    5.0
2010-01-07    6.0
2010-01-08    7.0
2010-01-09    8.0
2010-01-10    9.0
""", tso.get(engine, 'ts_test'))

    ts_slight_variation = ts_begin.copy()
    ts_slight_variation.iloc[3] = 0
    ts_slight_variation.iloc[6] = 0
    tso.insert(engine, ts_slight_variation, 'ts_test', 'celeste')

    assert_df("""
2010-01-01    0.0
2010-01-02    1.0
2010-01-03    2.0
2010-01-04    0.0
2010-01-05    4.0
2010-01-06    5.0
2010-01-07    0.0
2010-01-08    7.0
2010-01-09    8.0
2010-01-10    9.0
""", tso.get(engine, 'ts_test'))

    ts_longer = genserie(datetime(2010, 1, 3), 'D', 15)
    ts_longer.iloc[1] = 2.48
    ts_longer.iloc[3] = 3.14
    ts_longer.iloc[5] = ts_begin.iloc[7]

    tso.insert(engine, ts_longer, 'ts_test', 'test')

    assert_df("""
2010-01-01     0.00
2010-01-02     1.00
2010-01-03     0.00
2010-01-04     2.48
2010-01-05     2.00
2010-01-06     3.14
2010-01-07     4.00
2010-01-08     7.00
2010-01-09     6.00
2010-01-10     7.00
2010-01-11     8.00
2010-01-12     9.00
2010-01-13    10.00
2010-01-14    11.00
2010-01-15    12.00
2010-01-16    13.00
2010-01-17    14.00
""", tso.get(engine, 'ts_test'))

    # start testing manual overrides
    ts_begin = genserie(datetime(2010, 1, 1), 'D', 5, initval=[2])
    ts_begin.loc['2010-01-04'] = -1
    tso.insert(engine, ts_begin, 'ts_mixte', 'test')

    # -1 represents bogus upstream data
    assert_df("""
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
""", tso.get(engine, 'ts_mixte'))

    # refresh all the period + 1 extra data point
    ts_more = genserie(datetime(2010, 1, 2), 'D', 5, [2])
    ts_more.loc['2010-01-04'] = -1
    tso.insert(engine, ts_more, 'ts_mixte', 'test')

    assert_df("""
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
2010-01-06    2.0
""", tso.get(engine, 'ts_mixte'))

    # just append an extra data point
    # with no intersection with the previous ts
    ts_one_more = genserie(datetime(2010, 1, 7), 'D', 1, [3])
    tso.insert(engine, ts_one_more, 'ts_mixte', 'test')

    assert_df("""
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
2010-01-06    2.0
2010-01-07    3.0
""", tso.get(engine, 'ts_mixte'))

    hist = pd.read_sql('select id, parent from timeserie.ts_test order by id',
                        engine)
    assert_df("""
   id  parent
0   1     NaN
1   2     1.0
2   3     2.0
""", hist)

    hist = pd.read_sql('select id, parent from timeserie.ts_mixte order by id',
                        engine)
    assert_df("""
   id  parent
0   1     NaN
1   2     1.0
2   3     2.0
""", hist)

    allts = pd.read_sql("select name, table_name from registry "
                        "where name in ('ts_test', 'ts_mixte')",
                        engine)

    assert_df("""
       name          table_name
0   ts_test   timeserie.ts_test
1  ts_mixte  timeserie.ts_mixte
""", allts)

    assert_df("""
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
2010-01-06    2.0
2010-01-07    3.0
""", tso.get(engine, 'ts_mixte',
             revision_date=datetime.now()))


def test_bad_import(engine):
    # instantiate one time serie handler object
    tso = TimeSerie()

    # the data were parsed as date by pd.read_json()
    df_result = pd.read_csv(DATADIR / 'test_data.csv')
    df_result['Gas Day'] = df_result['Gas Day'].apply(parser.parse, dayfirst=True, yearfirst=False)
    df_result.set_index('Gas Day', inplace=True)
    ts = df_result['SC']

    tso.insert(engine, ts, 'SND_SC', 'test')
    result = tso.get(engine, 'SND_SC')
    assert result.dtype == 'float64'

    # insertion of empty ts
    ts = pd.Series(name='truc', dtype='object')
    tso.insert(engine, ts, 'empty_ts', 'test')
    assert tso.get(engine, 'empty_ts') is None

    # nan in ts
    # all na
    ts = genserie(datetime(2010, 1, 10), 'D', 10, [np.nan], name='truc')
    tso.insert(engine, ts, 'test_nan', 'test')
    assert tso.get(engine, 'test_nan') is None

    # mixe na
    ts = pd.Series([np.nan] * 5 + [3] * 5,
                   index=pd.date_range(start=datetime(2010, 1, 10),
                                       freq='D', periods=10), name='truc')
    tso.insert(engine, ts, 'test_nan', 'test')
    result = tso.get(engine, 'test_nan')

    tso.insert(engine, ts, 'test_nan', 'test')
    result = tso.get(engine, 'test_nan')
    assert_df("""
2010-01-15    3.0
2010-01-16    3.0
2010-01-17    3.0
2010-01-18    3.0
2010-01-19    3.0
""", result)

    # get_ts with name not in database

    tso.get(engine, 'inexisting_name', 'test')


def test_revision_date(engine):
    # instantiate one time serie handler object
    tso = TimeSerie()

    idate1 = datetime(2015, 1, 1, 15, 43, 23)
    with tso.newchangeset(engine, 'test', _insertion_date=idate1):

        ts = genserie(datetime(2010, 1, 4), 'D', 4, [1], name='truc')
        tso.insert(engine, ts, 'ts_through_time')
        assert idate1 == tso.latest_insertion_date(engine, 'ts_through_time')

    idate2 = datetime(2015, 1, 2, 15, 43, 23)
    with tso.newchangeset(engine, 'test', _insertion_date=idate2):

        ts = genserie(datetime(2010, 1, 4), 'D', 4, [2], name='truc')
        tso.insert(engine, ts, 'ts_through_time')
        assert idate2 == tso.latest_insertion_date(engine, 'ts_through_time')

    idate3 = datetime(2015, 1, 3, 15, 43, 23)
    with tso.newchangeset(engine, 'test', _insertion_date=idate3):

        ts = genserie(datetime(2010, 1, 4), 'D', 4, [3], name='truc')
        tso.insert(engine, ts, 'ts_through_time')
        assert idate3 == tso.latest_insertion_date(engine, 'ts_through_time')

    ts = tso.get(engine, 'ts_through_time')

    assert_df("""
2010-01-04    3.0
2010-01-05    3.0
2010-01-06    3.0
2010-01-07    3.0
""", ts)

    ts = tso.get(engine, 'ts_through_time',
                 revision_date=datetime(2015, 1, 2, 18, 43, 23) )

    assert_df("""
2010-01-04    2.0
2010-01-05    2.0
2010-01-06    2.0
2010-01-07    2.0
""", ts)

    ts = tso.get(engine, 'ts_through_time',
                 revision_date=datetime(2015, 1, 1, 18, 43, 23))

    assert_df("""
2010-01-04    1.0
2010-01-05    1.0
2010-01-06    1.0
2010-01-07    1.0
""", ts)

    ts = tso.get(engine, 'ts_through_time',
                 revision_date=datetime(2014, 1, 1, 18, 43, 23))

    assert ts is None


def test_snapshots(engine):
    tso = TimeSerie()
    tso._snapshot_interval = 4

    with engine.connect() as cnx:
        for tscount in range(1, 11):
            ts = genserie(datetime(2015, 1, 1), 'D', tscount, [1])
            diff = tso.insert(cnx, ts, 'growing', 'babar')
            assert diff.index[0] == diff.index[-1] == ts.index[-1]

    diff = tso.insert(engine, ts, 'growing', 'babar')
    assert diff is None

    df = pd.read_sql("select id from timeserie.growing where snapshot is not null",
                     engine)
    assert_df("""
   id
0   1
1   4
2   8
3  10
""", df)

    ts = tso.get(engine, 'growing')
    assert_df("""
2015-01-01    1.0
2015-01-02    1.0
2015-01-03    1.0
2015-01-04    1.0
2015-01-05    1.0
2015-01-06    1.0
2015-01-07    1.0
2015-01-08    1.0
2015-01-09    1.0
2015-01-10    1.0
""", ts)

    df = pd.read_sql("select id, diff, snapshot from timeserie.growing order by id", engine)
    for attr in ('diff', 'snapshot'):
        df[attr] = df[attr].apply(lambda x: 0 if x is None else len(x))

    assert_df("""
   id  diff  snapshot
0   1     0        32
1   2    32         0
2   3    32         0
3   4    32       125
4   5    32         0
5   6    32         0
6   7    32         0
7   8    32       249
8   9    32         0
9  10    32       311
""", df)

    table = tso._get_ts_table(engine, 'growing')
    snapid, snap = tso._find_snapshot(engine, table, ())
    assert snapid == 10
    assert (ts == snap).all()


def test_deletion(engine):
    tso = TimeSerie()

    ts_begin = genserie(datetime(2010, 1, 1), 'D', 11)
    ts_begin.iloc[-1] = np.nan
    tso.insert(engine, ts_begin, 'ts_del', 'test')

    ts = tso._build_snapshot_upto(engine, tso._get_ts_table(engine, 'ts_del'))
    assert ts.iloc[-1] == 9.0

    ts_begin.iloc[0] = np.nan
    ts_begin.iloc[3] = np.nan

    tso.insert(engine, ts_begin, 'ts_del', 'test')

    assert_df("""
2010-01-02    1.0
2010-01-03    2.0
2010-01-05    4.0
2010-01-06    5.0
2010-01-07    6.0
2010-01-08    7.0
2010-01-09    8.0
2010-01-10    9.0
""", tso.get(engine, 'ts_del'))

    ts2 = tso.get(engine, 'ts_del',
                 # force snapshot reconstruction feature
                 revision_date=datetime(2038, 1, 1))
    assert (tso.get(engine, 'ts_del') == ts2).all()

    ts_begin.iloc[0] = 42
    ts_begin.iloc[3] = 23

    tso.insert(engine, ts_begin, 'ts_del', 'test')

    assert_df("""
2010-01-01    42.0
2010-01-02     1.0
2010-01-03     2.0
2010-01-04    23.0
2010-01-05     4.0
2010-01-06     5.0
2010-01-07     6.0
2010-01-08     7.0
2010-01-09     8.0
2010-01-10     9.0
""", tso.get(engine, 'ts_del'))

    # now with string!

    ts_string = genserie(datetime(2010, 1, 1), 'D', 10, ['machin'])
    tso.insert(engine, ts_string, 'ts_string_del', 'test')

    ts_string[4] = None
    ts_string[5] = None

    tso.insert(engine, ts_string, 'ts_string_del', 'test')
    assert_df("""
2010-01-01    machin
2010-01-02    machin
2010-01-03    machin
2010-01-04    machin
2010-01-07    machin
2010-01-08    machin
2010-01-09    machin
2010-01-10    machin
""", tso.get(engine, 'ts_string_del'))

    ts_string[4] = 'truc'
    ts_string[6] = 'truc'

    tso.insert(engine, ts_string, 'ts_string_del', 'test')
    assert_df("""
2010-01-01    machin
2010-01-02    machin
2010-01-03    machin
2010-01-04    machin
2010-01-05      truc
2010-01-07      truc
2010-01-08    machin
2010-01-09    machin
2010-01-10    machin
""", tso.get(engine, 'ts_string_del'))

    # first insertion with only nan

    ts_begin = genserie(datetime(2010, 1, 1), 'D', 10, [np.nan])
    tso.insert(engine, ts_begin, 'ts_null', 'test')

    assert tso.get(engine, 'ts_null') is None


def test_multi_index(engine):
    tso = TimeSerie()

    appdate_0 = pd.DatetimeIndex(start=datetime(2015, 1, 1),
                                 end=datetime(2015, 1, 2),
                                 freq='D').values
    pubdate_0 = [pd.datetime(2015, 1, 11, 12, 0, 0)] * 2
    insertion_date_0 = [pd.datetime(2015, 1, 11, 12, 30, 0)] * 2

    multi = [
        appdate_0,
        np.array(pubdate_0),
        np.array(insertion_date_0)
    ]

    ts_multi = pd.Series(range(2), index=multi)
    ts_multi.index.rename(['b', 'c', 'a'], inplace=True)

    tso.insert(engine, ts_multi, 'ts_multi_simple', 'test')

    ts = tso.get(engine, 'ts_multi_simple')
    assert_df("""
                                                    ts_multi_simple
a                   b          c                                   
2015-01-11 12:30:00 2015-01-01 2015-01-11 12:00:00                0
                    2015-01-02 2015-01-11 12:00:00                1
""", pd.DataFrame(ts))

    diff = tso.insert(engine, ts_multi, 'ts_multi_simple', 'test')
    assert diff is None

    ts_multi_2 = pd.Series([0, 2], index=multi)
    ts_multi_2.index.rename(['b', 'c', 'a'], inplace=True)

    tso.insert(engine, ts_multi_2, 'ts_multi_simple', 'test')
    ts = tso.get(engine, 'ts_multi_simple')

    assert_df("""
                                                    ts_multi_simple
a                   b          c                                   
2015-01-11 12:30:00 2015-01-01 2015-01-11 12:00:00                0
                    2015-01-02 2015-01-11 12:00:00                2
""", pd.DataFrame(ts))

    # bigger ts
    appdate_0 = pd.DatetimeIndex(start=datetime(2015, 1, 1),
                                 end=datetime(2015, 1, 4),
                                 freq='D').values
    pubdate_0 = [pd.datetime(2015, 1, 11, 12, 0, 0)] * 4
    insertion_date_0 = [pd.datetime(2015, 1, 11, 12, 30, 0)] * 4

    appdate_1 = pd.DatetimeIndex(start=datetime(2015, 1, 1),
                                 end=datetime(2015, 1, 4),
                                 freq='D').values

    pubdate_1 = [pd.datetime(2015, 1, 21, 12, 0, 0)] * 4
    insertion_date_1 = [pd.datetime(2015, 1, 21, 12, 30, 0)] * 4

    multi = [
        np.concatenate([appdate_0, appdate_1]),
        np.array(pubdate_0 + pubdate_1),
        np.array(insertion_date_0 + insertion_date_1)
    ]

    ts_multi = pd.Series(range(8), index=multi)
    ts_multi.index.rename(['a', 'c', 'b'], inplace=True)

    tso.insert(engine, ts_multi, 'ts_multi', 'test')
    ts = tso.get(engine, 'ts_multi')

    assert_df("""
                                                    ts_multi
a          b                   c                            
2015-01-01 2015-01-11 12:30:00 2015-01-11 12:00:00         0
           2015-01-21 12:30:00 2015-01-21 12:00:00         4
2015-01-02 2015-01-11 12:30:00 2015-01-11 12:00:00         1
           2015-01-21 12:30:00 2015-01-21 12:00:00         5
2015-01-03 2015-01-11 12:30:00 2015-01-11 12:00:00         2
           2015-01-21 12:30:00 2015-01-21 12:00:00         6
2015-01-04 2015-01-11 12:30:00 2015-01-11 12:00:00         3
           2015-01-21 12:30:00 2015-01-21 12:00:00         7
    """, pd.DataFrame(ts.sort_index()))
    # Note: the columnns are returned according to the alphabetic order

    appdate_2 = pd.DatetimeIndex(start=datetime(2015, 1, 1),
                                 end=datetime(2015, 1, 4),
                                 freq='D').values
    pubdate_2 = [pd.datetime(2015, 1, 31, 12, 0, 0)] * 4
    insertion_date_2 = [pd.datetime(2015, 1, 31, 12, 30, 0)] * 4

    multi_2 = [
        np.concatenate([appdate_1, appdate_2]),
        np.array(pubdate_1 + pubdate_2),
        np.array(insertion_date_1 + insertion_date_2)
    ]

    ts_multi_2 = pd.Series([4] * 8, index=multi_2)
    ts_multi_2.index.rename(['a', 'c', 'b'], inplace=True)

    # A second ts is inserted with some index in common with the first
    # one: appdate_1, pubdate_1,and insertion_date_1. The value is set
    # at 4, which matches the previous value of the "2015-01-01" point.

    diff = tso.insert(engine, ts_multi_2, 'ts_multi', 'test')
    assert_df("""
                                                    ts_multi
a          b                   c                            
2015-01-01 2015-01-31 12:30:00 2015-01-31 12:00:00       4.0
2015-01-02 2015-01-21 12:30:00 2015-01-21 12:00:00       4.0
           2015-01-31 12:30:00 2015-01-31 12:00:00       4.0
2015-01-03 2015-01-21 12:30:00 2015-01-21 12:00:00       4.0
           2015-01-31 12:30:00 2015-01-31 12:00:00       4.0
2015-01-04 2015-01-21 12:30:00 2015-01-21 12:00:00       4.0
           2015-01-31 12:30:00 2015-01-31 12:00:00       4.0
        """, pd.DataFrame(diff.sort_index()))
    # the differential skips a value for "2015-01-01"
    # which does not change from the previous ts

    ts = tso.get(engine, 'ts_multi')
    assert_df("""
                                                    ts_multi
a          b                   c                            
2015-01-01 2015-01-11 12:30:00 2015-01-11 12:00:00         0
           2015-01-21 12:30:00 2015-01-21 12:00:00         4
           2015-01-31 12:30:00 2015-01-31 12:00:00         4
2015-01-02 2015-01-11 12:30:00 2015-01-11 12:00:00         1
           2015-01-21 12:30:00 2015-01-21 12:00:00         4
           2015-01-31 12:30:00 2015-01-31 12:00:00         4
2015-01-03 2015-01-11 12:30:00 2015-01-11 12:00:00         2
           2015-01-21 12:30:00 2015-01-21 12:00:00         4
           2015-01-31 12:30:00 2015-01-31 12:00:00         4
2015-01-04 2015-01-11 12:30:00 2015-01-11 12:00:00         3
           2015-01-21 12:30:00 2015-01-21 12:00:00         4
           2015-01-31 12:30:00 2015-01-31 12:00:00         4
        """, pd.DataFrame(ts.sort_index()))

    # the result ts have now 3 values for each point in 'a'
