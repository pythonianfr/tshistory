# coding: utf-8
from datetime import datetime, timedelta
from time import time
from dateutil import parser
import calendar

from pathlib2 import Path
import pandas as pd
import numpy as np
import pytest
from mock import patch

from tshistory.testutil import assert_group_equals, genserie, assert_df

DATADIR = Path(__file__).parent / 'data'


def test_changeset(engine, tsh):
    index = pd.date_range(start=datetime(2017, 1, 1), freq='D', periods=3)
    data = [1., 2., 3.]

    with patch('tshistory.tsio.datetime') as mock_date:
        mock_date.now.return_value = datetime(2020, 1, 1)
        with engine.connect() as cn:
            with tsh.newchangeset(cn, 'babar'):
                tsh.insert(cn, pd.Series(data, index=index), 'ts_values', author='WONTBEUSED')
                tsh.insert(cn, pd.Series(['a', 'b', 'c'], index=index), 'ts_othervalues')

        # bogus author won't show up
        assert tsh.log(engine)[0]['author'] == 'babar'

        g = tsh.get_group(engine, 'ts_values')
        g2 = tsh.get_group(engine, 'ts_othervalues')
        assert_group_equals(g, g2)

        with pytest.raises(AssertionError):
            tsh.insert(engine, pd.Series([2, 3, 4], index=index), 'ts_values')

        with engine.connect() as cn:
            data.append(data.pop(0))
            with tsh.newchangeset(cn, 'celeste'):
                tsh.insert(cn, pd.Series(data, index=index), 'ts_values')
                # below should be a noop
                tsh.insert(cn, pd.Series(['a', 'b', 'c'], index=index), 'ts_othervalues')

    g = tsh.get_group(engine, 'ts_values')
    assert ['ts_values'] == list(g.keys())

    assert_df("""
2017-01-01    2.0
2017-01-02    3.0
2017-01-03    1.0
""", tsh.get(engine, 'ts_values'))

    assert_df("""
2017-01-01    a
2017-01-02    b
2017-01-03    c
""", tsh.get(engine, 'ts_othervalues'))

    log = tsh.log(engine, names=['ts_values', 'ts_othervalues'])
    assert [
        {'author': 'babar',
         'rev': 1,
         'date': datetime(2020, 1, 1, 0, 0),
         'meta': {},
         'names': ['ts_values', 'ts_othervalues']},
        {'author': 'celeste',
         'rev': 2,
         'meta': {},
         'date': datetime(2020, 1, 1, 0, 0),
         'names': ['ts_values']}
    ] == log

    log = tsh.log(engine, names=['ts_othervalues'])
    assert len(log) == 1
    assert log[0]['rev'] == 1
    assert log[0]['names'] == ['ts_values', 'ts_othervalues']

    log = tsh.log(engine, fromrev=2)
    assert len(log) == 1

    log = tsh.log(engine, torev=1)
    assert len(log) == 1

    info = tsh.info(engine)
    assert {
        'changeset count': 2,
        'serie names': ['ts_othervalues', 'ts_values'],
        'series count': 2
    } == info


def test_strip(engine, tsh):
    for i in range(1, 5):
        pubdate = datetime(2017, 1, i)
        ts = genserie(datetime(2017, 1, 10), 'H', 1 + i)
        with tsh.newchangeset(engine, 'babar', _insertion_date=pubdate):
            tsh.insert(engine, ts, 'xserie')
        # also insert something completely unrelated
        tsh.insert(engine, genserie(datetime(2018, 1, 1), 'D', 1 + i), 'yserie', 'celeste')

    csida = tsh.changeset_at(engine, 'xserie', datetime(2017, 1, 3))
    assert csida is not None
    csidb = tsh.changeset_at(engine, 'xserie', datetime(2017, 1, 3, 1), mode='before')
    csidc = tsh.changeset_at(engine, 'xserie', datetime(2017, 1, 3, 1), mode='after')
    assert csidb < csida < csidc

    log = tsh.log(engine, names=['xserie', 'yserie'])
    assert [(idx, l['author']) for idx, l in enumerate(log, start=1)
    ] == [
        (1, 'babar'),
        (2, 'celeste'),
        (3, 'babar'),
        (4, 'celeste'),
        (5, 'babar'),
        (6, 'celeste'),
        (7, 'babar'),
        (8, 'celeste')
    ]

    h = tsh.get_history(engine, 'xserie')
    assert_df("""
insertion_date  value_date         
2017-01-01      2017-01-10 00:00:00    0.0
                2017-01-10 01:00:00    1.0
2017-01-02      2017-01-10 00:00:00    0.0
                2017-01-10 01:00:00    1.0
                2017-01-10 02:00:00    2.0
2017-01-03      2017-01-10 00:00:00    0.0
                2017-01-10 01:00:00    1.0
                2017-01-10 02:00:00    2.0
                2017-01-10 03:00:00    3.0
2017-01-04      2017-01-10 00:00:00    0.0
                2017-01-10 01:00:00    1.0
                2017-01-10 02:00:00    2.0
                2017-01-10 03:00:00    3.0
                2017-01-10 04:00:00    4.0
""", h)

    csid = tsh.changeset_at(engine, 'xserie', datetime(2017, 1, 3))
    with engine.connect() as cn:
        tsh.strip(cn, 'xserie', csid)

    assert_df("""
insertion_date  value_date         
2017-01-01      2017-01-10 00:00:00    0.0
                2017-01-10 01:00:00    1.0
2017-01-02      2017-01-10 00:00:00    0.0
                2017-01-10 01:00:00    1.0
                2017-01-10 02:00:00    2.0
""", tsh.get_history(engine, 'xserie'))

    assert_df("""
2017-01-10 00:00:00    0.0
2017-01-10 01:00:00    1.0
2017-01-10 02:00:00    2.0
""", tsh.get(engine, 'xserie'))

    # internal structure is ok
    with engine.connect() as cn:
        cn.execute('set search_path to "{}.timeserie"'.format(tsh.namespace))
        df = pd.read_sql("select id, diff, snapshot from xserie order by id", cn)
        for attr in ('diff', 'snapshot'):
            df[attr] = df[attr].apply(lambda x: False if x is None else True)

    assert_df("""
id   diff  snapshot
0   1  False      True
1   2   True      True
""", df)

    log = tsh.log(engine, names=['xserie', 'yserie'])
    # 5 and 7 have disappeared
    assert [l['author'] for l in log
    ] == ['babar', 'celeste', 'babar', 'celeste', 'celeste', 'celeste']

    log = tsh.log(engine, stripped=True, names=['xserie', 'yserie'])
    assert [list(l['meta'].values())[0][:-1] + 'X' for l in log if l['meta']
    ] == [
        'got stripped from X',
        'got stripped from X'
    ]


def test_tstamp_roundtrip(engine, tsh):
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

    tsh.insert(engine, ts, 'tztest', 'Babar')
    back = tsh.get(engine, 'tztest')

    # though un localized we understand it's been normalized to utc
    assert_df("""
2017-10-28 23:00:00+00:00    0.0
2017-10-29 00:00:00+00:00    1.0
2017-10-29 01:00:00+00:00    2.0
2017-10-29 02:00:00+00:00    3.0
""", back)

    assert (ts.index == back.index).all()
    assert str(back.index.dtype) == 'datetime64[ns, UTC]'


def test_multi_index_aware(engine, tsh):
    ts_multi_aware = genserie(
        start=pd.Timestamp(
            2017, 10, 28, 23
        ).tz_localize('UTC').tz_convert('Europe/Paris'),
        freq=['15T', '30T', '60T'],
        repeat=10,
        tz='Europe/Paris',
        name='ts_multi_aware',
    )
    ts_multi_aware.index.rename(['a', 'b', 'c'], inplace=True)

    assert_df("""
a                          b                          c                        
2017-10-29 01:00:00+02:00  2017-10-29 01:00:00+02:00  2017-10-29 01:00:00+02:00    0
2017-10-29 01:15:00+02:00  2017-10-29 01:30:00+02:00  2017-10-29 02:00:00+02:00    1
2017-10-29 01:30:00+02:00  2017-10-29 02:00:00+02:00  2017-10-29 02:00:00+01:00    2
2017-10-29 01:45:00+02:00  2017-10-29 02:30:00+02:00  2017-10-29 03:00:00+01:00    3
2017-10-29 02:00:00+02:00  2017-10-29 02:00:00+01:00  2017-10-29 04:00:00+01:00    4
2017-10-29 02:15:00+02:00  2017-10-29 02:30:00+01:00  2017-10-29 05:00:00+01:00    5
2017-10-29 02:30:00+02:00  2017-10-29 03:00:00+01:00  2017-10-29 06:00:00+01:00    6
2017-10-29 02:45:00+02:00  2017-10-29 03:30:00+01:00  2017-10-29 07:00:00+01:00    7
2017-10-29 02:00:00+01:00  2017-10-29 04:00:00+01:00  2017-10-29 08:00:00+01:00    8
2017-10-29 02:15:00+01:00  2017-10-29 04:30:00+01:00  2017-10-29 09:00:00+01:00    9
""", ts_multi_aware)

    tsh.insert(engine, ts_multi_aware, 'ts_multi_aware', 'test')
    ts_aware = tsh.get(engine, 'ts_multi_aware')

    assert_df("""
ts_multi_aware
a                         b                         c                                        
2017-10-28 23:00:00+00:00 2017-10-28 23:00:00+00:00 2017-10-28 23:00:00+00:00             0.0
2017-10-28 23:15:00+00:00 2017-10-28 23:30:00+00:00 2017-10-29 00:00:00+00:00             1.0
2017-10-28 23:30:00+00:00 2017-10-29 00:00:00+00:00 2017-10-29 01:00:00+00:00             2.0
2017-10-28 23:45:00+00:00 2017-10-29 00:30:00+00:00 2017-10-29 02:00:00+00:00             3.0
2017-10-29 00:00:00+00:00 2017-10-29 01:00:00+00:00 2017-10-29 03:00:00+00:00             4.0
2017-10-29 00:15:00+00:00 2017-10-29 01:30:00+00:00 2017-10-29 04:00:00+00:00             5.0
2017-10-29 00:30:00+00:00 2017-10-29 02:00:00+00:00 2017-10-29 05:00:00+00:00             6.0
2017-10-29 00:45:00+00:00 2017-10-29 02:30:00+00:00 2017-10-29 06:00:00+00:00             7.0
2017-10-29 01:00:00+00:00 2017-10-29 03:00:00+00:00 2017-10-29 07:00:00+00:00             8.0
2017-10-29 01:15:00+00:00 2017-10-29 03:30:00+00:00 2017-10-29 08:00:00+00:00             9.0
    """, pd.DataFrame(ts_aware.sort_index()))
    # Note: the columnns are returned according to the alphabetic order

    ts = tsh.get(engine, 'ts_multi_aware',
                 from_value_date=pd.Timestamp(2017, 10, 29, 0).tz_localize('UTC'),
                 to_value_date=pd.Timestamp(2017, 10, 29, 1).tz_localize('UTC'))
    assert_df("""
a                          b                          c                        
2017-10-29 00:00:00+00:00  2017-10-29 01:00:00+00:00  2017-10-29 03:00:00+00:00    4.0
2017-10-29 00:15:00+00:00  2017-10-29 01:30:00+00:00  2017-10-29 04:00:00+00:00    5.0
2017-10-29 00:30:00+00:00  2017-10-29 02:00:00+00:00  2017-10-29 05:00:00+00:00    6.0
2017-10-29 00:45:00+00:00  2017-10-29 02:30:00+00:00  2017-10-29 06:00:00+00:00    7.0
2017-10-29 01:00:00+00:00  2017-10-29 03:00:00+00:00  2017-10-29 07:00:00+00:00    8.0
    """, ts)


def test_differential(engine, tsh):
    ts_begin = genserie(datetime(2010, 1, 1), 'D', 10)
    tsh.insert(engine, ts_begin, 'ts_test', 'test')

    assert tsh.exists(engine, 'ts_test')
    assert not tsh.exists(engine, 'this_does_not_exist')

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
""", tsh.get(engine, 'ts_test'))

    # we should detect the emission of a message
    tsh.insert(engine, ts_begin, 'ts_test', 'babar')

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
""", tsh.get(engine, 'ts_test'))

    ts_slight_variation = ts_begin.copy()
    ts_slight_variation.iloc[3] = 0
    ts_slight_variation.iloc[6] = 0
    tsh.insert(engine, ts_slight_variation, 'ts_test', 'celeste')

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
""", tsh.get(engine, 'ts_test'))

    ts_longer = genserie(datetime(2010, 1, 3), 'D', 15)
    ts_longer.iloc[1] = 2.48
    ts_longer.iloc[3] = 3.14
    ts_longer.iloc[5] = ts_begin.iloc[7]

    tsh.insert(engine, ts_longer, 'ts_test', 'test')

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
""", tsh.get(engine, 'ts_test'))

    # start testing manual overrides
    ts_begin = genserie(datetime(2010, 1, 1), 'D', 5, initval=[2])
    ts_begin.loc['2010-01-04'] = -1
    tsh.insert(engine, ts_begin, 'ts_mixte', 'test')

    # -1 represents bogus upstream data
    assert_df("""
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
""", tsh.get(engine, 'ts_mixte'))

    # refresh all the period + 1 extra data point
    ts_more = genserie(datetime(2010, 1, 2), 'D', 5, [2])
    ts_more.loc['2010-01-04'] = -1
    tsh.insert(engine, ts_more, 'ts_mixte', 'test')

    assert_df("""
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
2010-01-06    2.0
""", tsh.get(engine, 'ts_mixte'))

    # just append an extra data point
    # with no intersection with the previous ts
    ts_one_more = genserie(datetime(2010, 1, 7), 'D', 1, [3])
    tsh.insert(engine, ts_one_more, 'ts_mixte', 'test')

    assert_df("""
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
2010-01-06    2.0
2010-01-07    3.0
""", tsh.get(engine, 'ts_mixte'))

    with engine.connect() as cn:
        cn.execute('set search_path to "{0}.timeserie", {0}, public'.format(tsh.namespace))
        hist = pd.read_sql('select id, parent from ts_test order by id',
                           cn)
        assert_df("""
   id  parent
0   1     NaN
1   2     1.0
2   3     2.0
""", hist)

        hist = pd.read_sql('select id, parent from ts_mixte order by id',
                           cn)
        assert_df("""
   id  parent
0   1     NaN
1   2     1.0
2   3     2.0
""", hist)

        allts = pd.read_sql("select name, table_name from registry "
                            "where name in ('ts_test', 'ts_mixte')",
                            cn)

        assert_df("""
name              table_name
0   ts_test   {0}.timeserie.ts_test
1  ts_mixte  {0}.timeserie.ts_mixte
""".format(tsh.namespace), allts)

        assert_df("""
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
2010-01-06    2.0
2010-01-07    3.0
""", tsh.get(cn, 'ts_mixte',
             revision_date=datetime.now()))


def test_bad_import(engine, tsh):
    # the data were parsed as date by pd.read_json()
    df_result = pd.read_csv(str(DATADIR / 'test_data.csv'))
    df_result['Gas Day'] = df_result['Gas Day'].apply(parser.parse, dayfirst=True, yearfirst=False)
    df_result.set_index('Gas Day', inplace=True)
    ts = df_result['SC']

    tsh.insert(engine, ts, 'SND_SC', 'test')
    result = tsh.get(engine, 'SND_SC')
    assert result.dtype == 'float64'

    # insertion of empty ts
    ts = pd.Series(name='truc', dtype='object')
    tsh.insert(engine, ts, 'empty_ts', 'test')
    assert tsh.get(engine, 'empty_ts') is None

    # nan in ts
    # all na
    ts = genserie(datetime(2010, 1, 10), 'D', 10, [np.nan], name='truc')
    tsh.insert(engine, ts, 'test_nan', 'test')
    assert tsh.get(engine, 'test_nan') is None

    # mixe na
    ts = pd.Series([np.nan] * 5 + [3] * 5,
                   index=pd.date_range(start=datetime(2010, 1, 10),
                                       freq='D', periods=10), name='truc')
    tsh.insert(engine, ts, 'test_nan', 'test')
    result = tsh.get(engine, 'test_nan')

    tsh.insert(engine, ts, 'test_nan', 'test')
    result = tsh.get(engine, 'test_nan')
    assert_df("""
2010-01-15    3.0
2010-01-16    3.0
2010-01-17    3.0
2010-01-18    3.0
2010-01-19    3.0
""", result)

    # get_ts with name not in database

    tsh.get(engine, 'inexisting_name', 'test')


def test_revision_date(engine, tsh):
    # we prepare a good joke for the end of the test
    ival = tsh._snapshot_interval
    tsh._snapshot_interval = 3

    for i in range(1, 5):
        with engine.connect() as cn:
            with tsh.newchangeset(cn, 'test',
                                  _insertion_date=datetime(2016, 1, i)):
                tsh.insert(cn, genserie(datetime(2017, 1, i), 'D', 3, [i]), 'revdate')

    # end of prologue, now some real meat
    idate0 = datetime(2015, 1, 1, 0, 0, 0)
    with tsh.newchangeset(engine, 'test', _insertion_date=idate0):

        ts = genserie(datetime(2010, 1, 4), 'D', 4, [0], name='truc')
        tsh.insert(engine, ts, 'ts_through_time')
        assert idate0 == tsh.latest_insertion_date(engine, 'ts_through_time')

    idate1 = datetime(2015, 1, 1, 15, 43, 23)
    with tsh.newchangeset(engine, 'test', _insertion_date=idate1):

        ts = genserie(datetime(2010, 1, 4), 'D', 4, [1], name='truc')
        tsh.insert(engine, ts, 'ts_through_time')
        assert idate1 == tsh.latest_insertion_date(engine, 'ts_through_time')

    idate2 = datetime(2015, 1, 2, 15, 43, 23)
    with tsh.newchangeset(engine, 'test', _insertion_date=idate2):

        ts = genserie(datetime(2010, 1, 4), 'D', 4, [2], name='truc')
        tsh.insert(engine, ts, 'ts_through_time')
        assert idate2 == tsh.latest_insertion_date(engine, 'ts_through_time')

    idate3 = datetime(2015, 1, 3, 15, 43, 23)
    with tsh.newchangeset(engine, 'test', _insertion_date=idate3):

        ts = genserie(datetime(2010, 1, 4), 'D', 4, [3], name='truc')
        tsh.insert(engine, ts, 'ts_through_time')
        assert idate3 == tsh.latest_insertion_date(engine, 'ts_through_time')

    ts = tsh.get(engine, 'ts_through_time')

    assert_df("""
2010-01-04    3.0
2010-01-05    3.0
2010-01-06    3.0
2010-01-07    3.0
""", ts)

    ts = tsh.get(engine, 'ts_through_time',
                 revision_date=datetime(2015, 1, 2, 18, 43, 23))

    assert_df("""
2010-01-04    2.0
2010-01-05    2.0
2010-01-06    2.0
2010-01-07    2.0
""", ts)

    ts = tsh.get(engine, 'ts_through_time',
                 revision_date=datetime(2015, 1, 1, 18, 43, 23))

    assert_df("""
2010-01-04    1.0
2010-01-05    1.0
2010-01-06    1.0
2010-01-07    1.0
""", ts)

    ts = tsh.get(engine, 'ts_through_time',
                 revision_date=datetime(2014, 1, 1, 18, 43, 23))

    assert ts is None

    # epilogue: back to the revdate issue
    assert_df("""
2017-01-01    1.0
2017-01-02    2.0
2017-01-03    3.0
2017-01-04    4.0
2017-01-05    4.0
2017-01-06    4.0
""", tsh.get(engine, 'revdate'))

    oldstate = tsh.get(engine, 'revdate', revision_date=datetime(2016, 1, 2))
    assert_df("""
2017-01-01    1.0
2017-01-02    2.0
2017-01-03    2.0
2017-01-04    2.0
""", oldstate)

    tsh._snapshot_interval = ival


def test_snapshots(engine, tsh):
    baseinterval = tsh._snapshot_interval
    tsh._snapshot_interval = 4

    with engine.connect() as cn:
        for tscount in range(1, 11):
            ts = genserie(datetime(2015, 1, 1), 'D', tscount, [1])
            diff = tsh.insert(cn, ts, 'growing', 'babar')
            assert diff.index[0] == diff.index[-1] == ts.index[-1]

    diff = tsh.insert(engine, ts, 'growing', 'babar')
    assert diff is None

    with engine.connect() as cn:
        cn.execute('set search_path to "{}.timeserie"'.format(tsh.namespace))
        df = pd.read_sql("select id from growing where snapshot is not null",
                         cn)
        assert_df("""
   id
0   1
1   4
2   8
3  10
""", df)

        ts = tsh.get(cn, 'growing')
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

        df = pd.read_sql("select id, diff, snapshot from growing order by id", cn)
        for attr in ('diff', 'snapshot'):
            df[attr] = df[attr].apply(lambda x: 0 if x is None else len(x))

        assert_df("""
id  diff  snapshot
0   1     0        35
1   2    36         0
2   3    36         0
3   4    36        47
4   5    36         0
5   6    36         0
6   7    36         0
7   8    36        59
8   9    36         0
9  10    36        67
""", df)

    table = tsh._get_ts_table(engine, 'growing')
    snapid, snap = tsh._find_snapshot(engine, table, ())
    assert snapid == 10
    assert (ts == snap).all()
    tsh._snapshot_interval = baseinterval


def test_deletion(engine, tsh):
    ts_begin = genserie(datetime(2010, 1, 1), 'D', 11)
    ts_begin.iloc[-1] = np.nan
    tsh.insert(engine, ts_begin, 'ts_del', 'test')

    ts = tsh._build_snapshot_upto(engine, tsh._get_ts_table(engine, 'ts_del'))
    assert ts.iloc[-1] == 9.0

    ts_begin.iloc[0] = np.nan
    ts_begin.iloc[3] = np.nan

    tsh.insert(engine, ts_begin, 'ts_del', 'test')

    assert_df("""
2010-01-02    1.0
2010-01-03    2.0
2010-01-05    4.0
2010-01-06    5.0
2010-01-07    6.0
2010-01-08    7.0
2010-01-09    8.0
2010-01-10    9.0
""", tsh.get(engine, 'ts_del'))

    ts2 = tsh.get(engine, 'ts_del',
                 # force snapshot reconstruction feature
                 revision_date=datetime(2038, 1, 1))
    assert (tsh.get(engine, 'ts_del') == ts2).all()

    ts_begin.iloc[0] = 42
    ts_begin.iloc[3] = 23

    tsh.insert(engine, ts_begin, 'ts_del', 'test')

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
""", tsh.get(engine, 'ts_del'))

    # now with string!

    ts_string = genserie(datetime(2010, 1, 1), 'D', 10, ['machin'])
    tsh.insert(engine, ts_string, 'ts_string_del', 'test')

    ts_string[4] = None
    ts_string[5] = None

    tsh.insert(engine, ts_string, 'ts_string_del', 'test')
    assert_df("""
2010-01-01    machin
2010-01-02    machin
2010-01-03    machin
2010-01-04    machin
2010-01-07    machin
2010-01-08    machin
2010-01-09    machin
2010-01-10    machin
""", tsh.get(engine, 'ts_string_del'))

    ts_string[4] = 'truc'
    ts_string[6] = 'truc'

    tsh.insert(engine, ts_string, 'ts_string_del', 'test')
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
""", tsh.get(engine, 'ts_string_del'))

    ts_string[ts_string.index] = np.nan
    tsh.insert(engine, ts_string, 'ts_string_del', 'test')

    erased = tsh.get(engine, 'ts_string_del')
    assert len(erased) == 0

    # first insertion with only nan

    ts_begin = genserie(datetime(2010, 1, 1), 'D', 10, [np.nan])
    tsh.insert(engine, ts_begin, 'ts_null', 'test')

    assert tsh.get(engine, 'ts_null') is None

    # exhibit issue with nans handling
    ts_repushed = genserie(datetime(2010, 1, 1), 'D', 11)
    ts_repushed[0:3] = np.nan

    assert_df("""
2010-01-01     NaN
2010-01-02     NaN
2010-01-03     NaN
2010-01-04     3.0
2010-01-05     4.0
2010-01-06     5.0
2010-01-07     6.0
2010-01-08     7.0
2010-01-09     8.0
2010-01-10     9.0
2010-01-11    10.0
Freq: D
""", ts_repushed)

    tsh.insert(engine, ts_repushed, 'ts_repushed', 'test')
    diff = tsh.insert(engine, ts_repushed, 'ts_repushed', 'test')
    assert diff is None

    # there is no difference
    assert 0 == len(tsh._compute_diff(ts_repushed, ts_repushed))

    ts_add = genserie(datetime(2010, 1, 1), 'D', 15)
    ts_add.iloc[0] = np.nan
    ts_add.iloc[13:] = np.nan
    ts_add.iloc[8] = np.nan
    diff = tsh._compute_diff(ts_repushed, ts_add)

    assert_df("""
2010-01-02     1.0
2010-01-03     2.0
2010-01-09     NaN
2010-01-12    11.0
2010-01-13    12.0""", diff.sort_index())
    # value on nan => value
    # nan on value => nan
    # nan on nan => Nothing
    # nan on nothing=> Nothing

    # full erasing
    # numeric
    ts_begin = genserie(datetime(2010, 1, 1), 'D', 4)
    tsh.insert(engine, ts_begin, 'ts_full_del', 'test')

    ts_begin.iloc[:] = np.nan
    tsh.insert(engine, ts_begin, 'ts_full_del', 'test')

    ts_end = genserie(datetime(2010, 1, 1), 'D', 4)
    tsh.insert(engine, ts_end, 'ts_full_del', 'test')

    # string

    ts_begin = genserie(datetime(2010, 1, 1), 'D', 4, ['text'])
    tsh.insert(engine, ts_begin, 'ts_full_del_str', 'test')

    ts_begin.iloc[:] = np.nan
    tsh.insert(engine, ts_begin, 'ts_full_del_str', 'test')

    ts_end = genserie(datetime(2010, 1, 1), 'D', 4, ['text'])
    tsh.insert(engine, ts_end, 'ts_full_del_str', 'test')


def test_multi_index(engine, tsh):
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

    tsh.insert(engine, ts_multi, 'ts_multi_simple', 'test')

    ts = tsh.get(engine, 'ts_multi_simple')
    assert_df("""
                                                    ts_multi_simple
a                   b          c                                   
2015-01-11 12:30:00 2015-01-01 2015-01-11 12:00:00              0.0
                    2015-01-02 2015-01-11 12:00:00              1.0
""", pd.DataFrame(ts))

    diff = tsh.insert(engine, ts_multi, 'ts_multi_simple', 'test')
    assert diff is None

    ts_multi_2 = pd.Series([0, 2], index=multi)
    ts_multi_2.index.rename(['b', 'c', 'a'], inplace=True)

    tsh.insert(engine, ts_multi_2, 'ts_multi_simple', 'test')
    ts = tsh.get(engine, 'ts_multi_simple')

    assert_df("""
                                                    ts_multi_simple
a                   b          c                                   
2015-01-11 12:30:00 2015-01-01 2015-01-11 12:00:00              0.0
                    2015-01-02 2015-01-11 12:00:00              2.0
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

    tsh.insert(engine, ts_multi, 'ts_multi', 'test')
    ts = tsh.get(engine, 'ts_multi')

    assert_df("""
                                                    ts_multi
a          b                   c                            
2015-01-01 2015-01-11 12:30:00 2015-01-11 12:00:00       0.0
           2015-01-21 12:30:00 2015-01-21 12:00:00       4.0
2015-01-02 2015-01-11 12:30:00 2015-01-11 12:00:00       1.0
           2015-01-21 12:30:00 2015-01-21 12:00:00       5.0
2015-01-03 2015-01-11 12:30:00 2015-01-11 12:00:00       2.0
           2015-01-21 12:30:00 2015-01-21 12:00:00       6.0
2015-01-04 2015-01-11 12:30:00 2015-01-11 12:00:00       3.0
           2015-01-21 12:30:00 2015-01-21 12:00:00       7.0
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

    diff = tsh.insert(engine, ts_multi_2, 'ts_multi', 'test')
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

    ts = tsh.get(engine, 'ts_multi')
    assert_df("""
                                                    ts_multi
a          b                   c                            
2015-01-01 2015-01-11 12:30:00 2015-01-11 12:00:00       0.0
           2015-01-21 12:30:00 2015-01-21 12:00:00       4.0
           2015-01-31 12:30:00 2015-01-31 12:00:00       4.0
2015-01-02 2015-01-11 12:30:00 2015-01-11 12:00:00       1.0
           2015-01-21 12:30:00 2015-01-21 12:00:00       4.0
           2015-01-31 12:30:00 2015-01-31 12:00:00       4.0
2015-01-03 2015-01-11 12:30:00 2015-01-11 12:00:00       2.0
           2015-01-21 12:30:00 2015-01-21 12:00:00       4.0
           2015-01-31 12:30:00 2015-01-31 12:00:00       4.0
2015-01-04 2015-01-11 12:30:00 2015-01-11 12:00:00       3.0
           2015-01-21 12:30:00 2015-01-21 12:00:00       4.0
           2015-01-31 12:30:00 2015-01-31 12:00:00       4.0
        """, pd.DataFrame(ts.sort_index()))

    # the result ts have now 3 values for each point in 'a'


def test_multi_index_get_history(engine, tsh):
    appdate = pd.DatetimeIndex(
        start=datetime(2015, 1, 1),
        end=datetime(2015, 1, 2),
        freq='D'
    ).values
    forecast_date = [pd.Timestamp(2015, 1, 11, 12, 0, 0)] * 2
    multi = [
        appdate,
        np.array(forecast_date),
    ]

    ts_multi = pd.Series(range(2), index=multi)
    ts_multi.index.rename(['app_date', 'fc_date'], inplace=True)

    tsh.insert(engine, ts_multi, 'ts_mi', 'Babar',
               _insertion_date=pd.datetime(2015, 1, 11, 12, 30, 0))


    ts = tsh.get_history(engine, 'ts_mi')
    assert_df("""
insertion_date       app_date    fc_date            
2015-01-11 12:30:00  2015-01-01  2015-01-11 12:00:00    0.0
                     2015-01-02  2015-01-11 12:00:00    1.0
""", ts)

    ts = tsh.get_history(engine, 'ts_mi', diffmode=True)

    assert_df("""
insertion_date       app_date    fc_date            
2015-01-11 12:30:00  2015-01-01  2015-01-11 12:00:00    0.0
                     2015-01-02  2015-01-11 12:00:00    1.0
""", ts)

    # new forecast
    appdate = pd.DatetimeIndex(
        start=datetime(2015, 1, 1),
        end=datetime(2015, 1, 2),
        freq='D'
    ).values
    forecast_date = [pd.Timestamp(2015, 1, 11, 13, 0, 0)] * 2
    multi = [
        appdate,
        np.array(forecast_date),
    ]

    ts_multi = pd.Series((x+.1 for x in range(2)), index=multi)
    ts_multi.index.rename(['app_date', 'fc_date'], inplace=True)

    tsh.insert(engine, ts_multi, 'ts_mi', 'Babar',
               _insertion_date=pd.datetime(2015, 1, 11, 13, 30, 0))

    ts = tsh.get_history(engine, 'ts_mi')
    assert_df("""
insertion_date       app_date    fc_date            
2015-01-11 12:30:00  2015-01-01  2015-01-11 12:00:00    0.0
                     2015-01-02  2015-01-11 12:00:00    1.0
2015-01-11 13:30:00  2015-01-01  2015-01-11 12:00:00    0.0
                                 2015-01-11 13:00:00    0.1
                     2015-01-02  2015-01-11 12:00:00    1.0
                                 2015-01-11 13:00:00    1.1
""", ts)

    ts = tsh.get_history(engine, 'ts_mi', diffmode=True)
    assert_df("""
insertion_date       app_date    fc_date            
2015-01-11 12:30:00  2015-01-01  2015-01-11 12:00:00    0.0
                     2015-01-02  2015-01-11 12:00:00    1.0
2015-01-11 13:30:00  2015-01-01  2015-01-11 13:00:00    0.1
                     2015-01-02  2015-01-11 13:00:00    1.1
""", ts)


def test_get_history(engine, tsh):
    for numserie in (1, 2, 3):
        with engine.connect() as cn:
            with tsh.newchangeset(cn, 'aurelien.campeas@pythonian.fr',
                                  _insertion_date=datetime(2017, 2, numserie)):
                tsh.insert(cn, genserie(datetime(2017, 1, 1), 'D', numserie), 'smallserie')

    ts = tsh.get(engine, 'smallserie')
    assert_df("""
2017-01-01    0.0
2017-01-02    1.0
2017-01-03    2.0
""", ts)

    logs = tsh.log(engine, names=['smallserie'])
    assert [
        {'author': 'aurelien.campeas@pythonian.fr',
         'meta': {},
         'date': datetime(2017, 2, 1, 0, 0),
         'names': ['smallserie']
        },
        {'author': 'aurelien.campeas@pythonian.fr',
         'meta': {},
         'date': datetime(2017, 2, 2, 0, 0),
         'names': ['smallserie']
        },
        {'author': 'aurelien.campeas@pythonian.fr',
         'meta': {},
         'date': datetime(2017, 2, 3, 0, 0),
         'names': ['smallserie']
        }
    ] == [{k: v for k, v in log.items() if k != 'rev'}
          for log in logs]
    histts = tsh.get_history(engine, 'smallserie')
    assert histts.name == 'smallserie'

    assert_df("""
insertion_date  value_date
2017-02-01      2017-01-01    0.0
2017-02-02      2017-01-01    0.0
                2017-01-02    1.0
2017-02-03      2017-01-01    0.0
                2017-01-02    1.0
                2017-01-03    2.0
""", histts)

    diffs = tsh.get_history(engine, 'smallserie', diffmode=True)
    assert_df("""
insertion_date  value_date
2017-02-01      2017-01-01    0.0
2017-02-02      2017-01-02    1.0
2017-02-03      2017-01-03    2.0
""", diffs)

    for idate in histts.index.get_level_values('insertion_date').unique():
        with engine.connect() as cn:
            with tsh.newchangeset(cn, 'aurelien.campeas@pythonian.f',
                                  _insertion_date=idate):
                tsh.insert(cn, histts[idate], 'smallserie2')

    # this is perfectly round-tripable
    assert (tsh.get(engine, 'smallserie2') == ts).all()
    assert (tsh.get_history(engine, 'smallserie2') == histts).all()

    # get history ranges
    tsa = tsh.get_history(engine, 'smallserie',
                          from_insertion_date=datetime(2017, 2, 2))
    assert_df("""
insertion_date  value_date
2017-02-02      2017-01-01    0.0
                2017-01-02    1.0
2017-02-03      2017-01-01    0.0
                2017-01-02    1.0
                2017-01-03    2.0
""", tsa)

    tsb = tsh.get_history(engine, 'smallserie',
                          to_insertion_date=datetime(2017, 2, 2))
    assert_df("""
insertion_date  value_date
2017-02-01      2017-01-01    0.0
2017-02-02      2017-01-01    0.0
                2017-01-02    1.0
""", tsb)

    tsc = tsh.get_history(engine, 'smallserie',
                          from_insertion_date=datetime(2017, 2, 2),
                          to_insertion_date=datetime(2017, 2, 2))
    assert_df("""
insertion_date  value_date
2017-02-02      2017-01-01    0.0
                2017-01-02    1.0
""", tsc)

    tsc = tsh.get_history(engine, 'smallserie',
                          from_insertion_date=datetime(2017, 2, 4),
                          to_insertion_date=datetime(2017, 2, 4))
    assert tsc is None

    tsc = tsh.get_history(engine, 'smallserie',
                          from_insertion_date=datetime(2016, 2, 1),
                          to_insertion_date=datetime(2017, 2, 2))
    assert_df("""
insertion_date  value_date
2017-02-01      2017-01-01    0.0
2017-02-02      2017-01-01    0.0
                2017-01-02    1.0
""", tsc)

    tsc = tsh.get_history(engine, 'smallserie',
                          from_insertion_date=datetime(2016, 2, 1),
                          to_insertion_date=datetime(2016, 12, 31))
    assert tsc is None

    # restrictions on value dates
    tsc = tsh.get_history(engine, 'smallserie',
                          from_value_date=datetime(2017, 1, 1),
                          to_value_date=datetime(2017, 1, 2))
    assert_df("""
insertion_date  value_date
2017-02-01      2017-01-01    0.0
2017-02-02      2017-01-01    0.0
                2017-01-02    1.0
2017-02-03      2017-01-01    0.0
                2017-01-02    1.0
""", tsc)

    diffs = tsh.get_history(engine, 'smallserie',
                            diffmode=True,
                            from_value_date=datetime(2017, 1, 1),
                            to_value_date=datetime(2017, 1, 2))
    assert_df("""
insertion_date  value_date
2017-02-01      2017-01-01    0.0
2017-02-02      2017-01-02    1.0
""", diffs)

    tsc = tsh.get_history(engine, 'smallserie',
                          from_value_date=datetime(2017, 1, 2))
    assert_df("""
insertion_date  value_date
2017-02-02      2017-01-02    1.0
2017-02-03      2017-01-02    1.0
                2017-01-03    2.0
""", tsc)

    tsc = tsh.get_history(engine, 'smallserie',
                          to_value_date=datetime(2017, 1, 2))
    assert_df("""
insertion_date  value_date
2017-02-01      2017-01-01    0.0
2017-02-02      2017-01-01    0.0
                2017-01-02    1.0
2017-02-03      2017-01-01    0.0
                2017-01-02    1.0
""", tsc)


def test_nr_gethistory(engine, tsh):
    s0 = pd.Series([-1, 0, 0, -1],
                   index=pd.DatetimeIndex(start=datetime(2016, 12, 29),
                                          end=datetime(2017, 1, 1),
                                          freq='D'))
    tsh.insert(engine, s0, 'foo', 'zogzog')

    s1 = pd.Series([1, 0, 0, 1],
                   index=pd.DatetimeIndex(start=datetime(2017, 1, 1),
                                          end=datetime(2017, 1, 4),
                                          freq='D'))
    idate = datetime(2016, 1, 1)
    for i in range(5):
        with engine.connect() as cn:
            with tsh.newchangeset(cn, 'aurelien.campeas@pythonian.f',
                                  _insertion_date=idate + timedelta(days=i)):
                tsh.insert(cn, s1 * i, 'foo')

    df = tsh.get_history(engine, 'foo',
                         datetime(2016, 1, 3),
                         datetime(2016, 1, 4),
                         datetime(2017, 1, 1),
                         datetime(2017, 1, 4))

    assert_df("""
insertion_date  value_date
2016-01-03      2017-01-01    2.0
                2017-01-02    0.0
                2017-01-03    0.0
                2017-01-04    2.0
2016-01-04      2017-01-01    3.0
                2017-01-02    0.0
                2017-01-03    0.0
                2017-01-04    3.0
""", df)


def test_add_na(engine, tsh):
    # a serie of NaNs won't be insert in base
    # in case of first insertion
    ts_nan = genserie(datetime(2010, 1, 1), 'D', 5)
    ts_nan[[True] * len(ts_nan)] = np.nan

    diff = tsh.insert(engine, ts_nan, 'ts_add_na', 'test')
    assert diff is None
    result = tsh.get(engine, 'ts_add_na')
    assert result is None

    # in case of insertion in existing data
    ts_begin = genserie(datetime(2010, 1, 1), 'D', 5)
    tsh.insert(engine, ts_begin, 'ts_add_na', 'test')

    ts_nan = genserie(datetime(2010, 1, 6), 'D', 5)
    ts_nan[[True] * len(ts_nan)] = np.nan
    ts_nan = pd.concat([ts_begin, ts_nan])

    diff = tsh.insert(engine, ts_nan, 'ts_add_na', 'test')
    assert diff is None

    result = tsh.get(engine, 'ts_add_na')
    assert len(result) == 5


def test_dtype_mismatch(engine, tsh):
    tsh.insert(engine,
               genserie(datetime(2015, 1, 1), 'D', 11).astype('str'),
               'error1',
               'test')

    with pytest.raises(Exception) as excinfo:
        tsh.insert(engine,
                   genserie(datetime(2015, 1, 1), 'D', 11),
                   'error1',
                   'test')

    assert 'Type error when inserting error1, new type is float64, type in base is object' == str(excinfo.value)

    tsh.insert(engine,
               genserie(datetime(2015, 1, 1), 'D', 11),
               'error2',
               'test')

    with pytest.raises(Exception) as excinfo:
        tsh.insert(engine,
                   genserie(datetime(2015, 1, 1), 'D', 11).astype('str'),
                   'error2',
                   'test')

    assert 'Type error when inserting error2, new type is object, type in base is float64' == str(excinfo.value)


def test_precision(engine, tsh):
    floaty = 0.123456789123456789
    ts = genserie(datetime(2015, 1, 1), 'D', 5, initval=[floaty])

    tsh.insert(engine, ts, 'precision', 'test')
    ts_round = tsh.get(engine, 'precision')
    assert 0.12345678912346 == ts_round.iloc[0]

    diff = tsh.insert(engine, ts_round, 'precision', 'test')
    assert diff is None  # the roundtriped series does not produce a diff when reinserted

    diff = tsh.insert(engine, ts, 'precision', 'test')  # neither does the original series
    assert diff is None


def test_get_from_to(engine, tsh):
    ts = genserie(datetime(2015, 1, 1), 'D', 365)
    tsh.insert(engine, ts, 'quitelong', 'aurelien.campeas@pythonian.fr')

    serie = tsh.get(engine, 'quitelong')
    assert serie.index[0] == pd.Timestamp('2015-01-01 00:00:00')
    assert serie.index[-1] == pd.Timestamp('2015-12-31 00:00:00')

    serie = tsh.get(engine, 'quitelong',
                    from_value_date=datetime(2015, 5, 1),
                    to_value_date=datetime(2015, 6, 1))
    assert serie.index[0] == pd.Timestamp('2015-05-01 00:00:00')
    assert serie.index[-1] == pd.Timestamp('2015-06-01 00:00:00')

    serie = tsh.get(engine, 'quitelong',
                    from_value_date=datetime(2016, 5, 1),
                    to_value_date=datetime(2016, 6, 1))
    assert serie.dtypes == np.dtype('float64')
    assert len(serie) == 0
    assert isinstance(serie.index, pd.DatetimeIndex)
    assert serie.index.freq is None


@pytest.mark.perf
def test_bigdata(engine, tracker, ptsh):
    tsh = ptsh

    def create_data():
        for year in range(2015, 2020):
            date = datetime(year, 1, 1)
            serie = genserie(date, '10Min', 6 * 24 * 365)
            with tsh.newchangeset(engine, 'aurelien.campeas@pythonian.fr',
                                  _insertion_date=date):
                tsh.insert(engine, serie, 'big')

    t0 = time()
    create_data()
    t1 = time() - t0
    tshclass = tsh.__class__.__name__

    with engine.connect() as cn:
        cn.execute('set search_path to "{}.timeserie"'.format(tsh.namespace))
        df = pd.read_sql('select id, diff, snapshot from big order by id', cn)

    for attr in ('diff', 'snapshot'):
        df[attr] = df[attr].apply(lambda x: 0 if x is None else len(x))

    size = df[['diff', 'snapshot']].sum().to_dict()
    tracker.append({'test': 'bigdata_insert',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': size['diff'],
                    'snapsize': size['snapshot']})

    t0 = time()
    tsh.get_history(engine, 'big')
    t1 = time() - t0
    tracker.append({'test': 'bigdata_history_all',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': None,
                    'snapsize': None})

    t0 = time()
    for year in (2015, 2017, 2019):
        for month in (1, 5, 9, 12):
            date = datetime(year, month, 1)
            tsh.get_history(engine, 'big',
                            from_insertion_date=date,
                            to_insertion_date=date + timedelta(days=31))
    t1 = time() - t0
    tracker.append({'test': 'bigdata_history_chunks',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': None,
                    'snapsize': None})


@pytest.mark.perf
def test_lots_of_diffs(engine, tracker, ptsh):
    tsh = ptsh

    def create_data():
        # one insert per day for 4 months
        for month in range(1, 4):
            days = calendar.monthrange(2017, month)[1]
            for day in range(1, days + 1):
                date = datetime(2017, month, day)
                serie = genserie(date, '10Min', 6 * 24)
                with engine.connect() as cn:
                    with tsh.newchangeset(cn, 'aurelien.campeas@pythonian.fr',
                                          _insertion_date=date.replace(year=2018)):
                        tsh.insert(cn, serie, 'manydiffs')

    t0 = time()
    create_data()
    t1 = time() - t0
    tshclass = tsh.__class__.__name__

    with engine.connect() as cn:
        cn.execute('set search_path to "{}.timeserie"'.format(tsh.namespace))
        df = pd.read_sql("select id, diff, snapshot from manydiffs order by id ",
                         cn)
    for attr in ('diff', 'snapshot'):
        df[attr] = df[attr].apply(lambda x: 0 if x is None else len(x))

    size = df[['diff', 'snapshot']].sum().to_dict()
    tracker.append({'test': 'manydiffs_insert',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': size['diff'],
                    'snapsize': size['snapshot']})

    t0 = time()
    tsh.get_history(engine, 'manydiffs')
    t1 = time() - t0
    tracker.append({'test': 'manydiffs_history_all',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': None,
                    'snapsize': None})

    t0 = time()
    for month in range(1, 3):
        for day in range(1, 5):
            date = datetime(2018, month, day)
            ts = tsh.get_history(engine, 'manydiffs',
                                 from_insertion_date=date,
                                 to_insertion_date=date + timedelta(days=31))
            assert ts is not None
    t1 = time() - t0
    tracker.append({'test': 'manydiffs_history_chunks',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': None,
                    'snapsize': None})

    t0 = time()
    for month in range(1, 3):
        for day in range(1, 5):
            date = datetime(2018, month, day)
            ts = tsh.get_history(engine, 'manydiffs',
                                 from_insertion_date=date,
                                 to_insertion_date=date + timedelta(days=31),
                                 from_value_date=date + timedelta(days=10),
                                 to_value_date=date + timedelta(days=20))
            assert ts is not None
    t1 = time() - t0
    tracker.append({'test': 'manydiffs_history_chunks_valuedate',
                    'class': tshclass,
                    'time': t1,
                    'diffsize': None,
                    'snapsize': None})
