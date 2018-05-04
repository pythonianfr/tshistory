from datetime import datetime, timedelta
from pathlib import Path
import pytz

from dateutil import parser
import pytest
import numpy as np
import pandas as pd

from tshistory.snapshot import Snapshot
from tshistory.testutil import (
    assert_df,
    assert_group_equals,
    genserie,
    tempattr
)

DATADIR = Path(__file__).parent / 'data'


def utcdt(*dt):
    return pd.Timestamp(datetime(*dt), tz='UTC')


def test_strip(engine, tsh):
    for i in range(1, 5):
        pubdate = utcdt(2017, 1, i)
        ts = genserie(datetime(2017, 1, 10), 'H', 1 + i)
        tsh.insert(engine, ts, 'xserie', 'babar', _insertion_date=pubdate)
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
insertion_date             value_date         
2017-01-01 00:00:00+00:00  2017-01-10 00:00:00    0.0
                           2017-01-10 01:00:00    1.0
2017-01-02 00:00:00+00:00  2017-01-10 00:00:00    0.0
                           2017-01-10 01:00:00    1.0
                           2017-01-10 02:00:00    2.0
2017-01-03 00:00:00+00:00  2017-01-10 00:00:00    0.0
                           2017-01-10 01:00:00    1.0
                           2017-01-10 02:00:00    2.0
                           2017-01-10 03:00:00    3.0
2017-01-04 00:00:00+00:00  2017-01-10 00:00:00    0.0
                           2017-01-10 01:00:00    1.0
                           2017-01-10 02:00:00    2.0
                           2017-01-10 03:00:00    3.0
                           2017-01-10 04:00:00    4.0
""", h)

    csid = tsh.changeset_at(engine, 'xserie', datetime(2017, 1, 3))
    with engine.connect() as cn:
        tsh.strip(cn, 'xserie', csid)

    assert_df("""
insertion_date             value_date         
2017-01-01 00:00:00+00:00  2017-01-10 00:00:00    0.0
                           2017-01-10 01:00:00    1.0
2017-01-02 00:00:00+00:00  2017-01-10 00:00:00    0.0
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
        df = pd.read_sql("select id, diff from xserie order by id", cn)
        df['diff'] = df['diff'].apply(lambda x: False if x is None else True)

    assert_df("""
id   diff
0   1  False
1   2   True
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


def test_chunks(engine, tsh):
    with tempattr(Snapshot, '_bucket_size', 2):
        ts = genserie(datetime(2010, 1, 1), 'D', 5)
        tsh.insert(engine, ts, 'chunks', 'test')

        # we expect 3 chunks
        sql = 'select parent, chunk from "{}.snapshot".chunks order by id'.format(
            tsh.namespace
        )
        chunks = engine.execute(sql).fetchall()
        assert len(chunks) == 3
        assert chunks[0].parent is None
        assert chunks[1].parent == 1
        assert chunks[2].parent == 2
        ts0 = tsh._deserialize(chunks[0].chunk, 'name')
        ts1 = tsh._deserialize(chunks[1].chunk, 'name')
        ts2 = tsh._deserialize(chunks[2].chunk, 'name')

        assert_df("""
2010-01-01    0.0
2010-01-02    1.0
""", ts0)

        assert_df("""
2010-01-03    2.0
2010-01-04    3.0
""", ts1)

        assert_df("""
2010-01-05    4.0
""", ts2)

        assert_df("""
2010-01-01    0.0
2010-01-02    1.0
2010-01-03    2.0
2010-01-04    3.0
2010-01-05    4.0
""", tsh.get(engine, 'chunks'))

        ts = pd.Series([4, 5, 6, 7, 8],
                       index=pd.date_range(start=datetime(2010, 1, 5),
                                           end=datetime(2010, 1, 9),
                                           freq='D'))

        tsh.insert(engine, ts, 'chunks', 'test')
        whole = tsh.get(engine, 'chunks')
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
""", whole)

        # we expect 6 chunks
        sql = 'select id, parent, chunk from "{}.snapshot".chunks order by id'.format(
            tsh.namespace
        )
        chunks = engine.execute(sql).fetchall()
        assert len(chunks) == 6
        assert chunks[4].parent == 4
        assert chunks[5].parent == 5
        assert {
            1: None,
            2: 1,
            3: 2, # head of first commit
            4: 2,
            5: 4,
            6: 5  # head of last commit
        } == {
            chunk.id: chunk.parent for chunk in chunks
        }

        ts3 = tsh._deserialize(chunks[3].chunk, 'name')
        ts4 = tsh._deserialize(chunks[4].chunk, 'name')
        ts5 = tsh._deserialize(chunks[5].chunk, 'name')

        assert_df("""
2010-01-05    4.0
2010-01-06    5.0
""", ts3)

        assert_df("""
2010-01-07    6.0
2010-01-08    7.0
""", ts4)

        assert_df("""
2010-01-09    8.0
""", ts5)

        # non-append edit
        whole[2] = 0
        whole[7] = 0

        tsh.insert(engine, whole, 'chunks', 'test')

        assert_df("""
2010-01-01    0.0
2010-01-02    1.0
2010-01-03    0.0
2010-01-04    3.0
2010-01-05    4.0
2010-01-06    5.0
2010-01-07    6.0
2010-01-08    0.0
2010-01-09    8.0
""", tsh.get(engine, 'chunks'))

        assert_df("""
2010-01-05    4.0
2010-01-06    5.0
2010-01-07    6.0
2010-01-08    0.0
2010-01-09    8.0
""", tsh.get(engine, 'chunks', from_value_date=datetime(2010, 1, 5)))

        # we expect 10 chunks
        # because we edit from the second chunk
        # and 4 new chunks have to be made
        sql = 'select id, parent, chunk from "{}.snapshot".chunks order by id'.format(
            tsh.namespace
        )
        chunks = engine.execute(sql).fetchall()
        assert len(chunks) == 10
        assert {
            1: None,
            2: 1,
            3: 2, # head of first commit
            4: 2,
            5: 4,
            6: 5, # head of second commit
            7: 1, # base of third commit (we lost many shared chunks)
            8: 7,
            9: 8,
            10: 9 # head of last commit
        } == {
            chunk.id: chunk.parent for chunk in chunks
        }

        # 2nd commit chunks without filtering
        snap = Snapshot(engine, tsh, 'chunks')
        chunks = {parent: len(ts) for parent, ts in snap.rawchunks(6)}
        assert chunks == {
            None: 2,
            1: 2,
            2: 2,
            4: 2,
            5: 1
        }
        # 2nd commit chunks with filtering
        chunks = {
            parent: len(ts)
            for parent, ts in snap.rawchunks(6, datetime(2010, 1, 5))
        }
        assert chunks == {2: 2, 4: 2, 5: 1}

        # 3rd commit chunks without filtering
        chunks = {parent: len(ts) for parent, ts in snap.rawchunks(10)}
        assert chunks == {
            None: 2,
            1: 2,
            7: 2,
            8: 2,
            9: 1
        }
        # 3rd commit chunks with filtering
        chunks = {
            parent: len(ts)
            for parent, ts in snap.rawchunks(10, datetime(2010, 1, 5))
        }
        assert chunks == {
            7: 2,
            8: 2,
            9: 1
        }


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
    # ival = Snapshot._interval
    # Snapshot._interval = 3

    for i in range(1, 5):
        with engine.connect() as cn:
            tsh.insert(cn, genserie(datetime(2017, 1, i), 'D', 3, [i]), 'revdate',
                       'test', _insertion_date=utcdt(2016, 1, i))

    # end of prologue, now some real meat
    idate0 = pd.Timestamp('2015-1-1 00:00:00', tz='UTC')
    ts = genserie(datetime(2010, 1, 4), 'D', 4, [0], name='truc')
    tsh.insert(engine, ts, 'ts_through_time',
               'test', _insertion_date=idate0)
    assert idate0 == tsh.latest_insertion_date(engine, 'ts_through_time')

    idate1 = pd.Timestamp('2015-1-1 15:45:23', tz='UTC')
    ts = genserie(datetime(2010, 1, 4), 'D', 4, [1], name='truc')
    tsh.insert(engine, ts, 'ts_through_time',
               'test', _insertion_date=idate1)
    assert idate1 == tsh.latest_insertion_date(engine, 'ts_through_time')

    idate2 = pd.Timestamp('2015-1-2 15:43:23', tz='UTC')
    ts = genserie(datetime(2010, 1, 4), 'D', 4, [2], name='truc')
    tsh.insert(engine, ts, 'ts_through_time',
               'test', _insertion_date=idate2)
    assert idate2 == tsh.latest_insertion_date(engine, 'ts_through_time')

    idate3 = pd.Timestamp('2015-1-3', tz='UTC')
    ts = genserie(datetime(2010, 1, 4), 'D', 4, [3], name='truc')
    tsh.insert(engine, ts, 'ts_through_time',
               'test', _insertion_date=idate3)
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

    # Snapshot._interval = ival


def _test_snapshots(engine, tsh):
    baseinterval = Snapshot._interval
    Snapshot._interval = 4

    with engine.connect() as cn:
        for tscount in range(1, 11):
            ts = genserie(datetime(2015, 1, 1), 'D', tscount, [1])
            diff = tsh.insert(cn, ts, 'growing', 'babar')
            assert diff.index[0] == diff.index[-1] == ts.index[-1]

    diff = tsh.insert(engine, ts, 'growing', 'babar')
    assert diff is None

    with engine.connect() as cn:
        cn.execute('set search_path to "{}.snapshot"'.format(tsh.namespace))
#         df = pd.read_sql("select cset from growing",
#                          cn)
#         assert_df("""
# cset
# 0     1
# 1     4
# 2     8
# 3    10
# """, df)

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

        df = pd.read_sql("select id, chunkhead from growing order by id", cn)
        df['chunkhead'] = df['chunkhead'].apply(lambda x: 0 if x is None else len(x))

        assert_df("""
id  chunkhead
0   1         35
1   4         47
2   8         59
3  10         67
""", df)

    # table = tsh._get_ts_table(engine, 'growing')
    _, snap = Snapshot(engine, tsh, 'growing').find()
    assert (ts == snap).all()
    Snapshot._interval = baseinterval


def test_deletion(engine, tsh):
    ts_begin = genserie(datetime(2010, 1, 1), 'D', 11)
    ts_begin.iloc[-1] = np.nan
    tsh.insert(engine, ts_begin, 'ts_del', 'test')

    ts = Snapshot(engine, tsh, 'ts_del').build_upto()
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
    assert 0 == len(tsh.diff(ts_repushed, ts_repushed))

    ts_add = genserie(datetime(2010, 1, 1), 'D', 15)
    ts_add.iloc[0] = np.nan
    ts_add.iloc[13:] = np.nan
    ts_add.iloc[8] = np.nan
    diff = tsh.diff(ts_repushed, ts_add)

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


def test_get_history(engine, tsh):
    for numserie in (1, 2, 3):
        with engine.connect() as cn:
            tsh.insert(cn, genserie(datetime(2017, 1, 1), 'D', numserie), 'smallserie',
                       'aurelien.campeas@pythonian.fr',
                       _insertion_date=utcdt(2017, 2, numserie))

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
         'date': pd.Timestamp('2017-02-01 00:00:00+0000', tz='UTC'),
         'names': ['smallserie']
        },
        {'author': 'aurelien.campeas@pythonian.fr',
         'meta': {},
         'date': pd.Timestamp('2017-02-02 00:00:00+0000', tz='UTC'),
         'names': ['smallserie']
        },
        {'author': 'aurelien.campeas@pythonian.fr',
         'meta': {},
         'date': pd.Timestamp('2017-02-03 00:00:00+0000', tz='UTC'),
         'names': ['smallserie']
        }
    ] == [{k: v for k, v in log.items() if k != 'rev'}
          for log in logs]
    histts = tsh.get_history(engine, 'smallserie')
    assert histts.name == 'smallserie'

    assert_df("""
insertion_date             value_date
2017-02-01 00:00:00+00:00  2017-01-01    0.0
2017-02-02 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
2017-02-03 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
                           2017-01-03    2.0
""", histts)

    diffs = tsh.get_history(engine, 'smallserie', diffmode=True)
    assert_df("""
insertion_date             value_date
2017-02-01 00:00:00+00:00  2017-01-01    0.0
2017-02-02 00:00:00+00:00  2017-01-02    1.0
2017-02-03 00:00:00+00:00  2017-01-03    2.0
""", diffs)

    for idate in histts.index.get_level_values('insertion_date').unique():
        with engine.connect() as cn:
            idate = idate.replace(tzinfo=pytz.timezone('UTC'))
            tsh.insert(cn, histts[idate], 'smallserie2',
                       'aurelien.campeas@pythonian.f', _insertion_date=idate)

    # this is perfectly round-tripable
    assert (tsh.get(engine, 'smallserie2') == ts).all()
    assert (tsh.get_history(engine, 'smallserie2') == histts).all()

    # get history ranges
    tsa = tsh.get_history(engine, 'smallserie',
                          from_insertion_date=datetime(2017, 2, 2))
    assert_df("""
insertion_date             value_date
2017-02-02 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
2017-02-03 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
                           2017-01-03    2.0
""", tsa)

    tsb = tsh.get_history(engine, 'smallserie',
                          to_insertion_date=datetime(2017, 2, 2))
    assert_df("""
insertion_date             value_date
2017-02-01 00:00:00+00:00  2017-01-01    0.0
2017-02-02 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
""", tsb)

    tsc = tsh.get_history(engine, 'smallserie',
                          from_insertion_date=datetime(2017, 2, 2),
                          to_insertion_date=datetime(2017, 2, 2))
    assert_df("""
insertion_date             value_date
2017-02-02 00:00:00+00:00  2017-01-01    0.0
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
insertion_date             value_date
2017-02-01 00:00:00+00:00  2017-01-01    0.0
2017-02-02 00:00:00+00:00  2017-01-01    0.0
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
insertion_date             value_date
2017-02-01 00:00:00+00:00  2017-01-01    0.0
2017-02-02 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
2017-02-03 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
""", tsc)

    diffs = tsh.get_history(engine, 'smallserie',
                            diffmode=True,
                            from_value_date=datetime(2017, 1, 1),
                            to_value_date=datetime(2017, 1, 2))
    assert_df("""
insertion_date             value_date
2017-02-01 00:00:00+00:00  2017-01-01    0.0
2017-02-02 00:00:00+00:00  2017-01-02    1.0
""", diffs)

    tsc = tsh.get_history(engine, 'smallserie',
                          from_value_date=datetime(2017, 1, 2))
    assert_df("""
insertion_date             value_date
2017-02-02 00:00:00+00:00  2017-01-02    1.0
2017-02-03 00:00:00+00:00  2017-01-02    1.0
                           2017-01-03    2.0
""", tsc)

    tsc = tsh.get_history(engine, 'smallserie',
                          to_value_date=datetime(2017, 1, 2))
    assert_df("""
insertion_date             value_date
2017-02-01 00:00:00+00:00  2017-01-01    0.0
2017-02-02 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
2017-02-03 00:00:00+00:00  2017-01-01    0.0
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
    idate = utcdt(2016, 1, 1)
    for i in range(5):
        with engine.connect() as cn:
            tsh.insert(cn, s1 * i, 'foo',
                       'aurelien.campeas@pythonian.f',
                       _insertion_date=idate + timedelta(days=i))

    df = tsh.get_history(engine, 'foo',
                         datetime(2016, 1, 3),
                         datetime(2016, 1, 4),
                         datetime(2017, 1, 1),
                         datetime(2017, 1, 4))

    assert_df("""
insertion_date             value_date
2016-01-03 00:00:00+00:00  2017-01-01    2.0
                           2017-01-02    0.0
                           2017-01-03    0.0
                           2017-01-04    2.0
2016-01-04 00:00:00+00:00  2017-01-01    3.0
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

    snap = Snapshot(engine, tsh, 'quitelong')
    if tsh.namespace == 'zzz':
        sql = 'select id, parent from "zzz.snapshot".quitelong order by id'
        chunks = engine.execute(sql).fetchall()
        # should be perfectly chained
        chunks = {
            chunk.id: chunk.parent for chunk in chunks
        }
        chunks.pop(1)
        assert all(k == v+1 for k, v in chunks.items())

        chunks = {parent: len(ts) for parent, ts in snap.rawchunks(73)}
        assert chunks == {None: 5, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5,
                          8: 5, 9: 5, 10: 5, 11: 5, 12: 5, 13: 5, 14: 5, 15: 5,
                          16: 5, 17: 5, 18: 5, 19: 5, 20: 5, 21: 5, 22: 5,
                          23: 5, 24: 5, 25: 5, 26: 5, 27: 5, 28: 5, 29: 5,
                          30: 5, 31: 5, 32: 5, 33: 5, 34: 5, 35: 5, 36: 5,
                          37: 5, 38: 5, 39: 5, 40: 5, 41: 5, 42: 5, 43: 5, 44: 5,
                          45: 5, 46: 5, 47: 5, 48: 5, 49: 5, 50: 5, 51: 5, 52: 5,
                          53: 5, 54: 5, 55: 5, 56: 5, 57: 5, 58: 5, 59: 5, 60: 5,
                          61: 5, 62: 5, 63: 5, 64: 5, 65: 5, 66: 5, 67: 5, 68: 5,
                          69: 5, 70: 5, 71: 5, 72: 5}
        chunks = {
            parent: len(ts)
            for parent, ts in snap.rawchunks(73, datetime(2015, 5, 1))
        }
        assert chunks == {24: 5, 25: 5, 26: 5, 27: 5, 28: 5, 29: 5, 30: 5, 31: 5,
                          32: 5, 33: 5, 34: 5, 35: 5, 36: 5, 37: 5, 38: 5, 39: 5,
                          40: 5, 41: 5, 42: 5, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5,
                          48: 5, 49: 5, 50: 5, 51: 5, 52: 5, 53: 5, 54: 5, 55: 5,
                          56: 5, 57: 5, 58: 5, 59: 5, 60: 5, 61: 5, 62: 5, 63: 5,
                          64: 5, 65: 5, 66: 5, 67: 5, 68: 5, 69: 5, 70: 5, 71: 5,
                          72: 5}

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
