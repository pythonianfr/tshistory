# coding: utf-8
from pathlib import Path
from datetime import datetime
from dateutil import parser

import pandas as pd
import numpy as np
from mock import patch
import pytest

from tshistory.tsio import TimeSerie

DATADIR = Path(__file__).parent / 'data'


def assert_group_equals(g1, g2):
    for (n1, s1), (n2, s2) in zip(sorted(g1.items()),
                                  sorted(g2.items())):
        assert n1 == n2
        assert s1.equals(s2)


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

    assert """
2017-01-01    2.0
2017-01-02    3.0
2017-01-03    1.0
""".strip() == tso.get(engine, 'ts_values').to_string().strip()

    assert """
2017-01-01    a
2017-01-02    b
2017-01-03    c
""".strip() == tso.get(engine, 'ts_othervalues').to_string().strip()

    assert tso.delete_last_changeset_for(engine, 'ts_values')

    assert """
2017-01-01    1.0
2017-01-02    2.0
2017-01-03    3.0
""".strip() == tso.get(engine, 'ts_values').to_string().strip()

    assert """
2017-01-01    a
2017-01-02    b
2017-01-03    c
""".strip() == tso.get(engine, 'ts_othervalues').to_string().strip()

    assert tso.delete_last_changeset_for(engine, 'ts_values')
    assert tso.get(engine, 'ts_values') is None
    assert tso.get(engine, 'ts_othervalues') is None

    assert not tso.delete_last_changeset_for(engine, 'ts_values')


def test_differential(engine):
    # instantiate one time serie handler object
    tso = TimeSerie()

    ts_begin = pd.Series(range(10))
    ts_begin.index = pd.date_range(start=datetime(2010, 1, 1), freq='D', periods=10)
    tso.insert(engine, ts_begin, 'ts_test', 'test')

    assert """
2010-01-01    0
2010-01-02    1
2010-01-03    2
2010-01-04    3
2010-01-05    4
2010-01-06    5
2010-01-07    6
2010-01-08    7
2010-01-09    8
2010-01-10    9
""".strip() == tso.get(engine, 'ts_test').to_string().strip()

    # we should detect the emission of a message
    tso.insert(engine, ts_begin, 'ts_test', 'babar')

    assert """
2010-01-01    0
2010-01-02    1
2010-01-03    2
2010-01-04    3
2010-01-05    4
2010-01-06    5
2010-01-07    6
2010-01-08    7
2010-01-09    8
2010-01-10    9
""".strip() == tso.get(engine, 'ts_test').to_string().strip()

    ts_slight_variation = ts_begin.copy()
    ts_slight_variation.iloc[3] = 0
    ts_slight_variation.iloc[6] = 0
    tso.insert(engine, ts_slight_variation, 'ts_test', 'celeste')

    assert """
2010-01-01    0
2010-01-02    1
2010-01-03    2
2010-01-04    0
2010-01-05    4
2010-01-06    5
2010-01-07    0
2010-01-08    7
2010-01-09    8
2010-01-10    9
""".strip() == tso.get(engine, 'ts_test').to_string().strip()

    ts_longer = pd.Series(range(15))
    ts_longer.index = pd.date_range(start=datetime(2010, 1, 3), freq='D', periods=15)
    ts_longer.iloc[1] = 2.48
    ts_longer.iloc[3] = 3.14
    ts_longer.iloc[5] = ts_begin.iloc[7]

    tso.insert(engine, ts_longer, 'ts_test', 'test')

    assert """
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
""".strip() == tso.get(engine, 'ts_test').to_string().strip()

    # start testing manual overrides
    ts_begin = pd.Series([2] * 5)
    ts_begin.index = pd.date_range(start=datetime(2010, 1, 1), freq='D', periods=5)
    ts_begin.loc['2010-01-04'] = -1
    tso.insert(engine, ts_begin, 'ts_mixte', 'test')

    # -1 represents bogus upstream data
    assert """
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
""".strip() == tso.get(engine, 'ts_mixte').to_string().strip()

    # refresh all the period + 1 extra data point
    ts_more = pd.Series([2] * 5)
    ts_more.index = pd.date_range(start=datetime(2010, 1, 2), freq='D', periods=5)
    ts_more.loc['2010-01-04'] = -1
    tso.insert(engine, ts_more, 'ts_mixte', 'test')

    assert """
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
2010-01-06    2.0
""".strip() == tso.get(engine, 'ts_mixte').to_string().strip()

    # just append an extra data point
    ts_one_more = pd.Series([3])  # with no intersection with the previous ts
    ts_one_more.index = pd.date_range(start=datetime(2010, 1, 7), freq='D', periods=1)
    tso.insert(engine, ts_one_more, 'ts_mixte', 'test')

    assert """
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
2010-01-06    2.0
2010-01-07    3.0
""".strip() == tso.get(engine, 'ts_mixte').to_string().strip()

    hist = pd.read_sql('select id, parent from timeserie.ts_test order by id',
                        engine)
    assert """
   id  parent
0   1     NaN
1   2     1.0
2   3     2.0
""".strip() == hist.to_string().strip()

    hist = pd.read_sql('select id, parent from timeserie.ts_mixte order by id',
                        engine)
    assert """
   id  parent
0   1     NaN
1   2     1.0
2   3     2.0
""".strip() == hist.to_string().strip()

    allts = pd.read_sql("select name, table_name from timeserie.registry "
                        "where name in ('ts_test', 'ts_mixte')",
                        engine)

    assert """
       name          table_name
0   ts_test   timeserie.ts_test
1  ts_mixte  timeserie.ts_mixte
""".strip() == allts.to_string().strip()

    assert """
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
2010-01-06    2.0
2010-01-07    3.0
""".strip() == tso.get(engine, 'ts_mixte',
                       revision_date=datetime.now()).to_string().strip()

    # test striping the last diff
    assert tso.delete_last_changeset_for(engine, 'ts_mixte')

    assert """
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
2010-01-06    2.0
""".strip() == tso.get(engine, 'ts_mixte').to_string().strip()


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
    ts = pd.Series([np.nan] * 10,
                   index=pd.date_range(start=datetime(2010, 1, 10),
                                       freq='D', periods=10), name='truc')
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
    assert """
2010-01-15    3.0
2010-01-16    3.0
2010-01-17    3.0
2010-01-18    3.0
2010-01-19    3.0
""".strip() == result.to_string().strip()

    # get_ts with name not in database

    tso.get(engine, 'inexisting_name', 'test')


def test_revision_date(engine):
    # instantiate one time serie handler object
    tso = TimeSerie()

    with patch('tshistory.tsio.datetime') as mock_date:
        mock_date.now.return_value = datetime(2015, 1, 1, 15, 43, 23)

        ts = pd.Series([1] * 4,
                       index=pd.date_range(start=datetime(2010, 1, 4),
                                           freq='D', periods=4), name='truc')
        tso.insert(engine, ts, 'ts_through_time', 'test')

    with patch('tshistory.tsio.datetime') as mock_date:
        mock_date.now.return_value = datetime(2015, 1, 2, 15, 43, 23)

        ts = pd.Series([2] * 4,
                       index=pd.date_range(start=datetime(2010, 1, 4),
                                           freq='D', periods=4), name='truc')
        tso.insert(engine, ts, 'ts_through_time', 'test')

    with patch('tshistory.tsio.datetime') as mock_date:
        mock_date.now.return_value = datetime(2015, 1, 3, 15, 43, 23)

        ts = pd.Series([3] * 4,
                       index=pd.date_range(start=datetime(2010, 1, 4),
                                           freq='D', periods=4), name='truc')
        tso.insert(engine, ts, 'ts_through_time', 'test')

    ts = tso.get(engine, 'ts_through_time')

    assert """
2010-01-04    3.0
2010-01-05    3.0
2010-01-06    3.0
2010-01-07    3.0
""".strip() == ts.to_string().strip()

    ts = tso.get(engine, 'ts_through_time',
                 revision_date=datetime(2015, 1, 2, 18, 43, 23) )

    assert """
2010-01-04    2.0
2010-01-05    2.0
2010-01-06    2.0
2010-01-07    2.0
""".strip() == ts.to_string().strip()

    ts = tso.get(engine, 'ts_through_time',
                 revision_date=datetime(2015, 1, 1, 18, 43, 23))

    assert """
2010-01-04    1.0
2010-01-05    1.0
2010-01-06    1.0
2010-01-07    1.0
""".strip() == ts.to_string().strip()

    ts = tso.get(engine, 'ts_through_time',
                 revision_date=datetime(2014, 1, 1, 18, 43, 23))

    assert ts is None


def test_snapshots(engine):
    tso = TimeSerie()
    tso._snapshot_interval = 5

    with engine.connect() as cnx:
        for tscount in range(11):
            ts = pd.Series([1] * tscount,
                           index=pd.date_range(datetime(2015, 1, 1),
                                               freq='D', periods=tscount))
            tso.insert(cnx, ts, 'growing', 'babar')

    df = pd.read_sql("select id from timeserie.growing where snapshot is not null",
                     engine)
    assert """
   id
0   1
1   5
2  10
""".strip() == df.to_string().strip()

    ts = tso.get(engine, 'growing')
    assert """
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
""".strip() == ts.to_string().strip()

    df = pd.read_sql("select id, diff, snapshot from timeserie.growing", engine)
    for attr in ('diff', 'snapshot'):
        df[attr] = df[attr].apply(lambda x: 0 if x is None else len(x))

    assert """
   id  diff  snapshot
0   1     0        68
1   2    68         0
2   3    68         0
3   5    68       187
4   4    68         0
5   6    68         0
6   7    68         0
7   8    68         0
8  10    68       342
9   9    68         0
""".strip() == df.to_string().strip()
