# coding: utf-8
from pathlib import Path
from datetime import datetime
from dateutil import parser

import pandas as pd
import numpy as np
from mock import patch

from tshistory.tsio import insert_ts, get_ts, delete_last_diff

DATADIR = Path(__file__).parent / 'data'


def test_differential(engine):

    ts_begin = pd.Series(range(10))
    ts_begin.index = pd.date_range(start=datetime(2010, 1, 1), freq='D', periods=10)
    insert_ts(engine, ts_begin, 'ts_test', 'test')

    assert """
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
""".strip() == get_ts(engine, 'ts_test').to_string().strip()

    # we should detect the emission of a message
    insert_ts(engine, ts_begin, 'ts_test', 'babar')

    assert """
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
""".strip() == get_ts(engine, 'ts_test').to_string().strip()

    ts_slight_variation = ts_begin.copy()
    ts_slight_variation.iloc[3] = 0
    ts_slight_variation.iloc[6] = 0
    insert_ts(engine, ts_slight_variation, 'ts_test', 'celeste')

    assert """
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
""".strip() == get_ts(engine, 'ts_test').to_string().strip()

    ts_longer = pd.Series(range(15))
    ts_longer.index = pd.date_range(start=datetime(2010, 1, 3), freq='D', periods=15)
    ts_longer.iloc[1] = 2.48
    ts_longer.iloc[3] = 3.14
    ts_longer.iloc[5] = ts_begin.iloc[7]

    insert_ts(engine, ts_longer, 'ts_test', 'test')

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
""".strip() == get_ts(engine, 'ts_test').to_string().strip()

    # start testing manual overrides
    ts_begin = pd.Series([2] * 5)
    ts_begin.index = pd.date_range(start=datetime(2010, 1, 1), freq='D', periods=5)
    ts_begin.loc['2010-01-04'] = -1
    insert_ts(engine, ts_begin, 'ts_mixte', 'test')

    # -1 represents bogus upstream data
    assert """
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
""".strip() == get_ts(engine, 'ts_mixte').to_string().strip()

    # refresh all the period + 1 extra data point
    ts_more = pd.Series([2] * 5)
    ts_more.index = pd.date_range(start=datetime(2010, 1, 2), freq='D', periods=5)
    ts_more.loc['2010-01-04'] = -1
    insert_ts(engine, ts_more, 'ts_mixte', 'test')

    assert """
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
2010-01-06    2.0
""".strip() == get_ts(engine, 'ts_mixte').to_string().strip()

    # just append an extra data point
    ts_one_more = pd.Series([3])  # with no intersection with the previous ts
    ts_one_more.index = pd.date_range(start=datetime(2010, 1, 7), freq='D', periods=1)
    insert_ts(engine, ts_one_more, 'ts_mixte', 'test')

    assert """
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
2010-01-06    2.0
2010-01-07    3.0
""".strip() == get_ts(engine, 'ts_mixte').to_string().strip()

    hist = pd.read_sql('select id, parent from ts_ts_test order by id',
                        engine)
    assert """
   id  parent
0   1     NaN
1   2     1.0
2   3     2.0
""".strip() == hist.to_string().strip()

    hist = pd.read_sql('select id, parent from ts_ts_mixte order by id',
                        engine)
    assert """
   id  parent
0   1     NaN
1   2     1.0
2   3     2.0
""".strip() == hist.to_string().strip()

    allts = pd.read_sql('select id, name, table_name from ts_registry',
                        engine)

    assert """
   id      name   table_name
0   1   ts_test   ts_ts_test
1   2  ts_mixte  ts_ts_mixte
""".strip() == allts.to_string().strip()

    assert """
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
2010-01-06    2.0
2010-01-07    3.0
""".strip() == get_ts(engine, 'ts_mixte',
                      revision_date=datetime.now()).to_string().strip()

    # test striping the last diff
    delete_last_diff(engine, 'ts_mixte')

    assert """
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
2010-01-06    2.0
""".strip() == get_ts(engine, 'ts_mixte').to_string().strip()


def test_bad_import(engine):
    # the data were parsed as date by pd.read_json()
    df_result = pd.read_csv(DATADIR / 'test_data.csv')
    df_result['Gas Day'] = df_result['Gas Day'].apply(parser.parse, dayfirst=True, yearfirst=False)
    df_result.set_index('Gas Day', inplace=True)
    ts = df_result['SC']
    insert_ts(engine, ts, 'SND_SC', 'test')
    result = get_ts(engine, 'SND_SC')
    assert result.dtype == 'float64'

    # insertion of empty ts
    ts = pd.Series(name='truc', dtype='object')
    insert_ts(engine, ts, 'empty_ts', 'test')
    assert get_ts(engine, 'empty_ts') is None

    # nan in ts
    # all na
    ts = pd.Series([np.nan] * 10,
                   index=pd.date_range(start=datetime(2010, 1, 10),
                                       freq='D', periods=10), name='truc')
    insert_ts(engine, ts, 'test_nan', 'test')
    assert get_ts(engine, 'test_nan') is None

    # mixe na
    ts = pd.Series([np.nan] * 5 + [3] * 5,
                   index=pd.date_range(start=datetime(2010, 1, 10),
                                       freq='D', periods=10), name='truc')
    insert_ts(engine, ts, 'test_nan', 'test')
    result = get_ts(engine, 'test_nan')

    insert_ts(engine, ts, 'test_nan', 'test')
    result = get_ts(engine, 'test_nan')
    assert """
2010-01-15    3.0
2010-01-16    3.0
2010-01-17    3.0
2010-01-18    3.0
2010-01-19    3.0
""".strip() == result.to_string().strip()

    # get_ts with name not in database

    get_ts(engine, 'inexisting_name', 'test')


def test_revision_date(engine):
    with patch('tshistory.tsio.datetime') as mock_date:
        mock_date.now.return_value = datetime(2015, 1, 1, 15, 43, 23)

        ts = pd.Series([1] * 4,
                       index=pd.date_range(start=datetime(2010, 1, 4),
                                           freq='D', periods=4), name='truc')
        insert_ts(engine, ts, 'ts_through_time', 'test')

    with patch('tshistory.tsio.datetime') as mock_date:
        mock_date.now.return_value = datetime(2015, 1, 2, 15, 43, 23)

        ts = pd.Series([2] * 4,
                       index=pd.date_range(start=datetime(2010, 1, 4),
                                           freq='D', periods=4), name='truc')
        insert_ts(engine, ts, 'ts_through_time', 'test')

    with patch('tshistory.tsio.datetime') as mock_date:
        mock_date.now.return_value = datetime(2015, 1, 3, 15, 43, 23)

        ts = pd.Series([3] * 4,
                       index=pd.date_range(start=datetime(2010, 1, 4),
                                           freq='D', periods=4), name='truc')
        insert_ts(engine, ts, 'ts_through_time', 'test')

    ts = get_ts(engine, 'ts_through_time')

    assert """
2010-01-04    3.0
2010-01-05    3.0
2010-01-06    3.0
2010-01-07    3.0
""".strip() == ts.to_string().strip()

    ts = get_ts(engine, 'ts_through_time',
                revision_date=datetime(2015, 1, 2, 18, 43, 23) )

    assert """
2010-01-04    2.0
2010-01-05    2.0
2010-01-06    2.0
2010-01-07    2.0
""".strip() == ts.to_string().strip()

    ts = get_ts(engine, 'ts_through_time',
                revision_date=datetime(2015, 1, 1, 18, 43, 23))

    assert """
2010-01-04    1.0
2010-01-05    1.0
2010-01-06    1.0
2010-01-07    1.0
""".strip() == ts.to_string().strip()

    ts = get_ts(engine, 'ts_through_time',
                revision_date=datetime(2014, 1, 1, 18, 43, 23))

    assert ts is None

