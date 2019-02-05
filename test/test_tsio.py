from datetime import datetime, timedelta
from pathlib import Path
import pytz

from dateutil import parser
import pytest
import numpy as np
import pandas as pd

from tshistory.snapshot import Snapshot
from tshistory.util import threadpool
from tshistory.tsio import TimeSerie
from tshistory.testutil import (
    assert_df,
    assert_hist,
    assert_hist_equals,
    assert_group_equals,
    assert_structures,
    genserie,
    tempattr
)

DATADIR = Path(__file__).parent / 'data'


def utcdt(*dt):
    return pd.Timestamp(datetime(*dt), tz='UTC')


def test_in_tx(engine):
    tsh = TimeSerie()

    assert tsh.type(engine, 'foo') == 'primary'

    with pytest.raises(TypeError) as err:
        tsh.insert(engine, 0, 0, 0)
    assert err.value.args[0] == 'You must use a transaction object'

    with engine.connect() as cn:
        with pytest.raises(TypeError) as err:
            tsh.insert(cn, 0, 0, 0)
    assert err.value.args[0] == 'You must use a transaction object'

    ts = genserie(datetime(2017, 10, 28, 23),
                  'H', 4, tz='UTC')
    with engine.begin() as cn:
        tsh.insert(cn, ts, 'test_tx', 'Babar')


def test_tstamp_roundtrip(engine, tsh):
    assert_structures(engine, tsh)
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

    tsh.insert(engine, ts, 'tztest', 'Babar',
               _insertion_date=utcdt(2018, 1, 1))
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

    ival = tsh.interval(engine, 'tztest')
    assert ival.left == pd.Timestamp('2017-10-28 23:00:00+0000', tz='UTC')
    assert ival.right == pd.Timestamp('2017-10-29 02:00:00+0000', tz='UTC')

    ts = genserie(datetime(2017, 10, 29, 1),
                  'H', 4, tz='UTC')
    ts.index = ts.index.tz_convert('Europe/Paris')
    tsh.insert(engine, ts, 'tztest', 'Celeste',
                   _insertion_date=utcdt(2018, 1, 3))

    ts = tsh.get(engine, 'tztest')
    assert_df("""
2017-10-28 23:00:00+00:00    0.0
2017-10-29 00:00:00+00:00    1.0
2017-10-29 01:00:00+00:00    0.0
2017-10-29 02:00:00+00:00    1.0
2017-10-29 03:00:00+00:00    2.0
2017-10-29 04:00:00+00:00    3.0
""", ts)

    hist = tsh.get_history(engine, 'tztest')
    assert_hist("""
insertion_date             value_date               
2018-01-01 00:00:00+00:00  2017-10-28 23:00:00+00:00    0.0
                           2017-10-29 00:00:00+00:00    1.0
                           2017-10-29 01:00:00+00:00    2.0
                           2017-10-29 02:00:00+00:00    3.0
2018-01-03 00:00:00+00:00  2017-10-28 23:00:00+00:00    0.0
                           2017-10-29 00:00:00+00:00    1.0
                           2017-10-29 01:00:00+00:00    0.0
                           2017-10-29 02:00:00+00:00    1.0
                           2017-10-29 03:00:00+00:00    2.0
                           2017-10-29 04:00:00+00:00    3.0
""", hist)

    hist = tsh.get_history(engine, 'tztest',
                           from_value_date=utcdt(2017, 10, 29, 1),
                           to_value_date=utcdt(2017, 10, 29, 3))
    assert_hist("""
insertion_date             value_date               
2018-01-01 00:00:00+00:00  2017-10-29 01:00:00+00:00    2.0
                           2017-10-29 02:00:00+00:00    3.0
2018-01-03 00:00:00+00:00  2017-10-29 01:00:00+00:00    0.0
                           2017-10-29 02:00:00+00:00    1.0
                           2017-10-29 03:00:00+00:00    2.0
""", hist)

    ival = tsh.interval(engine, 'tztest')
    assert ival.left == pd.Timestamp('2017-10-28 23:00:00+0000', tz='UTC')
    assert ival.right == pd.Timestamp('2017-10-29 04:00:00+0000', tz='UTC')
    assert_structures(engine, tsh)


def test_differential(engine, tsh):
    assert_structures(engine, tsh)
    ts_begin = genserie(datetime(2010, 1, 1), 'D', 10)
    tsh.insert(engine, ts_begin, 'ts_test', 'test')

    id1 = tsh.last_id(engine, 'ts_test')
    assert tsh._previous_cset(engine, 'ts_test', id1) is None

    assert tsh.exists(engine, 'ts_test')
    assert not tsh.exists(engine, 'this_does_not_exist')

    assert tsh.interval(engine, 'ts_test') == pd.Interval(
        datetime(2010, 1, 1, 0, 0), datetime(2010, 1, 10, 0, 0),
        closed='both'
    )

    with pytest.raises(ValueError):
        assert tsh.interval(engine, 'nosuchts')

    fetched = tsh.get(engine, 'ts_test')
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
""", fetched)
    assert fetched.name == 'ts_test'

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
    id2 = tsh.last_id(engine, 'ts_test')
    assert tsh._previous_cset(engine, 'ts_test', id2) == id1

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

    with engine.begin() as cn:
        tsh.insert(cn, ts_longer, 'ts_test', 'test')
    id3 = tsh.last_id(engine, 'ts_test')

    assert id1 < id2 < id3

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

    assert tsh.interval(engine, 'ts_test') == pd.Interval(
        datetime(2010, 1, 1, 0, 0), datetime(2010, 1, 17, 0, 0),
        closed='both'
    )

    # insert single data, in override of previous one
    ts_begin = genserie(datetime(2010, 1, 1), 'D', 5, initval=[2])
    ts_begin.loc['2010-01-04'] = -1
    tsh.insert(engine, ts_begin, 'ts_mixte', 'test')

    assert_df("""
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
""", tsh.get(engine, 'ts_mixte'))

    # add new series with one additional values
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

    assert_df("""
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
2010-01-06    2.0
    """, tsh.get(engine, 'ts_mixte',
                 from_value_date=datetime(2010, 1, 3),
                 to_value_date=datetime(2010, 1, 6)
    ))

    assert_df("""
2010-01-04   -1.0
2010-01-05    2.0
2010-01-06    2.0
2010-01-07    3.0
        """, tsh.get(engine, 'ts_mixte',
                     from_value_date=datetime(2010, 1, 4)
    ))

    assert_df("""
2010-01-01    2.0
2010-01-02    2.0
2010-01-03    2.0
2010-01-04   -1.0
2010-01-05    2.0
            """, tsh.get(engine, 'ts_mixte',
                         to_value_date=datetime(2010, 1, 5)
        ))

    with engine.begin() as cn:
        cn.execute('set search_path to "{0}.timeserie", {0}, public'.format(tsh.namespace))
        allts = pd.read_sql("select seriename, table_name from registry "
                            "where seriename in ('ts_test', 'ts_mixte')",
                            cn)

        assert_df("""
seriename table_name
0   ts_test    ts_test
1  ts_mixte   ts_mixte
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

    assert_structures(engine, tsh)


def test_serie_metadata(engine, tsh):
    serie = genserie(datetime(2010, 1, 1), 'D', 1, initval=[1])
    tsh.insert(engine, serie, 'ts-metadata', 'babar')

    initialmeta = tsh.metadata(engine, 'ts-metadata')
    assert initialmeta == {
        'index_dtype': '<M8[ns]',
        'index_names': [],
        'index_type': 'datetime64[ns]',
        'tzaware': False,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }

    tsh.update_metadata(engine, 'ts-metadata',
                        {'topic': 'banana spot price'}
    )
    assert tsh.metadata(engine, 'ts-metadata')['topic'] == 'banana spot price'

    with pytest.raises(AssertionError):
        tsh.update_metadata(engine, 'ts-metadata', {'tzaware': True})

    assert tsh.metadata(engine, 'ts-metadata') == {
        'index_dtype': '<M8[ns]',
        'index_names': [],
        'index_type': 'datetime64[ns]',
        'topic': 'banana spot price',
        'tzaware': False,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }


def test_changeset_metadata(engine, tsh):
    serie = genserie(datetime(2010, 1, 1), 'D', 1, initval=[1])
    tsh.insert(engine, serie, 'ts-cs-metadata', 'babar',
               {'foo': 'A', 'bar': 42})

    log = tsh.log(engine, names=['ts-cs-metadata'])
    meta = tsh.changeset_metadata(engine, log[0]['rev'])
    assert meta == {'foo': 'A', 'bar': 42}


def test_revision_date(engine, tsh):
    for i in range(1, 5):
        with engine.begin() as cn:
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


def test_point_deletion(engine, tsh):
    ts_begin = genserie(datetime(2010, 1, 1), 'D', 11)
    ts_begin.iloc[-1] = np.nan
    tsh.insert(engine, ts_begin, 'ts_del', 'test')

    _, ts = Snapshot(engine, tsh, 'ts_del').find()
    assert ts.iloc[-2] == 8.0

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
    with pytest.raises(ValueError):
        tsh.insert(engine, ts_string, 'ts_string_del', 'test')


def test_nan_first(engine, tsh):
    # first insertion with only nan
    ts_begin = genserie(datetime(2010, 1, 1), 'D', 10, [np.nan])
    assert tsh.insert(engine, ts_begin, 'ts_null', 'test') is None


def test_more_point_deletion(engine, tsh):
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
    with pytest.raises(ValueError):
        tsh.insert(engine, ts_begin, 'ts_full_del', 'test')

    ts_end = genserie(datetime(2010, 1, 1), 'D', 4)
    tsh.insert(engine, ts_end, 'ts_full_del', 'test')

    # string

    ts_begin = genserie(datetime(2010, 1, 1), 'D', 4, ['text'])
    tsh.insert(engine, ts_begin, 'ts_full_del_str', 'test')

    ts_begin = pd.Series([np.nan] * 4, name='ts_full_del_str',
                         index=ts_begin.index)

    with pytest.raises(ValueError):
        tsh.insert(engine, ts_begin, 'ts_full_del_str', 'test')

    ts_end = genserie(datetime(2010, 1, 1), 'D', 4, ['text'])
    tsh.insert(engine, ts_end, 'ts_full_del_str', 'test')


def test_deletion_over_horizon(engine, tsh):
    idate = utcdt(2018, 2, 1)
    ts = pd.Series(
        [1, 2, 3],
        index=pd.date_range(datetime(2018, 1, 1), freq='D', periods=3)
    )

    name = 'delete_over_hz'
    tsh.insert(engine, ts, name, 'Babar',
               _insertion_date=idate)

    ts = pd.Series(
        [np.nan, np.nan, np.nan],
        index=pd.date_range(datetime(2018, 1, 3), freq='D', periods=3)
    )

    tsh.insert(engine, ts, name, 'Celeste',
               _insertion_date=idate.replace(day=2))
    ival = tsh.interval(engine, name)
    assert ival.left == datetime(2018, 1, 1)
    assert ival.right == datetime(2018, 1, 2)

    ts = pd.Series(
        [np.nan, np.nan, np.nan],
        index=pd.date_range(datetime(2017, 12, 30), freq='D', periods=3)
    )
    tsh.insert(engine, ts, name, 'Arthur',
               _insertion_date=idate.replace(day=3))
    ival = tsh.interval(engine, name)
    assert ival.left == datetime(2018, 1, 2)
    assert ival.right == datetime(2018, 1, 2)


def test_get_history(engine, tsh):
    for numserie in (1, 2, 3):
        with engine.begin() as cn:
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

    assert_hist("""
insertion_date             value_date
2017-02-01 00:00:00+00:00  2017-01-01    0.0
2017-02-02 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
2017-02-03 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
                           2017-01-03    2.0
""", histts)

    diffs = tsh.get_history(engine, 'smallserie', diffmode=True)
    assert_hist("""
insertion_date             value_date
2017-02-01 00:00:00+00:00  2017-01-01    0.0
2017-02-02 00:00:00+00:00  2017-01-02    1.0
2017-02-03 00:00:00+00:00  2017-01-03    2.0
""", diffs)

    for idate in histts:
        with engine.begin() as cn:
            idate = idate.replace(tzinfo=pytz.timezone('UTC'))
            tsh.insert(cn, histts[idate], 'smallserie2',
                       'aurelien.campeas@pythonian.f', _insertion_date=idate)

    # this is perfectly round-tripable
    assert (tsh.get(engine, 'smallserie2') == ts).all()
    assert_hist_equals(tsh.get_history(engine, 'smallserie2'), histts)

    # get history ranges
    tsa = tsh.get_history(engine, 'smallserie',
                          from_insertion_date=datetime(2017, 2, 2))
    assert_hist("""
insertion_date             value_date
2017-02-02 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
2017-02-03 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
                           2017-01-03    2.0
""", tsa)

    tsb = tsh.get_history(engine, 'smallserie',
                          to_insertion_date=datetime(2017, 2, 2))
    assert_hist("""
insertion_date             value_date
2017-02-01 00:00:00+00:00  2017-01-01    0.0
2017-02-02 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
""", tsb)

    tsc = tsh.get_history(engine, 'smallserie',
                          from_insertion_date=datetime(2017, 2, 2),
                          to_insertion_date=datetime(2017, 2, 2))
    assert_hist("""
insertion_date             value_date
2017-02-02 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
""", tsc)

    tsc = tsh.get_history(engine, 'smallserie',
                          from_insertion_date=datetime(2017, 2, 4),
                          to_insertion_date=datetime(2017, 2, 4))
    assert tsc == {}

    tsc = tsh.get_history(engine, 'smallserie',
                          from_insertion_date=datetime(2016, 2, 1),
                          to_insertion_date=datetime(2017, 2, 2))
    assert_hist("""
insertion_date             value_date
2017-02-01 00:00:00+00:00  2017-01-01    0.0
2017-02-02 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
""", tsc)

    tsc = tsh.get_history(engine, 'smallserie',
                          from_insertion_date=datetime(2016, 2, 1),
                          to_insertion_date=datetime(2016, 12, 31))
    assert tsc == {}

    # restrictions on value dates
    tsc = tsh.get_history(engine, 'smallserie',
                          from_value_date=datetime(2017, 1, 1),
                          to_value_date=datetime(2017, 1, 2))
    assert_hist("""
insertion_date             value_date
2017-02-01 00:00:00+00:00  2017-01-01    0.0
2017-02-02 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
2017-02-03 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
""", tsc)

    tsc = tsh.get_history(engine, 'smallserie',
                          from_value_date=datetime(2017, 1, 2))
    assert_hist("""
insertion_date             value_date
2017-02-02 00:00:00+00:00  2017-01-02    1.0
2017-02-03 00:00:00+00:00  2017-01-02    1.0
                           2017-01-03    2.0
""", tsc)

    tsc = tsh.get_history(engine, 'smallserie',
                          to_value_date=datetime(2017, 1, 2))
    assert_hist("""
insertion_date             value_date
2017-02-01 00:00:00+00:00  2017-01-01    0.0
2017-02-02 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
2017-02-03 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
""", tsc)

    tsc = tsh.get_history(engine, 'no-such-series')
    assert tsc is None

    idates = tsh.insertion_dates(engine, 'smallserie')
    assert idates == [
        pd.Timestamp('2017-02-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2017-02-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2017-02-03 00:00:00+0000', tz='UTC')
    ]
    idates = tsh.insertion_dates(
        engine, 'smallserie',
        fromdate=utcdt(2017, 2, 2),
        todate=utcdt(2017, 2, 2)
    )
    assert idates == [
        pd.Timestamp('2017-02-02 00:00:00+0000', tz='UTC'),
    ]



def test_history_delta(engine, tsh):
    for d in range(1, 3):
        idate = utcdt(2018, 1, d)
        serie = genserie(idate - timedelta(hours=1), 'H', 6, initval=[d])
        tsh.insert(engine, serie, 'hd', 'aurelien.campeas@pythonian.fr',
                   _insertion_date=idate)

    assert_hist("""
insertion_date             value_date               
2018-01-01 00:00:00+00:00  2017-12-31 23:00:00+00:00    1.0
                           2018-01-01 00:00:00+00:00    1.0
                           2018-01-01 01:00:00+00:00    1.0
                           2018-01-01 02:00:00+00:00    1.0
                           2018-01-01 03:00:00+00:00    1.0
                           2018-01-01 04:00:00+00:00    1.0
2018-01-02 00:00:00+00:00  2017-12-31 23:00:00+00:00    1.0
                           2018-01-01 00:00:00+00:00    1.0
                           2018-01-01 01:00:00+00:00    1.0
                           2018-01-01 02:00:00+00:00    1.0
                           2018-01-01 03:00:00+00:00    1.0
                           2018-01-01 04:00:00+00:00    1.0
                           2018-01-01 23:00:00+00:00    2.0
                           2018-01-02 00:00:00+00:00    2.0
                           2018-01-02 01:00:00+00:00    2.0
                           2018-01-02 02:00:00+00:00    2.0
                           2018-01-02 03:00:00+00:00    2.0
                           2018-01-02 04:00:00+00:00    2.0
    """, tsh.get_history(engine, 'hd'))

    assert_hist("""
insertion_date             value_date               
2018-01-01 00:00:00+00:00  2017-12-31 23:00:00+00:00    1.0
                           2018-01-01 00:00:00+00:00    1.0
                           2018-01-01 01:00:00+00:00    1.0
                           2018-01-01 02:00:00+00:00    1.0
2018-01-02 00:00:00+00:00  2017-12-31 23:00:00+00:00    1.0
                           2018-01-01 00:00:00+00:00    1.0
                           2018-01-01 01:00:00+00:00    1.0
                           2018-01-01 02:00:00+00:00    1.0
                           2018-01-01 03:00:00+00:00    1.0
                           2018-01-01 04:00:00+00:00    1.0
                           2018-01-01 23:00:00+00:00    2.0
                           2018-01-02 00:00:00+00:00    2.0
                           2018-01-02 01:00:00+00:00    2.0
                           2018-01-02 02:00:00+00:00    2.0
""",  tsh.get_history(engine, 'hd', deltaafter=timedelta(hours=2)))

    assert_hist("""
insertion_date             value_date               
2018-01-01 00:00:00+00:00  2018-01-01 00:00:00+00:00    1.0
                           2018-01-01 01:00:00+00:00    1.0
2018-01-02 00:00:00+00:00  2018-01-02 00:00:00+00:00    2.0
                           2018-01-02 01:00:00+00:00    2.0
""",  tsh.get_history(engine, 'hd',
                      deltabefore=timedelta(hours=0),
                      deltaafter=timedelta(hours=1)))


def test_delta_na(engine, tsh):
    ldates = (
        utcdt(2015, 1, 20),
        utcdt(2015, 1, 21),
        utcdt(2015, 1, 22)
    )

    for idx, idate in enumerate(ldates):
        ts = pd.Series([idx] * 3, index = ldates)
        tsh.insert(engine, ts, 'without_na', 'arnaud',
                   _insertion_date=idate)

    assert_df("""
2015-01-20 00:00:00+00:00    2.0
2015-01-21 00:00:00+00:00    2.0
2015-01-22 00:00:00+00:00    2.0
""", tsh.get(engine, 'without_na'))

    assert_df("""
2015-01-20 00:00:00+00:00    0.0
2015-01-21 00:00:00+00:00    1.0
2015-01-22 00:00:00+00:00    2.0
    """, tsh.get_delta(engine, 'without_na', delta=timedelta(hours=0)))
    #as expected

    assert_hist("""
insertion_date             value_date               
2015-01-20 00:00:00+00:00  2015-01-20 00:00:00+00:00    0.0
                           2015-01-21 00:00:00+00:00    0.0
                           2015-01-22 00:00:00+00:00    0.0
2015-01-21 00:00:00+00:00  2015-01-20 00:00:00+00:00    1.0
                           2015-01-21 00:00:00+00:00    1.0
                           2015-01-22 00:00:00+00:00    1.0
2015-01-22 00:00:00+00:00  2015-01-20 00:00:00+00:00    2.0
                           2015-01-21 00:00:00+00:00    2.0
                           2015-01-22 00:00:00+00:00    2.0
    """, tsh.get_history(engine, 'without_na'))

    # now, the last insertion has Na as last value
    ldates = (
        utcdt(2015, 1, 20),
        utcdt(2015, 1, 21),
        utcdt(2015, 1, 22)
    )

    for idx, idate in enumerate(ldates):
        serie = pd.Series([float(idx)] * 3, index = ldates)
        if idx == 2:
            serie[-1] = np.nan
        tsh.insert(engine, serie, 'with_na', 'arnaud',
                   _insertion_date=idate)

    # the value at 2015-01-22 is hidden by the inserted nan
    assert_df("""
2015-01-20 00:00:00+00:00    2.0
2015-01-21 00:00:00+00:00    2.0
    """, tsh.get(engine, 'with_na'))

    # the last value is also correctly hidden
    assert_df("""
2015-01-20 00:00:00+00:00    0.0
2015-01-21 00:00:00+00:00    1.0
    """, tsh.get_delta(engine, 'with_na', delta=timedelta(hours=0)))

    # the value gathered by get_delta at value date 2015-01-22 is a
    # nan, so it masks the previous ones at the same date
    assert_hist("""
insertion_date             value_date               
2015-01-20 00:00:00+00:00  2015-01-20 00:00:00+00:00    0.0
                           2015-01-21 00:00:00+00:00    0.0
                           2015-01-22 00:00:00+00:00    0.0
2015-01-21 00:00:00+00:00  2015-01-20 00:00:00+00:00    1.0
                           2015-01-21 00:00:00+00:00    1.0
                           2015-01-22 00:00:00+00:00    1.0
2015-01-22 00:00:00+00:00  2015-01-20 00:00:00+00:00    2.0
                           2015-01-21 00:00:00+00:00    2.0
                           2015-01-22 00:00:00+00:00    NaN
        """, tsh.get_history(engine, 'with_na', _keep_nans=True))


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
        with engine.begin() as cn:
            tsh.insert(cn, s1 * i, 'foo',
                       'aurelien.campeas@pythonian.f',
                       _insertion_date=idate + timedelta(days=i))

    df = tsh.get_history(engine, 'foo',
                         datetime(2016, 1, 3),
                         datetime(2016, 1, 4),
                         datetime(2017, 1, 1),
                         datetime(2017, 1, 4))

    assert_hist("""
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
    assert 0.12345678912345678 == ts_round.iloc[0]

    diff = tsh.insert(engine, ts_round, 'precision', 'test')
    assert diff is None  # the roundtriped series does not produce a diff when reinserted

    diff = tsh.insert(engine, ts, 'precision', 'test')  # neither does the original series
    assert diff is None


def test_serie_deletion(engine, tsh):
    ts = genserie(datetime(2018, 1, 10), 'H', 10)
    tsh.insert(engine, ts, 'keepme', 'Babar')
    tsh.insert(engine, ts, 'deleteme', 'Celeste')
    ts = genserie(datetime(2018, 1, 12), 'H', 10)
    tsh.insert(engine, ts, 'keepme', 'Babar')
    tsh.insert(engine, ts, 'deleteme', 'Celeste')

    seriecount, csetcount, csetseriecount = assert_structures(engine, tsh)

    assert tsh.metadata(engine, 'deleteme') == {
        'tzaware': False,
        'index_type': 'datetime64[ns]',
        'value_type': 'float64',
        'index_dtype': '<M8[ns]',
        'index_names': [],
        'value_dtype': '<f8'
    }

    with engine.begin() as cn:
        tsh.delete(cn, 'deleteme')

    assert not tsh.exists(engine, 'deleteme')
    log = [entry['author']
           for entry in tsh.log(engine, names=('keepme', 'deleteme'))]
    assert log == ['Babar', 'Babar']

    seriecount2, csetcount2, csetseriecount2 = assert_structures(engine, tsh)

    assert csetcount - csetcount2  == 2
    assert csetseriecount - csetseriecount2 == 2
    assert seriecount - seriecount2 == 1
    assert tsh.metadata(engine, 'deleteme') is None

    ts = pd.Series(
        [1, 2, 3],
        index=pd.date_range(start=utcdt(2018, 1, 1),
                            freq='D', periods=3)
    )
    with engine.begin() as cn:
        tsh.insert(cn, ts, 'deleteme', 'Celeste')

    assert tsh.metadata(engine, 'deleteme') == {
        'tzaware': True,
        'index_type': 'datetime64[ns, UTC]',
        'value_type': 'float64',
        'index_dtype': '|M8[ns]',
        'index_names': [],
        'value_dtype': '<f8'
    }


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
    assert_hist("""
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
    with engine.begin() as cn:
        tsh.strip(cn, 'xserie', csid)

    assert_hist("""
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

    log = tsh.log(engine, names=['xserie', 'yserie'])
    # 5 and 7 have disappeared
    assert [l['author'] for l in log
    ] == ['babar', 'celeste', 'babar', 'celeste', 'celeste', 'celeste']

    log = tsh.log(engine, stripped=True, names=['xserie', 'yserie'])
    for l in log:
        if l['meta']:
            meta = l['meta']
            stripinfo = meta.get('tshistory.info')
            if stripinfo:
                assert stripinfo.startswith('got stripped from')


def test_long_name(engine, tsh):
    serie = genserie(datetime(2010, 1, 1), 'D', 40)

    name = 'a' * 64
    tsh.insert(engine, serie, name, 'babar')
    assert tsh.get(engine, name) is not None


def test_get_delta(engine, tsh):
    assert tsh.get_delta(engine, 'no-such-series',
                         delta=pd.Timedelta(days=2)) is None

    for idate in pd.DatetimeIndex(start=utcdt(2015, 1, 1),
                                  end=utcdt(2015, 1, 1, 3),
                                  freq='H'):
        ts = genserie(start=idate, freq='H', repeat=7)
        tsh.insert(engine, ts, 'republication', 'test',
                   _insertion_date=idate)

    hist = tsh.get_history(engine, 'republication')
    assert_hist("""
insertion_date             value_date               
2015-01-01 00:00:00+00:00  2015-01-01 00:00:00+00:00    0.0
                           2015-01-01 01:00:00+00:00    1.0
                           2015-01-01 02:00:00+00:00    2.0
                           2015-01-01 03:00:00+00:00    3.0
                           2015-01-01 04:00:00+00:00    4.0
                           2015-01-01 05:00:00+00:00    5.0
                           2015-01-01 06:00:00+00:00    6.0
2015-01-01 01:00:00+00:00  2015-01-01 00:00:00+00:00    0.0
                           2015-01-01 01:00:00+00:00    0.0
                           2015-01-01 02:00:00+00:00    1.0
                           2015-01-01 03:00:00+00:00    2.0
                           2015-01-01 04:00:00+00:00    3.0
                           2015-01-01 05:00:00+00:00    4.0
                           2015-01-01 06:00:00+00:00    5.0
                           2015-01-01 07:00:00+00:00    6.0
2015-01-01 02:00:00+00:00  2015-01-01 00:00:00+00:00    0.0
                           2015-01-01 01:00:00+00:00    0.0
                           2015-01-01 02:00:00+00:00    0.0
                           2015-01-01 03:00:00+00:00    1.0
                           2015-01-01 04:00:00+00:00    2.0
                           2015-01-01 05:00:00+00:00    3.0
                           2015-01-01 06:00:00+00:00    4.0
                           2015-01-01 07:00:00+00:00    5.0
                           2015-01-01 08:00:00+00:00    6.0
2015-01-01 03:00:00+00:00  2015-01-01 00:00:00+00:00    0.0
                           2015-01-01 01:00:00+00:00    0.0
                           2015-01-01 02:00:00+00:00    0.0
                           2015-01-01 03:00:00+00:00    0.0
                           2015-01-01 04:00:00+00:00    1.0
                           2015-01-01 05:00:00+00:00    2.0
                           2015-01-01 06:00:00+00:00    3.0
                           2015-01-01 07:00:00+00:00    4.0
                           2015-01-01 08:00:00+00:00    5.0
                           2015-01-01 09:00:00+00:00    6.0
""", hist)

    deltas = tsh.get_delta(engine,  'republication', delta=timedelta(hours=3))
    assert deltas.name == 'republication'

    assert_df("""
2015-01-01 03:00:00+00:00    3.0
2015-01-01 04:00:00+00:00    3.0
2015-01-01 05:00:00+00:00    3.0
2015-01-01 06:00:00+00:00    3.0
2015-01-01 07:00:00+00:00    4.0
2015-01-01 08:00:00+00:00    5.0
2015-01-01 09:00:00+00:00    6.0
""", deltas)

    deltas = tsh.get_delta(engine,  'republication', delta=timedelta(hours=5))
    assert_df("""
2015-01-01 05:00:00+00:00    5.0
2015-01-01 06:00:00+00:00    5.0
2015-01-01 07:00:00+00:00    5.0
2015-01-01 08:00:00+00:00    5.0
2015-01-01 09:00:00+00:00    6.0
""", deltas)

    hist = tsh.get_history(engine, 'republication',
                           deltabefore=-timedelta(hours=3),
                           deltaafter=timedelta(hours=3))
    assert_hist("""
insertion_date             value_date               
2015-01-01 00:00:00+00:00  2015-01-01 03:00:00+00:00    3.0
2015-01-01 01:00:00+00:00  2015-01-01 04:00:00+00:00    3.0
2015-01-01 02:00:00+00:00  2015-01-01 05:00:00+00:00    3.0
2015-01-01 03:00:00+00:00  2015-01-01 06:00:00+00:00    3.0
""", hist)

    hist = tsh.get_history(engine, 'republication',
                           deltabefore=-timedelta(hours=5),
                           deltaafter=timedelta(hours=5))

    assert_hist("""
insertion_date             value_date               
2015-01-01 00:00:00+00:00  2015-01-01 05:00:00+00:00    5.0
2015-01-01 01:00:00+00:00  2015-01-01 06:00:00+00:00    5.0
2015-01-01 02:00:00+00:00  2015-01-01 07:00:00+00:00    5.0
2015-01-01 03:00:00+00:00  2015-01-01 08:00:00+00:00    5.0
""", hist)


def test_get_delta_2_tzaware(engine, tsh):
    # maybe a more interesting example, each days we insert 7 data points
    for idx, idate in enumerate(pd.DatetimeIndex(start=utcdt(2015, 1, 1),
                                                 end=utcdt(2015, 1, 4),
                                                 freq='D')):
        ts = genserie(start=idate, freq='H', repeat=7)
        tsh.insert(engine, ts, 'repu2', 'test', _insertion_date=idate)

    deltas = tsh.get_delta(engine, 'repu2', delta=timedelta(hours=3))
    assert_df("""
2015-01-01 03:00:00+00:00    3.0
2015-01-01 04:00:00+00:00    4.0
2015-01-01 05:00:00+00:00    5.0
2015-01-01 06:00:00+00:00    6.0
2015-01-02 03:00:00+00:00    3.0
2015-01-02 04:00:00+00:00    4.0
2015-01-02 05:00:00+00:00    5.0
2015-01-02 06:00:00+00:00    6.0
2015-01-03 03:00:00+00:00    3.0
2015-01-03 04:00:00+00:00    4.0
2015-01-03 05:00:00+00:00    5.0
2015-01-03 06:00:00+00:00    6.0
2015-01-04 03:00:00+00:00    3.0
2015-01-04 04:00:00+00:00    4.0
2015-01-04 05:00:00+00:00    5.0
2015-01-04 06:00:00+00:00    6.0
""", deltas)

    deltas = tsh.get_delta(engine, 'repu2', delta=timedelta(hours=3),
                           from_value_date=datetime(2015,1,2),
                           to_value_date=datetime(2015, 1,3))
    assert_df("""
2015-01-02 03:00:00+00:00    3.0
2015-01-02 04:00:00+00:00    4.0
2015-01-02 05:00:00+00:00    5.0
2015-01-02 06:00:00+00:00    6.0
""", deltas)

    # which is basically the same as below
    hist = tsh.get_history(engine, 'repu2',
                           deltabefore=-timedelta(hours=3))
    assert_hist("""
insertion_date             value_date               
2015-01-01 00:00:00+00:00  2015-01-01 03:00:00+00:00    3.0
                           2015-01-01 04:00:00+00:00    4.0
                           2015-01-01 05:00:00+00:00    5.0
                           2015-01-01 06:00:00+00:00    6.0
2015-01-02 00:00:00+00:00  2015-01-02 03:00:00+00:00    3.0
                           2015-01-02 04:00:00+00:00    4.0
                           2015-01-02 05:00:00+00:00    5.0
                           2015-01-02 06:00:00+00:00    6.0
2015-01-03 00:00:00+00:00  2015-01-03 03:00:00+00:00    3.0
                           2015-01-03 04:00:00+00:00    4.0
                           2015-01-03 05:00:00+00:00    5.0
                           2015-01-03 06:00:00+00:00    6.0
2015-01-04 00:00:00+00:00  2015-01-04 03:00:00+00:00    3.0
                           2015-01-04 04:00:00+00:00    4.0
                           2015-01-04 05:00:00+00:00    5.0
                           2015-01-04 06:00:00+00:00    6.0
""", hist)

    # constrain the boundaries
    deltas = tsh.get_delta(engine, 'repu2',
                           delta=timedelta(hours=3),
                           from_value_date=utcdt(2015, 1, 1, 6),
                           to_value_date=utcdt(2015, 1, 3, 4))
    assert_df("""
2015-01-01 06:00:00+00:00    6.0
2015-01-02 03:00:00+00:00    3.0
2015-01-02 04:00:00+00:00    4.0
2015-01-02 05:00:00+00:00    5.0
2015-01-02 06:00:00+00:00    6.0
2015-01-03 03:00:00+00:00    3.0
2015-01-03 04:00:00+00:00    4.0
""", deltas)

    # out-of-bounds from/to constraint
    deltas = tsh.get_delta(engine, 'repu2',
                           delta=timedelta(hours=3),
                           from_value_date=utcdt(2014, 1, 1, 6),
                           to_value_date=utcdt(2014, 1, 3, 4))
    assert len(deltas) == 0
    assert isinstance(deltas, pd.Series)


def test_get_delta_2_tznaive(engine, tsh):
    # same as above, with naive dates
    for idx, idate in enumerate(pd.DatetimeIndex(start=utcdt(2015, 1, 1),
                                                 end=utcdt(2015, 1, 4),
                                                 freq='D')):
        ts = genserie(start=idate.replace(tzinfo=None), freq='H', repeat=7)
        tsh.insert(engine, ts, 'repu-tz-naive', 'test', _insertion_date=idate)

    deltas = tsh.get_delta(engine, 'repu-tz-naive', delta=timedelta(hours=3))
    assert_df("""
2015-01-01 03:00:00    3.0
2015-01-01 04:00:00    4.0
2015-01-01 05:00:00    5.0
2015-01-01 06:00:00    6.0
2015-01-02 03:00:00    3.0
2015-01-02 04:00:00    4.0
2015-01-02 05:00:00    5.0
2015-01-02 06:00:00    6.0
2015-01-03 03:00:00    3.0
2015-01-03 04:00:00    4.0
2015-01-03 05:00:00    5.0
2015-01-03 06:00:00    6.0
2015-01-04 03:00:00    3.0
2015-01-04 04:00:00    4.0
2015-01-04 05:00:00    5.0
2015-01-04 06:00:00    6.0
""", deltas)

    deltas = tsh.get_delta(engine, 'repu-tz-naive', delta=timedelta(hours=3),
                           from_value_date=datetime(2015,1,2),
                           to_value_date=datetime(2015, 1,3))
    assert_df("""
2015-01-02 03:00:00    3.0
2015-01-02 04:00:00    4.0
2015-01-02 05:00:00    5.0
2015-01-02 06:00:00    6.0
""", deltas)

    # which is basically the same as below
    hist = tsh.get_history(engine, 'repu-tz-naive',
                           deltabefore=-timedelta(hours=3))
    assert_hist("""
insertion_date             value_date         
2015-01-01 00:00:00+00:00  2015-01-01 03:00:00    3.0
                           2015-01-01 04:00:00    4.0
                           2015-01-01 05:00:00    5.0
                           2015-01-01 06:00:00    6.0
2015-01-02 00:00:00+00:00  2015-01-02 03:00:00    3.0
                           2015-01-02 04:00:00    4.0
                           2015-01-02 05:00:00    5.0
                           2015-01-02 06:00:00    6.0
2015-01-03 00:00:00+00:00  2015-01-03 03:00:00    3.0
                           2015-01-03 04:00:00    4.0
                           2015-01-03 05:00:00    5.0
                           2015-01-03 06:00:00    6.0
2015-01-04 00:00:00+00:00  2015-01-04 03:00:00    3.0
                           2015-01-04 04:00:00    4.0
                           2015-01-04 05:00:00    5.0
                           2015-01-04 06:00:00    6.0
""", hist)

    # constrain the boundaries
    deltas = tsh.get_delta(engine, 'repu-tz-naive',
                           delta=timedelta(hours=3),
                           from_value_date=datetime(2015, 1, 1, 6),
                           to_value_date=datetime(2015, 1, 3, 4))
    assert_df("""
2015-01-01 06:00:00    6.0
2015-01-02 03:00:00    3.0
2015-01-02 04:00:00    4.0
2015-01-02 05:00:00    5.0
2015-01-02 06:00:00    6.0
2015-01-03 03:00:00    3.0
2015-01-03 04:00:00    4.0
""", deltas)

    # out-of-bounds from/to constraint
    deltas = tsh.get_delta(engine, 'repu-tz-naive',
                           delta=timedelta(hours=3),
                           from_value_date=datetime(2014, 1, 1, 6),
                           to_value_date=datetime(2014, 1, 3, 4))
    assert len(deltas) == 0
    assert isinstance(deltas, pd.Series)


def test_rename(engine, tsh):
    if tsh.namespace == 'zzz':
        return  # this test can only run once

    serie = genserie(datetime(2020, 1, 1), 'D', 3)

    tsh.insert(engine, serie, 'foo', 'Babar')
    tsh.insert(engine, serie, 'bar', 'Babar')
    tsh.insert(engine, serie, 'quux', 'Babar')

    tsh.rename(engine, 'foo', 'new-foo')
    tsh.rename(engine, 'bar', 'new-bar')

    tsh._resetcaches()

    assert tsh.get(engine, 'foo') is None
    assert tsh.get(engine, 'bar') is None

    for name in ('quux', 'new-foo', 'new-bar'):
        assert tsh.exists(engine, name)
        assert tsh.get(engine, name) is not None

    # check we can safely re-use 'foo'
    serie = genserie(datetime(2025, 1, 1), 'D', 3)
    tsh.insert(engine, serie, 'foo', 'Babar')
    ts = tsh.get(engine, 'foo')
    assert_df("""
2025-01-01    0.0
2025-01-02    1.0
2025-01-03    2.0
""", ts)


def test_index_order(engine, tsh):
    ts = genserie(datetime(2020, 1, 1), 'D', 3)

    # will be sorted for us as needed
    tsh.insert(engine, ts.sort_index(ascending=False),
               'test_order', 'babar')


def test_parallel(engine, tsh):
    ts = genserie(datetime(2010, 1, 1), 'D', 10)

    pool = threadpool(4)

    args = [
        (ts, 'a', 'aurelien'),
        (ts, 'b', 'arnaud'),
        (ts, 'c', 'alain'),
        (ts, 'd', 'andre')
    ]

    errors = []
    ns = tsh.namespace
    def insert(ts, name, author):
        tsh = TimeSerie(namespace=ns)
        with engine.begin() as cn:
            try:
                tsh.insert(cn, ts, name, author)
            except Exception as e:
                errors.append(e)

    pool(insert, args)
    assert not len(errors)

    for name in 'abcd':
        assert len(tsh.get(engine, name))


def test_chunky_array(engine, tsh):
    ts = pd.Series(
        [3, 2, 1],
        index=[utcdt(2018, 1, i) for i in reversed(range(1, 4))]
    )

    assert ts.index.flags['C_CONTIGUOUS']
    ts = ts.sort_index()
    assert not ts.index.flags['C_CONTIGUOUS']

    with engine.begin() as cn:
        tsh.insert(cn, ts, 'chunky', 'Babar')

    # we're ok now
    ts_out = tsh.get(engine, 'chunky')
    assert_df("""
2018-01-01 00:00:00+00:00    1.0
2018-01-02 00:00:00+00:00    2.0
2018-01-03 00:00:00+00:00    3.0
""", ts_out)


def test_null_serie(engine, tsh):
    ts = pd.Series()

    tsh.insert(engine, ts, 'null', 'Babar')


def test_na_at_boundaries(engine, tsh):
    ts = pd.Series([np.nan] * 3 + [3] * 5 + [np.nan] * 2,
                   index=pd.date_range(start=datetime(2010, 1, 10),
                                       freq='D', periods=10))
    tsh.insert(engine, ts, 'test_nan', 'test')
    result = tsh.get(engine, 'test_nan')
    assert_df("""
2010-01-13    3.0
2010-01-14    3.0
2010-01-15    3.0
2010-01-16    3.0
2010-01-17    3.0
""", result)

    result = tsh.get(engine, 'test_nan', _keep_nans=True)
    assert_df("""
2010-01-13    3.0
2010-01-14    3.0
2010-01-15    3.0
2010-01-16    3.0
2010-01-17    3.0
""", result)

    ival = tsh.interval(engine, 'test_nan')
    assert ival.left == datetime(2010, 1, 13)
    assert ival.right == datetime(2010, 1, 17)

    # now, let's update with useless nans
    ts = pd.Series([np.nan] * 3 + [4] * 5 + [np.nan] * 2,
                   index=pd.date_range(start=datetime(2010, 1, 10),
                                       freq='D', periods=10))
    tsh.insert(engine, ts, 'test_nan', 'test')
    result = tsh.get(engine, 'test_nan', _keep_nans=True)
    # they don't show up
    assert_df("""
2010-01-13    4.0
2010-01-14    4.0
2010-01-15    4.0
2010-01-16    4.0
2010-01-17    4.0
""", result)

    ival = tsh.interval(engine, 'test_nan')
    assert ival.left == datetime(2010, 1, 13)
    assert ival.right == datetime(2010, 1, 17)

    # let's really shorten the series
    ts = pd.Series([np.nan] * 4 + [5] * 3 + [np.nan] * 3,
                   index=pd.date_range(start=datetime(2010, 1, 10),
                                       freq='D', periods=10))
    tsh.insert(engine, ts, 'test_nan', 'test')
    result = tsh.get(engine, 'test_nan', _keep_nans=True)
    assert_df("""
2010-01-13    NaN
2010-01-14    5.0
2010-01-15    5.0
2010-01-16    5.0
2010-01-17    NaN
""", result)

    ival = tsh.interval(engine, 'test_nan')
    assert ival.left == datetime(2010, 1, 14)
    assert ival.right == datetime(2010, 1, 16)


def test_no_series(engine, tsh):
    assert tsh.get(engine, 'inexisting_name') is None


def test_insert_errors(engine, tsh):
    ts = pd.Series([1, 2, 3],
                   index=pd.date_range(start=utcdt(2018, 1, 1),
                                       freq='D', periods=3))

    with pytest.raises(AssertionError):
        tsh.insert(engine, 42, 'error', 'Babar')

    with pytest.raises(AssertionError):
        tsh.insert(engine, ts, 42, 'Babar')

    with pytest.raises(AssertionError):
        tsh.insert(engine, ts, 'error', 42)

    with pytest.raises(AssertionError):
        tsh.insert(engine, ts, 'error', 'Babar', _insertion_date='2010-1-1')

    with pytest.raises(AssertionError):
        tsh.insert(engine, ts, 'error', 'Babar', metadata=42)


def test_index_with_nat(engine, tsh):
    index = list(pd.date_range(start=utcdt(2018, 1, 1),
                          freq='D', periods=3))
    index[1] = pd.NaT
    ts = pd.Series([1, 2, 3], index=index)

    with pytest.raises(AssertionError):
        tsh.insert(engine, ts, 'index_with_nat', 'Bugger')
