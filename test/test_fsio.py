from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from tshistory.testutil import (
    assert_df,
    assert_hist,
    genserie,
    tempconfig,
    utcdt
)
from tshistory.storage import FS1


def test_exists(engine, tsf):
    assert not tsf.exists(engine, 'hello')


def test_create_empty(engine, tsf):
    ts = pd.Series()

    tsf.update(
        engine,
        ts,
        'fs-first',
        'Babar'
    )


def test_create_initial(engine, tsf):
    ts = pd.Series(
        [1, 2, 3],
        index=pd.date_range(
            pd.Timestamp('2024-1-1', tz='utc'),
            periods=3,
            freq='D'
        )
    )

    assert not tsf.exists(engine, 'fs-first')

    diff = tsf.update(
        engine,
        ts,
        'fs-first',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-1', tz='utc')
    )

    assert_df("""
2024-01-01 00:00:00+00:00    1.0
2024-01-02 00:00:00+00:00    2.0
2024-01-03 00:00:00+00:00    3.0
""", diff)

    assert tsf.exists(engine, 'fs-first')

    out = tsf.get(engine, 'fs-first')
    assert_df("""
2024-01-01 00:00:00+00:00    1.0
2024-01-02 00:00:00+00:00    2.0
2024-01-03 00:00:00+00:00    3.0
""", out)

    idates = tsf.insertion_dates(engine, 'fs-first')
    assert idates == [
        pd.Timestamp('2024-01-01 00:00:00+0000', tz='UTC')
    ]

    ts = tsf.get(
        engine,
        'fs-first',
        revision_date=pd.Timestamp('2023-12-31', tz='UTC')
    )
    assert not len(ts)


def test_get_future_revdate(engine, tsf):
    ts = pd.Series(
        [1, 2, 3],
        index=pd.date_range(
            pd.Timestamp('2024-1-1', tz='utc'),
            periods=3,
            freq='D'
        )
    )

    tsf.update(
        engine,
        ts,
        'fs-future-revdate',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-1', tz='utc')
    )

    ts = tsf.get(
        engine,
        'fs-future-revdate',
        revision_date=pd.Timestamp('2025-1-1', tz='UTC')  # in the future
    )
    assert_df("""
2024-01-01 00:00:00+00:00    1.0
2024-01-02 00:00:00+00:00    2.0
2024-01-03 00:00:00+00:00    3.0
""", ts)


def test_create_naive(engine, tsf):
    ts = pd.Series(
        [1, 2, 3],
        index=pd.date_range(
            pd.Timestamp('2024-1-1'),
            periods=3,
            freq='D'
        )
    )

    assert not tsf.exists(engine, 'fs-naive')

    diff = tsf.update(
        engine,
        ts,
        'fs-naive',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-1', tz='utc')
    )

    assert_df("""
2024-01-01    1.0
2024-01-02    2.0
2024-01-03    3.0
""", diff)

    assert tsf.exists(engine, 'fs-naive')

    out = tsf.get(engine, 'fs-naive')
    assert_df("""
2024-01-01    1.0
2024-01-02    2.0
2024-01-03    3.0
""", out)

    out = tsf.get(engine, 'fs-naive', from_value_date=pd.Timestamp('2024-1-2', tz='utc'))
    assert_df("""
2024-01-02    2.0
2024-01-03    3.0
""", out)

    tsf.delete(engine, 'fs-naive')
    assert not tsf.exists(engine, 'fs-naive')
    tsf.delete(engine, 'no-such-series')


def test_replace(engine, tsf):
    ts0 = pd.Series(
        [1, 2, 3],
        index=pd.date_range(
            pd.Timestamp('2024-1-1', tz='utc'),
            periods=3,
            freq='D'
        )
    )

    tsf.replace(
        engine,
        ts0,
        'fs-replace',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-1', tz='utc')
    )

    ts = tsf.get(engine, 'fs-replace')
    assert_df("""
2024-01-01 00:00:00+00:00    1.0
2024-01-02 00:00:00+00:00    2.0
2024-01-03 00:00:00+00:00    3.0
""", ts)

    ts1 = pd.Series(
        [3, 4, 5],
        index=pd.date_range(
            pd.Timestamp('2024-1-2', tz='utc'),
            periods=3,
            freq='D'
        )
    )

    tsf.replace(
        engine,
        ts1,
        'fs-replace',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-2', tz='utc')
    )

    ts = tsf.get(engine, 'fs-replace')
    assert_df("""
2024-01-02 00:00:00+00:00    3.0
2024-01-03 00:00:00+00:00    4.0
2024-01-04 00:00:00+00:00    5.0
""", ts)


def test_find_rev(engine, tsf):
    ts0 = pd.Series(
        [1, 2, 3],
        index=pd.date_range(
            pd.Timestamp('2024-1-1', tz='utc'),
            periods=3,
            freq='D'
        )
    )

    tsf.replace(
        engine,
        ts0,
        'fs-replace',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-1 00:00:00', tz='utc')
    )

    ts = tsf.get(engine, 'fs-replace')
    assert_df("""
2024-01-01 00:00:00+00:00    1.0
2024-01-02 00:00:00+00:00    2.0
2024-01-03 00:00:00+00:00    3.0
""", ts)

    ts1 = pd.Series(
        [3, 4, 5],
        index=pd.date_range(
            pd.Timestamp('2024-1-2', tz='utc'),
            periods=3,
            freq='D'
        )
    )

    smalldelta = pd.Timedelta(microseconds=1)
    tsf.replace(
        engine,
        ts1,
        'fs-replace',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-1 00:00:00', tz='utc') + smalldelta
    )

    ts = tsf.get(engine, 'fs-replace')
    assert_df("""
2024-01-02 00:00:00+00:00    3.0
2024-01-03 00:00:00+00:00    4.0
2024-01-04 00:00:00+00:00    5.0
""", ts)


def test_update_get_keep_nans(engine, tsf):
    ts = pd.Series(
        [1, np.nan, 3],
        index=pd.date_range(
            pd.Timestamp('2024-1-1', tz='utc'),
            periods=3,
            freq='D'
        )
    )
    tsf.update(
        engine,
        ts,
        'fs-withnan',
        'Babar',
        keepnans=True
    )
    ts = tsf.get(engine, 'fs-withnan', _keep_nans=True)
    assert_df("""
2024-01-01 00:00:00+00:00    1.0
2024-01-02 00:00:00+00:00    NaN
2024-01-03 00:00:00+00:00    3.0
""", ts)

    ts = pd.Series(
        [np.nan, np.nan, np.nan],
        index=pd.date_range(
            pd.Timestamp('2024-1-1', tz='utc'),
            periods=3,
            freq='D'
        )
    )
    tsf.update(
        engine,
        ts,
        'fs-withnan',
        'Babar',
        keepnans=True
    )
    ts = tsf.get(engine, 'fs-withnan', _keep_nans=True)
    assert_df("""
2024-01-01 00:00:00+00:00   NaN
2024-01-02 00:00:00+00:00   NaN
2024-01-03 00:00:00+00:00   NaN
""", ts)
    assert tsf.interval(engine, 'fs-withnan') is None


def test_two_mono_chunk_revisions(engine, tsf):
    ts0 = pd.Series(
        [1, 2, 3],
        index=pd.date_range(
            pd.Timestamp('2024-1-1', tz='utc'),
            periods=3,
            freq='D'
        )
    )

    tsf.update(
        engine,
        ts0,
        'fs-2revs',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-1', tz='utc')
    )

    ts1 = pd.Series(
        [4, 5, 6],
        index=pd.date_range(
            pd.Timestamp('2024-1-4', tz='utc'),
            periods=3,
            freq='D'
        )
    )

    tsf.update(
        engine,
        ts1,
        'fs-2revs',
        'Celeste',
        insertion_date=pd.Timestamp('2024-1-2', tz='utc')
    )

    out = tsf.get(engine, 'fs-2revs')
    assert_df("""
2024-01-01 00:00:00+00:00    1.0
2024-01-02 00:00:00+00:00    2.0
2024-01-03 00:00:00+00:00    3.0
2024-01-04 00:00:00+00:00    4.0
2024-01-05 00:00:00+00:00    5.0
2024-01-06 00:00:00+00:00    6.0
""", out)

    idates = tsf.insertion_dates(engine, 'fs-2revs')
    assert idates == [
        pd.Timestamp('2024-01-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-02 00:00:00+0000', tz='UTC')
    ]

    ts = tsf.get(
        engine,
        'fs-2revs',
        revision_date=pd.Timestamp('2024-1-1 12:00:00+0000', tz='UTC')
    )
    assert_df("""
2024-01-01 00:00:00+00:00    1.0
2024-01-02 00:00:00+00:00    2.0
2024-01-03 00:00:00+00:00    3.0
""", ts)


def test_naive_two_mono_chunk_revisions(engine, tsf):
    ts0 = pd.Series(
        [1, 2, 3],
        index=pd.date_range(
            pd.Timestamp('2024-1-1'),
            periods=3,
            freq='D'
        )
    )

    tsf.update(
        engine,
        ts0,
        'fs-naive2revs',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-1', tz='utc')
    )

    ts1 = pd.Series(
        [4, 5, 6],
        index=pd.date_range(
            pd.Timestamp('2024-1-4'),
            periods=3,
            freq='D'
        )
    )

    tsf.update(
        engine,
        ts1,
        'fs-naive2revs',
        'Celeste',
        insertion_date=pd.Timestamp('2024-1-2', tz='utc')
    )

    out = tsf.get(engine, 'fs-naive2revs')
    assert_df("""
2024-01-01    1.0
2024-01-02    2.0
2024-01-03    3.0
2024-01-04    4.0
2024-01-05    5.0
2024-01-06    6.0
""", out)

    idates = tsf.insertion_dates(engine, 'fs-naive2revs')
    assert idates == [
        pd.Timestamp('2024-01-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-02 00:00:00+0000', tz='UTC')
    ]

    ts = tsf.get(
        engine,
        'fs-naive2revs',
        revision_date=pd.Timestamp('2024-1-1 12:00:00+0000', tz='UTC')
    )
    assert_df("""
2024-01-01    1.0
2024-01-02    2.0
2024-01-03    3.0
""", ts)


def test_two_overlapping_revisions(engine, tsf):
    ts0 = pd.Series(
        [1, 2, 3],
        index=pd.date_range(
            pd.Timestamp('2024-1-1', tz='utc'),
            periods=3,
            freq='D'
        )
    )

    tsf.update(
        engine,
        ts0,
        'fs-2overlap',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-1', tz='utc')
    )

    ts1 = pd.Series(
        [2.2, 3, 4],
        index=pd.date_range(
            pd.Timestamp('2024-1-2', tz='utc'),
            periods=3,
            freq='D'
        )
    )

    tsf.update(
        engine,
        ts1,
        'fs-2overlap',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-2', tz='utc')
    )

    ts = tsf.get(engine, 'fs-2overlap')
    assert_df("""
2024-01-01 00:00:00+00:00    1.0
2024-01-02 00:00:00+00:00    2.2
2024-01-03 00:00:00+00:00    3.0
2024-01-04 00:00:00+00:00    4.0
""", ts)


def test_one_multi_chunks_revision(engine, tsf):
    ts = pd.Series(
        [1] * 300,  # two chunks
        index=pd.date_range(
            pd.Timestamp('2024-1-1', tz='utc'),
            periods=300,
            freq='h'
        )
    )

    tsf.update(
        engine,
        ts,
        'fs-multichunks',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-1', tz='utc')
    )

    ts = tsf.get(engine, 'fs-multichunks')
    assert len(ts) == 300

    with engine.begin() as cn:
        sto = FS1(cn, tsf, 'fs-multichunks')
        if sto._max_bucket_size == 150:
            assert sto.tree_entries == 2


def test_two_chunks_three_revision(engine, tsf):
    ts = pd.Series(
        [1] * 300,  # two chunks
        index=pd.date_range(
            pd.Timestamp('2024-1-1', tz='utc'),
            periods=300,
            freq='h'
        )
    )

    tsf.update(
        engine,
        ts,
        'fs-2chunks3revs',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-1', tz='utc')
    )

    ts = ts * 2
    tsf.update(
        engine,
        ts,
        'fs-2chunks3revs',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-2', tz='utc')
    )

    ts = tsf.get(engine, 'fs-2chunks3revs')
    assert len(ts) == 300

    with engine.begin() as cn:
        sto = FS1(cn, tsf, 'fs-2chunks3revs')
        if sto._max_bucket_size == 150:
            assert sto.tree_entries == 4

    ts = ts * 2
    ts = ts[-10:]

    tsf.update(
        engine,
        ts,
        'fs-2chunks3revs',
        'Celeste',
        insertion_date=pd.Timestamp('2024-1-3', tz='utc')
    )

    ts = tsf.get(
        engine,
        'fs-2chunks3revs',
        from_value_date=pd.Timestamp('2024-1-13', tz='utc')
    )
    assert_df("""
2024-01-13 00:00:00+00:00    2.0
2024-01-13 01:00:00+00:00    2.0
2024-01-13 02:00:00+00:00    4.0
2024-01-13 03:00:00+00:00    4.0
2024-01-13 04:00:00+00:00    4.0
2024-01-13 05:00:00+00:00    4.0
2024-01-13 06:00:00+00:00    4.0
2024-01-13 07:00:00+00:00    4.0
2024-01-13 08:00:00+00:00    4.0
2024-01-13 09:00:00+00:00    4.0
2024-01-13 10:00:00+00:00    4.0
2024-01-13 11:00:00+00:00    4.0
""", ts)

    ts = tsf.get(
        engine,
        'fs-2chunks3revs',
        from_value_date=pd.Timestamp('2024-1-13', tz='utc'),
        to_value_date=pd.Timestamp('2024-1-13 03:00:00+00:00', tz='utc')
    )
    assert_df("""
2024-01-13 00:00:00+00:00    2.0
2024-01-13 01:00:00+00:00    2.0
2024-01-13 02:00:00+00:00    4.0
2024-01-13 03:00:00+00:00    4.0
""", ts)


def _prepare_revs(engine, tsf, name):
    for i in range(5):
        ts = pd.Series(
            [i],
            index=[pd.Timestamp('2024-1-1', tz='utc')]
        )
        tsf.update(
            engine,
            ts,
            name,
            'Babar',
            insertion_date=pd.Timestamp(f'2024-1-{i+1}', tz='utc')
        )


def test_get_revision_date(engine, tsf):
    name = 'fs-revdate'
    _prepare_revs(engine, tsf, name)

    ts = tsf.get(engine, 'fs-revdate', revision_date=pd.Timestamp('2023-12-31', tz='utc'))
    assert not len(ts)

    ts = tsf.get(engine, 'fs-revdate', revision_date=pd.Timestamp('2024-1-1', tz='utc'))
    assert_df("""
2024-01-01 00:00:00+00:00    0.0
""", ts)

    ts = tsf.get(engine, 'fs-revdate', revision_date=pd.Timestamp('2024-1-2', tz='utc'))
    assert_df("""
2024-01-01 00:00:00+00:00    1.0
""", ts)

    ts = tsf.get(engine, 'fs-revdate', revision_date=pd.Timestamp('2024-1-3', tz='utc'))
    assert_df("""
2024-01-01 00:00:00+00:00    2.0
""", ts)

    ts = tsf.get(engine, 'fs-revdate', revision_date=pd.Timestamp('2024-1-4', tz='utc'))
    assert_df("""
2024-01-01 00:00:00+00:00    3.0
""", ts)

    ts = tsf.get(engine, 'fs-revdate', revision_date=pd.Timestamp('2024-1-5', tz='utc'))
    assert_df("""
2024-01-01 00:00:00+00:00    4.0
""", ts)


def test_insertion_dates(engine, tsf):
    name = 'fs-idates'
    _prepare_revs(engine, tsf, name)

    idates = tsf.insertion_dates(engine, name)
    assert idates == [
        pd.Timestamp('2024-01-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-03 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-04 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-05 00:00:00+0000', tz='UTC')
    ]

    idates = tsf.insertion_dates(
        engine,
        name,
        from_insertion_date=pd.Timestamp('2023-12-31', tz='utc'),
        to_insertion_date=pd.Timestamp('2024-1-1', tz='utc')
    )
    assert idates == [
        pd.Timestamp('2024-01-01 00:00:00+0000', tz='UTC')
    ]

    idates = tsf.insertion_dates(
        engine,
        name,
        from_insertion_date=pd.Timestamp('2024-1-1', tz='utc'),
        to_insertion_date=pd.Timestamp('2024-1-3', tz='utc')
    )
    assert idates == [
        pd.Timestamp('2024-01-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-03 00:00:00+0000', tz='UTC')
    ]

    idates = tsf.insertion_dates(
        engine,
        name,
        from_insertion_date=pd.Timestamp('2024-1-1', tz='utc'),
        to_insertion_date=pd.Timestamp('2024-1-4', tz='utc')
    )
    assert idates == [
        pd.Timestamp('2024-01-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-03 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-04 00:00:00+0000', tz='UTC')
    ]

    idates = tsf.insertion_dates(
        engine,
        name,
        from_insertion_date=pd.Timestamp('2024-1-2', tz='utc'),
        to_insertion_date=pd.Timestamp('2024-1-5', tz='utc')
    )
    assert idates == [
        pd.Timestamp('2024-01-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-03 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-04 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-05 00:00:00+0000', tz='UTC')
    ]

    idates = tsf.insertion_dates(
        engine,
        name,
        from_insertion_date=pd.Timestamp('2024-1-4', tz='utc'),
        to_insertion_date=pd.Timestamp('2024-1-6', tz='utc')
    )
    assert idates == [
        pd.Timestamp('2024-01-04 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-05 00:00:00+0000', tz='UTC')
    ]

    idates = tsf.insertion_dates(
        engine,
        name,
        from_insertion_date=pd.Timestamp('2024-1-4', tz='utc')
    )
    assert idates == [
        pd.Timestamp('2024-01-04 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-05 00:00:00+0000', tz='UTC')
    ]

    idates = tsf.insertion_dates(
        engine,
        name,
        to_insertion_date=pd.Timestamp('2024-1-2', tz='utc')
    )
    assert idates == [
        pd.Timestamp('2024-01-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-02 00:00:00+0000', tz='UTC')
    ]

    assert tsf.latest_insertion_date(engine, name) == pd.Timestamp('2024-1-5', tz='utc')
    assert tsf.first_insertion_date(engine, name) == pd.Timestamp('2024-1-1', tz='utc')


def test_insertion_dates_2(engine, tsf):
    name = 'fs-idates2'
    for i in range(5):
        ts = pd.Series(
            [i],
            index=[pd.Timestamp(f'2024-1-{i+1}', tz='utc')]
        )
        tsf.update(
            engine,
            ts,
            name,
            'Babar',
            insertion_date=pd.Timestamp(f'2024-1-{i+1}', tz='utc')
        )

    idates = tsf.insertion_dates(
        engine,
        name,
        from_value_date=pd.Timestamp('2023-12-31', tz='utc'),
        to_value_date=pd.Timestamp('2024-1-2', tz='utc')
    )
    assert idates == [
        pd.Timestamp('2024-01-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-02 00:00:00+0000', tz='UTC')
    ]

    idates = tsf.insertion_dates(
        engine,
        name,
        from_value_date=pd.Timestamp('2024-1-2', tz='utc'),
        to_value_date=pd.Timestamp('2024-1-3', tz='utc')
    )
    assert idates == [
        pd.Timestamp('2024-01-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-03 00:00:00+0000', tz='UTC')
    ]


def test_insertion_dates_3(engine, tsf):
    for i in range(10):
        ts = pd.Series(
            np.array([1, 2, 3]) + i*2,
            pd.date_range(utcdt(2024, 4, 1+i), freq='d', periods=3)
        )
        tsf.update(
            engine,
            ts,
            'historical-series',
            'Babar',
            insertion_date=utcdt(2024, 4, 1+i)
        )

    revs = tsf.insertion_dates(engine, 'historical-series')
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

    revs = tsf.insertion_dates(
        engine,
        'historical-series',
        from_value_date=pd.Timestamp('2024-04-04'),
        to_value_date=pd.Timestamp('2024-04-05')
    )
    assert revs == [
        pd.Timestamp('2024-04-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-03 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-04 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-04-05 00:00:00+0000', tz='UTC'),
    ]


def test_log(engine, tsf):
    name = 'fs-logs'
    _prepare_revs(engine, tsf, name)

    log = tsf.log(engine, name)
    assert log == [
        {
            'date': pd.Timestamp('2024-01-01 00:00:00+0000', tz='UTC'),
            'author': 'Babar',
            'meta': {},
            'rev': 1
        },
        {
            'date': pd.Timestamp('2024-01-02 00:00:00+0000', tz='UTC'),
            'author': 'Babar',
            'meta': {},
            'rev': 2
        },
        {
            'date': pd.Timestamp('2024-01-03 00:00:00+0000', tz='UTC'),
            'author': 'Babar',
            'meta': {},
            'rev': 3
        },
        {
            'date': pd.Timestamp('2024-01-04 00:00:00+0000', tz='UTC'),
            'author': 'Babar',
            'meta': {},
            'rev': 4
        },
        {
            'date': pd.Timestamp('2024-01-05 00:00:00+0000', tz='UTC'),
            'author': 'Babar',
            'meta': {},
            'rev': 5
        }
    ]

    log = tsf.log(
        engine,
        name,
        fromdate=pd.Timestamp('2024-01-02', tz='utc'),
        todate=pd.Timestamp('2024-01-04', tz='utc'),
        limit=2
    )
    assert log == [
        {
            'author': 'Babar',
            'date': pd.Timestamp('2024-01-02 00:00:00+0000', tz='UTC'),
            'meta': {},
            'rev': 2
        },
        {
            'author': 'Babar',
            'date': pd.Timestamp('2024-01-03 00:00:00+0000', tz='UTC'),
            'meta': {},
            'rev': 3
        }
    ]


def test_history(engine, tsf):
    name = 'fs-history'
    _prepare_revs(engine, tsf, name)

    hist = tsf.history(engine, name)
    assert_hist("""
insertion_date             value_date               
2024-01-01 00:00:00+00:00  2024-01-01 00:00:00+00:00    0.0
2024-01-02 00:00:00+00:00  2024-01-01 00:00:00+00:00    1.0
2024-01-03 00:00:00+00:00  2024-01-01 00:00:00+00:00    2.0
2024-01-04 00:00:00+00:00  2024-01-01 00:00:00+00:00    3.0
2024-01-05 00:00:00+00:00  2024-01-01 00:00:00+00:00    4.0
""", hist)


def test_diffs(engine, tsf):
    name = 'fs-diffs'
    _prepare_revs(engine, tsf, name)

    # add one juicy one
    ts = pd.Series(
        [5, 5, 5],
        index=[
            pd.Timestamp('2024-1-1', tz='utc'),
            pd.Timestamp('2024-1-2', tz='utc'),
            pd.Timestamp('2024-1-3', tz='utc')
        ]
    )
    tsf.update(
        engine,
        ts,
        name,
        'Celeste',
        insertion_date=pd.Timestamp('2024-1-6', tz='utc')
    )

    diffs = tsf.diffs(engine, name)
    assert_hist("""
insertion_date             value_date               
2024-01-01 00:00:00+00:00  2024-01-01 00:00:00+00:00    0.0
2024-01-02 00:00:00+00:00  2024-01-01 00:00:00+00:00    1.0
2024-01-03 00:00:00+00:00  2024-01-01 00:00:00+00:00    2.0
2024-01-04 00:00:00+00:00  2024-01-01 00:00:00+00:00    3.0
2024-01-05 00:00:00+00:00  2024-01-01 00:00:00+00:00    4.0
2024-01-06 00:00:00+00:00  2024-01-01 00:00:00+00:00    5.0
                           2024-01-02 00:00:00+00:00    5.0
                           2024-01-03 00:00:00+00:00    5.0
""", diffs)


def test_erasure(engine, tsf):
    ts = pd.Series(
        list(range(11)),
        index=pd.date_range(
            pd.Timestamp('2024-1-1'),
            periods=11,
            freq='d'
        )
    )

    # create outright with a Nan at the end
    ts.iloc[-1] = np.nan
    assert ts.index.dtype.name == 'datetime64[ns]'
    tsf.update(
        engine, ts, 'ts_erase', 'test',
        keepnans=True, insertion_date=pd.Timestamp('2025-1-1', tz='utc')
    )

    # erase in the beginning and middle
    ts.iloc[0] = np.nan
    ts.iloc[3] = np.nan
    assert ts.index.dtype.name == 'datetime64[ns]'
    tsf.update(
        engine, ts, 'ts_erase', 'test',
        keepnans=True, insertion_date=pd.Timestamp('2025-1-2', tz='utc')
    )

    assert_df("""
2024-01-02    1.0
2024-01-03    2.0
2024-01-05    4.0
2024-01-06    5.0
2024-01-07    6.0
2024-01-08    7.0
2024-01-09    8.0
2024-01-10    9.0
""", tsf.get(engine, 'ts_erase'))

    ts2 = tsf.get(
        engine,
        'ts_erase',
        # in the future
        revision_date=pd.Timestamp('2038-1-1', tz='utc')
    )
    assert_df("""
2024-01-02    1.0
2024-01-03    2.0
2024-01-05    4.0
2024-01-06    5.0
2024-01-07    6.0
2024-01-08    7.0
2024-01-09    8.0
2024-01-10    9.0
""", ts2)

    assert_hist("""
insertion_date             value_date
2025-01-01 00:00:00+00:00  2024-01-01    0.0
                           2024-01-02    1.0
                           2024-01-03    2.0
                           2024-01-04    3.0
                           2024-01-05    4.0
                           2024-01-06    5.0
                           2024-01-07    6.0
                           2024-01-08    7.0
                           2024-01-09    8.0
                           2024-01-10    9.0
2025-01-02 00:00:00+00:00  2024-01-02    1.0
                           2024-01-03    2.0
                           2024-01-05    4.0
                           2024-01-06    5.0
                           2024-01-07    6.0
                           2024-01-08    7.0
                           2024-01-09    8.0
                           2024-01-10    9.0
""", tsf.history(engine, 'ts_erase'))

    ts.iloc[0] = 42
    ts.iloc[3] = 23
    tsf.update(engine, ts, 'ts_erase', 'test')

    assert_df("""
2024-01-01    42.0
2024-01-02     1.0
2024-01-03     2.0
2024-01-04    23.0
2024-01-05     4.0
2024-01-06     5.0
2024-01-07     6.0
2024-01-08     7.0
2024-01-09     8.0
2024-01-10     9.0
""", tsf.get(engine, 'ts_erase'))


def test_str_series(engine, tsf):
    ts = pd.Series(
        list('abc'),
        index=pd.date_range(
            pd.Timestamp('2024-1-1'),
            periods=3,
            freq='h'
        )
    )

    tsf.update(
        engine,
        ts,
        'fs-str',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-1', tz='utc')
    )

    assert_df("""
2024-01-01 00:00:00    a
2024-01-01 01:00:00    b
2024-01-01 02:00:00    c
""", tsf.get(engine, 'fs-str'))

    ts2 = pd.Series(
        ['€', 'ça', 'ôlala'],
        index=pd.date_range(
            pd.Timestamp('2024-1-1 02:00:00'),
            periods=3,
            freq='h'
        )
    )

    tsf.update(
        engine, ts2,
        'fs-str2',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-1', tz='utc')
    )

    assert_df("""
2024-01-01 02:00:00        €
2024-01-01 03:00:00       ça
2024-01-01 04:00:00    ôlala
""", tsf.get(engine, 'fs-str2'))

    tsf.update(
        engine,
        ts2,
        'fs-str',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-2', tz='utc')
    )

    assert_df("""
2024-01-01 00:00:00        a
2024-01-01 01:00:00        b
2024-01-01 02:00:00        €
2024-01-01 03:00:00       ça
2024-01-01 04:00:00    ôlala
""", tsf.get(engine, 'fs-str'))


def test_update_noparent(engine, tsf):
    ts = pd.Series(
        [1, 2, 3],
        index=pd.date_range(
            pd.Timestamp('2014-1-1', tz='utc'),
            periods=3,
            freq='d'
        )
    )

    tsf.update(
        engine,
        ts,
        'fs-noparent',
        'Babar'
    )

    ts2 = pd.Series(
        [1, 2, 3],
        index=pd.date_range(
            pd.Timestamp('2013-1-1', tz='utc'),
            periods=3,
            freq='d'
        )
    )

    tsf.update(
        engine,
        ts2,
        'fs-noparent',
        'Celeste'
    )

    assert_df("""
2013-01-01 00:00:00+00:00    1.0
2013-01-02 00:00:00+00:00    2.0
2013-01-03 00:00:00+00:00    3.0
2014-01-01 00:00:00+00:00    1.0
2014-01-02 00:00:00+00:00    2.0
2014-01-03 00:00:00+00:00    3.0
""", tsf.get(engine, 'fs-noparent'))


def test_history_naivequery(engine, tsf):
    for idx in (1, 2, 3):
        with engine.begin() as cn:
            tsf.update(
                cn,
                genserie(datetime(2017, 1, 1), 'd', idx),
                'fs-h2',
                'Babar',
                insertion_date=utcdt(2017, 2, idx)
            )

    # out of bounds (right)
    idates = tsf.insertion_dates(
        engine,
        'fs-h2',
        from_insertion_date=datetime(2017, 2, 4),
        to_insertion_date=datetime(2017, 2, 4)
    )
    assert idates == []
    h = tsf.history(engine, 'fs-h2',
                    from_insertion_date=datetime(2017, 2, 4),
                    to_insertion_date=datetime(2017, 2, 4))
    assert h == {}

    # out of bounds (left)
    idates = tsf.insertion_dates(
        engine,
        'fs-h2',
        from_insertion_date=datetime(2016, 2, 4),
        to_insertion_date=datetime(2016, 2, 4)
    )
    assert idates == []
    h = tsf.history(engine, 'fs-h2',
                    from_insertion_date=datetime(2016, 2, 4),
                    to_insertion_date=datetime(2016, 2, 4))
    assert h == {}

    h = tsf.history(engine, 'fs-h2')
    assert_hist("""
insertion_date             value_date
2017-02-01 00:00:00+00:00  2017-01-01    0.0
2017-02-02 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
2017-02-03 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
                           2017-01-03    2.0
""", h)

    # get history ranges
    h = tsf.history(engine, 'fs-h2',
                    from_insertion_date=datetime(2017, 2, 2))
    assert_hist("""
insertion_date             value_date
2017-02-02 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
2017-02-03 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
                           2017-01-03    2.0
""", h)

    h = tsf.history(engine, 'fs-h2',
                    to_insertion_date=datetime(2017, 2, 2))
    assert_hist("""
insertion_date             value_date
2017-02-01 00:00:00+00:00  2017-01-01    0.0
2017-02-02 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
""", h)

    h = tsf.history(engine, 'fs-h2',
                    from_insertion_date=datetime(2017, 2, 2),
                    to_insertion_date=datetime(2017, 2, 2))
    assert_hist("""
insertion_date             value_date
2017-02-02 00:00:00+00:00  2017-01-01    0.0
                           2017-01-02    1.0
""", h)


def test_strip(engine, tsf):
    for i in range(1, 5):
        pubdate = utcdt(2017, 1, i)
        ts = genserie(datetime(2017, 1, 10), 'h', 1 + i)
        tsf.update(engine, ts, 'fs-stripme', 'babar', insertion_date=pubdate)

    log = tsf.log(engine, 'fs-stripme')
    assert log == [
        {
            'author': 'babar',
            'date': pd.Timestamp('2017-01-01 00:00:00+0000', tz='UTC'),
            'meta': {},
            'rev': 1
        },
        {
            'author': 'babar',
            'date': pd.Timestamp('2017-01-02 00:00:00+0000', tz='UTC'),
            'meta': {},
            'rev': 2
        },
        {
            'author': 'babar',
            'date': pd.Timestamp('2017-01-03 00:00:00+0000', tz='UTC'),
            'meta': {},
            'rev': 3
        },
        {
            'author': 'babar',
            'date': pd.Timestamp('2017-01-04 00:00:00+0000', tz='UTC'),
            'meta': {},
            'rev': 4
        }
    ]

    h = tsf.history(engine, 'fs-stripme')
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

    idates = tsf.insertion_dates(engine, 'fs-stripme')
    assert len(idates) == 4

    with engine.begin() as cn:
        tsf.strip(cn, 'fs-stripme', datetime(2017, 1, 3))

    idates = tsf.insertion_dates(engine, 'fs-stripme')
    assert idates == [
        pd.Timestamp('2017-01-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2017-01-02 00:00:00+0000', tz='UTC')
    ]

    assert_hist("""
insertion_date             value_date         
2017-01-01 00:00:00+00:00  2017-01-10 00:00:00    0.0
                           2017-01-10 01:00:00    1.0
2017-01-02 00:00:00+00:00  2017-01-10 00:00:00    0.0
                           2017-01-10 01:00:00    1.0
                           2017-01-10 02:00:00    2.0
""", tsf.history(engine, 'fs-stripme'))

    assert_df("""
2017-01-10 00:00:00    0.0
2017-01-10 01:00:00    1.0
2017-01-10 02:00:00    2.0
""", tsf.get(engine, 'fs-stripme'))

    log = tsf.log(engine, 'fs-stripme')
    assert log == [
        {'author': 'babar',
         'date': pd.Timestamp('2017-01-01 00:00:00+0000', tz='UTC'),
         'meta': {},
         'rev': 1},
        {'author': 'babar',
         'date': pd.Timestamp('2017-01-02 00:00:00+0000', tz='UTC'),
         'meta': {},
         'rev': 2}
    ]


def test_rename(engine, tsf):
    ts = pd.Series(
        [1, 2, 3],
        index=pd.date_range(
            pd.Timestamp('2024-1-1', tz='utc'),
            periods=3,
            freq='D'
        )
    )
    tsf.update(
        engine,
        ts,
        'fs-rename',
        'Babar'
    )

    tsf.rename(engine, 'fs-rename', 'fs-renamed')

    assert not tsf.exists(engine, 'fs-rename')
    assert tsf.exists(engine, 'fs-renamed')

    assert tsf.get(engine, 'fs-rename') is None
    assert_df("""
2024-01-01 00:00:00+00:00    1.0
2024-01-02 00:00:00+00:00    2.0
2024-01-03 00:00:00+00:00    3.0
""", tsf.get(engine, 'fs-renamed'))


@pytest.fixture(scope='session')
def tsh1(engine):
    from tshistory import tsio, schema
    from tshistory.storage import FS1
    namespace = 'block1'
    schema.tsschema(namespace).create(engine, reset=True)
    dburi = 'postgresql://localhost:5433/postgres'
    datadir = Path(__file__).parent.parent / 'test' / 'data'
    datapath = datadir/namespace
    shutil.rmtree(datapath, ignore_errors=True)
    if not datapath.exists():
        datapath.mkdir()

    conf = (
        f'[dburi]\n'
        f'test = {dburi}\n'
        f'[storage]\n'
        f'test = filesystem1\n'
        f'test.path = {datadir/namespace}'
    )

    FS1._max_bucket_size = 1

    with tempconfig(conf.encode()):
        yield tsio.timeseriesfs1(namespace, None, uri=dburi)


def assert_nodes(engine, name, tsh, nodes):
    with engine.begin() as cn:
        cn.cache = {'series_path': {}}
        sto = FS1(cn, tsh, name)
        assert {
            idx: (node.parent, node.address, node.size)
            for idx, node in enumerate(sto.nodes(), start=1)
        } == nodes


def test_blocksize1(engine, tsh1):
    ts = pd.Series(
        [0, 1, 2, 3],
        index=pd.date_range(
            pd.Timestamp('2024-1-1', tz='utc'),
            periods=4,
            freq='h'
        )
    )
    ts.index = ts.index.tz_convert('Europe/Paris')
    tsh1.update(
        engine,
        ts,
        'fs-block1',
        'Babar',
    )

    assert_df("""
2024-01-01 00:00:00+00:00    0.0
2024-01-01 01:00:00+00:00    1.0
2024-01-01 02:00:00+00:00    2.0
2024-01-01 03:00:00+00:00    3.0
""", tsh1.get(engine, 'fs-block1'))

    assert_nodes(
        engine,
        'fs-block1',
        tsh1,
        {
            1: (0, 0, 29),
            2: (1, 29, 29),
            3: (2, 58, 29),
            4: (3, 87, 29)
        }
    )

    ts = pd.Series(
        [0, 1, 2, 3],
        index=pd.date_range(
            pd.Timestamp('2024-1-1 02:00:00', tz='utc'),
            periods=4,
            freq='h'
        )
    )
    tsh1.update(
        engine,
        ts,
        'fs-block1',
        'Babar',
    )

    assert_df("""
2024-01-01 00:00:00+00:00    0.0
2024-01-01 01:00:00+00:00    1.0
2024-01-01 02:00:00+00:00    0.0
2024-01-01 03:00:00+00:00    1.0
2024-01-01 04:00:00+00:00    2.0
2024-01-01 05:00:00+00:00    3.0
""", tsh1.get(engine, 'fs-block1'))


def test_small_steps(engine, tsh1):
    name = 'fs-smallsteps'
    ts = pd.Series(
        [0],
        [pd.Timestamp('2024-1-1', tz='utc')]
    )
    tsh1.update(
        engine,
        ts,
        name,
        'Babar',
    )

    assert_df("""
2024-01-01 00:00:00+00:00    0.0
""", tsh1.get(engine, name))

    # pure append update
    ts = pd.Series(
        [1],
        [pd.Timestamp('2024-1-2', tz='utc')]
    )
    tsh1.update(
        engine,
        ts,
        name,
        'Babar',
    )

    # looks good
    assert_df("""
2024-01-01 00:00:00+00:00    0.0
2024-01-02 00:00:00+00:00    1.0
""", tsh1.get(engine, name))

    assert_nodes(
        engine,
        name,
        tsh1,
        {
            1: (0, 0, 29),
            2: (1, 29, 29)
        }
    )

    # another pure append update
    ts = pd.Series(
        [2],
        [pd.Timestamp('2024-1-3', tz='utc')]
    )
    tsh1.update(
        engine,
        ts,
        name,
        'Babar',
    )

    # looks good
    assert_df("""
2024-01-01 00:00:00+00:00    0.0
2024-01-02 00:00:00+00:00    1.0
2024-01-03 00:00:00+00:00    2.0
""", tsh1.get(engine, name))

    assert_nodes(
        engine,
        name,
        tsh1,
        {
            1: (0, 0, 29),
            2: (1, 29, 29),
            3: (2, 58, 29)
        }
    )

    # edit second node
    ts = pd.Series(
        [11],
        [pd.Timestamp('2024-1-2', tz='utc')]
    )
    tsh1.update(
        engine,
        ts,
        name,
        'Babar',
    )

    assert_nodes(
        engine,
        name,
        tsh1,
        {
            1: (0, 0, 29),
            2: (1, 29, 29),
            3: (2, 58, 29),
            4: (1, 87, 29),
            5: (4, 116, 29)
        }
    )

    # looks good
    assert_df("""
2024-01-01 00:00:00+00:00     0.0
2024-01-02 00:00:00+00:00    11.0
2024-01-03 00:00:00+00:00     2.0
""", tsh1.get(engine, name))


def test_erasure_over_horizon(engine, tsh1):
    idate = utcdt(2018, 2, 1)
    ts = pd.Series(
        [1, 2, 3],
        index=pd.date_range(datetime(2018, 1, 1), freq='d', periods=3)
    )

    name = 'ersase_over_hz'
    tsh1.update(engine, ts, name, 'Babar',
               insertion_date=idate)

    # erase the rightmost part
    ts = pd.Series(
        [np.nan, np.nan, np.nan],
        index=pd.date_range(datetime(2018, 1, 3), freq='d', periods=3)
    )
    tsh1.update(engine, ts, name, 'Celeste',
               insertion_date=idate.replace(day=2),
               keepnans=True)

    # erase the leftmost part
    ts = pd.Series(
        [np.nan, np.nan, np.nan],
        index=pd.date_range(datetime(2017, 12, 30), freq='d', periods=3)
    )
    tsh1.update(engine, ts, name, 'Arthur',
               insertion_date=idate.replace(day=3),
               keepnans=True)


def test_long_name(engine, tsf):
    ts = pd.Series(
        [1., 2., 3.],
        index=pd.date_range(
            pd.Timestamp('2025-1-1', tz='utc'),
            periods=3,
            freq='D'
        )
    )

    name = 'aurélien' * 40  # should give us 320
    tsf.update(
        engine,
        ts,
        name,
        'Babar'
    )

    ts2 = tsf.get(engine, name)
    assert ts2.equals(ts)

    imeta = tsf.internal_metadata(engine, name)
    assert len(imeta['path']) == 229                  # as str
    assert len(imeta['path'].encode('utf-8')) == 255  # as bytes (which is the max we want)
