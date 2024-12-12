import pandas as pd

from tshistory.testutil import assert_df
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

    sto = FS1(tsf.root, 'fs-multichunks')
    assert sto.tree_entries == 2


def test_two_chunks_two_revision(engine, tsf):
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
        'fs-2chunks2revs',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-1', tz='utc')
    )

    ts = ts * 2
    tsf.update(
        engine,
        ts,
        'fs-2chunks2revs',
        'Babar',
        insertion_date=pd.Timestamp('2024-1-2', tz='utc')
    )

    ts = tsf.get(engine, 'fs-2chunks2revs')
    assert len(ts) == 300

    sto = FS1(tsf.root, 'fs-2chunks2revs')
    assert sto.tree_entries == 4
