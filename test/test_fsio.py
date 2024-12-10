import pandas as pd

from tshistory.testutil import assert_df


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
        'Babar'
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
        'Babar'
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
        'Celeste'
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
