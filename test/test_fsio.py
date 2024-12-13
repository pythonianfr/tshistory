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

    ts = tsf.get(
        engine,
        'fs-first',
        revision_date=pd.Timestamp('2023-12-31', tz='UTC')
    )
    assert not len(ts)


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

    sto = FS1(tsf.root, 'fs-2chunks3revs')
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


def test_get_revision_date(engine, tsf):
    for i in range(5):
        ts = pd.Series(
            [i],
            index=[pd.Timestamp('2024-1-1', tz='utc')]
        )
        tsf.update(
            engine,
            ts,
            'fs-revdate',
            'Babar',
            insertion_date=pd.Timestamp(f'2024-1-{i+1}', tz='utc')
        )

    idates = tsf.insertion_dates(engine, 'fs-revdate')
    assert idates == [
        pd.Timestamp('2024-01-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-02 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-03 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-04 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-05 00:00:00+0000', tz='UTC')
    ]

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

    idates = tsf.insertion_dates(
        engine,
        'fs-revdate',
        from_insertion_date=pd.Timestamp('2023-12-31', tz='utc'),
        to_insertion_date=pd.Timestamp('2024-1-1', tz='utc')
    )
    assert idates == [
        pd.Timestamp('2024-01-01 00:00:00+0000', tz='UTC')
    ]

    idates = tsf.insertion_dates(
        engine,
        'fs-revdate',
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
        'fs-revdate',
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
        'fs-revdate',
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
        'fs-revdate',
        from_insertion_date=pd.Timestamp('2024-1-4', tz='utc'),
        to_insertion_date=pd.Timestamp('2024-1-6', tz='utc')
    )
    assert idates == [
        pd.Timestamp('2024-01-04 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-05 00:00:00+0000', tz='UTC')
    ]

    idates = tsf.insertion_dates(
        engine,
        'fs-revdate',
        from_insertion_date=pd.Timestamp('2024-1-4', tz='utc')
    )
    assert idates == [
        pd.Timestamp('2024-01-04 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-05 00:00:00+0000', tz='UTC')
    ]

    idates = tsf.insertion_dates(
        engine,
        'fs-revdate',
        to_insertion_date=pd.Timestamp('2024-1-2', tz='utc')
    )
    assert idates == [
        pd.Timestamp('2024-01-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2024-01-02 00:00:00+0000', tz='UTC')
    ]

    log = tsf.log(engine, 'fs-revdate')
    assert log == [
        {
            'date': pd.Timestamp('2024-01-01 00:00:00+0000', tz='UTC'),
            'author': 'Babar',
            'meta': {},
            'rev': 0
        },
        {
            'date': pd.Timestamp('2024-01-02 00:00:00+0000', tz='UTC'),
            'author': 'Babar',
            'meta': {},
            'rev': 1
        },
        {
            'date': pd.Timestamp('2024-01-03 00:00:00+0000', tz='UTC'),
            'author': 'Babar',
            'meta': {},
            'rev': 2
        },
        {
            'date': pd.Timestamp('2024-01-04 00:00:00+0000', tz='UTC'),
            'author': 'Babar',
            'meta': {},
            'rev': 3
        },
        {
            'date': pd.Timestamp('2024-01-05 00:00:00+0000', tz='UTC'),
            'author': 'Babar',
            'meta': {},
            'rev': 4
        }
    ]

    log = tsf.log(
        engine,
        'fs-revdate',
        fromdate=pd.Timestamp('2024-01-02', tz='utc'),
        todate=pd.Timestamp('2024-01-04', tz='utc'),
        limit=2
    )
    assert log == [
        {
            'author': 'Babar',
            'date': pd.Timestamp('2024-01-02 00:00:00+0000', tz='UTC'),
            'meta': {},
            'rev': 1
        },
        {
            'author': 'Babar',
            'date': pd.Timestamp('2024-01-03 00:00:00+0000', tz='UTC'),
            'meta': {},
            'rev': 2
        }
    ]
