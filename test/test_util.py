from datetime import datetime
import io
from pathlib import Path

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, assume, settings

from tshistory import tsio
from tshistory import dbdiag
from tshistory.migrate import do_fix_indexes
from tshistory.sqlparser import (
    Index,
    parse_indexes,
    TSHISTORY_SQLFILES
)
from tshistory.util import (
    bisect_search,
    diff,
    fromjson,
    infer_freq,
    objects,
    patch,
    patchmany,
    unflatten,
)
from tshistory.testutil import (
    assert_df,
    create_index_issues,
    genserie,
    tables,
    utcdt
)


def test_objects():
    objs = objects('migrator')
    assert len(objs) > 0  # tshistory provides the base one

    objs = objects('tshclass')
    assert len(objs) > 0  # tshistory provides the base one


def test_unflatten():
    d = {
        'a': 42,
        'b.c': 'hello',
        'b.d': 'world'
    }
    assert unflatten(d) == {
        'a': 42,
        'b': {
            'c': 'hello',
            'd': 'world'
        }
    }


def test_unflatten2():
    d  = {
        'a.uri': 'http://series.com/api',
        'a.pkce.clientid': '123zogzog1',
        'a.pkce.scope': 'all.the.series'
    }
    u = unflatten(d)
    assert u == {
        'a':  {
            'pkce.clientid': '123zogzog1',
            'pkce.scope': 'all.the.series',
            'uri': 'http://series.com/api'
        }
    }


def test_patch():
    s1 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=4)
    )
    s2 = pd.Series(
        [12., 13., np.nan, 15.],
        index=pd.date_range(datetime(2020, 1, 1, 1), freq='h', periods=4)
    )
    p = patch(s1, s2)
    assert_df("""
2020-01-01 00:00:00     1.0
2020-01-01 01:00:00    12.0
2020-01-01 02:00:00    13.0
2020-01-01 03:00:00     NaN
2020-01-01 04:00:00    15.0
""", p)

    p2 = patchmany((s1, s2))
    assert_df("""
2020-01-01 00:00:00     1.0
2020-01-01 01:00:00    12.0
2020-01-01 02:00:00    13.0
2020-01-01 03:00:00     NaN
2020-01-01 04:00:00    15.0
""", p2)

    s3 = pd.Series([], dtype=s1.dtype)
    p = patch(s3, s3)
    assert len(p) == 0

    p3 = patchmany([p2])
    assert_df("""
2020-01-01 00:00:00     1.0
2020-01-01 01:00:00    12.0
2020-01-01 02:00:00    13.0
2020-01-01 03:00:00     NaN
2020-01-01 04:00:00    15.0
""", p3)


def test_patch_tzaware():
    s1 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(utcdt(2020, 6, 23, 22), freq='h', periods=4)
    )
    s2 = pd.Series(
        [3.1, 4., 5.],
        index=pd.date_range(utcdt(2020, 6, 24), freq='h', periods=3)
    )
    p = patch(s1, s2)
    assert_df("""
2020-06-23 22:00:00+00:00    1.0
2020-06-23 23:00:00+00:00    2.0
2020-06-24 00:00:00+00:00    3.1
2020-06-24 01:00:00+00:00    4.0
2020-06-24 02:00:00+00:00    5.0
""", p)

    assert s1.index.dtype.name == 'datetime64[ns, UTC]'
    assert s2.index.dtype.name == 'datetime64[ns, UTC]'
    assert p.index.dtype.name == 'datetime64[ns, UTC]'

    p2 = patchmany([s1, s2])
    assert_df("""
2020-06-23 22:00:00+00:00    1.0
2020-06-23 23:00:00+00:00    2.0
2020-06-24 00:00:00+00:00    3.1
2020-06-24 01:00:00+00:00    4.0
2020-06-24 02:00:00+00:00    5.0
""", p2)

    assert p2.index.dtype.name == 'datetime64[ns, UTC]'


def test_patch_one_empty():
    s1 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=4)
    )
    s2 = pd.Series(dtype='float64')
    p = patch(s1, s2)
    assert_df("""
2020-01-01 00:00:00    1.0
2020-01-01 01:00:00    2.0
2020-01-01 02:00:00    3.0
2020-01-01 03:00:00    4.0
""", p)


def test_patch_empty_one():
    s1 = pd.Series(dtype='float64')
    s2 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=4)
    )
    p = patch(s1, s2)
    assert_df("""
2020-01-01 00:00:00    1.0
2020-01-01 01:00:00    2.0
2020-01-01 02:00:00    3.0
2020-01-01 03:00:00    4.0
""", p)


def test_float_patchmany():
    s1 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=4)
    )
    s2 = pd.Series(
        [12., 13., np.nan, 15.],
        index=pd.date_range(datetime(2020, 1, 1, 1), freq='h', periods=4)
    )
    s3 = pd.Series(
        [0., 1., 2., 13., ],
        index=pd.date_range(datetime(2019, 12, 31, 23), freq='h', periods=4)
    )
    p = patchmany([s1, s2, s3])
    assert_df("""
2019-12-31 23:00:00     0.0
2020-01-01 00:00:00     1.0
2020-01-01 01:00:00     2.0
2020-01-01 02:00:00    13.0
2020-01-01 03:00:00     NaN
2020-01-01 04:00:00    15.0
""", p)

    s4 = pd.Series([], dtype=s1.dtype)
    p = patchmany([s4, s4, s4])
    assert len(p) == 0

    p = patchmany([s4, s1, s4])
    assert_df("""
2020-01-01 00:00:00    1.0
2020-01-01 01:00:00    2.0
2020-01-01 02:00:00    3.0
2020-01-01 03:00:00    4.0
""", p)


def test_string_patchmany():
    s1 = pd.Series(
        ['a', 'b', 'c', 'd'],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=4)
    )
    s2 = pd.Series(
        ['bb', 'cc', None, 'ee'],
        index=pd.date_range(datetime(2020, 1, 1, 1), freq='h', periods=4)
    )
    s3 = pd.Series(
        ['Z', 'a', 'b', 'cc'],
        index=pd.date_range(datetime(2019, 12, 31, 23), freq='h', periods=4)
    )
    p = patchmany([s1, s2, s3])
    assert_df("""
2019-12-31 23:00:00       Z
2020-01-01 00:00:00       a
2020-01-01 01:00:00       b
2020-01-01 02:00:00      cc
2020-01-01 03:00:00    None
2020-01-01 04:00:00      ee
""", p)


def test_diff():
    s1 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=4)
    )
    s2 = pd.Series(
        [12., 13., np.nan, 15.],
        index=pd.date_range(datetime(2020, 1, 1, 1), freq='h', periods=4)
    )
    s2[datetime(2019, 12, 31, 23)] = -1

    ds1s2 = diff(s1, s2)
    # tail ends come as new items
    # middle elements as updates
    assert_df("""
2019-12-31 23:00:00    -1.0
2020-01-01 01:00:00    12.0
2020-01-01 02:00:00    13.0
2020-01-01 03:00:00     NaN
2020-01-01 04:00:00    15.0
""", ds1s2)

    ds2s1 = diff(s2, s1)
    # only updates there
    assert_df("""
2020-01-01 00:00:00    1.0
2020-01-01 01:00:00    2.0
2020-01-01 02:00:00    3.0
2020-01-01 03:00:00    4.0
""", ds2s1)


def test_diff_nan_pure():
    s1 = pd.Series(
        [1, np.nan, 3],
        index=pd.date_range(datetime(2024, 1, 1), freq='h', periods=3)
    )
    s2 = pd.Series(
        [1, np.nan, 3, 4],
        index=pd.date_range(datetime(2024, 1, 1), freq='h', periods=4)
    )
    d = diff(s1, s2)
    # looks good
    assert_df("""
2024-01-01 03:00:00    4.0
""", d)

    s3 = pd.Series(
        [1, np.nan, np.nan, 4],
        index=pd.date_range(datetime(2024, 1, 1), freq='h', periods=4)
    )
    d = diff(s2, s3)
    # still good
    assert_df("""
2024-01-01 02:00:00   NaN
""", d)


def test_nan_mixed():
    n1 = pd.Series(
        [np.nan],
        index=pd.date_range(datetime(2024, 1, 1), freq='h', periods=1)
    )
    n2 = pd.Series(
        [np.nan, np.nan, 3],
        index=pd.date_range(datetime(2024, 1, 1), freq='h', periods=3)
    )
    d = diff(n1, n2)
    assert_df("""
2024-01-01 01:00:00    NaN
2024-01-01 02:00:00    3.0
""", d)

    n1 = pd.Series(
        [np.nan, 2, 3],
        index=pd.date_range(datetime(2024, 1, 1), freq='h', periods=3)
    )
    n1 = n1.drop(n1.index[1])  # erase the '2' point
    n2 = pd.Series(
        [np.nan, np.nan, 3],
        index=pd.date_range(datetime(2024, 1, 1), freq='h', periods=3)
    )
    d = diff(n1, n2)
    assert_df("""
2024-01-01 01:00:00   NaN
""", d)


def test_diff_duplicated():
    # with a duplicated row (left)
    s1 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=4)
    )
    dupe = pd.Series([5.], index=[datetime(2020, 1, 1, 3)])
    s1 = pd.concat([s1, dupe])
    s2 = pd.Series(
        [1., 2., 42., 4., .5],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=5)
    )
    with pytest.raises(ValueError):
        diff(s1, s2)


def test_infer_freq():
    s1 = pd.Series(
        [1., 2., 3., 4.],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=4)
    )
    d, q = infer_freq(s1)
    assert d == pd.Timedelta(hours=1)
    assert q == 1

    s2 = pd.Series(
        [1., 2., 3., None, 5.],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=5)
    )
    d, q = infer_freq(s2.dropna())
    assert d == pd.Timedelta(hours=1)
    assert q == 0.6666666666666666

    s3 = pd.Series(
        [1.],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=1)
    )
    with pytest.raises(AssertionError):
        infer_freq(s3)


def test_json():
    series = pd.Series(
        [1., 2., 3.],
        index=pd.date_range(datetime(2020, 1, 1), freq='h', periods=3)
    )
    jsonseries = series.to_json(date_format='iso')
    assert jsonseries == (
        '{"2020-01-01T00:00:00.000":1.0,'
        '"2020-01-01T01:00:00.000":2.0,'
        '"2020-01-01T02:00:00.000":3.0}'
    )

    series2 = pd.read_json(io.StringIO(jsonseries), typ='series', dtype=False)
    assert not getattr(series2.index.dtype, 'tz', False)
    assert series.equals(series2)

    series2 = fromjson(jsonseries, 'foo', tzaware=False)
    assert not getattr(series2.index.dtype, 'tz', False)
    assert series.equals(series2)


def test_bisect():
    values = [-4, -2, 1, 7]
    assert bisect_search(values, -5) == -1
    assert bisect_search(values, -4) == 0
    assert bisect_search(values, -3) == 0
    assert bisect_search(values, 0) == 1
    assert bisect_search(values, 1) == 2
    assert bisect_search(values, 3) == 2
    assert bisect_search(values, 7) == 3
    assert bisect_search(values, 8) == 4


def test_tables(engine, pure):
    with engine.begin() as cn:
        assert tables(cn) == [
            ('pure', 'basket'),
            ('pure', 'gr_oldmeta'),
            ('pure', 'group_registry'),
            ('pure', 'groupmap'),
            ('pure', 'registry'),
            ('pure', 'revision_metadata'),
            ('pure', 'tree'),
            ('pure', 'tree_series_map'),
            ('pure', 'ts_oldmeta'),
            ('pure-kvstore', 'kvstore'),
            ('pure-kvstore', 'things'),
            ('pure-kvstore', 'version'),
            ('pure-kvstore', 'vkvstore'),
            ('pure.group', 'registry'),
            ('pure.group', 'revision_metadata'),
        ]


def test_in_tx(tsh, engine):
    assert tsh.type(engine, 'foo') == 'primary'

    ts = genserie(datetime(2017, 10, 28, 23),
                  'h', 4, tz='UTC')
    with engine.begin() as cn:
        tsh.update(cn, ts, 'test_tx', 'Babar')


def test_timeseries_repr(tsh):
    if isinstance(tsh, tsio.timeseries):
        assert repr(tsh) == f'tsio.timeseries({tsh.namespace},othersources=None)'


def test_fix_missing_indexes(tsp, engine):
    """Test that we can create missing indexes with correct types"""
    # 1. Drop some indexes to simulate missing ones
    with engine.begin() as cn:
        # Drop a regular btree index
        cn.execute('DROP INDEX tsh.tsh_basket_kind_idx')

        # Drop a GIN index (for JSONB)
        cn.execute('DROP INDEX tsh.tsh_registry_metadata_idx')

        # Drop a GIST index (for ltree)
        cn.execute('DROP INDEX tsh.tsh_tree_path_idx')

        # Drop a foreign key index
        cn.execute('DROP INDEX tsh.tsh_ts_oldmeta_seriesid_idx')

    # 2. Verify they're missing
    report = dbdiag.diagnose_indexes(engine, 'tsh')
    assert 'MISSING' in report
    assert 'basket(kind)' in report
    assert 'registry(metadata)' in report
    assert 'tree(path)' in report
    assert 'ts_oldmeta(seriesid)' in report

    # 3. Fix the missing indexes
    expected_indexes = parse_indexes(TSHISTORY_SQLFILES, 'tsh')
    dbdiag.fix_indexes(engine, 'tsh', expected_indexes)

    # 4. Verify all fixed
    report = dbdiag.diagnose_indexes(engine, 'tsh')
    assert '✓ All indexes are correct!' in report

    # 5. Verify the indexes actually exist and work
    with engine.begin() as cn:
        # Check they exist in pg_indexes
        result = cn.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE schemaname='tsh'
            AND indexname IN ('tsh_basket_kind_idx', 'tsh_registry_metadata_idx',
                              'tsh_tree_path_idx', 'tsh_ts_oldmeta_seriesid_idx')
            ORDER BY indexname
        """).fetchall()

        assert len(result) == 4

        # Verify index types in definitions
        for name, definition in result:
            if name == 'tsh_registry_metadata_idx':
                assert 'gin' in definition.lower()
            elif name == 'tree_path_idx':
                assert 'gist' in definition.lower()
            # btree is default, may not appear in definition


# index testing is now available from testutil.create_index_issues


@given(
    to_duplicate_idx=st.sets(st.integers(0, 11), max_size=10),  # 12 indexes total (0-11)
    to_drop_idx=st.sets(st.integers(0, 11), max_size=10),
    to_misname_idx=st.sets(st.integers(0, 11), max_size=10)
)
@settings(max_examples=20, deadline=10000)
def test_migration_fix_indexes_with_hypothesis(
        tsp, engine, to_duplicate_idx, to_drop_idx, to_misname_idx
):
    """property-based test that migration handles any combination of index issues"""
    # ensure no overlap between drop and misname (can't rename a dropped index)
    assume(not (to_drop_idx & to_misname_idx))

    expected_indexes = parse_indexes(TSHISTORY_SQLFILES, 'tsh')

    # Create list of (table, columns) tuples for test manipulation
    all_index_keys = [(idx.table, idx.columns) for idx in expected_indexes]

    # convert indices to actual index tuples
    to_duplicate = [all_index_keys[i] for i in to_duplicate_idx]
    to_drop = [all_index_keys[i] for i in to_drop_idx]
    to_misname = [all_index_keys[i] for i in to_misname_idx]

    # diagnose initial clean state
    initial_report = dbdiag.diagnose_indexes(engine, 'tsh')
    assert '✓ All indexes are correct!' in initial_report

    create_index_issues(
        engine, 'tsh', to_duplicate, to_drop, to_misname, expected_indexes
    )

    issues_report = dbdiag.diagnose_indexes(engine, 'tsh')

    if to_duplicate or to_drop or to_misname:
        assert '✓ All indexes are correct!' not in issues_report

    do_fix_indexes(
        engine, 'tsh', interactive=False, indexes=expected_indexes
    )

    fixed_report = dbdiag.diagnose_indexes(engine, 'tsh')
    assert '✓ All indexes are correct!' in fixed_report

    do_fix_indexes(
        engine, 'tsh', interactive=False, indexes=expected_indexes
    )
    idempotent_report = dbdiag.diagnose_indexes(engine, 'tsh')
    assert idempotent_report == fixed_report

    # property 3: all expected indexes should exist with correct names and types
    actual = dbdiag.get_actual_indexes(engine, 'tsh')

    for expected_idx in expected_indexes:
        # Match by table, columns AND type (e.g., GIST vs btree on same column)
        matching = [idx for idx in actual
                   if idx.table == expected_idx.table
                   and idx.columns == expected_idx.columns
                   and idx.type == expected_idx.type]
        assert len(matching) == 1, f'expected exactly one {expected_idx.type} index for {expected_idx.table}({expected_idx.columns})'
        assert matching[0].name == expected_idx.name, f'wrong name for {expected_idx.type} index on {expected_idx.table}({expected_idx.columns})'


def test_sql_parser(tmp_path):
    sql_file = tmp_path / "test.sql"
    sql_file.write_text("""
    create index "test_idx" on "{ns}".mytable (col1);
    create index if not exists "test_gin_idx" on "{ns}".mytable using gin (metadata);
    create index multi_col_idx on "{ns}".mytable (col1, col2, col3);
    """)

    indexes = parse_indexes([sql_file], 'myns')

    assert indexes == [
        Index('test_idx', 'myns', 'mytable', ('col1',), 'btree'),
        Index('test_gin_idx', 'myns', 'mytable', ('metadata',), 'gin'),
        Index('multi_col_idx', 'myns', 'mytable',
              ('col1', 'col2', 'col3'), 'btree')
    ]


def test_sql_parser_namespace_replacement(tmp_path):
    sql_file = tmp_path / "test.sql"
    sql_file.write_text("""
    create index "{ns}_basket_kind_idx" on "{ns}".basket (kind);
    create index "{ns}_registry_metadata_idx" on "{ns}".registry using gin(metadata);
    """)

    indexes = parse_indexes([sql_file], 'pure')

    assert indexes == [
        Index('pure_basket_kind_idx', 'pure', 'basket', ('kind',), 'btree'),
        Index('pure_registry_metadata_idx', 'pure', 'registry',
              ('metadata',), 'gin')
    ]


def test_sql_parser_actual_files():
    tshistory_path = Path(__file__).parent.parent / 'tshistory'

    indexes = parse_indexes([
        tshistory_path / 'schema.sql',
        tshistory_path / 'registry.sql',
        tshistory_path / 'group.sql'
    ], 'tsh')

    assert len(indexes) > 10

    basket_idx = [idx for idx in indexes if idx.name == 'tsh_basket_kind_idx']
    assert len(basket_idx) == 1
    assert basket_idx[0] == Index(
        'tsh_basket_kind_idx', 'tsh', 'basket', ('kind',), 'btree'
    )

    tree_idx = [idx for idx in indexes if idx.name == 'tsh_tree_path_idx']
    assert len(tree_idx) == 1
    assert tree_idx[0] == Index(
        'tsh_tree_path_idx', 'tsh', 'tree', ('path',), 'gist'
    )

    gin_indexes = [idx for idx in indexes if idx.type == 'gin']
    assert len(gin_indexes) >= 4  # at least 4 GIN indexes


def test_parsed_indexes_match_database(tsp, engine):
    """Test that SQL parser correctly identifies indexes created by schema"""
    report = dbdiag.diagnose_indexes(engine, 'tsh')
    assert '✓ All indexes are correct!' in report


def test_auth_migration_bug(engine):
    from pathlib import Path
    import dbcache
    from dbcache.schema import init

    init(engine, 'auth', drop=True)

    with engine.begin() as cn:
        # simulate the exact live situation with multiple wrong-named duplicates
        cn.execute('drop index "auth"."auth_version_idate_idx"')
        cn.execute('drop index "auth"."auth_version_objid_idx"')

        # create the wrong indexes as they exist in production
        cn.execute('create index "version_idate_idx" ON "auth".version(idate)')
        cn.execute('create index "version_idate_idx1" ON "auth".version(idate)')
        cn.execute('create index "version_idate_idx2" ON "auth".version(idate)')
        cn.execute('create index "version_objid_idx" ON "auth".version(objid)')
        cn.execute('create index "version_objid_idx1" ON "auth".version(objid)')

    actual_before = dbdiag.get_actual_indexes(engine, 'auth')

    # Check initial state has wrong-named indexes
    wrong_before = [idx.name for idx in actual_before
                    if idx.table == 'version' and not idx.name.startswith('auth_')]
    assert 'version_idate_idx' in wrong_before
    assert 'version_idate_idx1' in wrong_before
    assert 'version_idate_idx2' in wrong_before
    assert 'version_objid_idx' in wrong_before
    assert 'version_objid_idx1' in wrong_before

    # verify unique indexes are now included
    unique_indexes = [idx for idx in actual_before if 'key' in idx.name]
    assert len(unique_indexes) > 0, "get_actual_indexes should now include unique indexes"

    # run what the migration runs
    dbcache_sql = Path(dbcache.__file__).parent / 'schema.sql'
    expected = parse_indexes([dbcache_sql], 'auth')

    from tshistory.migrate import do_fix_indexes
    do_fix_indexes(engine, 'auth', False, expected)

    actual_after = dbdiag.get_actual_indexes(engine, 'auth')

    # verify indexes are fixed
    version_indexes = [idx for idx in actual_after if idx.table == 'version']
    assert any(idx.name == 'auth_version_idate_idx' for idx in version_indexes)
    assert any(idx.name == 'auth_version_objid_idx' for idx in version_indexes)

    # should not have wrong-named indexes anymore (except constraint-backed ones)
    wrong_after = [idx.name for idx in actual_after
                   if idx.table == 'version' and not idx.name.startswith('auth_')
                   and not idx.name.endswith('_key')]  # Exclude constraint-backed indexes
    assert len(wrong_after) == 0

    # check that we can rename unique indexes
    with engine.begin() as cn:
        cn.execute('ALTER INDEX auth.things_key_key RENAME TO auth_things_key_key')

        result = cn.execute("""
            SELECT i.relname FROM pg_class i
            JOIN pg_namespace n ON n.oid = i.relnamespace
            WHERE i.relname = 'auth_things_key_key' AND n.nspname = 'auth'
        """).fetchone()
        assert result is not None, "Rename of unique constraint index failed!"

    # verify the renamed index is reflected
    actual_final = dbdiag.get_actual_indexes(engine, 'auth')
    final_names = [idx.name for idx in actual_final]
    assert 'auth_things_key_key' in final_names
    assert 'things_key_key' not in final_names


def test_find_wrong_indexes(engine):
    import dbcache
    from pathlib import Path
    from dbcache.schema import init
    from tshistory.migrate import do_fix_indexes

    init(engine, 'test_wrong', drop=True)

    with engine.begin() as cn:
        cn.execute('drop index "test_wrong"."test_wrong_version_idate_idx"')
        cn.execute('drop index "test_wrong"."test_wrong_version_objid_idx"')
        cn.execute('create index "version_idate_idx" ON "test_wrong".version(idate)')
        cn.execute('create index "version_objid_idx" ON "test_wrong".version(objid)')

        result = cn.execute("""
            SELECT i.relname
            FROM pg_class i
            JOIN pg_namespace n ON n.oid = i.relnamespace
            JOIN pg_index ix ON i.oid = ix.indexrelid
            JOIN pg_class t ON t.oid = ix.indrelid
            WHERE n.nspname = 'test_wrong'
            AND i.relname IN ('version_idate_idx', 'version_objid_idx')
        """).fetchall()

        assert len(result) == 2, "Should have 2 wrongly named indexes"

    dbcache_sql = Path(dbcache.__file__).parent / 'schema.sql'
    expected = parse_indexes([dbcache_sql], 'test_wrong')
    do_fix_indexes(engine, 'test_wrong', False, expected)

    with engine.begin() as cn:
        result = cn.execute("""
            SELECT i.relname
            FROM pg_class i
            JOIN pg_namespace n ON n.oid = i.relnamespace
            WHERE n.nspname = 'test_wrong'
            AND i.relname LIKE 'test_wrong_%'
            ORDER BY i.relname
        """).fetchall()

        names = [r[0] for r in result]
        assert 'test_wrong_version_idate_idx' in names
        assert 'test_wrong_version_objid_idx' in names
        assert 'version_idate_idx' not in names
        assert 'version_objid_idx' not in names


def test_indexes_created_in_correct_schema(engine):
    """Test that fix_indexes handles indexes correctly across schemas"""
    from pathlib import Path
    import dbcache
    from dbcache.schema import init
    from tshistory.migrate import do_fix_indexes

    init(engine, 'authtest', drop=True)

    dbcache_sql = Path(dbcache.__file__).parent / 'schema.sql'
    expected = parse_indexes([dbcache_sql], 'authtest')

    initial_indexes = dbdiag.get_actual_indexes(engine, 'authtest')
    initial_names = [idx.name for idx in initial_indexes]

    for name in initial_names:
        if not name.endswith('_key'):  # skip constraint-backed indexes
            assert name.startswith('authtest_'), f"Index {name} lacks namespace prefix"

    do_fix_indexes(engine, 'authtest', False, expected)

    final_indexes = dbdiag.get_actual_indexes(engine, 'authtest')
    final_names = [idx.name for idx in final_indexes]

    assert set(initial_names) == set(final_names)
