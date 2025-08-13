"""Database diagnostics for tshistory - index health checks"""

from collections import defaultdict


def get_expected_indexes(namespace='tsh'):
    """Hard-code the expected indexes based on our SQL files

    Returns dict of (table, columns) -> (index_name, index_type)
    where index_type is 'btree', 'gin', or 'gist'
    """
    # simple and explicit - no parsing needed
    return {
        # schema.sql
        ('basket', ('kind',)): (f'{namespace}_basket_kind_idx', 'btree'),
        ('ts_oldmeta', ('moment',)): (f'{namespace}_ts_oldmeta_moment_idx', 'btree'),
        ('ts_oldmeta', ('seriesid',)): (f'{namespace}_ts_oldmeta_seriesid_idx', 'btree'),
        ('tree', ('path',)): ('tree_path_idx', 'gist'),  # no namespace prefix, GIST for ltree
        ('tree_series_map', ('treeid',)): ('tree_series_map_idx', 'btree'),  # no namespace prefix

        # registry.sql
        ('registry', ('internal_metadata',)): (f'{namespace}_registry_internal_metadata_idx', 'gin'),
        ('registry', ('metadata',)): (f'{namespace}_registry_metadata_idx', 'gin'),
        ('revision_metadata', ('series',)): (f'{namespace}_revision_metadata_series_idx', 'btree'),

        # group.sql
        ('group_registry', ('name',)): (f'ix_{namespace}_group_registry_idx', 'btree'),
        ('group_registry', ('internal_metadata',)): (f'{namespace}_group_registry_internal_metadata_idx', 'gin'),
        ('group_registry', ('metadata',)): (f'{namespace}_group_registry_metadata_idx', 'gin'),
        ('groupmap', ('groupid',)): (f'ix_{namespace}_groupmap_group_idx', 'btree'),
        ('groupmap', ('seriesid',)): (f'ix_{namespace}_groupmap_series_idx', 'btree'),
        ('gr_oldmeta', ('moment',)): (f'{namespace}_gr_oldmeta_moment_idx', 'btree'),
        ('gr_oldmeta', ('groupid',)): (f'{namespace}_gr_oldmeta_groupid_idx', 'btree'),
    }


def get_actual_indexes(engine, namespace='tsh'):
    """Get all non-primary indexes from database"""
    query = """
    SELECT
        t.relname as table_name,
        i.relname as index_name,
        array_agg(a.attname ORDER BY array_position(ix.indkey, a.attnum)) as columns
    FROM pg_class t
    JOIN pg_namespace n ON n.oid = t.relnamespace
    JOIN pg_index ix ON t.oid = ix.indrelid
    JOIN pg_class i ON i.oid = ix.indexrelid
    JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
    WHERE n.nspname = %(namespace)s
      AND NOT ix.indisprimary  -- skip primary keys
      AND NOT ix.indisunique   -- skip unique constraints
    GROUP BY t.relname, i.relname
    ORDER BY t.relname, i.relname
    """

    indexes = []
    with engine.begin() as cn:
        for row in cn.execute(query, namespace=namespace).fetchall():
            indexes.append({
                'table': row.table_name,
                'name': row.index_name,
                'columns': tuple(row.columns)
            })
    return indexes


def find_issues(expected, actual):
    """Compare expected vs actual and find issues

    Returns dict with raw data about issues found
    """
    # group actual by (table, columns)
    by_columns = defaultdict(list)
    for idx in actual:
        key = (idx['table'], idx['columns'])
        by_columns[key].append(idx['name'])

    issues = {
        'duplicates': [],  # [(table, columns, [idx1, idx2, ...])]
        'wrong_name': [],  # [(table, columns, actual_name, expected_name)]
        'missing': [],     # [(table, columns, expected_name)]
    }

    # check each expected index
    for (table, columns), (expected_name, _index_type) in expected.items():
        actual_names = by_columns.get((table, columns), [])

        if not actual_names:
            issues['missing'].append((table, columns, expected_name))
        elif len(actual_names) > 1:
            issues['duplicates'].append((table, columns, actual_names))
            # also check if the name is wrong
            if expected_name not in actual_names:
                issues['wrong_name'].append((table, columns, actual_names[0], expected_name))
        elif actual_names[0] != expected_name:
            issues['wrong_name'].append((table, columns, actual_names[0], expected_name))

    return issues


def format_report(issues):
    """Format issues into human-readable report"""
    lines = []

    # count total issues
    total = len(issues['duplicates']) + len(issues['wrong_name']) + len(issues['missing'])

    if total == 0:
        return "✓ All indexes are correct!"

    lines.append(f"Found {total} index issues:\n")

    # duplicates
    if issues['duplicates']:
        lines.append(f"DUPLICATES ({len(issues['duplicates'])} tables with duplicates):")
        for table, columns, names in issues['duplicates']:
            col_str = ', '.join(columns)
            lines.append(f"  {table}({col_str}): {len(names)} indexes")
            lines.append(f"    {', '.join(names)}")
        lines.append("")

    # wrong names
    if issues['wrong_name']:
        lines.append(f"WRONG NAMES ({len(issues['wrong_name'])}):")
        for table, columns, actual, expected in issues['wrong_name']:
            col_str = ', '.join(columns)
            lines.append(f"  {table}({col_str}):")
            lines.append(f"    actual: {actual}")
            lines.append(f"    expected: {expected}")
        lines.append("")

    # missing
    if issues['missing']:
        lines.append(f"MISSING ({len(issues['missing'])}):")
        for table, columns, expected in issues['missing']:
            col_str = ', '.join(columns)
            lines.append(f"  {table}({col_str}): should have {expected}")

    return '\n'.join(lines)


def diagnose_indexes(engine, namespace='tsh'):
    """Main entry point - diagnose and return formatted report"""
    expected = get_expected_indexes(namespace)
    actual = get_actual_indexes(engine, namespace)
    issues = find_issues(expected, actual)
    return format_report(issues)


def ensure_expected_indexes(engine, namespace, indexes, dry_run=False):
    """Ensure all expected indexes exist with correct names

    This method:
    1. Creates any missing indexes with the correct type (gin, gist, btree)
    2. Renames existing indexes that have wrong names

    Returns list of SQL commands executed (or would execute if dry_run=True)
    """
    actual = get_actual_indexes(engine, namespace)

    # group actual indexes by (table, columns)
    by_columns = defaultdict(list)
    for idx in actual:
        key = (idx['table'], idx['columns'])
        by_columns[key].append(idx['name'])

    commands = []

    with engine.begin() as cn:
        for (table, columns), (expected_name, index_type) in indexes.items():
            # Check if table exists before trying to create index
            table_exists = cn.execute(
                'select exists ('
                '  select 1 from information_schema.tables '
                '  where table_schema = %(namespace)s '
                '  and table_name = %(table)s'
                ')',
                namespace=namespace,
                table=table
            ).scalar()

            if not table_exists:
                # Skip indexes for non-existent tables
                continue

            actual_names = by_columns.get((table, columns), [])

            if not actual_names:
                # missing - create it with the correct index type
                # properly quote all identifiers
                col_list = ', '.join(f'"{col}"' for col in columns)

                # use the explicit index type from our mapping
                if index_type == 'gin':
                    cmd = (
                        f'create index if not exists "{expected_name}" '
                        f'on "{namespace}"."{table}" using gin ({col_list})'
                    )
                elif index_type == 'gist':
                    cmd = (
                        f'create index if not exists "{expected_name}" '
                        f'on "{namespace}"."{table}" using gist ({col_list})'
                    )
                else:  # btree
                    cmd = (
                        f'create index if not exists "{expected_name}" '
                        f'on "{namespace}"."{table}" ({col_list})'
                    )

                commands.append(cmd)
                if not dry_run:
                    cn.execute(cmd)
                continue

            # check if expected name already exists
            if expected_name in actual_names:
                # already has correct name, nothing to rename
                continue

            # find the original index (typically the one without numeric suffix)
            # sort to get consistent ordering, pick first non-numbered one
            original = None
            for name in sorted(actual_names):
                # check if name ends with a digit (like idx1, idx2)
                if name and not name[-1].isdigit():
                    original = name
                    break

            # if no non-numbered index found, just use the first one
            if not original:
                original = sorted(actual_names)[0]

            # rename it to the expected name
            cmd = f'alter index "{namespace}"."{original}" rename to "{expected_name}"'
            commands.append(cmd)
            if not dry_run:
                cn.execute(cmd)

    return commands


def drop_duplicates(engine, namespace, indexes, dry_run=False):
    """Phase 2: Drop all duplicate indexes (keep only the expected ones)

    Returns list of SQL commands executed (or would execute if dry_run=True)
    """
    actual = get_actual_indexes(engine, namespace)

    # group actual indexes by (table, columns)
    by_columns = defaultdict(list)
    for idx in actual:
        key = (idx['table'], idx['columns'])
        by_columns[key].append(idx['name'])

    commands = []

    with engine.begin() as cn:
        for (table, columns), (expected_name, _index_type) in indexes.items():
            actual_names = by_columns.get((table, columns), [])

            # drop everything except the expected name
            for name in actual_names:
                if name != expected_name:
                    cmd = f'drop index if exists "{namespace}"."{name}"'
                    commands.append(cmd)
                    if not dry_run:
                        cn.execute(cmd)

    return commands


def fix_indexes(engine, namespace, indexes, dry_run=False):
    """Fix all index issues: ensure correct indexes then drop duplicates

    Convenience function that calls both phases.
    Returns list of all SQL commands executed.
    """
    commands = []
    commands.extend(ensure_expected_indexes(engine, namespace, indexes, dry_run))
    commands.extend(drop_duplicates(engine, namespace, indexes, dry_run))
    return commands
