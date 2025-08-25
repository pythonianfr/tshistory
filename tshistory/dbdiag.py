from collections import defaultdict

from tshistory.sqlparser import (
    Index,
    parse_indexes,
    TSHISTORY_SQLFILES
)


def get_actual_indexes(engine, namespace='tsh'):
    """Get all non-primary indexes from database"""
    query = """
    SELECT
        t.relname as table_name,
        i.relname as index_name,
        am.amname as index_type,
        array_agg(a.attname ORDER BY array_position(ix.indkey, a.attnum)) as columns
    FROM pg_class t
    JOIN pg_namespace n ON n.oid = t.relnamespace
    JOIN pg_index ix ON t.oid = ix.indrelid
    JOIN pg_class i ON i.oid = ix.indexrelid
    JOIN pg_am am ON am.oid = i.relam
    JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
    WHERE n.nspname = %(namespace)s
      AND NOT ix.indisprimary  -- skip primary keys
    GROUP BY t.relname, i.relname, am.amname
    ORDER BY t.relname, i.relname
    """

    indexes = []
    with engine.begin() as cn:
        for row in cn.execute(query, namespace=namespace).fetchall():
            indexes.append(
                Index(
                    name=row.index_name,
                    schema=namespace,
                    table=row.table_name,
                    columns=tuple(row.columns),
                    type=row.index_type
                )
            )
    return indexes


def find_issues(expected, actual):
    # group actual by (table, columns) to find duplicates
    by_columns = defaultdict(list)
    for idx in actual:
        key = (idx.table, idx.columns)
        by_columns[key].append(idx)

    issues = {
        'duplicates': [],
        'wrong_name': [],
        'missing': [],
    }

    # check each expected index
    for expected_idx in expected:
        matching = by_columns.get((expected_idx.table, expected_idx.columns), [])

        same_type = [m for m in matching if m.type == expected_idx.type]

        if not same_type:
            issues['missing'].append(
                (expected_idx.table, expected_idx.columns, expected_idx.name)
            )
        elif len(same_type) > 1:
            actual_names = [m.name for m in same_type]
            issues['duplicates'].append(
                (expected_idx.table, expected_idx.columns, actual_names)
            )
        elif same_type[0].name != expected_idx.name:
            issues['wrong_name'].append(
                (expected_idx.table, expected_idx.columns,
                 same_type[0].name, expected_idx.name)
            )

    return issues


def format_report(issues):
    lines = []
    total = len(issues['duplicates']) + len(issues['wrong_name']) + len(issues['missing'])

    if total == 0:
        return "✓ All indexes are correct!"

    lines.append(f"Found {total} index issues:\n")

    if issues['duplicates']:
        lines.append(f"DUPLICATES ({len(issues['duplicates'])} tables with duplicates):")
        for table, columns, names in issues['duplicates']:
            col_str = ', '.join(columns)
            lines.append(f"  {table}({col_str}): {len(names)} indexes")
            lines.append(f"    {', '.join(names)}")
        lines.append("")

    if issues['wrong_name']:
        lines.append(f"WRONG NAMES ({len(issues['wrong_name'])}):")
        for table, columns, actual, expected in issues['wrong_name']:
            col_str = ', '.join(columns)
            lines.append(f"  {table}({col_str}):")
            lines.append(f"    actual: {actual}")
            lines.append(f"    expected: {expected}")
        lines.append("")

    if issues['missing']:
        lines.append(f"MISSING ({len(issues['missing'])}):")
        for table, columns, expected in issues['missing']:
            col_str = ', '.join(columns)
            lines.append(f"  {table}({col_str}): should have {expected}")

    return '\n'.join(lines)


def diagnose_indexes(engine, namespace='tsh'):
    expected = parse_indexes(TSHISTORY_SQLFILES, namespace)
    actual = get_actual_indexes(engine, namespace)
    issues = find_issues(expected, actual)
    return format_report(issues)


# fixing part

def ensure_expected_indexes(engine, namespace, indexes):
    actual = get_actual_indexes(engine, namespace)

    # group actual indexes by (table, columns, type)
    # This allows different index types on same columns (e.g. btree + gist)
    by_columns_and_type = defaultdict(list)
    for idx in actual:
        key = (idx.table, idx.columns, idx.type)
        by_columns_and_type[key].append(idx)

    commands = []

    for idx in indexes:
        matching = by_columns_and_type.get((idx.table, idx.columns, idx.type), [])

        if not matching:
            # missing - create it with the correct index type
            # properly quote all identifiers
            col_list = ', '.join(f'"{col}"' for col in idx.columns)

            # use the explicit index type from our mapping
            if idx.type == 'gin':
                cmd = (
                    f'create index if not exists "{idx.name}" '
                    f'on "{namespace}"."{idx.table}" using gin ({col_list})'
                )
            elif idx.type == 'gist':
                cmd = (
                    f'create index if not exists "{idx.name}" '
                    f'on "{namespace}"."{idx.table}" using gist ({col_list})'
                )
            else:  # btree
                cmd = (
                    f'create index if not exists "{idx.name}" '
                    f'on "{namespace}"."{idx.table}" ({col_list})'
                )

            commands.append(cmd)
            continue

        # check if expected name already exists
        if any(m.name == idx.name for m in matching):
            # already has correct name, nothing to rename
            continue

        # just pick the first one to rename (others will be dropped)
        original = matching[0].name

        # rename it to the expected name
        cmd = f'alter index "{namespace}"."{original}" rename to "{idx.name}"'
        commands.append(cmd)

    with engine.begin() as cn:
        for cmd in commands:
            cn.execute(cmd)


def drop_duplicates(engine, namespace, indexes):
    actual = get_actual_indexes(engine, namespace)

    # group actual indexes by (table, columns)
    by_columns = defaultdict(list)
    for idx in actual:
        key = (idx.table, idx.columns)
        by_columns[key].append(idx)

    with engine.begin() as cn:
        result = cn.execute("""
            SELECT i.relname
            FROM pg_index ix
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_namespace n ON n.oid = i.relnamespace
            JOIN pg_constraint c ON c.conindid = i.oid
            WHERE n.nspname = %(namespace)s
        """, namespace=namespace).fetchall()

        constraint_backed = {row[0] for row in result}

    commands = []

    for idx in indexes:
        matching = by_columns.get((idx.table, idx.columns), [])

        same_type = [m for m in matching if m.type == idx.type]

        if len(same_type) > 1:
            has_expected = any(m.name == idx.name for m in same_type)
            has_constraint = any(m.name in constraint_backed for m in same_type)

            if has_expected and has_constraint:
                commands.append(f'drop index if exists "{namespace}"."{idx.name}"')

            for m in same_type:
                if m.name != idx.name:
                    if m.name in constraint_backed:
                        commands.append(
                            f'alter index "{namespace}"."{m.name}" rename to "{idx.name}"'
                        )
                    else:
                        commands.append(
                            f'drop index if exists "{namespace}"."{m.name}"'
                        )

    with engine.begin() as cn:
        for cmd in commands:
            cn.execute(cmd)


def fix_indexes(engine, namespace, indexes):
    """Fix all index issues: ensure correct indexes then drop duplicates

    Convenience function that calls both phases.
    """
    ensure_expected_indexes(engine, namespace, indexes)
    drop_duplicates(engine, namespace, indexes)
