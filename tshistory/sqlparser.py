from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import sqlglot


@dataclass
class Index:
    name: str
    schema: str
    table: str
    columns: Tuple[str, ...]
    type: str


def parse_indexes(sql_content: str, namespace: str = 'tsh') -> List[Index]:
    sql_content = sql_content.format(ns=namespace)
    indexes = []

    for statement in sqlglot.parse(sql_content, dialect='postgres'):
        if not (isinstance(statement, sqlglot.exp.Create) and statement.kind == 'INDEX'):
            continue

        index = statement.this
        table_ref = index.args['table']
        params = index.args['params']
        using = params.args.get('using')

        indexes.append(
            Index(
                name=index.this.name,
                schema=namespace,
                table=table_ref.name,
                columns=tuple(c.this.name for c in params.args.get('columns', [])),
                type=using.name if using else 'btree'
            )
        )

    return indexes


def parse_sql_file(filepath: str, namespace: str) -> List[Index]:
    path = Path(filepath)
    if not path.exists():
        return []
    return parse_indexes(path.read_text(), namespace)


def tshistory_indexes(namespace: str = 'tsh') -> List[Index]:
    base_path = Path(__file__).parent
    indexes = []

    indexes.extend(parse_sql_file(str(base_path / 'schema.sql'), namespace))
    indexes.extend(parse_sql_file(str(base_path / 'registry.sql'), namespace))
    indexes.extend(parse_sql_file(str(base_path / 'group.sql'), namespace))

    indexes.extend(parse_sql_file(
        str(base_path / 'registry.sql'), f'{namespace}.group'
    ))

    return indexes
