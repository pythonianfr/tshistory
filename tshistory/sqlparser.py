from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import sqlglot


TSHISTORY_PATH = Path(__file__).parent
TSHISTORY_SQLFILES = (
    TSHISTORY_PATH / 'schema.sql',
    TSHISTORY_PATH / 'registry.sql',
    TSHISTORY_PATH / 'group.sql'
)


@dataclass
class Index:
    name: str
    schema: str
    table: str
    columns: Tuple[str, ...]
    type: str


def parse_indexes(sql_files: Sequence[Path], namespace: str = 'tsh') -> List[Index]:
    indexes = []

    for filepath in sql_files:
        sql_content = filepath.read_text().format(ns=namespace)

        for statement in sqlglot.parse(sql_content, dialect='postgres'):
            if not isinstance(statement, sqlglot.exp.Create) or statement.kind != 'INDEX':
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
                    columns=tuple(c.this.name for c in params.args['columns']),
                    type=using.name if using else 'btree'
                )
            )

    return indexes
