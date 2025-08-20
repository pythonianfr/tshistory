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


def parse_indexes(sql_files: List[Path], namespace: str = 'tsh') -> List[Index]:
    indexes = []
    
    for filepath in sql_files:
        if not filepath.exists():
            continue
            
        sql_content = filepath.read_text().format(ns=namespace)
        
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
