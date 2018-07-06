
# sqlalchemy patch

from sqlalchemy.sql import elements, expression
from sqlalchemy.dialects.postgresql.base import PGDDLCompiler

elements.NONE_NAME = elements._NONE_NAME


def visit_create_index(self, create):
    preparer = self.preparer
    index = create.element
    self._verify_index_table(index)
    text = "CREATE "
    if index.unique:
        text += "UNIQUE "
    text += "INDEX "

    if self.dialect._supports_create_index_concurrently:
        concurrently = index.dialect_options['postgresql']['concurrently']
        if concurrently:
            text += "CONCURRENTLY "

    # PATCH
    if index.name is None or index.name is elements.NONE_NAME:
        # -> no name
        text += "ON %s" % preparer.format_table(index.table)
    else:
        text += "%s ON %s " % (
            self._prepared_index_name(index,
                                      include_schema=False),
            preparer.format_table(index.table)
        )
    # /PATCH

    using = index.dialect_options['postgresql']['using']
    if using:
        text += "USING %s " % preparer.quote(using)

    ops = index.dialect_options["postgresql"]["ops"]
    text += "(%s)" \
            % (
                ', '.join([
                    self.sql_compiler.process(
                        expr.self_group()
                        if not isinstance(expr, expression.ColumnClause)
                        else expr,
                        include_table=False, literal_binds=True) +
                    (
                        (' ' + ops[expr.key])
                        if hasattr(expr, 'key')
                        and expr.key in ops else ''
                    )
                    for expr in index.expressions
                ])
            )

    withclause = index.dialect_options['postgresql']['with']

    if withclause:
        text += " WITH (%s)" % (', '.join(
            ['%s = %s' % storage_parameter
             for storage_parameter in withclause.items()]))

    tablespace_name = index.dialect_options['postgresql']['tablespace']

    if tablespace_name:
        text += " TABLESPACE %s" % preparer.quote(tablespace_name)

    whereclause = index.dialect_options["postgresql"]["where"]

    if whereclause is not None:
        where_compiled = self.sql_compiler.process(
            whereclause, include_table=False,
            literal_binds=True)
        text += " WHERE " + where_compiled
    return text

PGDDLCompiler.visit_create_index = visit_create_index
