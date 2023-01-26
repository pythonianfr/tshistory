import uuid


def usym(basename):
    " produce a unique symbol "
    return f'{basename}{uuid.uuid4().hex}'


class and_:
    __slots__ = ('clauses', 'compiled')

    def __init__(self, *clauses):
        self.compiled = [
            clause.sql(self)
            for clause in clauses
        ]

    def where(self, clause, **kw):
        return clause, kw

    def sql(self, sqlquery):
        sqls = []
        kws = {}
        for sql, kw in self.compiled:
            sqls.append(sql)
            kws.update(kw)
        sqlquery.where(
            ' and '.join(sqls),
            **kws
        )


class tzaware:

    def sql(self, sqlquery):
        return sqlquery.where(
            'internal_metadata @> \'{"tzaware":true}\'::jsonb'
        )


class byname:
    __slots__ = ('query',)

    def __init__(self, query: str):
        self.query = query.replace(' ', '%%')

    def sql(self, sqlquery):
        vid = usym('name')
        return sqlquery.where(
            f'name like %({vid})s',
            **{vid: f'%%{self.query}%%'}
        )


class bymetakey:
    __slots__ = ('key',)

    def __init__(self, key: str):
        self.key = key

    def sql(self, sqlquery):
        vid = usym('key')
        return sqlquery.where(
            f'metadata ? %({vid})s',
            **{vid: self.key}
        )


class bymetaitem:
    __slots__ = ('key', 'value')

    def __init__(self, key: str, value: str):
        self.key = key
        self.value = value

    def sql(self, sqlquery):
        # NOTE: this is weak and injection prone
        # we need to find a robust workaround for
        # psycopg2 bugs
        vid = usym('value')
        if not isinstance(self.value, str):
            return sqlquery.where(
                f'metadata @> \'{{"{self.key}":%({vid})s}}\'::jsonb',
                **{vid: self.value}
            )

        return sqlquery.where(
            f'metadata @> \'{{"{self.key}":"{self.value}"}}\'::jsonb'
        )
