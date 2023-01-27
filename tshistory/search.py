import uuid


def usym(basename):
    " produce a unique symbol "
    return f'{basename}{uuid.uuid4().hex}'


class and_:
    __slots__ = ('sqls', 'kw')

    def __init__(self, *clauses):
        self.sqls = []
        self.kw = {}
        for clause in clauses:
            sql, kw = clause.sql()
            self.sqls.append(sql)
            self.kw.update(kw)

    def sql(self):
        return ' and '.join(self.sqls), self.kw


class or_:
    __slots__ = ('sqls', 'kw')

    def __init__(self, *clauses):
        self.sqls = []
        self.kw = {}
        for clause in clauses:
            sql, kw = clause.sql()
            self.sqls.append(sql)
            self.kw.update(kw)

    def sql(self):
        return ' or '.join(self.sqls), self.kw


class not_:
    __slots__ = ('clause',)

    def __init__(self, clause):
        self.clause = clause

    def sql(self):
        sql, kw = self.clause.sql()
        return f'not {sql}', kw


class tzaware:

    def sql(self):
        return 'internal_metadata @> \'{"tzaware":true}\'::jsonb', {}


class byname:
    __slots__ = ('query',)

    def __init__(self, query: str):
        self.query = query.replace(' ', '%%')

    def sql(self):
        vid = usym('name')
        return f'name like %({vid})s', {vid: f'%%{self.query}%%'}


class bymetakey:
    __slots__ = ('key',)

    def __init__(self, key: str):
        self.key = key

    def sql(self):
        vid = usym('key')
        return f'metadata ? %({vid})s', {vid: self.key}


class bymetaitem:
    __slots__ = ('key', 'value')

    def __init__(self, key: str, value: str):
        self.key = key
        self.value = value

    def sql(self):
        # NOTE: this is weak and injection prone
        # we need to find a robust workaround for
        # psycopg2 bugs
        vid = usym('value')
        if not isinstance(self.value, str):
            return (
                f'metadata @> \'{{"{self.key}":%({vid})s}}\'::jsonb',
                {vid: self.value}
            )

        return f'metadata @> \'{{"{self.key}":"{self.value}"}}\'::jsonb', {}
