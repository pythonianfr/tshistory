
class tzaware:

    def sql(self, sqlquery):
        sqlquery.where(
            'internal_metadata @> \'{"tzaware":true}\'::jsonb'
        )


class byname:
    __slots__ = ('query',)

    def __init__(self, query: str):
        self.query = query.replace(' ', '%%')

    def sql(self, sqlquery):
        sqlquery.where(
            'name like %(name)s',
            name=f'%%{self.query}%%'
        )


class bymetakey:
    __slots__ = ('key',)

    def __init__(self, key: str):
        self.key = key

    def sql(self, sqlquery):
        sqlquery.where(
            'metadata ? %(key)s',
            key=self.key
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
        if not isinstance(self.value, str):
            sqlquery.where(
                f'metadata @> \'{{"{self.key}":%(value)s}}\'::jsonb',
                value=self.value
            )
            return
        sqlquery.where(
            f'metadata @> \'{{"{self.key}":"{self.value}"}}\'::jsonb',
            value=self.value
        )
