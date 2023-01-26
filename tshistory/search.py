
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
