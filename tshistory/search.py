
class byname:
    __slots__ = ('query',)

    def __init__(self, query: str):
        self.query = query.replace(' ', '%%')

    def sql(self, sqlquery):
        sqlquery.where(
            'name like %(name)s',
            name=f'%%{self.query}%%'
        )
