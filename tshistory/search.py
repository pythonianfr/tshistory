import uuid
from psyl.lisp import parse


__all__ = [
    'query', 'and_', 'or_', 'not_', 'tzaware',
    'byname', 'bymetakey', 'bymetaitems',
    'lt', 'lte', 'gt', 'gte'
]


def usym(basename):
    " produce a unique symbol "
    return f'{basename}{uuid.uuid4().hex}'


_OPMAP = {
    'and': 'and_',
    'or': 'or_',
    'not': 'not_',
    '<': 'lt',
    '<=': 'lte',
    '>': 'gt',
    '>=': 'gte',
    '=': 'eq'
}


class query:

    def sql(self):
        raise NotImplementedError

    def __expr__(self):
        raise NotImplementedError

    def expr(self):
        return self.__expr__()

    def __repr__(self):
        return f'<query: {self.expr()}>'

    @staticmethod
    def fromexpr(expr):
        tree = parse(expr)
        return query._fromtree(tree)

    @staticmethod
    def _fromtree(tree):
        op = tree[0]
        klass = globals()[_OPMAP.get(op, op)]
        return klass._fromtree(tree)


class and_(query):
    __slots__ = ('items',)

    def __init__(self, *items):
        self.items = items

    def __expr__(self):
        return f'(and {" ".join(x.expr() for x in self.items)})'

    @classmethod
    def _fromtree(cls, tree):
        items = [
            query._fromtree(subtree)
            for subtree in tree[1:]
        ]
        return cls(*items)

    def sql(self):
        sqls = []
        kws = {}
        for item in self.items:
            sql, kw = item.sql()
            sqls.append(sql)
            kws.update(kw)
        return ' and '.join(sqls), kws


class or_(query):
    __slots__ = ('items',)

    def __init__(self, *items):
        self.items = items

    def __expr__(self):
        return f'(or {" ".join(x.expr() for x in self.items)})'

    @classmethod
    def _fromtree(cls, tree):
        items = [
            query._fromtree(subtree)
            for subtree in tree[1:]
        ]
        return cls(*items)

    def sql(self):
        sqls = []
        kws = {}
        for item in self.items:
            sql, kw = item.sql()
            sqls.append(sql)
            kws.update(kw)
        return ' or '.join(sqls), kws


class not_(query):
    __slots__ = ('item',)

    def __init__(self, item):
        self.item = item

    def __expr__(self):
        return f'(not {self.item.expr()})'

    @classmethod
    def _fromtree(cls, tree):
        return cls(query._fromtree(tree[1]))

    def sql(self):
        sql, kw = self.item.sql()
        return f'not {sql}', kw


class tzaware(query):

    def __expr__(self):
        return '(tzaware)'

    @classmethod
    def _fromtree(cls, _):
        return cls()

    def sql(self):
        return 'internal_metadata @> \'{"tzaware":true}\'::jsonb', {}


class byname(query):
    __slots__ = ('query',)

    def __init__(self, query: str):
        self.query = query

    def __expr__(self):
        return f'(byname "{self.query}")'

    @classmethod
    def _fromtree(cls, tree):
        return cls(tree[1])

    def sql(self):
        vid = usym('name')
        query = self.query.replace(' ', '%%')
        return f'name like %({vid})s', {vid: f'%%{query}%%'}


class bymetakey(query):
    __slots__ = ('key',)

    def __init__(self, key: str):
        self.key = key

    def __expr__(self):
        return f'(bymetakey "{self.key}")'

    @classmethod
    def _fromtree(cls, tree):
        return cls(tree[1])

    def sql(self):
        vid = usym('key')
        return f'metadata ? %({vid})s', {vid: self.key}


class bymetaitem(query):
    __slots__ = ('key', 'value')

    def __init__(self, key: str, value: str):
        self.key = key
        self.value = value

    def __expr__(self):
        if isinstance(self.value, str):
            return f'(bymetaitem "{self.key}" "{self.value}")'
        return f'(bymetaitem "{self.key}" {self.value})'

    @classmethod
    def _fromtree(cls, tree):
        return cls(*tree[1:])

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


class _comparator(query):
    __slots__ = ('key', 'value')
    _op = None
    _lispop = None

    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __expr__(self):
        if isinstance(self.value, str):
            return f'({self._op} "{self.key}" "{self.value}")'
        return f'({self._lispop or self._op} "{self.key}" {self.value})'

    @classmethod
    def _fromtree(cls, tree):
        return cls(*tree[1:])

    def sql(self):
        vid = usym('value')
        return (
            f'jsonb_path_match(metadata, \'$.{self.key} {self._op} %({vid})s\')',
            {vid: self.value}
        )


class lt(_comparator):
    __slots__ = ('key', 'value')
    _op = '<'


class lte(_comparator):
    __slots__ = ('key', 'value')
    _op = '<='


class gt(_comparator):
    __slots__ = ('key', 'value')
    _op = '>'


class gte(_comparator):
    __slots__ = ('key', 'value')
    _op = '>='


class eq(_comparator):
    __slots__ = ('key', 'value')
    _op = '=='
    _lispop = '='
