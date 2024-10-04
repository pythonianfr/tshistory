import json
import uuid
import typing
from psyl.lisp import parse

from tshistory.util import (
    all_subclasses,
    ensure_plugin_registration
)


__all__ = [
    'query', 'and_', 'or_', 'not_', 'tzaware',
    'byname', 'bymetakey', 'bymetaitem', 'bysource',
    'byeverything',
    'lt', 'lte', 'gt', 'gte'
]


def usym(basename):
    " produce a unique symbol "
    return f'{basename}{uuid.uuid4().hex}'


_OPMAP = {
    'by.everything': 'byeverything',
    'by.and': 'and_',
    'by.or': 'or_',
    'by.not': 'not_',
    'by.tzaware': 'tzaware',
    'by.name': 'byname',
    'by.metakey': 'bymetakey',
    'by.metaitem': 'bymetaitem',
    'by.internal-metaitem': 'byinternalmetaitem',
    'by.source': 'bysource',
    '<': 'lt',
    '<=': 'lte',
    '>': 'gt',
    '>=': 'gte',
    '=': 'eq'
}


def _has_bysource(tree):
    if not isinstance(tree, list):
        return tree

    op = tree[0]
    if op == 'by.source':
        return True

    for item in tree:
        if _has_bysource(item):
            return True

    return False


def prunebysource(sourcename: str, querytree: list) -> typing.Optional[list]:
    """
    Remove all subtrees associated with by.source expressions that
    do not match the given sourcename.

    """
    assert isinstance(querytree, list)

    def _prune(tree):
        if not isinstance(tree, list):
            return tree

        if not _has_bysource(tree):
            # optimise the case where there is no bysource filter
            # checking is cheaper than rewriting
            return tree

        op = tree[0]
        if op == 'by.source' and tree[1] != sourcename:
            return None  # pruned !

        newtree = []
        pruned = False
        for item in tree:
            newitem = _prune(item)
            if newitem is None:
                pruned = True
                continue
            newtree.append(newitem)

        if pruned and op in ('by.and', 'by.not'):
            return None

        if op in ('by.and', 'by.or'):
            # remove the and/or if they reign over a single clause
            if len(newtree[1:]) == 1:
                return newtree[1]

        return newtree

    return _prune(
        querytree
    )


def removebysource(querytree: list) -> typing.Optional[list]:
    """
    Remove all by.source expression and simplify the tree
    accordingly

    """
    assert isinstance(querytree, list)

    def _prune(tree):
        if not isinstance(tree, list):
            return tree

        if not _has_bysource(tree):
            # optimise the case where there is no bysource filter
            # checking is cheaper than rewriting
            return tree

        op = tree[0]
        if op == 'by.source':
            return None  # pruned !

        newtree = []
        for item in tree:
            newitem = _prune(item)
            if newitem is None:
                continue
            newtree.append(newitem)

        if op in ('by.and', 'by.or'):
            # remove the and/or if they reign over a single clause
            if len(newtree[1:]) == 1:
                return newtree[1]

        return newtree

    return _prune(
        querytree
    )




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
    def klassbyname(klassname):
        classmap = {
            klass.__name__: klass
            for klass in all_subclasses(query)
            if not klass.__name__.startswith('_')
        }
        return classmap[klassname]

    @staticmethod
    def _fromtree(tree):
        ensure_plugin_registration()
        op = tree[0]
        klass = query.klassbyname(_OPMAP[op])
        return klass._fromtree(tree)


class bysource(query):
    __slots__ = ('source',)

    def __init__(self, source: str):
        self.source = source

    def __expr__(self):
        return f'(by.source "{self.source}")'

    @staticmethod
    def __sig__():
        return {
            'source': 'str',
            'return': 'query'
        }

    @classmethod
    def _fromtree(cls, tree):
        return cls(tree[1])

    def sql(self, namespace='tsh'):
        return '', {}


class byeverything(query):

    @classmethod
    def _fromtree(cls, _):
        return cls()

    @staticmethod
    def __sig__():
        return {
            'return': 'query'
        }

    def sql(self, namespace='tsh'):
        return '', {}


class and_(query):
    __slots__ = ('items',)

    def __init__(self, *items):
        self.items = items

    def __expr__(self):
        return f'(by.and {" ".join(x.expr() for x in self.items)})'

    @staticmethod
    def __sig__():
        return {
            'items': 'Packed[query]',
            'return': 'query'
        }

    @classmethod
    def _fromtree(cls, tree):
        items = [
            query._fromtree(subtree)
            for subtree in tree[1:]
        ]
        return cls(*items)

    def sql(self, namespace='tsh'):
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
        return f'(by.or {" ".join(x.expr() for x in self.items)})'

    @staticmethod
    def __sig__():
        return {
            'items': 'Packed[query]',
            'return': 'query'
        }

    @classmethod
    def _fromtree(cls, tree):
        items = [
            query._fromtree(subtree)
            for subtree in tree[1:]
        ]
        return cls(*items)

    def sql(self, namespace='tsh'):
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
        return f'(by.not {self.item.expr()})'

    @staticmethod
    def __sig__():
        return {
            'item': 'query',
            'return': 'query'
        }

    @classmethod
    def _fromtree(cls, tree):
        return cls(query._fromtree(tree[1]))

    def sql(self, namespace='tsh'):
        sql, kw = self.item.sql()
        return f'not {sql}', kw


class tzaware(query):

    def __expr__(self):
        return '(by.tzaware)'

    @staticmethod
    def __sig__():
        return {
            'return': 'query'
        }

    @classmethod
    def _fromtree(cls, _):
        return cls()

    def sql(self, namespace='tsh'):
        return 'internal_metadata @> \'{"tzaware":true}\'::jsonb', {}


class byname(query):
    __slots__ = ('query',)

    def __init__(self, query: str):
        self.query = query

    def __expr__(self):
        return f'(by.name "{self.query}")'

    @staticmethod
    def __sig__():
        return {
            'query': 'str',
            'return': 'query'
        }

    @classmethod
    def _fromtree(cls, tree):
        return cls(tree[1])

    def sql(self, namespace='tsh'):
        vid = usym('name')
        query = self.query.replace(' ', '%%')
        return f'name like %({vid})s', {vid: f'%%{query}%%'}


class bymetakey(query):
    __slots__ = ('key',)

    def __init__(self, key: str):
        self.key = key

    def __expr__(self):
        return f'(by.metakey "{self.key}")'

    @staticmethod
    def __sig__():
        return {
            'key': 'str',
            'return': 'query'
        }

    @classmethod
    def _fromtree(cls, tree):
        return cls(tree[1])

    def sql(self, namespace='tsh'):
        vid = usym('key')
        return f'metadata ? %({vid})s', {vid: self.key}


class _comparator(query):
    __slots__ = ('key', 'value')
    _op = None
    _lispop = None
    _field = 'metadata'

    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __expr__(self):
        if isinstance(self.value, str):
            return f'({self._lispop or self._op} "{self.key}" "{self.value}")'
        return f'({self._lispop or self._op} "{self.key}" {self.value})'

    @staticmethod
    def __sig__():
        return {
            'key': 'str',
            'value': 'Union[str, Number, bool]',
            'return': 'query'
        }

    @classmethod
    def _fromtree(cls, tree):
        return cls(*tree[1:])

    def sql(self, namespace='tsh'):
        if isinstance(self.value, str):
            kid = usym('key')
            jid = usym('json')
            return (
                # issue: " (double quotes) in json vs plain sql means
                # something different - we have to use a concatenation
                # trick to have it
                f'jsonb_path_match({self._field}, '
                f'(\'$."\' || %({kid})s || \'" {self._op} $value\')::jsonpath, '
                f'%({jid})s)',
                {
                    kid: self.key,
                    jid: json.dumps({'value': self.value})
                }
            )

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


class bymetaitem(eq):
    __slots__ = ('key', 'value')

    def __expr__(self):
        if isinstance(self.value, str):
            return f'(by.metaitem "{self.key}" "{self.value}")'
        return f'(by.metaitem "{self.key}" {self.value})'


class byinternalmetaitem(eq):
    _field = 'internal_metadata'

    def __expr__(self):
        if isinstance(self.value, str):
            return f'(by.internal-metaitem "{self.key}" "{self.value}")'
        return f'(by.internal-metaitem "{self.key}" {self.value})'
