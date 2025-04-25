from psyl.lisp import (
    parse,
    serialize
)

from tshistory import search


def _serialize_roundtrip(searchobj):
    return search.query.fromexpr(searchobj.expr()).expr() == searchobj.expr()


def test_tzaware():
    s0 = search.tzaware()
    assert s0.expr() == '(by.tzaware)'
    assert _serialize_roundtrip(s0)


def test_byname():
    s1 = search.byname('foo bar')
    assert s1.expr() == '(by.name "foo bar")'
    assert _serialize_roundtrip(s1)


def test_or():
    s0 = search.tzaware()
    s1 = search.byname('foo bar')
    s2 = search.or_(s0, s1)
    assert s2.expr() == '(by.or (by.tzaware) (by.name "foo bar"))'
    assert _serialize_roundtrip(s2)


def test_and():
    s0 = search.tzaware()
    s1 = search.byname('foo bar')
    s3 = search.and_(s0, s1)
    assert s3.expr() == '(by.and (by.tzaware) (by.name "foo bar"))'
    assert _serialize_roundtrip(s3)


def test_not():
    s0 = search.tzaware()
    s1 = search.byname('foo bar')
    s3 = search.and_(s0, s1)
    s4 = search.not_(s3)
    assert s4.expr() == '(by.not (by.and (by.tzaware) (by.name "foo bar")))'
    assert _serialize_roundtrip(s4)


def test_bymetakey():
    s5 = search.bymetakey('key')
    assert s5.expr() == '(by.metakey "key")'
    assert _serialize_roundtrip(s5)


def test_bymetaitem_str():
    s6 = search.bymetaitem('key', 'value')
    assert s6.expr() == '(by.metaitem "key" "value")'
    assert _serialize_roundtrip(s6)


def test_bymetaitem_num():
    s7 = search.bymetaitem('key', 42)
    assert s7.expr() == '(by.metaitem "key" 42)'
    assert _serialize_roundtrip(s7)


def test_lt():
    s8 = search.lt('key', 42)
    assert s8.expr() == '(< "key" 42)'
    assert _serialize_roundtrip(s8)


def test_lte():
    s9 = search.lte('key', 42)
    assert s9.expr() == '(<= "key" 42)'
    assert _serialize_roundtrip(s9)


def test_gt():
    s10 = search.gt('key', 42)
    assert s10.expr() == '(> "key" 42)'
    assert _serialize_roundtrip(s10)


def test_gte():
    s11 = search.gte('key', 42)
    assert s11.expr() == '(>= "key" 42)'
    assert _serialize_roundtrip(s11)


def test_eq_num():
    s12 = search.eq('key', 42)
    assert s12.expr() == '(= "key" 42)'
    assert _serialize_roundtrip(s12)


def tests_eq_str():
    s13 = search.eq('key', "Hello")
    assert s13.expr() == '(= "key" "Hello")'
    assert _serialize_roundtrip(s13)


def test_bysource():
    s14 = search.bysource('remote')
    assert s14.expr() == '(by.source "remote")'
    assert _serialize_roundtrip(s14)


def test_byinternalmetaitem():
    s15 = search.byinternalmetaitem('key', 42)
    assert s15.expr() == '(by.internal-metaitem "key" 42)'
    assert _serialize_roundtrip(s15)


def test_search_types():
    types = {}
    for lispname, kname in search._OPMAP.items():
        if not getattr(search, kname, False):
            continue
        types[lispname] = search.query.klassbyname(kname).__sig__()

    assert types == {
        '<': {'key': 'str', 'return': 'query', 'value': 'Union[str, Number, bool]'},
        '<=': {'key': 'str', 'return': 'query', 'value': 'Union[str, Number, bool]'},
        '=': {'key': 'str', 'return': 'query', 'value': 'Union[str, Number, bool]'},
        '>': {'key': 'str', 'return': 'query', 'value': 'Union[str, Number, bool]'},
        '>=': {'key': 'str', 'return': 'query', 'value': 'Union[str, Number, bool]'},
        'by.and': {'items': 'Packed[query]', 'return': 'query'},
        'by.everything': {'return': 'query'},
        'by.internal-metaitem': {'key': 'str',
                                 'return': 'query',
                                 'value': 'Union[str, Number, bool]'},
        'by.metaitem': {'key': 'str',
                        'return': 'query',
                        'value': 'Union[str, Number, bool]'},
        'by.metakey': {'key': 'MetaKey', 'return': 'query'},
        'by.name': {'query': 'str', 'return': 'query'},
        'by.not': {'item': 'query', 'return': 'query'},
        'by.or': {'items': 'Packed[query]', 'return': 'query'},
        'by.source': {'source': 'Source', 'return': 'query'},
        'by.tzaware': {'return': 'query'}
    }


def test_prune_bysource():
    """Notion of by.source filter.

    Query without it: executed as is eveywhere.

    by.source "source" -> remove all by.source <source> that do not
    match "source"

    """
    assert search.prunebysource(
        'local',
        parse('(by.source "remote")')
    ) is None

    assert serialize(
        search.prunebysource(
            'local',
            parse(
                '(by.or '
                '  (by.name "foo")'
                '  (by.source "remote"))'
            )
        )
    ) == '(by.name "foo")'

    assert search.prunebysource(
        'local',
        parse(
            '(by.and '
            '  (by.name "foo")'
            '  (by.source "remote"))'
        )
    ) is None

    assert serialize(
        search.prunebysource(
            'remote',
            parse(
                '(by.or '
                '  (by.name "foo")'
                '  (by.source "local")'
                '  (by.and '
                '    (by.name "bar")'
                '    (by.source "remote")))'
            )
        )
    ) == '(by.or (by.name "foo") (by.and (by.name "bar") (by.source "remote")))'

    assert serialize(
        search.prunebysource(
            'remote',
            parse(
                '(by.or '
                '  (by.name "foo")'
                '  (by.source "remote")'
                '  (by.and '
                '    (by.name "bar")'
                '    (by.source "local")))'
            )
        )
    ) == '(by.or (by.name "foo") (by.source "remote"))'

    q = search.prunebysource(
        'local',
        parse(
            '(by.or '
            '  (by.not (by.and (by.name "basket.fed") (by.source "local")))'
            '  (by.source "remote"))'
        )
    )
    assert q == ['by.not', ['by.and', ['by.name', 'basket.fed'], ['by.source', 'local']]]
    q2 = search.removebysource(q)
    assert q2 == ['by.not', ['by.name', 'basket.fed']]
