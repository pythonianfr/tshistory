from datetime import datetime

import pytest

from tshistory.util import (
    bisect_search,
    nary_pack,
    nary_unpack
)
from tshistory.testutil import (
    genserie
)


def test_bisect():
    values = [-4, -2, 1, 7]
    assert bisect_search(values, -5) == -1
    assert bisect_search(values, -4) == 0
    assert bisect_search(values, -3) == 0
    assert bisect_search(values, 0) == 1
    assert bisect_search(values, 1) == 2
    assert bisect_search(values, 3) == 2
    assert bisect_search(values, 7) == 3
    assert bisect_search(values, 8) == 4


def test_pack_unpack():
    chunks = (
        b'a',
        b'bb',
        b'ccc',
        b'dd',
        b'e'
    )
    packed = nary_pack(*chunks)
    assert len(packed) == 33
    unpacked = nary_unpack(packed)
    assert chunks == unpacked

    chunks = (
        b'aurelien',
        b'campeas',
        b'develops',
        b'tshistory'
    )
    packed = nary_pack(*chunks)
    assert packed == (
        b'\x00\x00\x00\x04\x00\x00\x00\x08\x00\x00\x00\x07\x00'
        b'\x00\x00\x08\x00\x00\x00\t'
        b'aureliencampeasdevelopstshistory'
    )
    unpacked = nary_unpack(packed)
    assert chunks == unpacked

    chunks = tuple()
    packed = nary_pack(*chunks)
    unpacked = nary_unpack(packed)
    assert chunks == unpacked


def test_in_tx(tsh, engine):
    assert tsh.type(engine, 'foo') == 'primary'

    with pytest.raises(TypeError) as err:
        tsh.update(engine.connect(), 0, 0, 0)
    assert err.value.args[0] == 'You must use a transaction object'

    ts = genserie(datetime(2017, 10, 28, 23),
                  'H', 4, tz='UTC')
    with engine.begin() as cn:
        tsh.update(cn, ts, 'test_tx', 'Babar')
        # that old api is still there
        tsh.insert(cn, ts, 'test_tx', 'Babar')
