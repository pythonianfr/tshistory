from datetime import datetime
import os
import tempfile

import pandas as pd
import pytz

from tshistory.util import series_metadata
from tshistory.testutil import (
    assert_df,
    gengroup,
    genserie,
    utcdt
)
from tshistory.codecs import (
    iohelper,
    nary_pack,
    nary_unpack,
    node,
    pack_group,
    pack_group_history,
    pack_history,
    pack_many_series,
    pack_series,
    rev,
    unpack_group,
    unpack_group_history,
    unpack_history,
    unpack_many_series,
    unpack_series,
)


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


def test_pack_unpack_series():
    series1 = pd.Series(
        [1., 2., 3.],
        pd.date_range(utcdt(2021, 1, 1), freq='D', periods=3)
    )
    meta = series_metadata(series1)
    unpacked = unpack_series(
        'foo',
        pack_series(
            meta, series1
        )
    )
    assert_df("""
2021-01-01 00:00:00+00:00    1.0
2021-01-02 00:00:00+00:00    2.0
2021-01-03 00:00:00+00:00    3.0
""", unpacked)

    meta = series_metadata(series1)
    assert_df("""
2021-01-01 00:00:00+00:00    1.0
2021-01-02 00:00:00+00:00    2.0
2021-01-03 00:00:00+00:00    3.0
""", unpack_series(
        'foo', pack_series(
            meta, series1
        )
    ))

    empty = pd.Series(dtype='float64')
    meta = series_metadata(series1)
    packed = pack_series(
        meta, empty
    )
    unpacked = unpack_series(
        'foo', packed
    )
    assert len(unpacked) == 0
    assert meta == {
        'index_dtype': '|M8[ns]',
        'index_type': 'datetime64[ns, UTC]',
        'tzaware': True,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }


def test_pack_unpack_many_series():
    s1 = pd.Series(
        [1., 2., 3.],
        index=pd.date_range(utcdt(2020, 1, 1), freq='h', periods=3)
    )
    s2 = pd.Series(
        [1.1, 2.2, 3.3],
        index=pd.date_range(datetime(2020, 1, 2), freq='h', periods=3)
    )
    meta1 = {
        'tzaware': True,
        'index_type': 'datetime64[ns, UTC]',
        'value_type': 'float64',
        'index_dtype': '|M8[ns]',
        'value_dtype': '<f8'
    }
    meta2 = {
        'index_dtype': '<M8[ns]',
        'index_type': 'datetime64[ns]',
        'tzaware': False,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }
    packed = pack_many_series(
        [
            (meta1, s1),
            (meta2, s2)
        ]
    )
    unpacked = unpack_many_series(packed)
    assert unpacked[0].equals(s1)
    assert unpacked[1].equals(s2)


def test_pack_unpack_history(tsh, engine):
    for numserie in (1, 2, 3):
        with engine.begin() as cn:
            tsh.update(cn, genserie(datetime(2021, 1, 1), 'D', numserie),
                       'small-hist-naive',
                       'aurelien.campeas@pythonian.fr',
                       insertion_date=utcdt(2021, 2, numserie))

    hist = tsh.history(engine, 'small-hist-naive')
    meta = tsh.internal_metadata(engine, 'small-hist-naive')
    packed = pack_history(meta, hist)
    meta2, hist2 = unpack_history(packed)
    assert meta2 == meta
    for idate, series in hist.items():
        assert hist2[idate].equals(series)

    for numserie in (1, 2, 3):
        with engine.begin() as cn:
            tsh.update(cn, genserie(utcdt(2021, 1, 1), 'D', numserie),
                       'small-hist-tzaware',
                       'aurelien.campeas@pythonian.fr',
                       insertion_date=utcdt(2021, 2, numserie))

    hist = tsh.history(engine, 'small-hist-tzaware')
    meta = tsh.internal_metadata(engine, 'small-hist-tzaware')
    packed = pack_history(meta, hist)
    meta2, hist2 = unpack_history(packed)
    assert meta2 == meta
    for idate, series in hist.items():
        assert hist2[idate].equals(series)

# group

def test_pack_unpack_naive_group():
    df = gengroup(3, pd.Timestamp('2021-1-1'), 5, 'D', 2)
    packed = pack_group(df)
    unpacked = unpack_group(packed)

    assert unpacked.equals(df)

    packed = pack_group(df)
    unpacked = unpack_group(packed)

    assert unpacked.equals(df)


def test_pack_unpack_tzaware_group():
    df = gengroup(3, pd.Timestamp('2021-1-1', tz='UTC'), 5, 'D', 2)
    packed = pack_group(df)
    unpacked = unpack_group(packed)

    assert unpacked.equals(df)

    packed = pack_group(df)
    unpacked = unpack_group(packed)

    assert unpacked.equals(df)


def test_pack_naive_history_group():
    df1 = gengroup(3, pd.Timestamp('2021-1-1'), 5, 'D', 2)
    df2 = gengroup(3, pd.Timestamp('2021-1-2'), 5, 'D', 2)
    hist = {
        utcdt(2021, 1, 1): df1,
        utcdt(2021, 1, 2): df2,
    }
    packed = pack_group_history(hist)
    unpacked = unpack_group_history(packed)
    assert hist.keys() == unpacked.keys()
    for idate, group in hist.items():
        assert unpacked[idate].equals(hist[idate])


def test_pack_empty_history_group():
    df1 = gengroup(3, pd.Timestamp('2021-1-1', tz='UTC'), 5, 'D', 2)
    df2 = pd.DataFrame()
    hist = {
        utcdt(2021, 1, 1): df1,
        utcdt(2021, 1, 2): df2,
        utcdt(2021, 1, 3): df1 * 2
    }
    packed = pack_group_history(hist)
    unpacked = unpack_group_history(packed)
    assert hist.keys() == unpacked.keys()
    for idate, group in hist.items():
        assert unpacked[idate].equals(hist[idate])


def test_pack_tzaware_history_group():
    df1 = gengroup(3, pd.Timestamp('2021-1-1', tz='UTC'), 5, 'D', 2)
    df2 = gengroup(3, pd.Timestamp('2021-1-1', tz='UTC'), 5, 'D', 2)
    hist = {
        utcdt(2021, 1, 1): df1,
        utcdt(2021, 1, 2): df2,
    }
    packed = pack_group_history(hist)
    unpacked = unpack_group_history(packed)
    assert hist.keys() == unpacked.keys()
    for idate, group in hist.items():
        assert unpacked[idate].equals(hist[idate])


def test_make_snapshot_record():
    ts = pd.Series(
        [1., 2., 3.],
        index=pd.date_range(utcdt(2024, 1, 1), periods=3, freq='D')
    )
    meta = {
        'tzaware': True,
        'index_type': 'datetime64[ns, UTC]',
        'value_type': 'float64',
        'index_dtype': '|M8[ns]',
        'value_dtype': '<f8'
    }
    packed_ts = pack_series(meta, ts)
    bn = node.pack(
        utcdt(2020, 1, 1),
        utcdt(2020, 1, 2),
        0, # parent
        0, # start
        len(packed_ts) # offset
    )
    assert len(bn) == 26
    assert isinstance(bn, bytearray)

    n = node.unpack(
        pytz.utc,
        bytes(bn)
    )
    assert n.start == utcdt(2020, 1, 1)
    assert n.end == utcdt(2020, 1, 2)
    assert n.parent == 0
    assert n.address == 0
    assert n.size == 140  # Fast compression (level 1) produces slightly larger output

    bn = node.pack(
        utcdt(2020, 1, 1),
        utcdt(2020, 1, 2),
        1,
        0,
        len(packed_ts)
    )
    assert len(bn) == 26
    assert isinstance(bn, bytearray)

    n = node.unpack(
        pytz.utc,
        bytes(bn)
    )
    assert n.start == utcdt(2020, 1, 1)
    assert n.end == utcdt(2020, 1, 2)
    assert n.parent == 1
    assert n.address == 0
    assert n.size == 140  # Fast compression (level 1) produces slightly larger output


def test_version_record():
    br = rev.pack(
        utcdt(2024, 1, 1),
        utcdt(2023, 12, 31, 0),
        utcdt(2023, 12, 31, 2),
        0,
        1
    )
    assert len(br) == 32
    assert isinstance(br, bytearray)

    r = rev.unpack(
        pytz.utc,
        bytes(br)
    )
    assert r.revdate == utcdt(2024, 1, 1)
    assert r.diffstart == utcdt(2023, 12, 31, 0)
    assert r.diffend == utcdt(2023, 12, 31, 2)
    assert r.index == 0
    assert r.metaid == 1


def test_read_write_2_versions():
    meta = {
        'tzaware': True,
        'index_type': 'datetime64[ns, UTC]',
        'value_type': 'float64',
        'index_dtype': '|M8[ns]',
        'value_dtype': '<f8'
    }
    # this will be a pure append scenario
    ts1 = pd.Series(
        [1., 2., 3.],
        index=pd.date_range(utcdt(2024, 1, 1), periods=3, freq='D')
    )
    ts2 = pd.Series(
        [4., 5., 6.],
        index=pd.date_range(utcdt(2024, 1, 4), periods=3, freq='D')
    )

    with tempfile.TemporaryDirectory() as tmp:
        # Plan:
        # * write two snapshots and their blocks
        # * then write the associated revisions
        # * then read the full series (made of 2 revs)
        with open(tmp + '/revs', mode='x'):
            pass
        with open(tmp + '/tree', mode='x'):
            pass
        with open(tmp + '/chunks', mode='x'):
            pass

        with open(tmp + '/tree', 'wb') as tree:
            # write the snapshots (using prepared chunks)
            # v1
            packed1 = iohelper.serialize_ts(ts1, False)
            rec1 = node.pack(
                ts1.index[0],
                ts1.index[-1],
                0, # indice of the parent in the tree file (0 means no parent)
                0, # address of the block in the chunks file
                len(packed1) # offset of the block in the chunks file
            )
            tree.write(rec1)
            # v2
            packed2 = iohelper.serialize_ts(ts2, False)
            rec2 = node.pack(
                ts2.index[0],
                ts2.index[-1],
                1, # indicates the first block
                len(packed1),
                len(packed2)
            )
            tree.write(rec2)

        with open(tmp + '/chunks', 'wb') as chunks:
            # write the two chunks also
            chunks.write(packed1)
            chunks.write(packed2)

        # check the size
        assert os.stat(tmp + '/tree').st_size == 52
        assert os.stat(tmp + '/chunks').st_size == 89

        with open(tmp + '/revs', 'wb') as revs:
            # now, having written the tree let's write the revs
            br = rev.pack(
                utcdt(2024, 2, 1),
                ts1.index[0],
                ts1.index[-1],
                0, # index in tree obviously starts at zero
                42, # we don't care much about metaid ...
            )
            revs.write(br)
            br = rev.pack(
                utcdt(2024, 2, 2),
                ts2.index[0],
                ts2.index[-1],
                1, # second record in tree
                42
            )
            revs.write(br)

        # now, let's read the complete version back
        with open(tmp + '/revs', 'rb') as revs:
            # rev block is of size 32
            revs.seek(32) # seek to the beginning of the last block
            bytestr = revs.read(32)
            r = rev.unpack(
                pytz.utc,
                bytestr
            )
            assert r.index == 1

        # ok, let's dig the chunks from this blockid and rebuild the
        # whole series from chunks
        with open(tmp + '/tree', 'rb') as tree:
            # we start with using the tree index
            tree.seek(r.index * 26) # move to last block
            fixed = tree.read(26)
            node2 = node.unpack(pytz.utc, fixed)

            tree.seek(0)
            fixed = tree.read(26)
            node1 = node.unpack(pytz.utc, fixed)

        with open(tmp + '/chunks', 'rb') as chunks:
            chunks.seek(node2.address)
            chunk2 = chunks.read(node2.size)

            chunks.seek(node1.address)
            chunk1 = chunks.read(node2.size)

    fullts = iohelper.chunks_to_ts(meta, [chunk1, chunk2])
    assert_df("""
2024-01-01 00:00:00+00:00    1.0
2024-01-02 00:00:00+00:00    2.0
2024-01-03 00:00:00+00:00    3.0
2024-01-04 00:00:00+00:00    4.0
2024-01-05 00:00:00+00:00    5.0
2024-01-06 00:00:00+00:00    6.0
""", fullts)
