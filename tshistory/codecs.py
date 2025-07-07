import io
import json
import pytz
import struct
import zlib

import numpy as np
import pandas as pd
from typing import Any, Iterable



# binary

def numpy_serialize(series: pd.Series, isstr: bool = False) -> tuple[bytes, bytes]:
    # use `view` as a workarround for "cannot include dtype 'M' in a buffer"
    if len(series):
        bindex = np.ascontiguousarray(
            series.index.values
        ).view(np.uint8).data
    else:
        bindex = b''

    if isstr:
        # string separatd by 0 and nones/nans represented as 3 (ETX)
        END, ETX = b'\0'.decode(), b'\3'.decode()
        # first, safety belt
        for s in series.values:
            if not pd.isnull(s):
                assert END not in s and ETX not in s
        bvalues = b'\0'.join(
            b'\3' if pd.isnull(v) else v.encode('utf-8')
            for v in series.values
        )
    else:
        bvalues = series.values.data.tobytes()

    return bindex, bvalues


SIZE = struct.Struct('!L')


def binary_pack(bytes1: bytes, bytes2: bytes) -> bytes:
    """assemble two byte strings into a unique byte string
    storing the size of the first string first
    this will permit to destructure back the two
    original byte strings

    """
    bytes1_size = SIZE.pack(len(bytes1))
    return bytes1_size + bytes1 + bytes2


def binary_unpack(packedbytes: bytes) -> tuple[bytes, bytes]:
    """get a compressed bytes stream and return the two embedded
    bytes strings

    """
    [bytes1_size] = SIZE.unpack(packedbytes[:4])
    bytes2_offset = bytes1_size + 4
    return packedbytes[4:bytes2_offset], packedbytes[bytes2_offset:]


def nary_pack(*bytestr: bytes) -> bytes:
    sizes = [
        SIZE.pack(len(b))
        for b in bytestr
    ]
    sizes_size = SIZE.pack(len(sizes))
    stream = io.BytesIO()
    stream.write(sizes_size)
    stream.write(b''.join(sizes))
    for bstr in bytestr:
        stream.write(bstr)
    return stream.getvalue()


def nary_unpack(packedbytes: bytes) -> tuple[bytes, ...]:
    [sizes_size] = SIZE.unpack(packedbytes[:4])
    payloadoffset = 4 + sizes_size * 4
    sizes = struct.unpack(
        f'!{"L"*sizes_size}',
        packedbytes[4: payloadoffset]
    )
    fmt = ''.join('%ss' % size for size in sizes)
    return struct.unpack(fmt, packedbytes[payloadoffset:])


def numpy_deserialize(
    bindex: bytes,
    bvalues: bytes,
    metadata: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray | list[str | None]]:
    """produce a pandas series from serialized index and values (numpy
    arrays)

    """
    # Direct frombuffer is more efficient and correct
    index = np.frombuffer(
        bindex,
        metadata['index_dtype']
    )

    if metadata['value_type'] == 'object':  # str
        if not len(bvalues):
            values = []
        else:
            values = [
                v.decode('utf-8') if v != b'\3' else None
                for v in bvalues.split(b'\0')
            ]
    else:
        values = np.frombuffer(
            bvalues,
            metadata['value_dtype']
        )
    return index, values


def pack_series(
    metadata: dict[str, Any],
    series: pd.Series,
    compressor = zlib.compress
) -> bytes:
    """Transform a series, using associated metadata, into a binary format
    (using an optional serializer e.g. b85encode)
    """
    bindex, bvalues = numpy_serialize(
        series,
        metadata['value_type'] == 'object'
    )
    bmeta = json.dumps(metadata).encode('utf-8')
    return compressor(
        nary_pack(
            bmeta,
            bindex,
            bvalues
        )
    )


def unpack_series(
    name: str,
    bytestream: bytes,
    decompressor = zlib.decompress
) -> pd.Series:
    """Transform a binary string into a pandas series of the given name
    """
    bmeta, bindex, bvalues = nary_unpack(
        decompressor(bytestream)
    )
    meta = json.loads(bmeta)
    index, values = numpy_deserialize(
        bindex,
        bvalues,
        meta
    )
    series = pd.Series(
        values,
        index=index,
        name=name,
        dtype=meta['value_type']
    )
    if meta['tzaware']:
        series = series.tz_localize(pytz.utc)
    return series


def pack_many_series(
    serieslist: list[pd.Series],
    compressor = zlib.compress
) -> bytes:
    """Transform a series list, using associated metadata, into a binary format
    """
    binaries = []
    for (metadata, series) in serieslist:
        bindex, bvalues = numpy_serialize(
            series,
            metadata['value_type'] == 'object'
        )
        metadata['name'] = series.name
        bmeta = json.dumps(metadata).encode('utf-8')
        binaries.append(bmeta)
        binaries.append(bindex)
        binaries.append(bvalues)

    return compressor(
        nary_pack(
            *binaries
        )
    )


def unpack_many_series(
    bytestream: bytes,
    decompressor = zlib.decompress
) -> list[pd.Series]:
    """Transform a binary string into a pandas series of the given name
    """
    binaries = nary_unpack(
        decompressor(bytestream)
    )
    serieslist = []
    for bmeta, bindex, bvalues in zip(*[iter(binaries)]*3):
        meta = json.loads(bmeta)
        index, values = numpy_deserialize(
            bindex,
            bvalues,
            meta
        )
        name = meta['name']
        series = pd.Series(
            values,
            index=index,
            name=name
        )
        if meta['tzaware']:
            series = series.tz_localize(pytz.utc)
        serieslist.append(series)

    return serieslist


def pack_history(
    metadata: dict[str, Any],
    hist: dict[pd.Timestamp, pd.Series]
) -> bytes:
    byteslist = [json.dumps(metadata).encode('utf-8')]
    arr = np.array(
        [tstamp.to_datetime64() for tstamp in hist],
        dtype='datetime64[ns]'
    ).view(np.uint8).data
    byteslist.append(arr)
    isstr = metadata['value_type'] == 'object'
    for series in hist.values():
        index, values = numpy_serialize(
            series,
            isstr
        )
        byteslist.append(index)
        byteslist.append(values)
    stream = io.BytesIO(
        zlib.compress(
            nary_pack(*byteslist)
        )
    )
    return stream.getvalue()


def unpack_history(
    bytestring: bytes
) -> tuple[dict[str, Any], dict[pd.Timestamp, pd.Series]]:
    byteslist = nary_unpack(zlib.decompress(bytestring))
    metadata = json.loads(byteslist[0])
    idates = np.frombuffer(
        byteslist[1],
        '|M8[ns]' if metadata['tzaware'] else '<M8[ns]'
    )
    hist = {}
    for idx, (bindex, bvalues) in enumerate(zip(*[iter(byteslist[2:])]*2)):
        index, values = numpy_deserialize(
            bindex, bvalues, metadata
        )
        series = pd.Series(
            values, index=index
        )
        if metadata['tzaware']:
            series = series.tz_localize(pytz.utc)
        hist[pd.Timestamp(idates[idx]).tz_localize(pytz.utc)] = series
    return metadata, hist


# groups

def serialize_index(df: pd.DataFrame) -> tuple[bytes, bytes]:
    dtype = df.index.dtype.str.encode('utf-8')
    if len(df):
        return dtype, np.ascontiguousarray(
            df.index.values
        ).view(np.uint8).data
    return dtype, b''


def serialize_values(df: pd.DataFrame) -> list[bytes]:
    """ convert each values of a dataframe into a list
    a series takes 3 list entries, for:
    * the dtype
    * the name
    * the values
    """
    byteslist = []
    if df.columns.dtype.name != 'object':
        df.columns = [str(col) for col in df.columns]
    for col in df.columns:
        series = df[col]
        byteslist.append(
            series.dtype.name.encode('utf-8')
        )
        byteslist.append(
            series.name.encode('utf-8')
        )
        byteslist.append(
            series.values.data.tobytes()
        )
    return byteslist


def pack_group(df: pd.DataFrame) -> bytes:
    bidtype, bindex = serialize_index(df)
    out = [bidtype, bindex]
    out += serialize_values(df)
    return zlib.compress(nary_pack(*out))


def unpack_group(bytestr: bytes) -> pd.DataFrame:
    byteslist = nary_unpack(zlib.decompress(bytestr))
    bidtype, bindex = byteslist[0:2]
    if len(bindex):
        index = np.frombuffer(
            bindex,
            bidtype
        )
    else:
        df = pd.DataFrame(index=pd.DatetimeIndex([]))
        if bidtype.startswith(b'|'):
            df.index = df.index.tz_localize(pytz.utc)
        return df

    values = {}
    iterbseries = zip(*[iter(byteslist[2:])] * 3)
    for bdtype, bname, bvalues in iterbseries:
        name = bname.decode('utf-8')
        values[name] = np.frombuffer(
            bvalues,
            bdtype.decode('utf-8')
        )

    df = pd.DataFrame(values, index=index)
    if bidtype.startswith(b'|'):
        df.index = df.index.tz_localize(pytz.utc)

    return df


def pack_group_history(
    hist: dict[pd.Timestamp, pd.DataFrame]
) -> bytes:
    byteslist = []
    byteslist.append(
        np.array(
            [tstamp.to_datetime64() for tstamp in hist],
            dtype='datetime64[ns]'
        ).view(np.uint8).data
    )
    for df in hist.values():
        bidtype, bindex = serialize_index(df)
        out = [bidtype, bindex]
        byteslist += out
        values = serialize_values(df)
        bnbvalues = str(len(values)).encode('utf-8')
        byteslist.append(bnbvalues)
        byteslist += values
    stream = io.BytesIO(
        zlib.compress(
            nary_pack(*byteslist)
        )
    )
    return stream.getvalue()


def unpack_group_history(
    bytestring: bytes
) -> dict[pd.Timestamp, pd.DataFrame]:
    byteslist = nary_unpack(zlib.decompress(bytestring))
    idates = np.frombuffer(
        byteslist[0],
        '|M8[ns]'
    )
    idates = [
        pd.Timestamp(idate).tz_localize(pytz.utc)
        for idate in idates
    ]

    hist = {}
    cursor = 1
    dfidx = 0
    while cursor < len(byteslist):
        bidtype = byteslist[cursor]
        bindex = byteslist[cursor + 1]
        nbvalues = int(byteslist[cursor + 2].decode('utf-8'))
        content = byteslist[cursor + 3 : cursor + 3 + nbvalues]
        cursor = cursor + 3 + nbvalues
        if len(bindex):
            index = np.frombuffer(
                bindex,
                bidtype
            )
        else:
            hist[idates[dfidx]] = pd.DataFrame()
            dfidx += 1
            continue
        values = {}
        iter_value = zip(*[iter(content)] * 3)
        for bdtype, bname, bvalues in iter_value:
            name = bname.decode('utf-8')
            values[name] = np.frombuffer(
                bvalues,
                bdtype.decode('utf-8')
            )
        df = pd.DataFrame(values, index=index)
        if bidtype.startswith(b'|'):
            df.index = df.index.tz_localize(pytz.utc)
        hist[idates[dfidx]] = df
        dfidx += 1
    return hist


# file binary serialisation

class rev:
    _size = 32
    __slots__ = 'revdate_ns', 'diffstart_ns', 'diffend_ns', 'index', 'metaid', '_tz'
    parser = struct.Struct('!qqqII')

    def __init__(
        self,
        revdate_ns: int,
        diffstart_ns: int,
        diffend_ns: int,
        index: int,
        metaid: int,
        tz: pytz.BaseTzInfo | None
    ) -> None:
        self.revdate_ns = revdate_ns
        self.diffstart_ns = diffstart_ns
        self.diffend_ns = diffend_ns
        self.index = index
        self.metaid = metaid
        self._tz = tz

    # Lazy Timestamp creation for backward compatibility
    @property
    def revdate(self):
        return pd.Timestamp(self.revdate_ns, 'ns', tz=pytz.utc)

    @property
    def diffstart(self):
        return pd.Timestamp(self.diffstart_ns, 'ns', tz=self._tz)

    @property
    def diffend(self):
        return pd.Timestamp(self.diffend_ns, 'ns', tz=self._tz)

    def __repr__(self):
        return (
            f'rev({self.revdate},{self.diffstart},{self.diffend},'
            f'{self.index},{self.metaid})'
        )

    @staticmethod
    def pack(
        revdate: pd.Timestamp,
        diffstart: pd.Timestamp,
        diffend: pd.Timestamp,
        index: int,
        metaid: int
    ) -> bytearray:
        buff = bytearray(rev._size)
        rev.parser.pack_into(buff, 0, revdate.value, diffstart.value, diffend.value, index, metaid)
        return buff

    @staticmethod
    def unpack(
        tz: pytz.BaseTzInfo | None,
        bytestr: bytes
    ) -> 'rev':
        revdate, diffstart, diffend, index, metaid = rev.parser.unpack_from(bytestr, 0)
        return rev(revdate, diffstart, diffend, index, metaid, tz)


class node:
    _size = 26
    __slots__ = 'start_ns', 'end_ns', 'parent', 'address', 'size', '_tz'
    parser = struct.Struct('!qqIIh')

    def __init__(
        self,
        start_ns: int,
        end_ns: int,
        parent: int,
        address: int,
        size: int,
        tz: pytz.BaseTzInfo | None
    ) -> None:
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.parent = parent
        self.address = address
        self.size = size
        self._tz = tz

    # Lazy Timestamp creation for backward compatibility
    @property
    def start(self):
        return pd.Timestamp(self.start_ns, 'ns', tz=self._tz)

    @property
    def end(self):
        return pd.Timestamp(self.end_ns, 'ns', tz=self._tz)

    def __repr__(self):
        return f'node({self.start},{self.end},{self.parent},{self.address},{self.size})'

    @staticmethod
    def pack(
        start: pd.Timestamp,
        end: pd.Timestamp,
        parent: int,
        address: int,
        datasize: int
    ) -> bytearray:
        buff = bytearray(node._size)
        node.parser.pack_into(buff, 0, start.value, end.value, parent, address, datasize)
        return buff

    @staticmethod
    def unpack(
        tz: pytz.BaseTzInfo | None,
        bytestr: bytes
    ) -> 'node':
        start, end, parent, address, size = node.parser.unpack_from(bytestr, 0)
        return node(start, end, parent, address, size, tz)

    @staticmethod
    def unpack_many(
        tz: pytz.BaseTzInfo | None,
        bytestr: bytes
    ):
        for start, end, parent, address, size in node.parser.iter_unpack(bytestr):
            yield node(start, end, parent, address, size, tz)


class iohelper:

    @staticmethod
    def serialize_ts(
        ts: pd.Series,
        isstr: bool,
        compressor = zlib
    ) -> bytes:
        index, values = numpy_serialize(ts, isstr)
        return compressor.compress(binary_pack(index, values))

    @staticmethod
    def chunks_to_ts(
        metadata: dict[str, Any],
        chunks: Iterable[bytes],
        compressor = zlib
    ) -> pd.Series:
        nchunks = (
            binary_unpack(compressor.decompress(chunk))
            for chunk in chunks
        )
        bseparator = b'\0' if metadata['value_type'] == 'object' else b''
        indexchunks, valueschunks = zip(*nchunks)
        index, values = numpy_deserialize(
            b''.join(indexchunks),
            bseparator.join(valueschunks),
            metadata
        )

        assert len(values) == len(index)
        ts = pd.Series(values, index=index)
        assert ts.index.is_monotonic_increasing

        if metadata.get('tzaware', False):
            return ts.tz_localize('UTC')
        return ts
