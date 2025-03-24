from array import array
import io
import json
import pytz
import struct
import zlib

import numpy as np
import pandas as pd



# binary

def numpy_serialize(series, isstr=False):
    # use `view` as a workarround for "cannot include dtype 'M' in a buffer"
    if len(series):
        bindex = np.ascontiguousarray(
            series.index.values
        ).view(np.uint8).data.tobytes()
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


def binary_pack(bytes1, bytes2):
    """assemble two byte strings into a unique byte string
    storing the size of the first string first
    this will permit to destructure back the two
    original byte strings

    """
    bytes1_size = SIZE.pack(len(bytes1))
    return bytes1_size + bytes1 + bytes2


def binary_unpack(packedbytes):
    """get a compressed bytes stream and return the two embedded
    bytes strings

    """
    [bytes1_size] = SIZE.unpack(packedbytes[:4])
    bytes2_offset = bytes1_size + 4
    return packedbytes[4:bytes2_offset], packedbytes[bytes2_offset:]


def nary_pack(*bytestr):
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


def nary_unpack(packedbytes):
    [sizes_size] = SIZE.unpack(packedbytes[:4])
    payloadoffset = 4 + sizes_size * 4
    sizes = struct.unpack(
        f'!{"L"*sizes_size}',
        packedbytes[4: payloadoffset]
    )
    fmt = ''.join('%ss' % size for size in sizes)
    return struct.unpack(fmt, packedbytes[payloadoffset:])


def numpy_deserialize(bindex, bvalues, metadata):
    """produce a pandas series from serialized index and values (numpy
    arrays)

    """
    # array is a workaround for an obscure bug with pandas.isin
    index = np.frombuffer(
        array('d', bindex),
        metadata['index_dtype'] # NOTE: this is not sufficient
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


def pack_series(metadata, series, compressor=zlib.compress):
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


def unpack_series(name, bytestream, decompressor=zlib.decompress):
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


def pack_many_series(serieslist, compressor=zlib.compress):
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


def unpack_many_series(bytestream, decompressor=zlib.decompress):
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


def pack_history(metadata, hist):
    byteslist = [json.dumps(metadata).encode('utf-8')]
    arr = np.array(
        [tstamp.to_datetime64() for tstamp in hist],
        dtype='datetime64[ns]'
    ).view(np.uint8).data.tobytes()
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


def unpack_history(bytestring):
    byteslist = nary_unpack(zlib.decompress(bytestring))
    metadata = json.loads(byteslist[0])
    print('D', byteslist[1])
    idates = np.frombuffer(
        array('d', byteslist[1]),
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

def serialize_index(df):
    dtype = df.index.dtype.str.encode('utf-8')
    if len(df):
        return dtype, np.ascontiguousarray(
            df.index.values
        ).view(np.uint8).data.tobytes()
    return dtype, b''


def serialize_values(df):
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


def pack_group(df):
    bidtype, bindex = serialize_index(df)
    out = [bidtype, bindex]
    out += serialize_values(df)
    return zlib.compress(nary_pack(*out))


def unpack_group(bytestr):
    byteslist = nary_unpack(zlib.decompress(bytestr))
    bidtype, bindex = byteslist[0:2]
    if len(bindex):
        index = np.frombuffer(
            array('d', bindex),
            bidtype
        )
    else:
        return pd.DataFrame(index=pd.DatetimeIndex([]))

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


def pack_group_history(hist):
    byteslist = []
    byteslist.append(
        np.array(
            [tstamp.to_datetime64() for tstamp in hist],
            dtype='datetime64[ns]'
        ).view(np.uint8).data.tobytes()
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


def unpack_group_history(bytestring):
    byteslist = nary_unpack(zlib.decompress(bytestring))
    idates = np.frombuffer(
        array('d', byteslist[0]),'|M8[ns]'
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
                array('d', bindex),
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
    __slots__ = 'revdate', 'diffstart', 'diffend', 'index', 'metaid'
    parser = struct.Struct('!qqqII')

    def __init__(self, revdate, diffstart, diffend, index, metaid):
        self.revdate = revdate
        self.diffstart = diffstart
        self.diffend = diffend
        self.index = index
        self.metaid = metaid

    def __repr__(self):
        return (
            f'rev({self.revdate},{self.diffstart},{self.diffend},'
            f'{self.index},{self.metaid})'
        )

    @staticmethod
    def pack(revdate, diffstart, diffend, index, metaid):
        buff = bytearray(rev._size)
        rev.parser.pack_into(buff, 0, revdate.value, diffstart.value, diffend.value, index, metaid)
        return buff

    @staticmethod
    def unpack(tz, bytestr):
        buff = array('B', bytestr)
        revdate, diffstart, diffend, index, metaid = rev.parser.unpack_from(buff)
        return rev(
            pd.Timestamp(revdate, 'ns', tz=pytz.utc),
            pd.Timestamp(diffstart, 'ns', tz=tz),
            pd.Timestamp(diffend, 'ns', tz=tz),
            index,
            metaid
        )


class node:
    _size = 26
    __slots__ = 'start', 'end', 'parent', 'address', 'size'
    parser = struct.Struct('!qqIIh')

    def __init__(self, start, end, parent, address, size):
        self.start = start
        self.end = end
        self.parent = parent
        self.address = address
        self.size = size

    def __repr__(self):
        return f'node({self.start},{self.end},{self.parent},{self.address},{self.size})'

    @staticmethod
    def pack(start, end, parent, address, datasize):
        buff = bytearray(node._size)
        node.parser.pack_into(buff, 0, start.value, end.value, parent, address, datasize)
        return buff

    @staticmethod
    def unpack(tz, bytestr):
        buff = array('B', bytestr)
        start, end, parent, address, size = node.parser.unpack_from(buff)
        return node(
            pd.Timestamp(start, 'ns', tz=tz),
            pd.Timestamp(end, 'ns', tz=tz),
            parent,
            address,
            size
        )

    @staticmethod
    def unpack_many(tz, bytestr):
        buff = array('B', bytestr)
        for start, end, parent, address, size in node.parser.iter_unpack(buff):
            yield node(
                pd.Timestamp(start, 'ns', tz=tz),
                pd.Timestamp(end, 'ns', tz=tz),
                parent,
                address,
                size
            )


class iohelper:

    @staticmethod
    def serialize_ts(ts, isstr, compressor=zlib):
        index, values = numpy_serialize(ts, isstr)
        return compressor.compress(binary_pack(index, values))

    @staticmethod
    def chunks_to_ts(metadata, chunks, compressor=zlib):
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
