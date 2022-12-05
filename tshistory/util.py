import io
import os
import math
import struct
import json
from array import array
from collections import defaultdict
import logging
import threading
import tempfile
import shutil
import zlib
from functools import (
    partial,
    reduce
)
from contextlib import contextmanager
from pathlib import Path
from warnings import warn

import pytz
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64tz_dtype
from sqlalchemy.engine import url
from sqlalchemy.engine.base import Engine
from inireader import reader


def logme(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(level)
    return logger


def empty_series(tzaware, dtype='float64', name=None):
    return pd.Series(
        [],
        index=pd.DatetimeIndex(
            [],
            tz='UTC' if tzaware else None
        ),
        dtype=dtype,
        name=name
    )


@contextmanager
def tempdir(suffix='', prefix='tmp'):
    tmp = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
    try:
        yield Path(tmp)
    finally:
        shutil.rmtree(tmp)


def unflatten(flattened):
    """Build nested dictionaries from a flattened one, e.g
    foo.bar -> {'foo': {'bar': ...}}
    """
    nested = defaultdict(lambda: defaultdict(dict))
    for key, value in flattened.items():
        try:
            toplevel, newkey = [
                k.strip()
                for k in key.split('.', maxsplit=1)
            ]
        except ValueError:
            # nothing to unflatten
            nested[key] = value
            continue
        nested[toplevel][newkey] = value
    return nested


def get_cfg_path():
    if 'TSHISTORYCFGPATH' in os.environ:
        cfgpath = Path(os.environ['TSHISTORYCFGPATH'])
        if cfgpath.exists():
            return cfgpath
    cfgpath = Path('tshistory.cfg')
    if cfgpath.exists():
        return cfgpath
    cfgpath = Path('~/tshistory.cfg').expanduser()
    if cfgpath.exists():
        return cfgpath
    cfgpath = Path(
        os.environ.get('XDG_CONFIG_HOME', '~/.config'),
        'tshistory.cfg'
    ).expanduser()
    if cfgpath.exists():
        return cfgpath


def find_dburi(something: str) -> str:
    if something.startswith('http'):
        return something
    try:
        url.make_url(something)
    except Exception:
        pass
    else:
        return something

    # lookup in the env, then in cwd, then in the home
    cfgpath = get_cfg_path()
    if not cfgpath:
        raise Exception('could not use nor look up the db uri')

    try:
        cfg = reader(cfgpath)
        return cfg['dburi'][something]
    except Exception as exc:
        raise Exception(
            f'could not find the `{something}` entry in the '
            f'[dburi] section of the `{cfgpath.resolve()}` '
            f'conf file (cause: {exc.__class__.__name__} -> {exc})'
        )


# find available components

def find_most_specific_tshclass():
    try:
        from tshistory_refinery.tsio import timeseries
    except ImportError:
        from tshistory.tsio import timeseries
    return timeseries


def find_most_specific_http_client():
    try:
        from tshistory_refinery.http import RefineryClient as Client
    except ImportError:
        from tshistory.http.client import Client
    return Client


# tsio helpers

def ensuretz(adate):
    if adate is None:
        return
    if adate.tzinfo is None:
        return pd.Timestamp(adate, tz='UTC')
    return adate


def tzaware_serie(ts):
    return is_datetime64tz_dtype(ts.index)


def pruned_history(hist):
    if not hist:
        return hist
    idates = list(hist.keys())
    idate = idates[0]
    current = hist[idate]
    pruned = {
        idate: current
    }
    for idate in idates[1:]:
        newts = hist[idate]
        if not current.equals(newts):
            pruned[idate] = newts
            current = newts
    return pruned


def start_end(ts, notz=True):
    ts = ts.dropna()
    if not len(ts):
        return None, None
    start = ts.index.min()
    end = ts.index.max()
    if start.tzinfo is not None and notz:
        assert end.tzinfo is not None
        start = start.tz_convert('UTC').replace(tzinfo=None)
        end = end.tz_convert('UTC').replace(tzinfo=None)
    return start, end


def closed_overlaps(fromdate, todate):
    fromdate = "'-infinity'" if fromdate is None else '%(fromdate)s'
    todate = "'infinity'" if todate is None else '%(todate)s'
    return f'({fromdate}, {todate}) overlaps (tsstart, tsend + interval \'1 microsecond\')'


def inject_in_index(serie, revdate):
    mindex = [(revdate, valuestamp) for valuestamp in serie.index]
    serie.index = pd.MultiIndex.from_tuples(
        mindex,
        names=[
            'insertion_date', 'value_date'
        ]
    )


def compatible_date(tzaware, date):
    if date is None:
        return

    if not tzaware:
        return date.replace(tzinfo=None)

    if tzaware and date.tzinfo is None:
        return date.replace(tzinfo=pytz.utc)

    return date


# metadata

def series_metadata(ts):
    index = ts.index
    return {
        'tzaware': tzaware_serie(ts),
        'index_type': index.dtype.name,
        'index_dtype': index.dtype.str,
        'value_dtype': ts.dtypes.str,
        'value_type': ts.dtypes.name
    }


# serialisation helpers (binary, json)

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


def binary_pack(bytes1, bytes2):
    """assemble two byte strings into a unique byte string
    storing the size of the first string first
    this will permit to destructure back the two
    original byte strings

    """
    bytes1_size = struct.pack('!L', len(bytes1))
    return bytes1_size + bytes1 + bytes2


def binary_unpack(packedbytes):
    """get a compressed bytes stream and return the two embedded
    bytes strings

    """
    [bytes1_size] = struct.unpack(
        '!L', packedbytes[:4]
    )
    bytes2_offset = bytes1_size + 4
    return packedbytes[4:bytes2_offset], packedbytes[bytes2_offset:]


def nary_pack(*bytestr):
    sizes = [
        struct.pack('!L', len(b))
        for b in bytestr
    ]
    sizes_size = struct.pack('!L', len(sizes))
    stream = io.BytesIO()
    stream.write(sizes_size)
    stream.write(b''.join(sizes))
    for bstr in bytestr:
        stream.write(bstr)
    return stream.getvalue()


def nary_unpack(packedbytes):
    [sizes_size] = struct.unpack(
        '!L', packedbytes[:4]
    )
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
        name=name
    )
    if meta['tzaware']:
        series = series.tz_localize('UTC')
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
            series = series.tz_localize('UTC')
        serieslist.append(series)

    return serieslist


def pack_history(metadata, hist):
    byteslist = [json.dumps(metadata).encode('utf-8')]
    byteslist.append(
        np.array(
            [tstamp.to_datetime64() for tstamp in hist],
            dtype='datetime64'
        ).view(np.uint8).data.tobytes()
    )
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
            series = series.tz_localize('utc')
        hist[pd.Timestamp(idates[idx]).tz_localize('utc')] = series
    return metadata, hist


TO64 = set(
    (np.dtype('int'), np.dtype('int32'), np.dtype('int64'), np.dtype('float32'))
)


def num2float(pdobj):
    # get a Series or a Dataframe column
    if pdobj.dtype in TO64:
        return pdobj.astype('float64')
    return pdobj


def tojson(ts, precision=1e-14):
    return ts.to_json(date_format='iso',
                      double_precision=-int(math.log10(precision)))


def fromjson(jsonb, tsname, tzaware=False):
    if jsonb  ==  '{}':
        return empty_series(tzaware, name=tsname)
    series = _fromjson(jsonb, tsname).fillna(value=np.nan)
    if tzaware:
        if not getattr(series.index.dtype, 'tz', None):
            # from pandas 0.25 we are already good
            series.index = series.index.tz_localize('utc')
    else:
        if getattr(series.index.dtype, 'tz', None):
            # from pandas 0.25 we are now bad
            series.index = series.index.tz_localize(None)
    return series


def _fromjson(jsonb, tsname):
    wrapped = io.StringIO(jsonb) if isinstance(jsonb, str) else io.BytesIO(jsonb)
    result = pd.read_json(wrapped, typ='series', dtype=False)
    result.name = tsname
    result = num2float(result)
    return result

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
        return pd.DataFrame()

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
        df.index = df.index.tz_localize('UTC')

    return df


def pack_group_history(hist):
    byteslist = []
    byteslist.append(
        np.array(
            [tstamp.to_datetime64() for tstamp in hist],
            dtype='datetime64'
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
        pd.Timestamp(idate).tz_localize('utc')
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
            df.index = df.index.tz_localize('UTC')
        hist[idates[dfidx]] = df
        dfidx += 1
    return hist


# diff/patch utilities

def _populate(index, values, outindex, outvalues):
    mask = np.in1d(outindex, index, assume_unique=True)
    outvalues[
        mask
    ] = values


def patch(base, diff):
    """update base series with differing values from diff series:
    * new points added
    * updated points
    """
    assert base is not None
    assert diff is not None

    if len(base) == 0:
        return diff

    if len(diff) == 0:
        return base

    if base.dtype == 'object':
        basei = base.index
        diffi = diff.index
        newindex = basei.union(diffi)
        patched = pd.Series([0] * len(newindex), index=newindex)
        patched[basei] = base
        patched[diffi] = diff
        patched.name = base.name
        return patched

    index1 = base.index.values
    index2 = diff.index.values

    uindex = np.union1d(
        index1,
        index2
    )
    uvalues = np.zeros(len(uindex))

    _populate(index1, base.values, uindex, uvalues)
    _populate(index2, diff.values, uindex, uvalues)

    tz = base.index.dtype.tz.zone if is_datetime64tz_dtype(base.index) else None
    series = pd.Series(
        uvalues,
        index=uindex,
        name=base.name
    )
    if tz:
        series.index = series.index.tz_localize(tz)
    return series


def patchmany(series):
    assert len(series), 'patchmany wants at least one series'
    first = series[0]

    if len(series) == 1:
        return first

    if first.dtype == 'object':
        final = first
        for ts in series[1:]:
            final = patch(final, ts)
        return final

    series = [
        ts for ts in series
        if len(ts)
    ]
    if not series:
        return first

    uindex = reduce(
        np.union1d,
        (ts.index.values for ts in series)
    )
    uvalues = np.zeros(len(uindex))

    for ts in series:
        _populate(ts.index.values, ts.values, uindex, uvalues)

    tz = first.index.dtype.tz.zone if is_datetime64tz_dtype(first.index) else None
    series = pd.Series(
        uvalues,
        index=uindex,
        name=first.name
    )
    if tz:
        series.index = series.index.tz_localize(tz)
    return series


def diff(base, other, _precision=1e-14):
    if base is None:
        return other
    base = base.dropna()
    if not len(base):
        return other

    # intersection
    mask_overlap = np.isin(
        other.index.values,
        base.index.values,
        assume_unique=True
    )
    base_overlap = base[other.index[mask_overlap]]
    other_overlap = other[mask_overlap]

    # equal values at intersection
    if base.dtype == 'float64':
        mask_equal = np.isclose(base_overlap, other_overlap,
                                rtol=0, atol=_precision)
    else:
        mask_equal = base_overlap == other_overlap

    if base.dtype == 'float64':
        mask_na_equal = (
            np.isnan(base_overlap.values) &
            np.isnan(other_overlap.values)
        )
    else:
        mask_na_equal = base_overlap.isnull() & other_overlap.isnull()

    mask_equal = mask_equal | mask_na_equal

    # series of updated elements
    diff_overlap = other_overlap[~mask_equal]
    # series of new elements brought by the `other` side
    diff_new = other[~mask_overlap]
    diff_new = diff_new.dropna()

    return pd.concat([diff_overlap, diff_new]).sort_index()


# //ism helper

def threadpool(maxthreads):
    L = logging.getLogger('parallel')

    def run(func, argslist):
        count = 0
        threads = []
        L.debug('// run %s %s', func.__name__, len(argslist))

        # initial threads
        for count, args in enumerate(argslist, start=1):
            th = threading.Thread(target=func, args=args)
            threads.append(th)
            L.debug('// start thread %s', th.name)
            th.daemon = True
            th.start()
            if count == maxthreads:
                break

        while threads:
            for th in threads[:]:
                th.join(1. / maxthreads)
                if not th.is_alive():
                    threads.remove(th)
                    L.debug('// thread %s exited, %s remaining', th.name, len(threads))
                    if count < len(argslist):
                        newth = threading.Thread(target=func, args=argslist[count])
                        threads.append(newth)
                        L.debug('// thread %s started', newth.name)
                        newth.daemon = True
                        newth.start()
                        count += 1

    return run


# transaction wrapper

def _set_cache(txobj):
    txobj.cache = {
        'series_tablename': {},
        'metadata': {}
    }
    return txobj


def tx(func):
    " a decorator to check that the first method argument is a transaction "
    def check_tx_and_call(self, cn, *a, **kw):
        # safety belt to make sure important api points are tx-safe
        if not isinstance(cn, Engine):
            if not cn.in_transaction():
                raise TypeError('You must use a transaction object')
        else:
            with cn.begin() as txcn:
                return func(self, _set_cache(txcn), *a, **kw)

        return func(self, _set_cache(cn), *a, **kw)
    check_tx_and_call.__name__ = func.__name__
    return check_tx_and_call


# bisection

def bisect_search(values, value):
    """return an index j such that ``value`` is between values[j]
    and values[j+1].
    values must be monotonic increasing.

    j=-1 or j=len(values) is returned to indicate that ``value`` is
    out of range below and above respectively.

    thanks to https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    """
    n = len(values)
    first, last = values[0], values[-1]
    if value < first:
        return -1
    if value > last:
        return n
    if value == first:
        return 0
    if value == last:
        return n - 1

    jl = 0
    ju = n - 1
    while ju - jl > 1:
        jm = (ju+jl) >> 1
        if value >= values[jm]:
            jl = jm
        else:
            ju = jm

    return jl


# api extensions helper

def extend(klass):
    """Helper to add methods to the base api class

    e.g.

    .. code-block::python
     from tshistory.api import mainsource

     @extend(mainsource)
     def formula(name):
         return self.tsh.formula(self.engine, name)

    """

    def decorator(func):
        name = func.__name__
        if getattr(klass, name, None) is not None:
            warn(f'replacing already existing method {name} over {klass}')
        setattr(klass, name, func)
        return func

    return decorator
