from __future__ import annotations
import hashlib
import io
import math
import re
from collections import defaultdict
import logging
import threading
import tempfile
import shutil
from datetime import (
    datetime,
    timedelta
)
from importlib_metadata import entry_points
from contextlib import contextmanager
from pathlib import Path
from warnings import warn
from typing import (
    Any,
    Optional,
    Union,
    Generator,
    Callable
)

import pytz
import numpy as np
import pandas as pd
from sqlhelp import select, pgapi
from dbcache.api import kvstore


def logme(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(level)
    return logger


def empty_series(
    tzaware: bool,
    dtype: str = 'float64',
    name: Optional[str] = None
) -> pd.Series:
    return pd.Series(
        [],
        index=pd.DatetimeIndex(
            [],
            tz='UTC' if tzaware else None
        ),
        dtype=dtype,
        name=name
    )


# other

@contextmanager
def tempdir(
    suffix: str = '',
    prefix: str = 'tmp'
) -> Generator[Path, None, None]:
    tmp = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
    try:
        yield Path(tmp)
    finally:
        shutil.rmtree(tmp)


def unflatten(flattened: dict[str, Any]) -> dict[str, Any]:
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


# generic functions

def objects(name: str) -> list[Any]:
    objs = set([
        ep.load()
        for ep in entry_points(name=name)
    ])
    return list(objs)


_DONE = False

def ensure_plugin_registration() -> None:
    global _DONE
    if _DONE:
        return

    # the whole purpose of this entry point is to make sure things are
    # imported, hence objects from pluginsq registered in their proper
    # places
    for ep in entry_points(group='forceimports'):
        ep.load()
    _DONE = True


# versions

class NoVersion(Exception):
    pass


class VersionMismatch(Exception):
    pass


def read_versions(
    uri: str,
    namespace: str,
    version_string: str = 'tshistory-version'
) -> tuple[str, str]:
    from tshistory import __version__ as code_version
    store = kvstore(uri, f'{namespace}-kvstore')
    try:
        stored_version = store.get('tshistory-version')
    except Exception:
        raise NoVersion(
            f'version of the software ({code_version}) '
            f'and the db  differ. '
            'Please install it or run the `migrate` command'
        )

    return stored_version, code_version


def ensure_versions(uri: str, namespace: str) -> None:
    stored_version, code_version = read_versions(uri, namespace)
    if stored_version != code_version:
        raise VersionMismatch(
            f'version of the software ({code_version}) '
            f'and the db ({stored_version}@{namespace}) differ. '
            'Please run the `migrate` command'
        )


def run_migrations(
    uri: str,
    interactive: bool,
    initial: bool,
    force: bool,
    last: bool,
    namespace: str
) -> None:
    for migrator in sorted(objects('migrator'), key=lambda x: x._order):
        migrator(
            uri, namespace, interactive=interactive, start=initial, force=force, last=last
        ).run_migrations()


# find available components

def find_most_specific_tshclass(storage: str) -> Any:
    objs = sorted(
        [
            obj for obj in objects('tshclass')
            if obj.storage == storage
        ],
        key=lambda x:x.index
    )
    return objs[-1]


def find_most_specific_http_client() -> Any:
    objs = sorted(
        objects('httpclient'),
        key=lambda x:x.index
    )
    return objs[-1]


# tsio helpers

class ts(str):
    __slots__ = 'name', 'imeta', 'meta', 'source', 'kind'

    def __init__(self, _name: str, *_a: Any, **_kw: Any) -> None:
        # this is useless but it helps pytype understand
        # what is going on on the call sites
        # (pytype visibly doesnt know how to exploit __new__
        # for this purpose)
        super().__init__()

    def __new__(
        cls,
        name: str,
        imeta: Optional[dict[str, Any]] = None,
        meta: Optional[dict[str, Any]] = None,
        source: str = 'local',
        kind: str = 'primary'
    ) -> ts:
        obj = str.__new__(cls, name)
        obj.imeta = imeta
        obj.meta = meta
        obj.source = source
        obj.kind = kind
        return obj

    def to_json(self) -> dict[str, Any]:
        return {
            'name': self,
            'imeta': self.imeta,
            'meta': self.meta,
            'source': self.source,
            'kind': self.kind
        }


def to_nanoseconds(
    timestamp: Optional[Union[pd.Timestamp, datetime]]
) -> Optional[int]:
    """Convert various timestamp types to nanoseconds for fast comparison"""
    if timestamp is None:
        return None
    if hasattr(timestamp, 'value'):
        # pandas Timestamp
        return timestamp.value
    # datetime.datetime - convert to pandas first
    return pd.Timestamp(timestamp).value


def ensuretz(
    adate: Optional[Union[pd.Timestamp, datetime]]
) -> Optional[pd.Timestamp]:
    if adate is None:
        return
    if adate.tzinfo is None:
        return pd.Timestamp(adate, tz='UTC')
    return adate


def applytz(
    date: Union[pd.Timestamp, datetime],
    tzaware: bool
) -> pd.Timestamp:
    pd_date = pd.Timestamp(date)
    if pd_date.tzinfo is None and not tzaware:
        return pd_date
    elif pd_date.tzinfo is not None and tzaware:
        return pd_date
    elif pd_date.tzinfo is None and tzaware:
        return pd.Timestamp(pd_date, tz='UTC')
    else:
        return pd_date.tz_localize(None)


def tzaware_series(ts: pd.Series) -> bool:
    return isinstance(ts.index.dtype, pd.DatetimeTZDtype)

tzaware_serie = tzaware_series


def pruned_history(
    hist: dict[pd.Timestamp, pd.Series]
) -> dict[pd.Timestamp, pd.Series]:
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


def start_end(
    ts: pd.Series,
    notz: bool = True
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
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


def closed_overlaps(
    fromdate: Optional[pd.Timestamp],
    todate: Optional[pd.Timestamp]
) -> str:
    fromdate = "'-infinity'" if fromdate is None else '%(fromdate)s'
    todate = "'infinity'" if todate is None else '%(todate)s'
    return (
        f'({fromdate}, {todate}) '
        'overlaps '
        '(diffstart - interval \'1 microsecond\', '
        ' diffend + interval \'1 microsecond\')'
    )


def inject_in_index(
    series: pd.Series,
    revdate: pd.Timestamp
) -> None:
    mindex = [(revdate, valuestamp) for valuestamp in series.index]
    series.index = pd.MultiIndex.from_tuples(
        mindex,
        names=[
            'insertion_date', 'value_date'
        ]
    )


def compatible_date(
    tzaware: bool,
    date: Optional[Union[pd.Timestamp, datetime]]
) -> Optional[pd.Timestamp]:
    if date is None:
        return

    # Ensure we have a pandas Timestamp with nanosecond precision
    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)

    if not tzaware:
        return date.replace(tzinfo=None)

    if tzaware and date.tzinfo is None:
        return date.replace(tzinfo=pytz.utc)

    return date.tz_convert('utc')


def infer_freq(ts: pd.Series) -> tuple[pd.Timedelta, float]:
    assert len(ts) > 1, 'infer_freq needs a series of size > 1'
    index = ts.index.to_series()
    deltas = (index - index.shift(1)).dropna()
    freq = deltas.median()

    conform_intervals = sum(deltas == freq)
    return freq, conform_intervals / len(deltas)


def with_inferred_freq(
    ts: pd.Series,
    from_value_date: Optional[pd.Timestamp] = None,
    to_value_date: Optional[pd.Timestamp] = None
) -> pd.Series:
    if len(ts) < 3 :
        # we can't infer anything
        return ts

    ts_start = ts.index[0]
    ts_end = ts.index[-1]
    old_index = ts.index
    freq = infer_freq(ts)[0]
    tzaware = ts_start.tz is not None
    to_value_date = compatible_date(tzaware, to_value_date)
    from_value_date = compatible_date(tzaware, from_value_date)

    if from_value_date is None and to_value_date is None:
        new_index = pd.date_range(
            start=ts_start,
            end=ts_end,
            freq=freq
        )
        return ts.reindex(new_index.union(old_index))

    if from_value_date is None:
        new_index = pd.date_range(
            start=ts_start,
            end=to_value_date,
            freq=freq
        )
        return ts.reindex(new_index.union(old_index))

    if to_value_date is None:
        new_index = pd.date_range(
            start=ts_end,
            end=from_value_date,
            freq=-freq
        ).sort_values()
        return ts.reindex(new_index.union(old_index))

    # we have to build the index in two parts
    new_index = pd.date_range(
        start=ts_start,
        end=to_value_date,
        freq=freq
    )
    complement = pd.date_range(
        start=ts_start,
        end=from_value_date,
        freq=-freq
    )
    new_index = new_index.union(complement).sort_values()
    return ts.reindex(new_index.union(old_index))


def guard_insert(
    newts: pd.Series,
    name: str,
    author: str,
    metadata: Optional[dict[str, Any]],
    insertion_date: Optional[datetime]
) -> pd.Series:
    assert len(name), 'Name is an empty string'
    assert isinstance(author, str), 'Author is not a string'
    assert metadata is None or isinstance(metadata, dict), (
        f'Bad format for metadata ({repr(metadata)})'
    )
    if insertion_date is not None:
        assert isinstance(insertion_date, datetime), 'Bad format for insertion date'
        assert insertion_date.tzinfo is not None, (
            f'for "{name}", the specified revision date '
            f'"{insertion_date}" must be tzaware'
        )
    assert isinstance(newts, pd.Series), 'Not a pd.Series'
    index = newts.index
    assert isinstance(index, pd.DatetimeIndex), 'You must provide a DatetimeIndex'
    assert not index.duplicated().any(), 'There are some duplicates in the index'

    assert index.notna().all(), 'The index contains NaT entries'
    if index.tz is not None:
        newts.index = index.tz_convert('UTC')
    if not index.is_monotonic_increasing:
        newts = newts.sort_index()

    return num2float(newts)


def guard_query_dates(*dates: Optional[datetime]) -> None:
    assert all(
        isinstance(dt, datetime)
        for dt in filter(None, dates)
    ), 'all query dates must be datetime-compatible objects'


def hash64(text: str) -> int:
    # hasher from text to 64 bits ints
    seed = text.encode('utf-8')
    hash_digest = hashlib.shake_128(seed).digest(8)
    return int.from_bytes(hash_digest, byteorder='big', signed=True)


# tsio search helpers

def make_find_sqlquery(
    ns: str,
    target: str,
    items: list[str],
    query: Any,
    limit: int,
    meta: bool
) -> Any:
    if meta:
        items += ['internal_metadata', 'metadata']
    q = select(
        *items
    ).table(
        f'"{ns}".{target} as reg'
    ).order('name', 'asc')
    sql, kw = query.sql(ns)
    if sql:
        q.where(sql, **kw)
    if limit:
        q.limit(limit)

    return q


# timedelta (de)serialisation

def delta_isoformat(td: timedelta) -> str:
    return f'P{td.days}DT0H0M{td.seconds}S'


_DELTA = re.compile('P(.*)DT(.*)H(.*)M(.*)S')
def parse_delta(td: str) -> pd.Timedelta:
    match = _DELTA.match(td)
    if not match:
        raise Exception(f'unparseable time delta `{td}`')
    days, hours, minutes, seconds = match.groups()
    return pd.Timedelta(
        days=int(days), hours=int(hours),
        minutes=int(minutes), seconds=int(seconds)
    )


# metadata

def series_metadata(ts: pd.Series) -> dict[str, Any]:
    index = ts.index
    return {
        'tzaware': tzaware_series(ts),
        'index_type': index.dtype.name,
        'index_dtype': index.dtype.str,
        'value_dtype': ts.dtypes.str,
        'value_type': ts.dtypes.name
    }


# json serialisation

TO64 = set(
    (np.dtype('int'), np.dtype('int32'), np.dtype('int64'), np.dtype('float32'))
)


def num2float(pdobj: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    # get a Series or a Dataframe column
    if pdobj.dtype in TO64:
        return pdobj.astype('float64')
    return pdobj


def tojson(ts: pd.Series, precision: float = 1e-14) -> str:
    return ts.to_json(
        date_format='iso',
        date_unit='ns',
        double_precision=-int(math.log10(precision))
    )


def tojson2(ts: pd.Series) -> dict[str, Any]:
    return {
        k.isoformat(): v
        for k, v in ts.items()
    }


def fromjson(
    jsonb: Union[str, bytes],
    tsname: str,
    tzaware: bool = False
) -> pd.Series:
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


def _fromjson(
    jsonb: Union[str, bytes],
    tsname: str
) -> pd.Series:
    wrapped = io.StringIO(jsonb) if isinstance(jsonb, str) else io.BytesIO(jsonb)
    result = pd.read_json(wrapped, typ='series', dtype=False)
    result.name = tsname
    result = num2float(result)
    return result


# diff/patch utilities

def index_zone(ts: pd.Series) -> Optional[str]:
    if pd.__version__.startswith('2'):
        return ts.index.dtype.tz
    return ts.index.dtype.tz.zone


def _populate(
    index: np.ndarray,
    values: np.ndarray,
    outindex: np.ndarray,
    outvalues: np.ndarray
) -> None:
    mask = np.in1d(outindex, index, assume_unique=True)
    outvalues[
        mask
    ] = values


def patch(base: pd.Series, diff: pd.Series) -> pd.Series:
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
        patched = pd.Series([0] * len(newindex), index=newindex, dtype='object')
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

    tz = index_zone(base) if tzaware_series(base) else None
    series = pd.Series(
        uvalues,
        index=uindex,
        name=base.name,
        dtype='float64'
    )
    if tz:
        series.index = series.index.tz_localize(tz)
    return series


def patchmany(series: Union[list[pd.Series], tuple[pd.Series, ...]]) -> pd.Series:
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

    # build union index more efficiently
    all_indices = np.concatenate([ts.index.values for ts in series])
    uindex = np.unique(all_indices)
    uvalues = np.zeros(len(uindex))

    # single-pass population using searchsorted
    for ts in series:
        if len(ts) == 0:
            continue

        # find positions in uindex for this series' indices
        positions = np.searchsorted(uindex, ts.index.values)

        # verify positions are correct (safety check)
        mask = (positions < len(uindex)) & (uindex[positions] == ts.index.values)
        valid_positions = positions[mask]
        valid_values = ts.values[mask]

        # update values at found positions
        uvalues[valid_positions] = valid_values

    # assumption: all series are tzaware or naive
    tz = index_zone(first) if tzaware_series(first) else None
    series = pd.Series(
        uvalues,
        index=uindex,
        name=first.name
    )
    if tz:
        series.index = series.index.tz_localize(tz)
    return series


def diff(
    base: Optional[pd.Series],
    other: pd.Series,
    _precision: float = 1e-14
) -> pd.Series:
    if base is None:
        return other
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
        mask_equal = np.abs(base_overlap.values - other_overlap.values) < _precision
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
    diff_new = diff_new

    return pd.concat([diff_overlap, diff_new]).sort_index()


# stuff

def diffs(
    cn: Any,
    tsh: Any,
    name: str,
    tablename: str,
    from_idate: Optional[pd.Timestamp],
    to_idate: Optional[pd.Timestamp]
) -> Generator[tuple[int, pd.Timestamp, pd.Series], None, None]:
    from tshistory.codecs import iohelper
    meta = tsh.internal_metadata(cn, name)
    tzaware = tsh.tzaware(cn, name)
    if from_idate:
        ts = tsh.get(
            cn,
            name,
            revision_date=from_idate - timedelta(milliseconds=1)
        )
    else:
        ts = empty_series(tzaware)

    # revs
    revsql = select(
        'id', 'snapshot', 'insertion_date'
    ).table(f'"{tsh.namespace}.revision"."{tablename}"'
    ).order('id', direction='asc')
    if from_idate:
        revsql.where(
            'insertion_date >= %(fromdate)s',
            fromdate=from_idate
        )
    if to_idate:
        revsql.where(
            'insertion_date <= %(to_idate)s',
            to_idate=to_idate
        )

    allrevs = [
        (csid, snapshot, idate)
        for csid, snapshot, idate in revsql.do(cn).fetchall()
    ]

    chunksql = (
        f'select id, parent, chunk '
        f'from "{tsh.namespace}.snapshot"."{tablename}" '
    )

    # very greedy, let's try to limit this using
    # to_idate later ...
    chunks = {
        c.id: (c.parent, c.chunk)
        for c in cn.execute(
                chunksql
        ).fetchall()
    }

    _cache = {}

    def patched(top, prev, items):
        if not items:
            return prev
        items.reverse()
        # non-appends will cause costly recomputations
        # we might want to find a middle ground
        if len(_cache) > 100:
            for ci in list(_cache.keys())[:20]:
                _cache.pop(ci)
        out = _cache[top] = patch(
            prev,
            iohelper.chunks_to_ts(meta, items)
        )
        return out

    def buildseries(top):
        items = []

        parent = top
        while parent in chunks:
            grandpa, chunk = chunks[parent]
            items.append(chunk)  # bytes
            if grandpa in _cache:
                # found a known parent: let's patch and remember
                return patched(top, _cache[grandpa], items)
            parent = grandpa

        # initial revision or full new series
        return patched(top, empty_series(tzaware), items)

    for csid, snapid, idate in allrevs:
        current = buildseries(snapid)
        tsdiff = diff(ts, current)
        yield csid, idate, tsdiff
        ts = current


# //ism helper

def threadpool(maxthreads: int) -> Callable[[Callable, list[tuple]], None]:
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

_required_keys = ('internal_metadata', 'series_tablename', 'series_path')

def ensure_cache(cnobj: Any) -> Any:
    """Ensure cache exists and has all required keys. Safe to call multiple times."""
    if not hasattr(cnobj, 'cache'):
        cnobj.cache = {}

    # ensure all required cache keys exist
    for key in _required_keys:
        if key not in cnobj.cache:
            cnobj.cache[key] = {}

    return cnobj


def tx(func: Callable) -> Callable:
    " a decorator to check that the first method argument is a transaction "
    def check_tx_and_call(self, cn, *a, **kw):
        # safety belt to make sure important api points are tx-safe
        if isinstance(cn, pgapi.pgdb):
            with cn.begin() as txcn:
                return func(self, ensure_cache(txcn), *a, **kw)

        return func(self, ensure_cache(cn), *a, **kw)
    check_tx_and_call.__name__ = func.__name__
    return check_tx_and_call


# bisection

def bisect_search(values: Union[np.ndarray, list[float]], value: float) -> int:
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


# python subclasses helper

def isleaf(cls: type) -> bool:
    return not cls.__subclasses__()


def all_subclasses(cls: type) -> set[type]:
    return set(
        cls.__subclasses__()
    ).union(
        s for c in cls.__subclasses__()
        for s in all_subclasses(c)
    )


def leafclasses(cls: type) -> set[type]:
    return {
        klass
        for klass in all_subclasses(cls)
        if isleaf(klass)
    }


# api extensions helper

def extend(klass: type) -> Callable[[Callable], Callable]:
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


# series replication helper

def replicate_series(
    tsa_origin: Any,
    tsa_target: Any,
    origname: str,
    targetname: Optional[str] = None,
    from_insertion_date: Optional[pd.Timestamp] = None,
    insertion_freq_offset: Optional[str] = None
) -> None:
    if not tsa_origin.exists(origname):
        raise ValueError(f'"{origname}" is unknown.')

    targetname = origname if targetname is None else targetname

    insertion_dates = tsa_origin.insertion_dates(
        origname,
        from_insertion_date=from_insertion_date
    )
    if tsa_target.exists(targetname):
        last_known_insertion_date = tsa_target.insertion_dates(targetname)[-1]
        insertion_dates = [
            insertion_date
            for insertion_date in insertion_dates
            if insertion_date > last_known_insertion_date
        ]

    if insertion_freq_offset is not None and len(insertion_dates) > 1:
        ts_insertion = pd.Series(data=0, index=insertion_dates)
        ifreq, _ = infer_freq(ts_insertion)
        # convert the resample_freq to timedelta
        dates = pd.date_range(
            pd.Timestamp('2000-1-1'),
            periods=2,
            freq=insertion_freq_offset
        )
        resample_delta = dates[1] - dates[0]

        if ifreq < resample_delta:
            ts_insertion = ts_insertion.resample(insertion_freq_offset).mean()
            insertion_dates = ts_insertion.index

    for idate in insertion_dates:
        ts = tsa_origin.get(origname, revision_date=idate)
        tsa_target.update(
            targetname,
            ts,
            insertion_date=idate,
            author='replication'
        )

    metadata = tsa_origin.metadata(origname)
    tsa_target.replace_metadata(targetname, metadata)


def replicate_basket(
    tsa_origin: Any,
    tsa_target: Any,
    basket_name: str,
    from_insertion_date: Optional[pd.Timestamp] = None,
    insertion_freq_offset: Optional[str] = None,
    prefix: str = '',
    suffix: str = ''
) -> None:
    series = tsa_origin.basket(basket_name)
    poolrun = threadpool(16)

    poolrun(
        replicate_series,
        [
            (
                tsa_origin,
                tsa_target,
                origname,
                prefix + origname + suffix,
                from_insertion_date,
                insertion_freq_offset
            )
            for origname in series
        ]
    )


# checkdiff helper

def checkdiffs_for_name(
    engine: Any,
    tsa: Any,
    name: str
) -> None:
    tsh = tsa.tsh
    with engine.begin() as cn:
        cn.cache = {'series_tablename': {}}
        tablename = tsh._series_to_tablename(cn, name)

    things = engine.execute(
        f'select insertion_date, diffstart, diffend '
        f'from "tsh.revision"."{tablename}" '
        f'order by insertion_date'
    )
    h = tsa.history(name, diffmode=True, keepnans=True)
    tzaware = tsa.tsh.tzaware(engine, name)

    for idate, start, end in things.fetchall():
        if not tzaware:
            start = pd.Timestamp(start).tz_localize(None)
            end = pd.Timestamp(end).tz_localize(None)
        print(idate)
        ts = h[idate]
        if ts.index[0] != start:
            print('-> start', ts.index[0], start)
        if ts.index[-1] != end:
            print('-> end', ts.index[0], end)
