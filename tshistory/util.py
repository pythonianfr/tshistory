import os
import math
import zlib
import hashlib
import logging
import threading
import tempfile
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_datetimetz
from sqlalchemy.sql.expression import text
from sqlalchemy.engine import url
from inireader import reader


@contextmanager
def tempdir(suffix='', prefix='tmp'):
    tmp = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
    try:
        yield Path(tmp)
    finally:
        shutil.rmtree(tmp)


def get_cfg_path():
    if 'TSHISTORYCFGPATH' is os.environ:
        cfgpath = Path(os.environ['TSHISTORYCFGPATH'])
        if cfgpath.exists():
            return cfgpath
    cfgpath = Path('tshistory.cfg')
    if cfgpath.exists():
        return cfgpath
    cfgpath = Path('~/tshistory.cfg').expanduser()
    if cfgpath.exists():
        return cfgpath


def find_dburi(something: str) -> str:
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


def tzaware_serie(ts):
    return is_datetimetz(ts.index)


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
    fromdate = "'-infinity'" if fromdate is None else ':fromdate'
    todate = "'infinity'" if todate is None else ':todate'
    return text(
        '({}, {}) overlaps (start, "end" + interval \'1 microsecond\')'.format(
            fromdate, todate
        )
    )


def subset(ts, fromdate, todate):
    if fromdate is None and todate is None:
        return ts
    if isinstance(fromdate, tuple):
        fromdate = fromdate[0]
    if isinstance(todate, tuple):
        todate = todate[0]
    return ts.loc[fromdate:todate]


def inject_in_index(serie, revdate):
    mindex = [(revdate, valuestamp) for valuestamp in serie.index]
    serie.index = pd.MultiIndex.from_tuples(mindex, names=[
        'insertion_date', 'value_date']
    )


def num2float(pdobj):
    # get a Series or a Dataframe column
    if str(pdobj.dtype).startswith('int'):
        return pdobj.astype('float64')
    return pdobj


def tojson(ts, precision=1e-14):
    return ts.to_json(date_format='iso',
                      double_precision=-int(math.log10(precision)))


def fromjson(jsonb, tsname, tzaware=False):
    series = _fromjson(jsonb, tsname).fillna(value=np.nan)
    if tzaware:
        series.index = series.index.tz_localize('utc')
    return series


def _fromjson(jsonb, tsname):
    if jsonb == '{}':
        return pd.Series(name=tsname)

    result = pd.read_json(jsonb, typ='series', dtype=False)
    result.name = tsname
    result = num2float(result)
    return result


class SeriesServices(object):
    _precision = 1e-14

    # diff handling

    def patch(self, base, diff):
        assert base is not None
        assert diff is not None
        newindex = base.index.union(diff.index).sort_values()
        patched = pd.Series([0] * len(newindex), index=newindex)
        patched[base.index] = base
        patched[diff.index] = diff
        patched.name = base.name
        return patched

    def diff(self, base, other):
        if base is None:
            return other
        base = base[~base.isnull()]
        if not len(base):
            return other

        mask_overlap = other.index.isin(base.index)
        base_overlap = base[other.index[mask_overlap]]
        other_overlap = other[mask_overlap]

        if base.dtype == 'float64':
            mask_equal = np.isclose(base_overlap, other_overlap,
                                    rtol=0, atol=self._precision)
        else:
            mask_equal = base_overlap == other_overlap

        mask_na_equal = base_overlap.isnull() & other_overlap.isnull()
        mask_equal = mask_equal | mask_na_equal

        diff_overlap = other[mask_overlap][~mask_equal]
        diff_new = other[~mask_overlap]
        diff_new = diff_new[~diff_new.isnull()]
        return pd.concat([diff_overlap, diff_new])


    # serialization

    def _make_tablename(self, seriename):
        """ compute the unqualified (no namespace) table name
        from a serie name, to allow arbitrary serie names
        """
        # postgresql table names are limited to 63 chars.
        if len(seriename) > 63:
            return hashlib.sha1(seriename.encode('utf-8')).hexdigest()
        return seriename


def rename_series(engine, serie_map, namespace='tsh'):
    from tshistory.schema import tsschema
    schema = tsschema(namespace)
    schema.define()

    reg = schema.registry
    with engine.begin() as cn:
        for old, new in serie_map.items():
            print('rename', old, '->', new)
            sql = reg.update().where(
                reg.c.seriename == old
            ).values(
                seriename=new
            )
            cn.execute(sql)


def delete_series(engine, series, namespace='tsh'):
    from tshistory.tsio import TimeSerie
    tsh = TimeSerie(namespace=namespace)

    for name in series:
        with engine.begin() as cn:
            if not tsh.exists(cn, name):
                print('skipping unknown', name)
                continue
            print('delete', name)
            tsh.delete(cn, name)


def threadpool(maxthreads):
    L = logging.getLogger('parallel')

    def run(func, argslist):
        count = 0
        threads = []
        L.info('// run %s %s', func.__name__, len(argslist))

        # initial threads
        for count, args in enumerate(argslist, start=1):
            th = threading.Thread(target=func, args=args)
            threads.append(th)
            L.info('// start thread %s', th.name)
            th.daemon = True
            th.start()
            if count == maxthreads:
                break

        while threads:
            for th in threads[:]:
                th.join(1. / maxthreads)
                if not th.is_alive():
                    threads.remove(th)
                    L.info('// thread %s exited, %s remaining', th.name, len(threads))
                    if count < len(argslist):
                        newth = threading.Thread(target=func, args=argslist[count])
                        threads.append(newth)
                        L.info('// thread %s started', newth.name)
                        newth.daemon = True
                        newth.start()
                        count += 1

    return run
