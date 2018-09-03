import math
import zlib
import hashlib

import numpy as np
import pandas as pd
from pandas.api.types import is_datetimetz
from sqlalchemy.sql.expression import text


def tzaware_serie(ts):
    return is_datetimetz(ts.index)


def start_end(ts):
    start = ts.index.min()
    end = ts.index.max()
    if start.tzinfo is not None:
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


def fromjson(jsonb, tsname):
    return _fromjson(jsonb, tsname).fillna(value=np.nan)


def _fromjson(jsonb, tsname):
    if jsonb == '{}':
        return pd.Series(name=tsname)

    result = pd.read_json(jsonb, typ='series', dtype=False)
    result.name = tsname
    if isinstance(result.index, pd.DatetimeIndex):
        result = num2float(result)
        return result

    # multi index case
    columns = result.index.values.tolist()
    columns.remove(tsname)
    result = pd.read_json(jsonb, typ='frame',
                          convert_dates=columns)
    result.set_index(sorted(columns), inplace=True)
    return num2float(result.iloc[:, 0])  # get a Series object


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

    reg = schema.registry
    with engine.connect() as cn:
        for old, new in serie_map.items():
            print('{} -> {}'.format(old, new))
            sql = reg.update().where(
                reg.c.seriename == old
            ).values(
                seriename=new
            )
            cn.execute(sql)

