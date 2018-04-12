import math
import zlib

import numpy as np
import pandas as pd
from pandas.api.types import is_datetimetz


def mindate(ts):
    if isinstance(ts.index, pd.MultiIndex):
        return ts.index.min()[0]
    return ts.index.min()


def maxdate(ts):
    if isinstance(ts.index, pd.MultiIndex):
        return ts.index.max()[0]
    return ts.index.max()


def tzaware_serie(ts):
    if isinstance(ts.index, pd.MultiIndex):
        tzaware = [is_datetimetz(ts.index.get_level_values(idx_name))
                   for idx_name in ts.index.names]
        assert all(tzaware) or not any(tzaware), (
            'all your indexes must be '
            'either tzaware or none of them'
        )
        return all(tzaware)
    return is_datetimetz(ts.index)


def subset(ts, fromdate, todate):
    if fromdate is None and todate is None:
        return ts
    if isinstance(fromdate, tuple):
        fromdate = fromdate[0]
    if isinstance(todate, tuple):
        todate = todate[0]
    if isinstance(ts.index, pd.MultiIndex):
        if not ts.index.lexsort_depth:
            ts.sort_index(inplace=True)
    return ts.loc[fromdate:todate]


def inject_in_index(serie, revdate):
    if isinstance(serie.index, pd.MultiIndex):
        mindex = [(revdate, *rest) for rest in serie.index]
        serie.index = pd.MultiIndex.from_tuples(mindex, names=[
            'insertion_date', *serie.index.names]
        )
        return
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
    if not isinstance(ts.index, pd.MultiIndex):
        return ts.to_json(date_format='iso',
                          double_precision=-int(math.log10(precision)))

    # multi index case
    return ts.to_frame().reset_index().to_json(date_format='iso')


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
        patched = pd.Series([0.0], index=base.index.union(diff.index))
        patched[base.index] = base
        patched[diff.index] = diff
        patched.sort_index(inplace=True)
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

    def _serialize(self, ts):
        if ts is None:
            return None
        return zlib.compress(tojson(ts, self._precision).encode('utf-8'))

    def _deserialize(self, ts, name):
        return fromjson(zlib.decompress(ts).decode('utf-8'), name)
