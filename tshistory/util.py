import pandas as pd
from pandas.api.types import is_datetimetz


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
