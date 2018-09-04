from datetime import datetime
from contextlib import contextmanager

import pandas as pd

from tshistory.util import inject_in_index


def utcdt(*dt):
    return pd.Timestamp(datetime(*dt), tz='UTC')


def remove_metadata(tsrepr):
    if 'Freq' in tsrepr or 'Name' in tsrepr:
        return tsrepr[:tsrepr.rindex('\n')]
    return tsrepr


def assert_df(expected, df):
    exp = remove_metadata(expected.strip())
    got = remove_metadata(df.to_string().strip())
    assert exp == got


def assert_hist(expected, dfdict):
    # copy to avoid side effects
    series = [(key, serie.copy()) for key, serie in dfdict.items()]
    for revdate, serie in series:
        inject_in_index(serie, revdate)

    series = pd.concat([serie for _, serie in series])
    assert_df(expected, series)


def assert_hist_equals(h1, h2):
    assert h1.keys() == h2.keys()
    for k in h1:
        assert (h1[k] == h2[k]).all()


def assert_group_equals(g1, g2):
    for (n1, s1), (n2, s2) in zip(sorted(g1.items()),
                                  sorted(g2.items())):
        assert n1 == n2
        assert s1.equals(s2)


def assert_structures(engine, tsh):
    seriecount = engine.execute(
        'select count(*) from "{}".registry'.format(tsh.namespace)
    ).scalar()
    csetcount = engine.execute(
        'select count(*) from "{}".changeset'.format(tsh.namespace)
    ).scalar()
    csetseriecount = engine.execute(
        'select count(*) from "{}".changeset_series'.format(tsh.namespace)
    ).scalar()
    assert csetcount == csetseriecount
    return seriecount, csetcount, csetseriecount


def genserie(start, freq, repeat, initval=None, tz=None, name=None):
    if initval is None:
        values = range(repeat)
    else:
        values = initval * repeat

    if isinstance(freq, (list, tuple)):
        idx = []
        for i in range(len(freq)):
            idx.append(pd.date_range(start=start,
                                     freq=freq[i],
                                     periods=repeat,
                                     tz=tz))
        return pd.Series(values, name=name, index=idx)

    else:
        return pd.Series(values,
                         name=name,
                         index=pd.date_range(start=start,
                                             freq=freq,
                                             periods=repeat,
                                             tz=tz))

@contextmanager
def tempattr(obj, attr, value):
    oldvalue = getattr(obj, attr)
    setattr(obj, attr, value)
    yield
    setattr(obj, attr, oldvalue)
