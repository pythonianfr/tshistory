import pandas as pd


def remove_metadata(tsrepr):
    if 'Freq' in tsrepr or 'Name' in tsrepr:
        return tsrepr[:tsrepr.rindex('\n')]
    return tsrepr


def assert_df(expected, df):
    exp = remove_metadata(expected.strip())
    got = remove_metadata(df.to_string().strip())
    assert exp == got


def assert_group_equals(g1, g2):
    for (n1, s1), (n2, s2) in zip(sorted(g1.items()),
                                  sorted(g2.items())):
        assert n1 == n2
        assert s1.equals(s2)


def genserie(start, freq, repeat, initval=None, tz=None, name=None):
    if initval is None:
        values = range(repeat)
    else:
        values = initval * repeat
    return pd.Series(values,
                     name=name,
                     index=pd.date_range(start=start,
                                         freq=freq,
                                         periods=repeat,
                                         tz=tz))
