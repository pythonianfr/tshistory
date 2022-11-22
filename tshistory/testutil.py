import io
from datetime import datetime
from contextlib import contextmanager
from functools import partial

import pandas as pd
import responses
import pytest
import webtest

from tshistory.http import app
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


def hist_to_df(dfdict):
    # copy to avoid side effects
    series = [(key, serie.copy()) for key, serie in dfdict.items()]
    for revdate, serie in series:
        inject_in_index(serie, revdate)

    return pd.concat([serie for _, serie in series])


def assert_hist(expected, dfdict):
    series = hist_to_df(dfdict)
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


def gen_value_ranges(start, end, lag):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    lag = pd.Timedelta(lag)
    return [
        (start, end),
        (start - lag, end + lag),
        (start + lag, end - lag),
        (None, None),
        (start, None),
        (None, end),
    ]


def _dt_indexed_df_from_csv(csv_path, index_label='datetime'):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    tz_info = pd.to_datetime(df[index_label].iloc[0]).tzinfo
    df.index = pd.to_datetime(df[index_label], utc=bool(tz_info))
    df.index.name = index_label
    return df.drop(columns=index_label)


def ts_from_csv(csv_path, index_label='datetime'):
    return _dt_indexed_df_from_csv(csv_path, index_label).iloc[:, 0]


def hist_from_csv(csv_path, index_label='datetime'):
    df = _dt_indexed_df_from_csv(csv_path, index_label)
    hist = {}
    for i_date, ts in df.items():
        i_date = pd.Timestamp(i_date)
        if not i_date.tzinfo:
            i_date = i_date.tz_localize('utc')
        hist[i_date] = ts.dropna()
    return hist


def gengroup(n_scenarios, from_date, length, freq, seed=0):
    """ Return a dataframe with ncol = n_scenarios, nrow = length, the
    seed is the value in the upper-left corner wich increases by one by
    going to the right and the bottom

    """
    index = pd.date_range(
        start=from_date,
        freq=freq,
        periods=length
    )
    mat = []
    for irow in range(length):
        mat.append(
            [
                nb + seed + irow
                for nb in range(n_scenarios)
            ]
        )
    return pd.DataFrame(mat, index=index)


@contextmanager
def tempattr(obj, attr, value):
    oldvalue = getattr(obj, attr)
    setattr(obj, attr, value)
    yield
    setattr(obj, attr, oldvalue)


# api stuff

def read_request_bridge(client, request):
    resp = client.get(request.url,
                      params=request.body,
                      headers=request.headers)
    return (resp.status_code, resp.headers, resp.body)


def write_request_bridge(method):
    def bridge(request):
        resp = method(request.url,
                      params=request.body,
                      headers=request.headers)
        return (resp.status_code, resp.headers, resp.body)
    return bridge


def with_tester(uri, resp, wsgitester):
    resp.add_callback(
        responses.GET, uri + '/series/state',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.PUT, uri + '/series/state',
        callback=write_request_bridge(wsgitester.put)
    )

    resp.add_callback(
        responses.GET, uri + '/series/supervision',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.DELETE, uri + '/series/state',
        callback=write_request_bridge(wsgitester.delete)
    )

    resp.add_callback(
        responses.PUT, uri + '/series/strip',
        callback=write_request_bridge(wsgitester.put)
    )

    resp.add_callback(
        responses.GET, uri + '/series/insertion_dates',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.GET, uri + '/series/staircase',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.GET, uri + '/series/block_staircase',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.GET, uri + '/series/history',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.GET, uri + '/series/catalog',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.PATCH, uri + '/series/state',
        callback=write_request_bridge(wsgitester.patch)
    )

    resp.add_callback(
        responses.GET, uri + '/series/metadata',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.PUT, uri + '/series/metadata',
        callback=write_request_bridge(wsgitester.put)
    )

    resp.add_callback(
        responses.GET, uri + '/series/log',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.GET, uri + '/series/formula',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.PATCH, uri + '/series/formula',
        callback=write_request_bridge(wsgitester.patch)
    )

    resp.add_callback(
        responses.POST, uri + '/series/eval_formula',
        callback=write_request_bridge(wsgitester.post)
    )

    resp.add_callback(
        responses.GET, uri + '/series/xl',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.GET, uri + '/series/formula_components',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.GET, uri + '/group/state',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.PATCH, uri + '/group/state',
        callback=write_request_bridge(wsgitester.patch)
    )

    resp.add_callback(
        responses.PUT, uri + '/group/state',
        callback=write_request_bridge(wsgitester.put)
    )

    resp.add_callback(
        responses.DELETE, uri + '/group/state',
        callback=write_request_bridge(wsgitester.delete)
    )

    resp.add_callback(
        responses.GET, uri + '/group/insertion_dates',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.GET, uri + '/group/history',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.GET, uri + '/group/metadata',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.PUT, uri + '/group/metadata',
        callback=write_request_bridge(wsgitester.put)
    )

    resp.add_callback(
        responses.GET, uri + '/group/catalog',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.GET, uri + '/group/formula',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.PUT, uri + '/group/formula',
        callback=write_request_bridge(wsgitester.put)
    )

    resp.add_callback(
        responses.GET, uri + '/group/boundformula',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.PUT, uri + '/group/boundformula',
        callback=write_request_bridge(wsgitester.put)
    )

    resp.add_callback(
        responses.PUT, uri + '/cache/policy',
        callback=write_request_bridge(wsgitester.put)
    )

    resp.add_callback(
        responses.PATCH, uri + '/cache/policy',
        callback=write_request_bridge(wsgitester.patch)
    )

    resp.add_callback(
        responses.DELETE, uri + '/cache/policy',
        callback=write_request_bridge(wsgitester.delete)
    )

    resp.add_callback(
        responses.PUT, uri + '/cache/mapping',
        callback=write_request_bridge(wsgitester.put)
    )

    resp.add_callback(
        responses.DELETE, uri + '/cache/mapping',
        callback=write_request_bridge(wsgitester.delete)
    )

    resp.add_callback(
        responses.GET, uri + '/cache/cacheable',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.GET, uri + '/cache/policies',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.GET, uri + '/cache/policy-series',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.GET, uri + '/cache/series-policy',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.GET, uri + '/cache/series-has-cache',
        callback=partial(read_request_bridge, wsgitester)
    )

    resp.add_callback(
        responses.DELETE, uri + '/cache/series-has-cache',
        callback=write_request_bridge(wsgitester.delete)
    )

    resp.add_callback(
        responses.PUT, uri + '/cache/refresh-policy-now',
        callback=write_request_bridge(wsgitester.put)
    )


class WebTester(webtest.TestApp):

    def _check_status(self, status, res):
        try:
            super(WebTester, self)._check_status(status, res)
        except:
            print(res.errors)
            # raise <- default behaviour on 4xx is silly

    def _gen_request(self, method, url, params,
                     headers=None,
                     extra_environ=None,
                     status=None,
                     upload_files=None,
                     expect_errors=False,
                     content_type=None):
        """
        Do a generic request.
        PATCH: *bypass* all transformation as params comes
               straight from a prepared (python-requests) request.
        """
        environ = self._make_environ(extra_environ)

        environ['REQUEST_METHOD'] = str(method)
        url = str(url)
        url = self._remove_fragment(url)
        req = self.RequestClass.blank(url, environ)

        if isinstance(params, str):
            params = params.encode('utf-8')
        req.environ['wsgi.input'] = io.BytesIO(params)
        req.content_length = len(params) if params else 0
        if headers:
            req.headers.update(headers)
        return self.do_request(req, status=status,
                               expect_errors=expect_errors)


def make_tsx(uri,
             initschemafunc,
             tsioclass,
             httpclass,
             clientclass=None,
             passthru=None):
    from tshistory import api as tsh_api

    @pytest.fixture(params=['pg', 'http'])
    def tsx(request, engine):
        initschemafunc(engine)

        tsa = tsh_api.timeseries(
            str(engine.url),
            handler=tsioclass
        )

        if request.param == 'pg':
            # direct mode
            yield tsa

        else:
            wsgitester = WebTester(
                app.make_app(
                    tsa,
                    httpclass
                )
            )
            with responses.RequestsMock(assert_all_requests_are_fired=False) as resp:
                with_tester(uri, resp, wsgitester)
                if passthru:
                    passthru(resp)
                # will query the app created above (which in turn uses
                # the direct mode tsa)
                http_tsa = tsh_api.timeseries(
                    uri,
                    handler=tsioclass,
                    clientclass=clientclass
                )

                yield http_tsa

    return tsx
