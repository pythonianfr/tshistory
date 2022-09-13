import io
import json
import zlib
from datetime import datetime as dt, timedelta

import pandas as pd
import webtest

from tshistory import util, tsio
from tshistory.testutil import (
    assert_df,
    assert_hist,
    utcdt,
    gengroup,
    genserie,
    hist_from_csv,
    ts_from_csv,
)


DBURI = 'postgresql://localhost:5433/postgres'



def test_error(http):
    series_in = genserie(pd.Timestamp('2018-1-1'), 'H', 3)
    res = http.patch('/series/state', params={
        'name': 'test-error',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series_in)
    })

    v2 = pd.Series(
        ['a', 'b', 'c'],
        pd.date_range(pd.Timestamp('2020-1-1'), freq='D', periods=3)
    )
    res = http.patch('/series/state', params={
        'name': 'test-error',
        'series': util.tojson(v2),
        'author': 'Babar',
        'tzaware': util.tzaware_serie(v2)
    })
    assert res.status_code == 418
    assert res.body == (
        b'Type error when inserting test-error, new type is object, '
        b'type in base is float64'
    )
    res = http.delete('/series/state', params={
        'name': 'test-error'
    })

    # no input validation error (nr) on the delta parameter
    res = http.get('/series/staircase', params={
        'name': 'test-error',
        'delta': str(timedelta(days=2))
    })
    assert res.json == {'message': '`test-error` does not exists'}


def test_no_series(http):
    res = http.get('/series/state?name=no-such-series')
    assert res.status_code == 404
    assert res.json == {
        'message': '`no-such-series` does not exists'
    }

    res = http.get('/series/metadata?name=no-such-series')
    assert res.status_code == 404
    assert res.json == {
        'message': '`no-such-series` does not exists'
    }

    res = http.get('/series/history?name=no-such-series')
    assert res.status_code == 404
    assert res.json == {
        'message': '`no-such-series` does not exists'
    }

    res = http.get('/series/insertion_dates?name=no-such-series')
    assert res.status_code == 404
    assert res.json == {
        'message': '`no-such-series` does not exists'
    }

    res = http.get('/series/staircase', params={
        'name': 'no-such-series',
        'delta': pd.Timedelta(hours=3)
    })

    assert res.status_code == 404
    assert res.json == {
        'message': '`no-such-series` does not exists'
    }


def test_naive(http):
    series_in = genserie(pd.Timestamp('2018-1-1'), 'H', 3)
    res = http.patch('/series/state', params={
        'name': 'test-naive',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series_in)
    })

    assert res.status_code == 201
    assert res.body == (
        b'{"2018-01-01T00:00:00.000Z":0.0,'
        b'"2018-01-01T01:00:00.000Z":1.0,'
        b'"2018-01-01T02:00:00.000Z":2.0}'
    )

    res = http.get('/series/metadata?name=test-naive&all=1')
    meta = res.json
    meta.pop('supervision_status', None)
    assert meta == {
        'index_dtype': '<M8[ns]',
        'index_type': 'datetime64[ns]',
        'tzaware': False,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }

    res = http.get('/series/state?name=test-naive')
    series = util.fromjson(res.body, 'test', meta['tzaware'])
    assert_df("""
2018-01-01 00:00:00    0.0
2018-01-01 01:00:00    1.0
2018-01-01 02:00:00    2.0
""", series)


def test_base(http):
    # insert
    series_in = genserie(utcdt(2018, 1, 1), 'H', 3)
    res = http.patch('/series/state', params={
        'name': 'test',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series_in)
    })

    assert res.status_code == 201
    assert res.body == (
        b'{"2018-01-01T00:00:00.000Z":0.0,'
        b'"2018-01-01T01:00:00.000Z":1.0,'
        b'"2018-01-01T02:00:00.000Z":2.0}'
    )

    res = http.patch('/series/state', params={
        'name': 'test',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series_in)
    })
    assert res.status_code == 200
    assert res.body == b'{}'

    # catalog
    res = http.get('/series/catalog')
    assert res.status_code == 200
    assert res.json == {
        'postgres': [
            ['test-naive', 'primary'],
            ['test', 'primary']
        ]
    }

    # metadata
    res = http.get('/series/metadata?name=test')
    meta = res.json
    assert meta == {}

    res = http.get('/series/metadata?name=test&all=1')
    meta = res.json
    meta.pop('supervision_status', None)
    assert meta == {
        'index_dtype': '|M8[ns]',
        'index_type': 'datetime64[ns, UTC]',
        'tzaware': True,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }

    res = http.put('/series/metadata', params={
        'metadata': json.dumps({
            'freq': 'D',
            'description': 'banana spot price'
        }),
        'name': 'test'
    })
    assert res.status_code == 200
    res = http.get('/series/metadata?name=test')
    meta2 = res.json
    assert meta2 == {
        'freq': 'D',
        'description': 'banana spot price'
    }

    # metadata: delete by uploading an empty dict
    res = http.put('/series/metadata', params={
        'metadata': json.dumps({}),
        'name': 'test'
    })
    assert res.status_code == 200
    res = http.get('/series/metadata?name=test')
    meta2 = res.json
    assert meta2 == {}

    # get
    res = http.get('/series/state?name=test')
    series = util.fromjson(res.body, 'test', meta['tzaware'])
    assert_df("""
2018-01-01 00:00:00+00:00    0.0
2018-01-01 01:00:00+00:00    1.0
2018-01-01 02:00:00+00:00    2.0
""", series)


    # reinsert
    series_in = genserie(utcdt(2018, 1, 1, 3), 'H', 1, [3])
    res = http.patch('/series/state', params={
        'name': 'test',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 13),
        'tzaware': util.tzaware_serie(series_in)
    })

    assert res.status_code == 200

    res = http.get('/series/state?name=test')
    series = util.fromjson(res.body, 'test', meta['tzaware'])
    assert_df("""
2018-01-01 00:00:00+00:00    0.0
2018-01-01 01:00:00+00:00    1.0
2018-01-01 02:00:00+00:00    2.0
2018-01-01 03:00:00+00:00    3.0
""", series)

    # checkout a past state
    res = http.get('/series/state', params={
        'name': 'test',
        'insertion_date': utcdt(2018, 1, 1, 10)
    })
    series = util.fromjson(res.body, 'test', meta['tzaware'])
    assert_df("""
2018-01-01 00:00:00+00:00    0.0
2018-01-01 01:00:00+00:00    1.0
2018-01-01 02:00:00+00:00    2.0
""", series)

    # checkout too far in the past
    res = http.get('/series/state', params={
        'name': 'test',
        'insertion_date': utcdt(2018, 1, 1, 0)
    })
    assert res.json == {}

    # history
    res = http.get('/series/history?name=test')
    df = pd.read_json(io.BytesIO(res.body))

    # we real client would need to handle timestamp
    # tz-awareness
    assert_df("""
2018-01-01 10:00:00  2018-01-01 13:00:00
2018-01-01 00:00:00                  0.0                    0
2018-01-01 01:00:00                  1.0                    1
2018-01-01 02:00:00                  2.0                    2
2018-01-01 03:00:00                  NaN                    3
""", df)

    res = http.get('/series/history?name=test&format=tshpack')
    meta, hist = util.unpack_history(res.body)
    assert_hist("""
insertion_date             value_date               
2018-01-01 10:00:00+00:00  2018-01-01 00:00:00+00:00    0.0
                           2018-01-01 01:00:00+00:00    1.0
                           2018-01-01 02:00:00+00:00    2.0
2018-01-01 13:00:00+00:00  2018-01-01 00:00:00+00:00    0.0
                           2018-01-01 01:00:00+00:00    1.0
                           2018-01-01 02:00:00+00:00    2.0
                           2018-01-01 03:00:00+00:00    3.0
""", hist)

    # diff mode
    res = http.get('/series/history', params={
        'name': 'test',
        'diffmode': True
    })
    df = pd.read_json(io.BytesIO(res.body))

    assert_df("""
2018-01-01 10:00:00  2018-01-01 13:00:00
2018-01-01 00:00:00                  0.0                  NaN
2018-01-01 01:00:00                  1.0                  NaN
2018-01-01 02:00:00                  2.0                  NaN
2018-01-01 03:00:00                  NaN                  3.0
""", df)

    # empty range
    res = http.get('/series/history', params={
        'name': 'test',
        'from_insertion_date': utcdt(2018, 1, 1, 11),
        'to_insertion_date': utcdt(2018, 1, 1, 12),
    })
    df = pd.read_json(io.BytesIO(res.body))
    assert len(df) == 0

    # insertion dates subset
    res = http.get('/series/history', params={
        'name': 'test',
        'from_insertion_date': utcdt(2018, 1, 1, 10),
        'to_insertion_date': utcdt(2018, 1, 1, 12),
    })
    df = pd.read_json(io.BytesIO(res.body))

    assert_df("""
                     2018-01-01 10:00:00
2018-01-01 00:00:00                    0
2018-01-01 01:00:00                    1
2018-01-01 02:00:00                    2
""", df)

    # value dates subset
    res = http.get('/series/history', params={
        'name': 'test',
        'from_value_date': utcdt(2018, 1, 1, 2),
        'to_value_date': utcdt(2018, 1, 1, 3),
    })
    df = pd.read_json(io.BytesIO(res.body))

    assert_df("""
                     2018-01-01 10:00:00  2018-01-01 13:00:00
2018-01-01 02:00:00                  2.0                    2
2018-01-01 03:00:00                  NaN                    3
""", df)

    # state/get from from/to value date restriction
    res = http.get('/series/state', params={
        'name': 'test',
        'from_value_date': utcdt(2018, 1, 1, 1),
        'to_value_date': utcdt(2018, 1, 1, 2)
    })
    assert res.json == {
        '2018-01-01T01:00:00.000Z': 1.0,
        '2018-01-01T02:00:00.000Z': 2.0
    }

    series_in = genserie(utcdt(2019, 1, 1), 'H', 3)
    res = http.patch('/series/state', params={
        'name': 'test',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 2),
        'tzaware': util.tzaware_serie(series_in),
        'replace': True
    })

    assert res.status_code == 200
    res = http.get('/series/state', params={
        'name': 'test'
    })
    assert res.json == {
        '2019-01-01T00:00:00.000Z': 0.0,
        '2019-01-01T01:00:00.000Z': 1.0,
        '2019-01-01T02:00:00.000Z': 2.0
    }

    res = http.get('/series/metadata', params={
        'name': 'test',
        'type': 'interval'
    })
    assert res.json == [
        True,
        '2019-01-01T00:00:00+00:00',
        '2019-01-01T02:00:00+00:00'
    ]

    res = http.get('/series/metadata', params={
        'name': 'test',
        'type': 'type'
    })
    assert res.json == 'primary'

    res = http.get('/series/insertion_dates', params={
        'name': 'test'
    })
    idates = [
        pd.Timestamp(t, tz='UTC')
        for t in res.json['insertion_dates']
    ]
    assert idates == [
        pd.Timestamp('2018-01-01 10:00:00', tz='UTC'),
        pd.Timestamp('2018-01-01 13:00:00', tz='UTC'),
        pd.Timestamp('2018-01-02 00:00:00', tz='UTC')
    ]


def test_delete(http):
    series_in = genserie(utcdt(2018, 1, 1), 'H', 3)
    res = http.patch('/series/state', params={
        'name': 'test',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series_in)
    })

    res = http.delete('/series/state', params={
        'name': 'no-such-series'
    })
    assert res.status_code == 404
    res = http.delete('/series/state', params={
        'name': 'test'
    })
    assert res.status_code == 204
    res = http.get('/series/catalog')
    assert 'test' not in res.json


def test_rename(http):
    series_in = genserie(utcdt(2018, 1, 1), 'H', 3)
    res = http.patch('/series/state', params={
        'name': 'test',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series_in)
    })
    res = http.put('/series/state', params={
        'name': 'no-such-series',
        'newname': 'no-better'
    })
    assert res.status_code == 404
    res = http.put('/series/state', params={
        'name': 'test',
        'newname': 'test2'
    })
    assert res.status_code == 204
    res = http.get('/series/catalog')
    assert res.json == {
        'postgres': [
            ['test-naive', 'primary'],
            ['test2', 'primary']
        ]
    }

    assert 'test' not in res.json

    res = http.patch('/series/state', params={
        'name': 'test3',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series_in)
    })
    res = http.put('/series/state', params={
        'name': 'test2',
        'newname': 'test3'
    })
    assert res.status_code == 409
    assert res.json == {
        'message': '`test3` does exists'
    }



def test_strip(http):
    series_in = genserie(utcdt(2021, 1, 1), 'H', 3)
    res = http.patch('/series/state', params={
        'name': 'stripme',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2021, 1, 1),
        'tzaware': util.tzaware_serie(series_in)
    })
    series_in = genserie(utcdt(2021, 1, 2), 'H', 3)
    res = http.patch('/series/state', params={
        'name': 'stripme',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2021, 1, 2),
        'tzaware': util.tzaware_serie(series_in)
    })

    res = http.put('/series/strip', params={
        'name': 'stripme',
        'insertion_date': utcdt(2021, 1, 2)
    })
    assert res.status_code == 204

    res = http.get('/series/insertion_dates', params={
        'name': 'stripme'
    })
    idates = json.loads(res.text)['insertion_dates']
    assert len(idates) == 1


def test_staircase(http):
    # each days we insert 7 data points
    for idate in pd.date_range(start=utcdt(2015, 1, 1),
                               end=utcdt(2015, 1, 4),
                               freq='D'):
        series = genserie(start=idate, freq='H', repeat=7)
        http.patch('/series/state', params={
            'name': 'staircase',
            'series': util.tojson(series),
            'author': 'Babar',
            'insertion_date': idate,
            'tzaware': util.tzaware_serie(series)
        })

    res = http.get('/series/staircase', params={
        'name': 'staircase',
        'delta': pd.Timedelta(hours=3),
        'from_value_date': utcdt(2015, 1, 1, 4),
        'to_value_date': utcdt(2015, 1, 2, 5),
    })
    series = util.fromjson(res.body, 'test', True)

    assert_df("""
2015-01-01 04:00:00+00:00    4.0
2015-01-01 05:00:00+00:00    5.0
2015-01-01 06:00:00+00:00    6.0
2015-01-02 03:00:00+00:00    3.0
2015-01-02 04:00:00+00:00    4.0
2015-01-02 05:00:00+00:00    5.0
""", series)

    res = http.get('/series/staircase', params={
        'name': 'staircase',
        'delta': pd.Timedelta(hours=3),
        'from_value_date': utcdt(2015, 1, 1, 4),
        'to_value_date': utcdt(2015, 1, 2, 5),
        'format': 'tshpack'
    })
    meta, index, values = util.nary_unpack(zlib.decompress(res.body))
    meta = json.loads(meta)
    index, values = util.numpy_deserialize(index, values, meta)
    series = pd.Series(values, index=index)
    series = series.tz_localize('UTC')

    assert_df("""
2015-01-01 04:00:00+00:00    4.0
2015-01-01 05:00:00+00:00    5.0
2015-01-01 06:00:00+00:00    6.0
2015-01-02 03:00:00+00:00    3.0
2015-01-02 04:00:00+00:00    4.0
2015-01-02 05:00:00+00:00    5.0
""", series)


def test_block_staircase(http):
    hist = hist_from_csv(io.StringIO("""
datetime,               2020-01-01 08:00+0, 2020-01-02 08:00+0, 2020-01-03 08:00+0
2020-01-03 00:00+01:00, 1.0,                10.0,               100.0
2020-01-03 04:00+01:00, 2.0,                20.0,               200.0
2020-01-03 08:00+01:00, 3.0,                30.0,               300.0
2020-01-03 16:00+01:00, 4.0,                40.0,               400.0
2020-01-04 00:00+01:00, 5.0,                50.0,               500.0
2020-01-04 04:00+01:00, 6.0,                60.0,               600.0
2020-01-04 08:00+01:00, 7.0,                70.0,               700.0
2020-01-04 16:00+01:00, 8.0,                80.0,               800.0
"""))
    for idate, ts in hist.items():
        http.patch('/series/state', params={
            'name': 'test_b_staircase',
            'series': util.tojson(ts),
            'author': 'test_http',
            'insertion_date': idate,
            'tzaware': util.tzaware_serie(ts)
        })
    sc_kwargs = dict(
        from_value_date=pd.Timestamp('2020-01-03', tz='CET').to_pydatetime(),
        to_value_date=pd.Timestamp('2020-01-05', tz='CET').to_pydatetime(),
        revision_freq=json.dumps({'days': 1}),
        revision_time=json.dumps({'hour': 10}),
        revision_tz='CET',
        maturity_offset=json.dumps({'hours': 24}),
        maturity_time=json.dumps({'hour': 4}),
    )
    expected_ts = ts_from_csv(io.StringIO("""
datetime,               value
2020-01-03 00:00+01:00, 1.0
2020-01-03 04:00+01:00, 20.0
2020-01-03 08:00+01:00, 30.0
2020-01-03 16:00+01:00, 40.0
2020-01-04 00:00+01:00, 50.0
2020-01-04 04:00+01:00, 600.0
2020-01-04 08:00+01:00, 700.0
2020-01-04 16:00+01:00, 800.0
"""))

    # test query with 'json' format
    res = http.get('/series/block_staircase', params=dict(
        name='test_b_staircase', **sc_kwargs
    ))
    computed_ts = util.fromjson(res.body, 'test_b_staircase', True)
    pd.testing.assert_series_equal(computed_ts, expected_ts, check_names=False)

    # test query with 'tsh_pack' format
    res = http.get('/series/block_staircase', params=dict(
        name='test_b_staircase', **sc_kwargs, format='tshpack'
    ))
    computed_ts = util.unpack_series('test_b_staircase', res.body)
    pd.testing.assert_series_equal(computed_ts, expected_ts, check_names=False)


def test_get_fast_path(http):
    series_in = genserie(utcdt(2018, 1, 1), 'H', 3)
    res = http.patch('/series/state', params={
        'name': 'test_fast',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series_in)
    })

    assert res.status_code == 201

    out = http.get('/series/state', params={
        'name': 'test_fast',
        'format': 'tshpack'
    })
    meta, index, values = util.nary_unpack(zlib.decompress(out.body))
    meta = json.loads(meta)
    index, values = util.numpy_deserialize(index, values, meta)
    series = pd.Series(values, index=index)
    series = series.tz_localize('UTC')

    assert_df("""
2018-01-01 00:00:00+00:00    0.0
2018-01-01 01:00:00+00:00    1.0
2018-01-01 02:00:00+00:00    2.0
""", series)

    meta.pop('supervision_status', None)
    assert meta == {
        'tzaware': True,
        'index_type': 'datetime64[ns, UTC]',
        'value_type': 'float64',
        'index_dtype': '|M8[ns]',
        'value_dtype': '<f8'
    }


def test_multisource(http, engine):
    series = genserie(utcdt(2020, 1, 1), 'D', 3)
    res = http.patch('/series/state', params={
        'name': 'test-multi',
        'series': util.tojson(series),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series)
    })

    assert res.status_code == 201

    tsh = tsio.timeseries('other')
    tsh.update(
        engine,
        series,
        'test-other-source',
        'Babar'
    )

    out = http.get('/series/state', params={
        'name': 'test-multi',
    })
    assert out.json == {
        '2020-01-01T00:00:00.000Z': 0.0,
        '2020-01-02T00:00:00.000Z': 1.0,
        '2020-01-03T00:00:00.000Z': 2.0
    }

    out = http.get('/series/state', params={
        'name': 'test-other-source',
    })
    assert out.json == {
        '2020-01-01T00:00:00.000Z': 0.0,
        '2020-01-02T00:00:00.000Z': 1.0,
        '2020-01-03T00:00:00.000Z': 2.0
    }

    res = http.patch('/series/state', params={
        'name': 'test-multi',
        'series': util.tojson(series),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series)
    })
    assert res.status_code == 200

    res = http.patch('/series/state', params={
        'name': 'test-other-source',
        'series': util.tojson(series),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series)
    })
    assert res.status_code == 405
    assert res.json == {'message': 'not allowed to update to a secondary source'}

    res = http.get('/series/metadata?name=test-other-source', params={
        'all': True
    })
    meta = res.json
    assert meta == {
        'tzaware': True,
        'index_type': 'datetime64[ns, UTC]',
        'value_type': 'float64',
        'index_dtype': '|M8[ns]',
        'value_dtype': '<f8'
    }

    res = http.put('/series/metadata', params={
        'metadata': json.dumps({
            'description': 'banana spot price'
        }),
        'name': 'test-other-source'
    })
    assert res.status_code == 405
    assert res.json == {
        'message': 'not allowed to update metadata to a secondary source'
    }

    res = http.delete('/series/state', params={
        'name': 'test-other-source'
    })
    assert res.status_code == 405
    assert res.json == {
        'message': 'not allowed to delete to a secondary source'
    }

    res = http.delete('/series/state', params={
        'name': 'test-other-source'
    })
    assert res.status_code == 405
    assert res.json == {
        'message': 'not allowed to delete to a secondary source'
    }

    res = http.put('/series/state', params={
        'name': 'test-other-source',
        'newname': 'test2-other-source'
    })
    assert res.status_code == 405
    assert res.json == {
        'message': 'not allowed to rename to a secondary source'
    }

    # catalog
    res = http.get('/series/catalog')
    assert res.status_code == 200
    assert res.json == {
        'postgres@other': [
            ['test-other-source', 'primary']
        ],
        'postgres': [
            ['test-naive', 'primary'],
            ['test2', 'primary'],
            ['test3', 'primary'],
            ['stripme', 'primary'],
            ['staircase', 'primary'],
            ['test_b_staircase', 'primary'],
            ['test_fast', 'primary'],
            ['test-multi', 'primary']
        ]
    }
    res = http.get('/series/catalog', params={
        'allsources': False
    })
    assert res.status_code == 200
    assert 'postgres' in res.json
    assert 'postgres@other' not in res.json


def test_group(http):
    df = gengroup(
        n_scenarios=3,
        from_date=dt(2021, 1, 1),
        length=5,
        freq='D',
        seed=2.
    )

    df.columns = ['a', 'b', 'c']

    bgroup = util.pack_group(df)
    res = http.patch('/group/state', {
        'name': 'test_group',
        'author': 'Babar',
        'format': 'tshpack',
        'replace': json.dumps(True),
        'insertion_date': str(utcdt(2022, 3, 1)),
        # We need to send the date as an str because of a weird behavior
        # in webtest.app.TestApp._gen_request (v 2.0.35) triggered by the type
        # of the bgroup (binary)
        'bgroup': webtest.Upload('bgroup', bgroup)
    })
    assert res.status_code == 201

    res = http.get('/group/state', {'name': 'test_group'})
    df2 = util.unpack_group(res.body)
    assert df.equals(df2)

    bgroup = util.pack_group(df*2)
    res = http.patch('/group/state', {
        'name': 'test_group',
        'author': 'Babar',
        'format': 'tshpack',
        'replace': json.dumps(True),
        'insertion_date': str(utcdt(2022, 3, 2)),
        'bgroup': webtest.Upload('bgroup', bgroup)
    })
    assert res.status_code == 200

    res = http.get('/group/state', {'name': 'test_group'})
    df3 = util.unpack_group(res.body)

    res = http.get('/group/insertion_dates', params={
        'name': 'test_group'
    })
    idates = [
        pd.Timestamp(t, tz='UTC')
        for t in res.json['insertion_dates']
    ]
    assert idates == [
        pd.Timestamp('2022-03-01 00:00:00+0000', tz='UTC'),
        pd.Timestamp('2022-03-02 00:00:00+0000', tz='UTC')
    ]

    res = http.get('/group/history',params={
        'name': 'test_group'
    })
    hist = util.unpack_group_history(res.body)

    assert hist[idates[0]].equals(df)
    assert hist[idates[1]].equals(df3)

    res = http.get('/group/catalog')
    assert res.json == {
        'postgres': [['test_group', 'primary']]
    }

    res = http.get('/group/metadata', {'name': 'test_group'})
    assert res.json == {}

    res = http.put('/group/metadata', {
        'name': 'test_group',
        'metadata': json.dumps({'foo': 'bar'})
    })
    assert res.status_code == 200

    res = http.get('/group/metadata', {'name': 'test_group'})
    assert res.json == {'foo': 'bar'}

    res = http.delete('/group/state', {'name': 'test_group'})
    assert res.status_code == 204

    res = http.get('/group/catalog')
    assert res.json == {}


def test_log(http):
    series = genserie(utcdt(2020, 1, 1), 'D', 5)
    for d in range(5):
        res = http.patch('/series/state', params={
            'name': 'test-log',
            'series': util.tojson(series),
            'author': 'Babar',
            'insertion_date': utcdt(2020, 1, d + 1).isoformat(),
            'metadata': json.dumps({'comment': f'day {d+1}'}),
            'tzaware': util.tzaware_serie(series)
        })
        assert res.status_code in (201, 200)
        series[d] = 42


    out = http.get('/series/state', params={
        'name': 'test-log',
        'insertion_date': utcdt(2020, 1, 2)
        }
    )
    assert out.json == {
        '2020-01-01T00:00:00.000Z': 42.0,
        '2020-01-02T00:00:00.000Z': 1.0,
        '2020-01-03T00:00:00.000Z': 2.0,
        '2020-01-04T00:00:00.000Z': 3.0,
        '2020-01-05T00:00:00.000Z': 4.0
    }

    res = http.get('/series/log', params={
        'name': 'test-log'
    })

    assert res.json == [
        {'author': 'Babar',
         'date': '2020-01-01T00:00:00+00:00',
         'meta': {'comment': 'day 1'},
         'rev': 1},
        {'author': 'Babar',
         'date': '2020-01-02T00:00:00+00:00',
         'meta': {'comment': 'day 2'},
         'rev': 2},
        {'author': 'Babar',
         'date': '2020-01-03T00:00:00+00:00',
         'meta': {'comment': 'day 3'},
         'rev': 3},
        {'author': 'Babar',
         'date': '2020-01-04T00:00:00+00:00',
         'meta': {'comment': 'day 4'},
         'rev': 4},
        {'author': 'Babar',
         'date': '2020-01-05T00:00:00+00:00',
         'meta': {'comment': 'day 5'},
         'rev': 5}
    ]

    res = http.get('/series/log', params={
        'name': 'test-log',
        'limit': 2
    })
    assert res.json == [
        {'author': 'Babar',
         'date': '2020-01-04T00:00:00+00:00',
         'meta': {'comment': 'day 4'},
         'rev': 4},
        {'author': 'Babar',
         'date': '2020-01-05T00:00:00+00:00',
         'meta': {'comment': 'day 5'},
         'rev': 5}
    ]

    res = http.get('/series/log', params={
        'name': 'test-log',
        'fromdate': utcdt(2020, 1, 2).isoformat(),
        'todate': utcdt(2020, 1, 3).isoformat()
    })
    assert res.json == [
        {'author': 'Babar',
         'date': '2020-01-02T00:00:00+00:00',
         'meta': {'comment': 'day 2'},
         'rev': 2},
        {'author': 'Babar',
         'date': '2020-01-03T00:00:00+00:00',
         'meta': {'comment': 'day 3'},
         'rev': 3}
    ]
