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
    genserie
)


DBURI = 'postgresql://localhost:5433/postgres'



def test_error(client):
    series_in = genserie(pd.Timestamp('2018-1-1'), 'H', 3)
    res = client.patch('/series/state', params={
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
    res = client.patch('/series/state', params={
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
    res = client.delete('/series/state', params={
        'name': 'test-error'
    })

    # no input validation error (nr) on the delta parameter
    res = client.get('/series/staircase', params={
        'name': 'test-error',
        'delta': str(timedelta(days=2))
    })
    assert res.json == {'message': '`test-error` does not exists'}


def test_no_series(client):
    res = client.get('/series/state?name=no-such-series')
    assert res.status_code == 404
    assert res.json == {
        'message': '`no-such-series` does not exists'
    }

    res = client.get('/series/metadata?name=no-such-series')
    assert res.status_code == 404
    assert res.json == {
        'message': '`no-such-series` does not exists'
    }

    res = client.get('/series/history?name=no-such-series')
    assert res.status_code == 404
    assert res.json == {
        'message': '`no-such-series` does not exists'
    }

    res = client.get('/series/insertion_dates?name=no-such-series')
    assert res.status_code == 404
    assert res.json == {
        'message': '`no-such-series` does not exists'
    }

    res = client.get('/series/staircase', params={
        'name': 'no-such-series',
        'delta': pd.Timedelta(hours=3)
    })

    assert res.status_code == 404
    assert res.json == {
        'message': '`no-such-series` does not exists'
    }


def test_naive(client):
    series_in = genserie(pd.Timestamp('2018-1-1'), 'H', 3)
    res = client.patch('/series/state', params={
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

    res = client.get('/series/metadata?name=test-naive&all=1')
    meta = res.json
    meta.pop('supervision_status', None)
    assert meta == {
        'index_dtype': '<M8[ns]',
        'index_type': 'datetime64[ns]',
        'tzaware': False,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }

    res = client.get('/series/state?name=test-naive')
    series = util.fromjson(res.body, 'test', meta['tzaware'])
    assert_df("""
2018-01-01 00:00:00    0.0
2018-01-01 01:00:00    1.0
2018-01-01 02:00:00    2.0
""", series)


def test_base(client):
    # insert
    series_in = genserie(utcdt(2018, 1, 1), 'H', 3)
    res = client.patch('/series/state', params={
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

    res = client.patch('/series/state', params={
        'name': 'test',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series_in)
    })
    assert res.status_code == 200
    assert res.body == b'{}'

    # catalog
    res = client.get('/series/catalog')
    assert res.status_code == 200
    assert res.json == {
        'db://localhost:5433/postgres!tsh': [
            ['test-naive', 'primary'],
            ['test', 'primary']
        ]
    }

    # metadata
    res = client.get('/series/metadata?name=test')
    meta = res.json
    assert meta == {}

    res = client.get('/series/metadata?name=test&all=1')
    meta = res.json
    meta.pop('supervision_status', None)
    assert meta == {
        'index_dtype': '|M8[ns]',
        'index_type': 'datetime64[ns, UTC]',
        'tzaware': True,
        'value_dtype': '<f8',
        'value_type': 'float64'
    }

    res = client.put('/series/metadata', params={
        'metadata': json.dumps({
            'freq': 'D',
            'description': 'banana spot price'
        }),
        'name': 'test'
    })
    assert res.status_code == 200
    res = client.get('/series/metadata?name=test')
    meta2 = res.json
    assert meta2 == {
        'freq': 'D',
        'description': 'banana spot price'
    }

    # metadata: delete by uploading an empty dict
    res = client.put('/series/metadata', params={
        'metadata': json.dumps({}),
        'name': 'test'
    })
    assert res.status_code == 200
    res = client.get('/series/metadata?name=test')
    meta2 = res.json
    assert meta2 == {}

    # get
    res = client.get('/series/state?name=test')
    series = util.fromjson(res.body, 'test', meta['tzaware'])
    assert_df("""
2018-01-01 00:00:00+00:00    0.0
2018-01-01 01:00:00+00:00    1.0
2018-01-01 02:00:00+00:00    2.0
""", series)


    # reinsert
    series_in = genserie(utcdt(2018, 1, 1, 3), 'H', 1, [3])
    res = client.patch('/series/state', params={
        'name': 'test',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 13),
        'tzaware': util.tzaware_serie(series_in)
    })

    assert res.status_code == 200

    res = client.get('/series/state?name=test')
    series = util.fromjson(res.body, 'test', meta['tzaware'])
    assert_df("""
2018-01-01 00:00:00+00:00    0.0
2018-01-01 01:00:00+00:00    1.0
2018-01-01 02:00:00+00:00    2.0
2018-01-01 03:00:00+00:00    3.0
""", series)

    # checkout a past state
    res = client.get('/series/state', params={
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
    res = client.get('/series/state', params={
        'name': 'test',
        'insertion_date': utcdt(2018, 1, 1, 0)
    })
    assert res.json == {}

    # history
    res = client.get('/series/history?name=test')
    df = pd.read_json(res.body)

    # we real client would need to handle timestamp
    # tz-awareness
    assert_df("""
2018-01-01 10:00:00  2018-01-01 13:00:00
2018-01-01 00:00:00                  0.0                    0
2018-01-01 01:00:00                  1.0                    1
2018-01-01 02:00:00                  2.0                    2
2018-01-01 03:00:00                  NaN                    3
""", df)

    res = client.get('/series/history?name=test&format=tshpack')
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
    res = client.get('/series/history', params={
        'name': 'test',
        'diffmode': True
    })
    df = pd.read_json(res.body)

    assert_df("""
2018-01-01 10:00:00  2018-01-01 13:00:00
2018-01-01 00:00:00                  0.0                  NaN
2018-01-01 01:00:00                  1.0                  NaN
2018-01-01 02:00:00                  2.0                  NaN
2018-01-01 03:00:00                  NaN                  3.0
""", df)

    # empty range
    res = client.get('/series/history', params={
        'name': 'test',
        'from_insertion_date': utcdt(2018, 1, 1, 11),
        'to_insertion_date': utcdt(2018, 1, 1, 12),
    })
    df = pd.read_json(res.body)
    assert len(df) == 0

    # insertion dates subset
    res = client.get('/series/history', params={
        'name': 'test',
        'from_insertion_date': utcdt(2018, 1, 1, 10),
        'to_insertion_date': utcdt(2018, 1, 1, 12),
    })
    df = pd.read_json(res.body)

    assert_df("""
                     2018-01-01 10:00:00
2018-01-01 00:00:00                    0
2018-01-01 01:00:00                    1
2018-01-01 02:00:00                    2
""", df)

    # value dates subset
    res = client.get('/series/history', params={
        'name': 'test',
        'from_value_date': utcdt(2018, 1, 1, 2),
        'to_value_date': utcdt(2018, 1, 1, 3),
    })
    df = pd.read_json(res.body)

    assert_df("""
                     2018-01-01 10:00:00  2018-01-01 13:00:00
2018-01-01 02:00:00                  2.0                    2
2018-01-01 03:00:00                  NaN                    3
""", df)

    # state/get from from/to value date restriction
    res = client.get('/series/state', params={
        'name': 'test',
        'from_value_date': utcdt(2018, 1, 1, 1),
        'to_value_date': utcdt(2018, 1, 1, 2)
    })
    assert res.json == {
        '2018-01-01T01:00:00.000Z': 1.0,
        '2018-01-01T02:00:00.000Z': 2.0
    }

    series_in = genserie(utcdt(2019, 1, 1), 'H', 3)
    res = client.patch('/series/state', params={
        'name': 'test',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 2),
        'tzaware': util.tzaware_serie(series_in),
        'replace': True
    })

    assert res.status_code == 200
    res = client.get('/series/state', params={
        'name': 'test'
    })
    assert res.json == {
        '2019-01-01T00:00:00.000Z': 0.0,
        '2019-01-01T01:00:00.000Z': 1.0,
        '2019-01-01T02:00:00.000Z': 2.0
    }

    res = client.get('/series/metadata', params={
        'name': 'test',
        'type': 'interval'
    })
    assert res.json == [
        True,
        '2019-01-01T00:00:00+00:00',
        '2019-01-01T02:00:00+00:00'
    ]

    res = client.get('/series/metadata', params={
        'name': 'test',
        'type': 'type'
    })
    assert res.json == 'primary'

    res = client.get('/series/insertion_dates', params={
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


def test_delete(client):
    series_in = genserie(utcdt(2018, 1, 1), 'H', 3)
    res = client.patch('/series/state', params={
        'name': 'test',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series_in)
    })

    res = client.delete('/series/state', params={
        'name': 'no-such-series'
    })
    assert res.status_code == 404
    res = client.delete('/series/state', params={
        'name': 'test'
    })
    assert res.status_code == 204
    res = client.get('/series/catalog')
    assert 'test' not in res.json


def test_rename(client):
    series_in = genserie(utcdt(2018, 1, 1), 'H', 3)
    res = client.patch('/series/state', params={
        'name': 'test',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series_in)
    })
    res = client.put('/series/state', params={
        'name': 'no-such-series',
        'newname': 'no-better'
    })
    assert res.status_code == 404
    res = client.put('/series/state', params={
        'name': 'test',
        'newname': 'test2'
    })
    assert res.status_code == 204
    res = client.get('/series/catalog')
    assert res.json == {
        'db://localhost:5433/postgres!tsh': [
            ['test-naive', 'primary'],
            ['test2', 'primary']
        ]
    }

    assert 'test' not in res.json

    res = client.patch('/series/state', params={
        'name': 'test3',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series_in)
    })
    res = client.put('/series/state', params={
        'name': 'test2',
        'newname': 'test3'
    })
    assert res.status_code == 409
    assert res.json == {
        'message': '`test3` does exists'
    }



def test_strip(client):
    series_in = genserie(utcdt(2021, 1, 1), 'H', 3)
    res = client.patch('/series/state', params={
        'name': 'stripme',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2021, 1, 1),
        'tzaware': util.tzaware_serie(series_in)
    })
    series_in = genserie(utcdt(2021, 1, 2), 'H', 3)
    res = client.patch('/series/state', params={
        'name': 'stripme',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2021, 1, 2),
        'tzaware': util.tzaware_serie(series_in)
    })

    res = client.put('/series/strip', params={
        'name': 'stripme',
        'insertion_date': utcdt(2021, 1, 2)
    })
    assert res.status_code == 204

    res = client.get('/series/insertion_dates', params={
        'name': 'stripme'
    })
    idates = json.loads(res.text)['insertion_dates']
    assert len(idates) == 1


def test_staircase(client):
    # each days we insert 7 data points
    for idate in pd.date_range(start=utcdt(2015, 1, 1),
                               end=utcdt(2015, 1, 4),
                               freq='D'):
        series = genserie(start=idate, freq='H', repeat=7)
        client.patch('/series/state', params={
            'name': 'staircase',
            'series': util.tojson(series),
            'author': 'Babar',
            'insertion_date': idate,
            'tzaware': util.tzaware_serie(series)
        })

    res = client.get('/series/staircase', params={
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

    res = client.get('/series/staircase', params={
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


def test_get_fast_path(client):
    series_in = genserie(utcdt(2018, 1, 1), 'H', 3)
    res = client.patch('/series/state', params={
        'name': 'test_fast',
        'series': util.tojson(series_in),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series_in)
    })

    assert res.status_code == 201

    out = client.get('/series/state', params={
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


def test_multisource(client, engine):
    series = genserie(utcdt(2020, 1, 1), 'D', 3)
    res = client.patch('/series/state', params={
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

    out = client.get('/series/state', params={
        'name': 'test-multi',
    })
    assert out.json == {
        '2020-01-01T00:00:00.000Z': 0.0,
        '2020-01-02T00:00:00.000Z': 1.0,
        '2020-01-03T00:00:00.000Z': 2.0
    }

    out = client.get('/series/state', params={
        'name': 'test-other-source',
    })
    assert out.json == {
        '2020-01-01T00:00:00.000Z': 0.0,
        '2020-01-02T00:00:00.000Z': 1.0,
        '2020-01-03T00:00:00.000Z': 2.0
    }

    res = client.patch('/series/state', params={
        'name': 'test-multi',
        'series': util.tojson(series),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series)
    })
    assert res.status_code == 200

    res = client.patch('/series/state', params={
        'name': 'test-other-source',
        'series': util.tojson(series),
        'author': 'Babar',
        'insertion_date': utcdt(2018, 1, 1, 10),
        'tzaware': util.tzaware_serie(series)
    })
    assert res.status_code == 405
    assert res.json == {'message': 'not allowed to update to a secondary source'}

    res = client.get('/series/metadata?name=test-other-source', params={
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

    res = client.put('/series/metadata', params={
        'metadata': json.dumps({
            'description': 'banana spot price'
        }),
        'name': 'test-other-source'
    })
    assert res.status_code == 405
    assert res.json == {
        'message': 'not allowed to update metadata to a secondary source'
    }

    res = client.delete('/series/state', params={
        'name': 'test-other-source'
    })
    assert res.status_code == 405
    assert res.json == {
        'message': 'not allowed to delete to a secondary source'
    }

    res = client.delete('/series/state', params={
        'name': 'test-other-source'
    })
    assert res.status_code == 405
    assert res.json == {
        'message': 'not allowed to delete to a secondary source'
    }

    res = client.put('/series/state', params={
        'name': 'test-other-source',
        'newname': 'test2-other-source'
    })
    assert res.status_code == 405
    assert res.json == {
        'message': 'not allowed to rename to a secondary source'
    }

    # catalog
    res = client.get('/series/catalog')
    assert res.status_code == 200
    assert res.json == {
        'db://localhost:5433/postgres!other': [
            ['test-other-source', 'primary']
        ],
        'db://localhost:5433/postgres!tsh': [
            ['test-naive', 'primary'],
            ['test2', 'primary'],
            ['test3', 'primary'],
            ['stripme', 'primary'],
            ['staircase', 'primary'],
            ['test_fast', 'primary'],
            ['test-multi', 'primary']
        ]
    }
    res = client.get('/series/catalog', params={
        'allsources': False
    })
    assert res.status_code == 200
    assert 'db://localhost:5433/postgres!tsh' in res.json
    assert 'db://localhost:5433/postgres!other' not in res.json


def test_group(client):
    df = gengroup(
        n_scenarios=3,
        from_date=dt(2021, 1, 1),
        length=5,
        freq='D',
        seed=2.
    )

    df.columns = ['a', 'b', 'c']

    bgroup = util.pack_group(df)
    res = client.patch('/group/state', {
        'name': 'test_group',
        'author': 'Babar',
        'format': 'tshpack',
        'replace': json.dumps(True),
        'bgroup': webtest.Upload('bgroup', bgroup)
    })
    assert res.status_code == 201

    res = client.get('/group/state', {'name': 'test_group'})
    df2 = util.unpack_group(res.body)
    assert df.equals(df2)

    res = client.get('/group/catalog')
    assert res.json == {
        'db://localhost:5433/postgres!tsh': [['test_group', 'primary']]
    }

    res = client.get('/group/metadata', {'name': 'test_group'})
    assert res.json == {}

    res = client.put('/group/metadata', {
        'name': 'test_group',
        'metadata': json.dumps({'foo': 'bar'})
    })
    assert res.status_code == 200

    res = client.get('/group/metadata', {'name': 'test_group'})
    assert res.json == {'foo': 'bar'}

    res = client.delete('/group/state', {'name': 'test_group'})
    assert res.status_code == 204

    res = client.get('/group/catalog')
    assert res.json == {}


def test_log(client):
    series = genserie(utcdt(2020, 1, 1), 'D', 5)
    for d in range(5):
        res = client.patch('/series/state', params={
            'name': 'test-log',
            'series': util.tojson(series),
            'author': 'Babar',
            'insertion_date': utcdt(2020, 1, d + 1).isoformat(),
            'metadata': json.dumps({'comment': f'day {d+1}'}),
            'tzaware': util.tzaware_serie(series)
        })
        assert res.status_code in (201, 200)
        series[d] = 42


    out = client.get('/series/state', params={
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

    res = client.get('/series/log', params={
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

    res = client.get('/series/log', params={
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

    res = client.get('/series/log', params={
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
