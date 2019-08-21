from datetime import datetime

import pandas as pd
import numpy as np

from tshistory.util import _set_cache
from tshistory.snapshot import Snapshot
from tshistory.testutil import (
    assert_df,
    assert_hist,
    assert_group_equals,
    genserie,
    tempattr,
    utcdt
)


def chunksize(snap, head, from_value_date=None):
    return {
        parent: len(snap._chunks_to_ts([rawchunk]))
        for _, parent, rawchunk in snap.rawchunks(
                head, from_value_date=from_value_date
        )
    }


def test_chunks(engine, tsh):
    _set_cache(engine)
    with tempattr(Snapshot, '_max_bucket_size', 2):
        ts = genserie(datetime(2010, 1, 1), 'D', 5)
        tsh.insert(engine, ts, 'chunks', 'test')

        assert_df("""
2010-01-01    0.0
2010-01-02    1.0
2010-01-03    2.0
2010-01-04    3.0
2010-01-05    4.0
""", tsh.get(engine, 'chunks'))

        # we expect 3 chunks
        sql = 'select parent, chunk from "{}.snapshot".chunks order by id'.format(
            tsh.namespace
        )
        chunks = engine.execute(sql).fetchall()
        assert len(chunks) == 3
        assert chunks[0].parent is None
        assert chunks[1].parent == 1
        assert chunks[2].parent == 2
        snap = Snapshot(engine, tsh, 'chunks')
        ts0 = snap._chunks_to_ts([chunks[0].chunk])
        ts1 = snap._chunks_to_ts([chunks[1].chunk])
        ts2 = snap._chunks_to_ts([chunks[2].chunk])

        assert_df("""
2010-01-01    0.0
2010-01-02    1.0
""", ts0)

        assert_df("""
2010-01-03    2.0
2010-01-04    3.0
""", ts1)

        assert_df("""
2010-01-05    4.0
""", ts2)

        ts = pd.Series([4, 5, 6, 7, 8],
                       index=pd.date_range(start=datetime(2010, 1, 5),
                                           end=datetime(2010, 1, 9),
                                           freq='D'))

        tsh.insert(engine, ts, 'chunks', 'test')
        whole = tsh.get(engine, 'chunks')
        assert_df("""
2010-01-01    0.0
2010-01-02    1.0
2010-01-03    2.0
2010-01-04    3.0
2010-01-05    4.0
2010-01-06    5.0
2010-01-07    6.0
2010-01-08    7.0
2010-01-09    8.0
""", whole)

        # we expect 5 chunks
        sql = 'select id, parent, chunk from "{}.snapshot".chunks order by id'.format(
            tsh.namespace
        )
        chunks = engine.execute(sql).fetchall()
        assert len(chunks) == 5
        assert chunks[4].parent == 4
        assert {
            1: None,
            2: 1,
            3: 2,
            4: 3,
            5: 4
        } == {
            chunk.id: chunk.parent for chunk in chunks
        }

        snap = Snapshot(engine, tsh, 'chunks')
        ts0 = snap._chunks_to_ts([chunks[0].chunk])
        ts1 = snap._chunks_to_ts([chunks[1].chunk])
        ts2 = snap._chunks_to_ts([chunks[2].chunk])
        ts3 = snap._chunks_to_ts([chunks[3].chunk])
        ts4 = snap._chunks_to_ts([chunks[4].chunk])

        assert_df("""
2010-01-01    0.0
2010-01-02    1.0
""", ts0)

        assert_df("""
2010-01-03    2.0
2010-01-04    3.0
""", ts1)

        assert_df("""
2010-01-05    4.0
""", ts2)

        assert_df("""
2010-01-06    5.0
2010-01-07    6.0
""", ts3)

        assert_df("""
2010-01-08    7.0
2010-01-09    8.0
""", ts4)

        # non-append edit
        whole[2] = 0
        whole[7] = 0

        tsh.insert(engine, whole, 'chunks', 'test')

        assert_df("""
2010-01-01    0.0
2010-01-02    1.0
2010-01-03    0.0
2010-01-04    3.0
2010-01-05    4.0
2010-01-06    5.0
2010-01-07    6.0
2010-01-08    0.0
2010-01-09    8.0
""", tsh.get(engine, 'chunks'))

        assert_df("""
2010-01-05    4.0
2010-01-06    5.0
2010-01-07    6.0
2010-01-08    0.0
2010-01-09    8.0
""", tsh.get(engine, 'chunks', from_value_date=datetime(2010, 1, 5)))

        # we expect 9 chunks
        # because we edit from the second chunk
        # and 3 new chunks have to be made
        sql = 'select id, parent, chunk from "{}.snapshot".chunks order by id'.format(
            tsh.namespace
        )
        chunks = engine.execute(sql).fetchall()
        assert len(chunks) == 9
        assert {
            1: None,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 1, # base of third commit (loss of shared chunks)
            7: 6,
            8: 7,
            9: 8
        } == {
            chunk.id: chunk.parent for chunk in chunks
        }

        # 2nd commit chunks without filtering
        snap = Snapshot(engine, tsh, 'chunks')
        chunks = chunksize(snap, 5)
        assert chunks == {
            None: 2,
            1: 2,
            2: 1,
            3: 2,
            4: 2
        }

        # 2nd commit chunks with filtering
        chunks = chunksize(snap, 6, datetime(2010, 1, 5))
        assert chunks == {1: 2}

        # 3rd commit chunks without filtering
        chunks = chunksize(snap, 9)
        assert chunks == {
            None: 2,
            1: 2,
            6: 2,
            7: 2,
            8: 1
        }

        # 3rd commit chunks with filtering
        chunks = chunksize(snap, 9, datetime(2010, 1, 5))
        assert chunks == {
            6: 2,
            7: 2,
            8: 1
        }


def test_append(engine, tsh):
    if tsh.namespace == 'z-z':
        return

    for x, dt in enumerate(pd.date_range(start=utcdt(2018, 1, 1),
                                         freq='D', periods=10)):
        ts = genserie(dt, 'D', 1, [x], name='daily')
        tsh.insert(engine, ts, 'append', 'aurelien.campeas@pythonian.fr',
                   insertion_date=dt)

    sql = 'select id, parent, chunk from "{}.snapshot".append order by id'.format(
        tsh.namespace
    )
    chunks = engine.execute(sql).fetchall()
    c = {
        chunk.id: chunk.parent for chunk in chunks
    }
    # nice linked list
    assert c == {
        1: None, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9
    }


def test_prepend(engine, tsh):
    if tsh.namespace == 'z-z':
        return

    serie = genserie(datetime(2010, 1, 1), 'D', 40)

    assert 40 == len(serie)
    ts_insert = serie[2:]
    tsh.insert(engine, ts_insert, 'prepend', 'test')
    assert 38 == len(tsh.get(engine, 'prepend'))

    tsh.insert(engine, serie, 'prepend', 'test')
    assert 40 == len(tsh.get(engine, 'prepend'))

    sql = 'select id, parent, chunk from "{}.snapshot".prepend order by id'.format(
        tsh.namespace
    )
    chunks = engine.execute(sql).fetchall()
    c = {
        chunk.id: chunk.parent for chunk in chunks
    }

    # ugly un-linked list
    assert c == {1: None, 2: None}


def test_get_from_to(engine, tsh):
    ts = genserie(datetime(2015, 1, 1), 'D', 365)
    tsh.insert(engine, ts, 'quitelong', 'aurelien.campeas@pythonian.fr')

    snap = Snapshot(engine, tsh, 'quitelong')
    if tsh.namespace == 'z-z':
        sql = 'select id, parent from "z-z.snapshot".quitelong order by id'
        chunks = engine.execute(sql).fetchall()
        # should be perfectly chained
        chunks = {
            chunk.id: chunk.parent for chunk in chunks
        }
        chunks.pop(1)
        assert all(k == v+1 for k, v in chunks.items())

        snap = Snapshot(engine, tsh, 'quitelong')
        chunks = chunksize(snap, 73)
        assert chunks == {None: 5, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5,
                          8: 5, 9: 5, 10: 5, 11: 5, 12: 5, 13: 5, 14: 5, 15: 5,
                          16: 5, 17: 5, 18: 5, 19: 5, 20: 5, 21: 5, 22: 5,
                          23: 5, 24: 5, 25: 5, 26: 5, 27: 5, 28: 5, 29: 5,
                          30: 5, 31: 5, 32: 5, 33: 5, 34: 5, 35: 5, 36: 5,
                          37: 5, 38: 5, 39: 5, 40: 5, 41: 5, 42: 5, 43: 5, 44: 5,
                          45: 5, 46: 5, 47: 5, 48: 5, 49: 5, 50: 5, 51: 5, 52: 5,
                          53: 5, 54: 5, 55: 5, 56: 5, 57: 5, 58: 5, 59: 5, 60: 5,
                          61: 5, 62: 5, 63: 5, 64: 5, 65: 5, 66: 5, 67: 5, 68: 5,
                          69: 5, 70: 5, 71: 5, 72: 5}
        chunks = chunksize(snap, 73, datetime(2015, 5, 1))
        assert chunks == {24: 5, 25: 5, 26: 5, 27: 5, 28: 5, 29: 5, 30: 5, 31: 5,
                          32: 5, 33: 5, 34: 5, 35: 5, 36: 5, 37: 5, 38: 5, 39: 5,
                          40: 5, 41: 5, 42: 5, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5,
                          48: 5, 49: 5, 50: 5, 51: 5, 52: 5, 53: 5, 54: 5, 55: 5,
                          56: 5, 57: 5, 58: 5, 59: 5, 60: 5, 61: 5, 62: 5, 63: 5,
                          64: 5, 65: 5, 66: 5, 67: 5, 68: 5, 69: 5, 70: 5, 71: 5,
                          72: 5}

    serie = tsh.get(engine, 'quitelong')
    assert serie.index[0] == pd.Timestamp('2015-01-01 00:00:00')
    assert serie.index[-1] == pd.Timestamp('2015-12-31 00:00:00')

    serie = tsh.get(engine, 'quitelong',
                    from_value_date=datetime(2015, 5, 1),
                    to_value_date=datetime(2015, 6, 1))
    assert serie.index[0] == pd.Timestamp('2015-05-01 00:00:00')
    assert serie.index[-1] == pd.Timestamp('2015-06-01 00:00:00')

    serie = tsh.get(engine, 'quitelong',
                    from_value_date=datetime(2016, 5, 1),
                    to_value_date=datetime(2016, 6, 1))
    assert serie.dtypes == np.dtype('float64')
    assert len(serie) == 0
    assert isinstance(serie.index, pd.DatetimeIndex)
    assert serie.index.freq is None
