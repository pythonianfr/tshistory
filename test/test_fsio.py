import pandas as pd


def test_exists(engine, tsf):
    assert not tsf.exists(engine, 'hello')


def test_create_empty(engine, tsf):
    ts = pd.Series()

    tsf.update(
        engine,
        ts,
        'fs-first',
        'Babar'
    )

