

def test_exists(engine, tsf):
    assert not tsf.exists(engine, 'hello')

