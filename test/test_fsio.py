import pytest


def test_exists(engine, tsf):
    with pytest.raises(AttributeError):
        assert not tsf.exists(engine, 'hello')

