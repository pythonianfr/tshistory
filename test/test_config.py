import tempfile

import pytest

from tshistory.config import configuration
from tshistory.testutil import (
    tempconfig,
    tempenv
)


def test_no_config():
    with tempfile.TemporaryDirectory() as d:
        with tempenv(HOME=d):
            with pytest.raises(Exception) as err:
                configuration()
            assert err.value.args[0] == 'No `tshistory.cfg` file could be found.'


def test_empty_config():
    with tempconfig(b''):
        with pytest.raises(AssertionError) as err:
            configuration()
        assert err.value.args[0] == 'No [dburi] section'


def test_simple_config():
    with tempconfig(b'[dburi]\nfoo=postgresql:///refinery'):
        cfg = configuration()
        assert cfg.cfg._data == {
            'dburi': {'foo': 'postgresql:///refinery'},
            'sources': {},
            'auth': {},
            'server-auth': {}
        }


def test_sources_auth():
    with tempconfig(
            b'[dburi]\n'
            b'foo=postgresql:///refinery\n'
            b'[sources]\n'
            b'foo.bar=https://bar.io\n'
            b'[auth]\n'
            b'bar.uri=https://bar.io\n'
            b'bar.login=babar\n'
            b'bar.password=celeste\n'
    ):
        cfg = configuration()
        assert cfg.cfg._data == {
            'dburi': {'foo': 'postgresql:///refinery'},
            'sources': {'foo.bar': 'https://bar.io'},
            'auth': {
                'bar.uri': 'https://bar.io',
                'bar.login': 'babar',
                'bar.password': 'celeste'},
            'server-auth': {}
        }


def test_sources_server_auth():
    with tempconfig(
            b'[dburi]\n'
            b'foo=postgresql:///refinery\n'
            b'[sources]\n'
            b'foo.bar=https://bar.io\n'
            b'[auth]\n'
            b'bar.uri=https://bar.io\n'
            b'bar.login=babar\n'
            b'bar.password=celeste\n'
            b'[server-auth]\n'
            b'client_id=xxx\n'
            b'client_secret=yyy\n'
            b'domain=zzz\n'
            b'authorize_uri=https://foo.io/authorize\n'
            b'audience=https://foo.io/api\n'
    ):
        cfg = configuration()
        assert cfg.cfg._data == {
            'dburi': {'foo': 'postgresql:///refinery'},
            'sources': {'foo.bar': 'https://bar.io'},
            'auth': {
                'bar.uri': 'https://bar.io',
                'bar.login': 'babar',
                'bar.password': 'celeste'},
            'server-auth': {
                'client_id': 'xxx',
                'client_secret': 'yyy',
                'domain': 'zzz',
                'authorize_uri': 'https://foo.io/authorize',
                'audience': 'https://foo.io/api'}
        }
