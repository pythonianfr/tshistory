import os
from pathlib import Path

from inireader import reader
from sqlalchemy.engine import url

from tshistory.util import unflatten


class configuration:
    optsections = ('sources', 'auth', 'server-auth')
    defaults = {
        'sources': {},
        'auth': {},
        'server-auth': {}
    }

    @staticmethod
    def path():
        if 'TSHISTORYCFGPATH' in os.environ:
            cfgpath = Path(os.environ['TSHISTORYCFGPATH'])
            if cfgpath.exists():
                return cfgpath

        cfgpath = Path('tshistory.cfg')
        if cfgpath.exists():
            return cfgpath

        cfgpath = Path('~/tshistory.cfg').expanduser()
        if cfgpath.exists():
            return cfgpath

        cfgpath = Path(
            os.environ.get('XDG_CONFIG_HOME', '~/.config'),
            'tshistory.cfg'
        ).expanduser()
        if cfgpath.exists():
            return cfgpath

    def __init__(self, adict=None):
        # NOTE: we want a validation step
        if adict is not None:
            self.cfg = adict
        else:
            cfgpath = configuration.path()
            if not cfgpath:
                raise Exception('No `tshistory.cfg` file could be found.')
            self.cfg = reader(cfgpath)
        self._finish()

    def __getitem__(self, name):
        return self.cfg[name]

    def __contains__(self, name):
        return name in self.cfg

    def _finish(self):
        assert 'dburi' in self.cfg, 'No [dburi] section'
        for sec in self.optsections:
            if sec not in self.cfg:
                self.cfg._data[sec] = self.defaults[sec]

    def find_dburi(self, something: str) -> str:
        if something.startswith('http'):
            return something
        try:
            url.make_url(something)
        except Exception:
            pass
        else:
            return something

        try:
            return self.cfg['dburi'][something]
        except Exception as exc:
            raise Exception(
                f'could not find the `{something}` entry in the '
                f'[dburi] section of the tshistory.cfg '
                f'conf file (cause: {exc.__class__.__name__} -> {exc})'
            )

    def find_first_uri(self):
        assert 'dburi' in self.cfg, 'Your tshistory.cfg file does not contain a [dburi] section.'
        return next(
            iter(
                self.cfg['dburi'].values()
            )
        )

    def find_first_uriname(self):
        assert 'dburi' in self.cfg, 'Your tshistory.cfg file does not contain a [dburi] section.'
        return next(
            iter(
                self.cfg['dburi'].keys()
            )
        )

    def find_sources(self, uri):
        # The [db] section may contain several name -> uri entries. We find
        # the matching name and then we can find the associated sources.
        for localname, dburi in self.cfg['dburi'].items():
            if uri == dburi:
                break
        else:
            raise Exception(f'No match for {uri} in the tshistory.cfg file.')

        allsources = unflatten(self.cfg['sources'])
        sources = {}
        for name, source in allsources.get(localname, {}).items():
            uri, ns = source.split(',')
            sources[name] = (uri.strip(), ns.strip())
        return sources

    def auth(self, uri):
        try:
            return next(
                iter(
                    self.cfg['auth'].keys()
                )
            )
        except KeyError:
            return {}
