from datetime import datetime, timedelta
from urllib.parse import urlparse
from typing import (
    Dict,
    Optional,
    Union
)
from collections import defaultdict

from sqlalchemy import create_engine
import pandas as pd

from tshistory.tsio import timeseries as tshclass


class timeseries:

    def __new__(cls, uri,
                namespace='tsh',
                handler=tshclass,
                sources=()):
        parseduri = urlparse(uri)
        if parseduri.scheme.startswith('postgres'):
            return dbtimeseries(
                uri,
                namespace,
                tshclass=handler,
                othersources=altsources(handler, sources)
            )
        elif parseduri.scheme.startswith('http'):
            try:
                from tshistory_client.api import Client
            except ImportError:
                raise NotImplementedError(
                    f'to handle `{uri}` you should install `tshistory_client`'
                )
            return Client(uri)  # in the default namespace !
        raise NotImplementedError(uri)


class dbtimeseries:
    __slots__ = (
        'uri', 'namespace',
        'engine', 'tsh',
        'othersources'
    )

    def __repr__(self):
        return (
            f'timeseries('
            f'uri={self.uri},'
            f'ns={self.namespace},'
            f'sources={self.othersources or nil})'
        )

    def __init__(self,
                 uri: str,
                 namespace: str='tsh',
                 tshclass: type=tshclass,
                 othersources=None):
        self.uri = uri
        self.namespace = namespace
        self.engine = create_engine(uri)
        self.tsh = tshclass(namespace, othersources)
        self.othersources = othersources

    def update(self,
               name: str,
               updatets: pd.Series,
               author: str,
               metadata: Optional[dict]=None,
               insertion_date: Optional[datetime]=None,
               **kw) -> Optional[pd.Series]:

        # give a chance to say *no*
        self.othersources.update(name)

        return self.tsh.update(
            self.engine,
            updatets,
            name,
            author,
            metadata=metadata,
            insertion_date=insertion_date,
            **kw
        )

    def replace(self,
                name: str,
                updatets: pd.Series,
                author: str,
                metadata: Optional[dict]=None,
                insertion_date: Optional[datetime]=None,
                **kw) -> Optional[pd.Series]:

        # give a chance to say *no*
        self.othersources.replace(name)

        return self.tsh.replace(
            self.engine,
            updatets,
            name,
            author,
            metadata=metadata,
            insertion_date=insertion_date,
            **kw
        )

    def exists(self, name):
        if (not self.tsh.exists(self.engine, name) and
            not self.othersources.exists(name)):
            return False

        return True

    def get(self, name: str,
            revision_date: Optional[datetime]=None,
            from_value_date: Optional[datetime]=None,
            to_value_date: Optional[datetime]=None) -> Optional[pd.Series]:

        ts = self.tsh.get(
            self.engine,
            name,
            revision_date=revision_date,
            from_value_date=from_value_date,
            to_value_date=to_value_date
        )

        if ts is None:
            ts = self.othersources.get(
                name,
                revision_date=revision_date,
                from_value_date=from_value_date,
                to_value_date=to_value_date
            )
        return ts

    def history(self,
                name: str,
                from_insertion_date: Optional[datetime]=None,
                to_insertion_date: Optional[datetime]=None,
                from_value_date: Optional[datetime]=None,
                to_value_date: Optional[datetime]=None,
                diffmode: bool=False,
                _keep_nans: bool=False) -> Dict[datetime, pd.Series]:

        hist = self.tsh.history(
            self.engine,
            name,
            from_insertion_date=from_insertion_date,
            to_insertion_date=to_insertion_date,
            from_value_date=from_value_date,
            to_value_date=to_value_date,
            diffmode=diffmode,
            _keep_nans=_keep_nans
        )

        if not hist:
            hist = self.othersources.history(
                name,
                from_insertion_date=from_insertion_date,
                to_insertion_date=to_insertion_date,
                from_value_date=from_value_date,
                to_value_date=to_value_date,
                diffmode=diffmode,
                _keep_nans=_keep_nans
            )
        return hist

    def staircase(self,
                  name: str,
                  delta: timedelta,
                  from_value_date: Optional[datetime]=None,
                  to_value_date: Optional[datetime]=None) -> Optional[pd.Series]:
        sc = self.tsh.staircase(
            self.engine,
            name,
            delta,
            from_value_date=from_value_date,
            to_value_date=to_value_date
        )

        if sc is None:
            sc = self.othersources.staircase(
                name,
                delta,
                from_value_date=from_value_date,
                to_value_date=to_value_date
            )
        return sc

    def catalog(self, allsources=True):
        parsed = urlparse(self.uri)
        instancename = f'db://{parsed.netloc.split("@")[-1]}{parsed.path}'
        cat = defaultdict(list)
        for name, kind in self.tsh.list_series(self.engine).items():
            cat[(instancename, self.namespace)].append((name, kind))
        if allsources:
            for key, val in self.othersources.catalog(allsources).items():
                assert key not in cat, f'{key} already in {cat}'
                cat[key] = val
        return cat

    def interval(self, name: str) -> pd.Interval:
        try:
            ival = self.tsh.interval(self.engine, name)
        except ValueError:
            return self.othersources.interval(name)
        return ival

    def metadata(self,
                 name: str,
                 all: bool=False):

        meta = self.tsh.metadata(self.engine, name)
        if not meta:
            meta = self.othersources.metadata(name)
        if all:
            return meta
        for key in self.tsh.metakeys:
            meta.pop(key, None)
        return meta

    def update_metadata(self,
                        name: str,
                        metadata: dict):

        # give a chance to say *no*
        self.othersources.update_metadata(name)
        self.tsh.update_metadata(
            self.engine,
            name,
            metadata
        )

    def type(self, name: str):
        return self.tsh.type(self.engine, name)

    def rename(self,
               currname: str,
               newname: str):

        # give a chance to say *no*
        self.othersources.rename(currname)
        return self.tsh.rename(self.engine, currname, newname)

    def delete(self, name: str):
        # give a chance to say *no*
        self.othersources.delete(name)
        return self.tsh.delete(self.engine, name)


class source:
    __slots__ = 'uri', 'namespace', 'tsa'

    def __init__(self, uri, namespace, tshclass):
        self.uri = uri
        self.namespace = namespace
        self.tsa = timeseries(uri, namespace, tshclass)

    def __repr__(self):
        return f'source(uri={self.uri},ns={self.namespace})'


class altsources:
    " Class to handle some operations allowed on secondary sources "
    __slots__ = ('sources',)

    def __init__(self, tshclass, sources=()):
        self.sources = [
            source(
                src_uri,
                src_namespace,
                tshclass
            )
            for src_uri, src_namespace in sources
        ]

    def __repr__(self):
        return str(self.sources)

    def _findsourcefor(self, name):
        for source in self.sources:
            if source.tsa.exists(name):
                return source

    def exists(self, name):
        for source in self.sources:
            if source.tsa.exists(name):
                return True
        return False

    def get(self, name: str,
            revision_date: Optional[datetime]=None,
            from_value_date: Optional[datetime]=None,
            to_value_date: Optional[datetime]=None) -> Optional[pd.Series]:
        source = self._findsourcefor(name)
        if source is None:
            return
        return source.tsa.get(
            name,
            revision_date=revision_date,
            from_value_date=from_value_date,
            to_value_date=to_value_date
        )

    def history(self,
                name: str,
                from_insertion_date: Optional[datetime]=None,
                to_insertion_date: Optional[datetime]=None,
                from_value_date: Optional[datetime]=None,
                to_value_date: Optional[datetime]=None,
                diffmode: bool=False,
                _keep_nans: bool=False) -> Dict[datetime, pd.Series]:
        source = self._findsourcefor(name)
        if source is None:
            return
        return source.tsa.history(
            name,
            from_insertion_date=from_insertion_date,
            to_insertion_date=to_insertion_date,
            from_value_date=from_value_date,
            to_value_date=to_value_date,
            diffmode=diffmode,
            _keep_nans=_keep_nans
        )

    def metadata(self, name: str):
        source = self._findsourcefor(name)
        if source is None:
            return
        meta = source.tsa.metadata(name, all=True)
        return meta

    def interval(self, name: str) -> pd.Interval:
        source = self._findsourcefor(name)
        if source is None:
            raise ValueError(f'no interval for series: {name}')
        ival = source.tsa.interval(name)
        if ival is None:
            raise ValueError(f'no interval for series: {name}')
        return ival

    def update(self, name):
        if self.exists(name):
            raise ValueError(
                'not allowed to update to a secondary source'
            )

    def replace(self, name):
        if self.exists(name):
            raise ValueError(
                'not allowed to replace to a secondary source'
            )

    def update_metadata(self, name):
        if self.exists(name):
            raise ValueError(
                'not allowed to update metadata to a secondary source'
            )

    def rename(self, name):
        if self.exists(name):
            raise ValueError(
                'not allowed to rename to a secondary source'
            )

    def delete(self, name: str):
        if self.exists(name):
            raise ValueError(
                'not allowed to delete to a secondary source'
            )

    def catalog(self, allsources=False):
        cat = {}
        for source in self.sources:
            cat.update(source.tsa.catalog(allsources))
        return cat
