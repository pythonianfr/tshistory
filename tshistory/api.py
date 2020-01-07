from datetime import datetime, timedelta
from urllib.parse import urlparse
from typing import (
    Dict,
    Optional,
    Union
)
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
                diffmode: bool=False) -> Dict[datetime, pd.Series]:

        hist = self.tsh.history(
            self.engine,
            name,
            from_insertion_date=from_insertion_date,
            to_insertion_date=to_insertion_date,
            from_value_date=from_value_date,
            to_value_date=to_value_date,
            diffmode=diffmode
        )

        if hist is None:
            hist = self.othersources.history(
                name,
                from_insertion_date=from_insertion_date,
                to_insertion_date=to_insertion_date,
                from_value_date=from_value_date,
                to_value_date=to_value_date,
                diffmode=diffmode
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

    def catalog(self):
        parsed = urlparse(self.uri)
        instancename = f'db://{parsed.netloc.split("@")[-1]}{parsed.path}'
        local = {
            name: (kind, instancename, self.namespace)
            for name, kind in self.tsh.list_series(self.engine).items()
        }
        others = self.othersources.catalog()
        local.update(**others)
        return local

    def interval(self, name: str) -> pd.Interval:
        ival = self.tsh.interval(self.engine, name)
        if ival is None:
            ival = self.othersources.interval(name)
        return ival

    def metadata(self,
                 name: str,
                 all: bool=False):

        meta = self.othersources.metadata(name, all)
        if not meta:
            meta = self.tsh.metadata(self.engine, name)
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

    # formula api extension

    def register_formula(self, name, formula,
                         reject_unknown=True, update=False):

        if not getattr(self.tsh, 'register_formula', False):
            raise TypeError(f'the {self.tsh} handler cannot register formulae')

        self.tsh.register_formula(
            self.engine,
            name,
            formula,
            reject_unknown=reject_unknown,
            update=update
        )


class source:
    __slots__ = ('engine', 'tsh', 'uri', 'namespace')

    def __init__(self, uri, namespace, tshclass):
        self.uri = uri
        self.namespace = namespace
        self.engine = create_engine(uri)
        self.tsh = tshclass(namespace)


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

    def _findsourcefor(self, name):
        for source in self.sources:
            if source.tsh.exists(source.engine, name):
                return source

    def exists(self, name):
        for source in self.sources:
            if source.tsh.exists(source.engine, name):
                return True
        return False

    def get(self, name: str,
            revision_date: Optional[datetime]=None,
            from_value_date: Optional[datetime]=None,
            to_value_date: Optional[datetime]=None) -> Optional[pd.Series]:
        source = self._findsourcefor(name)
        if source is None:
            return
        return source.tsh.get(
            source.engine,
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
                diffmode: bool=False) -> Dict[datetime, pd.Series]:
        source = self._findsourcefor(name)
        if source is None:
            return
        return source.tsh.history(
            source.engine,
            name,
            from_insertion_date=from_insertion_date,
            to_insertion_date=to_insertion_date,
            from_value_date=from_value_date,
            to_value_date=to_value_date,
            diffmode=diffmode
        )

    def metadata(self,
                 name: str,
                 all: bool=False):
        source = self._findsourcefor(name)
        if source is None:
            return
        meta = source.tsh.metadata(source.engine, name)
        if all:
            return meta
        for key in source.tsh.metakeys:
            meta.pop(key, None)
        return meta

    def interval(self, name: str) -> pd.Interval:
        source = self._findsourcefor(name)
        if source is None:
            return
        return source.interval(name)

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

    def catalog(self):
        cat = {}
        for source in self.sources:
            parsed = urlparse(source.uri)
            instancename = f'db://{parsed.netloc.split("@")[-1]}{parsed.path}'
            cat.update(**{
                name: (kind, instancename, source.namespace)
                for name, kind in source.tsh.list_series(source.engine).items()
            })
        return cat
