from datetime import datetime, timedelta
from typing import (
    Dict,
    Optional,
    Union
)
from sqlalchemy import create_engine
import pandas as pd

from tshistory.tsio import timeseries as tshclass


class timeseries:
    __slots__ = (
        'uri', 'namespace',
        'engine', 'tsh'
    )

    def __init__(self,
                 uri: str,
                 namespace: str='tsh',
                 tshclass: type=tshclass):
        self.uri = uri
        self.namespace = namespace
        self.engine = create_engine(uri)
        self.tsh = tshclass(namespace)

    def update(self,
               name: str,
               updatets: pd.Series,
               author: str,
               metadata: Optional[dict]=None,
               insertion_date: Optional[datetime]=None,
               **kw) -> Optional[pd.Series]:
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
        return self.tsh.exists(self.engine, name)

    def get(self, name: str,
            revision_date: Optional[datetime]=None,
            from_value_date: Optional[datetime]=None,
            to_value_date: Optional[datetime]=None) -> Optional[pd.Series]:
        return self.tsh.get(
            self.engine,
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
        return self.tsh.history(
            self.engine,
            name,
            from_insertion_date=from_insertion_date,
            to_insertion_date=to_insertion_date,
            from_value_date=from_value_date,
            to_value_date=to_value_date,
            diffmode=diffmode
        )

    def staircase(self,
                  name: str,
                  delta: timedelta,
                  from_value_date: Optional[datetime]=None,
                  to_value_date: Optional[datetime]=None) -> Optional[pd.Series]:
        return self.tsh.staircase(
            self.engine,
            name,
            delta,
            from_value_date=from_value_date,
            to_value_date=to_value_date
        )

    def catalog(self):
        return self.tsh.list_series(self.engine)

    def interval(self, name: str) -> pd.Interval:
        return self.tsh.interval(self.engine, name)

    def metadata(self,
                 name: str,
                 all: bool=False):
        meta = self.tsh.metadata(self.engine, name)
        if all:
            return meta
        for key in self.tsh.metakeys:
            meta.pop(key, None)
        return meta

    def update_metadata(self,
                        name: str,
                        metadata: dict):
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
        return self.tsh.rename(self.engine, currname, newname)

    def delete(self, name: str):
        return self.tsh.delete(self.engine, name)



class source:
    __slots__ = ('engine', 'tsh', 'uri', 'namespace')

    def __init__(self, uri, namespace, tshclass):
        self.uri = uri
        self.namespace = namespace
        self.engine = create_engine(uri)
        self.tsh = tshclass(namespace)


class multisourcetimeseries(timeseries):
    __slots__ = (
        'uri', 'namespace',
        'mainsource', 'sources'
    )

    @property
    def engine(self):
        return self.mainsource.engine

    @property
    def tsh(self):
        return self.mainsource.tsh

    def __init__(self,
                 uri: str,
                 namespace: str='tsh',
                 tshclass: type=tshclass):
        self.uri = uri
        self.namespace = namespace
        self.mainsource = source(uri, namespace, tshclass)
        self.sources = [self.mainsource]

    def addsource(self, uri, namespace, tshclass=None):
        self.sources.append(
            source(uri, namespace, tshclass or self.mainsource.tsh.__class__)
        )

    def _findsourcefor(self, name):
        for source in self.sources:
            with source.engine.begin() as cn:
                if source.tsh.exists(cn, name):
                    return source

    def _findwritesourcefor(self, name):
        for source in self.sources[1:]:
            with source.engine.begin() as cn:
                if source.tsh.exists(cn, name):
                    return None
        return self.mainsource

    def exists(self, name):
        for source in self.sources:
            with source.engine.begin() as cn:
                if source.tsh.exists(cn, name):
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
        for key in self.tsh.metakeys:
            meta.pop(key, None)
        return meta

    def update(self,
               name: str,
               updatets: pd.Series,
               author: str,
               metadata: Optional[dict]=None,
               insertion_date: Optional[datetime]=None,
               **kw) -> Optional[pd.Series]:
        source = self._findwritesourcefor(name)
        if source:
            return source.tsh.update(
                source.engine,
                updatets,
                name,
                author,
                metadata=metadata,
                insertion_date=insertion_date,
                **kw
            )

        raise ValueError(
            'not allowed to update to a secondary source'
        )

    def replace(self,
                name: str,
                newts: pd.Series,
                author: str,
                metadata: Optional[dict]=None,
                insertion_date: Optional[datetime]=None,
                **kw) -> Optional[pd.Series]:
        source = self._findwritesourcefor(name)
        if source:
            return source.tsh.replace(
                source.engine,
                newts,
                name,
                author,
                metadata=metadata,
                insertion_date=insertion_date,
                **kw
            )

        raise ValueError(
            'not allowed to replace to a secondary source'
        )

    def update_metadata(self,
                        name: str,
                        metadata: dict):
        source = self._findwritesourcefor(name)
        if source:
            return source.tsh.update_metadata(
                source.engine,
                name,
                metadata
            )

        raise ValueError(
            'not allowed to update metadata to a secondary source'
        )

    def rename(self,
               currname: str,
               newname: str):
        source = self._findwritesourcefor(currname)
        if source:
            return self.tsh.rename(source.engine, currname, newname)

        raise ValueError(
            'not allowed to rename to a secondary source'
        )

    def delete(self, name: str):
        source = self._findwritesourcefor(name)
        if source:
            return self.tsh.delete(source.engine, name)

        raise ValueError(
            'not allowed to delete to a secondary source'
        )
