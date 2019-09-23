from datetime import datetime
from typing import (
    Dict,
    Optional,
    Union
)
from sqlalchemy import create_engine
import pandas as pd

from tshistory.tsio import timeseries as dbtimeseries


class timeseries:
    __slots__ = (
        'uri', 'namespace',
        'engine', 'tsh'
    )

    def __init__(self,
                 uri: str,
                 namespace: str='tsh'):
        self.uri = uri
        self.namespace = namespace
        self.engine = create_engine(uri)
        self.tsh = dbtimeseries(namespace)

    def update(self,
               updatets: pd.Series,
               name: str,
               author: str,
               metadata: Optional[dict]=None,
               insertion_date: Optional[datetime]=None) -> Optional[pd.Series]:
        with self.engine.begin() as cn:
            return self.tsh.update(
                cn,
                updatets,
                name,
                author,
                metadata=metadata,
                insertion_date=insertion_date
            )

    def replace(self,
                updatets: pd.Series,
                name: str,
                author: str,
                metadata: Optional[dict]=None,
                insertion_date: Optional[datetime]=None) -> Optional[pd.Series]:
        with self.engine.begin() as cn:
            return self.tsh.replace(
                cn,
                updatets,
                name,
                author,
                metadata=metadata,
                insertion_date=insertion_date
            )

    def exists(self, name):
        with self.engine.begin() as cn:
            return self.tsh.exists(cn, name)

    def get(self, name: str,
            revision_date: Optional[datetime]=None,
            from_value_date: Optional[datetime]=None,
            to_value_date: Optional[datetime]=None) -> Optional[pd.Series]:
        with self.engine.begin() as cn:
            return self.tsh.get(
                cn,
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
        with self.engine.begin() as cn:
            return self.tsh.history(
                cn,
                name,
                from_insertion_date=from_insertion_date,
                to_insertion_date=to_insertion_date,
                from_value_date=from_value_date,
                to_value_date=to_value_date,
                diffmode=diffmode
            )

    def catalog(self):
        with self.engine.begin() as cn:
            return self.tsh.list_series(cn)

    def interval(self, name: str) -> pd.Interval:
        with self.engine.begin() as cn:
            return self.tsh.interval(cn, name)

    def metadata(self,
                 name: str,
                 all: bool=False):
        with self.engine.begin() as cn:
            meta = self.tsh.metadata(cn, name)
            if all:
                return meta
            for key in self.tsh.metakeys:
                meta.pop(key, None)
            return meta

    def update_metadata(self,
                        name: str,
                        metadata: dict):
        with self.engine.begin() as cn:
            self.tsh.update_metadata(
                cn,
                name,
                metadata
            )

    def type(self, name: str):
        with self.engine.begin() as cn:
            return self.tsh.type(cn, name)



class source:
    __slots__ = ('engine', 'tsh', 'uri', 'namespace')

    def __init__(self, uri, namespace):
        self.uri = uri
        self.namespace = namespace
        self.engine = create_engine(uri)
        self.tsh = dbtimeseries(namespace)


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
                 namespace: str='tsh'):
        self.uri = uri
        self.namespace = namespace
        self.mainsource = source(uri, namespace)
        self.sources = [self.mainsource]

    def addsource(self, uri, namespace):
        self.sources.append(
            source(uri, namespace)
        )

    def _findsourcefor(self, name):
        for source in self.sources:
            with source.engine.begin() as cn:
                if source.tsh.exists(cn, name):
                    return source

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
        with source.engine.begin() as cn:
            return source.tsh.get(
                cn,
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
        with source.engine.begin() as cn:
            return source.tsh.history(
                cn,
                name,
                from_insertion_date=from_insertion_date,
                to_insertion_date=to_insertion_date,
                from_value_date=from_value_date,
                to_value_date=to_value_date,
                diffmode=diffmode
            )

    def update(self,
               updatets: pd.Series,
               name: str,
               author: str,
               metadata: Optional[dict]=None,
               insertion_date: Optional[datetime]=None) -> Optional[pd.Series]:
        source = self._findsourcefor(name)
        if source is None or source == self.sources[0]:
            # creation or main source update
            with source.engine.begin() as cn:
                return source.tsh.update(
                    cn,
                    updatets,
                    name,
                    author,
                    metadata=metadata,
                    insertion_date=insertion_date
                )

        raise ValueError(
            'not allowed to update to a secondary source '
            f'{source.uri} {source.namespace}'
        )


    def replace(self,
                newts: pd.Series,
                name: str,
                author: str,
                metadata: Optional[dict]=None,
                insertion_date: Optional[datetime]=None) -> Optional[pd.Series]:
        source = self._findsourcefor(name)
        if source is None or source == self.sources[0]:
            # creation or main source update
            with source.engine.begin() as cn:
                return source.tsh.replace(
                    cn,
                    newts,
                    name,
                    author,
                    metadata=metadata,
                    insertion_date=insertion_date
                )

        raise ValueError(
            'not allowed to replace to a secondary source '
            f'{source.uri} {source.namespace}'
        )
