from datetime import datetime, timedelta
from urllib.parse import urlparse
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union
)
from collections import defaultdict

from sqlalchemy import create_engine
import pandas as pd

from tshistory.util import ensuretz
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
    """Postgres-backed implementation of the API

    The api documentation is carried by this object.
    The http client provides exactly the same methods.

    """
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
        self.engine = create_engine(uri, max_overflow=100)
        self.tsh = tshclass(namespace, othersources)
        self.othersources = othersources

    def update(self,
               name: str,
               updatets: pd.Series,
               author: str,
               metadata: Optional[dict]=None,
               insertion_date: Optional[datetime]=None,
               **kw) -> Optional[pd.Series]:
        """Update a series named by <name> with the input pandas series.

        This creates a new version of the series. Only the _changes_
        between the last version and the provided series are part of
        the new version.

        A series made of the changed points is returned.  If there was
        no change, None is returned and no new version is created.

        New points are added, changed points are changed,
        points with NaN are considered to be _erased_.

        The `author` is mandatory.
        The `metadata` dictionary allows to associate any metadata
        with the new series revision.

        It is possible to force an `insertion_date`, which can only be
        higher than the previous `insertion_date`.

        """
        insertion_date = ensuretz(insertion_date)

        # check local existence
        if not self.tsh.exists(self.engine, name):
            # give a chance to say *no*
            self.othersources.forbidden(
                name,
                'not allowed to update to a secondary source'
            )

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
                replacets: pd.Series,
                author: str,
                metadata: Optional[dict]=None,
                insertion_date: Optional[datetime]=None,
                **kw) -> Optional[pd.Series]:
        """Replace a series named by <name> with the input pandas series.

        This creates a new version of the series. The series is completely
        replaced with the provided values.

        The `author` is mandatory.
        The `metadata` dictionary allows to associate any metadata
        with the new series revision.

        It is possible to force an `insertion_date`, which can only be
        higher than the previous `insertion_date`.

        """
        insertion_date = ensuretz(insertion_date)

        # check local existence
        if not self.tsh.exists(self.engine, name):
            # give a chance to say *no*
            self.othersources.forbidden(
                name,
                'not allowed to replace to a secondary source'
            )

        return self.tsh.replace(
            self.engine,
            replacets,
            name,
            author,
            metadata=metadata,
            insertion_date=insertion_date,
            **kw
        )

    def exists(self, name: str) -> bool:
        """Checks the existence of a series with a given name.

        """
        if (not self.tsh.exists(self.engine, name) and
            not self.othersources.exists(name)):
            return False

        return True

    def get(self, name: str,
            revision_date: Optional[datetime]=None,
            from_value_date: Optional[datetime]=None,
            to_value_date: Optional[datetime]=None,
            _keep_nans: bool=False) -> Optional[pd.Series]:
        """Get a series by name.

        By default one gets the latest version.

        By specifying `revision_date` one can get the closest version
        matching the given date.

        The `from_value_date` and `to_value_date` parameters permit to
        specify a narrower date range (by default all points are
        provided).

        If the series does not exists, a None is returned.

        """
        revision_date = ensuretz(revision_date)

        ts = self.tsh.get(
            self.engine,
            name,
            revision_date=revision_date,
            from_value_date=from_value_date,
            to_value_date=to_value_date,
            _keep_nans=_keep_nans
        )

        if ts is None:
            ts = self.othersources.get(
                name,
                revision_date=revision_date,
                from_value_date=from_value_date,
                to_value_date=to_value_date,
                _keep_nans=_keep_nans
            )
        return ts

    def insertion_dates(self,
                        name: str,
                        from_insertion_date: Optional[datetime]=None,
                        to_insertion_date: Optional[datetime]=None):
        """Get the list of all insertion dates.

        """
        from_insertion_date = ensuretz(from_insertion_date)
        to_insertion_date = ensuretz(to_insertion_date)

        if self.tsh.exists(self.engine, name):
            return self.tsh.insertion_dates(
                self.engine,
                name,
                from_insertion_date,
                to_insertion_date
            )

        return self.othersources.insertion_dates(
            name,
            from_insertion_date,
            to_insertion_date
        )

    def history(self,
                name: str,
                from_insertion_date: Optional[datetime]=None,
                to_insertion_date: Optional[datetime]=None,
                from_value_date: Optional[datetime]=None,
                to_value_date: Optional[datetime]=None,
                diffmode: bool=False,
                _keep_nans: bool=False) -> Dict[datetime, pd.Series]:
        """Get all versions of a series in the form of a dict from insertion
        dates to series version.

        It is possible to restrict the versions range by specifying
        `from_insertion_date` and `to_insertion_date`.

        It is possible to restrict the values range by specifying
        `from_value_date` and `to_value_date`.

        If `diffmode` is set to True, we don't get full series values
        between two consecutive insertion date but only the difference
        series (with new points, updated points and deleted
        points). This is typically more costly to compute but can be
        much more compact, and it encodes the same information as with
        `diffmode` set to False.

        """
        from_insertion_date = ensuretz(from_insertion_date)
        to_insertion_date = ensuretz(to_insertion_date)

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
        """Compute a series whose value dates are the most recent
        constrained to be `delta` time after the insertion dates of
        the series.

        This kind of query typically makes sense for forecast series
        where the relationship between insertion date and value date
        is sound.

        """

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

    def catalog(self, allsources: bool=True) -> Dict[str,List[Tuple[str,str]]]:
        """Produces a catalog of all series in the form of a mapping from
        source to a list of (name, kind) pair.

        By default it provides the series from all sources.

        If `allsources` is False, only the main source is listed.

        """
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
        """Return a pandas interval object which provides the smallest and
        highest value date of a series.

        """
        try:
            ival = self.tsh.interval(self.engine, name)
        except ValueError:
            return self.othersources.interval(name)
        return ival

    def metadata(self,
                 name: str,
                 all: bool=False) -> Dict[str, Any]:
        """Return a series metadata dictionary.

        If `all` is True, internal metadata will be provided.

        """
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
                        metadata: dict) -> None:
        """Update a series metadata with a dictionary from strings to anything
        json-serializable.

        Internal keys are not allowed and any attempt to update them
        will raise.

        """
        # give a chance to say *no*
        self.othersources.forbidden(
            name,
            'not allowed to update metadata to a secondary source'
        )

        self.tsh.update_metadata(
            self.engine,
            name,
            metadata
        )

    def type(self, name: str) -> str:
        """Return the type of a series, for instance 'primary' or 'formula'.

        """
        if self.tsh.exists(self.engine, name):
            return self.tsh.type(self.engine, name)

        return self.othersources.type(name)

    def log(self,
            name: str,
            limit: Optional[int]=None,
            fromdate: Optional[pd.Timestamp]=None,
            todate: Optional[pd.Timestamp]=None) -> List[Dict[str, Any]]:
        """Return a list of revisions for a given series, in reverse
        chronological order, with filters.

        Revisions are dicts of:
        * rev: revision id (int)
        * author: author name
        * date: timestamp of the revision
        * meta: the revision metadata

        """

        return self.tsh.log(
            self.engine,
            name,
            limit=limit,
            fromdate=fromdate,
            todate=todate
        )

    def rename(self,
               currname: str,
               newname: str) -> None:
        """Rename a series.

        The target name must be available.

        """
        # give a chance to say *no*
        self.othersources.forbidden(
            currname,
            'not allowed to rename to a secondary source'
        )

        return self.tsh.rename(self.engine, currname, newname)

    def delete(self, name: str):
        """Delete a series.

        This is an irreversible operation.

        """
        # give a chance to say *no*
        self.othersources.forbidden(
            name,
            'not allowed to delete to a secondary source'
        )

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
            try:
                if source.tsa.exists(name):
                    return source
            except:
                print(f'source {source} currently unavailable')

    def exists(self, name):
        for source in self.sources:
            try:
                if source.tsa.exists(name):
                    return True
            except:
                print(f'source {source} currently unavailable')
        return False

    def get(self, name: str,
            revision_date: Optional[datetime]=None,
            from_value_date: Optional[datetime]=None,
            to_value_date: Optional[datetime]=None,
            _keep_nans: bool=False) -> Optional[pd.Series]:
        source = self._findsourcefor(name)
        if source is None:
            return
        return source.tsa.get(
            name,
            revision_date=revision_date,
            from_value_date=from_value_date,
            to_value_date=to_value_date,
            _keep_nans=_keep_nans
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

    def type(self, name: str):
        source = self._findsourcefor(name)
        if source is None:
            return

        return source.tsa.type(name)

    def interval(self, name: str) -> pd.Interval:
        source = self._findsourcefor(name)
        if source is None:
            raise ValueError(f'no interval for series: {name}')
        ival = source.tsa.interval(name)
        if ival is None:
            raise ValueError(f'no interval for series: {name}')
        return ival

    def insertion_dates(self,
                        name: str,
                        from_insertion_date: Optional[datetime]=None,
                        to_insertion_date: Optional[datetime]=None):
        source = self._findsourcefor(name)
        if source is None:
            # let's be nice in all cases
            return []
        return source.tsa.insertion_dates(name)

    def forbidden(self, name, msg):
        if self.exists(name):
            raise ValueError(msg)

    def catalog(self, allsources=False):
        cat = {}
        for source in self.sources:
            try:
                cat.update(source.tsa.catalog(allsources))
            except:
                import traceback as tb; tb.print_exc()
                print(f'source {source} temporarily unavailable')
        return cat
