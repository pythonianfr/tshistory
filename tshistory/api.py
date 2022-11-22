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

from tshistory.util import (
    ensuretz,
    find_most_specific_tshclass,
    find_most_specific_http_client,
    threadpool
)
from tshistory.tsio import timeseries as tshclass


NONETYPE = type(None)


class timeseries:

    def __new__(cls, uri,
                namespace='tsh',
                handler=None,
                sources=(),
                clientclass=None):
        parseduri = urlparse(uri)
        if parseduri.scheme.startswith('postgres'):
            if handler is None:
                handler = find_most_specific_tshclass()
            return mainsource(
                uri,
                namespace,
                tshclass=handler,
                othersources=altsources(handler, sources)
            )
        elif parseduri.scheme.startswith('http'):
            if clientclass is None:
                clientclass = find_most_specific_http_client()
            return clientclass(uri)

        raise NotImplementedError(uri)


class mainsource:
    """API façade for the main source (talks directly to the storage)

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

    def _instancename(self):
        parsed = urlparse(self.uri)
        if self.tsh.namespace == 'tsh':
            return f'{parsed.path[1:]}'
        else:
            return f'{parsed.path[1:]}@{self.tsh.namespace}'

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
            _keep_nans: bool=False,
            **kw) -> Optional[pd.Series]:
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
            _keep_nans=_keep_nans,
            **kw
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
                        to_insertion_date: Optional[datetime]=None,
                        from_value_date: Optional[datetime]=None,
                        to_value_date: Optional[datetime]=None,
                        **kw):
        """Get the list of all insertion dates.

        """
        from_insertion_date = ensuretz(from_insertion_date)
        to_insertion_date = ensuretz(to_insertion_date)

        if self.tsh.exists(self.engine, name):
            return self.tsh.insertion_dates(
                self.engine,
                name,
                from_insertion_date=from_insertion_date,
                to_insertion_date=to_insertion_date,
                from_value_date=from_value_date,
                to_value_date=to_value_date,
                **kw
            )

        return self.othersources.insertion_dates(
            name,
            from_insertion_date,
            to_insertion_date,
            from_value_date,
            to_value_date,
            **kw
        )

    def history(self,
                name: str,
                from_insertion_date: Optional[datetime]=None,
                to_insertion_date: Optional[datetime]=None,
                from_value_date: Optional[datetime]=None,
                to_value_date: Optional[datetime]=None,
                diffmode: bool=False,
                _keep_nans: bool=False,
                **kw) -> Dict[datetime, pd.Series]:
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
            _keep_nans=_keep_nans,
            **kw
        )

        if hist is None:
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

    def block_staircase(
        self,
        name,
        from_value_date: Optional[datetime] = None,
        to_value_date: Optional[datetime] = None,
        revision_freq: Dict[str, int] = None,
        revision_time: Dict[str, int] = None,
        revision_tz: str = 'UTC',
        maturity_offset: Dict[str, int] = None,
        maturity_time: Dict[str, int] = None,
    ):
        """Staircase series by block

        Computes a series rebuilt from successive blocks of history, each linked to a
        distinct revision date. The revision dates are taken at regular time intervals
        determined by `revision_freq`, `revision_time` and `revision_tz`. The time lag
        between revision dates and value dates of each block is determined by
        `maturity_offset` and `maturity_time`.

        name: str unique identifier of the series
        from_value_date: pandas.Timestamp from which values are retrieved
        to_value_date: pandas.Timestamp to which values are retrieved
        revision_freq: dict giving revision frequency, of which keys must be taken from
            ['years', 'months', 'weeks', 'bdays', 'days', 'hours', 'minutes', 'seconds']
            and values as integers. Default is daily frequency, i.e. {'days': 1}
        revision_time: dict giving revision time, of which keys should be taken from
            ['year', 'month', 'day', 'weekday', 'hour', 'minute', 'second'] and values
            must be integers. It is only used for revision date initialisation. The next
            revision dates are then obtained by successively adding `revision_freq`.
            Default is {'hour': 0}
        revision_tz: str giving time zone in which revision date and time are expressed.
            Default is 'UTC'
        maturity_offset: dict giving time lag between each revision date and start time
            of related block values. Its keys must be taken from ['years', 'months',
            'weeks', 'bdays', 'days', 'hours', 'minutes', 'seconds'] and values as
            integers. Default is {}, i.e. the revision date is the block start date
        maturity_time: dict fixing start time of each block, of which keys should be
            taken from ['year', 'month', 'day', 'hour', 'minute', 'second'] and values
            must be integers. The start date of each block is thus obtained by adding
            `maturity_offset` to revision date and then applying `maturity_time`.
            Default is {}, i.e. block start date is just the revision date shifted by
            `maturity_offset`
        """
        bsc = self.tsh.block_staircase(
            self.engine,
            name,
            from_value_date=from_value_date,
            to_value_date=to_value_date,
            revision_freq=revision_freq,
            revision_time=revision_time,
            revision_tz=revision_tz,
            maturity_offset=maturity_offset,
            maturity_time=maturity_time,
        )
        if bsc is None:
            bsc = self.othersources.block_staircase(
                name,
                from_value_date=from_value_date,
                to_value_date=to_value_date,
                revision_freq=revision_freq,
                revision_time=revision_time,
                revision_tz=revision_tz,
                maturity_offset=maturity_offset,
                maturity_time=maturity_time,
            )
        return bsc


    def catalog(self, allsources: bool=True) -> Dict[str,List[Tuple[str,str]]]:
        """Produces a catalog of all series in the form of a mapping from
        source to a list of (name, kind) pair.

        By default it provides the series from all sources.

        If `allsources` is False, only the main source is listed.

        """
        instancename = self._instancename()
        cat = defaultdict(list)
        for name, kind in self.tsh.list_series(self.engine).items():
            cat[(instancename, self.namespace)].append((name, kind))
        if allsources:
            for key, val in self.othersources.catalog(False).items():
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
                        metadata: dict) -> NONETYPE:
        """Update a series metadata with a dictionary from strings to anything
        json-serializable.

        Internal keys are not allowed and any attempt to update them
        will raise.

        """
        with self.engine.begin() as cn:
            if self.tsh.exists(cn, name):
                return self.tsh.update_metadata(cn, name, metadata)

        self.othersources.forbidden(
            name,
            'not allowed to update metadata to a secondary source'
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
               newname: str) -> NONETYPE:
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
        with self.engine.begin() as cn:
            if self.tsh.exists(cn, name):
                return self.tsh.delete(cn, name)

        # give a chance to say *no*
        self.othersources.forbidden(
            name,
            'not allowed to delete to a secondary source'
        )

    def strip(self, name: str, insertion_date: datetime) -> NONETYPE:
        """Remove revisions after a specific insertion date.

        This is an irreversible operation.

        """

        insertion_date = ensuretz(insertion_date)
        if not self.tsh.exists(self.engine, name):
            raise Exception(f'no series {name} exists')

        with self.engine.begin() as cn:
            csid = self.tsh.changeset_at(
                cn, name, insertion_date, 'after'
            )
            if csid is not None:
                return self.tsh.strip(cn, name, csid)

    # groups

    def group_exists(self, name: str) -> bool:
        """Checks the existence of a group with a given name.

        """
        with self.engine.begin() as cn:
            return self.tsh.group_exists(cn, name)

    def group_type(self, name: str) -> str:
        """Return the type of a group, for instance 'primary', 'formula' or
        'bound'

        """
        with self.engine.begin() as cn:
            return self.tsh.group_type(cn, name)

    def group_rename(self, oldname: str, newname: str) -> NONETYPE:
        """Rename a group.

        The target name must be available.

        """
        with self.engine.begin() as cn:
            self.tsh.group_rename(cn, oldname, newname)

    def group_get(self,
                  name: str,
                  revision_date: Optional[pd.Timestamp]=None,
                  from_value_date: Optional[pd.Timestamp]=None,
                  to_value_date=None
    ) -> Optional[pd.DataFrame]:
        """Get a group by name.

        By default one gets the latest version.

        By specifying `revision_date` one can get the closest version
        matching the given date.

        The `from_value_date` and `to_value_date` parameters permit to
        specify a narrower date range (by default all points are
        provided).

        If the group does not exists, a None is returned.

        """
        with self.engine.begin() as cn:
            return self.tsh.group_get(
                cn,
                name,
                revision_date,
                from_value_date,
                to_value_date
            )

    def group_insertion_dates(self,
                              name: str,
                              from_insertion_date: Optional[pd.Timestamp]=None,
                              to_insertion_date:Optional[pd.Timestamp]=None
    ) -> List[pd.Timestamp]:
        """Get the list of all insertion dates for any given group
        """
        with self.engine.begin() as cn:
            return self.tsh.group_insertion_dates(
                cn,
                name,
                from_insertion_date=from_insertion_date,
                to_insertion_date=to_insertion_date
            )

    def group_history(self,
                      name: str,
                      from_value_date: Optional[pd.Timestamp]=None,
                      to_value_date: Optional[pd.Timestamp]=None,
                      from_insertion_date: Optional[pd.Timestamp]=None,
                      to_insertion_date: Optional[pd.Timestamp]=None,
    ) -> Dict[pd.Timestamp, pd.DataFrame]:
        """Get all versions of a group in the form of a dict from insertion
              dates to dataframe.

              It is possible to restrict the versions range by specifying
              `from_insertion_date` and `to_insertion_date`.

              It is possible to restrict the values range by specifying
              `from_value_date` and `to_value_date`.
        """
        with self.engine.begin() as cn:
            return self.tsh.group_history(
                cn,
                name,
                from_value_date=from_value_date,
                to_value_date=to_value_date,
                from_insertion_date=from_insertion_date,
                to_insertion_date=to_insertion_date,
            )

    def group_replace(self,
                      name: str,
                      df: pd.DataFrame,
                      author: str,
                      insertion_date: Optional[pd.Timestamp]=None) -> NONETYPE:
        """Replace a group named by <name> with the input dataframe.

        This creates a new version of the group. The group is completely
        replaced with the provided values.

        The `author` is mandatory.
        The `metadata` dictionary allows to associate any metadata
        with the new group revision.

        It is possible to force an `insertion_date`, which can only be
        higher than the previous `insertion_date`.

        """
        with self.engine.begin() as cn:
            self.tsh.group_replace(
                cn,
                df,
                name,
                author,
                insertion_date
            )

    def group_delete(self, name: str) -> NONETYPE:
        """Delete a group.

        This is an irreversible operation.

        """
        with self.engine.begin() as cn:
            self.tsh.group_delete(cn, name)

    def group_metadata(self,
                       name: str,
                       all: bool=False) -> Dict[str, Any]:
        """Return a group metadata dictionary.

        """
        meta = self.tsh.group_metadata(self.engine, name)
        if meta is None:
            return
        if all:
            return meta
        for key in self.tsh.metakeys:
            meta.pop(key, None)
        return meta

    def update_group_metadata(self, name: str, meta: Dict[str, Any]) -> NONETYPE:
        """Update a group metadata with a dictionary from strings to anything
        json-serializable.

        """
        with self.engine.begin() as cn:
            self.tsh.update_group_metadata(cn, name, meta)

    def group_catalog(self) -> Dict[str,List[Tuple[str,str]]]:
        """Produces a catalog of all groups in the form of a mapping from
        source to a list of (name, kind) pair.

        """
        instancename = self._instancename()
        cat = defaultdict(list)
        for name, kind in self.tsh.list_groups(self.engine).items():
            cat[(instancename, self.namespace)].append((name, kind))
        return cat


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
                        to_insertion_date: Optional[datetime]=None,
                        from_value_date: Optional[datetime]=None,
                        to_value_date: Optional[datetime]=None,
                        **kw) -> List[pd.Timestamp]:
        source = self._findsourcefor(name)
        if source is None:
            # let's be nice in all cases
            return []
        return source.tsa.insertion_dates(
            name,
            from_insertion_date=from_insertion_date,
            to_insertion_date=to_insertion_date,
            from_value_date=from_value_date,
            to_value_date=to_value_date,
            **kw
        )

    def forbidden(self, name, msg):
        if self.exists(name):
            raise ValueError(msg)

    def catalog(self, allsources=False):
        cats = []
        pool = threadpool(len(self.sources))
        def getcat(source):
            try:
                cats.append(
                    source.tsa.catalog(allsources)
                )
            except:
                import traceback as tb; tb.print_exc()
                print(f'source {source} temporarily unavailable')

        pool(getcat, [(s,) for s in self.sources])
        cat = {}
        for c in cats:
            cat.update(c)
        return cat
