from datetime import datetime
import logging
import hashlib
import uuid
import json
from pathlib import Path

import pandas as pd
import numpy as np

from sqlhelp import sqlfile, select, insert

from tshistory import search
from tshistory.util import (
    closed_overlaps,
    compatible_date,
    diff,
    empty_series,
    ensuretz,
    num2float,
    patch,
    pruned_history,
    series_metadata,
    start_end,
    ts,
    tx
)
from tshistory.storage import Postgres


L = logging.getLogger('tshistory.tsio')
SERIESSCHEMA = Path(__file__).parent / 'series.sql'


class timeseries:
    index = 0
    namespace = 'tsh'
    schema = None
    metakeys = {
        'tablename',
        'tzaware',
        'index_type',
        'index_dtype',
        'value_dtype',
        'value_type'
    }
    create_lock_id = None
    delete_lock_id = None
    storageclass = Postgres

    def __init__(self, namespace='tsh', othersources=None,
                 _groups=True):
        self.namespace = namespace
        self.create_lock_id = sum(ord(c) for c in namespace)
        self.delete_lock_id = sum(ord(c) for c in namespace)
        self.othersources = othersources
        # primary series for groups in a simple timeseries store
        # in its own namespace
        if _groups:
            self.tsh_group = timeseries(
                namespace=f'{self.namespace}.group',
                _groups=False
            )

    def __repr__(self):
        return (
            f'tsio.timeseries({self.namespace},othersources={self.othersources})'
        )

    @tx
    def update(self, cn, updatets, name, author,
               metadata=None,
               insertion_date=None,
               **k):
        """Create a new revision of a given time series
        with update semantics:
        * new points will be added to the existing series
        * existing points will be updated when value differs
        * nans will be interpreted as point erasure

        updatets: pandas.Series with date index
        name: str unique identifier of the serie
        author: str free-form author name
        metadata: optional dict for changeset metadata
        """
        assert isinstance(name, str), 'Name is not a string'
        name = name.strip()
        tablename = self._series_to_tablename(cn, name)
        if not len(updatets):
            if tablename is None:
                return self._guard_insert(
                    updatets, name, author, metadata,
                    insertion_date
                )
            # known series: we can do a bit better
            return empty_series(
                self.tzaware(cn, name),
                dtype=updatets.dtype
            )

        updatets = self._guard_insert(
            updatets, name, author, metadata,
            insertion_date
        )
        updatets.name = name
        assert ('<M8[ns]' == updatets.index.dtype or
                'datetime' in str(updatets.index.dtype) and not
                isinstance(updatets.index, pd.MultiIndex))

        if tablename is None:
            seriesmeta = self._series_initial_meta(cn, name, updatets)
            return self._create(cn, updatets, name, author, seriesmeta,
                                metadata, insertion_date)

        return self._update(cn, updatets, name, author,
                            metadata, insertion_date)

    @tx
    def replace(self, cn, newts, name, author,
                metadata=None,
                insertion_date=None,
                **k):
        """Create a new revision of a given time series
        and do a wholesale replacement of the series
        with the provided one (no update semantics)

        newts: pandas.Series with date index
        name: str unique identifier of the serie
        author: str free-form author name
        metadata: optional dict for changeset metadata
        """
        # nans have no replacement semantics -> drop them
        assert isinstance(name, str), 'Name is not a string'
        name = name.strip()
        newts = newts.dropna()
        newts = self._guard_insert(
            newts, name, author, metadata,
            insertion_date
        )
        if not len(newts):
            return empty_series(
                newts.index.tz is not None,
                dtype=newts.dtype
            )

        assert ('<M8[ns]' == newts.index.dtype or
                'datetime' in str(newts.index.dtype) and not
                isinstance(newts.index, pd.MultiIndex))

        newts.name = name
        tablename = self._series_to_tablename(cn, name)

        if tablename is None:
            seriesmeta = self._series_initial_meta(cn, name, newts)
            return self._create(cn, newts, name, author, seriesmeta,
                                metadata, insertion_date)

        self._validate(cn, newts, name)

        # check that we don't insert a duplicate of current value
        current = self.get(cn, name)
        if current.equals(newts):
            L.info('no difference in %s by %s (for ts of size %s)',
                   name, author, len(newts))
            return

        # compute series start/end stamps
        start, end = start_end(newts)
        head = self.storageclass(cn, self, name).create(newts)
        self._new_revision(
            cn, name, head, start, end,
            author, insertion_date, metadata
        )
        L.info('inserted series (size=%s) for ts %s by %s',
               len(newts), name, author)
        return newts


    def list_series(self, cn):
        """Return the mapping of all series to their type"""
        sql = f'select name from "{self.namespace}".registry '
        return {
            row.name: 'primary'
            for row in cn.execute(sql)
        }

    def tzaware(self, cn, name):
        return cn.execute(
            'select internal_metadata->\'tzaware\' '
            f'from "{self.namespace}".registry '
            'where name = %(name)s',
            name=name
        ).scalar()

    @tx
    def get(self, cn, name, revision_date=None,
            from_value_date=None, to_value_date=None,
            _keep_nans=False,
            **kw):
        """Compute and return the serie of a given name

        revision_date: datetime filter to get previous versions of the
        serie

        """
        if not self.exists(cn, name):
            return

        self._guard_query_dates(
            revision_date, from_value_date, to_value_date
        )

        csetfilter = []
        if revision_date:
            csetfilter.append(
                lambda q: q.where(
                    'insertion_date <= %(idate)s', idate=revision_date
                )
            )

        meta = self.internal_metadata(cn, name)
        # munge query to satisfy pandas idiocy
        if from_value_date or to_value_date:
            tzaware = meta['tzaware']
            if from_value_date:
                from_value_date = compatible_date(tzaware, from_value_date)
            if to_value_date:
                to_value_date = compatible_date(tzaware, to_value_date)

        snap = self.storageclass(cn, self, name)
        try:
            _, current = snap.find(csetfilter=csetfilter,
                                   from_value_date=from_value_date,
                                   to_value_date=to_value_date)
        except ValueError as err:
            raise ValueError(
                f'name: "{name}", revdate: {revision_date} '
                f'(from "{err}")'
            )

        if current is None:
            return empty_series(
                meta['tzaware'],
                dtype=meta['value_type'],
                name=name
            )

        if not _keep_nans:
            current = current.dropna()
        current.name = name
        return current

    @tx
    def internal_metadata(self, cn, name):
        if name in cn.cache['internal_metadata']:
            return cn.cache['internal_metadata'][name]
        meta = cn.cache['internal_metadata'][name] = cn.execute(
            f'select internal_metadata '
            f'from "{self.namespace}".registry '
            f'where name = %(name)s',
            name=name
        ).scalar()
        return meta

    @tx
    def update_internal_metadata(self, cn, name, metadata):
        imeta = self.internal_metadata(cn, name) or {}
        imeta.update(metadata)
        cn.execute(
            f'update "{self.namespace}".registry '
            'set internal_metadata = %(metadata)s '
            'where name = %(name)s',
            metadata=json.dumps(imeta),
            name=name
        )

    @tx
    def metadata(self, cn, name):
        return cn.execute(
            f'select metadata from "{self.namespace}".registry '
            'where name = %(name)s',
            name=name
        ).scalar()

    @tx
    def update_metadata(self, cn, name, metadata):
        assert isinstance(metadata, dict)
        existing_metadata = self.metadata(cn, name) or {}

        existing_metadata.update(metadata)
        cn.execute(
            f'update "{self.namespace}".registry '
            'set metadata = %(metadata)s '
            'where registry.name = %(name)s',
            metadata=json.dumps(existing_metadata),
            name=name
        )

    @tx
    def replace_metadata(self, cn, name, metadata):
        assert isinstance(metadata, dict)
        cn.execute(
            f'update "{self.namespace}".registry '
            'set metadata = %(metadata)s '
            'where registry.name = %(name)s',
            metadata=json.dumps(metadata),
            name=name
        )

    def changeset_metadata(self, cn, csid):
        assert isinstance(csid, int)
        q = select(
            'metadata'
        ).table(
            f'"{self.namespace}".changeset'
        ).where(
            'id = %(csid)s', csid=csid
        )
        return q.do(cn).scalar()

    def type(self, cn, name):
        return 'primary'

    @tx
    def history(self, cn, name,
                from_insertion_date=None,
                to_insertion_date=None,
                from_value_date=None,
                to_value_date=None,
                diffmode=False,
                _keep_nans=False,
                **kw):
        tablename = self._series_to_tablename(cn, name)
        if tablename is None:
            return

        self._guard_query_dates(
            from_insertion_date, to_insertion_date,
            from_value_date, to_value_date
        )
        revs = self._revisions(
            cn, name,
            from_insertion_date,
            to_insertion_date,
            from_value_date,
            to_value_date
        )

        if not revs:
            return {}

        if diffmode:
            # compute the previous serie value
            first_csid = revs[0][0]
            previous_csid = self._previous_cset(cn, name, first_csid)
            revs.insert(0, (previous_csid, None))

        snapshot = self.storageclass(cn, self, name)

        # careful there with naive series vs inputs
        if from_value_date or to_value_date:
            tzaware = self.tzaware(cn, name)
            if from_value_date:
                from_value_date = compatible_date(tzaware, from_value_date)
            if to_value_date:
                to_value_date = compatible_date(tzaware, to_value_date)

        series = snapshot.findall(
            revs,
            from_value_date,
            to_value_date
        )

        if diffmode:
            diffs = []
            for (_revdate_a, serie_a), (revdate_b, serie_b) in zip(series, series[1:]):
                if serie_a is None:
                    # when we scan the entirety of the history: there exists no "previous" serie
                    # we therefore consider the first serie as a diff to the "null" serie
                    diffs.append((revdate_b, serie_b))
                else:
                    series_diff = diff(serie_a, serie_b)
                    if len(series_diff):
                        diffs.append((revdate_b, series_diff))
            series = diffs
        else:
            series = [
                (idate, ts if _keep_nans else ts.dropna())
                for idate, ts in series
            ]

        hist = {
            idate: ts
            for idate, ts in series if len(series)
        }

        if from_value_date or to_value_date:
            # now it's possible that the extremities cut
            # yields similar series for successive idates
            # and we are not interested in that
            hist = pruned_history(hist)

        return hist

    @tx
    def staircase(self, cn, name, delta,
                  from_value_date=None,
                  to_value_date=None):
        """ compute a series whose value dates are bounded to be
        `delta` time after the insertion dates and where we
        keep the most recent ones
        """
        if not self.exists(cn, name):
            return

        self._guard_query_dates(
            from_value_date, to_value_date
        )
        base = self.get(
            cn, name,
            from_value_date=from_value_date,
            to_value_date=to_value_date,
            _keep_nans=True
        )
        tzaware = self.tzaware(cn, name)
        if not len(base):
            return empty_series(tzaware, name=name)

        chunks = []
        for vdate in base.index:
            idate = ensuretz(vdate)
            ts = self.get(
                cn,
                name,
                revision_date=idate - delta,
                from_value_date=vdate,
                to_value_date=vdate,
                _keep_nans=True
            )
            if ts is not None and len(ts):
                chunks.append(ts)

        if chunks:
            return pd.concat(chunks).dropna()
        return empty_series(tzaware, name=name)

    @tx
    def block_staircase(
        self,
        cn,
        name,
        from_value_date=None,
        to_value_date=None,
        revision_freq=None,
        revision_time=None,
        revision_tz='UTC',
        maturity_offset=None,
        maturity_time=None,
    ):
        if not self.exists(cn, name):
            return
        self._guard_query_dates(from_value_date, to_value_date)

        latest_ts = self.get(
            cn, name,
            from_value_date=from_value_date,
            to_value_date=to_value_date,
            _keep_nans=True
        )
        tzaware = self.tzaware(cn, name)
        if not len(latest_ts):
            return empty_series(tzaware, name=name)

        def replacement_offset(offset, name):
            """pandas.DateOffset that replaces datetime parameters"""
            if not isinstance(offset, dict):
                raise TypeError(
                    f'Expected replacement offset `{name}` as dict but {type(offset)} was'
                    'given'
                )
            if not offset:  # return null offset
                return pd.DateOffset(hours=0)
            allowed_keys = ['year', 'month', 'day', 'weekday', 'hour', 'minute', 'second']
            for k in offset:
                if k not in allowed_keys:
                    raise ValueError(
                        f'Could not convert replacement offset `{name}` from dict with key '
                        f'{k}, allowed keys are {allowed_keys}'
                    )
            return pd.DateOffset(**offset)

        def shift_offset(offset, name):
            """pandas.DateOffset that shifts datetime parameters"""
            if not isinstance(offset, dict):
                raise TypeError(
                    f'Expected shift offset `{name}` as dict but {type(offset)} was given'
                )
            if not offset:  # return null offset
                return pd.DateOffset(hours=0)
            allowed_keys = [
                'years', 'months', 'weeks', 'bdays', 'days', 'hours', 'minutes', 'seconds'
            ]
            for k in offset:
                if k not in allowed_keys:
                    raise ValueError(
                        f'Could not convert `{name}` from dict with key {k}, ' +
                        f'allowed keys are {allowed_keys}'
                    )
            if 'bdays' in offset:
                if len(offset) > 1:
                    raise ValueError(
                        f'Shift offset `{name}` cannot combine \'bdays\' with other offset'
                        'units'
                    )
                return pd.offsets.BusinessDay(offset['bdays'])
            else:
                return pd.DateOffset(**offset)

        from_value_date = from_value_date or latest_ts.index.min()
        to_value_date = to_value_date or latest_ts.index.max()
        from_value_date = compatible_date(tzaware, from_value_date)
        to_value_date = compatible_date(tzaware, to_value_date)

        revision_freq = revision_freq or {'days': 1}
        revision_time = revision_time or {'hour': 0}
        revision_tz = revision_tz or 'UTC'
        maturity_offset = maturity_offset or {}
        maturity_time = maturity_time or {}

        sc_kwargs = dict(
            revision_freq=revision_freq,
            revision_time=revision_time,
            revision_tz=revision_tz,
            maturity_offset=maturity_offset,
            maturity_time=maturity_time,
        )

        revision_freq = shift_offset(revision_freq, name='revision_freq')
        revision_time = replacement_offset(revision_time, name='revision_time')
        maturity_offset = shift_offset(maturity_offset, name='maturity_offset')
        maturity_time = replacement_offset(maturity_time, name='maturity_time')
        if hasattr(maturity_time, 'weekday'):
            # do not use weekday on maturity time because pd.DateOffset(weekday=n) does
            # not preserve week number
            raise ValueError('Parameter \'weekday\' cannot be used for `maturity_time`')

        def get_block_start(rev_date):
            block_start = (rev_date + maturity_offset) + maturity_time
            if not tzaware:
                block_start = block_start.tz_localize(None)
            return block_start

        if tzaware:
            from_v_date_aware = pd.Timestamp(from_value_date).tz_convert(revision_tz)
        else:
            from_v_date_aware = pd.Timestamp(from_value_date).tz_localize(revision_tz)

        # roll back to earliest revision date to consider
        init_rev_date = (
            ((from_v_date_aware + maturity_time) - maturity_offset) + revision_time
        )
        init_block_start = get_block_start(init_rev_date)
        while init_block_start > from_value_date:
            prev_rev_date = init_rev_date - revision_freq
            prev_block_start = get_block_start(prev_rev_date)
            if not (prev_block_start < init_block_start):
                raise BlockStaircaseRevisionError(
                    sc_kwargs=sc_kwargs,
                    revision_dates=[prev_rev_date, init_rev_date],
                    block_start_dates=[prev_block_start, init_block_start]
                )
            init_rev_date = prev_rev_date
            init_block_start = prev_block_start

        # assemble blocks by looping over successive revisions
        revision_date = init_rev_date
        block_start = init_block_start
        res_ts = empty_series(tzaware, name=name)
        while block_start <= to_value_date:
            chunk = self.get(
                cn,
                name,
                revision_date=revision_date,
                from_value_date=max(block_start, from_value_date),
                to_value_date=to_value_date,
            )
            if chunk is not None and len(chunk):
                res_ts = patch(res_ts, chunk)
            next_rev_date = revision_date + revision_freq
            next_block_start = get_block_start(next_rev_date)
            if not (block_start < next_block_start):
                raise BlockStaircaseRevisionError(
                    sc_kwargs=sc_kwargs,
                    revision_dates=[revision_date, next_rev_date],
                    block_start_dates=[block_start, next_block_start],
                )
            revision_date = next_rev_date
            block_start = next_block_start

        if tzaware:
            try:
                res_ts = res_ts.tz_convert(revision_tz)
            except:
                import ipdb; ipdb.set_trace()
        return res_ts

    @tx
    def exists(self, cn, name):
        return self._series_to_tablename(cn, name) is not None

    @tx
    def latest_insertion_date(self, cn, name):
        tablename = self._series_to_tablename(cn, name)
        q = select('max(insertion_date)').table(
            f'"{self.namespace}.revision"."{tablename}"'
        )
        idate = pd.Timestamp(
            q.do(cn).scalar()
        )
        if not pd.isnull(idate):
            return idate.astimezone('UTC')

    @tx
    def first_insertion_date(self, cn, name):
        tablename = self._series_to_tablename(cn, name)
        q = select('min(insertion_date)').table(
            f'"{self.namespace}.revision"."{tablename}"'
        )
        idate = pd.Timestamp(
            q.do(cn).scalar()
        )
        if not pd.isnull(idate):
            return idate.astimezone('UTC')

    @tx
    def insertion_dates(self, cn, name,
                        from_insertion_date=None,
                        to_insertion_date=None,
                        from_value_date=None,
                        to_value_date=None,
                        **kw):
        self._guard_query_dates(
            from_insertion_date, to_insertion_date,
            from_value_date, to_value_date
        )
        revs = self._revisions(
            cn, name,
            from_insertion_date=from_insertion_date,
            to_insertion_date=to_insertion_date,
            from_value_date=from_value_date,
            to_value_date=to_value_date
        )

        return [
            idate
            for _cset, idate in revs
        ]

    @tx
    def last_id(self, cn, name):
        snapshot = self.storageclass(cn, self, name)
        return snapshot.last_id()

    @tx
    def changeset_at(self, cn, name, revdate, mode='strict'):
        operators = {
            'strict': '=',
            'before': '<=',
            'after': '>='
        }
        tablename = self._series_to_tablename(cn, name)
        assert mode in operators
        q = select(
            'id'
        ).table(
            f'"{self.namespace}.revision"."{tablename}"',
        ).where(
            f'insertion_date {operators[mode]} %(revdate)s',
            revdate=revdate
        ).order('insertion_date', 'asc'
        ).limit(1)
        return q.do(cn).scalar()

    @tx
    def rename(self, cn, oldname, newname):
        sql = (f'update "{self.namespace}".registry '
               'set name = %(newname)s '
               'where name = %(oldname)s')
        cn.execute(sql, oldname=oldname, newname=newname)

    @tx
    def delete(self, cn, name):
        tablename = self._series_to_tablename(cn, name)
        if tablename is None:
            print('not deleting unknown series', name, self.namespace)
            return
        # serialize all deletions to avoid deadlocks
        cn.execute(
            f'select pg_advisory_xact_lock({self.delete_lock_id})'
        )
        rid = cn.execute(
            f'select id from "{self.namespace}".registry '
            'where name = %(name)s',
            name=name
        ).scalar()
        # drop series tables
        cn.execute(
            f'drop table "{self.namespace}.revision"."{tablename}" cascade'
        )
        cn.execute(
            f'drop table "{self.namespace}.snapshot"."{tablename}" cascade'
        )
        cn.execute(
            f'delete from "{self.namespace}".registry '
            'where id = %(rid)s',
            rid=rid
        )

    @tx
    def strip(self, cn, name, csid):
        # wipe the diffs
        tablename = self._series_to_tablename(cn, name)
        sql = (f'delete from "{self.namespace}.revision"."{tablename}" '
               'where id >= %(csid)s')
        cn.execute(sql, csid=csid)
        snapshot = self.storageclass(cn, self, name)
        snapshot.reclaim()

    def info(self, cn):
        """Gather global statistics on the current tshistory repository
        """
        sql = f'select count(*) from "{self.namespace}".registry'
        stats = {'series count': cn.execute(sql).scalar()}
        sql = (f'select distinct name from "{self.namespace}".registry '
               'order by name')
        stats['serie names'] = [row for row, in cn.execute(sql).fetchall()]
        return stats

    @tx
    def log(self, cn, name,
            limit=None, authors=None,
            fromdate=None, todate=None):
        """Build a structure showing the history of a series in the db,
        per changeset, in chronological order.
        """
        if not self.exists(cn, name):
            return []

        log = []
        q = self._log_series_query(
            cn, name, limit, authors,
            fromdate, todate
        )
        rset = q.do(cn)
        for csetid, author, revdate, meta in rset.fetchall():
            log.append({'rev': csetid, 'author': author,
                        'date': pd.Timestamp(revdate).tz_convert('utc'),
                        'meta': meta if meta else {}})

        log.sort(key=lambda rev: rev['rev'])
        return log

    @tx
    def interval(self, cn, name, notz=False):
        tablename = self._series_to_tablename(cn, name)
        if tablename is None:
            raise ValueError(f'no interval for series: {name}')
        sql = (f'select tsstart, tsend '
               f'from "{self.namespace}.revision"."{tablename}" '
               f'order by id desc limit 1')
        res = cn.execute(sql).fetchone()
        start, end = res.tsstart, res.tsend
        tz = None
        if self.tzaware(cn, name) and not notz:
            tz = 'UTC'
        start, end = pd.Timestamp(start, tz=tz), pd.Timestamp(end, tz=tz)
        return pd.Interval(left=start, right=end, closed='both')

    _find_items = ['name']

    @tx
    def find(self, cn, query, limit=None, meta=False, source='local'):
        items = self._find_items[:]
        if meta:
            items += ['internal_metadata', 'metadata']
        q = select(
            *items
        ).table(
            f'"{self.namespace}".registry as reg'
        ).order('name', 'asc')
        sql, kw = query.sql(self.namespace)
        if sql:
            q.where(sql, **kw)
        if limit:
            q.limit(limit)

        return self._finish_find(cn, q, meta, source)

    def _finish_find(self, cn, q, meta, source):
        if not meta:
            return [
                ts(name, source=source)
                for name, in q.do(cn).fetchall()
            ]

        return [
            ts(name, imeta, umeta, source)
            for name, imeta, umeta in q.do(cn).fetchall()
        ]

    @tx
    def register_basket(self, cn, name, query):
        insert(
            f'"{self.namespace}".basket'
        ).values(
            name=name,
            query=query
        ).do(cn)

    @tx
    def basket(self, cn, name):
        query = self.basket_definition(cn, name)
        return self.find(cn, search.query.fromexpr(query))

    @tx
    def basket_definition(self, cn, name):
        query = select(
            'query'
        ).table(
            f'"{self.namespace}".basket'
        ).where(
            name=name
        ).do(cn).scalar()

        return query


    @tx
    def list_baskets(self, cn):
        q = select('name').table(f'"{self.namespace}".basket').order('name')
        return [
            name for name, in
            q.do(cn).fetchall()
        ]

    @tx
    def delete_basket(self, cn, name):
        cn.execute(
            f'delete from "{self.namespace}".basket where name=%(name)s',
            name=name
        )

    # /API
    # Helpers

    # creation / update

    def _guard_insert(self, newts, name, author, metadata, insertion_date):
        assert len(name), 'Name is an empty string'
        assert isinstance(author, str), 'Author is not a string'
        assert metadata is None or isinstance(metadata, dict), (
            f'Bad format for metadata ({repr(metadata)})'
        )
        assert (insertion_date is None or
                isinstance(insertion_date, datetime)), 'Bad format for insertion date'
        assert isinstance(newts, pd.Series), 'Not a pd.Series'
        index = newts.index
        assert isinstance(index, pd.DatetimeIndex), 'You must provide a DatetimeIndex'
        assert not index.duplicated().any(), 'There are some duplicates in the index'

        assert index.notna().all(), 'The index contains NaT entries'
        if index.tz is not None:
            newts.index = index.tz_convert('UTC')
        if not index.is_monotonic_increasing:
            newts = newts.sort_index()

        return num2float(newts)

    def _guard_query_dates(self, *dates):
        assert all(
            isinstance(dt, datetime)
            for dt in filter(None, dates)
        ), 'all query dates must be datetime-compatible objects'

    def _create(self, cn, newts, name, author, seriesmeta,
                metadata=None, insertion_date=None):
        start, end = start_end(newts, notz=False)
        if start is None:
            assert end is None
            # this is just full of nans
            return None
        # chop off unwanted nans
        newts = newts.loc[start:end]
        if len(newts) == 0:
            return None

        # at creation time we take an exclusive lock to avoid
        # a deadlock on created tables against the changeset-series fk
        cn.execute(
            f'select pg_advisory_xact_lock({self.create_lock_id})'
        )

        self._make_ts_table(cn, name)
        self._register_serie(cn, name, seriesmeta)

        snapshot = self.storageclass(cn, self, name)
        head = snapshot.create(newts)
        start, end = start_end(newts)

        self._new_revision(
            cn, name, head, start, end,
            author, insertion_date, metadata
        )

        L.info('first insertion of %s (size=%s) by %s',
               name, len(newts), author)
        return newts

    def _update(self, cn, newts, name, author,
                metadata=None, insertion_date=None):
        self._validate(cn, newts, name)

        snapshot = self.storageclass(cn, self, name)
        # NOTE: there is a potential for a small i/o optimisation
        # there as we get a bunch of chunks, and in .update we will
        # get _mostly_ the same chunks: some reads might be avoided by
        # being clever.
        series_diff = diff(
            snapshot.last(newts.index.min(),
                          newts.index.max()),
            newts
        )
        if not len(series_diff):
            L.info('no difference in %s by %s (for ts of size %s)',
                   name, author, len(newts))
            return empty_series(
                self.tzaware(cn, name),
                name=name
            )

        # compute series start/end stamps
        tsstart, tsend = start_end(series_diff)
        ival = self.interval(cn, name, notz=True)
        start = min(tsstart or ival.left, ival.left)
        end = max(tsend or ival.right, ival.right)

        if pd.isnull(series_diff[0]) or pd.isnull(series_diff[-1]):
            # we *might* be shrinking, let's look at the full series
            # and yes, shrinkers have a slow path
            last = snapshot.last()
            patched = patch(last, series_diff).dropna()
            if not len(patched):
                raise ValueError('complete erasure of a series is forbidden')
            start = patched.index[0]
            end = patched.index[-1]

        head = snapshot.update(series_diff)

        self._new_revision(
            cn, name, head, start, end,
            author, insertion_date, metadata
        )
        L.info('inserted diff (size=%s) for ts %s by %s',
               len(series_diff), name, author)
        return series_diff

    def _new_revision(self, cn, name, head, tsstart, tsend,
                      author, insertion_date, metadata):
        tablename = self._series_to_tablename(cn, name)
        if insertion_date is not None:
            assert insertion_date.tzinfo is not None, (
                f'for "{name}", the specified revision date '
                f'"{insertion_date}" must be tzaware'
            )
            idate = pd.Timestamp(insertion_date)
        else:
            idate = pd.Timestamp(datetime.utcnow(), tz='UTC')
        latest_idate = self.latest_insertion_date(cn, name)
        if latest_idate:
            assert idate > latest_idate, (
                f'"{name}" already has a newer revision than "{idate}"'
            )
        if metadata:
            metadata = json.dumps(metadata)

        q = insert(
            f'"{self.namespace}.revision"."{tablename}" '
        ).values(
            snapshot=head,
            tsstart=tsstart,
            tsend=tsend,
            author=author,
            insertion_date=idate,
            metadata=metadata
        )
        q.do(cn)

    # serie table handling

    def _make_tablename(self, cn, name):
        """ compute the unqualified (no namespace) table name
        from a serie name, to allow arbitrary serie names
        """
        # default
        tablename = name
        # postgresql table names are limited to 63 chars.
        if len(name) > 63:
            tablename = hashlib.sha1(name.encode('utf-8')).hexdigest()

        # collision detection (collision can happen after a rename)
        if cn.execute(f'select internal_metadata->\'tablename\' '
                      f'from "{self.namespace}".registry '
                      f'where internal_metadata->>\'tablename\' = %(tablename)s',
                      tablename=tablename).scalar():
            tablename = str(uuid.uuid4())

        cn.cache['series_tablename'][name] = tablename
        return tablename

    def _series_to_tablename(self, cn, name):
        tablename = cn.cache['series_tablename'].get(name)
        if tablename is not None:
            return tablename

        tablename = cn.execute(
            f'select internal_metadata->\'tablename\' '
            f'from "{self.namespace}".registry '
            f'where name = %(name)s',
            name=name
        ).scalar()
        if tablename is None:
            # bogus series name
            return
        cn.cache['series_tablename'][name] = tablename
        return tablename

    def _make_ts_table(self, cn, name):
        tablename = self._make_tablename(cn, name)
        table = sqlfile(
            SERIESSCHEMA,
            namespace=self.namespace,
            tablename=tablename
        )
        cn.execute(table)

    def _series_initial_meta(self, _cn, _name, ts):
        return series_metadata(ts)

    def _register_serie(self, cn, name, seriesmeta):
        tablename = self._series_to_tablename(cn, name)
        seriesmeta['tablename'] = tablename
        cn.execute(
            f'insert into "{self.namespace}".registry '
            '(name, internal_metadata, metadata) '
            'values (%s, %s, %s) '
            'returning id',
            name,
            json.dumps(seriesmeta),
            json.dumps({})
        ).scalar()

    # changeset handling

    def _previous_cset(self, cn, name, csid):
        tablename = self._series_to_tablename(cn, name)
        sql = (f'select id from "{self.namespace}.revision"."{tablename}" '
               'where id < %(csid)s '
               'order by id desc limit 1')
        return cn.execute(sql, csid=csid).scalar()

    # insertion handling

    def _validate(self, cn, ts, name):
        if ts.isnull().all():
            # ts erasure
            return
        tstype = ts.dtype
        meta = self.internal_metadata(cn, name)
        if tstype != meta['value_type']:
            m = (f'Type error when inserting {name}, '
                 f'new type is {tstype}, type in base is {meta["value_type"]}')
            raise Exception(m)
        if ts.index.dtype.name != meta['index_type']:
            raise Exception(
                'Incompatible index types: '
                f'ref=`{meta["index_type"]}`, new=`{ts.index.dtype.name}`'
            )

    def _revisions(self, cn, name,
                   from_insertion_date=None,
                   to_insertion_date=None,
                   from_value_date=None,
                   to_value_date=None,
                   qcallback=None):
        tablename = self._series_to_tablename(cn, name)
        q = select(
            'id', 'insertion_date'
        ).table(
            f'"{self.namespace}.revision"."{tablename}"'
        )

        if from_insertion_date:
            q.where(
                'insertion_date >= %(from_idate)s',
                from_idate=from_insertion_date
            )
        if to_insertion_date:
            q.where(
                'insertion_date <= %(to_idate)s ',
                to_idate=to_insertion_date
            )

        if from_value_date or to_value_date:
            # work around weirdness/crash in postgres, which will spit
            #   (psycopg2.errors.AmbiguousFunction)
            #   function pg_catalog.overlaps(timestamp with time zone, unknown,
            #   timestamp without time zone, timestamp without time zone)
            #   is not unique'
            # if only a tzaware from_value_date is provided.
            if not(from_value_date and to_value_date):
                if from_value_date:
                    if from_value_date.tzinfo is not None:
                        from_value_date = from_value_date.tz_localize(None)
                else:
                    if to_value_date.tzinfo is not None:
                        to_value_date = to_value_date.tz_localize(None)

            q.where(
                closed_overlaps(from_value_date, to_value_date),
                fromdate=from_value_date,
                todate=to_value_date
            )

        if qcallback:
            qcallback(q)

        q.order('id')
        return [
            (csid, pd.Timestamp(idate).astimezone('UTC'))
            for csid, idate in q.do(cn).fetchall()
        ]

    def _log_series_query(self, cn, name,
                          limit=None, authors=None,
                          fromdate=None, todate=None):
        tablename = self._series_to_tablename(cn, name)
        q = select(
            'id', 'author', 'insertion_date', 'metadata',
            opt='distinct'
        ).table(
            f'"{self.namespace}.revision"."{tablename}"'
        )

        if authors:
            q.where(
                'author in %(authors)s',
                author=tuple(authors)
            )
        if fromdate:
            q.where('insertion_date >= %(fromdate)s', fromdate=fromdate)
        if todate:
            q.where('insertion_date <= %(todate)s', todate=todate)

        q.order('id', 'desc')
        if limit:
            q.limit(limit)
        return q

    # groups

    @tx
    def group_type(self, _cn, _name):
        return 'primary'

    @tx
    def group_exists(self, cn, name):
        return bool(
            cn.execute(
                f'select id from "{self.namespace}".group_registry '
                'where name = %(name)s',
                name=name
            ).scalar()
        )

    @tx
    def list_groups(self, cn):
        cat = {
            name: 'primary'
            for name, in cn.execute(
                f'select name from "{self.namespace}".group_registry'
            ).fetchall()
        }
        return cat

    @tx
    def group_internal_metadata(self, cn, name):
        return cn.execute(
            f'select internal_metadata from "{self.namespace}".group_registry '
            'where name = %(name)s',
            name=name
        ).scalar()

    @tx
    def group_metadata(self, cn, name):
        return cn.execute(
            f'select metadata from "{self.namespace}".group_registry '
            'where name = %(name)s',
            name=name
        ).scalar()

    @tx
    def group_rename(self, cn, oldname, newname):
        sql = (f'update "{self.namespace}".group_registry '
               'set name = %(newname)s '
               'where name = %(oldname)s')
        cn.execute(sql, oldname=oldname, newname=newname)

    @tx
    def update_group_metadata(self, cn, name, metadata, internal=False):
        assert isinstance(metadata, dict)
        assert internal or not set(metadata.keys()) & self.metakeys
        meta = self.group_metadata(cn, name)
        # remove al but internal stuff
        newmeta = {
            key: meta[key]
            for key in self.metakeys
            if meta.get(key) is not None
        }
        newmeta.update(metadata)
        sql = (
            f'update "{self.namespace}".group_registry '
            'set metadata = %(metadata)s '
            f'where name = %(name)s'
        )
        cn.execute(
            sql,
            metadata=json.dumps(newmeta),
            name=name
        )

    def _group_info(self, cn, name):
        ns = self.namespace
        sql = (
            f'select gm.name, sr.name '
            f'from "{ns}".groupmap as gm '
            f'join "{ns}".group_registry as gr on gr.id = gm.groupid '
            f'join "{ns}.group".registry as sr on sr.id = gm.seriesid '
            f'where gr.name = %(name)s'
        )
        return cn.execute(sql, name=name).fetchall()

    def _create_group_item(self, cn, group_id, colname,
                           series, author, insertion_date):
        # create unique id for groupmap series
        seriename = str(uuid.uuid4())
        # insert series
        self.tsh_group.replace(
            cn, series, seriename, author,
            insertion_date=insertion_date
        )

        # get registry id of inserted series
        sql = (
            f'select id from "{self.namespace}.group".registry '
            'where name = %(sn)s'
        )
        registry_id = cn.execute(sql, sn=seriename).scalar()

        # insert infos in groupmap table
        sql = (
            f'insert into "{self.namespace}".groupmap (name, groupid, seriesid) '
            'values (%(colname)s, %(group_id)s, %(registry_id)s)'
        )
        cn.execute(
            sql,
            colname=colname,
            group_id=group_id,
            registry_id=registry_id
        )

    def _check_group_columns(self, name, infos, df):
        colnames = df.columns
        colrefs = [col for col, _ in infos]

        dupes = colnames.duplicated()
        if dupes.any():
            duplicated = colnames[dupes]
            str_dupes = ', '.join(duplicated)
            raise Exception(
                f'group update error for `{name}`: `{str_dupes}` columns are duplicated'
            )

        col_plus = set(colnames) - set(colrefs)
        if len(col_plus):
            str_plus = ', '.join(col_plus)
            raise Exception(
                f'group update error for `{name}`: `{str_plus}` columns are in excess'
            )

        col_minus = set(colrefs) - (set(colnames))
        if len(col_minus):
            str_minus = ', '.join(col_minus)
            raise Exception(
                f'group update error for `{name}`: `{str_minus}` columns are missing'
            )

    @tx
    def group_replace(self, cn, df, name, author,
                      insertion_date=None):
        assert isinstance(df, pd.DataFrame), (
            f'group `{name}` must be updated with a dataframe'
        )
        gtype = self.group_type(cn, name)
        if gtype != 'primary' and self.group_exists(cn, name):
            raise ValueError(
                f'cannot group-replace `{name}`: '
                f'this name has type `{gtype}`'
            )

        if df.columns.dtype != np.dtype('O'):
            df.columns = df.columns.astype('str')
        if insertion_date is None:
            insertion_date = pd.Timestamp(datetime.utcnow(), tz='UTC')

        infos = self._group_info(cn, name)

        if not len(infos):
            # first insertion -> register group
            sql = (
                f'insert into "{self.namespace}".group_registry (name)'
                'values (%(name)s)'
                'returning id'
            )
            group_id = cn.execute(sql, name=name).scalar()
            for colname in df.columns:
                self._create_group_item(
                    cn,
                    group_id,
                    colname,
                    df[colname],
                    author,
                    insertion_date
                )
            tsmeta = cn.execute(
                'select tsr.internal_metadata '
                f'from "{self.namespace}".group_registry as gr, '
                f'     "{self.namespace}".groupmap as gm,'
                f'     "{self.namespace}.group".registry as tsr '
                'where gr.name = %(name)s and '
                '      gr.id = gm.groupid and '
                '      gm.seriesid = tsr.id '
                'limit 1',
                name=name
            ).scalar()
            cn.execute(
                f'update "{self.namespace}".group_registry '
                'set internal_metadata = %(imeta)s, '
                '    metadata = %(metadata)s '
                f'where name = %(name)s',
                imeta=json.dumps(tsmeta),
                metadata=json.dumps({}),
                name=name
            )
            return

        # update
        self._check_group_columns(name, infos, df)
        for colname, itemname in infos:
            ts = df[colname]
            self.tsh_group.replace(
                cn,
                ts,
                itemname,
                author,
                insertion_date=insertion_date
            )

    @tx
    def group_get(self, cn, name,
                  revision_date=None,
                  from_value_date=None,
                  to_value_date=None):
        if not self.group_exists(cn, name):
            return None

        series_name_id = {
            seriesname: sid
            for sid, seriesname in self._group_info(cn, name)
        }
        allseries = []
        for name in series_name_id:
            allseries.append(
                self.tsh_group.get(
                    cn,
                    name,
                    revision_date=revision_date,
                    from_value_date=from_value_date,
                    to_value_date=to_value_date
                )
            )
        df = pd.concat(allseries, axis=1)
        return df.rename(columns=series_name_id)

    @tx
    def group_delete(self, cn, name):
        if not self.group_exists(cn, name):
            return

        infos = self._group_info(cn, name)
        sql = (
            f'delete from "{self.namespace}".group_registry '
            'where name = %(name)s'
        )
        cn.execute(sql, name=name)
        # deletion of the orphan series
        seriesnames = (sn for _, sn in infos)
        for sn in seriesnames:
            self.tsh_group.delete(cn, sn)

    @tx
    def group_insertion_dates(self, cn, name, **bounds):
        if not self.group_exists(cn, name):
            return None
        infos = self._group_info(cn, name)
        one_series_name = infos[0][1]
        return self.tsh_group.insertion_dates(cn, one_series_name, **bounds)

    @tx
    def group_history(self, cn, name, **bounds):
        if not self.group_exists(cn, name):
            return None
        infos = self._group_info(cn, name)
        # infos: list of tuples (scenario-name(external), series-name(internal))
        series_history = {}
        for scenario_name, series_name in infos:
            series_history[scenario_name] = self.tsh_group.history(
                cn,
                series_name,
                **bounds
            )
        # Now we just need to invert the keys order of series_history
        # we have: {'scenario': {idate: {ts}}}; we want: {idate: {scenario: {ts}}}
        scenarios = [name for name, _ in infos]
        renaming = {
            id: scenario
            for scenario, id in infos
        }
        history_group = {}
        # all the idates are the same for all series
        for idate in series_history[list(series_history)[0]]:
            series = [
                series_history[scenario][idate]
                for scenario in scenarios
            ]
            history_group[idate] = pd.concat(series, axis=1).rename(columns=renaming)

        return history_group


class BlockStaircaseRevisionError(Exception):
    def __init__(self, sc_kwargs, revision_dates, block_start_dates):
        msg = (
            'Revision and maturity arguments {} of `block_staircase` resulted in '
            'successive revisions {} with non-increasing block start dates {}'
        )
        super().__init__(msg.format(
            sc_kwargs,
            [str(rd) for rd in revision_dates],
            [str(bs) for bs in block_start_dates]
        ))
