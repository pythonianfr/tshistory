from datetime import datetime
import logging
import hashlib
import uuid
import json
from pathlib import Path
from deprecated import deprecated

import pandas as pd

from sqlhelp import sqlfile, select, insert

from tshistory.util import (
    bisect_search,
    closed_overlaps,
    num2float,
    pruned_history,
    SeriesServices,
    start_end,
    tx,
    tzaware_serie
)
from tshistory.snapshot import Snapshot

L = logging.getLogger('tshistory.tsio')
SERIESSCHEMA = Path(__file__).parent / 'series.sql'


class timeseries(SeriesServices):
    namespace = 'tsh'
    schema = None
    metakeys = {
        'tzaware',
        'index_type',
        'index_dtype',
        'value_dtype',
        'value_type'
    }
    create_lock_id = None
    delete_lock_id = None

    def __init__(self, namespace='tsh'):
        self.namespace = namespace
        self.create_lock_id = sum(ord(c) for c in namespace)
        self.delete_lock_id = sum(ord(c) for c in namespace)

    @tx
    def update(self, cn, updatets, name, author,
               metadata=None,
               insertion_date=None):
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
        if not len(updatets):
            return
        updatets = self._guard_insert(
            updatets, name, author, metadata,
            insertion_date
        )

        assert ('<M8[ns]' == updatets.index.dtype or
                'datetime' in str(updatets.index.dtype) and not
                isinstance(updatets.index, pd.MultiIndex))

        updatets.name = name
        tablename = self._series_to_tablename(cn, name)

        if tablename is None:
            seriesmeta = self._series_initial_meta(cn, name, updatets)
            return self._create(cn, updatets, name, author, seriesmeta,
                                metadata, insertion_date)

        return self._update(cn, tablename, updatets, name, author,
                            metadata, insertion_date)

    @deprecated('reason: use the equivalent `update` method')
    def insert(self, *a, **kw):
        return self.update(*a, **kw)

    @tx
    def replace(self, cn, newts, name, author,
                metadata=None,
                insertion_date=None):
        """Create a new revision of a given time series
        and do a wholesale replacement of the series
        with the provided one (no update semantics)

        newts: pandas.Series with date index
        name: str unique identifier of the serie
        author: str free-form author name
        metadata: optional dict for changeset metadata
        """
        newts = newts.dropna()
        if not len(newts):
            return
        newts = self._guard_insert(
            newts, name, author, metadata,
            insertion_date
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
        head = Snapshot(cn, self, name).create(newts)
        self._new_revision(
            cn, name, head, start, end,
            author, insertion_date, metadata
        )
        L.info('inserted series (size=%s) for ts %s by %s',
               len(newts), name, author)
        return newts


    def list_series(self, cn):
        """Return the mapping of all series to their type"""
        sql = f'select seriesname from "{self.namespace}".registry '
        return {
            row.seriesname: 'primary'
            for row in cn.execute(sql)
        }

    @tx
    def get(self, cn, name, revision_date=None,
            from_value_date=None, to_value_date=None,
            _keep_nans=False):
        """Compute and return the serie of a given name

        revision_date: datetime filter to get previous versions of the
        serie

        """
        if not self.exists(cn, name):
            return

        csetfilter = []
        if revision_date:
            csetfilter.append(
                lambda q: q.where(
                    f'insertion_date <= %(idate)s', idate=revision_date
                )
            )
        snap = Snapshot(cn, self, name)
        _, current = snap.find(csetfilter=csetfilter,
                               from_value_date=from_value_date,
                               to_value_date=to_value_date)

        if current is not None and not _keep_nans:
            current.name = name
            current = current.dropna()
        return current

    @tx
    def metadata(self, cn, name):
        """Return metadata dict of timeserie."""
        if name in cn.cache['metadata']:
            return cn.cache['metadata']
        sql = (f'select metadata from "{self.namespace}".registry '
               'where seriesname = %(name)s')
        meta = cn.cache['metadata'][name] = cn.execute(sql, name=name).scalar()
        return meta

    @tx
    def update_metadata(self, cn, name, metadata, internal=False):
        assert isinstance(metadata, dict)
        assert internal or not set(metadata.keys()) & self.metakeys
        meta = self.metadata(cn, name)
        # remove al but internal stuff
        newmeta = {
            key: meta[key]
            for key in self.metakeys
            if meta.get(key) is not None
        }
        newmeta.update(metadata)
        sql = (f'update "{self.namespace}".registry as reg '
               'set metadata = %(metadata)s '
               'where reg.seriesname = %(seriesname)s')
        cn.execute(
            sql,
            metadata=json.dumps(newmeta),
            seriesname=name
        )
        cn.cache['metadata'][name] = meta

    def changeset_metadata(self, cn, csid):
        assert isinstance(csid, int)
        q = select(
            'metadata'
        ).table(
            f'"{self.namespace}".changeset'
        ).where(
            f'id = %(csid)s', csid=csid
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
                _keep_nans=False):
        tablename = self._series_to_tablename(cn, name)
        if tablename is None:
            return

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

        snapshot = Snapshot(cn, self, name)
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
                    diff = self.diff(serie_a, serie_b)
                    if len(diff):
                        diffs.append((revdate_b, diff))
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

        base = self.get(
            cn, name,
            from_value_date=from_value_date,
            to_value_date=to_value_date,
            _keep_nans=True
        )
        if not len(base):
            return pd.Series(name=name)

        # prepare the needed revision dates
        fromidate = base.index.min() - delta
        toidate = base.index.max() - delta

        hcache = historycache(
            self, cn, name,
            from_value_date=from_value_date,
            to_value_date=to_value_date,
            to_insertion_date=toidate,
            tzaware=self.metadata(cn, name).get('tzaware')
        )

        return hcache.staircase(
            delta,
            from_value_date,
            to_value_date
        )

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
    def insertion_dates(self, cn, name,
                        fromdate=None, todate=None):
        revs = self._revisions(
            cn, name,
            from_insertion_date=fromdate,
            to_insertion_date=todate
        )

        return [
            idate
            for _cset, idate in revs
        ]

    @tx
    def last_id(self, cn, name):
        snapshot = Snapshot(cn, self, name)
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
        )
        return q.do(cn).scalar()

    @tx
    def rename(self, cn, oldname, newname):
        sql = (f'update "{self.namespace}".registry '
               'set seriesname = %(newname)s '
               'where seriesname = %(oldname)s')
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
        rid, tablename = cn.execute(
            f'select id, tablename from "{self.namespace}".registry '
            'where seriesname = %(seriesname)s',
            seriesname=name
        ).fetchone()
        # drop series tables
        cn.execute(
            f'drop table "{self.namespace}.revision"."{tablename}" cascade'
        )
        cn.execute(
            f'drop table "{self.namespace}.snapshot"."{tablename}" cascade'
        )
        cn.execute(f'delete from "{self.namespace}".registry '
                   'where id = %(rid)s',
                   rid=rid)

    @tx
    def strip(self, cn, name, csid):
        # wipe the diffs
        tablename = self._series_to_tablename(cn, name)
        sql = (f'delete from "{self.namespace}.revision"."{tablename}" '
               'where id >= %(csid)s')
        cn.execute(sql, csid=csid)
        snapshot = Snapshot(cn, self, name)
        snapshot.reclaim()

    def info(self, cn):
        """Gather global statistics on the current tshistory repository
        """
        sql = f'select count(*) from "{self.namespace}".registry'
        stats = {'series count': cn.execute(sql).scalar()}
        sql = (f'select distinct seriesname from "{self.namespace}".registry '
               'order by seriesname')
        stats['serie names'] = [row for row, in cn.execute(sql).fetchall()]
        return stats

    @tx
    def log(self, cn, name,
            limit=None, authors=None,
            fromdate=None, todate=None):
        """Build a structure showing the history of a series in the db,
        per changeset, in chronological order.
        """
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
            raise ValueError(f'no such serie: {name}')
        sql = (f'select tsstart, tsend '
               f'from "{self.namespace}.revision"."{tablename}" '
               f'order by id desc limit 1')
        res = cn.execute(sql).fetchone()
        start, end = res.tsstart, res.tsend
        tz = None
        if self.metadata(cn, name).get('tzaware') and not notz:
            tz = 'UTC'
        start, end = pd.Timestamp(start, tz=tz), pd.Timestamp(end, tz=tz)
        return pd.Interval(left=start, right=end, closed='both')

    # /API
    # Helpers

    # creation / update

    def _guard_insert(self, newts, name, author, metadata, insertion_date):
        assert isinstance(name, str), 'Name not a string'
        assert isinstance(author, str), 'Author not a string'
        assert metadata is None or isinstance(metadata, dict), 'Bad format for metadata'
        assert (insertion_date is None or
                isinstance(insertion_date, datetime)), 'Bad format for insertion date'
        assert isinstance(newts, pd.Series), 'Not a pd.Series'
        index = newts.index
        assert not index.duplicated().any(), 'There are some duplicates in the index'

        assert index.notna().all(), 'The index contains NaT entries'
        if index.tz is not None:
            newts.index = index.tz_convert('UTC')
        if not index.is_monotonic_increasing:
            newts = newts.sort_index()

        return num2float(newts)

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

        snapshot = Snapshot(cn, self, name)
        head = snapshot.create(newts)
        start, end = start_end(newts)

        self._new_revision(
            cn, name, head, start, end,
            author, insertion_date, metadata
        )

        L.info('first insertion of %s (size=%s) by %s',
               name, len(newts), author)
        return newts

    def _update(self, cn, tablename, newts, name, author,
                metadata=None, insertion_date=None):
        self._validate(cn, newts, name)
        snapshot = Snapshot(cn, self, name)
        diff = self.diff(snapshot.last(newts.index.min(),
                                       newts.index.max()),
                         newts)
        if not len(diff):
            L.info('no difference in %s by %s (for ts of size %s)',
                   name, author, len(newts))
            return

        # compute series start/end stamps
        tsstart, tsend = start_end(newts)
        ival = self.interval(cn, name, notz=True)
        start = min(tsstart or ival.left, ival.left)
        end = max(tsend or ival.right, ival.right)

        if pd.isnull(diff[0]) or pd.isnull(diff[-1]):
            # we *might* be shrinking, let's look at the full series
            # and yes, shrinkers have a slow path
            last = snapshot.last()
            patched = self.patch(last, diff).dropna()
            if not len(patched):
                raise ValueError('complete erasure of a series is forbidden')
            if pd.isnull(diff[0]):
                start = patched.index[0]
            if pd.isnull(diff[-1]):
                end = patched.index[-1]

        head = snapshot.update(diff)
        self._new_revision(
            cn, name, head, start, end,
            author, insertion_date, metadata
        )
        L.info('inserted diff (size=%s) for ts %s by %s',
               len(diff), name, author)
        return diff

    def _new_revision(self, cn, name, head, tsstart, tsend,
                      author, insertion_date, metadata):
        tablename = self._series_to_tablename(cn, name)
        if insertion_date is not None:
            assert insertion_date.tzinfo is not None
            idate = pd.Timestamp(insertion_date)
        else:
            idate = pd.Timestamp(datetime.utcnow(), tz='UTC')
        latest_idate = self.latest_insertion_date(cn, name)
        if latest_idate:
            assert idate > latest_idate
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
        if cn.execute(f'select tablename '
                      f'from "{self.namespace}".registry '
                      f'where tablename = %(seriesname)s',
                      seriesname=name).scalar():
            tablename = str(uuid.uuid4())

        cn.cache['series_tablename'][name] = tablename
        return tablename

    def _series_to_tablename(self, cn, name):
        tablename = cn.cache['series_tablename'].get(name)
        if tablename is not None:
            return tablename

        tablename = cn.execute(
            f'select tablename from "{self.namespace}".registry '
            f'where seriesname = %(seriesname)s',
            seriesname=name
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

    def _series_initial_meta(self, cn, name, ts):
        index = ts.index
        return {
            'tzaware': tzaware_serie(ts),
            'index_type': index.dtype.name,
            'index_dtype': index.dtype.str,
            'value_dtype': ts.dtypes.str,
            'value_type': ts.dtypes.name
        }

    def _register_serie(self, cn, name, seriesmeta):
        sql = (f'insert into "{self.namespace}".registry '
               '(seriesname, tablename, metadata) '
               'values (%s, %s, %s) '
               'returning id')
        tablename = self._series_to_tablename(cn, name)
        regid = cn.execute(
            sql,
            name,
            tablename,
            json.dumps(seriesmeta)
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
        meta = self.metadata(cn, name)
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


class historycache:

    def __init__(self, tsh, cn, name,
                 from_value_date=None,
                 to_value_date=None,
                 from_insertion_date=None,
                 to_insertion_date=None,
                 tzaware=True):
        self.name = name
        self.tzaware = tzaware
        self.hist = tsh.history(
            cn, name,
            from_value_date=from_value_date,
            to_value_date=to_value_date,
            from_insertion_date=from_insertion_date,
            to_insertion_date=to_insertion_date,
            _keep_nans=True
        )
        self.idates = list(self.hist.keys())
        self.naive_idates = [
            dt.replace(tzinfo=None)
            for dt in self.idates
        ]

    def _find_nearest_idate(self, revision_date):
        if self.tzaware:
            idates = self.idates
        else:
            idates = self.naive_idates
        idx = bisect_search(idates, revision_date)
        if idx == -1:
            return None
        if idx >= len(idates):
            idx = len(idates) - 1
        return self.idates[idx]

    def get(self, revision_date=None,
            from_value_date=None,
            to_value_date=None):

        if not len(self.hist):
            return pd.Series(name=self.name)

        if revision_date is None:
            return list(self.hist.values())[-1].dropna()

        idate = self._find_nearest_idate(revision_date)
        if idate:
            return self.hist[idate].loc[
                from_value_date:to_value_date
            ].dropna()

        return pd.Series(name=self.name)

    def staircase(self, delta,
                  from_value_date=None,
                  to_value_date=None):
        """ compute a series whose value dates are bounded to be
        `delta` time after the insertion dates and where we
        keep the most recent ones
        """
        base = self.get(
            from_value_date=from_value_date,
            to_value_date=to_value_date
        )
        if not len(base):
            return base

        chunks = []
        for vdate in base.index:
            ts = self.get(
                revision_date=vdate - delta,
                from_value_date=vdate,
                to_value_date=vdate
            )
            if ts is not None and len(ts):
                chunks.append(ts)

        ts = pd.Series()
        if chunks:
            ts = pd.concat(chunks)
        ts.name = self.name
        return ts
