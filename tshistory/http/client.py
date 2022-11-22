import json
import zlib

import inireader
import requests
import pandas as pd
import numpy as np
import pytz

from tshistory.util import (
    get_cfg_path,
    pack_group,
    pack_series,
    series_metadata,
    tzaware_serie,
    unflatten,
    unpack_history,
    unpack_group_history,
    unpack_group,
    unpack_many_series,
    unpack_series
)


def strft(dt):
    """Format dt object into str.

    We first make sure dt is localized (aka non-naive). If dt is naive
    UTC is automatically added as tzinfo.
    """
    is_naive = dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None
    if is_naive:
        dt = pytz.UTC.localize(dt)
    else:
        dt = pd.Timestamp(dt).tz_convert('UTC')

    return dt.isoformat()


def unwraperror(func):
    """Method decorator to transform HTTP 418 errors into proper
    exceptions
    """

    def wrapper(*a, **k):
        res = func(*a, **k)
        if isinstance(res, requests.models.Response):
            if res.status_code == 418:
                raise Exception(res.text)
            if res.status_code == 404:
                raise Exception('404 - please check your base uri')
            if res.status_code == 400:
                raise Exception(f'Bad Query: {res.text}')
        return res

    return wrapper


def get_auth(uri, config):
    if 'auth' not in config:
        return ()
    for name, items in unflatten(config['auth']).items():
        if items['uri'] == uri:
            return items

    print(f'found no auth items for this uri: `{uri}`')
    return ()


class Client:
    __slots__ = 'uri', 'auth', 'session'

    def __init__(self, uri):
        self.uri = uri
        self.session = requests.Session()
        auth = get_auth(
            uri,
            inireader.reader(get_cfg_path())
        )
        if 'login' in auth:
            self.session.auth = auth['login'], auth['password']

    def __repr__(self):
        return f"tshistory-http-client(uri='{self.uri}')"

    @unwraperror
    def exists(self, name):
        res = self.session.get(f'{self.uri}/series/metadata', params={
            'name': name
        })
        if res.status_code in (200, 404):
            meta = res.json()
            if 'message' in meta and meta['message'].endswith('does not exists'):
                return False
            return True

        return res

    @unwraperror
    def _insert(self, name, series, author,
                metadata=None,
                insertion_date=None,
                supervision=False,
                replace=False):
        meta = series_metadata(series)
        qdata = {
            'name': name,
            'author': author,
            'insertion_date': insertion_date.isoformat() if insertion_date else None,
            'tzaware': meta['tzaware'],
            'replace': replace,
            'supervision': supervision,
            'format': 'tshpack'
        }
        if metadata:
            qdata['metadata'] = json.dumps(metadata)

        res = self.session.patch(
            f'{self.uri}/series/state',
            data=qdata,
            files={
                'bseries': pack_series(meta, series)
            }
        )

        assert res.status_code in (200, 201, 405, 418), f'{res.status_code} {res.text}'
        if res.status_code == 405:
            raise ValueError(res.json()['message'])

        if res.status_code in (200, 201):
            return unpack_series(name, res.content)

        return res

    def update(self, name, series, author,
               metadata=None, insertion_date=None, manual=False):
        return self._insert(
            name, series, author,
            metadata=metadata,
            insertion_date=insertion_date,
            supervision=manual
        )

    @unwraperror
    def replace(self, name, series, author,
                metadata=None, insertion_date=None, manual=False):
        return self._insert(
            name, series, author,
            metadata=metadata,
            insertion_date=insertion_date,
            replace=True,
            supervision=manual
        )

    @unwraperror
    def metadata(self, name, all=False):
        res = self.session.get(f'{self.uri}/series/metadata', params={
            'name': name,
            'all': int(all)
        })
        if res.status_code == 200:
            return res.json()
        if res.status_code == 404:
            return None

        return res

    @unwraperror
    def update_metadata(self, name, metadata):
        assert isinstance(metadata, dict)
        res = self.session.put(f'{self.uri}/series/metadata', data={
            'name': name,
            'metadata': json.dumps(metadata)
        })

        return res

    @unwraperror
    def get(self, name,
            revision_date=None,
            from_value_date=None,
            to_value_date=None,
            nocache=False,
            live=False,
            _keep_nans=False):
        args = {
            'name': name,
            'format': 'tshpack',
            'nocache': nocache,
            'live': live,
            '_keep_nans': _keep_nans
        }
        if revision_date:
            args['insertion_date'] = strft(revision_date)
        if from_value_date:
            args['from_value_date'] = strft(from_value_date)
        if to_value_date:
            args['to_value_date'] = strft(to_value_date)
        res = self.session.get(
            f'{self.uri}/series/state', params=args
        )
        if res.status_code == 404:
            return None
        if res.status_code == 200:
            return unpack_series(name, res.content)

        return res

    @unwraperror
    def insertion_dates(self, name,
                        from_insertion_date=None,
                        to_insertion_date=None,
                        from_value_date=None,
                        to_value_date=None,
                        nocache=False):
        args = {
            'name': name,
            'nocache': nocache
        }
        if from_insertion_date:
            args['from_insertion_date'] = strft(from_insertion_date)
        if to_insertion_date:
            args['to_insertion_date'] = strft(to_insertion_date)
        if from_value_date:
            args['from_value_date'] = strft(from_value_date)
        if to_value_date:
            args['to_value_date'] = strft(to_value_date)

        res = self.session.get(
            f'{self.uri}/series/insertion_dates', params=args
        )
        if res.status_code == 404:
            return None
        if res.status_code == 200:
            return [
                pd.Timestamp(t, tz='UTC')
                for t in res.json()['insertion_dates']
            ]

        return res

    @unwraperror
    def staircase(self, name, delta,
                  from_value_date=None,
                  to_value_date=None):
        args = {
            'name': name,
            'delta': delta,
            'format': 'tshpack'
        }
        if from_value_date:
            args['from_value_date'] = strft(from_value_date)
        if to_value_date:
            args['to_value_date'] = strft(to_value_date)
        res = self.session.get(
            f'{self.uri}/series/staircase', params=args
        )
        if res.status_code == 404:
            return None
        if res.status_code == 200:
            return unpack_series(name, res.content)

        return res


    @unwraperror
    def block_staircase(
        self,
        name,
        from_value_date=None,
        to_value_date=None,
        revision_freq=None,
        revision_time=None,
        revision_tz='UTC',
        maturity_offset=None,
        maturity_time=None,
    ):
        args = {'name': name, 'format': 'tshpack'}

        if from_value_date is not None:
            args['from_value_date'] = strft(from_value_date)
        if to_value_date is not None:
            args['to_value_date'] = strft(to_value_date)
        if revision_freq is not None:
            args['revision_freq'] = json.dumps(revision_freq)
        if revision_time is not None:
            args['revision_time'] = json.dumps(revision_time)
        if revision_tz is not None:
            args['revision_tz'] = revision_tz
        if maturity_offset is not None:
            args['maturity_offset'] = json.dumps(maturity_offset)
        if maturity_time is not None:
            args['maturity_time'] = json.dumps(maturity_time)

        res = self.session.get(f'{self.uri}/series/block_staircase', params=args)
        if res.status_code == 404:
            return None
        if res.status_code == 200:
            ts = unpack_series(name, res.content)
            if tzaware_serie(ts) and revision_tz:
                ts = ts.tz_convert(revision_tz)
            return ts
        return res


    @unwraperror
    def history(self, name,
                from_insertion_date=None,
                to_insertion_date=None,
                from_value_date=None,
                to_value_date=None,
                diffmode=False,
                nocache=False,
                _keep_nans=False):
        args = {
            'name': name,
            'format': 'tshpack',
            'diffmode': json.dumps(diffmode),
            'nocache': json.dumps(nocache),
            '_keep_nans': json.dumps(_keep_nans)
        }
        if from_insertion_date:
            args['from_insertion_date'] = strft(from_insertion_date)
        if to_insertion_date:
            args['to_insertion_date'] = strft(to_insertion_date)
        if from_value_date:
            args['from_value_date'] = strft(from_value_date)
        if to_value_date:
            args['to_value_date'] = strft(to_value_date)
        res = self.session.get(
            f'{self.uri}/series/history', params=args
        )
        if res.status_code == 404:
            return None
        if res.status_code == 200:
            _meta, hist = unpack_history(res.content)
            for series in hist.values():
                series.name = name
            return hist

        return res

    @unwraperror
    def type(self, name):
        res = self.session.get(f'{self.uri}/series/metadata', params={
            'name': name,
            'type': 'type'
        })
        assert res.status_code in (200, 404)
        if res.status_code == 200:
            return res.json()

        if res.status_code == 418:
            return res

    @unwraperror
    def interval(self, name):
        res = self.session.get(f'{self.uri}/series/metadata', params={
            'name': name,
            'type': 'interval'
        })
        if res.status_code == 200:
            tzaware, left, right = res.json()
            tz = 'utc' if tzaware else None
            return pd.Interval(
                pd.Timestamp(left, tz=tz),
                pd.Timestamp(right, tz=tz),
                closed='both'
            )

        return res
        # raise ValueError(f'no interval for series: {name}')

    @unwraperror
    def log(self, name, limit=None, fromdate=None, todate=None):
        query = {
            'name': name
        }
        if limit:
            query['limit'] = limit
        if fromdate:
            query['fromdate'] = fromdate.isoformat()
        if todate:
            query['todate'] = todate.isoformat()
        res = self.session.get(f'{self.uri}/series/log', params=query)
        if res.status_code == 200:
            logs = []
            for item in res.json():
                item['date'] = pd.Timestamp(item['date'])
                logs.append(item)
            return logs

        return res

    @unwraperror
    def catalog(self, allsources=True):
        res = self.session.get(f'{self.uri}/series/catalog', params={
            'allsources': allsources
        })
        tuplify = lambda x: (x, 'tsh') if '@' not in x else (x, x.split('@')[1])
        if res.status_code == 200:
            return {
                tuplify(k): v
                for k, v in res.json().items()
            }

        return res

    @unwraperror
    def rename(self, oldname, newname):
        res = self.session.put(
            f'{self.uri}/series/state',
            data={'name': oldname, 'newname': newname}
        )
        if res.status_code == 204:
            return

        return res

    @unwraperror
    def strip(self, name, insertion_date):
        res = self.session.put(
            f'{self.uri}/series/strip',
            data={'name': name,
                  'insertion_date': insertion_date}
        )
        if res.status_code == 204:
            return

        return res

    @unwraperror
    def delete(self, name):
        res = self.session.delete(
            f'{self.uri}/series/state',
            data={'name': name}
        )
        if res.status_code in (204, 404):
            return

        return res

    # groups

    @unwraperror
    def group_replace(self, name, df, author,
                      insertion_date=None,
                      metadata=None):
        if not isinstance(df, pd.DataFrame):
            raise Exception(f'group `{name}` must be updated with a dataframe')

        if len(df.columns) != len(set(df.columns)):
            raise Exception(
                f'group update error for `{name}`: you have redundant columns'
            )

        if df.columns.dtype != np.dtype('O'):
            df.columns = df.columns.astype('str')

        qdata = {
            'name': name,
            'author': author,
            'insertion_date': insertion_date.isoformat() if insertion_date else None,
            'replace': json.dumps(True),
            'format': 'tshpack'
        }
        if metadata:
            qdata['metadata'] = json.dumps(metadata)
        res = self.session.patch(
            f'{self.uri}/group/state',
            data=qdata,
            files={
                'bgroup': pack_group(df)
            }
        )
        if res.status_code in (200, 201):
            return
        if res.status_code == 418:
            raise Exception(res.text)

        return res

    @unwraperror
    def group_get(self, name,
                  revision_date=None,
                  from_value_date=None,
                  to_value_date=None):
        args = {
            'name': name,
            'format': 'tshpack'
        }
        if revision_date:
            args['insertion_date'] = strft(revision_date)
        if from_value_date:
            args['from_value_date'] = strft(from_value_date)
        if to_value_date:
            args['to_value_date'] = strft(to_value_date)
        res = self.session.get(
            f'{self.uri}/group/state', params=args
        )
        if res.status_code == 404:
            return None
        if res.status_code == 200:
            return unpack_group(res.content)

        return res

    @unwraperror
    def group_insertion_dates(self, name,
                        from_insertion_date=None,
                        to_insertion_date=None):
        args = {
            'name': name,
        }
        if from_insertion_date:
            args['from_insertion_date'] = strft(from_insertion_date)
        if to_insertion_date:
            args['to_insertion_date'] = strft(to_insertion_date)

        res = requests.get(
            f'{self.uri}/group/insertion_dates', params=args
        )
        if res.status_code == 404:
            return None
        if res.status_code == 200:
            return [
                pd.Timestamp(t, tz='UTC')
                for t in res.json()['insertion_dates']
            ]

        return res

    @unwraperror
    def group_history(self, name,
                from_insertion_date=None,
                to_insertion_date=None,
                from_value_date=None,
                to_value_date=None):
        args = {
            'name': name,
            'format': 'tshpack',

        }
        if from_insertion_date:
            args['from_insertion_date'] = strft(from_insertion_date)
        if to_insertion_date:
            args['to_insertion_date'] = strft(to_insertion_date)
        if from_value_date:
            args['from_value_date'] = strft(from_value_date)
        if to_value_date:
            args['to_value_date'] = strft(to_value_date)
        res = requests.get(
            f'{self.uri}/group/history', params=args
        )
        if res.status_code == 404:
            return None
        if res.status_code == 200:
            hist = unpack_group_history(res.content)
            for series in hist.values():
                series.name = name
            return hist

        return res

    @unwraperror
    def group_catalog(self, allsources=True):
        res = self.session.get(f'{self.uri}/group/catalog', params={
            'allsources': allsources
        })
        if res.status_code == 200:
            tuplify = lambda x: (x, 'tsh') if '@' not in x else (x, x.split('@')[1])
            return {
                tuplify(k): [(a, b) for a, b in v]
                for k, v in res.json().items()
            }

        return res

    @unwraperror
    def group_type(self, name):
        res = self.session.get(f'{self.uri}/group/metadata', params={
            'name': name,
            'type': 'type'
        })
        assert res.status_code in (200, 404)
        if res.status_code == 200:
            return res.json()

        if res.status_code == 418:
            return res

    @unwraperror
    def group_metadata(self, name, all=False):
        res = self.session.get(f'{self.uri}/group/metadata', params={
            'name': name,
            'type': 'standard',
            'all': all
        })
        assert res.status_code in (200, 404)
        if res.status_code == 200:
            return res.json()

        if res.status_code == 418:
            return res

    @unwraperror
    def update_group_metadata(self, name, meta):
        assert isinstance(meta, dict)
        res = self.session.put(f'{self.uri}/group/metadata', data={
            'name': name,
            'metadata': json.dumps(meta)
        })

        assert res.status_code != 404

        return res

    @unwraperror
    def group_exists(self, name):
        res = self.session.get(f'{self.uri}/group/metadata', params={
            'name': name
        })
        if res.status_code in (200, 404):
            meta = res.json()
            if 'message' in meta and meta['message'].endswith('does not exists'):
                return False
            return True

        return res

    @unwraperror
    def group_delete(self, name):
        res = self.session.delete(
            f'{self.uri}/group/state',
            data={'name': name}
        )
        if res.status_code in (204, 404):
            return

        return res

    @unwraperror
    def group_rename(self, oldname, newname):
        res = self.session.put(
            f'{self.uri}/group/state',
            data={'name': oldname, 'newname': newname}
        )
        if res.status_code == 204:
            return

        return res
