import json
import warnings
from datetime import timedelta

import inireader
import requests
import pandas as pd
import numpy as np
from requests_auth import (
    OAuth2AuthorizationCodePKCE,
    OAuth2ClientCredentials
)

from tshistory.tsio import timeseries
from tshistory.util import (
    diff,
    get_cfg_path,
    guard_insert,
    guard_query_dates,
    logme,
    pack_group,
    pack_series,
    parse_delta,
    pruned_history,
    series_metadata,
    ts,
    tzaware_series,
    unpack_group_history,
    unpack_group,
    unpack_series
)
from tshistory.http.util import get_auth


def strft(dt):
    """Format dt object into str.

    We first make sure dt is localized (aka non-naive). If dt is naive
    UTC is automatically added as tzinfo.
    """
    is_naive = dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None
    if not is_naive:
        dt = pd.Timestamp(dt).tz_convert('UTC')

    return dt.isoformat()


def setup_cache():
    from requests_auth import OAuth2, JsonTokenFileCache
    OAuth2.token_cache = JsonTokenFileCache('.tshistory.token_cache.json')


def oauth2_auth(auth):
    setup_cache()
    domain = auth['domain']
    meta = requests.get(
        f'https://{domain}/.well-known/openid-configuration'
    ).json()
    tokenurl = meta['token_endpoint']
    clientid = auth['client_id']
    clientsecret = auth['client_secret']
    return OAuth2ClientCredentials(
        audience=auth['uri'],
        token_url=tokenurl,
        client_id=clientid,
        client_secret=clientsecret
    )


def pkce_auth(uri, auth):
    setup_cache()
    domain = auth['domain']
    meta = requests.get(
        f'https://{domain}/.well-known/openid-configuration'
    ).json()
    return OAuth2AuthorizationCodePKCE(
        authorization_url=meta['authorization_endpoint'],
        token_url=meta['token_endpoint'],
        redirect_uri_endpoint='pkce',
        audience=auth['uri'],
        client_id=auth['client_id']
    )


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
            if res.status_code == 401:
                raise Exception('401 - Unauthorized. Check your tshistory.cfg file.')
            if res.status_code == 403:
                raise Exception(f'403 - Unauthorized. {res.text}')
            if res.status_code == 413:
                raise Exception('413 - Payload to big for the web server.')
            if res.status_code >= 500:
                raise Exception('The server could not process your query.')
        return res

    return wrapper


class httpclient:
    index = 0
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
        elif 'pkce' in auth:
            self.session.auth = pkce_auth(uri, auth)
        elif 'client_id' in auth:
            self.session.auth = oauth2_auth(auth)

    def __repr__(self):
        return f"tshistory-http-client(uri='{self.uri}')"

    @unwraperror
    def exists(self, name):
        res = self.session.get(f'{self.uri}/series/metadata', params={
            'name': name,
            'type': 'exists'
        })
        if res.status_code == 200:
            return True
        elif res.status_code == 404:
            return False

        return res

    @unwraperror
    def _insert(self, name, series, author,
                metadata=None,
                insertion_date=None,
                keepnans=False,
                supervision=False,
                replace=False):
        guard_insert(
            series, name, author, metadata,
            insertion_date
        )
        meta = series_metadata(series)
        qdata = {
            'name': name,
            'author': author,
            'insertion_date': insertion_date.isoformat() if insertion_date else None,
            'keepnans': keepnans,
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

        if res.status_code == 405:
            raise ValueError(res.json()['message'])

        if res.status_code in (200, 201):
            return unpack_series(name, res.content)

        return res

    @unwraperror
    def update(self, name, series, author,
               metadata=None, insertion_date=None, keepnans=False, manual=False):
        return self._insert(
            name, series, author,
            metadata=metadata,
            insertion_date=insertion_date,
            keepnans=keepnans,
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
    def source(self, name):
        res = self.session.get(f'{self.uri}/series/source', params={
            'name': name
        })

        if res.status_code == 200:
            return res.json()
        elif res.status_code == 404:
            return None

        return res

    @unwraperror
    def metadata(self, name, all=False):
        if all is not None:
            warnings.warn(
                'The `all` parameter is deprecated and has now no effect. '
                'You should use .internal_metadata instead',
                DeprecationWarning
            )

        res = self.session.get(f'{self.uri}/series/metadata', params={
            'name': name,
            'all': all
        })
        if res.status_code == 200:
            return res.json()
        if res.status_code == 404:
            return None

        return res

    @unwraperror
    def internal_metadata(self, name):
        res = self.session.get(f'{self.uri}/series/metadata', params={
            'name': name,
            'type': 'internal'
        })
        if res.status_code == 200:
            return res.json()
        if res.status_code == 404:
            # bw compat for old servers
            res = self.session.get(f'{self.uri}/series/metadata', params={
                'name': name,
                'all': True
            })
            if res.status_code == 404:
                return None
            if res.status_code == 200:
                meta = res.json()
                for key in list(meta):
                    # list call above to help against a weird
                    # `dictionary changed size during iteration`
                    if key not in timeseries.metakeys:
                        meta.pop(key, None)
                return meta

        return res

    @unwraperror
    def update_metadata(self, name, metadata):
        assert isinstance(metadata, dict)
        existing_metadata = self.metadata(name)
        if existing_metadata is None:
            return
        existing_metadata.update(metadata)

        res = self.session.patch(f'{self.uri}/series/metadata', data={
            'name': name,
            'metadata': json.dumps(existing_metadata)
        })
        if res.status_code == 204:
            return None

        return res

    @unwraperror
    def replace_metadata(self, name, metadata):
        assert isinstance(metadata, dict)
        res = self.session.put(f'{self.uri}/series/metadata', data={
            'name': name,
            'metadata': json.dumps(metadata)
        })

        return res

    @unwraperror
    def inferred_freq(self, name,
                      revision_date=None,
                      from_value_date=None,
                      to_value_date=None):
        args = {
            'name': name
        }
        if revision_date:
            args['revision_date'] = strft(revision_date)
        if from_value_date:
            args['from_value_date'] = strft(from_value_date)
        if to_value_date:
            args['to_value_date'] = strft(to_value_date)

        res = self.session.get(
            f'{self.uri}/series/freq', params=args
        )
        if res.status_code == 404:
            return None
        if res.status_code == 200:
            data = res.json()
            if data is None:
                return
            ifreq = data['inferred_freq']
            return parse_delta(ifreq[0]), float(ifreq[1])
        return res

    @unwraperror
    def get(self, name,
            revision_date=None,
            from_value_date=None,
            to_value_date=None,
            nocache=False,
            live=False,
            inferred_freq=False,
            _keep_nans=False):
        guard_query_dates(
            revision_date, from_value_date, to_value_date
        )
        args = {
            'name': name,
            'format': 'tshpack',
            'nocache': nocache,
            'live': live,
            'inferred_freq': inferred_freq,
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
        guard_query_dates(
            from_insertion_date, to_insertion_date,
            from_value_date, to_value_date
        )
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
        guard_query_dates(
            from_value_date, to_value_date
        )
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
    def block_staircase(self, name,
                        from_value_date=None,
                        to_value_date=None,
                        revision_freq=None,
                        revision_time=None,
                        revision_tz='UTC',
                        maturity_offset=None,
                        maturity_time=None):

        guard_query_dates(from_value_date, to_value_date)
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
            if tzaware_series(ts) and revision_tz:
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
        guard_query_dates(
            from_insertion_date, to_insertion_date,
            from_value_date, to_value_date
        )

        if not self.exists(name):
            return

        idates = self.insertion_dates(
            name,
            from_insertion_date,
            to_insertion_date,
            from_value_date,
            to_value_date,
            nocache
        )
        base = None
        if diffmode:
            base = self.get(
                name,
                revision_date=idates[0] - timedelta(seconds=1),
                from_value_date=from_value_date,
                to_value_date=to_value_date
            )

        hist = {}
        for idate in idates:
            ts = self.get(
                name,
                revision_date=idate,
                from_value_date=from_value_date,
                to_value_date=to_value_date,
                nocache=nocache
            )
            if diffmode:
                oldbase = base
                base = ts
                ts = diff(oldbase, ts)
            hist[idate] = ts

        if from_value_date or to_value_date:
            hist = pruned_history(hist)

        return hist

    @unwraperror
    def type(self, name):
        res = self.session.get(f'{self.uri}/series/metadata', params={
            'name': name,
            'type': 'type'
        })
        if res.status_code == 200:
            return res.json()

        if res.status_code == 418:
            return res

        # 404 -> we tried to delete a non-existent series, do nothing

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
        res = self.session.get(
            f'{self.uri}/series/catalog',
            params={
                'allsources': allsources
            },
            timeout=(2, 3)
        )
        tuplify = lambda x: (x, 'tsh') if '@' not in x else (x, x.split('@')[1])
        if res.status_code == 200:
            return {
                tuplify(k): v
                for k, v in res.json().items()
            }
        elif res.status_code in (500, 502, 503, 504):
            logme('tshistory.http.client.catalog').warning(
                'remote at %s cannot return a catalog',
                self.uri
            )
            return {}

        return res

    @unwraperror
    def find(self, q, limit=None, meta=False, _source='local'):
        assert isinstance(q, str)
        res = self.session.get(f'{self.uri}/series/find', params={
            'query': q,
            'limit': limit,
            'meta': meta,
            'source': _source
        })

        if res.status_code == 200:
            return [
                ts(item['name'], item['imeta'], item['meta'], kind=item['kind'])
                for item in res.json()
            ]

        return res

    @unwraperror
    def rename(self, oldname, newname, propagate=True):
        res = self.session.put(
            f'{self.uri}/series/state',
            data={'name': oldname, 'newname': newname, 'propagate': json.dumps(propagate)}
        )
        if res.status_code == 204:
            return
        if res.status_code == 404:
            # oldname didn't exist
            return
        if res.status_code == 409:
            # newname already exists
            raise ValueError(
                f'`{newname}` already exists.'
            )

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

    # basket

    @unwraperror
    def register_basket(self, name, query):
        res = self.session.put(
            f'{self.uri}/series/basket',
            data={
                'name': name,
                'query': query
            }
        )
        if res.status_code == 200:
            return

        return res

    @unwraperror
    def basket(self, name):
        res = self.session.get(
            f'{self.uri}/series/basket',
            params={'name': name}
        )
        if res.status_code == 200:
            return [
                ts(item['name'], item['imeta'], item['meta'])
                for item in res.json()
            ]

        return res

    @unwraperror
    def basket_definition(self, name):
        res = self.session.get(
            f'{self.uri}/series/basket-definition',
            params={'name': name}
        )
        if res.status_code == 200:
            return res.json()

        return res

    @unwraperror
    def list_baskets(self):
        res = self.session.get(
            f'{self.uri}/series/baskets'
        )
        if res.status_code == 200:
            return res.json()

    @unwraperror
    def delete_basket(self, name):
        res = self.session.delete(
            f'{self.uri}/series/basket',
            data={
                'name': name,
            }
        )
        if res.status_code == 200:
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

        res = self.session.get(
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
        res = self.session.get(
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
        if res.status_code == 200:
            return res.json()

        if res.status_code == 418:
            return res

        # 404 -> we tried to delete a non-existent group, do nothing

    @unwraperror
    def group_metadata(self, name, all=False):
        if all is not None:
            warnings.warn(
                'The `all` parameter is deprecated and has now no effect. '
                'You should use .internal_metadata instead',
                DeprecationWarning
            )

        res = self.session.get(f'{self.uri}/group/metadata', params={
            'name': name,
            'type': 'standard',
            'all': all
        })
        if res.status_code == 200:
            return res.json()

        if res.status_code == 418:
            return res

        # 404 -> we tried to read a non-existent group, do nothing

    @unwraperror
    def group_internal_metadata(self, name):
        res = self.session.get(f'{self.uri}/group/metadata', params={
            'name': name,
            'type': 'internal'
        })
        if res.status_code == 200:
            return res.json()

        if res.status_code == 418:
            return res

        # 404 -> we tried to delete a non-existent group, do nothing

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
