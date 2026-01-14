import orjson as json
import warnings
from datetime import (
    datetime,
    timedelta
)
from typing import Optional

import requests
import pandas as pd
import numpy as np
from requests_auth import (
    OAuth2AuthorizationCodePKCE,
    OAuth2ClientCredentials
)

from tshistory.config import (
    configuration,
    NoConfigFile
)
from tshistory.util import (
    diff,
    guard_insert,
    guard_query_dates,
    logme,
    parse_delta,
    pruned_history,
    series_metadata,
    ts,
    tzaware_series,
)
from tshistory.codecs import (
    pack_group,
    pack_series,
    unpack_group_history,
    unpack_group,
    unpack_series
)
from tshistory.http.util import get_auth


# HTTP timeout constants
CONNECT_TIMEOUT = 2   # seconds - fail fast if connection is slow
READ_TIMEOUT = 60     # seconds - safety net for hung connections
DEFAULT_TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)


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
    issuer = auth.get('issuer')
    if issuer:
        discovery_url = f'{issuer}/.well-known/openid-configuration'
    else:
        domain = auth['domain']
        discovery_url = f'https://{domain}/.well-known/openid-configuration'
    meta = requests.get(discovery_url).json()
    tokenurl = meta['token_endpoint']
    clientid = auth['client_id']
    clientsecret = auth['client_secret']
    return OAuth2ClientCredentials(
        token_url=tokenurl,
        client_id=clientid,
        client_secret=clientsecret
    )


def pkce_auth(uri, auth):
    setup_cache()
    issuer = auth.get('issuer')
    if issuer:
        discovery_url = f'{issuer}/.well-known/openid-configuration'
    else:
        domain = auth['domain']
        discovery_url = f'https://{domain}/.well-known/openid-configuration'
    meta = requests.get(discovery_url).json()

    pkce_params = {
        'authorization_url': meta['authorization_endpoint'],
        'token_url': meta['token_endpoint'],
        'redirect_uri_endpoint': 'pkce',
        'client_id': auth['client_id'],
        'scope': 'openid profile email'
    }

    # confidential client: pass client_secret for token exchange
    if 'client_secret' in auth:
        pkce_params['client_secret'] = auth['client_secret']

    return OAuth2AuthorizationCodePKCE(**pkce_params)


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
            if res.status_code == 405:
                raise ValueError(res.json().get('message', res.text))
            if res.status_code == 401:
                raise Exception('401 - Unauthorized. Check your tshistory.cfg file.')
            if res.status_code == 403:
                raise Exception(f'403 - Unauthorized. {res.text}')
            if res.status_code == 413:
                raise Exception('413 - Payload to big for the web server.')
            if res.status_code >= 500:
                raise Exception(f'The server could not process your query. {res.text}')
        return res

    return wrapper


def healthcheck(session, uri):
    if not uri.endswith('/'):
        uri += '/'
    r = session.get(uri + 'versions', timeout=DEFAULT_TIMEOUT)
    if r.status_code != 200:
        if r.status_code == 401:
            print(
                'Access forbidden. '
                'Do you have a config file with the credentials ?'
            )
            return
        print(
            'The server is not answering. Your uri may be wrong. '
            'Or you are talking to an old version'
        )


class httpclient:
    index = 0
    __slots__ = 'uri', 'auth', 'session'

    def __init__(self, uri):
        self.uri = uri
        self.session = requests.Session()

        # look up the config, and accept to work
        # without one (which entails no auth)
        cfg = None
        try:
            cfg = configuration()
        except NoConfigFile:
            print('No config file found -> No auth method available.')
        except:
            raise

        if cfg is not None:
            auth = get_auth(uri, cfg)
            if 'login' in auth:
                self.session.auth = auth['login'], auth['password']
            elif 'pkce' in auth:
                self.session.auth = pkce_auth(uri, auth)
            elif 'client_id' in auth:
                self.session.auth = oauth2_auth(auth)

            # add custom headers if configured
            if 'extraheaders' in auth:
                try:
                    headers = json.loads(auth['extraheaders'])
                except json.JSONDecodeError as e:
                    raise ValueError(f'invalid extraheaders JSON: {e}')
                self.session.headers.update(headers)

        # immediately check the uri
        healthcheck(self.session, uri)

    def __repr__(self):
        return f"tshistory-http-client(uri='{self.uri}')"

    @unwraperror
    def info(self):
        res = self.session.get(f'{self.uri}/global/properties', params={
            'property': 'info'
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            return res.json()

        return res

    @unwraperror
    def sources(self):
        res = self.session.get(f'{self.uri}/global/properties', params={
            'property': 'sources'
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            return res.json()

        return res

    @unwraperror
    def exists(self, name: str):
        res = self.session.get(f'{self.uri}/series/metadata', params={
            'name': name,
            'type': 'exists'
        }, timeout=DEFAULT_TIMEOUT)
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
            },
            timeout=DEFAULT_TIMEOUT
        )

        if res.status_code in (200, 201):
            return unpack_series(name, res.content)

        return res

    @unwraperror
    def update(self,
               name: str,
               series: pd.Series,
               author: str,
               metadata: Optional[dict]=None,
               insertion_date: Optional[datetime]=None,
               keepnans: bool=False,
               manual: bool=False):
        return self._insert(
            name, series, author,
            metadata=metadata,
            insertion_date=insertion_date,
            keepnans=keepnans,
            supervision=manual
        )

    @unwraperror
    def replace(self,
                name:str,
                series: pd.Series,
                author: str,
                metadata: Optional[dict]=None,
                insertion_date: Optional[datetime]=None,
                manual: bool=False):
        return self._insert(
            name, series, author,
            metadata=metadata,
            insertion_date=insertion_date,
            replace=True,
            supervision=manual
        )

    @unwraperror
    def source(self, name: str):
        res = self.session.get(f'{self.uri}/series/source', params={
            'name': name
        }, timeout=DEFAULT_TIMEOUT)

        if res.status_code == 200:
            return res.json()
        elif res.status_code == 404:
            return None

        return res

    @unwraperror
    def metadata(self, name: str, all: bool=False):
        if all is not None:
            warnings.warn(
                'The `all` parameter is deprecated and has now no effect. '
                'You should use .internal_metadata instead',
                DeprecationWarning
            )

        res = self.session.get(f'{self.uri}/series/metadata', params={
            'name': name,
            'all': all
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            return res.json()
        if res.status_code == 404:
            return None

        return res

    @unwraperror
    def old_metadata(self, name: str):
        res = self.session.get(f'{self.uri}/series/metadata', params={
            'name': name,
            'type': 'archive'
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            return [
                (pd.Timestamp(stamp), meta, user)
                for stamp, meta, user in res.json()
            ]
        if res.status_code == 404:
            return None

        return res

    @unwraperror
    def path_series(self, pathname: str):
        res = self.session.get(f'{self.uri}/series/tree-path', params={
            'type': 'pathname',
            'name': pathname
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            return res.json()

        return res

    @unwraperror
    def series_path(self, pathname: str):
        res = self.session.get(f'{self.uri}/series/tree-path', params={
            'type': 'seriesname',
            'name': pathname
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            return res.json()

        return res

    @unwraperror
    def tree(self):
        res = self.session.get(f'{self.uri}/series/tree', timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            return res.json()

        return res

    @unwraperror
    def delete_path(self, path: str):
        res = self.session.delete(f'{self.uri}/series/tree-path', data={
            'path': path
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            return res.json()

        return res

    @unwraperror
    def rename_path(self, path: str, newpath: str):
        res = self.session.put(f'{self.uri}/series/tree-path', data={
            'path': path,
            'newpath': newpath
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            return res.json()

        return res

    @unwraperror
    def set_series_path(self, name: str, path: Optional[str]):
        res = self.session.patch(f'{self.uri}/series/tree-path', data={
            'name': name,
            'path': path
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            return

        return res

    @unwraperror
    def internal_metadata(self, name: str):
        res = self.session.get(f'{self.uri}/series/metadata', params={
            'name': name,
            'type': 'internal'
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            return res.json()
        if res.status_code == 404:
            # bw compat for old servers
            res = self.session.get(f'{self.uri}/series/metadata', params={
                'name': name,
                'all': True
            }, timeout=DEFAULT_TIMEOUT)
            if res.status_code == 404:
                return None
            if res.status_code == 200:
                meta = res.json()
                return meta

        return res

    @unwraperror
    def update_metadata(self, name: str, metadata: dict):
        assert isinstance(metadata, dict)
        existing_metadata = self.metadata(name)
        if existing_metadata is None:
            return
        existing_metadata.update(metadata)

        res = self.session.patch(f'{self.uri}/series/metadata', data={
            'name': name,
            'metadata': json.dumps(existing_metadata)
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 204:
            return None

        return res

    @unwraperror
    def replace_metadata(self, name: str, metadata: dict):
        assert isinstance(metadata, dict)
        res = self.session.put(f'{self.uri}/series/metadata', data={
            'name': name,
            'metadata': json.dumps(metadata)
        }, timeout=DEFAULT_TIMEOUT)

        return res

    @unwraperror
    def inferred_freq(self,
                      name: str,
                      revision_date: Optional[datetime]=None,
                      from_value_date: Optional[datetime]=None,
                      to_value_date: Optional[datetime]=None):
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
            f'{self.uri}/series/freq', params=args, timeout=DEFAULT_TIMEOUT
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
    def get(self,
            name: str,
            revision_date: Optional[datetime]=None,
            from_value_date: Optional[datetime]=None,
            to_value_date: Optional[datetime]=None,
            nocache: Optional[bool]=False,
            live: Optional[bool]=False,
            inferred_freq: Optional[bool]=False,
            keepnans: Optional[bool]=False):
        guard_query_dates(
            revision_date, from_value_date, to_value_date
        )
        args = {
            'name': name,
            'format': 'tshpack',
            'nocache': nocache,
            'live': live,
            'inferred_freq': inferred_freq,
            '_keep_nans': keepnans
        }
        if revision_date:
            args['insertion_date'] = strft(revision_date)
        if from_value_date:
            args['from_value_date'] = strft(from_value_date)
        if to_value_date:
            args['to_value_date'] = strft(to_value_date)
        res = self.session.get(
            f'{self.uri}/series/state', params=args, timeout=DEFAULT_TIMEOUT
        )
        if res.status_code == 404:
            return None
        if res.status_code == 200:
            return unpack_series(name, res.content)

        return res

    @unwraperror
    def insertion_dates(self,
                        name: str,
                        from_insertion_date: Optional[datetime]=None,
                        to_insertion_date: Optional[datetime]=None,
                        from_value_date: Optional[datetime]=None,
                        to_value_date: Optional[datetime]=None,
                        limit: Optional[int]=None,
                        nocache: bool=False):
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
        if limit:
            args['limit'] = limit

        res = self.session.get(
            f'{self.uri}/series/insertion_dates', params=args, timeout=DEFAULT_TIMEOUT
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
    def staircase(self,
                  name: str,
                  delta: pd.Timedelta,
                  from_value_date: Optional[datetime]=None,
                  to_value_date: Optional[datetime]=None):
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
            f'{self.uri}/series/staircase', params=args, timeout=DEFAULT_TIMEOUT
        )
        if res.status_code == 404:
            return None
        if res.status_code == 200:
            return unpack_series(name, res.content)

        return res


    @unwraperror
    def block_staircase(self,
                        name: str,
                        from_value_date: Optional[datetime] = None,
                        to_value_date: Optional[datetime] = None,
                        revision_freq: Optional[dict[str, int]] = None,
                        revision_time: Optional[dict[str, int]] = None,
                        revision_tz: str = 'UTC',
                        maturity_offset: Optional[dict[str, int]] = None,
                        maturity_time: Optional[dict[str, int]] = None):

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

        res = self.session.get(f'{self.uri}/series/block_staircase', params=args, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 404:
            return None
        if res.status_code == 200:
            ts = unpack_series(name, res.content)
            if tzaware_series(ts) and revision_tz:
                ts = ts.tz_convert(revision_tz)
            return ts
        return res


    @unwraperror
    def history(self,
                name: str,
                from_insertion_date: Optional[datetime]=None,
                to_insertion_date: Optional[datetime]=None,
                from_value_date: Optional[datetime]=None,
                to_value_date: Optional[datetime]=None,
                diffmode: bool=False,
                nocache: bool=False,
                keepnans: bool=False):
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
            nocache=nocache
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
    def type(self, name: str):
        res = self.session.get(f'{self.uri}/series/metadata', params={
            'name': name,
            'type': 'type'
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            return res.json()

        if res.status_code == 418:
            return res

        # 404 -> we tried to delete a non-existent series, do nothing

    @unwraperror
    def interval(self, name: str):
        res = self.session.get(f'{self.uri}/series/metadata', params={
            'name': name,
            'type': 'interval'
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            tzaware, left, right = res.json()
            tz = 'utc' if tzaware else None
            if left is None:
                return None
            return pd.Interval(
                pd.Timestamp(left, tz=tz),
                pd.Timestamp(right, tz=tz),
                closed='both'
            )

        return res
        # raise ValueError(f'no interval for series: {name}')

    @unwraperror
    def log(self, name: str,
            limit: Optional[int]=None,
            fromdate: Optional[datetime]=None,
            todate: Optional[datetime]=None):
        query = {
            'name': name
        }
        if limit:
            query['limit'] = limit
        if fromdate:
            query['fromdate'] = fromdate.isoformat()
        if todate:
            query['todate'] = todate.isoformat()
        res = self.session.get(f'{self.uri}/series/log', params=query, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            logs = []
            for item in res.json():
                item['date'] = pd.Timestamp(item['date'])
                logs.append(item)
            return logs

        return res

    @unwraperror
    def catalog(self, allsources: bool=True):
        res = self.session.get(
            f'{self.uri}/series/catalog',
            params={
                'allsources': allsources
            },
            timeout=DEFAULT_TIMEOUT
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
    def find(self,
             q: Optional[str]=None,
             limit: Optional[int]=None,
             meta: bool=False,
             sources: list=[],
             _source: str='local'):
        res = self.session.get(f'{self.uri}/series/find', params={
            'query': q,
            'limit': limit,
            'meta': meta,
            'sources': ','.join(sources),
            '_source': _source
        }, timeout=DEFAULT_TIMEOUT)

        if res.status_code == 200:
            return [
                ts(
                    item['name'], item['imeta'], item['meta'],
                    kind=item['kind'], source=item['source']
                )
                for item in res.json()
            ]

        return res

    @unwraperror
    def rename(self, oldname: str, newname: str, propagate: bool=True):
        res = self.session.put(
            f'{self.uri}/series/state',
            data={'name': oldname, 'newname': newname, 'propagate': json.dumps(propagate)},
            timeout=DEFAULT_TIMEOUT
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
    def strip(self, name: str, insertion_date: datetime):
        res = self.session.put(
            f'{self.uri}/series/strip',
            data={'name': name,
                  'insertion_date': insertion_date},
            timeout=DEFAULT_TIMEOUT
        )
        if res.status_code == 204:
            return

        return res

    @unwraperror
    def delete(self, name: str):
        res = self.session.delete(
            f'{self.uri}/series/state',
            data={'name': name},
            timeout=DEFAULT_TIMEOUT
        )
        if res.status_code in (204, 404):
            return

        return res

    # basket

    @unwraperror
    def register_basket(self, name: str, query: str, group: bool=False):
        res = self.session.put(
            f'{self.uri}/series/basket',
            data={
                'name': name,
                'query': query,
                'group': group
            },
            timeout=DEFAULT_TIMEOUT
        )
        if res.status_code == 200:
            return

        return res

    @unwraperror
    def basket(self,
               name: str,
               limit: Optional[int]=None,
               meta: Optional[dict]=None,
               sources: list=[],
               group: bool=False):
        res = self.session.get(f'{self.uri}/series/basket', params={
            'name': name,
            'limit': limit,
            'meta': meta,
            'sources': ','.join(sources),
            'group': group
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            return [
                ts(item['name'], item['imeta'], item['meta'], item['source'])
                for item in res.json()
            ]

        return res

    @unwraperror
    def basket_definition(self, name: str, group: bool=False):
        res = self.session.get(
            f'{self.uri}/series/basket-definition',
            params={'name': name, 'group': group}, timeout=DEFAULT_TIMEOUT
        )
        if res.status_code == 200:
            return res.json()

        return res

    @unwraperror
    def list_baskets(self, group: bool=False):
        res = self.session.get(
            f'{self.uri}/series/baskets',
            params={'group': group}, timeout=DEFAULT_TIMEOUT
        )
        if res.status_code == 200:
            return res.json()

    @unwraperror
    def rename_basket(self, oldname: str, newname: str, group: bool=False):
        res = self.session.patch(
            f'{self.uri}/series/basket',
            data={
                'oldname': oldname,
                'newname': newname,
                'group': group
            },
            timeout=DEFAULT_TIMEOUT
        )
        if res.status_code == 200:
            return
        if res.status_code == 404:
            raise ValueError(f'basket `{oldname}` does not exist')
        if res.status_code == 409:
            raise ValueError(f'basket `{newname}` already exists')

        return res

    @unwraperror
    def delete_basket(self, name: str, group: bool=False):
        res = self.session.delete(
            f'{self.uri}/series/basket',
            data={
                'name': name,
                'group': group
            },
            timeout=DEFAULT_TIMEOUT
        )
        if res.status_code == 200:
            return

        return res

    # groups

    @unwraperror
    def _group_insert(self, name, df, author,
                      insertion_date=None,
                      metadata=None,
                      replace=True):
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
            'replace': json.dumps(replace),
            'format': 'tshpack'
        }
        if metadata:
            qdata['metadata'] = json.dumps(metadata)
        res = self.session.patch(
            f'{self.uri}/group/state',
            data=qdata,
            files={
                'bgroup': pack_group(df)
            },
            timeout=DEFAULT_TIMEOUT
        )
        if res.status_code in (200, 201):
            return
        if res.status_code == 418:
            raise Exception(res.text)

        return res

    @unwraperror
    def group_update(self,
                     name: str,
                     df: pd.DataFrame,
                     author: str,
                     insertion_date: Optional[datetime]=None,
                     metadata: Optional[dict]=None):
        return self._group_insert(
            name, df, author,
            insertion_date=insertion_date,
            metadata=metadata,
            replace=False
        )

    @unwraperror
    def group_replace(self,
                      name: str,
                      df: pd.DataFrame,
                      author: str,
                      insertion_date: Optional[datetime]=None,
                      metadata: Optional[dict]=None):
        return self._group_insert(
            name, df, author,
            insertion_date=insertion_date,
            metadata=metadata,
            replace=True
        )

    @unwraperror
    def group_get(self, name: str,
                  revision_date: Optional[datetime]=None,
                  from_value_date: Optional[datetime]=None,
                  to_value_date: Optional[datetime]=None):
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
            f'{self.uri}/group/state', params=args, timeout=DEFAULT_TIMEOUT
        )
        if res.status_code == 404:
            return None
        if res.status_code == 200:
            return unpack_group(res.content)

        return res

    @unwraperror
    def group_insertion_dates(self,
                              name: str,
                              from_insertion_date: Optional[datetime]=None,
                              to_insertion_date: Optional[datetime]=None):
        args = {
            'name': name,
        }
        if from_insertion_date:
            args['from_insertion_date'] = strft(from_insertion_date)
        if to_insertion_date:
            args['to_insertion_date'] = strft(to_insertion_date)

        res = self.session.get(
            f'{self.uri}/group/insertion_dates', params=args, timeout=DEFAULT_TIMEOUT
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
    def group_history(self,
                      name: str,
                      from_insertion_date: Optional[datetime]=None,
                      to_insertion_date: Optional[datetime]=None,
                      from_value_date: Optional[datetime]=None,
                      to_value_date: Optional[datetime]=None):
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
            f'{self.uri}/group/history', params=args, timeout=DEFAULT_TIMEOUT
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
    def group_catalog(self, allsources: bool=True):
        res = self.session.get(f'{self.uri}/group/catalog', params={
            'allsources': allsources
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            tuplify = lambda x: (x, 'tsh') if '@' not in x else (x, x.split('@')[1])
            return {
                tuplify(k): [(a, b) for a, b in v]
                for k, v in res.json().items()
            }

        return res

    @unwraperror
    def group_type(self, name: str):
        res = self.session.get(f'{self.uri}/group/metadata', params={
            'name': name,
            'type': 'type'
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            return res.json()

        if res.status_code == 418:
            return res

        # 404 -> we tried to delete a non-existent group, do nothing

    @unwraperror
    def group_log(self,
                  name: str,
                  limit: Optional[int]=None,
                  fromdate: Optional[datetime]=None,
                  todate: Optional[datetime]=None):
        query = {
            'name': name
        }
        if limit:
            query['limit'] = limit
        if fromdate:
            query['fromdate'] = fromdate.isoformat()
        if todate:
            query['todate'] = todate.isoformat()
        res = self.session.get(f'{self.uri}/group/log', params=query, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            logs = []
            for item in res.json():
                item['date'] = pd.Timestamp(item['date'])
                logs.append(item)
            return logs

        return res

    @unwraperror
    def group_source(self, name: str):
        res = self.session.get(f'{self.uri}/group/source', params={
            'name': name
        }, timeout=DEFAULT_TIMEOUT)

        if res.status_code == 200:
            return res.json()
        elif res.status_code == 404:
            return None

        return res

    @unwraperror
    def group_metadata(self, name: str, all: bool=False):
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
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            return res.json()

        if res.status_code == 418:
            return res

        # 404 -> we tried to read a non-existent group, do nothing

    @unwraperror
    def group_old_metadata(self, name: str):
        res = self.session.get(f'{self.uri}/group/metadata', params={
            'name': name,
            'type': 'archive'
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            return [
                (pd.Timestamp(stamp), meta, user)
                for stamp, meta, user in res.json()
            ]
        if res.status_code == 404:
            return None

        return res

    @unwraperror
    def group_internal_metadata(self, name: str):
        res = self.session.get(f'{self.uri}/group/metadata', params={
            'name': name,
            'type': 'internal'
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code == 200:
            return res.json()

        if res.status_code == 418:
            return res

        # 404 -> we tried to delete a non-existent group, do nothing

    @unwraperror
    def replace_group_metadata(self, name: str, meta: dict):
        assert isinstance(meta, dict)
        res = self.session.put(f'{self.uri}/group/metadata', data={
            'name': name,
            'metadata': json.dumps(meta)
        }, timeout=DEFAULT_TIMEOUT)

        assert res.status_code != 404

        return res

    @unwraperror
    def update_group_metadata(self, name: str, meta: dict):
        assert isinstance(meta, dict)
        res = self.session.patch(f'{self.uri}/group/metadata', data={
            'name': name,
            'metadata': json.dumps(meta)
        }, timeout=DEFAULT_TIMEOUT)

        assert res.status_code != 404

        return res

    @unwraperror
    def group_exists(self, name: str):
        res = self.session.get(f'{self.uri}/group/metadata', params={
            'name': name
        }, timeout=DEFAULT_TIMEOUT)
        if res.status_code in (200, 404):
            meta = res.json()
            if 'message' in meta and meta['message'].endswith('does not exists'):
                return False
            return True

        return res

    @unwraperror
    def group_delete(self, name: str):
        res = self.session.delete(
            f'{self.uri}/group/state',
            data={'name': name},
            timeout=DEFAULT_TIMEOUT
        )
        if res.status_code in (204, 404):
            return

        return res

    @unwraperror
    def group_rename(self, oldname: str, newname: str):
        res = self.session.put(
            f'{self.uri}/group/state',
            data={'name': oldname, 'newname': newname},
            timeout=DEFAULT_TIMEOUT
        )
        if res.status_code == 204:
            return

        return res

    @unwraperror
    def group_find(self,
                   q: str,
                   limit: Optional[int]=None,
                   meta: bool=False,
                   sources: list=[],
                   _source: str='local'):
        assert isinstance(q, str)
        res = self.session.get(f'{self.uri}/group/find', params={
            'query': q,
            'limit': limit,
            'meta': meta,
            'sources': ','.join(sources),
            'source': _source
        }, timeout=DEFAULT_TIMEOUT)

        if res.status_code == 200:
            return [
                ts(
                    item['name'], item['imeta'], item['meta'],
                    kind=item['kind'], source=item['source']
                )
                for item in res.json()
            ]

        return res
