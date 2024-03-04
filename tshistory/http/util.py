import simplejson as json
from functools import wraps
import logging
import traceback as tb

from flask import (
    make_response,
    request
)
from werkzeug.exceptions import HTTPException
import pandas as pd

from tshistory import util


def get_auth(uri, config):
    if 'auth' not in config:
        return {}

    for name, items in util.unflatten(config['auth']).items():
        if items['uri'] == uri:
            return items

    print(f'found no auth items for this uri: `{uri}`')
    return {}


# null wsgi security wrapper

class nosecurity:
    """A wsgi middleware that provides no security at all, running all
    api calls as "admin".

    """
    __slots__ = 'app', 'role'

    def __init__(self, app, role='admin'):
        self.app = app
        self.role = role

    def __call__(self, environ, start_response):
        environ['ROLE'] = self.role
        return self.app(environ, start_response)


def required_roles(*roles):

    def decorator(func):
        def wrapper(*a, **kw):
            role = request.environ.get('ROLE') or 'guest'
            if role not in roles:
                user = request.environ.get('USER')
                return f'permission denied for user "{user}"', 403
            return func(*a, **kw)

        return wrapper
    return decorator


def utcdt(dtstr):
    return pd.Timestamp(dtstr)


def todict(dictstr):
    if dictstr is None:
        return None
    return json.loads(dictstr)


def enum(*enum):
    " an enum input type "

    def _str(val):
        if val not in enum:
            raise ValueError(f'Possible choices are in {enum}')
        return val
    _str.__schema__ = {'type': 'enum'}
    return _str


L = logging.getLogger('tshistory-server')

def onerror(func):
    @wraps(func)
    def wrapper(*a, **k):
        try:
            return func(*a, **k)
        except Exception as err:
            if isinstance(err, HTTPException):
                raise
            L.exception('oops')
            tb.print_exc()
            response = make_response(str(err))
            response.headers['Content-Type'] = 'text/plain'
            response.status_code = 418
            return response

    return wrapper


def series_response(format, series, metadata, code):
    if format == 'json':
        if series is not None:
            response = make_response(
                # no series.to_json because it switches the series to
                # utc before serialization and we don't want that
                json.dumps({
                    stamp.isoformat(): val
                    for stamp, val in series.items()
                }, ignore_nan=True)
            )
        else:
            response = make_response('null')
        response.headers['Content-Type'] = 'text/json'
        response.status_code = code
        return response

    assert format == 'tshpack'
    response = make_response(
        util.pack_series(metadata, series)
    )
    response.headers['Content-Type'] = 'application/octet-stream'
    response.status_code = code
    return response


def group_response(format, df, code):
    if format == 'json':
        # HACK: with naive dates in the index we have to play a bit
        # see https://github.com/pandas-dev/pandas/issues/12997
        # this should be fixed in pandas 1.5
        if df.index.dtype.name == 'datetime64[ns]':
            df.index = df.index.strftime('%Y-%m-%dT%H:%M:%S')
            jsondf = df.to_json()
        else:
            jsondf = df.to_json(date_format='iso')
        response = make_response(
            jsondf
        )
        response.headers['Content-Type'] = 'text/json'
        response.status_code = code
        return response

    response = make_response(
        util.pack_group(df)
    )
    response.headers['Content-Type'] = 'application/octet-stream'
    response.status_code = code
    return response
