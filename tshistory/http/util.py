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

from tshistory import (
    codecs,
    util
)


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


def force_naive(maybe_datetime):
    if maybe_datetime is None:
        return None
    return maybe_datetime.tz_localize(None)


def convert_bounds(from_value_date, to_value_date, tzone, tzaware):
    if not tzaware:
        return force_naive(from_value_date), force_naive(to_value_date)
    if from_value_date and from_value_date.tz is None and tzone is not None:
        from_value_date = from_value_date.tz_localize(tzone)
    if to_value_date and to_value_date.tz is None and tzone is not None:
        to_value_date = to_value_date.tz_localize(tzone)
    return from_value_date, to_value_date


def prune_bounds(series, from_value_date, to_value_date, exclude='none'):
    if exclude == 'none':
        return series
    assert exclude in ('left', 'right', 'both')
    if exclude == 'left' and from_value_date is None:
        return series
    if exclude == 'right' and to_value_date is None:
        return series
    if exclude == 'left':
        to_prune = [from_value_date]
    elif exclude == 'right':
        to_prune = [to_value_date]
    elif exclude == 'both':
        to_prune = [from_value_date, to_value_date]
    return _prune(series, to_prune)


def _prune(series, datetimes):
    mask = series.index.isin(datetimes)
    if mask.any():
        series = series[~mask]
    return series


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


def series_to_json(series):
    """ replace series.to_json because it switches
    the series to utc before serialization """
    return json.dumps({
        stamp.isoformat(): val
        for stamp, val in series.items()
    }, ignore_nan=True)


def group_to_json(df):
    result = {}
    for col in df.columns:
        result[col] = {
            stamp.isoformat(): val
            for stamp, val in df[col].items()
        }
    return json.dumps(result, ignore_nan=True)


def series_response(format, series, metadata, code):
    if format == 'json':
        if series is not None:
            response = make_response(

                series_to_json(series)
            )
        else:
            response = make_response('null')
        response.headers['Content-Type'] = 'text/json'
        response.status_code = code
        return response

    assert format == 'tshpack'
    response = make_response(
        codecs.pack_series(metadata, series)
    )
    response.headers['Content-Type'] = 'application/octet-stream'
    response.status_code = code
    return response


def group_response(format, df, code):
    if format == 'json':
        # HACK: with naive dates in the index we have to play a bit
        # see https://github.com/pandas-dev/pandas/issues/12997
        # this should be fixed in pandas 1.5
        response = make_response(
            group_to_json(df)
        )
        response.headers['Content-Type'] = 'text/json'
        response.status_code = code
        return response

    response = make_response(
        codecs.pack_group(df)
    )
    response.headers['Content-Type'] = 'application/octet-stream'
    response.status_code = code
    return response
