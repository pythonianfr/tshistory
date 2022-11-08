import json
from functools import wraps
import logging
import traceback as tb

from flask import make_response
from werkzeug.exceptions import HTTPException
import pandas as pd

from tshistory import util


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
                series.to_json(orient='index',
                               date_format='iso')
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
