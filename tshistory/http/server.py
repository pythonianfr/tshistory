import json

import pandas as pd
import werkzeug
from flask import (
    Blueprint,
    request,
    make_response
)
from flask_restx import (
    Api as baseapi,
    inputs,
    Resource,
    reqparse
)

from tshistory import api as tsapi, util

from tshistory.http.util import (
    enum,
    onerror,
    series_response,
    group_response,
    todict,
    utcdt
)


def no_content():
    # see https://github.com/flask-restful/flask-restful/issues/736
    resp = make_response('', 204)
    resp.headers.clear()
    return resp


base = reqparse.RequestParser()

base.add_argument(
    'name', type=str, required=True,
    help='timeseries name'
)

update = base.copy()
update.add_argument(
    'series', type=str,
    help='json representation of the series'
)
update.add_argument(
    'author', type=str, required=True,
    help='author of the insertion'
)
update.add_argument(
    'insertion_date', type=utcdt, default=None,
    help='insertion date can be forced'
)
update.add_argument(
    'tzaware', type=inputs.boolean, default=True,
    help='tzaware series'
)
update.add_argument(
    'metadata', type=todict, default=None,
    help='metadata associated with this insertion'
)
update.add_argument(
    'replace', type=inputs.boolean, default=False,
    help='replace the current series entirely with the provided series '
    '(no update semantics)'
)
update.add_argument(
    # we are cheating a bit there, but that's life
    'supervision', type=inputs.boolean, default=False,
    help='tell if this is a supervised update'
)
update.add_argument(
    'bseries', type=werkzeug.datastructures.FileStorage,
    location='files',
    help='series in binary format (if "tshpack" is chosen)'
)
update.add_argument(
    'format', type=enum('json', 'tshpack'), default='json'
)

rename = base.copy()
rename.add_argument(
    'newname', type=str, required=True,
    help='new name of the series'
)

metadata = base.copy()
metadata.add_argument(
    'all', type=inputs.boolean, default=False,
    help='get all metadata, including internal'
)
metadata.add_argument(
    'type', type=enum('standard', 'type', 'interval'),
    default='standard',
    help='specify the kind of needed metadata'
)

put_metadata = base.copy()
put_metadata.add_argument(
    'metadata', type=str, required=True,
    help='set new metadata for a series'
)

insertion_dates = base.copy()
insertion_dates.add_argument(
    'from_insertion_date', type=utcdt, default=None
)
insertion_dates.add_argument(
    'to_insertion_date', type=utcdt, default=None
)
insertion_dates.add_argument(
    'from_value_date', type=utcdt, default=None
)
insertion_dates.add_argument(
    'to_value_date', type=utcdt, default=None
)
insertion_dates.add_argument(
    'nocache', type=inputs.boolean, default=False
)

get = base.copy()
get.add_argument(
    'insertion_date', type=utcdt, default=None,
    help='insertion date can be forced'
)
get.add_argument(
    'from_value_date', type=utcdt, default=None
)
get.add_argument(
    'to_value_date', type=utcdt, default=None
)
get.add_argument(
    'nocache', type=inputs.boolean, default=False,
    help='ask for the uncached series'
)
get.add_argument(
    'live', type=inputs.boolean, default=False,
    help='patch the cached series with the freshest data'
)
get.add_argument(
    '_keep_nans', type=inputs.boolean, default=False,
    help='keep erasure information'
)
get.add_argument(
    'format', type=enum('json', 'tshpack'), default='json'
)

delete = base.copy()

history = base.copy()
history.add_argument(
    'from_insertion_date', type=utcdt, default=None
)
history.add_argument(
    'to_insertion_date', type=utcdt, default=None
)
history.add_argument(
    'from_value_date', type=utcdt, default=None
)
history.add_argument(
    'to_value_date', type=utcdt, default=None
)
history.add_argument(
    'diffmode', type=inputs.boolean, default=False
)
history.add_argument(
    'nocache', type=inputs.boolean, default=False
)
history.add_argument(
    '_keep_nans', type=inputs.boolean, default=False
)
history.add_argument(
    'format', type=enum('json', 'tshpack'), default='json'
)

staircase = base.copy()
staircase.add_argument(
    'delta', type=lambda v: pd.Timedelta(v), required=True,
    help='time delta in iso 8601 duration'
)
staircase.add_argument(
    'from_value_date', type=utcdt, default=None
)
staircase.add_argument(
    'to_value_date', type=utcdt, default=None
)
staircase.add_argument(
    'format', type=enum('json', 'tshpack'), default='json'
)

block_staircase = base.copy()
block_staircase.add_argument(
    'from_value_date', type=utcdt, default=None
)
block_staircase.add_argument(
    'to_value_date', type=utcdt, default=None
)
block_staircase.add_argument(
    'revision_freq', type=todict, default=None
)
block_staircase.add_argument(
    'revision_time', type=todict, default=None
)
block_staircase.add_argument(
    'revision_tz', type=str, default='UTC'
)
block_staircase.add_argument(
    'maturity_offset', type=todict, default=None
)
block_staircase.add_argument(
    'maturity_time', type=todict, default=None
)
block_staircase.add_argument(
    'format', type=enum('json', 'tshpack'), default='json'
)

catalog = reqparse.RequestParser()
catalog.add_argument(
    'allsources', type=inputs.boolean, default=True
)

strip = base.copy()
strip.add_argument(
    'insertion_date', type=utcdt, default=None
)

log = base.copy()
log.add_argument(
    'limit', type=int, default=None,
    help='number of revisions from the most recent'
)
log.add_argument(
    'fromdate', type=utcdt, default=None,
    help='minimal date'
)
log.add_argument(
    'todate', type=utcdt, default=None,
    help='maximal date'
)

# groups

groupbase = reqparse.RequestParser()
groupbase.add_argument(
    'name', type=str, required=True,
    help='group name'
)

groupupdate = groupbase.copy()
groupupdate.add_argument(
    'author', type=str, required=True,
    help='author of the insertion'
)
groupupdate.add_argument(
    'insertion_date', type=utcdt, default=None,
    help='insertion date can be forced'
)
groupupdate.add_argument(
    'bgroup', type=werkzeug.datastructures.FileStorage,
    location='files',
    help='series group in binary format'
)

grouprename = groupbase.copy()
grouprename.add_argument(
    'newname', type=str, required=True,
    help='new name of the group'
)

groupget = groupbase.copy()
groupget.add_argument(
    'insertion_date', type=utcdt, default=None,
    help='insertion date can be forced'
)
groupget.add_argument(
    'from_value_date', type=utcdt, default=None
)
groupget.add_argument(
    'to_value_date', type=utcdt, default=None
)
groupget.add_argument(
    '_keep_nans', type=inputs.boolean, default=False,
    help='keep erasure information'
)
groupget.add_argument(
    'format', type=enum('json', 'tshpack'), default='json'
)

group_insertion_dates = base.copy()
group_insertion_dates.add_argument(
    'from_insertion_date', type=utcdt, default=None
)
group_insertion_dates.add_argument(
    'to_insertion_date', type=utcdt, default=None
)

group_history = base.copy()
group_history.add_argument(
    'from_insertion_date', type=utcdt, default=None
)
group_history.add_argument(
    'to_insertion_date', type=utcdt, default=None
)
group_history.add_argument(
    'from_value_date', type=utcdt, default=None
)
group_history.add_argument(
    'to_value_date', type=utcdt, default=None
)
group_history.add_argument(
    'format', type=enum('json', 'tshpack'), default='json'
)


groupcatalog = reqparse.RequestParser()
groupcatalog.add_argument(
    'allsources', type=inputs.boolean, default=True
)

groupdelete = groupbase.copy()

groupmetadata = groupbase.copy()
groupmetadata.add_argument(
    'all', type=inputs.boolean, default=False,
    help='get all metadata, including internal'
)
groupmetadata.add_argument(
    'type', type=enum('standard', 'type'),
    default='standard',
    help='specify the kind of needed metadata'
)

put_groupmetadata = groupbase.copy()
put_groupmetadata.add_argument(
    'metadata', type=str, required=True,
    help='set new metadata for a series group'
)


class httpapi:
    __slots__ = 'tsa', 'bp', 'api', 'nss', 'nsg'

    def __init__(self,
                 tsa,
                 title='tshistory api',
                 description=(
                     'reading and updating time series state, '
                     'histoy, formulas and metadata'
                 )):

        # warn against playing proxy games
        assert isinstance(tsa, tsapi.mainsource)
        self.tsa = tsa

        self.bp = Blueprint(
            'tshistory-server',
            __name__,
            template_folder='tsh_templates',
            static_folder='tsh_static',
        )

        # api & ns

        class Api(baseapi):

            # see https://github.com/flask-restful/flask-restful/issues/67
            def _help_on_404(self, message=None):
                return message or 'No such thing.'

        self.api = Api(
            self.bp,
            version='1.0',
            title=title,
            description=description
        )
        self.api.namespaces.pop(0)  # wipe the default namespace

        self.nss = self.api.namespace(
            'series',
            description='Time Series Operations'
        )
        self.nsg = self.api.namespace(
            'group',
            description='Group Operations'
        )

        self.routes()

    # routes

    def routes(self):

        tsa = self.tsa
        api = self.api
        nss = self.nss
        nsg = self.nsg

        @nss.route('/metadata')
        class timeseries_metadata(Resource):

            @api.expect(metadata)
            @onerror
            def get(self):
                args = metadata.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                if args.type == 'standard':
                    meta = tsa.metadata(args.name, all=args.all)
                    return meta, 200
                if args.type == 'type':
                    stype = tsa.type(args.name)
                    return stype, 200

                assert args.type == 'interval'
                try:
                    ival = tsa.interval(args.name)
                except ValueError:
                    return no_content()
                tzaware = tsa.metadata(args.name, all=True).get('tzaware', False)
                return (tzaware,
                        ival.left.isoformat(),
                        ival.right.isoformat()), 200

            @api.expect(put_metadata)
            @onerror
            def put(self):
                args = put_metadata.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                metadata = json.loads(args.metadata)
                try:
                    tsa.update_metadata(args.name, metadata)
                except ValueError as err:
                    if err.args[0].startswith('not allowed to'):
                        api.abort(405, err.args[0])
                    raise

                return '', 200


        @nss.route('/state')
        class timeseries_state(Resource):

            @api.expect(update)
            @onerror
            def patch(self):
                args = update.parse_args()
                if args.format == 'json':
                    series = util.fromjson(
                        args.series,
                        args.name,
                        args.tzaware
                    )
                else:
                    assert args.format == 'tshpack'
                    series = util.unpack_series(
                        args.name,
                        args.bseries.stream.read()
                    )

                exists = tsa.exists(args.name)
                try:
                    if args.replace:
                        diff = tsa.replace(
                            args.name, series, args.author,
                            metadata=args.metadata,
                            insertion_date=args.insertion_date,
                            manual=args.supervision
                        )
                    else:
                        diff = tsa.update(
                            args.name, series, args.author,
                            metadata=args.metadata,
                            insertion_date=args.insertion_date,
                            manual=args.supervision
                        )
                except ValueError as err:
                    if err.args[0].startswith('not allowed to'):
                        api.abort(405, err.args[0])
                    raise

                return series_response(
                    args.format,
                    diff,
                    tsa.metadata(args.name, all=True),
                    200 if exists else 201
                )

            @api.expect(rename)
            @onerror
            def put(self):
                args = rename.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')
                if tsa.exists(args.newname):
                    api.abort(409, f'`{args.newname}` does exists')

                try:
                    tsa.rename(args.name, args.newname)
                except ValueError as err:
                    if err.args[0].startswith('not allowed to'):
                        api.abort(405, err.args[0])
                    raise

                return no_content()

            @api.expect(get)
            @onerror
            def get(self):
                args = get.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                series = tsa.get(
                    args.name,
                    revision_date=args.insertion_date,
                    from_value_date=args.from_value_date,
                    to_value_date=args.to_value_date,
                    nocache=args.nocache,
                    live=args.live,
                    _keep_nans=args._keep_nans
                )
                # the fast path will need it
                # also it is read from a cache filled at get time
                # so very cheap call
                metadata = tsa.metadata(args.name, all=True)
                assert metadata is not None, f'series {args.name} has no metadata'

                return series_response(
                    args.format,
                    series,
                    metadata,
                    200
                )

            @api.expect(delete)
            @onerror
            def delete(self):
                args = delete.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                try:
                    tsa.delete(args.name)
                except ValueError as err:
                    if err.args[0].startswith('not allowed to'):
                        api.abort(405, err.args[0])
                    raise

                return no_content()

        @nss.route('/strip')
        class timeseries_strip(Resource):

            @api.expect(strip)
            @onerror
            def put(self):
                args = strip.parse_args()
                tsa.strip(args.name, args.insertion_date)
                return no_content()


        @nss.route('/insertion_dates')
        class timeseries_idates(Resource):

            @api.expect(history)
            @onerror
            def get(self):
                args = insertion_dates.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                idates = tsa.insertion_dates(
                    args.name,
                    from_insertion_date=args.from_insertion_date,
                    to_insertion_date=args.to_insertion_date,
                    from_value_date=args.from_value_date,
                    to_value_date=args.to_value_date,
                    nocache=args.nocache
                )
                response = make_response({'insertion_dates':
                    [
                        dt.isoformat() for dt in idates
                    ]
                })
                response.headers['Content-Type'] = 'text/json'
                return response

        @nss.route('/history')
        class timeseries_history(Resource):

            @api.expect(history)
            @onerror
            def get(self):
                args = history.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                hist = tsa.history(
                    args.name,
                    from_insertion_date=args.from_insertion_date,
                    to_insertion_date=args.to_insertion_date,
                    from_value_date=args.from_value_date,
                    to_value_date=args.to_value_date,
                    diffmode=args.diffmode,
                    nocache=args.nocache,
                    _keep_nans=args._keep_nans
                )
                metadata = tsa.metadata(args.name, all=True)

                if args.format == 'json':
                    if hist is not None:
                        response = make_response(
                            pd.DataFrame(hist).to_json()
                        )
                    else:
                        response = make_response('null')
                    response.headers['Content-Type'] = 'text/json'
                    return response

                response = make_response(
                    util.pack_history(metadata, hist)
                )
                response.headers['Content-Type'] = 'application/octet-stream'
                return response

        @nss.route('/staircase')
        class timeseries_staircase(Resource):

            @api.expect(staircase)
            @onerror
            def get(self):
                args = staircase.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                series = tsa.staircase(
                    args.name, delta=args.delta,
                    from_value_date=args.from_value_date,
                    to_value_date=args.to_value_date,
                )
                metadata = tsa.metadata(args.name, all=True)

                if args.format == 'json':
                    if series is not None:
                        response = make_response(
                            series.to_json(orient='index', date_format='iso')
                        )
                    else:
                        response = make_response('null')
                    response.headers['Content-Type'] = 'text/json'
                    return response

                response = make_response(
                    util.pack_series(metadata, series)
                )
                response.headers['Content-Type'] = 'application/octet-stream'
                return response

        @nss.route('/block_staircase')
        class timeseries_block_staircase(Resource):

            @api.expect(block_staircase)
            @onerror
            def get(self):
                args = block_staircase.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                series = tsa.block_staircase(
                    args.name,
                    from_value_date=args.from_value_date,
                    to_value_date=args.to_value_date,
                    revision_freq=args.revision_freq,
                    revision_time=args.revision_time,
                    revision_tz=args.revision_tz,
                    maturity_offset=args.maturity_offset,
                    maturity_time=args.maturity_time,
                )
                metadata = tsa.metadata(args.name, all=True)

                if args.format == 'json':
                    if series is not None:
                        response = make_response(
                            series.to_json(orient='index', date_format='iso')
                        )
                    else:
                        response = make_response('null')
                    response.headers['Content-Type'] = 'text/json'
                    return response

                response = make_response(
                    util.pack_series(metadata, series)
                )
                response.headers['Content-Type'] = 'application/octet-stream'
                return response

        @nss.route('/catalog')
        class timeseries_catalog(Resource):

            @api.expect(catalog)
            @onerror
            def get(self):
                args = catalog.parse_args()
                cat = {
                    f'{uri}': series
                    for (uri, ns), series in tsa.catalog(allsources=args.allsources).items()
                }
                return cat

        @nss.route('/log')
        class series_log(Resource):

            @api.expect(log)
            @onerror
            def get(self):
                args = log.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                logs = []
                for item in tsa.log(
                    args.name,
                    limit=args.limit,
                    fromdate=args.fromdate,
                    todate=args.todate):
                    item['date'] = item['date'].isoformat()
                    logs.append(item)

                return logs, 200

        # groups

        @nsg.route('/state')
        class timeseries_group_state(Resource):

            @api.expect(groupupdate)
            @onerror
            def patch(self):
                args = groupupdate.parse_args()

                df = util.unpack_group(
                    args.bgroup.stream.read()
                )

                exists = tsa.group_exists(args.name)
                tsa.group_replace(
                    args.name,
                    df,
                    args.author,
                    insertion_date=args.insertion_date,
                )

                return '', 200 if exists else 201

            @api.expect(groupget)
            @onerror
            def get(self):
                args = groupget.parse_args()

                df = tsa.group_get(
                    args.name,
                    revision_date=args.insertion_date,
                    from_value_date=args.from_value_date,
                    to_value_date=args.to_value_date
                )
                if df is None:
                    api.abort(404, f'`{args.name}` does not exists')

                return group_response(
                    args.format,
                    df,
                    200
                )

            @api.expect(grouprename)
            @onerror
            def put(self):
                args = grouprename.parse_args()
                if not tsa.group_exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')
                if tsa.group_exists(args.newname):
                    api.abort(409, f'`{args.newname}` does exists')

                try:
                    tsa.group_rename(args.name, args.newname)
                except ValueError as err:
                    if err.args[0].startswith('not allowed to'):
                        api.abort(405, err.args[0])
                    raise

                return no_content()

            @api.expect(groupdelete)
            @onerror
            def delete(self):
                args = groupdelete.parse_args()
                if not tsa.group_exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                try:
                    tsa.group_delete(args.name)
                except ValueError as err:
                    if err.args[0].startswith('not allowed to'):
                        api.abort(405, err.args[0])
                    raise

                return no_content()

        @nsg.route('/insertion_dates')
        class timeseries_group_idates(Resource):

            @api.expect(group_insertion_dates)
            @onerror
            def get(self):
                args = group_insertion_dates.parse_args()
                if not tsa.group_exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                idates = tsa.group_insertion_dates(
                    args.name,
                    from_insertion_date=args.from_insertion_date,
                    to_insertion_date=args.to_insertion_date,
                )
                response = make_response({'insertion_dates':
                    [
                        dt.isoformat() for dt in idates
                    ]
                })
                response.headers['Content-Type'] = 'text/json'
                return response

        @nsg.route('/history')
        class timeseries_group_history(Resource):

            @api.expect(group_history)
            @onerror
            def get(self):
                args = group_history.parse_args()
                if not tsa.group_exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                hist = tsa.group_history(
                    args.name,
                    from_insertion_date=args.from_insertion_date,
                    to_insertion_date=args.to_insertion_date,
                    from_value_date=args.from_value_date,
                    to_value_date=args.to_value_date,
                )

                response = make_response(
                    util.pack_group_history(hist)
                )
                response.headers['Content-Type'] = 'application/octet-stream'
                return response

        @nsg.route('/catalog')
        class timeseries_groupcatalog(Resource):

            @api.expect(groupcatalog)
            @onerror
            def get(self):
                cat = {
                    f'{uri}': series
                    for (uri, ns), series in tsa.group_catalog().items()
                }
                return cat

        @nsg.route('/metadata')
        class timeseries_groupmetadata(Resource):

            @api.expect(groupmetadata)
            @onerror
            def get(self):
                args = groupmetadata.parse_args()
                if not tsa.group_exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                if args.type == 'type':
                    stype = tsa.group_type(args.name)
                    return stype, 200

                assert args.type == 'standard'
                meta = tsa.group_metadata(args.name, all=args.all)
                return meta, 200

            @api.expect(put_groupmetadata)
            @onerror
            def put(self):
                args = put_groupmetadata.parse_args()
                if not tsa.group_exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                metadata = json.loads(args.metadata)
                try:
                    tsa.update_group_metadata(args.name, metadata)
                except ValueError as err:
                    if err.args[0].startswith('not allowed to'):
                        api.abort(405, err.args[0])
                    raise

                return '', 200
