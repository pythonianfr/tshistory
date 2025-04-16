import json

import pandas as pd
import werkzeug
from flask import (
    Blueprint,
    jsonify,
    make_response
)
from flask_restx import (
    Api as baseapi,
    inputs,
    Resource,
    reqparse
)
from dbcache import api as storeapi

from tshistory import (
    api as tsapi,
    codecs,
    config,
    util
)
from tshistory.http.util import (
    onerror,
    series_response,
    group_response,
    required_roles,
    todict,
    utcdt,
    convert_bounds,
    prune_bounds,
)


def no_content():
    # see https://github.com/flask-restful/flask-restful/issues/736
    resp = make_response('', 204)
    resp.headers.clear()
    return resp


def rawseries(value):
    # here we got a dict with stamps as strings and nulls as Nones
    # *or* maybe a string ...
    return value


def csv(value):
    if not value:
        return []
    return value.split(',')


properties = reqparse.RequestParser()
properties.add_argument(
    'property', type=str, choices=('info', 'sources',),
    required=True,
    help='get the global instance properties'
)

base = reqparse.RequestParser()

base.add_argument(
    'name', type=str, required=True,
    help='timeseries name'
)

update = base.copy()
update.add_argument(
    'series', type=rawseries,
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
    'keepnans', type=inputs.boolean, default=False,
    help='treat nans as point erasure or drop them'
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
    'tzone', type=str, default='UTC',
    help='Convert tz-aware series into this time zone before sending'
)
update.add_argument(
    'format', type=str, choices=('json', 'tshpack'), default='json'
)
update.add_argument(
    'dtype', type=str, default='float64',
    help='Force value-type for creation with nans series'
)

rename = base.copy()
rename.add_argument(
    'newname', type=str, required=True,
    help='new name of the series'
)
rename.add_argument(
    'propagate', type=inputs.boolean,
    default=True,
    help='propagate renaming to dependents'
)

source = base.copy()

metadata = base.copy()
metadata.add_argument(
    'all', type=inputs.boolean, default=False,
    help='get all metadata, including internal'
)
metadata.add_argument(
    'type', type=str,
    choices=('standard', 'internal', 'archive', 'type', 'exists', 'interval'),
    default='standard',
    help='specify the kind of needed metadata'
)

put_metadata = base.copy()
put_metadata.add_argument(
    'metadata', type=str, required=True,
    help='set new metadata for a series'
)

tree = reqparse.RequestParser()
tree.add_argument(
    'attribute', type=str,
    help='set the tree attribute'
)

treepath = reqparse.RequestParser()
treepath.add_argument(
    'name', type=str, required=True,
    help='path name or series name (depending on the value of the type parameter)'
)
treepath.add_argument(
    'type', type=str, required=True,
    choices=('pathname', 'seriesname'),
    help='describe the role of the name attribute'
)

treepath_delete = reqparse.RequestParser()
treepath_delete.add_argument(
    'path', type=str, required=True,
    help='path to delete'
)

inferred_freq = base.copy()
inferred_freq.add_argument(
    'revision_date', type=utcdt, default=None,
)
inferred_freq.add_argument(
    'from_value_date', type=utcdt, default=None
)
inferred_freq.add_argument(
    'to_value_date', type=utcdt, default=None
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
    'limit', type=int, default=None
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
    'tzone', type=str, default='UTC',
    help='Convert tz-aware series into this time zone before sending'
)
get.add_argument(
    'exclude', type=str, default='none',
    help='Exclude "left", "right" or "both" request bounds from the series index'
)
get.add_argument(
    'inferred_freq', type=inputs.boolean, default=False,
    help='re-index series on a inferred frequency'
)
get.add_argument(
    'format', type=str, choices=('json', 'tshpack'), default='json'
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
    'format', type=str, choices=('json', 'tshpack'), default='json'
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
    'format', type=str, choices=('json', 'tshpack'), default='json'
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
    'format', type=str, choices=('json', 'tshpack'), default='json'
)

catalog = reqparse.RequestParser()
catalog.add_argument(
    'allsources', type=inputs.boolean, default=True
)

find = reqparse.RequestParser()
find.add_argument(
    'query', type=str
)
find.add_argument(
    'limit', type=int
)
find.add_argument(
    'meta', type=inputs.boolean, default=False
)
find.add_argument(
    'sources', type=csv, default=[]
)
find.add_argument(
    '_source', type=str, default='local'
)

basket = reqparse.RequestParser()
basket.add_argument(
    'name', type=str
)
basket.add_argument(
    'limit', type=int
)
basket.add_argument(
    'meta', type=inputs.boolean, default=False
)
basket.add_argument(
    'sources', type=csv, default=[]
)

register_basket = reqparse.RequestParser()
register_basket.add_argument(
    'name', type=str
)
register_basket.add_argument(
    'query', type=str
)

nothing = reqparse.RequestParser()


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
    'replace', type=inputs.boolean,
    help='replace or update operation'
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
    'format', type=str, choices=('json', 'tshpack'), default='json'
)
groupget.add_argument(
    'tzone', type=str, default='UTC' ,
    help = 'Convert tz-aware group into this time zone before sending'
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
    'format', type=str, choices=('json', 'tshpack'), default='json'
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
    'type', type=str,
    choices=('standard', 'archive', 'internal', 'type'),
    default='standard',
    help='specify the kind of needed metadata'
)

put_groupmetadata = groupbase.copy()
put_groupmetadata.add_argument(
    'metadata', type=str, required=True,
    help='set new metadata for a series group'
)

groupsource = base.copy()

groupfind = reqparse.RequestParser()
groupfind.add_argument(
    'query', type=str
)
groupfind.add_argument(
    'limit', type=int
)
groupfind.add_argument(
    'meta', type=inputs.boolean, default=False
)
groupfind.add_argument(
    '_source', type=str, default='local'
)


class httpapi:
    __slots__ = 'tsa', 'bp', 'api', 'nsglobal', 'nss', 'nsg'

    def __init__(self,
                 tsa,
                 title='tshistory api',
                 description=(
                     'reading and updating time series state, '
                     'history, formulas and metadata'
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

        self.nsglobal = self.api.namespace(
            'global',
            description='Global Operations'
        )

        self.nss = self.api.namespace(
            'series',
            description='Time Series Operations'
        )
        self.nsg = self.api.namespace(
            'group',
            description='Group Operations'
        )

        self.routes()

        # ad-hoc stuff
        @self.bp.route('/versions')
        def versions():
            store = storeapi.kvstore(
                tsa.uri,
                namespace='tsh-kvstore'
            )
            return jsonify(store.all())

    # routes

    def routes(self):

        tsa = self.tsa
        api = self.api
        nsglobal = self.nsglobal
        nss = self.nss
        nsg = self.nsg

        cfg = config.configuration()

        @nsglobal.route('/properties')
        class global_properties(Resource):

            @api.doc(responses={200: 'Got content'})
            @api.expect(properties)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                """returns the secondary sources of the local instance.
                """
                args = properties.parse_args()
                if args.property == 'sources':
                    return tsa.sources(), 200
                elif args.property == 'info':
                    return tsa.info(), 200

                # we should never get there
                api.abort(400, 'Asked property does not exist')

        @nss.route('/source')
        class timeseries_source(Resource):

            @api.doc(responses={200: 'Got content', 404: 'Does not exist'})
            @api.expect(source)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                """returns the source of a series

                If it comes from a secondary source, it returns the source name.
                If it comes from the main source it returns the "local" string.
                """
                args = source.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                return tsa.source(args.name), 200

        @nss.route('/sources')
        class timeseries_sources(Resource):

            @api.doc(responses={200: 'Got content'})
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                """returns the sources of a Refinery"""
                return cfg.sources(), 200

        @nss.route('/metadata')
        class timeseries_metadata(Resource):

            @api.doc(responses={200: 'Got content', 404: 'Does not exist'})
            @api.expect(metadata)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                """get a series metadata

                The "type" field decides which kind of metadata is returned.
                * exists -> tells if the series exists (bool)
                * type -> returns the type of the series ("primary" or "formula")
                * standard -> return the user defined metadata (str -> scalar dict)
                * internal -> return the internal metadata (str -> scalar dict)

                """
                args = metadata.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                if args.type == 'exists':
                    return True, 200

                imeta = tsa.internal_metadata(args.name)
                if args.type == 'standard':
                    usermeta = tsa.metadata(args.name) or {}
                    #  bw compat pre internal-metadata for old clients
                    if args.all:
                        usermeta.update(imeta)
                    return usermeta, 200

                if args.type == 'archive':
                    metas = [
                        (stamp.isoformat(), meta)
                        for stamp, meta in tsa.old_metadata(args.name)
                    ]
                    return metas, 200

                if args.type == 'internal':
                    return imeta, 200

                if args.type == 'type':
                    stype = tsa.type(args.name)
                    return stype, 200

                tzaware = imeta.get('tzaware')
                assert args.type == 'interval'
                try:
                    ival = tsa.interval(args.name)
                    if ival is None:
                        return (tzaware, None, None), 200
                except ValueError:
                    return no_content()

                return (tzaware,
                        ival.left.isoformat(),
                        ival.right.isoformat()), 200

            @api.doc(responses={
                200: 'Got content',
                404: 'Does not exist',
                405: 'Not allowed'
            })
            @api.expect(put_metadata)
            @onerror
            @required_roles('admin', 'rw')
            def put(self):
                """replace the user metadata of a series

                The metadata must be provided as a json string.
                The format is a key-value mapping.
                Values must be scalars.
                """
                args = put_metadata.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                metadata = json.loads(args.metadata)
                try:
                    tsa.replace_metadata(args.name, metadata)
                except ValueError as err:
                    if err.args[0].startswith('not allowed to'):
                        api.abort(405, err.args[0])
                    raise

                return '', 200

            @api.doc(responses={204: 'Success', 404: 'Does not exist'})
            @api.expect(put_metadata)
            @onerror
            @required_roles('admin', 'rw')
            def patch(self):
                """update the user metadata of a series

                The unmodified entries are left unmodified.

                The metadata must be provided as a json string.
                The format is a key-value mapping.
                Values must be scalars.
                """
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

                return no_content()

        @nss.route('/metadata-keys')
        class timeseries_metadata_keys(Resource):

            @api.expect(nothing)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                """returns the sources of a Refinery"""
                return tsa.list_metadata_keys()

        @nss.route('/tree-attribute')
        class timeseries_tree_attribute(Resource):

            @api.expect(nothing)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                return tsa.tree_attribute()

            @api.expect(tree)
            @onerror
            @required_roles('admin')
            def put(self):
                args = tree.parse_args()
                return tsa.set_tree_attribute(args.attribute)

        @nss.route('/tree-path')
        class timeseries_tree_path(Resource):

            @api.expect(treepath)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                args = treepath.parse_args()
                if args.type == 'pathname':
                    return tsa.path_series(args.name)

                assert args.type == 'seriesname'
                return tsa.series_path(args.name)

            @api.expect(treepath_delete)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def delete(self):
                args = treepath_delete.parse_args()
                return tsa.delete_path(args.path)

        @nss.route('/tree')
        class timeseries_tree(Resource):

            @api.expect(nothing)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                return tsa.tree()

        @nss.route('/freq')
        class timeseries_freq(Resource):

            @api.doc(responses={200: 'Got Content', 404: 'Does not exist'})
            @api.expect(inferred_freq)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                """returns the inferred period and a quality indicator of a series

                The return format is a mapping from "inferred_freq" to a tuple.
                Tuple element 1 contains the freq as an ISO9601 time delta.
                Tuple element 2 contains the quality indicator as a float (from 0 to 1).

                Example:

                {
                 "inferred_freq": [
                  "P0DT0H0M3600S",
                  "1.0"
                 ]
                }
                """
                args = inferred_freq.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                freq_qa = tsa.inferred_freq(
                    args.name,
                    args.revision_date,
                    args.from_value_date,
                    args.to_value_date
                )

                if freq_qa is None:
                    return make_response('null')

                freq = util.delta_isoformat(freq_qa[0])
                response = make_response(
                    {
                        'inferred_freq': (freq, str(freq_qa[1]))
                    }
                )
                response.headers['Content-Type'] = 'text/json'
                return response

        @nss.route('/state')
        class timeseries_state(Resource):

            @api.doc(responses={
                200: 'Updated',
                201: 'Created',
                405: 'Not allowed'
            })
            @api.expect(update)
            @onerror
            @required_roles('admin', 'rw')
            def patch(self):
                """create or update a series

                The series field should receive a json mapping encoded series, like
                {
                 "2023-1-1T00:00:00": 42.5,
                 "2023-1-2T00:00:00": 172.3
                }

                The tzaware field must be set to indicate if the
                series is timezone aware or not.

                The replace boolean indicates if we are doing an
                "update" or a "replace" api call.

                The format fields accept another value than "json" but
                this is used by the Python client and will be left
                undocumented.

                """
                args = update.parse_args()
                if args.format == 'json':
                    meta = tsa.internal_metadata(args.name)
                    # creation
                    if not meta:
                        dtype = args.dtype
                    # existing
                    else:
                        dtype = meta and meta['value_dtype'] or None
                    # data given in parameter
                    if args.series is not None:
                        series = pd.Series(args.series, dtype=dtype)
                    # data given as file
                    else:
                        series = pd.Series(json.loads(args.bseries.stream.read()), dtype=dtype)
                    series.index = pd.to_datetime(
                        series.index,
                        utc=args.tzaware,
                        format='ISO8601',
                    )
                else:
                    assert args.format == 'tshpack'
                    series = codecs.unpack_series(
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
                            keepnans=args.keepnans,
                            manual=args.supervision
                        )
                except ValueError as err:
                    if err.args[0].startswith('not allowed to'):
                        api.abort(405, err.args[0])
                    raise

                if args.tzaware and args.tzone != 'UTC':
                    diff.index = diff.index.tz_convert(args.tzone)

                if diff is None:
                    # nothing happened
                    # possible cause is sending nans without erasure flag
                    # on creation
                    return no_content()

                return series_response(
                    args.format,
                    diff,
                    tsa.internal_metadata(args.name),
                    200 if exists else 201
                )

            @api.doc(responses={
                204: 'Success',
                404: 'Does not exist',
                405: 'Not allowed',
                409: 'Target already exists',
            })
            @api.expect(rename)
            @onerror
            @required_roles('admin', 'rw')
            def put(self):
                "rename a series"
                args = rename.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')
                if tsa.exists(args.newname):
                    api.abort(409, f'`{args.newname}` does exists')

                try:
                    tsa.rename(args.name, args.newname, args.propagate)
                except ValueError as err:
                    if err.args[0].startswith('not allowed to'):
                        api.abort(405, err.args[0])
                    raise

                return no_content()

            @api.doc(responses={200: 'Got content', 404: 'Does not exist'})
            @api.expect(get)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                """return a series in json format

                The return format is like this:
                {
                 "2023-1-1T00:00:00": 42.5,
                 "2023-1-2T00:00:00": 172.3
                }

                By default one gets the latest version of the series
                over its full horizon.

                By specifying the "insertion_date" argument, one can
                get the series version closest to the provided date
                (in ISO8601 string format).

                The "from_value_date" and "to_value_date" parameters
                allow to restrict the query horizon. The must be
                encoded as ISO8601 dates.

                The "nocache" parameter allows to read a computed
                series by bypassing its cache if it has one.

                The "live" parameter allows to read a computed series
                by getting its cached content (if it has a cache) and
                provide the latest uncached points.

                """
                args = get.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')
                metadata = tsa.internal_metadata(args.name)
                from_value_date, to_value_date = convert_bounds(
                    args.from_value_date,
                    args.to_value_date,
                    args.tzone,
                    metadata['tzaware']
                )

                series = tsa.get(
                    args.name,
                    revision_date=args.insertion_date,
                    from_value_date=from_value_date,
                    to_value_date=to_value_date,
                    nocache=args.nocache,
                    live=args.live,
                    inferred_freq=args.inferred_freq,
                    keepnans=args._keep_nans
                )

                # the fast path will need it
                # also it is read from a cache filled at get time
                # so very cheap call
                series = prune_bounds(
                    series,
                    from_value_date,
                    to_value_date,
                    args.exclude
                )
                if metadata['tzaware'] and args.tzone.upper() != 'UTC':
                    series.index = series.index.tz_convert(args.tzone)

                return series_response(
                    args.format,
                    series,
                    metadata,
                    200
                )

            @api.doc(responses={
                204: 'Sucess',
                404: 'Does not exist',
                405: 'Not allowed'
            })
            @api.expect(delete)
            @onerror
            @required_roles('admin', 'rw')
            def delete(self):
                """delete a series

                Warning: this is an irreversible operation.
                """
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

        @api.doc(responses={204: 'Success', 404: 'Does not exist'})
        @nss.route('/strip')
        class timeseries_strip(Resource):

            @api.expect(strip)
            @onerror
            @required_roles('admin', 'rw')
            def put(self):
                """strip a series

                Remove all versions starting from the "inssertion_date" parameter.
                This is an irreversible operation.
                """
                args = strip.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                tsa.strip(args.name, args.insertion_date)
                return no_content()


        @nss.route('/insertion_dates')
        class timeseries_idates(Resource):

            @api.doc(responses={200: 'Got content', 404: 'Does not exist'})
            @api.expect(insertion_dates)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                """return the revisions of a series

                It comes as a json list of ISO8601 string encoded dates.

                It is possible to restrict the horizon by using the
                "from_insertion_date" / "to_insertion_date" /
                "from_value_date" / "to_value_date" / "limit"
                parameters, all (but limit) encoded as ISO8601 strings.

                The "nocache" parameter allows to bypass the cache of
                a computed series (if it exists) and get the revisions
                of the live formula.

                """
                args = insertion_dates.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                idates = tsa.insertion_dates(
                    args.name,
                    from_insertion_date=args.from_insertion_date,
                    to_insertion_date=args.to_insertion_date,
                    from_value_date=args.from_value_date,
                    to_value_date=args.to_value_date,
                    limit=args.limit,
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
            @required_roles('admin', 'rw', 'ro')
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
                metadata = tsa.internal_metadata(args.name)

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
                    codecs.pack_history(metadata, hist)
                )
                response.headers['Content-Type'] = 'application/octet-stream'
                return response

        @nss.route('/staircase')
        class timeseries_staircase(Resource):

            @api.doc(responses={200: 'Got content', 404: 'Does not exist'})
            @api.expect(staircase)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                args = staircase.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                series = tsa.staircase(
                    args.name, delta=args.delta,
                    from_value_date=args.from_value_date,
                    to_value_date=args.to_value_date,
                )
                metadata = tsa.internal_metadata(args.name)

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
                    codecs.pack_series(metadata, series)
                )
                response.headers['Content-Type'] = 'application/octet-stream'
                return response

        @nss.route('/block_staircase')
        class timeseries_block_staircase(Resource):

            @api.doc(responses={200: 'Got content', 404: 'Does not exist'})
            @api.expect(block_staircase)
            @onerror
            @required_roles('admin', 'rw', 'ro')
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
                metadata = tsa.internal_metadata(args.name)

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
                    codecs.pack_series(metadata, series)
                )
                response.headers['Content-Type'] = 'application/octet-stream'
                return response

        @nss.route('/catalog')
        class timeseries_catalog(Resource):

            @api.doc(responses={200: 'Got content'})
            @api.expect(catalog)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                """returns the series catalog (deprecated)

                This is a deprecated method, you should use "/find" instead.
                """
                args = catalog.parse_args()
                cat = {
                    f'{uri}': series
                    for (uri, ns), series in tsa.catalog(allsources=args.allsources).items()
                }
                return cat

        @nss.route('/find')
        class timeseries_find(Resource):

            @api.doc(responses={200: 'Got content'})
            @api.expect(find)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                """return a list of series descriptor from a filter query

                A filter query is a lisp expression.
                Examples:
                * (by.everything) will return descriptors for all series
                * (by.name ".fcst") will return descriptos for all
                  series whose name contains the ".fcst" string

                The complete description of the filter language can be
                found in the main documentation.

                It is possible to specify a limit argument to limit
                the results. Results are sorted by series name.

                By setting the "meta" argument to true, one gets the
                internal and user metadata in the returned series
                descriptors.

                The series descriptor is an object with fixed fields.
                Without metadata it looks like this:

                {
                 "name": "series0",
                 "imeta": null,
                 "meta": null,
                 "source": "local",
                 "kind": "primary"
                }

                With metadata, we have this:

                {
                 "name": "series0",
                 "imeta": {
                  "tzaware": false,
                  "tablename": "series0",
                  "index_type": "datetime64[ns]",
                  "value_type": "float64",
                  "index_dtype": "<M8[ns]",
                  "value_dtype": "<f8",
                  "supervision_status": "supervised"
                 },
                 "meta": {
                  "foo": "bar"
                 },
                 "source": "local",
                 "kind": "primary"
                }
                """
                args = find.parse_args()
                return [
                    item.to_json()
                    for item in tsa.find(
                            args.query,
                            limit=args.limit,
                            meta=args.meta,
                            sources=args.sources,
                            _source=args._source
                    )
                ]

        @nss.route('/basket')
        class timeseries_basket(Resource):

            @api.expect(basket)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                args = basket.parse_args()
                return [
                    item.to_json()
                    for item in tsa.basket(
                            args.name,
                            limit=args.limit,
                            meta=args.meta,
                            sources=args.sources
                    )
                ]

            @api.expect(register_basket)
            @onerror
            @required_roles('admin', 'rw')
            def put(self):
                args = register_basket.parse_args()
                tsa.register_basket(
                    name=args.name,
                    query=args.query
                )
                return '', 200

            @api.expect(basket)
            @onerror
            @required_roles('admin', 'rw')
            def delete(self):
                args = basket.parse_args()
                return tsa.delete_basket(
                    args.name
                )

        @nss.route('/baskets')
        class timeseries_baskets(Resource):

            @api.expect(nothing)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                return tsa.list_baskets()

        @nss.route('/basket-definition')
        class timeseries_basket_def(Resource):

            @api.expect(basket)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                args = basket.parse_args()
                return tsa.basket_definition(
                    args.name
                )

        @nss.route('/log')
        class series_log(Resource):

            @api.doc(responses={200: 'Got content', 404: 'Does not exist'})
            @api.expect(log)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                """returns the insertion log of a series, as a list.

                Individual items as returned as such:

                {
                 "rev": 2,
                 "author": "webui",
                 "date": "2022-10-27T13:46:34.777338+00:00",
                 "meta": {
                  "edited": true
                 }
                }

                It is possible to specify a limit.

                Also the fromdate/todate parameters allow to restrict
                the versions horizon. Dates should be provided as
                ISO8601 strings.

                """
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

        @nsg.route('/source')
        class timeseries_group_source(Resource):

            @api.doc(responses={200: 'Got content', 404: 'Does not exist'})
            @api.expect(groupsource)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                """returns the source of a group

                If it comes from a secondary source, it returns the source name.
                If it comes from the main source it returns the "local" string.
                """
                args = groupsource.parse_args()
                if not tsa.group_exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                return tsa.group_source(args.name), 200

        @nsg.route('/state')
        class timeseries_group_state(Resource):

            @api.expect(groupupdate)
            @onerror
            @required_roles('admin', 'rw')
            def patch(self):
                args = groupupdate.parse_args()

                df = codecs.unpack_group(
                    args.bgroup.stream.read()
                )

                exists = tsa.group_exists(args.name)
                if args.replace:
                    tsa.group_replace(
                        args.name,
                        df,
                        args.author,
                        insertion_date=args.insertion_date,
                    )
                else:
                    tsa.group_update(
                        args.name,
                        df,
                        args.author,
                        insertion_date=args.insertion_date,
                    )

                return '', 200 if exists else 201

            @api.expect(groupget)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                args = groupget.parse_args()
                metadata = tsa.group_internal_metadata(args.name)
                if metadata is None:
                    api.abort(404, f'`{args.name}` does not exists')
                from_value_date, to_value_date = convert_bounds(
                    args.from_value_date,
                    args.to_value_date,
                    args.tzone,
                    metadata['tzaware']
                )
                df = tsa.group_get(
                    args.name,
                    revision_date=args.insertion_date,
                    from_value_date=from_value_date,
                    to_value_date=to_value_date
                )
                if metadata['tzaware'] and args.tzone.upper() != 'UTC':
                    df.index = df.index.tz_convert(args.tzone)
                return group_response(
                    args.format,
                    df,
                    200
                )

            @api.expect(grouprename)
            @onerror
            @required_roles('admin', 'rw')
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
            @required_roles('admin', 'rw')
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
            @required_roles('admin', 'rw', 'ro')
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
            @required_roles('admin', 'rw', 'ro')
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
                    codecs.pack_group_history(hist)
                )
                response.headers['Content-Type'] = 'application/octet-stream'
                return response

        @nsg.route('/catalog')
        class timeseries_groupcatalog(Resource):

            @api.expect(groupcatalog)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                args = groupcatalog.parse_args()
                cat = {
                    f'{uri}': series
                    for (uri, ns), series in tsa.group_catalog(allsources=args.allsources).items()
                }
                return cat

        @nsg.route('/metadata')
        class timeseries_groupmetadata(Resource):

            @api.expect(groupmetadata)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                args = groupmetadata.parse_args()
                if not tsa.group_exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                if args.type == 'type':
                    stype = tsa.group_type(args.name)
                    return stype, 200

                if args.type == 'archive':
                    metas = [
                        (stamp.isoformat(), meta)
                        for stamp, meta in tsa.group_old_metadata(args.name)
                    ]
                    return metas, 200

                if args.type == 'internal':
                    meta = tsa.group_internal_metadata(args.name)
                    return meta, 200

                assert args.type == 'standard'
                meta = tsa.group_metadata(args.name, all=args.all) or {}
                return meta, 200

            @api.expect(put_groupmetadata)
            @onerror
            @required_roles('admin', 'rw')
            def put(self):
                args = put_groupmetadata.parse_args()
                if not tsa.group_exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                metadata = json.loads(args.metadata)
                try:
                    tsa.replace_group_metadata(args.name, metadata)
                except ValueError as err:
                    if err.args[0].startswith('not allowed to'):
                        api.abort(405, err.args[0])
                    raise

                return '', 200

            @api.expect(put_groupmetadata)
            @onerror
            @required_roles('admin', 'rw')
            def patch(self):
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

        @nsg.route('/log')
        class group_log(Resource):

            @api.doc(responses={200: 'Got content', 404: 'Does not exist'})
            @api.expect(log)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                """returns the insertion log of a group, as a list.

                Individual items as returned as such:

                {
                 "rev": 2,
                 "author": "webui",
                 "date": "2022-10-27T13:46:34.777338+00:00",
                }

                It is possible to specify a limit.

                Also the fromdate/todate parameters allow to restrict
                the versions horizon. Dates should be provided as
                ISO8601 strings.

                """
                args = log.parse_args()
                if not tsa.group_exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                logs = []
                for item in tsa.group_log(
                    args.name,
                    limit=args.limit,
                    fromdate=args.fromdate,
                    todate=args.todate):
                    item['date'] = item['date'].isoformat()
                    logs.append(item)

                return logs, 200

        @nsg.route('/find')
        class group_find(Resource):

            @api.doc(responses={200: 'Got content'})
            @api.expect(groupfind)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                """return a list of group descriptor from a filter query

                A filter query is a lisp expression.
                Examples:
                * (by.everything) will return descriptors for all series
                * (by.name ".fcst") will return descriptos for all
                  series whose name contains the ".fcst" string

                The complete description of the filter language can be
                found in the main documentation.

                It is possible to specify a limit argument to limit
                the results. Results are sorted by series name.

                By setting the "meta" argument to true, one gets the
                internal and user metadata in the returned series
                descriptors.

                The group descriptor is an object with fixed fields.
                Without metadata it looks like this:

                {
                 "name": "series0",
                 "imeta": null,
                 "meta": null,
                 "source": "local",
                 "kind": "primary"
                }

                With metadata, we have this:

                {
                 "name": "series0",
                 "imeta": {
                  "tzaware": false,
                  "tablename": "series0",
                  "index_type": "datetime64[ns]",
                  "value_type": "float64",
                  "index_dtype": "<M8[ns]",
                  "value_dtype": "<f8",
                  "supervision_status": "supervised"
                 },
                 "meta": {
                  "foo": "bar"
                 },
                 "source": "local",
                 "kind": "primary"
                }
                """
                args = find.parse_args()
                return [
                    item.to_json()
                    for item in tsa.group_find(
                            args.query,
                            limit=args.limit,
                            meta=args.meta,
                            _source=args._source
                    )
                ]
