import orjson as json

import pandas as pd
import werkzeug
from flask import (
    Blueprint,
    jsonify,
    make_response,
    request
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

treepath_rename = treepath_delete.copy()
treepath_rename.add_argument(
    'newpath', type=str, required=True,
    help='new path name'
)

treepath_set = base.copy()
treepath_set.add_argument(
    'path', type=str, required=False, default=None,
    help='tree path (None to unset)'
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
basket.add_argument(
    'group', type=inputs.boolean, default=False
)

register_basket = reqparse.RequestParser()
register_basket.add_argument(
    'name', type=str
)
register_basket.add_argument(
    'query', type=str
)
register_basket.add_argument(
    'group', type=inputs.boolean, default=False
)

rename_basket = reqparse.RequestParser()
rename_basket.add_argument(
    'oldname', type=str
)
rename_basket.add_argument(
    'newname', type=str
)
rename_basket.add_argument(
    'group', type=inputs.boolean, default=False
)

nothing = reqparse.RequestParser()

list_baskets = reqparse.RequestParser()
list_baskets.add_argument(
    'group', type=inputs.boolean, default=False
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

            @api.doc(
                responses={200: 'Got content', 400: 'Invalid property'},
                description="""Get global instance properties - multiplexed endpoint

This single route provides access to 2 different global information types via the "property" parameter.

**sources** - List all configured secondary sources
  Returns: list of source names (strings)
  Example: `["meteo", "fundamentals"]`

**info** - Get information about all sources including local
  Returns: dict mapping source names to metadata
  Example:
  ```json
  {
    "local": {
      "type": "primary",
      "series_count": 1234
    },
    "meteo": {
      "type": "remote",
      "uri": "postgresql://meteo-db/tsdb"
    }
  }
  ```

**Parameters:**
- property: "sources" or "info" (required)
"""
            )
            @api.expect(properties)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                args = properties.parse_args()
                if args.property == 'sources':
                    return tsa.sources(), 200
                elif args.property == 'info':
                    return tsa.info(), 200

                # we should never get there
                api.abort(400, 'Asked property does not exist')

        @nss.route('/source')
        class timeseries_source(Resource):

            @api.doc(
                responses={200: 'Got content', 404: 'Does not exist'},
                description="""Return the source location of a series

Identifies which refinery instance stores a series:
- Returns "local" if stored in the main refinery instance
- Returns the source name (e.g., "remote-refinery") if stored in a secondary refinery

**Parameters:**
- name: series name

**Returns:** "local" | <source-name> (str)

**Example:** `?name=my-series`
"""
            )
            @api.expect(source)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                args = source.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                return tsa.source(args.name), 200

        @nss.route('/sources')
        class timeseries_sources(Resource):

            @api.doc(
                responses={200: 'Got content'},
                description="""List configured remote time series sources

Returns all secondary refinery instances configured as data sources.

**Returns:** list of [name, uri] pairs for each configured source

**Example:**
```json
[
  ["remote-prod", "postgresql://prod.example.com/tsdb"],
  ["remote-backup", "postgresql://backup.example.com/tsdb"]
]
```
"""
            )
            @api.expect(nothing)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                return cfg.sources(), 200

        @nss.route('/metadata')
        class timeseries_metadata(Resource):

            @api.doc(
                responses={200: 'Got content',
                           404: 'Does not exist'},
                description="""Get series metadata - multiplexed endpoint

This single route provides access to 6 different metadata operations via the "type" parameter.

**exists** - Test series existence
  Returns: HTTP 404 if series doesn't exist, OR true with HTTP 200 if it exists
  Use: lightweight existence check (client checks HTTP status, not return value)
  Example: `?name=my-series&type=exists`

**type** - Get series implementation type
  Returns: "primary" | "formula" | <plugin-type> (str)
  Use: determine if series is raw data, computed, or plugin-specific
  Example: `?name=my-series&type=type`

**standard** - Get user-defined metadata
  Returns: {key: value, ...} (dict) - may be empty {}
  Use: retrieve arbitrary metadata set by users via update_metadata()
  Example: `?name=my-series&type=standard`
  Deprecated: parameter "all" (formerly merged standard+internal, now use type=internal separately)

**internal** - Get system-managed metadata
  Returns: dict with keys: tzaware (bool), index_type (str), index_dtype (str),
           value_type (str), value_dtype (str), tablename (str),
           left (iso_date), right (iso_date), path (str), etc.
  Use: inspect technical properties (timezone, dtypes, time bounds, storage location)
  Example: `?name=my-series&type=internal`

**archive** - Get metadata modification history
  Returns: [[timestamp_iso, metadata_dict, username], ...] (list of tuples)
  Use: audit trail of metadata changes ordered by modification time (newest first)
  Example: `?name=my-series&type=archive`

**interval** - Get time range of available data
  Returns: [tzaware, min_date_iso, max_date_iso] (tuple)
          OR [tzaware, null, null] if series is empty
          OR HTTP 204 (No Content) on error
  Use: find temporal boundaries without loading series data
  Example: `?name=my-series&type=interval`
"""
            )
            @api.expect(metadata)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
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
                        (stamp.isoformat(), meta, user)
                        for stamp, meta, user in tsa.old_metadata(args.name)
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

            @api.doc(
                responses={
                    200: 'Success',
                    404: 'Does not exist',
                    405: 'Not allowed'
                },
                description="""Replace series metadata

Completely replaces user metadata with new values. Previous metadata is archived and can be retrieved via GET with type=archive.

**Parameters:**
- name: series name
- metadata: JSON string with new metadata dict (values must be scalars)

**Example metadata:**
```json
{
  "unit": "MW",
  "description": "Power production",
  "source": "EDF"
}
```
"""
            )
            @api.expect(put_metadata)
            @onerror
            @required_roles('admin', 'rw')
            def put(self):
                args = put_metadata.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                metadata = json.loads(args.metadata)
                user = request.environ.get('USER')
                try:
                    tsa.replace_metadata(args.name, metadata, user=user)
                except ValueError as err:
                    if err.args[0].startswith('not allowed to'):
                        api.abort(405, err.args[0])
                    raise

                return '', 200

            @api.doc(
                responses={204: 'Success', 404: 'Does not exist'},
                description="""Update series metadata

Updates specific metadata keys while preserving others. Previous metadata state is archived and can be retrieved via GET with type=archive.

**Parameters:**
- name: series name
- metadata: JSON string with metadata dict (only specified keys are updated, values must be scalars)

**Example:** If existing metadata is `{"unit": "MW", "source": "EDF"}` and you send `{"unit": "GW"}`, result will be `{"unit": "GW", "source": "EDF"}`.
"""
            )
            @api.expect(put_metadata)
            @onerror
            @required_roles('admin', 'rw')
            def patch(self):
                args = put_metadata.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                metadata = json.loads(args.metadata)
                user = request.environ.get('USER')
                try:
                    tsa.update_metadata(args.name, metadata, user=user)
                except ValueError as err:
                    if err.args[0].startswith('not allowed to'):
                        api.abort(405, err.args[0])
                    raise

                return no_content()

        @nss.route('/metadata-keys')
        class timeseries_metadata_keys(Resource):

            @api.doc(
                responses={200: 'Got content'},
                description="""List all metadata keys used across all series

Returns a sorted list of all unique metadata key names that exist in the registry.
Useful for metadata discovery and validation.

**Returns:** list of string keys

**Example:**
```json
["author", "category", "description", "unit"]
```
"""
            )
            @api.expect(nothing)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                return tsa.list_metadata_keys()

        @nss.route('/tree-path')
        class timeseries_tree_path(Resource):

            @api.doc(
                responses={200: 'Got content'},
                description="""Navigate series tree paths - bidirectional lookup

Supports two lookup modes via the `type` parameter:
- **pathname**: find series name from tree path
- **seriesname**: find tree path from series name

**Parameters:**
- type: "pathname" or "seriesname"
- name: path or series name (depending on type)

**Returns:** string (series name or path), or null if not found
"""
            )
            @api.expect(treepath)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                args = treepath.parse_args()
                if args.type == 'pathname':
                    return tsa.path_series(args.name)

                assert args.type == 'seriesname'
                return tsa.series_path(args.name)

            @api.doc(
                responses={200: 'Success'},
                description="""Rename a tree path

Renames a tree path and updates all series under that path.

**Parameters:**
- path: current path
- newpath: new path name
"""
            )
            @api.expect(treepath_rename)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def put(self):
                args = treepath_rename.parse_args()
                return tsa.rename_path(args.path, args.newpath)

            @api.doc(
                responses={200: 'Success'},
                description="""Delete a tree path

Removes a path from the tree structure. Series under this path are unaffected but lose their path association.

**Parameters:**
- path: tree path to delete
"""
            )
            @api.expect(treepath_delete)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def delete(self):
                args = treepath_delete.parse_args()
                return tsa.delete_path(args.path)

            @api.doc(
                responses={200: 'Success', 405: 'Invalid path'},
                description="""Set or unset tree path for a series

Assigns a series to a tree path for hierarchical organization.

**Parameters:**
- name: series name
- path: tree path (set to null to unset)
"""
            )
            @api.expect(treepath_set)
            @onerror
            @required_roles('admin', 'rw')
            def patch(self):
                args = treepath_set.parse_args()
                try:
                    return tsa.set_series_path(args.name, args.path)
                except ValueError as err:
                    if 'invalid tree path:' in str(err):
                        api.abort(405, str(err))
                    raise

        @nss.route('/tree')
        class timeseries_tree(Resource):

            @api.doc(
                responses={200: 'Got content'},
                description="""List all tree paths

Returns all available tree paths in the hierarchy. Paths use dot notation (ltree format).

**Returns:** list of path strings (dot-separated)

**Example:**
```json
["energy.electricity", "energy.gas", "weather.temperature"]
```
"""
            )
            @api.expect(nothing)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                return tsa.tree()

        @nss.route('/freq')
        class timeseries_freq(Resource):

            @api.doc(
                responses={200: 'Got content', 404: 'Does not exist'},
                description="""Infer the period frequency of a series with quality indicator

Analyzes the series index to detect its frequency pattern.

**Parameters:**
- name: series name
- revision_date, from_value_date, to_value_date: optional date filters

**Returns:** object with inferred_freq tuple [iso_duration, quality]
- Element 0: frequency as ISO8601 duration string
- Element 1: quality indicator from 0.0 (poor) to 1.0 (perfect)

**Example:**
```json
{
  "inferred_freq": ["P0DT0H0M3600S", "1.0"]
}
```

Returns null if frequency cannot be determined.
"""
            )
            @api.expect(inferred_freq)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
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

            @api.doc(
                responses={
                    200: 'Updated',
                    201: 'Created',
                    405: 'Not allowed'
                },
                description="""Create or update a series with data

Inserts new data points into a series or creates a new series. Returns the diff (changed points).

**Parameters:**
- name: series name
- author: user performing the operation
- series: JSON object mapping timestamps to values (format: {"2023-01-01T00:00:00": 42.5})
- tzaware: true if series has timezone-aware timestamps (required)
- replace: true for replace operation (overwrites), false for update (merges, default)
- insertion_date: override insertion timestamp (ISO8601, optional)
- metadata: initial metadata for new series (JSON string, optional)
- keepnans: preserve NaN values (default: false)
- supervision: mark as manual data entry (default: false)
- dtype: data type for new series (e.g., "float64", optional)

**Returns:** diff series showing only changed points (format=json default)

**Example request:**
```json
{
  "name": "power.prod",
  "author": "operator",
  "tzaware": true,
  "series": {
    "2023-01-01T00:00:00+00:00": 100.5,
    "2023-01-02T00:00:00+00:00": 105.2
  }
}
```
"""
            )
            @api.expect(update)
            @onerror
            @required_roles('admin', 'rw')
            def patch(self):
                args = update.parse_args()
                if args.format == 'json':
                    meta = tsa.internal_metadata(args.name)
                    # creation
                    if not meta:
                        dtype = args.dtype
                    # existing
                    else:
                        dtype = meta and meta['value_type'] or None
                    # data given in parameter
                    if args.series is not None:
                        if isinstance(args.series, str):
                            try:
                                series_data = json.loads(args.series)
                            except json.JSONDecodeError:
                                api.abort(400, 'Invalid JSON format for series data')
                            series = pd.Series(series_data, dtype=dtype)
                        else:
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

                if args.author == 'no-user':
                    user = request.environ.get('USER')
                else:
                    user = args.author
                exists = tsa.exists(args.name)
                try:
                    if args.replace:
                        diff = tsa.replace(
                            args.name, series, user,
                            metadata=args.metadata,
                            insertion_date=args.insertion_date,
                            manual=args.supervision
                        )
                    else:
                        diff = tsa.update(
                            args.name, series, user,
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

            @api.doc(
                responses={
                    204: 'Success',
                    404: 'Does not exist',
                    405: 'Not allowed',
                    409: 'Target already exists',
                },
                description="""Rename a series

Changes the name of an existing series.

**Parameters:**
- name: current series name
- newname: new series name
- propagate: update formula references (default: true)
"""
            )
            @api.expect(rename)
            @onerror
            @required_roles('admin', 'rw')
            def put(self):
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

            @api.doc(
                responses={200: 'Got content', 404: 'Does not exist'},
                description="""Get series data

Returns series data as a mapping of timestamps to values. By default returns the latest version over full horizon.

**Parameters:**
- name: series name
- insertion_date: get version at this timestamp (ISO8601, optional, default: latest)
- from_value_date, to_value_date: restrict time range (ISO8601, optional)
- tzone: timezone for index conversion (default: "UTC")
- nocache: bypass cache for computed series (default: false)
- live: for computed series, combine cached data with live uncached points (default: false)
- inferred_freq: include inferred frequency metadata (default: false)
- format: "json" or "tshpack" (default: json)

**Returns (format=json):** object mapping timestamps to values

**Example:**
```json
{
  "2023-01-01T00:00:00+00:00": 100.5,
  "2023-01-02T00:00:00+00:00": 105.2,
  "2023-01-03T00:00:00+00:00": 98.7
}
```
"""
            )
            @api.expect(get)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
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

            @api.doc(
                responses={
                    204: 'Success',
                    404: 'Does not exist',
                    405: 'Not allowed'
                },
                description="""Delete a series

Permanently removes a series and all its revisions. This is an irreversible operation.

**Parameters:**
- name: series name

**Warning:** This deletes all historical data and metadata. The operation cannot be undone.
"""
            )
            @api.expect(delete)
            @onerror
            @required_roles('admin', 'rw')
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

        @api.doc(responses={204: 'Success', 404: 'Does not exist'})
        @nss.route('/strip')
        class timeseries_strip(Resource):

            @api.doc(
                responses={204: 'Success', 404: 'Does not exist'},
                description="""Strip series revisions from a given date

Removes all revisions starting from the specified insertion date. This is an irreversible operation.

**Parameters:**
- name: series name
- insertion_date: remove all revisions from this date onward (ISO8601)

**Warning:** This permanently deletes data and cannot be undone.
"""
            )
            @api.expect(strip)
            @onerror
            @required_roles('admin', 'rw')
            def put(self):
                args = strip.parse_args()
                if not tsa.exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                tsa.strip(args.name, args.insertion_date)
                return no_content()


        @nss.route('/insertion_dates')
        class timeseries_idates(Resource):

            @api.doc(
                responses={200: 'Got content', 404: 'Does not exist'},
                description="""Get all revision timestamps for a series

Returns the list of insertion dates (revision history) for a series.

**Parameters:**
- name: series name
- from_insertion_date, to_insertion_date: filter by insertion date range (ISO8601, optional)
- from_value_date, to_value_date: filter by value date range (ISO8601, optional)
- limit: maximum number of dates to return (optional)
- nocache: bypass cache for computed series (default: false)

**Returns:** object with insertion_dates array

**Example:**
```json
{
  "insertion_dates": [
    "2022-01-15T10:30:00+00:00",
    "2022-01-16T14:20:00+00:00",
    "2022-01-17T09:15:00+00:00"
  ]
}
```
"""
            )
            @api.expect(insertion_dates)
            @onerror
            @required_roles('admin', 'rw', 'ro')
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

            @api.doc(
                responses={200: 'Got content', 404: 'Does not exist'},
                description="""Get complete revision history for a series

Returns all revisions of a series, optionally in diff mode.

**Parameters:**
- name: series name
- from_insertion_date, to_insertion_date: filter by insertion date range (ISO8601, optional)
- from_value_date, to_value_date: filter by value date range (ISO8601, optional)
- diffmode: return diffs between revisions instead of full series (default: false)
- nocache: bypass cache for computed series (default: false)
- format: "json" or "tshpack" (default: json)

**Returns (format=json):** dict mapping insertion dates to series data

**Example:**
```json
{
  "2022-01-15T10:30:00+00:00": {
    "2022-01-01T00:00:00": 100.0,
    "2022-01-02T00:00:00": 105.0
  },
  "2022-01-16T14:20:00+00:00": {
    "2022-01-01T00:00:00": 100.0,
    "2022-01-02T00:00:00": 105.0,
    "2022-01-03T00:00:00": 110.0
  }
}
```
"""
            )
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

            @api.doc(
                responses={200: 'Got content', 404: 'Does not exist'},
                description="""Build a staircase series for vintage data analysis

Constructs a series where each value is constrained by a time delta after its insertion date.
Useful for analyzing how data looked at different points in time.

**Parameters:**
- name: series name
- delta: time offset from insertion date (e.g., "1 day")
- from_value_date, to_value_date: optional date range filters
"""
            )
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

            @api.doc(
                responses={200: 'Got content', 404: 'Does not exist'},
                description="""Build a block staircase series with revision and maturity parameters

Advanced staircase construction for block-structured time series with configurable revision patterns and maturity offsets.

**Parameters:**
- name: series name
- from_value_date, to_value_date: optional date range filters
- revision_freq: revision frequency as dict (e.g., {"days": 1})
- revision_time: time of day for revisions as dict (e.g., {"hour": 9})
- revision_tz: timezone for revision times (default: "UTC")
- maturity_offset: maturity offset as dict
- maturity_time: time of day for maturity
"""
            )
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

            @api.doc(
                responses={200: 'Got content'},
                description="""Return the series catalog (deprecated)

This is a deprecated method, you should use "/find" instead.
"""
            )
            @api.expect(catalog)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                args = catalog.parse_args()
                cat = {
                    f'{uri}': series
                    for (uri, ns), series in tsa.catalog(allsources=args.allsources).items()
                }
                return cat

        @nss.route('/find')
        class timeseries_find(Resource):

            @api.doc(
                responses={200: 'Got content'},
                description="""Search for series using filter query language

Query series using lisp-like filter expressions. Results are sorted by series name.

**Parameters:**
- query: lisp filter expression (see examples below)
- limit: maximum number of results (optional)
- meta: include internal and user metadata in results (default: false)
- sources: filter by specific sources (optional)

**Query Examples:**
- `(by.everything)` - return all series
- `(by.name ".fcst")` - series whose name contains ".fcst"
- `(by.key "unit" "MW")` - series with metadata unit=MW

See main documentation for complete filter language reference.

**Returns:** list of series descriptors

**Descriptor format without metadata:**
```json
{
  "name": "series0",
  "imeta": null,
  "meta": null,
  "source": "local",
  "kind": "primary"
}
```

**Descriptor format with metadata (meta=true):**
```json
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
```
"""
            )
            @api.expect(find)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
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

            @api.doc(
                responses={200: 'Got content'},
                description="""Get series descriptors from a named basket

Returns series matching a basket's stored query. Baskets are saved queries for easy reuse.

**Parameters:**
- name: basket name
- limit: maximum results (optional)
- meta: include metadata in descriptors (default: false)
- sources: filter by sources (optional)
- group: basket is for groups (default: false)

**Returns:** list of series descriptors (same format as /find)

**Without metadata:**
```json
[{
  "name": "series0",
  "imeta": null,
  "meta": null,
  "source": "local",
  "kind": "primary"
}]
```

**With metadata (meta=true):**
```json
[{
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
    "unit": "MW",
    "description": "Power output"
  },
  "source": "local",
  "kind": "primary"
}]
```
"""
            )
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
                            sources=args.sources,
                            group=args.group
                    )
                ]

            @api.doc(
                responses={200: 'Success'},
                description="""Create or update a basket

Registers a basket with a name and filter query for reuse.

**Parameters:**
- name: basket name
- query: filter query in lisp format (e.g., "(by.name \".prod\")")
- group: basket is for groups instead of series (default: false)
"""
            )
            @api.expect(register_basket)
            @onerror
            @required_roles('admin', 'rw')
            def put(self):
                args = register_basket.parse_args()
                tsa.register_basket(
                    name=args.name,
                    query=args.query,
                    group=args.group
                )
                return '', 200

            @api.doc(
                responses={
                    200: 'Success',
                    404: 'Does not exist',
                    409: 'Target already exists'
                },
                description="""Rename a basket

**Parameters:**
- oldname: current basket name
- newname: new basket name
- group: basket is for groups (default: false)
"""
            )
            @api.expect(rename_basket)
            @onerror
            @required_roles('admin', 'rw')
            def patch(self):
                args = rename_basket.parse_args()
                if tsa.basket_definition(args.oldname, group=args.group) is None:
                    api.abort(404, f'basket `{args.oldname}` does not exist')
                if tsa.basket_definition(args.newname, group=args.group) is not None:
                    api.abort(409, f'basket `{args.newname}` already exists')
                tsa.rename_basket(
                    oldname=args.oldname,
                    newname=args.newname,
                    group=args.group
                )
                return '', 200

            @api.doc(
                responses={200: 'Success'},
                description="""Delete a basket

Removes a basket definition. Does not affect the series it references.

**Parameters:**
- name: basket name
- group: basket is for groups (default: false)
"""
            )
            @api.expect(basket)
            @onerror
            @required_roles('admin', 'rw')
            def delete(self):
                args = basket.parse_args()
                return tsa.delete_basket(
                    args.name,
                    group=args.group
                )

        @nss.route('/baskets')
        class timeseries_baskets(Resource):

            @api.doc(
                responses={200: 'Got content'},
                description="""List all basket names

Returns names of all registered baskets.

**Parameters:**
- group: list group baskets instead of series baskets (default: false)

**Returns:** list of basket names

**Example:**
```json
["production", "forecasts", "validated"]
```
"""
            )
            @api.expect(list_baskets)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                args = list_baskets.parse_args()
                return tsa.list_baskets(group=args.group)

        @nss.route('/basket-definition')
        class timeseries_basket_def(Resource):

            @api.doc(
                responses={200: 'Got content'},
                description="""Get basket query definition

Returns the filter query associated with a basket.

**Parameters:**
- name: basket name
- group: basket is for groups (default: false)

**Returns:** query string (lisp expression)

**Example:**
```json
"(by.name \".prod\")"
```
"""
            )
            @api.expect(basket)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                args = basket.parse_args()
                return tsa.basket_definition(
                    args.name,
                    group=args.group
                )

        @nss.route('/log')
        class series_log(Resource):

            @api.doc(
                responses={200: 'Got content', 404: 'Does not exist'},
                description="""Get insertion log history for a series

Returns the list of all insertion operations on a series with revision metadata.

**Parameters:**
- name: series name
- limit: maximum number of entries (optional)
- fromdate: start date filter (ISO8601)
- todate: end date filter (ISO8601)

**Returns:** list of log entries, each containing:
- rev: revision number
- author: user who made the insertion
- date: insertion timestamp (ISO8601)
- meta: insertion metadata (e.g., edited flag)

**Example:**
```json
[{
  "rev": 2,
  "author": "webui",
  "date": "2022-10-27T13:46:34.777338+00:00",
  "meta": {
    "edited": true
  }
}]
```
"""
            )
            @api.expect(log)
            @onerror
            @required_roles('admin', 'rw', 'ro')
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

        @nsg.route('/source')
        class timeseries_group_source(Resource):

            @api.doc(
                responses={200: 'Got content', 404: 'Does not exist'},
                description="""Return the source location of a group

Identifies which refinery instance stores a group:
- Returns "local" if stored in the main refinery instance
- Returns the source name (e.g., "remote-refinery") if stored in a secondary refinery

**Parameters:**
- name: group name

**Returns:** source name string ("local" or configured source name)
"""
            )
            @api.expect(groupsource)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                args = groupsource.parse_args()
                if not tsa.group_exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                return tsa.group_source(args.name), 200

        @nsg.route('/state')
        class timeseries_group_state(Resource):

            @api.doc(
                responses={200: 'Updated', 201: 'Created'},
                description="""Create or update a group of series

A group is a DataFrame where each column is a time series. Returns empty body (unlike series which return a diff).

**Parameters:**
- name: group name
- author: user creating/updating the group
- bgroup: binary-encoded DataFrame (format: tshpack)
- insertion_date: override insertion timestamp (ISO8601, optional)
- replace: true for replace operation, false for update (default)

**Returns:** empty body with HTTP 201 on creation, 200 on update
"""
            )
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

            @api.doc(
                responses={200: 'Got content', 404: 'Does not exist'},
                description="""Get a group of series

Returns a group as a dict of series (column_name → {timestamp → value}).

**Parameters:**
- name: group name
- insertion_date: retrieve version at this timestamp (ISO8601, optional)
- from_value_date, to_value_date: date range filters (ISO8601, optional)
- tzone: timezone for index conversion (default: "UTC")
- format: "json" or "tshpack" (default: json)

**Returns (format=json):**
```json
{
  "column_a": {
    "2021-01-01T00:00:00": 2.0,
    "2021-01-02T00:00:00": 3.0
  },
  "column_b": {
    "2021-01-01T00:00:00": 5.0,
    "2021-01-02T00:00:00": 6.0
  }
}
```
"""
            )
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

            @api.doc(
                responses={204: 'Success',
                           404: 'Does not exist',
                           409: 'Target already exists'},
                description="""Rename a group

Changes the name of an existing group.

**Parameters:**
- name: current group name
- newname: new group name
"""
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

            @api.doc(
                responses={204: 'Success', 404: 'Does not exist'},
                description="""Delete a group

Permanently removes a group. This is an irreversible operation.

**Parameters:**
- name: group name
"""
            )
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

            @api.doc(
                responses={200: 'Got content',
                           404: 'Does not exist'},
                description="""Get revision history timestamps for a group

Returns all insertion dates (revisions) for a group.

**Parameters:**
- name: group name
- from_insertion_date, to_insertion_date: filter by insertion date range (ISO8601, optional)

**Returns:** object with insertion_dates array

**Example:**
```json
{
  "insertion_dates": [
    "2022-03-01T00:00:00+00:00",
    "2022-03-02T00:00:00+00:00"
  ]
}
```
"""
            )
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

            @api.doc(
                responses={200: 'Got content',
                           404: 'Does not exist'},
                description="""Get complete revision history for a group

Returns all revisions of a group in binary format (tshpack).

**Parameters:**
- name: group name
- from_insertion_date, to_insertion_date: filter by insertion date range (ISO8601, optional)
- from_value_date, to_value_date: filter by value date range (ISO8601, optional)

**Returns:** binary-encoded history (format: tshpack, application/octet-stream)
"""
            )
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

            @api.doc(
                responses={200: 'Got content'},
                description="""List all groups (deprecated)

Returns dict mapping source URI to list of column names for each group.

**Deprecated:** This endpoint is deprecated. Use `/group/find` with `(by.everything)` instead.

**Parameters:**
- allsources: include groups from all configured sources (default: true)
"""
            )
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

            @api.doc(
                responses={200: 'Got content',
                           404: 'Does not exist'},
                description="""Get group metadata - multiplexed endpoint

This single route provides access to 4 different metadata operations via the "type" parameter.

**type** - Get group type (e.g., "primary", "bound")
  Returns: string with group type

**standard** - Get user-defined metadata (default if type not specified)
  Returns: dict of user metadata key-value pairs

**internal** - Get system metadata (tzaware, tablename, value_type, etc.)
  Returns: dict of internal metadata

**archive** - Get historical metadata changes
  Returns: list of [timestamp, metadata, user] tuples

**Parameters:**
- name: group name
- type: operation type (default: "standard")
"""
            )
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
                        (stamp.isoformat(), meta, user)
                        for stamp, meta, user in tsa.group_old_metadata(args.name)
                    ]
                    return metas, 200

                if args.type == 'internal':
                    meta = tsa.group_internal_metadata(args.name)
                    return meta, 200

                assert args.type == 'standard'
                meta = tsa.group_metadata(args.name, all=args.all) or {}
                return meta, 200

            @api.doc(
                responses={200: 'Success', 404: 'Does not exist', 405: 'Not allowed'},
                description="""Replace group metadata

Completely replaces user metadata with new values. Previous metadata is archived and can be retrieved via GET with type=archive.

**Parameters:**
- name: group name
- metadata: JSON string with new metadata dict
"""
            )
            @api.expect(put_groupmetadata)
            @onerror
            @required_roles('admin', 'rw')
            def put(self):
                args = put_groupmetadata.parse_args()
                if not tsa.group_exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                metadata = json.loads(args.metadata)
                user = request.environ.get('USER')
                try:
                    tsa.replace_group_metadata(args.name, metadata, user=user)
                except ValueError as err:
                    if err.args[0].startswith('not allowed to'):
                        api.abort(405, err.args[0])
                    raise

                return '', 200

            @api.doc(
                responses={200: 'Success',
                           404: 'Does not exist',
                           405: 'Not allowed'},
                description="""Update group metadata

Updates specific metadata keys while preserving others. Previous metadata state is archived and can be retrieved via GET with type=archive.

**Parameters:**
- name: group name
- metadata: JSON string with metadata dict (only specified keys are updated)
"""
            )
            @api.expect(put_groupmetadata)
            @onerror
            @required_roles('admin', 'rw')
            def patch(self):
                args = put_groupmetadata.parse_args()
                if not tsa.group_exists(args.name):
                    api.abort(404, f'`{args.name}` does not exists')

                metadata = json.loads(args.metadata)
                user = request.environ.get('USER')
                try:
                    tsa.update_group_metadata(args.name, metadata, user=user)
                except ValueError as err:
                    if err.args[0].startswith('not allowed to'):
                        api.abort(405, err.args[0])
                    raise

                return '', 200

        @nsg.route('/log')
        class group_log(Resource):

            @api.doc(
                responses={200: 'Got content', 404: 'Does not exist'},
                description="""Get insertion log history for a group

Returns the list of all insertion operations on a group with revision metadata.

**Parameters:**
- name: group name
- limit: maximum number of entries (optional)
- fromdate: start date filter (ISO8601)
- todate: end date filter (ISO8601)

**Returns:** list of log entries, each containing:
- rev: revision number
- author: user who made the insertion
- date: insertion timestamp (ISO8601)

**Example:**
```json
[{
  "rev": 2,
  "author": "webui",
  "date": "2022-10-27T13:46:34.777338+00:00"
}]
```
"""
            )
            @api.expect(log)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
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

            @api.doc(
                responses={200: 'Got content'},
                description="""Search for groups using filter query language

Query groups using lisp-like filter expressions. Results are sorted by group name.

**Parameters:**
- query: lisp filter expression (see examples below)
- limit: maximum number of results (optional)
- meta: include internal and user metadata in results (default: false)
- sources: filter by specific sources (optional)

**Query Examples:**
- `(by.everything)` - return all groups
- `(by.name ".prod")` - groups whose name contains ".prod"
- `(by.key "category" "forecast")` - groups with metadata category=forecast

See main documentation for complete filter language reference.

**Returns:** list of group descriptors

**Descriptor format without metadata:**
```json
{
  "name": "group0",
  "imeta": null,
  "meta": null,
  "source": "local",
  "kind": "primary"
}
```

**Descriptor format with metadata (meta=true):**
```json
{
  "name": "group0",
  "imeta": {
    "tzaware": false,
    "tablename": "group0",
    "index_type": "datetime64[ns]",
    "value_type": "float64",
    "index_dtype": "<M8[ns]",
    "value_dtype": "<f8"
  },
  "meta": {
    "category": "forecast",
    "unit": "MW"
  },
  "source": "local",
  "kind": "primary"
}
```
"""
            )
            @api.expect(find)
            @onerror
            @required_roles('admin', 'rw', 'ro')
            def get(self):
                args = find.parse_args()
                return [
                    item.to_json()
                    for item in tsa.group_find(
                            args.query,
                            limit=args.limit,
                            meta=args.meta,
                            sources=args.sources,
                            _source=args._source
                    )
                ]
