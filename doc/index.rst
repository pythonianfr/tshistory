.. toctree::
   :maxdepth: 2
   :caption: Contents:

Welcome to tshistory's documentation!
=====================================

This is a library to store/retrieve pandas timeseries to/from a
postgres database, tracking their successive versions.

Introduction
............

Purpose
-------

`tshistory` is targetted at applications using time series where
`backtesting`_ and `cross-validation`_ are an essential feature.

It provides exhaustivity and efficiency of the storage, with a simple
Python api.

It can be used as a building block for machine learning, model
optimization and validation, both for inputs and outputs.


## Principles

There are many ways to represent timeseries in a relational database,
and `tshistory` provides two things:

* a base python API which abstracts away the underlying storage

* a postgres model, which emphasizes the compact storage of successive
  states of series

The core idea of tshistory is to handle successive versions of
timeseries as they grow in time, allowing to get older states of any
series.


Basic usage
...........

Starting with a fresh database
------------------------------

You need a postgresql database. You can create one like this:

.. code-block:: bash

   createdb mydb

Then, initialize the `tshistory` tables, like this:

.. code-block:: bash

   tsh init-db postgresql://me:password@localhost/mydb

From this you're ready to go !


Creating a series
-----------------

However here's a simple example:

.. code-block:: python

 >>> import pandas as pd
 >>> from tshistory.api import timeseries
 >>>
 >>> tsa = timeseries('postgres://me:password@localhost/mydb')
 >>>
 >>> series = pd.Series([1, 2, 3],
 ...                    pd.date_range(start=pd.Timestamp(2017, 1, 1),
 ...                                  freq='D', periods=3))
 # db insertion
 >>> tsa.update('my_series', series, 'babar@pythonian.fr')
 ...
 2017-01-01    1.0
 2017-01-02    2.0
 2017-01-03    3.0
 Freq: D, Name: my_series, dtype: float64

 # note how our integers got turned into floats
 # (there are no provisions to handle integer series as of today)

 # retrieval
 >>> tsa.get('my_series')
 ...
 2017-01-01    1.0
 2017-01-02    2.0
 2017-01-03    3.0
 Name: my_series, dtype: float64

Note that we generally adopt the convention to name the time series
api object `tsa`.


Updating a series
-----------------

This is good. Now, let's insert more:

.. code-block:: python

 >>> series = pd.Series([2, 7, 8, 9],
 ...                    pd.date_range(start=pd.Timestamp(2017, 1, 2),
 ...                                  freq='D', periods=4))
 # db insertion
 >>> tsa.update('my_series', series, 'babar@pythonian.fr')
 ...
 2017-01-03    7.0
 2017-01-04    8.0
 2017-01-05    9.0
 Name: my_series, dtype: float64

 # you get back the *new information* you put inside
 # and this is why the `2` doesn't appear (it was already put
 # there in the first step)

 # db retrieval
 >>> tsa.get('my_series')
 ...
 2017-01-01    1.0
 2017-01-02    2.0
 2017-01-03    7.0
 2017-01-04    8.0
 2017-01-05    9.0
 Name: my_series, dtype: float64

It is important to note that the third value was *replaced*, and the two
last values were just *appended*.

As noted the point at `2017-1-2` wasn't a new information so it was
just ignored.


Retrieving history
------------------

We can access the whole history (or parts of it) in one call:

.. code-block:: python

 >>> history = tsa.history('my_series')
 ...
 >>>
 >>> for idate, series in history.items(): # it's a dict
 ...     print('insertion date:', idate)
 ...     print(series)
 ...
 insertion date: 2018-09-26 17:10:36.988920+02:00
 2017-01-01    1.0
 2017-01-02    2.0
 2017-01-03    3.0
 Name: my_series, dtype: float64
 insertion date: 2018-09-26 17:12:54.508252+02:00
 2017-01-01    1.0
 2017-01-02    2.0
 2017-01-03    7.0
 2017-01-04    8.0
 2017-01-05    9.0
 Name: my_series, dtype: float64


Note how this shows the full serie state for each insertion date.
Also the insertion date is timzeone aware.

It is possible to show the differences only:

.. code-block:: python

 >>> diffs = tsa.history('my_series', diffmode=True)
 ...
 >>> for idate, series in diffs.items():
 ...   print('insertion date:', idate)
 ...   print(series)
 ...
 insertion date: 2018-09-26 17:10:36.988920+02:00
 2017-01-01    1.0
 2017-01-02    2.0
 2017-01-03    3.0
 Name: my_series, dtype: float64
 insertion date: 2018-09-26 17:12:54.508252+02:00
 2017-01-03    7.0
 2017-01-04    8.0
 2017-01-05    9.0
 Name: my_series, dtype: float64


You can see a series metadata:

.. code-block:: python

 >>> tsa.metadata('series', internal=True)
 {'tzaware': False, 'index_type': 'datetime64[ns]', 'value_type': 'float64',
 'index_dtype': '<M8[ns]', 'value_dtype': '<f8'}


We built a series with naive time stamps, but timezone-aware
timestamps work well (and it is advised to use them !).


The API object
..............

In the few examples above we manipulate the time series through an
object that talks directly to the postgresql back end.

It is possible to also talk to a rest api using the same api, like
shown below and proceed exactly like in the above code examples:

.. code-block:: python

 >>> from tshistory.api import timeseries
 >>>
 >>> tsa = timeseries('http://my.timeseries.info/api')


Using an HTTP/REST end point
----------------------------

For this to work, one needs to use the the `tshistory_rest`_ and
`tshistory_client`_ packages.

The client package will be used transparently on a
`timeseries('http://.../api')` call, nothing more needs to be done
than install it.

For the rest api, you need to build a small [flask][flask] app and set
up the `tshistory_rest` blueprint like this (in an `app.py` module):

.. code-block:: python

 from flask import Flask

 from tshistory.api import timeseries
 from tshistory_rest.blueprint import blueprint as blueprint


 def make_app(dburi):
     app = Flask('my-timeseries-app')
     app.register_blueprint(
         blueprint(timeseries(dburi)),
         url_prefix='/api'
     )
     return app


Then, you can start it in development mode like this:

.. code-block:: python

 app = make_app('postgresql://me:password@localhost/mydb')
 app.run('192.168.1.1', 8080)


or just leave it to a wsgi container in e.g. a `wsgi.py` module:

.. code-block:: python

 from my_series_app.app import make_app

 app = make_app('postgresql://me:password@localhost/mydb')


Command line
............

Basic operations
----------------

A command line tool is provided, called `tsh`. It provides its usage
guidelines:

.. code-block:: bash

 $ tsh
 Usage: tsh [OPTIONS] COMMAND [ARGS]...

 Options:
   --help  Show this message and exit.

 Commands:
   check    coherence checks of the db
   get      show a serie in its current state
   history  show a serie full history
   info     show global statistics of the repository
   init-db  initialize an new db.
   log      show revision history of entire repository or...
   view     visualize time series through the web


`Info` provides an overview of the time series repository (number of
committed changes, number and series and their names).

.. code-block:: bash

 $ tsh info postgres://babar:babarpassword@dataserver:5432/banana_studies
 changeset count: 209
 series count:    144
 series names:    banana_spot_price, banana_trades, banana_turnover


`Log` provides the full history of editions to time series in the
repository.

.. code-block:: bash

 $ tsh log postgres://babar:babar@dataserver:5432/banana_studies --limit 3
 revision: 206
 author:   BABAR
 date:     2017-06-06 15:32:51.502507
 series:   banana_spot_price

 revision: 207
 author:   BABAR
 date:     2017-06-06 15:32:51.676507
 series:   banana_trades

 revision: 209
 author:   CELESTE
 date:     2017-06-06 15:32:51.977507
 series:   banana_turnover


All options of all commands can be obtained by using the `--help`
switch:

.. code-block:: bash

 $ tsh log --help
 Usage: tsh log [OPTIONS] DB_URI

 Options:
   -l, --limit TEXT
   --show-diff
   -s, --serie TEXT
   --from-rev TEXT
   --to-rev TEXT
   --help            Show this message and exit.



Extensions
..........

It is possible to augment the `tsh` command with new subcommands (or
augment, modify existing commands).

Any program doing so must define a new command and declare a setup
tools entry point named `tshistory:subcommand` as in e.g.:

.. code-block:: python

    entry_points={'tshistory.subcommands': [
        'view=tsview.command:view'
    ]}


For instance, the `tsview`_ python package provides such a
`view` subcommand for generic time series visualisation.

.. _tsview: https://bitbucket.org/pythonian/tsview
.. _tshistory_rest: https://bitbucket.org/pythonian/tshistory_rest
.. _tshistory_client: https://bitbucket.org/pythonian/tshistory_client
.. _backtesting: https://en.wikipedia.org/wiki/Backtesting
.. _cross-validation: https://en.wikipedia.org/wiki/Cross-validation_(statistics)
.. _flask: https://www.palletsprojects.com/p/flask/


Status
......

It is currently considered `beta` software even though it has been in
production for two years. It is still evolving. Schema/Database
changes come with migration procedure using the `tsh` utility.

When it is good: if you do mostly appends (and occasional edits in the
past) it will store data in a very compact way.

Reading any version of the series will always be the fastest (io-bound)
operation.

Alternative backend storage and storage strategies will be considered
in the future.


API Reference
=============

.. automodule:: tshistory.api
   :members: dbtimeseries

