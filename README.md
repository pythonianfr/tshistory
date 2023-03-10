TSHISTORY
=========

This is a library to store/retrieve pandas timeseries to/from a
postgres database, tracking their successive versions.

# Introduction

## Purpose

`tshistory` is targetted at applications using time series where
[backtesting][backtesting] and [cross-validation][cross-validation]
are an essential feature.

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


# Basic usage

## Starting with a fresh database

You need a postgresql database. You can create one like this:

```shell
 createdb mydb
```

Then, initialize the `tshistory` tables, like this:

```python
 tsh init-db postgresql://me:password@localhost/mydb
```

From this you're ready to go !


## Creating a series

However here's a simple example:

```python
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
```

Note that we generally adopt the convention to name the time series
api object `tsa`.


## Updating a series

This is good. Now, let's insert more:

```python
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
```

It is important to note that the third value was *replaced*, and the two
last values were just *appended*. As noted the point at `2017-1-2` wasn't a new information so it was
just ignored.

## Retrieving history

We can access the whole history (or parts of it) in one call:

```python
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
```

Note how this shows the full serie state for each insertion date. Also the insertion date is timzeone aware.

Specific versions of a series can be retrieved individually using the `get` method as follows:
```python
 >>> tsa.get('my_series', revision_date=pd.Timestamp('2018-09-26 17:11+02:00'))
 ...
 2017-01-01    1.0
 2017-01-02    2.0
 2017-01-03    3.0
 Name: my_series, dtype: float64
 >>>
 >>> tsa.get('my_series', revision_date=pd.Timestamp('2018-09-26 17:14+02:00'))
 ...
 2017-01-01    1.0
 2017-01-02    2.0
 2017-01-03    7.0
 2017-01-04    8.0
 2017-01-05    9.0
 Name: my_series, dtype: float64
```


It is possible to retrieve only the differences between successive insertions:

```python
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
```

You can see a series metadata:

```python
 >>> tsa.update_metadata('series', {'foo': 42})
 >>> tsa.metadata('series')
 {foo: 42}
```


## Staircase series

A staircase series can be defined as a series of which values originate from successive
revisions with a fixed time span between revision date and value date. This is
especially useful for backtesting.

### Basic staircase

Let us take an example assuming a series called `daily_series` has been created with
insertions given by the following table (considering row indices are value dates, and
columns indices are insertion dates):

|            | 2020-01-01<br>00:00+00 | 2020-01-02<br>00:00+00 | 2020-01-03<br>00:00+00 |
|-----------:|:----------------------:|:----------------------:|:----------------------:|
| 2020-01-01 | 1.1                    |                        |                        |
| 2020-01-02 | 2.1                    | 2.2                    |                        |
| 2020-01-03 | 3.1                    | 3.2                    | 3.3                    |
| 2020-01-04 |                        | 4.2                    | 4.3                    |
| 2020-01-05 |                        |                        | 5.3                    |

Supposing this series is a forecast published on a daily basis, we can for example
reconstruct the day-ahead forecast series, i.e. the values such that the time span
between revision date and value date is 1 day (or more) as follows:
```python
 >>> tsa.staircase('daily_series',
                   from_value_date=pd.Timestamp('2020-01-01'),
                   to_value_date=pd.Timestamp('2020-01-07'),
                   delta=pd.Timedelta(days=1))
 ...
 2020-01-02    2.1
 2020-01-03    3.2
 2020-01-04    4.3
 2020-01-05    5.3
 Name: daily_series, dtype: float64
```

The name "staircase" refers to the way in which these values are picked from the history:

|            | 2020-01-01<br>00:00+00 | 2020-01-02<br>00:00+00 | 2020-01-03<br>00:00+00 |
|-----------:|:----------------------:|:----------------------:|:----------------------:|
| 2020-01-01 |                        |                        |                        |
| 2020-01-02 | **2.1**                |                        |                        |
| 2020-01-03 |                        | **3.2**                |                        |
| 2020-01-04 |                        |                        | **4.3**                |
| 2020-01-05 |                        |                        | **5.3**                |


Now if instead we consider an hourly forecast series, we may want to define day-ahead
forecast as a staircase series with a daily revision occurring at 9am, and link each
revision to the 24 hours of the next day. More generally we may want to reconstruct a
staircase series where successive revisions each relate to several value dates. Such
cases should instead be handled using the `block_staircase` method described below.

### Block staircase

Let us take another example considering the series `hourly_series` with following insertions:

|                     | 2020-01-01<br>06:00+00 | 2020-01-01<br>14:00+00 | 2020-01-02<br>06:00+00 | 2020-01-02<br>14:00+00 |
|--------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|
| 2020-01-01 00:00+00 | 1.1                    | 1.2                    |                        |                        |
| 2020-01-01 08:00+00 | 2.1                    | 2.2                    |                        |                        |
| 2020-01-01 16:00+00 | 3.1                    | 3.2                    |                        |                        |
| 2020-01-02 00:00+00 | 4.1                    | 4.2                    | 4.3                    | 4.4                    |
| 2020-01-02 08:00+00 | 5.1                    | 5.2                    | 5.3                    | 5.4                    |
| 2020-01-02 16:00+00 | 6.1                    | 6.2                    | 6.3                    | 6.4                    |
| 2020-01-03 00:00+00 | 7.1                    | 7.2                    | 7.3                    | 7.4                    |
| 2020-01-03 08:00+00 | 8.1                    | 8.2                    | 8.3                    | 8.4                    |
| 2020-01-03 16:00+00 | 9.1                    | 9.2                    | 9.3                    | 9.4                    |
| 2020-01-04 00:00+00 |                        |                        | 10.3                   | 10.4                   |
| 2020-01-04 08:00+00 |                        |                        | 11.3                   | 11.4                   |
| 2020-01-04 16:00+00 |                        |                        | 12.3                   | 12.4                   |

Then the day-ahead forecast with revisions at 9am can be computed as follows:
```python
 >>> tsa.block_staircase('hourly_series',
                         from_value_date=pd.Timestamp('2020-01-01', tz="utc"),
                         to_value_date=pd.Timestamp('2020-01-05', tz="utc"),
                         revision_freq={'days': 1},
                         revision_time={'hour': 9},
                         revision_tz='utc',
                         maturity_offset={'days': 1},
                         maturity_time={'hour': 0})
 ...
 2020-01-02 00:00:00+00:00   4.1
 2020-01-02 08:00:00+00:00   5.1
 2020-01-02 16:00:00+00:00   6.1
 2020-01-03 00:00:00+00:00   7.3 
 2020-01-03 08:00:00+00:00   8.3 
 2020-01-03 16:00:00+00:00   9.3 
 2020-01-04 00:00:00+00:00   10.4
 2020-01-04 08:00:00+00:00   11.4
 2020-01-04 16:00:00+00:00   12.4
 Name: hourly_series, dtype: float64
```

Note that with `revision_time={'hour': 9}`, the method ends up picking values from the
two 6am insertions (except for the values of 2020-01-04 when latest available revision
is 2020-01-02 14:00). Taking revision time after 2pm, say
`revision_time={'hour': 20}`, would instead select values from the 2pm insertions only.

In general, the arguments of `block_staircase` should be used as follows:
* `from_value_date` and `to_value_date`: time range on which values are retrieved
* `revision_freq`: revision frequency, as a dictionary of integers of which keys must be taken from
`["years", "months", "weeks", "bdays", "days", "hours", "minutes", "seconds"]`
* `revision_time`: revision time, as a dictionary of integers of which keys should be
taken from `["year", "month", "day", "weekday", "hour", "minute", "second"]`. It is used
for revision date initialisation. The next revision dates are then obtained by
successively adding `revision_freq`.
* `revision_tz`: time zone in which revision date and time are expressed
* `maturity_offset`: time span between each revision date and start time
of related block of values, as dictionary of integers. Its keys must be taken from
`["years", "months", "weeks", "bdays", "days", "hours", "minutes", "seconds"]`. No lag
is considered if it is not specified, i.e. the revision date is the block start date
* `maturity_time`: start time of each block, as a dictionary of integers of which keys
should be taken from `["year", "month", "day", "hour", "minute", "second"]`. The start
date of each block is thus obtained by adding `maturity_offset` to revision date and
then applying `maturity_time`. If not specified block start date is just the revision
date shifted by `maturity_offset`

### Other use cases

The `block_staircase` method covers multiple use cases, such as week-ahead revisions or
revision by business day, as described in the following examples.

#### Week-ahead staircase

Consider a series named `weekly_series` with following insertions:

|                  | 2021-01-05<br>(Tue) | 2021-01-07<br>(Thu) | 2021-01-12<br>(Tue) | 2021-01-14<br>(Thu) |
|:-----------------|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
| 2021-01-11 (Mon) | 1.1                 | 1.2                 |                     |                     |
| 2021-01-12 (Tue) | 2.1                 | 2.2                 |                     |                     |
| 2021-01-13 (Wed) | 3.1                 | 3.2                 |                     |                     |
| 2021-01-14 (Thu) | 4.1                 | 4.2                 |                     |                     |
| 2021-01-15 (Fri) | 5.1                 | 5.2                 |                     |                     |
| 2021-01-16 (Sat) | 6.1                 | 6.2                 |                     |                     |
| 2021-01-17 (Sun) | 7.1                 | 7.2                 |                     |                     |
| 2021-01-18 (Mon) | 8.1                 | 8.2                 | 8.3                 | 8.4                 |
| 2021-01-19 (Tue) | 9.1                 | 9.2                 | 9.3                 | 9.4                 |
| 2021-01-20 (Wed) | 10.1                | 10.2                | 10.3                | 10.4                |
| 2021-01-21 (Thu) |                     |                     | 11.3                | 11.4                |
| 2021-01-22 (Fri) |                     |                     | 12.3                | 12.4                |
| 2021-01-23 (Sat) |                     |                     | 13.3                | 13.4                |
| 2021-01-24 (Sun) |                     |                     | 14.3                | 14.4                |
| 2021-01-25 (Mon) |                     |                     | 15.3                | 15.4                |
| 2021-01-26 (Tue) |                     |                     | 16.3                | 16.4                |
| 2021-01-27 (Wed) |                     |                     | 17.3                | 17.4                |

Then the week-ahead staircase with weekly revision on Friday can be retrieved as
follows:
```python
 >>> tsa.block_staircase('weekly_series',
                         from_value_date=pd.Timestamp('2021-01-10'),
                         to_value_date=pd.Timestamp('2021-01-30'),
                         revision_freq={'days': 7},
                         revision_time={'weekday': 4},
                         revision_tz='utc',
                         maturity_offset={'days': 3},
                         maturity_time={'hour': 0})
 ...
 2021-01-11   1.2
 2021-01-12   2.2
 2021-01-13   3.2
 2021-01-14   4.2
 2021-01-15   5.2
 2021-01-16   6.2
 2021-01-17   7.2
 2021-01-18   8.4
 2021-01-19   9.4
 2021-01-20   10.4
 2021-01-21   11.4
 2021-01-22   12.4
 2021-01-23   13.4
 2021-01-24   14.4
 2021-01-25   15.4
 2021-01-26   16.4
 2021-01-27   17.4
 Name: weekly_series, dtype: float64
```

It is also possible to retrieve a month-ahead staircase series taking instead
`revision_freq={'months': 1}` and, for example, `revision_time={'day': 15}` to perform
monthly revision every 15th day of the month.

#### Revision by business day

The block_staircase method allows to express revision frequency and/or maturity time
span in business days. Consider a series named `business_day_series` with these
insertions:

|                  | 2021-01-13<br>(Wed) | 2021-01-14<br>(Thu) | 2021-01-15<br>(Fri) | 2021-01-16<br>(Sat) | 2021-01-17<br>(Sun) | 2021-01-18<br>(Mon) |
|:-----------------|:-------------------:|:-------------------:|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
| 2021-01-13 (Wed) | 3.1                 |                     |                     |                     |                     |                     |
| 2021-01-14 (Thu) | 4.1                 | 4.2                 |                     |                     |                     |                     |
| 2021-01-15 (Fri) | 5.1                 | 5.2                 | 5.3                 |                     |                     |                     |
| 2021-01-16 (Sat) | 6.1                 | 6.2                 | 6.3                 | 6.4                 |                     |                     |
| 2021-01-17 (Sun) |                     | 7.2                 | 7.3                 | 7.4                 | 7.5                 |                     |
| 2021-01-18 (Mon) |                     |                     | 8.3                 | 8.4                 | 8.5                 | 9.6                 |
| 2021-01-19 (Tue) |                     |                     |                     | 9.4                 | 9.5                 | 11.6                |
| 2021-01-20 (Wed) |                     |                     |                     |                     | 10.5                | 12.6                |
| 2021-01-21 (Thu) |                     |                     |                     |                     |                     | 13.6                |

Then we can retrieve a business-day-ahead staircase series with revision every business
day as follows:
```python
 >>> tsa.block_staircase('business_day_series',
                         from_value_date=pd.Timestamp('2021-01-13'),
                         to_value_date=pd.Timestamp('2021-01-21'),
                         revision_freq={'bdays': 1},
                         revision_tz='utc',
                         maturity_offset={'bdays': 1})
 ...
 2021-01-14   4.1
 2021-01-15   5.2
 2021-01-16   6.2
 2021-01-17   7.2
 2021-01-18   8.3
 2021-01-19   11.6
 2021-01-20   12.6
 2021-01-21   13.6
 Name: weekly_series, dtype: float64
```

### Optimizing staircase computations with history cache

It may be useful in some cases to compute multiple staircase series from the same source
series. For example, given a series named `"my_forecast"`, we could reconstruct both the 
one-day-ahead and two-days-ahead staircase series by doing
```python
 >>> ts_1da = tsa.block_staircase('my_forecast',
                                  from_value_date=pd.Timestamp('2020-01-01', tz="utc"),
                                  to_value_date=pd.Timestamp('2022-01-01', tz="utc"),
                                  revision_freq={'days': 1},
                                  revision_time={'hour': 9},
                                  revision_tz='utc',
                                  maturity_offset={'days': 1},
                                  maturity_time={'hour': 0})
 >>> ts_2da = tsa.block_staircase('my_forecast',
                                  from_value_date=pd.Timestamp('2020-01-01', tz="utc"),
                                  to_value_date=pd.Timestamp('2022-01-01', tz="utc"),
                                  revision_freq={'days': 1},
                                  revision_time={'hour': 9},
                                  revision_tz='utc',
                                  maturity_offset={'days': 2},
                                  maturity_time={'hour': 0})
```

However, the `block_staircase` function makes use of the `history` function to
reconstruct staircase series. For this reason, the response time may be a bit long
depending on the time span of the staircase series and the number of insertions.

To optimize the total execution time, we could retrieve the history of `"my_forecast"`
series, store it in memory and reconstruct the staircase series from it. This can be
done using the `historycache` object from the `tshsitory.tsio` module, as follows
```python
 >>> from tshistory.tsio import historycache
 >>>
 >>> hist = tsa.history('my_forecast'
                        from_value_date=pd.Timestamp('2020-01-01', tz="utc"),
                        to_value_date=pd.Timestamp('2022-01-01', tz="utc"))
 >>> hcache = historycache('my_forecast_cache', hist=hist, tzaware=True)
 >>>
 >>> ts_1da = hcache.block_staircase(from_value_date=pd.Timestamp('2020-01-01', tz="utc"),
                                     to_value_date=pd.Timestamp('2022-01-01', tz="utc"),
                                     revision_freq={'days': 1},
                                     revision_time={'hour': 9},
                                     revision_tz='utc',
                                     maturity_offset={'days': 1},
                                     maturity_time={'hour': 0})
 >>> ts_2da = hcache.block_staircase(from_value_date=pd.Timestamp('2020-01-01', tz="utc"),
                                     to_value_date=pd.Timestamp('2022-01-01', tz="utc"),
                                     revision_freq={'days': 1},
                                     revision_time={'hour': 9},
                                     revision_tz='utc',
                                     maturity_offset={'days': 2},
                                     maturity_time={'hour': 0})
```

In the example above, the `history` function of tshistory API is called once. Then the
staircase series are computed using the history data stored in memory using the
`historycache` object, which avoids one API call and reduce execution time.

This example assumes the series `"my_forecast"` is timezone-aware. In the case of
timezone-naive series, the `tzaware` parameter of `historycache` should be adapted
accordingly.

The `historycache` class also provides a `staircase` method, so this technique can also
be used for basic staircase computation.

# The API object

In the few examples above we manipulate the time series through an
object that talks directly to the postgresql back end.

It is possible to also talk to a rest api using the same api, like
shown below and proceed exactly like in the above code examples:

```python
 >>> from tshistory.api import timeseries
 >>>
 >>> tsa = timeseries('http://my.timeseries.info/api')
```

## Using an HTTP/REST end point

For the rest api, you need to build a small [flask][flask] app like
this (in an `app.py` module):

```python
from flask import Flask

from tshistory.api import timeseries
from tshistory.http.server import blueprint as blueprint


def make_app(dburi):
    app = Flask('my-timeseries-app')
    app.register_blueprint(
        blueprint(timeseries(dburi)),
        url_prefix='/api'
    )
    return app
```

Then, you can start it in development mode like this:

```python
app = make_app('postgresql://me:password@localhost/mydb')
app.run('192.168.1.1', 8080)
```

or just leave it to a wsgi container in e.g. a `wsgi.py` module:

```python
from my_series_app.app import make_app

app = make_app('postgresql://me:password@localhost/mydb')
```

## API surface

For now we only provide a list of supported methods.


### Information access (read methods)

* catalog

* exists

* get

* history

* interval

* metadata

* staircase

* block_staircase

* type


### Information update (write methods)

* update

* update_metadata

* replace

* rename

* delete


# Command line

## Basic operations

A command line tool is provided, called `tsh`. It provides its usage
guidelines:

```shell
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
```

`Info` provides an overview of the time series repository (number of
committed changes, number and series and their names).

```shell
 $ tsh info postgres://babar:babarpassword@dataserver:5432/banana_studies
 changeset count: 209
 series count:    144
 series names:    banana_spot_price, banana_trades, banana_turnover
```

`Log` provides the full history of editions to time series in the
repository.

```shell
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
```

All options of all commands can be obtained by using the `--help`
switch:

```shell
 $ tsh log --help
 Usage: tsh log [OPTIONS] DB_URI

 Options:
   -l, --limit TEXT
   --show-diff
   -s, --serie TEXT
   --from-rev TEXT
   --to-rev TEXT
   --help            Show this message and exit.
```


## Extensions

It is possible to augment the `tsh` command with new subcommands (or
augment, modify existing commands).

Any program doing so must define a new command and declare a setup
tools entry point named `tshistory:subcommand` as in e.g.:

```python

    entry_points={'tshistory.subcommands': [
        'view=tsview.command:view'
    ]}
```

For instance, the [tsview][tsview] python package provides such a
`view` subcommand for generic time series visualisation.

[tsview]: https://hg.sr.ht/~pythonian/tsview
[backtesting]: https://en.wikipedia.org/wiki/Backtesting
[cross-validation]: https://en.wikipedia.org/wiki/Cross-validation_(statistics)
[flask]: https://www.palletsprojects.com/p/flask/


# Status

It is currently considered `beta` software even though it has been in
production for two years. It is still evolving. Schema/Database
changes come with migration procedure using the `tsh` utility.

When it is good: if you do mostly appends (and occasional edits in the
past) it will store data in a very compact way.

Reading any version of the series will always be the fastest (io-bound)
operation.

Alternative backend storage and storage strategies will be considered
in the future.

