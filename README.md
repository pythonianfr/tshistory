TSHISTORY
===========

This is a library to store/retrieve pandas timeseries to/from a
postgres database, tracking their successive versions.

[TOC]


# Principles

There are many ways to represent timeseries in a relational database,
and `tshistory` provides two things:

* a base python API which abstracts away the underlying storage

* a postgres model, which uses BYTEA fields to store chunks of the
  series data.

The core idea of tshistory is to handle successive versions of
timeseries as they grow in time, allowing to get older states of any
series.

Series state can be indexed by either a timestamp (which typically
matches the moment a new insertion took place) or a `changeset id`
which is a numeric index denoting the exact change leading to a given
version.


# Basic usage


Complete usage is shown in the test suite. However here's a simple
example:

```python
 from datetime import datetime
 from sqlalchemy import create_engine
 import pandas as pd
 from tshistory.tsio import TimeSerie

 engine = create_engine('postgres://me:password@localhost/mydb')
 tsh = TimeSerie()

 serie = pd.Series([1, 2, 3],
                  pd.date_range(start=datetime(2017, 1, 1),
                                freq='D', periods=3))
 # db insertion
 tsh.insert(engine, serie, 'my_serie', 'babar@pythonian.fr')

 # db retrieval
 assert tsh.get(engine, 'my_serie') == serie
```

This is good. Now, let's add more:

```python
 serie = pd.Series([7, 8, 9],
                  pd.date_range(start=datetime(2017, 1, 3),
                                freq='D', periods=3))
 # db insertion
 tsh.insert(engine, serie, 'my_serie', 'babar@pythonian.fr')

 # db retrieval
 stored = tsh.get(engine, 'my_serie')

 assert """
2017-01-01    1
2017-01-02    2
2017-01-03    7
2017-01-04    8
2017-01-04    9
Freq: D
""".strip() == stored.to_string().strip()
```

It is important to note that the third value was replaced, and the two
last values were just appended.


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
  dump     dump all time series revisions in a zip file
  get      show a serie in its current state
  history  show a serie full history
  info     show global statistics of the repository
  init-db  initialize an new db.
  log      show revision history of entire repository or...
  restore  restore zip file in a freshly initialized...
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

[tsview]: https://bitbucket.org/pythonian/tsview
