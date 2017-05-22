# TSHISTORY

This is a library to store/retrieve pandas timeseries to/from a
postgres database, tracking their successive versions.


# Principles

There are many ways to represent timeseries in a relational database,
and `tshistory` provides two things:

* a base python API which abstracts away the underlying storage

* a postgres model, which uses JSONB fields to store the bulk of the
  series data.

The core idea of tshistory is to handle successive versions of
timeseries as they grow in time, allowing to get older states of any
series.

Series state can be indexed by either a timestamp (which typically
matches the moment a new insertion took place) or a `changeset id`
which denotes the exact change leading to a given version.


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
 tsh.insert(engine, serie, 'my_serie', 'aurelien@campeas.fr')

 # db retrieval
 assert tsh.get(engine, 'my_serie') == serie
```

This is good. Now, let's add more:

```python
 serie = pd.Series([7, 8, 9],
                  pd.date_range(start=datetime(2017, 1, 3),
                                freq='D', periods=3))
 # db insertion
 tsh.insert(engine, serie, 'my_serie', 'aurelien@campeas.fr')

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

