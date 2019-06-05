create table "{namespace}.timeserie"."{tablename}" (
  id serial primary key,
  cset integer not null references "{namespace}".changeset(id),
  tsstart timestamp not null, -- whole series index min
  tsend timestamp not null,   -- whole series index max
  snapshot integer references "{namespace}.snapshot"."{tablename}"(id)
);

create index on "{namespace}.timeserie"."{tablename}"(cset);
create index on "{namespace}.timeserie"."{tablename}"(snapshot);
create index on "{namespace}.timeserie"."{tablename}"(tsstart);
create index on "{namespace}.timeserie"."{tablename}"(tsend);
