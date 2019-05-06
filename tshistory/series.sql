create table "{namespace}.timeserie"."{tablename}" (
  id serial primary key,
  cset integer not null references "{namespace}".changeset(id),
  tsstart timestamp not null,
  tsend timestamp not null,
  snapshot integer references "{namespace}.snapshot"."{tablename}"(id)
);

create index on "{namespace}.timeserie"."{tablename}"(cset);
create index on "{namespace}.timeserie"."{tablename}"(snapshot);
create index on "{namespace}.timeserie"."{tablename}"(tsstart);
create index on "{namespace}.timeserie"."{tablename}"(tsend);
