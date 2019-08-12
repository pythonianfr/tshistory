create table "{namespace}.revision"."{tablename}" (
  id serial primary key,
  tsstart timestamp not null, -- whole series index min
  tsend timestamp not null,   -- whole series index max
  snapshot integer references "{namespace}.snapshot"."{tablename}"(id),
  author text not null,
  insertion_date timestamptz not null,
  metadata jsonb
);

create index on "{namespace}.revision"."{tablename}"(snapshot);
create index on "{namespace}.revision"."{tablename}"(tsstart);
create index on "{namespace}.revision"."{tablename}"(tsend);
create index on "{namespace}.revision"."{tablename}"(insertion_date);
