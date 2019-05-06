create table "{namespace}.snapshot"."{tablename}" (
  id serial primary key,
  cstart timestamptz,
  cend timestamptz,
  chunk bytea,
  parent integer references "{namespace}.snapshot"."{tablename}"(id)
);

create index on "{namespace}.snapshot"."{tablename}"(cstart);
create index on "{namespace}.snapshot"."{tablename}"(cend);
create index on "{namespace}.snapshot"."{tablename}"(parent);
