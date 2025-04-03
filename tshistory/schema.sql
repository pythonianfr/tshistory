-- baskets

create table "{ns}".basket (
  id serial primary key,
  name text not null,
  "query" text not null,
  unique(name)
);


-- metadata-history

create table "{ns}".ts_oldmeta (
  moment timestamptz unique not null default now(),
  seriesid integer not null references "{ns}".registry (id) on delete cascade,
  metadata jsonb not null
);

create index on "{ns}".ts_oldmeta (moment);
create index on "{ns}".ts_oldmeta (seriesid);
