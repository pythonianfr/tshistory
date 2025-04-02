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


-- tree

create extension if not exists ltree;

create table "{ns}".tree (
  id serial primary key,
  path ltree
);

create index tree_path_idx on "{ns}".tree using gist (path);


create table "{ns}".tree_series_map (
  seriesid integer unique references "{ns}".registry (id),
  treeid integer references "{ns}".tree (id)
);

create index tree_series_map_idx on "{ns}".tree_series_map (treeid);
