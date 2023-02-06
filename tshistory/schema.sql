-- time series registry

create table "{ns}".registry (
  id serial primary key,
  name text not null,
  internal_metadata jsonb,
  metadata jsonb,
  unique(name)
);

create index on "{ns}".registry using gin(internal_metadata);
create index on "{ns}".registry using gin(metadata);


create table "{ns}".basket (
  id serial primary key,
  name text not null,
  "query" text not null,
  unique(name)
);
