-- time series registry

create table "{ns}".registry (
  id serial primary key,
  seriesname text not null,
  tablename text not null,
  internal_metadata jsonb,
  metadata jsonb,
  unique(seriesname),
  unique(tablename)
);

create index on "{ns}".registry using gin(internal_metadata);
create index on "{ns}".registry using gin(metadata);
