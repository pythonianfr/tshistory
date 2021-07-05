-- time series registry

create table "{ns}".registry (
  id serial primary key,
  seriesname text not null,
  tablename text not null,
  metadata jsonb,
  unique(seriesname),
  unique(tablename)
);
