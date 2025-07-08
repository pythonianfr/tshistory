-- time series registry

create table "{ns}".registry (
  id serial primary key,
  name text not null,
  internal_metadata jsonb not null,
  metadata jsonb not null default '{{}}',
  unique(name)
);

create index on "{ns}".registry using gin(internal_metadata);
create index on "{ns}".registry using gin(metadata);

-- commit author + metadata (for the fs backend)

create table "{ns}".revision_metadata (
  id serial primary key,
  series integer not null references "{ns}".registry(id) on delete cascade,
  author text not null,
  metadata jsonb
);

create index on "{ns}".revision_metadata(series);
