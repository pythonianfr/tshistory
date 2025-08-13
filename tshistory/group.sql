-- time series groups registry

create table "{ns}".group_registry (
  id serial primary key,
  name text not null unique,
  internal_metadata jsonb not null,
  metadata jsonb not null default '{{}}'
);

create index "ix_{ns}_group_registry_idx" on "{ns}".group_registry(name);
create index "{ns}_group_registry_internal_metadata_idx" on "{ns}".group_registry using gin(internal_metadata);
create index "{ns}_group_registry_metadata_idx" on "{ns}".group_registry using gin(metadata);

-- in the series <-> group mapping below
-- we don't give series names their member name
-- to avoid any risk of name conflict
create table "{ns}".groupmap (
  -- member name
  name text not null,
  -- mapping from group (in group_registry) to individual series
  -- (in "{ns}".group)
  groupid integer references "{ns}".group_registry(id) on delete cascade,
  seriesid integer references "{ns}.group".registry(id),
  unique(name, groupid)
);

create index "ix_{ns}_groupmap_group_idx" on "{ns}".groupmap(groupid);
create index "ix_{ns}_groupmap_series_idx" on "{ns}".groupmap(seriesid);


-- metadata-history

create table "{ns}".gr_oldmeta (
  moment timestamptz unique not null default now(),
  groupid integer not null references "{ns}".group_registry (id) on delete cascade,
  userid text default 'no-user',
  metadata jsonb not null
);

create index "{ns}_gr_oldmeta_moment_idx" on "{ns}".gr_oldmeta (moment);
create index "{ns}_gr_oldmeta_groupid_idx" on "{ns}".gr_oldmeta (groupid);
