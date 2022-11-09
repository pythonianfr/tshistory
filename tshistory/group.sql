-- time series groups registry

create table "{ns}".group_registry (
  id serial primary key,
  name text not null unique,
  metadata jsonb
);

create index "ix_{ns}_group_registry_idx" on "{ns}".group_registry(name);

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
