create table "{ns}".registry (
  id serial primary key,
  seriename text not null,
  table_name text not null,
  metadata jsonb,
  unique(seriename),
  unique(table_name)
);


create table "{ns}".changeset (
  id serial primary key,
  author text not null,
  insertion_date timestamptz not null,
  metadata jsonb
);

create index on "{ns}".changeset(author);
create index on "{ns}".changeset(insertion_date) ;


create table "{ns}".changeset_series (
  cset integer references "{ns}".changeset(id) on delete set null,
  serie integer references "{ns}".registry(id) on delete set null
);

create index on "{ns}".changeset_series(cset);
create index on "{ns}".changeset_series(serie);
