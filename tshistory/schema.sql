-- baskets

create table "{ns}".basket (
  id serial primary key,
  name text not null,
  "query" text not null,
  unique(name)
);
