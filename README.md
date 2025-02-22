CREATE EXTENSION IF NOT EXISTS vector;

create table document (
    id serial primary key,
    text text,
    source text,
    embedding VECTOR
);