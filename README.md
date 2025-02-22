CREATE EXTENSION IF NOT EXISTS vector;

create table document
(
id serial primary key,
text text,
source text,
embedding VECTOR,
created_at timestamp default now(),
updated_at timestamp default now()
);

CREATE INDEX idx_document_text ON document(text);
