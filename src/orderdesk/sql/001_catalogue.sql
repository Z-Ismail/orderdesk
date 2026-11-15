-- Catalogue, customers and customer-specific aliases.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE products (
    sku             text PRIMARY KEY,
    name            text NOT NULL,
    description     text NOT NULL,
    category        text NOT NULL,
    unit_of_measure text NOT NULL,
    pack_size       integer NOT NULL CHECK (pack_size > 0),
    list_price      numeric(10, 2) NOT NULL CHECK (list_price >= 0),
    active          boolean NOT NULL DEFAULT true,
    embedding       vector(384),  -- must match the embedding model's dimension
    -- Punctuation becomes spaces first, so 'No.8' indexes as 'no 8' and '18x29x39in' splits.
    search_tsv      tsvector GENERATED ALWAYS AS (
                        to_tsvector('english', regexp_replace(
                            name || ' ' || description || ' ' || category,
                            '[^[:alnum:]]+', ' ', 'g'))
                    ) STORED
);
CREATE INDEX products_search_tsv_idx ON products USING gin (search_tsv);
CREATE INDEX products_embedding_idx ON products USING hnsw (embedding vector_cosine_ops);

CREATE TABLE customers (
    customer_id              text PRIMARY KEY,
    name                     text NOT NULL,
    email_domain             text NOT NULL,
    contact_email            text NOT NULL,
    price_tier               text NOT NULL,
    account_status           text NOT NULL,
    default_delivery_address text
);
CREATE INDEX customers_email_domain_idx ON customers (lower(email_domain));

-- What a specific customer calls one of our products: their own code, or their own wording.
CREATE TABLE customer_aliases (
    id            bigserial PRIMARY KEY,
    customer_id   text NOT NULL REFERENCES customers (customer_id),
    customer_sku  text,
    customer_text text,
    sku           text NOT NULL REFERENCES products (sku),
    CHECK (customer_sku IS NOT NULL OR customer_text IS NOT NULL)
);
CREATE UNIQUE INDEX customer_aliases_sku_uq
    ON customer_aliases (customer_id, lower(customer_sku)) WHERE customer_sku IS NOT NULL;
CREATE UNIQUE INDEX customer_aliases_text_uq
    ON customer_aliases (customer_id, lower(customer_text)) WHERE customer_text IS NOT NULL;
