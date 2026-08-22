-- BeautyRAG schema: products + reviews, with pgvector embeddings and full-text search.
-- Applied via data/init_db.py (idempotent: safe to run multiple times).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS products (
    product_id      TEXT PRIMARY KEY,
    product_name    TEXT NOT NULL,
    brand_name      TEXT,
    category        TEXT,
    price_usd       NUMERIC,
    rating          NUMERIC,
    reviews_count   INTEGER,
    ingredients     TEXT,
    description     TEXT,
    embedding       VECTOR(384),
    search_vector   TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(product_name, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(brand_name, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(category, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(description, '')), 'C') ||
        setweight(to_tsvector('english', coalesce(ingredients, '')), 'D')
    ) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id       TEXT PRIMARY KEY,
    product_id      TEXT NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    author_id       TEXT,
    rating          SMALLINT,
    review_title    TEXT,
    review_text     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Full-text search index over products.
CREATE INDEX IF NOT EXISTS idx_products_search_vector ON products USING GIN (search_vector);

-- Vector similarity index. HNSW builds incrementally (unlike IVFFlat, it doesn't need
-- representative data present at creation time), so it's safe to create before ingest.
CREATE INDEX IF NOT EXISTS idx_products_embedding_hnsw ON products
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_reviews_product_id ON reviews (product_id);
