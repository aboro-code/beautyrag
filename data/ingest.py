"""Load the Sephora product/review CSVs into Postgres and generate product embeddings.

Source: Kaggle "Sephora Products and Skincare Reviews" dataset, downloaded into
data/raw/ via `kaggle datasets download -d nadyinky/sephora-products-and-skincare-reviews`.

Run: python -m data.ingest
"""

import argparse
import ast
import asyncio
from pathlib import Path

import asyncpg
import pandas as pd
from pgvector.asyncpg import register_vector
from sentence_transformers import SentenceTransformer

from app.config import settings

RAW_DIR = Path(__file__).parent / "raw"
REVIEW_FILES = [
    "reviews_0-250.csv",
    "reviews_250-500.csv",
    "reviews_500-750.csv",
    "reviews_750-1250.csv",
    "reviews_1250-end.csv",
]
MAX_REVIEWS_PER_PRODUCT = 8


def parse_list_field(value) -> str:
    """Dataset stores ingredients/highlights as a stringified Python list; flatten to plain text."""
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        items = ast.literal_eval(value)
        if isinstance(items, list):
            return ", ".join(str(i) for i in items)
    except (ValueError, SyntaxError):
        pass
    return value


def load_products() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "product_info.csv")

    df["ingredients_text"] = df["ingredients"].apply(parse_list_field)
    df["highlights_text"] = df["highlights"].apply(parse_list_field)
    df["category"] = (
        df[["primary_category", "secondary_category", "tertiary_category"]]
        .fillna("")
        .agg(" > ".join, axis=1)
        .str.strip(" >")
    )
    # No free-text product description exists in this dataset; highlights (short
    # marketing tags like "Fragrance-Free", "Vegan") is the closest analog.
    df["description"] = df["highlights_text"]

    df["embedding_text"] = (
        df["product_name"].fillna("")
        + " by "
        + df["brand_name"].fillna("")
        + ". Category: "
        + df["category"]
        + ". "
        + df["highlights_text"]
        + ". Ingredients: "
        + df["ingredients_text"].str.slice(0, 1000)
    )

    df["reviews_count"] = df["reviews"].fillna(0).astype(int)
    return df


def load_reviews(product_ids: set[str]) -> pd.DataFrame:
    frames = []
    usecols = [
        "product_id",
        "author_id",
        "rating",
        "review_title",
        "review_text",
        "total_feedback_count",
        "submission_time",
    ]
    for fname in REVIEW_FILES:
        chunk = pd.read_csv(RAW_DIR / fname, usecols=usecols)
        frames.append(chunk)
    reviews = pd.concat(frames, ignore_index=True)

    reviews = reviews[reviews["product_id"].isin(product_ids)]
    reviews = reviews.dropna(subset=["review_text"])
    reviews["total_feedback_count"] = reviews["total_feedback_count"].fillna(0)

    # Keep the most helpful/recent reviews per product rather than all ~1.1M rows.
    reviews = reviews.sort_values(
        ["product_id", "total_feedback_count", "submission_time"], ascending=[True, False, False]
    )
    reviews = reviews.groupby("product_id").head(MAX_REVIEWS_PER_PRODUCT)

    reviews = reviews.reset_index(drop=True)
    reviews["review_id"] = reviews["product_id"] + "-" + reviews.index.astype(str)
    return reviews


async def insert_products(conn: asyncpg.Connection, df: pd.DataFrame, model: SentenceTransformer) -> None:
    print(f"Embedding {len(df)} products...")
    embeddings = model.encode(
        df["embedding_text"].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    rows = [
        (
            row.product_id,
            row.product_name,
            row.brand_name if pd.notna(row.brand_name) else None,
            row.category or None,
            float(row.price_usd) if pd.notna(row.price_usd) else None,
            float(row.rating) if pd.notna(row.rating) else None,
            int(row.reviews_count),
            row.ingredients_text or None,
            row.description or None,
            embeddings[i],
        )
        for i, row in enumerate(df.itertuples(index=False))
    ]

    print(f"Inserting {len(rows)} products...")
    await conn.executemany(
        """
        INSERT INTO products (
            product_id, product_name, brand_name, category, price_usd,
            rating, reviews_count, ingredients, description, embedding
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (product_id) DO UPDATE SET
            product_name = EXCLUDED.product_name,
            brand_name = EXCLUDED.brand_name,
            category = EXCLUDED.category,
            price_usd = EXCLUDED.price_usd,
            rating = EXCLUDED.rating,
            reviews_count = EXCLUDED.reviews_count,
            ingredients = EXCLUDED.ingredients,
            description = EXCLUDED.description,
            embedding = EXCLUDED.embedding
        """,
        rows,
    )


async def insert_reviews(conn: asyncpg.Connection, df: pd.DataFrame) -> None:
    rows = [
        (
            row.review_id,
            row.product_id,
            str(row.author_id) if pd.notna(row.author_id) else None,
            int(row.rating) if pd.notna(row.rating) else None,
            row.review_title if pd.notna(row.review_title) else None,
            row.review_text if pd.notna(row.review_text) else None,
        )
        for row in df.itertuples(index=False)
    ]

    print(f"Inserting {len(rows)} reviews...")
    batch_size = 5000
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        await conn.executemany(
            """
            INSERT INTO reviews (review_id, product_id, author_id, rating, review_title, review_text)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (review_id) DO NOTHING
            """,
            batch,
        )
        print(f"  ...{min(i + batch_size, len(rows))}/{len(rows)}")


async def main(limit: int | None) -> None:
    products_df = load_products()
    if limit:
        products_df = products_df.head(limit)

    reviews_df = load_reviews(set(products_df["product_id"]))

    model = SentenceTransformer(settings.embedding_model_name)

    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn, statement_cache_size=0)
    await register_vector(conn)
    try:
        await insert_products(conn, products_df, model)
        await insert_reviews(conn, reviews_df)
    finally:
        await conn.close()

    print("Ingest complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only ingest the first N products (for a quick test run)")
    args = parser.parse_args()
    asyncio.run(main(args.limit))
