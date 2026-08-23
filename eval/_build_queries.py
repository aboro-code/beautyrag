"""One-off helper: builds eval/queries.py's labeled ground truth from the live catalog.

Relevance labels are derived from targeted category + keyword filters against product
metadata (not full manual review of all 8,494 products) -- a defensible proxy for hand
labeling given the weekend timebox, spot-checked for sanity. Run once; the output is
committed as static data so the eval itself doesn't depend on live filtering.
"""

import asyncio

import asyncpg

from app.config import settings

QUERIES: list[tuple[str, str]] = [
    ("vitamin C serum for brightening", "product_name ILIKE '%vitamin c%' AND category ILIKE '%serum%'"),
    (
        "hyaluronic acid serum for hydration",
        "product_name ILIKE '%hyaluronic acid%' AND category ILIKE '%serum%'",
    ),
    (
        "retinol cream for anti-aging",
        "product_name ILIKE '%retinol%' AND (category ILIKE '%moisturizer%' OR category ILIKE '%treatment%')",
    ),
    ("niacinamide serum for pores", "product_name ILIKE '%niacinamide%' AND category ILIKE '%serum%'"),
    (
        "clay mask for acne-prone skin",
        "category ILIKE '%mask%' AND (product_name ILIKE '%clay%' OR product_name ILIKE '%acne%')",
    ),
    (
        "gentle cleanser for sensitive skin",
        "category ILIKE '%cleanser%' AND (product_name ILIKE '%gentle%' OR product_name ILIKE '%sensitive%')",
    ),
    (
        "eye cream for dark circles",
        "category ILIKE '%eye%' AND (product_name ILIKE '%dark circle%' OR product_name ILIKE '%eye cream%')",
    ),
    (
        "exfoliating toner with AHA or BHA",
        "category ILIKE '%toner%' AND (product_name ILIKE '%aha%' OR product_name ILIKE '%bha%' OR product_name ILIKE '%exfoliat%')",
    ),
    (
        "body lotion for dry skin",
        "category ILIKE '%Body Lotions%' AND product_name ILIKE '%lotion%' AND product_name NOT ILIKE '%after%shave%'",
    ),
    ("matte foundation for oily skin", "category ILIKE '%foundation%' AND product_name ILIKE '%matte%'"),
    (
        "shampoo for damaged hair",
        "category ILIKE '%shampoo%' AND (product_name ILIKE '%damage%' OR product_name ILIKE '%repair%')",
    ),
    ("peptide serum for fine lines", "product_name ILIKE '%peptide%' AND category ILIKE '%serum%'"),
    ("charcoal face wash", "product_name ILIKE '%charcoal%' AND category ILIKE '%cleanser%'"),
    ("setting spray for makeup", "product_name ILIKE '%setting spray%'"),
    ("lip balm with SPF", "product_name ILIKE '%lip%' AND (product_name ILIKE '%spf%' OR description ILIKE '%spf%')"),
    (
        "fragrance-free sunscreen",
        "category ILIKE '%sun%' AND (description ILIKE '%fragrance-free%' OR description ILIKE '%fragrance free%')",
    ),
    ("vegan face moisturizer", "category ILIKE '%moisturizer%' AND description ILIKE '%vegan%'"),
    ("sheet mask for hydration", "product_name ILIKE '%sheet mask%'"),
]

MAX_RELEVANT_PER_QUERY = 25


async def main() -> None:
    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn)

    print("LABELED_QUERIES = [")
    for query_text, where_clause in QUERIES:
        rows = await conn.fetch(
            f"SELECT product_id FROM products WHERE {where_clause} LIMIT {MAX_RELEVANT_PER_QUERY}"
        )
        ids = [r["product_id"] for r in rows]
        print(f"    {(query_text, ids)!r},")
    print("]")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
