from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings import embed_query
from app.models import Product
from app.schemas import ProductResult

RRF_K = 60
CANDIDATE_POOL_SIZE = 50


def _to_result(product: Product, score: float) -> ProductResult:
    return ProductResult(
        product_id=product.product_id,
        product_name=product.product_name,
        brand_name=product.brand_name,
        category=product.category,
        price_usd=float(product.price_usd) if product.price_usd is not None else None,
        rating=float(product.rating) if product.rating is not None else None,
        reviews_count=product.reviews_count,
        score=score,
    )


async def vector_search(session: AsyncSession, query: str, limit: int) -> list[ProductResult]:
    """Rank products by pgvector cosine distance between the query embedding and product embeddings."""
    query_embedding = embed_query(query)
    distance = Product.embedding.cosine_distance(query_embedding)

    stmt = select(Product, distance.label("distance")).order_by(distance).limit(limit)
    rows = (await session.execute(stmt)).all()

    return [_to_result(product, score=1 - distance) for product, distance in rows]


async def keyword_search(session: AsyncSession, query: str, limit: int) -> list[ProductResult]:
    """Rank products by Postgres full-text search (ts_rank) over the products.search_vector column."""
    tsquery = func.plainto_tsquery("english", query)
    rank = func.ts_rank(Product.search_vector, tsquery)

    stmt = (
        select(Product, rank.label("rank"))
        .where(Product.search_vector.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()

    return [_to_result(product, score=rank) for product, rank in rows]


async def hybrid_search(session: AsyncSession, query: str, limit: int) -> list[ProductResult]:
    """Fuse vector and keyword rankings via Reciprocal Rank Fusion (RRF).

    Each retrieval mode contributes 1/(RRF_K + rank) per product; summing across
    both lists rewards products that rank well under either signal, which is why
    hybrid outperforms either method alone (see eval/run_eval.py results).
    """
    # Sequential, not concurrent: AsyncSession doesn't support overlapping operations
    # on the same connection.
    vector_results = await vector_search(session, query, CANDIDATE_POOL_SIZE)
    keyword_results = await keyword_search(session, query, CANDIDATE_POOL_SIZE)

    rrf_scores: dict[str, float] = {}
    products_by_id: dict[str, ProductResult] = {}

    for ranked_list in (vector_results, keyword_results):
        for rank, result in enumerate(ranked_list, start=1):
            rrf_scores[result.product_id] = rrf_scores.get(result.product_id, 0.0) + 1 / (RRF_K + rank)
            products_by_id.setdefault(result.product_id, result)

    fused_ids = sorted(rrf_scores, key=lambda pid: rrf_scores[pid], reverse=True)[:limit]

    return [
        products_by_id[pid].model_copy(update={"score": rrf_scores[pid]}) for pid in fused_ids
    ]
