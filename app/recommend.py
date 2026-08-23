from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Product
from app.schemas import ProductResult

CANDIDATE_POOL_SIZE = 20
RATING_WEIGHT = 0.15


async def recommend(session: AsyncSession, product_id: str, limit: int) -> list[ProductResult]:
    """Content-based 'similar products': rank the catalog by embedding cosine similarity
    to the source product, then re-rank the top candidates with a small rating boost so a
    well-reviewed near-match can edge out a marginally-closer but poorly-rated one.
    """
    source = await session.get(Product, product_id)
    if source is None:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")

    distance = Product.embedding.cosine_distance(source.embedding)
    stmt = (
        select(Product, distance.label("distance"))
        .where(Product.product_id != product_id)
        .order_by(distance)
        .limit(CANDIDATE_POOL_SIZE)
    )
    rows = (await session.execute(stmt)).all()

    scored = []
    for product, dist in rows:
        similarity = 1 - dist
        rating_boost = (float(product.rating) / 5) if product.rating is not None else 0.0
        combined_score = similarity * (1 - RATING_WEIGHT) + rating_boost * RATING_WEIGHT
        scored.append((combined_score, product))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [
        ProductResult(
            product_id=product.product_id,
            product_name=product.product_name,
            brand_name=product.brand_name,
            category=product.category,
            price_usd=float(product.price_usd) if product.price_usd is not None else None,
            rating=float(product.rating) if product.rating is not None else None,
            reviews_count=product.reviews_count,
            score=combined_score,
        )
        for combined_score, product in scored[:limit]
    ]
