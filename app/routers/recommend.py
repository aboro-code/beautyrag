import time

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.recommend import recommend as run_recommend
from app.schemas import RecommendResponse

router = APIRouter()


@router.get("/recommend/{product_id}", response_model=RecommendResponse)
async def recommend(
    product_id: str,
    limit: int = Query(10, ge=1, le=50),
    session: AsyncSession = Depends(get_session),
) -> RecommendResponse:
    """Content-based 'similar products': catalog neighbors of product_id by embedding
    cosine similarity, re-ranked with a small rating boost. Returns 404 if product_id
    isn't in the catalog.
    """
    start = time.perf_counter()
    results = await run_recommend(session, product_id, limit)
    return RecommendResponse(product_id=product_id, results=results, latency_ms=(time.perf_counter() - start) * 1000)
