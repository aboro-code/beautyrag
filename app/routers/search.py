import time
from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas import SearchResponse
from app.search import hybrid_search, keyword_search, vector_search

router = APIRouter()

_SEARCH_FNS = {
    "vector": vector_search,
    "keyword": keyword_search,
    "hybrid": hybrid_search,
}


@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, description="Natural-language product search query"),
    mode: Literal["vector", "keyword", "hybrid"] = Query("hybrid"),
    limit: int = Query(10, ge=1, le=50),
    session: AsyncSession = Depends(get_session),
) -> SearchResponse:
    """Search products by vector similarity, keyword full-text, or hybrid (RRF fusion of both).

    Exposing each mode separately (rather than only shipping hybrid) is what makes
    eval/run_eval.py's Precision@5/Recall@5/MRR comparison across modes possible.
    """
    start = time.perf_counter()
    results = await _SEARCH_FNS[mode](session, q, limit)
    latency_ms = (time.perf_counter() - start) * 1000

    return SearchResponse(query=q, mode=mode, results=results, latency_ms=latency_ms)
