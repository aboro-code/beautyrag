from typing import Literal

from pydantic import BaseModel


class ProductResult(BaseModel):
    product_id: str
    product_name: str
    brand_name: str | None = None
    category: str | None = None
    price_usd: float | None = None
    rating: float | None = None
    reviews_count: int | None = None
    score: float


class SearchResponse(BaseModel):
    query: str
    mode: Literal["vector", "keyword", "hybrid"]
    results: list[ProductResult]
    latency_ms: float
