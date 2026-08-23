import re
import time

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Review
from app.schemas import AskResponse, ProductResult
from app.search import hybrid_search

RETRIEVAL_LIMIT = 5
REVIEWS_PER_PRODUCT = 2
REVIEW_SNIPPET_CHARS = 300

CITATION_RE = re.compile(r"\[(P\d+)\]")

# Grounding is a hard requirement: the model must only draw on the retrieved
# context and must say so rather than invent a product when nothing fits.
SYSTEM_PROMPT = (
    "You are a skincare/beauty shopping assistant. Answer the user's question "
    "using ONLY the product information provided below. Do not use outside "
    "knowledge, and do not invent products, prices, or claims that aren't in the "
    "provided information. When you make a claim about a specific product, cite "
    "it inline right after the claim using its ID in square brackets, e.g. "
    "[P123456]. If none of the provided products actually answer the question, "
    "say plainly that you don't have enough information rather than guessing."
)


def _build_context(results: list[ProductResult], reviews_by_product: dict[str, list[str]]) -> str:
    blocks = []
    for r in results:
        lines = [
            f"Product ID: {r.product_id}",
            f"Name: {r.product_name}",
            f"Brand: {r.brand_name or 'Unknown'}",
            f"Category: {r.category or 'Unknown'}",
            f"Price: ${r.price_usd}" if r.price_usd is not None else "Price: Unknown",
            f"Rating: {r.rating}/5 ({r.reviews_count or 0} reviews)"
            if r.rating is not None
            else "Rating: Unknown",
        ]
        product_reviews = reviews_by_product.get(r.product_id, [])
        if product_reviews:
            lines.append("Sample reviews: " + " | ".join(product_reviews))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


async def _fetch_reviews(session: AsyncSession, product_ids: list[str]) -> dict[str, list[str]]:
    if not product_ids:
        return {}

    stmt = select(Review.product_id, Review.review_text).where(Review.product_id.in_(product_ids))
    rows = (await session.execute(stmt)).all()

    reviews_by_product: dict[str, list[str]] = {}
    for product_id, review_text in rows:
        if not review_text:
            continue
        bucket = reviews_by_product.setdefault(product_id, [])
        if len(bucket) < REVIEWS_PER_PRODUCT:
            bucket.append(review_text[:REVIEW_SNIPPET_CHARS])
    return reviews_by_product


async def _call_groq(context: str, question: str) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{settings.groq_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={
                "model": settings.groq_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Product information:\n\n{context}\n\nQuestion: {question}",
                    },
                ],
                "temperature": 0.2,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def ask(session: AsyncSession, question: str) -> AskResponse:
    """Retrieve grounding context via hybrid search, then generate a cited answer via Groq."""
    start = time.perf_counter()

    results = await hybrid_search(session, question, RETRIEVAL_LIMIT)

    if not results:
        return AskResponse(
            answer="I don't have enough information in the catalog to answer that.",
            cited_product_ids=[],
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    reviews_by_product = await _fetch_reviews(session, [r.product_id for r in results])
    context = _build_context(results, reviews_by_product)

    answer = await _call_groq(context, question)

    # Only trust citations the model actually emitted for products we actually
    # retrieved -- this is what keeps cited_product_ids grounded rather than a
    # rubber-stamp of everything that was fetched.
    retrieved_ids = {r.product_id for r in results}
    cited_ids = [pid for pid in dict.fromkeys(CITATION_RE.findall(answer)) if pid in retrieved_ids]

    return AskResponse(answer=answer, cited_product_ids=cited_ids, latency_ms=(time.perf_counter() - start) * 1000)
