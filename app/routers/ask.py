import time

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import get_cached_ask, set_cached_ask
from app.db import get_session
from app.rag import ask as run_ask
from app.rate_limit import enforce_rate_limit
from app.schemas import AskRequest, AskResponse

router = APIRouter()


@router.post("/ask", response_model=AskResponse, dependencies=[Depends(enforce_rate_limit)])
async def ask(request: AskRequest, session: AsyncSession = Depends(get_session)) -> AskResponse:
    """Grounded RAG Q&A: hybrid-search the catalog, then answer using only that retrieved
    context. The model is instructed to say it doesn't have enough information rather than
    inventing a product or claim, and cited_product_ids reflects only products the model
    actually cited in its answer -- not just whatever was retrieved.

    Responses are cached in Redis (keyed on the normalized question, short TTL) so a
    repeated question skips the Groq call entirely; rate limiting protects this
    LLM-calling endpoint from being hammered.
    """
    start = time.perf_counter()

    cached = await get_cached_ask(request.question)
    if cached is not None:
        cached["latency_ms"] = (time.perf_counter() - start) * 1000
        return AskResponse(**cached, cached=True)

    response = await run_ask(session, request.question)
    await set_cached_ask(request.question, response.model_dump(exclude={"cached"}))
    return response
