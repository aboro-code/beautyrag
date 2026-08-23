from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.rag import ask as run_ask
from app.schemas import AskRequest, AskResponse

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, session: AsyncSession = Depends(get_session)) -> AskResponse:
    """Grounded RAG Q&A: hybrid-search the catalog, then answer using only that retrieved
    context. The model is instructed to say it doesn't have enough information rather than
    inventing a product or claim, and cited_product_ids reflects only products the model
    actually cited in its answer -- not just whatever was retrieved.
    """
    return await run_ask(session, request.question)
