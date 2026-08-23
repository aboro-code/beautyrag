from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.cache import redis_client
from app.db import engine
from app.routers.ask import router as ask_router
from app.routers.recommend import router as recommend_router
from app.routers.search import router as search_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await engine.dispose()
    await redis_client.aclose()


app = FastAPI(title="BeautyRAG", lifespan=lifespan)
app.include_router(search_router)
app.include_router(ask_router)
app.include_router(recommend_router)


@app.get("/health")
async def health():
    """Liveness/readiness check: confirms the API can reach both Postgres and Redis."""
    db_ok = True
    redis_ok = True

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    try:
        await redis_client.ping()
    except Exception:
        redis_ok = False

    status = "ok" if db_ok and redis_ok else "degraded"
    return {"status": status, "database": db_ok, "redis": redis_ok}
