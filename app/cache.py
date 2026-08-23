import json

from redis.asyncio import Redis, from_url

from app.config import settings

redis_client: Redis = from_url(settings.redis_url, decode_responses=True)


def normalize_query(text: str) -> str:
    return " ".join(text.strip().lower().split())


async def get_cached_ask(question: str) -> dict | None:
    cached = await redis_client.get(f"ask:{normalize_query(question)}")
    return json.loads(cached) if cached else None


async def set_cached_ask(question: str, payload: dict) -> None:
    await redis_client.set(
        f"ask:{normalize_query(question)}", json.dumps(payload), ex=settings.ask_cache_ttl_seconds
    )
