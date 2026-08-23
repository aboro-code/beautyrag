from fastapi import HTTPException, Request

from app.cache import redis_client
from app.config import settings

WINDOW_SECONDS = 60


async def enforce_rate_limit(request: Request) -> None:
    """Basic fixed-window rate limiter (per client IP, backed by Redis) to keep the
    Groq-calling /ask endpoint from being hammered in a demo/interview setting.
    """
    client_ip = request.client.host if request.client else "unknown"
    key = f"ratelimit:{client_ip}"

    count = await redis_client.incr(key)
    if count == 1:
        await redis_client.expire(key, WINDOW_SECONDS)

    if count > settings.rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
