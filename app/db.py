from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

# statement_cache_size=0: required when connecting through Supabase's transaction-mode
# pooler (pgbouncer/Supavisor) -- it doesn't support asyncpg's server-side prepared
# statement caching across pooled connections, causing "prepared statement does not
# exist" errors on the second use of any query.
engine = create_async_engine(
    settings.database_url, pool_pre_ping=True, connect_args={"statement_cache_size": 0}
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncSession:
    """FastAPI dependency yielding a scoped async DB session."""
    async with AsyncSessionLocal() as session:
        yield session
