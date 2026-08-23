import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import settings

# Connecting through Supabase's transaction-mode pooler (pgbouncer/Supavisor) needs all
# three of these, or asyncpg raises "prepared statement does not exist" / "already
# exists":
# 1. statement_cache_size=0 -- asyncpg's own client-side prepared-statement cache isn't
#    valid across pooled connections that can be swapped server-side between queries.
# 2. NullPool -- SQLAlchemy's own connection pool must be disabled too, so the pooler
#    (which already does the real pooling) is the only thing multiplexing connections.
# 3. prepared_statement_name_func -- asyncpg names statements with a plain per-connection
#    counter ("__asyncpg_stmt_1__", "_2__", ...) that restarts at 1 for every new
#    connection. Since the pooler can route two different client connections to the same
#    backend session, two concurrent requests can both try to prepare "_1__" and collide.
#    Generating a globally-unique name per statement avoids that regardless of how the
#    pooler multiplexes things underneath.
engine = create_async_engine(
    settings.database_url,
    poolclass=NullPool,
    connect_args={
        "statement_cache_size": 0,
        "prepared_statement_name_func": lambda: f"__asyncpg_{uuid.uuid4()}__",
    },
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncSession:
    """FastAPI dependency yielding a scoped async DB session."""
    async with AsyncSessionLocal() as session:
        yield session
