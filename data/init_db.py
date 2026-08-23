"""Apply data/schema.sql against DATABASE_URL. Run once (idempotent, safe to re-run)."""

import asyncio
from pathlib import Path

import asyncpg

from app.config import settings


async def main() -> None:
    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    schema_sql = (Path(__file__).parent / "schema.sql").read_text()

    conn = await asyncpg.connect(dsn, statement_cache_size=0)
    try:
        await conn.execute(schema_sql)
        print("Schema applied successfully.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
