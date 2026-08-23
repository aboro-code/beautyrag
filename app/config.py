from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str
    database_url: str
    redis_url: str

    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    groq_model: str = "openai/gpt-oss-20b"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    ask_cache_ttl_seconds: int = 300
    rate_limit_per_minute: int = 30


settings = Settings()
