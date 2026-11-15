"""All configuration comes from environment variables (or .env). Nothing secret lives in code."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SecretStr prints as '**********', so the key can't leak via logs, errors or print()
    anthropic_api_key: SecretStr

    extraction_model: str = "claude-haiku-5-5"
    vision_model: str = "claude-haiku-5-5"  # used when a document is a scan
    max_output_tokens: int = 2000  # hard ceiling per call
    max_calls_per_run: int = 50  # hard ceiling per process run
    max_cost_usd_per_run: float = 0.50  # hard ceiling per process run

    database_url: str = "postgresql://orderdesk:orderdesk@localhost:5433/orderdesk"
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # 384 dimensions, runs locally, free


def get_settings() -> Settings:
    return Settings()
