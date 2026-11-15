"""All configuration comes from environment variables (or .env). Nothing secret lives in code."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SecretStr prints as '**********', so the key can't leak via logs, errors or print()
    # Optional so the public demo (which never calls Claude) can run without a key.
    anthropic_api_key: SecretStr = SecretStr("")

    extraction_model: str = "claude-haiku-5-5"
    vision_model: str = "claude-haiku-5-5"  # used when a document is a scan
    agent_model: str = "claude-sonnet-5-5"  # investigation needs stronger reasoning
    agent_max_steps: int = 8  # hard ceiling on tool-calling rounds per investigation
    max_output_tokens: int = 2000  # hard ceiling per call
    max_calls_per_run: int = 50  # hard ceiling per process run
    max_cost_usd_per_run: float = 0.50  # hard ceiling per process run

    database_url: str = "postgresql://orderdesk:orderdesk@localhost:5433/orderdesk"
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # 384 dimensions, runs locally, free

    # A searched match is only trusted if it's similar enough AND clearly ahead of the runner-up.
    # Calibrate with: uv run python evals/match_confidence_report.py
    match_min_similarity: float = 0.75
    match_min_margin: float = 0.02

    # Web login. Set SECURE_COOKIES=true when served over HTTPS (always, in production).
    session_hours: int = 12
    secure_cookies: bool = False

    # Public demo: recorded agent results, shared demo login, a "reset demo" button.
    # NEVER enable on a real deployment: reset deletes every document and decision.
    demo_mode: bool = False
    demo_email: str = "demo@orderdesk.app"
    demo_password: SecretStr = SecretStr("")

    erp_base_url: str = "http://localhost:8100"
    # Two keys on purpose: the agent only ever gets the read key (least privilege).
    erp_read_key: SecretStr = SecretStr("dev-read-key")
    erp_write_key: SecretStr = SecretStr("dev-write-key")


def get_settings() -> Settings:
    return Settings()
