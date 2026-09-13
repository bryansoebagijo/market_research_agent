"""
Central configuration, driven by environment variables (.env).

    from config import settings
    print(settings.gemini_model)
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # ---------- Reasoning LLM ----------
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"

    # ---------- Market data keys (optional) ----------
    twelve_data_api_key: str = ""
    alphavantage_api_key: str = ""

    # ---------- Agent behaviour ----------
    max_subquestions: int = 5
    max_critic_loops: int = 2

    # ---------- Evaluation ----------
    # The judge should differ from the model being judged, so it doesn't share its blind
    # spots. Set either key below and the judge switches to that provider automatically —
    # genuinely independent, no code change. With neither set it falls back to a
    # different Gemini model, and the evaluation report says plainly that the grade is
    # NOT provider-independent rather than implying more rigour than it has.
    anthropic_api_key: str = ""
    anthropic_judge_model: str = "claude-sonnet-5"
    openai_api_key: str = ""
    openai_judge_model: str = "gpt-4o"
    judge_model: str = "gemini-3.6-flash"  # same-family fallback
    judge_delay_seconds: float = 4.0  # pace judge calls against free-tier limits

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
