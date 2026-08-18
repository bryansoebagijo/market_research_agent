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
    gemini_model: str = "gemini-3.6-flash"

    # ---------- Market data keys (optional) ----------
    twelve_data_api_key: str = ""
    alphavantage_api_key: str = ""

    # ---------- Agent behaviour ----------
    max_subquestions: int = 5
    max_critic_loops: int = 2

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
