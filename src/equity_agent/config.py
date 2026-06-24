"""Configuration: secrets from environment (.env), project params from config.yaml.

Two distinct concerns, deliberately kept apart:

* :class:`Settings` — secrets and machine-local runtime (API keys, DATABASE_URL).
  Read from environment / ``.env``. Never committed.
* :class:`ProjectConfig` — non-secret project parameters (universe, timeframe).
  Read from ``config.yaml``. Committed and code-reviewed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class Settings(BaseSettings):
    """Secrets and runtime settings, loaded from environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    environment: str = "dev"
    log_level: str = "INFO"

    database_url: str = "sqlite:///data/equity_agent.db"

    # LLM (Phase 2) — comma-separated priority list with failover, e.g. "groq,deepseek".
    llm_provider: str = "groq"
    groq_api_key: str | None = None
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    google_api_key: str | None = None  # Gemini, kept as a fallback
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Execution broker (Phase 4) — IBKR (TWS/Gateway). Alpaca dropped (not in Ukraine).
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497  # 7497 = TWS paper, 4002 = Gateway paper
    ibkr_client_id: int = 1

    # Fundamentals & news (Phase 1)
    finnhub_api_key: str | None = None
    fmp_api_key: str | None = None

    # Macro (Phase 1)
    fred_api_key: str | None = None

    # Monitoring (Phase 0+)
    sentry_dsn: str | None = None


class ProjectConfig(BaseModel):
    """Non-secret project parameters, loaded from ``config.yaml``."""

    universe: list[str] = Field(min_length=1)
    benchmark: str = "SPY"
    regime_symbols: list[str] = Field(default_factory=list)
    timeframe: str = "1d"
    history_start: str = "2015-01-01"
    fred_series: list[str] = Field(default_factory=list)
    data_dir: Path = Path("data")

    @property
    def all_data_symbols(self) -> list[str]:
        """Every symbol we need market data for: traded + benchmark + regime."""
        seen: dict[str, None] = {}
        for sym in [*self.universe, self.benchmark, *self.regime_symbols]:
            seen.setdefault(sym, None)
        return list(seen)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> ProjectConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ProjectConfig(**raw)
