"""Structured logging + optional error monitoring (GlitchTip / Sentry)."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

from .config import get_settings

_CONFIGURED = False


def setup_logging(level: str | None = None) -> logging.Logger:
    """Idempotent root-logger setup. Returns the project logger."""
    global _CONFIGURED
    settings = get_settings()
    log_level = (level or settings.log_level).upper()

    if not _CONFIGURED:
        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%Y-%m-%d %H:%M:%S]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
        _CONFIGURED = True

    logger = logging.getLogger("equity_agent")
    logger.setLevel(log_level)
    return logger


def init_monitoring() -> bool:
    """Initialise Sentry/GlitchTip if a DSN is configured. Returns True if active."""
    settings = get_settings()
    if not settings.sentry_dsn:
        return False
    import sentry_sdk

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment,
        traces_sample_rate=0.0,
    )
    return True
