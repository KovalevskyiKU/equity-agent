"""HTTP API for the trading app (FastAPI). The frontend talks to this; the core
research/execution code is reused, not reimplemented."""

from .app import app, create_app

__all__ = ["app", "create_app"]
