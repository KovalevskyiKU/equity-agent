"""Command-line entry point: ``eqa <command>``."""

from __future__ import annotations

from datetime import date

import typer
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import __version__
from .config import load_config
from .data import YFinanceProvider, ingest_daily_bars
from .logging_setup import init_monitoring, setup_logging
from .signals.feature_store import build_feature_store
from .storage import init_db, session_scope
from .storage.models import DailyBar, Instrument

app = typer.Typer(add_completion=False, help="equity-agent CLI")


def _upsert_instrument(session: Session, symbol: str, role: str) -> None:
    inst = session.get(Instrument, symbol)
    if inst is None:
        session.add(Instrument(symbol=symbol, role=role))
    else:
        inst.role = role


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(f"equity-agent {__version__}")


@app.command()
def initdb() -> None:
    """Create tables and register the configured universe as instruments."""
    setup_logging()
    init_monitoring()
    init_db()
    cfg = load_config()
    with session_scope() as session:
        for sym in cfg.universe:
            _upsert_instrument(session, sym, role="traded")
        _upsert_instrument(session, cfg.benchmark, role="benchmark")
        for sym in cfg.regime_symbols:
            if sym != cfg.benchmark:
                _upsert_instrument(session, sym, role="regime")
    typer.echo("DB initialised; instruments registered.")


@app.command()
def ingest(
    start: str = typer.Option(None, help="ISO date; defaults to config.history_start"),
    end: str = typer.Option(None, help="ISO date; defaults to today"),
) -> None:
    """Fetch and store daily bars for the full data universe (traded + benchmark + regime)."""
    log = setup_logging()
    init_monitoring()
    init_db()
    cfg = load_config()
    start_d = date.fromisoformat(start) if start else date.fromisoformat(cfg.history_start)
    end_d = date.fromisoformat(end) if end else date.today()

    log.info("Ingesting %s -> %s for %s", start_d, end_d, cfg.all_data_symbols)
    result = ingest_daily_bars(cfg.all_data_symbols, start_d, end_d, YFinanceProvider())
    total = sum(result.values())
    log.info("Done. Inserted %d new bars across %d symbols.", total, len(result))


@app.command()
def features() -> None:
    """Build the per-symbol feature store (Parquet) from stored daily bars."""
    log = setup_logging()
    init_monitoring()
    init_db()
    counts = build_feature_store()
    total = sum(counts.values())
    log.info("Feature store built: %d rows across %d symbols", total, len(counts))


@app.command()
def status() -> None:
    """Show how many bars are stored per symbol."""
    init_db()
    with session_scope() as session:
        rows = session.execute(
            select(
                DailyBar.symbol,
                func.count(DailyBar.id),
                func.min(DailyBar.ts),
                func.max(DailyBar.ts),
            ).group_by(DailyBar.symbol)
        ).all()
    if not rows:
        typer.echo("No bars stored yet. Run `eqa ingest`.")
        return
    for symbol, n, first, last in rows:
        typer.echo(f"{symbol:8} {n:6} bars   {first} -> {last}")


if __name__ == "__main__":
    app()
