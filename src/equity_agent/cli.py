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
from .research.signal_eval import run as run_signal_eval
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
def research(
    horizon: list[int] = typer.Option(None, "--horizon", help="forward-return horizons, days"),
) -> None:
    """Score each feature's predictive edge (IC + quantile spread) vs forward returns."""
    setup_logging()
    init_monitoring()
    init_db()
    horizons = tuple(horizon) if horizon else (1, 5, 10)
    results = run_signal_eval(horizons=horizons)
    for h, table in results.items():
        if table.empty:
            typer.echo(f"\n[h={h}] no data — run `eqa ingest` and `eqa features` first.")
            continue
        typer.echo(f"\n=== forward horizon = {h} day(s) — features ranked by |IC| ===")
        typer.echo(table.to_string(index=False))


@app.command("kronos-signal")
def kronos_signal_cmd(
    symbol: str = typer.Argument(..., help="ticker, e.g. AAPL"),
    horizon: int = typer.Option(10, help="forecast horizon in trading days"),
    samples: int = typer.Option(20, help="number of Kronos sample paths"),
    lookback: int = typer.Option(256, help="history bars fed to the model"),
) -> None:
    """Current Kronos probabilistic signal for a symbol (uses latest stored bars)."""
    setup_logging()
    init_db()
    # Imported lazily — these pull in torch, which the rest of the CLI doesn't need.
    from .signals.feature_store import load_bars
    from .signals.kronos_signal import KronosForecaster

    bars = load_bars(symbol)
    if len(bars) < lookback:
        typer.echo(f"Not enough bars for {symbol} ({len(bars)} < {lookback}). Run `eqa ingest`.")
        raise typer.Exit(1)

    window = bars.tail(lookback)
    forecaster = KronosForecaster()
    sig = forecaster.signal(window, horizon=horizon, sample_count=samples)
    last_close = float(window["close"].iloc[-1])

    typer.echo(f"{symbol}  last_close={last_close:.2f}  horizon={horizon}d  samples={samples}")
    typer.echo(f"  p_up    = {sig['k_p_up']:.3f}")
    typer.echo(f"  exp_ret = {sig['k_exp_ret'] * 100:+.2f}%")
    typer.echo(f"  ret_std = {sig['k_ret_std'] * 100:.2f}%")


@app.command("kronos-eval")
def kronos_eval_cmd(
    symbol: str = typer.Argument(..., help="ticker, e.g. AAPL"),
    horizon: int = typer.Option(10, help="forward horizon in trading days"),
    points: int = typer.Option(60, help="historical as-of points to evaluate"),
    samples: int = typer.Option(12, help="Kronos sample paths per point"),
    step: int = typer.Option(4, help="trading days between as-of points"),
    lookback: int = typer.Option(256, help="history bars fed to the model"),
) -> None:
    """Indicative edge read: IC of the Kronos signal vs forward returns (compute-heavy)."""
    log = setup_logging()
    init_db()
    from .config import PROJECT_ROOT, load_config
    from .research.kronos_eval import evaluate_kronos, summarize

    df = evaluate_kronos(
        symbol, horizon=horizon, points=points, sample_count=samples, step=step, lookback=lookback
    )
    if df.empty:
        typer.echo(f"No data for {symbol}. Run `eqa ingest` first.")
        raise typer.Exit(1)

    reports_dir = PROJECT_ROOT / load_config().data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(reports_dir / f"kronos_eval_{symbol}_h{horizon}.csv", index=False)

    log.info("Evaluated %d points for %s (horizon=%d)", len(df), symbol, horizon)
    typer.echo(f"\n=== Kronos signal IC vs {horizon}d forward return — {symbol} ===")
    typer.echo(summarize(df).to_string(index=False))


@app.command()
def backtest(
    strategy: str = typer.Option("buy-hold", help="buy-hold | momentum (demo)"),
    lookback: int = typer.Option(20, help="lookback for the momentum demo"),
    fee_bps: float = typer.Option(1.0, help="per-side commission, bps"),
    slippage_bps: float = typer.Option(5.0, help="slippage vs open, bps"),
) -> None:
    """Backtest a baseline strategy on the universe vs SPY buy-and-hold."""
    setup_logging()
    init_db()
    from .backtest import strategy as strat
    from .backtest.engine import BacktestConfig, run_backtest
    from .backtest.metrics import return_summary
    from .backtest.panels import load_price_panels

    cfg = load_config()
    config = BacktestConfig(fee_bps=fee_bps, slippage_bps=slippage_bps)

    open_u, close_u = load_price_panels(cfg.universe)
    if open_u.empty:
        typer.echo("No price data. Run `eqa ingest` first.")
        raise typer.Exit(1)
    weights = (
        strat.momentum_long_flat(close_u, lookback=lookback)
        if strategy == "momentum"
        else strat.buy_and_hold_equal(close_u)
    )
    res = run_backtest(open_u, close_u, weights, config)

    open_b, close_b = load_price_panels([cfg.benchmark])
    bench = run_backtest(open_b, close_b, strat.single_asset(close_b, cfg.benchmark), config)

    strat_m = return_summary(res.returns)
    bench_m = return_summary(bench.returns)
    typer.echo(f"\n=== {strategy} (universe) vs {cfg.benchmark} buy-and-hold ===")
    typer.echo(f"{'metric':<14}{strategy:>14}{cfg.benchmark:>14}")
    for key in ("total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar"):
        typer.echo(f"{key:<14}{strat_m[key]:>14.3f}{bench_m[key]:>14.3f}")
    typer.echo(f"{'n_trades':<14}{res.n_trades:>14}{bench.n_trades:>14}")
    typer.echo(f"{'turnover':<14}{res.turnover:>14.2f}{bench.turnover:>14.2f}")


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
