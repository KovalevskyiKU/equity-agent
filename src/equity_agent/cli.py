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
    strategy: str = typer.Option("buy-hold", help="buy-hold | vol-target | momentum (demo)"),
    lookback: int = typer.Option(20, help="lookback for the momentum demo"),
    fee_bps: float = typer.Option(1.0, help="per-side commission, bps"),
    slippage_bps: float = typer.Option(5.0, help="slippage vs open, bps"),
    max_dd_stop: float = typer.Option(0.0, help="drawdown circuit breaker, e.g. 0.15 (0 = off)"),
) -> None:
    """Backtest a baseline strategy on the universe vs SPY buy-and-hold."""
    setup_logging()
    init_db()
    from .backtest import strategy as strat
    from .backtest.engine import BacktestConfig, run_backtest
    from .backtest.metrics import return_summary
    from .backtest.panels import load_price_panels

    cfg = load_config()
    config = BacktestConfig(
        fee_bps=fee_bps, slippage_bps=slippage_bps, max_drawdown_stop=max_dd_stop or None
    )

    open_u, close_u = load_price_panels(cfg.universe)
    if open_u.empty:
        typer.echo("No price data. Run `eqa ingest` first.")
        raise typer.Exit(1)
    if strategy == "momentum":
        weights = strat.momentum_long_flat(close_u, lookback=lookback)
    elif strategy == "vol-target":
        weights = strat.vol_target_weights(close_u)
    else:
        weights = strat.buy_and_hold_equal(close_u)
    res = run_backtest(open_u, close_u, weights, config)

    # Benchmarks are passive — no circuit breaker, just fees/slippage.
    bench_cfg = BacktestConfig(fee_bps=fee_bps, slippage_bps=slippage_bps)
    basket = run_backtest(open_u, close_u, strat.buy_and_hold_equal(close_u), bench_cfg)
    open_b, close_b = load_price_panels([cfg.benchmark])
    bench = run_backtest(open_b, close_b, strat.single_asset(close_b, cfg.benchmark), bench_cfg)

    strat_m = return_summary(res.returns)
    basket_m = return_summary(basket.returns)
    bench_m = return_summary(bench.returns)
    typer.echo(f"\n=== {strategy} vs basket vs {cfg.benchmark} (full history) ===")
    typer.echo(f"{'metric':<14}{strategy:>14}{'basket':>14}{cfg.benchmark:>14}")
    for key in ("total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar"):
        typer.echo(f"{key:<14}{strat_m[key]:>14.3f}{basket_m[key]:>14.3f}{bench_m[key]:>14.3f}")
    typer.echo(f"{'n_trades':<14}{res.n_trades:>14}{basket.n_trades:>14}{bench.n_trades:>14}")
    typer.echo(f"{'turnover':<14}{res.turnover:>14.2f}{basket.turnover:>14.2f}{bench.turnover:>14.2f}")


@app.command()
def news(
    symbol: str = typer.Argument(..., help="ticker, e.g. AAPL"),
    days: int = typer.Option(7, help="lookback days of news"),
    model: str = typer.Option("llama-3.3-70b-versatile", help="LLM model"),
    limit: int = typer.Option(0, help="max new articles to score (0 = all)"),
) -> None:
    """Fetch recent news, score sentiment with Gemini, store, show daily sentiment."""
    from datetime import timedelta

    log = setup_logging()
    init_monitoring()
    init_db()
    from .signals.sentiment import fetch_score_store, get_daily_sentiment

    end = date.today()
    start = end - timedelta(days=days)
    result = fetch_score_store(symbol, start, end, model=model, limit=limit or None)
    log.info("news[%s]: fetched=%d scored=%d cached=%d", symbol, *(
        result["fetched"], result["scored"], result["cached"]))

    daily = get_daily_sentiment(symbol)
    if daily.empty:
        typer.echo("No scored news yet.")
    else:
        typer.echo("\nDaily impact-weighted sentiment (last 10):")
        typer.echo(daily.tail(10).round(3).to_string())


@app.command()
def decide(
    symbol: str = typer.Argument(..., help="ticker, e.g. AAPL"),
    model: str = typer.Option("llama-3.3-70b-versatile", help="LLM model"),
    kronos: bool = typer.Option(True, help="include the Kronos signal (one model run)"),
    max_weight: float = typer.Option(0.34, help="max portfolio weight per name"),
) -> None:
    """Today's LLM trading decision for a symbol, from the current point-in-time signals."""
    setup_logging()
    init_monitoring()
    init_db()
    from .decision.agent import decide as run_decide
    from .signals.bundle import build_bundle

    bundle = build_bundle(symbol, with_kronos=kronos)
    out = run_decide(bundle, model=model, max_weight=max_weight)

    typer.echo(f"\nSignals for {symbol}: {bundle}")
    typer.echo(
        f"\nDECISION: {out.action}  weight={out.target_weight:.2f}  "
        f"confidence={out.confidence:.2f}"
    )
    typer.echo(f"Rationale: {out.rationale}")


@app.command("backtest-llm")
def backtest_llm_cmd(
    months: int = typer.Option(6, help="window length in months"),
    end: str = typer.Option(None, help="window end date (ISO); default = latest"),
    rebalance_days: int = typer.Option(5, help="trading days between decisions"),
    max_weight: float = typer.Option(0.34, help="max weight per name"),
    model: str = typer.Option("llama-3.3-70b-versatile", help="LLM model"),
    kronos: bool = typer.Option(False, help="include Kronos in each decision (slow)"),
    sentiment: bool = typer.Option(False, help="include sentiment (recent only)"),
    delay: float = typer.Option(8.0, help="seconds between LLM calls (free-tier TPM)"),
) -> None:
    """Backtest the LLM decision agent on a recent window vs SPY (spends Gemini quota)."""
    log = setup_logging()
    init_monitoring()
    init_db()
    from .backtest.llm_backtest import run_llm_backtest

    cfg = load_config()
    rep = run_llm_backtest(
        cfg.universe,
        cfg.benchmark,
        months=months,
        end=end,
        rebalance_days=rebalance_days,
        max_weight=max_weight,
        model=model,
        with_kronos=kronos,
        with_sentiment=sentiment,
        delay=delay,
    )
    log.info(
        "LLM decisions: %d calls, %d failed, over %d dates",
        rep.n_calls, rep.n_failed, rep.n_decision_dates,
    )
    if rep.n_failed:
        typer.echo(
            f"WARNING: {rep.n_failed}/{rep.n_decision_dates} decision dates FAILED "
            "— the LLM column is unreliable for this run."
        )
    window = f"{months}mo ending {end}" if end else f"last {months}mo"
    typer.echo(f"\n=== LLM vs equal-weight basket vs {cfg.benchmark} — {window} ===")
    typer.echo(f"{'metric':<14}{'LLM':>12}{'voltgt':>12}{'basket':>12}{cfg.benchmark:>12}")
    for key in ("total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar"):
        typer.echo(
            f"{key:<14}{rep.strategy[key]:>12.3f}{rep.voltarget[key]:>12.3f}"
            f"{rep.basket[key]:>12.3f}{rep.benchmark[key]:>12.3f}"
        )


@app.command("backtest-sweep")
def backtest_sweep_cmd(
    window_months: int = typer.Option(6, help="window length in months"),
    step_months: int = typer.Option(2, help="months between window starts"),
    fee_bps: float = typer.Option(1.0, help="per-side commission, bps"),
    slippage_bps: float = typer.Option(5.0, help="slippage vs open, bps"),
) -> None:
    """Rolling-window comparison of mechanical strategies over full history (no LLM)."""
    setup_logging()
    init_db()
    from .backtest.sweep import run_sweep

    cfg = load_config()
    per, agg = run_sweep(
        cfg.universe,
        cfg.benchmark,
        window_months=window_months,
        step_months=step_months,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    n_windows = per["window_end"].nunique()
    typer.echo(f"\n=== Rolling {window_months}mo windows, step {step_months}mo, n={n_windows} ===")
    typer.echo(agg.to_string(index=False))

    # Head-to-head: vol-target vs basket across windows.
    sharpe = per.pivot(index="window_end", columns="strategy", values="sharpe")
    mdd = per.pivot(index="window_end", columns="strategy", values="max_drawdown")
    beat = float((sharpe["voltgt"] > sharpe["basket"]).mean()) * 100
    shallower = float((mdd["voltgt"] > mdd["basket"]).mean()) * 100
    typer.echo(f"\nvol-target beats basket on Sharpe in {beat:.0f}% of windows")
    typer.echo(f"vol-target shallower drawdown than basket in {shallower:.0f}% of windows")


@app.command("backtest-kronos")
def backtest_kronos_cmd(
    months: int = typer.Option(12, help="recent window length in months"),
    rebalance_days: int = typer.Option(21, help="trading days between decisions"),
    samples: int = typer.Option(10, help="Kronos sample paths per decision"),
    horizon: int = typer.Option(10, help="forecast horizon in trading days"),
    max_weight: float = typer.Option(0.20, help="max weight per name"),
) -> None:
    """Backtest a mechanical Kronos P(up) rule vs vol-target / basket / SPY (no LLM, slow)."""
    log = setup_logging()
    init_db()
    from .backtest.kronos_rule import run_kronos_backtest

    cfg = load_config()
    res = run_kronos_backtest(
        cfg.universe,
        cfg.benchmark,
        months=months,
        rebalance_days=rebalance_days,
        samples=samples,
        horizon=horizon,
        max_weight=max_weight,
    )
    log.info("Kronos-rule backtest done (window %dmo)", months)
    cols = ["kronos", "voltgt", "basket", "spy"]
    typer.echo(f"\n=== Kronos rule vs vol-target / basket / {cfg.benchmark} — last {months}mo ===")
    typer.echo(f"{'metric':<14}" + "".join(f"{c:>12}" for c in cols))
    for key in ("total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar"):
        typer.echo(f"{key:<14}" + "".join(f"{res[c][key]:>12.3f}" for c in cols))


@app.command("paper-reset")
def paper_reset_cmd(cash: float = typer.Option(100000.0, help="starting cash")) -> None:
    """Reset the paper-trading account (wipes positions/trades/equity)."""
    setup_logging()
    init_db()
    from .execution.paper_broker import reset_account

    reset_account(cash)
    typer.echo(f"Paper account reset to ${cash:,.2f}")


@app.command("paper-run")
def paper_run_cmd(
    fee_bps: float = typer.Option(1.0, help="per-side commission, bps"),
    slippage_bps: float = typer.Option(5.0, help="slippage vs price, bps"),
    cash: float = typer.Option(100000.0, help="starting cash if no account yet"),
) -> None:
    """Run one daily paper rebalance to the core strategy weights."""
    log = setup_logging()
    init_monitoring()
    init_db()
    from .execution.runner import run_paper

    res = run_paper(fee_bps=fee_bps, slippage_bps=slippage_bps, starting_cash=cash)
    log.info(
        "Paper: cash=$%.2f positions=$%.2f equity=$%.2f (%d names)",
        res["cash"], res["positions_value"], res["equity"], int(res["n_positions"]),
    )


@app.command("paper-status")
def paper_status_cmd() -> None:
    """Show the paper account: cash, equity, open positions."""
    setup_logging()
    init_db()
    from sqlalchemy import desc, select

    from .storage import session_scope
    from .storage.models import Account, EquitySnapshot, Position

    with session_scope() as s:
        acc = s.get(Account, 1)
        if acc is None:
            typer.echo("No paper account. Run `eqa paper-reset` then `eqa paper-run`.")
            return
        positions = s.scalars(select(Position)).all()
        last = s.scalars(select(EquitySnapshot).order_by(desc(EquitySnapshot.ts))).first()
        typer.echo(f"cash:   ${acc.cash:,.2f}   (start ${acc.starting_cash:,.2f})")
        if last is not None:
            typer.echo(f"equity: ${last.equity:,.2f}   positions: ${last.positions_value:,.2f}")
        typer.echo("positions:")
        for p in positions:
            typer.echo(f"  {p.symbol:6} qty={p.qty:.4f}  avg=${p.avg_price:.2f}")


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
