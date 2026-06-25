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

    from .config import get_settings

    if get_settings().fred_api_key and cfg.fred_series:
        from .data.fred import ingest_fred

        fred = ingest_fred(cfg.fred_series, start_d, end_d)
        log.info("FRED: %d obs across %d series", sum(fred.values()), len(fred))


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
    typer.echo(
        "NOTE: 'basket'/'strategy' use TODAY'S universe over all history -> "
        f"survivorship-biased (inflated). {cfg.benchmark} (cap-weight) is the honest "
        "bar; for the corrected read use `factor-backtest-pit` (see docs/METHODOLOGY.md)."
    )
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
    typer.echo(
        "NOTE: uses TODAY'S universe over all history -> survivorship-biased. The "
        "honest, point-in-time read is `factor-backtest-pit` (see docs/METHODOLOGY.md)."
    )
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
    risk_off: bool = typer.Option(False, help="apply LLM news risk-off gate"),
    news_days: int = typer.Option(3, help="news lookback for the risk-off gate"),
) -> None:
    """Run one daily paper rebalance to the core strategy weights."""
    log = setup_logging()
    init_monitoring()
    init_db()
    from .execution.runner import run_paper

    res = run_paper(
        fee_bps=fee_bps, slippage_bps=slippage_bps, starting_cash=cash,
        risk_off=risk_off, news_days=news_days,
    )
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
def daily(
    risk_off: bool = typer.Option(True, help="apply the LLM news risk-off gate"),
) -> None:
    """Full daily cycle: ingest latest bars -> (features for non-spy cores) ->
    paper rebalance -> monitor snapshot."""
    from datetime import UTC, datetime, timedelta

    log = setup_logging()
    init_monitoring()
    init_db()
    from .data import YFinanceProvider, ingest_daily_bars
    from .execution.runner import run_paper
    from .signals.feature_store import build_feature_store

    cfg = load_config()
    today = datetime.now(UTC).date()
    ingest_daily_bars(cfg.all_data_symbols, today - timedelta(days=10), today, YFinanceProvider())
    # The per-symbol feature store feeds the LLM decision agent / research, not the
    # SPY core or the news risk-off gate — skip its (heavy) rebuild for the spy core.
    if cfg.core_strategy != "spy":
        build_feature_store()
    else:
        log.info("core=spy: skipping feature-store rebuild")
    res = run_paper(risk_off=risk_off)
    log.info(
        "Daily done: equity=$%.2f cash=$%.2f (%d names held)",
        res["equity"], res["cash"], int(res["n_positions"]),
    )

    # End-of-cycle monitor snapshot (drawdown / Sharpe / tracking vs benchmark).
    from .backtest.panels import load_price_panels
    from .dashboard.data import paper_overview
    from .monitoring import monitor_summary

    _, close_b = load_price_panels([cfg.benchmark])
    spy_close = close_b[cfg.benchmark] if not close_b.empty else None
    mon = monitor_summary(paper_overview()["equity_curve"], spy_close)
    if "total_return" in mon:
        log.info(
            "Monitor: total %.2f%%  maxDD %.2f%%  Sharpe %.2f%s",
            mon["total_return"] * 100,
            mon["max_drawdown"] * 100,
            mon["sharpe"],
            f"  excess vs {cfg.benchmark} {mon['excess_vs_spy'] * 100:+.2f}%"
            if "excess_vs_spy" in mon else "",
        )


@app.command()
def dashboard(port: int = typer.Option(8501, help="port to serve on")) -> None:
    """Launch the Streamlit dashboard (needs the [ui] extra)."""
    import subprocess
    import sys

    from .config import PROJECT_ROOT

    app_path = PROJECT_ROOT / "dashboard_app.py"
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)],
        check=False,
    )


@app.command()
def walkforward(
    horizon: int = typer.Option(10, help="forward horizon in trading days"),
    n_splits: int = typer.Option(6, help="walk-forward folds"),
    alpha: float = typer.Option(0.1, help="ridge regularisation"),
) -> None:
    """Payoff test: does a walk-forward ridge model on the feature cluster beat the basket OOS?"""
    from typing import cast

    setup_logging()
    init_db()
    from .research.wf_strategy import run_walkforward_strategy

    cfg = load_config()
    res = run_walkforward_strategy(horizon=horizon, n_splits=n_splits, alpha=alpha)
    if not res:
        typer.echo("No data. Run `eqa ingest` and `eqa features` first.")
        raise typer.Exit(1)

    typer.echo(
        f"\nOOS IC = {cast(float, res['oos_ic']):.4f} "
        f"(t={cast(float, res['oos_t']):.2f}, n={res['n_oos']}), {res['n_dates']} OOS days"
    )
    strat = cast(dict, res["strategy"])
    basket = cast(dict, res["basket"])
    spy = cast(dict, res["spy"])
    typer.echo(f"\n{'metric':<14}{'WF-model':>12}{'basket':>12}{cfg.benchmark:>12}")
    for key in ("total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar"):
        typer.echo(f"{key:<14}{strat[key]:>12.3f}{basket[key]:>12.3f}{spy[key]:>12.3f}")


@app.command("backtest-overlay")
def backtest_overlay_cmd(
    band: float = typer.Option(0.05, help="no-trade band on exposure (cuts churn)"),
    lookback: int = typer.Option(20, help="realized-vol lookback, days"),
) -> None:
    """SPY buy-hold vs vol-target overlay across target vols (total-return, net of costs).

    The one validated improvement: the overlay gives up absolute return for a better
    Sharpe/Calmar and a much shallower drawdown. Enable via config.risk_overlay.
    """
    setup_logging()
    init_db()
    from .backtest.overlay_backtest import run_overlay_comparison

    df = run_overlay_comparison(band=band, lookback=lookback)
    if df.empty:
        typer.echo("No price data. Run `eqa ingest` first.")
        raise typer.Exit(1)
    cfg = load_config()
    typer.echo(
        f"\n=== {cfg.benchmark} buy-hold vs vol-target overlay "
        "(total-return, net costs) ==="
    )
    typer.echo(df.to_string(index=False))


@app.command("live-run")
def live_run_cmd(
    execute: bool = typer.Option(
        False, "--execute", help="TRANSMIT real orders to IBKR (default: dry-run, plan only)"
    ),
    min_notional: float = typer.Option(50.0, help="skip orders below this notional"),
) -> None:
    """Plan (and with --execute, transmit) IBKR orders to reach the core target.

    Dry-run by default: it connects to TWS/Gateway, prints the orders it WOULD place,
    and sends nothing. Needs the [ibkr] extra and a running TWS/IB Gateway.
    """
    setup_logging()
    init_db()
    from .backtest.panels import load_price_panels
    from .config import get_settings
    from .execution.ibkr_broker import IBKRBroker
    from .execution.runner import compute_core_target

    cfg = load_config()
    s = get_settings()
    _, close_b = load_price_panels([cfg.benchmark])
    close_u = load_price_panels(cfg.universe)[1] if cfg.core_strategy != "spy" else None
    if close_b.empty or (close_u is not None and close_u.empty):
        typer.echo("No price data. Run `eqa ingest` first.")
        raise typer.Exit(1)
    import pandas as pd

    u_panel = close_u if close_u is not None else pd.DataFrame()
    target, prices = compute_core_target(cfg.core_strategy, u_panel, close_b, cfg.benchmark)

    broker = IBKRBroker(s.ibkr_host, s.ibkr_port, s.ibkr_client_id)
    try:
        broker.connect()
    except Exception as e:  # noqa: BLE001 - surface any connection problem clearly
        typer.echo(
            f"Could not connect to IBKR at {s.ibkr_host}:{s.ibkr_port} "
            f"- is TWS / IB Gateway running and the API enabled? ({e})"
        )
        raise typer.Exit(1) from e
    try:
        orders = broker.rebalance(target, prices, execute=execute, min_notional=min_notional)
    finally:
        broker.disconnect()

    mode = "TRANSMITTED" if execute else "DRY-RUN (nothing sent)"
    typer.echo(f"\n=== IBKR {mode} - core={cfg.core_strategy} ===")
    for o in orders:
        typer.echo(
            f"  {o.side:<4} {o.symbol:<6} qty={o.qty:g}  "
            f"~${o.est_notional:,.0f} @ {o.est_price:.2f}"
        )
    if not orders:
        typer.echo("  (no orders - already at target)")


@app.command("factor-ic")
def factor_ic_cmd(
    horizon: int = typer.Option(21, help="forward horizon in trading days (~1 month)"),
    min_names: int = typer.Option(30, help="min rankable names required per date"),
) -> None:
    """Cross-sectional (per-date) IC of price-only factors over the expanded universe.

    Monthly is the honest read (non-overlapping ~horizon windows); daily is shown
    for contrast (overlapping -> inflated t).
    """
    setup_logging()
    init_db()
    from .research.factor_eval import run_cross_sectional_ic

    cfg = load_config()
    res, n = run_cross_sectional_ic(cfg.universe, horizon=horizon, min_names=min_names)
    if not res:
        typer.echo("No price data. Run `eqa ingest` and `eqa features` first.")
        raise typer.Exit(1)

    typer.echo(f"\n=== Cross-sectional IC vs {horizon}d forward return - {n} names ===")
    typer.echo(
        f"{'factor':<16}{'freq':<9}{'mean_ic':>9}{'t_stat':>9}"
        f"{'ic_ir':>8}{'hit%':>7}{'n_dt':>6}{'n_nm':>7}"
    )
    for name, freqs in res.items():
        for freq, s in (("monthly", freqs["monthly"]), ("daily", freqs["daily"])):
            typer.echo(
                f"{name:<16}{freq:<9}{s.mean_ic:>9.4f}{s.t_stat:>9.2f}{s.ic_ir:>8.3f}"
                f"{s.hit_rate * 100:>7.1f}{s.n_dates:>6}{s.mean_n_names:>7.0f}"
            )


@app.command("factor-backtest")
def factor_backtest_cmd(
    q: float = typer.Option(0.2, help="top-quantile fraction (0.2 = top quintile)"),
    min_names: int = typer.Option(30, help="min rankable names required per rebalance"),
    fee_bps: float = typer.Option(1.0, help="per-side commission, bps"),
    slippage_bps: float = typer.Option(5.0, help="slippage vs open, bps"),
) -> None:
    """Monthly top-quantile factor portfolios vs the equal-weight basket + SPY (net of costs)."""
    setup_logging()
    init_db()
    from .backtest.factor_portfolio import run_factor_portfolios

    cfg = load_config()
    res = run_factor_portfolios(
        cfg.universe, cfg.benchmark, q=q, min_names=min_names,
        fee_bps=fee_bps, slippage_bps=slippage_bps,
    )
    if not res:
        typer.echo("No price data. Run `eqa ingest` and `eqa features` first.")
        raise typer.Exit(1)

    from typing import cast

    from .backtest.factor_portfolio import FactorBacktest

    spy = cast(dict, res["spy"])
    factors = cast(dict, res["factors"])
    keys = ("total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar")
    typer.echo(
        f"\n=== Monthly factor portfolios (top {q:.0%}) vs basket vs SPY "
        f"- {res['n_symbols']} names ==="
    )
    for name, fb in factors.items():
        fb = cast(FactorBacktest, fb)
        typer.echo(f"\n--- {name} ---  turnover={fb.turnover:.1f}  trades={fb.n_trades}")
        typer.echo(f"{'metric':<14}{'top-Q':>12}{'basket':>12}{'SPY':>12}")
        for key in keys:
            typer.echo(f"{key:<14}{fb.portfolio[key]:>12.3f}{fb.basket[key]:>12.3f}{spy[key]:>12.3f}")
        ls = fb.long_short
        typer.echo(
            f"long-short (idealized, gross): ann_ret={ls['ann_return']:.3f} "
            f"sharpe={ls['sharpe']:.2f} hit={ls['hit_rate'] * 100:.0f}% n={ls['n_periods']}"
        )


@app.command("ingest-fundamentals")
def ingest_fundamentals_cmd(
    union: bool = typer.Option(
        True, help="include point-in-time dropped names (else current only)"
    ),
) -> None:
    """Fetch point-in-time annual fundamentals (Finnhub as-reported) for the universe."""
    log = setup_logging()
    init_db()
    from .data.fundamentals import ingest_fundamentals
    from .data.sp500_history import ever_members, fetch_sp500_changes

    cfg = load_config()
    if union:
        changes = fetch_sp500_changes()
        symbols = ever_members(cfg.universe, changes, f"{cfg.history_start[:4]}-01-01")
    else:
        symbols = list(cfg.universe)
    log.info("Fetching fundamentals for %d symbols", len(symbols))
    res = ingest_fundamentals(symbols)
    got = sum(1 for v in res.values() if v > 0)
    log.info("Fundamentals: %d/%d symbols with data", got, len(symbols))


@app.command("factor-backtest-pit")
def factor_backtest_pit_cmd(
    q: float = typer.Option(0.2, help="top-quantile fraction (0.2 = top quintile)"),
    min_names: int = typer.Option(30, help="min rankable members required per rebalance"),
    fundamentals: bool = typer.Option(False, help="include value/quality fundamental factors"),
    fee_bps: float = typer.Option(1.0, help="per-side commission, bps"),
    slippage_bps: float = typer.Option(5.0, help="slippage vs open, bps"),
) -> None:
    """Survivorship-corrected factor backtest: rank only point-in-time index members.

    Reconstructs historical S&P 500 membership from Wikipedia (needs network) and
    masks each rebalance so only names actually in the index that day are ranked.
    Compare against `eqa factor-backtest` (today's members over all history).
    """
    setup_logging()
    init_db()
    from .backtest.factor_portfolio import run_pit_factor_portfolios
    from .data.sp500_history import ever_members, fetch_sp500_changes

    cfg = load_config()
    changes = fetch_sp500_changes()
    union = ever_members(cfg.universe, changes, f"{cfg.history_start[:4]}-01-01")
    res = run_pit_factor_portfolios(
        union, cfg.universe, changes, cfg.benchmark,
        q=q, min_names=min_names, fee_bps=fee_bps, slippage_bps=slippage_bps,
        with_fundamentals=fundamentals,
    )
    if not res:
        typer.echo("No price data. Run `eqa ingest` first.")
        raise typer.Exit(1)

    from typing import cast

    basket = cast(dict, res["basket"])
    spy = cast(dict, res["spy"])
    factors = cast(dict, res["factors"])
    keys = ("total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar")
    typer.echo(
        f"\n=== POINT-IN-TIME factor portfolios (top {q:.0%}) vs member basket vs SPY ==="
    )
    typer.echo(
        f"union universe={res['n_symbols']} names, "
        f"avg rankable members/rebalance={res['mean_members']:.0f}"
    )
    for name, fb in factors.items():
        fb = cast(dict, fb)
        typer.echo(f"\n--- {name} ---  turnover={fb['turnover']:.1f}  trades={fb['n_trades']}")
        typer.echo(f"{'metric':<14}{'top-Q':>12}{'basket':>12}{'SPY':>12}")
        port = cast(dict, fb["portfolio"])
        for key in keys:
            typer.echo(f"{key:<14}{port[key]:>12.3f}{basket[key]:>12.3f}{spy[key]:>12.3f}")
        ls = cast(dict, fb["long_short"])
        typer.echo(
            f"long-short (idealized, gross): ann_ret={ls['ann_return']:.3f} "
            f"sharpe={ls['sharpe']:.2f} hit={ls['hit_rate'] * 100:.0f}% n={ls['n_periods']}"
        )


@app.command()
def monitor() -> None:
    """Monitor the paper account: equity, last-run P&L, drawdown, Sharpe, tracking vs SPY."""
    setup_logging()
    init_db()
    from typing import cast

    import pandas as pd

    from .backtest.panels import load_price_panels
    from .dashboard.data import paper_overview
    from .monitoring import monitor_summary

    cfg = load_config()
    ov = paper_overview()
    if not ov["has_account"]:
        typer.echo("No paper account. Run `eqa paper-reset` then `eqa paper-run`.")
        return

    _, close_b = load_price_panels([cfg.benchmark])
    spy_close = close_b[cfg.benchmark] if not close_b.empty else None
    s = monitor_summary(cast(pd.DataFrame, ov["equity_curve"]), spy_close)
    if s.get("snapshots", 0) == 0:
        typer.echo("No equity snapshots yet. Run `eqa paper-run`.")
        return

    typer.echo(f"\n=== paper monitor ({s['snapshots']} snapshots) ===")
    typer.echo(f"equity        ${s['equity']:,.2f}")
    if "total_return" in s:
        typer.echo(f"total return  {s['total_return'] * 100:+.2f}%")
        typer.echo(f"last P&L      ${s['last_pnl']:+,.2f} ({s['last_pnl_pct'] * 100:+.2f}%)")
        typer.echo(f"max drawdown  {s['max_drawdown'] * 100:.2f}%")
        typer.echo(f"sharpe        {s['sharpe']:.2f}   ann vol {s['ann_vol'] * 100:.1f}%")
    if "excess_vs_spy" in s:
        typer.echo(
            f"vs {cfg.benchmark:<6}    SPY {s['spy_return'] * 100:+.2f}%  "
            f"-> excess {s['excess_vs_spy'] * 100:+.2f}%"
        )
    elif "total_return" in s:
        typer.echo(f"vs {cfg.benchmark}: need >=2 snapshots spanning trading days")


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
