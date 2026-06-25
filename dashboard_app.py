"""Streamlit dashboard. Launch: `python app.py`, `eqa dashboard`, or
`streamlit run dashboard_app.py`."""

import streamlit as st

from equity_agent.backtest.overlay_backtest import run_overlay_comparison
from equity_agent.backtest.panels import load_price_panels
from equity_agent.config import load_config
from equity_agent.dashboard.data import (
    factor_leaderboard,
    factor_performance,
    paper_overview,
    recent_news,
    strategy_curves,
)
from equity_agent.monitoring import monitor_summary
from equity_agent.storage.db import init_db

st.set_page_config(page_title="equity-agent", layout="wide")
init_db()
cfg = load_config()
st.title("equity-agent")
st.caption(
    f"core = {cfg.core_strategy} (tracks {cfg.benchmark}). Research verdict: no factor "
    f"beats {cfg.benchmark} once survivorship is removed — see docs/PHASE1_FINDINGS.md."
)


# Heavy compute is cached so the page stays responsive after first load.
@st.cache_data(show_spinner="Backtesting strategies…")
def _curves():  # type: ignore[no-untyped-def]
    return strategy_curves()


@st.cache_data(show_spinner="Backtesting factor portfolios…")
def _performance():  # type: ignore[no-untyped-def]
    return factor_performance()


@st.cache_data(show_spinner="Ranking current factor picks…")
def _leaderboard():  # type: ignore[no-untyped-def]
    return factor_leaderboard()


@st.cache_data(show_spinner="Loading benchmark…")
def _spy_close():  # type: ignore[no-untyped-def]
    _, cb = load_price_panels([cfg.benchmark])
    return cb[cfg.benchmark] if not cb.empty else None


@st.cache_data(show_spinner="Backtesting the risk overlay…")
def _overlay():  # type: ignore[no-untyped-def]
    return run_overlay_comparison()


portfolio_tab, predictions_tab, backtest_tab, overlay_tab, signals_tab = st.tabs(
    ["Paper portfolio", "Predictions", "Backtest (% profit)", "Risk overlay", "Signals & news"]
)

with portfolio_tab:
    ov = paper_overview()
    if not ov["has_account"]:
        st.info("No paper account yet. Run `eqa paper-reset` then `eqa paper-run`.")
    else:
        curve = ov["equity_curve"]
        s = monitor_summary(curve, _spy_close())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Equity", f"${s.get('equity', float(ov['cash'])):,.0f}")
        c2.metric("Total return", f"{s.get('total_return', 0.0) * 100:+.2f}%")
        c3.metric("Max drawdown", f"{s.get('max_drawdown', 0.0) * 100:.2f}%")
        if "excess_vs_spy" in s:
            c4.metric(f"Excess vs {cfg.benchmark}", f"{s['excess_vs_spy'] * 100:+.2f}%")
        else:
            c4.metric("Sharpe", f"{s.get('sharpe', float('nan')):.2f}")
        if len(curve):
            st.line_chart(curve.set_index("ts")["equity"])
        st.subheader("Positions")
        st.dataframe(ov["positions"], width="stretch")
        st.subheader("Recent trades")
        st.dataframe(ov["trades"], width="stretch")

with predictions_tab:
    st.write(
        "Top names each factor **favors right now** (today's universe). Informational: "
        "backtested net of costs, these factors do **not** beat "
        f"{cfg.benchmark} — this is what they pick, not a buy list."
    )
    board = _leaderboard()
    if not board:
        st.info("No price data. Run `eqa ingest`.")
    else:
        cols = st.columns(len(board))
        for col, (name, df) in zip(cols, board.items(), strict=False):
            col.subheader(name)
            col.dataframe(df, width="stretch", hide_index=True)

with backtest_tab:
    st.write(
        f"Monthly top-quintile factor portfolios vs **{cfg.benchmark}** (buy & hold). "
        "⚠️ Over today's universe → survivorship-biased (inflated). The honest, "
        "point-in-time read is `eqa factor-backtest-pit`."
    )
    perf = _performance()
    if perf.empty:
        st.info("No price data. Run `eqa ingest`.")
    else:
        st.dataframe(perf, width="stretch", hide_index=True)
    curves, metrics = _curves()
    if not curves.empty:
        st.subheader("Core vs basket vs benchmark — equity curves")
        st.line_chart(curves)
        st.dataframe(metrics, width="stretch", hide_index=True)

with overlay_tab:
    st.write(
        f"**The one validated improvement.** {cfg.benchmark} buy-and-hold vs a vol-target "
        "overlay (scale exposure by target/realized vol, rest in cash), total-return, net "
        "of costs. It gives up absolute return for a **better Sharpe/Calmar and ~half the "
        "drawdown** — crash insurance, with the edge concentrated in 2020/2022. Enable via "
        "`config.risk_overlay: vol_target`."
    )
    ov_df = _overlay()
    if ov_df.empty:
        st.info("No price data. Run `eqa ingest`.")
    else:
        st.dataframe(ov_df, width="stretch", hide_index=True)

with signals_tab:
    news = recent_news()
    if news.empty:
        st.info("No scored news. Run `eqa news SYMBOL` or `eqa paper-run --risk-off`.")
    else:
        st.dataframe(news, width="stretch")
