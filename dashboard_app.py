"""Streamlit dashboard. Run: `streamlit run dashboard_app.py` (or `eqa dashboard`)."""

import streamlit as st

from equity_agent.dashboard.data import paper_overview, recent_news, strategy_curves
from equity_agent.storage.db import init_db

st.set_page_config(page_title="equity-agent", layout="wide")
init_db()
st.title("equity-agent")

portfolio_tab, backtest_tab, signals_tab = st.tabs(
    ["Paper portfolio", "Strategy backtest", "Signals & news"]
)

with portfolio_tab:
    ov = paper_overview()
    if not ov["has_account"]:
        st.info("No paper account yet. Run `eqa paper-reset` then `eqa paper-run`.")
    else:
        curve = ov["equity_curve"]
        cash = float(ov["cash"])
        start = float(ov["starting_cash"]) or 1.0
        equity = float(curve["equity"].iloc[-1]) if len(curve) else cash
        c1, c2, c3 = st.columns(3)
        c1.metric("Equity", f"${equity:,.0f}")
        c2.metric("Cash", f"${cash:,.0f}")
        c3.metric("Return", f"{(equity / start - 1) * 100:+.1f}%")
        if len(curve):
            st.line_chart(curve.set_index("ts")["equity"])
        st.subheader("Positions")
        st.dataframe(ov["positions"], use_container_width=True)
        st.subheader("Recent trades")
        st.dataframe(ov["trades"], use_container_width=True)

with backtest_tab:
    curves, metrics = strategy_curves()
    if curves.empty:
        st.info("No price data. Run `eqa ingest`.")
    else:
        st.line_chart(curves)
        st.dataframe(metrics, use_container_width=True)

with signals_tab:
    news = recent_news()
    if news.empty:
        st.info("No scored news. Run `eqa news SYMBOL` or `eqa paper-run --risk-off`.")
    else:
        st.dataframe(news, use_container_width=True)
