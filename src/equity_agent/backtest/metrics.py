"""Performance metrics — engine-agnostic.

Return-based metrics take a pandas Series of periodic (e.g. daily) strategy
returns. Trade-based metrics take a Series/array of per-trade PnL. Nothing here
assumes a particular backtest engine, so these are shared by the Phase 3
simulator, the live tracker, and any research notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return float("nan")
    return float((1.0 + returns).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    n = len(returns)
    if n == 0:
        return float("nan")
    growth = float((1.0 + returns).prod())
    if growth <= 0.0:
        return float("nan")
    years = n / periods_per_year
    return growth ** (1.0 / years) - 1.0 if years > 0 else float("nan")


def annual_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("nan")
    dd = downside.std(ddof=1)
    if dd == 0 or np.isnan(dd):
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Most negative peak-to-trough of the cumulative equity curve (e.g. -0.25)."""
    if len(returns) == 0:
        return float("nan")
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns)
    growth = cagr(returns, periods_per_year)
    if mdd == 0 or np.isnan(mdd) or np.isnan(growth):
        return float("nan")
    return float(growth / abs(mdd))


def win_rate(trade_pnl: pd.Series) -> float:
    if len(trade_pnl) == 0:
        return float("nan")
    return float((trade_pnl > 0).mean())


def profit_factor(trade_pnl: pd.Series) -> float:
    gains = float(trade_pnl[trade_pnl > 0].sum())
    losses = float(-trade_pnl[trade_pnl < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return gains / losses


def expectancy(trade_pnl: pd.Series) -> float:
    """Average PnL per trade."""
    if len(trade_pnl) == 0:
        return float("nan")
    return float(trade_pnl.mean())


def return_summary(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> dict[str, float]:
    return {
        "n": float(len(returns)),
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods_per_year),
        "ann_vol": annual_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free, periods_per_year),
        "sortino": sortino_ratio(returns, risk_free, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
    }
