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


def capm_alpha_beta(
    returns: pd.Series,
    market: pd.Series,
    periods_per_year: int = TRADING_DAYS,
    risk_free: float = 0.0,
) -> dict[str, float]:
    """OLS of excess strategy returns on excess market returns — the standard edge test.

    ``r_s - rf = alpha + beta * (r_m - rf) + e``

    Comparing raw Sharpe (what we did before) is not a test of edge: in a strong bull
    decade any strategy with beta < 1 looks bad even with a positive alpha. This
    separates the two. Returns annualized alpha, beta, the **t-stat of alpha** (the
    thing that says whether the edge is real), the information ratio (alpha over
    residual risk), R^2 and n.
    """
    pair = pd.concat([returns, market], axis=1).dropna()
    n = len(pair)
    if n < 30:
        return {k: float("nan") for k in
                ("ann_alpha", "beta", "alpha_t", "info_ratio", "r2", "n")} | {"n": float(n)}

    rf = risk_free / periods_per_year
    y = pair.iloc[:, 0].to_numpy() - rf
    x = pair.iloc[:, 1].to_numpy() - rf
    X = np.column_stack([np.ones(n), x])

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_d, beta = float(coef[0]), float(coef[1])
    resid = y - X @ coef
    dof = n - 2
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_alpha = float(np.sqrt(sigma2 * xtx_inv[0, 0]))

    resid_vol = float(np.std(resid, ddof=2)) * np.sqrt(periods_per_year)
    ann_alpha = alpha_d * periods_per_year
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "ann_alpha": ann_alpha,
        "beta": beta,
        "alpha_t": alpha_d / se_alpha if se_alpha > 0 else float("nan"),
        "info_ratio": ann_alpha / resid_vol if resid_vol > 0 else float("nan"),
        "r2": 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan"),
        "n": float(n),
    }
