"""Binance spot execution adapter — same rebalance intent as IBKR/paper, for crypto.

Thin ``ccxt`` glue (optional ``[crypto-exec]`` dependency, lazy import) over the
shared, tested :func:`plan_orders` (fractional — crypto trades in fractions, not
whole shares). Symbol convention maps our ``BTC-USD`` to Binance ``BTC/USDT``.

SAFETY: transmits nothing unless ``execute=True``; the CLI defaults to a dry run.
Live trading is a deliberate, user-initiated action against the user's own keys.
"""

from __future__ import annotations

import logging
from typing import Any

from .orders import PlannedOrder, plan_orders

logger = logging.getLogger("equity_agent")


def to_ccxt_symbol(symbol: str) -> str:
    """Our ``BTC-USD`` -> Binance ``BTC/USDT``."""
    return symbol.replace("-USD", "/USDT")


def base_asset(symbol: str) -> str:
    """``BTC-USD`` -> ``BTC`` (the balance key on the exchange)."""
    return symbol.split("-")[0]


class BinanceBroker:
    """Minimal Binance spot adapter. Order math is in plan_orders; this is glue only."""

    def __init__(
        self, api_key: str | None = None, secret: str | None = None, exchange: Any = None
    ) -> None:
        self._key, self._secret = api_key, secret
        self._ex: Any = exchange  # injectable for tests

    def connect(self) -> None:
        if self._ex is None:
            import ccxt  # lazy: optional dependency

            self._ex = ccxt.binance(
                {"apiKey": self._key, "secret": self._secret, "enableRateLimit": True}
            )

    def positions(self, symbols: list[str]) -> dict[str, float]:
        """Held base-asset quantities, keyed in our ``BTC-USD`` convention."""
        bal = self._ex.fetch_balance().get("total", {})
        return {s: float(bal.get(base_asset(s), 0.0)) for s in symbols}

    def prices(self, symbols: list[str]) -> dict[str, float]:
        """Last price per symbol (our convention) from the exchange tickers."""
        out: dict[str, float] = {}
        for s in symbols:
            t = self._ex.fetch_ticker(to_ccxt_symbol(s))
            if t.get("last"):
                out[s] = float(t["last"])
        return out

    def equity_usdt(self, symbols: list[str], prices: dict[str, float]) -> float:
        """Total account value in USDT: free USDT + marked value of held bases."""
        bal = self._ex.fetch_balance().get("total", {})
        equity = float(bal.get("USDT", 0.0))
        for s in symbols:
            equity += float(bal.get(base_asset(s), 0.0)) * prices.get(s, 0.0)
        return equity

    def rebalance(
        self,
        target_weights: dict[str, float],
        prices: dict[str, float],
        *,
        execute: bool = False,
        min_notional: float = 10.0,
    ) -> list[PlannedOrder]:
        """Plan spot orders to reach ``target_weights``; transmit only if ``execute``."""
        symbols = list(target_weights)
        equity = self.equity_usdt(symbols, prices)
        orders = plan_orders(
            target_weights, prices, equity, self.positions(symbols),
            min_notional=min_notional, whole_shares=False,
        )
        if not execute:
            logger.info("Binance dry-run: %d orders planned (not transmitted)", len(orders))
            return orders
        for o in orders:
            self._ex.create_order(to_ccxt_symbol(o.symbol), "market", o.side.lower(), o.qty)
            logger.info("Binance order transmitted: %s %s %g", o.side, o.symbol, o.qty)
        return orders
