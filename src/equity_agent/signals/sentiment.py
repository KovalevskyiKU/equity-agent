"""LLM news sentiment (Gemini) → stored, then aggregated into a daily signal.

Each headline is scored once by Gemini into {sentiment, impact, rationale},
cached in the ``news_items`` table (keyed on url), and never re-scored. The
daily feature is an impact-weighted, causally-smoothed sentiment series, gated
by ``published_at`` so a backtest never sees a headline before it was published.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from pydantic import BaseModel
from sqlalchemy import select

from ..data.news import fetch_company_news
from ..llm import DEFAULT_MODEL, generate_structured
from ..storage.db import session_scope
from ..storage.models import NewsItem

logger = logging.getLogger("equity_agent")

_PROMPT = (
    "You are a financial news analyst. Score this news for the stock {symbol}.\n"
    "Return sentiment in [-1, 1] (very bearish .. very bullish for the share price) "
    "and impact in [0, 1] (how market-moving this specific item is for {symbol}).\n\n"
    "HEADLINE: {headline}\nSUMMARY: {summary}"
)


class SentimentOut(BaseModel):
    sentiment: float
    impact: float
    rationale: str


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def score_article(
    symbol: str, headline: str, summary: str, model: str = DEFAULT_MODEL
) -> SentimentOut:
    """One LLM call → validated sentiment score."""
    prompt = _PROMPT.format(symbol=symbol, headline=headline, summary=summary or "(none)")
    return generate_structured(prompt, SentimentOut, model=model)


def fetch_score_store(
    symbol: str,
    start: date,
    end: date,
    model: str = DEFAULT_MODEL,
    limit: int | None = None,
) -> dict[str, int]:
    """Fetch news, score new (uncached) articles with Gemini, store them. Returns counts."""
    articles = fetch_company_news(symbol, start, end)
    with session_scope() as session:
        known = set(session.scalars(select(NewsItem.url)).all())
    todo = [a for a in articles if a.url not in known]
    if limit is not None:
        todo = todo[:limit]

    scored = 0
    for art in todo:
        try:
            s = score_article(symbol, art.headline, art.summary, model)
        except Exception as e:  # noqa: BLE001 - one bad article shouldn't stop the batch
            logger.warning("[%s] scoring failed for %s: %s", symbol, art.url[:60], e)
            continue
        with session_scope() as session:
            session.add(
                NewsItem(
                    symbol=symbol,
                    published_at=art.published_at,
                    source=art.source,
                    title=art.headline,
                    summary=art.summary,
                    url=art.url,
                    sentiment=_clamp(s.sentiment, -1.0, 1.0),
                    impact=_clamp(s.impact, 0.0, 1.0),
                    llm_model=model,
                    llm_rationale=s.rationale,
                )
            )
        scored += 1

    return {"fetched": len(articles), "scored": scored, "cached": len(articles) - len(todo)}


def get_daily_sentiment(symbol: str, halflife_days: float = 3.0) -> pd.Series:
    """Impact-weighted daily sentiment, causally EWM-smoothed. Empty if no news."""
    with session_scope() as session:
        rows = session.execute(
            select(NewsItem.published_at, NewsItem.sentiment, NewsItem.impact).where(
                NewsItem.symbol == symbol, NewsItem.sentiment.isnot(None)
            )
        ).all()
    if not rows:
        return pd.Series(dtype=float, name="sentiment")

    df = pd.DataFrame(rows, columns=["published_at", "sentiment", "impact"])
    df["day"] = pd.to_datetime(df["published_at"]).dt.normalize()

    def _wavg(g: pd.DataFrame) -> float:
        w = g["impact"].clip(lower=0.0)
        total = float(w.sum())
        if total <= 0:
            return float(g["sentiment"].mean())
        return float((g["sentiment"] * w).sum() / total)

    daily = df.groupby("day", group_keys=True).apply(_wavg).sort_index()
    smoothed = daily.ewm(halflife=halflife_days).mean()
    smoothed.name = "sentiment"
    return smoothed
