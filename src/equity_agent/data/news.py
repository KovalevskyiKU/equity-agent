"""Finnhub company-news fetcher (free tier)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import requests

from ..config import get_settings

_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"


@dataclass
class NewsArticle:
    symbol: str
    published_at: datetime
    source: str
    headline: str
    summary: str
    url: str


def fetch_company_news(symbol: str, start: date, end: date) -> list[NewsArticle]:
    """Fetch company news for a symbol over [start, end]. Needs FINNHUB_API_KEY."""
    key = get_settings().finnhub_api_key
    if not key:
        raise RuntimeError("FINNHUB_API_KEY not set in .env")

    resp = requests.get(
        _COMPANY_NEWS_URL,
        params={"symbol": symbol, "from": start.isoformat(), "to": end.isoformat(), "token": key},
        timeout=30,
    )
    resp.raise_for_status()

    articles: list[NewsArticle] = []
    for item in resp.json():
        ts = item.get("datetime")
        url = item.get("url", "")
        if not ts or not url:
            continue
        articles.append(
            NewsArticle(
                symbol=symbol,
                published_at=datetime.fromtimestamp(int(ts), tz=UTC),
                source=item.get("source", ""),
                headline=item.get("headline", ""),
                summary=item.get("summary", ""),
                url=url,
            )
        )
    return articles
