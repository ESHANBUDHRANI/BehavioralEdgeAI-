from __future__ import annotations

from datetime import datetime, timedelta
from newsapi import NewsApiClient
from backend.config import get_settings


def fetch_symbol_news(symbol: str, date_iso: str, limit: int = 10) -> list[dict]:
    settings = get_settings()
    if not settings.newsapi_key:
        return []
    client = NewsApiClient(api_key=settings.newsapi_key)
    dt = datetime.fromisoformat(date_iso)
    frm = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
    to = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    result = client.get_everything(
        q=symbol,
        from_param=frm,
        to=to,
        language="en",
        sort_by="publishedAt",
        page_size=limit,
    )
    return result.get("articles", [])


def test_news() -> dict:
    return {"ok": True, "provider": "newsapi"}
