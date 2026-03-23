from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import yfinance as yf
from backend.config import get_settings


def _cache_file(symbol: str, interval: str) -> Path:
    settings = get_settings()
    clean = symbol.replace("^", "IDX_").replace("/", "_")
    return settings.cache_dir / f"yf_{clean}_{interval}.csv"


def fetch_ohlcv(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    cache_file = _cache_file(symbol, interval)
    settings = get_settings()
    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.utcnow() - mtime < timedelta(hours=settings.cache_ttl_hours):
            cached = pd.read_csv(cache_file)
            for col in ["Datetime", "Date", "datetime", "date"]:
                if col in cached.columns:
                    cached[col] = pd.to_datetime(cached[col], errors="coerce")
            # FIX: yfinance writes ticker name as first row in CSV cache
            # Drop any rows where the Close column contains a string (ticker name)
            for close_col in ["Close", "Adj Close", "close"]:
                if close_col in cached.columns:
                    cached = cached[pd.to_numeric(cached[close_col], errors="coerce").notna()]
                    break
            cached = cached.reset_index(drop=True)
            return cached

    data = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if data.empty:
        return pd.DataFrame()

    # FIX: flatten MultiIndex columns from newer yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    data = data.reset_index()
    data.to_csv(cache_file, index=False)
    return data


def test_data_provider() -> dict:
    return {"ok": True, "cache_ttl_hours": get_settings().cache_ttl_hours}
