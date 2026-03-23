from __future__ import annotations

import pandas as pd


def classify_trend(row: pd.Series) -> str:
    if row.get("sma50", 0) > row.get("sma200", 0) and row.get("adx", 0) > 20:
        return "bullish"
    if row.get("sma50", 0) > row.get("sma200", 0):
        return "weak_bullish"
    if row.get("sma50", 0) < row.get("sma200", 0) and row.get("adx", 0) > 20:
        return "bearish"
    return "weak_bearish"


def classify_momentum(row: pd.Series) -> str:
    rsi = row.get("rsi14", 50)
    if rsi >= 70:
        return "overbought"
    if rsi <= 30:
        return "oversold"
    hist = row.get("macd_hist", 0)
    return "increasing" if hist > 0 else "decreasing"


def classify_volatility(row: pd.Series) -> str:
    bw = row.get("bb_bandwidth", 0)
    if bw is None:
        return "normal"
    if bw < 5:
        return "low_compression"
    if bw > 20:
        return "high_expansion"
    return "normal"


def classify_market_regime(row: pd.Series, vix_value: float) -> str:
    adx = row.get("adx", 0)
    atr = row.get("atr14", 0)
    if vix_value > 25 and atr > 0:
        return "high_volatility"
    if vix_value < 15 and atr > 0:
        return "low_volatility"
    if adx > 25:
        return "trending"
    return "ranging"


def test_regime() -> dict:
    return {"ok": True, "labels": ["bullish", "weak_bullish", "bearish", "weak_bearish"]}
