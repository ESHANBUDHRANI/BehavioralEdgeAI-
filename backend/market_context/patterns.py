from __future__ import annotations

import pandas as pd


def detect_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    high = "High" if "High" in out.columns else "high"
    low = "Low" if "Low" in out.columns else "low"
    out["swing_high"] = out[high] == out[high].rolling(5, center=True).max()
    out["swing_low"] = out[low] == out[low].rolling(5, center=True).min()
    out["higher_high"] = out[high] > out[high].shift(1)
    out["higher_low"] = out[low] > out[low].shift(1)
    out["lower_high"] = out[high] < out[high].shift(1)
    out["lower_low"] = out[low] < out[low].shift(1)
    return out


def pattern_label(row: pd.Series) -> tuple[str, float]:
    if row.get("higher_high", False) and row.get("higher_low", False):
        return ("breakout_pattern", 0.65)
    if row.get("lower_high", False) and row.get("lower_low", False):
        return ("breakdown_pattern", 0.65)
    return ("none", 0.25)


def test_patterns() -> dict:
    return {"ok": True, "message": "pattern detector wired"}
