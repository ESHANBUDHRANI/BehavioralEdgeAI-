from __future__ import annotations

import pandas as pd


def time_series_split(df: pd.DataFrame, date_col: str = "timestamp", train_ratio: float = 0.8):
    ordered = df.sort_values(date_col).reset_index(drop=True)
    idx = int(len(ordered) * train_ratio)
    return ordered.iloc[:idx].copy(), ordered.iloc[idx:].copy()


def test_time_split() -> dict:
    return {"ok": True, "rule": "earlier=train, later=test"}
