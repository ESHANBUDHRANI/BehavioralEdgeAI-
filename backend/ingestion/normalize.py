from __future__ import annotations

from datetime import datetime
import pandas as pd


COLUMN_MAP = {
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "timestamp": "timestamp",
    "symbol": "symbol",
    "ticker": "symbol",
    "instrument": "symbol",
    "side": "buy_sell",
    "type": "buy_sell",
    "buy/sell": "buy_sell",
    "buy_sell": "buy_sell",
    "qty": "quantity",
    "shares": "quantity",
    "quantity": "quantity",
    "price": "price",
    "avg_price": "price",
    "fill_price": "price",
    "value": "value",
}

REQUIRED = {"timestamp", "symbol", "buy_sell", "quantity", "price"}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_")
        renamed[c] = COLUMN_MAP.get(key, key)
    out = df.rename(columns=renamed).copy()
    missing = REQUIRED - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(list(missing))}")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out["buy_sell"] = out["buy_sell"].astype(str).str.upper().str.strip()
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["timestamp", "symbol", "buy_sell", "quantity", "price"])
    out["trade_date"] = out["timestamp"].dt.date

    # FIX: original code used groupby aggregation which merged separate trades
    # at the same price on the same day, corrupting PnL and holding duration.
    # Now we only drop exact duplicate rows (all 5 key columns identical),
    # preserving legitimately separate trades that share a price.
    out = out.drop_duplicates(
        subset=["timestamp", "symbol", "buy_sell", "quantity", "price"]
    )
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def trade_preview(df: pd.DataFrame, limit: int = 30) -> list[dict]:
    payload = df.head(limit).copy()
    payload["timestamp"] = payload["timestamp"].apply(
        lambda x: x.isoformat() if isinstance(x, datetime) else str(x)
    )
    return payload.to_dict(orient="records")


def test_normalize() -> dict:
    return {"ok": True, "required_columns": sorted(REQUIRED)}
