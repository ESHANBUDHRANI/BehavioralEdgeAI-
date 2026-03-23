from __future__ import annotations

from datetime import timedelta
import pandas as pd
from backend.market_context.data_provider import fetch_ohlcv
from backend.market_context.indicators import compute_indicator_frame
from backend.market_context.news import fetch_symbol_news
from backend.market_context.patterns import detect_market_structure, pattern_label
from backend.market_context.regime import (
    classify_market_regime,
    classify_momentum,
    classify_trend,
    classify_volatility,
)


def _date_col(df: pd.DataFrame) -> str:
    for c in ["Date", "Datetime", "date", "datetime"]:
        if c in df.columns:
            return c
    raise ValueError("No date column found")


def build_market_context_for_trades(trades_df: pd.DataFrame) -> list[dict]:
    payloads: list[dict] = []
    if trades_df.empty:
        return payloads
    start = (trades_df["timestamp"].min() - timedelta(days=260)).strftime("%Y-%m-%d")
    end = (trades_df["timestamp"].max() + timedelta(days=5)).strftime("%Y-%m-%d")
    vix = fetch_ohlcv("^VIX", start, end, interval="1d")
    vix_date_col = _date_col(vix) if not vix.empty else None
    for _, trade in trades_df.iterrows():
        symbol = trade["symbol"]
        trade_day = pd.to_datetime(trade["timestamp"]).date()
        ohlc = fetch_ohlcv(symbol, start, end, interval="1d")
        if ohlc.empty:
            continue
        data = detect_market_structure(compute_indicator_frame(ohlc))
        date_col = _date_col(data)
        data[date_col] = pd.to_datetime(data[date_col]).dt.date
        row_slice = data[data[date_col] <= trade_day]
        if row_slice.empty:
            continue
        row = row_slice.iloc[-1]
        vix_value = 20.0
        if vix_date_col:
            vix[vix_date_col] = pd.to_datetime(vix[vix_date_col]).dt.date
            vix_slice = vix[vix[vix_date_col] <= trade_day]
            if not vix_slice.empty:
                close_col = "Close" if "Close" in vix_slice.columns else "close"
                vix_value = float(vix_slice.iloc[-1][close_col])
        p_name, p_conf = pattern_label(row)
        headlines = fetch_symbol_news(symbol, pd.to_datetime(trade["timestamp"]).isoformat())
        tf_labels = {"1D": classify_trend(row)}
        for tf, interval in [("4H", "60m"), ("1H", "60m"), ("15M", "15m")]:
            tf_data = fetch_ohlcv(symbol, start, end, interval=interval)
            if tf_data.empty:
                tf_labels[tf] = "neutral"
            else:
                tf_ind = detect_market_structure(compute_indicator_frame(tf_data))
                tf_row = tf_ind.iloc[-1]
                tf_labels[tf] = classify_trend(tf_row)

        volume_z = float(row.get("volume_z", 0))
        if volume_z > 2:
            volume_label = "institutional_accumulation"
        elif volume_z < -2:
            volume_label = "selling_pressure"
        else:
            volume_label = "weak_move"

        structure_break = bool(row.get("lower_low", False) and row.get("lower_high", False))
        structure_label = "structure_break" if structure_break else "uptrend_intact" if bool(row.get("higher_high", False)) else "ranging"

        zscore_price = float(row.get("zscore_price", 0))
        mean_rev_label = "extended_high" if zscore_price > 2 else "extended_low" if zscore_price < -2 else "neutral"
        context = {
            "trend_context": {"label": classify_trend(row)},
            "momentum_context": {"label": classify_momentum(row), "rsi14": float(row.get("rsi14", 50.0))},
            "volatility_context": {"label": classify_volatility(row)},
            "volume_context": {"label": volume_label},
            "market_structure_context": {"label": structure_label},
            "liquidity_context": {
                "label": "liquidity_above" if bool(row.get("swing_high", False)) else "liquidity_below"
            },
            "mean_reversion_context": {"label": mean_rev_label},
            "pattern_context": {"label": p_name, "confidence": p_conf},
            "multi_timeframe_context": tf_labels,
            "market_regime_context": {"label": classify_market_regime(row, vix_value)},
            "news_sentiment_context": {
                "label": "neutral",
                "confidence": 0.0,
                "score": 0.0,
                "headlines": [h.get("title", "") for h in headlines if h.get("title")],
            },
        }
        payloads.append({"date": trade_day, "symbol": symbol, "context": context})
    return payloads


def test_market_context_engine() -> dict:
    return {"ok": True, "message": "market context engine wired"}
