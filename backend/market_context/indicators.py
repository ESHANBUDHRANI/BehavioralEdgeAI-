from __future__ import annotations

import numpy as np
import pandas as pd
import ta


def _price_col(df: pd.DataFrame) -> str:
    for c in ["Close", "close", "Adj Close", "adj_close"]:
        if c in df.columns:
            return c
    raise ValueError("No close price column found")


def compute_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # FIX: yfinance newer versions return MultiIndex columns like ('Close', 'RELIANCE.NS')
    # Flatten them to simple column names
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    close = _price_col(df)
    high = "High" if "High" in df.columns else "high"
    low = "Low" if "Low" in df.columns else "low"
    volume = "Volume" if "Volume" in df.columns else "volume"
    out = df.copy()

    # Moving averages
    out["sma20"] = ta.trend.sma_indicator(out[close], window=20)
    out["sma50"] = ta.trend.sma_indicator(out[close], window=50)
    out["sma200"] = ta.trend.sma_indicator(out[close], window=200)
    out["ema9"] = ta.trend.ema_indicator(out[close], window=9)
    out["ema21"] = ta.trend.ema_indicator(out[close], window=21)
    out["ema50"] = ta.trend.ema_indicator(out[close], window=50)

    # MACD
    macd = ta.trend.MACD(out[close], window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()

    # ADX
    adx = ta.trend.ADXIndicator(out[high], out[low], out[close], window=14)
    out["adx"] = adx.adx()

    # RSI
    out["rsi14"] = ta.momentum.rsi(out[close], window=14)

    # Stochastic RSI
    stoch = ta.momentum.StochRSIIndicator(out[close], window=14, smooth1=3, smooth2=3)
    out["stochrsi_k"] = stoch.stochrsi_k()

    # ROC
    out["roc"] = ta.momentum.roc(out[close], window=14)

    # ATR
    out["atr14"] = ta.volatility.average_true_range(out[high], out[low], out[close], window=14)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(out[close], window=20, window_dev=2)
    out["bb_lower"] = bb.bollinger_lband()
    out["bb_mid"] = bb.bollinger_mavg()
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_bandwidth"] = bb.bollinger_wband()

    # Volatility and volume
    out["returns_std_20"] = out[close].pct_change().rolling(20).std()
    out["volume_ma20"] = out[volume].rolling(20).mean()
    out["volume_z"] = (
        (out[volume] - out[volume].rolling(20).mean())
        / out[volume].rolling(20).std()
    )

    # OBV
    out["obv"] = ta.volume.on_balance_volume(out[close], out[volume])

    # VWAP
    out["vwap"] = ta.volume.volume_weighted_average_price(
        out[high], out[low], out[close], out[volume]
    )

    # ADL (Accumulation/Distribution)
    out["adl"] = ta.volume.acc_dist_index(out[high], out[low], out[close], out[volume])

    # Z-score and distance from MAs
    out["zscore_price"] = (
        (out[close] - out[close].rolling(20).mean())
        / out[close].rolling(20).std()
    )
    out["distance_ema20"] = ((out[close] - out["ema21"]) / out["ema21"]) * 100.0
    out["distance_ema50"] = ((out[close] - out["ema50"]) / out["ema50"]) * 100.0

    return out.replace([np.inf, -np.inf], np.nan)


def test_indicators() -> dict:
    return {"ok": True, "message": "indicator computations wired"}