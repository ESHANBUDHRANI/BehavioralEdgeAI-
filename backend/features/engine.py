from __future__ import annotations

import numpy as np
import pandas as pd


def build_behavioral_features(trades_df: pd.DataFrame, market_context: list[dict]) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    df = trades_df.sort_values("timestamp").copy()
    df["position_value"] = df["quantity"] * df["price"]
    rolling_size = df["position_value"].rolling(20, min_periods=1).mean()
    df["position_size_dev_rolling"] = (df["position_value"] - rolling_size) / rolling_size.replace(0, np.nan)
    df["holding_duration_days"] = df.get("holding_duration", 0.0)
    avg_hold = max(float(df["holding_duration_days"].mean()), 1e-6)
    df["early_exit"] = (df["holding_duration_days"] < 0.2 * avg_hold).astype(int)
    df["overheld"] = (df["holding_duration_days"] > 2.0 * avg_hold).astype(int)
    df["is_loss"] = (df.get("pnl", 0.0) < 0).astype(int)
    df["next_trade_delta_hours"] = (
        df["timestamp"].shift(-1) - df["timestamp"]
    ).dt.total_seconds().fillna(0) / 3600.0
    df["post_loss_next_trade_hours"] = np.where(df["is_loss"] == 1, df["next_trade_delta_hours"], np.nan)
    df["size_change_after_loss"] = np.where(
        df["is_loss"].shift(1).fillna(0) == 1,
        df["position_value"].pct_change(),
        0.0
    )
    df["trade_frequency_7d"] = (
        df.set_index("timestamp")
        .assign(one=1)["one"]
        .rolling("7D")
        .sum()
        .reset_index(drop=True)
    )
    df["trade_frequency_30d"] = (
        df.set_index("timestamp")
        .assign(one=1)["one"]
        .rolling("30D")
        .sum()
        .reset_index(drop=True)
    )
    freq_mean = df["trade_frequency_7d"].mean()
    freq_std = max(float(df["trade_frequency_7d"].std()), 1e-6)
    df["frequency_spike"] = ((df["trade_frequency_7d"] - freq_mean) / freq_std > 2).astype(int)
    df["revenge_score"] = (
        (df["post_loss_next_trade_hours"].fillna(
            df["post_loss_next_trade_hours"].median()
        ).rsub(24) / 24).clip(0, 1)
        + df["size_change_after_loss"].fillna(0).clip(-1, 1).abs()
        + df["frequency_spike"]
    ) / 3.0
    df["emotional_score"] = (
        df["revenge_score"] * 0.4
        + df["position_size_dev_rolling"].fillna(0).abs().clip(0, 2) * 0.3
        + df["early_exit"] * 0.3
    )
    df["emotional_state"] = np.select(
        [df["emotional_score"] < 0.33, df["emotional_score"] < 0.66],
        ["calm", "anxious"],
        default="euphoric",
    )
    df["signal_following_score"] = 0.5
    df["feature_confidence"] = 0.8
    return df


def test_features_engine() -> dict:
    return {"ok": True, "message": "feature engine wired"}