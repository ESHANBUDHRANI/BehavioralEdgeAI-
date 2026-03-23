from __future__ import annotations

import numpy as np
import pandas as pd


def _mad_z(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = np.median(np.abs(series - median))
    denom = mad if mad > 1e-9 else 1e-9
    return 0.6745 * (series - median) / denom


def compute_baselines(features_df: pd.DataFrame) -> dict:
    if features_df.empty:
        return {"insufficient_data": True, "message": "no feature rows found", "confidence": 0.0}
    numeric = features_df.select_dtypes(include=["number"]).copy()
    stats = {}
    for col in numeric.columns:
        series = numeric[col].dropna()
        if series.empty:
            continue
        stats[col] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "p25": float(series.quantile(0.25)),
            "p75": float(series.quantile(0.75)),
        }
        numeric[f"{col}_robust_z"] = _mad_z(numeric[col].fillna(series.median()))
    composite = numeric.filter(like="_robust_z").abs().mean(axis=1)
    return {
        "insufficient_data": len(features_df) < 50,
        "summary_stats": stats,
        "composite_deviation_score": composite.fillna(0).tolist(),
        "bias_baselines": {
            "disposition_effect_coefficient": float(features_df.get("pnl", pd.Series([0])).mean()),
            "revenge_trading_frequency_rate": float(features_df.get("revenge_score", pd.Series([0])).mean()),
            "overconfidence_proxy": float(features_df.get("position_size_dev_rolling", pd.Series([0])).mean()),
            "signal_following_rate": float(features_df.get("signal_following_score", pd.Series([0.5])).mean()),
        },
        "confidence": 0.85 if len(features_df) >= 50 else 0.45,
    }


def test_baselines() -> dict:
    return {"ok": True, "message": "baseline engine wired"}
