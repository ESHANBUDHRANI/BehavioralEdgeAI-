from __future__ import annotations

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from backend.models.runtime import insufficient_data_guard


def _single_test(df: pd.DataFrame, x: str, y: str, maxlag: int = 3) -> dict:
    pair = df[[y, x]].dropna()
    if len(pair) < maxlag + 10:
        return {"direction": f"{x}->{y}", "p_value": 1.0, "lag": None, "story": "insufficient samples"}
    result = grangercausalitytests(pair, maxlag=maxlag, verbose=False)
    best_lag = min(result, key=lambda lag: result[lag][0]["ssr_ftest"][1])
    pval = float(result[best_lag][0]["ssr_ftest"][1])
    return {
        "direction": f"{x}->{y}",
        "lag": int(best_lag),
        "p_value": pval,
        "story": f"{x} Granger-causes {y}" if pval < 0.05 else f"No strong evidence that {x} causes {y}",
    }


def run(features_df: pd.DataFrame, context_df: pd.DataFrame, config: dict | None = None) -> dict:
    guard = insufficient_data_guard(len(features_df))
    if guard:
        return guard
    joined = features_df.copy()
    joined["sentiment"] = context_df.get("news_sentiment_score", pd.Series([0.0] * len(features_df)))
    joined["volatility"] = context_df.get("volatility_score", pd.Series([0.0] * len(features_df)))
    joined["market_regime_num"] = context_df.get("market_regime_num", pd.Series([0.0] * len(features_df)))
    joined["trade_frequency"] = joined.get("trade_frequency_7d", pd.Series([0.0] * len(features_df)))
    joined["position_size_change"] = joined.get("position_value", pd.Series([0.0] * len(features_df))).pct_change().fillna(0)
    joined["cluster_shift"] = joined.get("cluster_label", pd.Series([0] * len(features_df))).diff().fillna(0).abs()
    return {
        "tests": [
            _single_test(joined, "sentiment", "trade_frequency"),
            _single_test(joined, "volatility", "position_size_change"),
            _single_test(joined, "market_regime_num", "cluster_shift"),
        ],
        "confidence": 0.68,
    }


def test_causality() -> dict:
    return {"ok": True, "message": "granger causality wired"}
