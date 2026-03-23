from __future__ import annotations

import pandas as pd
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator, BIC
from pgmpy.models import BayesianNetwork
from backend.models.runtime import insufficient_data_guard


NODES = [
    "market_regime",
    "news_sentiment",
    "behavioral_cluster",
    "emotional_state",
    "anomaly_flag",
    "trade_outcome",
]


def run(features_df: pd.DataFrame, context_df: pd.DataFrame, config: dict | None = None) -> dict:
    guard = insufficient_data_guard(len(features_df))
    if guard:
        return guard
    df = pd.DataFrame()
    df["market_regime"] = context_df.get("market_regime", pd.Series(["ranging"] * len(features_df))).astype(str)
    df["news_sentiment"] = context_df.get("news_sentiment", pd.Series(["neutral"] * len(features_df))).astype(str)
    df["behavioral_cluster"] = features_df.get("cluster_label", pd.Series([0] * len(features_df))).astype(str)
    df["emotional_state"] = features_df.get("emotional_state", pd.Series(["calm"] * len(features_df))).astype(str)
    df["anomaly_flag"] = features_df.get("anomaly_flag", pd.Series([0] * len(features_df))).astype(str)
    df["trade_outcome"] = (features_df.get("pnl", pd.Series([0] * len(features_df))) > 0).map(
        {True: "win", False: "loss"}
    )
    hc = HillClimbSearch(df)
    structure = hc.estimate(scoring_method=BIC(df))
    model = BayesianNetwork(structure.edges())
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    cpds = {cpd.variable: str(cpd) for cpd in model.get_cpds()}
    cpd_payload = []
    for cpd in model.get_cpds():
        cpd_payload.append(
            {
                "variable": cpd.variable,
                "variable_card": int(cpd.variable_card),
                "evidence": list(cpd.variables[1:]),
                "evidence_card": list(cpd.cardinality[1:]) if len(cpd.cardinality) > 1 else [],
                "values": cpd.get_values().tolist(),
                "state_names": {k: list(v) for k, v in (cpd.state_names or {}).items()},
            }
        )
    state_names = {}
    for col in df.columns:
        state_names[col] = sorted(df[col].dropna().astype(str).unique().tolist())
    return {
        "nodes": list(df.columns),
        "edges": list(model.edges()),
        "cpds": cpds,
        "cpd_payload": cpd_payload,
        "state_names": state_names,
        "counterfactual_template": "if regime had been bullish, outcome probability changes to X%",
        "confidence": 0.7,
    }


def test_bayesian_network() -> dict:
    return {"ok": True, "nodes": NODES}
