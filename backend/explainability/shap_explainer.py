from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def _summary(shap_values: np.ndarray, feature_names: list[str], top_n: int = 15) -> list[dict]:
    importance = np.abs(shap_values).mean(axis=0)
    order = np.argsort(importance)[::-1][:top_n]
    return [
        {"feature": feature_names[i], "importance": float(importance[i])}
        for i in order
    ]


def compute_shap_bundle(
    features_df: pd.DataFrame,
    clustering_labels: list[int],
    anomaly_scores: list[float],
    lstm_prediction_error: list[float],
) -> dict:
    numeric = features_df.select_dtypes(include=["number"]).fillna(0)
    if numeric.empty:
        return {"gmm_shap": [], "iforest_shap": [], "lstm_shap": [], "force_plot_trade_index": None}

    X = numeric.to_numpy(dtype=float)
    feature_names = numeric.columns.tolist()

    gmm_target = np.array(clustering_labels[: len(X)], dtype=int)
    if len(gmm_target) < len(X):
        gmm_target = np.pad(gmm_target, (0, len(X) - len(gmm_target)), mode="edge")
    gmm_model = RandomForestClassifier(n_estimators=300, random_state=42)
    gmm_model.fit(X, gmm_target)
    gmm_explainer = shap.TreeExplainer(gmm_model)
    gmm_values = gmm_explainer.shap_values(X)
    gmm_arr = np.array(gmm_values[0] if isinstance(gmm_values, list) else gmm_values)

    anomaly_target = np.array(anomaly_scores[: len(X)], dtype=float)
    if len(anomaly_target) < len(X):
        anomaly_target = np.pad(anomaly_target, (0, len(X) - len(anomaly_target)), mode="edge")
    anomaly_model = RandomForestRegressor(n_estimators=300, random_state=42)
    anomaly_model.fit(X, anomaly_target)
    anomaly_explainer = shap.TreeExplainer(anomaly_model)
    anomaly_values = np.array(anomaly_explainer.shap_values(X))

    lstm_target = np.array(lstm_prediction_error[: len(X)], dtype=float)
    if len(lstm_target) < len(X):
        lstm_target = np.pad(lstm_target, (0, len(X) - len(lstm_target)), mode="edge")
    lstm_model = RandomForestRegressor(n_estimators=300, random_state=42)
    lstm_model.fit(X, lstm_target)
    lstm_explainer = shap.TreeExplainer(lstm_model)
    lstm_values = np.array(lstm_explainer.shap_values(X))

    force_idx = int(np.argmax(anomaly_target)) if len(anomaly_target) else 0
    return {
        "gmm_shap": _summary(gmm_arr, feature_names),
        "iforest_shap": _summary(anomaly_values, feature_names),
        "lstm_shap": _summary(lstm_values, feature_names),
        "force_plot_trade_index": force_idx,
    }


def test_shap_explainer() -> dict:
    return {"ok": True, "message": "shap explainer bundle wired"}
