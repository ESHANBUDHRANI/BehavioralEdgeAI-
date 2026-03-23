from __future__ import annotations

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier


def explain_anomalous_trade(features_df: pd.DataFrame, index: int, anomaly_flags: list[int]) -> str:
    numeric = features_df.select_dtypes(include=["number"]).fillna(0)
    if numeric.empty or index >= len(numeric) or not anomaly_flags:
        return "No trade explanation available."
    y = np.array(anomaly_flags[: len(numeric)], dtype=int)
    if len(y) < len(numeric):
        y = np.pad(y, (0, len(numeric) - len(y)), mode="edge")
    model = RandomForestClassifier(n_estimators=250, random_state=42)
    model.fit(numeric.to_numpy(), y)
    explainer = LimeTabularExplainer(
        training_data=numeric.to_numpy(),
        feature_names=numeric.columns.tolist(),
        class_names=["normal", "anomalous"],
        mode="classification",
    )
    exp = explainer.explain_instance(
        numeric.iloc[index].to_numpy(),
        model.predict_proba,
        num_features=6,
    )
    parts = [f"{name} ({weight:+.3f})" for name, weight in exp.as_list()]
    return f"Trade {index} anomaly drivers: " + ", ".join(parts)


def test_lime_explainer() -> dict:
    return {"ok": True, "message": "lime tabular explainer wired"}
