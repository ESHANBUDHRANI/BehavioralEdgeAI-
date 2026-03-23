from __future__ import annotations

import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from backend.models.runtime import insufficient_data_guard


def _auto_label(center: dict[str, float]) -> str:
    high_size = center.get("position_value", 0) > 0.5
    high_freq = center.get("trade_frequency_7d", 0) > 0.5
    high_revenge = center.get("revenge_score", 0) > 0.5
    if high_size and high_freq:
        return "aggressive_high_activity"
    if high_revenge:
        return "reactive_post_loss"
    if not high_size and not high_freq:
        return "conservative_selective"
    return "balanced_tactical"


def run(features_df: pd.DataFrame, context_df: pd.DataFrame | None = None, config: dict | None = None) -> dict:
    guard = insufficient_data_guard(len(features_df))
    if guard:
        return guard
    numeric = features_df.select_dtypes(include=["number"]).copy()
    top_features = numeric.var().sort_values(ascending=False).head(15).index.tolist()
    X_raw = numeric[top_features].fillna(0).to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42)
    gmm_labels = gmm.fit_predict(X)
    prob_matrix = gmm.predict_proba(X)
    hierarchical = AgglomerativeClustering(n_clusters=4, linkage="ward")
    h_labels = hierarchical.fit_predict(X)
    Z = linkage(X, method="ward")
    centers = []
    labels = []
    center_dicts = []
    for i in range(4):
        cluster_rows = X[gmm_labels == i]
        center = cluster_rows.mean(axis=0).tolist() if len(cluster_rows) else [0.0] * len(top_features)
        centers.append(center)
        feature_center = {feature: center[idx] for idx, feature in enumerate(top_features)}
        center_dicts.append(feature_center)
        labels.append(_auto_label(feature_center))
    try:
        sil = float(silhouette_score(X, gmm_labels))
    except Exception:  # noqa: BLE001
        sil = 0.0
    return {
        "cluster_probabilities": prob_matrix.tolist(),
        "gmm_labels": gmm_labels.tolist(),
        "hierarchical_labels": h_labels.tolist(),
        "dendrogram_data": Z[:, :4].tolist(),
        "cluster_centers": centers,
        "cluster_centers_by_feature": center_dicts,
        "cluster_plain_labels": labels,
        "selected_features": top_features,
        "silhouette_score": sil,
        "confidence": max(0.0, min(1.0, 0.55 + max(0.0, sil))),
    }


def test_clustering() -> dict:
    return {"ok": True, "message": "clustering model wired"}
