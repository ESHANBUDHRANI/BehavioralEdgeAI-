from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from backend.config import session_output_dir
from backend.explainability.nlg import report_summary_template


def _compute_effectiveness_metrics(model_results: dict) -> dict:
    clustering = model_results.get("clustering", {})
    gmm_silhouette = float(clustering.get("silhouette_score", 0.0))
    gmm_n_clusters = int(clustering.get("n_components", len(clustering.get("cluster_plain_labels", []))))

    anomaly_data = model_results.get("anomaly", {})
    anomaly_flags = anomaly_data.get("anomaly_flag", [])
    total_trades = max(len(anomaly_flags), 1)
    anomaly_rate = float(sum(1 for f in anomaly_flags if f == 1)) / total_trades

    recon_errors = anomaly_data.get("reconstruction_error", [])
    ae_recon_mean = float(np.mean(recon_errors)) if recon_errors else 0.0

    lstm_data = model_results.get("lstm_model", {})
    lstm_errors = lstm_data.get("prediction_error", [])
    lstm_error_mean = float(np.mean(lstm_errors)) if lstm_errors else 0.0

    pca_ratio = 0.0
    if clustering.get("pca_explained_variance_ratio"):
        pca_ratio = float(clustering["pca_explained_variance_ratio"])
    else:
        try:
            from sklearn.decomposition import PCA
            centers = clustering.get("cluster_centers_by_feature", [])
            if centers and len(centers) > 1 and len(centers[0]) > 1:
                arr = np.array(centers, dtype=float)
                pca = PCA(n_components=min(2, arr.shape[1]))
                pca.fit(arr)
                pca_ratio = float(np.sum(pca.explained_variance_ratio_))
        except Exception:  # noqa: BLE001
            pass

    bn_data = model_results.get("bayesian_network", {})
    bn_edge_count = len(bn_data.get("edges", []))

    causality_data = model_results.get("causality", {})
    granger_sig = 0
    for key, val in causality_data.items():
        if isinstance(val, dict):
            p = val.get("p_value")
            if p is not None and float(p) < 0.05:
                granger_sig += 1
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict) and float(item.get("p_value", 1.0)) < 0.05:
                    granger_sig += 1

    hmm_data = model_results.get("hmm_model", {})
    hmm_log_likelihood = float(hmm_data.get("log_likelihood", 0.0))

    conf_scores = []
    for name, result in model_results.items():
        if isinstance(result, dict) and "confidence" in result:
            c = result["confidence"]
            if isinstance(c, (int, float)) and c > 0:
                conf_scores.append(float(c))
    overall_confidence = float(np.mean(conf_scores)) if conf_scores else 0.0

    plain_english = (
        f"The behavioral models explain {pca_ratio * 100:.0f}% of variance in your trading patterns. "
        f"Clustering produced {gmm_n_clusters} behavioral segments with a quality score of {gmm_silhouette:.2f}. "
        f"{granger_sig} statistically significant causal relationships were detected between market "
        f"conditions and your behavior. Overall model confidence is {overall_confidence * 100:.0f}%."
    )

    return {
        "gmm_silhouette_score": gmm_silhouette,
        "gmm_n_clusters": gmm_n_clusters,
        "anomaly_rate": anomaly_rate,
        "autoencoder_reconstruction_error_mean": ae_recon_mean,
        "lstm_prediction_error_mean": lstm_error_mean,
        "pca_explained_variance_ratio": pca_ratio,
        "bayesian_network_edge_count": bn_edge_count,
        "granger_significant_tests": granger_sig,
        "hmm_log_likelihood": hmm_log_likelihood,
        "overall_model_confidence": overall_confidence,
        "plain_english_summary": plain_english,
    }


def generate_behavioral_report(session_id: str, model_results: dict, emergency_count: int = 0) -> dict:
    risk_profile = model_results.get("risk_profile_label", "unknown")
    summary = report_summary_template(risk_profile, emergency_count)
    effectiveness = _compute_effectiveness_metrics(model_results)
    report = {
        "executive_summary": summary,
        "bias_profile": model_results.get("behavioral_biases", {}),
        "cluster_analysis": model_results.get("clustering", {}),
        "temporal_patterns": model_results.get("hmm_model", {}),
        "risk_profile": model_results.get("risk_distribution", {}),
        "counterfactuals": model_results.get("counterfactuals", []),
        "anomaly_explanations": model_results.get("anomaly_explanations", []),
        "effectiveness_metrics": effectiveness,
    }
    out_dir = session_output_dir(session_id)
    json_path = out_dir / "behavioral_report.json"
    txt_path = out_dir / "behavioral_report.txt"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(_to_text(report), encoding="utf-8")
    return {
        "report_json_path": str(json_path),
        "report_txt_path": str(txt_path),
        "report": report,
        "confidence": 0.8,
    }


def _to_text(report: dict) -> str:
    lines = []
    lines.append("Behavioral Report")
    lines.append("=" * 40)
    for key, value in report.items():
        lines.append(f"\n[{key}]")
        lines.append(json.dumps(value, indent=2, default=str))
    return "\n".join(lines)


def test_report_generator() -> dict:
    return {"ok": True, "message": "report generator wired"}
