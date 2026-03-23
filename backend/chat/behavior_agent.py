from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from backend.database.repository import Repository
from backend.explainability.nlg import behavioral_summary_template, bias_action_recommendation

logger = logging.getLogger(__name__)


def _analysis_map(session_id: str) -> dict[str, dict[str, Any]]:
    """Load all analysis results for a session keyed by model name."""
    repo = Repository()
    rows = repo.get_analysis_results(session_id)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            out[row.model_name] = json.loads(row.result_json)
        except Exception:  # noqa: BLE001
            out[row.model_name] = {}
    return out


def _top_shap_rankings(shap_payload: dict[str, Any], n: int = 5) -> list[dict[str, Any]]:
    """Merge and deduplicate SHAP rankings across GMM, IForest, and LSTM."""
    merged: list[dict[str, Any]] = []
    for key in ("gmm_shap", "iforest_shap", "lstm_shap"):
        for item in shap_payload.get(key, []) or []:
            merged.append(
                {
                    "feature": item.get("feature"),
                    "importance": float(item.get("importance", 0.0)),
                }
            )
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in sorted(merged, key=lambda x: x["importance"], reverse=True):
        feat = item["feature"]
        if feat and feat not in seen:
            seen.add(feat)
            deduped.append(item)
    return deduped[:n]


_BIAS_THRESHOLDS: dict[str, list[tuple[float, str]]] = {
    "disposition_effect_coefficient": [(1.5, "high"), (1.0, "moderate"), (0.5, "low")],
    "revenge_trading_frequency_rate": [(0.3, "high"), (0.15, "moderate"), (0.0, "low")],
    "overconfidence_proxy": [(0.6, "moderate"), (0.3, "low"), (0.0, "low")],
    "signal_following_rate": [],
}

_BIAS_INTERPRETATIONS: dict[str, dict[str, str]] = {
    "disposition_effect_coefficient": {
        "severe": "Extreme tendency to hold losers and cut winners short.",
        "high": "Strong disposition effect — consistently selling winners too early and holding losers.",
        "moderate": "Moderate disposition bias — some tendency to hold losers longer than optimal.",
        "low": "Disposition effect is within normal bounds.",
    },
    "revenge_trading_frequency_rate": {
        "severe": "Very frequent revenge trades after losses, severely impacting returns.",
        "high": "High revenge trading frequency — losses frequently trigger impulsive follow-up trades.",
        "moderate": "Occasional revenge trading detected after loss sequences.",
        "low": "Revenge trading is not a significant pattern.",
    },
    "overconfidence_proxy": {
        "severe": "Extreme overconfidence — position sizing and frequency spike after wins.",
        "high": "High overconfidence markers detected in sizing behavior.",
        "moderate": "Moderate overconfidence — some tendency to increase risk after winning streaks.",
        "low": "Overconfidence is not a significant factor.",
    },
    "signal_following_rate": {
        "low": "Contrarian flag: you frequently trade against indicator signals.",
        "moderate": "Moderate signal adherence — mixed signal following and independent decisions.",
        "high": "Strong signal follower — trades align well with technical indicators.",
        "severe": "Extremely rigid signal following with almost no independent judgment.",
    },
}


def _classify_bias(name: str, value: float | None) -> dict[str, Any]:
    """Classify a single bias metric into severity and interpretation."""
    if value is None:
        return {"name": name, "score": 0.0, "severity": "unknown", "interpretation": "Data unavailable."}
    score = float(value)

    if name == "signal_following_rate":
        if score < 0.4:
            severity = "low"
        elif score < 0.65:
            severity = "moderate"
        else:
            severity = "high"
    else:
        thresholds = _BIAS_THRESHOLDS.get(name, [])
        severity = "severe"
        for threshold, label in thresholds:
            if score >= threshold:
                severity = label
                break
        else:
            severity = "low"

    interpretation = _BIAS_INTERPRETATIONS.get(name, {}).get(severity, "")
    return {"name": name, "score": score, "severity": severity, "interpretation": interpretation}


def _dominant_of(labels: list, default: str = "unknown") -> str:
    """Return the most common element from a list."""
    if not labels:
        return default
    counts = Counter(labels)
    return str(counts.most_common(1)[0][0])


_REQUIRED_MODELS = ["clustering", "hmm_model", "behavioral_biases", "baselines", "emotional_state", "shap"]


def compute_agent_confidence(session_id: str, analysis: dict[str, dict[str, Any]]) -> float:
    """Score confidence 0.0-1.0 based on data quality signals."""
    conf = 1.0
    repo = Repository()
    trades = repo.get_modeling_trades(session_id)
    all_trades = repo.get_trades(session_id)
    if len(trades) < 50:
        conf -= 0.2
    for key in _REQUIRED_MODELS:
        if key not in analysis or not analysis[key]:
            conf -= 0.15
            break
    for key in _REQUIRED_MODELS:
        if analysis.get(key, {}).get("insufficient_data"):
            conf -= 0.1
            break
    if len(all_trades) > 0 and len(all_trades) - len(trades) > 0.3 * len(all_trades):
        conf -= 0.1
    return max(0.1, round(conf, 2))


def _data_freshness_warning(session_id: str) -> str | None:
    """Return a warning if the analysis data is older than 30 days."""
    try:
        repo = Repository()
        session = repo.get_session(session_id)
        if session and session.created_at:
            age_days = (datetime.now(timezone.utc) - session.created_at.replace(tzinfo=timezone.utc)).days
            if age_days > 30:
                return (
                    f"Analysis was computed {age_days} days ago. Market conditions may have changed. "
                    "Consider re-uploading recent trade history."
                )
    except Exception:  # noqa: BLE001
        pass
    return None


def _build_bias_action_items(top_biases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each high/severe bias, derive a concrete action recommendation."""
    items: list[dict[str, Any]] = []
    for bias in top_biases:
        severity = bias["severity"]
        if severity not in ("high", "severe", "moderate"):
            continue
        rec = bias_action_recommendation(bias["name"], severity)
        if rec:
            items.append({
                "bias": bias["name"],
                "severity": severity,
                "action_recommendation": rec,
            })
    return items


def build_behavior_context(session_id: str) -> dict[str, Any]:
    """Build the full behavioral context from analysis results."""
    analysis = _analysis_map(session_id)
    clustering = analysis.get("clustering", {})
    hmm = analysis.get("hmm_model", {})
    biases = analysis.get("behavioral_biases", {})
    baselines = analysis.get("baselines", {})
    emotional = analysis.get("emotional_state", {})
    shap_payload = analysis.get("shap", {})

    bias_baselines = baselines.get("bias_baselines", {}) if isinstance(baselines, dict) else {}

    cluster_labels = clustering.get("cluster_plain_labels", [])
    dominant_cluster = cluster_labels[0] if cluster_labels else "unknown"

    hmm_state_labels = hmm.get("state_labels", [])
    user_state_seq = hmm.get("user_state_sequence", [])
    if user_state_seq:
        dominant_idx = Counter(user_state_seq).most_common(1)[0][0]
        dominant_hmm_state = (
            hmm_state_labels[dominant_idx]
            if isinstance(dominant_idx, int) and dominant_idx < len(hmm_state_labels)
            else f"state_{dominant_idx}"
        )
    else:
        dominant_hmm_state = "unknown"

    emotional_labels = emotional.get("emotional_state_label", [])
    total_e = max(len(emotional_labels), 1)
    emotional_distribution = {
        "calm_disciplined": emotional_labels.count("calm_disciplined") / total_e,
        "anxious_reactive": emotional_labels.count("anxious_reactive") / total_e,
        "euphoric_overconfident": emotional_labels.count("euphoric_overconfident") / total_e,
    }
    dominant_emotional_state = _dominant_of(emotional_labels, "calm_disciplined")

    shap_top = _top_shap_rankings(shap_payload, n=5)

    raw_biases = {
        "disposition_effect_coefficient": bias_baselines.get(
            "disposition_effect_coefficient",
            biases.get("disposition_effect_score"),
        ),
        "revenge_trading_frequency_rate": bias_baselines.get("revenge_trading_frequency_rate"),
        "overconfidence_proxy": bias_baselines.get("overconfidence_proxy"),
        "signal_following_rate": bias_baselines.get("signal_following_rate"),
    }
    top_biases = [_classify_bias(name, val) for name, val in raw_biases.items()]
    severity_order = {"severe": 0, "high": 1, "moderate": 2, "low": 3, "unknown": 4}
    top_biases.sort(key=lambda b: severity_order.get(b["severity"], 5))

    behavioral_summary = behavioral_summary_template(
        dominant_cluster=dominant_cluster,
        dominant_hmm_state=dominant_hmm_state,
        dominant_emotional_state=dominant_emotional_state,
        top_biases=top_biases,
        shap_top_features=shap_top,
    )

    behavior_context = {
        "cluster_descriptions": {
            "labels": cluster_labels,
            "centers_by_feature": clustering.get("cluster_centers_by_feature", []),
            "silhouette_score": clustering.get("silhouette_score"),
            "auto_labels": clustering.get("cluster_plain_labels", []),
        },
        "hmm_state_sequences": {
            "user_state_sequence": user_state_seq,
            # FIX: was reading "transition_matrix" but hmm_model outputs "user_transition_matrix"
            # hmm_model now outputs both keys; reading "user_transition_matrix" here
            "transition_probability_matrix": hmm.get("user_transition_matrix", []),
            "most_likely_state_path": user_state_seq,
            "state_labels": hmm_state_labels,
        },
        "behavioral_biases": {
            "loss_aversion_lambda": biases.get("loss_aversion_lambda"),
            "risk_seeking_alpha": biases.get("risk_seeking_alpha"),
            "disposition_effect_score": biases.get("disposition_effect_score"),
        },
        "bias_baselines": bias_baselines,
        "emotional_state": {
            "distribution": emotional_distribution,
            "dominant_state": dominant_emotional_state,
            "timeline": emotional.get("timeline_score", []),
        },
        "shap_feature_importance": shap_top,
    }

    agent_confidence = compute_agent_confidence(session_id, analysis)
    freshness_warning = _data_freshness_warning(session_id)
    bias_action_items = _build_bias_action_items(top_biases)

    result: dict[str, Any] = {
        "behavior_context": behavior_context,
        "behavioral_summary": behavioral_summary,
        "top_biases": top_biases,
        "dominant_cluster": dominant_cluster,
        "dominant_hmm_state": dominant_hmm_state,
        "dominant_emotional_state": dominant_emotional_state,
        "shap_top_features": shap_top,
        "bias_action_items": bias_action_items,
        "agent_confidence": agent_confidence,
        "source_models": ["GMM", "HMM", "Baselines", "EmotionalState", "SHAP"],
        "retrieval_hints": {
            "preferred_chunk_types": ["bias", "cluster_narrative", "hmm_narrative", "anomaly_trade", "report_section"],
            "top_features": [f["feature"] for f in shap_top if f.get("feature")],
        },
    }
    if freshness_warning:
        result["data_freshness_warning"] = freshness_warning
    return result


def run_behavior_agent(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: build behavioral context and attach to state."""
    ctx = build_behavior_context(state["session_id"])
    state.setdefault("agent_outputs", {})
    state["agent_outputs"]["behavior"] = ctx
    state["behavioral_profile"] = {
        "dominant_cluster": ctx["dominant_cluster"],
        "dominant_hmm_state": ctx["dominant_hmm_state"],
        "dominant_emotional_state": ctx["dominant_emotional_state"],
        "top_biases": ctx["top_biases"],
        "shap_top_features": ctx["shap_top_features"],
    }
    return state


def test_behavior_agent() -> dict:
    """Smoke test for behavior agent module."""
    return {"ok": True, "message": "behavior agent context builder wired"}
