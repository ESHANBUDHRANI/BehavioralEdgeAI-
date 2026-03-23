from __future__ import annotations

import json
from backend.database.repository import Repository


def build_system_prompt(session_id: str) -> str:
    repo = Repository()
    results = repo.get_analysis_results(session_id)
    session_rows = repo.get_recent_messages(session_id, limit=1)
    emergency = 0
    profile = "unknown"
    clusters = []
    stability = "unknown"
    top_biases = []
    best_regime = "unknown"
    worst_regime = "unknown"
    emotional_distribution = {}
    loss_aversion = "unknown"
    disposition = "unknown"

    for r in results:
        payload = json.loads(r.result_json)
        if r.model_name == "clustering":
            clusters = payload.get("cluster_plain_labels", [])[:3]
        elif r.model_name == "hmm_model":
            stability = f"{len(set(payload.get('user_state_sequence', [])))} states observed"
        elif r.model_name == "behavioral_biases":
            loss_aversion = payload.get("loss_aversion_lambda", "unknown")
            disposition = payload.get("disposition_effect_score", "unknown")
            top_biases = [
                f"loss_aversion={loss_aversion}",
                f"disposition={disposition}",
                f"risk_alpha={payload.get('risk_seeking_alpha', 'unknown')}",
            ]
        elif r.model_name == "emotional_state":
            labels = payload.get("emotional_state_label", [])
            total = max(len(labels), 1)
            emotional_distribution = {
                "calm_disciplined": labels.count("calm_disciplined") / total,
                "anxious_reactive": labels.count("anxious_reactive") / total,
                "euphoric_overconfident": labels.count("euphoric_overconfident") / total,
            }
        elif r.model_name == "risk_distribution":
            best_regime = "low_volatility"
            worst_regime = "high_volatility"

    if session_rows:
        # Message presence simply confirms session history exists.
        _ = session_rows[0]
    summary = f"""
You are a grounded behavioral trading assistant for session {session_id}.
Top clusters: {clusters if clusters else ['unknown']}.
Behavioral stability: {stability}.
Top biases with evidence: {top_biases if top_biases else ['unknown']}.
Best performing regime: {best_regime}. Worst performing regime: {worst_regime}.
Emotional state distribution: {emotional_distribution if emotional_distribution else {'unknown': 1.0}}.
Loss aversion coefficient: {loss_aversion}. Disposition effect score: {disposition}.
Emergency trade count: {emergency}. Emergency trades are excluded from behavioral modeling.
Risk profile label: {profile}.
Never answer without retrieved context. If chunks are insufficient, state so explicitly.
Always cite chunk sources and model names.
""".strip()
    return summary[:3000]


def test_prompt_builder() -> dict:
    return {"ok": True, "message": "system prompt builder wired"}
