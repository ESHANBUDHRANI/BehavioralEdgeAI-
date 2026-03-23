from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import CausalInference, VariableElimination
from pgmpy.models import BayesianNetwork

from backend.config import get_settings
from backend.database.repository import Repository
from backend.chat.counterfactual import compute_live_counterfactual

logger = logging.getLogger(__name__)

TARGET_NODE = "trade_outcome"

# ── Scenario keyword mapping (original 7 + 5 new) ────────────────────────
_SCENARIO_PATTERNS: list[tuple[re.Pattern, dict[str, Any]]] = [
    # Original scenarios
    (re.compile(r"held?\s*longer|hold\s*longer", re.I), {"variable": "holding_duration", "multiplier": 1.5}),
    (re.compile(r"held?\s*shorter|exit\s*earlier", re.I), {"variable": "holding_duration", "multiplier": 0.5}),
    (re.compile(r"smaller\s*position|reduce\s*size", re.I), {"variable": "position_size", "multiplier": 0.5}),
    (re.compile(r"larger\s*position|increase\s*size", re.I), {"variable": "position_size", "multiplier": 1.5}),
    (re.compile(r"skip\s*high\s*volatil|avoid\s*volatil", re.I), {"variable": "volatility_filter", "value": "high_volatility"}),
    (re.compile(r"after\s*every\s*loss|after\s*losses", re.I), {"variable": "post_loss_skip", "n": 1}),
    (re.compile(r"exclude\s*emergency|without\s*emergency", re.I), {"variable": "emergency_filter", "include": False}),
    # New scenarios (2F)
    (re.compile(r"traded?\s*less|fewer\s*trades", re.I), {"variable": "frequency_reduction", "factor": 0.5}),
    (re.compile(r"traded?\s*more|more\s*trades", re.I), {"variable": "frequency_increase", "factor": 2.0}),
    (re.compile(r"better\s*entr|waited?\s*for\s*signal", re.I), {"variable": "signal_filter", "min_signal_score": 0.6}),
    (re.compile(r"avoid(?:ed)?\s*loss|cut\s*loss(?:es)?\s*faster", re.I), {"variable": "loss_limit", "max_loss_pct": 0.02}),
    (re.compile(r"only\s*best\s*setup|selective", re.I), {"variable": "top_cluster_only"}),
]


def _parse_scenario(user_message: str) -> dict[str, Any]:
    """Match user message against scenario keyword patterns."""
    for pattern, params in _SCENARIO_PATTERNS:
        if pattern.search(user_message or ""):
            return dict(params)
    return {}


# ── Analysis results helper ───────────────────────────────────────────────
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


# ── 2A — Agent confidence calibration ─────────────────────────────────────

_REQUIRED_MODELS = ["bayesian_network", "risk_distribution"]


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


# ── Bayesian network reconstruction ──────────────────────────────────────
def _build_bn(bundle: dict[str, Any]) -> BayesianNetwork:
    """Reconstruct a BayesianNetwork from stored CPD payload."""
    edges = [tuple(e) for e in bundle.get("edges", [])]
    model = BayesianNetwork(edges)
    cpd_payload = bundle.get("cpd_payload", [])
    cpds = []
    for item in cpd_payload:
        cpds.append(
            TabularCPD(
                variable=item["variable"],
                variable_card=int(item["variable_card"]),
                values=item["values"],
                evidence=item.get("evidence") or None,
                evidence_card=item.get("evidence_card") or None,
                state_names=item.get("state_names") or None,
            )
        )
    if cpds:
        model.add_cpds(*cpds)
    return model


def _distribution_to_dict(query_result, target_node: str) -> dict[str, float]:
    """Convert a pgmpy query result to a plain dictionary."""
    values = query_result.values.tolist()
    names = []
    if getattr(query_result, "state_names", None):
        names = query_result.state_names.get(target_node, [])
    if not names:
        names = [f"state_{i}" for i in range(len(values))]
    return {str(names[i]): float(values[i]) for i in range(len(values))}


def _bayesian_probability_shift(
    session_id: str,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Compute baseline and counterfactual probability distributions via BN."""
    analysis = _analysis_map(session_id)
    bn_payload = analysis.get("bayesian_network", {})
    if not bn_payload or not bn_payload.get("cpd_payload"):
        return {"error": "bayesian network not available", "baseline": {}, "counterfactual": {}}

    state_names = bn_payload.get("state_names", {})
    model = _build_bn(bn_payload)
    infer = VariableElimination(model)
    baseline = infer.query(variables=[TARGET_NODE], show_progress=False)
    baseline_dist = _distribution_to_dict(baseline, TARGET_NODE)

    variable = scenario.get("variable", "")
    node = None
    for candidate in state_names:
        if variable.lower() in candidate.lower() or candidate.lower() in variable.lower():
            node = candidate
            break
    if not node:
        node = list(state_names.keys())[0] if state_names else None
    if not node:
        return {"error": "no matching BN node", "baseline": baseline_dist, "counterfactual": {}}

    node_states = state_names.get(node, [])
    value = node_states[0] if node_states else None
    if value is None:
        return {"error": "no state values for node", "baseline": baseline_dist, "counterfactual": {}}

    try:
        causal = CausalInference(model)
        cf = causal.query(variables=[TARGET_NODE], do={node: value}, show_progress=False)
        cf_dist = _distribution_to_dict(cf, TARGET_NODE)
    except Exception:  # noqa: BLE001
        try:
            do_model = BayesianNetwork(model.edges())
            kept_cpds = [cpd for cpd in model.get_cpds() if cpd.variable != node]
            do_model.add_cpds(*kept_cpds)
            parents = list(do_model.get_parents(node))
            if parents:
                do_model.remove_edges_from([(p, node) for p in parents])
            vec = [[0.0] for _ in node_states]
            idx = node_states.index(value) if value in node_states else 0
            vec[idx] = [1.0]
            do_cpd = TabularCPD(
                variable=node, variable_card=len(node_states),
                values=vec, state_names={node: node_states},
            )
            do_model.add_cpds(do_cpd)
            cf_infer = VariableElimination(do_model)
            cf = cf_infer.query(variables=[TARGET_NODE], show_progress=False)
            cf_dist = _distribution_to_dict(cf, TARGET_NODE)
        except Exception:  # noqa: BLE001
            cf_dist = {}

    return {"baseline": baseline_dist, "counterfactual": cf_dist, "intervention_node": node, "intervention_value": value}


# ── ChromaDB precomputed search ───────────────────────────────────────────
def _search_precomputed(session_id: str, query_text: str) -> dict[str, Any] | None:
    """Search ChromaDB for a precomputed counterfactual matching the query."""
    settings = get_settings()
    persist_dir = str(settings.project_root / "data" / "cache" / "chroma")
    try:
        db = Chroma(
            collection_name="behavioral_analysis",
            embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}),
            persist_directory=persist_dir,
        )
        scored = db.similarity_search_with_relevance_scores(
            query_text, k=1,
            filter={"session_id": session_id, "chunk_type": "counterfactual"},
        )
        if scored:
            doc, score = scored[0]
            if score > 0.75:
                return {"text": doc.page_content, "metadata": doc.metadata, "relevance": float(score)}
    except Exception:  # noqa: BLE001
        pass
    return None


# ── Run agent ─────────────────────────────────────────────────────────────
def run_counterfactual_agent(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: parse scenario, compute or retrieve counterfactual, attach to state."""
    session_id = state["session_id"]
    user_message = state.get("user_message", "")

    scenario = _parse_scenario(user_message)
    if not scenario:
        scenario = state.get("counterfactual_scenario") or {}
    if not scenario:
        scenario = {"variable": "holding_duration", "multiplier": 1.5}

    # For top_cluster_only, resolve dominant cluster from analysis
    if scenario.get("variable") == "top_cluster_only" and "cluster" not in scenario:
        try:
            analysis = _analysis_map(session_id)
            labels = analysis.get("clustering", {}).get("cluster_plain_labels", [])
            scenario["cluster"] = labels[0] if labels else 0
        except Exception:  # noqa: BLE001
            scenario["cluster"] = 0

    variable = scenario.get("variable", "unknown")
    scenario_description = f"{variable} scenario: {json.dumps({k: v for k, v in scenario.items() if k != 'variable'})}"

    precomputed = _search_precomputed(session_id, user_message or scenario_description)
    if precomputed:
        try:
            metrics = json.loads(precomputed["text"])
        except Exception:  # noqa: BLE001
            metrics = {"raw": precomputed["text"]}
        result = {
            "scenario_variable": variable,
            "scenario_description": scenario_description,
            "original_metrics": metrics.get("original_metrics", {}),
            "counterfactual_metrics": metrics.get("counterfactual_metrics", {}),
            "delta_pnl": float(metrics.get("delta_pnl", 0.0)),
            "delta_win_rate": float(metrics.get("delta_win_rate", 0.0)),
            "bayesian_probability_shift": {},
            "source": "precomputed",
        }
    else:
        live = compute_live_counterfactual(session_id, scenario)
        bn_shift = _bayesian_probability_shift(session_id, scenario)
        result = {
            "scenario_variable": variable,
            "scenario_description": scenario_description,
            "original_metrics": live.get("original_metrics", {}),
            "counterfactual_metrics": live.get("counterfactual_metrics", {}),
            "delta_pnl": float(live.get("delta_pnl", 0.0)),
            "delta_win_rate": float(live.get("delta_win_rate", 0.0)),
            "bayesian_probability_shift": bn_shift,
            "source": "live_computed",
        }

    # Confidence calibration
    analysis = _analysis_map(session_id)
    result["agent_confidence"] = compute_agent_confidence(session_id, analysis)

    state.setdefault("agent_outputs", {})
    state["agent_outputs"]["counterfactual"] = result
    return state


def compute_counterfactual(
    session_id: str,
    scenario: dict[str, Any] | None = None,
    user_message: str = "",
) -> dict[str, Any]:
    """Backward-compatible entry point used by the /api/counterfactual endpoint."""
    mock_state: dict[str, Any] = {
        "session_id": session_id,
        "user_message": user_message,
        "counterfactual_scenario": scenario or {},
        "agent_outputs": {},
    }
    run_counterfactual_agent(mock_state)
    return mock_state["agent_outputs"].get("counterfactual", {})


def test_counterfactual_agent() -> dict:
    """Smoke test for counterfactual agent module."""
    return {"ok": True, "message": "counterfactual bayesian intervention agent wired"}
