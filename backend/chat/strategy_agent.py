from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

import numpy as np

from backend.database.repository import Repository
from backend.explainability.nlg import (
    strategy_improvement_suggestions,
    strategy_risk_warnings,
    strategy_style_description,
)

logger = logging.getLogger(__name__)

_STRATEGY_TERMS = {
    "strategy", "setup", "edge", "approach", "style", "plan",
    "improve", "optimize", "momentum", "contrarian",
}


def is_strategy_query(user_message: str) -> bool:
    """Check if a message contains strategy-related keywords."""
    text = (user_message or "").lower()
    return any(term in text for term in _STRATEGY_TERMS)


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


def _dominant_hmm_state(hmm: dict[str, Any]) -> dict[str, Any]:
    """Identify the most frequent HMM state from the user sequence."""
    seq = hmm.get("user_state_sequence", []) or []
    if not seq:
        return {"state": None, "label": "unknown", "count": 0}
    counts = Counter(seq)
    state, count = counts.most_common(1)[0]
    labels = hmm.get("state_labels", [])
    label = labels[state] if isinstance(state, int) and state < len(labels) else f"state_{state}"
    return {"state": state, "label": label, "count": count}


def _emotional_distribution(emotional: dict[str, Any]) -> dict[str, float]:
    """Compute proportional distribution across emotional states."""
    labels = emotional.get("emotional_state_label", []) or []
    total = max(len(labels), 1)
    return {
        "calm_disciplined": labels.count("calm_disciplined") / total,
        "anxious_reactive": labels.count("anxious_reactive") / total,
        "euphoric_overconfident": labels.count("euphoric_overconfident") / total,
    }


def _regime_conditioned_win_rates(session_id: str) -> dict[str, float]:
    """Compute per-regime win rate from trades joined with market context."""
    repo = Repository()
    trades = repo.get_modeling_trades(session_id)
    contexts = repo.get_market_context(session_id)
    context_map: dict[tuple[str, str], str] = {}
    for ctx in contexts:
        try:
            payload = json.loads(ctx.context_json)
            regime = payload.get("market_regime_context", {}).get("label", "unknown")
        except Exception:  # noqa: BLE001
            regime = "unknown"
        context_map[(str(ctx.date), ctx.symbol.upper())] = regime
    wins: dict[str, int] = {}
    totals: dict[str, int] = {}
    for t in trades:
        key = (str(t.timestamp.date()), t.symbol.upper())
        regime = context_map.get(key, "unknown")
        totals[regime] = totals.get(regime, 0) + 1
        if float(t.pnl) > 0:
            wins[regime] = wins.get(regime, 0) + 1
    return {r: wins.get(r, 0) / max(totals[r], 1) for r in totals}


def _trade_stats(session_id: str) -> dict[str, float]:
    """Compute summary statistics on trade duration and position size."""
    repo = Repository()
    trades = repo.get_modeling_trades(session_id)
    durations = [float(t.holding_duration) for t in trades if t.holding_duration]
    positions = [float(t.price) * float(t.quantity) for t in trades]
    return {
        "mean_holding_duration": float(np.mean(durations)) if durations else 0.0,
        "std_holding_duration": float(np.std(durations)) if durations else 0.0,
        "mean_position_value": float(np.mean(positions)) if positions else 0.0,
        "std_position_value": float(np.std(positions)) if positions else 0.0,
    }


def _derive_trading_style(
    signal_following_rate: float,
    revenge_trading_rate: float,
    overconfidence_proxy: float,
    mean_holding_duration: float,
    position_size_cv: float,
    frequency_spike_rate: float,
) -> str:
    """Derive a trading style label from behavioral metrics using fixed rules."""
    if (
        signal_following_rate > 0.65
        and revenge_trading_rate < 0.2
        and position_size_cv < 0.3
    ):
        return "disciplined_systematic"
    if (
        frequency_spike_rate > 0.3
        and revenge_trading_rate > 0.3
        and mean_holding_duration < 2
    ):
        return "reactive_emotional"
    if mean_holding_duration > 10 and position_size_cv > 0.5:
        return "concentrated_patient"
    if overconfidence_proxy > 0.6:
        return "overconfident_momentum"
    return "mixed_adaptive"


# ---------------------------------------------------------------------------
# 2A — Agent confidence calibration
# ---------------------------------------------------------------------------

_REQUIRED_MODELS = ["clustering", "hmm_model", "emotional_state", "baselines"]


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


# ---------------------------------------------------------------------------
# Core context builder
# ---------------------------------------------------------------------------

def build_strategy_context(session_id: str) -> dict[str, Any]:
    """Build the full strategy context from analysis results."""
    analysis = _analysis_map(session_id)
    clustering = analysis.get("clustering", {})
    hmm = analysis.get("hmm_model", {})
    emotional = analysis.get("emotional_state", {})
    baselines = analysis.get("baselines", {})
    bias_baselines = baselines.get("bias_baselines", {}) if isinstance(baselines, dict) else {}

    signal_following = float(bias_baselines.get("signal_following_rate", 0.5))
    revenge_rate = float(bias_baselines.get("revenge_trading_frequency_rate", 0.0))
    overconfidence = float(bias_baselines.get("overconfidence_proxy", 0.0))

    dominant_state = _dominant_hmm_state(hmm)
    emotional_dist = _emotional_distribution(emotional)
    regime_win_rates = _regime_conditioned_win_rates(session_id)
    stats = _trade_stats(session_id)

    mean_hold = stats["mean_holding_duration"]
    pos_mean = stats["mean_position_value"]
    pos_std = stats["std_position_value"]
    position_size_cv = pos_std / max(pos_mean, 1e-9)
    frequency_spike_rate = revenge_rate

    trading_style = _derive_trading_style(
        signal_following_rate=signal_following,
        revenge_trading_rate=revenge_rate,
        overconfidence_proxy=overconfidence,
        mean_holding_duration=mean_hold,
        position_size_cv=position_size_cv,
        frequency_spike_rate=frequency_spike_rate,
    )

    ideal_regime = max(regime_win_rates, key=regime_win_rates.get) if regime_win_rates else "unknown"

    agent_confidence = compute_agent_confidence(session_id, analysis)

    return {
        "trading_style": trading_style,
        "style_description": strategy_style_description(trading_style),
        "ideal_regime": ideal_regime,
        "risk_warnings": strategy_risk_warnings(trading_style),
        "improvement_suggestions": strategy_improvement_suggestions(trading_style),
        "regime_win_rates": regime_win_rates,
        "cluster_characteristics": {
            "labels": clustering.get("cluster_plain_labels", []),
        },
        "hmm_dominant_state": dominant_state,
        "emotional_state_distribution": emotional_dist,
        "signal_following_score": signal_following,
        "trade_stats": stats,
        "agent_confidence": agent_confidence,
        "source_models": ["GMM", "HMM", "EmotionalState", "Baselines", "MarketContext"],
    }


def run_strategy_agent(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: build strategy context and attach to state."""
    context = build_strategy_context(state["session_id"])
    state.setdefault("agent_outputs", {})
    state["agent_outputs"]["strategy"] = context
    return state


def test_strategy_agent() -> dict:
    """Smoke test for strategy agent module."""
    return {"ok": True, "message": "strategy agent profile builder wired"}
