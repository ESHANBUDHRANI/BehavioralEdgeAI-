from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from backend.database.repository import Repository
from backend.explainability.nlg import risk_summary_template

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


def _regime_win_rates_with_counts(session_id: str) -> tuple[dict[str, float], dict[str, int]]:
    """Compute per-regime win rate and sample size from trades + market context."""
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
    rates = {regime: wins.get(regime, 0) / max(total, 1) for regime, total in totals.items()}
    return rates, totals


def _derive_risk_profile_label(var95: float | None, stress_coupling: float | None) -> str:
    """Classify overall risk profile using exact threshold rules."""
    v = var95 if var95 is not None else 0.0
    s = stress_coupling if stress_coupling is not None else 0.0
    if v < -0.05 and s > 0.7:
        return "aggressive"
    if v < -0.03 and s > 0.5:
        return "moderate"
    if v > -0.02 and s < 0.3:
        return "conservative"
    return "erratic"


# ---------------------------------------------------------------------------
# 2A — Agent confidence calibration
# ---------------------------------------------------------------------------

_REQUIRED_MODELS = ["risk_distribution", "garch_model", "behavioral_biases"]


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
# 2B — Data freshness check
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2D — Regime analysis strengthening
# ---------------------------------------------------------------------------

def _regime_confidence_label(count: int) -> str:
    """Classify regime statistical reliability by sample size."""
    if count >= 10:
        return "high"
    if count >= 5:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Core context builder
# ---------------------------------------------------------------------------

def build_risk_context(session_id: str) -> dict[str, Any]:
    """Build the full risk context from analysis results."""
    analysis = _analysis_map(session_id)
    risk_dist = analysis.get("risk_distribution", {})
    garch = analysis.get("garch_model", {})
    biases = analysis.get("behavioral_biases", {})

    var95 = risk_dist.get("var95")
    cvar95 = risk_dist.get("cvar95")
    skewness = risk_dist.get("skewness")
    kurtosis = risk_dist.get("kurtosis")
    tail_dep = risk_dist.get("tail_dependency_coefficient")
    stress_coupling = garch.get("stress_coupling_score")
    cond_vol = garch.get("conditional_volatility_user", [])
    cond_vol_mean = float(sum(cond_vol) / max(len(cond_vol), 1)) if cond_vol else None
    loss_aversion = biases.get("loss_aversion_lambda")

    regime_rates, regime_counts = _regime_win_rates_with_counts(session_id)
    best_regime = max(regime_rates, key=regime_rates.get) if regime_rates else "unknown"
    worst_regime = min(regime_rates, key=regime_rates.get) if regime_rates else "unknown"
    best_win_rate = regime_rates.get(best_regime, 0.0)
    worst_win_rate = regime_rates.get(worst_regime, 0.0)

    risk_profile_label = _derive_risk_profile_label(
        float(var95) if var95 is not None else None,
        float(stress_coupling) if stress_coupling is not None else None,
    )

    risk_summary = risk_summary_template(
        risk_profile_label=risk_profile_label,
        var95=float(var95) if var95 is not None else None,
        cvar95=float(cvar95) if cvar95 is not None else None,
        stress_coupling=float(stress_coupling) if stress_coupling is not None else None,
        best_regime=best_regime,
        best_win_rate=best_win_rate,
        worst_regime=worst_regime,
        worst_win_rate=worst_win_rate,
    )

    risk_context = {
        "var95": float(var95) if var95 is not None else None,
        "cvar95": float(cvar95) if cvar95 is not None else None,
        "skewness": float(skewness) if skewness is not None else None,
        "kurtosis": float(kurtosis) if kurtosis is not None else None,
        "tail_dependency": float(tail_dep) if tail_dep is not None else None,
        "stress_coupling": float(stress_coupling) if stress_coupling is not None else None,
        "conditional_volatility_mean": cond_vol_mean,
        "loss_aversion_lambda": float(loss_aversion) if loss_aversion is not None else None,
    }

    # Confidence calibration
    agent_confidence = compute_agent_confidence(session_id, analysis)

    # Data freshness
    freshness_warning = _data_freshness_warning(session_id)

    # Regime analysis strengthening
    regime_sample_sizes = regime_counts
    regime_confidence = {r: _regime_confidence_label(c) for r, c in regime_counts.items()}
    best_regime_caveat = None
    if regime_confidence.get(best_regime) == "low":
        n = regime_counts.get(best_regime, 0)
        best_regime_caveat = (
            f"Note: best regime identified from only {n} trades. "
            "This finding should be treated as preliminary."
        )

    result: dict[str, Any] = {
        "risk_context": risk_context,
        "risk_summary": risk_summary,
        "risk_profile_label": risk_profile_label,
        "var95": risk_context["var95"],
        "cvar95": risk_context["cvar95"],
        "stress_coupling": risk_context["stress_coupling"],
        "best_regime": best_regime,
        "worst_regime": worst_regime,
        "regime_win_rates": regime_rates,
        "regime_sample_sizes": regime_sample_sizes,
        "regime_confidence": regime_confidence,
        "agent_confidence": agent_confidence,
        "source_models": ["KDE", "GaussianCopula", "DCC-GARCH", "BehavioralBiases"],
        "retrieval_hints": {
            "preferred_chunk_types": ["risk_narrative", "risk", "counterfactual", "model_effectiveness", "report_section"],
            "priority_terms": ["VaR95", "CVaR95", "tail dependency", "stress coupling", "drawdown", "risk profile"],
        },
    }
    if best_regime_caveat:
        result["best_regime_caveat"] = best_regime_caveat
    if freshness_warning:
        result["data_freshness_warning"] = freshness_warning
    return result


def run_risk_agent(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: build risk context and attach to state."""
    context = build_risk_context(state["session_id"])
    state.setdefault("agent_outputs", {})
    state["agent_outputs"]["risk"] = context
    return state


def test_risk_agent() -> dict:
    """Smoke test for risk agent module."""
    return {"ok": True, "message": "risk agent context builder wired"}
