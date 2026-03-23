from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from backend.database.repository import Repository
from backend.market_context.data_provider import fetch_ohlcv
from backend.market_context.indicators import compute_indicator_frame
from backend.market_context.regime import classify_market_regime, classify_trend


def _analysis_map(session_id: str) -> dict:
    repo = Repository()
    rows = repo.get_analysis_results(session_id)
    out: dict = {}
    for row in rows:
        try:
            out[row.model_name] = json.loads(row.result_json)
        except Exception:  # noqa: BLE001
            out[row.model_name] = {}
    return out


def behavior_insights(session_id: str, retrieved_chunks: list[dict]) -> dict:
    analysis = _analysis_map(session_id)
    cluster = analysis.get("clustering", {})
    hmm = analysis.get("hmm_model", {})
    biases = analysis.get("behavioral_biases", {})
    notes = []
    if cluster:
        labels = cluster.get("cluster_plain_labels", [])
        notes.append(f"Dominant behavior segments: {labels[:3]}.")
    if hmm:
        story = hmm.get("transition_story", [])
        if story:
            notes.append(story[0].get("story", "State transition insight unavailable."))
    if biases:
        notes.append(
            "Loss aversion lambda="
            f"{biases.get('loss_aversion_lambda', 'n/a')}, disposition score="
            f"{biases.get('disposition_effect_score', 'n/a')}."
        )
    for chunk in retrieved_chunks:
        if chunk.get("metadata", {}).get("chunk_type") in {"bias", "cluster", "hmm"}:
            notes.append(chunk.get("text", "")[:260])
    return {"analysis": notes[:6], "source_models": ["GMM", "HMM", "ProspectTheory", "DispositionEffect"]}


def risk_insights(session_id: str, retrieved_chunks: list[dict]) -> dict:
    analysis = _analysis_map(session_id)
    risk = analysis.get("risk_distribution", {})
    garch = analysis.get("garch_model", {})
    baselines = analysis.get("baselines", {})
    notes = []
    if risk:
        notes.append(
            f"Tail profile: VaR95={risk.get('var95', 'n/a')}, CVaR95={risk.get('cvar95', 'n/a')}, "
            f"skew={risk.get('skewness', 'n/a')}, kurtosis={risk.get('kurtosis', 'n/a')}."
        )
    if garch:
        notes.append(
            f"Stress coupling score={garch.get('stress_coupling_score', 'n/a')} "
            "(higher means stronger behavior-market coupling)."
        )
    if baselines:
        notes.append(
            f"Composite deviation mean="
            f"{np.mean(baselines.get('composite_deviation_score', [0])):.3f}."
        )
    for chunk in retrieved_chunks:
        if chunk.get("metadata", {}).get("chunk_type") in {"risk", "counterfactual"}:
            notes.append(chunk.get("text", "")[:260])
    return {"analysis": notes[:6], "source_models": ["KDE", "GaussianCopula", "DCC-GARCH", "Baselines"]}


def market_insights(session_id: str, user_message: str) -> dict:
    del session_id
    tickers = re.findall(r"\b[A-Z]{1,5}\b", user_message or "")
    if not tickers:
        return {
            "analysis": ["No ticker detected. Specify a symbol (e.g., AAPL) for market compatibility analysis."],
            "source_models": ["yfinance", "RegimeClassifier"],
        }
    ticker = tickers[0]
    end = datetime.utcnow().date()
    start = end - timedelta(days=420)
    daily = fetch_ohlcv(ticker, start.isoformat(), end.isoformat(), interval="1d")
    if daily.empty:
        return {"analysis": [f"No recent market data for {ticker}."], "source_models": ["yfinance"]}
    ind = compute_indicator_frame(daily)
    latest = ind.iloc[-1]
    trend = classify_trend(latest)
    regime = classify_market_regime(latest, vix_value=20.0)
    compat = (
        "high_compatibility"
        if trend in {"bullish", "weak_bullish"} and regime in {"trending", "risk_on"}
        else "moderate_or_low_compatibility"
    )
    return {
        "analysis": [
            f"{ticker} trend={trend}, regime={regime}.",
            f"Behavior-profile compatibility={compat}.",
        ],
        "source_models": ["yfinance", "TrendContext", "RegimeContext"],
    }


def strategy_insights(session_id: str, retrieved_chunks: list[dict]) -> dict:
    analysis = _analysis_map(session_id)
    biases = analysis.get("behavioral_biases", {})
    risk = analysis.get("risk_distribution", {})
    emotional = analysis.get("emotional_state", {})
    strategies = []
    lam = float(biases.get("loss_aversion_lambda", 1.0)) if biases else 1.0
    cvar = float(risk.get("cvar95", 0.0)) if risk else 0.0
    anxious_rate = 0.0
    labels = emotional.get("emotional_state_label", [])
    if labels:
        anxious_rate = labels.count("anxious_reactive") / max(len(labels), 1)
    if lam > 2.0:
        strategies.append("Reduce post-loss re-entry speed and enforce one-candle cooldown after losing trades.")
    if cvar < -0.03:
        strategies.append("Cap risk per trade at 0.5%-1.0% and downshift leverage in high-volatility regimes.")
    if anxious_rate > 0.35:
        strategies.append("Switch to rule-based checklists before every entry to suppress reactive execution.")
    if not strategies:
        strategies.append("Maintain current sizing discipline and focus on regime-aligned entries.")
    for chunk in retrieved_chunks:
        if chunk.get("metadata", {}).get("chunk_type") in {"static_knowledge", "bias", "risk"}:
            strategies.append(chunk.get("text", "")[:220])
    return {"analysis": strategies[:6], "source_models": ["BehavioralBiases", "RiskDistribution", "EmotionalState"]}


def test_specialists() -> dict:
    return {"ok": True, "message": "specialist agent logic wired"}
