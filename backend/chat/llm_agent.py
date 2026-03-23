from __future__ import annotations

import json
import os
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from backend.chat.prompt_builder import build_system_prompt


def _behavior_section(b: dict[str, Any]) -> str:
    summary = b.get("behavioral_summary", "")
    biases = b.get("top_biases", [])
    bias_lines = ", ".join(f"{x['name']}: {x['severity']}" for x in biases) if biases else "none"
    cluster = b.get("dominant_cluster", "unknown")
    emotion = b.get("dominant_emotional_state", "unknown")
    shap = b.get("shap_top_features", [])
    shap_str = ", ".join(f["feature"] for f in shap[:5]) if shap else "unknown"
    return (
        f"=== BEHAVIORAL PROFILE ===\n"
        f"{summary}\n"
        f"Top biases: {bias_lines}\n"
        f"Dominant cluster: {cluster}\n"
        f"Dominant emotional state: {emotion}\n"
        f"Key behavioral drivers (SHAP): {shap_str}"
    )


def _risk_section(r: dict[str, Any]) -> str:
    summary = r.get("risk_summary", "")
    var95 = r.get("var95")
    cvar95 = r.get("cvar95")
    label = r.get("risk_profile_label", "unknown")
    best = r.get("best_regime", "unknown")
    worst = r.get("worst_regime", "unknown")
    rates = r.get("regime_win_rates", {})
    best_rate = rates.get(best, 0.0)
    worst_rate = rates.get(worst, 0.0)
    return (
        f"=== RISK PROFILE ===\n"
        f"{summary}\n"
        f"VaR 95%: {var95} | CVaR 95%: {cvar95}\n"
        f"Risk profile: {label}\n"
        f"Best regime: {best} ({best_rate:.0%})\n"
        f"Worst regime: {worst} ({worst_rate:.0%})"
    )


def _market_section(m: dict[str, Any]) -> str:
    ticker = m.get("ticker", "unknown")
    regime = m.get("current_regime", "unknown")
    trend = m.get("current_trend", "unknown")
    rsi = m.get("current_rsi", "N/A")
    vol = m.get("current_volatility", "unknown")
    score = m.get("compatibility_score", 0)
    reasoning = m.get("compatibility_reasoning", "")
    return (
        f"=== CURRENT MARKET CONTEXT ===\n"
        f"Ticker: {ticker}\n"
        f"Regime: {regime} | Trend: {trend}\n"
        f"RSI: {rsi} | Volatility: {vol}\n"
        f"Compatibility score: {score}/100\n"
        f"{reasoning}"
    )


def _strategy_section(s: dict[str, Any]) -> str:
    style = s.get("trading_style", "unknown")
    desc = s.get("style_description", "")
    ideal = s.get("ideal_regime", "unknown")
    warnings = s.get("risk_warnings", [])
    warnings_str = "; ".join(warnings) if warnings else "none"
    return (
        f"=== STRATEGY PROFILE ===\n"
        f"Style: {style}\n"
        f"{desc}\n"
        f"Best suited regime: {ideal}\n"
        f"Warnings: {warnings_str}"
    )


def _counterfactual_section(c: dict[str, Any]) -> str:
    desc = c.get("scenario_description", "")
    delta_pnl = c.get("delta_pnl", 0.0)
    delta_wr = c.get("delta_win_rate", 0.0)
    bn_shift = c.get("bayesian_probability_shift", {})
    return (
        f"=== COUNTERFACTUAL RESULT ===\n"
        f"Scenario: {desc}\n"
        f"PnL delta: {delta_pnl:.2f} | Win rate delta: {delta_wr:.4f}\n"
        f"Bayesian probability shift: {json.dumps(bn_shift, default=str)}"
    )


def _knowledge_section(chunks: list[dict[str, Any]]) -> str:
    lines = ["=== RETRIEVED KNOWLEDGE ==="]
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source = meta.get("chunk_type", meta.get("source", "chunk"))
        text = chunk.get("text", "")
        lines.append(f"[SOURCE: {source}] {text[:800]}")
    return "\n".join(lines)


def _build_structured_context(agent_outputs: dict[str, Any], chunks: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    if "behavior" in agent_outputs:
        sections.append(_behavior_section(agent_outputs["behavior"]))
    if "risk" in agent_outputs:
        sections.append(_risk_section(agent_outputs["risk"]))
    if "market" in agent_outputs and not agent_outputs["market"].get("skipped"):
        sections.append(_market_section(agent_outputs["market"]))
    if "strategy" in agent_outputs:
        sections.append(_strategy_section(agent_outputs["strategy"]))
    if "counterfactual" in agent_outputs:
        sections.append(_counterfactual_section(agent_outputs["counterfactual"]))
    if chunks:
        sections.append(_knowledge_section(chunks))
    return "\n\n".join(sections)


def _build_history(history: list[dict[str, Any]]) -> str:
    last_10 = history[-10:] if history else []
    return "\n".join(f"{msg.get('role', '?')}: {msg.get('message', '')}" for msg in last_10)


def _build_sources(agent_outputs: dict[str, Any], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for agent_name, payload in agent_outputs.items():
        if payload is not None and not (isinstance(payload, dict) and payload.get("skipped")):
            sources.append({"type": "agent", "name": agent_name})
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        sources.append({
            "type": "chunk",
            "source": meta.get("source", "chunk"),
            "chunk_type": meta.get("chunk_type", "unknown"),
            "relevance": meta.get("relevance"),
        })
    return sources


def _derive_confidence(agent_outputs: dict[str, Any], chunks: list[dict[str, Any]]) -> str:
    active_agents = sum(
        1 for v in agent_outputs.values()
        if v is not None and not (isinstance(v, dict) and v.get("skipped"))
    )
    chunk_count = len(chunks)
    if active_agents >= 2 and chunk_count >= 3:
        return "high"
    if active_agents >= 1 or 1 <= chunk_count <= 2:
        return "medium"
    return "low"


_INSTRUCTION = (
    "Answer using only the context above. "
    "Cite which section your answer is based on using "
    "[BEHAVIORAL PROFILE], [RISK PROFILE], [MARKET], "
    "[STRATEGY], [COUNTERFACTUAL], or [KNOWLEDGE] tags. "
    "If the context does not contain enough information "
    "to answer confidently, say so explicitly. "
    "Never invent statistics or behavioral patterns "
    "not present in the context."
)


def run_llm_agent(state: dict[str, Any]) -> dict[str, Any]:
    system_prompt = build_system_prompt(state["session_id"])
    agent_outputs = state.get("agent_outputs", {})
    chunks = state.get("retrieved_chunks", [])
    sources = _build_sources(agent_outputs, chunks)
    confidence = _derive_confidence(agent_outputs, chunks)

    if not chunks and not any(
        v for v in agent_outputs.values()
        if v is not None and not (isinstance(v, dict) and v.get("skipped"))
    ):
        state["final_response"] = (
            "I do not have sufficient retrieved context to answer this question. "
            "Please ensure the analysis pipeline has completed for this session."
        )
        state["sources"] = sources
        state["confidence"] = "low"
        return state

    structured = _build_structured_context(agent_outputs, chunks)
    history = _build_history(state.get("conversation_history", []))
    user_message = state.get("user_message", "")

    # CHANGED: using Groq with llama-3.3-70b-versatile — free, fast, best quality
    # Reads GROQ_API_KEY from environment automatically
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    messages = [
        SystemMessage(content=f"{system_prompt}\n\nInstruction: {_INSTRUCTION}"),
        HumanMessage(content=(
            f"Context:\n{structured}\n\n"
            f"Conversation history:\n{history}\n\n"
            f"User question: {user_message}"
        )),
    ]

    reply = llm.invoke(messages)
    state["final_response"] = reply.content if hasattr(reply, "content") else str(reply)
    state["sources"] = sources
    state["confidence"] = confidence
    return state


def test_llm_agent() -> dict:
    return {"ok": True, "message": "llm agent wired with Groq llama-3.3-70b-versatile"}
