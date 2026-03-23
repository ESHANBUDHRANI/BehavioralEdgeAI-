"""
End-to-end integration test for the Behavioral Trading Analysis system.
Run with:  python backend/test_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import get_settings, session_output_dir
from backend.database.init_db import init_db
from backend.database.repository import Repository

# ── Result tracking ───────────────────────────────────────────────────────

results: list[dict] = []


def _record(step: str, status: str, notes: str = ""):
    results.append({"step": step, "status": status, "notes": notes})
    tag = "PASS" if status == "PASS" else f"FAIL: {notes}" if status == "FAIL" else f"SKIPPED: {notes}"
    print(f"  {step}: {tag}")


# ── 1. Generate synthetic trades ──────────────────────────────────────────

def generate_synthetic_trades() -> Path:
    np.random.seed(42)
    symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    n = 100
    base_date = datetime.now() - timedelta(days=365)
    rows = []
    for i in range(n):
        sym = symbols[i % len(symbols)]
        ts = base_date + timedelta(days=int(np.random.uniform(0, 360)))
        side = np.random.choice(["BUY", "SELL"])
        price = float(np.random.uniform(100, 3000))
        qty = int(np.random.randint(1, 100))
        pnl = float(np.random.normal(0, price * 0.02) * qty)
        holding = float(np.random.uniform(0.1, 30))
        rows.append({
            "timestamp": ts.isoformat(),
            "symbol": sym,
            "buy_sell": side,
            "quantity": qty,
            "price": round(price, 2),
            "pnl": round(pnl, 2),
            "holding_duration": round(holding, 2),
        })
    df = pd.DataFrame(rows)
    settings = get_settings()
    out_path = settings.data_dir / "test_trades_synthetic.csv"
    df.to_csv(out_path, index=False)
    return out_path


# ── 2. Ingestion ──────────────────────────────────────────────────────────

def run_ingestion(session_id: str, csv_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        repo = Repository()
        repo.create_session(session_id, csv_path.name)
        records = df.to_dict(orient="records")
        repo.insert_trades(session_id, records)
        repo.set_status(session_id, "analysis_started")
        _record("Ingestion", "PASS", f"{len(records)} trades inserted")
        return df
    except Exception as exc:
        _record("Ingestion", "FAIL", str(exc))
        traceback.print_exc()
        return None


# ── 3. Market context ─────────────────────────────────────────────────────

def run_market_context(session_id: str, trades_df: pd.DataFrame) -> list | None:
    try:
        from backend.market_context.engine import build_market_context_for_trades
        payload = build_market_context_for_trades(trades_df)
        repo = Repository()
        repo.save_market_context(session_id, payload)
        _record("Market Context", "PASS", f"{len(payload)} context items")
        return payload
    except Exception as exc:
        _record("Market Context", "FAIL", str(exc))
        traceback.print_exc()
        return None


# ── 4. Features ───────────────────────────────────────────────────────────

def run_features(trades_df: pd.DataFrame, market_payload: list) -> pd.DataFrame | None:
    try:
        from backend.features.engine import build_behavioral_features
        features_df = build_behavioral_features(trades_df, market_payload)
        _record("Features", "PASS", f"{features_df.shape[1]} features created")
        return features_df
    except Exception as exc:
        _record("Features", "FAIL", str(exc))
        traceback.print_exc()
        return None


# ── 5. Models ─────────────────────────────────────────────────────────────

def run_models(session_id: str, features_df: pd.DataFrame, context_df: pd.DataFrame, market_payload: list) -> dict:
    from backend.models import (
        clustering, hmm_model, anomaly, sentiment, bayesian_network,
        lstm_model, risk_distribution, causality, garch_model,
        behavioral_biases, emotional_state, tft_model,
    )
    from backend.baselines.statistics import compute_baselines

    model_results: dict = {}

    baselines = compute_baselines(features_df)
    model_results["baselines"] = baselines
    _record("Model baselines", "PASS", f"confidence={baselines.get('confidence', 'N/A')}")

    model_modules = [
        ("clustering", clustering),
        ("hmm_model", hmm_model),
        ("anomaly", anomaly),
        ("bayesian_network", bayesian_network),
        ("lstm_model", lstm_model),
        ("risk_distribution", risk_distribution),
        ("causality", causality),
        ("garch_model", garch_model),
        ("behavioral_biases", behavioral_biases),
        ("emotional_state", emotional_state),
        ("tft_model", tft_model),
    ]

    for name, module in model_modules:
        try:
            result = module.run(features_df, context_df, {})
            model_results[name] = result
            if result.get("insufficient_data"):
                _record(f"Model {name}", "SKIPPED", result.get("message", "insufficient data"))
            else:
                conf = result.get("confidence", "N/A")
                _record(f"Model {name}", "PASS", f"confidence={conf}")
        except Exception as exc:
            _record(f"Model {name}", "FAIL", str(exc))
            model_results[name] = {"error": str(exc), "confidence": 0}
            traceback.print_exc()

    try:
        news_items = [
            {"symbol": m["symbol"], "date": str(m["date"]), "headline": h}
            for m in market_payload
            for h in m.get("context", {}).get("news_sentiment_context", {}).get("headlines", [])
            if h
        ]
        sent_result = sentiment.run(features_df, context_df, {"news_items": news_items})
        model_results["sentiment"] = sent_result
        _record("Model sentiment", "PASS", f"confidence={sent_result.get('confidence', 'N/A')}")
    except Exception as exc:
        _record("Model sentiment", "FAIL", str(exc))
        model_results["sentiment"] = {"error": str(exc), "confidence": 0}

    repo = Repository()
    for name, result in model_results.items():
        repo.save_analysis_result(session_id, name, result)

    return model_results


# ── 6. Explainability ─────────────────────────────────────────────────────

def run_explainability(session_id: str, features_df: pd.DataFrame, model_results: dict) -> dict | None:
    try:
        from backend.explainability.shap_explainer import compute_shap_bundle
        from backend.explainability.lime_explainer import explain_anomalous_trade
        from backend.explainability.report_generator import generate_behavioral_report

        shap_data = compute_shap_bundle(
            features_df=features_df,
            clustering_labels=model_results.get("clustering", {}).get("gmm_labels", []),
            anomaly_scores=model_results.get("anomaly", {}).get("anomaly_confidence", [0.0] * len(features_df)),
            lstm_prediction_error=model_results.get("lstm_model", {}).get("prediction_error", []),
        )
        model_results["shap"] = shap_data

        anomaly_indices = [
            i for i, f in enumerate(model_results.get("anomaly", {}).get("anomaly_flag", []))
            if f == 1
        ][:5]
        model_results["anomaly_explanations"] = [
            explain_anomalous_trade(
                features_df, i,
                model_results.get("anomaly", {}).get("anomaly_flag", []),
            )
            for i in anomaly_indices
        ]

        report = generate_behavioral_report(session_id, model_results)
        effectiveness = report.get("report", {}).get("effectiveness_metrics", {})
        has_metrics = bool(effectiveness.get("plain_english_summary"))
        _record("Explainability", "PASS", f"effectiveness_metrics={'present' if has_metrics else 'missing'}")
        return report
    except Exception as exc:
        _record("Explainability", "FAIL", str(exc))
        traceback.print_exc()
        return None


# ── 7. Charts ─────────────────────────────────────────────────────────────

def run_charts(session_id: str, trades_df: pd.DataFrame, features_df: pd.DataFrame, model_results: dict) -> int:
    try:
        from backend.visualizations.charts import generate_all_charts
        paths = generate_all_charts(session_id, trades_df, features_df, model_results)
        chart_dir = session_output_dir(session_id) / "charts"
        html_files = list(chart_dir.glob("*.html"))
        _record("Charts", "PASS", f"{len(html_files)}/15 generated")
        return len(html_files)
    except Exception as exc:
        _record("Charts", "FAIL", str(exc))
        traceback.print_exc()
        return 0


# ── 8. RAG index ──────────────────────────────────────────────────────────

def run_rag_index(session_id: str, report: dict, model_results: dict) -> int:
    try:
        from backend.chat.index_builder import (
            build_narrative_chunks,
            build_precomputed_counterfactual_chunks,
            build_session_index,
        )

        chunks: list[dict] = []
        report_data = report.get("report", {}) if report else {}
        for section, content in report_data.items():
            chunks.append({
                "text": f"{section}: {json.dumps(content, default=str)[:500]}",
                "metadata": {
                    "session_id": session_id,
                    "chunk_type": "report_section",
                    "source": f"report:{section}",
                    "confidence": 0.8,
                },
            })

        chunks += build_narrative_chunks(session_id, model_results)
        chunks += build_precomputed_counterfactual_chunks(session_id)
        result = build_session_index(session_id, chunks)
        n = result.get("indexed_chunks", 0)
        _record("RAG Index", "PASS", f"{n} chunks indexed")
        return n
    except Exception as exc:
        _record("RAG Index", "FAIL", str(exc))
        traceback.print_exc()
        return 0


# ── 9. Chat routing ──────────────────────────────────────────────────────

def run_chat_tests(session_id: str):
    from backend.chat.graph import ChatState

    turns = [
        {
            "message": "What are my worst trading biases?",
            "expected_intent": "bias_query",
            "expected_agents": ["behavior"],
        },
        {
            "message": "What is my risk profile?",
            "expected_intent": "risk_profile_query",
            "expected_agents": ["risk"],
        },
        {
            "message": "What if I had smaller positions?",
            "expected_intent": "whatif_counterfactual",
            "expected_agents": ["counterfactual"],
        },
        {
            "message": "Should I look at RELIANCE?",
            "expected_intent": "recommendation_request",
            "expected_agents": ["market", "strategy"],
        },
        {
            "message": "What trading style am I?",
            "expected_intent": "general_explanation",
            "expected_agents": ["strategy"],
        },
    ]

    from backend.chat.graph import intent_node, _route_after_intent

    all_pass = True
    for turn in turns:
        state: dict = {
            "session_id": session_id,
            "user_message": turn["message"],
            "intent": "general_explanation",
            "retrieved_chunks": [],
            "agent_outputs": {},
            "conversation_history": [],
            "behavioral_profile": {},
            "final_response": "",
            "sources": [],
            "confidence": "low",
        }
        try:
            state = intent_node(state)
            detected_intent = state["intent"]
            route = _route_after_intent(state)
            matched = detected_intent == turn["expected_intent"]

            print(f"    Turn: '{turn['message'][:40]}...'")
            print(f"      Intent: {detected_intent} (expected {turn['expected_intent']}) {'OK' if matched else 'MISMATCH'}")
            print(f"      Route: {route}")

            if not matched:
                all_pass = False
        except Exception as exc:
            print(f"    Turn FAILED: {exc}")
            all_pass = False

    # Now test full agent execution for first turn (bias_query -> behavior_agent)
    try:
        from backend.chat.behavior_agent import run_behavior_agent
        state_full: dict = {
            "session_id": session_id,
            "user_message": "What are my worst trading biases?",
            "intent": "bias_query",
            "retrieved_chunks": [],
            "agent_outputs": {},
            "conversation_history": [],
            "behavioral_profile": {},
            "final_response": "",
            "sources": [],
            "confidence": "low",
        }
        state_full = run_behavior_agent(state_full)
        behavior_out = state_full["agent_outputs"].get("behavior", {})
        has_summary = bool(behavior_out.get("behavioral_summary"))
        has_biases = bool(behavior_out.get("top_biases"))
        print(f"    Behavior agent output: summary={'yes' if has_summary else 'no'}, biases={'yes' if has_biases else 'no'}")
        if not has_summary or not has_biases:
            all_pass = False
    except Exception as exc:
        print(f"    Behavior agent FAILED: {exc}")
        all_pass = False

    # Test risk agent
    try:
        from backend.chat.risk_agent import run_risk_agent
        state_risk: dict = {
            "session_id": session_id,
            "user_message": "What is my risk profile?",
            "intent": "risk_profile_query",
            "retrieved_chunks": [],
            "agent_outputs": {},
            "conversation_history": [],
            "behavioral_profile": {},
            "final_response": "",
            "sources": [],
            "confidence": "low",
        }
        state_risk = run_risk_agent(state_risk)
        risk_out = state_risk["agent_outputs"].get("risk", {})
        has_label = bool(risk_out.get("risk_profile_label"))
        print(f"    Risk agent output: label={risk_out.get('risk_profile_label', 'MISSING')}, var95={risk_out.get('var95', 'MISSING')}")
        if not has_label:
            all_pass = False
    except Exception as exc:
        print(f"    Risk agent FAILED: {exc}")
        all_pass = False

    # Test strategy agent
    try:
        from backend.chat.strategy_agent import run_strategy_agent
        state_strat: dict = {
            "session_id": session_id,
            "user_message": "What trading style am I?",
            "intent": "general_explanation",
            "retrieved_chunks": [],
            "agent_outputs": {},
            "conversation_history": [],
            "behavioral_profile": {},
            "final_response": "",
            "sources": [],
            "confidence": "low",
        }
        state_strat = run_strategy_agent(state_strat)
        strat_out = state_strat["agent_outputs"].get("strategy", {})
        print(f"    Strategy agent output: style={strat_out.get('trading_style', 'MISSING')}")
    except Exception as exc:
        print(f"    Strategy agent FAILED: {exc}")
        all_pass = False

    _record("Chat Routing", "PASS" if all_pass else "FAIL", "see details above")


# ── Build context_df helper ──────────────────────────────────────────────

def _build_context_df(market_payload: list) -> pd.DataFrame:
    rows = []
    for item in market_payload:
        ctx = item.get("context", {})
        regime = ctx.get("market_regime_context", {}).get("label", "unknown")
        sentiment_ctx = ctx.get("news_sentiment_context", {})
        rows.append({
            "market_regime": regime,
            "news_sentiment": sentiment_ctx.get("label", "neutral"),
            "news_sentiment_score": float(sentiment_ctx.get("score", 0.0)),
            "volatility_score": 1.0 if ctx.get("volatility_context", {}).get("label") == "high_expansion" else 0.0,
            "market_regime_num": 1.0 if regime == "trending" else 0.0,
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BEHAVIORAL TRADING ANALYSIS — INTEGRATION TEST")
    print("=" * 60)

    init_db()
    session_id = uuid.uuid4().hex
    print(f"\nSession ID: {session_id}\n")

    # 1. Synthetic trades
    print("[1/9] Generating synthetic trades...")
    csv_path = generate_synthetic_trades()
    print(f"  Saved to {csv_path}")

    # 2. Ingestion
    print("\n[2/9] Running ingestion...")
    trades_df = run_ingestion(session_id, csv_path)
    if trades_df is None:
        print("ABORT: Ingestion failed.")
        _print_summary()
        return

    # 3. Market context
    print("\n[3/9] Running market context...")
    market_payload = run_market_context(session_id, trades_df)
    if market_payload is None:
        market_payload = []

    # 4. Features
    print("\n[4/9] Running feature engineering...")
    features_df = run_features(trades_df, market_payload)
    if features_df is None:
        print("ABORT: Feature engineering failed.")
        _print_summary()
        return

    # 5. Models
    print("\n[5/9] Running models...")
    context_df = _build_context_df(market_payload)
    model_results = run_models(session_id, features_df, context_df, market_payload)

    # 6. Explainability
    print("\n[6/9] Running explainability...")
    report = run_explainability(session_id, features_df, model_results)

    # 7. Charts
    print("\n[7/9] Generating charts...")
    run_charts(session_id, trades_df, features_df, model_results)

    # 8. RAG index
    print("\n[8/9] Building RAG index...")
    run_rag_index(session_id, report, model_results)

    # 9. Chat routing
    print("\n[9/9] Testing chat routing and agents...")
    run_chat_tests(session_id)

    _print_summary()


def _print_summary():
    print("\n" + "=" * 60)
    print(f"{'Step':<30} | {'Status':<8} | Notes")
    print("-" * 60)
    overall = "PASS"
    for r in results:
        status = r["status"]
        if status == "FAIL":
            overall = "FAIL"
        print(f"{r['step']:<30} | {status:<8} | {r['notes'][:40]}")
    print("-" * 60)
    print(f"{'Overall':<30} | {overall:<8} |")
    print("=" * 60)


if __name__ == "__main__":
    main()
