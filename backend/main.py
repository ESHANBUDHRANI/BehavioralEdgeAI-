from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sklearn.decomposition import PCA

from backend.config import get_settings, session_output_dir
from backend.database.init_db import init_db
from backend.database.repository import Repository
from backend.ingestion.normalize import normalize_columns, trade_preview
from backend.ingestion.ocr_parser import extract_image_table
from backend.ingestion.pdf_parser import extract_pdf_tables
from backend.ingestion.position_reconstruction import reconstruct_positions_fifo
from backend.market_context.engine import build_market_context_for_trades
from backend.market_context.data_provider import fetch_ohlcv
from backend.features.engine import build_behavioral_features
from backend.baselines.statistics import compute_baselines
from backend.models import (
    clustering,
    hmm_model,
    anomaly,
    sentiment,
    bayesian_network,
    lstm_model,
    risk_distribution,
    causality,
    garch_model,
    behavioral_biases,
    emotional_state,
    tft_model,
)
from backend.explainability.shap_explainer import compute_shap_bundle
from backend.explainability.lime_explainer import explain_anomalous_trade
from backend.explainability.report_generator import generate_behavioral_report
from backend.visualizations.charts import generate_all_charts
from backend.chat.graph import build_graph
from backend.chat.memory import get_recent, save_message
from backend.chat.counterfactual import run_counterfactual
from backend.chat.index_builder import (
    build_narrative_chunks,
    build_precomputed_counterfactual_chunks,
    build_session_index,
    build_static_knowledge_index,
)

_STATIC_INDEX_READY = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _STATIC_INDEX_READY
    try:
        build_static_knowledge_index()
        _STATIC_INDEX_READY = True
    except Exception:
        _STATIC_INDEX_READY = False
    yield


app = FastAPI(title="Local Behavioral Trading Analysis", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
repo = Repository()
progress_store: dict[str, dict[str, Any]] = {}


def _parse_upload(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".pdf":
        raw = extract_pdf_tables(path)
    elif path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        raw = extract_image_table(path)
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use PDF, image, or CSV.")
    return normalize_columns(raw)


def _set_progress(session_id: str, stage: str, pct: int) -> None:
    progress_store[session_id] = {
        "session_id": session_id,
        "stage": stage,
        "progress": pct,
        "updated_at": datetime.utcnow().isoformat(),
    }


def _build_context_df(market_payload: list[dict], target_length: int) -> pd.DataFrame:
    rows = []
    for item in market_payload:
        regime = item["context"]["market_regime_context"]["label"]
        sentiment_ctx = item["context"]["news_sentiment_context"]
        rows.append(
            {
                "market_regime": regime,
                "news_sentiment": sentiment_ctx.get("label", "neutral"),
                "news_sentiment_score": float(sentiment_ctx.get("score", 0.0)),
                "volatility_score": 1.0 if item["context"]["volatility_context"]["label"] == "high_expansion" else 0.0,
                "market_regime_num": 1.0 if regime == "trending" else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            {
                "market_regime": ["unknown"] * target_length,
                "news_sentiment": ["neutral"] * target_length,
                "news_sentiment_score": [0.0] * target_length,
                "volatility_score": [0.0] * target_length,
                "market_regime_num": [0.0] * target_length,
            }
        )
    elif len(df) < target_length:
        last_row = df.iloc[[-1]]
        pad = pd.concat([last_row] * (target_length - len(df)), ignore_index=True)
        df = pd.concat([df, pad], ignore_index=True)
    elif len(df) > target_length:
        df = df.iloc[:target_length].reset_index(drop=True)
    return df


def _inject_sentiment_into_context(market_payload: list[dict], sentiment_map: dict) -> None:
    for item in market_payload:
        key = f"{item.get('symbol')}|{item.get('date')}"
        s = sentiment_map.get(key)
        if not s:
            continue
        item["context"]["news_sentiment_context"]["label"] = s.get("label", "neutral")
        item["context"]["news_sentiment_context"]["confidence"] = float(s.get("confidence", 0.0))
        item["context"]["news_sentiment_context"]["score"] = float(s.get("score", 0.0))


def _build_effectiveness_panel(features_df: pd.DataFrame, model_results: dict) -> dict:
    numeric = features_df.select_dtypes(include=["number"]).fillna(0)
    pca_ratio = 0.0
    if len(numeric) > 3 and numeric.shape[1] > 1:
        pca = PCA(n_components=min(3, numeric.shape[1]))
        pca.fit(numeric.to_numpy(dtype=float))
        pca_ratio = float(np.sum(pca.explained_variance_ratio_))
    anomaly_flags = model_results.get("anomaly", {}).get("anomaly_flag", [])
    anomaly_rate = float(np.mean(anomaly_flags)) if anomaly_flags else 0.0
    lstm_error = model_results.get("lstm_model", {}).get("prediction_error", [])
    avg_lstm_err = float(np.mean(lstm_error)) if lstm_error else 0.0
    silhouette = float(model_results.get("clustering", {}).get("silhouette_score", 0.0))
    recon_err = model_results.get("anomaly", {}).get("reconstruction_error", [])
    recon_p95 = float(np.quantile(recon_err, 0.95)) if recon_err else 0.0
    explain_pct = max(0.0, min(100.0, pca_ratio * 100.0))
    return {
        "gmm_silhouette_score": silhouette,
        "autoencoder_reconstruction_error_p95": recon_p95,
        "isolation_forest_anomaly_rate": anomaly_rate,
        "lstm_prediction_error_mean": avg_lstm_err,
        "pca_explained_variance_ratio": pca_ratio,
        "plain_english": f"This model suite explains approximately {explain_pct:.1f}% of behavioral variance.",
    }


def _build_rag_chunks(session_id: str, report: dict, model_results: dict) -> list[dict]:
    chunks: list[dict] = []
    for section, content in report.items():
        text = f"{section}: {json.dumps(content, default=str)}"
        chunk_type = "report_section"
        if "bias" in section:
            chunk_type = "bias"
        if "risk" in section:
            chunk_type = "risk"
        if "counterfactual" in section:
            chunk_type = "counterfactual"
        chunks.append(
            {
                "text": text,
                "metadata": {
                    "session_id": session_id,
                    "chunk_type": chunk_type,
                    "time_period": "full_session",
                    "symbol": "ALL",
                    "confidence": 0.8,
                    "source": f"report:{section}",
                },
            }
        )
    for idx, text in enumerate(model_results.get("anomaly_explanations", [])):
        chunks.append(
            {
                "text": text,
                "metadata": {
                    "session_id": session_id,
                    "chunk_type": "anomaly_trade",
                    "time_period": "session",
                    "symbol": "ALL",
                    "confidence": 0.75,
                    "source": f"lime:anomaly:{idx}",
                },
            }
        )
    for idx, c in enumerate(model_results.get("behavioral_biases", {}).items()):
        chunks.append(
            {
                "text": f"Bias metric {c[0]} = {c[1]}",
                "metadata": {
                    "session_id": session_id,
                    "chunk_type": "bias",
                    "time_period": "session",
                    "symbol": "ALL",
                    "confidence": 0.85,
                    "source": f"model:behavioral_biases:{idx}",
                },
            }
        )
    for idx, entry in enumerate(model_results.get("model_effectiveness", {}).items()):
        chunks.append(
            {
                "text": f"Model effectiveness {entry[0]} = {entry[1]}",
                "metadata": {
                    "session_id": session_id,
                    "chunk_type": "model_effectiveness",
                    "time_period": "session",
                    "symbol": "ALL",
                    "confidence": 0.9,
                    "source": f"model_effectiveness:{idx}",
                },
            }
        )
    return chunks


def _analysis_pipeline(session_id: str) -> None:
    try:
        _set_progress(session_id, "Fetching market data...", 20)
        repo.set_status(session_id, "market_context")
        trades = repo.get_modeling_trades(session_id)
        trades_df = pd.DataFrame(
            [
                {
                    "id": t.id,
                    "timestamp": t.timestamp,
                    "symbol": t.symbol,
                    "buy_sell": t.side,
                    "quantity": t.quantity,
                    "price": t.price,
                    "pnl": t.pnl,
                    "holding_duration": t.holding_duration,
                }
                for t in trades
            ]
        )
        market_payload = build_market_context_for_trades(trades_df)
        out = session_output_dir(session_id)

        _set_progress(session_id, "Running behavioral models...", 50)
        repo.set_status(session_id, "modeling")
        features_df = build_behavioral_features(trades_df, market_payload)
        baselines = compute_baselines(features_df)
        context_df = _build_context_df(market_payload, target_length=len(features_df))

        sentiment_result = sentiment.run(
            features_df,
            context_df,
            {
                "news_items": [
                    {"symbol": m["symbol"], "date": str(m["date"]), "headline": h}
                    for m in market_payload
                    for h in m.get("context", {}).get("news_sentiment_context", {}).get("headlines", [])
                    if h
                ]
            },
        )
        _inject_sentiment_into_context(market_payload, sentiment_result.get("daily_symbol_sentiment", {}))
        context_df = _build_context_df(market_payload, target_length=len(features_df))

        repo.save_market_context(session_id, market_payload)
        (out / "market_context.json").write_text(json.dumps(market_payload, default=str, indent=2), encoding="utf-8")

        model_results = {
            "baselines": baselines,
            "clustering": clustering.run(features_df, context_df, {}),
            "hmm_model": hmm_model.run(features_df, context_df, {}),
            "anomaly": anomaly.run(features_df, context_df, {}),
            "sentiment": sentiment_result,
            "bayesian_network": bayesian_network.run(features_df, context_df, {}),
            "lstm_model": lstm_model.run(features_df, context_df, {}),
            "risk_distribution": risk_distribution.run(features_df, context_df, {}),
            "causality": causality.run(features_df, context_df, {}),
            "garch_model": garch_model.run(features_df, context_df, {}),
            "behavioral_biases": behavioral_biases.run(features_df, context_df, {}),
            "emotional_state": emotional_state.run(features_df, context_df, {}),
            "tft_model": tft_model.run(features_df, context_df, {"session_id": session_id}),
        }
        model_results["model_effectiveness"] = _build_effectiveness_panel(
            features_df=features_df,
            model_results=model_results,
        )
        for name, result in model_results.items():
            repo.save_analysis_result(session_id, name, result)

        _set_progress(session_id, "Generating insights...", 80)
        anomaly_scores = model_results["anomaly"].get("anomaly_confidence", [0.0] * len(features_df))
        shap_data = compute_shap_bundle(
            features_df=features_df,
            clustering_labels=model_results.get("clustering", {}).get("gmm_labels", []),
            anomaly_scores=anomaly_scores,
            lstm_prediction_error=model_results.get("lstm_model", {}).get("prediction_error", []),
        )
        model_results["shap"] = shap_data
        anomaly_indices = [i for i, f in enumerate(model_results["anomaly"].get("anomaly_flag", [])) if f == 1][:10]
        model_results["anomaly_explanations"] = [
            explain_anomalous_trade(
                features_df,
                i,
                model_results.get("anomaly", {}).get("anomaly_flag", []),
            )
            for i in anomaly_indices
        ]
        report = generate_behavioral_report(session_id, model_results)
        chart_paths = generate_all_charts(session_id, trades_df, features_df, model_results)
        chunks = _build_rag_chunks(session_id, report.get("report", {}), model_results)
        chunks += build_narrative_chunks(session_id, model_results)
        chunks += build_precomputed_counterfactual_chunks(session_id)
        build_session_index(session_id, chunks)
        (out / "analysis_results.json").write_text(
            json.dumps({"model_results": model_results, "report": report, "charts": chart_paths}, default=str, indent=2),
            encoding="utf-8",
        )
        repo.set_status(session_id, "complete")
        _set_progress(session_id, "Completed", 100)
    except Exception as exc:
        repo.set_status(session_id, "failed")
        _set_progress(session_id, f"failed: {exc}", 100)


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    settings = get_settings()
    session_id = uuid.uuid4().hex
    target = settings.uploads_dir / f"{session_id}_{file.filename}"
    content = await file.read()
    target.write_bytes(content)
    repo.create_session(session_id=session_id, filename=file.filename or target.name)
    _set_progress(session_id, "Extracting trades...", 10)
    normalized = _parse_upload(target)
    clean_df, open_positions = reconstruct_positions_fifo(normalized)
    records = clean_df.to_dict(orient="records")
    repo.insert_trades(session_id, records)
    output_dir = session_output_dir(session_id)
    (output_dir / "trades.json").write_text(json.dumps(records, default=str, indent=2), encoding="utf-8")
    repo.set_status(session_id, "awaiting_emergency_flags")
    _set_progress(session_id, "Awaiting emergency trade review...", 15)
    return {
        "session_id": session_id,
        "trade_preview": trade_preview(clean_df, limit=50),
        "open_positions": open_positions,
    }


@app.post("/api/emergency/{session_id}")
async def set_emergency_flags(session_id: str, payload: dict, background_tasks: BackgroundTasks):
    trade_ids = payload.get("trade_ids", [])
    reason = payload.get("reason", "financial_emergency")
    repo.mark_emergency_trades(session_id, trade_ids, reason)
    repo.set_status(session_id, "analysis_started")
    background_tasks.add_task(_analysis_pipeline, session_id)
    return {"session_id": session_id, "flagged_trade_count": len(trade_ids), "analysis_started": True}


@app.get("/api/progress/{session_id}")
async def get_progress_stream(session_id: str):
    async def event_stream():
        for _ in range(600):
            item = progress_store.get(session_id, {"stage": "waiting", "progress": 0})
            yield f"data: {json.dumps(item)}\n\n"
            if item.get("progress", 0) >= 100:
                break
            await asyncio.sleep(1)
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/analysis/{session_id}")
async def get_analysis(session_id: str):
    out = session_output_dir(session_id) / "analysis_results.json"
    if not out.exists():
        return {"session_id": session_id, "status": "pending", "results": []}
    return json.loads(out.read_text(encoding="utf-8"))


@app.get("/api/charts/{session_id}")
async def get_charts(session_id: str):
    chart_dir = session_output_dir(session_id) / "charts"
    chart_paths = [str(p) for p in chart_dir.glob("*.html")]
    return {"session_id": session_id, "charts": sorted(chart_paths)}


@app.get("/api/chart-file/{session_id}/{filename}")
async def get_chart_file(session_id: str, filename: str):
    from fastapi.responses import HTMLResponse
    chart_dir = session_output_dir(session_id) / "charts"
    chart_path = chart_dir / filename
    if not chart_path.exists() or chart_path.suffix != ".html":
        raise HTTPException(status_code=404, detail="Chart not found")
    return HTMLResponse(content=chart_path.read_text(encoding="utf-8"))


@app.get("/api/report/{session_id}")
async def get_report(session_id: str):
    out = session_output_dir(session_id)
    json_path = out / "behavioral_report.json"
    txt_path = out / "behavioral_report.txt"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="report not ready")
    return {
        "session_id": session_id,
        "report_json": json.loads(json_path.read_text(encoding="utf-8")),
        "report_text": txt_path.read_text(encoding="utf-8") if txt_path.exists() else "",
    }


@app.post("/api/chat/{session_id}")
async def chat(session_id: str, payload: dict):
    user_message = payload.get("message", "")
    if not user_message:
        raise HTTPException(status_code=400, detail="message required")
    graph = build_graph()
    recent = get_recent(session_id, limit=10)
    state = {
        "session_id": session_id,
        "user_message": user_message,
        "intent": "general_explanation",
        "retrieved_chunks": [],
        "conversation_history": recent,
        "behavioral_profile": {},
        "agent_outputs": {},
        "final_response": "",
        "sources": [],
        "confidence": "low",
    }
    save_message(session_id, "user", user_message, "unknown", [])
    result = graph.invoke(state)
    response = result.get("final_response", "No response.")
    intent = result.get("intent", "general_explanation")
    sources = result.get("sources", [])
    confidence = result.get("confidence", "low")
    chunk_refs = [
        s.get("source", "analysis_chunk")
        for s in sources if s.get("type") == "chunk"
    ] if sources else []
    save_message(session_id, "assistant", response, intent, chunk_refs)
    return {
        "session_id": session_id,
        "intent": intent,
        "response": response,
        "sources": sources,
        "confidence": confidence,
    }


@app.get("/api/chat/history/{session_id}")
async def chat_history(session_id: str):
    return {"session_id": session_id, "history": get_recent(session_id, limit=200)}


@app.post("/api/counterfactual/{session_id}")
async def counterfactual(session_id: str, payload: dict):
    result = run_counterfactual(session_id, payload)
    return {"session_id": session_id, "result": result}


@app.get("/api/market/{symbol}")
async def market(symbol: str):
    end = datetime.utcnow().date()
    start = end - timedelta(days=365)
    df = fetch_ohlcv(symbol, start.isoformat(), end.isoformat(), interval="1d")
    if df.empty:
        raise HTTPException(status_code=404, detail="no market data found")
    return {"symbol": symbol, "rows": len(df), "latest": df.tail(1).to_dict(orient="records")[0]}


@app.get("/api/health")
async def health():
    import httpx
    ollama_ok = False
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get("http://localhost:11434/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False
    return {"status": "ok", "ollama_reachable": ollama_ok, "static_knowledge_indexed": _STATIC_INDEX_READY}