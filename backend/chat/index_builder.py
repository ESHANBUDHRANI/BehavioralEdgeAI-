from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from backend.config import get_settings, session_output_dir


# ── Shared embeddings factory ─────────────────────────────────────────────
# CHANGED: using HuggingFace sentence-transformers — 100% free, runs locally
# Downloads ~90MB model on first run, then cached forever

def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


# ── Session index ─────────────────────────────────────────────────────────

def build_session_index(session_id: str, chunks: list[dict]) -> dict:
    settings = get_settings()
    embeddings = _get_embeddings()
    persist_dir = settings.project_root / "data" / "cache" / "chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)
    texts = [c["text"] for c in chunks]
    metadatas = [
        (
            {
                "chunk_type": "report_section",
                "time_period": "session",
                "symbol": "ALL",
                "confidence": 0.5,
                "source": "unknown",
            }
            | c.get("metadata", {})
            | {"session_id": session_id}
        )
        for c in chunks
    ]
    ids = [f"{session_id}_{i}" for i in range(len(chunks))]
    vectordb = Chroma(
        collection_name="behavioral_analysis",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    if texts:
        vectordb.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return {"indexed_chunks": len(chunks), "persist_directory": str(persist_dir)}


def build_narrative_chunks(session_id: str, model_results: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate per-cluster, HMM, risk, and strategy narrative chunks."""
    chunks: list[dict[str, Any]] = []

    # 1. Per-cluster narrative
    clustering = model_results.get("clustering", {})
    cluster_labels = clustering.get("cluster_plain_labels", [])
    gmm_labels = clustering.get("gmm_labels", [])
    for idx, label in enumerate(cluster_labels):
        trade_count = gmm_labels.count(idx) if gmm_labels else 0
        text = (
            f"Cluster {idx} ('{label}'): contains {trade_count} trades. "
            f"This cluster is characterized by the dominant features "
            f"identified in the GMM center analysis."
        )
        chunks.append({
            "text": text,
            "metadata": {
                "chunk_type": "cluster_narrative",
                "cluster_id": idx,
                "session_id": session_id,
                "confidence": 0.8,
                "source": f"narrative:cluster:{idx}",
            },
        })

    # 2. HMM transition narrative
    hmm = model_results.get("hmm_model", {})
    transition_story = hmm.get("transition_story", [])
    state_labels = hmm.get("state_labels", [])
    user_seq = hmm.get("user_state_sequence", [])
    if transition_story:
        story_text = " ".join(str(s) for s in transition_story[:5])
    elif user_seq and state_labels:
        from collections import Counter
        transitions: list[str] = []
        for i in range(1, len(user_seq)):
            prev = user_seq[i - 1]
            cur = user_seq[i]
            prev_l = state_labels[prev] if isinstance(prev, int) and prev < len(state_labels) else str(prev)
            cur_l = state_labels[cur] if isinstance(cur, int) and cur < len(state_labels) else str(cur)
            transitions.append(f"{prev_l}->{cur_l}")
        most_common = Counter(transitions).most_common(3)
        story_text = "Most common state transitions: " + ", ".join(
            f"{t} ({c} times)" for t, c in most_common
        )
    else:
        story_text = "HMM transition data not available."
    chunks.append({
        "text": f"HMM behavioral state transitions: {story_text}",
        "metadata": {
            "chunk_type": "hmm_narrative",
            "session_id": session_id,
            "confidence": 0.75,
            "source": "narrative:hmm_transitions",
        },
    })

    # 3. Risk profile narrative
    risk_dist = model_results.get("risk_distribution", {})
    garch = model_results.get("garch_model", {})
    var95 = risk_dist.get("var95", "N/A")
    cvar95 = risk_dist.get("cvar95", "N/A")
    stress = garch.get("stress_coupling_score", "N/A")
    tail_dep = risk_dist.get("tail_dependency_coefficient", "N/A")
    risk_text = (
        f"Risk profile summary: VaR 95% = {var95}, CVaR 95% = {cvar95}, "
        f"stress coupling = {stress}, tail dependency = {tail_dep}."
    )
    chunks.append({
        "text": risk_text,
        "metadata": {
            "chunk_type": "risk_narrative",
            "session_id": session_id,
            "confidence": 0.85,
            "source": "narrative:risk_profile",
        },
    })

    # 4. Strategy profile narrative
    try:
        from backend.chat.strategy_agent import build_strategy_context
        strategy = build_strategy_context(session_id)
        strategy_text = (
            f"Trading style: {strategy.get('trading_style', 'unknown')}. "
            f"{strategy.get('style_description', '')} "
            f"Ideal regime: {strategy.get('ideal_regime', 'unknown')}. "
            f"Warnings: {'; '.join(strategy.get('risk_warnings', []))}. "
            f"Suggestions: {'; '.join(strategy.get('improvement_suggestions', []))}."
        )
    except Exception:  # noqa: BLE001
        strategy_text = "Strategy profile could not be computed at index time."
    chunks.append({
        "text": strategy_text,
        "metadata": {
            "chunk_type": "strategy_narrative",
            "session_id": session_id,
            "confidence": 0.8,
            "source": "narrative:strategy_profile",
        },
    })

    return chunks


def build_precomputed_counterfactual_chunks(session_id: str) -> list[dict[str, Any]]:
    """Compute 5 standard counterfactual scenarios and return as indexable chunks."""
    from backend.chat.counterfactual import compute_live_counterfactual

    scenarios = [
        {"variable": "position_size", "multiplier": 0.5, "label": "Half position size after every loss"},
        {"variable": "volatility_filter", "value": "high_volatility", "label": "Exclude high-volatility trades"},
        {"variable": "holding_duration", "multiplier": 1.5, "label": "Hold 50% longer"},
        {"variable": "emergency_filter", "include": True, "label": "Include emergency trades"},
        {"variable": "post_loss_skip", "n": 1, "label": "Skip one trade after every loss"},
    ]

    chunks: list[dict[str, Any]] = []
    for scenario in scenarios:
        try:
            result = compute_live_counterfactual(session_id, scenario)
            text = json.dumps({
                "scenario": scenario.get("label", scenario["variable"]),
                "original_metrics": result.get("original_metrics", {}),
                "counterfactual_metrics": result.get("counterfactual_metrics", {}),
                "delta_pnl": result.get("delta_pnl", 0.0),
                "delta_win_rate": result.get("delta_win_rate", 0.0),
            }, default=str)
        except Exception as exc:  # noqa: BLE001
            text = json.dumps({"scenario": scenario.get("label", ""), "error": str(exc)})
        chunks.append({
            "text": text,
            "metadata": {
                "chunk_type": "counterfactual",
                "scenario": scenario.get("label", scenario["variable"]),
                "session_id": session_id,
                "confidence": 0.8,
                "source": f"precomputed_counterfactual:{scenario['variable']}",
            },
        })
    return chunks


# ── Static knowledge index ────────────────────────────────────────────────

def _chunk_text(text: str, max_tokens: int = 500, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        start += max_tokens - overlap
    return chunks


def build_static_knowledge_index() -> dict:
    settings = get_settings()
    persist_dir = settings.project_root / "data" / "cache" / "chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = _get_embeddings()
    try:
        db = Chroma(
            collection_name="static_knowledge",
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        existing = db.get()
        if existing and existing.get("ids") and len(existing["ids"]) > 0:
            return {"status": "already_indexed", "chunk_count": len(existing["ids"])}
    except Exception:  # noqa: BLE001
        pass

    docs: list[dict[str, Any]] = []
    knowledge_dir = settings.static_knowledge_dir
    if not knowledge_dir.exists():
        return {"status": "no_static_knowledge_dir", "chunk_count": 0}

    for path in sorted(knowledge_dir.glob("**/*")):
        if not path.is_file() or path.suffix.lower() not in (".txt", ".md"):
            continue
        raw = path.read_text(encoding="utf-8")
        text_chunks = _chunk_text(raw, max_tokens=500, overlap=100)
        for i, chunk in enumerate(text_chunks):
            docs.append({
                "text": chunk,
                "metadata": {
                    "source": path.name,
                    "chunk_type": "static_knowledge",
                    "time_period": "evergreen",
                    "symbol": "ALL",
                    "confidence": 0.8,
                    "chunk_index": i,
                },
            })

    if not docs:
        return {"status": "no_documents", "chunk_count": 0}

    texts = [d["text"] for d in docs]
    metadatas = [d["metadata"] | {"session_id": "static_knowledge"} for d in docs]
    ids = [f"static_{i}" for i in range(len(docs))]

    db = Chroma(
        collection_name="static_knowledge",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return {"status": "indexed", "chunk_count": len(docs)}


def test_index_builder() -> dict:
    return {"ok": True, "message": "chroma index builder wired with HuggingFace embeddings"}
