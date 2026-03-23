from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from backend.config import get_settings
from backend.chat.behavior_agent import run_behavior_agent
from backend.chat.counterfactual_agent import run_counterfactual_agent
from backend.chat.llm_agent import run_llm_agent
from backend.chat.market_agent import detect_ticker, run_market_agent
from backend.chat.risk_agent import run_risk_agent
from backend.chat.strategy_agent import is_strategy_query, run_strategy_agent


class ChatState(TypedDict):
    session_id: str
    user_message: str
    intent: str
    retrieved_chunks: list
    agent_outputs: dict
    conversation_history: list
    behavioral_profile: dict
    final_response: str
    sources: list
    confidence: str


def intent_node(state: ChatState) -> ChatState:
    state.setdefault("agent_outputs", {})
    text = (state.get("user_message") or "").lower()
    if any(k in text for k in ("bias", "discipline", "behavior")):
        intent = "bias_query"
    elif any(k in text for k in ("what if", "counterfactual", "if i had", "what would")):
        intent = "whatif_counterfactual"
    elif any(k in text for k in ("risk", "drawdown", "tail", "stress")):
        if any(k in text for k in ("stress test", "stress scenario")):
            intent = "stress_test"
        else:
            intent = "risk_profile_query"
    elif any(k in text for k in ("recommend", "should i", "stocks match", "look at")):
        intent = "recommendation_request"
    elif any(k in text for k in ("why", "underperform", "performance", "lost money")):
        intent = "performance_explanation"
    else:
        intent = "general_explanation"
    state["intent"] = intent
    return state


def behavior_node(state: ChatState) -> ChatState:
    return run_behavior_agent(state)

def risk_node(state: ChatState) -> ChatState:
    return run_risk_agent(state)

def counterfactual_node(state: ChatState) -> ChatState:
    return run_counterfactual_agent(state)

def market_node(state: ChatState) -> ChatState:
    return run_market_agent(state)

def strategy_node(state: ChatState) -> ChatState:
    return run_strategy_agent(state)


def retrieval_node(state: ChatState) -> ChatState:
    settings = get_settings()
    persist_dir = str(settings.project_root / "data" / "cache" / "chroma")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    session_db = Chroma(
        collection_name="behavioral_analysis",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    behavior_ctx = state.get("agent_outputs", {}).get("behavior", {})
    hints = behavior_ctx.get("retrieval_hints", {}) if isinstance(behavior_ctx, dict) else {}
    top_features = hints.get("top_features", [])

    prefix = ""
    if top_features:
        prefix = "Behavior drivers: " + ", ".join(top_features[:4]) + ". "
    query_text = prefix + state.get("user_message", "")

    intent = state.get("intent", "general_explanation")
    session_id = state["session_id"]

    chunk_type = None
    if intent == "bias_query":
        chunk_type = "bias"
    elif intent == "risk_profile_query":
        chunk_type = "risk_narrative"
    elif intent == "whatif_counterfactual":
        chunk_type = "counterfactual"

    if chunk_type:
        user_filter = {
            "$and": [
                {"session_id": {"$eq": session_id}},
                {"chunk_type": {"$eq": chunk_type}},
            ]
        }
    else:
        user_filter = {"session_id": {"$eq": session_id}}

    scored = session_db.similarity_search_with_relevance_scores(query_text, k=4, filter=user_filter)
    if not scored or max(s for _, s in scored) < 0.40:
        scored = session_db.similarity_search_with_relevance_scores(
            query_text, k=4, filter={"session_id": {"$eq": session_id}},
        )

    static_scored: list = []
    try:
        static_db = Chroma(
            collection_name="static_knowledge",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        static_scored = static_db.similarity_search_with_relevance_scores(query_text, k=2)
    except Exception:
        pass

    combined = list(scored) + list(static_scored)
    combined.sort(key=lambda pair: pair[1], reverse=True)
    combined = combined[:6]

    preferred = hints.get("preferred_chunk_types", [])
    if preferred:
        pref_items = [(d, s) for d, s in combined if d.metadata.get("chunk_type") in preferred]
        other_items = [(d, s) for d, s in combined if d.metadata.get("chunk_type") not in preferred]
        combined = (pref_items + other_items)[:6]

    docs = [
        {
            "text": doc.page_content,
            "metadata": {**doc.metadata, "relevance": float(score)},
        }
        for doc, score in combined
    ]
    state["retrieved_chunks"] = docs
    return state


def llm_node(state: ChatState) -> ChatState:
    return run_llm_agent(state)


def _route_after_intent(state: ChatState) -> str:
    intent = state.get("intent", "general_explanation")
    if intent in ("bias_query", "performance_explanation"):
        return "behavior_node"
    if intent in ("risk_profile_query", "stress_test"):
        return "risk_node"
    if intent == "whatif_counterfactual":
        return "counterfactual_node"
    if intent == "recommendation_request":
        return "market_node"
    if intent == "general_explanation" and is_strategy_query(state.get("user_message", "")):
        return "strategy_node"
    return "retrieval_node"


def _route_after_market(state: ChatState) -> str:
    return "strategy_node"

def _route_after_strategy(state: ChatState) -> str:
    return "retrieval_node"


def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("intent_node", intent_node)
    graph.add_node("behavior_node", behavior_node)
    graph.add_node("risk_node", risk_node)
    graph.add_node("counterfactual_node", counterfactual_node)
    graph.add_node("market_node", market_node)
    graph.add_node("strategy_node", strategy_node)
    graph.add_node("retrieval_node", retrieval_node)
    graph.add_node("llm_node", llm_node)
    graph.set_entry_point("intent_node")
    graph.add_conditional_edges("intent_node", _route_after_intent)
    graph.add_edge("behavior_node", "retrieval_node")
    graph.add_edge("risk_node", "retrieval_node")
    graph.add_edge("counterfactual_node", "retrieval_node")
    graph.add_conditional_edges("market_node", _route_after_market)
    graph.add_conditional_edges("strategy_node", _route_after_strategy)
    graph.add_edge("retrieval_node", "llm_node")
    graph.add_edge("llm_node", END)
    return graph.compile()


def test_graph() -> dict:
    return {"ok": True, "message": "langgraph graph wired with Groq + HuggingFace embeddings"}