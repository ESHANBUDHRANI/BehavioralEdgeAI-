from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any

from backend.database.repository import Repository
from backend.market_context.data_provider import fetch_ohlcv
from backend.market_context.indicators import compute_indicator_frame
from backend.market_context.regime import (
    classify_market_regime,
    classify_trend,
    classify_volatility,
)
from backend.explainability.nlg import market_compatibility_template

logger = logging.getLogger(__name__)

_TICKER_RE = re.compile(r"\b[A-Z]{2,5}(?:\.NS|\.BSE)?\b")

_NSE_NAME_MAP: dict[str, str] = {
    "reliance": "RELIANCE.NS", "tcs": "TCS.NS", "infosys": "INFY.NS",
    "infy": "INFY.NS", "hdfc": "HDFCBANK.NS", "hdfcbank": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS", "icicibank": "ICICIBANK.NS", "sbi": "SBIN.NS",
    "sbin": "SBIN.NS", "kotak": "KOTAKBANK.NS", "bharti": "BHARTIARTL.NS",
    "airtel": "BHARTIARTL.NS", "itc": "ITC.NS", "hindunilvr": "HINDUNILVR.NS",
    "hul": "HINDUNILVR.NS", "bajfinance": "BAJFINANCE.NS", "bajaj finance": "BAJFINANCE.NS",
    "maruti": "MARUTI.NS", "asian paints": "ASIANPAINT.NS", "asianpaint": "ASIANPAINT.NS",
    "wipro": "WIPRO.NS", "hcltech": "HCLTECH.NS", "hcl": "HCLTECH.NS",
    "titan": "TITAN.NS", "sunpharma": "SUNPHARMA.NS", "sun pharma": "SUNPHARMA.NS",
    "ultracemco": "ULTRACEMCO.NS", "ultratech": "ULTRACEMCO.NS",
    "nestleindia": "NESTLEIND.NS", "nestle": "NESTLEIND.NS",
    "adani": "ADANIENT.NS", "adanient": "ADANIENT.NS",
    "adaniports": "ADANIPORTS.NS", "powergrid": "POWERGRID.NS",
    "ntpc": "NTPC.NS", "ongc": "ONGC.NS", "tatamotors": "TATAMOTORS.NS",
    "tata motors": "TATAMOTORS.NS", "tatasteel": "TATASTEEL.NS",
    "tata steel": "TATASTEEL.NS", "jswsteel": "JSWSTEEL.NS",
    "jsw steel": "JSWSTEEL.NS", "m&m": "M&M.NS", "mahindra": "M&M.NS",
    "techm": "TECHM.NS", "tech mahindra": "TECHM.NS",
    "hindalco": "HINDALCO.NS", "grasim": "GRASIM.NS",
    "drlreddy": "DRREDDY.NS", "dr reddy": "DRREDDY.NS", "drreddy": "DRREDDY.NS",
    "cipla": "CIPLA.NS", "divislab": "DIVISLAB.NS", "divis lab": "DIVISLAB.NS",
    "eichermot": "EICHERMOT.NS", "eicher": "EICHERMOT.NS",
    "heromotoco": "HEROMOTOCO.NS", "hero": "HEROMOTOCO.NS",
    "apollohosp": "APOLLOHOSP.NS", "apollo": "APOLLOHOSP.NS",
    "britannia": "BRITANNIA.NS", "coalindia": "COALINDIA.NS",
    "coal india": "COALINDIA.NS", "upl": "UPL.NS",
    "indusindbk": "INDUSINDBK.NS", "indusind": "INDUSINDBK.NS",
    "bpcl": "BPCL.NS", "tataconsum": "TATACONSUM.NS",
    "ltim": "LTIM.NS", "lt": "LT.NS", "larsen": "LT.NS",
    "axisbank": "AXISBANK.NS", "axis bank": "AXISBANK.NS",
}

_STOPWORDS = {
    "what", "which", "stock", "stocks", "match", "style", "risk", "best",
    "with", "for", "your", "my", "and", "the", "should", "look", "buy",
    "sell", "hold", "trade", "about", "how", "does", "this", "that",
    "can", "will", "would", "have", "been", "from", "into",
}


def detect_ticker(user_message: str) -> str | None:
    """Extract a stock ticker from user message via name map or regex."""
    text = (user_message or "").lower()
    for name, ticker in _NSE_NAME_MAP.items():
        if name in text:
            return ticker

    candidates = []
    for token in _TICKER_RE.findall(user_message or ""):
        t = token.upper()
        if t.lower() in _STOPWORDS:
            continue
        candidates.append(t)
    if not candidates:
        return None

    end = datetime.utcnow().date()
    start = end - timedelta(days=60)
    for ticker in candidates:
        for suffix in ["", ".NS", ".BSE"]:
            full = ticker + suffix if not ticker.endswith((".NS", ".BSE")) else ticker
            try:
                df = fetch_ohlcv(full, start.isoformat(), end.isoformat(), interval="1d")
                if not df.empty:
                    return full
            except Exception:  # noqa: BLE001
                continue
    return None


def _regime_win_rates(session_id: str) -> dict[str, float]:
    """Compute win rate per market regime for the session."""
    repo = Repository()
    trades = repo.get_modeling_trades(session_id)
    contexts = repo.get_market_context(session_id)
    ctx_map: dict[tuple[str, str], str] = {}
    for ctx in contexts:
        try:
            payload = json.loads(ctx.context_json)
            regime = payload.get("market_regime_context", {}).get("label", "unknown")
        except Exception:  # noqa: BLE001
            regime = "unknown"
        ctx_map[(str(ctx.date), ctx.symbol.upper())] = regime
    wins: dict[str, int] = {}
    totals: dict[str, int] = {}
    for t in trades:
        key = (str(t.timestamp.date()), t.symbol.upper())
        regime = ctx_map.get(key, "unknown")
        totals[regime] = totals.get(regime, 0) + 1
        if float(t.pnl) > 0:
            wins[regime] = wins.get(regime, 0) + 1
    return {r: wins.get(r, 0) / max(totals[r], 1) for r in totals}


# ---------------------------------------------------------------------------
# 2A — Agent confidence calibration
# ---------------------------------------------------------------------------

def compute_agent_confidence(session_id: str) -> float:
    """Score confidence 0.0-1.0 based on data quality signals for market agent."""
    conf = 1.0
    repo = Repository()
    trades = repo.get_modeling_trades(session_id)
    all_trades = repo.get_trades(session_id)
    if len(trades) < 50:
        conf -= 0.2
    analysis_rows = repo.get_analysis_results(session_id)
    has_garch = any(r.model_name == "garch_model" for r in analysis_rows)
    if not has_garch:
        conf -= 0.15
    for r in analysis_rows:
        if r.model_name == "garch_model":
            try:
                data = json.loads(r.result_json)
                if data.get("insufficient_data"):
                    conf -= 0.1
            except Exception:  # noqa: BLE001
                pass
            break
    if len(all_trades) > 0 and len(all_trades) - len(trades) > 0.3 * len(all_trades):
        conf -= 0.1
    return max(0.1, round(conf, 2))


# ---------------------------------------------------------------------------
# 2E — RSI bucket win rates
# ---------------------------------------------------------------------------

def _rsi_bucket_win_rates(session_id: str) -> tuple[dict[str, float], str, str]:
    """Compute user win rate bucketed by RSI at time of trade.

    Returns (bucket_rates, best_bucket, worst_bucket).
    """
    repo = Repository()
    trades = repo.get_modeling_trades(session_id)
    contexts = repo.get_market_context(session_id)
    rsi_map: dict[tuple[str, str], float] = {}
    for ctx in contexts:
        try:
            payload = json.loads(ctx.context_json)
            rsi_val = payload.get("momentum_context", {}).get("rsi14")
            if rsi_val is not None:
                rsi_map[(str(ctx.date), ctx.symbol.upper())] = float(rsi_val)
        except Exception:  # noqa: BLE001
            pass

    buckets: dict[str, list[bool]] = {"oversold": [], "neutral": [], "overbought": []}
    for t in trades:
        key = (str(t.timestamp.date()), t.symbol.upper())
        rsi = rsi_map.get(key)
        if rsi is None:
            continue
        if rsi < 35:
            buckets["oversold"].append(float(t.pnl) > 0)
        elif rsi > 65:
            buckets["overbought"].append(float(t.pnl) > 0)
        else:
            buckets["neutral"].append(float(t.pnl) > 0)

    rates: dict[str, float] = {}
    for bucket, results in buckets.items():
        rates[bucket] = float(sum(results) / max(len(results), 1)) if results else 0.0

    best = max(rates, key=rates.get) if rates else "neutral"
    worst = min(rates, key=rates.get) if rates else "neutral"
    return rates, best, worst


def run_market_agent(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: detect ticker, fetch market data, compute compatibility."""
    ticker = detect_ticker(state.get("user_message", ""))
    state.setdefault("agent_outputs", {})
    session_id = state["session_id"]

    if not ticker:
        state["agent_outputs"]["market"] = {"ticker": None, "skipped": True}
        return state

    end = datetime.utcnow().date()
    start = end - timedelta(days=420)
    ohlcv = fetch_ohlcv(ticker, start.isoformat(), end.isoformat(), interval="1d")
    if ohlcv.empty:
        state["agent_outputs"]["market"] = {"ticker": ticker, "skipped": True, "reason": "no OHLCV data"}
        return state

    ind = compute_indicator_frame(ohlcv)
    latest = ind.iloc[-1]

    vix_df = fetch_ohlcv("^VIX", start.isoformat(), end.isoformat(), interval="1d")
    vix_val = 20.0
    if not vix_df.empty:
        for col in ["Close", "close", "Adj Close"]:
            if col in vix_df.columns:
                vix_val = float(vix_df.iloc[-1][col])
                break

    current_regime = classify_market_regime(latest, vix_value=vix_val)
    current_trend = classify_trend(latest)
    current_volatility = classify_volatility(latest)
    current_rsi = float(latest.get("rsi14", 50.0))

    risk_out = state.get("agent_outputs", {}).get("risk", {})
    if risk_out and risk_out.get("best_regime"):
        best_regime = risk_out["best_regime"]
        worst_regime = risk_out["worst_regime"]
        regime_rates = risk_out.get("regime_win_rates", {})
        stress_coupling = risk_out.get("stress_coupling") or 0.0
    else:
        regime_rates = _regime_win_rates(session_id)
        best_regime = max(regime_rates, key=regime_rates.get) if regime_rates else "unknown"
        worst_regime = min(regime_rates, key=regime_rates.get) if regime_rates else "unknown"
        analysis = {}
        try:
            repo = Repository()
            for row in repo.get_analysis_results(session_id):
                if row.model_name == "garch_model":
                    analysis = json.loads(row.result_json)
                    break
        except Exception:  # noqa: BLE001
            pass
        stress_coupling = float(analysis.get("stress_coupling_score", 0.0))

    # Base compatibility score
    score = 50.0
    if current_regime == best_regime:
        score += 35.0
    if current_regime == worst_regime:
        score -= 35.0
    if current_volatility == "high_expansion" and stress_coupling > 0.6:
        score -= 20.0
    if current_volatility == "low_compression" and stress_coupling < 0.3:
        score += 15.0

    # RSI bucket adjustment
    rsi_bucket_rates, best_rsi_bucket, worst_rsi_bucket = _rsi_bucket_win_rates(session_id)
    if current_rsi < 35:
        current_rsi_bucket = "oversold"
    elif current_rsi > 65:
        current_rsi_bucket = "overbought"
    else:
        current_rsi_bucket = "neutral"

    if current_rsi_bucket == best_rsi_bucket:
        score += 10.0
    if current_rsi_bucket == worst_rsi_bucket:
        score -= 10.0

    score = max(0.0, min(100.0, score))

    rsi_compat_note = (
        f"Current RSI ({current_rsi:.1f}) places this stock in the '{current_rsi_bucket}' zone. "
        f"Your historical win rates by RSI bucket: "
        + ", ".join(f"{k}: {v:.0%}" for k, v in rsi_bucket_rates.items())
        + "."
    )

    best_win_rate = regime_rates.get(best_regime, 0.0)
    match_or_conflict = "match" if score >= 50 else "conflict with"
    reasoning = market_compatibility_template(
        ticker=ticker,
        current_regime=current_regime,
        current_volatility=current_volatility,
        best_regime=best_regime,
        best_win_rate=best_win_rate,
        match_or_conflict=match_or_conflict,
    )

    agent_confidence = compute_agent_confidence(session_id)

    state["agent_outputs"]["market"] = {
        "ticker": ticker,
        "current_regime": current_regime,
        "current_trend": current_trend,
        "current_volatility": current_volatility,
        "current_rsi": current_rsi,
        "compatibility_score": score,
        "compatibility_reasoning": reasoning,
        "rsi_bucket_win_rates": rsi_bucket_rates,
        "current_rsi_bucket": current_rsi_bucket,
        "rsi_compatibility_note": rsi_compat_note,
        "agent_confidence": agent_confidence,
        "source_models": ["yfinance", "MarketRegimeClassifier"],
    }
    state["market_context"] = state["agent_outputs"]["market"]
    return state


def test_market_agent() -> dict:
    """Smoke test for market agent module."""
    return {"ok": True, "message": "market agent compatibility logic wired"}
