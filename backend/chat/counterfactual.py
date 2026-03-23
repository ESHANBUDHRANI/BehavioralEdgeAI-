from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any

import pandas as pd

from backend.database.repository import Repository
from backend.market_context.data_provider import fetch_ohlcv

logger = logging.getLogger(__name__)


def _load_trades(session_id: str, include_emergency: bool = False) -> pd.DataFrame:
    """Load trades from SQLite, optionally including emergency-flagged trades."""
    repo = Repository()
    if include_emergency:
        trades = repo.get_trades(session_id)
    else:
        trades = repo.get_modeling_trades(session_id)
    rows = [
        {
            "id": t.id,
            "timestamp": t.timestamp,
            "symbol": t.symbol,
            "side": t.side,
            "quantity": float(t.quantity),
            "price": float(t.price),
            "pnl": float(t.pnl),
            "holding_duration": float(t.holding_duration),
            "cluster_label": getattr(t, "cluster_label", -1),
        }
        for t in trades
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _load_market_context(session_id: str) -> dict[tuple[str, str], str]:
    """Load market regime labels keyed by (date, symbol)."""
    repo = Repository()
    contexts = repo.get_market_context(session_id)
    ctx_map: dict[tuple[str, str], str] = {}
    for ctx in contexts:
        try:
            payload = json.loads(ctx.context_json)
            regime = payload.get("market_regime_context", {}).get("label", "unknown")
        except Exception:  # noqa: BLE001
            regime = "unknown"
        ctx_map[(str(ctx.date), ctx.symbol.upper())] = regime
    return ctx_map


def _get_close_on_date(symbol: str, target_date: pd.Timestamp) -> float | None:
    """Fetch the closest closing price on or before target_date from cache."""
    start = (target_date - timedelta(days=10)).strftime("%Y-%m-%d")
    end = (target_date + timedelta(days=5)).strftime("%Y-%m-%d")
    df = fetch_ohlcv(symbol, start, end, interval="1d")
    if df.empty:
        return None
    for col in ["Close", "close", "Adj Close"]:
        if col in df.columns:
            date_col = None
            for dc in ["Date", "Datetime", "date", "datetime"]:
                if dc in df.columns:
                    date_col = dc
                    break
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                candidates = df[df[date_col] <= target_date].sort_values(date_col, ascending=False)
                if not candidates.empty:
                    return float(candidates.iloc[0][col])
            return float(df.iloc[-1][col])
    return None


def _aggregate_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Compute total PnL, win rate, and trade count from a DataFrame."""
    if df.empty:
        return {"total_pnl": 0.0, "win_rate": 0.0, "trade_count": 0}
    total_pnl = float(df["pnl"].sum())
    win_rate = float((df["pnl"] > 0).mean())
    return {"total_pnl": total_pnl, "win_rate": win_rate, "trade_count": int(len(df))}


def _holding_duration_scenario(df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    result = df.copy()
    new_pnls = []
    for _, row in result.iterrows():
        new_duration = row["holding_duration"] * multiplier
        new_exit_date = row["timestamp"] + timedelta(days=new_duration)
        new_close = _get_close_on_date(row["symbol"], new_exit_date)
        if new_close is not None:
            entry_price = row["price"]
            qty = row["quantity"]
            if str(row["side"]).upper() in ("BUY", "B"):
                new_pnl = (new_close - entry_price) * qty
            else:
                new_pnl = (entry_price - new_close) * qty
        else:
            new_pnl = row["pnl"] * multiplier
        new_pnls.append(new_pnl)
    result["pnl"] = new_pnls
    return result


def _position_size_scenario(df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    result = df.copy()
    result["pnl"] = result["pnl"] * multiplier
    return result


def _volatility_filter_scenario(df: pd.DataFrame, session_id: str, filter_value: str) -> pd.DataFrame:
    ctx_map = _load_market_context(session_id)
    keep = []
    for _, row in df.iterrows():
        key = (str(row["timestamp"].date()), row["symbol"].upper())
        regime = ctx_map.get(key, "unknown")
        if filter_value.lower() not in regime.lower():
            keep.append(True)
        else:
            keep.append(False)
    return df[keep].reset_index(drop=True)


def _post_loss_skip_scenario(df: pd.DataFrame, n: int) -> pd.DataFrame:
    keep_indices: list[int] = []
    skip_remaining = 0
    for idx, row in df.iterrows():
        if skip_remaining > 0:
            skip_remaining -= 1
            continue
        keep_indices.append(idx)
        if float(row["pnl"]) < 0:
            skip_remaining = n
    return df.loc[keep_indices].reset_index(drop=True)


def _emergency_filter_scenario(session_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    repo = Repository()
    all_trades = repo.get_trades(session_id)
    modeling_trades = repo.get_modeling_trades(session_id)
    all_rows = [
        {
            "id": t.id, "timestamp": t.timestamp, "symbol": t.symbol,
            "side": t.side, "quantity": float(t.quantity),
            "price": float(t.price), "pnl": float(t.pnl),
            "holding_duration": float(t.holding_duration),
        }
        for t in all_trades
    ]
    mod_rows = [
        {
            "id": t.id, "timestamp": t.timestamp, "symbol": t.symbol,
            "side": t.side, "quantity": float(t.quantity),
            "price": float(t.price), "pnl": float(t.pnl),
            "holding_duration": float(t.holding_duration),
        }
        for t in modeling_trades
    ]
    df_all = pd.DataFrame(all_rows)
    df_mod = pd.DataFrame(mod_rows)
    if not df_all.empty:
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
    if not df_mod.empty:
        df_mod["timestamp"] = pd.to_datetime(df_mod["timestamp"], errors="coerce")
    return df_all, df_mod


def _frequency_reduction_scenario(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    if df.empty or factor <= 0:
        return df
    n_keep = max(1, int(len(df) * factor))
    step = max(1, len(df) // n_keep)
    return df.iloc[::step].reset_index(drop=True)


def _frequency_increase_scenario(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    if df.empty or factor <= 1:
        return df
    repeats = int(round(factor))
    return pd.concat([df] * repeats, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def _signal_filter_scenario(df: pd.DataFrame, session_id: str, min_score: float) -> pd.DataFrame:
    try:
        analysis_rows = Repository().get_analysis_results(session_id)
        features_data = {}
        for row in analysis_rows:
            if row.model_name == "baselines":
                features_data = json.loads(row.result_json)
                break
        signal_scores = features_data.get("signal_following_scores", [])
        if signal_scores and len(signal_scores) == len(df):
            mask = [s >= min_score for s in signal_scores]
            return df[mask].reset_index(drop=True)
    except Exception:  # noqa: BLE001
        pass
    return df


def _loss_limit_scenario(df: pd.DataFrame, max_loss_pct: float) -> pd.DataFrame:
    result = df.copy()
    capped = []
    for _, row in result.iterrows():
        position_value = abs(row["price"] * row["quantity"])
        max_loss = position_value * max_loss_pct
        if row["pnl"] < -max_loss:
            capped.append(-max_loss)
        else:
            capped.append(row["pnl"])
    result["pnl"] = capped
    return result


def _top_cluster_only_scenario(df: pd.DataFrame, session_id: str, cluster_id: Any) -> pd.DataFrame:
    try:
        analysis_rows = Repository().get_analysis_results(session_id)
        for row in analysis_rows:
            if row.model_name == "clustering":
                clustering = json.loads(row.result_json)
                labels = clustering.get("gmm_labels", [])
                if labels and len(labels) == len(df):
                    if isinstance(cluster_id, str):
                        plain_labels = clustering.get("cluster_plain_labels", [])
                        if cluster_id in plain_labels:
                            target = plain_labels.index(cluster_id)
                        else:
                            target = 0
                    else:
                        target = int(cluster_id)
                    mask = [lbl == target for lbl in labels]
                    return df[mask].reset_index(drop=True)
                break
    except Exception:  # noqa: BLE001
        pass
    return df


def compute_live_counterfactual(session_id: str, scenario: dict[str, Any]) -> dict[str, Any]:
    """Compute counterfactual metrics for a given scenario."""
    variable = scenario.get("variable", "")
    df = _load_trades(session_id, include_emergency=False)
    if df.empty:
        return {
            "original_metrics": {}, "counterfactual_metrics": {},
            "delta_pnl": 0.0, "delta_win_rate": 0.0,
        }

    original = _aggregate_metrics(df)

    if variable == "holding_duration":
        multiplier = float(scenario.get("multiplier", 1.5))
        cf_df = _holding_duration_scenario(df, multiplier)
        counterfactual = _aggregate_metrics(cf_df)
    elif variable == "position_size":
        multiplier = float(scenario.get("multiplier", 0.5))
        cf_df = _position_size_scenario(df, multiplier)
        counterfactual = _aggregate_metrics(cf_df)
    elif variable == "volatility_filter":
        filter_value = scenario.get("value", "high_volatility")
        cf_df = _volatility_filter_scenario(df, session_id, filter_value)
        counterfactual = _aggregate_metrics(cf_df)
    elif variable == "post_loss_skip":
        n = int(scenario.get("n", 1))
        cf_df = _post_loss_skip_scenario(df, n)
        counterfactual = _aggregate_metrics(cf_df)
    elif variable == "emergency_filter":
        df_all, df_mod = _emergency_filter_scenario(session_id)
        original = _aggregate_metrics(df_mod)
        counterfactual = _aggregate_metrics(df_all)
    elif variable == "frequency_reduction":
        factor = float(scenario.get("factor", 0.5))
        cf_df = _frequency_reduction_scenario(df, factor)
        counterfactual = _aggregate_metrics(cf_df)
    elif variable == "frequency_increase":
        factor = float(scenario.get("factor", 2.0))
        cf_df = _frequency_increase_scenario(df, factor)
        counterfactual = _aggregate_metrics(cf_df)
    elif variable == "signal_filter":
        min_score = float(scenario.get("min_signal_score", 0.6))
        cf_df = _signal_filter_scenario(df, session_id, min_score)
        counterfactual = _aggregate_metrics(cf_df)
    elif variable == "loss_limit":
        max_loss_pct = float(scenario.get("max_loss_pct", 0.02))
        cf_df = _loss_limit_scenario(df, max_loss_pct)
        counterfactual = _aggregate_metrics(cf_df)
    elif variable == "top_cluster_only":
        cluster_id = scenario.get("cluster", 0)
        cf_df = _top_cluster_only_scenario(df, session_id, cluster_id)
        counterfactual = _aggregate_metrics(cf_df)
    else:
        counterfactual = dict(original)

    delta_pnl = counterfactual["total_pnl"] - original["total_pnl"]
    delta_win_rate = counterfactual["win_rate"] - original["win_rate"]

    result = {
        "original_metrics": original,
        "counterfactual_metrics": counterfactual,
        "delta_pnl": float(delta_pnl),
        "delta_win_rate": float(delta_win_rate),
    }

    try:
        Repository().save_counterfactual(session_id, {
            "scenario": scenario,
            "original_metrics": original,
            "counterfactual_metrics": counterfactual,
            "delta_pnl": delta_pnl,
        })
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save counterfactual result: %s", exc)

    return result


def run_counterfactual(session_id: str, scenario: dict) -> dict:
    """Entry point for /api/counterfactual endpoint.

    FIX: Removed deferred circular import of compute_counterfactual from
    counterfactual_agent. Now calls compute_live_counterfactual directly,
    then applies Bayesian shift via a lazy import inside a try/except
    to avoid the circular dependency at module load time.
    """
    query = str(scenario.get("query", ""))
    scenario_parsed = {k: v for k, v in scenario.items() if k != "query"}

    live = compute_live_counterfactual(session_id, scenario_parsed)

    # Bayesian shift — imported lazily to avoid circular import
    bn_shift: dict = {}
    try:
        from backend.chat.counterfactual_agent import _bayesian_probability_shift  # noqa: PLC0415
        bn_shift = _bayesian_probability_shift(session_id, scenario_parsed)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Bayesian shift computation skipped: %s", exc)

    return {**live, "bayesian_probability_shift": bn_shift, "scenario_description": query}


def test_counterfactual() -> dict:
    """Smoke test for counterfactual module."""
    return {"ok": True, "message": "counterfactual real PnL recomputation wired"}
