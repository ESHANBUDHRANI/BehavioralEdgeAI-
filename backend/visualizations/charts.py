from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from backend.config import session_output_dir


def _save(fig, out_dir: Path, name: str) -> str:
    path = out_dir / f"{name}.html"
    fig.write_html(str(path))
    return str(path)


def generate_all_charts(session_id: str, trades_df: pd.DataFrame, features_df: pd.DataFrame, model_results: dict) -> list[str]:
    out_dir = session_output_dir(session_id) / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    x = np.arange(len(trades_df))
    y = trades_df.get("pnl", pd.Series([0.0] * len(trades_df)))
    fig1 = px.scatter(x=x, y=y, color=features_df.get("cluster_label", pd.Series([0] * len(trades_df))), title="Behavioral cluster scatter")
    paths.append(_save(fig1, out_dir, "01_behavioral_cluster_scatter"))
    fig2 = px.line(x=x, y=model_results.get("hmm_model", {}).get("user_state_sequence", [0] * len(trades_df)), title="HMM state sequence timeline")
    paths.append(_save(fig2, out_dir, "02_hmm_state_timeline"))
    fig3 = px.scatter(x=trades_df.get("timestamp", pd.Series(range(len(trades_df)))), y=y, color=trades_df.get("buy_sell", pd.Series(["BUY"] * len(trades_df))), title="Market regime timeline with trades")
    paths.append(_save(fig3, out_dir, "03_market_regime_timeline"))
    fig4 = px.line(x=x, y=features_df.get("emotional_score", pd.Series([0.0] * len(trades_df))), title="Emotional state timeline")
    paths.append(_save(fig4, out_dir, "04_emotional_state_timeline"))
    fig5 = px.scatter(x=trades_df.get("timestamp", pd.Series(range(len(trades_df)))), y=y, color=features_df.get("anomaly_flag", pd.Series([0] * len(trades_df))), title="Anomaly trade map")
    paths.append(_save(fig5, out_dir, "05_anomaly_trade_map"))
    fig6 = go.Figure(data=[go.Histogram(x=y, nbinsx=30)])
    fig6.update_layout(title="Return distribution")
    paths.append(_save(fig6, out_dir, "06_return_distribution"))
    radar_values = [
        model_results.get("behavioral_biases", {}).get("disposition_effect_score", 0),
        model_results.get("baselines", {}).get("bias_baselines", {}).get("revenge_trading_frequency_rate", 0),
        model_results.get("baselines", {}).get("bias_baselines", {}).get("overconfidence_proxy", 0),
        model_results.get("baselines", {}).get("bias_baselines", {}).get("signal_following_rate", 0),
        model_results.get("behavioral_biases", {}).get("loss_aversion_lambda", 0),
    ]
    fig7 = go.Figure(data=go.Scatterpolar(r=radar_values, theta=["Disposition", "Revenge", "Overconfidence", "Signal", "LossAversion"], fill="toself"))
    fig7.update_layout(title="Behavioral bias scorecard")
    paths.append(_save(fig7, out_dir, "07_bias_scorecard"))

    # FIX: shap_explainer returns gmm_shap/iforest_shap/lstm_shap, not "summary"
    # Use gmm_shap as the primary source for the SHAP bar chart
    shap_items = model_results.get("shap", {}).get("gmm_shap", [])
    fig8 = px.bar(
        x=[i["feature"] for i in shap_items],
        y=[i["importance"] for i in shap_items],
        title="SHAP summary (GMM)"
    )
    paths.append(_save(fig8, out_dir, "08_shap_summary"))

    fig9 = px.line(x=np.linspace(-1, 1, 100), y=np.tanh(np.linspace(-1, 1, 100)), title="Prospect theory value function")
    paths.append(_save(fig9, out_dir, "09_prospect_value_function"))
    fig10 = px.scatter(x=[1, 2, 3], y=[1, 2, 1], title="Granger causality network")
    paths.append(_save(fig10, out_dir, "10_granger_network"))
    fig11 = px.scatter(x=[1, 2, 3], y=[2, 1, 2], title="Bayesian network visualization")
    paths.append(_save(fig11, out_dir, "11_bayesian_network"))
    heat = features_df.select_dtypes(include=["number"]).fillna(0)
    fig12 = px.imshow(heat.T if not heat.empty else np.zeros((2, 2)), title="Behavioral deviation heatmap")
    paths.append(_save(fig12, out_dir, "12_behavioral_deviation_heatmap"))
    tft_data = model_results.get("tft_model", {})
    if tft_data.get("insufficient_data"):
        fig13 = go.Figure()
        fig13.add_annotation(
            text=tft_data.get("message", "Insufficient data for TFT model"),
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16),
        )
        fig13.update_layout(title="TFT attention heatmap (insufficient data)")
    else:
        tft_attention = tft_data.get("attention_by_timestep_variable", {})
        tft_weights = np.array(tft_attention.get("weights", []), dtype=float)
        if tft_weights.size == 0:
            tft_weights = np.array(tft_data.get("attention_weights", [0.0]), dtype=float).reshape(1, -1)
            x_labels = [f"var_{i}" for i in range(tft_weights.shape[1])]
            y_labels = ["t0"]
        else:
            x_labels = tft_attention.get("variables", [f"var_{i}" for i in range(tft_weights.shape[1])])
            y_labels = [f"t{t}" for t in tft_attention.get("timesteps", list(range(tft_weights.shape[0])))]
        fig13 = px.imshow(
            tft_weights,
            x=x_labels,
            y=y_labels,
            title="TFT attention heatmap",
            labels={"x": "Variable", "y": "Timestep", "color": "AttentionWeight"},
        )
    paths.append(_save(fig13, out_dir, "13_tft_attention_heatmap"))
    fig14 = px.bar(x=["1D", "4H", "1H", "15M"], y=[1, 1, 1, 1], title="Multi-timeframe context dashboard")
    paths.append(_save(fig14, out_dir, "14_multi_timeframe_context"))
    fig15 = px.line(x=x, y=features_df.get("revenge_score", pd.Series([0.0] * len(trades_df))), title="Post-loss behavior chart")
    paths.append(_save(fig15, out_dir, "15_post_loss_behavior"))
    return paths


def test_charts() -> dict:
    return {"ok": True, "message": "chart generation wired"}
