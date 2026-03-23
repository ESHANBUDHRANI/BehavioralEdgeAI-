from __future__ import annotations

from typing import Any


def bias_template(model_name: str, metric_name: str, metric_value: float) -> str:
    return f"[{model_name}] {metric_name}: {metric_value:.3f}. Higher values indicate stronger behavioral bias."


def cluster_template(model_name: str, cluster_id: int, description: str) -> str:
    return f"[{model_name}] Cluster {cluster_id}: {description}."


def counterfactual_template(model_name: str, statement: str) -> str:
    return f"[{model_name}] Counterfactual: {statement}"


def report_summary_template(risk_profile: str, emergency_count: int) -> str:
    return (
        f"Risk profile is {risk_profile}. "
        f"Emergency trades excluded from modeling: {emergency_count}."
    )


# ---------------------------------------------------------------------------
# Behavioral summary NLG
# ---------------------------------------------------------------------------

def behavioral_summary_template(
    dominant_cluster: str,
    dominant_hmm_state: str,
    dominant_emotional_state: str,
    top_biases: list[dict[str, Any]],
    shap_top_features: list[dict[str, Any]],
) -> str:
    bias_parts = []
    for b in top_biases[:3]:
        bias_parts.append(f"{b['name']} ({b['severity']})")
    bias_str = ", ".join(bias_parts) if bias_parts else "no significant biases detected"

    feature_str = ", ".join(f["feature"] for f in shap_top_features[:3]) if shap_top_features else "unknown"

    return (
        f"Your dominant behavioral cluster is '{dominant_cluster}' with a primary emotional state "
        f"of '{dominant_emotional_state}'. "
        f"The HMM identifies your most frequent trading state as '{dominant_hmm_state}'. "
        f"Key biases detected: {bias_str}. "
        f"The most influential features driving your behavior are {feature_str}."
    )


# ---------------------------------------------------------------------------
# Risk summary NLG
# ---------------------------------------------------------------------------

def risk_summary_template(
    risk_profile_label: str,
    var95: float | None,
    cvar95: float | None,
    stress_coupling: float | None,
    best_regime: str,
    best_win_rate: float,
    worst_regime: str,
    worst_win_rate: float,
) -> str:
    var_str = f"{var95:.4f}" if var95 is not None else "N/A"
    cvar_str = f"{cvar95:.4f}" if cvar95 is not None else "N/A"
    stress_str = f"{stress_coupling:.2f}" if stress_coupling is not None else "N/A"
    return (
        f"Your risk profile is classified as '{risk_profile_label}'. "
        f"Value-at-Risk (95%) is {var_str} and Conditional VaR is {cvar_str}, "
        f"with a stress-coupling score of {stress_str}. "
        f"You perform best in '{best_regime}' conditions (win rate {best_win_rate:.0%}) "
        f"and worst in '{worst_regime}' conditions (win rate {worst_win_rate:.0%})."
    )


# ---------------------------------------------------------------------------
# Market compatibility NLG
# ---------------------------------------------------------------------------

def market_compatibility_template(
    ticker: str,
    current_regime: str,
    current_volatility: str,
    best_regime: str,
    best_win_rate: float,
    match_or_conflict: str,
) -> str:
    return (
        f"{ticker} is currently in a {current_regime} regime with "
        f"{current_volatility} volatility. Based on your trade history, "
        f"you perform best in {best_regime} conditions with a "
        f"win rate of {best_win_rate:.0%}. This stock's current "
        f"conditions {match_or_conflict} your behavioral strengths."
    )


# ---------------------------------------------------------------------------
# Strategy style NLG
# ---------------------------------------------------------------------------

_STYLE_DESCRIPTIONS: dict[str, str] = {
    "disciplined_systematic": (
        "You exhibit a disciplined, systematic trading approach with strong signal adherence "
        "and consistent position sizing. Your emotional profile is predominantly calm, and you "
        "rarely deviate from your trading framework."
    ),
    "reactive_emotional": (
        "Your trading behavior shows strong reactive tendencies, with frequent trade bursts "
        "following losses and elevated emotional intensity. Position sizes and timing fluctuate "
        "significantly in response to recent outcomes."
    ),
    "concentrated_patient": (
        "You display a patient, concentrated style with longer holding periods and larger position "
        "sizing variance. You tend to wait for high-conviction setups but may over-concentrate "
        "when confident."
    ),
    "overconfident_momentum": (
        "Your style leans toward momentum-chasing with overconfidence markers — increasing size "
        "after wins and underestimating drawdown risk. Signal adherence is moderate but position "
        "management shows bias toward recent winners."
    ),
    "mixed_adaptive": (
        "Your trading style is adaptive and does not fit a single archetype. You shift between "
        "approaches depending on market conditions, which can be a strength but also introduces "
        "inconsistency."
    ),
}

_STYLE_RISK_WARNINGS: dict[str, list[str]] = {
    "disciplined_systematic": [
        "Rigid adherence to signals may cause missed opportunities in regime transitions.",
        "Over-optimization to past patterns risks performance degradation when market structure shifts.",
    ],
    "reactive_emotional": [
        "Revenge trading after losses significantly erodes risk-adjusted returns.",
        "Elevated trade frequency during drawdowns amplifies transaction costs and slippage.",
    ],
    "concentrated_patient": [
        "Heavy concentration in few positions creates tail-risk exposure during market shocks.",
        "Long holding periods may lead to excessive drawdowns if stop-loss discipline is weak.",
    ],
    "overconfident_momentum": [
        "Increasing position size after wins without adjusting for volatility inflates drawdown risk.",
        "Momentum chasing in late-cycle regimes often results in sharp mean-reversion losses.",
    ],
    "mixed_adaptive": [
        "Frequent style switching can generate conflicting signals and reduce conviction.",
        "Lack of a consistent edge makes it harder to evaluate whether losses stem from strategy or execution.",
    ],
}

_STYLE_IMPROVEMENT: dict[str, list[str]] = {
    "disciplined_systematic": [
        "Introduce a regime-detection overlay to reduce signal reliance during transitional phases.",
        "Add a volatility-scaled position sizing rule to dampen exposure in high-volatility regimes.",
        "Periodically review and prune underperforming signals to prevent strategy decay.",
    ],
    "reactive_emotional": [
        "Implement a mandatory cooldown period (e.g. 24 hours) after every losing trade before new entries.",
        "Cap daily trade count to reduce impulsive activity during drawdowns.",
        "Use a pre-trade checklist that forces signal confirmation before execution.",
    ],
    "concentrated_patient": [
        "Set a hard maximum allocation per position (e.g. 10% of portfolio) to limit concentration risk.",
        "Add trailing stops to protect accumulated gains on long-duration holdings.",
        "Diversify across uncorrelated regimes or sectors to smooth equity curve.",
    ],
    "overconfident_momentum": [
        "Apply a fixed position-sizing rule that ignores recent P&L streaks.",
        "Backtest your momentum signals in mean-reverting regimes to understand failure modes.",
        "Track and review trades where you increased size after consecutive wins — measure their actual win rate.",
    ],
    "mixed_adaptive": [
        "Commit to a single strategy archetype for a defined evaluation period (e.g. 30 trades) before switching.",
        "Journal every trade with a clear thesis and review which theses produce positive expectancy.",
        "Identify the one or two regimes where your win rate is highest and allocate more capital there.",
    ],
}


def strategy_style_description(style: str) -> str:
    return _STYLE_DESCRIPTIONS.get(style, _STYLE_DESCRIPTIONS["mixed_adaptive"])


def strategy_risk_warnings(style: str) -> list[str]:
    return _STYLE_RISK_WARNINGS.get(style, _STYLE_RISK_WARNINGS["mixed_adaptive"])


def strategy_improvement_suggestions(style: str) -> list[str]:
    return _STYLE_IMPROVEMENT.get(style, _STYLE_IMPROVEMENT["mixed_adaptive"])


# ---------------------------------------------------------------------------
# Bias action-item NLG (used by behavior_agent decision policy)
# ---------------------------------------------------------------------------

_BIAS_ACTION_ITEMS: dict[str, dict[str, str]] = {
    "disposition_effect_coefficient": {
        "high": (
            "Consider setting hard exit rules before entering a trade. "
            "Define your exit price at entry time."
        ),
        "severe": (
            "Your disposition effect is extreme. Implement automatic stop-loss "
            "and take-profit orders at entry for every position, and review "
            "held-loser duration weekly."
        ),
    },
    "revenge_trading_frequency_rate": {
        "high": (
            "Implement a mandatory cooling-off period of 24 hours after any "
            "trade with a loss exceeding your average loss size."
        ),
        "severe": (
            "Revenge trading is severely impacting returns. Enforce a hard "
            "daily-trade-count cap and disable order entry for 24 hours "
            "after consecutive losses."
        ),
    },
    "overconfidence_proxy": {
        "moderate": (
            "Your position sizes increase after wins. Consider capping size "
            "increases at 10% above your baseline regardless of recent performance."
        ),
        "high": (
            "Strong overconfidence detected. Implement a fixed position-sizing "
            "rule that ignores recent P&L and reduce maximum single-trade "
            "allocation."
        ),
        "severe": (
            "Extreme overconfidence. Freeze position sizing at the 30-day "
            "moving average and introduce a peer review step before "
            "above-average size trades."
        ),
    },
    "signal_following_rate": {
        "low": (
            "You frequently trade against technical signals. Track whether "
            "your contrarian trades outperform your signal-aligned trades "
            "to validate this approach."
        ),
    },
}


def bias_action_recommendation(bias_name: str, severity: str) -> str | None:
    """Return a concrete action recommendation for a bias at a given severity.

    Returns None if no recommendation is defined for the combination.
    """
    return _BIAS_ACTION_ITEMS.get(bias_name, {}).get(severity)


def test_nlg() -> dict:
    """Smoke test for NLG module."""
    return {"ok": True, "message": "nlg templates wired"}
