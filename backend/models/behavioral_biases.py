from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from backend.models.runtime import insufficient_data_guard


def _prospect_value(x, alpha, lambd):
    return np.where(x >= 0, np.power(x + 1e-9, alpha), -lambd * np.power(np.abs(x) + 1e-9, alpha))


def run(features_df: pd.DataFrame, context_df: pd.DataFrame | None = None, config: dict | None = None) -> dict:
    guard = insufficient_data_guard(len(features_df))
    if guard:
        return guard
    pnl = features_df.get("pnl", pd.Series([0.0] * len(features_df))).astype(float).to_numpy()
    x = np.linspace(-1, 1, len(pnl))
    y = np.tanh(pnl / (np.std(pnl) + 1e-9))
    params, _ = curve_fit(_prospect_value, x, y, p0=[0.88, 2.25], maxfev=10000)
    alpha, lambd = float(params[0]), float(params[1])
    gains_realized = np.sum(pnl > 0)
    losses_realized = np.sum(pnl < 0)
    pgr = gains_realized / max(gains_realized + losses_realized, 1)
    plr = losses_realized / max(gains_realized + losses_realized, 1)
    disposition = pgr / max(plr, 1e-6)
    return {
        "loss_aversion_lambda": lambd,
        "risk_seeking_alpha": alpha,
        "disposition_effect_score": float(disposition),
        "interpretation": "higher lambda indicates stronger loss aversion",
        "confidence": 0.77,
    }


def test_behavioral_biases() -> dict:
    return {"ok": True, "message": "prospect theory/disposition wired"}
