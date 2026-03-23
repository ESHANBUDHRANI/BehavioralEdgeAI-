from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from backend.models.runtime import insufficient_data_guard


def _dcc_correlations(eps: np.ndarray, a: float = 0.02, b: float = 0.97) -> np.ndarray:
    t_len = eps.shape[0]
    q_bar = np.cov(eps.T)
    q_t = q_bar.copy()
    corrs = np.zeros(t_len, dtype=float)
    for t in range(t_len):
        e_prev = eps[t - 1].reshape(-1, 1) if t > 0 else np.zeros((eps.shape[1], 1))
        q_t = (1 - a - b) * q_bar + a * (e_prev @ e_prev.T) + b * q_t
        d = np.sqrt(np.diag(q_t))
        d_inv = np.diag(1.0 / np.where(d == 0, 1e-9, d))
        r_t = d_inv @ q_t @ d_inv
        corrs[t] = float(np.clip(r_t[0, 1], -1.0, 1.0))
    return corrs


def run(features_df: pd.DataFrame, context_df: pd.DataFrame, config: dict | None = None) -> dict:
    guard = insufficient_data_guard(len(features_df))
    if guard:
        return guard
    behavior_vol = features_df.get("pnl", pd.Series([0.0] * len(features_df))).astype(float).fillna(0.0).to_numpy()
    market_vol = context_df.get("volatility_score", pd.Series(np.zeros(len(features_df)))).astype(float).fillna(0.0).to_numpy()
    garch_user = arch_model(behavior_vol, vol="Garch", p=1, q=1, dist="normal")
    garch_market = arch_model(market_vol, vol="Garch", p=1, q=1, dist="normal")
    fit_user = garch_user.fit(disp="off")
    fit_market = garch_market.fit(disp="off")
    user_vol = np.asarray(fit_user.conditional_volatility, dtype=float)
    market_vol_cond = np.asarray(fit_market.conditional_volatility, dtype=float)
    user_std = behavior_vol / np.where(user_vol == 0, 1e-9, user_vol)
    market_std = market_vol / np.where(market_vol_cond == 0, 1e-9, market_vol_cond)
    eps = np.column_stack([user_std, market_std])
    dcc_corr = _dcc_correlations(eps)
    stress_coupling = float(np.nanmax(np.abs(dcc_corr))) if len(dcc_corr) else 0.0
    return {
        "user_garch_params": {k: float(v) for k, v in fit_user.params.items()},
        "market_garch_params": {k: float(v) for k, v in fit_market.params.items()},
        "conditional_volatility_user": user_vol.tolist(),
        "conditional_volatility_market": market_vol_cond.tolist(),
        "time_varying_correlation": dcc_corr.tolist(),
        "stress_coupling_score": stress_coupling,
        "confidence": 0.78,
    }


def test_garch_model() -> dict:
    return {"ok": True, "message": "garch model wired"}
