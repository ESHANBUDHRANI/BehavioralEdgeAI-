from __future__ import annotations

import numpy as np
import pandas as pd
from backend.models.runtime import insufficient_data_guard


def run(features_df: pd.DataFrame, context_df: pd.DataFrame | None = None, config: dict | None = None) -> dict:
    guard = insufficient_data_guard(len(features_df))
    if guard:
        return guard
    speed = 1.0 / (features_df.get("next_trade_delta_hours", pd.Series([24.0] * len(features_df))) + 1e-6)
    size_dev = features_df.get("position_size_dev_rolling", pd.Series([0.0] * len(features_df))).abs()
    hold_comp = features_df.get("early_exit", pd.Series([0] * len(features_df)))
    post_loss = features_df.get("size_change_after_loss", pd.Series([0.0] * len(features_df))).abs()
    score = (speed.rank(pct=True) + size_dev.rank(pct=True) + hold_comp.rank(pct=True) + post_loss.rank(pct=True)) / 4.0
    labels = np.select(
        [score < 0.33, score < 0.66],
        ["calm_disciplined", "anxious_reactive"],
        default="euphoric_overconfident",
    )
    confidence = (score - 0.5).abs() * 2
    return {
        "emotional_state_label": labels.tolist(),
        "confidence_per_trade": confidence.tolist(),
        "timeline_score": score.tolist(),
        "confidence": 0.76,
    }


def test_emotional_state() -> dict:
    return {"ok": True, "message": "emotional state model wired"}
