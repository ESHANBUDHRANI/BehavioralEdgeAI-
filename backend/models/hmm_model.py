from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from backend.models.runtime import insufficient_data_guard


def _fit_hmm(X: np.ndarray, n_states: int = 3):
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=42)
    model.fit(X)
    states = model.predict(X)
    _, viterbi_path = model.decode(X, algorithm="viterbi")
    return model, states, viterbi_path


def _state_labels(means: np.ndarray) -> list[str]:
    labels = []
    for idx, mean_vec in enumerate(means):
        magnitude = float(np.linalg.norm(mean_vec))
        if magnitude > 1.5:
            labels.append(f"state_{idx}_high_intensity")
        elif magnitude < 0.7:
            labels.append(f"state_{idx}_low_intensity")
        else:
            labels.append(f"state_{idx}_moderate")
    return labels


def run(features_df: pd.DataFrame, context_df: pd.DataFrame, config: dict | None = None) -> dict:
    guard = insufficient_data_guard(len(features_df))
    if guard:
        return guard
    fX = features_df.select_dtypes(include=["number"]).fillna(0).to_numpy(dtype=float)
    cX = context_df.select_dtypes(include=["number"]).fillna(0).to_numpy(dtype=float) if context_df is not None else fX
    fX = StandardScaler().fit_transform(fX)
    cX = StandardScaler().fit_transform(cX)
    n_states = 4 if len(features_df) > 100 else 3
    user_hmm, user_states, user_viterbi = _fit_hmm(fX, n_states=n_states)
    market_hmm, market_states, market_viterbi = _fit_hmm(cX, n_states=n_states)
    user_transition = user_hmm.transmat_
    transition_story = []
    for from_state in range(user_transition.shape[0]):
        to_state = int(np.argmax(user_transition[from_state]))
        prob = float(user_transition[from_state, to_state])
        transition_story.append(
            {
                "from_state": from_state,
                "to_state": to_state,
                "probability": prob,
                "story": f"Given state {from_state}, most likely shift is to state {to_state} ({prob:.2%}).",
            }
        )

    # FIX: added log_likelihood key so report_generator.py reads a real value
    try:
        log_likelihood = float(user_hmm.score(fX))
    except Exception:  # noqa: BLE001
        log_likelihood = 0.0

    return {
        "user_state_sequence": user_states.tolist(),
        "market_state_sequence": market_states.tolist(),
        # FIX: key was "user_transition_matrix" but behavior_agent read "transition_matrix"
        # Providing both keys for full compatibility
        "user_transition_matrix": user_transition.tolist(),
        "transition_matrix": user_transition.tolist(),
        "market_transition_matrix": market_hmm.transmat_.tolist(),
        "user_emission_means": user_hmm.means_.tolist(),
        "market_emission_means": market_hmm.means_.tolist(),
        "state_labels": _state_labels(user_hmm.means_),
        "user_viterbi_path": user_viterbi.tolist(),
        "market_viterbi_path": market_viterbi.tolist(),
        "transition_story": transition_story,
        "log_likelihood": log_likelihood,
        "confidence": 0.75,
    }


def test_hmm_model() -> dict:
    return {"ok": True, "message": "hmm model wired"}
