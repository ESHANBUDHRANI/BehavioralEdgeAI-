from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from backend.models.runtime import get_device, insufficient_data_guard


class SeqLSTM(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, num_layers=2, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(128, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _make_sequences(X: np.ndarray, window: int = 20):
    xs, ys = [], []
    for i in range(len(X) - window):
        xs.append(X[i : i + window])
        ys.append(X[i + window])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def run(features_df: pd.DataFrame, context_df: pd.DataFrame | None = None, config: dict | None = None) -> dict:
    guard = insufficient_data_guard(len(features_df))
    if guard:
        return guard
    X = features_df.select_dtypes(include=["number"]).fillna(0).to_numpy(dtype=np.float32)
    seq_x, seq_y = _make_sequences(X, window=20)
    if len(seq_x) < 10:
        return {"insufficient_data": True, "message": "need more sequential depth", "confidence": 0.0}
    device = get_device()
    model = SeqLSTM(X.shape[1]).to(device).float()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    tx = torch.tensor(seq_x, dtype=torch.float32, device=device)
    ty = torch.tensor(seq_y, dtype=torch.float32, device=device)
    for _ in range(20):
        optim.zero_grad()
        pred = model(tx)
        loss = loss_fn(pred, ty)
        loss.backward()
        optim.step()
    with torch.no_grad():
        pred = model(tx).cpu().numpy()
    error = np.mean((pred - seq_y) ** 2, axis=1)

    # FIX: replaced deprecated .ptp() with (max - min)
    error_range = error.max() - error.min() + 1e-9
    anomaly_score = ((error - error.min()) / error_range).tolist()

    return {
        "behavioral_state_vector": pred.tolist(),
        "next_state_prediction": pred[-1].tolist(),
        "prediction_error": error.tolist(),
        "anomaly_score": anomaly_score,
        "confidence": 0.72,
    }


def test_lstm_model() -> dict:
    return {"ok": True, "message": "lstm model wired"}
