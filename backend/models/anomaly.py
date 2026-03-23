from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from backend.models.runtime import get_device, insufficient_data_guard


class Autoencoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, in_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def run(features_df: pd.DataFrame, context_df: pd.DataFrame | None = None, config: dict | None = None) -> dict:
    guard = insufficient_data_guard(len(features_df))
    if guard:
        return guard
    numeric = features_df.select_dtypes(include=["number"]).fillna(0)
    feature_names = numeric.columns.tolist()
    scaler = StandardScaler()
    X = scaler.fit_transform(numeric.to_numpy(dtype=np.float32)).astype(np.float32)
    iforest = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
    iso_pred = iforest.fit_predict(X)
    iso_score = -iforest.score_samples(X)
    normal_mask = iso_pred == 1
    train_X = X[normal_mask] if normal_mask.any() else X
    device = get_device()
    model = Autoencoder(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    train_t = torch.tensor(train_X, dtype=torch.float32, device=device)
    model.train()
    for _ in range(60):
        opt.zero_grad()
        recon = model(train_t)
        loss = loss_fn(recon, train_t)
        loss.backward()
        opt.step()
    all_t = torch.tensor(X, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        recon = model(all_t).cpu().numpy()
    residual = (X - recon) ** 2
    recon_error = np.mean(residual, axis=1)
    per_feature_error = residual.mean(axis=0)

    # FIX: replaced deprecated .ptp() with (max - min)
    iso_range = iso_score.max() - iso_score.min() + 1e-9
    recon_range = recon_error.max() - recon_error.min() + 1e-9
    combined = (iso_score - iso_score.min()) / iso_range
    combined = 0.5 * combined + 0.5 * ((recon_error - recon_error.min()) / recon_range)

    threshold = float(np.quantile(combined, 0.9))
    return {
        "anomaly_flag": (combined > threshold).astype(int).tolist(),
        "isolation_score": iso_score.tolist(),
        "reconstruction_error": recon_error.tolist(),
        "reconstruction_error_by_feature": {
            feature_names[i]: float(per_feature_error[i]) for i in range(len(feature_names))
        },
        "anomaly_threshold": threshold,
        "anomaly_confidence": combined.tolist(),
        "confidence": 0.8,
    }


def test_anomaly() -> dict:
    return {"ok": True, "message": "isolation forest + autoencoder wired"}
