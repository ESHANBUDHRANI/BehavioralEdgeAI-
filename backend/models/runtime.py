from __future__ import annotations

import torch


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_float32_tensor(x, device: str):
    return torch.tensor(x, dtype=torch.float32, device=device)


def insufficient_data_guard(n_rows: int) -> dict | None:
    if n_rows < 50:
        return {
            "insufficient_data": True,
            "message": "insufficient data: need at least 50 trades",
            "confidence": 0.0,
        }
    return None


def test_runtime() -> dict:
    return {"ok": True, "device": get_device()}
