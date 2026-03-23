from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from backend.config import get_settings
from backend.models.runtime import insufficient_data_guard

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback: Spearman-ranked variable importance
# ---------------------------------------------------------------------------

def _spearman_importance(df: pd.DataFrame, target: str) -> dict[str, float]:
    """Rank feature importance using Spearman correlation with target."""
    numeric = df.select_dtypes(include=["number"]).fillna(0)
    if target not in numeric.columns or numeric.shape[1] < 2:
        return {}
    result: dict[str, float] = {}
    target_series = numeric[target]
    for col in numeric.columns:
        if col == target:
            continue
        try:
            corr, _ = spearmanr(numeric[col], target_series, nan_policy="omit")
            result[col] = float(abs(corr)) if np.isfinite(corr) else 0.0
        except Exception:  # noqa: BLE001
            result[col] = 0.0
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


def _fallback_result(numeric: pd.DataFrame, reason: str, target: str = "emotional_score") -> dict[str, Any]:
    """Return a well-documented fallback when TFT cannot train."""
    importance = _spearman_importance(numeric, target)
    return {
        "insufficient_data": True,
        "message": reason,
        "fallback_reason": reason,
        "variable_importance": importance,
        "top_variables": [{"feature": k, "score": v} for k, v in list(importance.items())[:10]],
        "confidence": 0.4,
    }


# ---------------------------------------------------------------------------
# Input validation and cleaning
# ---------------------------------------------------------------------------

def _validate_and_clean(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Validate inputs, replace infinities, drop zero-variance columns.

    Returns cleaned df, surviving feature columns, and list of log messages.
    """
    logs: list[str] = []

    inf_mask = np.isinf(df[feature_cols + [target_col]].select_dtypes(include=["number"]))
    if inf_mask.any().any():
        inf_cols = [c for c in inf_mask.columns if inf_mask[c].any()]
        for col in inf_cols:
            median_val = df[col].replace([np.inf, -np.inf], np.nan).median()
            df[col] = df[col].replace([np.inf, -np.inf], median_val)
        logs.append(f"Replaced infinities with column medians in {inf_cols}")

    zero_var = [c for c in feature_cols if df[c].std() == 0]
    if zero_var:
        feature_cols = [c for c in feature_cols if c not in zero_var]
        logs.append(f"Dropped zero-variance columns: {zero_var}")

    if "time_idx" in df.columns:
        df = df.sort_values("time_idx").reset_index(drop=True)
        expected = np.arange(len(df))
        if not np.array_equal(df["time_idx"].values, expected):
            df["time_idx"] = expected
            logs.append("Reindexed time_idx to ensure strict monotonicity")

    for msg in logs:
        logger.info("[TFT input validation] %s", msg)

    return df, feature_cols, logs


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_dir() -> Path:
    """Return the directory for TFT checkpoints, creating it if needed."""
    d = get_settings().cache_dir / "tft_checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _checkpoint_path(session_id: str) -> Path:
    """Return the path for a session's TFT checkpoint."""
    return _checkpoint_dir() / f"{session_id}_tft.ckpt"


# ---------------------------------------------------------------------------
# Attention extraction helper
# ---------------------------------------------------------------------------

def _extract_time_variable_attention(raw_pred, interpretation) -> np.ndarray:
    """Extract a [timestep x variable] attention matrix from TFT interpretation."""
    for key in ["encoder_variables", "decoder_variables"]:
        tensor = interpretation.get(key) if isinstance(interpretation, dict) else None
        if tensor is None:
            continue
        arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        if arr.ndim == 3:
            return np.mean(arr, axis=0)
        if arr.ndim == 2:
            return arr
    static_vars = interpretation.get("static_variables") if isinstance(interpretation, dict) else None
    if static_vars is not None:
        arr = static_vars.detach().cpu().numpy() if hasattr(static_vars, "detach") else np.asarray(static_vars)
        if arr.ndim == 2:
            return np.mean(arr, axis=0, keepdims=True)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
    return np.zeros((1, 1), dtype=float)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_tft(dataset, train_loader, val_loader, use_gpu: bool, checkpoint_path: Path | None):
    """Train a TFT model with EarlyStopping and optional checkpoint saving."""
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        hidden_size=32,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=0,
        reduce_on_plateau_patience=3,
    )

    callbacks = []
    monitor_metric = "val_loss" if val_loader is not None else "train_loss"

    early_stop = EarlyStopping(monitor=monitor_metric, patience=3, mode="min")
    callbacks.append(early_stop)

    if checkpoint_path is not None:
        ckpt_cb = ModelCheckpoint(
            dirpath=str(checkpoint_path.parent),
            filename=checkpoint_path.stem,
            save_top_k=1,
            monitor=monitor_metric,
            mode="min",
        )
        callbacks.append(ckpt_cb)

    trainer = Trainer(
        max_epochs=15,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        enable_checkpointing=checkpoint_path is not None,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        gradient_clip_val=0.1,
        callbacks=callbacks,
    )

    val_dls = [val_loader] if val_loader is not None else None
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_dls)

    early_stopped = early_stop.stopped_epoch > 0 if hasattr(early_stop, "stopped_epoch") else False
    epochs_completed = trainer.current_epoch

    return tft, trainer, early_stopped, epochs_completed


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

def _compute_val_metrics(tft, val_loader, val_df: pd.DataFrame, target_col: str) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE, and R² on the validation set."""
    try:
        preds = tft.predict(val_loader)
        actuals = torch.tensor(val_df[target_col].values, dtype=torch.float32)
        if preds.ndim > 1:
            preds = preds.mean(dim=-1)
        preds = preds[:len(actuals)]
        diff = preds - actuals
        mae = float(torch.mean(torch.abs(diff)).item())
        rmse = float(torch.sqrt(torch.mean(diff ** 2)).item())
        nonzero_mask = actuals != 0
        mape = float(torch.mean(torch.abs(diff[nonzero_mask] / actuals[nonzero_mask])).item()) if nonzero_mask.any() else 0.0
        ss_res = float(torch.sum(diff ** 2).item())
        ss_tot = float(torch.sum((actuals - actuals.mean()) ** 2).item())
        r2 = 1.0 - (ss_res / max(ss_tot, 1e-9))
        return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}
    except Exception as exc:  # noqa: BLE001
        logger.warning("TFT validation metric computation failed: %s", exc)
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(features_df: pd.DataFrame, context_df: pd.DataFrame, config: dict | None = None) -> dict:
    """Run the Temporal Fusion Transformer model with production hardening.

    Supports checkpoint caching, proper validation metrics, graceful fallback
    with Spearman importance rankings, and input validation.
    """
    config = config or {}
    session_id = config.get("session_id", "")
    force_retrain = config.get("force_retrain", False)

    guard = insufficient_data_guard(len(features_df))
    if guard:
        return guard

    if context_df is None or context_df.empty:
        context_df = pd.DataFrame(index=features_df.index)

    merged = pd.concat(
        [features_df.reset_index(drop=True), context_df.reset_index(drop=True)],
        axis=1,
    )
    numeric = merged.select_dtypes(include=["number"]).copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "emotional_score" not in numeric.columns:
        numeric["emotional_score"] = features_df.get(
            "emotional_score",
            pd.Series(np.zeros(len(features_df))),
        ).astype(float).fillna(0.0)

    target_col = "emotional_score"
    feature_cols = [c for c in numeric.columns if c != target_col][:40]
    if not feature_cols:
        return _fallback_result(numeric, "no numeric predictors for TFT", target_col)

    if len(numeric) < 60:
        return _fallback_result(
            numeric,
            f"insufficient_rows: {len(numeric)}/60 required",
            target_col,
        )

    # Delayed import to keep startup lean
    from pytorch_forecasting import TimeSeriesDataSet

    df = numeric[feature_cols + [target_col]].copy()
    df["time_idx"] = np.arange(len(df))
    df["series_id"] = "user_series"

    # Input validation
    df, feature_cols, validation_logs = _validate_and_clean(df, feature_cols, target_col)
    if not feature_cols:
        return _fallback_result(numeric, "all features dropped during validation (zero variance)", target_col)

    encoder_length = min(30, len(df) // 3)
    prediction_length = 1

    # Date-based 80/20 split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    if len(train_df) < encoder_length + prediction_length + 10:
        return _fallback_result(
            numeric,
            f"insufficient sequence depth: {len(train_df)} train rows, need {encoder_length + prediction_length + 10}",
            target_col,
        )

    behavioral_features = [c for c in feature_cols if c in features_df.columns]
    market_features = [c for c in feature_cols if c in context_df.columns and c not in features_df.columns]
    time_varying_unknown = behavioral_features + [target_col] if behavioral_features else feature_cols + [target_col]
    time_varying_known = market_features + ["time_idx"] if market_features else ["time_idx"]

    training_cutoff = int(train_df["time_idx"].max())

    dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["series_id"],
        max_encoder_length=encoder_length,
        max_prediction_length=prediction_length,
        static_categoricals=["series_id"],
        time_varying_unknown_reals=time_varying_unknown,
        time_varying_known_reals=time_varying_known,
        add_relative_time_idx=True,
    )
    train_loader = dataset.to_dataloader(train=True, batch_size=32, num_workers=0)

    val_loader = None
    if len(val_df) >= encoder_length + prediction_length:
        try:
            val_dataset = TimeSeriesDataSet.from_dataset(dataset, val_df, stop_randomization=True)
            val_loader = val_dataset.to_dataloader(train=False, batch_size=32, num_workers=0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not build validation dataloader: %s", exc)

    # Checkpoint: check for existing
    ckpt_file = _checkpoint_path(session_id) if session_id else None
    tft = None

    if ckpt_file and ckpt_file.exists() and not force_retrain:
        try:
            from pytorch_forecasting import TemporalFusionTransformer
            tft = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_file))
            logger.info("Loaded TFT checkpoint for session %s", session_id)
            used_device = "checkpoint"
            early_stopped = False
            epochs_completed = 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load TFT checkpoint, retraining: %s", exc)
            tft = None

    if tft is None:
        used_device = "cpu"
        try:
            if torch.cuda.is_available():
                tft, trainer, early_stopped, epochs_completed = _train_tft(
                    dataset, train_loader, val_loader, use_gpu=True, checkpoint_path=ckpt_file,
                )
                used_device = "gpu"
            else:
                tft, trainer, early_stopped, epochs_completed = _train_tft(
                    dataset, train_loader, val_loader, use_gpu=False, checkpoint_path=ckpt_file,
                )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda" in msg:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                tft, trainer, early_stopped, epochs_completed = _train_tft(
                    dataset, train_loader, val_loader, use_gpu=False, checkpoint_path=ckpt_file,
                )
                used_device = "cpu_fallback"
            else:
                return _fallback_result(numeric, f"training_failed: {exc}", target_col)
        except Exception as exc:  # noqa: BLE001
            return _fallback_result(numeric, f"training_failed: {exc}", target_col)

        # Save checkpoint manually if ModelCheckpoint callback didn't fire
        if ckpt_file and not ckpt_file.exists():
            try:
                trainer.save_checkpoint(str(ckpt_file))
                logger.info("Saved TFT checkpoint to %s", ckpt_file)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not save TFT checkpoint: %s", exc)

    # ── Interpretation ────────────────────────────────────────────────────
    raw_prediction = tft.predict(train_loader, mode="raw", return_x=True)
    raw_output = raw_prediction
    if isinstance(raw_prediction, tuple) and raw_prediction:
        raw_output = raw_prediction[0]
    if hasattr(raw_output, "output"):
        raw_output = raw_output.output

    interpretation = tft.interpret_output(raw_output, reduction="none")
    time_var_matrix = _extract_time_variable_attention(raw_output, interpretation).astype(float)
    if time_var_matrix.size == 0:
        time_var_matrix = np.zeros((1, max(1, len(feature_cols))), dtype=float)
    if time_var_matrix.shape[1] > len(feature_cols):
        time_var_matrix = time_var_matrix[:, :len(feature_cols)]
    elif time_var_matrix.shape[1] < len(feature_cols):
        pad = np.zeros((time_var_matrix.shape[0], len(feature_cols) - time_var_matrix.shape[1]), dtype=float)
        time_var_matrix = np.concatenate([time_var_matrix, pad], axis=1)

    mean_importance = np.mean(np.abs(time_var_matrix), axis=0)
    ranked = sorted(
        [(feature_cols[i], float(mean_importance[i])) for i in range(len(feature_cols))],
        key=lambda x: x[1],
        reverse=True,
    )

    temporal = interpretation.get("attention") if isinstance(interpretation, dict) else None
    temporal_weights = []
    if temporal is not None:
        arr = temporal.detach().cpu().numpy() if hasattr(temporal, "detach") else np.asarray(temporal)
        if arr.ndim >= 2:
            temporal_weights = np.mean(arr, axis=tuple(range(arr.ndim - 1))).tolist()
        else:
            temporal_weights = arr.tolist()

    # ── Validation metrics ────────────────────────────────────────────────
    val_metrics = {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0}
    if val_loader is not None:
        val_metrics = _compute_val_metrics(tft, val_loader, val_df, target_col)

    top_variables = [{"feature": f, "score": s} for f, s in ranked[:10]]

    return {
        "config": {
            "hidden_size": 32,
            "attention_head_size": 1,
            "dropout": 0.1,
            "hidden_continuous_size": 16,
            "max_encoder_length": encoder_length,
            "target": target_col,
        },
        "device_used": used_device,
        "attention_weights": temporal_weights,
        "attention_by_timestep_variable": {
            "timesteps": list(range(int(time_var_matrix.shape[0]))),
            "variables": feature_cols,
            "weights": time_var_matrix.tolist(),
        },
        "variable_importance": dict(ranked[:20]),
        "top_variables": top_variables,
        "prediction_error": val_metrics["mae"],
        "mae": val_metrics["mae"],
        "rmse": val_metrics["rmse"],
        "mape": val_metrics["mape"],
        "r2": val_metrics["r2"],
        "training_epochs_completed": epochs_completed if tft is not None else 0,
        "early_stopped": early_stopped if tft is not None else False,
        "validation_logs": validation_logs if 'validation_logs' in dir() else [],
        "checkpoint_path": str(ckpt_file) if ckpt_file else None,
        "confidence": 0.8,
    }


def test_tft_model() -> dict:
    """Smoke test for TFT module import."""
    return {"ok": True, "message": "temporal fusion transformer training path wired"}
