from __future__ import annotations

from collections import defaultdict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from backend.config import get_settings
from backend.models.runtime import get_device


_MODEL = None
_TOKENIZER = None


def _load_model():
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _TOKENIZER, _MODEL
    settings = get_settings()
    model_dir = settings.models_pretrained_dir / "finbert"
    model_name = str(model_dir) if model_dir.exists() else "ProsusAI/finbert"
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name, cache_dir=settings.models_pretrained_dir)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=settings.models_pretrained_dir)
    _MODEL.to(get_device())
    _MODEL.eval()
    return _TOKENIZER, _MODEL


def score_headlines(news_items: list[dict]) -> dict:
    tokenizer, model = _load_model()
    grouped = defaultdict(list)
    for item in news_items:
        grouped[(item["symbol"], item["date"])].append(item["headline"])
    output = {}
    labels = ["negative", "neutral", "positive"]
    for key, headlines in grouped.items():
        symbol, date = key
        out_key = f"{symbol}|{date}"
        if not headlines:
            output[out_key] = {"label": "neutral", "score": 0.0, "confidence": 0.0}
            continue
        enc = tokenizer(headlines, truncation=True, padding=True, return_tensors="pt").to(get_device())
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).mean(dim=0)
        idx = int(torch.argmax(probs).item())
        output[out_key] = {
            "label": labels[idx],
            "score": float(probs[2] - probs[0]),
            "confidence": float(torch.max(probs).item()),
        }
    return output


def run(features_df, context_df=None, config=None) -> dict:
    config = config or {}
    news_items = config.get("news_items", [])
    if not news_items:
        return {"daily_symbol_sentiment": {}, "message": "no news headlines available", "confidence": 0.1}
    scored = score_headlines(news_items)
    avg_conf = 0.0
    if scored:
        avg_conf = sum(float(v.get("confidence", 0.0)) for v in scored.values()) / len(scored)
    return {"daily_symbol_sentiment": scored, "confidence": float(avg_conf)}


def test_sentiment() -> dict:
    return {"ok": True, "message": "finbert sentiment wiring ready"}
