from __future__ import annotations

import json
from datetime import datetime
from typing import Iterable
from sqlalchemy import select
from backend.database.models import (
    AnalysisResult,
    ConversationHistory,
    Counterfactual,
    EmergencyTrade,
    MarketContext,
    SessionModel,
    Trade,
)
from backend.database.session import SessionLocal


class Repository:
    def create_session(self, session_id: str, filename: str) -> None:
        with SessionLocal() as db:
            db.add(SessionModel(session_id=session_id, filename=filename))
            db.commit()

    def set_status(self, session_id: str, status: str) -> None:
        with SessionLocal() as db:
            row = db.get(SessionModel, session_id)
            if row:
                row.analysis_status = status
                db.commit()

    def get_session(self, session_id: str) -> SessionModel | None:
        with SessionLocal() as db:
            return db.get(SessionModel, session_id)

    def insert_trades(self, session_id: str, trades: Iterable[dict]) -> None:
        with SessionLocal() as db:
            for item in trades:
                db.add(
                    Trade(
                        session_id=session_id,
                        timestamp=item["timestamp"],
                        symbol=item["symbol"],
                        side=item["buy_sell"],
                        quantity=float(item["quantity"]),
                        price=float(item["price"]),
                        pnl=float(item.get("pnl", 0.0)),
                        holding_duration=float(item.get("holding_duration", 0.0)),
                    )
                )
            db.commit()

    def get_trades(self, session_id: str) -> list[Trade]:
        with SessionLocal() as db:
            return list(
                db.scalars(
                    select(Trade).where(Trade.session_id == session_id).order_by(Trade.timestamp.asc())
                ).all()
            )

    def get_modeling_trades(self, session_id: str) -> list[Trade]:
        with SessionLocal() as db:
            excluded = {
                row.trade_id
                for row in db.scalars(
                    select(EmergencyTrade).where(EmergencyTrade.session_id == session_id)
                ).all()
            }
            rows = list(
                db.scalars(select(Trade).where(Trade.session_id == session_id).order_by(Trade.timestamp.asc())).all()
            )
            return [r for r in rows if r.id not in excluded]

    def mark_emergency_trades(self, session_id: str, trade_ids: list[int], reason: str) -> None:
        with SessionLocal() as db:
            for trade_id in trade_ids:
                db.add(EmergencyTrade(session_id=session_id, trade_id=trade_id, reason=reason))
            session_row = db.get(SessionModel, session_id)
            if session_row:
                session_row.emergency_trade_count = len(trade_ids)
            db.commit()

    def save_market_context(self, session_id: str, payloads: Iterable[dict]) -> None:
        with SessionLocal() as db:
            for item in payloads:
                db.add(
                    MarketContext(
                        session_id=session_id,
                        date=item["date"],
                        symbol=item["symbol"],
                        context_json=json.dumps(item["context"]),
                    )
                )
            db.commit()

    def get_market_context(self, session_id: str) -> list[MarketContext]:
        with SessionLocal() as db:
            return list(
                db.scalars(
                    select(MarketContext)
                    .where(MarketContext.session_id == session_id)
                    .order_by(MarketContext.date.asc())
                ).all()
            )

    def save_analysis_result(self, session_id: str, model_name: str, result: dict) -> None:
        with SessionLocal() as db:
            db.add(
                AnalysisResult(
                    session_id=session_id,
                    model_name=model_name,
                    result_json=json.dumps(result, default=str),
                    computed_at=datetime.utcnow(),
                )
            )
            db.commit()

    def get_analysis_results(self, session_id: str) -> list[AnalysisResult]:
        with SessionLocal() as db:
            return list(db.scalars(select(AnalysisResult).where(AnalysisResult.session_id == session_id)).all())

    def save_chat_message(
        self, session_id: str, role: str, message: str, intent: str, chunks_used: list[str] | None = None
    ) -> None:
        with SessionLocal() as db:
            db.add(
                ConversationHistory(
                    session_id=session_id,
                    role=role,
                    message=message,
                    intent=intent,
                    chunks_used=json.dumps(chunks_used or []),
                )
            )
            db.commit()

    def get_recent_messages(self, session_id: str, limit: int = 10) -> list[ConversationHistory]:
        with SessionLocal() as db:
            rows = list(
                db.scalars(
                    select(ConversationHistory)
                    .where(ConversationHistory.session_id == session_id)
                    .order_by(ConversationHistory.timestamp.desc())
                    .limit(limit)
                ).all()
            )
            rows.reverse()
            return rows

    def save_counterfactual(self, session_id: str, payload: dict) -> None:
        with SessionLocal() as db:
            db.add(
                Counterfactual(
                    session_id=session_id,
                    scenario=json.dumps(payload.get("scenario", {})),
                    original_metrics=json.dumps(payload.get("original_metrics", {})),
                    counterfactual_metrics=json.dumps(payload.get("counterfactual_metrics", {})),
                    delta_pnl=float(payload.get("delta_pnl", 0.0)),
                )
            )
            db.commit()

    def get_counterfactuals(self, session_id: str) -> list[Counterfactual]:
        with SessionLocal() as db:
            return list(db.scalars(select(Counterfactual).where(Counterfactual.session_id == session_id)).all())
