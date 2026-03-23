from __future__ import annotations

from datetime import datetime, date
from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SessionModel(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    filename: Mapped[str] = mapped_column(String, default="")
    risk_profile_label: Mapped[str] = mapped_column(String, default="unknown")
    behavioral_profile_json: Mapped[str] = mapped_column(Text, default="{}")
    emergency_trade_count: Mapped[int] = mapped_column(Integer, default=0)
    analysis_status: Mapped[str] = mapped_column(String, default="uploaded")


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    side: Mapped[str] = mapped_column(String)
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)
    holding_duration: Mapped[float] = mapped_column(Float, default=0.0)
    cluster_label: Mapped[int] = mapped_column(Integer, default=-1)
    emotional_state: Mapped[str] = mapped_column(String, default="unknown")
    anomaly_flag: Mapped[int] = mapped_column(Integer, default=0)
    anomaly_score: Mapped[float] = mapped_column(Float, default=0.0)


class EmergencyTrade(Base):
    __tablename__ = "emergency_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), index=True)
    trade_id: Mapped[int] = mapped_column(ForeignKey("trades.id"), index=True)
    reason: Mapped[str] = mapped_column(String, default="financial_emergency")


class MarketContext(Base):
    __tablename__ = "market_context"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    context_json: Mapped[str] = mapped_column(Text)


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), index=True)
    model_name: Mapped[str] = mapped_column(String, index=True)
    result_json: Mapped[str] = mapped_column(Text)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ConversationHistory(Base):
    __tablename__ = "conversation_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), index=True)
    role: Mapped[str] = mapped_column(String)
    message: Mapped[str] = mapped_column(Text)
    intent: Mapped[str] = mapped_column(String, default="general_explanation")
    chunks_used: Mapped[str] = mapped_column(Text, default="[]")
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Counterfactual(Base):
    __tablename__ = "counterfactuals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), index=True)
    scenario: Mapped[str] = mapped_column(Text)
    original_metrics: Mapped[str] = mapped_column(Text)
    counterfactual_metrics: Mapped[str] = mapped_column(Text)
    delta_pnl: Mapped[float] = mapped_column(Float, default=0.0)
