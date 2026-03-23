from __future__ import annotations

from backend.database.repository import Repository


def save_message(session_id: str, role: str, message: str, intent: str, chunks_used: list[str] | None = None) -> None:
    Repository().save_chat_message(session_id, role, message, intent, chunks_used)


def get_recent(session_id: str, limit: int = 10) -> list[dict]:
    rows = Repository().get_recent_messages(session_id, limit=limit)
    return [
        {
            "role": r.role,
            "message": r.message,
            "intent": r.intent,
            "chunks_used": r.chunks_used,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
        }
        for r in rows
    ]


def test_memory() -> dict:
    return {"ok": True, "message": "chat memory wired"}
