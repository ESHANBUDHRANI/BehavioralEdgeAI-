from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    uploads_dir: Path
    outputs_dir: Path
    cache_dir: Path
    models_pretrained_dir: Path
    static_knowledge_dir: Path
    sqlite_url: str
    newsapi_key: str
    cache_ttl_hours: int


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_settings() -> Settings:
    root = get_project_root()
    data_dir = root / "data"
    uploads = data_dir / "uploads"
    outputs = data_dir / "outputs"
    cache = data_dir / "cache"
    models_pretrained = root / "models_pretrained"
    static_knowledge = root / "static_knowledge"
    for path in [data_dir, uploads, outputs, cache, models_pretrained, static_knowledge]:
        path.mkdir(parents=True, exist_ok=True)
    db_path = root / "backend" / "database" / "app.db"
    sqlite_url = f"sqlite:///{db_path.as_posix()}"
    return Settings(
        project_root=root,
        data_dir=data_dir,
        uploads_dir=uploads,
        outputs_dir=outputs,
        cache_dir=cache,
        models_pretrained_dir=models_pretrained,
        static_knowledge_dir=static_knowledge,
        sqlite_url=sqlite_url,
        newsapi_key=os.getenv("NEWSAPI_KEY", ""),
        cache_ttl_hours=24,
    )


def get_device() -> str:
    try:
        import torch  # Local import to avoid hard dependency during DB init.

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def session_output_dir(session_id: str) -> Path:
    settings = get_settings()
    path = settings.outputs_dir / session_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "charts").mkdir(parents=True, exist_ok=True)
    return path
