from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def frontend_directory() -> Path:
    """Frontend lives beside the repository, not inside it."""
    repository_root = Path(__file__).resolve().parents[2]
    return repository_root.parent / "frontend"


def mount_frontend(app: FastAPI) -> None:
    # Статика — фронт
    frontend_path = frontend_directory()
    if frontend_path.exists():
        app.mount(
            "/",
            StaticFiles(directory=str(frontend_path), html=True),
            name="frontend",
        )
