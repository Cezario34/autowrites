from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def resolve_frontend_directory(repository_root: Path) -> Path:
    """Каталог frontend внутри репозитория, иначе соседний с репозиторием."""
    bundled = repository_root / "frontend"
    if bundled.is_dir():
        return bundled
    return repository_root.parent / "frontend"


def frontend_directory() -> Path:
    repository_root = Path(__file__).resolve().parents[2]
    return resolve_frontend_directory(repository_root)


def mount_frontend(app: FastAPI) -> None:
    # Статика — фронт
    frontend_path = frontend_directory()
    if frontend_path.exists():
        app.mount(
            "/",
            StaticFiles(directory=str(frontend_path), html=True),
            name="frontend",
        )
