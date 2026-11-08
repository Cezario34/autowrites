from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles


class FrontendFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        url_path = scope.get("path") or ""
        if url_path == "/api" or url_path.startswith("/api/"):
            raise HTTPException(status_code=404, detail="Маршрут не найден")
        return await super().get_response(path, scope)


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
            FrontendFiles(directory=str(frontend_path), html=True),
            name="frontend",
        )
