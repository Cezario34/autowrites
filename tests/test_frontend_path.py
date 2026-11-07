from pathlib import Path

from app.infrastructure.static import resolve_frontend_directory


def test_prefers_frontend_inside_repository(tmp_path: Path):
    bundled = tmp_path / "frontend"
    bundled.mkdir()

    assert resolve_frontend_directory(tmp_path) == bundled


def test_uses_sibling_frontend_when_repository_has_none(tmp_path: Path):
    repository = tmp_path / "autowrites"
    repository.mkdir()

    assert resolve_frontend_directory(repository) == tmp_path / "frontend"
