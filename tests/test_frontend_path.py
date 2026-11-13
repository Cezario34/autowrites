from pathlib import Path

from app.infrastructure.static import resolve_frontend_directory


def test_prefers_frontend_inside_repository(tmp_path: Path):
    bundled = tmp_path / "frontend"
    bundled.mkdir()

    assert resolve_frontend_directory(tmp_path) == bundled


def test_page_saves_text_and_shows_plan_beside_names():
    page = Path("frontend/index.html").read_text(encoding="utf-8")
    assert "Сохранить как текст" in page
    assert "Дополнить сюжет и персонажей" in page
    assert "/supplement" in page
    assert "глава 1" in page
    assert "глава 2" in page
    assert "<h2>План</h2>" in page
    assert "Вывести сюжет" in page
    assert 'title.textContent = "Глава "' in page
    assert 'document.createElement("details")' in page
    assert 'document.createElement("summary")' in page
    assert "item.appearance" in page
    assert "Внешность" in page


def test_uses_sibling_frontend_when_repository_has_none(tmp_path: Path):
    repository = tmp_path / "autowrites"
    repository.mkdir()

    assert resolve_frontend_directory(repository) == tmp_path / "frontend"
