from pathlib import Path

from app.infrastructure.static import resolve_frontend_directory


def test_prefers_frontend_inside_repository(tmp_path: Path):
    bundled = tmp_path / "frontend"
    bundled.mkdir()

    assert resolve_frontend_directory(tmp_path) == bundled


def test_page_saves_text_and_shows_plan_beside_names():
    page = Path("frontend/index.html").read_text(encoding="utf-8")
    assert "Сохранить и дополнить" in page
    assert "Проверить текст" in page
    assert "Предложить главу" in page
    assert "/suggest" in page
    assert "GPT-5.6" in page
    assert "длиннее 5 000 и короче 10 000 символов" in page
    assert "GPT-6.1 Sol" not in page
    assert "Проверить по книге" not in page
    assert "Сохранить в базу" not in page
    assert "/api/check-chapter" in page
    assert "/api/save-chapter" in page
    assert "/review" in page
    assert "/supplement" in page
    assert "глава 1" in page
    assert "глава 2" in page
    assert "<h2>План</h2>" in page
    assert "Вывести сюжет" in page
    assert ">Удалить</button>" in page
    assert 'method: "DELETE"' in page
    assert 'title.textContent = "Глава "' in page
    assert 'document.createElement("details")' in page
    assert 'document.createElement("summary")' in page
    assert "item.appearance" in page
    assert "Внешность" in page


def test_uses_sibling_frontend_when_repository_has_none(tmp_path: Path):
    repository = tmp_path / "autowrites"
    repository.mkdir()

    assert resolve_frontend_directory(repository) == tmp_path / "frontend"
