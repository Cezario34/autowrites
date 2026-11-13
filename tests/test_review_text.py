import json

from fastapi.testclient import TestClient

from app.core.config import settings
from app.infrastructure.book_files import library
from app.infrastructure.books import book_store
from app.infrastructure.openai_client import openai_client
from app.main import app

client = TestClient(app)


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(book_store, "path", tmp_path / "books.json")
    monkeypatch.setattr(library, "root", tmp_path / "books")
    monkeypatch.setattr(settings, "openai_api_key", "test-key")


def test_review_sends_the_pasted_text_with_both_json_files(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    created = client.post("/api/books", json={"title": "Север"})
    book_id = created.json()["id"]
    folder = tmp_path / "books" / "Север"
    (folder / "characters.json").write_text(
        json.dumps(
            {
                "персонажи": [
                    {
                        "имя": "Ульрика",
                        "роль": "героиня",
                        "описание": "Упала с башни.",
                        "внешность": "",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (folder / "plot.json").write_text(
        json.dumps(
            {"сюжет": [{"глава": "1", "описание": "Ульрика падает с башни."}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    seen = {}

    def complete(system, user, json_mode=False, temperature=0.2):
        seen["system"] = system
        seen["user"] = user
        seen["json_mode"] = json_mode
        seen["temperature"] = temperature
        return "Расхождения с сюжетом. Хаген уже в форте, а в тексте он ещё в пути."

    monkeypatch.setattr(openai_client, "complete", complete)
    chapter = "Хаген только выезжает из леса."
    response = client.post(
        f"/api/books/{book_id}/review",
        json={"text": chapter},
    )
    assert response.status_code == 200
    assert response.json()["review"].startswith("Расхождения с сюжетом")
    assert chapter in seen["user"]
    assert "Ульрика" in seen["user"]
    assert "падает с башни" in seen["user"]
    assert seen["json_mode"] is False
    assert "слабые места" in seen["system"].casefold()
    assert "Расхождения с сюжетом" in seen["system"]


def test_empty_review_text_does_not_call_openai(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    created = client.post("/api/books", json={"title": "Юг"})
    book_id = created.json()["id"]

    def complete(*args, **kwargs):
        raise AssertionError("вызов не нужен")

    monkeypatch.setattr(openai_client, "complete", complete)
    response = client.post(f"/api/books/{book_id}/review", json={"text": "   "})
    assert response.status_code == 400
    assert response.json()["detail"] == "Глава пустая"
