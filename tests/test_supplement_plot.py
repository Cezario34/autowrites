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


def test_saved_chapters_update_plot_then_characters(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    created = client.post("/api/books", json={"title": "Север"})
    book_id = created.json()["id"]
    folder = tmp_path / "books" / "Север"
    (folder / "глава 1.txt").write_text(
        "Ульрика падает с башни и ломает платье.",
        encoding="utf-8",
    )
    (folder / "глава 2.txt").write_text(
        "Хаген уходит бить пещерных кошек.",
        encoding="utf-8",
    )
    calls = []

    def complete_json(system, user):
        calls.append(user)
        if "краткий сюжет" in system:
            if "Это глава 1." in user:
                return json.dumps(
                    {"сюжет": [{"глава": "1", "описание": "Ульрика падает с башни."}]},
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "сюжет": [
                        {"глава": "1", "описание": "Ульрика падает с башни."},
                        {"глава": "Глава 2", "описание": "Хаген уходит бить кошек."},
                    ]
                },
                ensure_ascii=False,
            )
        if "ломает платье" in user:
            return json.dumps(
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
            )
        return json.dumps(
            {
                "персонажи": [
                    {
                        "имя": "Ульрика",
                        "роль": "героиня",
                        "описание": "Упала с башни.",
                        "внешность": "",
                    },
                    {
                        "имя": "Хаген",
                        "роль": "конунг",
                        "описание": "Ушёл бить кошек.",
                        "внешность": "",
                    },
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(openai_client, "complete_json", complete_json)
    saved = client.post(f"/api/books/{book_id}/supplement")
    assert saved.status_code == 200
    body = saved.json()
    assert [item["number"] for item in body["chapters"]] == ["1", "2"]
    assert [item["plot_updated"] for item in body["chapters"]] == [True, True]
    assert [item["characters_updated"] for item in body["chapters"]] == [True, True]
    assert "Это глава 1." in calls[0]
    assert "ломает платье" in calls[1]
    assert "Это глава 2." in calls[2]
    assert "пещерных кошек" in calls[3]
    plot = json.loads((folder / "plot.json").read_text(encoding="utf-8"))
    assert plot == {
        "сюжет": [
            {"глава": "1", "описание": "Ульрика падает с башни."},
            {"глава": "2", "описание": "Хаген уходит бить кошек."},
        ]
    }
    assert "ломает платье" not in (folder / "plot.json").read_text(encoding="utf-8")
    names = [item["name"] for item in body["characters"]]
    assert names == ["Ульрика", "Хаген"]

    before_plot = (folder / "plot.json").read_text(encoding="utf-8")
    before_people = (folder / "characters.json").read_text(encoding="utf-8")
    calls.clear()
    again = client.post(f"/api/books/{book_id}/supplement")
    assert again.status_code == 200
    assert [item["number"] for item in again.json()["chapters"]] == ["2"]
    assert again.json()["chapters"][0]["plot_updated"] is False
    assert again.json()["chapters"][0]["characters_updated"] is False
    assert len(calls) == 2
    assert "Это глава 2." in calls[0]
    assert (folder / "plot.json").read_text(encoding="utf-8") == before_plot
    assert (folder / "characters.json").read_text(encoding="utf-8") == before_people


def test_plot_rewrite_does_not_touch_files_or_characters(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    created = client.post("/api/books", json={"title": "Юг"})
    book_id = created.json()["id"]
    folder = tmp_path / "books" / "Юг"
    (folder / "plot.json").write_text(
        json.dumps(
            {"сюжет": [{"глава": "1", "описание": "Ульрика падает с башни."}]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (folder / "глава 2.txt").write_text("Хаген уходит.", encoding="utf-8")
    stored_plot = (folder / "plot.json").read_text(encoding="utf-8")
    stored_people = (folder / "characters.json").read_text(encoding="utf-8")
    calls = {"n": 0}

    def complete_json(system, user):
        calls["n"] += 1
        calls["user"] = user
        return json.dumps(
            {"сюжет": [{"глава": "2", "описание": "Хаген уходит."}]},
            ensure_ascii=False,
        )

    monkeypatch.setattr(openai_client, "complete_json", complete_json)
    response = client.post(f"/api/books/{book_id}/supplement")
    assert response.status_code == 400
    assert "не трогал" in response.json()["detail"]
    assert calls["n"] == 3
    assert "Пропала глава 1" in calls["user"]
    assert "файл не менялся" in calls["user"]
    assert (folder / "plot.json").read_text(encoding="utf-8") == stored_plot
    assert (folder / "characters.json").read_text(encoding="utf-8") == stored_people


def test_supplement_asks_to_save_a_chapter_file_first(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    created = client.post("/api/books", json={"title": "Восток"})
    book_id = created.json()["id"]

    def complete_json(system, user):
        raise AssertionError("вызов не нужен")

    monkeypatch.setattr(openai_client, "complete_json", complete_json)
    response = client.post(f"/api/books/{book_id}/supplement")
    assert response.status_code == 400
    assert response.json()["detail"] == "Сначала сохрани главу как текст"
