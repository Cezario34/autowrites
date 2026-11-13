import json

from fastapi.testclient import TestClient

from app.core.config import settings
from app.infrastructure import style_examples
from app.infrastructure.book_files import library
from app.infrastructure.books import book_store
from app.infrastructure.openai_client import openai_client
from app.infrastructure.vector_store import vector_store
from app.main import app
from app.services.rag_service import rag_service

client = TestClient(app)


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(book_store, "path", tmp_path / "books.json")
    monkeypatch.setattr(library, "root", tmp_path / "books")
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    examples = tmp_path / "examples"
    examples.mkdir()
    monkeypatch.setattr(style_examples, "DIRECTORY", examples)
    return examples


def test_all_fragments_returns_every_piece_in_order(monkeypatch):
    def documents_where(where):
        assert where == {"book_id": "book"}
        return [
            ("второй кусок", {"kind": "fragment", "chapter_id": "a", "chunk_index": 1}),
            ("карточка", {"kind": "character", "chapter_id": "a", "chunk_index": 0}),
            ("первый кусок", {"kind": "fragment", "chapter_id": "a", "chunk_index": 0}),
            ("", {"kind": "fragment", "chapter_id": "b", "chunk_index": 0}),
            ("из другой главы", {"kind": "fragment", "chapter_id": "b", "chunk_index": 2}),
        ]

    monkeypatch.setattr(vector_store, "documents_where", documents_where)
    assert rag_service.all_fragments("book") == [
        "первый кусок",
        "второй кусок",
        "из другой главы",
    ]


def test_suggest_sends_all_fragments_examples_and_sol_model(tmp_path, monkeypatch):
    examples = _isolate(tmp_path, monkeypatch)
    (examples / "диалоги.txt").write_text("— Куда ты?\n— В башню.\n", encoding="utf-8")
    (examples / "описания.txt").write_text("Ветер нёс соль по камням.\n", encoding="utf-8")
    created = client.post("/api/books", json={"title": "Форт"})
    book_id = created.json()["id"]
    folder = tmp_path / "books" / "Форт"
    (folder / "characters.json").write_text(
        json.dumps(
            {
                "персонажи": [
                    {
                        "имя": "Марта",
                        "роль": "жительница форта",
                        "описание": "Завела Ульрику в башню.",
                        "внешность": "пустой взгляд",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (folder / "plot.json").write_text(
        json.dumps(
            {"сюжет": [{"глава": "1", "описание": "Ульрика входит в форт."}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (folder / "plan.txt").write_text("Дойти до жилой башни.", encoding="utf-8")
    (folder / "глава 1.txt").write_text("Ульрика вошла в ворота.", encoding="utf-8")
    monkeypatch.setattr(
        rag_service,
        "all_fragments",
        lambda book_id: ["Ульрика шагнула с башни.", "Хаген ждал внизу."],
    )
    seen = {}

    def complete(system, user, *, json_mode=False, temperature=0.2, model=None):
        seen["system"] = system
        seen["user"] = user
        seen["json_mode"] = json_mode
        seen["temperature"] = temperature
        seen["model"] = model
        return "Марта открыла дверь башни."

    monkeypatch.setattr(openai_client, "complete", complete)
    response = client.post(
        f"/api/books/{book_id}/suggest",
        json={"prompt": "Ульрика остаётся в башне на ночь", "text": "Она стояла у порога."},
    )
    assert response.status_code == 200
    assert response.json()["chapter"] == "Марта открыла дверь башни."
    assert response.json()["model"] == "gpt-5.6"
    assert seen["model"] == "gpt-5.6"
    assert seen["temperature"] is None
    assert seen["json_mode"] is False
    assert "Ульрика остаётся в башне на ночь" in seen["user"]
    assert "Она стояла у порога." in seen["user"]
    assert "Ульрика входит в форт." in seen["user"]
    assert "Марта" in seen["user"]
    assert "Дойти до жилой башни." in seen["user"]
    assert "Ульрика вошла в ворота." in seen["user"]
    assert "Ульрика шагнула с башни." in seen["user"]
    assert "Хаген ждал внизу." in seen["user"]
    assert "— Куда ты?" in seen["user"]
    assert "Ветер нёс соль по камням." in seen["user"]
    assert "только текст главы" in seen["system"]
    assert "5000" in seen["system"]


def test_suggested_chapter_is_cut_to_five_thousand_characters(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    created = client.post("/api/books", json={"title": "Север"})
    book_id = created.json()["id"]
    monkeypatch.setattr(rag_service, "all_fragments", lambda book_id: [])

    def complete(system, user, *, json_mode=False, temperature=0.2, model=None):
        return ("Ульрика шла по двору. " * 400).strip()

    monkeypatch.setattr(openai_client, "complete", complete)
    response = client.post(
        f"/api/books/{book_id}/suggest",
        json={"prompt": "Довести её до башни"},
    )
    assert response.status_code == 200
    chapter = response.json()["chapter"]
    assert len(chapter) <= 5000
    assert chapter.endswith(".")
    assert response.json()["model"] == "gpt-5.6"


def test_empty_direction_does_not_call_openai(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    created = client.post("/api/books", json={"title": "Юг"})
    book_id = created.json()["id"]

    def complete(*args, **kwargs):
        raise AssertionError("вызов не нужен")

    monkeypatch.setattr(openai_client, "complete", complete)
    response = client.post(
        f"/api/books/{book_id}/suggest",
        json={"prompt": "   "},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Напиши, куда вести сюжет"
