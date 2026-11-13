import json

from fastapi.testclient import TestClient

from app.core.config import settings
from app.infrastructure.book_files import library
from app.infrastructure.books import book_store
from app.infrastructure.vector_store import VectorStore
from app.main import app

client = TestClient(app)


def _isolate_book_files(tmp_path, monkeypatch):
    monkeypatch.setattr(book_store, "path", tmp_path / "books.json")
    monkeypatch.setattr(library, "root", tmp_path / "books")


def test_create_list_and_reject_blank_title(tmp_path, monkeypatch):
    _isolate_book_files(tmp_path, monkeypatch)

    blank = client.post("/api/books", json={"title": "   "})
    assert blank.status_code == 400
    assert blank.json() == {"detail": "Название книги пустое"}

    too_long = client.post("/api/books", json={"title": "А" * 201})
    assert too_long.status_code == 400
    assert too_long.json() == {"detail": "Название книги слишком длинное"}

    first = client.post("/api/books", json={"title": "  Первая  "})
    assert first.status_code == 201
    assert first.json()["title"] == "Первая"
    assert first.json()["chunk_count"] == 0

    second = client.post("/api/books", json={"title": "Вторая"})
    assert second.status_code == 201

    listed = client.get("/api/books")
    assert listed.status_code == 200
    assert [item["title"] for item in listed.json()] == ["Вторая", "Первая"]
    for title in ("Первая", "Вторая"):
        folder = tmp_path / "books" / title
        assert (folder / "plan.txt").read_text(encoding="utf-8") == ""
        assert (folder / "text.txt").read_text(encoding="utf-8") == ""
        assert json.loads((folder / "characters.json").read_text(encoding="utf-8")) == {
            "characters": []
        }


def test_unknown_book_has_no_character_list(tmp_path, monkeypatch):
    _isolate_book_files(tmp_path, monkeypatch)
    response = client.get("/api/books/missing-book/characters")
    assert response.status_code == 400
    assert response.json() == {"detail": "Книга не найдена"}


def test_each_book_has_its_own_character_file_and_a_plan(tmp_path, monkeypatch):
    _isolate_book_files(tmp_path, monkeypatch)
    created = client.post("/api/books", json={"title": "Север"})
    book_id = created.json()["id"]
    folder = tmp_path / "books" / "Север"
    assert (folder / "characters.json").is_file()
    assert (folder / "plan.txt").read_text(encoding="utf-8") == ""
    assert (folder / "text.txt").read_text(encoding="utf-8") == ""
    (folder / "characters.json").write_text(
        json.dumps(
            {
                "characters": [
                    {"name": "Ингрид", "description": "Леди."},
                    {"name": "Ульрик", "description": "Конунг."},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (folder / "plan.txt").write_text(
        "1. Башня\n2. Конунг",
        encoding="utf-8",
    )
    other = client.post("/api/books", json={"title": "Юг"})

    characters = client.get(f"/api/books/{book_id}/characters")
    assert characters.status_code == 200
    assert [item["name"] for item in characters.json()["characters"]] == [
        "Ингрид",
        "Ульрик",
    ]
    assert client.get(f"/api/books/{other.json()['id']}/characters").json() == {
        "characters": []
    }
    plan = client.get(f"/api/books/{book_id}/plan")
    assert plan.status_code == 200
    assert plan.json() == {"text": "1. Башня\n2. Конунг"}
    other_folder = tmp_path / "books" / "Юг"
    assert (other_folder / "plan.txt").read_text(encoding="utf-8") == ""
    assert (other_folder / "text.txt").read_text(encoding="utf-8") == ""
    assert json.loads((other_folder / "characters.json").read_text(encoding="utf-8")) == {
        "characters": []
    }


def test_save_as_text_writes_the_whole_chapter(tmp_path, monkeypatch):
    _isolate_book_files(tmp_path, monkeypatch)
    created = client.post("/api/books", json={"title": "Север: быт"})
    book_id = created.json()["id"]
    chapter = "Первая строка.\n\nВторая строка."

    saved = client.post(f"/api/books/{book_id}/text", json={"text": chapter})
    assert saved.status_code == 200
    assert saved.text == chapter
    assert saved.headers["content-type"].startswith("text/plain")
    assert "filename*=UTF-8''" in saved.headers["content-disposition"]
    assert (tmp_path / "books" / "Север быт" / "text.txt").read_text(encoding="utf-8") == chapter

    empty = client.post(f"/api/books/{book_id}/text", json={"text": "   "})
    assert empty.status_code == 400
    assert empty.json() == {"detail": "Глава пустая"}


def test_unknown_book_is_rejected_before_generation(tmp_path, monkeypatch):
    _isolate_book_files(tmp_path, monkeypatch)
    called = {"stream": False}

    def stream(*args, **kwargs):
        called["stream"] = True
        yield "не должно"

    from app.infrastructure.ollama import ollama_client

    monkeypatch.setattr(ollama_client, "generate_stream", stream)
    for path in ("/api/check-chapter", "/api/save-chapter"):
        response = client.post(
            path,
            json={"text": "Текст главы.", "book_id": "missing-book"},
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "Книга не найдена"}
    assert called["stream"] is False


def test_search_stays_inside_the_selected_book(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    store = VectorStore()
    store.add(
        ids=["a-0", "b-0"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["только первая книга", "только вторая книга"],
        metadatas=[
            {"book_id": "book-a", "chapter_id": "c1", "chunk_index": 0},
            {"book_id": "book-b", "chapter_id": "c2", "chunk_index": 0},
        ],
    )

    first = store.query([1.0, 0.0], n_results=5, where={"book_id": "book-a"})
    second = store.query([1.0, 0.0], n_results=5, where={"book_id": "book-b"})
    empty = store.query([1.0, 0.0], n_results=5, where={"book_id": "book-c"})

    assert first["documents"] == [["только первая книга"]]
    assert second["documents"] == [["только вторая книга"]]
    assert empty["documents"] == [[]]
    assert store.count_where({"book_id": "book-a"}) == 1
    assert store.count_where({"book_id": "book-c"}) == 0
