from fastapi.testclient import TestClient

from app.core.config import settings
from app.infrastructure.books import book_store
from app.infrastructure.vector_store import VectorStore
from app.main import app

client = TestClient(app)


def test_create_list_and_reject_blank_title(tmp_path, monkeypatch):
    monkeypatch.setattr(book_store, "path", tmp_path / "books.json")

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


def test_unknown_book_is_rejected_before_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(book_store, "path", tmp_path / "books.json")
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
