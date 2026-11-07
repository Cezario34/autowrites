import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.application.books import create_book
from app.infrastructure.books import book_store
from app.infrastructure.ollama import ollama_client
from app.infrastructure.vector_store import vector_store
from app.main import app
from app.services.embedding_service import embedding_service

client = TestClient(app)


@pytest.fixture
def book(tmp_path, monkeypatch):
    monkeypatch.setattr(book_store, "path", tmp_path / "books.json")
    return create_book("Роман")


def _patch_pipeline(monkeypatch, stream):
    monkeypatch.setattr(
        embedding_service,
        "embed_batch",
        lambda texts: [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(embedding_service, "embed", lambda text: [0.1, 0.2])
    monkeypatch.setattr(vector_store, "add", lambda **kwargs: None)
    monkeypatch.setattr(
        vector_store,
        "query",
        lambda *args, **kwargs: {"documents": [["ранее"]]},
    )
    monkeypatch.setattr(ollama_client, "generate_stream", stream)


def _events(response):
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_streams_stages_and_tokens(monkeypatch, caplog, book):
    saved = {}

    def add(**kwargs):
        saved["metadatas"] = kwargs["metadatas"]

    def query(*args, **kwargs):
        saved["where"] = kwargs.get("where")
        return {"documents": [["ранее"]]}

    def stream(prompt, system=None, temperature=0.3):
        assert "ранее" in prompt
        assert "Герой" in prompt
        assert "Роман" in prompt
        yield "Часть "
        yield "ответа"

    _patch_pipeline(monkeypatch, stream)
    monkeypatch.setattr(vector_store, "add", add)
    monkeypatch.setattr(vector_store, "query", query)
    caplog.set_level(logging.INFO, logger="app")

    response = client.post(
        "/api/process-chapter",
        json={
            "text": "  Герой вошёл в дом.  ",
            "book_id": book.id,
            "add_to_store": True,
        },
    )
    events = _events(response)
    messages = [event["message"] for event in events if event["type"] == "stage"]

    assert messages == [
        "Идёт обработка…",
        "Нарезаю главу…",
        "Считаю эмбеддинги…",
        "Сохраняю главу…",
        "Ищу контекст…",
        "Генерирую ответ…",
    ]
    assert "".join(event["text"] for event in events if event["type"] == "token") == (
        "Часть ответа"
    )
    meta = next(event for event in events if event["type"] == "meta")
    assert meta["book_id"] == book.id
    assert meta["book_title"] == "Роман"
    assert meta["context_chunks"] == ["ранее"]
    assert meta["chunks_added"] >= 1
    assert meta["original_length"] == len("  Герой вошёл в дом.  ")
    assert saved["where"] == {"book_id": book.id}
    assert saved["metadatas"][0]["book_id"] == book.id
    assert events[-1] == {"type": "done"}
    assert "этап: нарезка" in caplog.text
    assert "этап: эмбеддинги" in caplog.text
    assert "этап: запись в базу" in caplog.text
    assert "этап: поиск" in caplog.text
    assert "этап: генерация" in caplog.text
    assert "этап: готово" in caplog.text


def test_skips_store_stages_when_not_requested(monkeypatch, book):
    def stream(prompt, system=None, temperature=0.3):
        yield "ок"

    _patch_pipeline(monkeypatch, stream)
    response = client.post(
        "/api/process-chapter",
        json={"text": "Короткий абзац.", "book_id": book.id, "add_to_store": False},
    )
    messages = [
        event["message"] for event in _events(response) if event["type"] == "stage"
    ]
    assert "Нарезаю главу…" not in messages
    assert "Сохраняю главу…" not in messages
    assert messages[0] == "Идёт обработка…"
    assert "Генерирую ответ…" in messages


def test_empty_chapter_stays_bad_request(book):
    response = client.post(
        "/api/process-chapter",
        json={"text": "   ", "book_id": book.id},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Глава пустая"}


def test_model_error_is_logged_and_sent(monkeypatch, caplog, book):
    def stream(prompt, system=None, temperature=0.3):
        raise ConnectionError("ollama недоступна")
        yield ""

    _patch_pipeline(monkeypatch, stream)
    caplog.set_level(logging.INFO, logger="app")
    response = client.post(
        "/api/process-chapter",
        json={"text": "Текст главы.", "book_id": book.id, "add_to_store": False},
    )
    events = _events(response)
    assert events[-1] == {"type": "error", "detail": "ollama недоступна"}
    assert "этап: ошибка" in caplog.text


def test_check_text_is_unchanged():
    response = client.post("/api/check-text", json={"text": "  hello  "})
    assert response.status_code == 200
    assert response.json()["processed_text"] == "hello"
    assert response.json()["original_length"] == 9
