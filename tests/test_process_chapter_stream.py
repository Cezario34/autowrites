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


def test_check_uses_existing_rag_and_does_not_save(monkeypatch, caplog, book):
    saved = {"added": False}

    def add(**kwargs):
        saved["added"] = True

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
        "/api/check-chapter",
        json={"text": "  Герой вошёл в дом.  ", "book_id": book.id},
    )
    events = _events(response)
    messages = [event["message"] for event in events if event["type"] == "stage"]

    assert messages == [
        "Идёт проверка…",
        "Ищу в книге…",
        "Генерирую ответ…",
    ]
    assert "".join(event["text"] for event in events if event["type"] == "token") == (
        "Часть ответа"
    )
    meta = next(event for event in events if event["type"] == "meta")
    assert meta["book_id"] == book.id
    assert meta["book_title"] == "Роман"
    assert meta["context_chunks"] == ["ранее"]
    assert meta["original_length"] == len("  Герой вошёл в дом.  ")
    assert "chunks_added" not in meta
    assert saved["where"] == {"book_id": book.id}
    assert saved["added"] is False
    assert events[-1] == {"type": "done"}
    assert "этап: поиск" in caplog.text
    assert "этап: генерация" in caplog.text
    assert "этап: запись в базу" not in caplog.text


def test_save_stores_model_fragments_without_analysis(monkeypatch, caplog, book):
    saved = {}

    def add(**kwargs):
        saved["documents"] = kwargs["documents"]
        saved["metadatas"] = kwargs["metadatas"]

    def generate(prompt, system=None, temperature=0.3):
        saved["system"] = system
        saved["temperature"] = temperature
        assert "Герой вошёл" in prompt
        assert "Не анализируй" in prompt
        return '["Герой вошёл в дом.", "Он закрыл дверь."]'

    def stream(*args, **kwargs):
        raise AssertionError("при сохранении разбор не нужен")

    def query(*args, **kwargs):
        raise AssertionError("при сохранении поиск не нужен")

    _patch_pipeline(monkeypatch, stream)
    monkeypatch.setattr(ollama_client, "generate", generate)
    monkeypatch.setattr(vector_store, "add", add)
    monkeypatch.setattr(vector_store, "query", query)
    caplog.set_level(logging.INFO, logger="app")

    response = client.post(
        "/api/save-chapter",
        json={"text": "Герой вошёл в дом. Он закрыл дверь.", "book_id": book.id},
    )
    events = _events(response)
    messages = [event["message"] for event in events if event["type"] == "stage"]
    assert messages == [
        "Идёт сохранение…",
        "Модель разбивает текст…",
        "Считаю эмбеддинги…",
        "Сохраняю фрагменты…",
    ]
    assert not any(event["type"] == "token" for event in events)
    stored = next(event for event in events if event["type"] == "saved")
    assert stored["fragments"] == ["Герой вошёл в дом.", "Он закрыл дверь."]
    assert stored["chunks_added"] == 2
    assert stored["book_title"] == "Роман"
    assert saved["documents"] == stored["fragments"]
    assert [item["book_id"] for item in saved["metadatas"]] == [book.id, book.id]
    assert "JSON-массивом" in saved["system"]
    assert saved["temperature"] == 0.1
    assert events[-1] == {"type": "done"}
    assert "этап: разбиение" in caplog.text
    assert "этап: запись в базу" in caplog.text


def test_save_reports_unparsed_model_output(monkeypatch, book):
    def generate(prompt, system=None, temperature=0.3):
        return "Здесь есть противоречие, но это не JSON."

    def add(**kwargs):
        raise AssertionError("битый ответ нельзя сохранять")

    _patch_pipeline(monkeypatch, lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(ollama_client, "generate", generate)
    monkeypatch.setattr(vector_store, "add", add)

    response = client.post(
        "/api/save-chapter",
        json={"text": "Текст главы.", "book_id": book.id},
    )
    events = _events(response)
    assert events[-1] == {"type": "error", "detail": "Модель вернула не список фрагментов"}


def test_empty_chapter_stays_bad_request(book):
    for path in ("/api/check-chapter", "/api/save-chapter"):
        response = client.post(path, json={"text": "   ", "book_id": book.id})
        assert response.status_code == 400
        assert response.json() == {"detail": "Глава пустая"}


def test_model_error_is_logged_and_sent(monkeypatch, caplog, book):
    def stream(prompt, system=None, temperature=0.3):
        raise ConnectionError("ollama недоступна")
        yield ""

    _patch_pipeline(monkeypatch, stream)
    caplog.set_level(logging.INFO, logger="app")
    response = client.post(
        "/api/check-chapter",
        json={"text": "Текст главы.", "book_id": book.id},
    )
    events = _events(response)
    assert events[-1] == {"type": "error", "detail": "ollama недоступна"}
    assert "этап: ошибка" in caplog.text


def test_check_text_is_unchanged():
    response = client.post("/api/check-text", json={"text": "  hello  "})
    assert response.status_code == 200
    assert response.json()["processed_text"] == "hello"
    assert response.json()["original_length"] == 9
