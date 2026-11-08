import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.application.books import create_book
from app.infrastructure.books import book_store
from app.infrastructure.characters import character_store
from app.infrastructure.ollama import ollama_client
from app.infrastructure.vector_store import vector_store
from app.main import app
from app.services.embedding_service import embedding_service
from app.services.rag_service import rag_service

client = TestClient(app)


@pytest.fixture
def book(tmp_path, monkeypatch):
    monkeypatch.setattr(book_store, "path", tmp_path / "books.json")
    monkeypatch.setattr(character_store, "path", tmp_path / "characters.json")
    return create_book("Роман")


def _patch_pipeline(monkeypatch, stream):
    monkeypatch.setattr(
        embedding_service,
        "embed_batch",
        lambda texts: [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(embedding_service, "embed", lambda text: [0.1, 0.2])
    monkeypatch.setattr(vector_store, "add", lambda **kwargs: None)
    monkeypatch.setattr(vector_store, "upsert", lambda **kwargs: None)
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

    prompts = []

    def stream(prompt, system=None, temperature=0.3, response_format=None, num_predict=None):
        prompts.append((prompt, system, temperature, num_predict))
        yield "Герой вошёл в дом. Часть "
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
        "Сверяю с сохранённым текстом…",
        "Проверяю логику нового отрывка…",
    ]
    answer = "".join(event["text"] for event in events if event["type"] == "token")
    assert answer.startswith(
        "По уже сохранённому тексту (фрагментов из базы: 1)\n\nГерой вошёл в дом. Часть ответа"
    )
    assert "\n\nЛогика нового отрывка\n\nГерой вошёл в дом. Часть ответа" in answer
    assert "Вы прислали" not in answer
    rag_prompt, logic_prompt = prompts
    assert "ранее" in rag_prompt[0]
    assert "Герой" in rag_prompt[0]
    assert "Роман" in rag_prompt[0]
    assert "не цитируй" in rag_prompt[0]
    assert "ранее" not in logic_prompt[0]
    assert "Герой" in logic_prompt[0]
    assert "внутреннюю логику" in logic_prompt[0]
    assert "не цитируй" in logic_prompt[0]
    assert rag_prompt[2] == logic_prompt[2] == 0.3
    assert rag_prompt[3] == logic_prompt[3] == 600
    assert rag_prompt[1].startswith("Ты проверяешь присланный отрывок")
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
    assert "этап: сверка с базой" in caplog.text
    assert "этап: логика нового отрывка" in caplog.text
    assert "этап: запись в базу" not in caplog.text


def test_model_html_error_does_not_reach_the_page(monkeypatch, book):
    def stream(*args, **kwargs):
        raise RuntimeError(
            "<!DOCTYPE HTML><html><body><h1>Error response</h1>"
            "<p>Error code: 404</p> Nothing matches the given URI.</body></html>"
            " (status code: 404)"
        )
        yield ""

    _patch_pipeline(monkeypatch, stream)
    monkeypatch.setattr(
        vector_store,
        "query",
        lambda *args, **kwargs: {"documents": [[]]},
    )

    response = client.post(
        "/api/check-chapter",
        json={"text": "Герой вошёл в дом.", "book_id": book.id},
    )
    events = _events(response)
    detail = events[-1]["detail"]
    assert events[-1]["type"] == "error"
    assert "404" in detail
    assert "<html" not in detail.casefold()
    assert "<!doctype" not in detail.casefold()


def test_check_does_not_treat_a_stock_reply_as_analysis(monkeypatch, book):
    def stream(prompt, system=None, temperature=0.3, response_format=None, num_predict=None):
        assert "fsdf" in prompt
        assert num_predict == 600
        yield "Противоречий нет. Логика держится."

    _patch_pipeline(monkeypatch, stream)
    monkeypatch.setattr(
        vector_store,
        "query",
        lambda *args, **kwargs: {"documents": [["ранее сохранённый кусок"]]},
    )

    response = client.post(
        "/api/check-chapter",
        json={"text": "fsdf", "book_id": book.id},
    )
    answer = "".join(event["text"] for event in _events(response) if event["type"] == "token")
    assert answer.count("Вы прислали: «fsdf»") == 2
    assert answer.count("Модель этот текст не разбирала.") == 2
    assert "[1] ранее сохранённый кусок" in answer
    assert "фрагментов из базы: 1)\n\nВы прислали: «fsdf»" in answer
    assert "Логика нового отрывка\n\nВы прислали: «fsdf»" in answer


def test_check_retries_until_the_reply_quotes_the_excerpt(monkeypatch, book):
    sent = "Герой вошёл в комнату. У него были зелёные глаза и он сказал, что никогда не был в этом городе."
    calls = {"n": 0}

    def stream(prompt, system=None, temperature=0.3, response_format=None, num_predict=None):
        calls["n"] += 1
        assert num_predict == 600
        assert sent in prompt
        assert sent not in (system or "")
        assert "Противоречий нет" not in prompt
        assert "Противоречий нет" not in (system or "")
        if calls["n"] % 2 == 1:
            yield "Противоречий нет. Логика держится."
            return
        yield (
            sent
            + " Это противоречит уже записанной сцене: герой там уже живёт и помнит этот город."
        )

    _patch_pipeline(monkeypatch, stream)
    monkeypatch.setattr(
        vector_store,
        "query",
        lambda *args, **kwargs: {"documents": [["ранее"]]},
    )

    response = client.post(
        "/api/check-chapter",
        json={"text": sent, "book_id": book.id},
    )
    answer = "".join(event["text"] for event in _events(response) if event["type"] == "token")
    assert calls["n"] == 4
    assert "Вы прислали" not in answer
    assert "уже живёт и помнит этот город" in answer
    assert answer.count("Противоречий нет") == 0


def test_check_searches_the_whole_excerpt_and_skips_character_cards(monkeypatch, book):
    calls = []

    def query(embedding, n_results=None, where=None):
        calls.append(where)
        return {
            "documents": [["фрагмент сюжета", "Персонаж: Анна\nЖдёт."]],
            "metadatas": [[{"kind": "fragment"}, {"kind": "character"}]],
        }

    monkeypatch.setattr(vector_store, "query", query)
    monkeypatch.setattr(embedding_service, "embed", lambda text: [len(text)])

    chunks = rag_service.saved_fragments("А" * 2000 + "Б" * 2000 + "В" * 2000, book.id)

    assert chunks == ["фрагмент сюжета"]
    assert len(calls) == 3
    assert calls[0] == {"book_id": book.id}


def _save_reply(prompt, system=None, temperature=0.3, people=None):
    if "персонаж" in prompt.casefold():
        body = people
        if body is None:
            body = [{"name": "Герой", "description": "Вошёл в дом и закрыл дверь."}]
        return json.dumps(body, ensure_ascii=False)
    assert "Не анализируй" in prompt
    return '["Герой вошёл в дом.", "Он закрыл дверь."]'


def test_save_stores_fragments_and_character_cards(monkeypatch, caplog, book):
    saved = {"adds": [], "upserts": [], "systems": []}

    def add(**kwargs):
        saved["adds"].append(kwargs)

    def upsert(**kwargs):
        saved["upserts"].append(kwargs)

    def generate(prompt, system=None, temperature=0.3, response_format=None):
        saved["systems"].append(system)
        assert temperature == 0.1
        if "персонаж" in prompt.casefold():
            assert response_format == "json"
        else:
            assert response_format is None
        assert "Герой вошёл" in prompt
        return _save_reply(prompt, system, temperature)

    def stream(*args, **kwargs):
        raise AssertionError("при сохранении разбор не нужен")

    def query(*args, **kwargs):
        raise AssertionError("при сохранении поиск не нужен")

    _patch_pipeline(monkeypatch, stream)
    monkeypatch.setattr(ollama_client, "generate", generate)
    monkeypatch.setattr(vector_store, "add", add)
    monkeypatch.setattr(vector_store, "upsert", upsert)
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
        "Выписываю персонажей…",
        "Считаю эмбеддинги…",
        "Сохраняю фрагменты…",
    ]
    assert not any(event["type"] == "token" for event in events)
    stored = next(event for event in events if event["type"] == "saved")
    assert stored["fragments"] == ["Герой вошёл в дом.", "Он закрыл дверь."]
    assert stored["chunks_added"] == 2
    assert stored["characters"] == [
        {"name": "Герой", "description": "Вошёл в дом и закрыл дверь."}
    ]
    assert saved["adds"][0]["documents"] == stored["fragments"]
    assert [item["kind"] for item in saved["adds"][0]["metadatas"]] == [
        "fragment",
        "fragment",
    ]
    card = saved["upserts"][0]
    assert card["documents"] == ["Персонаж: Герой\nВошёл в дом и закрыл дверь."]
    assert card["metadatas"] == [
        {"book_id": book.id, "kind": "character", "name": "Герой"}
    ]
    assert "---" in saved["systems"][0]
    assert "JSON-массивом объектов" in saved["systems"][1]
    assert events[-1] == {"type": "done"}
    listed = client.get(f"/api/books/{book.id}/characters")
    assert listed.status_code == 200
    assert listed.json() == {"characters": stored["characters"]}
    assert "этап: персонажи" in caplog.text


def test_second_save_merges_character_json_and_database(monkeypatch, book):
    calls = {"people": []}

    def generate(prompt, system=None, temperature=0.3, response_format=None):
        people = calls["people"] or None
        return _save_reply(prompt, people=people)

    _patch_pipeline(monkeypatch, lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(ollama_client, "generate", generate)
    cards = {}

    def upsert(**kwargs):
        for item_id, document in zip(kwargs["ids"], kwargs["documents"]):
            cards[item_id] = document

    monkeypatch.setattr(vector_store, "upsert", upsert)

    client.post(
        "/api/save-chapter",
        json={"text": "Герой вошёл в дом.", "book_id": book.id},
    )
    calls["people"] = [{"name": "герой", "description": "Ждёт утра."}]
    response = client.post(
        "/api/save-chapter",
        json={"text": "Герой вошёл в дом.", "book_id": book.id},
    )
    stored = next(event for event in _events(response) if event["type"] == "saved")
    assert stored["characters"] == [
        {
            "name": "Герой",
            "description": "Вошёл в дом и закрыл дверь. Ждёт утра.",
        }
    ]
    assert list(cards.values()) == [
        "Персонаж: Герой\nВошёл в дом и закрыл дверь. Ждёт утра."
    ]
    assert len(cards) == 1


def test_save_keeps_fragments_when_characters_are_unparsed(monkeypatch, caplog, book):
    saved = {}

    def generate(prompt, system=None, temperature=0.3, response_format=None):
        if "персонаж" in prompt.casefold():
            return "Это описание героя, но не JSON."
        return '["Текст главы."]'

    def add(**kwargs):
        saved["documents"] = kwargs["documents"]

    _patch_pipeline(monkeypatch, lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(ollama_client, "generate", generate)
    monkeypatch.setattr(vector_store, "add", add)
    monkeypatch.setattr(
        vector_store,
        "upsert",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("карточки не из чего писать")),
    )
    caplog.set_level(logging.ERROR, logger="app")

    response = client.post(
        "/api/save-chapter",
        json={"text": "Текст главы.", "book_id": book.id},
    )
    events = _events(response)
    stored = next(event for event in events if event["type"] == "saved")
    assert stored["fragments"] == ["Текст главы."]
    assert stored["characters"] == []
    assert saved["documents"] == ["Текст главы."]
    assert events[-1] == {"type": "done"}
    assert "Это описание героя, но не JSON." in caplog.text
    listed = client.get(f"/api/books/{book.id}/characters")
    assert listed.json() == {"characters": []}


def test_save_cuts_the_chapter_when_the_model_only_reviews(monkeypatch, caplog, book):
    saved = {}

    def generate(prompt, system=None, temperature=0.3, response_format=None):
        return "Противоречий нет. Логика держится."

    def add(**kwargs):
        saved["documents"] = kwargs["documents"]

    _patch_pipeline(monkeypatch, lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(ollama_client, "generate", generate)
    monkeypatch.setattr(vector_store, "add", add)
    caplog.set_level(logging.ERROR, logger="app")

    chapter = "Герой вошёл в дом.\n\nОн закрыл дверь."
    response = client.post(
        "/api/save-chapter",
        json={"text": chapter, "book_id": book.id},
    )
    events = _events(response)
    stored = next(event for event in events if event["type"] == "saved")
    assert stored["split_by"] == "chapter"
    assert stored["fragments"] == [chapter]
    assert saved["documents"] == [chapter]
    assert "Противоречий нет" in stored["note"]
    messages = [event["message"] for event in events if event["type"] == "stage"]
    assert any("Режу главу сам" in message for message in messages)
    assert events[-1] == {"type": "done"}
    assert "режу главу сам" in caplog.text


def test_trailing_slash_still_reaches_save(book):
    response = client.post(
        "/api/save-chapter/",
        json={"text": "   ", "book_id": book.id},
        follow_redirects=False,
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Глава пустая"}


def test_unknown_api_post_is_not_method_not_allowed():
    response = client.post("/api/no-such-route", json={})
    assert response.status_code == 404
    assert response.json() == {"detail": "Маршрут не найден"}


def test_empty_chapter_stays_bad_request(book):
    for path in ("/api/check-chapter", "/api/save-chapter"):
        response = client.post(path, json={"text": "   ", "book_id": book.id})
        assert response.status_code == 400
        assert response.json() == {"detail": "Глава пустая"}


def test_long_excerpt_is_reviewed_once_without_rewriting_it(monkeypatch, book):
    chapter = "Начало главы про башню. " + ("Дальше идёт другая сцена. " * 40)
    calls = {"n": 0}

    def stream(prompt, system=None, temperature=0.3, response_format=None, num_predict=None):
        calls["n"] += 1
        assert num_predict == 600
        assert "не цитируй" in prompt
        yield "Герой смотрит в окно. Это другая сцена, не падение с башни."

    _patch_pipeline(monkeypatch, stream)
    monkeypatch.setattr(
        vector_store,
        "query",
        lambda *args, **kwargs: {"documents": [["я упала с башни"]]},
    )

    response = client.post(
        "/api/check-chapter",
        json={"text": chapter, "book_id": book.id},
    )
    answer = "".join(event["text"] for event in _events(response) if event["type"] == "token")
    assert calls["n"] == 2
    assert "Герой смотрит в окно." in answer
    assert "Вы прислали" not in answer


def test_model_error_is_logged_and_sent(monkeypatch, caplog, book):
    def stream(prompt, system=None, temperature=0.3, num_predict=None):
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
