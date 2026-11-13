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


def _book(title="Север"):
    created = client.post("/api/books", json={"title": title})
    assert created.status_code == 201
    return created.json()["id"], tmp_folder(title)


def tmp_folder(title):
    return library.root / title


def test_supplement_replaces_the_file_only_when_a_fact_is_added(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    book_id, folder = _book()
    original = {
        "персонажи": [
            {
                "имя": "Ульф",
                "роль": "лесной охотник",
                "описание": "Проводник.",
                "внешность": "Тёмная борода.",
            }
        ]
    }
    path = folder / "characters.json"
    path.write_text(json.dumps(original, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    seen = {}

    def complete_json(system, user):
        seen["system"] = system
        seen["user"] = user
        return json.dumps(
            {
                "персонажи": [
                    {
                        "имя": "Ульф",
                        "роль": "лесной охотник",
                        "описание": "Проводник. Честный и прямой.",
                        "внешность": "Тёмная борода.",
                    },
                    {
                        "имя": "Марта",
                        "роль": "молодая женщина форта",
                        "описание": "Помогает по хозяйству.",
                        "внешность": "",
                    },
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(openai_client, "complete_json", complete_json)
    chapter = "Ульф честный. {Марта} помогает по хозяйству."
    saved = client.post(
        f"/api/books/{book_id}/characters/supplement",
        json={"text": chapter},
    )
    assert saved.status_code == 200
    assert saved.json()["updated"] is True
    assert [item["name"] for item in saved.json()["characters"]] == ["Ульф", "Марта"]
    written = json.loads(path.read_text(encoding="utf-8"))
    ulf = written["персонажи"][0]
    assert ulf["описание"] == "Проводник. Честный и прямой."
    assert ulf["внешность"] == "Тёмная борода."
    assert written["персонажи"][1]["имя"] == "Марта"
    assert "Ульф" in seen["user"]
    assert chapter in seen["user"]
    assert "только JSON" in seen["system"]

    before = path.read_text(encoding="utf-8")

    def same_json(system, user):
        return json.dumps(written, ensure_ascii=False)

    monkeypatch.setattr(openai_client, "complete_json", same_json)
    again = client.post(
        f"/api/books/{book_id}/characters/supplement",
        json={"text": "В главе нет новых сведений."},
    )
    assert again.status_code == 200
    assert again.json()["updated"] is False
    assert path.read_text(encoding="utf-8") == before


def test_supplement_does_not_replace_a_card_the_model_rewrote(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    book_id, folder = _book("Юг")
    path = folder / "characters.json"
    original = path.read_text(encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "персонажи": [
                    {
                        "имя": "Герта",
                        "роль": "старая женщина форта",
                        "описание": "Готовит и штопает.",
                        "внешность": "Кутается в шаль.",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    stored = path.read_text(encoding="utf-8")

    def complete_json(system, user):
        return json.dumps(
            {"персонажи": [{"имя": "Герта", "роль": "старуха", "описание": "Скептична.", "внешность": ""}]},
            ensure_ascii=False,
        )

    monkeypatch.setattr(openai_client, "complete_json", complete_json)
    response = client.post(
        f"/api/books/{book_id}/characters/supplement",
        json={"text": "Герта скептична."},
    )
    assert response.status_code == 200
    assert response.json()["updated"] is False
    assert path.read_text(encoding="utf-8") == stored
    assert original != stored


def test_lost_character_reply_is_retried_and_then_saved(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    book_id, folder = _book("Форт")
    path = folder / "characters.json"
    original = {
        "персонажи": [
            {
                "имя": "Герта",
                "роль": "старая женщина форта",
                "описание": "Готовит и штопает.",
                "внешность": "Кутается в шаль.",
            }
        ]
    }
    path.write_text(json.dumps(original, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    calls = {"n": 0}

    def complete_json(system, user):
        calls["n"] += 1
        calls["user"] = user
        if calls["n"] == 1:
            return json.dumps(
                {
                    "персонажи": [
                        {
                            "имя": "Марта",
                            "роль": "жительница форта",
                            "описание": "Завела в башню.",
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
                        "имя": "Герта",
                        "роль": "старая женщина форта",
                        "описание": "Готовит и штопает. Скептична.",
                        "внешность": "Кутается в шаль.",
                    }
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(openai_client, "complete_json", complete_json)
    response = client.post(
        f"/api/books/{book_id}/characters/supplement",
        json={"text": "Герта скептична."},
    )
    assert response.status_code == 200
    assert response.json()["updated"] is True
    assert calls["n"] == 2
    assert "Пропала карточка «Герта»" in calls["user"]
    assert "файл не менялся" in calls["user"]
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["персонажи"][0]["описание"] == "Готовит и штопает. Скептична."
    assert written["персонажи"][0]["внешность"] == "Кутается в шаль."


def test_new_character_is_saved_when_an_old_field_was_rewritten(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    book_id, folder = _book("Башня")
    path = folder / "characters.json"
    path.write_text(
        json.dumps(
            {
                "персонажи": [
                    {
                        "имя": "Герта",
                        "роль": "старая женщина форта",
                        "описание": "Готовит и штопает.",
                        "внешность": "Кутается в шаль.",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    calls = {"n": 0}

    def complete_json(system, user):
        calls["n"] += 1
        return json.dumps(
            {
                "персонажи": [
                    {
                        "имя": "Герта",
                        "роль": "старуха",
                        "описание": "Скептична.",
                        "внешность": "",
                    },
                    {
                        "имя": "Ульф",
                        "роль": "лесной охотник",
                        "описание": "Пришёл в форт.",
                        "внешность": "Тёмная борода.",
                    },
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(openai_client, "complete_json", complete_json)
    response = client.post(
        f"/api/books/{book_id}/characters/supplement",
        json={"text": "Ульф пришёл, Герта скептична."},
    )
    assert response.status_code == 200
    assert response.json()["updated"] is True
    assert calls["n"] == 1
    written = json.loads(path.read_text(encoding="utf-8"))
    assert [item["имя"] for item in written["персонажи"]] == ["Герта", "Ульф"]
    assert written["персонажи"][0]["роль"] == "старая женщина форта"
    assert written["персонажи"][0]["описание"] == "Готовит и штопает."
    assert written["персонажи"][0]["внешность"] == "Кутается в шаль."
    assert written["персонажи"][1]["описание"] == "Пришёл в форт."


def test_revealed_name_overwrites_the_provisional_card(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    book_id, folder = _book("Форт")
    path = folder / "characters.json"
    original = {
        "персонажи": [
            {
                "имя": "старуха",
                "роль": "жительница форта",
                "описание": "Одна из женщин, вышедших навстречу Ульрике.",
                "внешность": "совсем старуха, кутающаяся в дырявую шаль",
            },
            {
                "имя": "девка",
                "роль": "жительница форта",
                "описание": "Одна из женщин, вышедших навстречу Ульрике.",
                "внешность": "забитого вида, с пустым взглядом",
            },
            {
                "имя": "Герта",
                "роль": "жительница форта",
                "описание": "Старая женщина, которая завела Ульрику в жилую башню и напомнила, что у них еще три овцы.",
                "внешность": "старая",
            },
            {
                "имя": "Марта",
                "роль": "жительница форта",
                "описание": "Молодая женщина, которая завела Ульрику в жилую башню; испуганно всхлипнула, когда Ульрика объявила об урезании пайков.",
                "внешность": "молодая",
            },
        ]
    }
    path.write_text(json.dumps(original, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def without_identity(system, user):
        kept = [item for item in original["персонажи"] if item["имя"] != "девка"]
        kept[-1] = {**kept[-1], "внешность": "забитого вида, с пустым взглядом"}
        return json.dumps({"персонажи": kept}, ensure_ascii=False)

    monkeypatch.setattr(openai_client, "complete_json", without_identity)
    refused = client.post(
        f"/api/books/{book_id}/characters/supplement",
        json={"text": "Девка назвалась Мартой."},
    )
    assert refused.status_code == 400
    assert path.read_text(encoding="utf-8") == json.dumps(original, ensure_ascii=False, indent=2) + "\n"

    def with_identity(system, user):
        assert "тождества" in system
        kept = [item for item in original["персонажи"] if item["имя"] != "девка"]
        kept[-1] = {**kept[-1], "внешность": "забитого вида, с пустым взглядом"}
        return json.dumps(
            {
                "персонажи": kept,
                "тождества": [{"было": "девка", "стало": "Марта"}],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(openai_client, "complete_json", with_identity)
    saved = client.post(
        f"/api/books/{book_id}/characters/supplement",
        json={"text": "Девка назвалась Мартой."},
    )
    assert saved.status_code == 200
    assert saved.json()["updated"] is True
    written = json.loads(path.read_text(encoding="utf-8"))
    names = [item["имя"] for item in written["персонажи"]]
    assert names == ["старуха", "Герта", "Марта"]
    assert written["персонажи"][0] == original["персонажи"][0]
    assert written["персонажи"][1] == original["персонажи"][2]
    assert written["персонажи"][2]["внешность"] == "забитого вида, с пустым взглядом"
    assert "девка" not in path.read_text(encoding="utf-8")


def test_missing_key_and_empty_chapter_do_not_call_openai(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    book_id, _folder = _book("Восток")
    called = {"n": 0}
    real_complete = openai_client.complete_json

    def complete_json(system, user):
        called["n"] += 1
        raise AssertionError("вызов не нужен")

    monkeypatch.setattr(openai_client, "complete_json", complete_json)
    empty = client.post(
        f"/api/books/{book_id}/characters/supplement",
        json={"text": "   "},
    )
    assert empty.status_code == 400
    assert empty.json()["detail"] == "Глава пустая"

    monkeypatch.setattr(openai_client, "complete_json", real_complete)
    monkeypatch.setattr(settings, "openai_api_key", "")
    missing = client.post(
        f"/api/books/{book_id}/characters/supplement",
        json={"text": "Глава про торф."},
    )
    assert missing.status_code == 400
    assert missing.json()["detail"] == "В окружении нет ключа OpenAI"
    assert called["n"] == 0
