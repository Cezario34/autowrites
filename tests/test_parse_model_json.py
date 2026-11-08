import pytest

from app.services.rag_service import (
    CharacterParseError,
    SplitError,
    parse_characters,
    parse_fragments,
)


def test_parse_fragments_from_object_after_prose():
    raw = (
        "Готово.\n"
        '```json\n'
        '{"fragments": ["Первый кусок.", "Второй кусок."]}\n'
        "```"
    )
    assert parse_fragments(raw) == ["Первый кусок.", "Второй кусок."]


def test_parse_fragments_split_by_marker():
    raw = "Герой вошёл в дом.\n---\nОн закрыл дверь."
    assert parse_fragments(raw) == ["Герой вошёл в дом.", "Он закрыл дверь."]


def test_parse_fragments_from_numbered_lines():
    raw = "1. Герой вошёл в дом.\n2. Он закрыл дверь."
    assert parse_fragments(raw) == ["Герой вошёл в дом.", "Он закрыл дверь."]


def test_parse_fragments_from_array_inside_text():
    assert parse_fragments('Вот фрагменты:\n["Один.", "Два."]') == ["Один.", "Два."]


def test_parse_characters_from_object():
    raw = '{"characters": [{"name": "Анна", "description": "Ждёт."}]}'
    parsed = parse_characters(raw)
    assert [(item.name, item.description) for item in parsed] == [("Анна", "Ждёт.")]


def test_plain_prose_is_still_rejected(caplog):
    caplog.set_level("ERROR", logger="app")
    with pytest.raises(SplitError, match="не список фрагментов") as error:
        parse_fragments("Здесь нет JSON, только разбор главы.")
    assert "Ответ модели:" in str(error.value)
    assert "Здесь нет JSON, только разбор главы." in caplog.text
    with pytest.raises(CharacterParseError, match="не список персонажей"):
        parse_characters("Анна — главная героиня, но это не JSON.")
