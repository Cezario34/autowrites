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


def test_parse_fragments_from_array_inside_text():
    assert parse_fragments('Вот фрагменты:\n["Один.", "Два."]') == ["Один.", "Два."]


def test_parse_characters_from_object():
    raw = '{"characters": [{"name": "Анна", "description": "Ждёт."}]}'
    parsed = parse_characters(raw)
    assert [(item.name, item.description) for item in parsed] == [("Анна", "Ждёт.")]


def test_plain_prose_is_still_rejected():
    with pytest.raises(SplitError, match="не список фрагментов"):
        parse_fragments("Здесь нет JSON, только разбор главы.")
    with pytest.raises(CharacterParseError, match="не список персонажей"):
        parse_characters("Анна — главная героиня, но это не JSON.")
