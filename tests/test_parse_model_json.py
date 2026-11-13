import pytest

from app.infrastructure.characters import Character, merge_characters
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
    assert [(item.name, item.role, item.description) for item in parsed] == [
        ("Анна", "", "Ждёт.")
    ]


def test_parse_characters_from_russian_shapes():
    listed = parse_characters(
        '{"персонажи": [{"имя": "Герта (Грета)", "роль": "старая женщина форта", "описание": "Живёт в форте."}]}'
    )
    assert [(item.name, item.role, item.description) for item in listed] == [
        ("Герта (Грета)", "старая женщина форта", "Живёт в форте.")
    ]
    by_name = parse_characters(
        '{"Ульрик": {"роль": "конунг", "описание": "Правит."}}'
    )
    assert [(item.name, item.role) for item in by_name] == [("Ульрик", "конунг")]
    with_looks = parse_characters(
        '{"персонажи": [{"имя": "Ульф", "роль": "лесной охотник", '
        '"описание": "Проводник.", "внешность": "Высокий, с тёмной бородой."}]}'
    )
    assert [
        (item.name, item.role, item.description, item.appearance)
        for item in with_looks
    ] == [("Ульф", "лесной охотник", "Проводник.", "Высокий, с тёмной бородой.")]


def test_merge_keeps_written_appearance_when_the_new_card_omits_it():
    kept = merge_characters(
        [
            Character(
                name="Ульф",
                role="лесной охотник",
                description="Проводник.",
                appearance="Высокий, с тёмной бородой.",
            )
        ],
        [Character(name="ульф", description="Честный и прямой.")],
    )
    assert len(kept) == 1
    assert kept[0].appearance == "Высокий, с тёмной бородой."
    assert kept[0].description == "Проводник. Честный и прямой."
    assert kept[0].role == "лесной охотник"


def test_plain_prose_is_still_rejected(caplog):
    caplog.set_level("ERROR", logger="app")
    with pytest.raises(SplitError, match="не список фрагментов") as error:
        parse_fragments("Здесь нет JSON, только разбор главы.")
    assert "Ответ модели:" in str(error.value)
    assert "Здесь нет JSON, только разбор главы." in caplog.text
    with pytest.raises(CharacterParseError, match="не список персонажей"):
        parse_characters("Анна — главная героиня, но это не JSON.")
