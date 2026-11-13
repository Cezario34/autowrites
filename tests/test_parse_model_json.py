import pytest

from app.infrastructure.characters import (
    Character,
    character_reply_from_model_text,
    merge_characters,
    supplement_decision,
)
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


def test_revealed_name_replaces_the_provisional_card():
    old_woman = Character(
        "старуха",
        "жительница форта",
        "Одна из женщин, вышедших навстречу Ульрике.",
        "совсем старуха, кутающаяся в дырявую шаль",
    )
    girl = Character(
        "девка",
        "жительница форта",
        "Одна из женщин, вышедших навстречу Ульрике.",
        "забитого вида, с пустым взглядом",
    )
    gerta = Character(
        "Герта",
        "жительница форта",
        "Старая женщина, которая завела Ульрику в жилую башню и напомнила, что у них еще три овцы.",
        "старая",
    )
    marta = Character(
        "Марта",
        "жительница форта",
        "Молодая женщина, которая завела Ульрику в жилую башню; испуганно всхлипнула, когда Ульрика объявила об урезании пайков.",
        "молодая",
    )
    rewritten = Character(
        marta.name,
        marta.role,
        marta.description,
        "забитого вида, с пустым взглядом",
    )
    decision, records = supplement_decision(
        [old_woman, girl, gerta, marta],
        [old_woman, gerta, rewritten],
        [("девка", "Марта")],
    )
    assert decision == "added"
    assert [item.name for item in records] == ["старуха", "Герта", "Марта"]
    assert records[0] == old_woman
    assert records[1] == gerta
    assert records[2].appearance == "забитого вида, с пустым взглядом"
    assert "молодая" not in records[2].appearance

    unchanged = supplement_decision(
        [old_woman, girl, gerta, marta],
        [old_woman, gerta, marta],
        [],
    )
    assert unchanged[0] == "lost"


def test_identity_moves_facts_when_the_model_leaves_both_cards():
    girl = Character("девка", "жительница форта", "Вышла навстречу.", "пустой взгляд")
    marta = Character("Марта", "жительница форта", "Завела Ульрику в башню.", "молодая")
    decision, records = supplement_decision(
        [girl, marta],
        [girl, marta],
        [("девка", "Марта")],
    )
    assert decision == "added"
    assert [item.name for item in records] == ["Марта"]
    assert records[0].appearance == "молодая пустой взгляд"
    assert "Вышла навстречу." in records[0].description


def test_previous_name_on_the_card_is_an_identity():
    people, identities = character_reply_from_model_text(
        '{"персонажи":[{"имя":"Марта","роль":"жительница форта","описание":"Завела в башню.","внешность":"пустой взгляд","прежнее имя":"девка"}],"тождества":[]}'
    )
    assert identities == [("девка", "Марта")]
    assert people[0].name == "Марта"


def test_plain_prose_is_still_rejected(caplog):
    caplog.set_level("ERROR", logger="app")
    with pytest.raises(SplitError, match="не список фрагментов") as error:
        parse_fragments("Здесь нет JSON, только разбор главы.")
    assert "Ответ модели:" in str(error.value)
    assert "Здесь нет JSON, только разбор главы." in caplog.text
    with pytest.raises(CharacterParseError, match="не список персонажей"):
        parse_characters("Анна — главная героиня, но это не JSON.")
