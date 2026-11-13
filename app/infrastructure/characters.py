import json
import threading
from dataclasses import dataclass

from app.infrastructure.book_files import CHARACTERS_FILE, library


@dataclass(frozen=True)
class Character:
    name: str
    role: str = ""
    description: str = ""
    appearance: str = ""


_LIST_KEYS = ("персонажи", "characters", "persons", "items", "result")
_NAME_KEYS = ("имя", "name")
_ROLE_KEYS = ("роль", "role")
_DESCRIPTION_KEYS = ("описание", "description")
_APPEARANCE_KEYS = ("внешность", "appearance")


def _field(item: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def character_from_item(item: dict, fallback_name: str = "") -> Character | None:
    if not isinstance(item, dict):
        return None
    name = _field(item, _NAME_KEYS) or fallback_name.strip()
    if not name:
        return None
    return Character(
        name=name,
        role=_field(item, _ROLE_KEYS),
        description=_field(item, _DESCRIPTION_KEYS),
        appearance=_field(item, _APPEARANCE_KEYS),
    )


def characters_from_payload(payload: object) -> list[Character]:
    """Читает список, объект с массивом или словарь «имя → карточка»."""
    if isinstance(payload, list):
        return _from_list(payload)
    if not isinstance(payload, dict):
        return []
    for key in _LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return _from_list(value)
    if _field(payload, _NAME_KEYS):
        one = character_from_item(payload)
        return [one] if one else []
    mapped: list[Character] = []
    for key, value in payload.items():
        if not isinstance(value, dict):
            return []
        character = character_from_item(value, fallback_name=str(key))
        if character is None:
            return []
        mapped.append(character)
    return mapped


def is_empty_character_payload(payload: object) -> bool:
    if payload in ([], {}):
        return True
    if isinstance(payload, dict):
        for key in _LIST_KEYS:
            if key in payload and payload[key] == []:
                return True
    return False


def _from_list(items: list) -> list[Character]:
    characters: list[Character] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        character = character_from_item(item)
        if character:
            characters.append(character)
    return characters


def characters_document(characters: list[Character]) -> str:
    return json.dumps(
        {"персонажи": [character_record(character) for character in characters]},
        ensure_ascii=False,
        indent=2,
    )


def characters_from_model_text(raw: str) -> list[Character]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    payload = json.loads(text)
    characters = characters_from_payload(payload)
    if characters or is_empty_character_payload(payload):
        return characters
    raise json.JSONDecodeError("JSON не список персонажей", text, 0)


def _card_key(name: str) -> str:
    return name.casefold().strip()


def _keeps_text(old: str, new: str) -> bool:
    stored = old.strip()
    return not stored or stored in new.strip()


def _same_card(left: Character, right: Character) -> bool:
    return (
        left.role.strip() == right.role.strip()
        and left.description.strip() == right.description.strip()
        and left.appearance.strip() == right.appearance.strip()
    )


def supplement_decision(
    existing: list[Character],
    proposed: list[Character],
) -> tuple[str, list[Character]]:
    """«same», если новых сведений нет; «added», если список только вырос; «lost», если ответ стёр уже записанное."""
    existing_by: dict[str, Character] = {}
    for character in existing:
        key = _card_key(character.name)
        if key:
            existing_by[key] = character
    ordered: list[Character] = []
    proposed_by: dict[str, int] = {}
    for character in proposed:
        key = _card_key(character.name)
        if not key:
            continue
        slot = proposed_by.get(key)
        if slot is None:
            proposed_by[key] = len(ordered)
            ordered.append(character)
        else:
            ordered[slot] = character
    for key, current in existing_by.items():
        slot = proposed_by.get(key)
        incoming = ordered[slot] if slot is not None else None
        if incoming is None or not (
            _keeps_text(current.role, incoming.role)
            and _keeps_text(current.description, incoming.description)
            and _keeps_text(current.appearance, incoming.appearance)
        ):
            return "lost", existing
    added = any(key not in existing_by for key in proposed_by)
    if not added:
        added = any(
            not _same_card(current, ordered[proposed_by[key]])
            for key, current in existing_by.items()
        )
    if not added:
        return "same", existing
    return "added", ordered


def character_record(character: Character) -> dict[str, str]:
    return {
        "имя": character.name,
        "роль": character.role,
        "описание": character.description,
        "внешность": character.appearance,
    }


def _append_fact(current: str, incoming: str) -> str:
    extra = incoming.strip()
    if extra and extra not in current:
        return f"{current} {extra}".strip()
    return current


def merge_characters(
    existing: list[Character],
    incoming: list[Character],
) -> list[Character]:
    """Склеивает карточки по имени, не теряя уже записанные факты."""
    ordered: list[Character] = []
    index: dict[str, int] = {}
    for character in [*existing, *incoming]:
        key = character.name.casefold().strip()
        if not key:
            continue
        if key not in index:
            index[key] = len(ordered)
            ordered.append(character)
            continue
        current = ordered[index[key]]
        role = current.role.strip() or character.role.strip()
        description = _append_fact(current.description, character.description)
        appearance = _append_fact(current.appearance, character.appearance)
        if (
            role != current.role
            or description != current.description
            or appearance != current.appearance
        ):
            ordered[index[key]] = Character(
                name=current.name,
                role=role,
                description=description,
                appearance=appearance,
            )
    return ordered


class CharacterStore:
    """JSON персонажей лежит в папке книги рядом с plan.txt."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def list_for_book(self, book_id: str) -> list[Character]:
        with self._lock:
            raw = library.read(book_id, CHARACTERS_FILE)
            if raw.strip():
                return characters_from_payload(json.loads(raw))
            return list(self._legacy_combined().get(book_id, []))

    def replace(self, book_id: str, characters: list[Character]) -> None:
        with self._lock:
            path = library.folder(book_id) / CHARACTERS_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {"персонажи": [character_record(character) for character in characters]},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)

    def _legacy_combined(self) -> dict[str, list[Character]]:
        """Старый общий characters.json, если в папке книги файла ещё нет."""
        path = library.root.parent / "characters.json"
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            stored_id: characters_from_payload(items)
            for stored_id, items in payload.get("books", {}).items()
        }


character_store = CharacterStore()
