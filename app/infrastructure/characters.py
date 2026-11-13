import json
import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger("app.characters")

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


def _json_from_model_text(raw: str) -> object:
    text = raw.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def characters_from_model_text(raw: str) -> list[Character]:
    payload = _json_from_model_text(raw)
    characters = characters_from_payload(payload)
    if characters or is_empty_character_payload(payload):
        return characters
    raise json.JSONDecodeError("JSON не список персонажей", raw.strip(), 0)


def identities_from_payload(payload: object) -> list[tuple[str, str]]:
    """Пары «прежнее имя → имя, которое остаётся» из ответа модели."""
    if not isinstance(payload, dict):
        return []
    found: list[tuple[str, str]] = []
    for key in ("тождества", "identities"):
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            was = _field(item, ("было", "from"))
            became = _field(item, ("стало", "to"))
            if was and became:
                found.append((was, became))
    people = None
    for key in _LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            people = value
            break
    if people is None:
        return found
    for item in people:
        if not isinstance(item, dict):
            continue
        became = _field(item, _NAME_KEYS)
        previous = item.get("прежние имена", item.get("прежнее имя"))
        names = previous if isinstance(previous, list) else [previous]
        for name in names:
            if isinstance(name, str) and name.strip() and became:
                found.append((name.strip(), became))
    return found


def character_reply_from_model_text(raw: str) -> tuple[list[Character], list[tuple[str, str]]]:
    payload = _json_from_model_text(raw)
    characters = characters_from_payload(payload)
    if not characters and not is_empty_character_payload(payload):
        raise json.JSONDecodeError("JSON не список персонажей", raw.strip(), 0)
    return characters, identities_from_payload(payload)


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


def _prefer_rewritten(rewritten: Character, older: Character) -> Character:
    """Новая карточка заменяет текст. Факт старой дописывается, только если его ещё нет."""
    return Character(
        name=rewritten.name,
        role=rewritten.role.strip() or older.role.strip(),
        description=_append_fact(rewritten.description, older.description),
        appearance=_append_fact(rewritten.appearance, older.appearance),
    )


def _resolve_name(key: str, alias_to_final: dict[str, str]) -> str:
    seen: set[str] = set()
    while key in alias_to_final and key not in seen:
        seen.add(key)
        key = alias_to_final[key]
    return key


def merge_same_person(
    existing: list[Character],
    proposed: list[Character],
    identities: list[tuple[str, str]],
) -> tuple[list[Character], set[str]]:
    """Склеивает карточки, если глава назвала прежнее имя и настоящее одним человеком."""
    existing_by: dict[str, Character] = {}
    for character in existing:
        key = _card_key(character.name)
        if key:
            existing_by[key] = character
    alias_to_final: dict[str, str] = {}
    final_spelling: dict[str, str] = {}
    for was, became in identities:
        was_key = _card_key(was)
        became_key = _card_key(became)
        if not was_key or not became_key or was_key == became_key:
            continue
        if was_key not in existing_by:
            continue
        alias_to_final[was_key] = became_key
        final_spelling[became_key] = became.strip()
    absorbed = {
        key for key in alias_to_final if _resolve_name(key, alias_to_final) != key
    }
    ordered: list[Character] = []
    index: dict[str, int] = {}
    pending: dict[str, Character] = {}
    for character in proposed:
        key = _card_key(character.name)
        if not key:
            continue
        target = _resolve_name(key, alias_to_final)
        if target != key:
            if target not in index:
                name = final_spelling.get(target, character.name)
                held = pending.get(target, Character(name, "", "", ""))
                pending[target] = _prefer_rewritten(held, character)
                continue
            ordered[index[target]] = _prefer_rewritten(ordered[index[target]], character)
            continue
        if target not in index:
            index[target] = len(ordered)
            ordered.append(character)
            continue
        ordered[index[target]] = _prefer_rewritten(character, ordered[index[target]])
    for target, held in pending.items():
        if target not in index:
            index[target] = len(ordered)
            ordered.append(held)
            continue
        ordered[index[target]] = _prefer_rewritten(ordered[index[target]], held)
    for was_key in absorbed:
        target = _resolve_name(was_key, alias_to_final)
        source = existing_by[was_key]
        if target not in index:
            previous = existing_by.get(target)
            name = final_spelling.get(target, previous.name if previous else source.name)
            base = previous or Character(name, "", "", "")
            if _card_key(base.name) != target:
                base = Character(name, base.role, base.description, base.appearance)
            index[target] = len(ordered)
            ordered.append(_prefer_rewritten(base, source))
            continue
        ordered[index[target]] = _prefer_rewritten(ordered[index[target]], source)
    return ordered, absorbed


def _keep_written_text(
    current: Character,
    incoming: Character,
    restored: list[str],
) -> Character:
    """Роль и описание не заменяются. Внешность можно только дополнить."""
    role = incoming.role
    description = incoming.description
    rewritten: list[str] = []
    if not _keeps_text(current.role, incoming.role):
        role = current.role
        rewritten.append("роль")
    if not _keeps_text(current.description, incoming.description):
        description = current.description
        rewritten.append("описание")
    appearance = _append_fact(current.appearance, incoming.appearance)
    if current.appearance.strip() and not _keeps_text(current.appearance, incoming.appearance):
        restored.append(f"У «{current.name}» внешность не стёрта.")
    if rewritten:
        restored.append(f"У «{current.name}» оставлено прежнее: {', '.join(rewritten)}.")
    return Character(
        name=current.name,
        role=role,
        description=description,
        appearance=appearance,
    )


def _keep_appearance(
    current: Character,
    incoming: Character,
    restored: list[str],
) -> Character:
    """При склейке имён роль и описание можно переписать, внешность — только дописать."""
    appearance = _append_fact(current.appearance, incoming.appearance)
    if current.appearance.strip() and not _keeps_text(current.appearance, incoming.appearance):
        restored.append(f"У «{current.name}» внешность не стёрта.")
    return Character(
        name=incoming.name,
        role=incoming.role,
        description=incoming.description,
        appearance=appearance,
    )


def supplement_decision(
    existing: list[Character],
    proposed: list[Character],
    identities: list[tuple[str, str]] | None = None,
) -> tuple[str, list[Character], str]:
    """«same», если новых сведений нет; «added», если список вырос или люди склеились; «lost», если ответ стёр уже записанное. Третье значение — что именно потеряно."""
    existing_by: dict[str, Character] = {}
    for character in existing:
        key = _card_key(character.name)
        if key:
            existing_by[key] = character
    absorbed: set[str] = set()
    targets: set[str] = set()
    if identities:
        proposed, absorbed = merge_same_person(existing, proposed, identities)
        alias_to_final = {}
        for was, became in identities:
            was_key = _card_key(was)
            became_key = _card_key(became)
            if was_key in absorbed and became_key:
                alias_to_final[was_key] = became_key
        targets = {_resolve_name(key, alias_to_final) for key in absorbed}
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
    problems: list[str] = []
    restored: list[str] = []
    for key, current in existing_by.items():
        if key in absorbed:
            continue
        slot = proposed_by.get(key)
        incoming = ordered[slot] if slot is not None else None
        if incoming is None:
            problems.append(f"Пропала карточка «{current.name}».")
            continue
        if key in targets:
            ordered[slot] = _keep_appearance(current, incoming, restored)
            continue
        ordered[slot] = _keep_written_text(current, incoming, restored)
    if problems:
        return "lost", existing, " ".join(problems)
    if restored:
        logger.info("этап: защита карточек, %s", " ".join(restored))
    added = bool(absorbed) or any(key not in existing_by for key in proposed_by)
    if not added:
        added = any(
            not _same_card(current, ordered[proposed_by[key]])
            for key, current in existing_by.items()
        )
    if not added:
        return "same", existing, ""
    return "added", ordered, ""


def character_record(character: Character) -> dict[str, str]:
    return {
        "имя": character.name,
        "роль": character.role,
        "описание": character.description,
        "внешность": character.appearance,
    }


def _append_fact(current: str, incoming: str) -> str:
    """Старый текст остаётся. Новый дописывается, только если его ещё нет."""
    stored = current.strip()
    extra = incoming.strip()
    if not stored:
        return extra
    if not extra or extra in stored:
        return stored
    if stored in extra:
        return extra
    return f"{stored} {extra}"


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
