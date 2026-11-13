import json
import threading
from dataclasses import dataclass

from app.infrastructure.book_files import CHARACTERS_FILE, library


@dataclass(frozen=True)
class Character:
    name: str
    description: str


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
        extra = character.description.strip()
        if extra and extra not in current.description:
            ordered[index[key]] = Character(
                name=current.name,
                description=f"{current.description} {extra}".strip(),
            )
    return ordered


def _cards(items: list[dict]) -> list[Character]:
    return [
        Character(name=item["name"], description=item.get("description") or "")
        for item in items
    ]


class CharacterStore:
    """JSON персонажей лежит в папке книги: books/<id>/characters.json."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def list_for_book(self, book_id: str) -> list[Character]:
        with self._lock:
            raw = library.read(book_id, CHARACTERS_FILE)
            if raw.strip():
                payload = json.loads(raw)
                return _cards(payload.get("characters", []))
            return list(self._legacy_combined().get(book_id, []))

    def replace(self, book_id: str, characters: list[Character]) -> None:
        with self._lock:
            path = library.folder(book_id) / CHARACTERS_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "characters": [
                            {
                                "name": character.name,
                                "description": character.description,
                            }
                            for character in characters
                        ]
                    },
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
            stored_id: _cards(items)
            for stored_id, items in payload.get("books", {}).items()
        }


character_store = CharacterStore()
