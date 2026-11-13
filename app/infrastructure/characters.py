import json
import threading
from dataclasses import dataclass
from pathlib import Path

from app.infrastructure.books import default_books_path
from app.infrastructure.book_files import book_file


def default_characters_dir() -> Path:
    return default_books_path().parent / "characters"


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
    """Отдельный JSON на книгу: characters/<id книги>.json."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or default_characters_dir()
        self._lock = threading.Lock()

    def list_for_book(self, book_id: str) -> list[Character]:
        with self._lock:
            path = book_file(self.directory, book_id, ".json")
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                return _cards(payload.get("characters", []))
            return list(self._legacy().get(book_id, []))

    def replace(self, book_id: str, characters: list[Character]) -> None:
        with self._lock:
            path = book_file(self.directory, book_id, ".json")
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
                ),
                encoding="utf-8",
            )
            temporary.replace(path)

    def _legacy(self) -> dict[str, list[Character]]:
        """Старый общий characters.json, если отдельного файла книги ещё нет."""
        path = self.directory.parent / "characters.json"
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            stored_id: _cards(items)
            for stored_id, items in payload.get("books", {}).items()
        }


character_store = CharacterStore()
