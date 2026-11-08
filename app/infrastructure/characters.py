import json
import threading
from dataclasses import dataclass
from pathlib import Path

from app.infrastructure.books import default_books_path


def default_characters_path() -> Path:
    return default_books_path().with_name("characters.json")


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


class CharacterStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_characters_path()
        self._lock = threading.Lock()

    def list_for_book(self, book_id: str) -> list[Character]:
        with self._lock:
            return list(self._read().get(book_id, []))

    def replace(self, book_id: str, characters: list[Character]) -> None:
        with self._lock:
            books = self._read()
            books[book_id] = list(characters)
            self._write(books)

    def _read(self) -> dict[str, list[Character]]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        books: dict[str, list[Character]] = {}
        for book_id, items in payload.get("books", {}).items():
            books[book_id] = [
                Character(
                    name=item["name"],
                    description=item.get("description") or "",
                )
                for item in items
            ]
        return books

    def _write(self, books: dict[str, list[Character]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "books": {
                        book_id: [
                            {"name": character.name, "description": character.description}
                            for character in characters
                        ]
                        for book_id, characters in books.items()
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.path)


character_store = CharacterStore()
