import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings


def default_books_path() -> Path:
    return Path(settings.chroma_persist_dir).expanduser().resolve().parent / "books.json"


@dataclass(frozen=True)
class Book:
    id: str
    title: str
    created_at: str


class BookStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_books_path()
        self._lock = threading.Lock()

    def list_books(self) -> list[Book]:
        with self._lock:
            books = self._read()
        books.sort(key=lambda book: book.created_at, reverse=True)
        return books

    def get(self, book_id: str) -> Book | None:
        with self._lock:
            for book in self._read():
                if book.id == book_id:
                    return book
        return None

    def create(self, title: str) -> Book:
        book = Book(
            id=str(uuid.uuid4()),
            title=title,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            books = self._read()
            books.append(book)
            self._write(books)
        return book

    def _read(self) -> list[Book]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [
            Book(
                id=item["id"],
                title=item["title"],
                created_at=item["created_at"],
            )
            for item in payload.get("books", [])
        ]

    def _write(self, books: list[Book]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "books": [
                        {
                            "id": book.id,
                            "title": book.title,
                            "created_at": book.created_at,
                        }
                        for book in books
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.path)


book_store = BookStore()
