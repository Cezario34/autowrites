import threading
from pathlib import Path

from app.infrastructure.books import default_books_path

CHARACTERS_FILE = "characters.json"
PLAN_FILE = "plan.txt"
TEXT_FILE = "text.txt"
EMPTY_CHARACTERS = '{\n  "characters": []\n}\n'


def default_library_dir() -> Path:
    return default_books_path().parent / "books"


def book_folder(root: Path, book_id: str) -> Path:
    from app.infrastructure.books import book_store

    book = book_store.get(book_id)
    raw = book.folder if book is not None and book.folder else book_id
    name = Path(raw).name
    if not name or name != raw or name in {".", ".."}:
        raise ValueError(book_id)
    return root / name


class BookLibrary:
    """Папка одной книги: characters.json, plan.txt и text.txt."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or default_library_dir()
        self._lock = threading.Lock()

    def folder(self, book_id: str) -> Path:
        return book_folder(self.root, book_id)

    def prepare(self, book_id: str) -> Path:
        """Пустые файлы, чтобы в них сразу можно было вставить текст."""
        with self._lock:
            folder = self.folder(book_id)
            folder.mkdir(parents=True, exist_ok=True)
            characters = folder / CHARACTERS_FILE
            if not characters.exists():
                characters.write_text(EMPTY_CHARACTERS, encoding="utf-8")
            for filename in (PLAN_FILE, TEXT_FILE):
                path = folder / filename
                if not path.exists():
                    path.write_text("", encoding="utf-8")
            return folder

    def read(self, book_id: str, filename: str) -> str:
        path = self.folder(book_id) / filename
        if path.is_file():
            return path.read_text(encoding="utf-8")
        legacy = self._legacy(book_id, filename)
        if legacy is not None and legacy.is_file():
            return legacy.read_text(encoding="utf-8")
        return ""

    def write(self, book_id: str, filename: str, text: str) -> Path:
        path = self.folder(book_id) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def _legacy(self, book_id: str, filename: str) -> Path | None:
        parent = self.root.parent
        if filename == CHARACTERS_FILE:
            return parent / "characters" / f"{book_id}.json"
        if filename == PLAN_FILE:
            return parent / "plans" / f"{book_id}.txt"
        if filename == TEXT_FILE:
            return parent / "texts" / f"{book_id}.txt"
        return None


library = BookLibrary()
