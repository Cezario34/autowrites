import re
import threading
from pathlib import Path

from app.infrastructure.books import default_books_path

CHARACTERS_FILE = "characters.json"
PLOT_FILE = "plot.json"
PLAN_FILE = "plan.txt"
TEXT_FILE = "text.txt"
EMPTY_CHARACTERS = '{\n  "персонажи": []\n}\n'
EMPTY_PLOT = '{\n  "сюжет": []\n}\n'
_CHAPTER_FILE = re.compile(r"^глава (\d+)\.txt$", re.IGNORECASE)


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
    """Папка одной книги: characters.json, plot.json, plan.txt и text.txt."""

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
            plot = folder / PLOT_FILE
            if not plot.exists():
                plot.write_text(EMPTY_PLOT, encoding="utf-8")
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

    def write_next_chapter(self, book_id: str, text: str) -> Path:
        """Пишет текст в «глава 1.txt», «глава 2.txt» и дальше, не затирая прошлые."""
        with self._lock:
            folder = self.folder(book_id)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / _next_chapter_name(folder)
            path.write_text(text, encoding="utf-8")
            return path

    def saved_chapters(self, book_id: str) -> list[tuple[str, str]]:
        """Главы из файлов «глава 1.txt», «глава 2.txt» по возрастанию номера."""
        folder = self.folder(book_id)
        if not folder.is_dir():
            return []
        found: list[tuple[int, str]] = []
        for path in folder.iterdir():
            if not path.is_file():
                continue
            match = _CHAPTER_FILE.match(path.name)
            if match:
                found.append((int(match.group(1)), path.read_text(encoding="utf-8")))
        found.sort()
        return [(str(number), text) for number, text in found]

    def _legacy(self, book_id: str, filename: str) -> Path | None:
        parent = self.root.parent
        if filename == CHARACTERS_FILE:
            return parent / "characters" / f"{book_id}.json"
        if filename == PLAN_FILE:
            return parent / "plans" / f"{book_id}.txt"
        if filename == TEXT_FILE:
            return parent / "texts" / f"{book_id}.txt"
        return None


def _next_chapter_name(folder: Path) -> str:
    highest = 0
    for path in folder.iterdir():
        if not path.is_file():
            continue
        match = _CHAPTER_FILE.match(path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"глава {highest + 1}.txt"


library = BookLibrary()
