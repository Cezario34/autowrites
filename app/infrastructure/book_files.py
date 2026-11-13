from pathlib import Path

from app.infrastructure.books import default_books_path


def book_file(directory: Path, book_id: str, suffix: str) -> Path:
    name = Path(book_id).name
    if not name or name != book_id or name in {".", ".."}:
        raise ValueError(book_id)
    return directory / f"{name}{suffix}"


class TextFolder:
    def __init__(self, directory: Path, suffix: str = ".txt") -> None:
        self.directory = directory
        self.suffix = suffix

    def read(self, book_id: str) -> str:
        path = book_file(self.directory, book_id, self.suffix)
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")

    def write(self, book_id: str, text: str) -> Path:
        path = book_file(self.directory, book_id, self.suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path


def default_plans_dir() -> Path:
    return default_books_path().parent / "plans"


def default_texts_dir() -> Path:
    return default_books_path().parent / "texts"


plan_files = TextFolder(default_plans_dir())
chapter_files = TextFolder(default_texts_dir())
