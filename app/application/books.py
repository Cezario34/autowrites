import logging
from dataclasses import dataclass

from app.infrastructure.book_files import PLAN_FILE, TEXT_FILE, library
from app.infrastructure.books import Book, book_store
from app.infrastructure.characters import Character, character_store
from app.infrastructure.vector_store import vector_store

logger = logging.getLogger("app.books")

MAX_TITLE_LENGTH = 200


class EmptyBookTitleError(Exception):
    """Название книги пустое или состоит только из пробелов."""


class BookTitleTooLongError(Exception):
    """Название книги длиннее допустимого."""


class UnknownBookError(Exception):
    """Книга с таким id не создана."""


class EmptyChapterError(Exception):
    """В файле нет текста."""


@dataclass(frozen=True)
class BookDetails:
    id: str
    title: str
    created_at: str
    chunk_count: int


def _details(book: Book, chunk_count: int | None = None) -> BookDetails:
    if chunk_count is None:
        stored = vector_store.count_where({"book_id": book.id})
        people = len(character_store.list_for_book(book.id))
        chunk_count = max(stored - people, 0)
    return BookDetails(
        id=book.id,
        title=book.title,
        created_at=book.created_at,
        chunk_count=chunk_count,
    )


def list_books() -> list[BookDetails]:
    return [_details(book) for book in book_store.list_books()]


def create_book(title: str) -> BookDetails:
    cleaned = title.strip()
    if not cleaned:
        raise EmptyBookTitleError
    if len(cleaned) > MAX_TITLE_LENGTH:
        raise BookTitleTooLongError
    book = book_store.create(cleaned)
    library.prepare(book.id)
    logger.info("книга создана: %s (%s)", book.title, book.id)
    return _details(book, chunk_count=0)


def list_characters(book_id: str) -> list[Character]:
    require_book(book_id)
    return character_store.list_for_book(book_id)


def read_plan(book_id: str) -> str:
    require_book(book_id)
    return library.read(book_id, PLAN_FILE)


def save_chapter_text(book_id: str, text: str) -> tuple[Book, str]:
    """Кладёт весь присланный текст в обычный txt этой книги."""
    book = require_book(book_id)
    if not text.strip():
        raise EmptyChapterError
    library.write(book_id, TEXT_FILE, text)
    logger.info("текст книги сохранён: %s, символов=%s", book.title, len(text))
    return book, text


def require_book(book_id: str) -> Book:
    book = book_store.get(book_id)
    if book is None:
        logger.info("этап: отказ, книга не найдена: %s", book_id)
        raise UnknownBookError
    return book
