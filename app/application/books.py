import logging
from dataclasses import dataclass

from app.infrastructure.books import Book, book_store
from app.infrastructure.vector_store import vector_store

logger = logging.getLogger("app.books")

MAX_TITLE_LENGTH = 200


class EmptyBookTitleError(Exception):
    """Название книги пустое или состоит только из пробелов."""


class BookTitleTooLongError(Exception):
    """Название книги длиннее допустимого."""


class UnknownBookError(Exception):
    """Книга с таким id не создана."""


@dataclass(frozen=True)
class BookDetails:
    id: str
    title: str
    created_at: str
    chunk_count: int


def _details(book: Book, chunk_count: int | None = None) -> BookDetails:
    if chunk_count is None:
        chunk_count = vector_store.count_where({"book_id": book.id})
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
    logger.info("книга создана: %s (%s)", book.title, book.id)
    return _details(book, chunk_count=0)


def require_book(book_id: str) -> Book:
    book = book_store.get(book_id)
    if book is None:
        logger.info("этап: отказ, книга не найдена: %s", book_id)
        raise UnknownBookError
    return book
