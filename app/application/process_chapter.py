import logging
from collections.abc import Iterator

from app.application.books import require_book
from app.services.rag_service import rag_service

logger = logging.getLogger("app.chapter")


class EmptyChapterError(Exception):
    """Глава пустая или состоит только из пробелов."""


def _require_text_and_book(text: str, book_id: str):
    book = require_book(book_id)
    if not text.strip():
        logger.info("этап: отказ, пустая глава")
        raise EmptyChapterError
    return book


def open_check_stream(text: str, book_id: str) -> Iterator[dict]:
    book = _require_text_and_book(text, book_id)
    return rag_service.iter_check_chapter(
        text,
        book_id=book.id,
        book_title=book.title,
    )


def open_save_stream(text: str, book_id: str) -> Iterator[dict]:
    book = _require_text_and_book(text, book_id)
    return rag_service.iter_save_chapter(
        text,
        book_id=book.id,
        book_title=book.title,
    )
