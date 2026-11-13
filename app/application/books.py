import json
import logging
from dataclasses import dataclass

from app.core.prompts import (
    PLOT_SUPPLEMENT_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    SUPPLEMENT_SYSTEM_PROMPT,
)
from app.infrastructure.book_files import CHARACTERS_FILE, PLAN_FILE, PLOT_FILE, library
from app.infrastructure.books import Book, book_store
from app.infrastructure.characters import (
    Character,
    character_store,
    characters_document,
    characters_from_model_text,
    supplement_decision,
)
from app.infrastructure.plot import (
    PlotChapter,
    plot_document,
    plot_from_model_text,
    plot_store,
    supplement_plot_decision,
)
from app.infrastructure.openai_client import (
    MissingOpenAIKeyError,
    OpenAIRequestError,
    openai_client,
)
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


class CharacterSupplementError(Exception):
    """Ответ OpenAI нельзя записать вместо текущего JSON."""


class PlotSupplementError(Exception):
    """Ответ OpenAI нельзя записать вместо сюжета."""


class NoSavedChapterError(Exception):
    """В папке книги ещё нет файла главы."""


@dataclass(frozen=True)
class SavedChapterResult:
    number: str
    plot_updated: bool
    characters_updated: bool


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


def read_plot(book_id: str) -> list[PlotChapter]:
    require_book(book_id)
    return sorted(plot_store.list_for_book(book_id), key=lambda chapter: int(chapter.number))


def supplement_plot(
    book_id: str,
    number: str,
    chapter: str,
) -> tuple[list[PlotChapter], bool]:
    """Сокращает главу до событий и заменяет plot.json, только если сведения новые."""
    require_book(book_id)
    if not chapter.strip():
        raise EmptyChapterError
    existing = plot_store.list_for_book(book_id)
    stored = library.read(book_id, PLOT_FILE).strip() or plot_document(existing)
    try:
        raw = openai_client.complete_json(
            PLOT_SUPPLEMENT_SYSTEM_PROMPT,
            f"Текущий сюжет:\n{stored}\n\nЭто глава {number}. В поле «глава» поставь \"{number}\".\n\nТекст главы:\n{chapter}",
        )
        proposed = plot_from_model_text(raw)
    except MissingOpenAIKeyError:
        raise
    except OpenAIRequestError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PlotSupplementError("Модель вернула не сюжет") from exc
    decision, records = supplement_plot_decision(existing, proposed)
    logger.info(
        "этап: дополнение сюжета, глава=%s, было=%s, стало=%s, решение=%s",
        number,
        len(existing),
        len(proposed),
        decision,
    )
    if decision == "lost":
        raise PlotSupplementError("Модель изменила уже записанный сюжет. Файл не трогал.")
    present = {item.number for item in (records if decision == "added" else existing)}
    if number not in present:
        raise PlotSupplementError("Модель не добавила главу в сюжет. Файл не трогал.")
    if decision == "same":
        return existing, False
    plot_store.replace(book_id, records)
    return records, True


def supplement_saved_chapters(
    book_id: str,
) -> tuple[list[SavedChapterResult], list[PlotChapter], list[Character]]:
    """По файлам глав обновляет сюжет, затем теми же текстами дополняет персонажей."""
    require_book(book_id)
    files = library.saved_chapters(book_id)
    saved = [(number, text) for number, text in files if text.strip()]
    if not files:
        raise NoSavedChapterError
    if not saved:
        raise EmptyChapterError
    known = {chapter.number for chapter in plot_store.list_for_book(book_id)}
    pending = [(number, text) for number, text in saved if number not in known]
    if not pending:
        pending = [saved[-1]]
    results: list[SavedChapterResult] = []
    for number, text in pending:
        _, plot_updated = supplement_plot(book_id, number, text)
        _, characters_updated = supplement_characters(book_id, text)
        results.append(
            SavedChapterResult(
                number=number,
                plot_updated=plot_updated,
                characters_updated=characters_updated,
            )
        )
    return results, plot_store.list_for_book(book_id), character_store.list_for_book(book_id)


def supplement_characters(book_id: str, chapter: str) -> tuple[list[Character], bool]:
    """Просит OpenAI дополнить JSON персонажей и заменяет файл, только если сведения новые."""
    require_book(book_id)
    if not chapter.strip():
        raise EmptyChapterError
    existing = character_store.list_for_book(book_id)
    stored = library.read(book_id, CHARACTERS_FILE).strip()
    if not stored:
        stored = characters_document(existing)
    try:
        raw = openai_client.complete_json(
            SUPPLEMENT_SYSTEM_PROMPT,
            f"Текущие карточки:\n{stored}\n\nГлава:\n{chapter}",
        )
        proposed = characters_from_model_text(raw)
    except MissingOpenAIKeyError:
        raise
    except OpenAIRequestError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CharacterSupplementError("Модель вернула не список персонажей") from exc
    decision, records = supplement_decision(existing, proposed)
    logger.info(
        "этап: дополнение персонажей, было=%s, стало=%s, решение=%s",
        len(existing),
        len(proposed),
        decision,
    )
    if decision == "lost":
        raise CharacterSupplementError(
            "Модель изменила уже записанные карточки. Файл не трогал."
        )
    if decision == "same":
        return existing, False
    character_store.replace(book_id, records)
    return records, True


def review_chapter(book_id: str, text: str) -> str:
    """Проверяет вставленный текст по JSON персонажей и сюжета."""
    require_book(book_id)
    if not text.strip():
        raise EmptyChapterError
    people = library.read(book_id, CHARACTERS_FILE).strip() or characters_document([])
    plot = library.read(book_id, PLOT_FILE).strip() or plot_document([])
    review = openai_client.complete(
        REVIEW_SYSTEM_PROMPT,
        f"Персонажи:\n{people}\n\nСюжет:\n{plot}\n\nТекст:\n{text}",
        temperature=0.4,
    ).strip()
    if not review:
        raise OpenAIRequestError("Модель не прислала проверку")
    logger.info("этап: проверка текста, символов=%s", len(review))
    return review


def save_chapter_text(book_id: str, text: str) -> tuple[Book, str, str]:
    """Кладёт главу в следующий файл: глава 1.txt, глава 2.txt и дальше."""
    book = require_book(book_id)
    if not text.strip():
        raise EmptyChapterError
    path = library.write_next_chapter(book_id, text)
    logger.info(
        "глава сохранена: %s, файл=%s, символов=%s",
        book.title,
        path.name,
        len(text),
    )
    return book, text, path.name


def require_book(book_id: str) -> Book:
    book = book_store.get(book_id)
    if book is None:
        logger.info("этап: отказ, книга не найдена: %s", book_id)
        raise UnknownBookError
    return book
