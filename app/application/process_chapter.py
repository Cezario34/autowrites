import logging
from collections.abc import Iterator
from dataclasses import dataclass

from app.services.rag_service import RAGResult, rag_service

logger = logging.getLogger("app.chapter")


class EmptyChapterError(Exception):
    """Глава пустая или состоит только из пробелов."""


@dataclass(frozen=True)
class ProcessedChapter:
    answer: str
    context_chunks: list[str]
    chunks_added: int
    original_length: int


def open_chapter_stream(text: str, add_to_store: bool = True) -> Iterator[dict]:
    if not text.strip():
        logger.info("этап: отказ, пустая глава")
        raise EmptyChapterError
    return rag_service.iter_process_chapter(text, add_to_store=add_to_store)


def process_chapter(text: str, add_to_store: bool = True) -> ProcessedChapter:
    parts: list[str] = []
    context_chunks: list[str] = []
    chunks_added = 0
    for event in open_chapter_stream(text, add_to_store=add_to_store):
        if event["type"] == "token":
            parts.append(event["text"])
        elif event["type"] == "meta":
            context_chunks = list(event["context_chunks"])
            chunks_added = event["chunks_added"]

    result = RAGResult(
        answer="".join(parts),
        context_chunks=context_chunks,
        chunks_added=chunks_added,
    )
    return ProcessedChapter(
        answer=result.answer,
        context_chunks=result.context_chunks,
        chunks_added=result.chunks_added,
        original_length=len(text),
    )
