from dataclasses import dataclass

from app.services.rag_service import RAGResult, rag_service


class EmptyChapterError(Exception):
    """Глава пустая или состоит только из пробелов."""


@dataclass(frozen=True)
class ProcessedChapter:
    answer: str
    context_chunks: list[str]
    chunks_added: int
    original_length: int


def process_chapter(text: str, add_to_store: bool = True) -> ProcessedChapter:
    if not text.strip():
        raise EmptyChapterError

    result: RAGResult = rag_service.process_chapter(
        text=text,
        add_to_store=add_to_store,
    )

    return ProcessedChapter(
        answer=result.answer,
        context_chunks=result.context_chunks,
        chunks_added=result.chunks_added,
        original_length=len(text),
    )