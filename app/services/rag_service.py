import json
import logging
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from app.core.config import settings
from app.core.prompts import (
    CHECK_CHAPTER_PROMPT,
    SPLIT_CHAPTER_PROMPT,
    SPLIT_SYSTEM_PROMPT,
)
from app.infrastructure.ollama import ollama_client
from app.infrastructure.vector_store import vector_store
from app.services.chunker import split_text
from app.services.embedding_service import embedding_service

logger = logging.getLogger("app.rag")


@dataclass
class RAGResult:
    answer: str
    context_chunks: list[str]
    chunks_added: int


class SplitError(Exception):
    """Модель не вернула пригодный список фрагментов."""


def _stage(stage: str, message: str) -> dict[str, str]:
    return {"type": "stage", "stage": stage, "message": message}


def parse_fragments(raw: str) -> list[str]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SplitError("Модель вернула не список фрагментов") from exc
    if isinstance(payload, dict):
        payload = payload.get("fragments") or payload.get("chunks")
    if not isinstance(payload, list):
        raise SplitError("Модель вернула не список фрагментов")

    fragments: list[str] = []
    for item in payload:
        if isinstance(item, str):
            piece = item.strip()
        elif isinstance(item, dict):
            piece = str(item.get("text") or item.get("content") or "").strip()
        else:
            piece = ""
        if piece:
            fragments.append(piece)
    if not fragments:
        raise SplitError("Модель не вернула фрагменты")
    return fragments


class RAGService:
    def _save_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        book_id: str,
        chapter_id: str | None = None,
    ) -> int:
        if not chunks:
            return 0

        chapter_id = chapter_id or str(uuid.uuid4())
        ids = [f"{book_id}-{chapter_id}-{i}" for i in range(len(chunks))]
        metadatas = [
            {"book_id": book_id, "chapter_id": chapter_id, "chunk_index": i}
            for i in range(len(chunks))
        ]
        vector_store.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        return len(chunks)

    def add_chapter(
        self,
        text: str,
        book_id: str,
        chapter_id: str | None = None,
    ) -> int:
        """Нарезает главу, делает эмбеддинги и сохраняет в векторную базу книги."""
        chunks = split_text(text)
        if not chunks:
            return 0

        embeddings = embedding_service.embed_batch(chunks)
        return self._save_chunks(chunks, embeddings, book_id, chapter_id)

    def query(
        self,
        text: str,
        book_id: str,
        n_results: int | None = None,
    ) -> list[str]:
        """Ищет релевантные чанки по тексту внутри одной книги."""
        # Берём только начало текста для эмбеддинга (чтобы не превысить контекст)
        query_text = text[:2000]
        query_embedding = embedding_service.embed(query_text)
        results = vector_store.query(
            query_embedding,
            n_results=n_results,
            where={"book_id": book_id},
        )

        documents = results.get("documents", [[]])[0]
        return documents or []

    def iter_check_chapter(
        self,
        text: str,
        book_id: str,
        book_title: str,
    ) -> Iterator[dict]:
        """Проверяет главу только по уже сохранённым фрагментам книги."""
        logger.info(
            "этап: проверка, книга=%s (%s), символов=%s",
            book_title,
            book_id,
            len(text),
        )
        yield _stage("start", "Идёт проверка…")
        yield _stage("search", "Ищу в книге…")
        context_chunks = self.query(text, book_id, n_results=settings.top_k)
        logger.info("этап: поиск, фрагментов=%s", len(context_chunks))
        context = (
            "\n\n---\n\n".join(context_chunks)
            if context_chunks
            else "Контекст отсутствует."
        )
        prompt = CHECK_CHAPTER_PROMPT.format(
            book=book_title,
            context=context,
            chapter=text,
        )
        yield {
            "type": "meta",
            "book_id": book_id,
            "book_title": book_title,
            "context_chunks": context_chunks,
            "original_length": len(text),
        }
        yield _stage("generate", "Генерирую ответ…")
        logger.info("этап: генерация, модель=%s", settings.llm_model)

        parts: list[str] = []
        for piece in ollama_client.generate_stream(prompt):
            parts.append(piece)
            yield {"type": "token", "text": piece}

        logger.info("этап: готово, символов ответа=%s", sum(len(part) for part in parts))
        yield {"type": "done"}

    def iter_save_chapter(
        self,
        text: str,
        book_id: str,
        book_title: str,
    ) -> Iterator[dict]:
        """Просит модель разрезать главу и сохраняет фрагменты без разбора."""
        logger.info(
            "этап: сохранение, книга=%s (%s), символов=%s",
            book_title,
            book_id,
            len(text),
        )
        yield _stage("start", "Идёт сохранение…")
        yield _stage("split", "Модель разбивает текст…")
        logger.info("этап: разбиение, модель=%s", settings.llm_model)
        raw = ollama_client.generate(
            SPLIT_CHAPTER_PROMPT.format(
                book=book_title,
                chapter=text,
                chunk_size=settings.chunk_size,
            ),
            system=SPLIT_SYSTEM_PROMPT,
            temperature=0.1,
        )
        fragments = parse_fragments(raw)
        logger.info("этап: фрагментов=%s", len(fragments))
        yield _stage("embed", "Считаю эмбеддинги…")
        logger.info(
            "этап: эмбеддинги, чанков=%s, модель=%s",
            len(fragments),
            settings.embedding_model,
        )
        embeddings = embedding_service.embed_batch(fragments)
        yield _stage("store", "Сохраняю фрагменты…")
        chunks_added = self._save_chunks(fragments, embeddings, book_id)
        logger.info("этап: запись в базу, чанков=%s", chunks_added)
        yield {
            "type": "saved",
            "book_id": book_id,
            "book_title": book_title,
            "fragments": fragments,
            "chunks_added": chunks_added,
            "original_length": len(text),
        }
        logger.info("этап: сохранено, чанков=%s", chunks_added)
        yield {"type": "done"}

    def check_chapter(self, text: str, book_id: str, book_title: str) -> RAGResult:
        parts: list[str] = []
        context_chunks: list[str] = []
        for event in self.iter_check_chapter(text, book_id, book_title):
            if event["type"] == "token":
                parts.append(event["text"])
            elif event["type"] == "meta":
                context_chunks = list(event["context_chunks"])
        return RAGResult(
            answer="".join(parts),
            context_chunks=context_chunks,
            chunks_added=0,
        )


# Синглтон
rag_service = RAGService()
