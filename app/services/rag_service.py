import logging
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from app.core.config import settings
from app.core.prompts import CHECK_CHAPTER_PROMPT
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


def _stage(stage: str, message: str) -> dict[str, str]:
    return {"type": "stage", "stage": stage, "message": message}


class RAGService:
    def _save_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        chapter_id: str | None = None,
    ) -> int:
        if not chunks:
            return 0

        chapter_id = chapter_id or str(uuid.uuid4())
        ids = [f"{chapter_id}-{i}" for i in range(len(chunks))]
        metadatas = [
            {"chapter_id": chapter_id, "chunk_index": i} for i in range(len(chunks))
        ]
        vector_store.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        return len(chunks)

    def add_chapter(self, text: str, chapter_id: str | None = None) -> int:
        """Нарезает главу, делает эмбеддинги и сохраняет в векторную базу."""
        chunks = split_text(text)
        if not chunks:
            return 0

        embeddings = embedding_service.embed_batch(chunks)
        return self._save_chunks(chunks, embeddings, chapter_id)

    def query(self, text: str, n_results: int | None = None) -> list[str]:
        """Ищет релевантные чанки по тексту."""
        # Берём только начало текста для эмбеддинга (чтобы не превысить контекст)
        query_text = text[:2000]
        query_embedding = embedding_service.embed(query_text)
        results = vector_store.query(query_embedding, n_results=n_results)

        documents = results.get("documents", [[]])[0]
        return documents or []

    def iter_process_chapter(
        self,
        text: str,
        add_to_store: bool = True,
    ) -> Iterator[dict]:
        """
        Полный цикл с событиями для стрима:
        1. (опционально) добавляет главу в базу
        2. ищет релевантный контекст
        3. формирует промпт
        4. отдаёт ответ модели по кускам
        """
        logger.info(
            "этап: старт, символов=%s, в базу=%s",
            len(text),
            add_to_store,
        )
        yield _stage("start", "Идёт обработка…")

        chunks_added = 0
        if add_to_store:
            yield _stage("chunk", "Нарезаю главу…")
            chunks = split_text(text)
            logger.info("этап: нарезка, чанков=%s", len(chunks))
            if chunks:
                yield _stage("embed", "Считаю эмбеддинги…")
                logger.info(
                    "этап: эмбеддинги, чанков=%s, модель=%s",
                    len(chunks),
                    settings.embedding_model,
                )
                embeddings = embedding_service.embed_batch(chunks)
                yield _stage("store", "Сохраняю главу…")
                chunks_added = self._save_chunks(chunks, embeddings)
                logger.info("этап: запись в базу, чанков=%s", chunks_added)
            else:
                logger.info("этап: запись в базу пропущена, чанков=0")

        yield _stage("search", "Ищу контекст…")
        context_chunks = self.query(text, n_results=settings.top_k)
        logger.info("этап: поиск, фрагментов=%s", len(context_chunks))
        context = (
            "\n\n---\n\n".join(context_chunks)
            if context_chunks
            else "Контекст отсутствует."
        )
        prompt = CHECK_CHAPTER_PROMPT.format(context=context, chapter=text)

        yield {
            "type": "meta",
            "chunks_added": chunks_added,
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

    def process_chapter(self, text: str, add_to_store: bool = True) -> RAGResult:
        parts: list[str] = []
        context_chunks: list[str] = []
        chunks_added = 0
        for event in self.iter_process_chapter(text, add_to_store=add_to_store):
            if event["type"] == "token":
                parts.append(event["text"])
            elif event["type"] == "meta":
                context_chunks = list(event["context_chunks"])
                chunks_added = event["chunks_added"]
        return RAGResult(
            answer="".join(parts),
            context_chunks=context_chunks,
            chunks_added=chunks_added,
        )


# Синглтон
rag_service = RAGService()
