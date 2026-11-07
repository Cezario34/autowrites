import uuid
from dataclasses import dataclass

from app.core.config import settings
from app.core.prompts import CHECK_CHAPTER_PROMPT
from app.infrastructure.ollama import ollama_client
from app.infrastructure.vector_store import vector_store
from app.services.chunker import split_text
from app.services.embedding_service import embedding_service


@dataclass
class RAGResult:
    answer: str
    context_chunks: list[str]
    chunks_added: int


class RAGService:
    def add_chapter(self, text: str, chapter_id: str | None = None) -> int:
        """Нарезает главу, делает эмбеддинги и сохраняет в векторную базу."""
        chunks = split_text(text)
        if not chunks:
            return 0

        embeddings = embedding_service.embed_batch(chunks)
        chapter_id = chapter_id or str(uuid.uuid4())

        ids = [f"{chapter_id}-{i}" for i in range(len(chunks))]
        metadatas = [{"chapter_id": chapter_id, "chunk_index": i} for i in range(len(chunks))]

        vector_store.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        return len(chunks)

    def query(self, text: str, n_results: int | None = None) -> list[str]:
        """Ищет релевантные чанки по тексту."""
        # Берём только начало текста для эмбеддинга (чтобы не превысить контекст)
        query_text = text[:2000]
        query_embedding = embedding_service.embed(query_text)
        results = vector_store.query(query_embedding, n_results=n_results)

        documents = results.get("documents", [[]])[0]
        return documents or []

    def process_chapter(self, text: str, add_to_store: bool = True) -> RAGResult:
        """
        Полный цикл:
        1. (опционально) добавляет главу в базу
        2. ищет релевантный контекст
        3. формирует промпт
        4. получает ответ от модели
        """
        chunks_added = 0
        if add_to_store:
            chunks_added = self.add_chapter(text)

        context_chunks = self.query(text, n_results=settings.top_k)
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "Контекст отсутствует."

        prompt = CHECK_CHAPTER_PROMPT.format(
            context=context,
            chapter=text,
        )

        answer = ollama_client.generate(prompt)

        return RAGResult(
            answer=answer,
            context_chunks=context_chunks,
            chunks_added=chunks_added,
        )


# Синглтон
rag_service = RAGService()