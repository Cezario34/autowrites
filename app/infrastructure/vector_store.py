from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings


class VectorStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def count_where(self, where: dict[str, Any]) -> int:
        found = self.collection.get(where=where, include=["metadatas"])
        return len(found.get("ids") or [])

    def documents_where(self, where: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        found = self.collection.get(where=where, include=["documents", "metadatas"])
        documents = found.get("documents") or []
        metadatas = found.get("metadatas") or []
        rows: list[tuple[str, dict[str, Any]]] = []
        for index, document in enumerate(documents):
            meta = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            rows.append((document or "", meta))
        return rows

    def query(
        self,
        query_embedding: list[float],
        n_results: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        n_results = n_results or settings.top_k
        if where is not None:
            available = self.count_where(where)
            if available == 0:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            n_results = min(n_results, available)
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

    def delete_where(self, where: dict[str, Any]) -> None:
        if self.count_where(where) == 0:
            return
        self.collection.delete(where=where)

    def count(self) -> int:
        return self.collection.count()

    def delete_collection(self) -> None:
        self.client.delete_collection(settings.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


# Синглтон
vector_store = VectorStore()