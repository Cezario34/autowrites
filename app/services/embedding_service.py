from app.infrastructure.ollama import ollama_client


class EmbeddingService:
    def embed(self, text: str) -> list[float]:
        return ollama_client.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return ollama_client.embed_batch(texts)


# Синглтон
embedding_service = EmbeddingService()