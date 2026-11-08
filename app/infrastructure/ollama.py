from collections.abc import Iterator

import ollama

from app.core.config import settings
from app.core.prompts import SYSTEM_PROMPT


def _chunk_text(chunk: object) -> str:
    message = chunk.message if hasattr(chunk, "message") else chunk["message"]
    if isinstance(message, dict):
        return message.get("content") or ""
    return getattr(message, "content", None) or ""


class OllamaClient:
    def __init__(self) -> None:
        self.client = ollama.Client(host=settings.ollama_base_url)
        self.llm_model = settings.llm_model
        self.embedding_model = settings.embedding_model

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        # Если системный промпт не передали — берём дефолтный
        system_prompt = system if system is not None else SYSTEM_PROMPT
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _options(self, temperature: float) -> dict[str, float | int]:
        return {"temperature": temperature, "num_ctx": 16384}

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        response_format: str | None = None,
    ) -> str:
        # Ответ собираем из потока: иначе тело из нескольких JSON-строк
        # падает с «Extra data» при разборе одним объектом.
        return "".join(
            self.generate_stream(
                prompt,
                system=system,
                temperature=temperature,
                response_format=response_format,
            )
        )

    def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        response_format: str | None = None,
    ) -> Iterator[str]:
        request: dict = {
            "model": self.llm_model,
            "messages": self._messages(prompt, system),
            "stream": True,
            "options": self._options(temperature),
        }
        if response_format:
            request["format"] = response_format
        stream = self.client.chat(**request)
        for chunk in stream:
            text = _chunk_text(chunk)
            if text:
                yield text

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings(
            model=self.embedding_model,
            prompt=text,
        )
        return response["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


# Синглтон
ollama_client = OllamaClient()
