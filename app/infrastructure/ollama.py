import logging
from collections.abc import Iterator

import ollama

from app.core.config import settings
from app.core.prompts import SYSTEM_PROMPT

logger = logging.getLogger("app.ollama")


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

    def _options(
        self,
        temperature: float,
        num_predict: int | None = None,
    ) -> dict[str, float | int]:
        options: dict[str, float | int] = {"temperature": temperature, "num_ctx": 16384}
        if num_predict is not None:
            options["num_predict"] = num_predict
        return options

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        response_format: str | None = None,
        num_predict: int | None = None,
    ) -> str:
        # Ответ собираем из потока: иначе тело из нескольких JSON-строк
        # падает с «Extra data» при разборе одним объектом.
        parts = list(
            self.generate_stream(
                prompt,
                system=system,
                temperature=temperature,
                response_format=response_format,
                num_predict=num_predict,
            )
        )
        text = "".join(parts)
        logger.info(
            "этап: ответ модели, кусков=%s, символов=%s, format=%s\n%s",
            len(parts),
            len(text),
            response_format or "-",
            text,
        )
        return text

    def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        response_format: str | None = None,
        num_predict: int | None = None,
    ) -> Iterator[str]:
        """Тот же вызов, что и ollama run: chat и шаблон модели из Modelfile."""
        system_prompt = SYSTEM_PROMPT if system is None else system
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        logger.info(
            "этап: запрос модели, chat, символов=%s, system=%s",
            len(prompt),
            system_prompt.splitlines()[0] if system_prompt else "-",
        )
        request: dict = {
            "model": self.llm_model,
            "messages": messages,
            "stream": True,
            "options": self._options(temperature, num_predict),
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
