import logging
from collections.abc import Iterator

import ollama
from ollama import ResponseError

from app.core.config import settings
from app.core.prompts import SYSTEM_PROMPT

logger = logging.getLogger("app.ollama")


def _chunk_text(chunk: object) -> str:
    message = chunk.message if hasattr(chunk, "message") else chunk["message"]
    if isinstance(message, dict):
        return message.get("content") or ""
    return getattr(message, "content", None) or ""


def _response_text(chunk: object) -> str:
    if hasattr(chunk, "response"):
        return chunk.response or ""
    if isinstance(chunk, dict):
        return chunk.get("response") or ""
    return ""


def _plain_prompt(system: str, prompt: str) -> str:
    if system:
        return f"{system}\n\n{prompt}"
    return prompt


class OllamaClient:
    def __init__(self) -> None:
        self.client = ollama.Client(host=settings.ollama_base_url)
        self.llm_model = settings.llm_model
        self.embedding_model = settings.embedding_model

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
        parts = list(
            self.generate_stream(
                prompt,
                system=system,
                temperature=temperature,
                response_format=response_format,
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
    ) -> Iterator[str]:
        system_prompt = SYSTEM_PROMPT if system is None else system
        plain = _plain_prompt(system_prompt, prompt)
        logger.info(
            "этап: запрос модели, символов=%s, начало задания=%s",
            len(prompt),
            " ".join(prompt.split())[:120],
        )
        try:
            yield from self._raw_stream(plain, temperature, response_format)
            return
        except ResponseError as exc:
            if exc.status_code != 404:
                raise
            logger.error(
                "этап: /api/generate нет, остаюсь на /api/chat. Шаблон модели может выкинуть текст."
            )
        yield from self._chat_stream(prompt, system_prompt, temperature, response_format)

    def _raw_stream(
        self,
        prompt: str,
        temperature: float,
        response_format: str | None,
    ) -> Iterator[str]:
        # raw обходит шаблон Modelfile: иначе модель не видит присланный текст
        # и отвечает заготовкой вроде «Противоречий нет».
        request: dict = {
            "model": self.llm_model,
            "prompt": prompt,
            "raw": True,
            "stream": True,
            "think": False,
            "options": self._options(temperature),
        }
        if response_format:
            request["format"] = response_format
        stream = self.client.generate(**request)
        for chunk in stream:
            text = _response_text(chunk)
            if text:
                yield text

    def _chat_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        response_format: str | None,
    ) -> Iterator[str]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        request: dict = {
            "model": self.llm_model,
            "messages": messages,
            "stream": True,
            "think": False,
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
