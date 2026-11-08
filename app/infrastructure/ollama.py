import logging
from collections.abc import Iterator

import ollama

from app.core.config import settings
from app.core.prompts import SYSTEM_PROMPT

logger = logging.getLogger("app.ollama")

# Свой шаблон на один запрос: иначе Modelfile модели подставляет
# готовую фразу вроде «Противоречий нет» и затирает разбор.
OWN_TEMPLATE = "{{ if .System }}{{ .System }}\n\n{{ end }}{{ .Prompt }}"


def _generate_piece(chunk: object) -> str:
    if hasattr(chunk, "response"):
        return chunk.response or ""
    if isinstance(chunk, dict):
        return chunk.get("response") or ""
    return ""


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
        system_prompt = system if system is not None else SYSTEM_PROMPT
        logger.info(
            "этап: запрос модели, свой шаблон, символов=%s, system=%s",
            len(prompt),
            (system_prompt or "").splitlines()[0] if system_prompt else "-",
        )
        request: dict = {
            "model": self.llm_model,
            "prompt": prompt,
            "system": system_prompt,
            "template": OWN_TEMPLATE,
            "raw": False,
            "stream": True,
            "options": self._options(temperature),
        }
        if response_format:
            request["format"] = response_format
        stream = self.client.generate(**request)
        for chunk in stream:
            text = _generate_piece(chunk)
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
