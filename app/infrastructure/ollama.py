import ollama
from app.core.config import settings
from app.core.prompts import SYSTEM_PROMPT


class OllamaClient:
    def __init__(self) -> None:
        self.client = ollama.Client(host=settings.ollama_base_url)
        self.llm_model = settings.llm_model
        self.embedding_model = settings.embedding_model

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
    ) -> str:
        messages = []

        # Если системный промпт не передали — берём дефолтный
        system_prompt = system if system is not None else SYSTEM_PROMPT
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=self.llm_model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_ctx": 16384,
            },
        )
        return response["message"]["content"]

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