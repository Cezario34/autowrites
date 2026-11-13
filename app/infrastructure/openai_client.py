import logging

from app.core.config import settings

logger = logging.getLogger("app.openai")


class MissingOpenAIKeyError(Exception):
    """В окружении нет OPENAI_API_KEY."""


class OpenAIRequestError(Exception):
    """Запрос к OpenAI не удался."""


class OpenAIClient:
    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float | None = 0.2,
        model: str | None = None,
    ) -> str:
        key = settings.openai_api_key.strip()
        if not key:
            raise MissingOpenAIKeyError
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=settings.openai_base_url)
        chosen = (model or settings.openai_model).strip()
        logger.info(
            "этап: запрос openai, модель=%s, символов=%s",
            chosen,
            len(user),
        )
        request = {
            "model": chosen,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "timeout": 120,
        }
        if temperature is not None:
            request["temperature"] = temperature
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:
            detail = str(exc).strip().splitlines()[0][:300]
            if "sk-" in detail or key in detail:
                detail = "OpenAI не ответил"
            logger.error("этап: openai не ответил: %s", detail or exc.__class__.__name__)
            raise OpenAIRequestError(detail or "OpenAI не ответил") from exc
        content = ""
        if response.choices:
            content = response.choices[0].message.content or ""
        logger.info(
            "этап: ответ openai, символов=%s\n%s",
            len(content),
            content,
        )
        return content

    def complete_json(self, system: str, user: str) -> str:
        return self.complete(system, user, json_mode=True)


openai_client = OpenAIClient()
