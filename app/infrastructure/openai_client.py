import logging

from app.core.config import settings

logger = logging.getLogger("app.openai")


class MissingOpenAIKeyError(Exception):
    """В окружении нет OPENAI_API_KEY."""


class OpenAIRequestError(Exception):
    """Запрос к OpenAI не удался."""


class OpenAIClient:
    def complete_json(self, system: str, user: str) -> str:
        key = settings.openai_api_key.strip()
        if not key:
            raise MissingOpenAIKeyError
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=settings.openai_base_url)
        logger.info(
            "этап: запрос openai, модель=%s, символов=%s",
            settings.openai_model,
            len(user),
        )
        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                timeout=120,
            )
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


openai_client = OpenAIClient()
