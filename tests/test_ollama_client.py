from ollama import ResponseError

from app.infrastructure.ollama import ollama_client


class _GenerateChunk:
    def __init__(self, text: str) -> None:
        self.response = text


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _ChatChunk:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


def _offline(monkeypatch):
    ollama_client._reading_model_name = None

    def show(model):
        raise ResponseError("нет модели", 404)

    monkeypatch.setattr(ollama_client.client, "show", show)


def test_generate_sends_the_text_outside_the_model_template(monkeypatch):
    seen = {}
    _offline(monkeypatch)

    def generate(**kwargs):
        seen.update(kwargs)
        yield _GenerateChunk("Мыло щипало ")
        yield _GenerateChunk("кожу, это и есть разбор.")

    def chat(**kwargs):
        raise AssertionError("сырой запрос не должен падать в chat")

    monkeypatch.setattr(ollama_client.client, "generate", generate)
    monkeypatch.setattr(ollama_client.client, "chat", chat)

    assert ollama_client.generate(
        "Мыло щипало кожу.",
        system="Разбери цитатами.",
        temperature=0.2,
    ) == "Мыло щипало кожу, это и есть разбор."
    assert seen["raw"] is True
    assert seen["think"] is False
    assert "Мыло щипало кожу." in seen["prompt"]
    assert seen["prompt"].startswith("Разбери цитатами.")
    assert seen["options"]["temperature"] == 0.2


def test_generate_falls_back_to_chat_when_generate_route_is_missing(monkeypatch):
    seen = {}
    _offline(monkeypatch)

    def generate(**kwargs):
        raise ResponseError("Nothing matches the given URI.", 404)

    def chat(**kwargs):
        seen.update(kwargs)
        yield _ChatChunk("Разбор из chat.")

    monkeypatch.setattr(ollama_client.client, "generate", generate)
    monkeypatch.setattr(ollama_client.client, "chat", chat)

    assert ollama_client.generate("Мыло щипало кожу.", system="Разбери.", temperature=0.2) == (
        "Разбор из chat."
    )
    assert seen["think"] is False
    assert seen["messages"][1]["content"] == "Мыло щипало кожу."
