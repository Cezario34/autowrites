from app.infrastructure.ollama import OWN_TEMPLATE, ollama_client


def test_generate_uses_our_template_and_joins_the_reply(monkeypatch):
    class Chunk:
        def __init__(self, text: str) -> None:
            self.response = text

    seen = {}

    def generate(**kwargs):
        seen.update(kwargs)
        yield Chunk("Цитата из ")
        yield Chunk("отрывка.")

    def chat(*args, **kwargs):
        raise AssertionError("чат модели подставляет её собственный шаблон")

    monkeypatch.setattr(ollama_client.client, "generate", generate)
    monkeypatch.setattr(ollama_client.client, "chat", chat)

    assert ollama_client.generate(
        "Мыло щипало кожу.",
        system="Разбери цитатами.",
        temperature=0.2,
    ) == "Цитата из отрывка."
    assert seen["raw"] is False
    assert seen["template"] == OWN_TEMPLATE
    assert seen["system"] == "Разбери цитатами."
    assert seen["prompt"] == "Мыло щипало кожу."
    assert seen["stream"] is True
    assert seen["options"]["temperature"] == 0.2
