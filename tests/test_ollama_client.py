from app.infrastructure.ollama import ollama_client


def test_generate_uses_chat_and_joins_the_reply(monkeypatch):
    class Message:
        def __init__(self, content: str) -> None:
            self.content = content

    class Chunk:
        def __init__(self, content: str) -> None:
            self.message = Message(content)

    seen = {}

    def chat(**kwargs):
        seen.update(kwargs)
        yield Chunk("Цитата из ")
        yield Chunk("отрывка.")

    def generate(**kwargs):
        raise AssertionError("адрес /api/generate у этого сервера отвечает 404")

    monkeypatch.setattr(ollama_client.client, "chat", chat)
    monkeypatch.setattr(ollama_client.client, "generate", generate)

    assert ollama_client.generate(
        "Мыло щипало кожу.",
        system="Разбери цитатами.",
        temperature=0.2,
    ) == "Цитата из отрывка."
    assert seen["stream"] is True
    assert seen["messages"][0] == {"role": "system", "content": "Разбери цитатами."}
    assert seen["messages"][1] == {"role": "user", "content": "Мыло щипало кожу."}
    assert seen["options"]["temperature"] == 0.2
