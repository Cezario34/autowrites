from app.infrastructure.ollama import ollama_client


def test_generate_joins_streamed_pieces(monkeypatch):
    class Message:
        content = ""

        def __init__(self, content: str) -> None:
            self.content = content

    class Chunk:
        def __init__(self, content: str) -> None:
            self.message = Message(content)

    def chat(*args, **kwargs):
        assert kwargs["stream"] is True
        yield Chunk('["Герой ')
        yield Chunk('вошёл."]')

    monkeypatch.setattr(ollama_client.client, "chat", chat)
    assert ollama_client.generate("глава", system="только json", temperature=0.1) == (
        '["Герой вошёл."]'
    )
