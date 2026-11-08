from app.infrastructure.ollama import ollama_client


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _ChatChunk:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


def test_generate_uses_chat_like_ollama_run(monkeypatch):
    seen = {}

    def chat(**kwargs):
        seen.update(kwargs)
        yield _ChatChunk("Мыло щипало ")
        yield _ChatChunk("кожу, это и есть разбор.")

    def generate(**kwargs):
        raise AssertionError("разбор идёт через chat, шаблон модели применяет Ollama")

    monkeypatch.setattr(ollama_client.client, "chat", chat)
    monkeypatch.setattr(ollama_client.client, "generate", generate)

    assert ollama_client.generate(
        "Мыло щипало кожу.",
        system="Разбери цитатами.",
        temperature=0.2,
    ) == "Мыло щипало кожу, это и есть разбор."
    assert seen["model"] == ollama_client.llm_model
    assert seen["stream"] is True
    assert "raw" not in seen
    assert "think" not in seen
    assert seen["messages"][0] == {"role": "system", "content": "Разбери цитатами."}
    assert seen["messages"][1] == {"role": "user", "content": "Мыло щипало кожу."}
    assert seen["options"]["temperature"] == 0.2
    assert seen["options"]["num_ctx"] == 16384
