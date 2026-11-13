from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # Модели
    llm_model: str = "qwen32b-rag"
    embedding_model: str = "nomic-embed-text"

    # RAG
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5

    # Chroma
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "chapters"

    # OpenAI. Ключ читается из OPENAI_API_KEY, модель можно сменить через OPENAI_MODEL.
    # Главу предлагает отдельная модель, её можно сменить через OPENAI_CHAPTER_MODEL.
    openai_api_key: str = ""
    openai_model: str = "gpt-5.4-mini"
    openai_chapter_model: str = "gpt-6.1-sol"
    openai_base_url: str = "https://api.openai.com/v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()