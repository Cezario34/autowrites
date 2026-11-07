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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()