from pydantic import BaseModel


class ChapterRequest(BaseModel):
    text: str
    add_to_store: bool = True


class ChapterResponse(BaseModel):
    answer: str
    context_chunks: list[str]
    chunks_added: int
    original_length: int