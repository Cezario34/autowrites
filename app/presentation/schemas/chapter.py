from pydantic import BaseModel


class ChapterRequest(BaseModel):
    text: str
    book_id: str


class ChapterResponse(BaseModel):
    answer: str
    context_chunks: list[str]
    chunks_added: int
    original_length: int