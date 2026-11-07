from pydantic import BaseModel


class BookCreateRequest(BaseModel):
    title: str


class BookResponse(BaseModel):
    id: str
    title: str
    created_at: str
    chunk_count: int
