from pydantic import BaseModel


class BookCreateRequest(BaseModel):
    title: str


class BookResponse(BaseModel):
    id: str
    title: str
    created_at: str
    chunk_count: int


class CharacterResponse(BaseModel):
    name: str
    description: str


class CharacterListResponse(BaseModel):
    characters: list[CharacterResponse]


class PlanResponse(BaseModel):
    text: str


class TextFileRequest(BaseModel):
    text: str
