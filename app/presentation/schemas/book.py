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
    role: str = ""
    description: str = ""
    appearance: str = ""


class CharacterListResponse(BaseModel):
    characters: list[CharacterResponse]


class CharacterSupplementResponse(BaseModel):
    updated: bool
    characters: list[CharacterResponse]


class PlotChapterResponse(BaseModel):
    number: str
    description: str


class SavedChapterResultResponse(BaseModel):
    number: str
    plot_updated: bool
    characters_updated: bool


class BookSupplementResponse(BaseModel):
    chapters: list[SavedChapterResultResponse]
    plot: list[PlotChapterResponse]
    characters: list[CharacterResponse]


class PlanResponse(BaseModel):
    text: str


class TextFileRequest(BaseModel):
    text: str
