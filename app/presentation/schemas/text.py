from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str


class TextResponse(BaseModel):
    original_length: int
    processed_text: str
    message: str
