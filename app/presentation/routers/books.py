from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.application.books import (
    BookTitleTooLongError,
    EmptyBookTitleError,
    EmptyChapterError,
    UnknownBookError,
    create_book,
    list_books,
    list_characters,
    read_plan,
    save_chapter_text,
)
from app.presentation.schemas.book import (
    BookCreateRequest,
    BookResponse,
    CharacterListResponse,
    CharacterResponse,
    PlanResponse,
    TextFileRequest,
)

router = APIRouter()


def _response(book) -> BookResponse:
    return BookResponse(
        id=book.id,
        title=book.title,
        created_at=book.created_at,
        chunk_count=book.chunk_count,
    )


@router.get("/api/books", response_model=list[BookResponse])
async def list_books_endpoint():
    return [_response(book) for book in list_books()]


@router.post("/api/books", response_model=BookResponse, status_code=201)
async def create_book_endpoint(request: BookCreateRequest):
    try:
        book = create_book(request.title)
    except EmptyBookTitleError:
        raise HTTPException(status_code=400, detail="Название книги пустое") from None
    except BookTitleTooLongError:
        raise HTTPException(
            status_code=400,
            detail="Название книги слишком длинное",
        ) from None
    return _response(book)


@router.get("/api/books/{book_id}/characters", response_model=CharacterListResponse)
async def list_characters_endpoint(book_id: str):
    try:
        characters = list_characters(book_id)
    except UnknownBookError:
        raise HTTPException(status_code=400, detail="Книга не найдена") from None
    return CharacterListResponse(
        characters=[
            CharacterResponse(
                name=character.name,
                role=character.role,
                description=character.description,
            )
            for character in characters
        ]
    )


@router.get("/api/books/{book_id}/plan", response_model=PlanResponse)
async def read_plan_endpoint(book_id: str):
    try:
        text = read_plan(book_id)
    except UnknownBookError:
        raise HTTPException(status_code=400, detail="Книга не найдена") from None
    return PlanResponse(text=text)


def _text_filename(title: str) -> str:
    cleaned = "".join(char if char not in '\\/:*?"<>|' else " " for char in title)
    cleaned = " ".join(cleaned.split()) or "текст"
    return f"{cleaned}.txt"


@router.post("/api/books/{book_id}/text")
async def save_text_endpoint(book_id: str, request: TextFileRequest):
    try:
        book, text = save_chapter_text(book_id, request.text)
    except UnknownBookError:
        raise HTTPException(status_code=400, detail="Книга не найдена") from None
    except EmptyChapterError:
        raise HTTPException(status_code=400, detail="Глава пустая") from None
    filename = quote(_text_filename(book.title))
    return Response(
        content=text,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )
