from fastapi import APIRouter, HTTPException

from app.application.books import (
    BookTitleTooLongError,
    EmptyBookTitleError,
    create_book,
    list_books,
)
from app.presentation.schemas.book import BookCreateRequest, BookResponse

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
