from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.application.books import (
    BookTitleTooLongError,
    CharacterSupplementError,
    EmptyBookTitleError,
    EmptyChapterError,
    EmptyDirectionError,
    NoSavedChapterError,
    PlotSupplementError,
    UnknownBookError,
    create_book,
    list_books,
    list_characters,
    read_plan,
    read_plot,
    review_chapter,
    save_chapter_text,
    suggest_chapter,
    supplement_characters,
    supplement_saved_chapters,
)
from app.infrastructure.openai_client import MissingOpenAIKeyError, OpenAIRequestError
from app.presentation.schemas.book import (
    BookCreateRequest,
    BookResponse,
    BookSupplementResponse,
    CharacterListResponse,
    CharacterResponse,
    CharacterSupplementResponse,
    PlanResponse,
    PlotChapterResponse,
    TextReviewResponse,
    PlotResponse,
    SavedChapterResultResponse,
    SuggestChapterRequest,
    SuggestChapterResponse,
    TextFileRequest,
)

router = APIRouter()


def _character_payload(characters) -> list[CharacterResponse]:
    return [
        CharacterResponse(
            name=character.name,
            role=character.role,
            description=character.description,
            appearance=character.appearance,
        )
        for character in characters
    ]


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
    return CharacterListResponse(characters=_character_payload(characters))


@router.post(
    "/api/books/{book_id}/characters/supplement",
    response_model=CharacterSupplementResponse,
)
async def supplement_characters_endpoint(book_id: str, request: TextFileRequest):
    try:
        characters, updated = supplement_characters(book_id, request.text)
    except UnknownBookError:
        raise HTTPException(status_code=400, detail="Книга не найдена") from None
    except EmptyChapterError:
        raise HTTPException(status_code=400, detail="Глава пустая") from None
    except MissingOpenAIKeyError:
        raise HTTPException(
            status_code=400,
            detail="В окружении нет ключа OpenAI",
        ) from None
    except OpenAIRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    except CharacterSupplementError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return CharacterSupplementResponse(
        updated=updated,
        characters=_character_payload(characters),
    )


@router.post(
    "/api/books/{book_id}/supplement",
    response_model=BookSupplementResponse,
)
async def supplement_saved_chapters_endpoint(book_id: str):
    try:
        results, plot, characters = supplement_saved_chapters(book_id)
    except UnknownBookError:
        raise HTTPException(status_code=400, detail="Книга не найдена") from None
    except NoSavedChapterError:
        raise HTTPException(
            status_code=400,
            detail="Сначала сохрани главу как текст",
        ) from None
    except EmptyChapterError:
        raise HTTPException(status_code=400, detail="Глава пустая") from None
    except MissingOpenAIKeyError:
        raise HTTPException(
            status_code=400,
            detail="В окружении нет ключа OpenAI",
        ) from None
    except OpenAIRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    except (PlotSupplementError, CharacterSupplementError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return BookSupplementResponse(
        chapters=[
            SavedChapterResultResponse(
                number=item.number,
                plot_updated=item.plot_updated,
                characters_updated=item.characters_updated,
            )
            for item in results
        ],
        plot=[
            PlotChapterResponse(number=item.number, description=item.description)
            for item in plot
        ],
        characters=_character_payload(characters),
    )


@router.get("/api/books/{book_id}/plot", response_model=PlotResponse)
async def read_plot_endpoint(book_id: str):
    try:
        chapters = read_plot(book_id)
    except UnknownBookError:
        raise HTTPException(status_code=400, detail="Книга не найдена") from None
    return PlotResponse(
        chapters=[
            PlotChapterResponse(number=chapter.number, description=chapter.description)
            for chapter in chapters
        ]
    )


@router.get("/api/books/{book_id}/plan", response_model=PlanResponse)
async def read_plan_endpoint(book_id: str):
    try:
        text = read_plan(book_id)
    except UnknownBookError:
        raise HTTPException(status_code=400, detail="Книга не найдена") from None
    return PlanResponse(text=text)


@router.post("/api/books/{book_id}/review", response_model=TextReviewResponse)
async def review_chapter_endpoint(book_id: str, request: TextFileRequest):
    try:
        review = review_chapter(book_id, request.text)
    except UnknownBookError:
        raise HTTPException(status_code=400, detail="Книга не найдена") from None
    except EmptyChapterError:
        raise HTTPException(status_code=400, detail="Глава пустая") from None
    except MissingOpenAIKeyError:
        raise HTTPException(
            status_code=400,
            detail="В окружении нет ключа OpenAI",
        ) from None
    except OpenAIRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    return TextReviewResponse(review=review)


@router.post("/api/books/{book_id}/suggest", response_model=SuggestChapterResponse)
async def suggest_chapter_endpoint(book_id: str, request: SuggestChapterRequest):
    try:
        chapter, model = suggest_chapter(book_id, request.prompt, request.text)
    except UnknownBookError:
        raise HTTPException(status_code=400, detail="Книга не найдена") from None
    except EmptyDirectionError:
        raise HTTPException(
            status_code=400,
            detail="Напиши, куда вести сюжет",
        ) from None
    except MissingOpenAIKeyError:
        raise HTTPException(
            status_code=400,
            detail="В окружении нет ключа OpenAI",
        ) from None
    except OpenAIRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    return SuggestChapterResponse(chapter=chapter, model=model)


@router.post("/api/books/{book_id}/text")
async def save_text_endpoint(book_id: str, request: TextFileRequest):
    try:
        _book, text, filename = save_chapter_text(book_id, request.text)
    except UnknownBookError:
        raise HTTPException(status_code=400, detail="Книга не найдена") from None
    except EmptyChapterError:
        raise HTTPException(status_code=400, detail="Глава пустая") from None
    filename = quote(filename)
    return Response(
        content=text,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )
