from fastapi import APIRouter, HTTPException

from app.application.process_chapter import EmptyChapterError, process_chapter
from app.presentation.schemas.chapter import ChapterRequest, ChapterResponse

router = APIRouter()


@router.post("/api/process-chapter", response_model=ChapterResponse)
async def process_chapter_endpoint(request: ChapterRequest):
    try:
        result = process_chapter(
            text=request.text,
            add_to_store=request.add_to_store,
        )
    except EmptyChapterError:
        raise HTTPException(status_code=400, detail="Глава пустая") from None

    return ChapterResponse(
        answer=result.answer,
        context_chunks=result.context_chunks,
        chunks_added=result.chunks_added,
        original_length=result.original_length,
    )