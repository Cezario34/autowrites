import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.application.books import UnknownBookError
from app.application.process_chapter import (
    EmptyChapterError,
    open_check_stream,
    open_save_stream,
)
from app.presentation.schemas.chapter import ChapterRequest

logger = logging.getLogger("app.api")
router = APIRouter()


def _public_error(exc: Exception) -> str:
    message = str(exc).strip()
    lowered = message.casefold()
    if "<!doctype" in lowered or "<html" in lowered:
        return "Сервер модели не принял запрос (404). Разбор текста не начался."
    return message or "Не удалось обработать главу"


def _ndjson(events: Iterator[dict]) -> StreamingResponse:
    def body():
        try:
            for event in events:
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:
            logger.exception("этап: ошибка")
            message = _public_error(exc)
            yield json.dumps(
                {"type": "error", "detail": message},
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        body(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _open(request: ChapterRequest, opener):
    try:
        return opener(text=request.text, book_id=request.book_id)
    except UnknownBookError:
        raise HTTPException(status_code=400, detail="Книга не найдена") from None
    except EmptyChapterError:
        raise HTTPException(status_code=400, detail="Глава пустая") from None


@router.post("/api/check-chapter")
async def check_chapter_endpoint(request: ChapterRequest):
    return _ndjson(_open(request, open_check_stream))


@router.post("/api/save-chapter")
async def save_chapter_endpoint(request: ChapterRequest):
    return _ndjson(_open(request, open_save_stream))
