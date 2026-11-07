import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.application.process_chapter import EmptyChapterError, open_chapter_stream
from app.presentation.schemas.chapter import ChapterRequest

logger = logging.getLogger("app.api")
router = APIRouter()


@router.post("/api/process-chapter")
async def process_chapter_endpoint(request: ChapterRequest):
    try:
        events = open_chapter_stream(
            text=request.text,
            add_to_store=request.add_to_store,
        )
    except EmptyChapterError:
        raise HTTPException(status_code=400, detail="Глава пустая") from None

    def ndjson():
        try:
            for event in events:
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:
            logger.exception("этап: ошибка")
            yield json.dumps(
                {"type": "error", "detail": str(exc)},
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        ndjson(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
