from fastapi import APIRouter, HTTPException

from app.application.check_text import EmptyTextError, check_text as run_check_text
from app.presentation.schemas.text import TextRequest, TextResponse

router = APIRouter()


@router.post("/api/check-text", response_model=TextResponse)
async def check_text(request: TextRequest):
    try:
        result = run_check_text(request.text)
    except EmptyTextError:
        raise HTTPException(status_code=400, detail="Текст пустой") from None

    return TextResponse(
        original_length=result.original_length,
        processed_text=result.processed_text,
        message=result.message,
    )
