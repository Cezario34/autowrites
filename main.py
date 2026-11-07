from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(title="AutoWriter API", version="0.1.0")

# CORS — на случай если фронт будет отдельно
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    original_length: int
    processed_text: str
    message: str

@app.post("/api/check-text", response_model=TextResponse)
async def check_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст пустой")

    # Пока просто эхо + базовая информация
    length = len(request.text)
    processed = request.text.strip()

    return TextResponse(
        original_length=length,
        processed_text=processed,
        message="Текст получен. Полноценная обработка будет добавлена позже."
    )

@app.get("/api/health")
async def health():
    return {"status": "ok"}

# Статика — фронт
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")