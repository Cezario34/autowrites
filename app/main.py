from fastapi import FastAPI

from app.core.logging import configure_logging
from app.infrastructure.cors import add_cors
from app.infrastructure.static import mount_frontend
from app.presentation.routers.books import router as books_router
from app.presentation.routers.check_text import router as check_text_router
from app.presentation.routers.health import router as health_router
from app.presentation.routers.process_chapter import router as process_chapter_router

configure_logging()

# после других include_router
app = FastAPI(title="AutoWriter API", version="0.1.0")

add_cors(app)
app.include_router(books_router)
app.include_router(check_text_router)
app.include_router(health_router)
app.include_router(process_chapter_router)

mount_frontend(app)
