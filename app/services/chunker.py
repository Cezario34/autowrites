from app.core.config import settings


def split_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Нарезает текст на чанки с перекрытием.
    Старается резать по абзацам, если возможно.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if not text.strip():
        return []

    # Сначала пробуем резать по абзацам
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        # Если абзац сам по себе больше chunk_size — режем его жёстко
        if len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            chunks.extend(_split_long_text(paragraph, chunk_size, chunk_overlap))
            continue

        # Если текущий чанк + абзац влезает — добавляем
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{paragraph}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Начинаем новый чанк с перекрытием
            overlap_text = _get_overlap(current_chunk, chunk_overlap)
            current_chunk = f"{overlap_text}\n\n{paragraph}".strip()

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [c for c in chunks if c]


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Жёсткая нарезка длинного текста по символам с перекрытием."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def _get_overlap(text: str, overlap: int) -> str:
    """Берёт хвост текста для перекрытия."""
    if not text or overlap <= 0:
        return ""
    return text[-overlap:].strip()