from dataclasses import dataclass


class EmptyTextError(Exception):
    """Submitted text has no non-whitespace content."""


@dataclass(frozen=True)
class CheckedText:
    original_length: int
    processed_text: str
    message: str


def check_text(text: str) -> CheckedText:
    if not text.strip():
        raise EmptyTextError

    # Пока просто эхо + базовая информация
    return CheckedText(
        original_length=len(text),
        processed_text=text.strip(),
        message="Текст получен. Полноценная обработка будет добавлена позже.",
    )
