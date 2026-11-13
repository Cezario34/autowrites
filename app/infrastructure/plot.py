import json
import re
import threading
from dataclasses import dataclass

from app.infrastructure.book_files import PLOT_FILE, library

_LIST_KEYS = ("сюжет", "plot", "главы", "chapters")
_NUMBER_KEYS = ("глава", "номер", "chapter", "number")
_DESCRIPTION_KEYS = ("описание", "description")
_NUMBER = re.compile(r"\d+")


@dataclass(frozen=True)
class PlotChapter:
    number: str
    description: str


def chapter_number(value: str) -> str:
    match = _NUMBER.search(value or "")
    if not match:
        return ""
    return str(int(match.group(0)))


def _field(item: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def plot_from_item(item: dict) -> PlotChapter | None:
    if not isinstance(item, dict):
        return None
    number = chapter_number(_field(item, _NUMBER_KEYS))
    description = _field(item, _DESCRIPTION_KEYS)
    if not number or not description:
        return None
    return PlotChapter(number=number, description=description)


def plot_from_payload(payload: object) -> list[PlotChapter]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = None
        for key in _LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                items = value
                break
        if items is None and _field(payload, _NUMBER_KEYS):
            items = [payload]
        if items is None:
            return []
    else:
        return []
    chapters: list[PlotChapter] = []
    for item in items:
        chapter = plot_from_item(item)
        if chapter:
            chapters.append(chapter)
    return chapters


def is_empty_plot_payload(payload: object) -> bool:
    if payload in ([], {}):
        return True
    if isinstance(payload, dict):
        for key in _LIST_KEYS:
            if key in payload and payload[key] == []:
                return True
    return False


def plot_record(chapter: PlotChapter) -> dict[str, str]:
    return {"глава": chapter.number, "описание": chapter.description}


def plot_document(chapters: list[PlotChapter]) -> str:
    return json.dumps(
        {"сюжет": [plot_record(chapter) for chapter in chapters]},
        ensure_ascii=False,
        indent=2,
    )


def plot_from_model_text(raw: str) -> list[PlotChapter]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    payload = json.loads(text)
    chapters = plot_from_payload(payload)
    if chapters or is_empty_plot_payload(payload):
        return chapters
    raise json.JSONDecodeError("JSON не сюжет", text, 0)


def supplement_plot_decision(
    existing: list[PlotChapter],
    proposed: list[PlotChapter],
) -> tuple[str, list[PlotChapter]]:
    """«same», если новых событий нет; «added», если сюжет только вырос; «lost», если ответ стёр уже записанное."""
    existing_by = {chapter.number: chapter for chapter in existing}
    ordered: list[PlotChapter] = []
    proposed_by: dict[str, int] = {}
    for chapter in proposed:
        slot = proposed_by.get(chapter.number)
        if slot is None:
            proposed_by[chapter.number] = len(ordered)
            ordered.append(chapter)
        else:
            ordered[slot] = chapter
    for number, current in existing_by.items():
        slot = proposed_by.get(number)
        incoming = ordered[slot] if slot is not None else None
        stored = current.description.strip()
        if incoming is None or (stored and stored not in incoming.description.strip()):
            return "lost", existing
    added = any(number not in existing_by for number in proposed_by)
    if not added:
        added = any(
            current.description.strip() != ordered[proposed_by[number]].description.strip()
            for number, current in existing_by.items()
        )
    if not added:
        return "same", existing
    return "added", ordered


class PlotStore:
    """Краткий сюжет лежит в папке книги рядом с characters.json."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def list_for_book(self, book_id: str) -> list[PlotChapter]:
        with self._lock:
            raw = library.read(book_id, PLOT_FILE)
            if not raw.strip():
                return []
            return plot_from_payload(json.loads(raw))

    def replace(self, book_id: str, chapters: list[PlotChapter]) -> None:
        with self._lock:
            path = library.folder(book_id) / PLOT_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(plot_document(chapters) + "\n", encoding="utf-8")
            temporary.replace(path)


plot_store = PlotStore()
