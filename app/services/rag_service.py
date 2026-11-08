import hashlib
import json
import logging
import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from app.core.config import settings
from app.core.prompts import (
    CHARACTER_CHAPTER_PROMPT,
    CHARACTER_SYSTEM_PROMPT,
    CHECK_AGAINST_RAG_PROMPT,
    CHECK_LOGIC_PROMPT,
    CHECK_RETRY_PROMPT,
    CHECK_SYSTEM_PROMPT,
    SPLIT_CHAPTER_PROMPT,
    SPLIT_SYSTEM_PROMPT,
)
from app.infrastructure.characters import (
    Character,
    character_store,
    merge_characters,
)
from app.infrastructure.ollama import ollama_client
from app.infrastructure.vector_store import vector_store
from app.services.chunker import split_text
from app.services.embedding_service import embedding_service

logger = logging.getLogger("app.rag")


@dataclass
class RAGResult:
    answer: str
    context_chunks: list[str]
    chunks_added: int


class SplitError(Exception):
    """Модель не вернула пригодный список фрагментов."""


class CharacterParseError(Exception):
    """Модель не вернула пригодный список персонажей."""


def _stage(stage: str, message: str) -> dict[str, str]:
    return {"type": "stage", "stage": stage, "message": message}


def _fill(template: str, **values: object) -> str:
    """Подставляет поля, не трогая фигурные скобки внутри самого текста."""
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def _opening(text: str, limit: int = 48) -> str:
    return " ".join(text.split())[:limit]


def _is_stock_reply(reply: str) -> bool:
    folded = " ".join(reply.split()).casefold().strip(" .!")
    if not folded:
        return True
    if folded.startswith("противоречий нет"):
        return True
    return "логика держится" in folded and len(folded) < 160


def _reply_skips_text(reply: str, source: str) -> bool:
    if _is_stock_reply(reply):
        return True
    compact_source = " ".join(source.split())
    compact_reply = " ".join(reply.split())
    if not compact_source:
        return False
    if len(compact_source) <= 80:
        return compact_source.casefold() not in compact_reply.casefold()
    return compact_source[:48].casefold() not in compact_reply.casefold()


def _check_system(text: str) -> str:
    return CHECK_SYSTEM_PROMPT + "\n\nПрисланный текст, его нужно разобрать:\n" + text


def _ask_about_text(prompt: str, source: str, task: str, context: str) -> str:
    reply = "".join(
        ollama_client.generate_stream(
            prompt,
            system=_check_system(source),
            temperature=0.2,
        )
    )
    if not _reply_skips_text(reply, source):
        return reply
    logger.error("этап: ответ не про присланный текст, прошу процитировать его")
    return "".join(
        ollama_client.generate_stream(
            _fill(CHECK_RETRY_PROMPT, chapter=source, context=context, task=task),
            system=_check_system(source),
            temperature=0.2,
        )
    )


def _present_check(reply: str, source: str, context: list[str] | None = None) -> str:
    if not _reply_skips_text(reply, source):
        return reply
    return _unread_notice(source, reply, context)


def _unread_notice(source: str, reply: str, context: list[str] | None = None) -> str:
    sent = " ".join(source.split())
    lines = [f"Вы прислали: «{sent[:500]}»", "", "Модель этот текст не разбирала."]
    stock = " ".join(reply.split())
    if stock:
        lines.append(f"Она ответила заготовкой: «{stock[:180]}».")
    note = getattr(ollama_client, "template_note", "")
    if note:
        lines.extend(["", note])
    if context:
        lines.extend(["", "Из базы для сверки бралось:"])
        lines.extend(f"[{index}] {chunk}" for index, chunk in enumerate(context, start=1))
    return "\n".join(lines)


def _text_windows(text: str, size: int = 2000) -> list[str]:
    """Начало, середина и конец длинного отрывка, чтобы сверка не смотрела только в начало."""
    stripped = text.strip()
    if len(stripped) <= size:
        return [stripped] if stripped else []
    middle = max(0, len(stripped) // 2 - size // 2)
    windows = [stripped[:size], stripped[middle : middle + size], stripped[-size:]]
    unique: list[str] = []
    for window in windows:
        if window not in unique:
            unique.append(window)
    return unique


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _model_payload(raw: str):
    text = _strip_fences(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, (list, dict)):
            return payload
    raise json.JSONDecodeError("В ответе нет JSON", text, 0)


def _payload_list(payload, keys: tuple[str, ...]):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        lists = [value for value in payload.values() if isinstance(value, list)]
        if len(lists) == 1:
            return lists[0]
    return None


def _strings_from_payload(payload) -> list[str] | None:
    items = _payload_list(payload, ("fragments", "chunks", "items", "result"))
    if not isinstance(items, list):
        return None
    fragments: list[str] = []
    for item in items:
        if isinstance(item, str):
            piece = item.strip()
        elif isinstance(item, dict):
            piece = str(item.get("text") or item.get("content") or "").strip()
        else:
            piece = ""
        if piece:
            fragments.append(piece)
    return fragments


def _explain(title: str, reason: str, raw: str) -> str:
    logger.error(
        "этап: %s, причина=%s, символов ответа=%s\n%s",
        title,
        reason,
        len(raw or ""),
        raw,
    )
    return f"{title}\nПричина: {reason}.\nОтвет модели:\n{raw}"


def _fragments_from_whole_json(text: str) -> list[str] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _strings_from_payload(payload)


def _fragments_marked(text: str) -> list[str]:
    parts = re.split(r"(?m)^[ \t]*---[ \t]*$", text)
    return [part.strip() for part in parts if part.strip()]


def _fragments_from_list_markup(text: str) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    marker = re.compile(r"^\s*(?:\d+[\.\)]|[-*•])\s+\S")
    for line in text.splitlines():
        if marker.match(line):
            if current:
                pieces.append("\n".join(current).strip())
            current = [re.sub(r"^\s*(?:\d+[\.\)]|[-*•])\s+", "", line)]
        elif current:
            current.append(line)
    if current:
        pieces.append("\n".join(current).strip())
    return [piece for piece in pieces if piece]


def parse_fragments(raw: str) -> list[str]:
    """Берёт фрагменты, на которые модель разрезала обычный текст."""
    text = _strip_fences(raw).strip()
    whole = _fragments_from_whole_json(text)
    if whole:
        return whole
    if whole == []:
        raise SplitError(_explain("Модель не вернула фрагменты", "JSON-список пустой", raw))
    marked = _fragments_marked(text)
    if len(marked) >= 2:
        return marked
    try:
        found = _strings_from_payload(_model_payload(text))
    except json.JSONDecodeError:
        found = None
    if found:
        return found
    listed = _fragments_from_list_markup(text)
    if len(listed) >= 2:
        return listed
    raise SplitError(
        _explain(
            "Модель вернула не список фрагментов",
            f"целый JSON не список, частей по --- {len(marked)}, пунктов списка {len(listed)}",
            raw,
        )
    )


def parse_characters(raw: str) -> list[Character]:
    try:
        payload = _model_payload(raw)
    except json.JSONDecodeError as exc:
        raise CharacterParseError(
            _explain("Модель вернула не список персонажей", "в ответе нет JSON", raw)
        ) from exc
    payload = _payload_list(payload, ("characters", "persons", "items", "result"))
    if not isinstance(payload, list):
        raise CharacterParseError(
            _explain("Модель вернула не список персонажей", "JSON не список персонажей", raw)
        )

    characters: list[Character] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        description = str(item.get("description") or "").strip()
        if name:
            characters.append(Character(name=name, description=description))
    if payload and not characters:
        raise CharacterParseError(
            _explain("Модель не вернула персонажей", "в списке нет имён", raw)
        )
    return characters


def character_document(character: Character) -> str:
    if character.description:
        return f"Персонаж: {character.name}\n{character.description}"
    return f"Персонаж: {character.name}"


def character_vector_id(book_id: str, name: str) -> str:
    digest = hashlib.sha1(name.casefold().strip().encode()).hexdigest()[:12]
    return f"{book_id}-character-{digest}"


class RAGService:
    def _save_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        book_id: str,
        chapter_id: str | None = None,
    ) -> int:
        if not chunks:
            return 0

        chapter_id = chapter_id or str(uuid.uuid4())
        ids = [f"{book_id}-{chapter_id}-{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "book_id": book_id,
                "chapter_id": chapter_id,
                "chunk_index": i,
                "kind": "fragment",
            }
            for i in range(len(chunks))
        ]
        vector_store.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        return len(chunks)

    def add_chapter(
        self,
        text: str,
        book_id: str,
        chapter_id: str | None = None,
    ) -> int:
        """Нарезает главу, делает эмбеддинги и сохраняет в векторную базу книги."""
        chunks = split_text(text)
        if not chunks:
            return 0

        embeddings = embedding_service.embed_batch(chunks)
        return self._save_chunks(chunks, embeddings, book_id, chapter_id)

    def query(
        self,
        text: str,
        book_id: str,
        n_results: int | None = None,
    ) -> list[str]:
        """Ищет релевантные чанки по тексту внутри одной книги."""
        # Берём только начало текста для эмбеддинга (чтобы не превысить контекст)
        query_text = text[:2000]
        query_embedding = embedding_service.embed(query_text)
        results = vector_store.query(
            query_embedding,
            n_results=n_results,
            where={"book_id": book_id},
        )

        documents = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        kept: list[str] = []
        for index, document in enumerate(documents):
            meta = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            if meta.get("kind") == "character" or not document:
                continue
            kept.append(document)
        return kept

    def saved_fragments(self, text: str, book_id: str) -> list[str]:
        """Ищет сохранённые фрагменты по началу, середине и концу нового отрывка."""
        found: list[str] = []
        for window in _text_windows(text):
            for chunk in self.query(window, book_id, n_results=settings.top_k):
                if chunk not in found:
                    found.append(chunk)
        return found[: settings.top_k * 2]

    def iter_check_chapter(
        self,
        text: str,
        book_id: str,
        book_title: str,
    ) -> Iterator[dict]:
        """Сверяет отрывок с базой и отдельно проверяет его внутреннюю логику."""
        logger.info(
            "этап: проверка, книга=%s (%s), символов=%s",
            book_title,
            book_id,
            len(text),
        )
        yield _stage("start", "Идёт проверка…")
        yield _stage("search", "Ищу в книге…")
        context_chunks = self.saved_fragments(text, book_id)
        logger.info("этап: поиск, фрагментов=%s", len(context_chunks))
        context = (
            "\n\n---\n\n".join(
                f"[{index}] {chunk}" for index, chunk in enumerate(context_chunks, start=1)
            )
            if context_chunks
            else "В базе по этой книге пока нет сохранённых фрагментов."
        )
        yield {
            "type": "meta",
            "book_id": book_id,
            "book_title": book_title,
            "context_chunks": context_chunks,
            "original_length": len(text),
        }
        yield _stage("rag", "Сверяю с сохранённым текстом…")
        logger.info("этап: сверка с базой, модель=%s", settings.llm_model)
        yield {
            "type": "token",
            "text": f"По уже сохранённому тексту (фрагментов из базы: {len(context_chunks)})\n\n",
        }
        rag_reply = _ask_about_text(
            _fill(
                CHECK_AGAINST_RAG_PROMPT,
                book=book_title,
                context=context,
                chapter=text,
            ),
            text,
            task="Сверь присланный текст с фрагментами базы и назови противоречия.",
            context=context,
        )
        logger.info("этап: сверка, прислано=%r, ответ=\n%s", _opening(text, 80), rag_reply)
        yield {
            "type": "token",
            "text": _present_check(rag_reply, text, context_chunks),
        }

        yield _stage("logic", "Проверяю логику нового отрывка…")
        logger.info("этап: логика нового отрывка, модель=%s", settings.llm_model)
        yield {"type": "token", "text": "\n\nЛогика нового отрывка\n\n"}
        logic_reply = _ask_about_text(
            _fill(CHECK_LOGIC_PROMPT, book=book_title, chapter=text),
            text,
            task="Проверь внутреннюю логику только присланного текста.",
            context="База для этой части не нужна.",
        )
        logger.info("этап: логика, прислано=%r, ответ=\n%s", _opening(text, 80), logic_reply)
        yield {"type": "token", "text": _present_check(logic_reply, text)}
        yield {"type": "done"}

    def iter_save_chapter(
        self,
        text: str,
        book_id: str,
        book_title: str,
    ) -> Iterator[dict]:
        """Режет главу на фрагменты и отдельно сохраняет карточки персонажей."""
        logger.info(
            "этап: сохранение, книга=%s (%s), символов=%s",
            book_title,
            book_id,
            len(text),
        )
        yield _stage("start", "Идёт сохранение…")
        yield _stage("split", "Модель разбивает текст…")
        logger.info("этап: разбиение, модель=%s", settings.llm_model)
        raw = ollama_client.generate(
            _fill(
                SPLIT_CHAPTER_PROMPT,
                book=book_title,
                chapter=text,
                chunk_size=settings.chunk_size,
            ),
            system=SPLIT_SYSTEM_PROMPT,
            temperature=0.1,
        )
        split_by = "model"
        note = ""
        try:
            fragments = parse_fragments(raw)
        except SplitError as exc:
            fragments = split_text(text)
            split_by = "chapter"
            note = " ".join(raw.split())
            logger.error(
                "этап: модель не разрезала текст, режу главу сам, фрагментов=%s\n%s",
                len(fragments),
                exc,
            )
            shown = note if len(note) <= 160 else note[:159] + "…"
            yield _stage("split", f"Модель ответила: {shown}. Режу главу сам.")
        if not fragments:
            raise SplitError("Глава пустая, сохранять нечего")
        logger.info("этап: фрагментов=%s, нарезка=%s", len(fragments), split_by)
        yield _stage("characters", "Выписываю персонажей…")
        logger.info("этап: персонажи, модель=%s", settings.llm_model)
        raw_characters = ollama_client.generate(
            _fill(CHARACTER_CHAPTER_PROMPT, book=book_title, chapter=text),
            system=CHARACTER_SYSTEM_PROMPT,
            temperature=0.1,
            response_format="json",
        )
        try:
            found = parse_characters(raw_characters)
        except CharacterParseError as exc:
            found = []
            logger.error(
                "этап: персонажи не разобраны, фрагменты всё равно сохраняю\n%s",
                exc,
            )
            yield _stage(
                "characters",
                "Персонажей из ответа не разобрал. Фрагменты сохраняю.",
            )
        characters = merge_characters(character_store.list_for_book(book_id), found)
        logger.info(
            "этап: персонажей в главе=%s, всего=%s",
            len(found),
            len(characters),
        )
        yield _stage("embed", "Считаю эмбеддинги…")
        logger.info(
            "этап: эмбеддинги, чанков=%s, персонажей=%s, модель=%s",
            len(fragments),
            len(found),
            settings.embedding_model,
        )
        embeddings = embedding_service.embed_batch(fragments)
        yield _stage("store", "Сохраняю фрагменты…")
        chunks_added = self._save_chunks(fragments, embeddings, book_id)
        if found:
            stored_people = [
                character
                for character in characters
                if character.name.casefold().strip()
                in {item.name.casefold().strip() for item in found}
            ]
            people_embeddings = embedding_service.embed_batch(
                [character_document(character) for character in stored_people]
            )
            vector_store.upsert(
                ids=[
                    character_vector_id(book_id, character.name)
                    for character in stored_people
                ],
                embeddings=people_embeddings,
                documents=[character_document(character) for character in stored_people],
                metadatas=[
                    {
                        "book_id": book_id,
                        "kind": "character",
                        "name": character.name,
                    }
                    for character in stored_people
                ],
            )
        character_store.replace(book_id, characters)
        logger.info("этап: запись в базу, чанков=%s", chunks_added)
        yield {
            "type": "saved",
            "book_id": book_id,
            "book_title": book_title,
            "fragments": fragments,
            "chunks_added": chunks_added,
            "characters": [
                {"name": character.name, "description": character.description}
                for character in characters
            ],
            "original_length": len(text),
            "split_by": split_by,
            "note": note,
        }
        logger.info(
            "этап: сохранено, чанков=%s, персонажей=%s",
            chunks_added,
            len(characters),
        )
        yield {"type": "done"}

    def check_chapter(self, text: str, book_id: str, book_title: str) -> RAGResult:
        parts: list[str] = []
        context_chunks: list[str] = []
        for event in self.iter_check_chapter(text, book_id, book_title):
            if event["type"] == "token":
                parts.append(event["text"])
            elif event["type"] == "meta":
                context_chunks = list(event["context_chunks"])
        return RAGResult(
            answer="".join(parts),
            context_chunks=context_chunks,
            chunks_added=0,
        )


# Синглтон
rag_service = RAGService()
