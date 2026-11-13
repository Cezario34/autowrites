from pathlib import Path

DIRECTORY = Path(__file__).resolve().parents[2] / "examples"
DIALOGUES_FILE = "диалоги.txt"
DESCRIPTIONS_FILE = "описания.txt"


def read_style_file(filename: str) -> str:
    path = DIRECTORY / filename
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def read_dialogues() -> str:
    return read_style_file(DIALOGUES_FILE)


def read_descriptions() -> str:
    return read_style_file(DESCRIPTIONS_FILE)
