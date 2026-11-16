import os
from pathlib import Path

from .config import settings
from .document_reader import SUPPORTED_EXTENSIONS


def is_safe_document(path: str) -> bool:
    """True only for a PDF/DOCX that really lives inside the data folder.

    resolve() follows '..' and symlinks first, so tricks like
    data/../../Windows/notepad.exe are caught.
    """
    resolved = Path(path).resolve()
    data_dir = settings.data_dir.resolve()
    return (
        resolved.is_file()
        and resolved.is_relative_to(data_dir)
        and resolved.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def open_file(path: str):
    if not is_safe_document(path):
        return "Refused: Athena only opens PDF/DOCX files inside its data folder."
    try:
        os.startfile(path)  # Windows only
        return True
    except Exception as e:
        return str(e)
