import re
import unicodedata

import pymupdf


class DocumentReadError(Exception):
    """A document could not be opened or parsed."""


def clean_pdf_text(text: str) -> str:
    # NFKC turns ligatures ("ﬁ" -> "fi") and other presentation forms into
    # plain letters; without it "conﬁdently" never matches a search for
    # "confidently". Non-breaking hyphens/spaces become normal ones too.
    text = unicodedata.normalize("NFKC", text)
    text = re.sub("[‐‑‒–]", "-", text)  # hyphen/dash variants
    # Re-join words hyphenated across a line break: "infor-\nmation" -> "information"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # Drop trailing spaces before line breaks
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf_pages(path: str) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for pages that contain text.

    Page numbers are 1-based, as a human would cite them. Raises
    DocumentReadError instead of returning an error string, so a broken
    file can never be indexed as if it were content.
    """
    try:
        with pymupdf.open(path) as doc:
            pages = []
            for index, page in enumerate(doc):
                text = clean_pdf_text(page.get_text("text"))
                if text:
                    pages.append((index + 1, text))
            return pages
    except Exception as exc:  # pymupdf raises several unrelated types
        raise DocumentReadError(f"Could not read PDF {path}: {exc}") from exc


def read_pdf(path: str) -> str:
    """Whole document as one string (used by the 'read <file>' command)."""
    return "\n\n".join(
        f"========== PAGE {number} ==========\n\n{text}"
        for number, text in read_pdf_pages(path)
    )
