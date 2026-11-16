"""Turn any supported document into sections: (page, heading, text).

Every format ends up in the same shape, so chunking, indexing and citing
work the same way for all of them:

    Document(title, sections=[Section(page=3, heading="2.1 Govern", text=...), ...])

`page` is a 1-based page (PDF) or slide (PPTX) number, or None for formats
without pages. `heading` is the nearest heading above the text, carried
across pages, so a chunk from the middle of "Appendix B" still knows it
belongs to Appendix B. Headings make retrieval much better for questions
like "what does day 7 of the framework say?", because the heading words
are indexed with every chunk under it.
"""

import logging
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from .pdf_reader import DocumentReadError, clean_pdf_text, read_pdf, read_pdf_pages

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt", ".pptx", ".html", ".htm"}

# A page is (page_number, text). Kept for callers that only need text.
Page = tuple[int | None, str]

# A PDF page with fewer characters than this is treated as scanned and OCR'd.
OCR_MIN_CHARS = 25


@dataclass
class Section:
    page: int | None
    heading: str | None
    text: str


@dataclass
class Document:
    title: str
    sections: list[Section] = field(default_factory=list)
    ocr_pages: list[int] = field(default_factory=list)

    def pages(self) -> list[Page]:
        return [(s.page, s.text) for s in self.sections]


def _title_from_filename(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[_+]+", " ", stem).strip()


def _short(heading: str, limit: int = 120) -> str:
    heading = re.sub(r"\s+", " ", heading).strip()
    return heading if len(heading) <= limit else heading[: limit - 1] + "…"


# ---------------------------------------------------------------- PDF ----

def _looks_like_heading(text: str) -> bool:
    words = text.split()
    return (
        1 <= len(words) <= 14
        and len(text) <= 100
        and not text.endswith((".", ",", ";", ":"))
        and any(c.isalpha() for c in text)
        and not re.fullmatch(r"[\d\s.\-–—|]+", text)
    )


def _line_text(spans: list[dict]) -> str:
    """Rebuild a line from its characters, adding spaces at visible gaps.

    Some PDFs (e.g. NIST AI 100-1) position every word separately and
    contain no space characters at all, so joining the text naively gives
    "notallincidentsandfailures". A gap wider than ~15% of the font size
    between two characters is a word break.
    """
    out: list[str] = []
    prev_x1 = None
    for span in spans:
        gap_threshold = 0.15 * span["size"]
        for char in span["chars"]:
            c = char["c"]
            x0, x1 = char["bbox"][0], char["bbox"][2]
            if prev_x1 is not None and x0 - prev_x1 > gap_threshold and out and out[-1] != " " and c != " ":
                out.append(" ")
            out.append(c)
            prev_x1 = x1
    return re.sub(r"[ \t]+", " ", "".join(out)).strip()


def _pdf_lines(page) -> list[tuple[str, float, bool]]:
    """(text, font size, bold) for every text line on the page."""
    lines = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if any(c["c"].strip() for c in s["chars"])]
            if not spans:
                continue
            text = _line_text(line["spans"])
            if not text:
                continue
            size = max(s["size"] for s in spans)
            bold = all("bold" in s["font"].lower() or s["flags"] & 16 for s in spans)
            lines.append((text, round(size, 1), bold))
    return lines


def _ocr_page(page) -> str:
    from .ocr import ocr_image_bytes  # lazy: loads an ONNX model

    pixmap = page.get_pixmap(dpi=200)
    return ocr_image_bytes(pixmap.tobytes("png"))


def read_pdf_document(path: str, ocr: bool = True) -> Document:
    import pymupdf

    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise DocumentReadError(f"Could not read PDF {path}: {exc}") from exc

    with doc:
        page_lines = [_pdf_lines(page) for page in doc]

        # Body text size = the most common size, weighted by characters.
        sizes = Counter()
        for lines in page_lines:
            for text, size, _ in lines:
                sizes[size] += len(text)
        body = sizes.most_common(1)[0][0] if sizes else 10.0

        title = (doc.metadata or {}).get("title", "").strip()
        result = Document(title="")
        heading: str | None = None

        def add_section(number: int, heading: str | None, buffer: list[str]) -> None:
            text = clean_pdf_text("\n".join(buffer))
            if text:
                result.sections.append(Section(number, heading, text))

        for index, (page, lines) in enumerate(zip(doc, page_lines, strict=True)):
            number = index + 1
            if sum(len(t) for t, _, _ in lines) < OCR_MIN_CHARS:
                if not ocr:
                    continue
                text = clean_pdf_text(_ocr_page(page))
                if text:
                    result.ocr_pages.append(number)
                    result.sections.append(Section(number, heading, text))
                continue

            buffer: list[str] = []
            for text, size, bold in lines:
                is_heading = _looks_like_heading(text) and (
                    size >= body * 1.18 or (bold and size >= body * 0.98 and len(text) <= 70)
                )
                if is_heading:
                    add_section(number, heading, buffer)  # text under the previous heading
                    buffer = []
                    heading = _short(text)
                    if not title and size >= body * 1.5:
                        title = _short(text)
                buffer.append(text)  # headings stay in the text too
            add_section(number, heading, buffer)

    result.title = title or _title_from_filename(path)
    return result


# --------------------------------------------------------------- DOCX ----

def read_docx_document(path: str) -> Document:
    from docx import Document as DocxDocument

    try:
        docx = DocxDocument(path)
    except Exception as exc:
        raise DocumentReadError(f"Could not read DOCX {path}: {exc}") from exc

    result = Document(title=(docx.core_properties.title or "").strip())
    heading, buffer = None, []

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            result.sections.append(Section(None, heading, text))
        buffer.clear()

    for paragraph in docx.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name if paragraph.style is not None else "").lower()
        if style.startswith("heading") or style == "title":
            flush()
            heading = _short(text)
            if style == "title" and not result.title:
                result.title = text
        buffer.append(text)
    flush()

    # Tables carry real content in many business documents.
    for table in docx.tables:
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        text = "\n".join(r for r in rows if r.strip(" |"))
        if text:
            result.sections.append(Section(None, "Table", text))

    result.title = result.title or _title_from_filename(path)
    return result


# ------------------------------------------------------ Markdown / TXT ----

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def read_markdown_document(path: str) -> Document:
    text = _read_text_file(path)
    result = Document(title="")
    heading, buffer = None, []

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            result.sections.append(Section(None, heading, body))
        buffer.clear()

    for line in text.splitlines():
        match = _MD_HEADING.match(line)
        if match:
            flush()
            heading = _short(re.sub(r"[*_`]", "", match.group(2)))
            if not result.title:
                result.title = heading
        buffer.append(line)
    flush()
    result.title = result.title or _title_from_filename(path)
    return result


def read_text_document(path: str) -> Document:
    text = _read_text_file(path).strip()
    first_line = text.splitlines()[0].strip() if text else ""
    title = first_line if 0 < len(first_line) <= 100 else _title_from_filename(path)
    return Document(title=title, sections=[Section(None, None, text)] if text else [])


def _read_text_file(path: str) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with open(path, encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise DocumentReadError(f"Could not decode text file {path}")


# --------------------------------------------------------------- PPTX ----

def read_pptx_document(path: str) -> Document:
    from pptx import Presentation

    try:
        deck = Presentation(path)
    except Exception as exc:
        raise DocumentReadError(f"Could not read PPTX {path}: {exc}") from exc

    result = Document(title=(deck.core_properties.title or "").strip())
    for number, slide in enumerate(deck.slides, start=1):
        title_shape = slide.shapes.title
        slide_title = title_shape.text.strip() if title_shape is not None and title_shape.has_text_frame else ""
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                texts.extend(p.text.strip() for p in shape.text_frame.paragraphs if p.text.strip())
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    texts.append(" | ".join(c.text.strip() for c in row.cells))
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                texts.append(f"Speaker notes: {notes}")
        text = "\n".join(texts).strip()
        if text:
            result.sections.append(Section(number, _short(slide_title) if slide_title else None, text))
        if not result.title and slide_title:
            result.title = slide_title
    result.title = result.title or _title_from_filename(path)
    return result


# --------------------------------------------------------------- HTML ----

def read_html_document(path: str) -> Document:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_read_text_file(path), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    root = soup.find("article") or soup.find("main") or soup.body or soup
    result = Document(title="")
    heading, buffer = None, []

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            result.sections.append(Section(None, heading, text))
        buffer.clear()

    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "td", "th", "blockquote"]):
        text = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
        if not text:
            continue
        if element.name in ("h1", "h2", "h3", "h4"):
            flush()
            heading = _short(text)
            title = title or text
        buffer.append(text)
    flush()
    result.title = title or _title_from_filename(path)
    return result


# ------------------------------------------------------------ dispatch ----

_READERS = {
    ".pdf": read_pdf_document,
    ".docx": read_docx_document,
    ".md": read_markdown_document,
    ".markdown": read_markdown_document,
    ".txt": read_text_document,
    ".pptx": read_pptx_document,
    ".html": read_html_document,
    ".htm": read_html_document,
}


def read_any(path: str) -> Document:
    extension = os.path.splitext(path)[1].lower()
    reader = _READERS.get(extension)
    if reader is None:
        raise ValueError(f"Unsupported file type: {extension}")
    document = reader(path)
    # Same text normalisation for every format (ligatures, full-width forms).
    for section in document.sections:
        section.text = unicodedata.normalize("NFKC", section.text)
    document.title = unicodedata.normalize("NFKC", document.title)
    return document


def read_document_pages(path: str) -> list[Page]:
    """Text per page. Kept for older callers; new code uses read_any()."""
    extension = os.path.splitext(path)[1].lower()
    if extension == ".pdf":
        return read_pdf_pages(path)
    if extension == ".docx":
        doc = read_docx_document(path)
        text = "\n".join(s.text for s in doc.sections if s.heading != "Table")
        return [(None, text)] if text else []
    return read_any(path).pages()


def read_document(path: str) -> str:
    """Whole document as one string (the 'read <file>' command)."""
    if os.path.splitext(path)[1].lower() == ".pdf":
        return read_pdf(path)
    return "\n\n".join(text for _, text in read_any(path).pages())
