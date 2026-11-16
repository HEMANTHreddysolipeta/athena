"""Section-aware reading for every supported format."""

import docx
import pymupdf
import pytest

from athena.chunker import chunk_sections, contextualize
from athena.document_reader import Section, read_any
from athena.pdf_reader import clean_pdf_text


def test_pdf_headings_are_detected_and_carried_across_pages(tmp_path):
    path = tmp_path / "guide.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 60), "Password Rules", fontsize=18)
    page.insert_text((50, 100), "Passwords must be at least fifteen characters long.", fontsize=10)
    page2 = doc.new_page()
    page2.insert_text((50, 60), "More body text that continues the same section.", fontsize=10)
    doc.save(path)

    result = read_any(str(path))
    assert result.sections[0].heading == "Password Rules"
    assert result.sections[0].page == 1
    # Page 2 has no heading of its own: it inherits the one above.
    assert result.sections[-1].page == 2
    assert result.sections[-1].heading == "Password Rules"


def test_pdf_words_positioned_without_space_characters_get_spaces(tmp_path):
    """Regression: NIST AI 100-1 positions each word with no space chars,
    which used to come out as 'notallincidents'."""
    path = tmp_path / "spaced.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    x = 50
    for word in ["not", "all", "incidents"]:
        page.insert_text((x, 100), word, fontsize=10)
        x += pymupdf.get_text_length(word, fontsize=10) + 4  # a gap, no space char
    doc.save(path)
    assert "not all incidents" in read_any(str(path)).sections[0].text


def test_ligatures_and_unicode_hyphens_are_normalised():
    assert clean_pdf_text("conﬁdently non‑breaking") == "confidently non-breaking"


def test_docx_headings_and_tables(tmp_path):
    path = tmp_path / "spec.docx"
    d = docx.Document()
    d.add_heading("Spec", 0)
    d.add_heading("Pricing", 1)
    d.add_paragraph("Plans start at 499 rupees.")
    table = d.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Tier"
    table.rows[0].cells[1].text = "Pro"
    d.save(path)

    result = read_any(str(path))
    assert result.title == "Spec"
    pricing = next(s for s in result.sections if s.heading == "Pricing")
    assert "499 rupees" in pricing.text
    assert any(s.heading == "Table" and "Tier | Pro" in s.text for s in result.sections)


def test_markdown_sections_follow_headings(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Guide\nintro\n\n## Install\nrun pip install\n\n## Usage\ncall ask()\n", encoding="utf-8")
    result = read_any(str(path))
    assert result.title == "Guide"
    assert [s.heading for s in result.sections] == ["Guide", "Install", "Usage"]
    assert "pip install" in result.sections[1].text


def test_txt_uses_first_line_as_title(tmp_path):
    path = tmp_path / "v.txt"
    path.write_text("Vector database\n\nA vector database stores embeddings.", encoding="utf-8")
    result = read_any(str(path))
    assert result.title == "Vector database"
    assert result.sections[0].page is None


def test_pptx_slides_are_pages_with_titles(tmp_path):
    from pptx import Presentation

    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[1])
    slide.shapes.title.text = "Quarterly results"
    slide.placeholders[1].text = "Revenue grew 12 percent"
    slide.notes_slide.notes_text_frame.text = "Mention the new client"
    path = tmp_path / "deck.pptx"
    deck.save(path)

    section = read_any(str(path)).sections[0]
    assert (section.page, section.heading) == (1, "Quarterly results")
    assert "12 percent" in section.text and "Mention the new client" in section.text


def test_html_drops_navigation_and_scripts(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>RAG</title><script>evil()</script></head><body>"
        "<nav>Home | About</nav><article><h1>RAG</h1><p>Retrieval first.</p>"
        "<h2>History</h2><p>Introduced in 2020.</p></article></body></html>",
        encoding="utf-8",
    )
    result = read_any(str(path))
    text = " ".join(s.text for s in result.sections)
    assert "evil" not in text and "About" not in text
    assert next(s for s in result.sections if s.heading == "History").text.endswith("Introduced in 2020.")


def test_scanned_page_is_ocrd(tmp_path, monkeypatch):
    """A page without a text layer goes through OCR (stubbed here: the real
    engine is exercised by the eval on demo_corpus/Scanned_*.pdf)."""
    import athena.ocr

    monkeypatch.setattr(athena.ocr, "ocr_image_bytes", lambda png: "Text read by OCR")
    path = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(path)

    result = read_any(str(path))
    assert result.ocr_pages == [1]
    assert result.sections[0].text == "Text read by OCR"


def test_chunks_keep_page_and_heading():
    sections = [Section(3, "Govern", "word " * 300), Section(4, "Map", "short text")]
    chunks = chunk_sections(sections, chunk_size=400, overlap=50)
    assert len(chunks) > 2
    assert all(c[:2] == (3, "Govern") for c in chunks[:-1])
    assert chunks[-1] == (4, "Map", "short text")


def test_contextualize_prefixes_title_and_heading():
    assert contextualize("AI RMF", "Govern", "text") == "AI RMF > Govern\ntext"
    assert contextualize("AI RMF", None, "text") == "AI RMF\ntext"
    assert contextualize("AI RMF", "AI RMF", "text") == "AI RMF\ntext"


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "x.exe"
    path.write_bytes(b"MZ")
    with pytest.raises(ValueError):
        read_any(str(path))
