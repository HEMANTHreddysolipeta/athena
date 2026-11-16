import docx
import pytest

from athena.document_reader import read_document_pages
from athena.pdf_reader import DocumentReadError, read_pdf_pages


def test_pdf_pages_are_numbered_from_one_and_blank_pages_skipped(make_pdf):
    path = make_pdf("a.pdf", ["first page", "", "third page"])
    pages = read_pdf_pages(path)
    assert [n for n, _ in pages] == [1, 3]
    assert "third page" in pages[1][1]


def test_broken_pdf_raises_instead_of_returning_error_text(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf")
    with pytest.raises(DocumentReadError):
        read_pdf_pages(str(bad))


def test_docx_is_one_unnumbered_page(tmp_path):
    path = tmp_path / "note.docx"
    d = docx.Document()
    d.add_paragraph("Hello")
    d.add_paragraph("")
    d.add_paragraph("World")
    d.save(path)
    assert read_document_pages(str(path)) == [(None, "Hello\nWorld")]


def test_unsupported_extension(tmp_path):
    path = tmp_path / "x.xyz"
    path.write_text("hi")
    with pytest.raises(ValueError):
        read_document_pages(str(path))
