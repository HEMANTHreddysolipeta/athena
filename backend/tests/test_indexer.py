import shutil

import pytest

from athena.folder_indexer import index_folder
from athena.indexer import EmptyDocumentError, IndexStatus, index_document


def test_index_stores_chunks_with_page_metadata(store, make_pdf):
    path = make_pdf("guide.pdf", ["alpha beta gamma", "delta epsilon"])
    result = index_document(path, store)

    assert result.status == IndexStatus.INDEXED
    assert store.count() == result.chunks == 2
    hit = store.search("delta epsilon", k=1)[0]
    assert hit.metadata["page"] == 2
    assert hit.metadata["filename"] == "guide.pdf"


def test_reindexing_same_file_does_not_duplicate(store, make_pdf):
    path = make_pdf("guide.pdf", ["alpha beta gamma"])
    index_document(path, store)
    second = index_document(path, store)

    assert second.status == IndexStatus.UNCHANGED
    assert store.count() == 1


def test_identical_copy_is_skipped_as_duplicate(store, make_pdf, tmp_path):
    original = make_pdf("resume.pdf", ["skills python sql"])
    copy = tmp_path / "resume (1).pdf"
    shutil.copy(original, copy)

    index_document(original, store)
    result = index_document(str(copy), store)

    assert result.status == IndexStatus.DUPLICATE
    assert store.count() == 1


def test_changed_file_replaces_old_chunks(store, make_pdf):
    path = make_pdf("notes.pdf", ["old content one", "old content two"])
    index_document(path, store)
    make_pdf("notes.pdf", ["new content"])  # overwrite same path

    result = index_document(path, store)

    assert result.status == IndexStatus.INDEXED
    assert store.count() == 1
    assert "new content" in store.search("content", k=1)[0].text


def test_scanned_pdf_without_text_is_an_error(store, make_pdf):
    path = make_pdf("scan.pdf", ["", ""])
    with pytest.raises(EmptyDocumentError):
        index_document(path, store)
    assert store.count() == 0


def test_folder_run_survives_a_bad_file(store, make_pdf, tmp_path):
    make_pdf("good.pdf", ["useful text"])
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")

    report = index_folder(tmp_path, store)

    assert report.counts["indexed"] == 1
    assert report.counts["failed"] == 1
    assert store.count() == 1


def test_folder_readme_and_license_files_are_not_indexed(tmp_path):
    from athena.folder_indexer import find_documents

    (tmp_path / "README.md").write_text("# data folder")
    (tmp_path / "SOURCES.md").write_text("# licenses")
    (tmp_path / "notes.md").write_text("# real notes")
    assert [p.name for p in find_documents(tmp_path)] == ["notes.md"]
