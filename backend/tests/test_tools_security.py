import pytest

from athena import search, tools


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "notes.pdf").write_bytes(b"%PDF-1.4")
    (data / "setup.exe").write_bytes(b"MZ")
    outside = tmp_path / "secret.pdf"
    outside.write_bytes(b"%PDF-1.4")

    # Settings is frozen (immutable), so swap the module-level references.
    fake = type("S", (), {"data_dir": data})()
    monkeypatch.setattr(tools, "settings", fake)
    monkeypatch.setattr(search, "SEARCH_FOLDERS", [data])
    return data, outside


def test_find_only_searches_the_data_folder(sandbox):
    data, outside = sandbox
    names = {f["name"] for f in search.find_files("")}
    assert "notes.pdf" in names
    assert "secret.pdf" not in names


def test_pdf_inside_data_is_allowed(sandbox):
    data, _ = sandbox
    assert tools.is_safe_document(str(data / "notes.pdf"))


@pytest.mark.parametrize("make_path", [
    lambda data, outside: str(outside),                        # outside the folder
    lambda data, outside: str(data / ".." / "secret.pdf"),     # '..' traversal
    lambda data, outside: str(data / "setup.exe"),             # executable inside
    lambda data, outside: str(data / "missing.pdf"),           # doesn't exist
])
def test_unsafe_paths_are_refused(sandbox, make_path, monkeypatch):
    data, outside = sandbox
    launched = []
    monkeypatch.setattr(tools.os, "startfile", lambda p: launched.append(p), raising=False)

    assert "Refused" in tools.open_file(make_path(data, outside))
    assert launched == []
