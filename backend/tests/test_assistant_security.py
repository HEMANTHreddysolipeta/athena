"""Who can reach Athena, and which paths the assistant may read or change.

Everything runs in throwaway folders: ATHENA_ASSISTANT_ROOT points the
assistant's "home folder" at tmp_path, never at the real one.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

import app as app_module
from athena.assistant import policy, security

TOKEN = os.environ["ATHENA_TOKEN"]


@pytest.fixture
def anon():
    return TestClient(app_module.app)


# ------------------------------------------------------------ server access

@pytest.mark.parametrize("method,path", [
    ("get", "/api/health"), ("get", "/api/documents"), ("get", "/api/metrics"),
    ("get", "/api/conversations"), ("get", "/api/assistant/storage"),
    ("get", "/api/assistant/activity"), ("post", "/api/assistant/storage/scan"),
])
def test_every_endpoint_needs_the_token(anon, method, path):
    assert getattr(anon, method)(path).status_code == 401
    assert getattr(anon, method)(path, headers={"x-athena-token": "wrong"}).status_code == 401


def test_chat_needs_the_token(anon):
    assert anon.post("/api/chat/stream", json={"message": "/storage"}).status_code == 401
    assert anon.post("/chat", json={"message": "/help"}).status_code == 401


def test_sign_in_sets_a_strict_http_only_cookie(anon):
    assert anon.post("/api/session", json={"token": "nope"}).status_code == 401
    assert anon.get("/api/session").json() == {"authenticated": False}

    response = anon.post("/api/session", json={"token": TOKEN})
    assert response.status_code == 200
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie

    assert anon.get("/api/session").json() == {"authenticated": True}
    assert anon.get("/api/assistant/activity").status_code == 200  # cookie alone is enough
    anon.delete("/api/session")
    anon.cookies.clear()
    assert anon.get("/api/assistant/activity").status_code == 401


def test_other_host_names_are_refused_even_with_the_token():
    """DNS rebinding: evil.example resolves to 127.0.0.1, the browser sends Host: evil.example."""
    client = TestClient(app_module.app, base_url="http://evil.example")
    response = client.get("/api/documents", headers={"x-athena-token": TOKEN})
    assert response.status_code == 403


def test_posts_from_other_websites_are_refused():
    client = TestClient(app_module.app, headers={"x-athena-token": TOKEN})
    bad = client.post("/api/assistant/storage/scan/cancel", headers={"origin": "https://evil.example"})
    assert bad.status_code == 403
    ok = client.post("/api/assistant/storage/scan/cancel", headers={"origin": "http://127.0.0.1:8000"})
    assert ok.status_code == 200


def test_token_is_random_persistent_and_private(tmp_path, monkeypatch):
    monkeypatch.delenv("ATHENA_TOKEN")
    monkeypatch.setenv("ATHENA_HOME", str(tmp_path / "athena"))
    monkeypatch.setattr(security, "_token", None)
    first = security.get_token()
    assert len(first) >= 40
    monkeypatch.setattr(security, "_token", None)
    assert security.get_token() == first  # read back from the file
    assert (tmp_path / "athena" / "token").read_text().strip() == first
    if sys.platform != "win32":
        assert oct((tmp_path / "athena" / "token").stat().st_mode & 0o777) == "0o600"


@pytest.mark.parametrize("value,host", [
    ("127.0.0.1:8000", "127.0.0.1"), ("localhost", "localhost"),
    ("[::1]:8000", "::1"), ("Evil.Example:80", "evil.example"), (None, ""),
])
def test_host_parsing(value, host):
    assert security.host_of(value) == host


# ------------------------------------------------------------------- paths

@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / "home"
    for folder in ("Documents/notes", "Downloads", "AppData/Local", ".ssh", "projects"):
        (root / folder).mkdir(parents=True)
    (root / "Documents" / "notes" / "a.txt").write_text("hi")
    (root / "AppData" / "Local" / "settings.ini").write_text("x")
    (root / ".ssh" / "id_ed25519").write_text("key")
    (root / ".gitconfig").write_text("[user]")
    (root / "NTUSER.DAT").write_text("x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not yours")
    monkeypatch.setenv("ATHENA_ASSISTANT_ROOT", str(root))
    return root


def test_reading_is_limited_to_the_home_folder(home):
    assert policy.check_read(home / "Documents" / "notes" / "a.txt").name == "a.txt"
    assert policy.check_read(home / "AppData" / "Local" / "settings.ini")  # sizes may be read
    for bad in (home.parent / "outside" / "secret.txt",
                home / ".." / "outside" / "secret.txt",
                home / "Documents" / ".." / ".." / "outside"):
        with pytest.raises(policy.PolicyError):
            policy.check_read(bad)


@pytest.mark.parametrize("bad", ["", "Documents/notes", "\\\\server\\share\\x", "//server/share",
                                 "C:relative.txt"])
def test_odd_paths_are_refused(home, bad):
    with pytest.raises(policy.PolicyError):
        policy.check_read(bad)


def test_missing_paths_are_refused(home):
    with pytest.raises(policy.PolicyError):
        policy.check_read(home / "nope.txt")


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS alternate data streams")
def test_alternate_data_streams_are_refused(home):
    with pytest.raises(policy.PolicyError):
        policy.check_read(str(home / "Documents" / "notes" / "a.txt") + ":hidden")


def test_links_cannot_lead_outside(home):
    link = home / "Documents" / "shortcut"
    target = home.parent / "outside"
    try:
        if sys.platform == "win32":
            import _winapi
            _winapi.CreateJunction(str(target), str(link))  # no admin rights needed
        else:
            os.symlink(target, link)
    except OSError:
        pytest.skip("can't create links here")
    with pytest.raises(policy.PolicyError):
        policy.check_read(link / "secret.txt")


@pytest.mark.parametrize("relative", [
    "AppData/Local/settings.ini", ".ssh/id_ed25519", ".gitconfig", "NTUSER.DAT",
    "Documents", "Downloads", "AppData", ".",
])
def test_protected_paths_cannot_be_changed(home, relative):
    with pytest.raises(policy.PolicyError):
        policy.check_change(home / relative)


def test_ordinary_files_can_be_changed(home):
    assert policy.check_change(home / "Documents" / "notes" / "a.txt")
    assert policy.check_change(home / "Documents" / "notes")
    assert policy.check_change(home / "projects")


def test_athenas_own_folder_cannot_be_changed(home, monkeypatch):
    own = home / "projects" / "athena-data"
    monkeypatch.setenv("ATHENA_HOME", str(own))
    (own / "activity.log").parent.mkdir(parents=True, exist_ok=True)
    (own / "activity.log").write_text("")
    with pytest.raises(policy.PolicyError):
        policy.check_change(own / "activity.log")


def test_display_uses_tilde(home):
    assert policy.display(home / "Documents") == os.path.join("~", "Documents")
    assert policy.display(home) == "~"
