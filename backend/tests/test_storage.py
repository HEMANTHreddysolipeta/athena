"""Storage report on a small fake home folder (never the real one)."""

import hashlib
import os
import time

import pytest
from fastapi.testclient import TestClient

import app as app_module
from athena import engine
from athena.assistant import storage

MB = 1024 * 1024


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / "home"

    def write(relative: str, size: int, seed: bytes = b"x", days_old: int = 0):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((seed * (size // len(seed) + 1))[:size])
        if days_old:
            old = time.time() - days_old * 86400
            os.utime(path, (old, old))
        return path

    write("Documents/report.pdf", 2 * MB, b"report")
    write("Pictures/report copy.pdf", 2 * MB, b"report")        # duplicate of the above
    write("Pictures/other.pdf", 2 * MB, b"different")           # same size, other content
    write("Downloads/setup-old.exe", 3 * MB, b"old", days_old=60)
    write("Downloads/setup-new.exe", 1 * MB, b"new")
    write("Downloads/notes-old.txt", 1000, b"n", days_old=90)   # old, but not an installer
    write("AppData/Local/Temp/junk.tmp", 5 * MB, b"t")
    write("AppData/Local/Temp/junk-copy.tmp", 5 * MB, b"t")     # same content, but not personal
    write("projects/app/node_modules/big.js", 2 * MB, b"report")  # dependency copy, ignored
    write(".cache/huggingface/model.bin", 4 * MB, b"m")
    monkeypatch.setenv("ATHENA_ASSISTANT_ROOT", str(root))
    monkeypatch.setenv("LOCALAPPDATA", str(root / "AppData" / "Local"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(storage, "FOLDER_MIN_BYTES", 0)
    return root


def run_scan(svc: storage.StorageService) -> dict:
    assert svc.start()
    svc._thread.join(timeout=60)
    assert svc.state == "done", svc.error
    return svc.summary()


def snapshot(root):
    return {p: (p.stat().st_mtime_ns, hashlib.md5(p.read_bytes()).hexdigest())
            for p in root.rglob("*") if p.is_file()}


def test_scan_measures_and_finds_cleanup_candidates(home):
    svc = storage.StorageService()
    report = run_scan(svc)

    total = sum(p.stat().st_size for p in home.rglob("*") if p.is_file())
    assert report["bytes"] == total
    assert report["files"] == 10
    sizes = {os.path.basename(f["path"]): f["bytes"] for f in report["home"]["folders"]}
    assert sizes["AppData"] == 10 * MB and sizes["Downloads"] == 4 * MB + 1000

    assert report["largest_files"][0]["bytes"] == 5 * MB

    dups = report["duplicates"]
    assert dups["complete"] and dups["count"] == 1
    names = sorted(os.path.basename(p) for p in dups["groups"][0]["paths"])
    assert names == ["report copy.pdf", "report.pdf"]   # not node_modules, not AppData
    assert dups["savings"] == 2 * MB

    installers = [os.path.basename(f["path"]) for f in report["old_installers"]["files"]]
    assert installers == ["setup-old.exe"]

    hogs = {h["id"]: h["bytes"] for h in report["space_hogs"]}
    assert hogs["temp"] == 10 * MB and hogs["huggingface"] == 4 * MB and hogs["downloads"] > 0


def test_scan_changes_nothing(home):
    before = snapshot(home)
    run_scan(storage.StorageService())
    assert snapshot(home) == before


def test_report_survives_a_restart(home):
    run_scan(storage.StorageService())
    again = storage.StorageService()
    assert again.summary()["files"] == 10


def test_folder_drill_down(home):
    svc = storage.StorageService()
    run_scan(svc)
    folder = svc.folder(str(home / "Downloads"))
    assert [os.path.basename(f["path"]) for f in folder["files"]][:2] == ["setup-old.exe", "setup-new.exe"]
    assert [c["name"] for c in folder["crumbs"]] == ["~", "Downloads"]
    with pytest.raises(storage.policy.PolicyError):
        svc.folder(str(home.parent))


def test_duplicate_budget_stops_early(home):
    svc = storage.StorageService()
    run_scan(svc)
    scanner = storage._Scanner(home, svc._cancel)
    scanner.run()
    found, complete = storage.find_duplicates(scanner.same_size, svc._cancel, {}, budget=MB)
    assert not complete and found == []


def test_scan_can_be_cancelled(home):
    svc = storage.StorageService()
    svc._cancel.set()
    scanner = storage._Scanner(home, svc._cancel)
    with pytest.raises(storage.ScanCancelled):
        scanner.run()


def test_api_flow(home, monkeypatch):
    monkeypatch.setattr(storage, "_service", storage.StorageService())
    opened = []
    monkeypatch.setattr("athena.assistant.api.subprocess.Popen", lambda args: opened.append(args))
    client = TestClient(app_module.app, headers={"x-athena-token": os.environ["ATHENA_TOKEN"]})

    overview = client.get("/api/assistant/storage").json()
    assert overview["drives"] and overview["drives"][0]["total"] > 0
    assert client.get("/api/assistant/storage/folder", params={"path": str(home)}).status_code == 409

    assert client.post("/api/assistant/storage/scan").json()["started"]
    storage._service._thread.join(timeout=60)
    assert client.get("/api/assistant/storage").json()["report"]["files"] == 10
    assert client.get("/api/assistant/storage/folder", params={"path": str(home.parent)}).status_code == 403

    assert client.post("/api/assistant/reveal", json={"path": str(home / "Documents" / "report.pdf")}).json()["ok"]
    assert opened and "report.pdf" in str(opened[0])
    assert client.post("/api/assistant/reveal", json={"path": str(home.parent)}).status_code == 403
    assert len(opened) == 1

    actions = [e["action"] for e in client.get("/api/assistant/activity").json()["entries"]]
    assert "reveal" in actions and "storage.scan.done" in actions


def test_storage_chat_command(home, monkeypatch):
    monkeypatch.setattr(storage, "_service", storage.StorageService())
    text = engine.process("/storage")
    assert "free of" in text and "No scan" in text


@pytest.mark.parametrize("message", [
    "hows my storage situation", "not in the document in my laptop whats the storage",
    "hows my laptop", "how much space is left on my laptop", "is my C drive full?",
    "check my computer",
])
def test_plain_storage_questions_go_to_the_storage_report(message):
    assert engine.parse_command(message) == ("storage", "")


@pytest.mark.parametrize("message", [
    "What is the minimum password length?", "What does AAL2 require?",
    "summarise the solar cells document", "find the password rules in 800-63B",
])
def test_document_questions_still_go_to_the_documents(message):
    assert engine.parse_command(message) is None


def test_folder_questions_in_chat(home, monkeypatch):
    svc = storage.StorageService()
    monkeypatch.setattr(storage, "_service", svc)
    assert engine.parse_command("whars un app data") is None  # no scan yet: no guessing
    run_scan(svc)
    for message in ("whars un app data", "what's in AppData", "show me downloads"):
        name, path = engine.parse_command(message)
        assert name == "folder", message
    assert os.path.basename(engine.parse_command("what's in AppData")[1]) == "AppData"
    text = engine.process("what's in AppData")
    assert "10.0 MB" in text and "Temporary files" in text and "won't delete" in text
    assert "Downloads" in engine.process("/folder downloads")
    assert engine.parse_command("What is the minimum password length?") is None


def test_follow_up_folder_questions_stay_in_context(home, monkeypatch):
    """The real conversation: "what's in app data", then "whats in local"
    must mean AppData\\Local, not the ~\\.local folder."""
    (home / ".local" / "bin").mkdir(parents=True)
    (home / ".local" / "bin" / "tool.exe").write_bytes(b"x" * 1000)
    svc = storage.StorageService()
    monkeypatch.setattr(storage, "_service", svc)
    run_scan(svc)
    _, appdata = engine.parse_command("whars un app data")
    name, local = engine.parse_command("whats in local", near_folder=appdata)
    assert name == "folder" and local == str(home / "AppData" / "Local")
    # No question word needed when the folder is inside the one being looked at.
    assert engine.parse_command("local has 95 gb wht is it", near_folder=appdata) == ("folder", local)
    assert engine.parse_command("temp", near_folder=local) == ("folder", str(home / "AppData" / "Local" / "Temp"))
    # Without context, a dot-folder only matches when the dot is typed.
    assert engine.parse_command("whats in .local")[1] == str(home / ".local")
    text = engine.process("what's in AppData")
    assert "Local" in text and "data apps keep on this computer only" in text


def test_chat_remembers_the_folder(home, monkeypatch, store):
    from athena import database
    database.initialize_database()
    monkeypatch.setattr(app_module, "get_store", lambda: store)
    svc = storage.StorageService()
    monkeypatch.setattr(storage, "_service", svc)
    run_scan(svc)
    client = TestClient(app_module.app, headers={"x-athena-token": os.environ["ATHENA_TOKEN"]})

    def say(message, cid=None):
        events = client.post("/api/chat/stream", json={"message": message, "conversation_id": cid}).text
        cid = cid or __import__("json").loads(events.split("data: ", 1)[1].split("\n", 1)[0])["id"]
        return cid, events

    cid, first = say("what's in app data")
    assert "AppData" in first
    _, second = say("whats in local", cid)
    assert "Temp" in second and ".local" not in second
