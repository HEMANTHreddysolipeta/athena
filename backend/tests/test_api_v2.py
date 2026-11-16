"""The v1 API: documents, uploads, streaming chat, conversations, metrics, guards."""

import json

import pytest
from fastapi.testclient import TestClient

import app as app_module
from athena import database, rag
from athena.config import settings


@pytest.fixture
def client(store, monkeypatch, tmp_path):
    database.initialize_database()
    monkeypatch.setattr(app_module, "get_store", lambda: store)
    monkeypatch.setattr(rag, "get_store", lambda: store)
    app_module._recent.clear()
    return TestClient(app_module.app)


@pytest.fixture
def fake_llm(monkeypatch):
    def stream(prompt):
        if "STANDALONE QUESTION" in prompt:
            yield "What is the minimum password length?"
            return
        yield from ["Passwords need ", "15 characters [1]."]

    monkeypatch.setattr(rag.llm, "generate_stream", stream)
    monkeypatch.setattr(rag.llm, "generate", lambda p: "".join(stream(p)))


def read_events(response) -> list[tuple[str, dict]]:
    events = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def upload(client, name, content):
    return client.post("/api/documents", files={"file": (name, content, "application/octet-stream")})


def test_upload_lists_and_deletes_a_document(client):
    r = upload(client, "rules.md", b"# Password rules\nPasswords need 15 characters.")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "indexed" and r.json()["title"] == "Password rules"

    docs = client.get("/api/documents").json()
    assert [d["filename"] for d in docs] == ["rules.md"]

    r = client.delete("/api/documents/rules.md")
    assert r.json() == {"filename": "rules.md", "removed_from_index": True, "deleted_file": True}
    assert client.get("/api/documents").json() == []


def test_upload_rejects_unsupported_types_and_strips_paths(client):
    assert upload(client, "virus.exe", b"MZ").status_code == 415
    r = upload(client, "../../evil.md", b"# Evil\nhello")
    assert r.status_code == 200
    assert r.json()["filename"] == "evil.md"  # saved inside the upload folder, not two levels up
    assert (settings.upload_dir / "evil.md").is_file()


def test_streaming_chat_sends_sources_tokens_and_a_grounded_answer(client, fake_llm):
    upload(client, "rules.md", b"# Password rules\nPasswords need a minimum of 15 characters.")
    r = client.post("/api/chat/stream", json={"message": "How long must passwords be?"})
    events = read_events(r)
    kinds = [e for e, _ in events]
    assert kinds[0] == "conversation" and "sources" in kinds and kinds[-1] == "done"
    done = events[-1][1]
    assert done["found"] and done["cited"] == [1]
    assert done["grounding"]["supported"] is True
    assert "".join(d["text"] for e, d in events if e == "token") == "Passwords need 15 characters [1]."

    conversation_id = events[0][1]["id"]
    messages = client.get(f"/api/conversations/{conversation_id}").json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["meta"]["sources"][0]["filename"] == "rules.md"

    metrics = client.get("/api/metrics").json()
    assert metrics["queries"] >= 1 and metrics["grounded_answers"] >= 1


def test_follow_up_in_a_conversation_is_rewritten(client, fake_llm):
    upload(client, "rules.md", b"# Password rules\nPasswords need a minimum of 15 characters.")
    first = read_events(client.post("/api/chat/stream", json={"message": "Tell me the password rules"}))
    conversation_id = first[0][1]["id"]
    second = read_events(client.post("/api/chat/stream",
                                     json={"message": "and how long must it be?", "conversation_id": conversation_id}))
    assert ("rewrite", {"question": "What is the minimum password length?"}) in second


def test_slash_commands_do_not_reach_the_llm(client, monkeypatch):
    monkeypatch.setattr(rag.llm, "generate_stream", lambda p: pytest.fail("LLM called"))
    events = read_events(client.post("/api/chat/stream", json={"message": "/help"}))
    assert events[-1][1]["command"] == "help"
    assert "/find" in events[-1][1]["answer"]


def test_question_starting_with_find_is_a_question_not_a_command(client, fake_llm):
    """Regression: the Milestone 1 router treated this as a file search."""
    upload(client, "rules.md", b"# Password rules\nPasswords need a minimum of 15 characters.")
    events = read_events(client.post("/api/chat/stream", json={"message": "find the password length rule"}))
    assert "command" not in events[-1][1]
    assert events[-1][1]["found"]


def test_conversations_can_be_listed_and_deleted(client, fake_llm):
    events = read_events(client.post("/api/chat/stream", json={"message": "/help"}))
    conversation_id = events[0][1]["id"]
    assert conversation_id in [c["id"] for c in client.get("/api/conversations").json()]
    assert client.delete(f"/api/conversations/{conversation_id}").status_code == 200
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404


def test_unknown_conversation_is_404(client):
    r = client.post("/api/chat/stream", json={"message": "hi", "conversation_id": "nope"})
    assert r.status_code == 404


def test_chat_is_rate_limited(client, monkeypatch):
    monkeypatch.setattr(app_module, "CHAT_RATE_PER_MINUTE", 2)
    codes = [client.post("/api/chat/stream", json={"message": "/help"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_api_key_is_enforced_when_set(client, monkeypatch):
    monkeypatch.setattr(app_module, "API_KEY", "s3cret")
    assert client.get("/api/documents").status_code == 401
    assert client.get("/api/documents", headers={"x-api-key": "s3cret"}).status_code == 200


def test_health_reports_llm_and_index(client, monkeypatch):
    monkeypatch.setattr(app_module, "is_available", lambda: (False, "Ollama is not reachable"))
    body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["index"]["chunks"] == 0


def test_url_to_private_address_is_refused(client):
    r = client.post("/api/documents/url", json={"url": "http://127.0.0.1:11434/api/tags"})
    assert r.status_code == 400
    assert "private" in r.json()["detail"] or "local" in r.json()["detail"]


def test_url_with_other_scheme_is_refused(client):
    r = client.post("/api/documents/url", json={"url": "file:///C:/Windows/win.ini"})
    assert r.status_code == 400
