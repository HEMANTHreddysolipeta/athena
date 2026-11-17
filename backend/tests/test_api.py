from fastapi.testclient import TestClient

import app as app_module
from athena.llm import LLMError

client = TestClient(app_module.app, headers={"x-athena-token": "test-token-0123456789abcdefghijklmnop"})  # no `with`: lifespan (DB init) is not run


def test_home():
    assert client.get("/").status_code == 200


def test_chat_returns_engine_response(monkeypatch):
    monkeypatch.setattr(app_module, "process", lambda m: f"echo: {m}")
    r = client.post("/chat", json={"message": "hi"})
    assert r.status_code == 200
    assert r.json() == {"response": "echo: hi"}


def test_empty_message_is_rejected():
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_llm_outage_is_503(monkeypatch):
    def down(_):
        raise LLMError("Ollama is not reachable")

    monkeypatch.setattr(app_module, "process", down)
    r = client.post("/chat", json={"message": "hi"})
    assert r.status_code == 503
    assert "not reachable" in r.json()["detail"]


def test_unexpected_error_is_500_without_leaking_details(monkeypatch):
    def boom(_):
        raise KeyError("secret internals")

    monkeypatch.setattr(app_module, "process", boom)
    r = client.post("/chat", json={"message": "hi"})
    assert r.status_code == 500
    assert "secret" not in r.text
