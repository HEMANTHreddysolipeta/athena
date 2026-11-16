import pytest
import requests

from athena import llm


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.text = payload, status, str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self._payload


def test_sends_timeout_and_disables_thinking(monkeypatch):
    seen = {}

    def fake_post(url, json, timeout):
        seen.update(json=json, timeout=timeout)
        return FakeResponse({"response": "hi"})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    assert llm.generate("q") == "hi"
    assert seen["timeout"] == llm.settings.llm_timeout_seconds
    assert seen["json"]["think"] is False
    assert seen["json"]["options"]["num_ctx"] == llm.settings.llm_num_ctx


def test_think_blocks_are_stripped(monkeypatch):
    monkeypatch.setattr(llm.requests, "post",
                        lambda *a, **k: FakeResponse({"response": "<think>hmm\n</think>\nAnswer [1]"}))
    assert llm.generate("q") == "Answer [1]"


@pytest.mark.parametrize("error,message", [
    (requests.Timeout(), "did not answer"),
    (requests.ConnectionError(), "not reachable"),
])
def test_network_errors_become_llm_errors(monkeypatch, error, message):
    def fail(*a, **k):
        raise error

    monkeypatch.setattr(llm.requests, "post", fail)
    with pytest.raises(llm.LLMError, match=message):
        llm.generate("q")


def test_http_error_becomes_llm_error(monkeypatch):
    monkeypatch.setattr(llm.requests, "post",
                        lambda *a, **k: FakeResponse({"error": "model not found"}, status=404))
    with pytest.raises(llm.LLMError, match="404"):
        llm.generate("q")
