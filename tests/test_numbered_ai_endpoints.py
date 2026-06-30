import asyncio
import importlib
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import predict


class _Resp:
    def __init__(self, status, text):
        self.status = status
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return self._text


class _SeqSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers or {}, "json": json or {}})
        status, text = self.responses.pop(0)
        return _Resp(status, text)


def _ok_text():
    return '{"choices":[{"message":{"content":"{\\"predictions\\": []}"}}]}'


def test_numbered_endpoint_candidates_read_env_and_code_models(monkeypatch):
    monkeypatch.setenv("GEMINI_API_URL", "https://slot1.example/v1")
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.setenv("GEMINI_API_URL_2", "https://slot2.example/v1")
    monkeypatch.setenv("GEMINI_API_KEY_2", "k2")
    monkeypatch.setitem(predict.AI_ENDPOINT_MODEL_SLOTS["gemini"], 2, "gemini-slot-2-model")

    eps = predict._endpoint_candidates_for_ai("gemini")

    assert [e["slot"] for e in eps[:2]] == [1, 2]
    assert eps[0]["model"] == predict.DEFAULT_MODELS["gemini"]
    assert eps[1]["model"] == "gemini-slot-2-model"
    assert eps[1]["url"] == "https://slot2.example/v1"
    assert eps[1]["key"] == "k2"


def test_numbered_endpoint_round_robin(monkeypatch):
    monkeypatch.setenv("GPT_API_URL", "https://slot1.example/v1")
    monkeypatch.setenv("GPT_API_KEY", "k1")
    monkeypatch.setenv("GPT_API_URL_2", "https://slot2.example/v1")
    monkeypatch.setenv("GPT_API_KEY_2", "k2")
    monkeypatch.setitem(predict.AI_ENDPOINT_MODEL_SLOTS["gpt"], 2, "gpt-slot-2-model")
    predict.AI_ENDPOINT_RR_CURSOR.clear()

    first = predict._ordered_endpoints_for_ai("gpt")
    second = predict._ordered_endpoints_for_ai("gpt")

    assert first[0]["slot"] == 1
    assert second[0]["slot"] == 2


def test_async_call_failovers_to_second_numbered_endpoint(monkeypatch):
    monkeypatch.setenv("GPT_API_URL", "https://slot1.example/v1")
    monkeypatch.setenv("GPT_API_KEY", "k1")
    monkeypatch.setenv("GPT_API_URL_2", "https://slot2.example/v1")
    monkeypatch.setenv("GPT_API_KEY_2", "k2")
    monkeypatch.setitem(predict.AI_ENDPOINT_MODEL_SLOTS["gpt"], 2, "gpt-slot-2-model")
    predict.AI_ENDPOINT_RR_CURSOR.clear()

    session = _SeqSession([(429, "busy"), (200, _ok_text())])
    ai, obj, st = asyncio.run(predict.async_call_ai_json(session, "gpt", "sys", "prompt", "phase1", [1]))

    assert ai == "gpt"
    assert st["ok"] is True
    assert st["endpoint_slot"] == 2
    assert session.calls[0]["url"] == "https://slot1.example/v1/chat/completions"
    assert session.calls[1]["url"] == "https://slot2.example/v1/chat/completions"
    assert session.calls[1]["json"]["model"] == "gpt-slot-2-model"
