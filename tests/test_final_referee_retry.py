import asyncio

import scripts.predict as predict


def _run(coro):
    return asyncio.run(coro)


def test_retryable_status_classification():
    assert predict._is_retryable_ai_status({"status": "http_503"}) is True
    assert predict._is_retryable_ai_status({"status": "http_429"}) is True
    assert predict._is_retryable_ai_status({"status": "timeout"}) is True
    assert predict._is_retryable_ai_status({"status": "parse_failed"}) is True
    assert predict._is_retryable_ai_status({"status": "error"}) is True
    # Non-transient: must not retry
    assert predict._is_retryable_ai_status({"status": "no_key"}) is False
    assert predict._is_retryable_ai_status({"status": "no_url"}) is False
    assert predict._is_retryable_ai_status({"status": "http_400"}) is False
    assert predict._is_retryable_ai_status({"status": "aiohttp_missing"}) is False


def test_final_retry_recovers_after_transient_503(monkeypatch):
    calls = {"n": 0}

    async def fake_call(session, ai_name, system_text, prompt, phase, expected):
        calls["n"] += 1
        if calls["n"] < 3:
            return ai_name, {}, {"ok": False, "status": "http_503"}
        return ai_name, {"predictions": []}, {"ok": True, "status": "ok"}

    monkeypatch.setattr(predict, "async_call_ai_json", fake_call)
    monkeypatch.setattr(predict, "AI_FINAL_RETRY_BASE_DELAY", 1)

    name, obj, st = _run(
        predict.async_call_ai_json_with_retry(None, "gemini", "sys", "p", "final", [1], max_retries=2)
    )

    assert calls["n"] == 3
    assert st["ok"] is True
    assert st.get("retry_succeeded_on_attempt") == 3


def test_final_retry_stops_on_non_retryable(monkeypatch):
    calls = {"n": 0}

    async def fake_call(session, ai_name, system_text, prompt, phase, expected):
        calls["n"] += 1
        return ai_name, {}, {"ok": False, "status": "no_key"}

    monkeypatch.setattr(predict, "async_call_ai_json", fake_call)

    name, obj, st = _run(
        predict.async_call_ai_json_with_retry(None, "gemini", "sys", "p", "final", [1], max_retries=3)
    )

    # no_key is permanent: must not retry
    assert calls["n"] == 1
    assert st["ok"] is False
    assert st["status"] == "no_key"


def test_final_retry_exhausts_and_returns_last_failure(monkeypatch):
    calls = {"n": 0}

    async def fake_call(session, ai_name, system_text, prompt, phase, expected):
        calls["n"] += 1
        return ai_name, {}, {"ok": False, "status": "http_503"}

    monkeypatch.setattr(predict, "async_call_ai_json", fake_call)
    monkeypatch.setattr(predict, "AI_FINAL_RETRY_BASE_DELAY", 1)

    name, obj, st = _run(
        predict.async_call_ai_json_with_retry(None, "gemini", "sys", "p", "final", [1], max_retries=2)
    )

    # 1 initial + 2 retries = 3 attempts, then give up
    assert calls["n"] == 3
    assert st["ok"] is False
    assert st["status"] == "http_503"
