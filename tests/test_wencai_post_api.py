import json
import asyncio
import os
import re
import sys

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import fetch_data


class FakeResponse:
    def __init__(self, payload, status=200):
        self.status = status
        self._text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return self._text


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.payload)


@pytest.fixture
def wencai_env(monkeypatch):
    monkeypatch.setenv("WENCAI_AUTHORIZATION", "test-auth")


def test_wencai_uses_authenticated_json_post_and_random_device_id(wencai_env):
    session = FakeSession({
        "code": 0,
        "msg": "",
        "data": {"matches": {"1": [], "2": []}},
    })

    result = asyncio.run(fetch_data.scrape_wencai_jczq_async(session, "2026-07-18"))

    assert result == []
    assert len(session.calls) == 1
    url, kwargs = session.calls[0]
    assert url == "https://edu.wencaivip.cn/api/v1.reference/matches"
    assert kwargs["headers"]["Authorization"] == "test-auth"
    assert kwargs["headers"]["Origin"] == "http://m.wencai51.cn"
    assert "iPhone" not in kwargs["headers"]["User-Agent"]
    assert re.fullmatch(r"wc-[0-9a-f]{32}", kwargs["json"]["cid"])
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        kwargs["json"]["i"],
    )


def test_wencai_rejects_business_error_without_parsing_matches(wencai_env, capsys):
    session = FakeSession({"code": 301, "msg": "非法请求！", "data": ""})

    result = asyncio.run(fetch_data.scrape_wencai_jczq_async(session, "2026-07-18"))

    assert result == []
    output = capsys.readouterr().out
    assert "code=301" in output
    assert "非法请求" in output


def test_wencai_requires_runtime_credentials(monkeypatch, capsys):
    monkeypatch.delenv("WENCAI_AUTHORIZATION", raising=False)
    session = FakeSession({})

    result = asyncio.run(fetch_data.scrape_wencai_jczq_async(session, "2026-07-18"))

    assert result == []
    assert session.calls == []
    assert "缺少运行凭证" in capsys.readouterr().out


def test_wencai_reports_non_json_response(wencai_env, capsys):
    session = FakeSession("<html>blocked</html>")

    result = asyncio.run(fetch_data.scrape_wencai_jczq_async(session, "2026-07-18"))

    assert result == []
    assert "不是有效 JSON" in capsys.readouterr().out
