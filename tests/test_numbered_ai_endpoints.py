"""Legacy numbered settings must never create another provider endpoint."""
import asyncio
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from scripts import predict


class _Resp:
    status = 503
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        return False
    async def text(self):
        return 'provider unavailable'


class _Session:
    def __init__(self):
        self.calls = []
    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append((url, headers, json))
        return _Resp()


@pytest.fixture
def provider_env(monkeypatch):
    for name in ('GPT', 'GROK', 'GEMINI'):
        monkeypatch.setenv(name + '_API_URL', 'https://' + name.lower() + '.example/v1')
        monkeypatch.setenv(name + '_API_KEY', name + '-key')
        monkeypatch.setenv(name + '_MODEL', name + '-selected-model')
        for slot in range(1, 6):
            monkeypatch.setenv(f'{name}_MODEL_{slot}', 'obsolete-model')
            for suffix in (str(slot), '_' + str(slot)):
                monkeypatch.setenv(name + '_API_URL' + suffix, 'https://obsolete.example/v1')
                monkeypatch.setenv(name + '_API_KEY' + suffix, 'obsolete-key')
    monkeypatch.setattr(predict, 'AI_MOCK_MODE', False)


@pytest.mark.parametrize('name', ['gpt', 'grok', 'gemini'])
def test_only_dedicated_unnumbered_credentials_and_model_are_used(provider_env, name):
    endpoints = predict._endpoint_candidates_for_ai(name)
    assert len(endpoints) == 1
    endpoint = endpoints[0]
    assert endpoint['url'] == f'https://{name}.example/v1'
    assert endpoint['key'] == name.upper() + '-key'
    assert endpoint['model'] == name.upper() + '-selected-model'
    assert predict.get_url_for_ai(name) == endpoint['url']
    assert predict.get_key_for_ai(name) == endpoint['key']


@pytest.mark.parametrize('name', ['gpt', 'grok', 'gemini'])
@pytest.mark.parametrize('missing', ['URL', 'KEY'])
def test_missing_primary_config_never_uses_numbered_or_other_provider(provider_env, monkeypatch, name, missing):
    monkeypatch.delenv(name.upper() + '_API_' + missing)
    assert predict._endpoint_candidates_for_ai(name) == []
    session = _Session()
    _, output, status = asyncio.run(predict.async_call_ai_json(
        session, name, 'system', 'prompt', 'panel_analysis', [1]))
    assert session.calls == []
    assert output == {}
    assert status['ok'] is False


@pytest.mark.parametrize('name,expected', [
    ('gpt', '熊猫-按量-gpt-6-astra'),
    ('grok', 'grok-4.7'),
    ('gemini', 'gemini-3.8-flash-high'),
])
def test_blank_model_uses_current_single_default(provider_env, monkeypatch, name, expected):
    monkeypatch.setenv(name.upper() + '_MODEL', '  ')
    assert predict._endpoint_candidates_for_ai(name)[0]['model'] == expected
    assert predict._model_for(name) == expected


@pytest.mark.parametrize('phase', ['phase1', 'final', 'single_pass', 'panel_analysis', 'panel_final'])
def test_failure_never_rotates_or_fails_over_to_numbered_endpoint(provider_env, phase):
    session = _Session()
    for _ in range(2):
        _, output, status = asyncio.run(predict.async_call_ai_json(
            session, 'gpt', 'system', 'prompt', phase, [1]))
        assert status['status'] == 'http_503'
        assert status['endpoint_total'] == 1
        assert output == {}
    assert len(session.calls) == 2
    for url, headers, payload in session.calls:
        assert url == 'https://gpt.example/v1/chat/completions'
        assert headers['Authorization'] == 'Bearer GPT-key'
        assert payload['model'] == 'GPT-selected-model'
