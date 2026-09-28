import asyncio
import json
import ssl
import socket

import aiohttp
import pytest
from test_single_pass import predict


@pytest.mark.parametrize('exc, expected', [
    (aiohttp.ClientConnectorSSLError(None, ssl.SSLError(1, 'peer alert')), 'tls_handshake_error'),
    (aiohttp.ClientConnectorCertificateError(None, ssl.SSLCertVerificationError(1, 'invalid')), 'tls_certificate_error'),
    (aiohttp.ClientConnectorError(None, socket.gaierror(-2, 'lookup failed')), 'dns_error'),
    (aiohttp.ClientConnectorError(None, ConnectionRefusedError(111, 'refused')), 'connect_error'),
])
def test_aiohttp_wrapped_transport_errors(exc, expected):
    assert predict.transport_failure(exc)['status'] == expected


@pytest.mark.parametrize('exc, expected', [
    (ssl.SSLError(1, '[SSL: TLSV1_ALERT_INTERNAL_ERROR] peer alert'), 'tls_handshake_error'),
    (ssl.SSLCertVerificationError(1, 'certificate verify failed'), 'tls_certificate_error'),
])
def test_transport_classifies_ssl_without_exposing_endpoint_or_key(monkeypatch, exc, expected):
    endpoint = {'name': 'gpt_1', 'model': 'original-model', 'slot': 1,
                'url': 'https://private.fixture.invalid/v1', 'key': 'secret-fixture-value'}
    monkeypatch.setattr(predict, '_endpoint_candidates_for_ai', lambda n: [endpoint])
    monkeypatch.setattr(predict, 'AI_MOCK_MODE', False)
    monkeypatch.setattr(predict, 'aiohttp', aiohttp)
    class Session:
        def post(self, url, **kwargs):
            assert url == 'https://private.fixture.invalid/v1/chat/completions'
            assert kwargs['json']['model'] == 'original-model'
            assert 'ssl' not in kwargs  # Certificate verification remains enabled.
            raise exc
    _, output, status = asyncio.run(predict.async_call_ai_json(Session(), 'gpt', 'system', 'prompt', 'panel_analysis', [1]))
    assert output == {}
    assert status['status'] == expected
    assert status['error_type'] == type(exc).__name__
    assert 'private.fixture.invalid' not in json.dumps(status)
    assert 'secret-fixture-value' not in json.dumps(status)


def test_failed_summary_explains_connection_stage_and_preserves_diagnostics(tmp_path, monkeypatch, capsys):
    import main
    summary = tmp_path / 'summary.md'
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))
    metadata = {'run_status': 'failed', 'total_matches': 8, 'successful_matches': 0,
                'request_count': 4, 'cache_hits': 0,
                'failure_summary': {'tls_handshake_error': 4, 'endpoint_unavailable': 12}}
    main.write_run_diagnostics(str(tmp_path / 'data'), metadata)
    assert json.loads((tmp_path / 'data/ai_phase_results/last_run.json').read_text()) == metadata
    assert 'TLS' in summary.read_text(encoding='utf-8')
    assert 'tls_handshake_error: 4' in capsys.readouterr().out


def bounded_response(monkeypatch, content, phase):
    """Exercise the actual transport parser without contacting a provider."""
    monkeypatch.setattr(predict, 'AI_MOCK_MODE', False)
    monkeypatch.setattr(predict, 'AI_SAVE_RAW_RESPONSE', False)
    monkeypatch.setattr(predict, 'aiohttp', aiohttp)
    monkeypatch.setattr(predict, '_endpoint_candidates_for_ai', lambda name: [{
        'name': 'offline', 'slot': 1, 'url': 'https://offline.invalid/v1',
        'model': 'original-model', 'key': 'fake',
    }])

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def text(self):
            return json.dumps({'choices': [{
                'finish_reason': 'stop', 'message': {'content': content},
            }]})

    class Session:
        def post(self, *args, **kwargs):
            return Response()

    return asyncio.run(predict.async_call_ai_json(
        Session(), 'gemini', 'system', 'prompt', phase, [1],
    ))


@pytest.mark.parametrize('phase', ['single_pass', 'panel_analysis', 'panel_final'])
@pytest.mark.parametrize('wrapper', ['plain', 'fence', 'bom'])
def test_bounded_json_preserves_chinese_quotes_and_literal_markup(monkeypatch, phase, wrapper):
    content = '[{"match":1,"predicted_score":"1-2","reason":"所谓“强阵”与‘落败’待核实；```json```及<think>原文</think>均为引用"}]'
    if wrapper == 'fence':
        content = '```json\n' + content + '\n```'
    elif wrapper == 'bom':
        content = '\ufeff' + content
    _, output, status = bounded_response(monkeypatch, content, phase)
    assert status['ok'] is True
    assert output == [{'match': 1, 'predicted_score': '1-2',
                       'reason': '所谓“强阵”与‘落败’待核实；```json```及<think>原文</think>均为引用'}]


@pytest.mark.parametrize('content', [
    '```json\n{"predictions":[{"match":1,"predicted_score":"1-2"}]',
    '{"predictions":[{"match":1,"predicted_score":"1-2"}]} trailing prose',
    '{“predictions”:[{"match":1,"predicted_score":"1-2"}]}',
    '{"reason":"bad\x00literal","match":1}',
])
def test_bounded_json_rejects_invalid_content_without_repair(monkeypatch, content):
    _, output, status = bounded_response(monkeypatch, content, 'panel_final')
    assert output == {}
    assert status['status'] == 'parse_failed'
