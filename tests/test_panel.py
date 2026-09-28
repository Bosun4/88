"""Contract tests for the original analysts -> referee sequence."""
import asyncio
import copy
import json
from datetime import datetime, timezone
import pytest
from test_single_pass import predict, row, setup_engine


def run_engine(monkeypatch, tmp_path, handler, budget='180'):
    setup_engine(monkeypatch, tmp_path)
    monkeypatch.setenv('AI_PANEL_MAX_CALLS', budget)
    monkeypatch.setattr(predict, 'async_call_ai_json', handler)
    from panel import run_panel
    return lambda evidence: asyncio.run(run_panel(predict, evidence))


def test_analysts_finish_before_referee_and_keep_model_identity(monkeypatch, tmp_path):
    done = []
    async def call(session, name, system, prompt, phase, expected):
        if phase == 'panel_final':
            assert set(done) == {'gpt', 'grok'}
            assert 'analyst_outputs' in prompt
        else:
            await asyncio.sleep(0)
            done.append(name)
        return name, {'predictions': [row(expected[0])]}, {'ok': True, 'status': 'ok'}
    run = run_engine(monkeypatch, tmp_path, call)
    result = run([{'match': 1, 'fixture_id': 'a'}])
    assert result[1]['source_model'] == 'gemini'
    assert set(result[1]['phase1_model_outputs']) == {'gpt', 'grok'}
    assert result[1]['ai_call_status']['gemini']['final']['row_status'] == 'ok'
    assert predict._LAST_AI_RUN_METADATA['request_count'] == 3
    assert predict._LAST_AI_RUN_METADATA['successful_matches'] == 1


def test_all_analysts_failed_skips_referee_and_reports_failure(monkeypatch, tmp_path):
    calls = []
    async def call(session, name, *args):
        calls.append(name)
        return name, {}, {'ok': False, 'status': 'http_524'}
    run = run_engine(monkeypatch, tmp_path, call)
    result = run([{'match': 1}])
    assert calls == ['gpt', 'grok']
    assert result[1]['final_direction'] == 'abstain'
    assert predict._LAST_AI_RUN_METADATA['run_status'] == 'failed'


def test_shared_endpoint_tls_failure_stops_remaining_fixtures(monkeypatch, tmp_path):
    calls = []
    async def call(session, name, *args):
        calls.append(name)
        return name, {}, {'ok': False, 'status': 'tls_handshake_error', 'error_type': 'ClientConnectorSSLError'}
    monkeypatch.setattr(predict, 'get_url_for_ai', lambda name: 'https://fixture.invalid/v1')
    run = run_engine(monkeypatch, tmp_path, call)
    result = run([{'match': i} for i in range(1, 9)])
    assert 1 <= len(calls) <= 4
    assert all(r['final_direction'] == 'abstain' for r in result.values())
    metadata = predict._LAST_AI_RUN_METADATA
    assert metadata['run_status'] == 'failed'
    assert metadata['failure_summary']['endpoint_unavailable'] >= 12
    assert metadata['failure_summary']['tls_handshake_error'] >= 1


def test_failed_referee_never_promotes_analyst(monkeypatch, tmp_path):
    async def call(session, name, system, prompt, phase, expected):
        return name, {'predictions': [row(expected[0])]} if name != 'gemini' else {}, {'ok': name != 'gemini', 'status': 'http_524' if name == 'gemini' else 'ok'}
    result = run_engine(monkeypatch, tmp_path, call)([{'match': 1}])
    assert result[1]['final_direction'] == 'abstain'
    assert result[1]['phase1_model_outputs']['gpt']['predicted_score'] == '2-1'


@pytest.mark.parametrize('grok_status', ['http_503', 'tls_handshake_error'])
def test_one_model_failure_does_not_disable_healthy_origin(monkeypatch, tmp_path, grok_status):
    calls = []
    async def call(session, name, system, prompt, phase, expected):
        calls.append(name)
        if name == 'grok':
            return name, {}, {'ok': False, 'status': grok_status}
        return name, {'predictions': [row(expected[0])]}, {'ok': True, 'status': 'ok'}
    monkeypatch.setattr(predict, 'get_url_for_ai', lambda name:
                        'https://grok.fixture.invalid/v1' if name == 'grok' and grok_status.startswith('tls')
                        else 'https://shared.fixture.invalid/v1')
    result = run_engine(monkeypatch, tmp_path, call)([{'match': 1}, {'match': 2}])
    assert all(r['source_model'] == 'gemini' for r in result.values())
    assert calls.count('gpt') == 2 and calls.count('gemini') == 2


def test_budget_reserves_whole_match_instead_of_stranding_all_referees(monkeypatch, tmp_path):
    calls = []
    async def call(session, name, system, prompt, phase, expected):
        calls.append((name, expected))
        return name, {'predictions': [row(expected[0])]}, {'ok': True}
    run = run_engine(monkeypatch, tmp_path, call, '3')
    result = run([{'match': 1}, {'match': 2}])
    assert len(calls) == 3
    assert result[1]['final_direction'] == 'home'
    assert result[2]['final_direction'] == 'abstain'
    assert predict._LAST_AI_RUN_METADATA['run_status'] == 'partial'


def test_recapture_reuses_cache_but_quote_change_invalidates(monkeypatch, tmp_path):
    calls = []
    async def call(session, name, system, prompt, phase, expected):
        calls.append(name)
        return name, {'predictions': [row(expected[0])]}, {'ok': True}
    run = run_engine(monkeypatch, tmp_path, call)
    evidence = [{'match': 1, 'captured_at': '2026-09-28T10:00:00Z', 'sp_home': 2.1}]
    run(evidence)
    evidence[0]['captured_at'] = '2026-09-28T10:01:00Z'
    run(evidence)
    assert len(calls) == 3
    assert predict._LAST_AI_RUN_METADATA['cache_hits'] == 3
    evidence[0]['sp_home'] = 2.2
    run(evidence)
    assert len(calls) == 6


def test_duplicate_referee_identifier_abstains(monkeypatch, tmp_path):
    async def call(session, name, system, prompt, phase, expected):
        return name, {'predictions': [row(1)] * (2 if name == 'gemini' else 1)}, {'ok': True}
    result = run_engine(monkeypatch, tmp_path, call)([{'match': 1}])
    assert result[1]['final_direction'] == 'abstain'


def test_self_reported_sources_are_not_accepted_as_retrieval(monkeypatch, tmp_path):
    async def call(session, name, system, prompt, phase, expected):
        r = row(1)
        r['external_fact_table'] = [{'claim': '阵容已确认', 'source_url': 'https://invented.example/news'}]
        r['web_research'] = {'used': True, 'sources': [{'url': 'https://invented.example/news', 'title': '虚构'}]}
        return name, {'predictions': [r]}, {'ok': True}
    result = run_engine(monkeypatch, tmp_path, call)([{'match': 1}])
    assert result[1]['external_fact_table'] == []
    assert result[1]['web_research']['used'] is False
    assert 'unverified_model_sources_removed' in result[1]['validation_warnings']


def test_total_goals_quotes_do_not_create_a_tradeable_over_under_market():
    assert 'over_under' not in predict._extract_market_odds({'a0': 8})
    assert 'over_under' not in predict._extract_market_odds({f'a{i}': 8 for i in range(8)})


def test_review_has_no_hidden_ai_request(monkeypatch, tmp_path):
    import self_learn
    from unittest.mock import Mock
    post = Mock(side_effect=AssertionError('review must be deterministic'))
    monkeypatch.setattr(self_learn.requests, 'post', post)
    monkeypatch.setattr(self_learn, 'GPT_API_KEY', 'configured')
    monkeypatch.setattr(self_learn, 'evaluate_matches', lambda *a: {'ledger': {'samples': 1, 'bettable': {'roi_pct': 0}, 'direction_accuracy_pct': 0}, 'reviews': []})
    monkeypatch.setattr(self_learn.ml, 'coaching_summary', lambda *a: 'summary')
    source = tmp_path / 'pred.json'
    source.write_text('{"matches":{"today":[]}}')
    self_learn.self_learn(str(source), str(tmp_path / 'diary.json'))
    assert post.call_count == 0


@pytest.mark.parametrize('value', ['invalid', 17, ['invalid'], True])
def test_malformed_source_metadata_does_not_discard_a_valid_prediction(value):
    from panel import verified_sources
    prediction = {'raw_item': {'external_fact_table': value, 'web_research': value}}
    result = verified_sources(prediction, {})
    assert result['external_fact_table'] == []
    assert result['web_research']['used'] is False


def test_runner_exception_marks_failed_instead_of_no_eligible_evidence(monkeypatch, tmp_path):
    setup_engine(monkeypatch, tmp_path)
    monkeypatch.setattr(predict, 'AI_RUN_MODE', 'panel')
    async def broken(*args):
        raise RuntimeError('unexpected runner failure')
    monkeypatch.setattr(predict, 'run_ai_native_web', broken)
    output, _ = predict.run_predictions({'matches': [future_match()]})
    assert output[0]['prediction']['is_abstain']
    assert predict._LAST_AI_RUN_METADATA['run_status'] == 'failed'


def future_match():
    return {'fixture_id': 'panel-integration', 'match_id': 'panel-integration',
            'kickoff_at': '2099-09-28T20:00:00Z', 'event_date': '2099-09-28',
            'home_team': 'A', 'away_team': 'B', 'league': '英超',
            'sp_home': 2.1, 'sp_draw': 3.2, 'sp_away': 3.1}


@pytest.mark.parametrize('referee_ok', [True, False])
def test_full_panel_adapter_and_publication_keep_roles(monkeypatch, tmp_path, referee_ok):
    from main import publish_prediction_outputs
    setup_engine(monkeypatch, tmp_path)
    monkeypatch.setattr(predict, 'AI_RUN_MODE', 'panel')
    async def call(session, name, system, prompt, phase, expected):
        ok = referee_ok or name != 'gemini'
        return name, {'predictions': [row(expected[0])]} if ok else {}, {'ok': ok, 'status': 'ok' if ok else 'http_524'}
    monkeypatch.setattr(predict, 'async_call_ai_json', call)
    output, top4 = predict.run_predictions({'matches': [future_match()]})
    pred = output[0]['prediction']
    assert pred['is_abstain'] is not referee_ok
    assert set(pred['phase1_model_outputs']) == {'gpt', 'grok'}
    assert pred['ai_run_metadata']['request_count'] == 3
    assert pred['ai_run_metadata']['run_status'] == ('completed' if referee_ok else 'failed')
    if referee_ok:
        paths = publish_prediction_outputs(str(tmp_path / 'published'), '2099-09-28', 'panel', {'matches': {'today': output}, 'top4': top4}, datetime.now(timezone.utc))
        saved = json.loads(open(paths['live'], encoding='utf-8').read())
        assert saved['matches']['today'][0]['prediction']['predicted_score'] == '2-1'


def test_kickoff_is_rechecked_before_referee(monkeypatch, tmp_path):
    import panel
    checks = []
    def gate(*args):
        checks.append(1)
        return 'eligible' if len(checks) <= 2 else 'already_started'
    monkeypatch.setattr(panel, 'prematch_status', gate)
    calls = []
    async def call(session, name, system, prompt, phase, expected):
        calls.append(name)
        return name, {'predictions': [row(expected[0])]}, {'ok': True}
    result = run_engine(monkeypatch, tmp_path, call)([{'match': 1, 'identity': future_match()}])
    assert calls == ['gpt', 'grok']
    assert result[1]['final_direction'] == 'abstain'
    assert result[1]['ai_call_status']['gemini']['final']['status'] == 'already_started'


def test_run_deadline_prevents_another_stage(monkeypatch, tmp_path):
    import panel
    clock = [0]
    monkeypatch.setattr(panel.time, 'monotonic', lambda: clock[0])
    monkeypatch.setenv('AI_PANEL_MAX_SECONDS', '60')
    calls = []
    async def call(session, name, system, prompt, phase, expected):
        calls.append(name)
        if name == 'grok':
            clock[0] = 61
        return name, {'predictions': [row(expected[0])]}, {'ok': True}
    result = run_engine(monkeypatch, tmp_path, call)([{'match': 1}])
    assert calls == ['gpt', 'grok']
    assert result[1]['ai_call_status']['gemini']['final']['status'] == 'run_deadline_exhausted'
