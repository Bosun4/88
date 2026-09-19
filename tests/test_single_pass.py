"""Regression checks for billing bounds, row identity and probability fidelity."""
import asyncio
import copy
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import predict
from main import publish_prediction_outputs
from single_pass import run_single_pass


def row(idx):
    return {'match': idx, 'predicted_score': '2-1', 'final_direction': 'home',
            'direction_probs': {'home': 80, 'draw': 12, 'away': 8},
            'top3': [{'score': '2-1', 'prob': 30}, {'score': '1-0', 'prob': 20},
                     {'score': '2-0', 'prob': 15}],
            'risk_score_candidates': [{'score': '1-2', 'reason': '反击风险'}],
            'recommendation': {'tier': 'D', 'is_recommended': False, 'bet_action': 'observe'}}


def setup_engine(monkeypatch, tmp_path, max_calls='12'):
    monkeypatch.setattr(predict, 'AI_MOCK_MODE', False)
    monkeypatch.setattr(predict, 'aiohttp', None)
    monkeypatch.setattr(predict, 'AI_CHUNK_SIZE', 2)
    monkeypatch.setattr(predict, 'AI_CHUNK_CONCURRENCY', 2)
    monkeypatch.setattr(predict, 'AI_MODEL_CONCURRENCY', 2)
    monkeypatch.setattr(predict, '_save_snapshot', lambda *a, **kw: None)
    monkeypatch.setenv('AI_PRIMARY_MODEL', 'gpt')
    monkeypatch.setenv('AI_PERSISTENT_CACHE_ENABLED', 'true')
    monkeypatch.setenv('AI_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('AI_SINGLE_PASS_MAX_CALLS', max_calls)


def test_one_request_per_batch_and_cache_isolation(monkeypatch, tmp_path):
    setup_engine(monkeypatch, tmp_path)
    calls = []
    async def fake(session, name, system, prompt, phase, expected):
        calls.append(expected)
        return name, {'predictions': [row(i) for i in expected]}, {'ok': True, 'status': 'ok'}
    monkeypatch.setattr(predict, 'async_call_ai_json', fake)
    evidence = [{'match': i, 'fixture_id': f'test-{i}'} for i in range(1, 6)]
    output = asyncio.run(run_single_pass(predict, evidence))
    assert len(calls) == 3
    assert len(output) == 5
    assert predict._LAST_AI_RUN_METADATA['request_count'] == 3
    assert output[1]['ai_call_status']['gpt']['single_pass']['match_ids'] == [1, 2]
    output[1]['ai_call_status']['gpt']['single_pass']['ok'] = False
    assert output[2]['ai_call_status']['gpt']['single_pass']['ok'] is True
    asyncio.run(run_single_pass(predict, evidence))
    assert len(calls) == 3
    assert predict._LAST_AI_RUN_METADATA['request_count'] == 0
    assert predict._LAST_AI_RUN_METADATA['cache_hits'] == 3
    changed = copy.deepcopy(evidence)
    changed[0]['sp_home'] = 2.2
    asyncio.run(run_single_pass(predict, changed))
    assert len(calls) == 4


def test_failure_and_budget_never_retry(monkeypatch, tmp_path):
    setup_engine(monkeypatch, tmp_path, '1')
    calls = []
    async def fail(*args):
        calls.append(1)
        return 'gpt', {}, {'ok': False, 'status': 'timeout'}
    monkeypatch.setattr(predict, 'async_call_ai_json', fail)
    evidence = [{'match': i} for i in range(1, 6)]
    output = asyncio.run(run_single_pass(predict, evidence))
    assert len(calls) == 1
    assert all(r['final_direction'] == 'abstain' for r in output.values())
    # Negative cache also blocks repeating the failed batch within its TTL.
    asyncio.run(run_single_pass(predict, evidence[:2]))
    assert len(calls) == 1


def test_duplicate_or_missing_ids_do_not_bind_by_position(monkeypatch, tmp_path):
    setup_engine(monkeypatch, tmp_path)
    async def fake(*args):
        return 'gpt', {'predictions': [row(1), row(1), {k: v for k, v in row(2).items() if k != 'match'}]}, {'ok': True}
    monkeypatch.setattr(predict, 'async_call_ai_json', fake)
    output = asyncio.run(run_single_pass(predict, [{'match': 1}, {'match': 2}]))
    assert all(r['final_direction'] == 'abstain' for r in output.values())


@pytest.mark.parametrize('bad_id', [True, 1.0, '[1] unrelated', 'match1garbage', None])
def test_malformed_model_ids_are_rejected(monkeypatch, tmp_path, bad_id):
    setup_engine(monkeypatch, tmp_path)

    async def fake(*args):
        return 'gpt', {'predictions': [row(bad_id), row(2)]}, {'ok': True}

    monkeypatch.setattr(predict, 'async_call_ai_json', fake)
    output = asyncio.run(run_single_pass(predict, [{'match': 1}, {'match': 2}]))
    assert output[1]['final_direction'] == 'abstain'
    assert output[2]['final_direction'] == 'home'


@pytest.mark.parametrize('chunk_size', [1, 2])
def test_duplicate_input_rows_use_one_request(monkeypatch, tmp_path, chunk_size):
    setup_engine(monkeypatch, tmp_path)
    monkeypatch.setattr(predict, 'AI_CHUNK_SIZE', chunk_size)
    monkeypatch.setenv('AI_PERSISTENT_CACHE_ENABLED', 'false')
    calls = []

    async def fake(session, name, system, prompt, phase, expected):
        calls.append(expected)
        return name, {'predictions': [row(i) for i in expected]}, {'ok': True}

    monkeypatch.setattr(predict, 'async_call_ai_json', fake)
    evidence = [{'match': 1, 'fixture_id': 'a'}] * 2
    output = asyncio.run(run_single_pass(predict, evidence))
    assert calls == [[1]]
    assert output[1]['final_direction'] == 'home'


def test_conflicting_input_ids_fail_before_call(monkeypatch, tmp_path):
    setup_engine(monkeypatch, tmp_path)
    calls = []

    async def fake(*args):
        calls.append(1)
        return 'gpt', {}, {'ok': False}

    monkeypatch.setattr(predict, 'async_call_ai_json', fake)
    with pytest.raises(ValueError, match='conflicting.*match'):
        asyncio.run(run_single_pass(predict, [
            {'match': 1, 'fixture_id': 'a'}, {'match': 1, 'fixture_id': 'b'},
        ]))
    assert calls == []


def test_interrupted_request_is_not_billed_again(monkeypatch, tmp_path):
    setup_engine(monkeypatch, tmp_path)
    calls = []

    async def interrupted(*args):
        calls.append(1)
        raise asyncio.CancelledError()

    monkeypatch.setattr(predict, 'async_call_ai_json', interrupted)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_single_pass(predict, [{'match': 1}]))
    output = asyncio.run(run_single_pass(predict, [{'match': 1}]))
    assert calls == [1]
    assert output[1]['final_direction'] == 'abstain'


@pytest.mark.parametrize('setting,value', [
    ('AI_TEMPERATURE_PHASE1', 0.77),
    ('AI_MAX_OUTPUT_TOKENS', 9876),
    ('AI_USE_RESPONSE_FORMAT', False),
])
def test_request_setting_change_invalidates_cache(monkeypatch, tmp_path, setting, value):
    setup_engine(monkeypatch, tmp_path)
    monkeypatch.setattr(predict, setting, True if setting == 'AI_USE_RESPONSE_FORMAT' else 0)
    calls = []

    async def fake(*args):
        calls.append(1)
        return 'gpt', {'predictions': [row(1)]}, {'ok': True}

    monkeypatch.setattr(predict, 'async_call_ai_json', fake)
    asyncio.run(run_single_pass(predict, [{'match': 1}]))
    monkeypatch.setattr(predict, setting, value)
    asyncio.run(run_single_pass(predict, [{'match': 1}]))
    assert calls == [1, 1]


def test_changed_evidence_timestamp_invalidates_cache(monkeypatch, tmp_path):
    setup_engine(monkeypatch, tmp_path)
    calls = []

    async def fake(*args):
        calls.append(1)
        return 'gpt', {'predictions': [row(1)]}, {'ok': True}

    monkeypatch.setattr(predict, 'async_call_ai_json', fake)
    for timestamp in ('2026-09-18T12:00:00Z', '2026-09-19T12:00:00Z'):
        asyncio.run(run_single_pass(predict, [
            {'match': 1, 'source': {'update_time': timestamp}},
        ]))
    assert calls == [1, 1]


def test_zero_call_budget_dispatches_nothing(monkeypatch, tmp_path):
    setup_engine(monkeypatch, tmp_path, '0')
    calls = []

    async def fake(*args):
        calls.append(1)
        return 'gpt', {}, {'ok': False}

    monkeypatch.setattr(predict, 'async_call_ai_json', fake)
    output = asyncio.run(run_single_pass(predict, [{'match': 1}]))
    assert calls == []
    assert output[1]['final_direction'] == 'abstain'
    assert predict._LAST_AI_RUN_METADATA['request_count'] == 0


def test_overlapping_runs_do_not_reset_active_run_state(monkeypatch, tmp_path):
    setup_engine(monkeypatch, tmp_path)
    calls = []

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def fake(*args):
            calls.append(1)
            predict.AI_CALL_STATUS['gpt'] = {'active': True}
            entered.set()
            await release.wait()
            return 'gpt', {'predictions': [row(1)]}, {'ok': True}

        monkeypatch.setattr(predict, 'async_call_ai_json', fake)
        first = asyncio.create_task(run_single_pass(predict, [{'match': 1}]))
        await entered.wait()
        try:
            with pytest.raises(RuntimeError, match='already running'):
                await run_single_pass(predict, [{'match': 1}])
            assert predict.AI_CALL_STATUS['gpt'] == {'active': True}
        finally:
            release.set()
            await first

    asyncio.run(scenario())
    assert calls == [1]


def test_failed_run_clears_previous_run_metadata(monkeypatch, tmp_path):
    setup_engine(monkeypatch, tmp_path)
    monkeypatch.setattr(predict, '_LAST_AI_RUN_METADATA', {'run_id': 'previous'})
    monkeypatch.setitem(predict.AI_CALL_STATUS['gpt'], 'old', {'ok': True})
    with pytest.raises(ValueError):
        asyncio.run(run_single_pass(predict, [{'match': True}]))
    assert predict._LAST_AI_RUN_METADATA.get('run_id') != 'previous'
    assert predict._LAST_AI_RUN_METADATA['run_status'] == 'failed'
    assert predict.AI_CALL_STATUS['gpt'] == {}


def test_batch_exception_settles_siblings_before_next_run(monkeypatch, tmp_path):
    setup_engine(monkeypatch, tmp_path)
    monkeypatch.setattr(predict, 'AI_CHUNK_SIZE', 1)
    monkeypatch.setenv('AI_PERSISTENT_CACHE_ENABLED', 'false')

    async def scenario():
        entered, release, sibling_done = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def fake(session, name, system, prompt, phase, expected):
            if expected == [1]:
                await entered.wait()
                raise RuntimeError('unexpected transport exception')
            entered.set()
            try:
                await release.wait()
            finally:
                sibling_done.set()
            return name, {'predictions': [row(2)]}, {'ok': True}

        monkeypatch.setattr(predict, 'async_call_ai_json', fake)
        try:
            with pytest.raises(RuntimeError, match='unexpected transport'):
                await run_single_pass(predict, [{'match': 1}, {'match': 2}])
            assert sibling_done.is_set()
        finally:
            release.set()

    asyncio.run(scenario())


def test_cache_is_rechecked_after_acquiring_disk_lock(monkeypatch, tmp_path):
    setup_engine(monkeypatch, tmp_path)
    import single_pass
    real_open = single_pass.os.open
    calls = []

    def racing_open(path, flags):
        # Another process finished after our first read, before lock acquisition.
        single_pass._save_cache(path.with_suffix('.json'),
                                {'predictions': [row(1)]}, {'ok': True})
        return real_open(path, flags)

    async def fake(*args):
        calls.append(1)
        return 'gpt', {}, {'ok': False}

    monkeypatch.setattr(single_pass.os, 'open', racing_open)
    monkeypatch.setattr(predict, 'async_call_ai_json', fake)
    output = asyncio.run(run_single_pass(predict, [{'match': 1}]))
    assert calls == []
    assert output[1]['final_direction'] == 'home'
    assert predict._LAST_AI_RUN_METADATA['cache_hits'] == 1


@pytest.mark.parametrize('probabilities', [
    {'home': 80},
    {'home': 80, 'draw': None, 'away': 20},
    {'home': 80, 'draw': True, 'away': 19},
    {'home': 80, 'draw': -1, 'away': 21},
    {'home': 80, 'draw': float('nan'), 'away': 20},
    {'home': 80, 'draw': 80, 'away': 80},
])
def test_invalid_direction_vector_cannot_generate_ev(monkeypatch, tmp_path, probabilities):
    setup_engine(monkeypatch, tmp_path)
    monkeypatch.setattr(predict, 'AI_RUN_MODE', 'single_pass')

    async def fake(*args):
        prediction = row(1)
        prediction['direction_probs'] = probabilities
        return 'gpt', {'predictions': [prediction]}, {'ok': True}

    monkeypatch.setattr(predict, 'async_call_ai_json', fake)
    output = asyncio.run(run_single_pass(predict, [{'match': 1}]))
    adapted = predict.adapt_ai_to_frontend(output[1], {})
    assert adapted['probabilities_available'] is False
    assert predict._build_bet_candidates(adapted, {'one_x_two': {'home': 2.0}}) == []


def test_one_percent_survives_single_pass_and_frontend(monkeypatch, tmp_path):
    setup_engine(monkeypatch, tmp_path)
    monkeypatch.setattr(predict, 'AI_RUN_MODE', 'single_pass')

    async def fake(*args):
        prediction = row(1)
        prediction['direction_probs'] = {'home': 1, 'draw': 49, 'away': 50}
        prediction['top3'][0]['prob'] = 1
        return 'gpt', {'predictions': [prediction]}, {'ok': True}

    monkeypatch.setattr(predict, 'async_call_ai_json', fake)
    output = asyncio.run(run_single_pass(predict, [{'match': 1}]))
    adapted = predict.adapt_ai_to_frontend(output[1], {})
    assert adapted['direction_probs']['home'] == 1
    assert adapted['top3'][0]['prob'] == 1
    legs = predict._build_bet_candidates(adapted, {
        'one_x_two': {'home': 2.0}, 'correct_score': {'2-1': 20},
    })
    assert len(legs) == 2
    assert all(leg['p_model'] == .01 for leg in legs)


def test_concurrency_and_call_budget_under_contention(monkeypatch, tmp_path):
    setup_engine(monkeypatch, tmp_path, '3')
    monkeypatch.setattr(predict, 'AI_CHUNK_SIZE', 1)
    active = peak = calls = 0

    async def fake(session, name, system, prompt, phase, expected):
        nonlocal active, peak, calls
        active += 1
        calls += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return name, {'predictions': [row(i) for i in expected]}, {'ok': True}

    monkeypatch.setattr(predict, 'async_call_ai_json', fake)
    output = asyncio.run(run_single_pass(predict, [{'match': i} for i in range(1, 11)]))
    assert peak == 2
    assert calls == 3
    assert len(output) == 10
    assert sum(r['final_direction'] == 'abstain' for r in output.values()) == 7


@pytest.mark.parametrize('finish_reason', ['length', 'stop', None])
def test_single_pass_transport_rejects_truncated_output(monkeypatch, tmp_path, finish_reason):
    setup_engine(monkeypatch, tmp_path)
    from types import SimpleNamespace

    monkeypatch.setattr(predict, 'aiohttp', SimpleNamespace(ClientTimeout=lambda **kw: kw))
    monkeypatch.setattr(predict, 'AI_SAVE_RAW_RESPONSE', False)
    monkeypatch.setattr(predict, '_endpoint_candidates_for_ai', lambda name: [{
        'name': 'offline', 'slot': 1, 'url': 'https://offline.invalid/v1',
        'model': 'offline', 'key': 'fake',
    }])

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def text(self):
            return json.dumps({'choices': [{
                'finish_reason': finish_reason,
                'message': {'content': '{"predictions":[' + json.dumps(row(1))},
            }]})

    class Session:
        def post(self, *args, **kwargs):
            return Response()

    _, _, status = asyncio.run(predict.async_call_ai_json(
        Session(), 'gpt', 'system', 'prompt', 'single_pass', [1],
    ))
    assert status['ok'] is False


def test_probabilities_and_risk_d_survive_adapter(monkeypatch):
    monkeypatch.setattr(predict, 'AI_RUN_MODE', 'single_pass')
    ai = predict.normalize_ai_predictions({'predictions': [row(1)]}, [1], 'gpt', 'single_pass')[1]
    adapted = predict.adapt_ai_to_frontend(ai, {'home_team': 'A', 'away_team': 'B', 'league': '英超'})
    assert adapted['direction_probs']['home'] == 80
    assert adapted['top3'][0]['prob'] == 30
    assert adapted['risk_score_candidates'][0]['score'] == '1-2'
    assert adapted['predicted_score'] == '2-1'
    assert adapted['grok_score'] == ''
    assert adapted['gemini_score'] == ''
    legs = predict._build_bet_candidates(adapted, {'one_x_two': {'home': 2.0}})
    home = next(x for x in legs if x['market'] == 'one_x_two')
    assert home['p_model'] == .8
    missing = {**adapted, 'direction_probs': {}, 'top3': [], 'top_score_candidates': []}
    assert predict._build_bet_candidates(missing, {'one_x_two': {'home': 2.0}}) == []


def test_compact_prompt_preserves_all_market_quotes_and_source_context():
    from single_pass import compact_evidence
    snapshot = json.loads((__import__('pathlib').Path(__file__).resolve().parents[1]
                           / 'data/predictions.json').read_text(encoding='utf-8'))
    evidence = predict.build_evidence_packet(snapshot['matches']['today'][0], 1)
    compact = compact_evidence(evidence)
    for key in ('correct_score_odds', 'lottery_market_1x2', 'total_goals_odds',
                'league_context', 'match'):
        assert compact[key] == evidence[key]
    assert len(json.dumps(compact)) < len(json.dumps(evidence)) * .6
    assert 'adjacent_score_audit_table' in evidence['score_cluster_diagnostics_v203']
    assert 'adjacent_score_audit_table' not in compact['score_cluster_diagnostics_v203']


def test_publication_rechecks_kickoff_and_preserves_risk(tmp_path):
    now = datetime(2099, 9, 19, 20, tzinfo=timezone.utc)
    match = {'match_id': 'x', 'kickoff_at': (now-timedelta(minutes=1)).isoformat(),
             'prediction': {'predicted_score': '2-1', 'recommend_gate_pass': True,
                            'risk_score_candidates': [{'score': '1-2'}]}}
    payload = {'matches': {'today': [match]}, 'top4': [copy.deepcopy(match)]}
    paths = publish_prediction_outputs(str(tmp_path/'data'), '2099-09-19', 'test', payload, now)
    saved = json.loads(open(paths['live'], encoding='utf-8').read())
    assert saved['top4'] == []
    prediction = saved['matches']['today'][0]['prediction']
    assert prediction['prematch_status'] == 'already_started'
    assert prediction['bet_action'] == 'no_bet'
    assert prediction['risk_score_candidates'] == [{'score': '1-2'}]


@pytest.mark.parametrize('prob', [0.5, 1, 0, None, True, float('nan'), -1, 101])
def test_score_percentage_survives_every_adapter(monkeypatch, tmp_path, prob):
    setup_engine(monkeypatch, tmp_path)
    monkeypatch.setattr(predict, 'AI_RUN_MODE', 'single_pass')
    expected = prob if isinstance(prob, (int, float)) and not isinstance(prob, bool) and 0 <= prob <= 100 else None

    async def fake(*args):
        prediction = row(1)
        prediction['top3'] = [{'score': '0-0', 'prob': 5}, {'score': '2-1', 'prob': prob}]
        prediction['risk_score_candidates'] = [{'score': '1-2', 'prob': prob}]
        return 'gpt', {'predictions': [prediction]}, {'ok': True}

    monkeypatch.setattr(predict, 'async_call_ai_json', fake)
    output = asyncio.run(run_single_pass(predict, [{'match': 1}]))
    adapted = predict.adapt_ai_to_frontend(output[1], {})
    assert adapted['top3'][1]['prob'] == expected
    assert dict(adapted['top_score_candidates'])['2-1'] == expected
    assert adapted['score_model_prob'] == expected  # main score, not first candidate
    assert predict._normalize_risk_score_candidates([{'score': '1-2', 'prob': prob}])[0].get('prob') == expected
    assert predict._candidate_score_prob({'score': '1-2', 'prob': prob}) == (expected or 0)
    legs = predict._build_bet_candidates(adapted, {'correct_score': {'2-1': 20}})
    assert (legs[0]['p_model'] if legs else None) == (round(expected / 100, 3) if expected else None)


def test_direction_gate_keeps_one_percent():
    assert predict._extract_prob_map_0_100({'home': 1, 'draw': 49, 'away': 50}) == {'home': 1, 'draw': 49, 'away': 50}


@pytest.mark.parametrize('finish', ['stop', 'length', None])
def test_single_pass_sse_accepts_only_completed_content(finish):
    events = [
        {'choices': [{'delta': {'reasoning_content': 'private analysis'}}]},
        {'choices': [{'delta': {'content': '{"predictions":'}}]},
        {'choices': [{'delta': {'content': '[]}'}, 'finish_reason': finish}]},
    ]
    stream = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events) + 'data: [DONE]\n'
    if finish == 'stop':
        assert predict._extract_response_text(predict._single_pass_sse_payload(stream)) == '{"predictions":[]}'
    else:
        with pytest.raises(ValueError, match='stream_incomplete'):
            predict._single_pass_sse_payload(stream)


def test_publish_lock_rerun_and_risk_settlement_end_to_end(tmp_path):
    from forward_ledger.ledger import read_verified_entries
    from forward_ledger.scoring import score_ledger_with_actuals
    now = datetime.now(timezone.utc)
    match = {'fixture_id': 'integration-fixture', 'kickoff_at': '2099-09-19T20:00:00Z',
             'prediction': {'predicted_score': '2-1', 'final_direction': 'home',
                            'recommendation_tier': 'D', 'recommend_gate_pass': False,
                            'risk_score_candidates': [{'score': '1-2'}]}}
    payload = {'matches': {'today': [match]}, 'top4': []}
    data_dir = tmp_path/'data'
    publish_prediction_outputs(str(data_dir), '2099-09-19', 'test', copy.deepcopy(payload), now)
    ledger = data_dir/'forward_ledger.jsonl'
    first = ledger.read_bytes()
    publish_prediction_outputs(str(data_dir), '2099-09-19', 'test', copy.deepcopy(payload), now)
    assert ledger.read_bytes() == first
    assert len(read_verified_entries(ledger)) == 1
    actuals = tmp_path/'actuals.csv'
    actuals.write_text('fixture_id,event_date,actual_score\nintegration-fixture,2099-09-19,1-2\n')
    result = score_ledger_with_actuals(str(ledger), str(actuals), str(tmp_path/'score.csv'), str(tmp_path/'score.md'))
    assert result[0]['settlement_status'] == 'settled'
    assert result[0]['exact_score_hit'] is False
    assert result[0]['risk_candidate_covered'] is True
    assert result[0]['strict_forward'] is True


@pytest.mark.parametrize("mode", ["single_pass", "fast_batch", "deep_research", "post_review"])
def test_run_modes_import_without_network(mode):
    from pathlib import Path
    import os
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, 'scripts'); import predict; print(predict.AI_RUN_MODE)"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "AI_RUN_MODE": mode, "VMAX_ALLOW_AUTO_INSTALL": "false"},
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith(mode)
