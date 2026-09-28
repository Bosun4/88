"""Production panel must not receive result-odds or mathematical score anchors."""
import json
import pytest
from test_panel import future_match
from test_single_pass import predict, row, setup_engine


def market():
    return {**future_match(), 'sp_home': 2.0, 'sp_draw': 4.0, 'sp_away': 4.0,
            'give_ball': -1, 'hhad_win': 3.0, 'hhad_same': 3.0, 'hhad_lose': 3.0,
            'w21': 8, 'l12': 12, 's11': 7, 'a2': 4, 'a3': 5,
            'crs_change': {'w21': -1}, 'ttg_change': {'a3': 1},
            'baseface': '1X2热门主胜', 'expert_intro': '低赔必胜',
            'poisson': {'score': '9-0'}, 'market_implied': {'home_fair_prob': .99}}


def test_panel_evidence_excludes_direction_pricing_and_legacy_hints(monkeypatch):
    monkeypatch.setattr(predict, 'AI_RUN_MODE', 'panel')
    def forbidden(*args):
        raise AssertionError('legacy directional compiler invoked')
    monkeypatch.setattr(predict, '_shin_devig_3way', forbidden)
    evidence = predict.build_evidence_packet(market(), 1)
    text = json.dumps(evidence, ensure_ascii=False)
    for field in ['lottery_market_1x2', 'market_implied', 'dual_market_divergence',
                  'home_fair_prob', '低赔必胜', '"score": "9-0"']:
        assert field not in text
    assert evidence['correct_score_odds']['2-1'] == 8
    assert evidence['market_movements']['correct_score']['w21'] == -1
    assert evidence['handicap']['odds']['home_cover'] == 3


def test_swapping_threeway_odds_cannot_flip_score_evidence(monkeypatch):
    monkeypatch.setattr(predict, 'AI_RUN_MODE', 'panel')
    first = market()
    second = {**first, 'sp_home': 4.0, 'sp_away': 2.0}
    assert predict.build_evidence_packet(first, 1) == predict.build_evidence_packet(second, 1)


def test_margin_distinguishes_overround_from_theoretical_hold_and_missing(monkeypatch):
    monkeypatch.setattr(predict, 'AI_RUN_MODE', 'panel')
    match = {**market(), 'sp_home': 2, 'sp_draw': 3, 'sp_away': 4}
    audit = predict.build_evidence_packet(match, 1)['market_margin_audit']['result_market']
    assert audit['overround_pct'] == pytest.approx(8.3333, abs=.0001)
    assert audit['theoretical_payout_pct'] == pytest.approx(92.3077, abs=.0001)
    assert audit['theoretical_hold_pct'] == pytest.approx(7.6923, abs=.0001)
    partial = predict.build_evidence_packet({**match, 'sp_away': 0}, 1)['market_margin_audit']['result_market']
    assert partial['available'] is False and partial['overround_pct'] is None
    # A partial score list omits 'other' outcomes: no full-market margin claim.
    assert predict.build_evidence_packet(match, 1)['market_margin_audit']['correct_score']['available'] is False


def test_panel_output_does_not_restore_probabilities_or_legacy_score_hints(monkeypatch, tmp_path):
    setup_engine(monkeypatch, tmp_path)
    monkeypatch.setattr(predict, 'AI_RUN_MODE', 'panel')
    def forbidden(*args):
        raise AssertionError('legacy result-odds tail guard invoked')
    monkeypatch.setattr(predict, 'apply_weak_home_tail_risk_protection', forbidden)
    prompts = []
    async def call(session, name, system, prompt, phase, expected):
        prompts.append(prompt)
        answer = row(expected[0])
        answer['market_risk_audit'] = {'status': 'hypothesis', 'evidence': ['让球与比分分歧'],
                                       'alternative_explanations': ['正常风控'], 'confirmation_needed': ['连续报价']}
        return name, {'predictions': [answer]}, {'ok': True, 'status': 'ok'}
    monkeypatch.setattr(predict, 'async_call_ai_json', call)
    output, _ = predict.run_predictions({'matches': [market()]})
    pred = output[0]['prediction']
    assert pred['predicted_score'] == '2-1'
    assert pred['direction_probs'] == {'home': None, 'draw': None, 'away': None}
    assert not pred['probabilities_available']
    assert all(candidate['prob'] is None for candidate in pred['top3'])
    assert pred['market_risk_audit']['status'] == 'hypothesis'
    assert not pred.get('all_tail_scores') and not pred.get('bet_recommendation')
    assert pred['market_margin_audit']['result_market']['overround_pct'] == 0
    assert all('lottery_market_1x2' not in p and 'home_fair_prob' not in p for p in prompts)
    assert 'market_risk_audit' in prompts[-1]


def test_score_output_keeps_source_gate(monkeypatch, tmp_path):
    setup_engine(monkeypatch, tmp_path)
    monkeypatch.setattr(predict, 'AI_RUN_MODE', 'panel')
    async def call(session, name, system, prompt, phase, expected):
        answer = row(expected[0])
        answer.update(reason='官方首发确认因此提升评级', evidence_quality_score=90)
        answer['recommendation'] = {'tier': 'A', 'is_recommended': True, 'bet_action': 'main', 'bet_confidence': 85}
        return name, {'predictions': [answer]}, {'ok': True, 'status': 'ok'}
    monkeypatch.setattr(predict, 'async_call_ai_json', call)
    output, top = predict.run_predictions({'matches': [market()]})
    assert output[0]['prediction']['recommend_gate_pass'] is False
    assert top == []
