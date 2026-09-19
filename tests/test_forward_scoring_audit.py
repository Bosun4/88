import csv
import json
from forward_ledger.ledger import create_ledger_from_prediction
from forward_ledger.scoring import score_ledger_with_actuals


def test_unverifiable_records_cannot_be_scored_by_bare_id(tmp_path):
    pred = tmp_path / 'p.json'
    ledger = tmp_path / 'l.jsonl'
    actual = tmp_path / 'a.csv'
    pred.write_text(json.dumps({'predictions': [{'match_id': 'x', 'predicted_score': '2-0'}]}), encoding='utf-8')
    create_ledger_from_prediction(str(pred), str(ledger))
    actual.write_text('match_id,actual_score\nx,2-0\n', encoding='utf-8')
    rows = score_ledger_with_actuals(str(ledger), str(actual), str(tmp_path / 's.csv'), str(tmp_path / 's.md'))
    assert rows[0]['settlement_status'] == 'unresolved'
    assert rows[0].get('exact_score_hit') is None


def test_forward_report_has_separate_d_and_side_denominators(tmp_path):
    pred = tmp_path / 'p.json'
    ledger = tmp_path / 'l.jsonl'
    actual = tmp_path / 'a.csv'
    pred.write_text(json.dumps({'matches': {'today': [
        {'fixture_id': 'x', 'kickoff_at': '2099-05-15T18:00:00Z', 'prediction': {
            'predicted_score': '2-0', 'final_direction': 'home', 'recommendation_tier': 'D',
            'risk_score_candidates': [{'score': '1-1'}]}}]}}), encoding='utf-8')
    create_ledger_from_prediction(str(pred), str(ledger))
    actual.write_text('fixture_id,event_date,actual_score\nx,2099-05-15,1-1\n', encoding='utf-8')
    md = tmp_path / 's.md'
    rows = score_ledger_with_actuals(str(ledger), str(actual), str(tmp_path / 's.csv'), str(md))
    assert rows[0]['risk_candidate_covered'] is True
    assert rows[0]['exact_score_hit'] is False
    assert 'side_risk_score' in md.read_text(encoding='utf-8')
    assert 'risk_d' in md.read_text(encoding='utf-8')


def test_forward_scoring_keeps_api_provider_identity(tmp_path):
    pred, ledger, actual = (tmp_path / x for x in ('p.json', 'l.jsonl', 'a.csv'))
    row = {'match_id': 'wencai:123', 'source': 'wencai', 'api_football_fixture_id': 55,
           'home_id': 42, 'away_id': 49, 'league_id': 39,
           'kickoff_at': '2099-09-20T07:30:00+08:00',
           'odds_captured_at': '2026-01-01T00:00:00Z', 'sp_home': 1.22,
           'prediction': {'predicted_score': '2-0', 'final_direction': 'home',
                          'recommendation': {'tier': 'A', 'bet_action': 'main'}}}
    pred.write_text(json.dumps({'matches': {'today': [row]}}), encoding='utf-8')
    create_ledger_from_prediction(str(pred), str(ledger))
    actual.write_text('source,fixture_id,kickoff_at,home_id,away_id,actual_score\napi_football,55,2099-09-19T23:30:00Z,42,49,2-0\n', encoding='utf-8')
    rows = score_ledger_with_actuals(str(ledger), str(actual), str(tmp_path / 's.csv'), str(tmp_path / 's.md'))
    assert rows[0]['settlement_status'] == 'settled'
    assert abs(rows[0]['profit'] - .22) < 1e-9
    assert rows[0]['strict_forward'] is True
    assert rows[0]['evaluation_scope'] == 'strict_forward'
