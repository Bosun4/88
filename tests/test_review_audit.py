import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_self_learn_current_artifact_no_identity_stays_unresolved(tmp_path, monkeypatch):
    import self_learn as sl
    source = Path(__file__).resolve().parents[1] / 'data/predictions.json'
    monkeypatch.setattr(sl, 'PRED_FILE', str(source))
    monkeypatch.setattr(sl, 'DIARY_FILE', str(tmp_path / 'diary.json'))
    monkeypatch.setattr(sl, 'fetch_actual_results', lambda day: (_ for _ in ()).throw(AssertionError('no event date')))
    result = sl.self_learn()
    assert result['ledger']['samples'] == 0
    assert len(result['unresolved']) == 26


def test_post_review_no_pair_only_binding_and_separate_risk_books(tmp_path):
    from post_review import review_predictions, summarize_reviews
    p = tmp_path / 'p.json'
    a = tmp_path / 'a.json'
    row = {'match_id': 'f1', 'home_team': 'A', 'away_team': 'B', 'event_date': '2026-09-20',
           'prediction': {'final_direction': 'home', 'predicted_score': '2-0',
                          'recommendation_tier': 'D', 'recommend_gate_pass': True,
                          'risk_score_candidates': [{'score': '1-1'}]}}
    p.write_text(json.dumps({'matches': {'today': [row, row]}}), encoding='utf-8')
    a.write_text(json.dumps([{'home_team': 'A', 'away_team': 'B', 'actual_score': '1-1'}]), encoding='utf-8')
    result = review_predictions(str(p), str(a))
    assert len(result) == 1
    assert result[0]['settlement_status'] == 'unresolved'
    a.write_text(json.dumps([{**row, 'actual_score': '1-1'}]), encoding='utf-8')
    result = review_predictions(str(p), str(a))
    assert result[0]['recommend_gate_pass'] is False
    report = summarize_reviews(result)
    assert report['risk_d']['samples'] == 1
    assert report['main_score']['hits'] == 0
    assert report['side_risk_score']['hits'] == 1


def test_api_extra_time_uses_regulation_score():
    from fixture_identity import api_actual
    result = api_actual({'fixture': {'id': 1, 'date': '2026-09-20T19:00:00Z', 'status': {'short': 'AET'}},
                         'teams': {}, 'goals': {'home': 2, 'away': 1},
                         'score': {'fulltime': {'home': 1, 'away': 1}}})
    assert result['actual_score'] == '1-1'


def test_self_learn_exact_result_retains_stored_prices_and_unresolved(tmp_path, monkeypatch):
    import self_learn as sl
    monkeypatch.setattr(sl, 'GPT_API_KEY', '')
    row = {'fixture_id': '42', 'source': 'api_football', 'home_id': 1, 'away_id': 2, 'home_team': 'A', 'away_team': 'B',
           'kickoff_at': '2026-09-20T19:00:00Z', 'sp_home': 1.22,
           'prediction': {'final_direction': 'home', 'predicted_score': '2-0',
                          'recommendation': {'tier': 'A', 'bet_action': 'main'}}}
    actual = {'fixture': {'id': 42, 'date': row['kickoff_at'], 'status': {'short': 'FT'}},
              'teams': {'home': {'id': 1, 'name': 'A'}, 'away': {'id': 2, 'name': 'B'}},
              'goals': {'home': 2, 'away': 0}}
    source = tmp_path / 'p.json'
    source.write_text(json.dumps({'matches': {'today': [row, row, {**row, 'fixture_id': '43'}]}}), encoding='utf-8')
    diary = sl.self_learn(str(source), str(tmp_path / 'd.json'), lambda day: [actual])
    assert diary['ledger']['samples'] == 1
    assert diary['ledger']['bettable']['pnl'] == .22
    assert diary['ledger']['strict_forward']['staked'] == 0
    assert len(diary['unresolved']) == 1
    assert json.loads((tmp_path / 'd.json').read_text(encoding='utf-8')) == diary


def test_verify_uses_same_binding_and_metrics(tmp_path, monkeypatch):
    import verify
    import self_learn as sl
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sl, 'GPT_API_KEY', '')
    monkeypatch.setattr(verify, 'get_yesterday', lambda: '2026-09-20')
    monkeypatch.setattr(verify, 'fetch_actual_results', lambda day: [])
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/predictions.json').write_text(json.dumps({'matches': {'today': [{'home_team': 'A', 'away_team': 'B'}]}}), encoding='utf-8')
    result = verify.verify_and_learn()
    assert result['ledger']['samples'] == 0
    assert len(result['unresolved']) == 1


def test_unknown_prediction_does_not_inflate_score_denominators():
    from post_review import score_prediction, summarize_reviews
    review = score_prediction({'prediction': {'is_abstain': True}}, '0-0')
    assert review['direction_hit'] is None
    assert review['goal_band_hit'] is None
    assert review['btts_hit'] is None
    assert summarize_reviews([review])['direction']['samples'] == 0
