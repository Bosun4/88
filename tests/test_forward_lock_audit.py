import json
from pathlib import Path
import pytest
from forward_ledger.ledger import create_ledger_from_prediction


def test_current_artifact_preserved_as_unverifiable_abstentions(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'data/predictions.json'
    expected = len(json.loads(source.read_text(encoding='utf-8'))['matches']['today'])
    ledger = tmp_path / 'ledger.jsonl'
    assert create_ledger_from_prediction(str(source), str(ledger)) == expected
    entries = [json.loads(s) for s in ledger.read_text(encoding='utf-8').splitlines()]
    assert len(entries) == expected > 0
    assert all(not e['strict_forward'] and e['is_abstain'] for e in entries)
    assert all(e['lock_status'] == 'unverifiable_missing_kickoff' for e in entries)
    assert all(e['entry_sha256'] for e in entries)


def test_lock_immutable_hash_and_late_prediction(tmp_path):
    source = tmp_path / 'pred.json'
    ledger = tmp_path / 'ledger.jsonl'
    row = {'match_id': '42', 'kickoff_at': '2099-09-20T19:00:00+08:00',
           'prediction_completed_at': '2026-01-01T00:00:00Z',
           'home_team': 'A', 'away_team': 'B',
           'prediction': {'predicted_score': '2-0', 'final_direction': 'home'}}
    source.write_text(json.dumps({'matches': {'today': [row]}}), encoding='utf-8')
    assert create_ledger_from_prediction(str(source), str(ledger)) == 1
    first = ledger.read_bytes()
    entry = json.loads(first)
    assert entry['strict_forward'] is True
    assert create_ledger_from_prediction(str(source), str(ledger)) == 0
    assert ledger.read_bytes() == first
    row['prediction']['predicted_score'] = '3-0'
    source.write_text(json.dumps({'matches': {'today': [row]}}), encoding='utf-8')
    with pytest.raises(ValueError, match='locked'):
        create_ledger_from_prediction(str(source), str(ledger))
    assert ledger.read_bytes() == first
    assert create_ledger_from_prediction(str(source), str(ledger), on_conflict='keep') == 0
    assert ledger.read_bytes() == first
    entry['predicted_score'] = '4-0'
    ledger.write_text(json.dumps(entry) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        create_ledger_from_prediction(str(source), str(ledger))


def test_late_and_unknown_schema_are_explicit(tmp_path):
    source = tmp_path / 'pred.json'
    ledger = tmp_path / 'ledger.jsonl'
    source.write_text(json.dumps({'matches': {'today': [{'match_id': 'late', 'kickoff_at': '2000-01-01T00:00:00Z'}]}}), encoding='utf-8')
    assert create_ledger_from_prediction(str(source), str(ledger)) == 1
    assert json.loads(ledger.read_text())['lock_status'] == 'after_kickoff'
    source.write_text(json.dumps({'unexpected': [{'match_id': 'x'}]}), encoding='utf-8')
    with pytest.raises(ValueError):
        create_ledger_from_prediction(str(source), str(ledger))
