"""Append-only, hash-verified prediction snapshots. Never backdate a lock."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from forward_ledger.hash_utils import sha256_file
from scripts.fixture_identity import event_date, fixture_key, kickoff_time, parse_time, prediction_rows
from scripts.metrics_ledger import score_channels


def _digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def read_verified_entries(path):
    """Validate the complete hash chain before append or scoring.

    Legacy entries without hashes remain readable but cannot be strict forward.
    """
    entries = []
    previous = ''
    path = Path(path)
    if not path.exists():
        return entries
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        digest = entry.get('entry_sha256')
        if digest:
            payload = {k: v for k, v in entry.items() if k != 'entry_sha256'}
            if _digest(payload) != digest or entry.get('previous_entry_sha256', '') != previous:
                raise ValueError('Ledger hash verification failed')
            previous = digest
        elif entry.get('schema_version') == 2:
            raise ValueError('Ledger hash missing')
        else:
            entry['strict_forward'] = False
            entry['lock_status'] = 'legacy_unverified'
        entries.append(entry)
    return entries


def _entry(row, data, source, sha, index, count, locked_at):
    pred = row.get('prediction', row)
    if not isinstance(pred, dict):
        pred = {}
    meta = data.get('metadata') or {} if isinstance(data, dict) else {}
    kickoff = kickoff_time(row)
    key = fixture_key(row)
    completed_raw = (row.get('prediction_completed_at') or pred.get('prediction_completed_at')
                     or meta.get('created_at_utc'))
    completed = parse_time(completed_raw)
    if not kickoff:
        status = 'unverifiable_missing_kickoff'
    elif locked_at >= kickoff:
        status = 'after_kickoff'
    elif not key:
        status = 'unverifiable_missing_identity'
    elif completed_raw and (not completed or completed > locked_at or completed >= kickoff):
        status = 'unverifiable_prediction_time'
    else:
        # Locking the bytes now is itself proof of existence before kickoff.
        status = 'locked_pre_kickoff'
    strict = status == 'locked_pre_kickoff'
    main, side = score_channels(pred)
    matrix = pred.get('matrix_shadow_layer') or {}
    probs = pred.get('probabilities') or {}
    top = matrix.get('matrix_top_scores', pred.get('matrix_top_scores', []))
    top = [x.get('score') if isinstance(x, dict) else x[0] if isinstance(x, list) else x for x in top] if isinstance(top, list) else []
    identity = {k: row[k] for k in ('fixture_id', 'match_id', 'api_football_fixture_id', 'source', 'fixture_source', 'league', 'league_id', 'season', 'home_id', 'away_id') if k in row}
    row_hash = _digest(row)
    # Unknown events are retained individually by source+index, never silently lost.
    lock_key = key or f'unverifiable:{sha}:{index}'
    return {
        **identity, 'schema_version': 2, 'fixture_key': key, 'lock_key': lock_key,
        'prediction_file': Path(source).name, 'prediction_sha256': sha,
        'prediction_row_sha256': row_hash, 'prediction_snapshot': row,
        'created_at_utc': completed_raw, 'locked_at_utc': locked_at.isoformat(),
        'match_count': count, 'engine_commit': meta.get('engine_commit', 'unknown'),
        'match_id': row.get('match_id') or row.get('fixture_id') or '',
        'home_team': row.get('home_team', ''), 'away_team': row.get('away_team', ''),
        'kickoff_at': row.get('kickoff_at') or row.get('kickoff_at_utc'),
        'event_date': event_date(row), 'lock_status': status, 'strict_forward': strict,
        'is_abstain': bool(pred.get('is_abstain') or not strict),
        'no_bet_reason': pred.get('no_bet_reason') or (status if not strict else None),
        'predicted_score': main or '', 'final_direction': pred.get('final_direction', ''),
        'confidence': pred.get('confidence'),
        'home_win_pct': probs.get('home', pred.get('home_win_pct')),
        'draw_pct': probs.get('draw', pred.get('draw_pct')),
        'away_win_pct': probs.get('away', pred.get('away_win_pct')),
        'recommendation_tier': pred.get('recommendation_tier') or (pred.get('recommendation') or {}).get('tier') or 'D',
        'risk_score_candidates': side, 'tail_risk_flags': pred.get('tail_risk_flags', []),
        'matrix_top_scores': top,
        'matrix_recommended_score': matrix.get('recommended_score', pred.get('matrix_recommended_score')),
        'matrix_recommended_direction': matrix.get('recommended_direction', pred.get('matrix_recommended_direction')),
        'matrix_disagreement_flags': matrix.get('disagreement_flags', pred.get('matrix_disagreement_flags', [])),
        'sub50_tiebreaker_warning': pred.get('sub50_tiebreaker_warning', False),
        'score_cluster': pred.get('score_cluster', []),
        'score_moderation_applied': pred.get('score_moderation_applied', False),
        'original_predicted_score': pred.get('original_predicted_score'),
    }


def create_ledger_from_prediction(prediction_json: str, output_jsonl: str, *, on_conflict="error"):
    """Return newly appended count; identical rerun is idempotent.

    Conflicts raise before writing, or on_conflict="keep" retains the first
    locked event unchanged for recurring production publication.
    Missing kickoff yields an explicit abstention, not an empty ledger.
    """
    if on_conflict not in {"error", "keep"}:
        raise ValueError("on_conflict must be error or keep")
    source = Path(prediction_json)
    raw = source.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    if isinstance(data, dict) and not any(k in data for k in ('matches', 'predictions', 'results')):
        raise ValueError('Unsupported prediction payload schema')
    rows = prediction_rows(data)
    if not rows:
        raise ValueError('Prediction payload has no matches to lock')
    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    guard = out.with_name(out.name + '.lock')
    try:
        handle = os.open(str(guard), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError('Ledger is locked by another writer') from exc
    try:
        os.close(handle)
        existing = read_verified_entries(out)
        by_key = {e.get('lock_key'): e for e in existing if e.get('lock_key')}
        previous = existing[-1].get('entry_sha256', '') if existing else ''
        locked_at = datetime.now(timezone.utc)
        pending = []
        for index, row in enumerate(rows):
            entry = _entry(row, data, prediction_json, sha, index, len(rows), locked_at)
            old = by_key.get(entry['lock_key'])
            if old:
                if (on_conflict == 'error'
                        and old.get('prediction_row_sha256') != entry['prediction_row_sha256']):
                    raise ValueError(f"Event already locked: {entry['lock_key']}")
                continue
            entry['previous_entry_sha256'] = previous
            entry['entry_sha256'] = _digest(entry)
            previous = entry['entry_sha256']
            pending.append(entry)
            by_key[entry['lock_key']] = entry
        if pending:
            with out.open('a', encoding='utf-8') as stream:
                for entry in pending:
                    stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        return len(pending)
    finally:
        guard.unlink(missing_ok=True)
