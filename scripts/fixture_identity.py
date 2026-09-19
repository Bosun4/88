"""Shared fixture identity. `date`/`id` are capture-day/row IDs, never identity.

Public API: prediction_rows(payload), fixture_key(row), match_actual(row, actuals),
dedupe_predictions(rows), parse_time(value), kickoff_time(row), event_date(row),
api_actual(fixture). Missing/ambiguous identity returns None, never a fuzzy match.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json


def parse_time(value):
    """Parse only timezone-aware ISO timestamps; never guess a timezone."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return dt.astimezone(timezone.utc) if dt.tzinfo else None
    except (ValueError, TypeError):
        return None


def kickoff_time(row):
    return parse_time(row.get('kickoff_at') or row.get('kickoff_at_utc'))


def event_date(row):
    """Canonical UTC kickoff day; explicit event date only if no kickoff exists."""
    kickoff = kickoff_time(row)
    if kickoff:
        return kickoff.date().isoformat()
    value = row.get('event_date') or row.get('fixture_date')
    if value:
        try:
            return datetime.strptime(str(value), '%Y-%m-%d').date().isoformat()
        except ValueError:
            return None
    return None


def provider_ids(row):
    """Known event identifiers by provider; never compare cross-provider IDs."""
    identities = {}
    for field in ('match_id', 'fixture_id'):
        value = row.get(field)
        if value in (None, '', 0, '0'):
            continue
        text = str(value)
        if ':' in text:
            namespace, value = text.split(':', 1)
        else:
            namespace = str(row.get('fixture_source') or row.get('source') or 'unspecified')
        identities[namespace] = str(value)
    if row.get('api_football_fixture_id') not in (None, '', 0, '0'):
        identities['api_football'] = str(row['api_football_fixture_id'])
    return identities

def prediction_rows(payload):
    """Normalize current matches.today, legacy lists, and flat predictions."""
    if isinstance(payload, list):
        return [dict(x) for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        raise ValueError('Prediction payload must be an object or list')
    source = payload.get('matches', payload.get('predictions', payload.get('results', [])))
    if isinstance(source, dict):
        rows = [x for group in source.values() if isinstance(group, list) for x in group]
    elif isinstance(source, list):
        rows = source
    else:
        raise ValueError('Unsupported prediction rows schema')
    return [dict(x) for x in rows if isinstance(x, dict)]


def _text(value):
    return str(value or '').strip().casefold()


def _entity(row, side):
    identity = row.get(f'{side}_id')
    if identity not in (None, '', 0, '0'):
        return 'id:' + str(identity)
    return _text(row.get(f'{side}_team'))


def fixture_key(row):
    """Stable event key, scoped by event date; no business-day fallback."""
    day = event_date(row)
    if not day:
        return None
    stable = row.get('match_id') or row.get('fixture_id') or row.get('api_football_fixture_id')
    if stable:
        ids = provider_ids(row)
        # Primary source ID remains unchanged when provider enrichment arrives.
        namespace, value = next(iter(ids.items()))
        parts = ['fixture', namespace, value, day]
    else:
        home, away = _entity(row, 'home'), _entity(row, 'away')
        league = _text(row.get('league_id') or row.get('league'))
        if not (home and away and league):
            return None
        parts = ['event', league, str(row.get('season') or ''), home, away, day]
    return json.dumps(parts, ensure_ascii=False, separators=(',', ':'))


def same_fixture(pred, actual):
    pk, ak = kickoff_time(pred), kickoff_time(actual)
    if pk and ak:
        if pk != ak:
            return False
    elif not event_date(pred) or event_date(pred) != event_date(actual):
        return False
    p_ids, a_ids = provider_ids(pred), provider_ids(actual)
    common = p_ids.keys() & a_ids.keys()
    if any(p_ids[k] != a_ids[k] for k in common):
        return False
    confirmed = bool(common)
    for field in ('season', 'league_id'):
        if pred.get(field) and actual.get(field) and _text(pred[field]) != _text(actual[field]):
            return False
    for side in ('home', 'away'):
        p_id, a_id = pred.get(side + '_id'), actual.get(side + '_id')
        if p_id not in (None, '', 0, '0') and a_id not in (None, '', 0, '0'):
            if str(p_id) != str(a_id):
                return False
        elif not confirmed and _entity(pred, side) != _entity(actual, side):
            return False
    if confirmed:
        return True
    league_match = any(pred.get(field) and actual.get(field)
                       and _text(pred[field]) == _text(actual[field])
                       for field in ('league_id', 'league'))
    return bool(league_match and all(_entity(pred, side) and _entity(actual, side)
                                    and _entity(pred, side) == _entity(actual, side)
                                    for side in ('home', 'away')))

def match_actual(row, actuals, market='1x2'):
    """Return exactly one compatible regulation-time result, else None."""
    candidates = []
    for actual in actuals:
        if actual.get('market', '1x2') not in ('1x2', 'regulation', '90min'):
            continue
        if market not in ('1x2', 'regulation', '90min'):
            continue
        if same_fixture(row, actual):
            candidates.append(actual)
    # Conflicting duplicate source rows are deliberately unresolved.
    if not candidates:
        return None
    scores = {x.get('actual_score') for x in candidates}
    return candidates[0] if len(scores) == 1 and None not in scores else None


def dedupe_predictions(rows):
    """First saved version per event; unknown identity retained for abstention."""
    seen = set()
    result = []
    for row in rows:
        key = fixture_key(row)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        result.append(row)
    return result


def api_actual(item):
    """API-Football result normalized to regulation-time 1x2 identity."""
    fixture = item.get('fixture') or {}
    status = (fixture.get('status') or {}).get('short')
    if status not in ('FT', 'AET', 'PEN'):
        return None
    # AET/PEN final goals include extra time: only fulltime is a 90-min score.
    goals = (item.get('score') or {}).get('fulltime')
    if not isinstance(goals, dict) or goals.get('home') is None or goals.get('away') is None:
        goals = item.get('goals') if status == 'FT' else None
    if not goals or any(goals.get(s) is None for s in ('home', 'away')):
        return None
    teams = item.get('teams') or {}
    league = item.get('league') or {}
    return {
        'fixture_id': fixture.get('id'), 'api_football_fixture_id': fixture.get('id'),
        'source': 'api_football', 'kickoff_at': fixture.get('date'),
        'league_id': league.get('id'), 'season': league.get('season'),
        'home_id': (teams.get('home') or {}).get('id'),
        'away_id': (teams.get('away') or {}).get('id'),
        'home_team': (teams.get('home') or {}).get('name'),
        'away_team': (teams.get('away') or {}).get('name'),
        'actual_score': f"{goals['home']}-{goals['away']}",
        'home_goals': goals['home'], 'away_goals': goals['away'],
        'status': status, 'market': '1x2',
    }
