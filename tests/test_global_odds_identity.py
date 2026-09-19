"""Strict event identity, quote freshness and provenance regression checks."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import global_odds as odds

NOW = datetime(2026, 9, 19, 12, tzinfo=timezone.utc)
KICKOFF = '2026-09-19T19:00:00Z'


def event():
    return {'id': 'event-1', 'sport_key': 'soccer_epl', 'commence_time': KICKOFF,
            'home_team': 'Arsenal', 'away_team': 'Manchester United',
            'bookmakers': [{'key': 'pinnacle', 'title': 'Pinnacle',
                           'last_update': '2026-09-19T11:55:00Z', 'markets': [
                               {'key': 'h2h', 'outcomes': [
                                   {'name': 'Arsenal', 'price': 2.0},
                                   {'name': 'Draw', 'price': 3.5},
                                   {'name': 'Manchester United', 'price': 4.0}]}]}]}


def match():
    return {'home_team': 'Arsenal', 'away_team': 'Manchester United',
            'league': '英超', 'kickoff_at': KICKOFF}


def run(monkeypatch, events, target=None):
    monkeypatch.setattr(odds, 'ODDS_API_KEY', 'test')
    monkeypatch.setattr(odds, '_fetch_sport', lambda key: deepcopy(events))
    monkeypatch.setattr(odds, 'translate_team_name', lambda name: name)
    rows = [target or match()]
    count = odds.enrich_with_global_odds(rows, now=NOW)
    return count, rows[0]


def test_retains_exact_event_and_underlying_quote(monkeypatch):
    count, result = run(monkeypatch, [event()])
    assert count == 1
    evidence = result['global_odds_evidence']
    assert evidence['event_id'] == 'event-1'
    assert evidence['market'] == 'h2h'
    assert evidence['captured_at'] == NOW.isoformat()
    assert evidence['quotes'][0]['bookmaker'] == 'pinnacle'
    assert evidence['quotes'][0]['last_update'] == '2026-09-19T11:55:00+00:00'
    assert result['global_home'] == 2.0


@pytest.mark.parametrize('change', ['opponent', 'league', 'date', 'stale', 'future',
                                    'no_quote_time', 'no_event_id', 'no_target_time',
                                    'ambiguous', 'reversed', 'partial_market', 'nan'])
def test_refuses_uncertain_events_or_quotes(monkeypatch, change):
    candidate, target = event(), match()
    if change == 'opponent':
        candidate['away_team'] = 'Manchester City'
    elif change == 'league':
        candidate['sport_key'] = 'soccer_uefa_champs_league'
    elif change == 'date':
        candidate['commence_time'] = '2026-09-30T19:00:00Z'
    elif change in ('stale', 'future', 'no_quote_time'):
        candidate['bookmakers'][0]['last_update'] = {
            'stale': '2026-09-01T11:55:00Z', 'future': '2026-09-19T13:00:00Z',
            'no_quote_time': None}[change]
    elif change == 'no_event_id':
        candidate.pop('id')
    elif change == 'no_target_time':
        target.pop('kickoff_at')
    elif change == 'reversed':
        candidate['home_team'], candidate['away_team'] = candidate['away_team'], candidate['home_team']
    elif change == 'partial_market':
        candidate['bookmakers'][0]['markets'][0]['outcomes'].pop()
    elif change == 'nan':
        candidate['bookmakers'][0]['markets'][0]['outcomes'][0]['price'] = float('nan')
    events = [candidate]
    if change == 'ambiguous':
        other = deepcopy(candidate)
        other['id'] = 'event-2'
        events.append(other)
    target['global_home'] = 9.99  # A failed refresh must not leave old prices live.
    count, result = run(monkeypatch, events, target)
    assert count == 0
    assert 'global_home' not in result
    assert result['global_odds_evidence']['quality'] == 'unavailable'
