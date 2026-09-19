"""Offline regressions for audit 04/05; timestamps are synthetic fixtures."""
import asyncio
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fetch_data
from test_wencai_post_api import FakeSession


def test_fetch_preserves_stable_identity_and_aware_times(monkeypatch):
    monkeypatch.setenv('WENCAI_AUTHORIZATION', 'test-auth')
    kickoff = datetime(2099, 9, 19, 13, tzinfo=timezone.utc)
    item = {'home': 'Arsenal', 'guest': 'Chelsea', 'cup': '英超',
            'stime': str(int(kickoff.timestamp())), 'id': 123,
            'week': '周六', 'week_no': '001'}
    session = FakeSession({'code': 0, 'data': {'matches': {'1': [item]}}})
    a = asyncio.run(fetch_data.scrape_wencai_jczq_async(session, '2099-09-19'))[0]
    b = asyncio.run(fetch_data.scrape_wencai_jczq_async(session, '2099-09-18'))[0]
    assert a['stime'] == item['stime']
    assert datetime.fromisoformat(a['kickoff_at']) == kickoff
    assert datetime.fromisoformat(a['captured_at']).tzinfo
    assert a['match_id'] == b['match_id'] == 'wencai:123'


def test_fallback_has_no_fabricated_statistics():
    stats = fetch_data.generate_stats_from_context(
        {'home_rank': 1, 'sp_home': 1.2, 'sp_away': 9}, 'home')
    assert stats['data_available'] is False
    assert stats['quality'] == 'unavailable'
    assert not any(k in stats for k in ('played', 'wins', 'form', 'goals_for'))


def test_stime_unknown_overflow_and_milliseconds_are_not_guessed():
    for value in (None, 'garbage', 0, True, 9999999999999999, '2026-09-19 19:00'):
        assert fetch_data._business_day_from_stime(value) is None


def test_started_fixture_is_not_returned(monkeypatch):
    monkeypatch.setenv('WENCAI_AUTHORIZATION', 'test-auth')
    item = {'home': 'A', 'guest': 'B', 'cup': '英超',
            'stime': int(datetime(2020, 1, 1, 13, tzinfo=timezone.utc).timestamp())}
    session = FakeSession({'data': {'matches': {'1': [item]}}})
    assert asyncio.run(fetch_data.scrape_wencai_jczq_async(session, '2020-01-01')) == []


def test_enrichment_resolves_fixture_before_season_stats(monkeypatch):
    calls = []
    fixture = {'fixture': {'id': 55, 'date': '2099-09-19T13:00:00Z'},
               'league': {'id': 39, 'season': 2099, 'round': 'Regular Season - 4'},
               'teams': {'home': {'id': 42, 'name': 'Arsenal'},
                         'away': {'id': 49, 'name': 'Chelsea'}}}

    async def api(session, endpoint, params, sema):
        calls.append((endpoint, params))
        if endpoint == '/fixtures' and 'date' in params:
            return [fixture]
        return []

    monkeypatch.setattr(fetch_data, 'API_FOOTBALL_KEY', 'test')
    monkeypatch.setattr(fetch_data, 'async_fetch_api', api)
    match = {'home_team': '阿森纳', 'away_team': '切尔西', 'league': '英超',
             'kickoff_at': '2099-09-19T13:00:00Z', 'match_id': 'wencai:123'}
    result = asyncio.run(fetch_data.enrich_match_data(None, match, 0, '2099-09-19', None))
    assert result['match_id'] == 'wencai:123'
    assert result['api_football_fixture_id'] == 55
    assert result['season'] == 2099
    assert result['league_id'] == 39
    stats_calls = [params for endpoint, params in calls if endpoint == '/teams/statistics']
    assert len(stats_calls) == 2
    assert all(p['season'] == 2099 and p['league'] == 39 for p in stats_calls)
    assert '/teams' not in [e for e, p in calls]
    assert result['league_evidence']['fixture']['source'] == 'api_football'


def test_enrichment_refuses_wrong_opponent_and_does_not_fetch_stats(monkeypatch):
    calls = []

    async def api(session, endpoint, params, sema):
        calls.append(endpoint)
        return [{'fixture': {'id': 99, 'date': '2099-09-19T13:00:00Z'},
                 'league': {'id': 39, 'season': 2099},
                 'teams': {'home': {'id': 42, 'name': 'Arsenal'},
                           'away': {'id': 50, 'name': 'Manchester City'}}}]

    monkeypatch.setattr(fetch_data, 'API_FOOTBALL_KEY', 'test')
    monkeypatch.setattr(fetch_data, 'async_fetch_api', api)
    match = {'home_team': 'Arsenal', 'away_team': 'Manchester United',
             'league': '英超', 'kickoff_at': '2099-09-19T13:00:00Z'}
    result = asyncio.run(fetch_data.enrich_match_data(None, match, 0, '2099-09-19', None))
    assert result['home_stats']['data_available'] is False
    assert result.get('league_id') is None
    assert calls == ['/fixtures']


def test_real_stats_are_scoped_and_incomplete_stats_are_unavailable():
    raw = {'team': {'id': 42}, 'league': {'id': 39, 'season': 2026},
           'fixtures': {k: {'total': v} for k, v in
                        [('played', 4), ('wins', 2), ('draws', 1), ('loses', 1)]},
           'goals': {'for': {'total': {'total': 7}}, 'against': {'total': {'total': 4}}}}
    observed = fetch_data._api_stats(raw, 42, 39, 2026, {'source': 'test'})
    assert observed['data_available'] is True and observed['estimated'] is False
    assert observed['played'] == 4
    assert fetch_data._api_stats(raw, 42, 39, 2024, {})['data_available'] is False
    raw['fixtures']['draws'].pop('total')
    assert fetch_data._api_stats(raw, 42, 39, 2026, {})['data_available'] is False
