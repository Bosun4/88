"""League evidence is sourced context, not a rank-based motivation prediction."""
import importlib
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def build(match):
    module = importlib.import_module('league_context')
    return module.build_league_context(match)


def test_missing_evidence_is_explicitly_unknown():
    result = build({'league': '英超', 'home_rank': 1, 'away_rank': 18})
    assert result['stage']['value'] == 'unknown'
    assert result['home']['points']['value'] is None
    assert result['home']['motivation']['value'] == 'unknown'
    assert result['home']['rotation']['value'] == 'unknown'
    json.dumps(result, allow_nan=False)


def source(**extra):
    return {'source': 'fixture_test', 'source_url': 'https://example.test/evidence',
            'captured_at': '2026-09-19T10:00:00Z', 'league_id': 39,
            'season': 2026, **extra}


def sample():
    rows = [{'team': {'id': team}, 'rank': rank, 'points': points,
             'all': {'played': 34}, 'description': desc,
             'update': '2026-09-19T09:00:00Z'}
            for team, rank, points, desc in [
                (1, 1, 80, 'Promotion - Champions League (League phase)'),
                (2, 2, 73, 'Promotion - Champions League (League phase)'),
                (3, 3, 60, 'Promotion - Europa League (League phase)'),
                (4, 4, 40, None), (5, 18, 35, 'Relegation')]]
    rows += [{'team': {'id': 100 + rank}, 'rank': rank, 'points': 36,
              'all': {'played': 34}, 'description': None}
             for rank in range(5, 18)]
    rows += [{'team': {'id': 100 + rank}, 'rank': rank, 'points': 25,
              'all': {'played': 34}, 'description': 'Relegation'}
             for rank in (19, 20)]
    return {'league': '英超', 'league_id': 39, 'season': 2026,
            'home_id': 2, 'away_id': 4, 'kickoff_at': '2026-09-19T19:00:00Z',
            'api_football_fixture_id': 55,
            'league_evidence': {
                'fixture': source(fixture_id=55, home_id=2, away_id=4,
                                  kickoff_at='2026-09-19T19:00:00Z', round='Regular Season - 35'),
                'rounds': source(rounds=[f'Regular Season - {n}' for n in range(1, 39)]),
                'standings': source(groups=[rows]),
            }}


def test_points_gaps_remaining_and_stage_use_season_evidence():
    result = build(sample())
    assert result['stage']['value'] == 'late'
    assert result['round']['value'] == 35
    assert result['home']['points']['value'] == 73
    assert result['home']['remaining_matches']['value'] == 4
    assert result['home']['title_gap']['value'] == 7
    assert result['away']['europe_gap']['value'] == 20
    assert result['away']['relegation_cushion']['value'] == 5
    assert result['away']['motivation']['value'] == 'unknown'
    assert result['home']['title_gap']['sources']
    json.dumps(result, allow_nan=False)


def scheduled(identifier, when, status='NS', team_id=2):
    return {'fixture': {'id': identifier, 'date': when, 'status': {'short': status}},
            'teams': {'home': {'id': team_id}, 'away': {'id': 99}},
            'league': {'id': 2, 'name': 'Cup', 'season': 2026}}


def test_schedule_gaps_and_rotation_require_sourced_team_evidence():
    match = sample()
    match['league_evidence']['home_schedule'] = source(team_id=2, fixtures=[
        scheduled(20, '2026-09-16T19:00:00Z', 'FT'),
        scheduled(21, '2026-09-22T19:00:00Z'),
        scheduled(22, '2026-09-20T19:00:00Z', 'PST'),
        scheduled(23, '2026-09-18T19:00:00Z', 'FT', team_id=123),
    ])
    match['league_evidence']['home_rotation'] = source(team_id=2, statement='Coach confirms two changes')
    result = build(match)
    assert result['home']['rest_days']['value'] == 3
    assert result['home']['next_match_in_days']['value'] == 3
    assert result['home']['fixtures_within_7_days']['value'] == 2
    assert result['home']['rotation']['value'] == 'Coach confirms two changes'
    assert result['away']['rest_days']['value'] is None


@pytest.mark.parametrize('bad', ['old_season', 'postkickoff', 'missing_source', 'duplicate', 'stale'])
def test_unverified_standings_never_become_current_facts(bad):
    match = sample()
    table = match['league_evidence']['standings']
    if bad == 'old_season':
        table['season'] = 2024
    elif bad == 'postkickoff':
        table['captured_at'] = '2026-09-19T20:00:00Z'
    elif bad == 'missing_source':
        table.pop('source')
    elif bad == 'duplicate':
        table['groups'].append(table['groups'][0])
    elif bad == 'stale':
        table['captured_at'] = '2026-09-01T10:00:00Z'
    assert build(match)['home']['points']['value'] is None


def test_unknown_worldcup_is_not_current_knockout_or_team_form():
    import league_intel
    text, _, knockout, _ = league_intel.build_league_intelligence(
        {'league': '世界杯', 'home_team': '法国', 'away_team': '巴西'})
    assert not knockout
    assert 'WC2026' not in text and 'WC-FORM' not in text
    assert 'unknown' in text
    motivation = '\n'.join(league_intel.analyze_motivation(
        {'league': '英超', 'home_rank': 1, 'away_rank': 18}, 'eng_top'))
    assert 'DESPERATE' not in motivation and 'TITLE' not in motivation


def test_both_import_paths():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    package = importlib.import_module('scripts.league_context')
    assert package.build_league_context(sample()) == build(sample())


@pytest.mark.parametrize('captured', ['2026-09-19T20:00:00Z', '2026-09-01T10:00:00Z',
                                    '2026-09-19T10:00:00', None])
def test_rotation_rejects_nonprematch_stale_or_naive_evidence(captured):
    match = sample()
    match['league_evidence']['home_rotation'] = source(
        captured_at=captured, team_id=2, statement='Coach confirms changes')
    assert build(match)['home']['rotation']['value'] == 'unknown'


def test_split_round_schedule_does_not_invent_remaining_matches():
    match = sample()
    match['league_evidence']['rounds']['rounds'].append('Championship Round - 1')
    assert build(match)['home']['remaining_matches']['value'] is None
    assert build(match)['stage']['value'] == 'unknown'
