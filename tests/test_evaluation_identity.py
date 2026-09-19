import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_identity_deduplicates_capture_days_but_not_event_dates():
    import fixture_identity as fi
    a = dict(match_id='42', kickoff_at='2026-09-20T19:00:00+08:00',
             home_team='A', away_team='B', league='L', date='2026-09-18')
    b = {**a, 'date': '2026-09-19'}
    c = {**a, 'kickoff_at': '2027-09-20T19:00:00+08:00'}
    assert fi.fixture_key(a) == fi.fixture_key(b)
    assert fi.fixture_key(a) != fi.fixture_key(c)
    assert len(fi.dedupe_predictions([a, b, c])) == 2
    assert fi.match_actual(a, [{**c, 'actual_score': '2-0'}]) is None
    assert fi.match_actual(a, [{**a, 'actual_score': '2-0'}])['actual_score'] == '2-0'
    assert fi.fixture_key({'id': 1, 'date': '2026-09-18', 'home_team': 'A', 'away_team': 'B'}) is None


def test_team_event_binding_allows_result_season_metadata_without_id():
    import fixture_identity as fi
    pred = {'league': 'L', 'event_date': '2026-09-20', 'home_team': 'A', 'away_team': 'B'}
    result = {**pred, 'season': '2627', 'actual_score': '1-0'}
    assert fi.match_actual(pred, [result]) is result
    assert fi.match_actual({**pred, 'season': '2526'}, [result]) is None
    assert fi.match_actual(pred, [{**result, 'home_team': 'B', 'away_team': 'A'}]) is None


def test_cross_provider_ids_timezone_and_localized_labels():
    import fixture_identity as fi
    pred = {'match_id': 'wencai:123', 'source': 'wencai', 'api_football_fixture_id': 55,
            'kickoff_at': '2026-09-21T01:00:00+08:00', 'event_date': '2026-09-21',
            'league': '英超', 'league_id': 39, 'season': 2026,
            'home_id': 42, 'away_id': 49, 'home_team': '阿森纳', 'away_team': '切尔西'}
    actual = fi.api_actual({'fixture': {'id': 55, 'date': '2026-09-20T17:00:00Z', 'status': {'short': 'FT'}},
                           'league': {'id': 39, 'name': 'Premier League', 'season': 2026},
                           'teams': {'home': {'id': 42, 'name': 'Arsenal'}, 'away': {'id': 49, 'name': 'Chelsea'}},
                           'goals': {'home': 2, 'away': 0}})
    assert fi.match_actual(pred, [actual]) is actual
    assert fi.match_actual({**pred, 'api_football_fixture_id': 99}, [actual]) is None
    assert fi.match_actual({**pred, 'home_id': 49, 'away_id': 42}, [actual]) is None
    before_enrich = {k: v for k, v in pred.items() if k != 'api_football_fixture_id'}
    assert fi.fixture_key(before_enrich) == fi.fixture_key(pred)
    unrelated = {**actual, 'source': 'other', 'api_football_fixture_id': None, 'fixture_id': 55}
    assert fi.match_actual({'match_id': 'wencai:55', 'kickoff_at': pred['kickoff_at']}, [unrelated]) is None
