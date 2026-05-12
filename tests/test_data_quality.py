import sys
sys.path.insert(0, 'scripts')
from data_quality import assess_match_quality, summarize_quality


def test_quality_low_when_missing_odds_api_and_ids():
    m = {'home_team':'A','away_team':'B','league':'L','sp_home':2.0,'sp_draw':3.0,'sp_away':4.0}
    q = assess_match_quality(m)
    assert q['data_quality_score'] < 100
    assert 'the_odds_api_missing_or_budget_skipped' in q['data_warnings']


def test_summarize_quality():
    matches=[{'data_quality_score':80,'data_quality_level':'high'}, {'data_quality_score':50,'data_quality_level':'low'}]
    s=summarize_quality(matches)
    assert s['avg_score'] == 65
    assert s['low_quality_matches'] == 1
