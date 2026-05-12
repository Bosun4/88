import sys
sys.path.insert(0, 'scripts')
from validators import parse_score, result_from_score, validate_prediction


def test_parse_score_variants():
    assert parse_score('2-1') == (2, 1)
    assert parse_score('2:1') == (2, 1)
    assert parse_score('2：1') == (2, 1)
    assert parse_score('bad') is None


def test_result_from_score():
    assert result_from_score('2-1') == '主胜'
    assert result_from_score('1-1') == '平局'
    assert result_from_score('0-1') == '客胜'


def test_validate_prediction_conflicts():
    warnings = validate_prediction({'predicted_score': '3-2', 'result': '平局', 'over_under_2_5': '小'}, {'data_quality_score': 90})
    assert any('score_result_conflict' in w for w in warnings)
    assert any('score_total_conflict' in w for w in warnings)
