import sys
sys.path.insert(0, 'scripts')
from pipeline_modes import forbid_backtest_context_in_prediction, sanitize_for_prediction


def test_leakage_guard_flags_actual_result(monkeypatch):
    monkeypatch.setenv('VMAX_PIPELINE_MODE', 'predict')
    warnings = forbid_backtest_context_in_prediction({'home_team':'A','actual_score':'2-1','nested':{'roi':1.2}})
    assert 'forbidden_live_prediction_field:actual_score' in warnings
    assert 'forbidden_live_prediction_field:roi' in warnings


def test_sanitize_for_prediction_removes_result_fields():
    clean = sanitize_for_prediction({'home_team':'A','actual_score':'2-1','prediction_snapshot':{'x':1}})
    assert 'actual_score' not in clean
    assert 'prediction_snapshot' not in clean
    assert clean['_sanitized_for_prediction'] is True
