import sys
sys.path.insert(0, 'scripts')
from verify import find_prediction_history


def test_missing_history_returns_none():
    assert find_prediction_history('2099-01-01') is None
