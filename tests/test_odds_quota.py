import sys
sys.path.insert(0, 'scripts')
import odds_quota


def test_odds_quota_daily_limit(tmp_path, monkeypatch):
    p = tmp_path / 'quota.json'
    monkeypatch.setattr(odds_quota, 'DAILY_BUDGET', 3)
    monkeypatch.setattr(odds_quota, 'MONTHLY_LIMIT', 100)
    for _ in range(3):
        ok, reason, _ = odds_quota.can_spend(1, path=str(p))
        assert ok, reason
        odds_quota.record_spend(1, path=str(p))
    ok, reason, _ = odds_quota.can_spend(1, path=str(p))
    assert not ok
    assert 'daily budget exceeded' in reason
