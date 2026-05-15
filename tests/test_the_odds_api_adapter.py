import os
import tempfile
from market_sentinel.adapters.the_odds_api import (
    get_mock_events,
    parse_the_odds_api_events_to_snapshots,
    run_the_odds_api_adapter
)

def test_mock_events_parsing():
    events = get_mock_events()
    snapshots = parse_the_odds_api_events_to_snapshots(events)
    
    assert len(snapshots) > 0
    
    h2h_snaps = [s for s in snapshots if s.market_type == "1x2"]
    assert len(h2h_snaps) == 3 # home, draw, away
    assert any(s.selection == "home" for s in h2h_snaps)
    assert any(s.implied_prob_fair > 0 for s in h2h_snaps)
    
    spreads_snaps = [s for s in snapshots if s.market_type == "asian_handicap"]
    assert len(spreads_snaps) == 2
    assert any(s.line == -0.5 for s in spreads_snaps)

def test_mock_cli():
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "snapshots/test.jsonl")
        run_the_odds_api_adapter("soccer", "uk", "h2h,spreads", out_path, mock=True)
        
        assert os.path.exists(out_path)
        alert_json = os.path.join(tmp, "alerts/test_alerts.json")
        alert_md = os.path.join(tmp, "alerts/test_alerts.md")
        assert os.path.exists(alert_json)
        assert os.path.exists(alert_md)
