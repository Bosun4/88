from market_sentinel.fair_probs import fair_probs_from_1x2
from market_sentinel.divergence import euro_asian_divergence_index, classify_divergence
from market_sentinel.steam import detect_late_steam

def test_fair_probs():
    odds = {"h": 2.0, "d": 3.4, "a": 3.8}
    probs = fair_probs_from_1x2(odds, method="power")
    total = probs["h"] + probs["d"] + probs["a"]
    assert abs(total - 1.0) < 0.01

def test_divergence():
    di, di_abs = euro_asian_divergence_index(0.65, -0.5, 2.0, 1.9)
    sev = classify_divergence(di_abs)
    assert sev in ["watch", "warning", "critical"]

def test_late_steam():
    res = detect_late_steam(0.55, 0.44, 20000, 5000, 5)
    flags = res["late_steam_flags"]
    assert "steam_critical" in flags
    assert "sharp_confirmed" in flags
    
    res2 = detect_late_steam(0.50, 0.49, 1000, 500, 5)
    assert not res2["late_steam_flags"]


def test_power_fair_probs_precisely_sum_to_one():
    probs = fair_probs_from_1x2({"h": 1.83, "d": 3.75, "a": 4.20}, method="power")
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_late_steam_zero_move_direction_none():
    res = detect_late_steam(0.50, 0.50, 1000, 1000, 5)
    assert res["late_steam_flags"] == []
    assert res["steam_direction"] == "none"
    assert "towards none" in res["steam_evidence"]
