import scripts.predict as predict
from scripts import league_intel


def test_package_import_build_evidence_keeps_enhanced_prompt_facts():
    match = {
        "home_team": "沙特",
        "away_team": "乌拉圭",
        "league": "世界杯",
        "match_num": "周一015",
        "sp_home": 8.45,
        "sp_draw": 4.35,
        "sp_away": 1.28,
        "give_ball": 1.0,
        "vote": {"win": "11", "same": "20", "lose": "69"},
        "change": {},
        "a0": 11.0,
        "a1": 4.5,
        "a2": 3.35,
        "a3": 3.55,
        "a4": 5.65,
        "a5": 10.5,
        "a6": 19.0,
        "a7": 28.0,
        "l01": 5.4,
        "l02": 5.3,
        "l12": 7.0,
        "l03": 7.5,
        "l13": 11.0,
        "l23": 35.0,
    }

    evidence = predict.build_evidence_packet(match, 1)

    assert "v206_shadow_pre_inject_error" not in evidence.get("data_quality", {})
    assert evidence["score_cluster_diagnostics_v203"]["available"] is True
    assert "market_microstructure_v203" in evidence
    assert "local_quantitative_intelligence" in evidence
    assert evidence["local_quantitative_intelligence"]["world_cup_round_gate"]["round"] == "unknown"
    assert "dual_market_divergence_calibration" in evidence


def test_world_cup_round_gate_distinguishes_round_three_margin_compression():
    match = {
        "home_team": "法国",
        "away_team": "突尼斯",
        "league": "世界杯小组赛",
        "baseface": "本场为2026世界杯小组赛第三轮，法国已提前出线，突尼斯必须取胜才有晋级希望。",
    }

    gate = league_intel.world_cup_round_gate(match)

    assert gate["round"] == "R3"
    assert gate["round_policy"] == "qualification_state_gate"
    assert "控分机械理解为小球" in gate["must_not_assume"][1]
    assert any("已出线强队vs有动机方" in item for item in gate["confidence_controls"])
