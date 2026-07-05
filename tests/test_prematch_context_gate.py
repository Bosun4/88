import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import predict
from tests.test_prematch_factor_gate import _base_ai_row


def test_weak_home_low_score_in_high_draw_league_is_capped_to_c():
    row = _base_ai_row(
        final_direction="home",
        predicted_score="1-0",
        direction_probs={"home": 46, "draw": 29, "away": 25},
        top3=[{"score": "1-0", "prob": 18}, {"score": "1-1", "prob": 16}],
        recommendation={"tier": "B", "is_recommended": True, "bet_confidence": 66, "risk_level": "medium", "risk_tags": []},
        money_flow={"sharp_money_direction": "home", "reverse_line_movement": False},
        web_research={"used": False, "sources": []},
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "瑞超", "s11": 6.8, "sp_home": 2.05})

    assert front["predicted_score"] == "1-0"
    assert front["final_direction"] == "home"
    assert front["recommend_gate_pass"] is False
    assert front["recommendation"]["tier"] == "D"
    assert "prematch_v2_weak_home_win_needs_confirmation" in front["pre_match_factor_audit"]["rules_applied"]


def test_cup_cross_context_favorite_requires_lineup_and_motivation_confirmation():
    row = _base_ai_row(
        final_direction="home",
        predicted_score="2-0",
        direction_probs={"home": 58, "draw": 24, "away": 18},
        top3=[{"score": "2-0", "prob": 17}, {"score": "1-0", "prob": 14}],
        recommendation={"tier": "B", "is_recommended": True, "bet_confidence": 68, "risk_level": "medium", "risk_tags": []},
        money_flow={"sharp_money_direction": "home", "reverse_line_movement": False},
        web_research={"used": False, "sources": []},
    )
    match = {"league": "亚冠乙", "sp_home": 1.55, "s11": 9.0, "information": {}, "intelligence": {}}
    front = predict.adapt_ai_to_frontend(row, match)

    assert front["predicted_score"] == "2-0"
    assert front["final_direction"] == "home"
    assert front["recommend_gate_pass"] is False
    assert front["recommendation"]["tier"] == "C"
    assert "external_fact_without_source" in front["validation_warnings"]
    assert "missing_external_confirmation" in front["validation_warnings"]
    assert "prematch_v2_cup_cross_context_lineup_motivation_required" in front["pre_match_factor_audit"]["rules_applied"]
    assert "prematch_v2_cross_region_requires_external_confirmation" in front["pre_match_factor_audit"]["rules_applied"]


def test_cup_favorite_with_web_lineup_and_motivation_can_pass():
    row = _base_ai_row(
        final_direction="home",
        predicted_score="2-0",
        direction_probs={"home": 62, "draw": 21, "away": 17},
        top3=[{"score": "2-0", "prob": 19}, {"score": "3-0", "prob": 12}],
        recommendation={"tier": "B", "is_recommended": True, "bet_confidence": 70, "risk_level": "medium", "risk_tags": []},
        money_flow={"sharp_money_direction": "home", "reverse_line_movement": False},
        web_research={"used": True, "sources": [{"title": "lineup", "url": "https://example.com", "claim": "官方首发与战意确认"}]},
        reason="晋级战意明确，官方首发确认。",
    )
    match = {"league": "亚冠乙", "sp_home": 1.55, "s11": 10.5, "information": {"official_lineup": True}, "intelligence": {"note": "必须晋级"}}
    front = predict.adapt_ai_to_frontend(row, match)

    assert front["predicted_score"] == "2-0"
    assert front["final_direction"] == "home"
    assert front["recommend_gate_pass"] is True
    assert "prematch_v2_cup_cross_context_lineup_motivation_required" not in front.get("pre_match_factor_audit", {}).get("rules_applied", [])


def test_worldcup_defaults_to_knockout_gate_even_without_round_label():
    row = _base_ai_row(
        final_direction="home",
        predicted_score="3-0",
        direction_probs={"home": 70, "draw": 18, "away": 12},
        top3=[{"score": "3-0", "prob": 18}, {"score": "1-1", "prob": 12}],
        recommendation={"tier": "A", "is_recommended": True, "bet_confidence": 78, "risk_level": "medium", "risk_tags": []},
        web_research={"used": False, "sources": []},
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.45, "sp_draw": 4.2, "sp_away": 7.8})
    flags = front["pre_match_factor_audit"]["match_context_flags"]

    assert flags["worldcup_knockout"] is True
    assert flags["worldcup_r3"] is False
    assert "prematch_v2_worldcup_ko_blowout_requires_confirmation" in front["pre_match_factor_audit"]["rules_applied"]
    assert not any("worldcup_r3" in r for r in front["pre_match_factor_audit"]["rules_applied"])


def test_worldcup_knockout_text_suppresses_group_r3_gate():
    row = _base_ai_row(
        final_direction="away",
        predicted_score="0-2",
        direction_probs={"home": 12, "draw": 20, "away": 68},
        top3=[{"score": "0-2", "prob": 18}, {"score": "1-1", "prob": 12}],
        recommendation={"tier": "A", "is_recommended": True, "bet_confidence": 76, "risk_level": "medium", "risk_tags": []},
        web_research={"used": False, "sources": []},
        reason="淘汰赛阶段，文本里即使提到旧小组赛第三轮也不得触发R3。",
    )
    match = {"league": "世界杯", "baseface": "1/8决赛", "intelligence": {"note": "旧报告提到小组赛第三轮已出线"}}
    front = predict.adapt_ai_to_frontend(row, match)
    flags = front["pre_match_factor_audit"]["match_context_flags"]

    assert flags["worldcup_knockout"] is True
    assert flags["worldcup_r3"] is False
    assert not any("worldcup_r3" in r for r in front["pre_match_factor_audit"]["rules_applied"])


def test_explicit_worldcup_group_r3_can_still_trigger_legacy_exception():
    row = _base_ai_row(
        final_direction="away",
        predicted_score="0-3",
        direction_probs={"home": 10, "draw": 18, "away": 72},
        top3=[{"score": "0-3", "prob": 18}, {"score": "1-1", "prob": 12}],
        recommendation={"tier": "A", "is_recommended": True, "bet_confidence": 76, "risk_level": "medium", "risk_tags": []},
        web_research={"used": False, "sources": []},
        reason="小组赛第三轮，热门已出线但仍被判断大胜。",
    )
    match = {"league": "世界杯", "baseface": "小组赛第三轮", "intelligence": {"note": "热门已出线轮换"}, "sp_away": 1.40, "sp_home": 8.0}
    front = predict.adapt_ai_to_frontend(row, match)
    flags = front["pre_match_factor_audit"]["match_context_flags"]

    assert flags["worldcup_knockout"] is False
    assert flags["worldcup_r3"] is True
    assert any("worldcup_r3" in r for r in front["pre_match_factor_audit"]["rules_applied"])
