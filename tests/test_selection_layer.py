import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import predict
from tests.test_prematch_factor_gate import _base_ai_row


def test_clean_a_grade_becomes_main_pick():
    row = _base_ai_row(
        final_direction="home",
        predicted_score="2-0",
        direction_probs={"home": 64, "draw": 20, "away": 16},
        top3=[{"score": "2-0", "prob": 20}, {"score": "3-0", "prob": 12}],
        recommendation={"tier": "A", "is_recommended": True, "bet_confidence": 76, "risk_level": "low", "risk_tags": []},
        money_flow={"sharp_money_direction": "home", "reverse_line_movement": False},
        web_research={"used": True, "sources": [{"title": "x", "url": "https://example.com", "claim": "ok"}]},
    )
    match = {"league": "葡超", "s11": 10.0, "s22": 20.0, "information": {"official_lineup": True}, "change": {"sp_home": "down"}}
    front = predict.adapt_ai_to_frontend(row, match)

    assert front["recommend_gate_pass"] is True
    assert front["selection_layer"] == "主推"
    assert front["selection_stake_unit"] == 1.0


def test_weak_home_draw_guard_becomes_hedged_draw_layer():
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

    assert front["selection_layer"] == "防平"
    assert "平局" in front["selection_hedge_suggestions"]
    assert front["recommend_gate_pass"] is False


def test_away_without_confirmation_is_observation_not_silent():
    row = _base_ai_row()
    front = predict.adapt_ai_to_frontend(row, {"league": "韩职", "s11": 8.8})

    assert front["final_direction"] == "away"
    assert front["selection_layer"] in {"观察", "放弃"}
    assert front["selection_layer"] == "放弃" if front["recommendation_tier"] == "D" else "观察"
    assert front["selection_stake_unit"] == 0.0


def test_cup_context_becomes_observation_when_not_hard_d():
    row = _base_ai_row(
        final_direction="home",
        predicted_score="2-0",
        direction_probs={"home": 58, "draw": 24, "away": 18},
        top3=[{"score": "2-0", "prob": 17}, {"score": "1-0", "prob": 14}],
        recommendation={"tier": "B", "is_recommended": True, "bet_confidence": 68, "risk_level": "medium", "risk_tags": []},
        money_flow={"sharp_money_direction": "home", "reverse_line_movement": False},
        web_research={"used": False, "sources": []},
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "亚冠乙", "sp_home": 1.55, "s11": 9.0})

    assert front["selection_layer"] in {"观察", "防平"}
    assert any("首发" in x or "临场" in x for x in front.get("selection_hedge_suggestions", []) + front.get("selection_layer_reasons", []))



def test_structured_confirmed_away_can_recover_to_small_observation_stake():
    row = _base_ai_row(
        final_direction="away",
        predicted_score="0-1",
        direction_probs={"home": 30, "draw": 27, "away": 43},
        top3=[{"score": "0-1", "prob": 17}, {"score": "1-1", "prob": 12}],
        recommendation={"tier": "C", "is_recommended": False, "bet_confidence": 62, "risk_level": "medium", "risk_tags": []},
        money_flow={"sharp_money_direction": "away", "reverse_line_movement": False},
        web_research={"used": True, "sources": [
            {"title": "lineup", "url": "https://example.com/1", "claim": "官方首发确认"},
            {"title": "market", "url": "https://example.com/2", "claim": "盘口资金支持客队"},
        ]},
        reason="官方首发确认，战意明确，sharp资金支持客队。",
    )
    match = {"league": "韩职", "s11": 9.5, "information": {"official_lineup": True}, "intelligence": {"note": "必须抢分"}, "change": {"away": "down"}}
    front = predict.adapt_ai_to_frontend(row, match)

    assert front["final_direction"] == "away"
    assert front["selection_layer"] == "小注"
    assert front["selection_stake_unit"] == 0.25
    assert front["selection_confirmation_bonus"] >= 22


def test_structured_rotation_risk_blocks_without_lineup():
    row = _base_ai_row(
        final_direction="home",
        predicted_score="2-0",
        direction_probs={"home": 58, "draw": 24, "away": 18},
        recommendation={"tier": "A", "is_recommended": True, "bet_confidence": 74, "risk_level": "low", "risk_tags": []},
        money_flow={"sharp_money_direction": "home", "reverse_line_movement": False},
        web_research={"used": True, "sources": [{"title": "x", "url": "https://example.com", "claim": "ok"}]},
        reason="强队可能轮换，部分主力轮休。",
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "英超", "s11": 10.0, "intelligence": {"h_inj": "轮休 轮换"}})

    assert "prematch_v2_rotation_risk_requires_lineup" in front["recommendation_downgrade_reasons"]
    assert front["selection_layer"] in {"观察", "防平"}
