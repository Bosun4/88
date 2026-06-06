import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import predict
from tests.test_prematch_factor_gate import _base_ai_row


def test_friendly_home_clean_sheet_gets_btts_and_2_2_tail_protection():
    row = _base_ai_row(
        final_direction="home",
        predicted_score="1-0",
        direction_probs={"home": 54, "draw": 25, "away": 21},
        goal_band="0-1",
        btts="no",
        top3=[{"score": "1-0", "prob": 18}, {"score": "1-1", "prob": 14}],
        recommendation={"tier": "B", "is_recommended": True, "bet_confidence": 66, "risk_level": "medium", "risk_tags": []},
        money_flow={"sharp_money_direction": "home", "reverse_line_movement": False},
        web_research={"used": False, "sources": []},
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "国际赛", "sp_home": 1.58, "s11": 7.0, "s22": 13.0})

    assert front["predicted_score"] == "1-0"
    assert front["final_direction"] == "home"
    assert "prematch_v2_friendly_clean_sheet_trap_guard" in front["pre_match_factor_audit"]["rules_applied"]
    scores = {x["score"] for x in front["risk_score_candidates"] if isinstance(x, dict)}
    assert {"1-1", "2-1", "2-2"}.issubset(scores)
    assert "friendly_clean_sheet_fragility" in front["tail_risk_flags"]
    assert front["selection_layer"] in {"防平", "观察", "放弃"}
    assert any("2-2" in x for x in front.get("selection_hedge_suggestions", []) + front.get("selection_layer_reasons", []))


def test_friendly_away_clean_sheet_gets_home_goal_and_draw_tail_protection():
    row = _base_ai_row(
        final_direction="away",
        predicted_score="0-2",
        direction_probs={"home": 22, "draw": 24, "away": 54},
        goal_band="2",
        btts="no",
        top3=[{"score": "0-2", "prob": 17}, {"score": "1-2", "prob": 14}],
        recommendation={"tier": "B", "is_recommended": True, "bet_confidence": 65, "risk_level": "medium", "risk_tags": []},
        money_flow={"sharp_money_direction": "away", "reverse_line_movement": False},
        web_research={"used": False, "sources": []},
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "国际友谊", "sp_away": 1.48, "s11": 7.0, "s22": 15.0})

    assert front["predicted_score"] == "0-2"
    assert front["final_direction"] == "away"
    assert front["pre_match_factor_audit"]["league_dna"]["key"] == "friendly_context"
    assert "prematch_v2_friendly_clean_sheet_trap_guard" in front["pre_match_factor_audit"]["rules_applied"]
    scores = {x["score"] for x in front["risk_score_candidates"] if isinstance(x, dict)}
    assert {"1-1", "1-2", "2-2"}.issubset(scores)
    assert "friendly_btts_late_goal_risk" in front["tail_risk_flags"]


def test_friendly_b_tier_low_price_home_favorite_is_capped_without_lineup_confirmation():
    row = _base_ai_row(
        final_direction="home",
        predicted_score="2-1",
        direction_probs={"home": 61, "draw": 23, "away": 16},
        goal_band="3",
        btts="yes",
        top3=[{"score": "2-1", "prob": 16}, {"score": "1-1", "prob": 13}],
        recommendation={"tier": "B", "is_recommended": True, "bet_confidence": 66, "risk_level": "medium", "risk_tags": []},
        money_flow={"sharp_money_direction": "home", "reverse_line_movement": False},
        web_research={"used": False, "sources": []},
    )
    front = predict.adapt_ai_to_frontend(row, {
        "league": "国际赛",
        "sp_home": 1.17,
        "sp_draw": 5.56,
        "sp_away": 10.5,
        "vote": {"win": "76", "same": "16", "lose": "8"},
    })

    assert front["predicted_score"] == "2-1"
    assert front["final_direction"] == "home"
    assert "prematch_v2_friendly_favorite_overheat_cap" in front["pre_match_factor_audit"]["rules_applied"]
    scores = {x["score"] for x in front["risk_score_candidates"] if isinstance(x, dict)}
    assert {"1-1", "2-1", "2-2"}.issubset(scores)
    assert "friendly_favorite_overheat" in front["tail_risk_flags"]
    assert front["selection_layer"] == "防平"
    assert front["selection_stake_unit"] == 0.25
