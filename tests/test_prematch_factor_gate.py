import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import predict


def _base_ai_row(**overrides):
    row = {
        "match": 1,
        "final_direction": "away",
        "predicted_score": "1-2",
        "direction_probs": {"home": 31, "draw": 27, "away": 42},
        "goal_band": "3",
        "btts": "yes",
        "top3": [{"score": "1-2", "prob": 17}, {"score": "1-1", "prob": 15}],
        "anchor_audit": {},
        "recommendation": {
            "tier": "A",
            "is_recommended": True,
            "top4_priority": 1,
            "bet_confidence": 74,
            "risk_level": "low",
            "risk_tags": [],
        },
        "recommendation_components": {
            "direction_edge": 76,
            "score_cluster_strength": 74,
            "goal_band_strength": 70,
            "btts_alignment": 68,
            "sharp_alignment": 66,
            "web_source_quality": 0,
            "market_conflict_penalty": 0,
        },
        "money_flow": {
            "public_money_direction": "unclear",
            "sharp_money_direction": "unclear",
            "sharp_confidence": 0,
            "reverse_line_movement": False,
        },
        "web_research": {"used": False, "sources": []},
        "reason": "赛前判断包含战意、体能、伤停，但无真实联网确认。",
    }
    row.update(overrides)
    return row


def test_away_win_without_web_or_sharp_is_no_bet_but_keeps_score_and_direction():
    row = _base_ai_row()
    front = predict.adapt_ai_to_frontend(row, {"league": "美职", "home_team": "A", "away_team": "B", "s11": 7.0})

    assert front["predicted_score"] == "1-2"
    assert front["final_direction"] == "away"
    assert front["recommend_gate_pass"] is False
    assert front["recommendation"]["tier"] == "D"
    assert "prematch_v2_away_win_without_external_market_confirmation" in front["recommendation_downgrade_reasons"]
    assert front["pre_match_factor_audit"]["league_dna"]["key"] == "美职"


def test_draw_defense_caps_non_draw_weak_edge_with_draw_cluster():
    row = _base_ai_row(
        final_direction="home",
        predicted_score="2-1",
        direction_probs={"home": 45, "draw": 29, "away": 26},
        recommendation={"tier": "A", "is_recommended": True, "bet_confidence": 72, "risk_level": "low", "risk_tags": []},
        money_flow={"sharp_money_direction": "home", "reverse_line_movement": False},
        web_research={"used": True, "sources": [{"title": "x", "url": "https://example.com", "claim": "ok"}]},
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "瑞超", "s11": 6.6, "information": {"official_lineup": True}})

    assert front["predicted_score"] == "2-1"
    assert front["final_direction"] == "home"
    assert front["recommend_gate_pass"] is False
    assert "prematch_v2_draw_defense_gate" in front["recommendation_downgrade_reasons"]
    assert "prematch_v2_strong_draw_cluster_no_bet" in front["recommendation_downgrade_reasons"]


def test_clean_confirmed_case_keeps_recommendation():
    row = _base_ai_row(
        final_direction="home",
        predicted_score="2-0",
        direction_probs={"home": 62, "draw": 21, "away": 17},
        top3=[{"score": "2-0", "prob": 19}, {"score": "3-0", "prob": 11}],
        recommendation={"tier": "A", "is_recommended": True, "bet_confidence": 76, "risk_level": "low", "risk_tags": []},
        money_flow={"sharp_money_direction": "home", "reverse_line_movement": False},
        web_research={"used": True, "sources": [{"title": "x", "url": "https://example.com", "claim": "ok"}]},
    )
    match = {"league": "葡超", "s11": 9.5, "s22": 18.0, "information": {"official_lineup": True}}
    front = predict.adapt_ai_to_frontend(row, match)

    assert front["predicted_score"] == "2-0"
    assert front["final_direction"] == "home"
    assert front["recommend_gate_pass"] is True
    assert front["recommendation"]["tier"] == "A"
    assert front["pre_match_factor_audit"]["data_quality_score"] >= 70
