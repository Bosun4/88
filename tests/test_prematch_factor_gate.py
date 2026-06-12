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
    assert front["recommend_gate_pass"] is True
    assert front["recommendation"]["tier"] == "A"
    assert front["selection_layer"] in {"观察", "防平"}
    assert "prematch_v2_away_win_without_external_market_confirmation" in front["pre_match_factor_audit"]["rules_applied"]
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
    assert "prematch_v2_draw_defense_gate" in front["pre_match_factor_audit"]["rules_applied"]
    assert "prematch_v2_high_draw_league_non_draw_cap" in front["pre_match_factor_audit"]["rules_applied"]


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


def test_league_dna_profiles_cover_priority_leagues():
    for league in ["世界杯", "英超", "西甲", "意甲", "荷甲", "MLS", "沙特"]:
        front = predict.adapt_ai_to_frontend(_base_ai_row(), {"league": league, "s11": 8.0})
        assert front["pre_match_factor_audit"]["league_dna"]["key"] == league


def test_frontend_contract_aliases_are_stable():
    row = _base_ai_row(
        final_direction="home",
        predicted_score="2-1",
        goal_band="3",
        btts="yes",
        recommendation={"tier": "B", "is_recommended": True, "bet_confidence": 72, "risk_level": "medium", "risk_tags": []},
        reason="AI读盘理由",
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "葡超", "s11": 8.0})

    assert front["btts_ai"] == "yes"
    assert front["ai_btts"] == "yes"
    assert front["over_under_2_5"] == "大"
    assert front["ai_over25"] == "大"
    assert front["ai_score_reason"] == "AI读盘理由"
    assert front["ai_confidence"] == 72


def test_timeline_rerank_without_real_timeline_is_no_bet_but_keeps_score():
    row = _base_ai_row(
        final_direction="home",
        predicted_score="2-1",
        recommendation={"tier": "A", "is_recommended": True, "bet_confidence": 78, "risk_level": "low", "risk_tags": []},
        market_timeline_audit={
            "available": False,
            "timeline_unavailable": True,
            "rerank_applied": True,
            "missing_timeline_points": ["T-60m", "T-30m"],
            "evidence_summary": "无真实时间序列，只是静态赔率推断",
        },
    )

    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "s11": 5.0})

    assert front["predicted_score"] == "2-1"
    assert front["final_direction"] == "home"
    assert front["recommendation"]["tier"] == "D"
    assert front["recommendation"]["is_recommended"] is False
    assert front["recommend_gate_pass"] is False
    assert "timeline_rerank_without_real_timeline" in front["recommendation_downgrade_reasons"]
    assert "market_timeline_audit" in front
