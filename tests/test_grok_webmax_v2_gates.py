# -*- coding: utf-8 -*-
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import predict


def _row(**overrides):
    row = {
        "final_direction": "home",
        "predicted_score": "2-0",
        "direction_probs": {"home": 62, "draw": 22, "away": 16},
        "top3": [{"score": "2-0", "prob": 18}, {"score": "1-0", "prob": 14}],
        "anchor_audit": {},
        "score_cluster_audit": {},
        "sharp_money_audit": {},
        "recommendation_components": {
            "direction_edge": 80,
            "score_cluster_strength": 75,
            "goal_band_strength": 72,
            "btts_alignment": 70,
            "sharp_alignment": 60,
            "web_source_quality": 80,
            "market_conflict_penalty": 0,
        },
        "web_research": {"used": True, "sources": [{"title": "bad", "url": "#", "claim": "阵容完整"}]},
        "external_fact_table": [],
        "evidence_quality_score": 85,
        "recommendation": {"tier": "A", "is_recommended": True, "bet_action": "main", "bet_confidence": 78},
        "reason": "官方首发与战意明确，适合主推。",
    }
    row.update(overrides)
    return row


def test_evidence_quality_is_capped_when_url_is_invalid():
    front = predict.adapt_ai_to_frontend(_row(), {"league": "世界杯", "sp_home": 1.45, "s11": 7.2})
    assert front["evidence_quality_score"] <= 40
    assert front["recommend_gate_pass"] is False
    assert "external_source_url_missing_or_invalid" in front["validation_warnings"]


def test_no_valid_source_caps_evidence_quality_even_if_model_claims_high_score():
    row = _row(web_research={"used": False, "sources": []}, evidence_quality_score=92)
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.45, "s11": 7.2})
    assert front["evidence_quality_score"] <= 30
    assert front["recommendation"]["tier"] == "C"
    assert front["recommend_gate_pass"] is False


def test_valid_sources_preserve_high_evidence_quality():
    row = _row(
        web_research={"used": True, "sources": [{"title": "official", "url": "https://example.com/team-news", "claim": "官方首发确认"}]},
        external_fact_table=[{"claim": "官方首发确认", "source_url": "https://example.com/team-news"}],
        evidence_quality_score=88,
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.45, "s11": 7.2})
    assert front["evidence_quality_score"] == 88
    assert "external_fact_without_source" not in front.get("validation_warnings", [])


def test_final_direction_not_top_probability_caps_recommendation():
    row = _row(
        final_direction="draw",
        predicted_score="2-2",
        direction_probs={"home": 40, "draw": 35, "away": 25},
        top3=[{"score": "2-2", "prob": 15}, {"score": "1-1", "prob": 20}],
        web_research={"used": True, "sources": [{"title": "heat", "url": "https://example.com/heat", "claim": "高温风险"}]},
        evidence_quality_score=75,
        recommendation={"tier": "B", "is_recommended": True, "bet_action": "small", "bet_confidence": 62},
        reason="主胜造热，但高温和大球曲线支持2-2。",
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.23, "s11": 8.0})
    assert front["predicted_score"] == "2-2"
    assert front["final_direction"] == "draw"
    assert front["recommend_gate_pass"] is False
    assert front["recommendation"]["tier"] == "C"
    assert "direction_probability_not_supporting_final_direction" in front["validation_warnings"]
    assert "top3_probability_order_conflict" in front["validation_warnings"]


def test_contrarian_steam_claim_without_valid_market_source_is_observe_only():
    row = _row(
        final_direction="draw",
        predicted_score="1-1",
        direction_probs={"home": 38, "draw": 36, "away": 26},
        web_research={"used": False, "sources": []},
        evidence_quality_score=80,
        recommendation={"tier": "B", "is_recommended": True, "bet_action": "small", "bet_confidence": 70},
        reason="主胜反向Steam造热，大热必死，必须反打下盘。",
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.30, "s11": 6.8})
    assert front["recommend_gate_pass"] is False
    assert front["recommendation"]["tier"] == "C"
    assert "contrarian_market_claim_without_valid_market_source" in front["validation_warnings"]


def test_contrarian_steam_claim_with_valid_market_source_can_remain_small():
    row = _row(
        final_direction="draw",
        predicted_score="1-1",
        direction_probs={"home": 36, "draw": 37, "away": 27},
        web_research={"used": True, "sources": [{"title": "odds", "url": "https://example.com/odds", "claim": "market snapshot shows home drift", "source_type": "market_snapshot"}]},
        external_fact_table=[{"claim": "home drift", "source_url": "https://example.com/odds", "category": "market_snapshot"}],
        evidence_quality_score=82,
        recommendation={"tier": "B", "is_recommended": True, "bet_action": "small", "bet_confidence": 58},
        reason="主胜反向Steam造热，但外部market_snapshot确认主胜漂移。",
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.80, "s11": 6.8})
    assert "contrarian_market_claim_without_valid_market_source" not in front.get("validation_warnings", [])


def test_gate_pass_observe_action_is_synchronized_to_not_recommended():
    row = _row(
        web_research={"used": True, "sources": [{"title": "ok", "url": "https://example.com", "claim": "ok"}]},
        evidence_quality_score=80,
        recommendation={"tier": "B", "is_recommended": True, "bet_action": "observe", "bet_confidence": 65},
        reason="盘口结构可看，但只观察。",
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.7, "s11": 7.0})
    assert front["recommend_gate_pass"] is False
    assert front["recommendation"]["is_recommended"] is False
    assert "gate_action_not_bettable" in front["recommend_gate_reasons"]


def test_select_top4_excludes_observe_even_when_ai_marks_recommended():
    rows = []
    for i, action in enumerate(["main", "small", "observe", "main", "small"], 1):
        rows.append({
            "home_team": f"H{i}",
            "away_team": f"A{i}",
            "prediction": {
                "predicted_score": "1-0",
                "recommend_gate_pass": action != "observe",
                "recommendation": {"tier": "B", "is_recommended": True, "bet_action": action, "bet_confidence": 70 - i},
            },
        })
    top = predict.select_top4(rows)
    assert all(x["prediction"]["recommendation"]["bet_action"] != "observe" for x in top)
