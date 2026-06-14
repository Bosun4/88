# -*- coding: utf-8 -*-
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import predict


def test_grok_webmax_instruction_replaces_timeline_storytelling():
    instr = predict._web_research_instruction("grok")
    assert "Grok Web-Max" in instr
    assert "external_fact_table" in instr
    assert "source_conflict_audit" in instr
    assert "evidence_quality_score" in instr
    assert "minimum_evidence_needed" in instr
    assert "禁止编造盘口时间序列" in instr
    assert "当前赔率只能作为 market_snapshot" in instr
    assert "T-60m" in instr
    assert "T-30m" in instr
    assert "临场回补" in instr
    assert "资金持续流入" in instr


def test_final_referee_prompts_enforce_grok_webmax_source_quality():
    ev = [{"match": 1}]
    for prompt in [
        predict.build_gemini_final_prompt(ev, {"gpt": {}, "grok": {}}, {}),
        predict.build_fallback_referee_prompt(ev, {"gpt": {}, "grok": {}}, {}),
        predict.build_family_debate_referee_prompt(ev, {"gpt": {}, "grok": {}}, {}),
    ]:
        assert "Grok Web-Max" in prompt
        assert "external_fact_table" in prompt
        assert "source_conflict_audit" in prompt
        assert "evidence_quality_score" in prompt
        assert "不得升为 main" in prompt
        assert "禁止" in prompt and "时间序列" in prompt


def test_external_fact_fields_survive_normalize_and_frontend_adapt():
    obj = {
        "predictions": [
            {
                "match": 1,
                "final_direction": "home",
                "predicted_score": "2-0",
                "direction_probs": {"home": 62, "draw": 23, "away": 15},
                "top3": [{"score": "2-0", "prob": 16, "logic": "主胜"}],
                "anchor_audit": {},
                "recommendation_components": {},
                "score_cluster_audit": {},
                "sharp_money_audit": {},
                "web_research": {"used": True, "sources": [{"title": "official", "url": "https://example.com/news", "claim": "官方首发确认", "published_at": "2026-06-13"}]},
                "external_fact_table": [
                    {
                        "category": "lineup",
                        "claim": "主队官方首发确认",
                        "source_type": "official",
                        "source_title": "official",
                        "source_url": "https://example.com/news",
                        "published_at": "2026-06-13",
                        "freshness": "same_day",
                        "confidence": "high",
                        "impact_direction": "upgrade_home",
                        "why_it_matters": "首发完整",
                    }
                ],
                "source_conflict_audit": {"has_conflict": False, "conflicts": []},
                "evidence_quality_score": 88,
                "minimum_evidence_needed": [],
                "external_facts_decision_impact": {"direction_impact": "supports_home", "recommendation_impact": "hold"},
                "recommendation": {"tier": "B", "is_recommended": True, "bet_action": "small", "bet_confidence": 62},
                "reason": "官方首发确认，支持主队方向。",
            }
        ]
    }
    row = predict.normalize_ai_predictions(obj, [1], "grok", "phase1")[1]
    assert row["external_fact_table"][0]["claim"] == "主队官方首发确认"
    assert row["source_conflict_audit"]["has_conflict"] is False
    assert row["evidence_quality_score"] == 88
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.55, "s11": 8.0})
    assert front["external_fact_table"][0]["source_url"] == "https://example.com/news"
    assert front["evidence_quality_score"] == 88


def test_grok_external_fact_fields_reach_final_referee_prompts():
    grok_row = {
        "match": 1,
        "final_direction": "home",
        "predicted_score": "2-0",
        "direction_probs": {"home": 62, "draw": 23, "away": 15},
        "top3": [{"score": "2-0", "prob": 16, "logic": "主胜"}],
        "anchor_audit": {},
        "score_cluster_audit": {},
        "sharp_money_audit": {},
        "web_research": {"used": True, "sources": [{"title": "official", "url": "https://example.com/news", "claim": "官方首发确认"}]},
        "external_fact_table": [{"category": "lineup", "claim": "主队官方首发确认", "source_url": "https://example.com/news"}],
        "source_conflict_audit": {"has_conflict": False, "conflicts": []},
        "evidence_quality_score": 88,
        "minimum_evidence_needed": ["二次确认客队核心停赛"],
        "external_facts_decision_impact": {"direction_impact": "supports_home", "recommendation_impact": "hold"},
        "recommendation": {"tier": "B", "is_recommended": True, "bet_action": "small", "bet_confidence": 62},
        "reason": "官方首发确认，支持主队方向。",
    }
    phase1 = {"gpt": {}, "grok": {1: grok_row}}
    ev = [{"match": 1}]

    for prompt in [
        predict.build_gemini_final_prompt(ev, phase1, {}),
        predict.build_fallback_referee_prompt(ev, phase1, {}),
        predict.build_family_debate_referee_prompt(ev, phase1, {}),
    ]:
        assert "主队官方首发确认" in prompt
        assert "https://example.com/news" in prompt
        assert "二次确认客队核心停赛" in prompt
        assert "supports_home" in prompt


def test_missing_external_source_context_forces_observe_gate():
    row = {
        "final_direction": "home",
        "predicted_score": "2-0",
        "direction_probs": {"home": 62, "draw": 23, "away": 15},
        "top3": [{"score": "2-0", "prob": 16, "logic": "主胜"}],
        "anchor_audit": {},
        "recommendation_components": {},
        "score_cluster_audit": {},
        "sharp_money_audit": {},
        "web_research": {"used": False, "sources": []},
        "recommendation": {"tier": "B", "is_recommended": True, "bet_action": "main", "bet_confidence": 70},
        "reason": "主队核心伤停影响已经消除，首发完整且战意明确，适合主推。",
    }
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.45, "s11": 7.0})
    assert front["recommend_gate_pass"] is False
    assert front["recommendation_tier"] == "C"
    assert front["recommendation"]["bet_action"] == "observe"
    assert "external_fact_without_source" in front["validation_warnings"]
    assert "missing_external_confirmation" in front["validation_warnings"]
