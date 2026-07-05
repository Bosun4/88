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
    assert "Bet365/William Hill/威廉希尔/Pinnacle/竞彩/百家均值" in instr
    assert "赔率升水降水与大小球变化不一致" in instr


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


def test_gemini_final_referee_gets_independent_web_and_market_tasks():
    prompt = predict.build_gemini_final_prompt([{"match": 1}], {"gpt": {}, "grok": {}}, {})

    assert "独立联网裁判职责" in prompt
    assert "即使 Grok 无法联网" in prompt
    assert "必须独立执行 Web-Augmented Match Research" in prompt
    assert "final_web_audit.web_used_by_final=false" in prompt
    assert "全市场终审职责" in prompt
    assert "1X2 欧赔/竞彩" in prompt
    assert "HHAD/亚盘让球语义" in prompt
    assert "总进球/大小球" in prompt
    assert "正确比分赔率簇" in prompt
    assert "不得把赔率、亚盘、比分簇任务只交给 GPT/Grok" in prompt
    assert "庄家逆向读盘职责" in prompt
    assert "Bet365、William Hill/威廉希尔、Pinnacle/低抽水基准、竞彩与百家均值" in prompt
    assert "节奏/xG/战术脏活" in prompt
    assert "日本型反击爆冷" in prompt
    assert "摩洛哥型低位铁桶" in prompt
    assert "克罗地亚型韧性拖平" in prompt
    assert "比分淘汰协议" in prompt
    assert "0-0、1-1、2-2、1-2、2-1" in prompt
    assert "世界杯淘汰赛三分法" in prompt
    assert "加时点球" in prompt


def test_full_spectrum_audit_schema_and_consistency_requirements():
    schema = predict._canonical_output_schema_text()
    judge = predict.build_consistency_judge_prompt([{"match": 1}], {1: {"match": 1, "final_direction": "home", "predicted_score": "2-0"}})

    for key in [
        "gemini_independent_research",
        "bookmaker_cross_audit",
        "tempo_xg_tactical_audit",
        "worldcup_upset_audit",
        "score_elimination_audit",
        "dirty_work_checklist",
    ]:
        assert key in schema
    assert "bet365" in schema
    assert "william_hill" in schema
    assert "ninety_minute_vs_advance_semantics" in schema
    assert "extra_time_penalty" in schema
    assert "0-0" in schema and "1-1" in schema and "2-2" in schema and "1-2" in schema and "2-1" in schema
    assert "bookmaker_cross_audit/tempo_xg_tactical_audit/score_elimination_audit 必须存在" in judge
    assert "score_elimination_audit 必须覆盖 0-0/1-1/2-2/1-2/2-1" in judge


def test_output_schema_keeps_no_bet_out_of_final_direction():
    schema = predict._canonical_output_schema_text()

    assert '"final_direction": "home/draw/away"' in schema
    assert "final_direction 只能是 home/draw/away" in schema
    assert "no_bet/observe 只能写在 recommendation.bet_action" in schema
    assert "系统级 abstain 只用于程序兜底" in schema


def test_phase1_addendum_uses_no_bet_as_recommendation_action_only():
    assert "允许 final_direction=abstain/no_bet" not in predict.PHASE1_ROLE_SPLIT_ADDENDUM
    assert "final_direction 只能是 home/draw/away" in predict.PHASE1_ROLE_SPLIT_ADDENDUM
    assert "bet_action=observe/no_bet" in predict.PHASE1_ROLE_SPLIT_ADDENDUM


def test_grok_phase1_prompt_preserves_schema_despite_not_final_referee():
    prompt = predict.build_phase1_prompt([{"match": 1}], "grok")

    assert "Grok重点【Web-Max 外部事实与资金背离审判员" in prompt
    assert "external_fact_table/source_conflict_audit/evidence_quality_score" in prompt
    assert "暂定 final_direction/predicted_score" in prompt
    assert "不要做最终比分预测" not in prompt
    assert '"final_direction": "home/draw/away"' in prompt


def test_consistency_judge_checks_external_fact_source_integrity():
    prompt = predict.build_consistency_judge_prompt([{"match": 1}], {1: {"match": 1, "final_direction": "home", "predicted_score": "2-0"}})

    assert "final_direction 只能是 home/draw/away" in prompt
    assert "no_bet/observe 只能存在于 recommendation.bet_action" in prompt
    assert "external_fact_table 非空时每条必须有 source_url" in prompt
    assert "source_conflict_audit/evidence_quality_score/external_facts_decision_impact" in prompt
    assert "repair 只能降级 recommendation 为 observe/no_bet" in prompt


def test_invalid_ai_final_direction_is_warned_and_score_direction_preserved():
    obj = {
        "predictions": [
            {
                "match": 1,
                "final_direction": "no_bet",
                "predicted_score": "1-1",
                "top3": [{"score": "1-1"}],
                "anchor_audit": {},
                "score_cluster_audit": {},
                "sharp_money_audit": {},
                "recommendation_components": {},
                "recommendation": {"is_recommended": False, "bet_action": "no_bet"},
            }
        ]
    }

    row = predict.normalize_ai_predictions(obj, [1], "gemini", "final")[1]

    assert row["final_direction"] == "draw"
    assert row["raw_ai_direction"] == "draw"
    assert "invalid_final_direction_protocol_fixed:no_bet->draw" in row["validation_warnings"]


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
                "bookmaker_cross_audit": {"bet365": "主胜降水但威廉平赔未抬死", "water_movement": "当前快照，不编时间线"},
                "tempo_xg_tactical_audit": {"tempo": "high", "xg_signal": "主队高压创造质量更好"},
                "worldcup_upset_audit": {"japan_type_counter": "unclear", "morocco_type_low_block": "no", "croatia_type_resilience": "no"},
                "score_elimination_audit": {"0-0": "reject", "1-1": "reject", "2-2": "reject", "1-2": "reject", "2-1": "keep"},
                "dirty_work_checklist": {"lineup": True, "odds_sources": True, "xg_tempo": True},
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
    assert front["bookmaker_cross_audit"]["bet365"] == "主胜降水但威廉平赔未抬死"
    assert front["tempo_xg_tactical_audit"]["tempo"] == "high"
    assert front["score_elimination_audit"]["2-1"] == "keep"


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
