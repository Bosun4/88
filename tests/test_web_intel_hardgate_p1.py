# -*- coding: utf-8 -*-
"""P1 联网情报链硬闸测试：新鲜度 / 来源冲突 / evidence_quality<50 从君子协定升级为代码硬闸。

基线缺陷（origin/main ac3e566）：
- 新鲜度(published_at/freshness)无任何代码强制，仅 prompt 文字要求。
- source_conflict_audit 是 AI 自填字段，gate 零校验。
- evidence_quality<50 仅 append warning，不降 tier / 不拦 gate（与 valid_sources==0 的硬降级形成漏洞）。
"""
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
        "web_research": {"used": True, "sources": [{"title": "official", "url": "https://example.com/team-news", "claim": "官方首发确认", "published_at": "2026-06-16"}]},
        "external_fact_table": [{"claim": "官方首发确认", "source_url": "https://example.com/team-news", "category": "lineup", "published_at": "2026-06-16", "freshness": "same_day"}],
        "evidence_quality_score": 85,
        "recommendation": {"tier": "A", "is_recommended": True, "bet_action": "main", "bet_confidence": 78},
        "reason": "官方首发与战意明确，适合主推。",
    }
    row.update(overrides)
    return row


# ---------- 硬闸 1：stale 来源 ----------

def test_stale_external_fact_caps_recommendation():
    """所有 external_fact 都 stale/过旧 且依赖外部事实 → 不得升 main，tier capped C。"""
    row = _row(
        external_fact_table=[{"claim": "三年前的旧首发新闻", "source_url": "https://example.com/old", "category": "lineup", "published_at": "2023-01-01", "freshness": "stale"}],
        web_research={"used": True, "sources": [{"title": "old", "url": "https://example.com/old", "claim": "旧首发", "published_at": "2023-01-01"}]},
        reason="依据首发阵容判断主推。",
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.45, "s11": 7.2})
    assert front["recommend_gate_pass"] is False
    assert predict._tier_cap_value(front["recommendation"]["tier"]) <= predict._tier_cap_value("C")
    assert "external_facts_all_stale" in front["validation_warnings"]


def test_fresh_external_fact_is_not_penalized_for_staleness():
    """same_day 新鲜来源不应被新鲜度闸误杀。"""
    front = predict.adapt_ai_to_frontend(_row(), {"league": "世界杯", "sp_home": 1.45, "s11": 7.2})
    assert "external_facts_all_stale" not in front.get("validation_warnings", [])


# ---------- 硬闸 2：来源冲突 ----------

def test_unresolved_source_conflict_caps_recommendation():
    """source_conflict_audit.has_conflict=true 且 conflicts 非空 → 不得升 main，tier capped C。"""
    row = _row(
        source_conflict_audit={"has_conflict": True, "conflicts": [{"topic": "首发", "a": "官方说A首发", "b": "记者说A伤缺"}]},
        reason="依据首发阵容判断主推。",
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.45, "s11": 7.2})
    assert front["recommend_gate_pass"] is False
    assert predict._tier_cap_value(front["recommendation"]["tier"]) <= predict._tier_cap_value("C")
    assert "unresolved_source_conflict" in front["validation_warnings"]


def test_no_conflict_is_not_penalized():
    """无冲突时不应被冲突闸误杀。"""
    row = _row(source_conflict_audit={"has_conflict": False, "conflicts": []})
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.45, "s11": 7.2})
    assert "unresolved_source_conflict" not in front.get("validation_warnings", [])


# ---------- 硬闸 3：evidence_quality < 50 硬降级 ----------

def test_low_evidence_quality_hard_caps_recommendation():
    """0 < evidence_quality < 50 → tier capped C + gate_pass=False（基线只 warning 不降级）。"""
    row = _row(evidence_quality_score=35)
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.45, "s11": 7.2})
    assert front["recommend_gate_pass"] is False
    assert predict._tier_cap_value(front["recommendation"]["tier"]) <= predict._tier_cap_value("C")
    assert "external_evidence_quality_below_50" in front["validation_warnings"]


def test_high_evidence_quality_not_capped_by_quality_gate():
    """高 quality 不被质量闸降级。"""
    front = predict.adapt_ai_to_frontend(_row(evidence_quality_score=85), {"league": "世界杯", "sp_home": 1.45, "s11": 7.2})
    assert "external_evidence_quality_below_50" not in front.get("validation_warnings", [])
