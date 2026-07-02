#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""内容质量三连修测试 (2026-07-02)

1. model_consensus 恒为 None (硬编码) → 按 phase1 各模型方向与终审方向一致数计算;
   缺席模型不计入分母 (GPT http_524 缺席时按在场模型算)。
2. ou_score_conflict=True 时输出 ou_display 协调字段:
   前端可显示"市场倾向大/小(与比分预期存在分歧)"而非裸矛盾。
3. http_524 (Cloudflare 网关超时) 必须可重试/可failover。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import predict as P


# ---------- 1) model_consensus ----------

def _mk_ai_r(final_score="2-0", phase1=None):
    return {
        "predicted_score": final_score,
        "phase1_model_outputs": phase1 or {},
    }


def test_consensus_counts_agreeing_models():
    ai_r = _mk_ai_r("2-0", {
        "gpt": {"predicted_score": "1-0"},    # home = agree
        "grok": {"predicted_score": "2-1"},   # home = agree
    })
    c, t = P._compute_model_consensus(ai_r, "home")
    assert (c, t) == (2, 2)


def test_consensus_excludes_absent_models():
    """GPT 缺席(无phase1行) → 分母只算在场模型"""
    ai_r = _mk_ai_r("2-0", {
        "grok": {"predicted_score": "0-0"},   # draw = disagree
    })
    c, t = P._compute_model_consensus(ai_r, "home")
    assert (c, t) == (0, 1)


def test_consensus_empty_phase1_returns_none():
    c, t = P._compute_model_consensus(_mk_ai_r("2-0", {}), "home")
    assert c is None and t == 0


def test_consensus_direction_from_direction_field_fallback():
    """phase1 行没有比分但有 final_direction 时也能计"""
    ai_r = _mk_ai_r("2-0", {
        "gpt": {"final_direction": "home"},
    })
    c, t = P._compute_model_consensus(ai_r, "home")
    assert (c, t) == (1, 1)


# ---------- 2) OU 冲突展示协调 ----------

def test_ou_conflict_display_present():
    # 市场大球倾向 vs 比分2-0(小) → 冲突, 必须有协调文案
    match_obj = {"a0": 15.0, "a1": 8.0, "a2": 5.0, "a3": 3.2, "a4": 3.0, "a5": 5.5, "a6": 9.0, "a7": 15.0}
    out = P.derive_ou_head(match_obj, "2-0")
    assert out.get("ou_score_conflict") is True
    disp = out.get("ou_display") or ""
    assert disp and ("分歧" in disp or "冲突" in disp)


def test_ou_no_conflict_display_simple():
    # 市场大球 + 比分3-1(大) → 不冲突, 文案简洁无"分歧"
    match_obj = {"a0": 15.0, "a1": 8.0, "a2": 5.0, "a3": 3.2, "a4": 3.0, "a5": 5.5, "a6": 9.0, "a7": 15.0}
    out = P.derive_ou_head(match_obj, "3-1")
    assert out.get("ou_score_conflict") is False
    disp = out.get("ou_display") or ""
    assert disp and "分歧" not in disp


def test_ou_abstain_no_display_crash():
    out = P.derive_ou_head({}, "弃权")
    assert out.get("over_under_2_5") is None



# ---------- 3) http_524 可重试 ----------

def test_http_524_is_retryable():
    assert P._is_retryable_ai_status({"status": "http_524"})


def test_http_522_is_retryable():
    assert P._is_retryable_ai_status({"status": "http_522"})


def test_http_400_not_retryable():
    assert not P._is_retryable_ai_status({"status": "http_400"})
