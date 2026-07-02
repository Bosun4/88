#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPT输出截断导致 parse_failed 的修复测试 (2026-07-02)

实测: 2026-07-02 线上 6/6 场 GPT last_status=parse_failed。
raw_excerpt 显示返回体是合法JSON开头但中途截断(端点输出token上限)。
旧 _repair_truncated_json 救不了「截在key中间」「截在转义符中间」两种姿势,
且每次失败都触发 122s 的 fallback 裁判 → 单场慢 28%。

修复:
1. _repair_truncated_json 增加 trim-to-last-parseable-prefix 回退:
   逐步回退到最近的结构安全点(逗号/括号边界)再闭合。
2. 请求显式带 max_tokens(AI_MAX_OUTPUT_TOKENS, 默认关闭=沿用端点默认,
   可在workflow调大避免截断)。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import predict as P


def _ok(obj):
    return isinstance(obj, (dict, list)) and bool(obj)


def test_truncated_mid_string_still_recovers():
    txt = '{"critic_reports": [{"match": 7, "target_findings": [{"target_model": "grok", "comment": "将抓包字段包装成 web_research 来源 ac'
    assert _ok(P._json_loads_best_effort_object(txt))


def test_truncated_mid_key_recovers():
    """截在 key 中间: {"...": [{"target_model": "grok", "comm ← 旧版返回{}"""
    txt = '{"critic_reports": [{"match": 7, "target_findings": [{"target_model": "grok", "comm'
    obj = P._json_loads_best_effort_object(txt)
    assert _ok(obj)
    assert "critic_reports" in obj


def test_truncated_mid_escape_recovers():
    """截在转义符中间: "abc\\ ← 旧版返回{}"""
    txt = '{"critic_reports": [{"match": 7, "comment": "abc\\'
    obj = P._json_loads_best_effort_object(txt)
    assert _ok(obj)


def test_truncated_after_colon_recovers():
    txt = '{"critic_reports": [{"match": 7, "target_findings": [{"target_model":'
    obj = P._json_loads_best_effort_object(txt)
    assert _ok(obj)


def test_recovered_prefix_keeps_complete_items():
    """恢复后应保留截断前完整的条目(match=1完整, match=2截断)"""
    txt = ('{"predictions": [{"match": 1, "score": "2-1", "direction": "home"},'
           ' {"match": 2, "score": "1-')
    obj = P._json_loads_best_effort_object(txt)
    assert _ok(obj)
    preds = obj.get("predictions") or []
    assert any(p.get("match") == 1 and p.get("score") == "2-1" for p in preds if isinstance(p, dict))


def test_valid_json_untouched():
    txt = '{"a": [1, 2], "b": {"c": "d"}}'
    assert P._json_loads_best_effort_object(txt) == {"a": [1, 2], "b": {"c": "d"}}


def test_max_output_tokens_env_exists():
    """AI_MAX_OUTPUT_TOKENS 配置存在; 默认0=不注入(沿用端点默认)"""
    assert hasattr(P, "AI_MAX_OUTPUT_TOKENS")
    assert P.AI_MAX_OUTPUT_TOKENS >= 0
