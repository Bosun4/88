#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase1/critic 瞬时失败重试测试 (2026-07-02)

背景: 用户反馈 grok 经常"该模型本轮未返回可展示分析"。
根因: final/fallback 裁决有 retry(AI_FINAL_RETRY_MAX=2), 但 phase1/critic 零重试,
     单模型一次 429/timeout/空补全 就整轮缺席。
修复: phase1/critic 统一走 async_call_ai_json_with_retry, 次数由 AI_PHASE1_RETRY_MAX 控制(默认1)。
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import predict


def test_phase1_retry_env_default():
    # 默认至少1次重试(瞬时故障自动恢复), 且可环境覆盖
    assert predict.AI_PHASE1_RETRY_MAX >= 1


def test_retry_wrapper_recovers_transient_phase1_failure(monkeypatch):
    """第一次 timeout, 第二次成功 → phase1 应拿到结果"""
    calls = {"n": 0}

    async def fake_call(session, ai_name, system_text, prompt, phase, expected):
        calls["n"] += 1
        if calls["n"] == 1:
            return ai_name, {}, {"ok": False, "status": "timeout"}
        return ai_name, {"predictions": []}, {"ok": True, "status": "ok"}

    monkeypatch.setattr(predict, "async_call_ai_json", fake_call)
    name, obj, st = asyncio.run(
        predict.async_call_ai_json_with_retry(None, "grok", "sys", "prompt", "phase1", [1], 1)
    )
    assert st["ok"] is True
    assert calls["n"] == 2
    assert st.get("retry_succeeded_on_attempt") == 2


def test_retry_wrapper_gives_up_on_permanent_failure(monkeypatch):
    """no_key 属永久失败 → 不重试, 快速返回"""
    calls = {"n": 0}

    async def fake_call(session, ai_name, system_text, prompt, phase, expected):
        calls["n"] += 1
        return ai_name, {}, {"ok": False, "status": "no_key"}

    monkeypatch.setattr(predict, "async_call_ai_json", fake_call)
    name, obj, st = asyncio.run(
        predict.async_call_ai_json_with_retry(None, "grok", "sys", "prompt", "phase1", [1], 3)
    )
    assert st["ok"] is False
    assert calls["n"] == 1  # 不该浪费重试在永久失败上


def test_run_one_chunk_uses_retry_for_phase1():
    """_run_one_chunk 源码必须让 phase1/critic 走 retry 包装(防回归)"""
    import inspect
    src = inspect.getsource(predict._run_one_chunk)
    # phase1 与 critic 的任务构造都应引用 retry 包装
    assert src.count("async_call_ai_json_with_retry") >= 3  # phase1 + critic + final(原有)


def test_legacy_analysis_placeholder_mentions_status():
    """模型缺席时提示语应携带失败状态便于排障(而非笼统一句)"""
    ai_r = {
        "phase1_model_outputs": {},
        "phase1_status": {"grok": {"ok": False, "status": "timeout", "endpoint": "svip"}},
    }
    msg = predict._legacy_model_analysis(ai_r, "grok")
    assert "timeout" in msg
