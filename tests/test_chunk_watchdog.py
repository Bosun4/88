#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单场看门狗超时测试 (2026-07-03)

背景: 2026-07-02晚间线上run, Chunk3/5的Gemini final挂住无响应,
AI_FINAL_READ_TIMEOUT=7200(2h)+重试3次 → 整个run吊死1h40m后被人工取消。
正常单场链路仅5-7分钟。

修复: _run_one_chunk 外层包 asyncio.wait_for 看门狗
(AI_CHUNK_WATCHDOG_SECONDS, 默认2400s=40min, 0=禁用),
超时该场标记失败继续下一场, 不吊死全局。
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import predict as P


def test_watchdog_env_exists_and_default():
    assert hasattr(P, "AI_CHUNK_WATCHDOG_SECONDS")
    assert P.AI_CHUNK_WATCHDOG_SECONDS == 2400  # 默认40分钟


def test_watchdog_wraps_hung_chunk():
    """挂住的chunk被看门狗掐掉, 返回空dict而不是永远等待"""
    async def _hung_chunk(*a, **k):
        await asyncio.sleep(999)
        return {1: {"ok": True}}

    async def _run():
        return await P._run_chunk_with_watchdog(_hung_chunk(), chunk_desc="test", watchdog=0.1)

    rows = asyncio.run(_run())
    assert rows == {}


def test_watchdog_passes_fast_chunk():
    async def _fast_chunk():
        return {1: {"predicted_score": "2-0"}}

    async def _run():
        return await P._run_chunk_with_watchdog(_fast_chunk(), chunk_desc="test", watchdog=5)

    rows = asyncio.run(_run())
    assert rows == {1: {"predicted_score": "2-0"}}


def test_watchdog_disabled_when_zero():
    """watchdog=0 → 不限制(兼容旧行为)"""
    async def _fast_chunk():
        return {2: {"predicted_score": "1-1"}}

    async def _run():
        return await P._run_chunk_with_watchdog(_fast_chunk(), chunk_desc="test", watchdog=0)

    rows = asyncio.run(_run())
    assert rows == {2: {"predicted_score": "1-1"}}


@pytest.mark.parametrize('concurrency', [1, 2])
def test_runner_times_out_hung_match_and_keeps_other_results(monkeypatch, concurrency):
    monkeypatch.setattr(P, 'AI_RUN_MODE', 'fast_batch')
    monkeypatch.setattr(P, 'AI_MOCK_MODE', True)
    monkeypatch.setattr(P, 'AI_CHUNK_SIZE', 1)
    monkeypatch.setattr(P, 'AI_CHUNK_CONCURRENCY', concurrency)
    monkeypatch.setattr(P, 'AI_CHUNK_WATCHDOG_SECONDS', 0.02)
    monkeypatch.setattr(P, '_save_snapshot', lambda *args, **kwargs: '')

    async def chunk(session, run_id, chunk_id, evidence):
        if evidence[0]['match'] == 1:
            await asyncio.Event().wait()
        return {2: {'predicted_score': '2-0'}}

    monkeypatch.setattr(P, '_run_one_chunk', chunk)
    assert asyncio.run(P.run_ai_native_web([{'match': 1}, {'match': 2}])) == {
        2: {'predicted_score': '2-0'}}
