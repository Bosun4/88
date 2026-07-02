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


def test_slot_worker_uses_watchdog_in_source():
    """源码守卫: 三条执行路径(_slot_worker/顺序/chunk并发)都必须走看门狗"""
    src = Path(P.__file__).read_text(encoding="utf-8")
    assert src.count("_run_chunk_with_watchdog(") >= 4  # 定义1 + 调用≥3
