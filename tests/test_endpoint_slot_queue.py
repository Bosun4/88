"""Fixture concurrency is independent of the removed endpoint slot queue."""
import asyncio
import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from scripts import predict


@pytest.mark.parametrize('concurrency', [1, 2])
def test_legacy_batches_respect_concurrency_and_process_each_match_once(monkeypatch, concurrency):
    monkeypatch.setattr(predict, 'AI_RUN_MODE', 'fast_batch')
    monkeypatch.setattr(predict, 'AI_CHUNK_SIZE', 1)
    monkeypatch.setattr(predict, 'AI_CHUNK_CONCURRENCY', concurrency)
    monkeypatch.setattr(predict, 'AI_MOCK_MODE', True)
    monkeypatch.setattr(predict, '_save_snapshot', lambda *args, **kwargs: '')
    evidence = [{'match': i} for i in range(1, 8)]
    active = peak = 0
    calls = []

    async def fake_chunk(session, run_id, chunk_id, evidence_batch):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        match = evidence_batch[0]['match']
        calls.append(match)
        await asyncio.sleep(0)
        active -= 1
        return {match: {'match': match}}

    monkeypatch.setattr(predict, '_run_one_chunk', fake_chunk)
    result = asyncio.run(predict.run_ai_native_web(evidence))
    assert sorted(result) == list(range(1, 8))
    assert sorted(calls) == list(range(1, 8))
    assert peak == concurrency
