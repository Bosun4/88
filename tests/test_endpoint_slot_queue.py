import asyncio
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import predict


def test_endpoint_slot_override_pins_numbered_endpoint(monkeypatch):
    monkeypatch.setenv("GPT_API_URL", "https://slot1.example/v1")
    monkeypatch.setenv("GPT_API_KEY", "k1")
    monkeypatch.setenv("GPT_API_URL_2", "https://slot2.example/v1")
    monkeypatch.setenv("GPT_API_KEY_2", "k2")
    monkeypatch.setitem(predict.AI_ENDPOINT_MODEL_SLOTS["gpt"], 2, "gpt-slot-2-model")
    predict.AI_ENDPOINT_RR_CURSOR.clear()

    token = predict.AI_ENDPOINT_SLOT_OVERRIDE.set(2)
    try:
        eps = predict._ordered_endpoints_for_ai("gpt")
    finally:
        predict.AI_ENDPOINT_SLOT_OVERRIDE.reset(token)

    assert [ep["slot"] for ep in eps] == [2]
    assert eps[0]["url"] == "https://slot2.example/v1"
    assert eps[0]["model"] == "gpt-slot-2-model"


def test_slot_queue_assigns_one_match_per_slot_then_refills(monkeypatch):
    evidence = [{"match": i, "home": f"H{i}", "away": f"A{i}"} for i in range(1, 8)]
    calls = []

    async def fake_run_one_chunk(session, run_id, chunk_id, evidence_batch):
        slot = predict.AI_ENDPOINT_SLOT_OVERRIDE.get()
        match = evidence_batch[0]["match"]
        calls.append((slot, match, chunk_id))
        await asyncio.sleep(0.001)
        return {match: {"match": match, "slot": slot}}

    monkeypatch.setattr(predict, "_run_one_chunk", fake_run_one_chunk)
    monkeypatch.setattr(predict, "_make_run_id", lambda evidence_all: "test-run")
    monkeypatch.setattr(predict, "_save_snapshot", lambda *args, **kwargs: "")
    monkeypatch.setattr(predict, "AI_ENDPOINT_SLOT_QUEUE", True)
    monkeypatch.setattr(predict, "AI_ENDPOINT_SLOT_WORKERS", 5)
    monkeypatch.setattr(predict, "AI_ENDPOINT_MAX_SLOTS", 5)
    monkeypatch.setattr(predict, "AI_MOCK_MODE", True)

    result = asyncio.run(predict.run_ai_native_web(evidence))

    assert sorted(result) == list(range(1, 8))
    assert sorted({slot for slot, _match, _chunk in calls[:5]}) == [1, 2, 3, 4, 5]
    assert len(calls) == 7
    assert all(1 <= slot <= 5 for slot, _match, _chunk in calls)
