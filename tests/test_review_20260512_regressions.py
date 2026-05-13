import asyncio
import importlib


def test_odds_overround_removal_normalizes_probabilities():
    predict = importlib.import_module("scripts.predict")
    packet = predict.build_evidence_packet(
        {"home_team": "A", "away_team": "B", "sp_home": 2.50, "sp_draw": 3.20, "sp_away": 2.90},
        1,
    )
    implied = packet["market_implied"]
    total = implied["home_fair_prob"] + implied["draw_fair_prob"] + implied["away_fair_prob"]
    assert abs(total - 1.0) <= 0.001
    assert implied["overround"] > 0


def test_build_evidence_packet_runtime_function_has_no_prediction_leakage():
    predict = importlib.import_module("scripts.predict")
    packet = predict.build_evidence_packet(
        {
            "home_team": "塞尔塔",
            "away_team": "莱万特",
            "league": "西甲",
            "match_num": "周二003",
            "sp_home": 2.1,
            "sp_draw": 3.2,
            "sp_away": 3.4,
            "a2": 3.4,
            "a3": 3.8,
            "w21": 8.0,
            "l23": 30.0,
            "prediction": {"predicted_score": "2-1"},
        },
        1,
    )
    assert packet["identity"]["home_team"] == "塞尔塔"
    assert "prediction" not in packet
    assert "predicted_score" not in packet.get("context_raw_fields", {})
    assert packet.get("evidence_compiler_version") == "v20.3.0_sharp_cluster_full"


def test_predicted_score_and_final_direction_are_closed_by_score():
    predict = importlib.import_module("scripts.predict")
    normalized = predict.normalize_ai_predictions(
        {
            "predictions": [
                {
                    "match": 1,
                    "final_direction": "home",
                    "predicted_score": "1-2",
                    "direction_probs": {"home": 55, "draw": 25, "away": 20},
                    "anchor_audit": {},
                }
            ]
        },
        [1],
        "unit",
        "final",
    )
    row = normalized[1]
    assert row["final_direction"] == "away"
    assert "dir_score_conflict_protocol_fixed:home->away" in row["validation_warnings"]


def test_json_output_parser_handles_fenced_json_and_rejects_empty():
    predict = importlib.import_module("scripts.predict")
    parsed = predict._json_loads_best_effort_object('```json\n{"predictions":[{"match":1,"predicted_score":"2-1"}]}\n```')
    assert parsed["predictions"][0]["predicted_score"] == "2-1"
    assert predict._json_loads_best_effort_object("not json") == {}


class _FakeResponse:
    status = 200

    async def text(self):
        return '{"choices":[{"message":{"content":"not json"}}]}'

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def post(self, *args, **kwargs):
        return _FakeResponse()


def test_async_ai_call_marks_invalid_json_as_parse_failed(monkeypatch=None):
    predict = importlib.import_module("scripts.predict")
    original_key = predict.get_key_for_ai
    original_url = predict.get_url_for_ai
    original_aiohttp = predict.aiohttp
    class _FakeAiohttp:
        class ClientTimeout:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
    try:
        predict.get_key_for_ai = lambda _name: "unit-test-key"
        predict.get_url_for_ai = lambda _name: "https://example.test/v1/chat/completions"
        predict.aiohttp = _FakeAiohttp
        _ai, obj, status = asyncio.run(
            predict.async_call_ai_json(_FakeSession(), "gpt", "system", "prompt", "phase1", [1])
        )
    finally:
        predict.get_key_for_ai = original_key
        predict.get_url_for_ai = original_url
        predict.aiohttp = original_aiohttp
    assert obj == {}
    assert status["ok"] is False
    assert status["status"] == "parse_failed"
    assert status["parse_error"] == "empty_or_invalid_json_object"


def test_goal_band_high_score_tail_protection():
    predict = importlib.import_module("scripts.predict")
    assert predict._score_goal_band("2-3") == "4+"
    normalized = predict.normalize_ai_predictions(
        {"predictions": [{"match": 1, "final_direction": "away", "predicted_score": "2-3", "direction_probs": {"home": 24, "draw": 28, "away": 48}, "anchor_audit": {}}]},
        [1],
        "unit",
        "final",
    )
    assert normalized[1]["goal_band"] == "4+"
    assert normalized[1]["btts"] == "yes"
