import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import asyncio
import importlib
from scripts import predict

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

def test_tail_risk_protection_for_weak_home_favorite():
    obj = {
        "predictions": [
            {
                "match": 1,
                "final_direction": "home",
                "predicted_score": "2-1",
                "direction_probs": {"home": 48, "draw": 28, "away": 24},
                "goal_band": "3",
                "btts": "yes",
                "top3": [{"score": "2-1", "prob": 18, "logic": "weak home favorite primary score"}],
                "recommendation": {
                    "tier": "A",
                    "is_recommended": True,
                    "top4_priority": 1,
                    "bet_confidence": 78,
                    "direction_stability": "medium",
                    "score_stability": "medium",
                    "risk_level": "low",
                    "risk_tags": [],
                    "why_recommended": "test fixture",
                },
                "score_cluster_audit": {},
                "sharp_money_audit": {},
                "recommendation_components": {},
                "anchor_audit": {},
                "web_research": {"used": False, "sources": []},
                "reason": "BTTS=yes with non-negligible away fight-back risk",
            }
        ]
    }

    rows = predict.normalize_ai_predictions(obj, [1], "gemini", "final")
    row = rows[1]

    assert row["predicted_score"] == "2-1"
    assert row["final_direction"] == "home"
    risk_scores = {c["score"] for c in row["risk_score_candidates"]}
    assert {"1-2", "2-2", "2-3"}.issubset(risk_scores)
    assert "weak_home_favorite_btts_tail" in row["tail_risk_flags"]
    assert row["confidence_downgrade_reason"] == "Weak home favorite with BTTS tail risk"
    assert row["recommendation"]["bet_confidence"] <= 60
    assert "weak_home_favorite_btts_tail_protection_applied" in row["validation_warnings"]

    frontend = predict.adapt_ai_to_frontend(row, {})
    assert frontend["confidence"] <= 60
    assert {"1-2", "2-2", "2-3"}.issubset({c["score"] for c in frontend["risk_score_candidates"]})


def _matrix_shadow_fixture():
    return {
        "home_team": "A",
        "away_team": "B",
        "sp_home": 2.10,
        "sp_draw": 3.25,
        "sp_away": 3.60,
        "a0": 9.5,
        "a1": 5.4,
        "a2": 3.3,
        "a3": 3.9,
        "a4": 6.8,
        "a5": 13.5,
        "a6": 28.0,
        "a7": 60.0,
        "w10": 7.5,
        "w20": 9.5,
        "w21": 8.0,
        "w30": 18.0,
        "w31": 14.0,
        "w32": 24.0,
        "w40": 42.0,
        "w41": 32.0,
        "w42": 45.0,
        "w50": 80.0,
        "w51": 70.0,
        "w52": 90.0,
        "s00": 10.0,
        "s11": 6.5,
        "s22": 13.0,
        "s33": 45.0,
        "l01": 10.5,
        "l02": 16.0,
        "l12": 11.0,
        "l03": 35.0,
        "l13": 26.0,
        "l23": 36.0,
        "l04": 80.0,
        "l14": 70.0,
        "l24": 90.0,
        "l05": 120.0,
        "l15": 110.0,
        "l25": 130.0,
        "crs_win": 55.0,
        "crs_same": 90.0,
        "crs_lose": 70.0,
    }


def _ai_row_fixture():
    return {
        "match": 1,
        "final_direction": "home",
        "predicted_score": "2-1",
        "direction_probs": {"home": 52, "draw": 27, "away": 21},
        "goal_band": "3",
        "btts": "yes",
        "top3": [{"score": "2-1", "prob": 18, "logic": "fixture"}],
        "recommendation": {"tier": "A", "is_recommended": True, "bet_confidence": 74, "risk_level": "medium"},
        "anchor_audit": {},
        "reason": "fixture",
    }


def test_matrix_shadow_fields_exist_and_do_not_override_core_decision_fields():
    row = predict.adapt_ai_to_frontend(_ai_row_fixture(), _matrix_shadow_fixture())

    for key in [
        "matrix_direction_probs",
        "matrix_top_scores",
        "matrix_goal_probs",
        "matrix_lambda_home",
        "matrix_lambda_away",
        "matrix_shape_verdict",
        "matrix_recommended_score",
        "matrix_recommended_direction",
        "matrix_disagreement_flags",
        "matrix_shadow_error",
    ]:
        assert key in row

    assert row["predicted_score"] == "2-1"
    assert row["final_direction"] == "home"
    assert row["confidence"] == 74
    assert row["result"] == "主胜"
    assert row["display_direction"] == "主胜"
    assert row["home_win_pct"] == 52
    assert row["draw_pct"] == 27
    assert row["away_win_pct"] == 21


def test_shadow_1x2_fair_probs_sum_to_100():
    fair = predict.fair_probs_from_1x2_shadow(2.10, 3.25, 3.60)["fair_probs"]
    assert abs(sum(fair.values()) - 100.0) <= 0.02


def test_matrix_goal_probs_sum_to_100_and_top_scores_nonempty():
    pred = predict.attach_matrix_shadow_fields(_ai_row_fixture(), _matrix_shadow_fixture())
    assert abs(sum(float(v) for v in pred["matrix_goal_probs"].values()) - 100.0) <= 0.05
    assert pred["matrix_top_scores"]


def test_matrix_shadow_source_has_no_team_date_or_result_hardcoding():
    import inspect
    source = "\n".join(
        inspect.getsource(getattr(predict, name))
        for name in [
            "fair_probs_from_1x2_shadow",
            "fair_probs_from_ttg_shadow",
            "crs_implied_probabilities_shadow",
            "build_unified_score_matrix_shadow",
            "attach_matrix_shadow_fields",
        ]
    )
    forbidden_tokens = ["塞尔塔", "莱万特", "2026", "2025", "周二003", "2-1命中", "赛果"]
    assert not any(token in source for token in forbidden_tokens)
import sys
import os

sys.path.append(os.path.abspath("/root/.openclaw/workspace/repos/88"))
from scripts.predict import apply_weak_home_tail_risk_protection

def test_tail_guard_threshold_expands_weak_home_shadow_candidates_without_overriding_final():
    # home_pct=54.5, away_pct=22.5 => should trigger the new <=55.0 / >=22.0 threshold
    row = {
        "final_direction": "home",
        "predicted_score": "2-1",
        "direction_probs": {"home": 54.5, "draw": 23.0, "away": 22.5},
        "raw_item": {"btts": "yes"}
    }
    
    # Run the protection
    res = apply_weak_home_tail_risk_protection(row)
    
    candidates = [c.get("score") for c in res.get("risk_score_candidates", [])]
    flags = res.get("tail_risk_flags", [])
    
    # Check flags and candidates
    assert "weak_home_favorite_btts_tail" in flags, "Missing weak_home_favorite_btts_tail flag"
    
    matched_candidates = set(candidates).intersection({"1-2", "2-2", "2-3"})
    assert len(matched_candidates) >= 2, f"Expected at least two of 1-2/2-2/2-3, got {candidates}"
    
    # Check it didn't modify final_direction or predicted_score
    assert res.get("final_direction") == "home", "final_direction was modified!"
    assert res.get("predicted_score") == "2-1", "predicted_score was modified!"

if __name__ == "__main__":
    test_tail_guard_threshold_expands_weak_home_shadow_candidates_without_overriding_final()
    print("test_tail_guard_threshold_expands_weak_home_shadow_candidates_without_overriding_final passed")
