import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import predict


def test_protocol_preserves_explicit_final_direction_on_score_conflict():
    row = {
        "final_direction": "home",
        "predicted_score": "1-2",
        "top3": [{"score": "1-2", "prob": 18}],
        "anchor_audit": {},
    }

    predict._protocol_enforce_prediction(row)

    assert row["final_direction"] == "home"
    assert row["score_direction_conflict"] is True
    assert "protocol_score_direction_conflict_preserved:home!=away" in row["validation_warnings"]


def test_normalize_top3_does_not_force_predicted_score_to_rank_one():
    row = {
        "predicted_score": "2-0",
        "top3": [
            {"score": "2-1", "prob": 22, "logic": "btts candidate"},
            {"score": "2-0", "prob": 18, "logic": "clean sheet"},
            {"score": "1-0", "prob": 12},
        ],
    }

    top = predict._normalize_top3(row, "2-0")

    assert [x["score"] for x in top[:3]] == ["2-1", "2-0", "1-0"]
    assert top[0]["logic"] == "btts candidate"


def test_score_shape_selector_promotes_btts_clean_sheet_candidate_without_changing_direction():
    row = {
        "final_direction": "home",
        "predicted_score": "2-0",
        "direction_probs": {"home": 62, "draw": 22, "away": 16},
        "goal_band": "2",
        "btts": "yes",
        "top3": [
            {"score": "2-0", "prob": 18},
            {"score": "2-1", "prob": 17},
            {"score": "1-0", "prob": 12},
        ],
        "recommendation": {"tier": "A", "is_recommended": True, "bet_confidence": 70},
    }

    out = predict._score_shape_selector(row, {})

    assert out["final_direction"] == "home"
    assert out["predicted_score"] == "2-1"
    assert out["btts"] == "yes"
    assert out["score_shape_calibrated"] is True
    assert out["score_shape_selector"]["reason"] == "score_shape_selector_btts_clean_sheet_uplift"
    assert out["top3"][0]["score"] == "2-1"


def test_score_shape_selector_draw_high_band_promotes_existing_high_draw_candidate():
    row = {
        "final_direction": "draw",
        "predicted_score": "1-1",
        "direction_probs": {"home": 31, "draw": 38, "away": 31},
        "goal_band": "4+",
        "btts": "yes",
        "top3": [
            {"score": "1-1", "prob": 17},
            {"score": "2-2", "prob": 15},
            {"score": "0-0", "prob": 9},
        ],
        "tail_risk_flags": ["high_btts_tail"],
        "recommendation": {"tier": "B", "is_recommended": False, "bet_confidence": 43},
    }

    out = predict._score_shape_selector(row, {})

    assert out["final_direction"] == "draw"
    assert out["predicted_score"] == "2-2"
    assert out["goal_band"] == "4+"
    assert out["score_shape_selector"]["reason"] == "score_shape_selector_draw_band_rerank"
