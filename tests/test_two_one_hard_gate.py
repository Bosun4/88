import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import predict


def _home_21_row(**overrides):
    row = {
        "match": 1,
        "final_direction": "home",
        "predicted_score": "2-1",
        "direction_probs": {"home": 49, "draw": 27, "away": 24},
        "goal_band": "3",
        "btts": "yes",
        "top3": [{"score": "2-1", "prob": 18}],
        "risk_score_candidates": [{"score": "1-2", "prob": 7}, {"score": "2-2", "prob": 10}, {"score": "2-3", "prob": 4}],
        "recommendation": {"tier": "A", "is_recommended": True, "bet_confidence": 78, "risk_level": "low", "risk_tags": []},
        "anchor_audit": {},
        "reason": "BTTS=yes, away fight-back visible",
    }
    row.update(overrides)
    return row


def test_home_2_1_with_weak_home_btts_tail_is_forced_no_bet_without_changing_score():
    row = predict.apply_weak_home_tail_risk_protection(_home_21_row())
    front = predict.adapt_ai_to_frontend(row, {})

    assert front["predicted_score"] == "2-1"
    assert front["final_direction"] == "home"
    assert front["recommend_gate_pass"] is False
    assert front["recommendation"]["tier"] == "D"
    assert front["recommendation"]["is_recommended"] is False
    assert front["confidence"] <= 49
    assert "two_one_home_hard_no_bet" in front["recommendation_downgrade_reasons"]
    assert "two_one_home_hard_no_bet_gate_applied" in front["validation_warnings"]
    assert front["no_bet_reason"]
    assert front["sub50_tiebreaker_warning"] is True


def test_home_2_1_matrix_draw_away_warning_blocks_recommendation():
    row = _home_21_row(direction_probs={"home": 55, "draw": 24, "away": 21}, btts="no", risk_score_candidates=[])
    row.update({
        "matrix_direction_probs": {"home": 43, "draw": 31, "away": 26},
        "matrix_disagreement_flags": {"matrix_draw_risk_warning": True, "matrix_away_tail_warning": True},
    })
    gated = predict.apply_two_one_home_hard_no_bet_gate(row)

    assert gated["predicted_score"] == "2-1"
    assert gated["final_direction"] == "home"
    assert gated["recommend_gate_pass"] is False
    assert gated["recommendation"]["tier"] == "D"
    assert "matrix_draw_or_away_tail_warning" in gated["recommend_gate_reasons"]


def test_home_2_1_strong_clean_case_is_not_blocked_by_hard_gate():
    row = _home_21_row(
        direction_probs={"home": 61, "draw": 22, "away": 17},
        btts="no",
        risk_score_candidates=[],
        top3=[{"score": "2-1", "prob": 19}, {"score": "2-0", "prob": 14}],
        recommendation={"tier": "A", "is_recommended": True, "bet_confidence": 72, "risk_level": "medium", "risk_tags": []},
    )
    gated = predict.apply_two_one_home_hard_no_bet_gate(row)

    assert gated["recommendation"]["tier"] == "A"
    assert gated["recommendation"].get("is_recommended") is True
    assert "two_one_home_hard_no_bet_gate_applied" not in gated.get("validation_warnings", [])
