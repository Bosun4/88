import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import predict

FIXTURE = ROOT / "tests" / "fixtures" / "upset_sandbox_matches.json"


def _score_dir(score: str) -> str:
    h, a = [int(x) for x in score.split("-")]
    return "home" if h > a else "away" if h < a else "draw"


def _actual_without_leak(match):
    clean = dict(match)
    return clean.pop("actual_score"), clean


def test_fixture_actual_score_field_is_removed_before_evidence_packets():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for idx, match in enumerate(data["matches"], 1):
        _actual, clean = _actual_without_leak(match)
        packet = predict.build_evidence_packet(clean, idx)
        dumped = json.dumps(packet, ensure_ascii=False)
        # Correct-score odds may naturally contain strings like "1-0"; the leak
        # guard is that post-match field names and result labels are absent.
        assert "actual_score" not in dumped
        assert "expected_profile" not in dumped


def test_alaves_barca_style_upset_is_flagged_as_high_away_favorite_upset_risk():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    actual, match = _actual_without_leak(data["matches"][0])
    fair = predict.fair_probs_from_1x2_shadow(match["sp_home"], match["sp_draw"], match["sp_away"])["fair_probs"]
    matrix = predict.build_unified_score_matrix_shadow(match)

    assert actual == "1-0"
    assert fair["away"] >= 65.0
    assert fair["home"] <= 16.0
    assert matrix["direction_probs"]["away"] > matrix["direction_probs"]["home"]
    assert any(row["score"] in {"0-1", "0-2", "1-2"} for row in matrix["top_scores"][:6])


def test_weak_home_2_1_trap_goes_no_bet_but_keeps_score_direction():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    actual, match = _actual_without_leak(data["matches"][1])
    ai = {
        "match": 1,
        "final_direction": "home",
        "predicted_score": "2-1",
        "direction_probs": {"home": 49, "draw": 27, "away": 24},
        "goal_band": "3",
        "btts": "yes",
        "top3": [{"score": "2-1", "prob": 18}],
        "risk_score_candidates": [{"score": "1-2"}, {"score": "2-2"}, {"score": "2-3"}],
        "recommendation": {"tier": "A", "is_recommended": True, "bet_confidence": 76, "risk_tags": []},
        "recommendation_components": {"direction_edge": 75, "score_cluster_strength": 70, "sharp_alignment": 65, "web_source_quality": 60},
        "anchor_audit": {},
        "reason": "weak home 2-1 trap sandbox",
    }
    front = predict.adapt_ai_to_frontend(ai, match)

    assert actual == "2-3"
    assert front["predicted_score"] == "2-1"
    assert front["final_direction"] == "home"
    assert front["recommend_gate_pass"] is False
    assert front["recommendation"]["tier"] == "D"
    assert "two_one_home_hard_no_bet_gate_applied" in front["validation_warnings"]


def test_normal_strong_home_is_not_forced_no_bet_by_tail_gate():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    actual, match = _actual_without_leak(data["matches"][2])
    ai = {
        "match": 1,
        "final_direction": "home",
        "predicted_score": "2-0",
        "direction_probs": {"home": 68, "draw": 20, "away": 12},
        "goal_band": "2",
        "btts": "no",
        "top3": [{"score": "2-0", "prob": 18}, {"score": "2-1", "prob": 12}],
        "recommendation": {"tier": "A", "is_recommended": True, "bet_confidence": 78, "risk_tags": []},
        "recommendation_components": {"direction_edge": 84, "score_cluster_strength": 76, "sharp_alignment": 70, "web_source_quality": 60},
        "anchor_audit": {},
    }
    front = predict.adapt_ai_to_frontend(ai, match)

    assert actual == "2-0"
    assert front["predicted_score"] == "2-0"
    assert front["recommendation"]["tier"] in {"A", "B"}
    assert front["recommend_gate_pass"] is True
