import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import predict


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
