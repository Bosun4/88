import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.post_review import score_prediction, review_predictions


def test_score_prediction_classifies_ai_miss_but_gate_blocked():
    row = {
        "match_num": "周六002",
        "league": "澳超",
        "home_team": "奥克兰FC",
        "away_team": "悉尼FC",
        "prediction": {
            "predicted_score": "1-1",
            "final_direction": "draw",
            "recommend_gate_pass": False,
            "recommendation_tier": "C",
            "confidence": 35,
        },
    }
    scored = score_prediction(row, "1-0")
    assert scored["score_hit"] is False
    assert scored["direction_hit"] is False
    assert scored["goal_band_hit"] is False
    assert scored["btts_hit"] is False
    assert scored["classification"] == "ai_miss_but_gate_blocked"


def test_review_predictions_matches_by_team_pair(tmp_path):
    pred_file = tmp_path / "predictions.json"
    actual_file = tmp_path / "actuals.json"
    pred_file.write_text(json.dumps({
        "matches": {
            "today": [{
                "match_num": "周六002",
                "home_team": "奥克兰FC",
                "away_team": "悉尼FC",
                "prediction": {"predicted_score": "1-1", "final_direction": "draw", "recommend_gate_pass": False},
            }]
        }
    }, ensure_ascii=False), encoding="utf-8")
    actual_file.write_text(json.dumps({
        "actuals": [{"home_team": "奥克兰FC", "away_team": "悉尼FC", "actual_score": "1-0"}]
    }, ensure_ascii=False), encoding="utf-8")
    rows = review_predictions(str(pred_file), str(actual_file))
    assert len(rows) == 1
    assert rows[0]["classification"] == "ai_miss_but_gate_blocked"
