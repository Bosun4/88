import json
import os
import csv
from forward_ledger.hash_utils import sha256_file
from forward_ledger.ledger import create_ledger_from_prediction
from forward_ledger.scoring import score_ledger_with_actuals

def test_ledger_pipeline(tmp_path):
    pred_file = tmp_path / "pred.json"
    pred_data = {
        "metadata": {"created_at_utc": "2026-05-15T12:00:00Z", "engine_commit": "abc1234"},
        "predictions": [
            {
                "match_id": "m1",
                "predicted_score": "2-0",
                "final_direction": "home",
                "risk_score_candidates": ["1-0", "3-0"],
                "score_moderation_applied": True,
                "original_predicted_score": "3-0",
                "score_cluster": ["2-0", "3-0"]
            }
        ]
    }
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(pred_data, f)
        
    ledger_file = tmp_path / "ledger.jsonl"
    cnt = create_ledger_from_prediction(str(pred_file), str(ledger_file))
    assert cnt == 1
    
    sha = sha256_file(str(pred_file))
    with open(ledger_file, "r") as f:
        entry = json.loads(f.readline())
        assert entry["prediction_sha256"] == sha
        assert entry["predicted_score"] == "2-0"
        
    actual_csv = tmp_path / "actual.csv"
    with open(actual_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["match_id", "actual_score"])
        writer.writerow(["m1", "2-0"])
        
    out_csv = tmp_path / "scored.csv"
    out_md = tmp_path / "scored.md"
    
    scored = score_ledger_with_actuals(str(ledger_file), str(actual_csv), str(out_csv), str(out_md))
    assert scored[0]["exact_score_hit"] is True
    assert scored[0]["direction_hit"] is True
    assert scored[0]["score_moderation_helped"] is True
    assert scored[0]["score_moderation_hurt"] is False
    assert scored[0]["score_cluster_covered"] is True
