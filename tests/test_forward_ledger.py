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


def test_ledger_normalizes_flat_matrix_and_dict_risk_candidates(tmp_path):
    pred_file = tmp_path / "pred_flat.json"
    pred_data = {
        "metadata": {"created_at_utc": "2026-05-15T12:00:00Z", "engine_commit": "abc1234"},
        "predictions": [
            {
                "match_id": "m2",
                "predicted_score": "2-1",
                "final_direction": "home",
                "risk_score_candidates": [{"score": "1-2"}, {"score": "2-2"}, "2-3"],
                "matrix_top_scores": [{"score": "2-1"}, {"score": "2-2"}],
                "matrix_recommended_score": "2-2",
                "matrix_recommended_direction": "draw",
                "matrix_disagreement_flags": {"matrix_draw_risk_warning": True},
                "no_bet_reason": "2-1 hard gate",
                "sub50_tiebreaker_warning": True,
            }
        ]
    }
    pred_file.write_text(json.dumps(pred_data), encoding="utf-8")
    ledger_file = tmp_path / "ledger.jsonl"
    assert create_ledger_from_prediction(str(pred_file), str(ledger_file)) == 1
    entry = json.loads(ledger_file.read_text(encoding="utf-8").splitlines()[0])
    assert entry["risk_score_candidates"] == ["1-2", "2-2", "2-3"]
    assert entry["matrix_top_scores"] == ["2-1", "2-2"]
    assert entry["matrix_recommended_score"] == "2-2"
    assert entry["matrix_recommended_direction"] == "draw"
    assert entry["matrix_disagreement_flags"] == {"matrix_draw_risk_warning": True}
    assert entry["no_bet_reason"] == "2-1 hard gate"
    assert entry["sub50_tiebreaker_warning"] is True

    actual_csv = tmp_path / "actual2.csv"
    with open(actual_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["match_id", "actual_score"])
        writer.writerow(["m2", "2-2"])
    out_csv = tmp_path / "scored2.csv"
    out_md = tmp_path / "scored2.md"
    scored = score_ledger_with_actuals(str(ledger_file), str(actual_csv), str(out_csv), str(out_md))
    assert scored[0]["risk_candidate_covered"] is True
    assert scored[0]["matrix_top_scores_covered"] is True
