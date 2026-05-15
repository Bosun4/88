import json
import os
from datetime import datetime, timezone
from dataclasses import asdict
from forward_ledger.hash_utils import sha256_file
from forward_ledger.schema import LedgerEntry

def create_ledger_from_prediction(prediction_json: str, output_jsonl: str):
    sha = sha256_file(prediction_json)
    with open(prediction_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    meta = data.get("metadata", {})
    created_at = meta.get("created_at_utc", datetime.now(timezone.utc).isoformat())
    engine_commit = meta.get("engine_commit", "unknown")
    preds = data.get("predictions", [])
    match_count = len(preds)
    
    entries = []
    for p in preds:
        matrix = p.get("matrix_shadow_layer", {})
        
        # safely extract candidates
        risk_cands = p.get("risk_score_candidates", [])
        if not isinstance(risk_cands, list): risk_cands = []
        
        flags = p.get("tail_risk_flags", [])
        if not isinstance(flags, list): flags = []
        
        top_matrix = [m.get("score") for m in matrix.get("matrix_top_scores", []) if isinstance(m, dict)]
        
        entry = LedgerEntry(
            prediction_file=os.path.basename(prediction_json),
            prediction_sha256=sha,
            created_at_utc=created_at,
            match_count=match_count,
            engine_commit=engine_commit,
            match_id=p.get("match_id", ""),
            home_team=p.get("home_team", ""),
            away_team=p.get("away_team", ""),
            predicted_score=p.get("predicted_score", ""),
            final_direction=p.get("final_direction", ""),
            confidence=p.get("confidence", 0),
            home_win_pct=p.get("probabilities", {}).get("home", 0.0),
            draw_pct=p.get("probabilities", {}).get("draw", 0.0),
            away_win_pct=p.get("probabilities", {}).get("away", 0.0),
            
            risk_score_candidates=risk_cands,
            tail_risk_flags=flags,
            
            matrix_recommended_score=matrix.get("recommended_score"),
            matrix_recommended_direction=matrix.get("recommended_direction"),
            matrix_top_scores=top_matrix,
            matrix_disagreement_flags=matrix.get("disagreement_flags", []),
            
            sub50_tiebreaker_warning=p.get("sub50_tiebreaker_warning", False),
            no_bet_reason=p.get("no_bet_reason"),
            score_cluster=p.get("score_cluster", []),
            
            score_moderation_applied=p.get("score_moderation_applied", False),
            original_predicted_score=p.get("original_predicted_score")
        )
        entries.append(entry)
        
    with open(output_jsonl, 'a', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
            
    return len(entries)
