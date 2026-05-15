import json
import csv
import os

def _get_direction(score: str) -> str:
    if not score or "-" not in score: return "unknown"
    try:
        h, a = map(int, score.split("-"))
        if h > a: return "home"
        if a > h: return "away"
        return "draw"
    except:
        return "unknown"

def _get_total_goals(score: str) -> int:
    try:
        h, a = map(int, score.split("-"))
        return h + a
    except:
        return -1

def _get_btts(score: str) -> bool:
    try:
        h, a = map(int, score.split("-"))
        return h > 0 and a > 0
    except:
        return False

def _score_distance(score1: str, score2: str) -> int:
    try:
        h1, a1 = map(int, score1.split("-"))
        h2, a2 = map(int, score2.split("-"))
        return abs(h1 - h2) + abs(a1 - a2)
    except:
        return 999

def score_ledger_with_actuals(ledger_jsonl: str, actual_results_csv: str, output_csv: str, output_md: str):
    actuals = {}
    with open(actual_results_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            actuals[row["match_id"]] = row["actual_score"]
            
    scored_entries = []
    
    with open(ledger_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            entry = json.loads(line)
            match_id = entry.get("match_id")
            
            if match_id in actuals:
                actual = actuals[match_id]
                entry["actual_score"] = actual
                entry["actual_direction"] = _get_direction(actual)
                
                pred = entry.get("predicted_score")
                entry["direction_hit"] = entry.get("final_direction") == entry["actual_direction"]
                entry["exact_score_hit"] = pred == actual
                entry["goal_band_hit"] = _get_total_goals(pred) == _get_total_goals(actual)
                entry["btts_hit"] = _get_btts(pred) == _get_btts(actual)
                
                entry["risk_candidate_covered"] = actual in entry.get("risk_score_candidates", [])
                entry["matrix_top_scores_covered"] = actual in entry.get("matrix_top_scores", [])
                entry["score_cluster_covered"] = actual in entry.get("score_cluster", [])
                
                entry["score_moderation_helped"] = False
                entry["score_moderation_hurt"] = False
                
                if entry.get("score_moderation_applied"):
                    orig = entry.get("original_predicted_score")
                    orig_dist = _score_distance(orig, actual)
                    new_dist = _score_distance(pred, actual)
                    
                    if new_dist < orig_dist or pred == actual:
                        entry["score_moderation_helped"] = True
                    elif orig_dist < new_dist or orig == actual:
                        entry["score_moderation_hurt"] = True
                        
            scored_entries.append(entry)
            
    if not scored_entries: return
    
    keys = scored_entries[0].keys()
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(scored_entries)
        
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("# Forward Ledger Scoring Report\n\n")
        f.write(f"Total Matches Scored: {len([e for e in scored_entries if e.get('actual_score')])}\n")
        # further aggregate logic can be added here
        
    return scored_entries
