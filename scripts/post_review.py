#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-match review utilities for the vMAX prediction loop.

This module is intentionally deterministic and secret-free.  It scores saved
predictions against final scores and classifies whether the recommendation gate
helped, hurt, or correctly allowed a hit.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

VALID_DIRS = {"home", "draw", "away"}


def parse_score(score: Any) -> Optional[Tuple[int, int]]:
    text = str(score or "").strip()
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def direction_from_score(score: Any) -> str:
    parsed = parse_score(score)
    if not parsed:
        return "unknown"
    home, away = parsed
    if home > away:
        return "home"
    if away > home:
        return "away"
    return "draw"


def goal_band(score: Any) -> str:
    parsed = parse_score(score)
    if not parsed:
        return "unknown"
    total = parsed[0] + parsed[1]
    return "4+" if total >= 4 else str(total)


def btts(score: Any) -> str:
    parsed = parse_score(score)
    if not parsed:
        return "unknown"
    return "yes" if parsed[0] > 0 and parsed[1] > 0 else "no"


def normalize_prediction_match(row: Dict[str, Any]) -> Dict[str, Any]:
    pred = row.get("prediction", row)
    if not isinstance(pred, dict):
        pred = {}
    return {
        "match_id": row.get("match_id") or row.get("id") or row.get("match_num") or pred.get("match"),
        "match_num": row.get("match_num", ""),
        "league": row.get("league", ""),
        "home_team": row.get("home_team") or row.get("home") or row.get("home_name", ""),
        "away_team": row.get("away_team") or row.get("away") or row.get("away_name", ""),
        "prediction": pred,
    }


def score_prediction(prediction_row: Dict[str, Any], actual_score: str, actual_direction: Optional[str] = None) -> Dict[str, Any]:
    item = normalize_prediction_match(prediction_row)
    pred = item["prediction"]
    pred_score = str(pred.get("predicted_score", "")).strip()
    pred_direction = str(pred.get("final_direction") or direction_from_score(pred_score)).strip().lower()
    actual_direction = (actual_direction or direction_from_score(actual_score)).strip().lower()

    score_hit = bool(pred_score and pred_score == actual_score)
    direction_hit = bool(pred_direction in VALID_DIRS and pred_direction == actual_direction)
    goal_band_hit = goal_band(pred_score) == goal_band(actual_score)
    btts_hit = btts(pred_score) == btts(actual_score)
    gate_pass = bool(pred.get("recommend_gate_pass"))
    tier = str(pred.get("recommendation_tier") or (pred.get("recommendation") or {}).get("tier") or "D").upper()
    confidence = pred.get("confidence")
    display_confidence = pred.get("display_confidence", confidence)

    if direction_hit and gate_pass:
        classification = "effective_recommended_hit"
    elif (not direction_hit) and gate_pass:
        classification = "serious_gate_pass_miss"
    elif direction_hit and not gate_pass:
        classification = "over_conservative_gate_blocked_hit"
    else:
        classification = "ai_miss_but_gate_blocked"

    return {
        **{k: item[k] for k in ["match_id", "match_num", "league", "home_team", "away_team"]},
        "predicted_score": pred_score,
        "actual_score": actual_score,
        "predicted_direction": pred_direction,
        "actual_direction": actual_direction,
        "score_hit": score_hit,
        "direction_hit": direction_hit,
        "goal_band_hit": goal_band_hit,
        "btts_hit": btts_hit,
        "recommend_gate_pass": gate_pass,
        "recommendation_tier": tier,
        "confidence": confidence,
        "display_confidence": display_confidence,
        "classification": classification,
    }


def _load_prediction_rows(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("matches"), dict):
        rows: List[Dict[str, Any]] = []
        for day_rows in obj["matches"].values():
            if isinstance(day_rows, list):
                rows.extend(x for x in day_rows if isinstance(x, dict))
        return rows
    if isinstance(obj, dict) and isinstance(obj.get("predictions"), list):
        return [x for x in obj["predictions"] if isinstance(x, dict)]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def _load_actuals(path: Path) -> Dict[str, Dict[str, str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("actuals", obj) if isinstance(obj, dict) else obj
    actuals: Dict[str, Dict[str, str]] = {}
    if not isinstance(rows, list):
        return actuals
    for row in rows:
        if not isinstance(row, dict):
            continue
        score = row.get("actual_score") or row.get("score")
        if not score:
            continue
        keys = [row.get("match_id"), row.get("match_num")]
        if row.get("home_team") and row.get("away_team"):
            keys.append(f"{row.get('home_team')}||{row.get('away_team')}")
        for key in keys:
            if key:
                actuals[str(key)] = {"actual_score": str(score), "actual_direction": str(row.get("actual_direction") or "")}
    return actuals


def review_predictions(predictions_path: str, actuals_path: str) -> List[Dict[str, Any]]:
    rows = _load_prediction_rows(Path(predictions_path))
    actuals = _load_actuals(Path(actuals_path))
    scored: List[Dict[str, Any]] = []
    for row in rows:
        item = normalize_prediction_match(row)
        keys = [item.get("match_id"), item.get("match_num"), f"{item.get('home_team')}||{item.get('away_team')}"]
        actual = next((actuals[str(k)] for k in keys if k and str(k) in actuals), None)
        if actual:
            scored.append(score_prediction(row, actual["actual_score"], actual.get("actual_direction") or None))
    return scored


def main() -> int:
    ap = argparse.ArgumentParser(description="Score vMAX predictions against actual results")
    ap.add_argument("predictions")
    ap.add_argument("actuals")
    ap.add_argument("--output", "-o", default="")
    args = ap.parse_args()
    scored = review_predictions(args.predictions, args.actuals)
    text = json.dumps({"count": len(scored), "reviews": scored}, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
