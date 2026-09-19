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

try:
    from .fixture_identity import dedupe_predictions, fixture_key, match_actual, prediction_rows
    from .metrics_ledger import score_channels
except ImportError:
    from fixture_identity import dedupe_predictions, fixture_key, match_actual, prediction_rows
    from metrics_ledger import score_channels

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
    main_score, side_scores = score_channels(pred)
    pred_score = main_score or ""
    pred_direction = str(pred.get("final_direction") or direction_from_score(pred_score)).strip().lower()
    actual_direction = (actual_direction or direction_from_score(actual_score)).strip().lower()

    score_hit = bool(pred_score and pred_score == actual_score)
    direction_hit = (pred_direction == actual_direction) if pred_direction in VALID_DIRS and actual_direction in VALID_DIRS else None
    goal_band_hit = (goal_band(pred_score) == goal_band(actual_score)) if main_score and parse_score(actual_score) else None
    btts_hit = (btts(pred_score) == btts(actual_score)) if main_score and parse_score(actual_score) else None
    gate_pass = bool(pred.get("recommend_gate_pass"))
    tier = str(pred.get("recommendation_tier") or (pred.get("recommendation") or {}).get("tier") or "D").upper()
    gate_pass = gate_pass and tier in {"S", "A", "B"} and not pred.get("is_abstain")
    confidence = pred.get("confidence")
    display_confidence = pred.get("display_confidence", confidence)

    if direction_hit is None:
        classification = "analysis_unavailable"
    elif direction_hit and gate_pass:
        classification = "effective_recommended_hit"
    elif (not direction_hit) and gate_pass:
        classification = "serious_gate_pass_miss"
    elif direction_hit and not gate_pass:
        classification = "over_conservative_gate_blocked_hit"
    else:
        classification = "ai_miss_but_gate_blocked"

    return {
        **{k: item[k] for k in ["match_id", "match_num", "league", "home_team", "away_team"]},
        "fixture_key": fixture_key(prediction_row),
        "settlement_status": "settled",
        "evaluation_scope": "retrospective_unlocked",
        "side_risk_scores": side_scores,
        "side_risk_hit": actual_score in side_scores if side_scores else None,
        "main_score_available": bool(main_score),
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


def _load_prediction_rows(path):
    return prediction_rows(json.loads(path.read_text(encoding="utf-8")))


def _load_actuals(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("actuals", obj) if isinstance(obj, dict) else obj
    return [r for r in rows if isinstance(r, dict) and parse_score(r.get('actual_score'))] if isinstance(rows, list) else []


def review_predictions(predictions_path, actuals_path):
    rows = dedupe_predictions(_load_prediction_rows(Path(predictions_path)))
    actuals = _load_actuals(Path(actuals_path))
    scored = []
    for row in rows:
        actual = match_actual(row, actuals)
        if actual:
            scored.append(score_prediction(row, actual['actual_score']))
        else:
            item = normalize_prediction_match(row)
            scored.append({**{k: v for k, v in item.items() if k != 'prediction'},
                           'fixture_key': fixture_key(row), 'settlement_status': 'unresolved',
                           'unresolved_reason': 'missing_or_ambiguous_event_result',
                           'evaluation_scope': 'retrospective_unlocked'})
    return scored


def summarize_reviews(reviews):
    settled = [r for r in reviews if r.get('settlement_status') == 'settled']
    def metric(rows, key):
        rows = [r for r in rows if r.get(key) is not None]
        hits = sum(bool(r[key]) for r in rows)
        return {'samples': len(rows), 'hits': hits,
                'accuracy_pct': round(hits / len(rows) * 100, 1) if rows else None}
    d_rows = [r for r in settled if r.get('recommendation_tier') == 'D']
    return {'settled': len(settled), 'unresolved': len(reviews) - len(settled),
            'direction': metric(settled, 'direction_hit'),
            'main_score': metric([r for r in settled if r.get('main_score_available')], 'score_hit'),
            'side_risk_score': metric(settled, 'side_risk_hit'),
            'risk_d': {**metric(d_rows, 'direction_hit'),
                       'main_score': metric([r for r in d_rows if r.get('main_score_available')], 'score_hit'),
                       'side_risk_score': metric(d_rows, 'side_risk_hit')},
            'effective_recommended': metric([r for r in settled if r.get('recommend_gate_pass')], 'direction_hit')}

def main() -> int:
    ap = argparse.ArgumentParser(description="Score vMAX predictions against actual results")
    ap.add_argument("predictions")
    ap.add_argument("actuals")
    ap.add_argument("--output", "-o", default="")
    args = ap.parse_args()
    scored = review_predictions(args.predictions, args.actuals)
    text = json.dumps({"count": len(scored), "summary": summarize_reviews(scored), "reviews": scored}, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
