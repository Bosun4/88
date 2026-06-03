#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vMAX audit library (deterministic, secret-free).

Pure scoring/metric functions layered on top of post_review.score_prediction.
Adds probability-quality metrics used by the betting-model evaluation literature:
  - RPS (Ranked Probability Score) for ordered 1X2 outcomes (Constantinou & Fenton 2012)
  - Brier score (multiclass) + reliability/ECE buckets
  - Confidence calibration table (does AI-confidence match real hit rate?)
  - Grouped hit-rate breakdowns (league / tier / featured)

This module deliberately performs NO network IO and reads NO secrets.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---- ordered outcome helpers -------------------------------------------------
DIR_INDEX = {"home": 0, "draw": 1, "away": 2}


def _norm_probs(h: Any, d: Any, a: Any) -> Optional[Tuple[float, float, float]]:
    try:
        vals = [float(h), float(d), float(a)]
    except (TypeError, ValueError):
        return None
    s = sum(vals)
    if s <= 0:
        return None
    # accept either 0-1 or 0-100 inputs
    if s > 1.5:
        vals = [v / 100.0 for v in vals]
        s = sum(vals)
    return tuple(v / s for v in vals)  # type: ignore


def rps_1x2(probs: Tuple[float, float, float], actual_dir: str) -> Optional[float]:
    """Ranked Probability Score for 3 ordered outcomes (home<draw<away order).

    Lower is better. Range [0,1]. Uses the standard cumulative formulation.
    """
    if actual_dir not in DIR_INDEX:
        return None
    outcome = [0.0, 0.0, 0.0]
    outcome[DIR_INDEX[actual_dir]] = 1.0
    cum_p = 0.0
    cum_o = 0.0
    total = 0.0
    for i in range(2):  # r-1 cumulative terms
        cum_p += probs[i]
        cum_o += outcome[i]
        total += (cum_p - cum_o) ** 2
    return total / 2.0


def brier_1x2(probs: Tuple[float, float, float], actual_dir: str) -> Optional[float]:
    if actual_dir not in DIR_INDEX:
        return None
    outcome = [0.0, 0.0, 0.0]
    outcome[DIR_INDEX[actual_dir]] = 1.0
    return sum((probs[i] - outcome[i]) ** 2 for i in range(3))


def extract_probs(pred: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """Prefer explicit 1X2 percentages; fall back to matrix_direction_probs."""
    p = _norm_probs(pred.get("home_win_pct"), pred.get("draw_pct"), pred.get("away_win_pct"))
    if p:
        return p
    m = pred.get("matrix_direction_probs") or {}
    if isinstance(m, dict):
        return _norm_probs(m.get("home"), m.get("draw"), m.get("away"))
    return None


# ---- aggregate metrics -------------------------------------------------------
def summarize(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(scored)
    if n == 0:
        return {"count": 0}
    dir_hits = sum(1 for s in scored if s.get("direction_hit"))
    score_hits = sum(1 for s in scored if s.get("score_hit"))
    band_hits = sum(1 for s in scored if s.get("goal_band_hit"))
    btts_hits = sum(1 for s in scored if s.get("btts_hit"))
    rps_vals = [s["rps"] for s in scored if s.get("rps") is not None]
    brier_vals = [s["brier"] for s in scored if s.get("brier") is not None]
    return {
        "count": n,
        "direction_hit_rate": round(dir_hits / n, 4),
        "exact_score_hit_rate": round(score_hits / n, 4),
        "goal_band_hit_rate": round(band_hits / n, 4),
        "btts_hit_rate": round(btts_hits / n, 4),
        "mean_rps": round(sum(rps_vals) / len(rps_vals), 4) if rps_vals else None,
        "mean_brier": round(sum(brier_vals) / len(brier_vals), 4) if brier_vals else None,
    }


def group_by(scored: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for s in scored:
        buckets.setdefault(str(s.get(key, "?")), []).append(s)
    return {k: summarize(v) for k, v in sorted(buckets.items())}


def calibration_table(scored: List[Dict[str, Any]], bins: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Bucket by display/confidence; compare claimed confidence vs real direction hit-rate."""
    bins = bins or [0, 40, 50, 60, 70, 80, 101]
    out: List[Dict[str, Any]] = []
    for lo, hi in zip(bins, bins[1:]):
        members = [s for s in scored if isinstance(s.get("confidence"), (int, float)) and lo <= s["confidence"] < hi]
        if not members:
            continue
        n = len(members)
        hit = sum(1 for s in members if s.get("direction_hit"))
        avg_conf = sum(s["confidence"] for s in members) / n
        out.append({
            "confidence_band": f"{lo}-{hi - 1}",
            "n": n,
            "avg_claimed_confidence": round(avg_conf, 1),
            "actual_direction_hit_rate_pct": round(hit / n * 100, 1),
            "calibration_gap_pct": round(avg_conf - hit / n * 100, 1),
        })
    return out


def ece(scored: List[Dict[str, Any]], bins: Optional[List[int]] = None) -> Optional[float]:
    """Expected Calibration Error over confidence bands (weighted by n)."""
    table = calibration_table(scored, bins)
    if not table:
        return None
    total = sum(r["n"] for r in table)
    return round(sum(r["n"] * abs(r["calibration_gap_pct"]) for r in table) / total / 100.0, 4)


def enrich_scored_row(scored_row: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    """Attach RPS/Brier + featured flag to a post_review scored row."""
    probs = extract_probs(pred)
    adir = scored_row.get("actual_direction")
    scored_row["rps"] = rps_1x2(probs, adir) if probs and adir in DIR_INDEX else None
    scored_row["brier"] = brier_1x2(probs, adir) if probs and adir in DIR_INDEX else None
    if scored_row["rps"] is not None:
        scored_row["rps"] = round(scored_row["rps"], 4)
    if scored_row["brier"] is not None:
        scored_row["brier"] = round(scored_row["brier"], 4)
    return scored_row
