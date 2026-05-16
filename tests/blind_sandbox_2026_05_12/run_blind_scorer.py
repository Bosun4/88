import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "blind_sandbox_2026_05_12"
ACTUAL_CSV = OUT_DIR / "actual_results.csv"
PRED_JSON = OUT_DIR / "blind_predictions.json"
SCORED_CSV = OUT_DIR / "blind_scored.csv"
REPORT_MD = OUT_DIR / "blind_report.md"

def direction_from_score(score: str) -> str:
    m = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", str(score))
    if not m: return "unknown"
    h, a = int(m.group(1)), int(m.group(2))
    if h > a: return "home"
    if h < a: return "away"
    return "draw"

def total_goals(score: str) -> int:
    m = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", str(score))
    if not m: return -1
    return int(m.group(1)) + int(m.group(2))

def btts_from_score(score: str) -> str:
    m = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", str(score))
    if not m: return "unknown"
    return "yes" if int(m.group(1)) > 0 and int(m.group(2)) > 0 else "no"

def goal_band_from_total(t: int) -> str:
    if t <= 1: return "0-1"
    if t <= 3: return "2-3"
    return "4+"

def score_predictions():
    actual = {}
    with ACTUAL_CSV.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            actual[r["match_code"]] = r

    preds_data = json.loads(PRED_JSON.read_text(encoding="utf-8"))
    out = []

    for p in preds_data.get("matches", []):
        code = str(p.get("match_code"))
        a = actual.get(code)
        if not a: continue

        final = p.get("final", {})
        pred_score = str(final.get("score", ""))
        pred_dir = final.get("direction") or direction_from_score(pred_score)
        pred_band = final.get("goal_band") or goal_band_from_total(total_goals(pred_score))
        pred_btts = final.get("btts") or btts_from_score(pred_score)

        out.append({
            "match_code": code,
            "home_team": p.get("home_team"),
            "away_team": p.get("away_team"),
            "predicted_score": pred_score,
            "actual_score": a["actual_score"],
            "predicted_direction": pred_dir,
            "actual_direction": a["actual_direction"],
            "direction_hit": str(pred_dir == a["actual_direction"]),
            "exact_score_hit": str(pred_score == a["actual_score"]),
            "predicted_goal_band": pred_band,
            "actual_goal_band": goal_band_from_total(int(a["total_goals"])),
            "goal_band_hit": str(pred_band == goal_band_from_total(int(a["total_goals"]))),
            "predicted_btts": pred_btts,
            "actual_btts": a["actual_btts"],
            "btts_hit": str(pred_btts == a["actual_btts"]),
            "confidence": final.get("confidence", "")
        })

    with SCORED_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    n = len(out)
    direction = sum(r["direction_hit"] == "True" for r in out)
    exact = sum(r["exact_score_hit"] == "True" for r in out)
    band = sum(r["goal_band_hit"] == "True" for r in out)
    btts = sum(r["btts_hit"] == "True" for r in out)

    pct = lambda n, d: 0.0 if d == 0 else round(n / d * 100, 2)

    md = [
        "# Blind Sandbox Retest Report — 2026-05-12",
        "",
        f"- Matches: {n}",
        f"- Direction hit: {direction}/{n} = {pct(direction,n)}%",
        f"- Exact score hit: {exact}/{n} = {pct(exact,n)}%",
        f"- Goal band hit: {band}/{n} = {pct(band,n)}%",
        f"- BTTS hit: {btts}/{n} = {pct(btts,n)}%",
        "",
        "## Match Details"
    ]

    for r in out:
        md.append(f"- {r['match_code']} {r['home_team']} vs {r['away_team']}: pred {r['predicted_score']} / actual {r['actual_score']} / dir_hit={r['direction_hit']}")

    REPORT_MD.write_text("\n".join(md), encoding="utf-8")

if __name__ == "__main__":
    score_predictions()
