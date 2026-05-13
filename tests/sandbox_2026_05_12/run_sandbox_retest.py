from __future__ import annotations
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "sandbox_2026_05_12"
ACTUAL_CSV = OUT_DIR / "actual_results.csv"
RETEST_JSON = OUT_DIR / "sandbox_retest_predictions.json"
SCORED_CSV = OUT_DIR / "sandbox_retest_scored.csv"
REPORT_MD = OUT_DIR / "sandbox_retest_report.md"

def direction_from_score(score: str) -> str:
    m = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", str(score))
    if not m:
        return "unknown"
    h, a = int(m.group(1)), int(m.group(2))
    if h > a:
        return "home"
    if h < a:
        return "away"
    return "draw"

def total_goals(score: str) -> int:
    m = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", str(score))
    if not m:
        return -1
    return int(m.group(1)) + int(m.group(2))

def btts_from_score(score: str) -> str:
    m = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", str(score))
    if not m:
        return "unknown"
    return "yes" if int(m.group(1)) > 0 and int(m.group(2)) > 0 else "no"

def goal_band_from_total(t: int) -> str:
    if t <= 1:
        return "0-1"
    if t <= 3:
        return "2-3"
    return "4+"

def load_actual() -> Dict[str, Dict[str, Any]]:
    rows = {}
    with ACTUAL_CSV.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["match_code"]] = r
    return rows

def score_rows(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    actual = load_actual()
    out = []
    for p in preds:
        code = str(p.get("match_code"))
        a = actual.get(code)
        final = p.get("final", {}) or {}
        pred_score = str(final.get("score", ""))
        pred_dir = final.get("direction") or direction_from_score(pred_score)
        pred_total = total_goals(pred_score)
        pred_band = final.get("goal_band") or goal_band_from_total(pred_total)
        pred_btts = final.get("btts") or btts_from_score(pred_score)

        if not a:
            continue
        row = {
            "match_code": code,
            "league": p.get("league", a.get("league")),
            "home_team": p.get("home_team", a.get("home_team")),
            "away_team": p.get("away_team", a.get("away_team")),
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
            "confidence": final.get("confidence", ""),
            "reason": final.get("reason", "")
        }
        out.append(row)
    return out

def write_scored(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise SystemExit("No scored rows")
    with SCORED_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def pct(n, d):
    return 0.0 if d == 0 else round(n / d * 100, 2)

def write_report(rows: List[Dict[str, Any]]) -> None:
    n = len(rows)
    direction = sum(r["direction_hit"] == "True" for r in rows)
    exact = sum(r["exact_score_hit"] == "True" for r in rows)
    band = sum(r["goal_band_hit"] == "True" for r in rows)
    btts = sum(r["btts_hit"] == "True" for r in rows)

    lines = []
    lines.append("# Sandbox Retest Report — 2026-05-12")
    lines.append("")
    lines.append(f"- Matches: {n}")
    lines.append(f"- Direction hit: {direction}/{n} = {pct(direction,n)}%")
    lines.append(f"- Exact score hit: {exact}/{n} = {pct(exact,n)}%")
    lines.append(f"- Goal band hit: {band}/{n} = {pct(band,n)}%")
    lines.append(f"- BTTS hit: {btts}/{n} = {pct(btts,n)}%")
    lines.append("")
    lines.append("## Match Details")
    lines.append("")
    for r in rows:
        lines.append(
            f"- {r['match_code']} {r['home_team']} vs {r['away_team']}: "
            f"pred {r['predicted_score']} / actual {r['actual_score']} / "
            f"dir_hit={r['direction_hit']} / band_hit={r['goal_band_hit']} / btts_hit={r['btts_hit']}"
        )

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

def main():
    if not RETEST_JSON.exists():
        raise SystemExit(f"Missing prediction file: {RETEST_JSON}")
    preds = json.loads(RETEST_JSON.read_text(encoding="utf-8"))
    if isinstance(preds, dict):
        preds = preds.get("matches", [])
    rows = score_rows(preds)
    write_scored(rows)
    write_report(rows)
    print(REPORT_MD)
    print(SCORED_CSV)

if __name__ == "__main__":
    main()
