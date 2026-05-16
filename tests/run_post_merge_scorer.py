import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "post_merge_blind_2026_05_12"
ACTUAL_CSV = OUT_DIR / "actual_results.csv"
PRED_JSON = OUT_DIR / "blind_predictions.json"
SCORED_CSV = OUT_DIR / "blind_scored.csv"
REPORT_MD = OUT_DIR / "post_merge_report.md"

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
        pred_dir = final.get("direction")
        pred_band = final.get("goal_band")
        pred_btts = final.get("btts")

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
            "actual_goal_band": "4+" if int(a["total_goals"])>3 else ("2-3" if int(a["total_goals"])>1 else "0-1"),
            "goal_band_hit": str(pred_band == ("4+" if int(a["total_goals"])>3 else ("2-3" if int(a["total_goals"])>1 else "0-1"))),
            "predicted_btts": pred_btts,
            "actual_btts": a["actual_btts"],
            "btts_hit": str(pred_btts == a["actual_btts"]),
            "confidence": final.get("confidence", ""),
            "risk_score_candidates": json.dumps(final.get("risk_score_candidates", [])),
            "tail_risk_flags": json.dumps(final.get("tail_risk_flags", [])),
            "confidence_downgrade_reason": final.get("confidence_downgrade_reason", "")
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
        "# Post Merge Blind Report — 2026-05-12",
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
        if "塞尔塔" in r["home_team"]:
            md.append(f"  - risk_score_candidates: {r['risk_score_candidates']}")
            md.append(f"  - tail_risk_flags: {r['tail_risk_flags']}")
            md.append(f"  - downgrade_reason: {r['confidence_downgrade_reason']}")

    REPORT_MD.write_text("\n".join(md), encoding="utf-8")

if __name__ == "__main__":
    score_predictions()
