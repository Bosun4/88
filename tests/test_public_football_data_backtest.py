import csv
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import predict

DATA_URLS = [
    "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2526/D1.csv",
    "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "https://www.football-data.co.uk/mmz4281/2526/F1.csv",
]
CACHE = ROOT / ".cache" / "football_data_2526"
REPORT = ROOT / "reports" / "sandbox" / "public_football_data_backtest.json"


def _fetch_csv(url: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / Path(url).name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=30) as resp:
            path.write_bytes(resp.read())
    return path.read_text(encoding="utf-8-sig", errors="ignore")


def _score_dir(h: int, a: int) -> str:
    return "home" if h > a else "away" if h < a else "draw"


def _actual_goal_band(total: int) -> str:
    if total <= 1:
        return "0-1"
    if total <= 3:
        return "2-3"
    return "4+"


def _rows(limit_per_file=220):
    rows = []
    for url in DATA_URLS:
        text = _fetch_csv(url)
        for r in csv.DictReader(io.StringIO(text)):
            if not r.get("HomeTeam") or not r.get("AwayTeam"):
                continue
            try:
                h, a = int(r["FTHG"]), int(r["FTAG"])
                avg_h, avg_d, avg_a = float(r["AvgH"]), float(r["AvgD"]), float(r["AvgA"])
            except Exception:
                continue
            rows.append({
                "league": r.get("Div"),
                "home_team": r["HomeTeam"],
                "away_team": r["AwayTeam"],
                "sp_home": avg_h,
                "sp_draw": avg_d,
                "sp_away": avg_a,
                "actual_score": f"{h}-{a}",
                "actual_direction": _score_dir(h, a),
                "actual_goal_band": _actual_goal_band(h + a),
            })
            if len([x for x in rows if x.get("league") == r.get("Div")]) >= limit_per_file:
                break
    return rows


def test_public_football_data_matrix_shadow_backtest_smoke():
    rows = _rows()
    assert len(rows) >= 1000
    metrics = {
        "n": 0,
        "direction_hit": 0,
        "goal_band_hit": 0,
        "exact_hit": 0,
        "heavy_fav_upsets": 0,
        "heavy_fav_upset_flagged_by_low_home_fair_or_matrix": 0,
    }
    sample_errors = []
    for row in rows[:1100]:
        actual_score = row.pop("actual_score")
        actual_direction = row.pop("actual_direction")
        actual_goal_band = row.pop("actual_goal_band")
        matrix = predict.build_unified_score_matrix_shadow(row)
        pred_score = matrix["top_scores"][0]["score"]
        pred_direction = predict._score_direction(pred_score)
        pred_goal_band = predict._score_goal_band(pred_score)
        fair = predict.fair_probs_from_1x2_shadow(row["sp_home"], row["sp_draw"], row["sp_away"])["fair_probs"]
        metrics["n"] += 1
        metrics["direction_hit"] += int(pred_direction == actual_direction)
        metrics["goal_band_hit"] += int(pred_goal_band == actual_goal_band)
        metrics["exact_hit"] += int(pred_score == actual_score)
        if fair["away"] >= 62 and actual_direction == "home":
            metrics["heavy_fav_upsets"] += 1
            # This is a risk-audit smoke metric, not a guarantee: heavy away favorite
            # upsets should at least be visible as low home probability, while the
            # matrix remains market-shaped and normally leans away.
            metrics["heavy_fav_upset_flagged_by_low_home_fair_or_matrix"] += int(fair["home"] <= 18 or matrix["direction_probs"].get("home", 0) <= 20)
        if pred_score != actual_score and len(sample_errors) < 12:
            sample_errors.append({
                "match": f"{row['home_team']} vs {row['away_team']}",
                "odds": [row["sp_home"], row["sp_draw"], row["sp_away"]],
                "pred_score": pred_score,
                "actual_score": actual_score,
                "matrix_dir": matrix["direction_probs"],
            })
        row["actual_score"] = actual_score
        row["actual_direction"] = actual_direction
        row["actual_goal_band"] = actual_goal_band
    metrics["direction_hit_rate"] = round(metrics["direction_hit"] / metrics["n"], 4)
    metrics["goal_band_hit_rate"] = round(metrics["goal_band_hit"] / metrics["n"], 4)
    metrics["exact_hit_rate"] = round(metrics["exact_hit"] / metrics["n"], 4)
    metrics["sample_errors"] = sample_errors
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    assert metrics["n"] >= 1000
    # Market-only score-matrix shadow is a diagnostic baseline, not a perfect
    # score predictor. These floors intentionally encode current observed public
    # data behavior while guarding against broken odds parsing/regressions.
    assert metrics["direction_hit_rate"] >= 0.35
    assert metrics["goal_band_hit_rate"] >= 0.08
    assert metrics["exact_hit_rate"] >= 0.08
    if metrics["heavy_fav_upsets"]:
        assert metrics["heavy_fav_upset_flagged_by_low_home_fair_or_matrix"] == metrics["heavy_fav_upsets"]
