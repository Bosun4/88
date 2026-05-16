from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "legacy" / "v18_1" / "predict_v18_1.py"
OUT_DIR = ROOT / "reports" / "v18_1_matrix_research"

PREFERRED_INPUTS = [
    ROOT / "reports" / "blind_sandbox_2026_05_12" / "prematch_input.json",
    ROOT / "reports" / "post_merge_blind_2026_05_12" / "prematch_input.json",
    ROOT / "reports" / "blind_sandbox_2026_05_12" / "blind_prematch_input.json",
]

PRED_JSON = OUT_DIR / "v18_1_blind_predictions.json"
PRED_SHA = OUT_DIR / "v18_1_blind_predictions.sha256"
ACTUAL_CSV = OUT_DIR / "actual_results.csv"
SCORED_CSV = OUT_DIR / "v18_1_blind_scored.csv"
REPORT_MD = OUT_DIR / "v18_1_blind_report.md"

LEAK_PATTERNS = [
    "actual_score", "actual_results", "final_score",
    "home_goals", "away_goals", "完场", "赛果", "真实赛果",
    "仁川联 0-1", "光州FC 0-1", "塞尔塔 2-3", "利雅胜利 1-1",
    "贝蒂斯 2-1", "圣旺红星 2-3", "南安普敦 1-1", "奥萨苏纳 1-2",
]

ACTUAL_ROWS = [
    ["2001","韩职","仁川联","浦项制铁","0-1","0","1","away","1","no","0-1"],
    ["2002","韩职","光州FC","首尔FC","0-1","0","1","away","1","no","0-0"],
    ["2003","西甲","塞尔塔","莱万特","2-3","2","3","away","5","yes","1-1"],
    ["2004","沙职","利雅胜利","利雅新月","1-1","1","1","draw","2","yes","1-0"],
    ["2005","西甲","贝蒂斯","埃尔切","2-1","2","1","home","3","yes","1-1"],
    ["2006","法甲","圣旺红星","罗德兹","2-3","2","3","away","5","yes","1-1"],
    ["2007","英冠","南安普敦","米堡","1-1","1","1","draw","2","yes","1-1"],
    ["2008","西甲","奥萨苏纳","马竞","1-2","1","2","away","3","yes","0-1"],
]

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def contains_leak(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [p for p in LEAK_PATTERNS if p in text]

def find_input() -> Path:
    for p in PREFERRED_INPUTS:
        if p.exists():
            leaks = contains_leak(p)
            if leaks:
                raise SystemExit(f"Input leak detected in {p}: {leaks}")
            return p
    raise SystemExit("No prematch_input.json found. Need a clean prematch input file.")

def normalize_raw(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict) and isinstance(obj.get("matches"), list):
        return {"matches": obj["matches"]}
    if isinstance(obj, list):
        return {"matches": obj}
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        return {"matches": obj["data"]}
    raise SystemExit("Unsupported prematch input format")

def score_direction(score: str) -> str:
    m = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", str(score))
    if not m: return "unknown"
    h, a = int(m.group(1)), int(m.group(2))
    return "home" if h > a else "away" if h < a else "draw"

def total_goals(score: str) -> int:
    m = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", str(score))
    return int(m.group(1)) + int(m.group(2)) if m else -1

def goal_band(t: int) -> str:
    if t <= 1: return "0-1"
    if t <= 3: return "2-3"
    return "4+"

def btts(score: str) -> str:
    m = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", str(score))
    if not m: return "unknown"
    return "yes" if int(m.group(1)) > 0 and int(m.group(2)) > 0 else "no"

def load_legacy_module():
    spec = importlib.util.spec_from_file_location("predict_v18_1", LEGACY)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def write_actual_after_prediction() -> None:
    with ACTUAL_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["match_code","league","home_team","away_team","actual_score","home_goals","away_goals","actual_direction","total_goals","actual_btts","half_time_score"])
        w.writerows(ACTUAL_ROWS)

def load_actual() -> Dict[str, Dict[str, str]]:
    with ACTUAL_CSV.open("r", encoding="utf-8") as f:
        return {r["match_code"]: r for r in csv.DictReader(f)}

def pct(n: int, d: int) -> float:
    return round(n / d * 100, 2) if d else 0.0

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = find_input()
    raw = normalize_raw(read_json(input_path))
    mod = load_legacy_module()

    preds, top4 = mod.run_predictions(raw, use_ai=False)

    out_matches: List[Dict[str, Any]] = []
    for idx, item in enumerate(preds, 1):
        p = item.get("prediction", {})
        code = str(item.get("match_code") or item.get("match_num") or item.get("num") or item.get("id") or idx)
        # 尽量从球队名顺序补 match_code，避免原始字段缺失
        home = str(item.get("home_team") or item.get("home") or "")
        away = str(item.get("away_team") or item.get("guest") or "")
        name_key = (home, away)
        known = {
            ("仁川联","浦项制铁"): "2001",
            ("光州FC","首尔FC"): "2002",
            ("塞尔塔","莱万特"): "2003",
            ("利雅胜利","利雅新月"): "2004",
            ("贝蒂斯","埃尔切"): "2005",
            ("圣旺红星","罗德兹"): "2006",
            ("南安普敦","米堡"): "2007",
            ("奥萨苏纳","马竞"): "2008",
        }
        code = known.get(name_key, code)
        out_matches.append({
            "match_code": code,
            "league": item.get("league") or item.get("cup") or "",
            "home_team": home,
            "away_team": away,
            "prediction": {
                "predicted_score": p.get("predicted_score"),
                "final_direction": p.get("final_direction"),
                "result": p.get("result"),
                "home_win_pct": p.get("home_win_pct"),
                "draw_pct": p.get("draw_pct"),
                "away_win_pct": p.get("away_win_pct"),
                "confidence": p.get("confidence"),
                "goal_range": p.get("goal_range"),
                "scenario": p.get("scenario"),
                "top_score_candidates": p.get("top_score_candidates"),
                "unified_matrix_top_scores": p.get("unified_matrix_top_scores"),
                "unified_goal_probs": p.get("unified_goal_probs"),
                "crs_shape": p.get("crs_shape"),
                "crs_moments": p.get("crs_moments"),
                "engine_version": p.get("engine_version"),
            }
        })

    payload = {
        "mode": "v18_1_matrix_shadow_use_ai_false",
        "input_path": str(input_path.relative_to(ROOT)),
        "actual_results_visible_during_prediction": False,
        "matches": out_matches,
    }

    PRED_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = hashlib.sha256(PRED_JSON.read_bytes()).hexdigest()
    PRED_SHA.write_text(f"{digest}  {PRED_JSON.name}\n", encoding="utf-8")

    # 预测锁定后才创建赛果
    write_actual_after_prediction()
    actual = load_actual()

    rows: List[Dict[str, Any]] = []
    for m in out_matches:
        code = str(m["match_code"])
        a = actual.get(code)
        if not a: continue

        p = m["prediction"]
        ps = str(p.get("predicted_score") or "")
        pd = str(p.get("final_direction") or score_direction(ps))
        pt = total_goals(ps)

        row = {
            "match_code": code,
            "league": m["league"],
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "predicted_score": ps,
            "actual_score": a["actual_score"],
            "predicted_direction": pd,
            "actual_direction": a["actual_direction"],
            "direction_hit": pd == a["actual_direction"],
            "exact_score_hit": ps == a["actual_score"],
            "predicted_goal_band": goal_band(pt),
            "actual_goal_band": goal_band(int(a["total_goals"])),
            "goal_band_hit": goal_band(pt) == goal_band(int(a["total_goals"])),
            "predicted_btts": btts(ps),
            "actual_btts": a["actual_btts"],
            "btts_hit": btts(ps) == a["actual_btts"],
            "confidence": p.get("confidence"),
            "scenario": p.get("scenario"),
        }
        rows.append(row)

    with SCORED_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    dh = sum(bool(r["direction_hit"]) for r in rows)
    sh = sum(bool(r["exact_score_hit"]) for r in rows)
    gh = sum(bool(r["goal_band_hit"]) for r in rows)
    bh = sum(bool(r["btts_hit"]) for r in rows)

    lines = [
        "# v18.1 Matrix Shadow Blind Report",
        "",
        f"- Mode: use_ai=False",
        f"- Input: {input_path.relative_to(ROOT)}",
        f"- Predictions sha256: {digest}",
        f"- Matches scored: {n}",
        f"- Direction hit: {dh}/{n} = {pct(dh,n)}%",
        f"- Exact score hit: {sh}/{n} = {pct(sh,n)}%",
        f"- Goal band hit: {gh}/{n} = {pct(gh,n)}%",
        f"- BTTS hit: {bh}/{n} = {pct(bh,n)}%",
        "",
        "## Match Details",
        "",
    ]

    for r in rows:
        lines.append(
            f"- {r['match_code']} {r['home_team']} vs {r['away_team']}: "
            f"pred {r['predicted_score']} / actual {r['actual_score']} / "
            f"dir_hit={r['direction_hit']} / band_hit={r['goal_band_hit']} / btts_hit={r['btts_hit']}"
        )

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("input:", input_path)
    print("predictions:", PRED_JSON)
    print("sha256:", digest)
    print("scored:", SCORED_CSV)
    print("report:", REPORT_MD)
    print(f"direction={pct(dh,n)} exact={pct(sh,n)} goal_band={pct(gh,n)} btts={pct(bh,n)}")

if __name__ == "__main__":
    main()
