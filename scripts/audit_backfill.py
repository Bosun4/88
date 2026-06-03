#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vMAX backfill audit orchestrator (deterministic, secret-free, no network).

Usage:
  python3 scripts/audit_backfill.py --pred data/history_2026-05-28_today_evening.json \
      --actuals reports/audit_backfill_20260531/actuals_2026-05-28.json \
      --out reports/audit_backfill_20260531/report_2026-05-28.md

Actuals JSON schema (hand/oracle-filled, cross-verified):
  [ {"match_num":"周四001","home_team":"爱尔兰","away_team":"卡塔尔","actual_score":"1-0"}, ... ]
Matching priority: match_num -> home||away -> home_team substring.
Rows with no actual are reported as "pending/unmatched" and excluded from rates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import post_review as PR
import audit_lib as A


def load_pred_rows(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        m = obj.get("matches")
        if isinstance(m, dict):
            for day in m.values():
                if isinstance(day, list):
                    rows.extend(x for x in day if isinstance(x, dict))
        elif isinstance(obj.get("predictions"), list):
            rows = [x for x in obj["predictions"] if isinstance(x, dict)]
    elif isinstance(obj, list):
        rows = [x for x in obj if isinstance(x, dict)]
    return rows


def index_actuals(actuals: List[Dict[str, Any]]) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for a in actuals:
        sc = a.get("actual_score")
        if not sc:
            continue
        for key in (a.get("match_num"), f"{a.get('home_team')}||{a.get('away_team')}"):
            if key:
                idx[str(key)] = sc
    return idx


def find_actual(row: Dict[str, Any], idx: Dict[str, str]) -> Optional[str]:
    keys = [row.get("match_num"), f"{row.get('home_team')}||{row.get('away_team')}"]
    for k in keys:
        if k and str(k) in idx:
            return idx[str(k)]
    return None


def is_featured(row: Dict[str, Any]) -> bool:
    return bool(row.get("is_recommended") or row.get("is_top4_candidate") or row.get("is_strict_recommended"))


def run(pred_path: Path, actuals_path: Path) -> Dict[str, Any]:
    rows = load_pred_rows(pred_path)
    idx = index_actuals(json.loads(actuals_path.read_text(encoding="utf-8")))
    scored: List[Dict[str, Any]] = []
    unmatched: List[str] = []
    for row in rows:
        actual = find_actual(row, idx)
        if not actual:
            unmatched.append(f"{row.get('match_num','?')} {row.get('home_team')} vs {row.get('away_team')}")
            continue
        pred = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
        s = PR.score_prediction(row, actual)
        A.enrich_scored_row(s, pred)
        s["featured"] = is_featured(row)
        s["confidence"] = pred.get("confidence")
        scored.append(s)
    return {
        "scored": scored,
        "unmatched": unmatched,
        "overall": A.summarize(scored),
        "by_tier": A.group_by(scored, "recommendation_tier"),
        "by_league": A.group_by(scored, "league"),
        "by_featured": A.group_by(scored, "featured"),
        "calibration": A.calibration_table(scored),
        "ece": A.ece(scored),
    }


def md_report(date_label: str, res: Dict[str, Any]) -> str:
    o = res["overall"]
    L: List[str] = [f"# vMAX 回溯审计 · {date_label}", ""]
    if o.get("count"):
        L += [
            f"- 样本场次: {o['count']}",
            f"- 方向命中率: {o['direction_hit_rate']*100:.1f}%",
            f"- 精确比分命中率: {o['exact_score_hit_rate']*100:.1f}%",
            f"- 进球带命中率: {o['goal_band_hit_rate']*100:.1f}%",
            f"- BTTS命中率: {o['btts_hit_rate']*100:.1f}%",
            f"- 平均RPS(越低越好): {o['mean_rps']}",
            f"- 平均Brier(越低越好): {o['mean_brier']}",
            f"- 信心校准误差ECE(越低越好): {res['ece']}",
            "",
        ]
    L += ["## 逐场明细", "", "| 场次 | 对阵 | 预测 | 实际 | 方向 | 比分 | 层 | 信心 | 精选 | RPS | 分类 |", "|---|---|---|---|:--:|:--:|---|---|:--:|---|---|"]
    for s in res["scored"]:
        L.append("| {mn} | {h} vs {a} | {ps} | {as_} | {dh} | {sh} | {t} | {c} | {f} | {r} | {cl} |".format(
            mn=s.get("match_num", ""), h=s.get("home_team"), a=s.get("away_team"),
            ps=s.get("predicted_score"), as_=s.get("actual_score"),
            dh="✅" if s.get("direction_hit") else "❌", sh="✅" if s.get("score_hit") else "❌",
            t=s.get("recommendation_tier"), c=s.get("confidence"),
            f="⭐" if s.get("featured") else "", r=s.get("rps"), cl=s.get("classification")))
    L += ["", "## 按推荐层", "", "| 层 | n | 方向 | 比分 | RPS |", "|---|---|---|---|---|"]
    for k, v in res["by_tier"].items():
        L.append(f"| {k} | {v['count']} | {v['direction_hit_rate']*100:.0f}% | {v['exact_score_hit_rate']*100:.0f}% | {v.get('mean_rps')} |")
    L += ["", "## 精选 vs 非精选", "", "| 精选 | n | 方向 | 比分 | RPS |", "|---|---|---|---|---|"]
    for k, v in res["by_featured"].items():
        L.append(f"| {'⭐是' if k=='True' else '否'} | {v['count']} | {v['direction_hit_rate']*100:.0f}% | {v['exact_score_hit_rate']*100:.0f}% | {v.get('mean_rps')} |")
    L += ["", "## 信心校准表(宣称信心 vs 实际命中)", "", "| 信心档 | n | 平均宣称 | 实际方向命中% | 校准缺口 |", "|---|---|---|---|---|"]
    for r in res["calibration"]:
        L.append(f"| {r['confidence_band']} | {r['n']} | {r['avg_claimed_confidence']} | {r['actual_direction_hit_rate_pct']}% | {r['calibration_gap_pct']:+.1f} |")
    if res["unmatched"]:
        L += ["", "## 未匹配/待补赛果", ""] + [f"- {u}" for u in res["unmatched"]]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--actuals", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    res = run(Path(args.pred), Path(args.actuals))
    label = args.label or Path(args.pred).stem
    md = md_report(label, res)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"report -> {args.out}")
    else:
        print(md)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
