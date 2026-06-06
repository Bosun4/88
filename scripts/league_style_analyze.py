#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authoritative league-style classification (by league field, not team guessing).

Tiers reflect real-world goal-scoring tendencies the user confirmed:
  - ATTACK (大开大合/大球): 德甲/挪超/瑞超/芬超/荷甲/德乙/荷乙/澳超
  - NEUTRAL (中性偏大): 沙职/美职
  - SMALL (小球/防守/传控): 英超/西甲/意甲/法甲/葡超/日职/韩职/解放者杯(南美防守)/欧战/英冠等
Then validates big-ball resonance hit-rate within each tier.
No network. No secrets.
"""
from __future__ import annotations
import json
from collections import defaultdict

M = json.load(open("reports/audit_backfill_20260531/matched_samples.json", encoding="utf-8"))

LEAGUE_STYLE = {
    # 进攻型 / 大开大合
    "德甲": "attack", "德乙": "attack", "挪超": "attack", "瑞超": "attack",
    "芬超": "attack", "荷甲": "attack", "荷乙": "attack", "澳超": "attack",
    # 中性偏大(进球中等偏多,但波动大)
    "沙职": "neutral", "美职": "neutral",
    # 小球 / 防守 / 传控
    "英超": "small", "西甲": "small", "意甲": "small", "法甲": "small", "法乙": "small",
    "葡超": "small", "日职": "small", "韩职": "small", "解放者杯": "small",
    "欧罗巴": "small", "欧协联": "small", "欧冠": "small", "英冠": "small",
    "英甲": "small", "英足总杯": "small", "意大利杯": "small", "亚冠乙": "small",
    "国际赛": "small",
}


def style(r):
    return LEAGUE_STYLE.get(str(r.get("league", "")), "unk")


for r in M:
    r["lstyle"] = style(r)


def stat(rows):
    res2 = [x for x in rows if x.get("bigball_resonance", 0) >= 2]
    res3 = [x for x in rows if x.get("bigball_resonance", 0) >= 3]
    h2 = sum(1 for x in res2 if x["actual_bigball"])
    h3 = sum(1 for x in res3 if x["actual_bigball"])
    return (len(rows), h2, len(res2), h3, len(res3))


print("=== 按联赛风格档位:大球真实命中率 ===")
print(f"{'档位':8s} {'样本':>4s} {'共振>=2大球率':>16s} {'共振=3大球率':>16s}")
groups = defaultdict(list)
for r in M:
    groups[r["lstyle"]].append(r)
for tier in ["attack", "neutral", "small", "unk"]:
    rows = groups[tier]
    if not rows:
        continue
    n, h2, n2, h3, n3 = stat(rows)
    p2 = f"{h2}/{n2}={round(h2/n2*100,1)}%" if n2 else "-"
    p3 = f"{h3}/{n3}={round(h3/n3*100,1)}%" if n3 else "-"
    print(f"{tier:8s} {n:>4d} {p2:>18s} {p3:>18s}")

print("\n=== 各联赛细分(仅匹配到赛果的) ===")
byl = defaultdict(list)
for r in M:
    byl[str(r.get("league"))].append(r)
for lg in sorted(byl, key=lambda x: -len(byl[x])):
    rows = byl[lg]
    res2 = [x for x in rows if x.get("bigball_resonance", 0) >= 2]
    if not res2:
        continue
    h = sum(1 for x in res2 if x["actual_bigball"])
    print(f"  {lg:6s}[{LEAGUE_STYLE.get(lg,'?'):7s}] 共振>=2: {h}/{len(res2)} 大球 = {round(h/len(res2)*100,1)}%")

# 进攻型档逐场明细(这是要做硬约束的核心档)
print("\n=== 进攻型档逐场(硬约束目标档) ===")
for r in sorted(groups["attack"], key=lambda x: -x.get("bigball_resonance", 0)):
    flag = "💥大球" if r["actual_bigball"] else "小球"
    print(f"  [{r.get('league')}] {r['home_team']}-{r['away_team']:9s} 共振{r['bigball_resonance']} a4={r['a4']} a7={r['a7']} | 预测{r['predicted_score']} 实际{r['actual']}({r['actual_total']}) {flag}")
