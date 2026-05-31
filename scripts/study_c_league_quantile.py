#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STUDY C — 相对联赛分位 vs 绝对阈值 (Per-League Quantile Signal)

用户方法学: 同样 a4=4.2, 在德甲(常态偏大)和西甲(常态偏小)含义完全不同.
绝对阈值会毒化跨联赛统计. 应改用"相对该联赛自身 a4 分布的压缩分位".

本研究验证: 把每场的 a4 换算成"它在本联赛 a4 分布里的分位"(越低=越被压),
再看"相对分位低" 是否比 "绝对a4低" 更能拎出真实大球.
口径: actual_bigball = 真实总进球>=4. 联赛样本<8 的归入风格档合并求分布.
无网络无密钥.
"""
from __future__ import annotations
import json, statistics as st
from collections import defaultdict

M = json.load(open("reports/audit_backfill_20260531/matched_samples.json", encoding="utf-8"))
M = [r for r in M if r.get("a4") and r.get("actual_total") is not None]

# 风格档(给薄样本联赛兜底)
BIG = {"挪超","瑞超","芬超","荷甲","荷乙","德甲","德乙","美职","沙职","解放者杯","澳超","日职"}
SMALL = {"英超","英冠","意甲","西甲","法甲","葡超","欧协联","欧冠","欧罗巴","意大利杯"}
def style(lg):
    if lg in BIG: return "BIG档"
    if lg in SMALL: return "SMALL档"
    return "NEU档"

# 1. 建每联赛 a4 分布(>=8场单列, 否则并入风格档)
from collections import Counter
cnt = Counter(str(r["league"]) for r in M)
def group_key(lg):
    return lg if cnt[lg] >= 8 else style(lg)

dist = defaultdict(list)
for r in M:
    dist[group_key(str(r["league"]))].append(r["a4"])
for k in dist: dist[k].sort()

def pct_rank(val, arr):
    # val 在 arr 中的分位(0=最低/最被压, 1=最高)
    below = sum(1 for x in arr if x < val)
    return below / len(arr)

for r in M:
    gk = group_key(str(r["league"]))
    r["a4_qtile"] = pct_rank(r["a4"], dist[gk])
    r["gk"] = gk

base = sum(1 for r in M if r["actual_bigball"]) / len(M)
print(f"=== STUDY C: 相对联赛分位 vs 绝对阈值 ===")
print(f"样本 {len(M)}, 基础大球率 {base*100:.1f}%\n")

def report(name, sel):
    if not sel: print(f"  {name}: 空"); return
    big = sum(1 for r in sel if r["actual_bigball"])
    print(f"  {name}: {big}/{len(sel)} = {big/len(sel)*100:.1f}%  (lift {(big/len(sel)-base)*100:+.1f}pp)")

print("=== 绝对阈值 (现状口径) ===")
report("a4<=4.2", [r for r in M if r["a4"]<=4.2])
report("a4<=4.5", [r for r in M if r["a4"]<=4.5])

print("\n=== 相对联赛分位 (越低=本联赛里越被压) ===")
report("a4分位<=20% (本联赛最被压的1/5)", [r for r in M if r["a4_qtile"]<=0.20])
report("a4分位<=33%", [r for r in M if r["a4_qtile"]<=0.33])
report("a4分位<=50%", [r for r in M if r["a4_qtile"]<=0.50])

print("\n=== 关键对照: 绝对低 但 联赛分位不低 (疑似诱盘) ===")
trap = [r for r in M if r["a4"]<=4.5 and r["a4_qtile"]>0.5]
report("a4<=4.5 但分位>50% (该联赛本就常压低)", trap)
gold = [r for r in M if r["a4_qtile"]<=0.33 and r["a4"]<=5.0]
report("分位<=33% 且 a4<=5.0 (相对+绝对双确认)", gold)

print("\n=== 各组联赛 a4 基线(中位/最低四分位) ===")
for gk in sorted(dist, key=lambda k:-len(dist[k])):
    arr = dist[gk]
    if len(arr)<4: continue
    print(f"  {gk:>8} n={len(arr):>2}  a4中位={st.median(arr):.2f}  最低1/4线={arr[len(arr)//4]:.2f}  最低值={arr[0]:.2f}")
