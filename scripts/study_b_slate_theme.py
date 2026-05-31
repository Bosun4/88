#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STUDY B — 版面主题探测 (Slate-Theme Clustering Audit)

问题: 体彩中心选场是否存在"成版大球 / 成版小球"的聚集性?
若真有聚集, 则"当日整版赔率基调"是一个免费的元信号, 可用于调每场先验.

口径:
- 按赛日 (history_*.json 的日期) 聚合每场的真实总进球与大球判定(actual_bigball)
- actual_bigball: 真实总进球 >=4 (沿用 matched_samples 既有口径)
- 检验三件事:
  1. 各赛日大球率的离散度 — 若体彩"成版", 赛日大球率应两极分化(很多接近0或接近高位),
     而非都挤在总体均值附近(那才是随机洒落)
  2. 聚集性统计检验 — 用赛日内方差 vs 二项随机期望方差对比(过度离散 overdispersion)
  3. 版面赔率基调能否预判赛日大球率 — 用赛日 a4 中位数 把赛日分成"偏大版/偏小版",
     看两组真实大球率是否分开

无网络, 无密钥. 纯本地 matched_samples.json.
"""
from __future__ import annotations
import json, math, statistics
from collections import defaultdict

SRC = "reports/audit_backfill_20260531/matched_samples.json"
M = json.load(open(SRC, encoding="utf-8"))

# 赛日 = src 文件里的日期段
def slate_of(r):
    s = r.get("src", "")
    # history_2026-05-08_today_evening.json -> 2026-05-08
    parts = s.split("_")
    return parts[1] if len(parts) > 1 else s

by_day = defaultdict(list)
for r in M:
    if r.get("actual_total") is None:
        continue
    by_day[slate_of(r)].append(r)

print(f"=== STUDY B: 版面主题聚集性 ===")
print(f"赛日数: {len(by_day)} | 总场次(有赛果): {sum(len(v) for v in by_day.values())}\n")

# 1. 各赛日大球率
print("=== 1. 各赛日真实大球率 (>=4球) ===")
rates = []
all_big = 0
all_n = 0
day_stats = []
for day in sorted(by_day):
    g = by_day[day]
    n = len(g)
    big = sum(1 for r in g if r.get("actual_bigball"))
    rate = big / n
    rates.append(rate)
    all_big += big
    all_n += n
    day_stats.append((day, n, big, rate))
    bar = "█" * round(rate * 20)
    print(f"  {day}  n={n:>2}  大球 {big:>2}/{n:<2} = {rate*100:>5.1f}%  {bar}")

base = all_big / all_n
print(f"\n  全样本基础大球率 p = {base*100:.1f}%  ({all_big}/{all_n})")

# 2. overdispersion 检验
# 若每场大球是独立同分布 Bernoulli(p), 各赛日大球数应服从 Binomial(n_day, p)
# 比较 观测的赛日率方差 vs 二项期望方差
print("\n=== 2. 过度离散检验 (聚集性) ===")
# 加权(按场次)的观测方差
obs_var = sum(len(by_day[d]) * (r - base) ** 2 for d, r in zip(sorted(by_day), [s[3] for s in day_stats])) / all_n
# 期望: 对 Binomial, Var(rate)=p(1-p)/n, 加权平均 = p(1-p)*mean(1/n)... 用更稳的 chi-square 分散指数
# Pearson 分散统计量 sum((big - n*p)^2 / (n*p(1-p)))  ~ chi2(df=days-1) if 随机
chi = 0.0
for d, n, big, rate in day_stats:
    exp = n * base
    var = n * base * (1 - base)
    if var > 0:
        chi += (big - exp) ** 2 / var
df = len(day_stats) - 1
disp = chi / df if df > 0 else float("nan")
print(f"  Pearson 分散指数 = chi2/df = {chi:.2f}/{df} = {disp:.2f}")
print(f"  解读: =1 随机洒落 | >1.5 明显成版聚集 | >2 强聚集")
# 赛日率的极差与标准差
print(f"  赛日大球率: 最低 {min(rates)*100:.0f}%  最高 {max(rates)*100:.0f}%  标准差 {statistics.pstdev(rates)*100:.1f}pp")

# 3. 版面赔率基调能否预判
print("\n=== 3. 版面赔率基调 → 赛日大球率 ===")
day_a4 = []
for d, n, big, rate in day_stats:
    a4s = [r["a4"] for r in by_day[d] if r.get("a4")]
    if len(a4s) >= 3:
        med = statistics.median(a4s)
        day_a4.append((d, med, rate, n))
if day_a4:
    med_all = statistics.median([x[1] for x in day_a4])
    print(f"  赛日 a4 中位数的中位数 = {med_all:.2f} (用作偏大版/偏小版分界)")
    big_slate = [x for x in day_a4 if x[1] <= med_all]   # a4 低=版面偏大球
    small_slate = [x for x in day_a4 if x[1] > med_all]
    def wrate(grp):
        tb = sum(round(x[2] * x[3]) for x in grp); tn = sum(x[3] for x in grp)
        return tb / tn if tn else 0, tn
    br, bn = wrate(big_slate); sr, sn = wrate(small_slate)
    print(f"  偏大版赛日 (a4中位<= {med_all:.2f}): {len(big_slate)}个赛日 {bn}场, 真实大球率 {br*100:.1f}%")
    print(f"  偏小版赛日 (a4中位>  {med_all:.2f}): {len(small_slate)}个赛日 {sn}场, 真实大球率 {sr*100:.1f}%")
    print(f"  >>> 基调分离度: {(br-sr)*100:+.1f}pp")
    print("  (分离越大, 说明'整版赔率基调'越能预判当天是大球日还是小球日)")
