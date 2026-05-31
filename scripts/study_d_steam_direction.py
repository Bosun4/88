#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STUDY D — 大小球线移动方向的预测力 (Totals Line Steam Direction)

用户巴黎vs阿森纳核心读法: 大小球线 3->2.25 下行 = 小球盘.
本研究用 football-data 五大联赛真实样本(开盘 vs 收盘 over/under 2.5 赔率)
实测: "线往哪走"能不能预判真实总进球大小.

口径:
- 开盘大球概率 p_open = 去抽水后的 over2.5 隐含概率 (B365>2.5 / B365<2.5)
- 收盘大球概率 p_close = 同上 (B365C>2.5 / B365C<2.5)
- steam = p_close - p_open  (>0=资金推向大球/线上行, <0=推向小球/线下行)
- 真实: FTHG+FTAG >=3 即 over2.5 命中
验证: 按 steam 分组, 看真实 over2.5 命中率是否随 steam 单调变化.
无网络. 纯本地 CSV.
"""
from __future__ import annotations
import csv, glob, statistics as st
from collections import defaultdict

def devig_over(o_over, o_under):
    try:
        po, pu = 1.0/float(o_over), 1.0/float(o_under)
        return po/(po+pu)
    except Exception:
        return None

rows = []
for path in glob.glob(".cache/football_data_2526/*.csv"):
    div = path.split("/")[-1].replace(".csv","")
    for r in csv.DictReader(open(path, encoding="utf-8", errors="ignore")):
        po = devig_over(r.get("B365>2.5"), r.get("B365<2.5"))
        pc = devig_over(r.get("B365C>2.5"), r.get("B365C<2.5"))
        try:
            tot = int(r["FTHG"]) + int(r["FTAG"])
        except Exception:
            continue
        if po is None or pc is None: continue
        rows.append({"div":div,"p_open":po,"p_close":pc,"steam":pc-po,
                     "tot":tot,"over25":tot>=3,"over35":tot>=4})

print(f"=== STUDY D: 大小球线移动方向预测力 ===")
print(f"五大联赛样本: {len(rows)} 场\n")
base25 = sum(1 for r in rows if r["over25"])/len(rows)
base35 = sum(1 for r in rows if r["over35"])/len(rows)
print(f"基础 over2.5率 {base25*100:.1f}% | over3.5(>=4球)率 {base35*100:.1f}%\n")

# 1. steam 分组 -> 真实 over2.5
print("=== 1. 线移动方向(steam=收盘-开盘大球概率) → 真实over2.5 ===")
buckets=[("强烈下行 steam<=-5%",lambda s:s<=-0.05),
         ("小幅下行 -5%~-1.5%",lambda s:-0.05<s<=-0.015),
         ("基本不动 ±1.5%",lambda s:-0.015<s<0.015),
         ("小幅上行 1.5%~5%",lambda s:0.015<=s<0.05),
         ("强烈上行 steam>=5%",lambda s:s>=0.05)]
for name,f in buckets:
    g=[r for r in rows if f(r["steam"])]
    if not g: continue
    o25=sum(1 for r in g if r["over25"])/len(g)
    o35=sum(1 for r in g if r["over35"])/len(g)
    print(f"  {name:>22}: n={len(g):>4}  真实over2.5={o25*100:>5.1f}%  >=4球={o35*100:>5.1f}%")

# 2. 对比: 收盘价单点 vs 移动方向, 谁更准
print("\n=== 2. 收盘静态价 vs 移动方向 的增量 ===")
# 控制收盘概率相近时, steam 还有没有额外信息
mid=[r for r in rows if 0.45<=r["p_close"]<=0.55]  # 收盘五五开的场
up=[r for r in mid if r["steam"]>=0.015]
dn=[r for r in mid if r["steam"]<=-0.015]
if up and dn:
    print(f"  收盘五五开(p_close 0.45-0.55) {len(mid)}场中:")
    print(f"    临场往大球走({len(up)}场): 真实over2.5 {sum(1 for r in up if r['over25'])/len(up)*100:.1f}%")
    print(f"    临场往小球走({len(dn)}场): 真实over2.5 {sum(1 for r in dn if r['over25'])/len(dn)*100:.1f}%")
    print(f"    >>> 同样收盘五五开, 仅凭'线往哪走'分离度 {(sum(1 for r in up if r['over25'])/len(up)-sum(1 for r in dn if r['over25'])/len(dn))*100:+.1f}pp")

# 3. 巴黎式: 线下行+收盘偏小 → 小球确认
print("\n=== 3. 巴黎式读法验证: 线下行 且 收盘偏小球 ===")
paris=[r for r in rows if r["steam"]<=-0.015 and r["p_close"]<0.5]
if paris:
    u=sum(1 for r in paris if not r["over25"])  # under命中
    print(f"  线下行+收盘大球概率<50%: {len(paris)}场, 真实小球(<=2球)率 {u/len(paris)*100:.1f}% (基础小球率{(1-base25)*100:.1f}%)")
