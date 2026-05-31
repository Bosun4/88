#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STUDY A — 两段式决策审计 (Tier-then-Score Decision Audit)

核心拷问: AI 现在是"读盘"还是"抄最低赔率比分"? 两段式(先定进球档位, 再档内选比分)
能不能比现状更准? 决策瘫痪点(0-0 vs 1-1; 4球带四选一)真实分布如何?

档位定义 (TIER, 按真实总进球):
  LOW   总进球 0-1   (闷局: 0-0/1-0/0-1)
  MID   总进球 2-3   (平衡: 1-1/2-0/2-1/...)
  HIGH  总进球 >=4   (踢穿/大球)

验证:
  A1. 档位先验基线 — 全样本三档真实占比 (盲猜MID该有多少命中)
  A2. 现状诊断 — 模型预测比分的精确命中率 & 方向命中率 (它现在多准)
  A3. 档位可分性 — 若用赔率(a4压缩/favourite)给"档位"投票, 能不能把HIGH档拎出来
  A4. 两段式天花板 — 假设档位判对, 第二段在该档落"该档历史众数比分", 精确命中率上界
  A5. 瘫痪点实证 — LOW档里 0-0 vs 1-1 真实占比; MID/HIGH里四选一(1-1/1-2/2-1/2-2)真实占比

无网络, 无密钥. 纯本地.
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict

M = json.load(open("reports/audit_backfill_20260531/matched_samples.json", encoding="utf-8"))
M = [r for r in M if r.get("actual_total") is not None and isinstance(r.get("actual"), str) and "-" in r["actual"]]

def tier_of_total(t):
    if t <= 1: return "LOW"
    if t <= 3: return "MID"
    return "HIGH"

def norm_score(s):
    # 把 actual / predicted_score 规约成 "h-a"
    try:
        h, a = str(s).replace("：", "-").replace(":", "-").split("-")[:2]
        return f"{int(h)}-{int(a)}"
    except Exception:
        return None

print("=== STUDY A: 两段式决策审计 ===")
print(f"有效样本: {len(M)} 场\n")

# A1 档位先验
print("=== A1. 档位先验基线 (真实总进球分档) ===")
tc = Counter(tier_of_total(r["actual_total"]) for r in M)
for t in ("LOW", "MID", "HIGH"):
    print(f"  {t}: {tc[t]:>3} ({tc[t]/len(M)*100:.1f}%)")
print(f"  >>> 盲猜永远押MID 的档位命中率 = {tc['MID']/len(M)*100:.1f}%")

# A2 现状诊断
print("\n=== A2. 现状: 模型预测的命中率 ===")
exact = 0; dir_hit = 0; tier_hit = 0; n_pred = 0
for r in M:
    ps = norm_score(r.get("predicted_score"))
    as_ = norm_score(r.get("actual"))
    if ps is None or as_ is None: continue
    n_pred += 1
    if ps == as_: exact += 1
    # 方向
    ph, pa = map(int, ps.split("-")); ah, aa = map(int, as_.split("-"))
    pd = (ph>pa)-(ph<pa); ad = (ah>aa)-(ah<aa)
    if pd == ad: dir_hit += 1
    # 档位
    if tier_of_total(ph+pa) == tier_of_total(ah+aa): tier_hit += 1
print(f"  可比预测: {n_pred} 场")
print(f"  精确比分命中: {exact}/{n_pred} = {exact/n_pred*100:.1f}%")
print(f"  方向命中:     {dir_hit}/{n_pred} = {dir_hit/n_pred*100:.1f}%")
print(f"  档位命中:     {tier_hit}/{n_pred} = {tier_hit/n_pred*100:.1f}%")
print(f"  (档位命中{tier_hit/n_pred*100:.0f}% vs 盲猜MID {tc['MID']/len(M)*100:.0f}% — 差值就是模型在'定档'上的真实增量)")

# A3 档位可分性: 用 a4 压缩 + favourite 投 HIGH 档
print("\n=== A3. 档位可分性: 赔率能否拎出 HIGH 档 ===")
def has(r,k): return r.get(k) is not None
for thr in (4.2, 4.5, 5.0, 5.3):
    sel = [r for r in M if has(r,"a4") and r["a4"] <= thr]
    if not sel: continue
    hi = sum(1 for r in sel if tier_of_total(r["actual_total"])=="HIGH")
    print(f"  a4<={thr}: {len(sel):>3}场 → 真实HIGH率 {hi/len(sel)*100:.1f}%  (基础HIGH率 {tc['HIGH']/len(M)*100:.1f}%)")

# A4 两段式天花板: 档判对后落该档众数比分
print("\n=== A4. 两段式天花板: 档内落众数比分 ===")
by_tier = defaultdict(Counter)
for r in M:
    as_ = norm_score(r.get("actual"))
    if as_: by_tier[tier_of_total(r["actual_total"])][as_] += 1
ceil = 0
for t in ("LOW","MID","HIGH"):
    c = by_tier[t]
    n = sum(c.values())
    top, topn = c.most_common(1)[0]
    ceil += topn
    print(f"  {t}({n}场): 众数比分 {top} 占 {topn}/{n}={topn/n*100:.1f}%  | 该档比分分布 {dict(c.most_common(4))}")
print(f"  >>> 若档位100%判对+档内落众数: 精确命中上界 = {ceil}/{len(M)} = {ceil/len(M)*100:.1f}%")
print(f"      (对比现状精确 {exact/n_pred*100:.1f}% — 这是两段式比分层的理论提升空间)")

# A5 决策瘫痪点
print("\n=== A5. 瘫痪点实证 ===")
low = [r for r in M if tier_of_total(r["actual_total"])=="LOW"]
lc = Counter(norm_score(r["actual"]) for r in low)
print(f"  [LOW档 {len(low)}场] 0-0 vs 1-0 vs 0-1 真实分布: {dict(lc.most_common())}")
z = lc.get("0-0",0)
print(f"    >>> 0-0 仅占 LOW档 {z}/{len(low)}={z/len(low)*100:.0f}%; 1-1属MID不在此档 — '0-0 vs 1-1纠结'本质是跨档错误!")
mid = [r for r in M if tier_of_total(r["actual_total"])=="MID"]
mc = Counter(norm_score(r["actual"]) for r in mid)
four = {k:mc.get(k,0) for k in ("1-1","2-0","0-2","2-1","1-2","3-0","0-3")}
print(f"  [MID档 {len(mid)}场] 比分分布: {dict(mc.most_common(6))}")
print(f"    四选一相关(1-1/2-1/1-2): 1-1={mc.get('1-1',0)} 2-1={mc.get('2-1',0)} 1-2={mc.get('1-2',0)}")
high = [r for r in M if tier_of_total(r["actual_total"])=="HIGH"]
hc = Counter(norm_score(r["actual"]) for r in high)
print(f"  [HIGH档 {len(high)}场] 比分分布: {dict(hc.most_common(8))}")
