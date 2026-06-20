#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析5届WC: 分轮(小组R1/R2/R3)+淘汰赛各阶段. 比分簇/平局/大球/点球加时. 真实数据."""
import json,datetime
from collections import defaultdict,Counter

M=json.load(open('reports/wc_research/_wc5_raw.json'))

# 每届小组赛48场(2006-2022都是32队/8组/每组3轮=48), 淘汰赛16场
# 按日期排序切轮次: 小组赛前48场按时间分3轮(每轮16); 其余为淘汰赛
by_wc=defaultdict(list)
for m in M:
    by_wc[m['wc']].append(m)

def parse(dt): return datetime.datetime.strptime(dt,"%Y%m%d")

rounds=defaultdict(list)   # key: '2022_R1' etc
ko=defaultdict(list)       # key: wc -> ko matches
for wc,ms in by_wc.items():
    ms.sort(key=lambda x:(x['date'],x['home']))
    group=ms[:48]; knock=ms[48:]
    # 小组赛按日期分组成轮: 取唯一日期, 前N天为R1...实际每轮16场
    for i,m in enumerate(group):
        r= i//16 +1
        rounds[f"{wc}_R{r}"].append(m)
        rounds[f"ALL_R{r}"].append(m)
    for m in knock:
        ko[wc].append(m); ko['ALL'].append(m)

def stats(ms):
    n=len(ms)
    if not n: return None
    tg=sum(m['tot'] for m in ms)
    dr=sum(1 for m in ms if m['draw'])
    big=sum(1 for m in ms if m['tot']>=4)
    over=sum(1 for m in ms if m['tot']>=3)  # over2.5
    zz=sum(1 for m in ms if m['hs']==0 and m['as']==0)
    return n,tg/n,dr/n,over/n,big/n,zz

print("=== 小组赛分轮(5届合计 ALL_Rx + 各届) ===")
print(f"{'bucket':10} {'N':>3} {'GPG':>5} {'draw%':>6} {'o2.5%':>6} {'4+%':>5} {'00':>3}")
for k in ['ALL_R1','ALL_R2','ALL_R3']:
    n,gpg,dr,ov,bg,zz=stats(rounds[k])
    print(f"{k:10} {n:3d} {gpg:5.2f} {dr*100:5.0f}% {ov*100:5.0f}% {bg*100:4.0f}% {zz:3d}")
print()
for wc in ['2006','2010','2014','2018','2022']:
    for r in [1,2,3]:
        n,gpg,dr,ov,bg,zz=stats(rounds[f"{wc}_R{r}"])
        print(f"{wc}_R{r:1}   {n:3d} {gpg:5.2f} {dr*100:5.0f}% {ov*100:5.0f}% {bg*100:4.0f}% {zz:3d}")
    print()

print("=== 比分簇分布(5届小组赛全部) ===")
allgroup=rounds['ALL_R1']+rounds['ALL_R2']+rounds['ALL_R3']
sc=Counter()
for m in allgroup:
    a,b=m['hs'],m['as']
    # 归一化为 高-低 表示比分形态
    hi,lo=max(a,b),min(a,b)
    sc[f"{hi}-{lo}"]+=1
tot=len(allgroup)
for s,c in sc.most_common(12):
    print(f"  {s}: {c} ({c/tot*100:.1f}%)")

print(f"\n=== 比分簇分布: 仅R3({len(rounds['ALL_R3'])}场) ===")
sc3=Counter()
for m in rounds['ALL_R3']:
    hi,lo=max(m['hs'],m['as']),min(m['hs'],m['as'])
    sc3[f"{hi}-{lo}"]+=1
t3=len(rounds['ALL_R3'])
for s,c in sc3.most_common(10):
    print(f"  {s}: {c} ({c/t3*100:.1f}%)")

print("\n=== 淘汰赛(5届合计) ===")
kn=ko['ALL']
n=len(kn)
pen=sum(1 for m in kn if m['pen'])
# 90分钟平局=进入加时(含点球): status里 pen 或 draw(加时后平进点球). ESPN score含加时
# 90min draw 无法直接分离, 用 pen 作点球率, draw(最终平)罕见
draw90_proxy=pen  # 点球必是120分钟平
tg=sum(m['tot'] for m in kn)
print(f"淘汰赛 N={n}, 场均进球(含加时){tg/n:.2f}, 点球决胜 {pen}场({pen/n*100:.0f}%)")
# 各阶段
ko_stage=defaultdict(list)
for wc,ms in by_wc.items():
    ms.sort(key=lambda x:(x['date'],x['home']))
    knock=ms[48:]
    # 16场: 8强16->8(R16,8场),4强(QF,4),半决赛(SF,2),3-4名+决赛(2)
    for i,m in enumerate(knock):
        if i<8: stg='R16'
        elif i<12: stg='QF'
        elif i<14: stg='SF'
        else: stg='F/3rd'
        ko_stage[stg].append(m)
for stg in ['R16','QF','SF','F/3rd']:
    ms=ko_stage[stg]; n=len(ms)
    if not n: continue
    pen=sum(1 for m in ms if m['pen'])
    tg=sum(m['tot'] for m in ms)
    print(f"  {stg:6} N={n:2d} 场均{tg/n:.2f} 点球{pen}场")

print("\n=== R3低分被爆案例(净胜1球及以下 + 大热门负/平) ===")
for m in rounds['ALL_R3']:
    if abs(m['hs']-m['as'])<=1 and m['tot']<=2:
        print(f"  {m['wc']} {m['home']} {m['hs']}-{m['as']} {m['away']}")
