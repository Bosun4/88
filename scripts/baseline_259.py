#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baseline_259.py — 项目88预测偏差基准回测(可复跑)。
真实赛果对账: 五大联赛(football-data.co.uk自动) + 世界杯(web抓取人工核验)。
固化为 prompt 改动前的对照基线。任何主链路改动后重跑此脚本对比。
用法: .venv/bin/python scripts/baseline_259.py
"""
from __future__ import annotations
import json, glob, sys
from pathlib import Path
from collections import defaultdict
import statistics as st

ROOT = Path(__file__).resolve().parent.parent
WC_ACTUALS = ROOT / 'reports/actuals_worldcup_202606/actuals_worldcup_202606.json'
AUTO_DIR = str(ROOT / 'reports/actuals_big5_202605')  # 五大联赛真实赛果(已落地仓库)
B5 = ('英超','西甲','意甲','法甲','德甲')

def dir_of(s):
    try: h,a = s.split('-'); h,a=int(h),int(a)
    except: return None
    return 'home' if h>a else 'away' if a>h else 'draw'

def f(v):
    try: return float(v)
    except: return None

def load_actuals():
    acts = {}
    for af in sorted(glob.glob(f'{AUTO_DIR}/actuals_*.json')):
        for x in json.load(open(af)):
            if x.get('actual_score'):
                acts[f"{x['home_team']}||{x['away_team']}"] = x['actual_score']
    wc = json.loads(WC_ACTUALS.read_text(encoding='utf-8'))['results']
    return acts, wc

def collect():
    acts, wc = load_actuals()
    seen=set(); rows=[]
    for fn in sorted(glob.glob(str(ROOT/'data/history_*.json'))):
        try: d=json.load(open(fn))
        except: continue
        for m in d.get('matches',{}).get('today',[]):
            lg=m.get('league'); k=f"{m.get('home_team')}||{m.get('away_team')}"
            is_wc = lg=='世界杯' and k in wc
            is_b5 = lg in B5 and k in acts
            if not (is_wc or is_b5): continue
            kk=(m.get('date'),k)
            if kk in seen: continue
            seen.add(kk)
            actual = wc[k] if is_wc else acts[k]
            actd = dir_of(actual)
            p = m.get('prediction',{})
            pdir = p.get('final_direction')
            if not actd or not pdir: continue
            sh,sd,sa = f(m.get('sp_home')),f(m.get('sp_draw')),f(m.get('sp_away'))
            favp = None
            if sh and sd and sa and 0 not in (sh,sd,sa):
                ih,idr,ia=1/sh,1/sd,1/sa; tot=ih+idr+ia
                favp = max(ih,idr,ia)/tot
            rows.append({'lg':'世界杯' if is_wc else lg,'actd':actd,'pdir':pdir,
                         'hit':pdir==actd,'favp':favp,'conf':p.get('confidence'),
                         'pred_score':p.get('final_ai_score') or p.get('predicted_score'),
                         'actual':actual,'h':m.get('home_team'),'a':m.get('away_team')})
    return rows

def main():
    rows = collect()
    n=len(rows); hit=sum(1 for r in rows if r['hit'])
    print(f'=== 项目88 基准回测 (prompt改动前对照线) ===')
    print(f'样本: {n}场 | 整体方向命中: {hit}/{n} = {hit*100/n:.1f}%')
    # 平局召回
    real_draw=[r for r in rows if r['actd']=='draw']
    caught=sum(1 for r in real_draw if r['pdir']=='draw')
    print(f'平局召回率: {caught}/{len(real_draw)} = {caught*100//max(len(real_draw),1)}%')
    pred_home=sum(1 for r in rows if r['pdir']=='home'); real_home=sum(1 for r in rows if r['actd']=='home')
    print(f'主胜: AI判{pred_home}({pred_home*100//n}%) vs 真实{real_home}({real_home*100//n}%)')
    # 按区分度
    print('\n按盘口区分度(最热方向隐含概率):')
    band=[(0,0.45,'低区分度'),(0.45,0.55,'中等'),(0.55,0.70,'较明确'),(0.70,1,'深盘大热')]
    fav_rows=[r for r in rows if r['favp'] is not None]
    for lo,hi,lab in band:
        g=[r for r in fav_rows if lo<=r['favp']<hi]
        h=sum(1 for r in g if r['hit'])
        print(f'  {lab:8}[{lo},{hi}): n={len(g)} 命中{h}={h*100//max(len(g),1)}%')
    # 按联赛
    print('\n按联赛:')
    byl=defaultdict(list)
    for r in rows: byl[r['lg']].append(r)
    for lg,g in sorted(byl.items(),key=lambda x:-len(x[1])):
        h=sum(1 for r in g if r['hit'])
        print(f'  {lg:6}: n={len(g)} 命中{h*100//len(g)}%')
    out=ROOT/'reports/baseline_259_snapshot.json'
    json.dump({'n':n,'overall_hit_pct':round(hit*100/n,1),
               'draw_recall_pct':round(caught*100/max(len(real_draw),1),1),
               'rows':rows},open(out,'w'),ensure_ascii=False,indent=1)
    print(f'\n快照已存: {out}')

if __name__=='__main__':
    main()
