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
    actuals = []
    for path in sorted(Path(AUTO_DIR).glob('actuals_*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        actuals.extend(x for x in data if isinstance(x, dict) and x.get('actual_score'))
    wc = json.loads(WC_ACTUALS.read_text(encoding='utf-8')).get('results', {}) if WC_ACTUALS.exists() else {}
    if isinstance(wc, list):
        actuals.extend(wc)
    return actuals, wc


def collect(return_audit=False):
    try:
        from .fixture_identity import dedupe_predictions, fixture_key, match_actual, prediction_rows
    except ImportError:
        from fixture_identity import dedupe_predictions, fixture_key, match_actual, prediction_rows
    acts, _legacy_wc = load_actuals()
    candidates = []
    for fn in sorted((ROOT / 'data').glob('history_*.json')):
        try:
            data = json.loads(fn.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        for match in prediction_rows(data):
            match['_snapshot_path'] = fn.name
            candidates.append(match)
    rows = []
    unresolved = []
    unique = dedupe_predictions(candidates)
    for m in unique:
        if m.get('league') not in (*B5, '世界杯'):
            continue
        actual_row = match_actual(m, acts)
        if not actual_row:
            unresolved.append({'fixture_key': fixture_key(m),
                               'home_team': m.get('home_team'), 'away_team': m.get('away_team'),
                               'snapshot_path': m['_snapshot_path'],
                               'reason': 'no_exact_result' if fixture_key(m) else 'missing_event_identity'})
            continue
        actual = actual_row['actual_score']
        actd = dir_of(actual)
        p = m.get('prediction') or {}
        pdir = p.get('final_direction')
        if not actd or pdir not in ('home', 'draw', 'away'):
            continue
        prices = [f(m.get('sp_' + d)) for d in ('home', 'draw', 'away')]
        favp = None
        if all(x and x > 1 for x in prices):
            inv = [1 / x for x in prices]
            favp = max(inv) / sum(inv)
        rows.append({'fixture_key': fixture_key(m), 'lg': m.get('league'),
                     'actd': actd, 'pdir': pdir, 'hit': pdir == actd,
                     'favp': favp, 'conf': p.get('confidence'),
                     'pred_score': p.get('final_ai_score') or p.get('predicted_score'),
                     'actual': actual, 'h': m.get('home_team'), 'a': m.get('away_team'),
                     'snapshot_path': m['_snapshot_path'],
                     'evaluation_scope': 'retrospective_unlocked'})
    if return_audit:
        return {'rows': rows, 'unresolved': unresolved, 'input_versions': len(candidates),
                'deduplicated_records': len(unique), 'strict_forward_samples': 0,
                'evaluation_scope': 'retrospective_unlocked'}
    return rows

def main():
    audit = collect(return_audit=True)
    rows = audit['rows']
    n=len(rows); hit=sum(1 for r in rows if r['hit'])
    print(f'=== 项目88 去重回顾统计（非严格前向；无赛事身份的旧赛果不结算） ===')
    print(f'样本: {n}场 | 方向命中: {hit}/{n} | 未结算: {len(audit["unresolved"])} | 严格前向: 0')
    # 平局召回
    real_draw=[r for r in rows if r['actd']=='draw']
    caught=sum(1 for r in real_draw if r['pdir']=='draw')
    print(f'平局召回率: {caught}/{len(real_draw)} = {caught*100//max(len(real_draw),1)}%')
    pred_home=sum(1 for r in rows if r['pdir']=='home'); real_home=sum(1 for r in rows if r['actd']=='home')
    print(f'主胜: AI判{pred_home}({pred_home*100//max(n,1)}%) vs 真实{real_home}({real_home*100//max(n,1)}%)')
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
    json.dump({**audit, 'n':n,'overall_hit_pct':round(hit*100/n,1) if n else None,
               'draw_recall_pct':round(caught*100/max(len(real_draw),1),1),
               'rows':rows},open(out,'w'),ensure_ascii=False,indent=1)
    print(f'\n快照已存: {out}')

if __name__=='__main__':
    main()
