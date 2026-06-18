#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""auto_actuals.py — 从 football-data.co.uk 自动生成 actuals JSON。
五大联赛真实赛果(免费无key)，输出 audit_backfill 兼容 schema:
  [{"match_num","home_team","away_team","actual_score"}]
匹配键: home_team||away_team (中文)，靠 team_aliases.json 中->英反查。
用法: python3 scripts/auto_actuals.py --pred <history.json> --out <actuals.json>
"""
from __future__ import annotations
import argparse, json, csv, io, urllib.request, re
from pathlib import Path

CODES = {'英超':'E0','西甲':'SP1','意甲':'I1','法甲':'F1','德甲':'D1'}
BASE = 'https://www.football-data.co.uk/mmz4281/2526/'
CACHE = '/tmp/fdcsv'  # 本地缓存优先, 避免重复打源站触发 503

def fetch_csv(code: str):
    import os
    cp = os.path.join(CACHE, f'{code}.csv')
    if os.path.exists(cp):
        return list(csv.DictReader(io.StringIO(open(cp,encoding='utf-8').read())))
    url = f'{BASE}{code}.csv'
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    raw = urllib.request.urlopen(req, timeout=25).read().decode('latin1')
    os.makedirs(CACHE, exist_ok=True)
    open(cp,'w',encoding='utf-8').write(raw)
    return list(csv.DictReader(io.StringIO(raw)))

def build_eng_results():
    """(EngHome,EngAway)->'fh-fa'"""
    res = {}
    for code in CODES.values():
        try: rows = fetch_csv(code)
        except Exception as e:
            print(f'  [warn] {code} 拉取失败: {str(e)[:80]}'); continue
        for r in rows:
            h,a = r.get('HomeTeam'),r.get('AwayTeam')
            try: fh,fa = int(r['FTHG']),int(r['FTAG'])
            except: continue
            if h and a: res[(h.strip(),a.strip())] = f'{fh}-{fa}'
    return res

def load_aliases(path: Path):
    a = json.loads(path.read_text(encoding='utf-8'))
    return {k:v for k,v in a.items() if not k.startswith('_')}

def cn_to_eng(cn: str, aliases: dict, eng_teams: set):
    """中文队名 -> CSV 英文队名。先查别名表(可能是关键词)，再在英文队集合里子串匹配。"""
    kw = aliases.get(cn)
    if kw:
        if kw in eng_teams: return kw
        for e in eng_teams:
            if kw.lower() in e.lower(): return e
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--aliases', default='reports/audit_backfill_20260531/team_aliases.json')
    args = ap.parse_args()

    pred = json.loads(Path(args.pred).read_text(encoding='utf-8'))
    rows = []
    m = pred.get('matches',{})
    if isinstance(m, dict):
        for day in m.values():
            if isinstance(day, list): rows.extend(day)
    aliases = load_aliases(Path(args.aliases))
    big5 = Path('reports/audit_backfill_20260531/team_aliases_big5.json')
    if big5.exists():
        aliases.update(load_aliases(big5))  # 五大联赛全队补全优先合并
    eng_res = build_eng_results()
    eng_teams = set()
    for h,a in eng_res: eng_teams.add(h); eng_teams.add(a)
    print(f'CSV 真实赛果对数: {len(eng_res)} | 英文队名: {len(eng_teams)}')

    out = []; matched=0; miss=[]
    for row in rows:
        if row.get('league') not in CODES: continue
        h,a = row.get('home_team'),row.get('away_team')
        eh,ea = cn_to_eng(h,aliases,eng_teams), cn_to_eng(a,aliases,eng_teams)
        score = eng_res.get((eh,ea)) if eh and ea else None
        if score:
            out.append({'match_num':row.get('match_num'),'home_team':h,'away_team':a,'actual_score':score})
            matched+=1
        else:
            miss.append(f"{row.get('league')} {h}({eh}) vs {a}({ea})")
    Path(args.out).write_text(json.dumps(out,ensure_ascii=False,indent=1),encoding='utf-8')
    print(f'已写 {matched} 场赛果 -> {args.out}')
    if miss:
        print(f'未匹配 {len(miss)} 场(非五大或别名缺失):')
        for x in miss[:30]: print('  ',x)

if __name__ == '__main__':
    main()
