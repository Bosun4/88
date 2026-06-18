#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓2026年3月+6月国际窗口友谊赛+预选, 按队聚合状态档. ESPN多赛事API. 真实数据."""
import json,urllib.request,time,datetime
from collections import defaultdict

LEAGUES=['fifa.friendly','fifa.worldq.concacaf','fifa.worldq.uefa','fifa.worldq.conmebol',
         'fifa.worldq.afc','fifa.worldq.caf','fifa.cofc','uefa.nations']

def fetch(lg,date):
    url=f"https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/scoreboard?dates={date}"
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    for _ in range(2):
        try: return json.load(urllib.request.urlopen(req,timeout=20))
        except Exception: time.sleep(0.5)
    return {}

def drange(s,e):
    s=datetime.date(*s); e=datetime.date(*e); o=[]
    while s<=e: o.append(s.strftime("%Y%m%d")); s+=datetime.timedelta(days=1)
    return o

windows=drange((2026,3,23),(2026,3,31))+drange((2026,6,1),(2026,6,8))
matches=[]; seen=set()
for lg in LEAGUES:
    for dt in windows:
        d=fetch(lg,dt)
        for ev in d.get('events',[]):
            comp=ev.get('competitions',[{}])[0]
            st=comp.get('status',{}).get('type',{})
            if not st.get('completed'): continue
            cs=comp.get('competitors',[])
            if len(cs)<2: continue
            try: sa=int(cs[0].get('score')); sb=int(cs[1].get('score'))
            except: continue
            na=cs[0].get('team',{}).get('displayName'); nb=cs[1].get('team',{}).get('displayName')
            ha=cs[0].get('homeAway'); 
            mid=(dt,na,nb)
            if mid in seen: continue
            seen.add(mid)
            matches.append({'lg':lg,'date':dt,'home':na,'hs':sa,'as':sb,'away':nb})
        time.sleep(0.15)
    print(lg,'done, total',len(matches))

json.dump(matches,open('reports/wc_research/_form_raw.json','w'),ensure_ascii=False)
print("TOTAL",len(matches))

# 按队聚合
TEAMS=['Brazil','Argentina','France','England','Spain','Germany','Portugal','Netherlands',
 'Belgium','Italy','Croatia','Uruguay','Mexico','United States','Canada','Morocco','Japan',
 'South Korea','Switzerland','Denmark','Colombia','Senegal','Poland','Serbia']
form=defaultdict(list)
for m in matches:
    for team,gf,ga,opp,loc in [(m['home'],m['hs'],m['as'],m['away'],'H'),(m['away'],m['as'],m['hs'],m['home'],'A')]:
        if team in TEAMS:
            res='W' if gf>ga else ('D' if gf==ga else 'L')
            form[team].append((m['date'],loc,gf,ga,opp,res))
print("\n=== 主要参赛队近期状态(3月+6月窗口) ===")
for t in TEAMS:
    g=sorted(form.get(t,[]))
    if not g: continue
    rec=''.join(x[5] for x in g)
    gf=sum(x[2] for x in g); ga=sum(x[3] for x in g)
    cs=sum(1 for x in g if x[3]==0)
    line=' | '.join(f"{x[0][4:]}{x[1]} {x[2]}-{x[3]} {x[4]}" for x in g)
    print(f"{t:15} [{rec}] GF{gf}/GA{ga} CS{cs}/{len(g)}: {line}")
