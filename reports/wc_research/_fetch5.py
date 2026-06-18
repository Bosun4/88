#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓5届世界杯(2006-2022)逐场真实赛果, 分轮+分阶段+比分簇+点球/加时. ESPN官方API. 无编造."""
import json,urllib.request,time
from collections import defaultdict,Counter

def fetch(date):
    url=f"https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={date}"
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    for _ in range(3):
        try:
            return json.load(urllib.request.urlopen(req,timeout=25))
        except Exception:
            time.sleep(1)
    return {}

def daterange(start,end):
    import datetime
    s=datetime.date(*start); e=datetime.date(*end); out=[]
    while s<=e:
        out.append(s.strftime("%Y%m%d")); s+=datetime.timedelta(days=1)
    return out

# 各届世界杯日期窗口
WCS={
 '2006':((2006,6,9),(2006,7,9)),
 '2010':((2010,6,11),(2010,7,11)),
 '2014':((2014,6,12),(2014,7,13)),
 '2018':((2018,6,14),(2018,7,15)),
 '2022':((2022,11,20),(2022,12,18)),
}

allmatches=[]
seen=set()
for wc,(s,e) in WCS.items():
    for dt in daterange(s,e):
        d=fetch(dt)
        for ev in d.get('events',[]):
            comp=ev.get('competitions',[{}])[0]
            st=comp.get('status',{}).get('type',{})
            if not st.get('completed'): continue
            stname=st.get('name','')
            cs=comp.get('competitors',[])
            if len(cs)<2: continue
            try:
                sa=int(cs[0].get('score')); sb=int(cs[1].get('score'))
            except: continue
            na=cs[0].get('team',{}).get('abbreviation') or cs[0].get('team',{}).get('displayName')
            nb=cs[1].get('team',{}).get('abbreviation') or cs[1].get('team',{}).get('displayName')
            mid=(wc,na,nb,sa,sb)
            if mid in seen: continue
            seen.add(mid)
            # leagueName/season slug 区分阶段
            notes=comp.get('notes',[])
            note=notes[0].get('headline','') if notes else ''
            pen='PEN' in stname
            so=[c.get('shootoutScore') for c in cs]
            allmatches.append({
                'wc':wc,'date':dt,'home':na,'hs':sa,'as':sb,'away':nb,
                'tot':sa+sb,'draw':sa==sb,'status':stname,'pen':pen,
                'shootout':so,'note':note
            })
        time.sleep(0.25)
    print(f"{wc}: cumulative {len(allmatches)} matches")

json.dump(allmatches,open('reports/wc_research/_wc5_raw.json','w'),ensure_ascii=False,indent=0)
print("TOTAL",len(allmatches),"saved")
PY_DONE=1
print("DONE")
