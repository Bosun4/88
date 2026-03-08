import requests,json,time,re
from datetime import datetime,timedelta
from bs4 import BeautifulSoup
from config import *
def get_today(offset=0):
    from zoneinfo import ZoneInfo
    return(datetime.now(ZoneInfo(TIMEZONE))+timedelta(days=offset)).strftime("%Y-%m-%d")
def scrape_500_jczq(date=None):
    date=date or get_today()
    url=C500_URL.format(date=date)
    print("  500.com: %s"%url)
    ms=[]
    try:
        r=requests.get(url,headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},timeout=20)
        r.encoding="gb2312"
        html=r.text
        soup=BeautifulSoup(html,"html.parser")
        rows=soup.select("tr[id]")
        if not rows:
            rows=soup.select("tbody tr")
        if not rows:
            rows=soup.find_all("tr")
        print("  found %d rows"%len(rows))
        for row in rows:
            tds=row.find_all("td")
            if len(tds)<4:continue
            text="|".join([td.get_text(strip=True) for td in tds])
            vs_td=None
            for td in tds:
                t=td.get_text(strip=True)
                if "vs" in t.lower() or " - " in t:
                    vs_td=t;break
            links=row.find_all("a")
            teams=[]
            for a in links:
                t=a.get_text(strip=True)
                if len(t)>1 and len(t)<20 and not t.isdigit():
                    teams.append(t)
            league=""
            match_num=""
            for td in tds[:3]:
                t=td.get_text(strip=True)
                if re.match(r"^(周[一二三四五六日])\d{3}$",t):
                    match_num=t
                elif len(t)>1 and len(t)<8 and not t.isdigit():
                    if not any(c in t for c in ["vs","VS","-"]):
                        league=t
            home=""
            away=""
            if vs_td and ("vs" in vs_td.lower()):
                parts=re.split(r"[vV][sS]",vs_td)
                if len(parts)==2:
                    home=parts[0].strip()
                    away=parts[1].strip()
            elif len(teams)>=2:
                home=teams[0]
                away=teams[1]
            if home and away:
                m={"home_team":home,"away_team":away,"league":league,"match_num":match_num,"source":"500","raw":text[:200]}
                ms.append(m)
                print("    %s: %s vs %s"%(league or"?",home,away))
    except Exception as e:
        print("  500.com error:%s"%str(e))
    return ms
def scrape_okooo_jczq(date=None):
    date=date or get_today()
    ms=[]
    try:
        r=requests.get(OKOOO_DETAIL,headers={"User-Agent":"Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)"},timeout=15)
        r.encoding="utf-8"
        soup=BeautifulSoup(r.text,"html.parser")
        items=soup.select(".match-info,.match-item,tr[data-mid],.game-row")
        if not items:
            items=soup.find_all("tr")
        for item in items:
            tds=item.find_all("td") if item.name=="tr" else [item]
            text=item.get_text("|",strip=True)
            if len(text)<10:continue
            links=item.find_all("a")
            teams=[]
            for a in links:
                t=a.get_text(strip=True)
                if 1<len(t)<20 and not t.isdigit():
                    teams.append(t)
            if len(teams)>=2:
                ms.append({"home_team":teams[0],"away_team":teams[1],"source":"okooo","raw":text[:200]})
    except Exception as e:
        print("  okooo error:%s"%str(e))
    return ms
def search_team_api(name):
    h={"x-apisports-key":API_FOOTBALL_KEY}
    try:
        r=requests.get(API_FOOTBALL_BASE+"/teams",headers=h,params={"search":name},timeout=10)
        d=r.json()
        if d.get("response") and len(d["response"])>0:
            team=d["response"][0]["team"]
            return{"id":team["id"],"name":team["name"],"logo":team.get("logo","")}
    except:pass
    return None
def fetch_stats(tid,lid=0,season=2024):
    if not tid:return{}
    h={"x-apisports-key":API_FOOTBALL_KEY}
    try:
        params={"team":tid,"season":season}
        if lid:params["league"]=lid
        r=requests.get(API_FOOTBALL_BASE+"/teams/statistics",headers=h,params=params,timeout=15);d=r.json()
        if d.get("response"):
            s=d["response"];return{"played":s["fixtures"]["played"].get("total",0),"wins":s["fixtures"]["wins"].get("total",0),"draws":s["fixtures"]["draws"].get("total",0),"losses":s["fixtures"]["loses"].get("total",0),"goals_for":s["goals"]["for"]["total"].get("total",0),"goals_against":s["goals"]["against"]["total"].get("total",0),"form":s.get("form",""),"clean_sheets":s["clean_sheet"].get("total",0),"avg_goals_for":s["goals"]["for"]["average"].get("total","0"),"avg_goals_against":s["goals"]["against"]["average"].get("total","0")}
    except Exception as e:print("[S]%s"%e)
    return{}
def fetch_h2h(hid,aid):
    if not hid or not aid:return[]
    h={"x-apisports-key":API_FOOTBALL_KEY}
    try:
        r=requests.get(API_FOOTBALL_BASE+"/fixtures/headtohead",headers=h,params={"h2h":str(hid)+"-"+str(aid),"last":10},timeout=15);d=r.json();rc=[]
        if d.get("response"):
            for m in d["response"]:rc.append({"date":m["fixture"]["date"][:10],"home":m["teams"]["home"]["name"],"away":m["teams"]["away"]["name"],"score":str(m["goals"]["home"])+"-"+str(m["goals"]["away"]),"league":m["league"]["name"]})
        return rc
    except:pass
    return[]
def fetch_odds():
    try:
        r=requests.get(ODDS_API_BASE+"/sports/soccer/odds",params={"apiKey":ODDS_API_KEY,"regions":"eu","markets":"h2h,spreads,totals","oddsFormat":"decimal"},timeout=15)
        if r.status_code!=200:print("[Odds]status:%d"%r.status_code);return{}
        d=r.json()
        if isinstance(d,dict):print("[Odds]:%s"%str(d.get("message",""))[:80]);return{}
        om={}
        for ev in d:
            k=ev["home_team"]+"_"+ev["away_team"];bks=[]
            for bk in ev.get("bookmakers",[])[:5]:
                mk={}
                for mt in bk.get("markets",[]):mk[mt["key"]]={o["name"]:o.get("price",0) for o in mt.get("outcomes",[])}
                bks.append({"name":bk["title"],"markets":mk})
            om[k]={"commence_time":ev.get("commence_time",""),"bookmakers":bks}
        return om
    except Exception as e:print("[O]:%s"%e)
    return{}
def collect_all(date=None):
    date=date or get_today();print("=== %s ==="%date)
    print("1. Scrape 500.com jczq...")
    matches=scrape_500_jczq(date)
    if not matches:
        print("2. Try okooo...")
        okooo=scrape_okooo_jczq(date)
        for o in okooo:
            if o.get("home_team") and o.get("away_team"):
                matches.append(o)
    print("  Total jczq matches: %d"%len(matches))
    print("3. Search teams in API-Football...")
    for i,m in enumerate(matches):
        print("  [%d/%d] %s vs %s"%(i+1,len(matches),m["home_team"],m["away_team"]))
        ht=search_team_api(m["home_team"])
        time.sleep(0.5)
        at=search_team_api(m["away_team"])
        time.sleep(0.5)
        m["home_id"]=ht["id"] if ht else 0
        m["away_id"]=at["id"] if at else 0
        m["home_logo"]=ht["logo"] if ht else ""
        m["away_logo"]=at["logo"] if at else ""
        m["league_logo"]=""
        m["id"]=i+1
        m["date"]=date
        if ht and ht["id"]:
            m["home_stats"]=fetch_stats(ht["id"]);time.sleep(0.3)
        else:m["home_stats"]={}
        if at and at["id"]:
            m["away_stats"]=fetch_stats(at["id"]);time.sleep(0.3)
        else:m["away_stats"]={}
        m["h2h"]=fetch_h2h(m.get("home_id"),m.get("away_id"));time.sleep(0.3)
    print("4. Odds...")
    odds=fetch_odds()
    return{"date":date,"matches":matches,"odds":odds,"standings":{},"fetch_time":datetime.utcnow().isoformat()}