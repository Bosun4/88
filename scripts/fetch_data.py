import requests,json,time
from datetime import datetime,timedelta
from bs4 import BeautifulSoup
from config import *
def get_today(offset=0):
    from zoneinfo import ZoneInfo
    return(datetime.now(ZoneInfo(TIMEZONE))+timedelta(days=offset)).strftime("%Y-%m-%d")
def fetch_matches(date=None):
    date=date or get_today();h={"x-apisports-key":API_FOOTBALL_KEY};ms=[]
    try:
        r=requests.get(API_FOOTBALL_BASE+"/fixtures",headers=h,params={"date":date},timeout=15)
        print("API status:"+str(r.status_code))
        d=r.json()
        print("API errors:"+str(d.get("errors",{})))
        print("API results:"+str(d.get("results",0)))
        if d.get("response"):
            for m in d["response"]:
                lid=m["league"]["id"]
                if lid in JINGCAI_LEAGUES:
                    ms.append({"id":m["fixture"]["id"],"date":m["fixture"]["date"],"league":m["league"]["name"],"league_id":lid,"league_logo":m["league"].get("logo",""),"home_team":m["teams"]["home"]["name"],"away_team":m["teams"]["away"]["name"],"home_logo":m["teams"]["home"].get("logo",""),"away_logo":m["teams"]["away"].get("logo",""),"home_id":m["teams"]["home"]["id"],"away_id":m["teams"]["away"]["id"],"status":m["fixture"]["status"]["short"],"home_goals":m["goals"]["home"],"away_goals":m["goals"]["away"]})
        print("Filtered jingcai: %d"%len(ms))
    except Exception as e:print("FETCH ERROR:"+str(e))
    return ms
def fetch_stats(tid,lid,season=2024):
    h={"x-apisports-key":API_FOOTBALL_KEY}
    try:
        r=requests.get(API_FOOTBALL_BASE+"/teams/statistics",headers=h,params={"team":tid,"league":lid,"season":season},timeout=15);d=r.json()
        if d.get("response"):
            s=d["response"];return{"played":s["fixtures"]["played"].get("total",0),"wins":s["fixtures"]["wins"].get("total",0),"draws":s["fixtures"]["draws"].get("total",0),"losses":s["fixtures"]["loses"].get("total",0),"goals_for":s["goals"]["for"]["total"].get("total",0),"goals_against":s["goals"]["against"]["total"].get("total",0),"form":s.get("form",""),"clean_sheets":s["clean_sheet"].get("total",0),"avg_goals_for":s["goals"]["for"]["average"].get("total","0"),"avg_goals_against":s["goals"]["against"]["average"].get("total","0")}
    except Exception as e:print("[S]%s:%s"%(tid,e))
    return{}
def fetch_h2h(hid,aid):
    h={"x-apisports-key":API_FOOTBALL_KEY}
    try:
        r=requests.get(API_FOOTBALL_BASE+"/fixtures/headtohead",headers=h,params={"h2h":str(hid)+"-"+str(aid),"last":10},timeout=15);d=r.json();rc=[]
        if d.get("response"):
            for m in d["response"]:rc.append({"date":m["fixture"]["date"][:10],"home":m["teams"]["home"]["name"],"away":m["teams"]["away"]["name"],"score":str(m["goals"]["home"])+"-"+str(m["goals"]["away"]),"league":m["league"]["name"]})
        return rc
    except Exception as e:print("[H]:%s"%e)
    return[]
def fetch_odds():
    try:
        r=requests.get(ODDS_API_BASE+"/sports/soccer/odds",params={"apiKey":ODDS_API_KEY,"regions":"eu","markets":"h2h,spreads,totals","oddsFormat":"decimal"},timeout=15)
        if r.status_code!=200:print("[Odds] status:%d"%r.status_code);return{}
        d=r.json()
        if isinstance(d,dict) and d.get("message"):print("[Odds]:"+d["message"]);return{}
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
def fetch_standings(comp="PL",season=2024):
    h={"X-Auth-Token":FOOTBALL_DATA_KEY}
    try:
        r=requests.get(FOOTBALL_DATA_BASE+"/competitions/"+comp+"/standings",headers=h,params={"season":season},timeout=15);d=r.json();st=[]
        if d.get("standings"):
            for g in d["standings"]:
                for t in g.get("table",[]):st.append({"position":t["position"],"team":t["team"]["name"],"played":t["playedGames"],"won":t["won"],"draw":t["draw"],"lost":t["lost"],"gf":t["goalsFor"],"ga":t["goalsAgainst"],"gd":t["goalDifference"],"points":t["points"]})
        return st
    except Exception as e:print("[FD]%s:%s"%(comp,e))
    return[]
def scrape_okooo(date=None):
    date=date or get_today();ms=[]
    try:
        r=requests.get(OKOOO_DETAIL,headers={"User-Agent":"Mozilla/5.0 (iPhone)"},timeout=15);r.encoding="utf-8"
        soup=BeautifulSoup(r.text,"html.parser")
        for row in soup.select("table tr,li"):
            t=row.get_text("|",strip=True)
            if t and len(t)>10:ms.append({"source":"okooo","raw":t[:200]})
    except:pass
    return ms
def scrape_500(date=None):
    date=date or get_today();ms=[]
    try:
        r=requests.get(C500_URL.format(date=date),headers={"User-Agent":"Mozilla/5.0"},timeout=15);r.encoding="gb2312"
        soup=BeautifulSoup(r.text,"html.parser")
        for row in soup.select("tr"):
            tds=row.select("td")
            if len(tds)>=5:ms.append({"source":"500","raw":"|".join([td.get_text(strip=True) for td in tds])[:300]})
    except:pass
    return ms
def collect_all(date=None):
    date=date or get_today();print("get %s..."%date)
    matches=fetch_matches(date);print("  %d matches"%len(matches))
    for i,m in enumerate(matches):
        print("  [%d/%d]%s v %s"%(i+1,len(matches),m["home_team"],m["away_team"]))
        m["home_stats"]=fetch_stats(m["home_id"],m["league_id"]);time.sleep(0.4)
        m["away_stats"]=fetch_stats(m["away_id"],m["league_id"]);time.sleep(0.4)
        m["h2h"]=fetch_h2h(m["home_id"],m["away_id"]);time.sleep(0.4)
    print("  odds...")
    odds=fetch_odds()
    print("  okooo...")
    okooo=scrape_okooo(date)
    print("  500...")
    c500=scrape_500(date)
    print("  standings...")
    standings={}
    for code,name in{"PL":"EPL","PD":"LaLiga","SA":"SerieA","BL1":"Bund","FL1":"L1"}.items():
        s=fetch_standings(code)
        if s:standings[name]=s
        time.sleep(0.5)
    return{"date":date,"matches":matches,"odds":odds,"okooo":okooo,"c500":c500,"standings":standings,"fetch_time":datetime.utcnow().isoformat()}
