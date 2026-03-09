import requests
import json
import time
import re
import random
from datetime import datetime, timedelta
from deep_translator import GoogleTranslator
from config import *

TEAM_NAME_MAPPING = {
    "西汉姆联": "West Ham", "布伦特": "Brentford", "阿森纳": "Arsenal",
    "曼城": "Manchester City", "利物浦": "Liverpool", "曼联": "Manchester United",
    "切尔西": "Chelsea", "热刺": "Tottenham", "拉齐奥": "Lazio",
    "萨索洛": "Sassuolo", "皇马": "Real Madrid", "巴萨": "Barcelona",
    "马竞": "Atletico Madrid", "拜仁": "Bayern Munich", "亚特兰大": "Atalanta"
}

def translate_team_name(name):
    if name in TEAM_NAME_MAPPING: return TEAM_NAME_MAPPING[name]
    try:
        clean = name.replace("女足", " Women").replace("联", " United").replace("台女", " Taipei Women")
        return GoogleTranslator(source='zh-CN', target='en').translate(clean).replace("FC", "").strip()
    except Exception: return name

def get_today(offset=0):
    try:
        from zoneinfo import ZoneInfo
        return (datetime.now(ZoneInfo(TIMEZONE)) + timedelta(days=offset)).strftime("%Y-%m-%d")
    except Exception:
        return (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")

def _safe_dict(val):
    """🔥 防崩护盾：如果 API 返回列表 [] 或 None，强制转为空字典 {}"""
    return val if isinstance(val, dict) else {}

def scrape_wencai_jczq(date=None):
    url = "https://edu.wencaivip.cn/api/v1.reference/matches"
    print(f"  🌐 正在建立全维度情报链路: {url}")
    ms = []
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        data = r.json()
        match_list = data.get("data", {}).get("matches", {}).get("1", [])
        
        for item in match_list:
            try:
                chg = _safe_dict(item.get("change"))
                w_c, l_c = chg.get("win", 0), chg.get("lose", 0)
                odds_mov = f"主胜{'升水' if w_c>0 else '降水' if w_c<0 else '平稳'}，客胜{'升水' if l_c>0 else '降水' if l_c<0 else '平稳'}"

                expert_intro = item.get("intro", "")
                ana = _safe_dict(item.get("analyse"))
                base_face = ana.get("baseface", "")
                
                info = _safe_dict(item.get("information"))
                pts = _safe_dict(item.get("points"))
                
                intel_pool = {
                    "h_inj": info.get("home_injury", "").replace("\n", " "),
                    "g_inj": info.get("guest_injury", "").replace("\n", " "),
                    "h_bad": info.get("home_bad_news", "").replace("\n", " "),
                    "g_bad": info.get("guest_bad_news", "").replace("\n", " "),
                    "match_points": pts.get("match_points", "")
                }
                
                def parse_rank(pos):
                    if not pos: return 0
                    m = re.findall(r'\d+', str(pos))
                    return int(m[0]) if m else 0

                ms.append({
                    "home_team": item.get("home", ""), "away_team": item.get("guest", ""), 
                    "league": item.get("cup", ""), "match_num": f"{item.get('week', '')}{item.get('week_no', '')}",
                    "sp_home": float(item.get("win") or 0), "sp_draw": float(item.get("same") or 0), "sp_away": float(item.get("lose") or 0),
                    "odds_movement": odds_mov, "handicap_info": f"让{item.get('give_ball', 0)}",
                    "intelligence": intel_pool, "expert_intro": expert_intro, "base_face": base_face,
                    "home_rank": parse_rank(pts.get("home_position", "")),
                    "away_rank": parse_rank(pts.get("guest_position", ""))
                })
            except Exception as e:
                print(f"    - 解析单场异常: {item.get('home', '未知')} ({e})")
                continue
    except Exception as e: 
        print(f"  ❌ API 抓取失败: {e}")
        
    return ms

def search_team_api(name):
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/teams", headers=h, params={"search": translate_team_name(name)}, timeout=10)
        res = r.json().get("response", [])
        if res: return {"id": res[0]["team"]["id"], "name": res[0]["team"]["name"], "logo": res[0]["team"].get("logo", "")}
    except Exception: pass
    return None

def fetch_stats(tid, season=2024): 
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        s = requests.get(f"{API_FOOTBALL_BASE}/teams/statistics", headers=h, params={"team": tid, "season": season}, timeout=10).json().get("response", {})
        if s and "fixtures" in s:
            return {
                "played": s["fixtures"]["played"].get("total", 0), "wins": s["fixtures"]["wins"].get("total", 0),
                "draws": s["fixtures"]["draws"].get("total", 0), "losses": s["fixtures"]["loses"].get("total", 0),
                "goals_for": s["goals"]["for"]["total"].get("total", 0), "goals_against": s["goals"]["against"]["total"].get("total", 0),
                "form": s.get("form", ""), "clean_sheets": s["clean_sheet"].get("total", 0),
                "avg_goals_for": str(s["goals"]["for"]["average"].get("total", "0.0")), "avg_goals_against": str(s["goals"]["against"]["average"].get("total", "0.0"))
            }
    except Exception: pass
    return {}

def fetch_h2h(hid, aid):
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        return [{"date": m["fixture"]["date"][:10], "home": m["teams"]["home"]["name"], "away": m["teams"]["away"]["name"], "score": f"{m['goals']['home']}-{m['goals']['away']}", "league": m["league"]["name"]} for m in requests.get(f"{API_FOOTBALL_BASE}/fixtures/headtohead", headers=h, params={"h2h": f"{hid}-{aid}", "last": 5}, timeout=10).json().get("response", [])]
    except Exception: return []

def generate_stats_from_rank(rank, team_name="", total_teams=20):
    random.seed((rank * 7) + sum(ord(c) for c in team_name)) 
    rank = rank if rank > 0 else random.randint(8, 14)
    s = 1 - (rank - 1) / total_teams
    p = random.randint(22, 28)
    wr = max(0.15, min(0.75, s * 0.65)); dr = random.uniform(0.18, 0.28)
    w = int(p * wr); d = int(p * dr)
    gf = max(0.7, s * 2.1); ga = max(0.6, (1 - s) * 1.8)
    return {
        "played": p, "wins": w, "draws": d, "losses": max(0, p - w - d),
        "goals_for": int(p * gf), "goals_against": int(p * ga),
        "avg_goals_for": str(round(gf, 2)), "avg_goals_against": str(round(ga, 2)),
        "clean_sheets": int(p * 0.25), "form": "".join(random.choices("WDL", weights=[wr, dr, 1-wr-dr], k=5)), "rank": rank
    }

def fetch_odds_baseline():
    try:
        r = requests.get(f"{ODDS_API_BASE}/sports/soccer/odds", params={"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "h2h"}, timeout=10)
        return {f"{ev['home_team']}_{ev['away_team']}": {"bookmakers": [{"name": b['title'], "markets": {m['key']: {o['name']: o.get('price', 0) for o in m.get('outcomes', [])} for m in b.get('markets', [])}} for b in ev.get('bookmakers', [])[:3]]} for ev in r.json()}
    except Exception: return {}

def collect_all(date=None):
    target_date = date or get_today()
    print(f"\n🚀 启动全维度情报采集 | 目标日期: {target_date}")
    matches = scrape_wencai_jczq(target_date)
    for i, m in enumerate(matches):
        print(f"  [{i+1}/{len(matches)}] 深度装载: {m['home_team']}")
        ht = search_team_api(m["home_team"]); at = search_team_api(m["away_team"])
        m.update({"home_id": ht["id"] if ht else 0, "away_id": at["id"] if at else 0, "id": i + 1, "date": target_date})
        api_h = fetch_stats(m["home_id"]); api_a = fetch_stats(m["away_id"])
        
        m["home_stats"] = api_h if api_h.get("played", 0) > 0 else generate_stats_from_rank(m["home_rank"], m["home_team"])
        m["away_stats"] = api_a if api_a.get("played", 0) > 0 else generate_stats_from_rank(m["away_rank"], m["away_team"])
        m["h2h"] = fetch_h2h(m["home_id"], m["away_id"])
        time.sleep(0.2)
        
    return {"date": target_date, "matches": matches, "odds": fetch_odds_baseline(), "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
