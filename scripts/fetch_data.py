import requests
import json
import time
import re
import random
from datetime import datetime, timedelta, timezone
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
        clean = name.replace("女足", " Women").replace("联", " United")
        return GoogleTranslator(source='zh-CN', target='en').translate(clean).replace("FC", "").strip()
    except Exception: return name

def _safe_dict(val): return val if isinstance(val, dict) else {}
def _get_float(val, default=999.0):
    try: return float(val) if val is not None else default
    except Exception: return default

def scrape_wencai_jczq(date_str):
    url = f"[https://edu.wencaivip.cn/api/v1.reference/matches?date=](https://edu.wencaivip.cn/api/v1.reference/matches?date=){date_str}"
    print(f"  🌐 正在建立问彩净水链路 [{date_str}]...")
    ms = []
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        data = r.json()
        match_dict = data.get("data", {}).get("matches", {})
        match_list = []
        for key in match_dict:
            if isinstance(match_dict[key], list): match_list.extend(match_dict[key])

        for item in match_list:
            try:
                t_type = item.get("types", "")
                if str(t_type) != "1" and str(t_type) != "足球": continue

                stime = item.get("stime")
                if not stime: continue
                
                dt = datetime.fromtimestamp(stime, timezone(timedelta(hours=8)))
                jc_date = (dt - timedelta(hours=11)).strftime("%Y-%m-%d")
                if jc_date != date_str: continue

                chg = _safe_dict(item.get("change"))
                w_c, l_c = _get_float(chg.get("win"), 0), _get_float(chg.get("lose"), 0)
                odds_mov = f"主胜{'升水' if w_c>0 else '降水' if w_c<0 else '平稳'}，客胜{'升水' if l_c>0 else '降水' if l_c<0 else '平稳'}"

                info = _safe_dict(item.get("information"))
                pts = _safe_dict(item.get("points"))
                
                # 🔥 核心防御：剔除冗长的专家观点，限制伤停字符串长度，防止新闻污染
                h_inj = info.get("home_injury", "").replace("\n", " ")[:150]
                g_inj = info.get("guest_injury", "").replace("\n", " ")[:150]
                
                intel_pool = {
                    "h_inj": h_inj if h_inj else "无重大伤停",
                    "g_inj": g_inj if g_inj else "无重大伤停"
                }
                
                v2_odds = {
                    "a1": _get_float(item.get("a1")), "a2": _get_float(item.get("a2")),
                    "a3": _get_float(item.get("a3")), "a4": _get_float(item.get("a4")),
                    "s11": _get_float(item.get("s11")), "s22": _get_float(item.get("s22")),
                    "w21": _get_float(item.get("w21")), "a5": _get_float(item.get("a5")), "a6": _get_float(item.get("a6"))
                }
                
                def parse_rank(pos):
                    if not pos: return 0
                    m = re.findall(r'\d+', str(pos))
                    return int(m[0]) if m else 0

                ms.append({
                    "home_team": item.get("home", ""), "away_team": item.get("guest", ""), 
                    "league": item.get("cup", ""), "match_num": f"{item.get('week', '')}{item.get('week_no', '')}",
                    "match_time": dt.strftime('%H:%M'),
                    "sp_home": _get_float(item.get("win"), 0), "sp_draw": _get_float(item.get("same"), 0), "sp_away": _get_float(item.get("lose"), 0),
                    "odds_movement": odds_mov, "handicap_info": f"让{item.get('give_ball', 0)}",
                    "intelligence": intel_pool, 
                    "home_rank": parse_rank(pts.get("home_position", "")),
                    "away_rank": parse_rank(pts.get("guest_position", "")),
                    "votes": _safe_dict(item.get("vote")),
                    "v2_odds_dict": v2_odds 
                })
            except Exception: continue
    except Exception as e: print(f"  ❌ API 抓取失败: {e}")
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

def fetch_odds_baseline(): return {}

def collect_all(date_str):
    print(f"\n🚀 启动数据脱水过滤 | 目标日期: {date_str}")
    matches = scrape_wencai_jczq(date_str)
    for i, m in enumerate(matches):
        print(f"  [{i+1}/{len(matches)}] 锁定并净化: {m['home_team']} vs {m['away_team']}")
        ht = search_team_api(m["home_team"]); at = search_team_api(m["away_team"])
        m.update({"home_id": ht["id"] if ht else 0, "away_id": at["id"] if at else 0, "id": i + 1, "date": date_str})
        api_h = fetch_stats(m["home_id"]); api_a = fetch_stats(m["away_id"])
        m["home_stats"] = api_h if api_h.get("played", 0) > 0 else {}
        m["away_stats"] = api_a if api_a.get("played", 0) > 0 else {}
        m["h2h"] = fetch_h2h(m["home_id"], m["away_id"])
        time.sleep(0.2)
    return {"date": date_str, "matches": matches, "odds": fetch_odds_baseline()}
