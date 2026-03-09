import requests
import json
import time
import re
from datetime import datetime, timedelta
from deep_translator import GoogleTranslator
from config import *

# ==================== 1. 队名映射与翻译系统 ====================
TEAM_NAME_MAPPING = {
    "西汉姆联": "West Ham", "布伦特": "Brentford", "阿森纳": "Arsenal",
    "曼城": "Manchester City", "利物浦": "Liverpool", "曼联": "Manchester United",
    "切尔西": "Chelsea", "热刺": "Tottenham", "拉齐奥": "Lazio",
    "萨索洛": "Sassuolo", "皇马": "Real Madrid", "巴萨": "Barcelona",
    "马竞": "Atletico Madrid", "拜仁": "Bayern Munich", "亚特兰大": "Atalanta"
}

def translate_team_name(name):
    if name in TEAM_NAME_MAPPING: 
        return TEAM_NAME_MAPPING[name]
    try:
        clean_n = name.replace("女足", " Women").replace("联", " United").replace("台女", " Taipei Women")
        translated = GoogleTranslator(source='zh-CN', target='en').translate(clean_n)
        return translated.replace("FC", "").strip()
    except: 
        return name

# ==================== 2. 问彩高级 JSON 解析引擎 ====================
def get_today(offset=0):
    try:
        from zoneinfo import ZoneInfo
        return (datetime.now(ZoneInfo(TIMEZONE)) + timedelta(days=offset)).strftime("%Y-%m-%d")
    except:
        return (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")

def scrape_wencai_jczq(date=None):
    url = "https://edu.wencaivip.cn/api/v1.reference/matches"
    print(f"  🌐 正在直连 Wencai 高级情报与盘口接口...")
    ms = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*"
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        data = r.json() 
        match_list = data.get("data", {}).get("matches", {}).get("1", [])
        
        for item in match_list:
            try:
                league = item.get("cup", "")
                m_num = f"{item.get('week', '')}{item.get('week_no', '')}"
                home = item.get("home", "")
                away = item.get("guest", "")
                
                sp_home = float(item.get("win") or 0)
                sp_draw = float(item.get("same") or 0)
                sp_away = float(item.get("lose") or 0)
                
                # 提取赔率异动 (1=升, -1=降, 0=稳)
                chg = item.get("change", {})
                w_c = chg.get("win", 0)
                l_c = chg.get("lose", 0)
                w_txt = "升水(机构看衰)" if w_c > 0 else "降水(机构防范)" if w_c < 0 else "平稳"
                l_txt = "升水(机构看衰)" if l_c > 0 else "降水(机构防范)" if l_c < 0 else "平稳"
                odds_movement = f"主胜{w_txt}，客胜{l_txt}"

                # 提取盘口
                give_ball = item.get("give_ball", 0)
                hd_str = f"让{give_ball} ({item.get('hhad_win', '-')}/{item.get('hhad_same', '-')}/{item.get('hhad_lose', '-')})"
                
                # 提取深度研报与推介
                expert_intro = item.get("intro", "")
                analyse_data = item.get("analyse", {})
                base_face = analyse_data.get("baseface", "") if isinstance(analyse_data, dict) else ""

                # 提取伤停情报
                info = item.get("information", {})
                intel = {
                    "h_inj": info.get("home_injury", "").replace("\n", " | "),
                    "g_inj": info.get("guest_injury", "").replace("\n", " | "),
                    "h_bad": info.get("home_bad_news", "").replace("\n", " "),
                    "g_bad": info.get("guest_bad_news", "").replace("\n", " "),
                    "match_points": item.get("points", {}).get("match_points", "")
                }
                
                # 排名解析
                def parse_rank(pos_str):
                    if not pos_str: return 0
                    m = re.search(r'\d+', str(pos_str))
                    return int(m.group()) if m else 0

                if home and away:
                    ms.append({
                        "home_team": home, "away_team": away, "league": league, "match_num": m_num,
                        "sp_home": sp_home, "sp_draw": sp_draw, "sp_away": sp_away,
                        "odds_movement": odds_movement, "handicap_info": hd_str,
                        "intelligence": intel, "expert_intro": expert_intro, "base_face": base_face,
                        "home_rank": parse_rank(item.get("points", {}).get("home_position")),
                        "away_rank": parse_rank(item.get("points", {}).get("guest_position")),
                        "source": "wencai_api"
                    })
            except:
                continue
    except Exception as e: 
        print(f"  ❌ API 解析错误: {str(e)}")
    return ms

def search_team_api(name):
    search_n = translate_team_name(name)
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/teams", headers=h, params={"search": search_n}, timeout=10)
        res = r.json().get("response", [])
        if res: return {"id": res[0]["team"]["id"], "name": res[0]["team"]["name"], "logo": res[0]["team"].get("logo", "")}
    except: pass
    return None

def fetch_stats(tid, season=2024): 
    if not tid: return {}
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/teams/statistics", headers=h, params={"team": tid, "season": season}, timeout=10)
        s = r.json().get("response", {})
        if s and "fixtures" in s:
            return {
                "played": s["fixtures"]["played"].get("total", 0),
                "wins": s["fixtures"]["wins"].get("total", 0),
                "draws": s["fixtures"]["draws"].get("total", 0),
                "losses": s["fixtures"]["loses"].get("total", 0),
                "goals_for": s["goals"]["for"]["total"].get("total", 0),
                "goals_against": s["goals"]["against"]["total"].get("total", 0),
                "form": s.get("form", ""),
                "clean_sheets": s["clean_sheet"].get("total", 0),
                "avg_goals_for": str(s["goals"]["for"]["average"].get("total", "0.0")),
                "avg_goals_against": str(s["goals"]["against"]["average"].get("total", "0.0"))
            }
    except: pass
    return {}

def fetch_h2h(hid, aid):
    if not hid or not aid: return []
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/fixtures/headtohead", headers=h, params={"h2h": f"{hid}-{aid}", "last": 5}, timeout=10)
        return [{"date": m["fixture"]["date"][:10], "home": m["teams"]["home"]["name"], "away": m["teams"]["away"]["name"], "score": f"{m['goals']['home']}-{m['goals']['away']}", "league": m["league"]["name"]} for m in r.json().get("response", [])]
    except: return []

def generate_stats_from_rank(rank, team_name="", total_teams=20):
    import random
    name_hash = sum(ord(c) for c in team_name) if team_name else 0
    random.seed((rank * 7) + name_hash) 
    rank = rank if rank > 0 else random.randint(8, 14)
    strength = 1 - (rank - 1) / total_teams
    played = random.randint(22, 28)
    win_r, draw_r = max(0.15, min(0.75, strength * 0.6)), random.uniform(0.18, 0.28)
    wins, draws = int(played * win_r), int(played * draw_r)
    return {
        "played": played, "wins": wins, "draws": draws, "losses": played - wins - draws,
        "goals_for": int(played * max(0.7, strength * 2.1)), "goals_against": int(played * max(0.6, (1 - strength) * 1.8)),
        "avg_goals_for": str(round(max(0.7, strength * 2.1), 2)), "avg_goals_against": str(round(max(0.6, (1 - strength) * 1.8), 2)),
        "clean_sheets": int(played * 0.22), "form": "".join(random.choices("WDL", weights=[win_r, draw_r, 1-win_r-draw_r], k=5)), 
        "rank": rank, "is_generated": True
    }

def collect_all(date=None):
    date = date or get_today()
    matches = scrape_wencai_jczq(date)
    for i, m in enumerate(matches):
        print(f"  [{i+1}/{len(matches)}] 深度透视: {m['home_team']}")
        ht = search_team_api(m["home_team"])
        at = search_team_api(m["away_team"])
        m.update({
            "home_id": ht["id"] if ht else 0, "away_id": at["id"] if at else 0,
            "home_logo": ht["logo"] if ht else "", "away_logo": at["logo"] if at else "",
            "id": i + 1, "date": date
        })
        api_h = fetch_stats(m["home_id"]) if m["home_id"] else {}
        api_a = fetch_stats(m["away_id"]) if m["away_id"] else {}
        m["home_stats"] = api_h if api_h.get("played", 0) > 0 else generate_stats_from_rank(m["home_rank"], m["home_team"])
        m["away_stats"] = api_a if api_a.get("played", 0) > 0 else generate_stats_from_rank(m["away_rank"], m["away_team"])
        m["h2h"] = fetch_h2h(m["home_id"], m["away_id"])
        time.sleep(0.3)
    return {"date": date, "matches": matches, "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
