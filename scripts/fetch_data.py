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
    "萨索洛": "Sassuolo", "尤文图斯": "Juventus", "罗马": "Roma",
    "AC米兰": "AC Milan", "国际米兰": "Inter", "那不勒斯": "Napoli",
    "西班牙人": "Espanyol", "奥维耶多": "Real Oviedo", "皇马": "Real Madrid",
    "巴萨": "Barcelona", "马竞": "Atletico Madrid", "朝鲜女": "North Korea W",
    "中国女": "China PR W", "敦刻尔克": "Dunkerque", "兰斯": "Reims",
    "通德拉": "Tondela", "里奥阿维": "Rio Ave", "孟加拉国女足": "Bangladesh W",
    "乌兹别克斯坦女足": "Uzbekistan W", "拜仁": "Bayern Munich", "亚特兰大": "Atalanta",
    "中国台女": "Chinese Taipei W", "日本女": "Japan W", "越南女": "Vietnam W"
}

def translate_team_name(name):
    if name in TEAM_NAME_MAPPING: 
        return TEAM_NAME_MAPPING[name]
    try:
        clean_n = name.replace("女足", " Women").replace("联", " United")
        translated = GoogleTranslator(source='zh-CN', target='en').translate(clean_n)
        return translated.replace("FC", "").strip()
    except: 
        return name

# ==================== 2. 问彩高级 JSON 解析引擎 (带情报提取) ====================
def get_today(offset=0):
    try:
        from zoneinfo import ZoneInfo
        return (datetime.now(ZoneInfo(TIMEZONE)) + timedelta(days=offset)).strftime("%Y-%m-%d")
    except:
        return (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")

def scrape_wencai_jczq(date=None):
    """
    全新升级版：直连 JSON 接口，不仅提取赔率，更提取伤停与利空情报喂给AI！
    """
    url = "https://edu.wencaivip.cn/api/v1.reference/matches"
    print(f"  🌐 正在直连 Wencai 高级情报接口...")
    ms = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*"
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        data = r.json() 
        football_matches = data.get("data", {}).get("matches", {}).get("1", [])
        
        if not football_matches:
            print("  ⚠️ 接口未返回足球比赛数据。")
            return ms
            
        print(f"  ✅ 极速解析到 {len(football_matches)} 场带有绝密情报的比赛数据！")
        
        for item in football_matches:
            try:
                # 1. 基础信息
                league = item.get("cup", "")
                match_num = f"{item.get('week', '')}{item.get('week_no', '')}"
                home = item.get("home", "")
                away = item.get("guest", "")
                
                # 2. 提取SP初赔
                sp_home = float(item.get("win") or 0)
                sp_draw = float(item.get("same") or 0)
                sp_away = float(item.get("lose") or 0)
                
                # 3. 提取排名与机构深度分析
                home_rank = 0
                away_rank = 0
                match_points = "" # 机构基本面提要
                
                points_data = item.get("points", {})
                if isinstance(points_data, dict):
                    h_pos = points_data.get("home_position", "")
                    a_pos = points_data.get("guest_position", "")
                    h_match = re.search(r'\d+', str(h_pos))
                    a_match = re.search(r'\d+', str(a_pos))
                    home_rank = int(h_match.group()) if h_match else 0
                    away_rank = int(a_match.group()) if a_match else 0
                    match_points = points_data.get("match_points", "")

                # 4. 🔥 新增：提取极其珍贵的伤停与利空情报
                info_data = item.get("information", {})
                home_injury = ""
                guest_injury = ""
                home_news = ""
                guest_news = ""
                
                if isinstance(info_data, dict):
                    home_injury = info_data.get("home_injury", "").replace("\n", " | ")
                    guest_injury = info_data.get("guest_injury", "").replace("\n", " | ")
                    home_news = info_data.get("home_bad_news", "").replace("\n", " ") # 重点提取利空
                    guest_news = info_data.get("guest_bad_news", "").replace("\n", " ")
                
                if home and away:
                    ms.append({
                        "home_team": home, 
                        "away_team": away, 
                        "league": league, 
                        "match_num": match_num,
                        "home_rank": home_rank, 
                        "away_rank": away_rank,
                        "sp_home": sp_home, 
                        "sp_draw": sp_draw, 
                        "sp_away": sp_away,
                        
                        # 注入高级情报池
                        "intelligence": {
                            "home_injury": home_injury,
                            "guest_injury": guest_injury,
                            "home_bad_news": home_news,
                            "guest_bad_news": guest_news,
                            "match_points": match_points[:200] # 取前200字核心防Token超载
                        },
                        
                        "source": "wencai_api", 
                        "raw_text": json.dumps(item)[:150]
                    })
                    print(f"    {match_num} {league}: {home}[{home_rank}] vs {away}[{away_rank}] (已加载伤停情报)")
            
            except Exception as e:
                continue
                
    except Exception as e: 
        print(f"  ❌ Wencai 接口请求或解析错误: {str(e)}")
        
    return ms

# ==================== 3. API-Football 数据交互 ====================
def search_team_api(name):
    search_n = translate_team_name(name)
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/teams", headers=h, params={"search": search_n}, timeout=10)
        res = r.json().get("response", [])
        if res: 
            return {"id": res[0]["team"]["id"], "name": res[0]["team"]["name"], "logo": res[0]["team"].get("logo", "")}
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
                "avg_goals_for": str(s["goals"]["for"]["average"].get("total", "0")),
                "avg_goals_against": str(s["goals"]["against"]["average"].get("total", "0"))
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

# ==================== 4. 核心兜底逻辑 ====================
def generate_stats_from_rank(rank, team_name="", total_teams=20):
    import random
    name_hash = sum(ord(c) for c in team_name) if team_name else 0
    random.seed((rank * 7) + name_hash) 
    
    rank = rank if rank > 0 else random.randint(5, 15)
    strength = 1 - (rank - 1) / total_teams
    played = random.randint(20, 30)
    win_r, draw_r = max(0.15, min(0.75, strength * 0.6)), random.uniform(0.15, 0.28)
    wins, draws = int(played * win_r), int(played * draw_r)
    gf_p, ga_p = max(0.6, strength * 2.2), max(0.5, (1 - strength) * 1.9)
    return {
        "played": played, "wins": wins, "draws": draws, "losses": played - wins - draws,
        "goals_for": int(played * gf_p), "goals_against": int(played * ga_p),
        "avg_goals_for": str(round(gf_p, 2)), "avg_goals_against": str(round(ga_p, 2)),
        "clean_sheets": int(played * 0.25), 
        "form": "".join(random.choices("WDL", weights=[win_r, draw_r, 1-win_r-draw_r], k=5)), 
        "rank": rank, "is_generated": True
    }

# ==================== 5. 外部赔率与指挥中枢 ====================
def fetch_odds():
    try:
        r = requests.get(f"{ODDS_API_BASE}/sports/soccer/odds", params={"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "h2h,spreads,totals"}, timeout=10)
        if r.status_code != 200: return {}
        om = {}
        for ev in r.json():
            k = f"{ev['home_team']}_{ev['away_team']}"
            bks = [{"name": b['title'], "markets": {m['key']: {o['name']: o.get('price', 0) for o in m.get('outcomes', [])} for m in b.get('markets', [])}} for b in ev.get('bookmakers', [])[:5]]
            om[k] = {"commence_time": ev.get("commence_time", ""), "bookmakers": bks}
        return om
    except: return {}

def collect_all(date=None):
    date = date or get_today()
    print(f"\n🚀 启动全量数据抓取中心 | 日期: {date}")
    
    matches = scrape_wencai_jczq(date)
    print(f"  - 发现有效赛事: {len(matches)} 场")
    
    for i, m in enumerate(matches):
        print(f"  [{i+1}/{len(matches)}] 深度装载: {m['home_team']} vs {m['away_team']}")
        
        ht = search_team_api(m["home_team"])
        at = search_team_api(m["away_team"])
        
        m.update({
            "home_id": ht["id"] if ht else 0, "away_id": at["id"] if at else 0,
            "home_logo": ht["logo"] if ht else "", "away_logo": at["logo"] if at else "",
            "home_name_en": ht["name"] if ht else m["home_team"], "away_name_en": at["name"] if at else m["away_team"],
            "id": i + 1, "date": date
        })

        api_h = fetch_stats(m["home_id"]) if m["home_id"] else {}
        api_a = fetch_stats(m["away_id"]) if m["away_id"] else {}
        
        if api_h and api_h.get("played", 0) > 0:
            m["home_stats"] = api_h
        else:
            m["home_stats"] = generate_stats_from_rank(m.get("home_rank", 10), m["home_team"])

        if api_a and api_a.get("played", 0) > 0:
            m["away_stats"] = api_a
        else:
            m["away_stats"] = generate_stats_from_rank(m.get("away_rank", 10), m["away_team"])
            
        m["h2h"] = fetch_h2h(m["home_id"], m["away_id"])
        time.sleep(0.3)

    print("3. 同步 Odds-API 赔率...")
    return {"date": date, "matches": matches, "odds": fetch_odds(), "fetch_time": get_today()}
