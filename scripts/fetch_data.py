import requests
import json
import time
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
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
    "乌兹别克斯坦女足": "Uzbekistan W", "拜仁": "Bayern Munich", "亚特兰大": "Atalanta"
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

# ==================== 2. 500.com 解析引擎 ====================
def get_today(offset=0):
    from zoneinfo import ZoneInfo
    return (datetime.now(ZoneInfo(TIMEZONE)) + timedelta(days=offset)).strftime("%Y-%m-%d")

def scrape_500_jczq(date=None):
    date = date or get_today()
    url = C500_URL.format(date=date)
    print(f"  🌐 正在连接 500.com: {url}")
    ms = []
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=20)
        r.encoding = "gb2312"
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.find_all("tr")
        print(f"  ✅ 发现 {len(rows)} 行网页数据")
        
        for row in rows:
            tds = row.find_all("td")
            if len(tds) < 4: continue
            
            full_text = "|".join([td.get_text(strip=True) for td in tds])
            league, match_num = "", ""
            
            for td in tds[:4]:
                t = td.get_text(strip=True)
                if re.match(r"^[\u4e00-\u9fff]+$", t) and 1 < len(t) < 6: 
                    league = t
                elif re.match(r"^\u5468[一二三四五六日]\d{3}$", t): 
                    match_num = t

            home, away = "", ""
            home_rank, away_rank = 0, 0

            # 方案A: VS 分割
            for td in tds:
                txt = td.get_text(strip=True)
                if "VS" in txt.upper() and len(txt) > 2:
                    parts = re.split(r'VS|vs', txt)
                    if len(parts) >= 2:
                        m_h = re.search(r'\[(\d+)\](.+)', parts[0].strip())
                        home_rank, home = (int(m_h.group(1)), m_h.group(2).strip()) if m_h else (0, re.sub(r'\[.*?\]', '', parts[0]).strip())
                        
                        m_a = re.search(r'\[(\d+)\](.+)', parts[1].strip())
                        away_rank, away = (int(m_a.group(1)), m_a.group(2).strip()) if m_a else (0, re.sub(r'\[.*?\]', '', parts[1]).strip())
                    break

            # 方案B: a标签降级提取
            if not home or not away:
                tls = []
                for a in row.find_all("a"):
                    t = a.get_text(strip=True)
                    if len(t) > 1 and t not in [match_num, league, "析", "亚", "欧"]:
                        tls.append(t)
                
                if len(tls) >= 2:
                    def cn(n):
                        m = re.search(r'\[(\d+)\](.+)', n)
                        return (int(m.group(1)), m.group(2).strip()) if m else (0, n.strip())
                    home_rank, home = cn(tls[0])
                    away_rank, away = cn(tls[1])

            if home and away:
                sps = re.findall(r"(\d+\.\d{2})", full_text)
                ms.append({
                    "home_team": home, "away_team": away, "league": league, "match_num": match_num,
                    "home_rank": home_rank, "away_rank": away_rank,
                    "sp_home": float(sps[0]) if len(sps)>0 else 0, 
                    "sp_draw": float(sps[1]) if len(sps)>1 else 0, 
                    "sp_away": float(sps[2]) if len(sps)>2 else 0,
                    "source": "500", "raw_text": full_text[:150]
                })
    except Exception as e: 
        print(f"  ❌ 500.com 解析错误: {str(e)}")
    return ms

# ==================== 3. API 数据交互 ====================
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

# ==================== 4. 核心兜底逻辑 (加入名字哈希防重) ====================
def generate_stats_from_rank(rank, team_name="", total_teams=20):
    import random
    # 使用名字哈希作为种子，确保只要名字不同，生成的兜底数据就绝对不一样！
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
    
    matches = scrape_500_jczq(date)
    print(f"  - 500网初始场次: {len(matches)}")
    
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
    return {"date": date, "matches": matches, "odds": fetch_odds(), "fetch_time": datetime.utcnow().isoformat()}
