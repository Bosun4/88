import requests
import json
import time
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from config import *

# ==================== 1. 队名翻译系统 (手动映射 + 自动翻译) ====================
# 保留并扩充你提供的映射字典
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
    "乌兹别克斯坦女足": "Uzbekistan W"
}

def translate_team_name(name):
    """双重翻译机制：先查字典，失败后调用Google翻译"""
    if name in TEAM_NAME_MAPPING:
        return TEAM_NAME_MAPPING[name]
    try:
        # 自动直译：针对没有录入字典的冷门球队或女足
        clean_n = name.replace("女足", " Women").replace("联", " United")
        translated = GoogleTranslator(source='zh-CN', target='en').translate(clean_n)
        return translated.replace("FC", "").strip()
    except:
        return name

# ==================== 2. 500.com 解析引擎 (完全还原你的原始逻辑) ====================
def get_today(offset=0):
    from zoneinfo import ZoneInfo
    return (datetime.now(ZoneInfo(TIMEZONE)) + timedelta(days=offset)).strftime("%Y-%m-%d")

def scrape_500_jczq(date=None):
    date = date or get_today()
    url = C500_URL.format(date=date)
    print(f"  🌐 正在连接 500.com 数据源: {url}")
    ms = []
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=20)
        r.encoding = "gb2312"
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.find_all("tr")
        
        for row in rows:
            tds = row.find_all("td")
            if len(tds) < 4: continue
            
            full_text = "|".join([td.get_text(strip=True) for td in tds])
            
            # 提取联赛与场次编号
            league, match_num = "", ""
            for td in tds[:4]:
                t = td.get_text(strip=True)
                if re.match(r"^[\u4e00-\u9fff]+$", t) and 1 < len(t) < 6:
                    league = t
                elif re.match(r"^\u5468[一二三四五六日]\d{3}$", t):
                    match_num = t

            # 核心还原：提取主客队排名与名字 (方案A & B)
            home, away = "", ""
            home_rank, away_rank = 0, 0

            # 方案A：VS 分割逻辑
            for td in tds:
                txt = td.get_text(strip=True)
                if "VS" in txt.upper() and len(txt) > 2:
                    parts = re.split(r'VS|vs', txt)
                    if len(parts) >= 2:
                        h_raw, a_raw = parts[0].strip(), parts[1].strip()
                        m_h = re.search(r'\[(\d+)\](.+)', h_raw)
                        home_rank, home = (int(m_h.group(1)), m_h.group(2).strip()) if m_h else (0, re.sub(r'\[.*?\]', '', h_raw).strip())
                        m_a = re.search(r'\[(\d+)\](.+)', a_raw)
                        away_rank, away = (int(m_a.group(1)), m_a.group(2).strip()) if m_a else (0, re.sub(r'\[.*?\]', '', a_raw).strip())
                    break

            # 方案B：降级 a 标签提取
            if not home or not away:
                team_links = [a.get_text(strip=True) for a in row.find_all("a") if len(a.get_text(strip=True)) > 1 and a.get_text(strip=True) not in [match_num, league, "析", "亚", "欧"]]
                if len(team_links) >= 2:
                    def clean_n(n):
                        m = re.search(r'\[(\d+)\](.+)', n)
                        return (int(m.group(1)), m.group(2).strip()) if m else (0, n.strip())
                    home_rank, home = clean_n(team_links[0])
                    away_rank, away = clean_n(team_links[1])

            # 提取赔率 SP 值
            if home and away:
                sp_nums = re.findall(r"(\d+\.\d{2})", full_text)
                sp_h = float(sp_nums[0]) if len(sp_nums) >= 1 else 0
                sp_d = float(sp_nums[1]) if len(sp_nums) >= 2 else 0
                sp_a = float(sp_nums[2]) if len(sp_nums) >= 3 else 0
                
                ms.append({
                    "home_team": home, "away_team": away, "league": league, "match_num": match_num,
                    "home_rank": home_rank, "away_rank": away_rank,
                    "sp_home": sp_h, "sp_draw": sp_d, "sp_away": sp_a,
                    "source": "500", "raw_text": full_text[:150]
                })
                print(f"    ✅ 抓取成功: {match_num} {home}[{home_rank}] vs {away}[{away_rank}]")
    except Exception as e:
        print(f"  ❌ 500.com 解析错误: {str(e)}")
    return ms

# ==================== 3. API-Football 数据交互 (ID/统计/交锋) ====================
def search_team_api(name):
    """带自动翻译补全的 API 搜索"""
    search_n = translate_team_name(name)
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/teams", headers=h, params={"search": search_n}, timeout=10)
        res = r.json().get("response", [])
        if res:
            t = res[0]["team"]
            return {"id": t["id"], "name": t["name"], "logo": t.get("logo", "")}
    except: pass
    return None

def fetch_stats(tid, season=2025):
    if not tid: return {}
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/teams/statistics", headers=h, params={"team": tid, "season": season}, timeout=15)
        s = r.json().get("response", {})
        if s:
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
        r = requests.get(f"{API_FOOTBALL_BASE}/fixtures/headtohead", headers=h, params={"h2h": f"{hid}-{aid}", "last": 10}, timeout=15)
        res = r.json().get("response", [])
        return [{"date": m["fixture"]["date"][:10], "home": m["teams"]["home"]["name"], "away": m["teams"]["away"]["name"], "score": f"{m['goals']['home']}-{m['goals']['away']}", "league": m["league"]["name"]} for m in res]
    except: return []

# ==================== 4. 核心兜底逻辑 (防 1-0 爆冷) ====================
def generate_stats_from_rank(rank, total_teams=20):
    """如果 API 没数据，根据 500 网爬到的排名生成合理的量化特征数据"""
    import random
    random.seed(rank * 7)
    rank = rank if rank > 0 else 10
    strength = 1 - (rank - 1) / total_teams
    played = random.randint(20, 30)
    win_r, draw_r = max(0.15, min(0.75, strength * 0.6)), random.uniform(0.15, 0.28)
    wins, draws = int(played * win_r), int(played * draw_r)
    gf_p, ga_p = max(0.6, strength * 2.2), max(0.5, (1 - strength) * 1.9)
    return {
        "played": played, "wins": wins, "draws": draws, "losses": played - wins - draws,
        "goals_for": int(played * gf_p), "goals_against": int(played * ga_p),
        "avg_goals_for": str(round(gf_p, 2)), "avg_goals_against": str(round(ga_p, 2)),
        "clean_sheets": int(played * 0.25), "form": "".join(random.choices("WDL", k=5)), "rank": rank, "is_generated": True
    }

# ==================== 5. 外部赔率与指挥中枢 ====================
def fetch_odds():
    try:
        r = requests.get(f"{ODDS_API_BASE}/sports/soccer/odds", params={"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "h2h,spreads,totals"}, timeout=15)
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
        
        # 1. 翻译并搜索 ID
        ht = search_team_api(m["home_team"])
        at = search_team_api(m["away_team"])
        
        m.update({
            "home_id": ht["id"] if ht else 0, "away_id": at["id"] if at else 0,
            "home_logo": ht["logo"] if ht else "", "away_logo": at["logo"] if at else "",
            "home_name_en": ht["name"] if ht else m["home_team"], "away_name_en": at["name"] if at else m["away_team"],
            "id": i + 1, "date": date
        })

        # 2. 抓取战绩数据 (带排名兜底)
        api_h = fetch_stats(m["home_id"]) if m["home_id"] else {}
        api_a = fetch_stats(m["away_id"]) if m["away_id"] else {}
        
        # 优先使用 API 数据，搜不到则用你爬到的“主客排名”生成数据
        if api_h and api_h.get("played", 0) > 0:
            m["home_stats"] = api_h
            print(f"    -> 主队: API数据装载成功")
        else:
            m["home_stats"] = generate_stats_from_rank(m.get("home_rank", 10))
            print(f"    -> 主队: 缺失API，已使用排名[{m['home_rank']}]兜底")

        if api_a and api_a.get("played", 0) > 0:
            m["away_stats"] = api_a
            print(f"    -> 客队: API数据装载成功")
        else:
            m["away_stats"] = generate_stats_from_rank(m.get("away_rank", 10))
            print(f"    -> 客队: 缺失API，已使用排名[{m['away_rank']}]兜底")
            
        m["h2h"] = fetch_h2h(m["home_id"], m["away_id"])
        time.sleep(0.5)

    print("3. 同步 Odds-API 赔率...")
    return {"date": date, "matches": matches, "odds": fetch_odds(), "fetch_time": datetime.utcnow().isoformat()}
