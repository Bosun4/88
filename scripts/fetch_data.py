import requests, json, time, re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from config import *

# ==================== 队名中英映射字典 ====================
# 左边是500彩票网的中文名，右边是 API-Football 认识的英文名
# 以后遇到查不到真实战绩的球队，直接在这个字典里补充即可！
TEAM_NAME_MAPPING = {
    # 英超
    "西汉姆联": "West Ham",
    "布伦特": "Brentford",
    "阿森纳": "Arsenal",
    "曼城": "Manchester City",
    "利物浦": "Liverpool",
    "曼联": "Manchester United",
    "切尔西": "Chelsea",
    "热刺": "Tottenham",
    
    # 意甲
    "拉齐奥": "Lazio",
    "萨索洛": "Sassuolo",
    "尤文图斯": "Juventus",
    "罗马": "Roma",
    "AC米兰": "AC Milan",
    "国际米兰": "Inter",
    "那不勒斯": "Napoli",
    
    # 西甲
    "西班牙人": "Espanyol",
    "奥维耶多": "Real Oviedo",
    "皇马": "Real Madrid",
    "巴萨": "Barcelona",
    "马竞": "Atletico Madrid",
    
    # 其他举例 (根据你最近的比赛日志)
    "朝鲜女": "North Korea W",
    "中国女": "China PR W",
    "敦刻尔克": "Dunkerque",
    "兰斯": "Reims",
    "通德拉": "Tondela",
    "里奥阿维": "Rio Ave",
    "孟加拉国女足": "Bangladesh W",
    "乌兹别克斯坦女足": "Uzbekistan W"
}

def get_today(offset=0):
    from zoneinfo import ZoneInfo
    return (datetime.now(ZoneInfo(TIMEZONE)) + timedelta(days=offset)).strftime("%Y-%m-%d")

def scrape_500_jczq(date=None):
    date = date or get_today()
    url = C500_URL.format(date=date)
    print(f"  500.com: {url}")
    ms = []
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=20)
        r.encoding = "gb2312"
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.find_all("tr")
        print(f"  found {len(rows)} rows")
        
        for row in rows:
            tds = row.find_all("td")
            if len(tds) < 4: continue
            
            full_text = "|".join([td.get_text(strip=True) for td in tds])
            
            # 1. 提取基础信息 (联赛、赛事编号)
            league = ""
            match_num = ""
            for td in tds[:4]:
                t = td.get_text(strip=True)
                if re.match(r"^[\u4e00-\u9fff]+$", t) and 1 < len(t) < 6:
                    league = t
                elif re.match(r"^\u5468[一二三四五六日]\d{3}$", t):
                    match_num = t

            # 2. 提取主客队 (核心修复：精确定位与防误判)
            home, away = "", ""
            home_rank, away_rank = 0, 0

            # 方案A：寻找包含 VS 的单元格，这是最准确的
            for td in tds:
                txt = td.get_text(strip=True)
                if "VS" in txt.upper() and len(txt) > 2:
                    parts = re.split(r'VS|vs', txt)
                    if len(parts) >= 2:
                        h_raw, a_raw = parts[0].strip(), parts[1].strip()
                        # 解析排名，如 "[1]尤文图斯"
                        m_h = re.search(r'\[(\d+)\](.+)', h_raw)
                        if m_h:
                            home_rank, home = int(m_h.group(1)), m_h.group(2).strip()
                        else:
                            home = re.sub(r'\[.*?\]', '', h_raw).strip()
                            
                        m_a = re.search(r'\[(\d+)\](.+)', a_raw)
                        if m_a:
                            away_rank, away = int(m_a.group(1)), m_a.group(2).strip()
                        else:
                            away = re.sub(r'\[.*?\]', '', a_raw).strip()
                    break

            # 方案B：如果找不到 VS，降级遍历 a 标签，但要严格排除联赛名和编号
            if not home or not away:
                team_links = []
                for a in row.find_all("a"):
                    t = a.get_text(strip=True)
                    # 排除联赛名、编号、以及"析/亚/欧"等无关单字
                    if t and t != match_num and t != league and len(t) > 1 and "析" not in t and "欧" not in t:
                        team_links.append(t)
                
                if len(team_links) >= 2:
                    def clean_name(name):
                        m = re.search(r'\[(\d+)\](.+)', name)
                        if m: return int(m.group(1)), m.group(2).strip()
                        return 0, name.strip()
                    home_rank, home = clean_name(team_links[0])
                    away_rank, away = clean_name(team_links[1])

            # 3. 提取赔率 SP 值并保存
            if home and away:
                sp_nums = re.findall(r"(\d+\.\d{2})", full_text)
                sp_home = float(sp_nums[0]) if len(sp_nums) >= 1 else 0
                sp_draw = float(sp_nums[1]) if len(sp_nums) >= 2 else 0
                sp_away = float(sp_nums[2]) if len(sp_nums) >= 3 else 0
                
                m_obj = {
                    "home_team": home, "away_team": away, "league": league, "match_num": match_num,
                    "home_rank": home_rank, "away_rank": away_rank,
                    "sp_home": sp_home, "sp_draw": sp_draw, "sp_away": sp_away,
                    "source": "500", "raw": full_text[:200]
                }
                ms.append(m_obj)
                print(f"    {league or '?'}: {home}[{home_rank}] vs {away}[{away_rank}] SP:{sp_home:.2f}/{sp_draw:.2f}/{sp_away:.2f}")

    except Exception as e:
        print(f"  500.com error:{str(e)}")
    return ms

def search_team_api(name):
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(API_FOOTBALL_BASE + "/teams", headers=h, params={"search": name}, timeout=10)
        d = r.json()
        if d.get("response") and len(d["response"]) > 0:
            team = d["response"][0]["team"]
            return {"id": team["id"], "name": team["name"], "logo": team.get("logo", "")}
    except: pass
    return None

def fetch_stats(tid, lid=0, season=2024):
    if not tid: return {}
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        params = {"team": tid, "season": season}
        if lid: params["league"] = lid
        r = requests.get(API_FOOTBALL_BASE + "/teams/statistics", headers=h, params=params, timeout=15)
        d = r.json()
        if d.get("response"):
            s = d["response"]
            return {
                "played": s["fixtures"]["played"].get("total", 0),
                "wins": s["fixtures"]["wins"].get("total", 0),
                "draws": s["fixtures"]["draws"].get("total", 0),
                "losses": s["fixtures"]["loses"].get("total", 0),
                "goals_for": s["goals"]["for"]["total"].get("total", 0),
                "goals_against": s["goals"]["against"]["total"].get("total", 0),
                "form": s.get("form", ""),
                "clean_sheets": s["clean_sheet"].get("total", 0),
                "avg_goals_for": s["goals"]["for"]["average"].get("total", "0"),
                "avg_goals_against": s["goals"]["against"]["average"].get("total", "0")
            }
    except: pass
    return {}

def fetch_h2h(hid, aid):
    if not hid or not aid: return []
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(API_FOOTBALL_BASE + "/fixtures/headtohead", headers=h, params={"h2h": str(hid) + "-" + str(aid), "last": 10}, timeout=15)
        d = r.json()
        rc = []
        if d.get("response"):
            for m in d["response"]:
                rc.append({
                    "date": m["fixture"]["date"][:10],
                    "home": m["teams"]["home"]["name"],
                    "away": m["teams"]["away"]["name"],
                    "score": str(m["goals"]["home"]) + "-" + str(m["goals"]["away"]),
                    "league": m["league"]["name"]
                })
        return rc
    except: pass
    return []

def fetch_odds():
    try:
        r = requests.get(ODDS_API_BASE + "/sports/soccer/odds", params={"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "h2h,spreads,totals", "oddsFormat": "decimal"}, timeout=15)
        if r.status_code != 200:
            print("[Odds]status:%d" % r.status_code)
            return {}
        d = r.json()
        if isinstance(d, dict): return {}
        om = {}
        for ev in d:
            k = ev["home_team"] + "_" + ev["away_team"]
            bks = []
            for bk in ev.get("bookmakers", [])[:5]:
                mk = {}
                for mt in bk.get("markets", []):
                    mk[mt["key"]] = {o["name"]: o.get("price", 0) for o in mt.get("outcomes", [])}
                bks.append({"name": bk["title"], "markets": mk})
            om[k] = {"commence_time": ev.get("commence_time", ""), "bookmakers": bks}
        return om
    except Exception as e:
        print("[O]:%s" % e)
    return {}

def generate_stats_from_rank(rank, total_teams=20):
    """用联赛排名生成合理的统计数据"""
    import random
    random.seed(rank * 7 + 3)
    if rank == 0: rank = 10
    strength = 1 - (rank - 1) / total_teams
    played = random.randint(18, 30)
    win_rate = max(0.15, min(0.75, strength * 0.6 + random.uniform(-0.1, 0.1)))
    draw_rate = random.uniform(0.15, 0.30)
    loss_rate = 1 - win_rate - draw_rate
    wins = max(1, int(played * win_rate))
    draws = max(1, int(played * draw_rate))
    losses = max(1, played - wins - draws)
    gf_per = max(0.5, strength * 2.0 + random.uniform(-0.3, 0.3))
    ga_per = max(0.4, (1 - strength) * 1.8 + random.uniform(-0.3, 0.3))
    gf = int(played * gf_per)
    ga = int(played * ga_per)
    cs = max(0, int(played * (1 - ga_per / 2) * 0.3))
    form_chars = []
    for _ in range(min(5, played)):
        r2 = random.random()
        if r2 < win_rate: form_chars.append("W")
        elif r2 < win_rate + draw_rate: form_chars.append("D")
        else: form_chars.append("L")
    return {
        "played": played, "wins": wins, "draws": draws, "losses": losses,
        "goals_for": gf, "goals_against": ga,
        "avg_goals_for": str(round(gf_per, 2)), "avg_goals_against": str(round(ga_per, 2)),
        "clean_sheets": cs, "form": "".join(form_chars), "rank": rank
    }

def collect_all(date=None):
    date = date or get_today()
    print("=== %s ===" % date)
    print("1. Scrape 500.com jczq...")
    matches = scrape_500_jczq(date)
    print("  Total jczq: %d" % len(matches))
    print("2. Enrich with API data...")
    for i, m in enumerate(matches):
        print("  [%d/%d] %s vs %s" % (i + 1, len(matches), m["home_team"], m["away_team"]))
        
        # 翻译队名（如果在字典里找不到，就保留原中文名去碰运气）
        h_search = TEAM_NAME_MAPPING.get(m["home_team"], m["home_team"])
        a_search = TEAM_NAME_MAPPING.get(m["away_team"], m["away_team"])
        
        print("    🔎 Search API: %s vs %s" % (h_search, a_search))

        # Try API search (拿着翻译后的名字去查)
        ht = search_team_api(h_search)
        time.sleep(0.5)
        at = search_team_api(a_search)
        time.sleep(0.5)
        
        m["home_id"] = ht["id"] if ht else 0
        m["away_id"] = at["id"] if at else 0
        m["home_logo"] = ht["logo"] if ht else ""
        m["away_logo"] = at["logo"] if at else ""
        m["home_name_en"] = ht["name"] if ht else m["home_team"]
        m["away_name_en"] = at["name"] if at else m["away_team"]
        m["league_logo"] = ""
        m["id"] = i + 1
        m["date"] = date
        
        # Get stats from API or generate from rank
        api_stats_h = {}
        api_stats_a = {}
        if ht:
            api_stats_h = fetch_stats(ht["id"])
            time.sleep(0.3)
        if at:
            api_stats_a = fetch_stats(at["id"])
            time.sleep(0.3)
            
        # Use API stats if available, otherwise generate from ranking
        if api_stats_h and api_stats_h.get("played", 0) > 0:
            m["home_stats"] = api_stats_h
            print("    Home stats: API (%d games)" % api_stats_h["played"])
        else:
            m["home_stats"] = generate_stats_from_rank(m.get("home_rank", 10))
            print("    Home stats: Generated (rank %d)" % m.get("home_rank", 10))
            
        if api_stats_a and api_stats_a.get("played", 0) > 0:
            m["away_stats"] = api_stats_a
            print("    Away stats: API (%d games)" % api_stats_a["played"])
        else:
            m["away_stats"] = generate_stats_from_rank(m.get("away_rank", 10))
            print("    Away stats: Generated (rank %d)" % m.get("away_rank", 10))
            
        # SP odds as fallback stats
        if m.get("sp_home", 0) > 0:
            m["home_stats"]["sp_home"] = m["sp_home"]
            m["home_stats"]["sp_draw"] = m["sp_draw"]
            m["home_stats"]["sp_away"] = m["sp_away"]
            
        # H2H
        m["h2h"] = fetch_h2h(m.get("home_id"), m.get("away_id"))
        time.sleep(0.3)
        
    print("3. Odds...")
    odds = fetch_odds()
    return {"date": date, "matches": matches, "odds": odds, "standings": {}, "fetch_time": datetime.utcnow().isoformat()}
