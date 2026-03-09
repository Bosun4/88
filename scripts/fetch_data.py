import requests
import json
import time
import re
import random
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
    "乌兹别克斯坦女足": "Uzbekistan W", "拜仁": "Bayern Munich", "亚特兰大": "Atalanta"
}

def translate_team_name(name):
    if name in TEAM_NAME_MAPPING: 
        return TEAM_NAME_MAPPING[name]
    try:
        clean_n = name.replace("女足", " Women").replace("联", " United")
        translated = GoogleTranslator(source='zh-CN', target='en').translate(clean_n)
        return translated.replace("FC", "").strip()
    except Exception: 
        return name

# ==================== 2. Wencai 高级情报解析引擎 ====================
def get_today(offset=0):
    try:
        from zoneinfo import ZoneInfo
        return (datetime.now(ZoneInfo(TIMEZONE)) + timedelta(days=offset)).strftime("%Y-%m-%d")
    except ImportError:
        return (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")

def _safe_dict(val):
    """🔥 终极护盾：防止问彩 API 诡异地返回空列表 [] 导致 .get() 崩溃"""
    return val if isinstance(val, dict) else {}

def scrape_wencai_jczq(date=None):
    """直连问彩高级接口，提取全维度情报"""
    url = "https://edu.wencaivip.cn/api/v1.reference/matches"
    print(f"  🌐 正在建立全维度情报链路: {url}")
    ms = []
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        data = r.json()
        
        # 足球数据在 "1" 里面
        match_list = data.get("data", {}).get("matches", {}).get("1", [])
        print(f"  ✅ 发现 {len(match_list)} 场竞彩足球赛事")
        
        for item in match_list:
            try:
                # 严密防御异常结构，提取升降水
                chg = _safe_dict(item.get("change"))
                w_c = chg.get("win", 0)
                l_c = chg.get("lose", 0)
                odds_mov = f"主胜{'升水' if w_c>0 else '降水' if w_c<0 else '平稳'}，客胜{'升水' if l_c>0 else '降水' if l_c<0 else '平稳'}"

                expert_intro = item.get("intro", "")
                ana = _safe_dict(item.get("analyse"))
                base_face = ana.get("baseface", "")
                
                info = _safe_dict(item.get("information"))
                pts = _safe_dict(item.get("points"))
                
                # 提取伤停与红黑榜
                intel_pool = {
                    "h_inj": info.get("home_injury", "").replace("\n", " ") if info.get("home_injury") else "无",
                    "g_inj": info.get("guest_injury", "").replace("\n", " ") if info.get("guest_injury") else "无",
                    "h_bad": info.get("home_bad_news", "").replace("\n", " ") if info.get("home_bad_news") else "无",
                    "g_bad": info.get("guest_bad_news", "").replace("\n", " ") if info.get("guest_bad_news") else "无",
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
                    "away_rank": parse_rank(pts.get("guest_position", "")),
                    "source": "wencai_api"
                })
            except Exception as e:
                print(f"    - 解析单场异常: {item.get('home', '未知')} ({e})")
                continue
    except Exception as e: 
        print(f"  ❌ Wencai API 抓取失败: {e}")
        
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
    except Exception: pass
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
    except Exception: pass
    return {}

def fetch_h2h(hid, aid):
    if not hid or not aid: return []
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/fixtures/headtohead", headers=h, params={"h2h": f"{hid}-{aid}", "last": 5}, timeout=10)
        return [{"date": m["fixture"]["date"][:10], "home": m["teams"]["home"]["name"], "away": m["teams"]["away"]["name"], "score": f"{m['goals']['home']}-{m['goals']['away']}", "league": m["league"]["name"]} for m in r.json().get("response", [])]
    except Exception: return []

# ==================== 4. 核心兜底逻辑 ====================
def generate_stats_from_rank(rank, team_name="", total_teams=20):
    # 使用名字哈希作为种子，确保兜底数据唯一恒定
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
    except Exception: return {}

def collect_all(date=None):
    date = date or get_today()
    print(f"\n🚀 启动全量数据抓取中心 | 日期: {date}")
    
    # 彻底替换为新的问彩接口
    matches = scrape_wencai_jczq(date)
    print(f"  - 问彩初始场次: {len(matches)}")
    
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
