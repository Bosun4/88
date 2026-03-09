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
    """安全翻译队名"""
    if name in TEAM_NAME_MAPPING: 
        return TEAM_NAME_MAPPING[name]
    try:
        clean_name = name.replace("女足", " Women").replace("联", " United").replace("台女", " Taipei Women")
        translated = GoogleTranslator(source='zh-CN', target='en').translate(clean_name)
        return translated.replace("FC", "").strip()
    except Exception as e: 
        print(f"      [翻译警告] {name} 翻译失败: {e}")
        return name

def get_today(offset=0):
    try:
        from zoneinfo import ZoneInfo
        now_time = datetime.now(ZoneInfo(TIMEZONE))
        return (now_time + timedelta(days=offset)).strftime("%Y-%m-%d")
    except Exception:
        return (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")

def scrape_wencai_jczq(date=None):
    """抓取 Wencai 高级接口数据 (含专家点评、伤停、异动)"""
    url = "https://edu.wencaivip.cn/api/v1.reference/matches"
    print(f"  🌐 正在建立全维度情报链路: {url}")
    ms = []
    
    headers = {
        "User-Agent": "Mozilla/5.0", 
        "Accept": "application/json"
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        response_json = r.json()
        match_list = response_json.get("data", {}).get("matches", {}).get("1", [])
        
        for item in match_list:
            try:
                # 提取赔率异动
                chg = item.get("change", {})
                w_c = chg.get("win", 0)
                l_c = chg.get("lose", 0)
                
                win_text = '升水' if w_c > 0 else '降水' if w_c < 0 else '平稳'
                lose_text = '升水' if l_c > 0 else '降水' if l_c < 0 else '平稳'
                odds_mov = f"主胜{win_text}，客胜{lose_text}"

                # 提取文本情报
                expert_intro = item.get("intro", "")
                
                analyse_data = item.get("analyse", {})
                base_face = analyse_data.get("baseface", "") if isinstance(analyse_data, dict) else ""
                
                # 提取伤停 (强制类型检查防止 'list' object 崩溃)
                info = item.get("information", {})
                if not isinstance(info, dict): 
                    info = {}
                    
                intel_pool = {
                    "h_inj": info.get("home_injury", "").replace("\n", " "),
                    "g_inj": info.get("guest_injury", "").replace("\n", " "),
                    "h_bad": info.get("home_bad_news", "").replace("\n", " "),
                    "g_bad": info.get("guest_bad_news", "").replace("\n", " "),
                }
                
                points_data = item.get("points", {})
                if isinstance(points_data, dict):
                    intel_pool["match_points"] = points_data.get("match_points", "")
                else:
                    intel_pool["match_points"] = ""
                
                # 提取排名
                def parse_rank(pos_value):
                    if not pos_value: return 0
                    nums = re.findall(r'\d+', str(pos_value))
                    return int(nums[0]) if nums else 0

                home_pos = points_data.get("home_position") if isinstance(points_data, dict) else ""
                away_pos = points_data.get("guest_position") if isinstance(points_data, dict) else ""

                ms.append({
                    "home_team": item.get("home", ""), 
                    "away_team": item.get("guest", ""), 
                    "league": item.get("cup", ""), 
                    "match_num": f"{item.get('week', '')}{item.get('week_no', '')}",
                    "sp_home": float(item.get("win") or 0), 
                    "sp_draw": float(item.get("same") or 0), 
                    "sp_away": float(item.get("lose") or 0),
                    "odds_movement": odds_mov, 
                    "handicap_info": f"让{item.get('give_ball', 0)}",
                    "intelligence": intel_pool, 
                    "expert_intro": expert_intro, 
                    "base_face": base_face,
                    "home_rank": parse_rank(home_pos),
                    "away_rank": parse_rank(away_pos)
                })
            except Exception as item_err:
                print(f"    - 解析跳过单场异常: {item_err}")
                continue
                
    except Exception as api_err: 
        print(f"  ❌ Wencai API 解析彻底失败: {api_err}")
        
    return ms

def search_team_api(name):
    """搜索 API-Football 球队 ID"""
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        translated_name = translate_team_name(name)
        r = requests.get(f"{API_FOOTBALL_BASE}/teams", headers=h, params={"search": translated_name}, timeout=10)
        res = r.json().get("response", [])
        if res: 
            return {
                "id": res[0]["team"]["id"], 
                "name": res[0]["team"]["name"], 
                "logo": res[0]["team"].get("logo", "")
            }
    except Exception: 
        pass
    return None

def fetch_stats(tid, season=2024): 
    """获取球队数据统计"""
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
    except Exception: 
        pass
    return {}

def fetch_h2h(hid, aid):
    """获取 H2H 历史记录"""
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/fixtures/headtohead", headers=h, params={"h2h": f"{hid}-{aid}", "last": 5}, timeout=10)
        results = []
        for m in r.json().get("response", []):
            results.append({
                "date": m["fixture"]["date"][:10], 
                "home": m["teams"]["home"]["name"], 
                "away": m["teams"]["away"]["name"], 
                "score": f"{m['goals']['home']}-{m['goals']['away']}", 
                "league": m["league"]["name"]
            })
        return results
    except Exception: 
        return []

def generate_stats_from_rank(rank, team_name="", total_teams=20):
    """基于排名和名字哈希生成稳定的兜底数据"""
    name_hash = sum(ord(c) for c in team_name) if team_name else 0
    random.seed((rank * 7) + name_hash) 
    
    rank = rank if rank > 0 else random.randint(8, 14)
    strength = 1 - (rank - 1) / total_teams
    played = random.randint(22, 28)
    
    win_r = max(0.15, min(0.75, strength * 0.65))
    draw_r = random.uniform(0.18, 0.28)
    
    wins = int(played * win_r)
    draws = int(played * draw_r)
    
    gf_per = max(0.7, strength * 2.1)
    ga_per = max(0.6, (1 - strength) * 1.8)
    
    form_list = random.choices("WDL", weights=[win_r, draw_r, 1-win_r-draw_r], k=5)
    
    return {
        "played": played, 
        "wins": wins, 
        "draws": draws, 
        "losses": max(0, played - wins - draws),
        "goals_for": int(played * gf_per), 
        "goals_against": int(played * ga_per),
        "avg_goals_for": str(round(gf_per, 2)), 
        "avg_goals_against": str(round(ga_per, 2)),
        "clean_sheets": int(played * 0.25), 
        "form": "".join(form_list), 
        "rank": rank,
        "is_generated": True
    }

def fetch_odds_baseline():
    """抓取 Odds-API 全球主流机构基准赔率"""
    try:
        r = requests.get(f"{ODDS_API_BASE}/sports/soccer/odds", params={"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "h2h"}, timeout=10)
        odds_map = {}
        for ev in r.json():
            key = f"{ev['home_team']}_{ev['away_team']}"
            bookmakers = []
            for b in ev.get('bookmakers', [])[:3]:
                markets = {}
                for m in b.get('markets', []):
                    outcomes = {}
                    for o in m.get('outcomes', []):
                        outcomes[o['name']] = o.get('price', 0)
                    markets[m['key']] = outcomes
                bookmakers.append({"name": b['title'], "markets": markets})
            odds_map[key] = {"bookmakers": bookmakers}
        return odds_map
    except Exception: 
        return {}

def collect_all(date=None):
    """主调度函数：拉取一切"""
    target_date = date or get_today()
    print(f"\n🚀 启动全维度情报采集 | 目标日期: {target_date}")
    
    matches = scrape_wencai_jczq(target_date)
    
    for i, m in enumerate(matches):
        print(f"  [{i+1}/{len(matches)}] 深度装载: {m['home_team']}")
        
        ht_data = search_team_api(m["home_team"])
        at_data = search_team_api(m["away_team"])
        
        m.update({
            "home_id": ht_data["id"] if ht_data else 0, 
            "away_id": at_data["id"] if at_data else 0, 
            "home_logo": ht_data["logo"] if ht_data else "",
            "away_logo": at_data["logo"] if at_data else "",
            "id": i + 1, 
            "date": target_date
        })
        
        api_h = fetch_stats(m["home_id"])
        api_a = fetch_stats(m["away_id"])
        
        # 数据对齐，保证绝不断裂
        m["home_stats"] = api_h if api_h.get("played", 0) > 0 else generate_stats_from_rank(m["home_rank"], m["home_team"])
        m["away_stats"] = api_a if api_a.get("played", 0) > 0 else generate_stats_from_rank(m["away_rank"], m["away_team"])
        m["h2h"] = fetch_h2h(m["home_id"], m["away_id"])
        
        time.sleep(0.2)
        
    odds_data = fetch_odds_baseline()
    
    return {
        "date": target_date, 
        "matches": matches, 
        "odds": odds_data, 
        "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
