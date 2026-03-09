 """
数据采集引擎 v5.5 (情报全维度满血版):
1. [Wencai API] 深度解析 edu.wencaivip.cn 高级接口，提取盘口、赔率、伤停、文字情报。
2. [Text Mining] 完整保留 专家点评(intro)、基本面研报(baseface)、投票分布(vote)。
3. [Odds Tracking] 解析 change 字典，追踪主/客胜赔率的实时升水(看衰)与降水(防范)动向。
4. [Safety Logic] 针对 points 和 information 字段加入 isinstance 校验，彻底根治 'list' object 崩溃。
5. [Fallback Engine] 采用队名哈希(Hash)种子生成恒定的模拟统计数据，确保 API 缺失时逻辑不断链。
"""
import requests
import json
import time
import re
import random
from datetime import datetime, timedelta
from deep_translator import GoogleTranslator
from config import *

# ==================== 1. 深度队名映射矩阵 ====================
# 用于将中文队名精准转化为 API-Football 能识别的唯一英文标识，确保统计数据 100% 匹配
TEAM_NAME_MAPPING = {
    "西汉姆联": "West Ham", "布伦特": "Brentford", "阿森纳": "Arsenal",
    "曼城": "Manchester City", "利物浦": "Liverpool", "曼联": "Manchester United",
    "切尔西": "Chelsea", "热刺": "Tottenham", "拉齐奥": "Lazio",
    "萨索洛": "Sassuolo", "尤文图斯": "Juventus", "罗马": "Roma",
    "AC米兰": "AC Milan", "国际米兰": "Inter", "那不勒斯": "Napoli",
    "西班牙人": "Espanyol", "奥维耶多": "Real Oviedo", "皇马": "Real Madrid",
    "巴萨": "Barcelona", "马竞": "Atletico Madrid", "朝鲜女": "North Korea W",
    "中国女": "China PR W", "敦刻尔克": "Dunkerque", "兰斯": "Reims",
    "通德拉": "Tondela", "里奥阿维": "Rio Ave", "拜仁": "Bayern Munich", 
    "亚特兰大": "Atalanta", "日本女": "Japan W", "越南女": "Vietnam W",
    "印度女": "India W", "中国台女": "Chinese Taipei W"
}

def translate_team_name(name):
    """队名翻译中枢：优先匹配硬编码映射，其次调用翻译引擎"""
    if name in TEAM_NAME_MAPPING: 
        return TEAM_NAME_MAPPING[name]
    try:
        # 清洗干扰项，提高翻译匹配率
        clean_n = name.replace("女足", " Women").replace("联", " United").replace("台女", " Taipei Women")
        translated = GoogleTranslator(source='zh-CN', target='en').translate(clean_n)
        return translated.replace("FC", "").strip()
    except Exception: 
        return name

# ==================== 2. 问彩高级 JSON 解析引擎 ====================
def get_today(offset=0):
    """获取目标日期字符串，支持偏移量"""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo(TIMEZONE))
    except:
        now = datetime.now()
    return (now + timedelta(days=offset)).strftime("%Y-%m-%d")

def scrape_wencai_jczq(date=None):
    """
    满血版抓取：直连问彩核心接口，捕获所有文字情报与盘口异动。
    """
    url = "https://edu.wencaivip.cn/api/v1.reference/matches"
    print(f"  🌐 正在建立全维度情报链路: {url}")
    ms = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*"
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        # 足球赛事存储在 "data" -> "matches" -> "1"
        raw_matches = data.get("data", {}).get("matches", {}).get("1", [])
        
        if not raw_matches:
            print("  ⚠️ Wencai API 未返回有效足球赛事。")
            return ms
            
        print(f"  ✅ 成功获取 {len(raw_matches)} 场赛事的原始数据包...")
        
        for item in raw_matches:
            try:
                # 1. 基础识别信息
                league = item.get("cup", "未知联赛")
                m_num = f"{item.get('week', '')}{item.get('week_no', '')}"
                home = item.get("home", "")
                away = item.get("guest", "")
                
                # 2. 赔率异动深度解析 (change 映射: 1=升水/利空, -1=降水/防范)
                chg = item.get("change", {})
                w_c = chg.get("win", 0)
                l_c = chg.get("lose", 0)
                w_txt = "升水(机构看衰)" if w_c > 0 else "降水(防范)" if w_c < 0 else "平稳"
                l_txt = "升水(机构看衰)" if l_c > 0 else "降水(防范)" if l_c < 0 else "平稳"
                odds_movement = f"主胜{w_txt}，客胜{l_txt}"

                # 3. 让球盘口封装
                give_ball = item.get("give_ball", 0)
                handicap_str = f"让{give_ball} ({item.get('hhad_win', '-')}/{item.get('hhad_same', '-')}/{item.get('hhad_lose', '-')})"

                # 4. 文本挖掘 (专家点评 + 深度研报)
                expert_intro = item.get("intro", "无专家点评")
                ana_dict = item.get("analyse", {})
                base_face = ana_dict.get("baseface", "暂无研报") if isinstance(ana_dict, dict) else "暂无研报"
                
                # 5. 🔥 伤停情报 (安全防御版：防止单场 item 格式异常)
                info_dict = item.get("information", {})
                if not isinstance(info_dict, dict): 
                    info_dict = {} # 强制转字典，防止 .get 报错
                
                intel_pool = {
                    "h_inj": info_dict.get("home_injury", "无汇报").replace("\n", " | "),
                    "g_inj": info_dict.get("guest_injury", "无汇报").replace("\n", " | "),
                    "h_bad": info_dict.get("home_bad_news", "无").replace("\n", " "),
                    "g_bad": info_dict.get("guest_bad_news", "无").replace("\n", " "),
                    "match_points": ""
                }
                
                # 6. 排名解析 (安全防御版)
                points_dict = item.get("points", {})
                home_rank = 0
                away_rank = 0
                if isinstance(points_dict, dict):
                    intel_pool["match_points"] = points_dict.get("match_points", "")
                    h_pos = points_dict.get("home_position", "")
                    a_pos = points_dict.get("guest_position", "")
                    # 正则提取数字
                    h_nums = re.findall(r'\d+', str(h_pos))
                    a_nums = re.findall(r'\d+', str(a_pos))
                    home_rank = int(h_nums[0]) if h_nums else 0
                    away_rank = int(a_nums[0]) if a_nums else 0

                ms.append({
                    "home_team": home, "away_team": away, "league": league, "match_num": m_num,
                    "sp_home": float(item.get("win") or 0), 
                    "sp_draw": float(item.get("same") or 0), 
                    "sp_away": float(item.get("lose") or 0),
                    "odds_movement": odds_movement,
                    "handicap_info": handicap_str,
                    "intelligence": intel_pool,
                    "expert_intro": expert_intro,
                    "base_face": base_face,
                    "votes": item.get("vote", {}),
                    "home_rank": home_rank,
                    "away_rank": away_rank,
                    "source": "wencai_api"
                })
            except Exception as e:
                print(f"    - 解析单场异常: {home} ({str(e)})")
                continue
                
    except Exception as e: 
        print(f"  ❌ 抓取引擎致命错误: {str(e)}")
        
    return ms

# ==================== 3. 统计数据库同步 (API-Football) ====================
def search_team_api(name):
    """搜索全球数据库 ID"""
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
    """抓取进球均值、近况等核心算力数据"""
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
    except Exception: pass
    return {}

def fetch_h2h(hid, aid):
    """抓取历史交锋数据"""
    if not hid or not aid: return []
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/fixtures/headtohead", headers=h, params={"h2h": f"{hid}-{aid}", "last": 5}, timeout=10)
        return [{"date": m["fixture"]["date"][:10], "home": m["teams"]["home"]["name"], "away": m["teams"]["away"]["name"], "score": f"{m['goals']['home']}-{m['goals']['away']}", "league": m["league"]["name"]} for m in r.json().get("response", [])]
    except Exception: return []

# ==================== 4. 稳健型统计兜底逻辑 ====================
def generate_stats_from_rank(rank, team_name="", total_teams=20):
    """
    当 API 缺失数据时，基于排名和队名哈希生成合理的统计数据。
    """
    name_hash = sum(ord(c) for c in team_name) if team_name else 0
    random.seed((rank * 7) + name_hash) 
    
    rank = rank if rank > 0 else random.randint(8, 14)
    strength = 1 - (rank - 1) / total_teams
    played = random.randint(22, 28)
    win_r, draw_r = max(0.15, min(0.75, strength * 0.65)), random.uniform(0.18, 0.28)
    wins, draws = int(played * win_r), int(played * draw_r)
    gf_p, ga_p = max(0.7, strength * 2.1), max(0.6, (1 - strength) * 1.8)
    
    return {
        "played": played, "wins": wins, "draws": draws, "losses": max(0, played - wins - draws),
        "goals_for": int(played * gf_p), "goals_against": int(played * ga_p),
        "avg_goals_for": str(round(gf_p, 2)), "avg_goals_against": str(round(ga_p, 2)),
        "clean_sheets": int(played * 0.25), 
        "form": "".join(random.choices("WDL", weights=[win_r, draw_r, 1-win_r-draw_r], k=5)), 
        "rank": rank, "is_generated": True
    }

# ==================== 5. 外部赔率基准 (Odds-API) ====================
def fetch_odds_baseline():
    """获取全球市场赔率作为 EV 计算锚点"""
    try:
        r = requests.get(f"{ODDS_API_BASE}/sports/soccer/odds", params={"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "h2h"}, timeout=10)
        if r.status_code != 200: return {}
        om = {}
        for ev in r.json():
            k = f"{ev['home_team']}_{ev['away_team']}"
            bks = [{"name": b['title'], "markets": {m['key']: {o['name']: o.get('price', 0) for o in m.get('outcomes', [])} for m in b.get('markets', [])}} for b in ev.get('bookmakers', [])[:3]]
            om[k] = {"bookmakers": bks}
        return om
    except Exception: return {}

# ==================== 6. 总调度中心 ====================
def collect_all(date=None):
    """
    满血主函数：建立深度情报池 -> 同步实时统计 -> 拉取外部赔率
    """
    target_date = date or get_today()
    print(f"\n🚀 启动全维度情报采集引擎 | 目标日期: {target_date}")
    
    # 步骤 1：抓取基础列表与深度文本
    matches = scrape_wencai_jczq(target_date)
    
    # 步骤 2：对每一场比赛进行深度装载
    for i, m in enumerate(matches):
        print(f"  [{i+1}/{len(matches)}] 深度装载: {m['home_team']} vs {m['away_team']}")
        
        # 定位 ID
        ht_data = search_team_api(m["home_team"])
        at_data = search_team_api(m["away_team"])
        
        m.update({
            "home_id": ht_data["id"] if ht_data else 0, 
            "away_id": at_data["id"] if at_data else 0,
            "home_logo": ht_data["logo"] if ht_data else "", 
            "away_logo": at_data["logo"] if at_data else "",
            "id": i + 1, "date": target_date
        })

        # 加载核心计算数据
        api_h = fetch_stats(m["home_id"]) if m["home_id"] else {}
        api_a = fetch_stats(m["away_id"]) if m["away_id"] else {}
        
        # 填充：实时数据优先，模拟数据兜底
        m["home_stats"] = api_h if api_h.get("played", 0) > 0 else generate_stats_from_rank(m["home_rank"], m["home_team"])
        m["away_stats"] = api_a if api_a.get("played", 0) > 0 else generate_stats_from_rank(m["away_rank"], m["away_team"])
        
        # 历史交锋
        m["h2h"] = fetch_h2h(m["home_id"], m["away_id"])
        
        # 频率控制
        time.sleep(0.35)

    print("  🔗 正在拉取全球 Odds-API 赔率基准...")
    return {
        "date": target_date, 
        "matches": matches, 
        "odds": fetch_odds_baseline(), 
        "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
