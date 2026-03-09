"""
数据采集引擎 v5.5 (情报全量版):
1. [Team Mapping] 建立中英文队名映射矩阵，对接全球足球数据库
2. [Wencai API] 深度解析 edu.wencaivip.cn 高级接口，提取核心盘口、赔率、伤停、文字情报
3. [Text Mining] 完整保留 专家点评(intro)、基本面研报(baseface)、投票分布(vote) 等非结构化文本
4. [Odds Tracking] 解析 change 字典，追踪主/客胜赔率的实时升水(看衰)与降水(防范)动向
5. [API-Football] 实时同步全球联赛进球均值、近况走势、历史交锋(H2H)数据
6. [Fallback Logic] 基于队名哈希(Hash)种子生成联赛排名预测数据，确保数据链条永不断裂
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
# 用于将中文赛事名称转化为 API-Football 能识别的唯一英文标识，确保统计数据 100% 匹配
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
    """队名翻译中枢：优先匹配映射表，其次调用 Google 翻译引擎"""
    if name in TEAM_NAME_MAPPING: 
        return TEAM_NAME_MAPPING[name]
    try:
        # 清洗竞彩特有的简称干扰项
        clean_n = name.replace("女足", " Women").replace("联", " United").replace("台女", " Taipei Women")
        translated = GoogleTranslator(source='zh-CN', target='en').translate(clean_n)
        return translated.replace("FC", "").strip()
    except Exception: 
        return name

# ==================== 2. 问彩高级 JSON 解析引擎 ====================
def get_today(offset=0):
    """获取目标日期字符串，支持偏移量（用于回测或获取次日赛事）"""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo(TIMEZONE))
    except:
        now = datetime.now()
    return (now + timedelta(days=offset)).strftime("%Y-%m-%d")

def scrape_wencai_jczq(date=None):
    """
    全量抓取： edu.wencaivip.cn 核心数据。
    捕获每一个字节的文字情报与盘口变化。
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
            print("  ⚠️ Wencai API 未返回有效足球赛事列表。")
            return ms
            
        print(f"  ✅ 成功获取 {len(raw_matches)} 场赛事的原始数据包...")
        
        for item in raw_matches:
            try:
                # A. 基础识别信息
                league = item.get("cup", "未知联赛")
                m_num = f"{item.get('week', '')}{item.get('week_no', '')}"
                home = item.get("home", "")
                away = item.get("guest", "")
                
                # B. 资金水位异动解析 (change 字典映射: 1=升水/利空, -1=降水/利好)
                chg = item.get("change", {})
                w_c = chg.get("win", 0)
                l_c = chg.get("lose", 0)
                w_txt = "升水(看衰)" if w_c > 0 else "降水(防范)" if w_c < 0 else "平稳"
                l_txt = "升水(看衰)" if l_c > 0 else "降水(防范)" if l_c < 0 else "平稳"
                odds_movement = f"主胜{w_txt}，客胜{l_txt}"

                # C. 让球盘口封装
                give_ball = item.get("give_ball", 0)
                handicap_str = f"让{give_ball} ({item.get('hhad_win', '-')}/{item.get('hhad_same', '-')}/{item.get('hhad_lose', '-')})"

                # D. 深度文本挖掘 (绝不偷工减料的关键点)
                # 获取专家对此场比赛的直接点评
                expert_intro = item.get("intro", "无专家点评")
                # 获取深度基本面研报
                ana_dict = item.get("analyse", {})
                base_face = ana_dict.get("baseface", "暂无深度基本面数据") if isinstance(ana_dict, dict) else ""
                
                # E. 伤停情报与多维度利空 (存入情报池)
                info_dict = item.get("information", {})
                intel_pool = {
                    "h_inj": info_dict.get("home_injury", "无伤停汇报").replace("\n", " | "),
                    "g_inj": info_dict.get("guest_injury", "无伤停汇报").replace("\n", " | "),
                    "h_bad": info_dict.get("home_bad_news", "无").replace("\n", " "),
                    "g_bad": info_dict.get("guest_bad_news", "无").replace("\n", " "),
                    "match_points": item.get("points", {}).get("match_points", "")
                }
                
                # F. 排名正则化 (处理 "英超18" 这种混合字符串)
                def clean_rank(val):
                    if not val: return 0
                    nums = re.findall(r'\d+', str(val))
                    return int(nums[0]) if nums else 0

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
                    "home_rank": clean_rank(item.get("points", {}).get("home_position")),
                    "away_rank": clean_rank(item.get("points", {}).get("guest_position")),
                    "source": "wencai_api"
                })
            except Exception as sub_e:
                print(f"    - 解析单场异常: {home} ({str(sub_e)})")
                continue
                
    except Exception as e: 
        print(f"  ❌ 数据抓取引擎致命错误: {str(e)}")
        
    return ms

# ==================== 3. 全球数据库实时同步 (API-Football) ====================
def search_team_api(name):
    """利用翻译后的队名定位其在全球数据库中的唯一 ID"""
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
    """获取球队详细统计指标：进球均值、零封率、近况形态"""
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
    """抓取两队历史 5 场交锋记录"""
    if not hid or not aid: return []
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/fixtures/headtohead", headers=h, params={"h2h": f"{hid}-{aid}", "last": 5}, timeout=10)
        return [{"date": m["fixture"]["date"][:10], "home": m["teams"]["home"]["name"], "away": m["teams"]["away"]["name"], "score": f"{m['goals']['home']}-{m['goals']['away']}", "league": m["league"]["name"]} for m in r.json().get("response", [])]
    except Exception: return []

# ==================== 4. 统计学稳健兜底逻辑 ====================
def generate_stats_from_rank(rank, team_name="", total_teams=20):
    """
    当 API 无法找到数据时，根据联赛排名生成模拟统计。
    使用队名哈希作为随机种子，确保同一支队在不同时段生成的数据是唯一且恒定的。
    """
    import random
    name_hash = sum(ord(c) for c in team_name) if team_name else 0
    random.seed((rank * 7) + name_hash) 
    
    rank = rank if rank > 0 else random.randint(8, 14)
    # 实力系数：排名越靠前，实力系数越高
    strength = 1 - (rank - 1) / total_teams
    played = random.randint(22, 30)
    win_r, draw_r = max(0.15, min(0.75, strength * 0.65)), random.uniform(0.18, 0.28)
    wins, draws = int(played * win_r), int(played * draw_r)
    gf_p, ga_p = max(0.7, strength * 2.2), max(0.6, (1 - strength) * 1.9)
    
    return {
        "played": played, "wins": wins, "draws": draws, "losses": max(0, played - wins - draws),
        "goals_for": int(played * gf_p), "goals_against": int(played * ga_p),
        "avg_goals_for": str(round(gf_p, 2)), "avg_goals_against": str(round(ga_p, 2)),
        "clean_sheets": int(played * 0.25), 
        "form": "".join(random.choices("WDL", weights=[win_r, draw_r, 1-win_r-draw_r], k=5)), 
        "rank": rank, "is_generated": True
    }

# ==================== 5. 跨平台赔率锚点 (Odds-API) ====================
def fetch_odds_baseline():
    """从全球 Odds-API 获取实时赔率作为 EV 计算的对标基准"""
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
    满血主函数：
    1. 抓取 wencai 情报。
    2. 深度同步进球统计与历史交锋。
    3. 整合 Odds-API 跨平台赔率。
    """
    target_date = date or get_today()
    print(f"\n🚀 启动全维度情报采集引擎 | 目标日期: {target_date}")
    
    # 步骤 1：捕获带有“专家点评”和“伤停”的比赛主列表
    matches = scrape_wencai_jczq(target_date)
    
    # 步骤 2：对每一场比赛进行“切片式”深度透视
    for i, m in enumerate(matches):
        print(f"  [{i+1}/{len(matches)}] 深度装载: {m['home_team']} vs {m['away_team']}")
        
        # 同步数据库 ID
        ht_data = search_team_api(m["home_team"])
        at_data = search_team_api(m["away_team"])
        
        m.update({
            "home_id": ht_data["id"] if ht_data else 0, 
            "away_id": at_data["id"] if at_data else 0,
            "home_logo": ht_data["logo"] if ht_data else "", 
            "away_logo": at_data["logo"] if at_data else "",
            "id": i + 1, "date": target_date
        })

        # 加载历史战绩统计 (核心计算泊松分布所需)
        api_h = fetch_stats(m["home_id"]) if m["home_id"] else {}
        api_a = fetch_stats(m["away_id"]) if m["away_id"] else {}
        
        # 数据对齐：实测数据优先，生成数据兜底
        m["home_stats"] = api_h if api_h.get("played", 0) > 0 else generate_stats_from_rank(m["home_rank"], m["home_team"])
        m["away_stats"] = api_a if api_a.get("played", 0) > 0 else generate_stats_from_rank(m["away_rank"], m["away_team"])
        
        # 获取两队历史交手真相
        m["h2h"] = fetch_h2h(m["home_id"], m["away_id"])
        
        # 严格遵守 API 访问速率限制
        time.sleep(0.35)

    print("  🔗 正在拉取全球 Odds-API 赔率交叉基准...")
    return {
        "date": target_date, 
        "matches": matches, 
        "odds": fetch_odds_baseline(), 
        "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
