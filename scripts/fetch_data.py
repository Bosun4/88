import os
import re
import json
import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from config import *

# ============================================================
#  球队译名映射字典 (提升 API 命中率)
# ============================================================
TEAM_NAME_MAPPING = {
    "西汉姆联": "West Ham", "布伦特": "Brentford", "阿森纳": "Arsenal",
    "曼城": "Manchester City", "利物浦": "Liverpool", "曼联": "Manchester United",
    "切尔西": "Chelsea", "热刺": "Tottenham", "拉齐奥": "Lazio",
    "萨索洛": "Sassuolo", "皇马": "Real Madrid", "巴萨": "Barcelona",
    "马竞": "Atletico Madrid", "拜仁": "Bayern Munich", "亚特兰大": "Atalanta",
    "国际米兰": "Inter Milan", "AC米兰": "AC Milan", "尤文图斯": "Juventus",
    "那不勒斯": "Napoli", "多特蒙德": "Borussia Dortmund", "莱比锡": "RB Leipzig",
    "巴黎": "Paris Saint Germain", "里昂": "Lyon", "马赛": "Marseille",
    "本菲卡": "Benfica", "波尔图": "FC Porto", "阿贾克斯": "Ajax",
    "费耶诺德": "Feyenoord", "塞维利亚": "Sevilla", "比利亚雷亚尔": "Villarreal",
    "毕尔巴鄂": "Athletic Bilbao", "皇家社会": "Real Sociedad",
    "狼队": "Wolverhampton", "纽卡斯尔": "Newcastle", "维拉": "Aston Villa",
    "唐卡斯特": "Doncaster", "维尔港": "Port Vale", "埃门": "Emmen", "坎布尔": "Cambuur"
}

# 全局并发控制器，防止 API-Football 封 IP (普通版限频 10次/秒)
CONCURRENCY_LIMIT = 8
sema = asyncio.Semaphore(CONCURRENCY_LIMIT)

def translate_team_name(name):
    if not name: return ""
    name = str(name).strip()
    if name in TEAM_NAME_MAPPING: return TEAM_NAME_MAPPING[name]
    try:
        from deep_translator import GoogleTranslator
        clean = name.replace("女足", " Women").replace("联", " United")
        return GoogleTranslator(source='zh-CN', target='en').translate(clean).replace("FC", "").strip()
    except:
        return name

def _safe_dict(val): return val if isinstance(val, dict) else {}
def _get_float(val, default=999.0):
    try: return float(val) if val is not None else default
    except: return default

async def scrape_wencai_jczq_async(session: aiohttp.ClientSession, date_str: str) -> dict:
    """直接解析问财底层的 JSON，自动隔离足球与篮球"""
    url = f"https://edu.wencaivip.cn/api/v1.reference/matches?date={date_str}"
    print(f"  正在抓取底层盘口数据 [{date_str}]...")
    
    football_matches = []
    basketball_matches = []
    
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    try:
        async with session.get(url, headers=headers, timeout=15) as r:
            if r.status != 200:
                print(f"  ❌ 抓取失败 HTTP {r.status}")
                return {"football": [], "basketball": []}
                
            data = await r.json()
            match_dict = data.get("data", {}).get("matches", {})
            match_list = []
            
            # 将嵌套的 "1" (足球) 和 "2" (篮球) 列表合并展开
            for key in match_dict:
                if isinstance(match_dict[key], list): 
                    match_list.extend(match_dict[key])
                    
            for item in match_list:
                try:
                    t_type = str(item.get("types") or "")
                    m_num = str(item.get("week", "")) + str(item.get("week_no", ""))
                    lg_t = str(item.get("cup") or "未知联赛")
                    home_t = str(item.get("home") or "未知主队")
                    away_t = str(item.get("guest") or "未知客队")
                    
                    # 提取深层情报
                    info = _safe_dict(item.get("information"))
                    analyse = _safe_dict(item.get("analyse"))
                    baseface = str(analyse.get("baseface") or "").strip()
                    expert_intro = str(item.get("intro") or "").strip()
                    
                    h_inj = str(info.get("home_injury") or "无重大伤停").replace("\n", " ").strip()[:150]
                    g_inj = str(info.get("guest_injury") or "无重大伤停").replace("\n", " ").strip()[:150]
                    home_bad = str(info.get("home_bad_news", "")).replace("\n", " ").strip()[:150]
                    guest_bad = str(info.get("guest_bad_news", "")).replace("\n", " ").strip()[:150]
                    
                    pts = _safe_dict(item.get("points"))
                    def parse_rank(pos):
                        if not pos: return 0
                        m2 = re.findall(r'\d+', str(pos))
                        return int(m2[0]) if m2 else 0

                    if t_type == "1" or t_type == "足球":
                        # ======== 处理足球数据 ========
                        chg = _safe_dict(item.get("change"))
                        w_c, l_c = _get_float(chg.get("win"), 0), _get_float(chg.get("lose"), 0)
                        odds_mov = f"主胜{'升水' if w_c>0 else '降水' if w_c<0 else '平稳'}，客胜{'升水' if l_c>0 else '降水' if l_c<0 else '平稳'}"
                        
                        v2_odds = {}
                        for k in ["a0","a1","a2","a3","a4","a5","a6","a7","s00","s11","s22","s33",
                                   "w10","w20","w21","w30","w31","w32","w40","w41","w42",
                                   "l01","l02","l12","l03","l13","l23",
                                   "ss","sp","sf","ps","pp","pf","fs","fp","ff"]:
                            val = item.get(k)
                            if val is not None:
                                v2_odds[k] = _get_float(val, 0)
                                
                        football_matches.append({
                            "home_team": home_t, "away_team": away_t,
                            "league": lg_t, "match_num": m_num,
                            "sp_home": _get_float(item.get("win"), 0),
                            "sp_draw": _get_float(item.get("same"), 0),
                            "sp_away": _get_float(item.get("lose"), 0),
                            "give_ball": _get_float(item.get("give_ball"), 0),
                            "change": _safe_dict(item.get("change")),
                            "vote": _safe_dict(item.get("vote")),
                            "odds_movement": odds_mov,
                            "intelligence": {"h_inj": h_inj, "g_inj": g_inj, "home_bad_news": home_bad, "guest_bad_news": guest_bad},
                            "expert_intro": expert_intro,
                            "baseface": baseface,
                            "had_analyse": analyse.get("had_analyse", []),
                            "home_rank": parse_rank(pts.get("home_position", item.get("home_position", ""))),
                            "away_rank": parse_rank(pts.get("guest_position", item.get("guest_position", ""))),
                            "v2_odds_dict": v2_odds,
                        })
                        print(f"    [⚽足球] {lg_t} {m_num}: {home_t} vs {away_t}")

                    elif t_type == "2" or t_type == "篮球":
                        # ======== 处理篮球数据 (安全隔离，留作未来篮球量化引擎扩展) ========
                        basketball_matches.append({
                            "home_team": home_t, "away_team": away_t,
                            "league": lg_t, "match_num": m_num,
                            "sp_win": _get_float(item.get("win"), 0),
                            "sp_lose": _get_float(item.get("lose"), 0),
                            "hdc_rq": _get_float(item.get("hdc_rq"), 0),  # 让分
                            "toalsp": _get_float(item.get("toalsp"), 0),  # 大小分盘口
                            "intelligence": {"h_inj": h_inj, "g_inj": g_inj},
                            "baseface": baseface
                        })
                        print(f"    [🏀篮球] {lg_t} {m_num}: {home_t} vs {away_t} (已隔离)")
                        
                except Exception as e:
                    continue
                    
    except Exception as e:
        print(f"  API抓取失败: {e}")
        
    print(f"  共抓取 ⚽{len(football_matches)}场足球, 🏀{len(basketball_matches)}场篮球")
    return {"football": football_matches, "basketball": basketball_matches}

async def async_fetch_api(session: aiohttp.ClientSession, endpoint: str, params: dict) -> dict:
    """通用的异步 API-Football 请求器"""
    if not API_FOOTBALL_KEY: return []
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    url = f"{API_FOOTBALL_BASE}{endpoint}"
    
    async with sema:
        try:
            async with session.get(url, headers=headers, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", [])
        except Exception as e:
            print(f"Async API Error [{endpoint}]: {e}")
        return []

def generate_stats_from_context(match, side):
    """API未命中时，通过赔率反推预期进球和胜率的容灾方案"""
    rank = int(match.get("home_rank" if side == "home" else "away_rank", 10) or 10)
    sp_h = float(match.get("sp_home", 0) or 0)
    sp_a = float(match.get("sp_away", 0) or 0)
    rank = max(1, min(20, rank if rank > 0 else 10))
    strength = 1.0 - (rank - 1) / 19.0
    if sp_h > 1 and sp_a > 1:
        if side == "home": odds_strength = (1/sp_h) / (1/sp_h + 1/sp_a)
        else: odds_strength = (1/sp_a) / (1/sp_h + 1/sp_a)
        strength = strength * 0.3 + odds_strength * 0.7
        
    played = 25
    win_rate = max(0.15, min(0.70, strength * 0.55 + 0.15))
    draw_rate = 0.25
    wins = max(1, round(played * win_rate))
    draws = max(1, round(played * draw_rate))
    losses = max(1, played - wins - draws)
    gf_per = max(0.6, strength * 1.6 + 0.4)
    ga_per = max(0.5, (1 - strength) * 1.5 + 0.4)
    
    if strength > 0.6: form = "WWDWW"
    elif strength > 0.4: form = "WDLWD"
    elif strength > 0.25: form = "LDWDL"
    else: form = "LLDLL"
    
    return {
        "played": played, "wins": wins, "draws": draws, "losses": losses,
        "goals_for": round(played * gf_per), "goals_against": round(played * ga_per),
        "avg_goals_for": str(round(gf_per, 2)),
        "avg_goals_against": str(round(ga_per, 2)),
        "clean_sheets": max(1, round(played * (1 - ga_per / 2.5) * 0.25)), 
        "form": form, "rank": rank,
    }

async def enrich_match_data(session: aiohttp.ClientSession, m: dict, i: int, date_str: str):
    """异步并发处理单场足球比赛的所有外部 API 请求"""
    m["id"] = i + 1
    m["date"] = date_str
    
    # 1. 搜索主客队 ID
    ht_task = async_fetch_api(session, "/teams", {"search": translate_team_name(m["home_team"])})
    at_task = async_fetch_api(session, "/teams", {"search": translate_team_name(m["away_team"])})
    ht_res, at_res = await asyncio.gather(ht_task, at_task)
    
    ht = ht_res[0]["team"] if ht_res else None
    at = at_res[0]["team"] if at_res else None
    
    m["home_id"] = ht["id"] if ht else 0
    m["away_id"] = at["id"] if at else 0

    # 2. 获取统计数据和 H2H 交锋
    tasks = []
    if m["home_id"]: tasks.append(async_fetch_api(session, "/teams/statistics", {"team": m["home_id"], "season": 2024}))
    else: tasks.append(asyncio.sleep(0))

    if m["away_id"]: tasks.append(async_fetch_api(session, "/teams/statistics", {"team": m["away_id"], "season": 2024}))
    else: tasks.append(asyncio.sleep(0))
        
    if m["home_id"] and m["away_id"]: tasks.append(async_fetch_api(session, "/fixtures/headtohead", {"h2h": f"{m['home_id']}-{m['away_id']}", "last": 5}))
    else: tasks.append(asyncio.sleep(0))

    results = await asyncio.gather(*tasks)
    
    # 解析主队 Stats
    api_h_raw = results[0]
    if api_h_raw and isinstance(api_h_raw, dict) and "fixtures" in api_h_raw:
        m["home_stats"] = {
            "played": api_h_raw["fixtures"]["played"].get("total", 0),
            "wins": api_h_raw["fixtures"]["wins"].get("total", 0),
            "draws": api_h_raw["fixtures"]["draws"].get("total", 0),
            "losses": api_h_raw["fixtures"]["loses"].get("total", 0),
            "avg_goals_for": str(api_h_raw["goals"]["for"]["average"].get("total", "0.0")),
            "avg_goals_against": str(api_h_raw["goals"]["against"]["average"].get("total", "0.0")),
            "form": api_h_raw.get("form", ""),
            "clean_sheets": api_h_raw["clean_sheet"].get("total", 0),
        }
    else:
        m["home_stats"] = generate_stats_from_context(m, "home")

    # 解析客队 Stats
    api_a_raw = results[1]
    if api_a_raw and isinstance(api_a_raw, dict) and "fixtures" in api_a_raw:
        m["away_stats"] = {
            "played": api_a_raw["fixtures"]["played"].get("total", 0),
            "wins": api_a_raw["fixtures"]["wins"].get("total", 0),
            "draws": api_a_raw["fixtures"]["draws"].get("total", 0),
            "losses": api_a_raw["fixtures"]["loses"].get("total", 0),
            "avg_goals_for": str(api_a_raw["goals"]["for"]["average"].get("total", "0.0")),
            "avg_goals_against": str(api_a_raw["goals"]["against"]["average"].get("total", "0.0")),
            "form": api_a_raw.get("form", ""),
            "clean_sheets": api_a_raw["clean_sheet"].get("total", 0),
        }
    else:
        m["away_stats"] = generate_stats_from_context(m, "away")

    # 解析 H2H
    h2h_raw = results[2]
    m["h2h"] = []
    if h2h_raw and isinstance(h2h_raw, list):
        m["h2h"] = [{"date": x["fixture"]["date"][:10], "score": f"{x['goals']['home']}-{x['goals']['away']}", "home": x["teams"]["home"]["name"], "away": x["teams"]["away"]["name"]} for x in h2h_raw]
        
    return m

async def async_collect_all(date_str: str) -> dict:
    """主控入口：并发抓取并组装所有数据"""
    print(f"\n=== 数据抓取 (全异步隔离版) | {date_str} ===")
    
    async with aiohttp.ClientSession() as session:
        # 1. 抓取基础数据并分类
        raw_data = await scrape_wencai_jczq_async(session, date_str)
        football_matches = raw_data.get("football", [])
        
        if not football_matches:
            return {"date": date_str, "matches": [], "odds": {}}
            
        print(f"\n  API-Football 并发补充足球数据开始...")
        # 2. 仅对足球数据进行海外 API 并发清洗
        tasks = [enrich_match_data(session, m, i, date_str) for i, m in enumerate(football_matches)]
        enriched_matches = await asyncio.gather(*tasks)
        
    return {"date": date_str, "matches": enriched_matches, "odds": {}}
