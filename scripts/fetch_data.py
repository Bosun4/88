import os
import re
import json
import asyncio
import requests
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from config import *

# ==================== 自动修复 aiohttp（关键修复） ====================
try:
    import aiohttp
except ModuleNotFoundError:
    print("🔧 检测到 aiohttp 缺失，正在自动安装...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])
    import aiohttp
    print("✅ aiohttp 安装成功")
# =====================================================================

# 极限球队映射字典
TEAM_NAME_MAPPING = {
    "西汉姆联": "West Ham", "布伦特": "Brentford", "阿森纳": "Arsenal",
    "曼城": "Manchester City", "利物浦": "Liverpool", "曼联": "Manchester United",
    "切尔西": "Chelsea", "热刺": "Tottenham", "拉齐奥": "Lazio",
    "萨索洛": "Sassuolo", "皇马": "Real Madrid", "巴萨": "Barcelona",
    "马竞": "Atletico Madrid", "拜仁": "Bayern Munich", "亚特兰大": "Atalanta",
    "国际米兰": "Inter Milan", "AC米兰": "AC Milan", "尤文图斯": "Juventus",
    "唐卡斯特": "Doncaster", "维尔港": "Port Vale", "埃门": "Emmen", "坎布尔": "Cambuur"
}

sema = asyncio.Semaphore(8)

def translate_team_name(name):
    if not name: return ""
    name = str(name).strip()
    if name in TEAM_NAME_MAPPING: return TEAM_NAME_MAPPING[name]
    try:
        from deep_translator import GoogleTranslator
        clean = name.replace("女足", " Women").replace("联", " United")
        return GoogleTranslator(source='zh-CN', target='en').translate(clean).replace("FC", "").strip()
    except: return name

def _safe_dict(val): return val if isinstance(val, dict) else {}
def _get_float(val, default=0.0):
    try: return float(val) if val is not None else default
    except: return default

async def scrape_wencai_jczq_async(session, date_str):
    """深度解析你提供的抓包 JSON，精准隔离篮球数据"""
    url = f"https://edu.wencaivip.cn/api/v1.reference/matches?date={date_str}"
    football_matches = []
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    
    try:
        async with session.get(url, headers=headers, timeout=15) as r:
            data = await r.json()
            matches_raw = data.get("data", {}).get("matches", {})
            # 仅提取 Key 为 "1" 的足球列表
            list_1 = matches_raw.get("1", [])
            for item in list_1:
                try:
                    m_num = str(item.get("week", "")) + str(item.get("week_no", ""))
                    info = _safe_dict(item.get("information"))
                    analyse = _safe_dict(item.get("analyse"))
                    pts = _safe_dict(item.get("points"))
                    
                    v2_odds = {}
                    for k in ["a0","a1","a2","a3","a4","a5","a6","a7","s00","s11","s22","s33",
                               "w10","w20","w21","w30","w31","w32","w40","w41","w42",
                               "l01","l02","l12","l03","l13","l23",
                               "ss","sp","sf","ps","pp","pf","fs","fp","ff"]:
                        val = item.get(k)
                        if val is not None: v2_odds[k] = _get_float(val)

                    football_matches.append({
                        "home_team": str(item.get("home", "未知")), 
                        "away_team": str(item.get("guest", "未知")),
                        "league": str(item.get("cup", "未知")), 
                        "match_num": m_num,
                        "sp_home": _get_float(item.get("win")),
                        "sp_draw": _get_float(item.get("same")),
                        "sp_away": _get_float(item.get("lose")),
                        "give_ball": _get_float(item.get("give_ball")),
                        "change": _safe_dict(item.get("change")),
                        "vote": _safe_dict(item.get("vote")),
                        "intelligence": {
                            "h_inj": str(info.get("home_injury", "无")), 
                            "g_inj": str(info.get("guest_injury", "无")),
                            "home_bad_news": str(info.get("home_bad_news", "")),
                            "guest_bad_news": str(info.get("guest_bad_news", ""))
                        },
                        "expert_intro": str(item.get("intro", "")),
                        "baseface": str(analyse.get("baseface", "")),
                        "had_analyse": analyse.get("had_analyse", []),
                        "home_rank": int(re.findall(r'\d+', str(pts.get("home_position", "10")))[0]),
                        "away_rank": int(re.findall(r'\d+', str(pts.get("guest_position", "10")))[0]),
                        "v2_odds_dict": v2_odds,
                        "home_id_original": item.get("home_id", 0),
                        "away_id_original": item.get("guest_id", 0)
                    })
                except: continue
    except Exception as e: print(f"  ❌ 抓取失败: {e}")
    return football_matches

async def async_fetch_api(session, endpoint, params):
    if not API_FOOTBALL_KEY: return []
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    async with sema:
        try:
            async with session.get(f"{API_FOOTBALL_BASE}{endpoint}", headers=headers, params=params, timeout=10) as r:
                if r.status == 200:
                    d = await r.json()
                    return d.get("response", [])
        except: return []
    return []

async def enrich_match_data(session, m, i, date_str):
    m["id"] = i + 1
    m["date"] = date_str
    
    # 并发搜 ID
    h_task = async_fetch_api(session, "/teams", {"search": translate_team_name(m["home_team"])})
    a_task = async_fetch_api(session, "/teams", {"search": translate_team_name(m["away_team"])})
    h_res, a_res = await asyncio.gather(h_task, a_task)
    
    m["home_id"] = h_res[0]["team"]["id"] if h_res else 0
    m["away_id"] = a_res[0]["team"]["id"] if a_res else 0

    # 并发搜 Stats & H2H
    tasks = []
    if m["home_id"]: tasks.append(async_fetch_api(session, "/teams/statistics", {"team": m["home_id"], "season": 2024}))
    else: tasks.append(asyncio.sleep(0))
    if m["away_id"]: tasks.append(async_fetch_api(session, "/teams/statistics", {"team": m["away_id"], "season": 2024}))
    else: tasks.append(asyncio.sleep(0))
    if m["home_id"] and m["away_id"]: tasks.append(async_fetch_api(session, "/fixtures/headtohead", {"h2h": f"{m['home_id']}-{m['away_id']}", "last": 5}))
    else: tasks.append(asyncio.sleep(0))

    results = await asyncio.gather(*tasks)
    
    # 解析主队
    if results[0] and isinstance(results[0], dict) and "fixtures" in results[0]:
        r = results[0]
        m["home_stats"] = {"played": r["fixtures"]["played"]["total"], "wins": r["fixtures"]["wins"]["total"], "avg_goals_for": str(r["goals"]["for"]["average"]["total"]), "form": r.get("form", ""), "clean_sheets": r["clean_sheet"]["total"]}
    else: m["home_stats"] = {"played": 25, "wins": 10, "avg_goals_for": "1.3", "form": "WDLWD", "clean_sheets": 5}
    
    # 解析客队
    if results[1] and isinstance(results[1], dict) and "fixtures" in results[1]:
        r = results[1]
        m["away_stats"] = {"played": r["fixtures"]["played"]["total"], "wins": r["fixtures"]["wins"]["total"], "avg_goals_for": str(r["goals"]["for"]["average"]["total"]), "form": r.get("form", ""), "clean_sheets": r["clean_sheet"]["total"]}
    else: m["away_stats"] = {"played": 25, "wins": 7, "avg_goals_for": "1.1", "form": "LDWDL", "clean_sheets": 3}

    m["h2h"] = [{"score": f"{x['goals']['home']}-{x['goals']['away']}"} for x in results[2] if isinstance(results[2], list)] if results[2] else []
    return m

async def async_collect_all(date_str):
    async with aiohttp.ClientSession() as session:
        matches = await scrape_wencai_jczq_async(session, date_str)
        if not matches: return {"date": date_str, "matches": []}
        tasks = [enrich_match_data(session, m, i, date_str) for i, m in enumerate(matches)]
        enriched = await asyncio.gather(*tasks)
    return {"date": date_str, "matches": enriched}