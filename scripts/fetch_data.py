import os
import re
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta, timezone
from config import *

TEAM_NAME_MAPPING = {
    "西汉姆联":"West Ham","布伦特":"Brentford","阿森纳":"Arsenal",
    "曼城":"Manchester City","利物浦":"Liverpool","曼联":"Manchester United",
    "切尔西":"Chelsea","热刺":"Tottenham","拉齐奥":"Lazio",
    "萨索洛":"Sassuolo","皇马":"Real Madrid","巴萨":"Barcelona",
    "马竞":"Atletico Madrid","拜仁":"Bayern Munich","亚特兰大":"Atalanta",
    "国际米兰":"Inter Milan","AC米兰":"AC Milan","尤文图斯":"Juventus",
    "那不勒斯":"Napoli","多特蒙德":"Borussia Dortmund","莱比锡":"RB Leipzig",
    "巴黎":"Paris Saint Germain","里昂":"Lyon","马赛":"Marseille",
    "本菲卡":"Benfica","波尔图":"FC Porto","阿贾克斯":"Ajax",
    "费耶诺德":"Feyenoord","塞维利亚":"Sevilla","比利亚雷亚尔":"Villarreal",
    "毕尔巴鄂":"Athletic Bilbao","皇家社会":"Real Sociedad",
    "狼队":"Wolverhampton","纽卡斯尔":"Newcastle","维拉":"Aston Villa",
    "唐卡斯特":"Doncaster","维尔港":"Port Vale","埃门":"Emmen","坎布尔":"Cambuur",
    "莱切斯特":"Leicester","伯恩利":"Burnley","富勒姆":"Fulham",
    "水晶宫":"Crystal Palace","伯恩茅斯":"Bournemouth","布莱顿":"Brighton",
    "诺丁汉森林":"Nottingham Forest","埃弗顿":"Everton","伊普斯":"Ipswich",
    "南安普顿":"Southampton","莱斯特城":"Leicester City",
}

def translate_team_name(name):
    if not name: return ""
    name = str(name).strip()
    if name in TEAM_NAME_MAPPING: return TEAM_NAME_MAPPING[name]
    try:
        from deep_translator import GoogleTranslator
        clean = name.replace("女足"," Women").replace("联"," United")
        return GoogleTranslator(source='zh-CN',target='en').translate(clean).replace("FC","").strip()
    except: return name

def _safe_dict(val): return val if isinstance(val, dict) else {}
def _get_float(val, default=0.0):
    try: return float(val) if val is not None else default
    except: return default

def generate_stats_from_context(match, side):
    """API未命中时，通过赔率+排名反推统计数据（容灾方案）"""
    rank = int(match.get("home_rank" if side=="home" else "away_rank", 10) or 10)
    sp_h = float(match.get("sp_home",0) or 0)
    sp_a = float(match.get("sp_away",0) or 0)
    rank = max(1, min(20, rank if rank > 0 else 10))
    strength = 1.0 - (rank - 1) / 19.0

    # 赔率反推强度（权重70%，比纯排名更准）
    if sp_h > 1 and sp_a > 1:
        if side == "home": odds_strength = (1/sp_h) / (1/sp_h + 1/sp_a)
        else: odds_strength = (1/sp_a) / (1/sp_h + 1/sp_a)
        strength = strength * 0.3 + odds_strength * 0.7

    played = 25
    win_rate = max(0.15, min(0.70, strength * 0.55 + 0.15))
    wins = max(1, round(played * win_rate))
    draws = max(1, round(played * 0.25))
    losses = max(1, played - wins - draws)
    gf = max(0.6, strength * 1.6 + 0.4)
    ga = max(0.5, (1-strength) * 1.5 + 0.4)

    if strength > 0.6: form = "WWDWW"
    elif strength > 0.4: form = "WDLWD"
    elif strength > 0.25: form = "LDWDL"
    else: form = "LLDLL"

    return {
        "played":played,"wins":wins,"draws":draws,"losses":losses,
        "goals_for":round(played*gf),"goals_against":round(played*ga),
        "avg_goals_for":str(round(gf,2)),"avg_goals_against":str(round(ga,2)),
        "clean_sheets":max(1,round(played*(1-ga/2.5)*0.25)),"form":form,
    }

async def scrape_wencai_jczq_async(session, date_str):
    """抓取问财数据，自动隔离足球与篮球（防止篮球数据污染泊松模型）"""
    url = f"https://edu.wencaivip.cn/api/v1.reference/matches?date={date_str}"
    football_matches = []

    headers = {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept":"application/json, text/plain, */*",
        "Accept-Language":"zh-CN,zh;q=0.9,en;q=0.8",
        "Origin":"https://edu.wencaivip.cn",
        "Referer":"https://edu.wencaivip.cn/"
    }

    try:
        async with session.get(url, headers=headers, timeout=15) as r:
            if r.status != 200:
                print(f"  ❌ 抓取失败 HTTP {r.status}")
                return []

            data = await r.json()
            if "data" not in data or not data["data"]:
                print(f"  ⚠️ 接口未返回数据: {str(data)[:100]}")
                return []

            matches_raw = data.get("data",{}).get("matches",{})

            # ===== 核心修复: 只取 "1" (足球)，跳过 "2" (篮球) =====
            # 篮球数据(124:101)如果流进泊松模型会直接溢出崩溃
            football_list = matches_raw.get("1", [])
            basketball_count = len(matches_raw.get("2", []))

            if basketball_count > 0:
                print(f"  🏀 已隔离 {basketball_count} 场篮球赛事（防止污染泊松模型）")

            if not football_list:
                print(f"  [INFO] 当日足球赛事列表为空")
                return []

            for item in football_list:
                try:
                    m_num = str(item.get("week","")) + str(item.get("week_no",""))
                    info = _safe_dict(item.get("information"))
                    analyse = _safe_dict(item.get("analyse"))
                    pts = _safe_dict(item.get("points"))
                    chg = _safe_dict(item.get("change"))

                    # 赔率变动描述
                    w_c = _get_float(chg.get("win"),0)
                    l_c = _get_float(chg.get("lose"),0)
                    odds_mov = f"主胜{'升水' if w_c>0 else '降水' if w_c<0 else '平稳'}，客胜{'升水' if l_c>0 else '降水' if l_c<0 else '平稳'}"

                    # 深度提取情报
                    h_inj = str(info.get("home_injury","无")).replace("\n"," ").strip()[:150]
                    g_inj = str(info.get("guest_injury","无")).replace("\n"," ").strip()[:150]
                    home_bad = str(info.get("home_bad_news","")).replace("\n"," ").strip()[:150]
                    guest_bad = str(info.get("guest_bad_news","")).replace("\n"," ").strip()[:150]

                    def parse_rank(pos):
                        if not pos: return 0
                        nums = re.findall(r'\d+', str(pos))
                        return int(nums[0]) if nums else 0

                    v2_odds = {}
                    for k in ["a0","a1","a2","a3","a4","a5","a6","a7",
                              "s00","s11","s22","s33",
                              "w10","w20","w21","w30","w31","w32","w40","w41","w42",
                              "l01","l02","l12","l03","l13","l23",
                              "ss","sp","sf","ps","pp","pf","fs","fp","ff"]:
                        val = item.get(k)
                        if val is not None: v2_odds[k] = _get_float(val)

                    football_matches.append({
                        "home_team": str(item.get("home","未知")),
                        "away_team": str(item.get("guest","未知")),
                        "league": str(item.get("cup","未知")),
                        "match_num": m_num,
                        "sp_home": _get_float(item.get("win")),
                        "sp_draw": _get_float(item.get("same")),
                        "sp_away": _get_float(item.get("lose")),
                        "give_ball": _get_float(item.get("give_ball")),
                        "change": chg,
                        "vote": _safe_dict(item.get("vote")),
                        "odds_movement": odds_mov,
                        "intelligence": {"h_inj":h_inj,"g_inj":g_inj,"home_bad_news":home_bad,"guest_bad_news":guest_bad},
                        "expert_intro": str(item.get("intro","")).strip(),
                        "baseface": str(analyse.get("baseface","")).strip(),
                        "had_analyse": analyse.get("had_analyse",[]),
                        "home_rank": parse_rank(pts.get("home_position",item.get("home_position",""))),
                        "away_rank": parse_rank(pts.get("guest_position",item.get("guest_position",""))),
                        "v2_odds_dict": v2_odds,
                    })
                except: continue

    except Exception as e:
        print(f"  ❌ 网络抓取异常: {e}")

    print(f"  ⚽ 足球赛事: {len(football_matches)} 场")
    return football_matches

async def async_fetch_api(session, endpoint, params, sema):
    if not API_FOOTBALL_KEY: return []
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    async with sema:
        try:
            async with session.get(f"{API_FOOTBALL_BASE}{endpoint}", headers=headers, params=params, timeout=10) as r:
                if r.status == 200:
                    d = await r.json()
                    return d.get("response",[])
        except: return []
    return []

async def enrich_match_data(session, m, i, date_str, sema):
    m["id"] = i + 1
    m["date"] = date_str

    h_task = async_fetch_api(session,"/teams",{"search":translate_team_name(m["home_team"])},sema)
    a_task = async_fetch_api(session,"/teams",{"search":translate_team_name(m["away_team"])},sema)
    h_res, a_res = await asyncio.gather(h_task, a_task)

    m["home_id"] = h_res[0]["team"]["id"] if h_res else 0
    m["away_id"] = a_res[0]["team"]["id"] if a_res else 0

    tasks = []
    if m["home_id"]: tasks.append(async_fetch_api(session,"/teams/statistics",{"team":m["home_id"],"season":2024},sema))
    else: tasks.append(asyncio.sleep(0))
    if m["away_id"]: tasks.append(async_fetch_api(session,"/teams/statistics",{"team":m["away_id"],"season":2024},sema))
    else: tasks.append(asyncio.sleep(0))
    if m["home_id"] and m["away_id"]: tasks.append(async_fetch_api(session,"/fixtures/headtohead",{"h2h":f"{m['home_id']}-{m['away_id']}","last":5},sema))
    else: tasks.append(asyncio.sleep(0))

    results = await asyncio.gather(*tasks)

    # 主队stats（含draws/losses）
    api_h = results[0]
    if api_h and isinstance(api_h, dict) and "fixtures" in api_h:
        m["home_stats"] = {
            "played": api_h["fixtures"]["played"].get("total",0),
            "wins": api_h["fixtures"]["wins"].get("total",0),
            "draws": api_h["fixtures"]["draws"].get("total",0),
            "losses": api_h["fixtures"]["loses"].get("total",0),
            "goals_for": api_h["goals"]["for"]["total"].get("total",0),
            "goals_against": api_h["goals"]["against"]["total"].get("total",0),
            "avg_goals_for": str(api_h["goals"]["for"]["average"].get("total","0.0")),
            "avg_goals_against": str(api_h["goals"]["against"]["average"].get("total","0.0")),
            "form": api_h.get("form",""),
            "clean_sheets": api_h["clean_sheet"].get("total",0),
        }
    else:
        m["home_stats"] = generate_stats_from_context(m, "home")

    # 客队stats
    api_a = results[1]
    if api_a and isinstance(api_a, dict) and "fixtures" in api_a:
        m["away_stats"] = {
            "played": api_a["fixtures"]["played"].get("total",0),
            "wins": api_a["fixtures"]["wins"].get("total",0),
            "draws": api_a["fixtures"]["draws"].get("total",0),
            "losses": api_a["fixtures"]["loses"].get("total",0),
            "goals_for": api_a["goals"]["for"]["total"].get("total",0),
            "goals_against": api_a["goals"]["against"]["total"].get("total",0),
            "avg_goals_for": str(api_a["goals"]["for"]["average"].get("total","0.0")),
            "avg_goals_against": str(api_a["goals"]["against"]["average"].get("total","0.0")),
            "form": api_a.get("form",""),
            "clean_sheets": api_a["clean_sheet"].get("total",0),
        }
    else:
        m["away_stats"] = generate_stats_from_context(m, "away")

    # H2H（含日期和队名，供经验规则引擎使用）
    h2h_raw = results[2]
    m["h2h"] = []
    if h2h_raw and isinstance(h2h_raw, list):
        m["h2h"] = [{"date":x["fixture"]["date"][:10],
                      "score":f"{x['goals']['home']}-{x['goals']['away']}",
                      "home":x["teams"]["home"]["name"],
                      "away":x["teams"]["away"]["name"]}
                     for x in h2h_raw if isinstance(x,dict) and "goals" in x]

    return m

async def async_collect_all(date_str):
    sema = asyncio.Semaphore(8)

    async with aiohttp.ClientSession() as session:
        matches = await scrape_wencai_jczq_async(session, date_str)
        if not matches: return {"date": date_str, "matches": []}

        print(f"  API-Football 并发补充数据中...")
        tasks = [enrich_match_data(session, m, i, date_str, sema) for i, m in enumerate(matches)]
        enriched = await asyncio.gather(*tasks)

    return {"date": date_str, "matches": enriched}