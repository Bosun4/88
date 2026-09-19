import os
import re
import json
import asyncio
import aiohttp
import uuid
import hashlib
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

# 国家队中→英映射（世界杯/国际赛事专用）。
# The Odds API soccer_fifa_world_cup 用英文国名；中文名走 GoogleTranslator 会因
# ".replace('联',' United')" 等规则被污染且联网不稳，故对国家队走显式 SSOT 映射。
NATIONAL_TEAM_MAPPING = {
    "阿根廷":"Argentina","巴西":"Brazil","法国":"France","德国":"Germany",
    "西班牙":"Spain","英格兰":"England","葡萄牙":"Portugal","荷兰":"Netherlands",
    "比利时":"Belgium","克罗地亚":"Croatia","意大利":"Italy","乌拉圭":"Uruguay",
    "哥伦比亚":"Colombia","墨西哥":"Mexico","美国":"USA","日本":"Japan",
    "韩国":"South Korea","沙特":"Saudi Arabia","伊朗":"Iran","澳大利亚":"Australia",
    "瑞士":"Switzerland","瑞典":"Sweden","挪威":"Norway","丹麦":"Denmark",
    "塞内加尔":"Senegal","摩洛哥":"Morocco","加纳":"Ghana","突尼斯":"Tunisia",
    "埃及":"Egypt","阿尔及利":"Algeria","阿尔及利亚":"Algeria","科特迪瓦":"Ivory Coast",
    "喀麦隆":"Cameroon","尼日利亚":"Nigeria","南非":"South Africa","刚果金":"DR Congo",
    "佛得角":"Cape Verde","加拿大":"Canada","厄瓜多尔":"Ecuador","巴拉圭":"Paraguay",
    "乌兹别克":"Uzbekistan","乌兹别克斯坦":"Uzbekistan","伊拉克":"Iraq","约旦":"Jordan",
    "卡塔尔":"Qatar","新西兰":"New Zealand","奥地利":"Austria","土耳其":"Turkey",
    "波黑":"Bosnia and Herzegovina","捷克":"Czech Republic","苏格兰":"Scotland",
    "海地":"Haiti","巴拿马":"Panama","库拉索":"Curacao","波兰":"Poland",
    "塞尔维亚":"Serbia","乌克兰":"Ukraine","威尔士":"Wales","哥斯达黎加":"Costa Rica",
}

def translate_team_name(name):
    if not name: return ""
    name = str(name).strip()
    if name in TEAM_NAME_MAPPING: return TEAM_NAME_MAPPING[name]
    if name in NATIONAL_TEAM_MAPPING: return NATIONAL_TEAM_MAPPING[name]
    try:
        from deep_translator import GoogleTranslator
        clean = name.replace("女足"," Women").replace("联"," United")
        return GoogleTranslator(source='zh-CN',target='en').translate(clean).replace("FC","").strip()
    except: return name

def _safe_dict(val): return val if isinstance(val, dict) else {}
def _get_float(val, default=0.0):
    try: return float(val) if val is not None else default
    except: return default


# ============================================================
# 抓取窗口过滤：只保留「今天 + 未来 N 天」的比赛，砍掉昨天及更远。
# 背景：问财接口 ?date=X 一次返回横跨多日的赛程（实测 6/17 查询返回
# 周二/三/四共 12 场）。全部进 enrich + AI 终审 = 重复抓 + token 浪费。
# 方案：用每场 stime（开赛 Unix 秒）按竞彩业务日口径（同 main.py 的
# VMAX_DATE_SHIFT_HOURS=11 偏移）算业务日，只留 [今天, 今天+days_ahead]。
# 默认 days_ahead=1 → 今天 + 明天两天。
# Fail-safe：stime 缺失/解析不出业务日 → 保留该场（宁可多跑不误杀真实比赛）。
# ------------------------------------------------------------
def _env_int(name, default):
    try:
        v = os.environ.get(name)
        return int(v) if v not in (None, "") else int(default)
    except (ValueError, TypeError):
        return int(default)


def _kickoff_from_stime(stime):
    """Wencai stime is Unix seconds; unknown formats remain unknown."""
    if isinstance(stime, bool) or not re.fullmatch(r"\d+(?:\.0+)?", str(stime)):
        return None
    try:
        ts = int(float(stime))
        if ts <= 0:
            return None
        return datetime.fromtimestamp(ts, timezone.utc)
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def _business_day_from_stime(stime, shift_hours=None):
    """由开赛 Unix 秒推算竞彩业务日 (YYYY-MM-DD)。解析失败返回 None。"""
    if shift_hours is None:
        shift_hours = _env_int("VMAX_DATE_SHIFT_HOURS", 11)
    kickoff = _kickoff_from_stime(stime)
    if kickoff is None:
        return None
    bj = timezone(timedelta(hours=8))
    dt = kickoff.astimezone(bj) - timedelta(hours=shift_hours)
    return dt.strftime("%Y-%m-%d")


def filter_matches_by_window(football_list, today=None, days_ahead=None, shift_hours=None):
    """只保留业务日 ∈ [today, today+days_ahead] 的场次。

    - today：竞彩业务日基准 (YYYY-MM-DD)。None 时按当前时间同口径推算。
    - days_ahead：今天之外再多留几天。None 时读 VMAX_FETCH_DAYS_AHEAD，默认 1。
    - stime 缺失/解析失败的场次保留（fail-safe，不误杀）。
    """
    if days_ahead is None:
        days_ahead = _env_int("VMAX_FETCH_DAYS_AHEAD", 1)
    if days_ahead < 0:
        days_ahead = 0
    if shift_hours is None:
        shift_hours = _env_int("VMAX_DATE_SHIFT_HOURS", 11)
    bj = timezone(timedelta(hours=8))
    if today is None:
        today = (datetime.now(bj) - timedelta(hours=shift_hours)).strftime("%Y-%m-%d")
    try:
        base = datetime.strptime(today, "%Y-%m-%d")
    except (ValueError, TypeError):
        return list(football_list)
    allowed = {(base + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(0, days_ahead + 1)}

    kept, dropped = [], 0
    for item in football_list:
        bizday = _business_day_from_stime((item or {}).get("stime"), shift_hours)
        if bizday is None or bizday in allowed:
            kept.append(item)
        else:
            dropped += 1
    if dropped:
        print(f"  🗓️ 抓取窗口过滤：保留 {len(kept)} 场（业务日 {sorted(allowed)}），砍掉 {dropped} 场窗口外比赛")
    return kept

def generate_stats_from_context(match, side):
    """Missing statistics cannot be reconstructed from odds or rankings."""
    return {
        "estimated": False, "data_available": False, "quality": "unavailable",
        "source": None, "data_note": "未取得可核验赛季战绩；赔率与排名不能反推比赛记录",
    }

async def scrape_wencai_jczq_async(session, date_str):
    """抓取问财数据，自动隔离足球与篮球（防止篮球数据污染泊松模型）"""
    # 2026-07-18 起，旧版无鉴权 GET (?date=...) 返回 code=301「非法请求」。
    # 新版接口要求 JSON POST；Authorization 从 Secret 注入，设备 UUID 与
    # client id 每次运行随机生成，UA 使用不含用户设备信息的通用桌面标识。
    url = "https://edu.wencaivip.cn/api/v1.reference/matches"
    football_matches = []

    authorization = os.environ.get("WENCAI_AUTHORIZATION", "").strip()
    if not authorization:
        print("  ❌ 问财新版接口缺少运行凭证: WENCAI_AUTHORIZATION")
        return []

    device_id = str(uuid.uuid4())
    client_id = f"wc-{uuid.uuid4().hex}"

    headers = {
        "User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept":"application/json, text/plain, */*",
        "Accept-Language":"zh-CN,zh;q=0.9,en;q=0.8",
        "Content-Type":"application/json",
        "Origin":"http://m.wencai51.cn",
        "Referer":"http://m.wencai51.cn/",
        "Authorization": authorization,
    }
    payload = {"i": device_id, "cid": client_id}

    try:
        async with session.post(url, headers=headers, json=payload, timeout=15) as r:
            if r.status != 200:
                print(f"  ❌ 抓取失败 HTTP {r.status}")
                return []

            # 服务端可能用非 JSON MIME 返回错误体，先读文本再解析以保留诊断。
            raw_text = await r.text()
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError:
                print(f"  ❌ 问财响应不是有效 JSON: {raw_text[:200]!r}")
                return []

            if data.get("code") not in (0, "0", None):
                print(f"  ❌ 问财业务拒绝 code={data.get('code')} msg={data.get('msg', '')}")
                return []

            if "data" not in data or not data["data"]:
                print(f"  ⚠️ 接口未返回数据: {str(data)[:200]}")
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

            # 抓取窗口过滤：问财接口一次返回跨多日赛程，只保留今天+未来 N 天（默认 1=今明两天）
            football_list = filter_matches_by_window(football_list, today=date_str)
            if not football_list:
                print(f"  [INFO] 窗口过滤后无符合赛事")
                return []

            captured = datetime.now(timezone.utc)
            for item in football_list:
                try:
                    kickoff = _kickoff_from_stime(item.get("stime"))
                    if kickoff and kickoff <= captured:
                        continue
                    source_id = item.get("id")
                    match_id = f"wencai:{source_id}" if source_id else None
                    if not match_id and kickoff and all(item.get(k) for k in ("cup", "home", "guest")):
                        identity = [item["cup"], item.get("season"), item["home"],
                                    item["guest"], kickoff.isoformat()]
                        match_id = "fixture:" + hashlib.sha256(
                            json.dumps(identity, ensure_ascii=False).encode("utf-8")
                        ).hexdigest()
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
                        "match_id": match_id,
                        "source_event_id": source_id,
                        "source": "wencai",
                        "source_url": url,
                        "stime": item.get("stime"),
                        "kickoff_at": kickoff.isoformat() if kickoff else None,
                        "captured_at": captured.isoformat(),
                        "kickoff_status": "known" if kickoff else "unknown",
                        "season": item.get("season"),
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

# Provider competition identifiers, never season assumptions.
API_LEAGUE_IDS = {
    "英超": 39, "英冠": 40, "西甲": 140, "意甲": 135, "德甲": 78,
    "德乙": 79, "法甲": 61, "法乙": 62, "荷甲": 88, "葡超": 94,
    "比甲": 144, "土超": 203, "苏超": 179, "日职": 98, "韩职": 292,
    "澳超": 188, "巴甲": 71, "阿甲": 128, "美职": 253, "挪超": 103,
    "瑞超": 113, "欧冠": 2, "欧罗巴": 3, "欧协联": 848, "世界杯": 1,
}


def _api_stats(raw, team_id, league_id, season, provenance):
    """Only return complete, correctly scoped provider statistics."""
    fallback = generate_stats_from_context({}, "")
    if not isinstance(raw, dict):
        return fallback
    if (raw.get("team", {}).get("id") != team_id
            or raw.get("league", {}).get("id") != league_id
            or raw.get("league", {}).get("season") != season):
        return fallback
    try:
        fixtures = raw["fixtures"]
        stats = {key: fixtures[src]["total"] for key, src in
                 (("played", "played"), ("wins", "wins"),
                  ("draws", "draws"), ("losses", "loses"))}
        stats.update({"goals_for": raw["goals"]["for"]["total"]["total"],
                      "goals_against": raw["goals"]["against"]["total"]["total"]})
        if any(type(v) is not int or v < 0 for v in stats.values()):
            return fallback
        if stats["wins"] + stats["draws"] + stats["losses"] != stats["played"]:
            return fallback
        stats.update({"form": raw.get("form"),
                      "avg_goals_for": raw["goals"]["for"].get("average", {}).get("total"),
                      "avg_goals_against": raw["goals"]["against"].get("average", {}).get("total"),
                      "clean_sheets": raw.get("clean_sheet", {}).get("total"),
                      "estimated": False, "data_available": True,
                      "quality": "observed", **provenance})
        return stats
    except (KeyError, TypeError):
        return fallback


async def enrich_match_data(session, m, i, date_str, sema):
    """Resolve a unique fixture before fetching season-scoped facts."""
    from fixture_identity import parse_time

    m.update({"id": i + 1, "date": date_str, "h2h": [],
              "home_stats": generate_stats_from_context(m, "home"),
              "away_stats": generate_stats_from_context(m, "away")})
    kickoff = parse_time(m.get("kickoff_at"))
    league_id = API_LEAGUE_IDS.get(m.get("league"))
    if not API_FOOTBALL_KEY or not kickoff or not league_id:
        return m
    # No general-purpose machine translation at an identity boundary.
    def name(value):
        value = TEAM_NAME_MAPPING.get(value, NATIONAL_TEAM_MAPPING.get(value, value))
        return str(value or "").strip().casefold()

    candidates = await async_fetch_api(
        session, "/fixtures", {"date": kickoff.date().isoformat(),
                               "league": league_id, "timezone": "UTC"}, sema)
    hits = []
    for row in candidates if isinstance(candidates, list) else []:
        if not isinstance(row, dict):
            continue
        f, league, teams = (row.get(k) or {} for k in ("fixture", "league", "teams"))
        if (f.get("id") and league.get("id") == league_id
                and parse_time(f.get("date")) == kickoff
                and all(name(m.get(f"{s}_team")) == name((teams.get(s) or {}).get("name"))
                        and (teams.get(s) or {}).get("id") for s in ("home", "away"))):
            hits.append(row)
    if len(hits) != 1 or not hits[0]["league"].get("season"):
        return m
    fixture = hits[0]
    season = fixture["league"]["season"]
    if m.get("season") and str(m["season"]) != str(season):
        return m
    m.update({"api_football_fixture_id": fixture["fixture"]["id"],
              "league_id": league_id, "season": season})
    for side in ("home", "away"):
        m[f"{side}_id"] = fixture["teams"][side]["id"]
    captured = datetime.now(timezone.utc).isoformat()
    common = {"source": "api_football", "captured_at": captured,
              "league_id": league_id, "season": season}
    def proof(endpoint, params):
        from urllib.parse import urlencode
        return {**common, "source_url": f"{API_FOOTBALL_BASE}{endpoint}?{urlencode(params)}"}

    base = {"league": league_id, "season": season}
    jobs = [("/teams/statistics", {**base, "team": m[f"{s}_id"]})
            for s in ("home", "away")]
    jobs += [("/standings", base), ("/fixtures/rounds", base)]
    for side in ("home", "away"):
        jobs.append(("/fixtures", {"team": m[f"{side}_id"],
                                  "from": (kickoff - timedelta(days=14)).date().isoformat(),
                                  "to": (kickoff + timedelta(days=14)).date().isoformat(),
                                  "timezone": "UTC"}))
    jobs.append(("/fixtures/headtohead", {"h2h": f"{m['home_id']}-{m['away_id']}",
                                        "from": "2000-01-01",
                                        "to": kickoff.date().isoformat()}))
    results = await asyncio.gather(*(async_fetch_api(session, e, p, sema) for e, p in jobs))
    # Timestamp the responses, not the request start.
    common["captured_at"] = datetime.now(timezone.utc).isoformat()
    evidence = {"fixture": {**proof("/fixtures", {"id": fixture["fixture"]["id"]}),
                            "fixture_id": fixture["fixture"]["id"],
                            "kickoff_at": kickoff.isoformat(),
                            "home_id": m["home_id"], "away_id": m["away_id"],
                            "round": fixture["league"].get("round")}}
    for index, side in enumerate(("home", "away")):
        m[f"{side}_stats"] = _api_stats(results[index], m[f"{side}_id"],
                                        league_id, season, proof(*jobs[index]))
    evidence["standings"] = {**proof(*jobs[2]), "groups": []}
    for entry in results[2] if isinstance(results[2], list) else []:
        league = entry.get("league") or {}
        if league.get("id") == league_id and league.get("season") == season:
            evidence["standings"]["groups"].extend(league.get("standings") or [])
    evidence["rounds"] = {**proof(*jobs[3]), "rounds": results[3]}
    for index, side in enumerate(("home", "away"), 4):
        evidence[f"{side}_schedule"] = {**proof(*jobs[index]), "team_id": m[f"{side}_id"],
                                        "fixtures": results[index]}
    m["league_evidence"] = evidence
    history = results[6] if isinstance(results[6], list) else []
    for row in history:
        f, goals = row.get("fixture") or {}, row.get("goals") or {}
        when = parse_time(f.get("date"))
        if (when and when < min(kickoff, datetime.now(timezone.utc))
                and (f.get("status") or {}).get("short") == "FT"
                and all(type(goals.get(s)) is int for s in ("home", "away"))):
            m["h2h"].append({"date": when.isoformat(),
                              "score": f"{goals['home']}-{goals['away']}",
                              "home": row["teams"]["home"]["name"],
                              "away": row["teams"]["away"]["name"],
                              **proof(*jobs[6])})
    m["h2h"] = sorted(m["h2h"], key=lambda x: x["date"], reverse=True)[:5]
    return m


async def async_collect_all(date_str):
    sema = asyncio.Semaphore(8)

    async with aiohttp.ClientSession() as session:
        matches = await scrape_wencai_jczq_async(session, date_str)
        if not matches: return {"date": date_str, "matches": []}

        print(f"  API-Football 并发补充数据中...")
        tasks = [enrich_match_data(session, m, i, date_str, sema) for i, m in enumerate(matches)]
        enriched = await asyncio.gather(*tasks)

    # 注入国际低抽水欧赔(点亮双轨市场背离防线);fail-safe,失败自动降级单轨
    try:
        from global_odds import enrich_with_global_odds_async
        await enrich_with_global_odds_async(enriched)
    except Exception as e:
        print(f"  [global_odds] 模块加载/执行失败,降级单轨: {type(e).__name__}: {str(e)[:80]}")

    return {"date": date_str, "matches": enriched}
