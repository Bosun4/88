# -*- coding: utf-8 -*-
"""
global_odds.py — 国际低抽水欧赔抓取与匹配(点亮双轨背离防线)

职责:
- 调 The Odds API(已配置 ODDS_API_KEY)拉取主流联赛 h2h(1X2)收盘欧赔。
- 优先取 Pinnacle,退而取所有 bookmaker 的中位数,作为"国际清算盘"基准。
- 以英文队名(经 translate_team_name 转换)模糊匹配,把 global_home/draw/away
  写进每个 match 对象,供 predict.build_evidence_packet 计算 Shin 偏斜度。

设计红线:
- 全程 fail-safe:任何异常/无 key/无外网/匹配不到 → 静默跳过,绝不打断主管线。
- 不修改既有竞彩赔率,只新增 global_* 字段。
- 按请求计费,所以每个 sport_key 只拉一次并缓存到本次进程。
"""
import asyncio
import difflib
from typing import Dict, List, Any, Optional

try:
    from config import ODDS_API_KEY, ODDS_API_BASE
except Exception:
    import os
    ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
    ODDS_API_BASE = "https://api.the-odds-api.com/v4"

try:
    from fetch_data import translate_team_name
except Exception:
    def translate_team_name(name):
        return str(name or "")


# 中文联赛名 → The Odds API sport_key(只覆盖 API 实际支持的联赛)
LEAGUE_SPORT_KEY = {
    "英超": "soccer_epl",
    "英冠": "soccer_efl_champ",
    "德甲": "soccer_germany_bundesliga",
    "德乙": "soccer_germany_bundesliga2",
    "西甲": "soccer_spain_la_liga",
    "意甲": "soccer_italy_serie_a",
    "法甲": "soccer_france_ligue_one",
    "法乙": "soccer_france_ligue_two",
    "荷甲": "soccer_netherlands_eredivisie",
    "葡超": "soccer_portugal_primeira_liga",
    "比甲": "soccer_belgium_first_div",
    "土超": "soccer_turkey_super_league",
    "苏超": "soccer_spl",
    "日职": "soccer_japan_j_league",
    "韩职": "soccer_korea_kleague1",
    "澳超": "soccer_australia_aleague",
    "巴甲": "soccer_brazil_campeonato",
    "阿甲": "soccer_argentina_primera_division",
    "美职": "soccer_usa_mls",
    "挪超": "soccer_norway_eliteserien",
    "瑞超": "soccer_sweden_allsvenskan",
    "世界杯": "soccer_fifa_world_cup",
    "欧冠": "soccer_uefa_champs_league",
    "欧罗巴": "soccer_uefa_europa_league",
    "欧协联": "soccer_uefa_europa_conference_league",
    "解放者杯": "soccer_conmebol_copa_libertadores",
    "南美解放者杯": "soccer_conmebol_copa_libertadores",
}

# Pinnacle 优先;其后取全场中位数
PREFERRED_BOOKMAKER = "pinnacle"


def _median(xs: List[float]) -> float:
    s = sorted(x for x in xs if isinstance(x, (int, float)) and x > 1.0)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _extract_1x2(event: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """从一个 event 抽 1X2 欧赔:优先 Pinnacle,否则各家中位数。"""
    home = event.get("home_team", "")
    away = event.get("away_team", "")
    if not home or not away:
        return None

    pin = {"home": [], "draw": [], "away": []}
    allbm = {"home": [], "draw": [], "away": []}
    for bm in event.get("bookmakers", []):
        is_pin = bm.get("key") == PREFERRED_BOOKMAKER
        for mk in bm.get("markets", []):
            if mk.get("key") != "h2h":
                continue
            for oc in mk.get("outcomes", []):
                nm, price = oc.get("name"), oc.get("price", 0)
                if not isinstance(price, (int, float)) or price <= 1.0:
                    continue
                sel = "home" if nm == home else ("away" if nm == away else ("draw" if nm == "Draw" else None))
                if sel is None:
                    continue
                allbm[sel].append(price)
                if is_pin:
                    pin[sel].append(price)

    src = pin if all(pin[k] for k in ("home", "draw", "away")) else allbm
    odds = {k: _median(src[k]) for k in ("home", "draw", "away")}
    if all(odds[k] > 1.0 for k in ("home", "draw", "away")):
        return {"home_team": home, "away_team": away, "odds": odds}
    return None


def _fetch_sport(sport_key: str) -> List[Dict[str, Any]]:
    """同步拉一个 sport_key 的赔率;失败返回 []。"""
    if not ODDS_API_KEY:
        return []
    import urllib.request
    import json as _json
    base = ODDS_API_BASE.rstrip("/")
    if not base.endswith("/v4"):
        base = base + "/v4" if "the-odds-api" in base else base
    url = (f"{base}/sports/{sport_key}/odds/"
           f"?apiKey={ODDS_API_KEY}&regions=eu,uk&markets=h2h&oddsFormat=decimal")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            return _json.loads(r.read().decode("utf-8")) or []
    except Exception as e:
        print(f"  [global_odds] {sport_key} 拉取失败: {type(e).__name__}: {str(e)[:80]}")
        return []


def _best_match(target: str, candidates: List[str], cutoff: float = 0.6) -> Optional[str]:
    if not target or not candidates:
        return None
    hit = difflib.get_close_matches(target.lower(), [c.lower() for c in candidates], n=1, cutoff=cutoff)
    if not hit:
        return None
    for c in candidates:
        if c.lower() == hit[0]:
            return c
    return None


def enrich_with_global_odds(matches: List[Dict[str, Any]]) -> int:
    """
    给 matches 注入 global_home/draw/away。返回成功匹配的场次数。
    全程 fail-safe:任何失败只影响该场缺省,不抛出。
    """
    if not ODDS_API_KEY:
        print("  [global_odds] 未配置 ODDS_API_KEY,跳过国际盘背离防线(单轨运行)")
        return 0

    # 按 sport_key 分组需要拉取的联赛
    need_keys = {}
    for m in matches:
        sk = LEAGUE_SPORT_KEY.get(str(m.get("league", "")).strip())
        if sk:
            need_keys.setdefault(sk, []).append(m)

    if not need_keys:
        print("  [global_odds] 本批次无 The Odds API 覆盖的联赛,跳过")
        return 0

    matched = 0
    for sk, group in need_keys.items():
        events = _fetch_sport(sk)
        parsed = [p for p in (_extract_1x2(e) for e in events) if p]
        if not parsed:
            continue
        en_names = []
        for p in parsed:
            en_names.append(p["home_team"])
        # 用 (home, away) 组合匹配,避免同队主客混淆
        for m in group:
            try:
                h_en = translate_team_name(m.get("home_team", ""))
                a_en = translate_team_name(m.get("away_team", ""))
                best = None
                best_score = 0.0
                for p in parsed:
                    hs = difflib.SequenceMatcher(None, h_en.lower(), p["home_team"].lower()).ratio()
                    as_ = difflib.SequenceMatcher(None, a_en.lower(), p["away_team"].lower()).ratio()
                    score = (hs + as_) / 2.0
                    if score > best_score:
                        best_score, best = score, p
                if best and best_score >= 0.6:
                    m["global_home"] = round(best["odds"]["home"], 3)
                    m["global_draw"] = round(best["odds"]["draw"], 3)
                    m["global_away"] = round(best["odds"]["away"], 3)
                    m["global_odds_source"] = "the_odds_api"
                    m["global_odds_match_score"] = round(best_score, 3)
                    matched += 1
            except Exception:
                continue

    print(f"  [global_odds] 国际欧赔匹配成功 {matched}/{len(matches)} 场,双轨背离防线已点亮")
    return matched


async def enrich_with_global_odds_async(matches: List[Dict[str, Any]]) -> int:
    """async 包装:把同步网络IO丢到线程池,避免阻塞事件循环。"""
    try:
        return await asyncio.to_thread(enrich_with_global_odds, matches)
    except Exception as e:
        print(f"  [global_odds] 注入异常,降级单轨: {type(e).__name__}: {str(e)[:80]}")
        return 0
