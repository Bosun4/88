# -*- coding: utf-8 -*-
"""
global_odds.py — 国际低抽水欧赔抓取与匹配(点亮双轨背离防线)

职责:
- 调 The Odds API 拉取主流联赛 h2h(1X2)赛前快照，并保留报价时间与来源。
- 优先取 Pinnacle,否则对完整且一小时内更新的 bookmaker 市场取中位数。
- 联赛、双方独立精确别名与带时区开赛时间联合匹配，把 global_home/draw/away
  写进每个 match 对象,供 predict.build_evidence_packet 计算 Shin 偏斜度。

设计红线:
- 全程 fail-safe:任何异常/无 key/无外网/匹配不到 → 静默跳过,绝不打断主管线。
- 不修改既有竞彩赔率,只新增 global_* 字段。
- 按请求计费,所以每个 sport_key 只拉一次并缓存到本次进程。
"""
import asyncio
import difflib
import math
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

try:
    from config import ODDS_API_KEY, ODDS_API_BASE
except Exception:
    import os
    ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
    ODDS_API_BASE = "https://api.the-odds-api.com/v4"

try:
    from fetch_data import TEAM_NAME_MAPPING, NATIONAL_TEAM_MAPPING
except Exception:
    TEAM_NAME_MAPPING, NATIONAL_TEAM_MAPPING = {}, {}

try:
    from fixture_identity import parse_time
except ImportError:
    from .fixture_identity import parse_time


def translate_team_name(name):
    """Identity aliases only; machine translation is not entity resolution."""
    return TEAM_NAME_MAPPING.get(name, NATIONAL_TEAM_MAPPING.get(name, str(name or "")))


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
MAX_QUOTE_AGE_SECONDS = 3600


def _median(xs: List[float]) -> float:
    s = sorted(x for x in xs if isinstance(x, (int, float)) and x > 1.0)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _extract_1x2(event: Dict[str, Any], now=None) -> Optional[Dict[str, Any]]:
    """Select complete fresh bookmaker markets and retain their lineage."""
    now = parse_time(now or datetime.now(timezone.utc))
    kickoff = parse_time(event.get("commence_time"))
    if not now or not kickoff or now >= kickoff or not event.get("id"):
        return None
    home = event.get("home_team", "")
    away = event.get("away_team", "")
    if not home or not away:
        return None

    quotes = []
    for bm in event.get("bookmakers", []):
        if not isinstance(bm, dict) or not bm.get("key"):
            continue
        for mk in bm.get("markets", []):
            if mk.get("key") != "h2h":
                continue
            updated = parse_time(mk.get("last_update", bm.get("last_update")))
            if not updated or not 0 <= (now - updated).total_seconds() <= MAX_QUOTE_AGE_SECONDS:
                continue
            prices = {}
            for oc in mk.get("outcomes", []):
                nm, price = oc.get("name"), oc.get("price", 0)
                sel = "home" if nm == home else ("away" if nm == away else ("draw" if nm == "Draw" else None))
                if (sel is None or sel in prices or type(price) not in (int, float)
                        or not math.isfinite(price) or price <= 1.0):
                    prices = {}
                    break
                prices[sel] = price
            if set(prices) == {"home", "draw", "away"}:
                quotes.append({"bookmaker": bm["key"], "bookmaker_title": bm.get("title"),
                               "event_id": event["id"], "market": "h2h",
                               "last_update": updated.isoformat(),
                               "captured_at": now.isoformat(), "odds": prices})
    if not quotes:
        return None
    # Repeated/conflicting bookmaker markets cannot be counted as independent books.
    if len({q["bookmaker"] for q in quotes}) != len(quotes):
        return None
    selected = [q for q in quotes if q["bookmaker"] == PREFERRED_BOOKMAKER] or quotes
    odds = {s: _median([q["odds"][s] for q in selected]) for s in ("home", "draw", "away")}
    return {"home_team": home, "away_team": away, "odds": odds,
            "event_id": event["id"], "sport_key": event.get("sport_key"),
            "kickoff_at": kickoff.isoformat(), "captured_at": now.isoformat(),
            "market": "h2h", "quotes": selected, "quality": "eligible",
            "aggregation": "pinnacle" if selected[0]["bookmaker"] == PREFERRED_BOOKMAKER
                           else "selection_median_complete_books",
            "last_update": min(q["last_update"] for q in selected)}


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
        # Request exceptions may embed the API-key query string.
        print(f"  [global_odds] {sport_key} 拉取失败: {type(e).__name__}")
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


def enrich_with_global_odds(matches: List[Dict[str, Any]], now=None) -> int:
    """Attach only unique, exact-side, same-league/time, fresh pre-match quotes.

    now is an explicit replay clock for offline tests; live capture time is taken
    after each HTTP response. Exact normalized aliases are the per-side threshold
    (1.0); fuzzy names never establish a sporting entity.
    """
    for match in matches:
        for key in list(match):
            if key.startswith("global_"):
                match.pop(key)
        match["global_odds_evidence"] = {
            "quality": "unavailable", "reason": "no_verified_fresh_quote",
            "source": "the_odds_api", "market": "h2h", "quotes": [],
        }
    if not ODDS_API_KEY:
        return 0
    groups = {}
    for match in matches:
        sport = LEAGUE_SPORT_KEY.get(str(match.get("league", "")).strip())
        if sport and parse_time(match.get("kickoff_at")):
            groups.setdefault(sport, []).append(match)
    matched = 0
    for sport, group in groups.items():
        events = _fetch_sport(sport)
        captured = parse_time(now or datetime.now(timezone.utc))
        if not captured or not isinstance(events, list):
            continue
        for match in group:
            kickoff = parse_time(match.get("kickoff_at"))
            if not kickoff or captured >= kickoff:
                continue
            names = {side: translate_team_name(match.get(f"{side}_team", "")).strip().casefold()
                     for side in ("home", "away")}
            candidates = [e for e in events if isinstance(e, dict)
                          and e.get("id") and e.get("sport_key") == sport
                          and parse_time(e.get("commence_time")) == kickoff
                          and all(names[s] and names[s] == str(e.get(f"{s}_team", "")).strip().casefold()
                                  for s in ("home", "away"))]
            # Ambiguity is checked before quote availability; a stale duplicate
            # is still a second possible event, not evidence of the first one.
            if len(candidates) != 1:
                continue
            try:
                quote = _extract_1x2(candidates[0], now=captured)
            except (TypeError, ValueError, KeyError, AttributeError):
                continue
            if not quote:
                continue
            quote.update({"source": "the_odds_api",
                          "source_url": f"{ODDS_API_BASE.rstrip('/')}/sports/{sport}/odds/",
                          "identity_method": "league_both_exact_aliases_kickoff",
                          "side_match_scores": {"home": 1.0, "away": 1.0}})
            for side in ("home", "draw", "away"):
                match[f"global_{side}"] = quote["odds"][side]
            match.update({"global_odds_source": "the_odds_api",
                          "global_odds_match_score": 1.0,
                          "global_odds_evidence": quote})
            matched += 1
    print(f"  [global_odds] verified fresh event quotes {matched}/{len(matches)}")
    return matched


async def enrich_with_global_odds_async(matches: List[Dict[str, Any]]) -> int:
    """async 包装:把同步网络IO丢到线程池,避免阻塞事件循环。"""
    try:
        return await asyncio.to_thread(enrich_with_global_odds, matches)
    except Exception as e:
        print(f"  [global_odds] 注入异常,降级单轨: {type(e).__name__}: {str(e)[:80]}")
        return 0
