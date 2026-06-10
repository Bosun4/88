# -*- coding: utf-8 -*-
"""
steam_tracker.py — 大小球线移动(steam)信号独立采集模块 [P1]

读盘范式 v2.x 的范式突破点：接回唯一在 1674 场上单调有效的独立信号——
over2.5 大小球线的移动方向(开盘 -> 临场)，作为定档逻辑的资金流佐证。

【设计原则 / 符合架构洁癖】
- 完全独立模块：不被 predict.py import，不改任何生产管线。本阶段只产出信号文件。
- 复用 global_odds 的 key 读取 / 联赛映射 / 队名匹配，不重造轮子。
- 实时双采，不碰 the-odds-api 历史付费墙(quota=10x)：
    T1 开盘快照 + T2 临场快照 -> steam = p_over_close - p_over_open
- graceful 降级：拿不到 totals / 联赛不覆盖 / 无 key -> 返回中性信号，绝不抛异常。

【输出】每场注入(不直接写 match，由调用方决定)：
    {
      "steam_available": bool,
      "steam_direction": "up" | "down" | "flat" | "unknown",   # over2.5 概率走向
      "steam_strength": float,     # |Δp_over| 原始位移幅度(0~1)
      "steam_weighted": float,     # 按联赛分层系数赋权后的强度
      "p_over_open": float|None,
      "p_over_close": float|None,
      "league_weight": float,      # STAGE1 标定的联赛分层系数
      "reason": str,
    }

【联赛分层系数来源】
STAGE1 用 football-data 1674 场 B365 开盘 vs B365C 收盘标定的 over2.5 漂移赋权
(法甲 +15pp / 德甲 +2.6pp ...)。系数 = 该联赛 steam 信号的可信度放大器。
未标定联赛默认 1.0(中性)。
"""
from __future__ import annotations

import os
import json
import time
import difflib
from typing import Any, Dict, List, Optional

# ---- 复用 global_odds 的接入约定(key / base / 联赛映射 / 队名翻译) ----
try:
    from global_odds import (
        ODDS_API_KEY,
        ODDS_API_BASE,
        LEAGUE_SPORT_KEY,
        sport_key_for_league,
        translate_team_name,
        _team_similarity,
        _best_match,
        _median,
    )
except Exception:  # pragma: no cover - 退化路径,保证模块可独立加载
    try:
        from scripts.global_odds import (  # type: ignore
            ODDS_API_KEY,
            ODDS_API_BASE,
            LEAGUE_SPORT_KEY,
            sport_key_for_league,
            translate_team_name,
            _team_similarity,
            _best_match,
            _median,
        )
    except Exception:
        ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
        ODDS_API_BASE = "https://api.the-odds-api.com/v4"
        LEAGUE_SPORT_KEY = {}

        def sport_key_for_league(league):
            return LEAGUE_SPORT_KEY.get(str(league or "").strip())

        def translate_team_name(name):
            return str(name or "")

        def _team_similarity(left, right):
            return difflib.SequenceMatcher(None, str(left or "").lower(), str(right or "").lower()).ratio()

        def _best_match(target, candidates, cutoff=0.6):
            if not target or not candidates:
                return None
            hit = difflib.get_close_matches(
                target.lower(), [c.lower() for c in candidates], n=1, cutoff=cutoff
            )
            if not hit:
                return None
            for c in candidates:
                if c.lower() == hit[0]:
                    return c
            return None

        def _median(xs):
            s = sorted(x for x in xs if isinstance(x, (int, float)) and x > 1.0)
            n = len(s)
            if n == 0:
                return 0.0
            mid = n // 2
            return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


# ---- 联赛分层赋权系数 (STAGE1 1674 场 B365 开/收盘标定) ----
# 值 = steam 信号在该联赛的可信度放大器；>1 放大、<1 收敛、1.0 中性。
# 注: 这是信号置信度权重,不是概率本身。法甲漂移最显著故权重最高。
LEAGUE_STEAM_WEIGHT: Dict[str, float] = {
    "法甲": 1.45,
    "德甲": 1.26,
    "荷甲": 1.22,
    "挪超": 1.20,
    "瑞超": 1.18,
    "美职": 1.15,
    "英超": 1.05,
    "意甲": 0.95,
    "西甲": 0.95,
}
DEFAULT_LEAGUE_WEIGHT = 1.0

# steam 方向判定阈值: |Δp_over| 小于此值视为 flat(噪声)
STEAM_FLAT_EPS = 0.015

# 大小球主线: over/under 2.5
TOTALS_MAIN_POINT = 2.5

PREFERRED_BOOKMAKER = "pinnacle"


def _neutral(reason: str) -> Dict[str, Any]:
    return {
        "steam_available": False,
        "steam_direction": "unknown",
        "steam_strength": 0.0,
        "steam_weighted": 0.0,
        "p_over_open": None,
        "p_over_close": None,
        "league_weight": DEFAULT_LEAGUE_WEIGHT,
        "reason": reason,
    }


def _devig_two_way(over_price: float, under_price: float) -> Optional[float]:
    """两路赔率去抽水 -> over 的公平概率。无效返回 None。"""
    if not (isinstance(over_price, (int, float)) and over_price > 1.0):
        return None
    if not (isinstance(under_price, (int, float)) and under_price > 1.0):
        return None
    io, iu = 1.0 / over_price, 1.0 / under_price
    s = io + iu
    if s <= 0:
        return None
    return io / s  # 归一化去 vig


def extract_over_prob(event: Dict[str, Any], point: float = TOTALS_MAIN_POINT) -> Optional[float]:
    """
    从一个 the-odds-api event 抽 over{point} 的去抽水公平概率。
    优先 Pinnacle,否则各家 over/under 中位数后去 vig。totals 缺失 -> None。
    """
    pin_over: List[float] = []
    pin_under: List[float] = []
    all_over: List[float] = []
    all_under: List[float] = []
    for bm in event.get("bookmakers", []) or []:
        is_pin = bm.get("key") == PREFERRED_BOOKMAKER
        for mk in bm.get("markets", []) or []:
            if mk.get("key") != "totals":
                continue
            for oc in mk.get("outcomes", []) or []:
                try:
                    pt = float(oc.get("point", -999) or -999)
                except (TypeError, ValueError):
                    continue
                if abs(pt - point) > 1e-6:
                    continue
                nm = (oc.get("name") or "").lower()
                price = oc.get("price", 0)
                if not isinstance(price, (int, float)) or price <= 1.0:
                    continue
                if nm == "over":
                    all_over.append(price)
                    if is_pin:
                        pin_over.append(price)
                elif nm == "under":
                    all_under.append(price)
                    if is_pin:
                        pin_under.append(price)

    if pin_over and pin_under:
        return _devig_two_way(_median(pin_over), _median(pin_under))
    if all_over and all_under:
        return _devig_two_way(_median(all_over), _median(all_under))
    return None


def fetch_totals_snapshot(sport_key: str, timeout: int = 12) -> List[Dict[str, Any]]:
    """实时拉一个 sport_key 的 totals 盘快照;失败/无 key 返回 []。每次 1 quota/region。"""
    if not ODDS_API_KEY or not sport_key:
        return []
    import urllib.request

    base = ODDS_API_BASE.rstrip("/")
    if "the-odds-api" in base and not base.endswith("/v4"):
        base = base + "/v4"
    url = (
        f"{base}/sports/{sport_key}/odds/"
        f"?apiKey={ODDS_API_KEY}&regions=eu,uk&markets=totals&oddsFormat=decimal"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")) or []
    except Exception as e:  # pragma: no cover - 网络路径
        print(f"  [steam] {sport_key} totals 拉取失败: {type(e).__name__}: {str(e)[:80]}")
        return []


def _match_event(home: str, away: str, events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """按翻译后英文主客队组合模糊匹配 event,避免只中主队导致错配。"""
    en_home = translate_team_name(home)
    en_away = translate_team_name(away)
    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for ev in events:
        ev_home = str(ev.get("home_team", "") or "")
        ev_away = str(ev.get("away_team", "") or "")
        if not ev_home or not ev_away:
            continue
        home_score = _team_similarity(en_home, ev_home)
        away_score = _team_similarity(en_away, ev_away)
        score = (home_score + away_score) / 2.0
        if score > best_score:
            best_score = score
            best = ev
    return best if best and best_score >= 0.6 else None


def compute_steam(
    p_open: Optional[float],
    p_close: Optional[float],
    league: str = "",
) -> Dict[str, Any]:
    """
    给定 over2.5 开盘/临场概率 -> steam 信号。纯函数,便于离线测试。
    p_open/p_close 任一缺失 -> 中性。
    """
    weight = LEAGUE_STEAM_WEIGHT.get(league, DEFAULT_LEAGUE_WEIGHT)
    if p_open is None or p_close is None:
        out = _neutral("缺少开盘或临场 over2.5 概率(totals 不可得或联赛不覆盖)")
        out["league_weight"] = weight
        return out

    delta = p_close - p_open
    strength = abs(delta)
    if strength < STEAM_FLAT_EPS:
        direction = "flat"
    elif delta > 0:
        direction = "up"      # 资金推大球
    else:
        direction = "down"    # 资金推小球

    return {
        "steam_available": True,
        "steam_direction": direction,
        "steam_strength": round(strength, 4),
        "steam_weighted": round(strength * weight, 4),
        "p_over_open": round(p_open, 4),
        "p_over_close": round(p_close, 4),
        "league_weight": weight,
        "reason": (
            f"over2.5 概率 {round(p_open,3)} -> {round(p_close,3)} (Δ={round(delta,3)}), "
            f"方向={direction}, 联赛={league or '未知'}(权重{weight})"
        ),
    }


# ---------------- 快照持久化(开盘轨用) ----------------

def save_snapshot(matches: List[Dict[str, Any]], out_path: str) -> int:
    """
    采集当前 totals 开盘快照并落盘(T1)。
    matches: [{home_team, away_team, league, match_id?}, ...]
    返回成功记录数。临场轨复用同结构,read_snapshot 后传入 compute。
    """
    snap: Dict[str, Any] = {"ts": int(time.time()), "over25": {}}
    by_league: Dict[str, List[Dict[str, Any]]] = {}
    for m in matches:
        by_league.setdefault(m.get("league", ""), []).append(m)

    n = 0
    for league, ms in by_league.items():
        sport_key = sport_key_for_league(league)
        if not sport_key:
            continue
        events = fetch_totals_snapshot(sport_key)
        if not events:
            continue
        for m in ms:
            ev = _match_event(m.get("home_team", ""), m.get("away_team", ""), events)
            if not ev:
                continue
            p_over = extract_over_prob(ev)
            if p_over is None:
                continue
            key = m.get("match_id") or f"{m.get('home_team')}__{m.get('away_team')}"
            snap["over25"][key] = p_over
            n += 1
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
    except Exception as e:  # pragma: no cover
        print(f"  [steam] 快照落盘失败: {e}")
    return n


def load_snapshot(path: str) -> Dict[str, float]:
    try:
        with open(path, encoding="utf-8") as f:
            return (json.load(f) or {}).get("over25", {}) or {}
    except Exception:
        return {}


def build_steam_signals(
    matches: List[Dict[str, Any]],
    open_snapshot_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    临场轨主入口：对每场拉临场 totals,结合开盘快照(若有)算 steam。
    返回 {match_key: steam_signal}。无 key / 无快照 -> 全中性,不抛异常。
    """
    open_over = load_snapshot(open_snapshot_path) if open_snapshot_path else {}
    out: Dict[str, Dict[str, Any]] = {}

    by_league: Dict[str, List[Dict[str, Any]]] = {}
    for m in matches:
        by_league.setdefault(m.get("league", ""), []).append(m)

    for league, ms in by_league.items():
        sport_key = sport_key_for_league(league)
        events = fetch_totals_snapshot(sport_key) if sport_key else []
        for m in ms:
            key = m.get("match_id") or f"{m.get('home_team')}__{m.get('away_team')}"
            ev = _match_event(m.get("home_team", ""), m.get("away_team", ""), events) if events else None
            p_close = extract_over_prob(ev) if ev else None
            p_open = open_over.get(key)
            out[key] = compute_steam(p_open, p_close, league)
    return out


if __name__ == "__main__":  # pragma: no cover
    print("== compute_steam 自检 ==")
    print(compute_steam(0.50, 0.62, "挪超"))   # up
    print(compute_steam(0.55, 0.40, "意甲"))   # down
    print(compute_steam(0.50, 0.505, "德甲"))  # flat
    print(compute_steam(None, 0.6, "芬超"))    # unknown(联赛不覆盖)
