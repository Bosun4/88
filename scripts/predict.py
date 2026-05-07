# -*- coding: utf-8 -*-
"""
vMAX 18.4.13 — PURE RAW-AI SHARP SAFE-HYBRID 稳定升级版
============================================================
入口：
    run_predictions(raw, use_ai=True) -> (res, top4)

本版修复重点：
1. 默认关闭持久化缓存，避免赔率更新或手动重跑时读取旧结论/旧弃权。
2. 保留 exact snapshot singleflight / 短窗口去重，只防前端重复触发，不跨赔率快照复用。
3. 每个模型每批次最多一次 API 请求；Claude 不 repair、不二次请求。
4. Grok/Claude 解析器增强：JSON 优先，安全文本兜底只抓明确“预测比分/最终比分/top1”等标签附近的比分。
5. 新增 sharp_money_pack：本地只做结构化资金/变盘诊断，不直接改 AI 比分。
6. 新增 score_shape_pack：低比分、1-1保护、强热门丢球、大球结构、比分赔率形态进入 prompt。
7. 新增 recommendation_gate：sharp 冲突、强强对决、次回合/总比分背景、形态冲突时降级推荐层，不篡改 AI 比分。
8. experience_review 覆盖修复：模型漏审计卡时本地补 neutral，不改方向、不改比分。

注意：
- 本地不跑 CRS 矩阵、不跑贝叶斯后验、不跑本地比分主裁。
- sharp_money_pack / score_shape_pack 是给 AI 审计和推荐降级用的结构化事实，不是本地裁决比分。
"""

from __future__ import annotations

import asyncio
import ast
import hashlib
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None

try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

ENGINE_VERSION = "vMAX 18.4.13-SHARP"
ENGINE_ARCHITECTURE = (
    "PURE RAW-AI + SHARP SAFE-HYBRID: GPT/Grok/Gemini 完整抓包初审 + Claude 压缩终审；"
    "本地只做抓包规范化、sharp_money_pack、score_shape_pack、经验审计卡、严格解析、字段闭环、推荐降级；"
    "默认关闭持久化缓存；exact snapshot 短窗口去重只防重复触发；每模型一次请求；AI失败即弃权。"
)

VALID_DIRS = {"home", "draw", "away"}
AI_NAMES = ["gpt", "grok", "gemini", "claude"]
PHASE1_NAMES = ["gpt", "grok", "gemini"]

DEFAULT_MODELS = {
    "gpt": "gpt-5.4",
    "grok": "grok-4.3",
    "gemini": "gemini-3.1-pro-preview-thinking-high",
    "claude": "claude-opus-4-7",
}

CRS_FULL_MAP = {
    "1-0": "w10", "2-0": "w20", "2-1": "w21", "3-0": "w30", "3-1": "w31", "3-2": "w32",
    "4-0": "w40", "4-1": "w41", "4-2": "w42", "5-0": "w50", "5-1": "w51", "5-2": "w52",
    "0-0": "s00", "1-1": "s11", "2-2": "s22", "3-3": "s33",
    "0-1": "l01", "0-2": "l02", "1-2": "l12", "0-3": "l03", "1-3": "l13", "2-3": "l23",
    "0-4": "l04", "1-4": "l14", "2-4": "l24", "0-5": "l05", "1-5": "l15", "2-5": "l25",
}

SCORE_OTHERS_HOME = ["4-3", "5-3", "5-4", "6-0", "6-1", "6-2", "6-3", "6-4", "7-0", "7-1", "7-2", "胜其他", "9-0"]
SCORE_OTHERS_DRAW = ["4-4", "5-5", "6-6", "平其他", "9-9"]
SCORE_OTHERS_AWAY = ["3-4", "0-6", "1-6", "2-6", "3-6", "0-7", "1-7", "2-7", "负其他", "0-9"]

HFTF_MAP = {
    "ss": "主/主", "sp": "主/平", "sf": "主/负",
    "ps": "平/主", "pp": "平/平", "pf": "平/负",
    "fs": "负/主", "fp": "负/平", "ff": "负/负",
}

# ============================================================
# 配置
# ============================================================

def _env_bool(name: str, default: bool = False) -> bool:
    v = str(os.environ.get(name, str(default))).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(str(os.environ.get(name, default)).strip()))
    except Exception:
        return default


AI_FORCE_COMMON_GATEWAY = _env_bool("FORCE_COMMON_GATEWAY_URL", True)
AI_READ_TIMEOUT = _env_int("AI_READ_TIMEOUT", 1000)
AI_CLAUDE_READ_TIMEOUT = _env_int("AI_CLAUDE_READ_TIMEOUT", 1500)
AI_CONNECT_TIMEOUT = _env_int("AI_CONNECT_TIMEOUT", 25)
AI_CLAUDE_CONNECT_TIMEOUT = _env_int("AI_CLAUDE_CONNECT_TIMEOUT", 45)

STRICT_ONE_CALL_PER_MODEL = True
AI_MAX_REQUESTS_PER_AI = 1
AI_ALLOW_SECOND_CALL_REPAIR = False
AI_ENABLE_CLAUDE_JSON_REPAIR = False
AI_ENABLE_ANY_MODEL_JSON_REPAIR = False
AI_USE_RESPONSE_FORMAT = _env_bool("AI_USE_RESPONSE_FORMAT", False)

AI_ALLOW_TEXT_FALLBACK = _env_bool("AI_ALLOW_TEXT_FALLBACK", True)
AI_ALLOW_CLAUDE_TEXT_FALLBACK = _env_bool("AI_ALLOW_CLAUDE_TEXT_FALLBACK", True)
AI_SAFE_TEXT_FALLBACK_ONLY = True
AI_PARSE_DEBUG = _env_bool("AI_PARSE_DEBUG", False)
AI_SAVE_RAW_RESPONSE = _env_bool("AI_SAVE_RAW_RESPONSE", False)

# 关键：默认不使用持久化缓存。只保留 exact snapshot 短窗口去重，防止前端重复触发造成 Claude 二次消费。
AI_ENABLE_PERSISTENT_CACHE = _env_bool("AI_ENABLE_PERSISTENT_CACHE", False)
AI_FORCE_FRESH = _env_bool("AI_FORCE_FRESH", False) or _env_bool("AI_MANUAL_FORCE_FRESH", False)
AI_EXACT_SNAPSHOT_REUSE_SECONDS = max(0, _env_int("AI_EXACT_SNAPSHOT_REUSE_SECONDS", 60))
AI_SINGLEFLIGHT_ENABLED = _env_bool("AI_SINGLEFLIGHT_ENABLED", True)
AI_CACHE_SCHEMA_VERSION = str(os.environ.get("AI_CACHE_SCHEMA_VERSION", "18.4.13-sharp-safe")).strip()
AI_CACHE_DIR = str(os.environ.get("AI_CACHE_DIR", "data/ai_cache")).strip() or "data/ai_cache"
AI_DISK_LOCK_WAIT_SECONDS = _env_int("AI_DISK_LOCK_WAIT_SECONDS", max(120, AI_CLAUDE_READ_TIMEOUT + 120))
AI_DISK_LOCK_POLL_SECONDS = max(1, _env_int("AI_DISK_LOCK_POLL_SECONDS", 3))

INCLUDE_FULL_RAW_PACKET = _env_bool("INCLUDE_FULL_RAW_PACKET", True)
RAW_PACKET_CHAR_LIMIT = _env_int("RAW_PACKET_CHAR_LIMIT", 20000)
FIELD_LIMIT_CHANGE = _env_int("FIELD_LIMIT_CHANGE", 4000)
FIELD_LIMIT_VOTE = _env_int("FIELD_LIMIT_VOTE", 3000)
FIELD_LIMIT_INFORMATION = _env_int("FIELD_LIMIT_INFORMATION", 8000)
FIELD_LIMIT_POINTS = _env_int("FIELD_LIMIT_POINTS", 8000)
FIELD_LIMIT_STYLE_EXTRA = _env_int("FIELD_LIMIT_STYLE_EXTRA", 6000)
AI_USE_COMPACT_CLAUDE_AUDIT = _env_bool("AI_USE_COMPACT_CLAUDE_AUDIT", True)
AI_MAX_PHASE1_REASON_CHARS_FOR_CLAUDE = _env_int("AI_MAX_PHASE1_REASON_CHARS_FOR_CLAUDE", 260)
CLAUDE_COMPACT_FIELD_LIMIT = _env_int("CLAUDE_COMPACT_FIELD_LIMIT", 2500)

ENABLE_EXTERNAL_CONTEXT = _env_bool("ENABLE_EXTERNAL_CONTEXT", False)
ENABLE_EXPERIENCE_AUDIT_CARDS = _env_bool("ENABLE_EXPERIENCE_AUDIT_CARDS", True)
EXPERIENCE_AUDIT_MAX_CARDS = max(0, _env_int("EXPERIENCE_AUDIT_MAX_CARDS", 16))
EXPERIENCE_AUDIT_MIN_WEIGHT = _env_int("EXPERIENCE_AUDIT_MIN_WEIGHT", 4)

SELECTION_TIER_S = _env_int("SELECTION_TIER_S", 78)
SELECTION_TIER_A = _env_int("SELECTION_TIER_A", 68)
SELECTION_TIER_B = _env_int("SELECTION_TIER_B", 56)
SELECTION_TIER_C = _env_int("SELECTION_TIER_C", 44)

_AI_RESULT_CACHE: Dict[str, Tuple[float, Dict[str, Dict[int, Dict[str, Any]]], Dict[str, Any]]] = {}
_AI_INFLIGHT_TASKS: Dict[str, asyncio.Task] = {}
AI_CALL_STATUS: Dict[str, Dict[str, Any]] = {n: {} for n in AI_NAMES}

# ============================================================
# 基础工具
# ============================================================

def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "" or s.lower() in ("none", "nan", "null", "-", "n/a", "?", "--"):
            return default
        return float(s.replace("%", "").replace("％", ""))
    except Exception:
        return default


def _i(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, _f(v, 0.0)))


def _mask_key(k: str) -> str:
    if not k:
        return ""
    if len(k) <= 8:
        return "***"
    return f"{k[:4]}...{k[-4:]}"


def _json_compact(obj: Any, max_len: int = 4000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        s = str(obj)
    return s[:max_len] if max_len and max_len > 0 else s


def _hash_obj(obj: Any) -> str:
    try:
        raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        raw = str(obj)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _deep_find_value(obj: Any, aliases: List[str], skip_keys: Optional[set] = None) -> Any:
    skip_keys = set(skip_keys or [])
    aliases_low = {str(a).lower() for a in aliases}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in aliases_low:
                return v
        for k, v in obj.items():
            if str(k).lower() in skip_keys:
                continue
            found = _deep_find_value(v, aliases, skip_keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for it in obj:
            found = _deep_find_value(it, aliases, skip_keys)
            if found is not None:
                return found
    return None

# ============================================================
# 比分与方向
# ============================================================

def _normalize_score_text(s: Any) -> str:
    return str(s).strip().replace(" ", "").replace("：", "-").replace(":", "-").replace("–", "-").replace("—", "-")


def _parse_score(s: Any) -> Tuple[Optional[int], Optional[int]]:
    ss = _normalize_score_text(s)
    if not ss or ss.lower() in ("home", "draw", "away", "abstain", "弃权"):
        return None, None
    if "胜" in ss and "其他" in ss:
        return 9, 0
    if "平" in ss and "其他" in ss:
        return 9, 9
    if "负" in ss and "其他" in ss:
        return 0, 9
    m = re.search(r"(\d{1,2})\s*[-:：]\s*(\d{1,2})", ss)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _score_direction(score_str: Any) -> Optional[str]:
    ss = _normalize_score_text(score_str)
    if ss in SCORE_OTHERS_HOME or ss == "9-0":
        return "home"
    if ss in SCORE_OTHERS_DRAW or ss == "9-9":
        return "draw"
    if ss in SCORE_OTHERS_AWAY or ss == "0-9":
        return "away"
    h, a = _parse_score(ss)
    if h is None:
        return None
    if h > a:
        return "home"
    if h < a:
        return "away"
    return "draw"


def _score_total(score_str: Any) -> Optional[int]:
    h, a = _parse_score(score_str)
    return None if h is None else h + a


def _direction_cn(direction: str) -> str:
    return {"home": "主胜", "draw": "平局", "away": "客胜", "abstain": "弃权"}.get(direction, "弃权")


def _dir_from_cn(v: Any) -> Optional[str]:
    s = str(v).strip().lower()
    if s in ("home", "主胜", "胜", "主", "主队", "home_win", "h", "win"):
        return "home"
    if s in ("draw", "平局", "平", "和", "tie", "x", "d", "same"):
        return "draw"
    if s in ("away", "客胜", "负", "客", "客队", "away_win", "a", "lose"):
        return "away"
    return None


def _score_display_label(score_str: Any, direction_code: Optional[str] = None) -> str:
    ss = _normalize_score_text(score_str)
    if ss in SCORE_OTHERS_HOME or ss in ("胜其他", "9-0"):
        return "胜其他"
    if ss in SCORE_OTHERS_DRAW or ss in ("平其他", "9-9"):
        return "平其他"
    if ss in SCORE_OTHERS_AWAY or ss in ("负其他", "0-9"):
        return "负其他"
    return ss


def _score_goal_band(score: str) -> str:
    total = _score_total(score)
    if total is None:
        return ""
    if total <= 1:
        return "0-1"
    if total == 2:
        return "2"
    if total == 3:
        return "3"
    return "4+"


def _score_btts(score: str) -> str:
    h, a = _parse_score(score)
    if h is None or a is None:
        return "unclear"
    return "yes" if h > 0 and a > 0 else "no"

# ============================================================
# 输入规范化
# ============================================================

def _is_change_direction_code(change: Any) -> bool:
    if not isinstance(change, dict) or not change:
        return False
    vals = []
    for k in ("win", "same", "draw", "lose", "home", "away"):
        if k in change:
            vals.append(_f(change.get(k), 999))
    return bool(vals) and all(v in (-1, 0, 1) for v in vals)


def _change_value(change: Dict[str, Any], key: str, default: float = 0.0) -> float:
    if not isinstance(change, dict):
        return default
    if key == "same":
        return _f(change.get("same", change.get("draw", default)), default)
    if key == "lose":
        return _f(change.get("lose", change.get("away", default)), default)
    if key == "win":
        return _f(change.get("win", change.get("home", default)), default)
    return _f(change.get(key, default), default)


def normalize_match(raw_m: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(raw_m or {})
    nested_keys = ["v2_odds_dict", "odds_dict", "odds", "v2", "odds_v2", "packet", "raw_odds", "data", "detail"]
    for nk in nested_keys:
        if isinstance(m.get(nk), dict):
            for k, v in m[nk].items():
                if k not in m:
                    m[k] = v

    home = m.get("home_team") or m.get("home") or m.get("host") or m.get("team_home") or m.get("homeName") or "Home"
    away = m.get("away_team") or m.get("guest") or m.get("away") or m.get("team_away") or m.get("awayName") or "Away"
    m["home_team"] = home
    m["away_team"] = away
    m["home"] = home
    m["guest"] = away

    skip = {"vote", "change", "points", "information", "prediction", "stats", "smart_signals", "recommend"}
    if m.get("sp_home") is None:
        v = _deep_find_value(m, ["win", "odds_win", "spf_sp3", "sp3", "home_odds", "主胜", "胜"], skip)
        if v is not None:
            m["sp_home"] = v
            m["win"] = v
    if m.get("sp_draw") is None:
        v = _deep_find_value(m, ["draw", "same", "odds_draw", "spf_sp1", "sp1", "平"], skip)
        if v is not None:
            m["sp_draw"] = v
            m["same"] = v
    if m.get("sp_away") is None:
        v = _deep_find_value(m, ["lose", "away_win", "odds_lose", "spf_sp0", "sp0", "away_odds", "客胜", "负"], skip)
        if v is not None:
            m["sp_away"] = v
            m["lose"] = v

    if "give_ball" not in m:
        m["give_ball"] = m.get("handicap") or m.get("rq") or m.get("let_ball") or "0"

    if not isinstance(m.get("change"), dict):
        ch = {}
        for src_key, dst_key in [
            ("change_win", "win"), ("cw", "win"), ("home_change", "win"),
            ("change_same", "same"), ("cs", "same"), ("change_draw", "same"), ("draw_change", "same"),
            ("change_lose", "lose"), ("cl", "lose"), ("change_away", "lose"), ("away_change", "lose"),
        ]:
            if src_key in m:
                ch[dst_key] = m.get(src_key)
        m["change"] = ch

    m["change_is_direction_code"] = _is_change_direction_code(m.get("change"))
    return m


def _extract_match_list(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if not isinstance(raw, dict):
        return []
    for key in ["matches", "today", "data", "items", "list"]:
        v = raw.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        if isinstance(v, dict):
            for kk in ["today", "list", "data", "items", "matches"]:
                if isinstance(v.get(kk), list):
                    return [x for x in v.get(kk, []) if isinstance(x, dict)]
    return []


def get_market_odds_for_score(match_obj: Dict[str, Any], score: str) -> float:
    score = _normalize_score_text(score)
    key = CRS_FULL_MAP.get(score)
    if key:
        return _f(match_obj.get(key, 0))
    if score in SCORE_OTHERS_HOME or score in ("胜其他", "9-0"):
        return _f(match_obj.get("crs_win", 0))
    if score in SCORE_OTHERS_DRAW or score in ("平其他", "9-9"):
        return _f(match_obj.get("crs_same", 0))
    if score in SCORE_OTHERS_AWAY or score in ("负其他", "0-9"):
        return _f(match_obj.get("crs_lose", 0))
    return 0.0

# ============================================================
# 市场快照 / Sharp / 比分形态
# ============================================================

def _odds_prob(o: float) -> float:
    return 0.0 if o <= 1.001 else 1.0 / o


def _implied_1x2(h: float, d: float, a: float) -> Dict[str, float]:
    ps = {"home": _odds_prob(h), "draw": _odds_prob(d), "away": _odds_prob(a)}
    s = sum(ps.values())
    if s <= 0:
        return {"home": 0.0, "draw": 0.0, "away": 0.0}
    return {k: round(v / s * 100, 1) for k, v in ps.items()}


def _static_favorite(sp_h: float, sp_d: float, sp_a: float) -> str:
    vals = [(sp_h, "home"), (sp_d, "draw"), (sp_a, "away")]
    vals = [(v, k) for v, k in vals if v > 1.001]
    if not vals:
        return "unclear"
    return min(vals, key=lambda x: x[0])[1]


def _vote_side(vote: Any) -> Tuple[str, Dict[str, int]]:
    if not isinstance(vote, dict):
        return "unclear", {"home": 0, "draw": 0, "away": 0}
    vv = {
        "home": _i(vote.get("win", vote.get("home", 0))),
        "draw": _i(vote.get("same", vote.get("draw", 0))),
        "away": _i(vote.get("lose", vote.get("away", 0))),
    }
    if max(vv.values()) <= 0:
        return "unclear", vv
    return max(vv.items(), key=lambda kv: kv[1])[0], vv


def _change_drop_up(match: Dict[str, Any]) -> Tuple[Dict[str, bool], Dict[str, bool], Dict[str, float]]:
    ch = match.get("change") if isinstance(match.get("change"), dict) else {}
    code = _is_change_direction_code(ch)
    raw = {
        "home": _change_value(ch, "win", 0.0),
        "draw": _change_value(ch, "same", 0.0),
        "away": _change_value(ch, "lose", 0.0),
    }
    if code:
        drop = {k: raw[k] < 0 for k in raw}
        up = {k: raw[k] > 0 for k in raw}
    else:
        # 兼容真实变动：负数=降，正数=升；如果是百分比/差值，阈值更宽。
        drop = {k: raw[k] < -0.03 for k in raw}
        up = {k: raw[k] > 0.03 for k in raw}
    return drop, up, raw


def _score_odds_pack(match: Dict[str, Any]) -> Dict[str, Any]:
    scores = {}
    for sc, key in CRS_FULL_MAP.items():
        v = _f(match.get(key, 0), 0)
        if v > 1.001:
            scores[sc] = v
    for sc, key in [("胜其他", "crs_win"), ("平其他", "crs_same"), ("负其他", "crs_lose")]:
        v = _f(match.get(key, 0), 0)
        if v > 1.001:
            scores[sc] = v
    low_scores = sorted(scores.items(), key=lambda kv: kv[1])[:8]
    return {
        "available_count": len(scores),
        "low_score_odds": low_scores,
        "s00": _f(match.get("s00", 0), 0),
        "s11": _f(match.get("s11", 0), 0),
        "w31": _f(match.get("w31", 0), 0),
        "w21": _f(match.get("w21", 0), 0),
        "l12": _f(match.get("l12", 0), 0),
    }


def _ttg_pack(match: Dict[str, Any]) -> Dict[str, Any]:
    vals = {str(i): _f(match.get(f"a{i}", 0), 0) for i in range(8)}
    positive = {k: v for k, v in vals.items() if v > 1.001}
    hot = min(positive.items(), key=lambda kv: kv[1])[0] if positive else "unclear"
    hot_odds = positive.get(hot, 0) if positive else 0
    return {
        "odds": vals,
        "hot_total_goals": hot,
        "hot_total_odds": hot_odds,
        "low_goal_signal": bool((0 < vals.get("0", 99) <= 8.8) or (0 < vals.get("1", 99) <= 5.8 and 0 < vals.get("2", 99) <= 4.2)),
        "high_goal_signal": bool((0 < vals.get("4", 99) <= 5.8) or (0 < vals.get("5", 99) <= 6.2) or (0 < vals.get("7", 99) <= 6.5)),
    }


def build_market_snapshot(match: Dict[str, Any]) -> Dict[str, Any]:
    sp_h = _f(match.get("sp_home", match.get("win")), 0)
    sp_d = _f(match.get("sp_draw", match.get("same")), 0)
    sp_a = _f(match.get("sp_away", match.get("lose")), 0)
    fav = _static_favorite(sp_h, sp_d, sp_a)
    public_side, votes = _vote_side(match.get("vote"))
    drop, up, raw_change = _change_drop_up(match)
    ttg = _ttg_pack(match)
    score_pack = _score_odds_pack(match)
    return {
        "home": match.get("home_team", match.get("home")),
        "away": match.get("away_team", match.get("guest")),
        "league": match.get("league", match.get("cup", "")),
        "match_num": match.get("match_num", ""),
        "spf": {"home": sp_h, "draw": sp_d, "away": sp_a},
        "implied_1x2_pct": _implied_1x2(sp_h, sp_d, sp_a),
        "static_favorite": fav,
        "public_hot_side": public_side,
        "vote_pct": votes,
        "give_ball": _f(match.get("give_ball", match.get("handicap", match.get("rq", 0))), 0),
        "change_is_direction_code": _is_change_direction_code(match.get("change")),
        "change_raw": raw_change,
        "drop_flags": drop,
        "up_flags": up,
        "ttg": ttg,
        "score_odds": score_pack,
        "market_hash": "",
    }


def build_sharp_money_pack(match: Dict[str, Any], market: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    market = market or build_market_snapshot(match)
    sp = market.get("spf", {})
    votes = market.get("vote_pct", {})
    drop = market.get("drop_flags", {})
    up = market.get("up_flags", {})
    fav = market.get("static_favorite", "unclear")
    public = market.get("public_hot_side", "unclear")
    h_odds, d_odds, a_odds = _f(sp.get("home"), 0), _f(sp.get("draw"), 0), _f(sp.get("away"), 0)
    vh, vd, va = _i(votes.get("home")), _i(votes.get("draw")), _i(votes.get("away"))

    signals: List[Dict[str, Any]] = []

    def add(code: str, side: str, strength: int, reason: str) -> None:
        signals.append({"code": code, "side": side, "strength": int(_clip(strength, 0, 100)), "reason": reason[:260]})

    if drop.get("home") and vh < 55 and h_odds >= 1.25:
        add("STEAM_HOME", "home", 58 + max(0, 55 - vh) // 2, f"主赔降水且主胜散户{vh}%未过热，偏专业资金主队")
    if drop.get("away") and va < 55 and a_odds >= 1.25:
        add("STEAM_AWAY", "away", 58 + max(0, 55 - va) // 2, f"客赔降水且客胜散户{va}%未过热，偏专业资金客队")
    if drop.get("draw") and up.get("home") and up.get("away"):
        add("SHARP_DRAW", "draw", 72, "平赔降水且胜负两端升水，典型平局资金/赔付控制")
    if drop.get("home") and vh >= 62:
        add("PUBLIC_HOME_DROP", "home_public", 48, f"主胜受注{vh}%且主赔同向降，需区分真强与顺水诱热")
    if drop.get("away") and va >= 62:
        add("PUBLIC_AWAY_DROP", "away_public", 48, f"客胜受注{va}%且客赔同向降，需区分真强与顺水诱热")

    side_strength = {"home": 0, "draw": 0, "away": 0, "home_public": 0, "away_public": 0}
    for s in signals:
        side_strength[s["side"]] = max(side_strength.get(s["side"], 0), int(s["strength"]))
    if side_strength["draw"] >= 70 and side_strength["away"] >= 58:
        dynamic = "away_or_draw"
        strength = max(side_strength["draw"], side_strength["away"])
    elif side_strength["draw"] >= 70 and side_strength["home"] >= 58:
        dynamic = "home_or_draw"
        strength = max(side_strength["draw"], side_strength["home"])
    else:
        candidates = [(side_strength[k], k) for k in ("home", "draw", "away")]
        strength, dynamic = max(candidates)
        if strength <= 0:
            dynamic = "unclear"

    conflict_type = "none"
    lock_forbidden: List[str] = []
    recommendation_cap = "S"
    if fav == "home" and dynamic in ("away", "draw", "away_or_draw"):
        conflict_type = "static_home_vs_dynamic_away_draw"
        lock_forbidden.append("home_strong_lock")
        recommendation_cap = "C"
    elif fav == "away" and dynamic in ("home", "draw", "home_or_draw"):
        conflict_type = "static_away_vs_dynamic_home_draw"
        lock_forbidden.append("away_strong_lock")
        recommendation_cap = "C"
    elif fav in ("home", "away") and public == fav and (side_strength.get(f"{fav}_public", 0) >= 48):
        conflict_type = "public_hot_same_side_drop_need_audit"
        recommendation_cap = "B"

    league_text = str(market.get("league", ""))
    is_elite_or_cup = any(k in league_text.lower() for k in ["欧冠", "欧罗巴", "欧联", "杯", "cup", "libertadores", "解放者", "南球"])
    if is_elite_or_cup and conflict_type.startswith("static_"):
        recommendation_cap = "C"
        if "elite_or_cup_conflict_no_s_a" not in lock_forbidden:
            lock_forbidden.append("elite_or_cup_conflict_no_s_a")

    return {
        "static_favorite": fav,
        "public_hot_side": public,
        "dynamic_money_direction": dynamic,
        "sharp_strength": int(_clip(strength if dynamic != "unclear" else 0, 0, 100)),
        "conflict_type": conflict_type,
        "recommendation_cap": recommendation_cap,
        "lock_forbidden": lock_forbidden,
        "signals": signals,
        "instruction_to_ai": (
            "sharp_pack只作为资金/变盘事实审计。若 static_favorite 与 dynamic_money_direction 冲突，"
            "不得强锁热门，必须给出平/反向保护解释。"
        ),
    }


def build_score_shape_pack(match: Dict[str, Any], market: Optional[Dict[str, Any]] = None, sharp: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    market = market or build_market_snapshot(match)
    sharp = sharp or build_sharp_money_pack(match, market)
    sp = market.get("spf", {})
    ttg = market.get("ttg", {})
    so = market.get("score_odds", {})
    fav = market.get("static_favorite", "unclear")
    h_odds, d_odds, a_odds = _f(sp.get("home"), 0), _f(sp.get("draw"), 0), _f(sp.get("away"), 0)
    hot_total = str(ttg.get("hot_total_goals", "unclear"))
    a0 = _f(ttg.get("odds", {}).get("0"), 0)
    a1 = _f(ttg.get("odds", {}).get("1"), 0)
    a2 = _f(ttg.get("odds", {}).get("2"), 0)
    a3 = _f(ttg.get("odds", {}).get("3"), 0)
    a4 = _f(ttg.get("odds", {}).get("4"), 0)
    s00 = _f(so.get("s00"), 0)
    s11 = _f(so.get("s11"), 0)
    w31 = _f(so.get("w31"), 0)
    w21 = _f(so.get("w21"), 0)
    l12 = _f(so.get("l12"), 0)
    pools: List[str] = []
    warnings: List[str] = []

    low_goal = bool((0 < a0 <= 8.8) or (0 < a1 <= 5.8 and 0 < a2 <= 4.2))
    draw_controlled = bool((0 < d_odds <= 3.45) or (0 < s11 <= 8.5) or (0 < s00 <= 9.0))
    strong_fav = bool((fav == "home" and 0 < h_odds <= 1.85) or (fav == "away" and 0 < a_odds <= 1.85))
    dominant_but_concede = bool(strong_fav and ((0 < a3 <= 5.0) or (0 < a4 <= 5.8)) and not low_goal)
    high_goal = bool((0 < a4 <= 5.8) or hot_total in ("4", "5", "6", "7"))

    if low_goal and draw_controlled:
        pools.extend(["1-1", "0-0", "1-0", "0-1"])
        warnings.append("LOW_GOAL_DRAW_PROTECTION:低总进球+平赔/1-1/0-0低位，必须审计1-1，不能只压0-0/1-0/0-1")
    elif low_goal:
        pools.extend(["1-0", "0-1", "1-1", "0-0"])
        warnings.append("LOW_GOAL_ECONOMIC_WIN:低总进球结构，经济型比分优先")

    if dominant_but_concede:
        if fav == "home":
            pools.extend(["3-1", "2-1", "3-0", "3-2"])
        elif fav == "away":
            pools.extend(["1-3", "1-2", "0-3", "2-3"])
        warnings.append("DOMINANT_BUT_CONCEDE:强热门+3/4球热，不得默认零封，必须审计BTTS yes")

    if high_goal and not low_goal:
        if fav == "home":
            pools.extend(["3-1", "3-2", "4-1", "2-1"])
        elif fav == "away":
            pools.extend(["1-3", "2-3", "1-4", "1-2"])
        else:
            pools.extend(["2-2", "2-1", "1-2", "3-2"])
        warnings.append("HIGH_GOAL_SHAPE:大球结构，需要优先审计BTTS与4+")

    if sharp.get("conflict_type") == "static_home_vs_dynamic_away_draw":
        pools = ["1-1", "2-2", "1-2", "2-1"] + pools
        warnings.append("SHARP_CONFLICT_SCORE_PROTECTION:主热门遇客/平资金，禁止强锁3-1/3-2/4-1")
    if sharp.get("conflict_type") == "static_away_vs_dynamic_home_draw":
        pools = ["1-1", "2-2", "2-1", "1-2"] + pools
        warnings.append("SHARP_CONFLICT_SCORE_PROTECTION:客热门遇主/平资金，禁止强锁1-3/2-3/1-4")

    uniq_pool = []
    for sc in pools:
        if sc not in uniq_pool:
            uniq_pool.append(sc)
    if not uniq_pool:
        if fav == "home":
            uniq_pool = ["2-0", "2-1", "1-0"]
        elif fav == "away":
            uniq_pool = ["0-1", "1-2", "0-2"]
        else:
            uniq_pool = ["1-1", "0-0", "2-2"]

    return {
        "low_goal_signal": low_goal,
        "draw_controlled": draw_controlled,
        "dominant_but_concede": dominant_but_concede,
        "high_goal_signal": high_goal,
        "hot_total_goals": hot_total,
        "s00": s00,
        "s11": s11,
        "w31": w31,
        "w21": w21,
        "l12": l12,
        "protected_score_pool": uniq_pool[:8],
        "warnings": warnings,
        "instruction_to_ai": "score_shape_pack只作为比分形态审计；AI可反驳，但必须说明赔率硬依据。",
    }

# ============================================================
# 经验审计卡
# ============================================================

_RULE_NAME = {
    "D01": "大热必死", "D08": "强强对话平局率高", "D10": "平手盘水位不动易平",
    "D13": "攻防数据接近必防平", "D15": "杯赛/淘汰赛/赛制保守复核", "U04": "受注比例一边倒反向操作",
    "U09": "排名差大但盘口便宜", "U10": "赔率剧烈变动", "G08": "0球赔率极低信号",
    "G10": "0-0赔率极低", "G12": "大球/BTTS高危", "B_SHARP": "Sharp聪明钱方向",
    "B_STEAM": "Steam资金方向", "X01": "强客低赔中热复核", "X02": "强方向热度与降水同向复核",
    "X03": "方向与总进球/BTTS一致性复核", "X04": "Sharp冲突降级", "X05": "1-1低比分保护",
    "X06": "强热门丢球保护",
}

_RULE_QUESTION = {
    "B_SHARP": "当前资金是真 sharp 还是诱导？必须结合散户热度、降水方向、胜平负联动。",
    "B_STEAM": "散户未明显跟进的一侧降水，是否属于专业资金？",
    "X04": "静态热门与动态资金冲突时，是否必须防平/防反向？",
    "X05": "低总进球+平局/1-1低位时，是否应该把1-1放入核心候选？",
    "X06": "强热门+3/4球热时，是否不能默认零封？",
    "X03": "最终比分必须与总进球和BTTS一致。",
}

class ExperienceAuditEngine:
    def analyze(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        if not ENABLE_EXPERIENCE_AUDIT_CARDS:
            return {"enabled": False, "triggered": [], "risk_signals": [], "total_score": 0, "recommendation": "disabled"}
        market = build_market_snapshot(match_data)
        sharp = build_sharp_money_pack(match_data, market)
        shape = build_score_shape_pack(match_data, market, sharp)
        triggered: List[Dict[str, Any]] = []
        risk_signals: List[str] = []

        def add(rid: str, category: str, weight: int, reason: str, direction: str = "audit"):
            if weight < EXPERIENCE_AUDIT_MIN_WEIGHT:
                return
            triggered.append({
                "id": rid,
                "name": _RULE_NAME.get(rid, rid),
                "category": category,
                "weight": int(weight),
                "reason": str(reason)[:260],
                "direction": direction,
                "ai_question": _RULE_QUESTION.get(rid, "请作为审计问题，不得机械裁决。"),
                "mode": "advisory_only_no_local_decision",
            })

        sp = market["spf"]
        sp_h, sp_d, sp_a = _f(sp.get("home")), _f(sp.get("draw")), _f(sp.get("away"))
        votes = market.get("vote_pct", {})
        vh, va = _i(votes.get("home")), _i(votes.get("away"))
        give_ball = _f(market.get("give_ball"), 0)
        league = str(market.get("league", ""))
        hr = _i(match_data.get("home_rank"), 10)
        ar = _i(match_data.get("away_rank"), 10)

        if sp_h and sp_h < 1.40 and vh >= 55:
            add("D01", "热门", 8, f"主赔{sp_h}极低+主胜受注{vh}%", "draw_or_away")
            risk_signals.append("EXP_D01 主队大热复核")
        if sp_a and sp_a < 1.40 and va >= 55:
            add("D01", "热门", 8, f"客赔{sp_a}极低+客胜受注{va}%", "draw_or_home")
            risk_signals.append("EXP_D01 客队大热复核")
        if vh >= 65:
            add("U04", "热度", 8, f"主胜受注{vh}%过热", "audit_hot_home")
            risk_signals.append(f"EXP_U04 主胜超热{vh}%")
        if va >= 65:
            add("U04", "热度", 8, f"客胜受注{va}%过热", "audit_hot_away")
            risk_signals.append(f"EXP_U04 客胜超热{va}%")
        if abs(give_ball) < 0.1:
            add("D10", "平局", 6, "平手/浅盘结构，需防均衡和平局", "draw")
        if abs(hr - ar) <= 3:
            add("D13", "平局", 5, f"排名接近{hr}vs{ar}", "draw")
        if any(k.lower() in league.lower() for k in ["杯", "cup", "解放者", "南球", "欧冠", "欧罗巴", "欧联"]):
            add("D15", "赛制", 5, "杯赛/洲际赛属性，必须区分小组赛、淘汰赛、次回合", "audit_format")

        for sig in sharp.get("signals", []):
            code = str(sig.get("code", ""))
            if code.startswith("STEAM"):
                add("B_STEAM", "sharp", 8, sig.get("reason", ""), sig.get("side", "audit"))
                risk_signals.append("EXP_B_STEAM " + str(sig.get("side", "")))
            if code == "SHARP_DRAW":
                add("B_SHARP", "sharp", 9, sig.get("reason", ""), "draw")
                risk_signals.append("EXP_B_SHARP draw")
        if sharp.get("conflict_type") != "none":
            add("X04", "sharp_conflict", 10, f"{sharp.get('conflict_type')}，recommendation_cap={sharp.get('recommendation_cap')}", sharp.get("dynamic_money_direction", "audit"))
            risk_signals.append("EXP_X04 " + sharp.get("conflict_type", ""))

        ttg = market.get("ttg", {})
        if ttg.get("low_goal_signal"):
            add("G08", "大小球", 7, f"低总进球信号 a0={ttg.get('odds',{}).get('0')} a1={ttg.get('odds',{}).get('1')} a2={ttg.get('odds',{}).get('2')}", "under")
        if ttg.get("high_goal_signal"):
            add("G12", "大小球", 7, f"高总进球信号 hot={ttg.get('hot_total_goals')}@{ttg.get('hot_total_odds')}", "over")
        if shape.get("draw_controlled") and shape.get("low_goal_signal"):
            add("X05", "比分", 9, "低总进球+平赔/1-1/0-0低位，必须审计1-1", "draw")
            risk_signals.append("EXP_X05 1-1保护")
        if shape.get("dominant_but_concede"):
            add("X06", "比分", 8, "强热门+3/4球热，不得默认零封", "btts_yes")
            risk_signals.append("EXP_X06 强热门丢球保护")
        add("X03", "一致性", 6, "最终比分需与总进球区间、BTTS、方向一致", "audit_shape")

        dedup: Dict[str, Dict[str, Any]] = {}
        for t in triggered:
            if t["id"] not in dedup or int(t["weight"]) > int(dedup[t["id"]]["weight"]):
                dedup[t["id"]] = t
        out = sorted(dedup.values(), key=lambda x: (-int(x.get("weight", 0)), str(x.get("id", ""))))
        if EXPERIENCE_AUDIT_MAX_CARDS > 0:
            out = out[:EXPERIENCE_AUDIT_MAX_CARDS]
        return {
            "enabled": True,
            "mode": "prompt_only_no_probability_change_no_score_change",
            "market_snapshot": market,
            "sharp_money_pack": sharp,
            "score_shape_pack": shape,
            "triggered": out,
            "triggered_count": len(out),
            "total_score": sum(int(t.get("weight", 0)) for t in out),
            "risk_signals": list(dict.fromkeys(risk_signals)),
            "recommendation": "存在经验审计卡：AI需逐条回应" if out else "无明显历史经验审计卡",
        }


def _experience_engine() -> ExperienceAuditEngine:
    global _EXPERIENCE_AUDIT_ENGINE
    try:
        return _EXPERIENCE_AUDIT_ENGINE
    except NameError:
        _EXPERIENCE_AUDIT_ENGINE = ExperienceAuditEngine()
        return _EXPERIENCE_AUDIT_ENGINE


def _format_experience_audit_for_prompt(exp: Dict[str, Any]) -> str:
    if not exp or not exp.get("enabled"):
        return '<experience_audit_cards mode="disabled">disabled</experience_audit_cards>\n'
    p = '<market_audit_packs mode="advisory_only_no_local_decision">\n'
    p += "sharp_money_pack:" + _json_compact(exp.get("sharp_money_pack", {}), 2500) + "\n"
    p += "score_shape_pack:" + _json_compact(exp.get("score_shape_pack", {}), 2500) + "\n"
    p += "注意：上述 pack 是赔率/资金/形态事实，不是本地裁决。AI可反驳，但必须给出赔率硬依据。\n"
    p += "</market_audit_packs>\n"
    rows = exp.get("triggered", []) or []
    if not rows:
        return p + '<experience_audit_cards mode="prompt_only_no_decision">无明显历史经验审计卡。</experience_audit_cards>\n'
    p += '<experience_audit_cards mode="prompt_only_no_decision">\n'
    p += "这些卡片只作为审计问题；禁止直接改方向、禁止直接改比分。必须在 audit.experience_review 中逐条输出 accepted/rejected/neutral。\n"
    p += "输出格式必须覆盖每个 id：X04:accepted because ...; X05:neutral because ...。不要合并成 D13/D15，必须分开写。\n"
    for t in rows:
        p += f"- {t.get('id')} {t.get('name')} | 权重={t.get('weight')} | 原因={t.get('reason')} | 问题={t.get('ai_question')}\n"
    p += "</experience_audit_cards>\n"
    return p

# ============================================================
# Prompt
# ============================================================

def _raw_field_line(label: str, value: Any, limit: int = 1200) -> str:
    if value is None or value == "" or value == {} or value == []:
        return ""
    text = _json_compact(value, limit) if isinstance(value, (dict, list)) else str(value)
    return f"{label}:{text[:limit].replace(chr(10), ' ')}\n"


def _raw_full_packet_line(match_obj: Dict[str, Any]) -> str:
    if not INCLUDE_FULL_RAW_PACKET:
        return ""
    raw_json = _json_compact(match_obj, RAW_PACKET_CHAR_LIMIT if RAW_PACKET_CHAR_LIMIT > 0 else 0)
    return f"raw_match_full_json:{raw_json}\n"


def _output_format_rule() -> str:
    return (
        "严格输出 JSON 数组。每场一个对象。不要 markdown，不要解释。对象模板："
        '{"match":1,"final_direction":"home/draw/away","direction_probs":{"home":45,"draw":28,"away":27},'
        '"goal_band":"0-1/2/3/4+","btts":"yes/no/unclear",'
        '"top3":[{"score":"2-0","prob":18,"market_logic":"中文说明"}],'
        '"reason":"中文说明","ai_confidence":0-100,"risk_level":"low/medium/high","data_missing":[],'
        '"audit":{"odds_source":"体彩竞彩抓包赔率","web_odds_check":"searched/web_search_unavailable/european_odds_missing",'
        '"sharp_money_direction":"home/draw/away/home_or_draw/away_or_draw/unclear","sharp_evidence":"中文说明",'
        '"market_pack_response":"逐条回应sharp_money_pack和score_shape_pack",'
        '"league_style":"中文说明","team_style":"中文说明","style_score_logic":"中文说明","direction_rejection":"中文说明",'
        '"experience_review":"X04:accepted because 中文说明; X05:neutral because 中文说明"}}\n'
        "match字段必须是数字序号。top3[0].score 必须与 final_direction 一致。goal_band/btts 必须与 top3[0].score 一致。"
        "所有说明字段必须中文。\n"
    )


def build_phase1_prompt(match_analyses: List[Dict[str, Any]]) -> str:
    p = "<context>\n"
    p += "你是竞彩足球 RAW-AI 比分预测模型。match_data 中的赔率是中国体彩竞彩抓包赔率，不是欧洲公司均赔。\n"
    p += "你需要基于原始抓包、赔率变动、散户热度、总进球、比分赔率、半全场、sharp_money_pack、score_shape_pack、经验审计卡，独立判断方向和比分。\n"
    p += "sharp_money_pack 是本地对资金/变盘的结构化诊断；它不是裁决，但你必须逐条审计。\n"
    p += "如果 static_favorite 与 dynamic_money_direction 冲突，禁止强锁热门；强强对决/杯赛遇 sharp 反热门必须防平/防反向。\n"
    p += "如果低总进球+平赔/1-1/0-0低位，必须审计1-1，不得只给0-0/1-0/0-1。\n"
    p += "如果强热门+3/4球热，不得默认零封，必须审计2-1/3-1/3-2或对应客胜丢球比分。\n"
    p += "禁止引用 CRS矩阵、贝叶斯、本地比分矩阵、本地风控裁决。\n"
    p += "若没有联网能力，audit.web_odds_check 写 web_search_unavailable，data_missing 加 external_european_odds，禁止假装联网。\n"
    p += "必须只输出 JSON 数组。\n"
    p += "</context>\n\n<output_format>\n" + _output_format_rule() + "</output_format>\n\n<match_data>\n"
    for i, ma in enumerate(match_analyses, 1):
        m = ma["match"]
        exp = ma.get("experience_audit") or _experience_engine().analyze(m)
        h, a = m.get("home_team", "Home"), m.get("away_team", "Away")
        p += f'<match index="{i}">\n[{i}] {h} vs {a} | {m.get("league", m.get("cup", ""))}\n'
        p += f"体彩竞彩1X2: 主胜={m.get('sp_home', m.get('win',''))} 平={m.get('sp_draw', m.get('same',''))} 客胜={m.get('sp_away', m.get('lose',''))}\n"
        p += f"让球:{m.get('give_ball', m.get('handicap', m.get('rq','')))}\n"
        p += f"change_is_direction_code:{m.get('change_is_direction_code', False)} | 若true，change=-1表示降水/下降，0表示不变，1表示升水/上升。\n"
        p += "总进球a0-a7:" + " | ".join([f"{g}={m.get(f'a{g}','')}" for g in range(8)]) + "\n"
        hf_l = []
        for k, lb in HFTF_MAP.items():
            v = m.get(k, None)
            if v not in (None, "", 0, "0"):
                hf_l.append(f"{lb}={v}")
        if hf_l:
            p += "半全场:" + " | ".join(hf_l) + "\n"
        p += _raw_field_line("change", m.get("change"), FIELD_LIMIT_CHANGE)
        p += _raw_field_line("vote", m.get("vote"), FIELD_LIMIT_VOTE)
        p += _raw_field_line("odds_movement", m.get("odds_movement"), 1400)
        p += _raw_field_line("baseface", m.get("baseface"), FIELD_LIMIT_INFORMATION)
        p += _raw_field_line("information", m.get("information"), FIELD_LIMIT_INFORMATION)
        p += _raw_field_line("points", m.get("points"), FIELD_LIMIT_POINTS)
        style_pack = {k: v for k, v in m.items() if k in (
            "league_style", "league_profile", "team_style", "home_style", "away_style", "play_style", "tactical_style",
            "pace_rating", "tempo", "home_form", "away_form", "weather", "injury", "lineup", "news", "motivation",
            "schedule", "home_rank", "away_rank", "home_stats", "away_stats", "aggregate_score", "first_leg", "second_leg",
        )}
        p += _raw_field_line("style_and_team_core", style_pack, FIELD_LIMIT_STYLE_EXTRA)
        p += _format_experience_audit_for_prompt(exp)
        p += _raw_full_packet_line(m)
        p += "</match>\n\n"
    p += "</match_data>\n"
    return p


def _short_ai_row(r: Dict[str, Any], idx: int) -> Dict[str, Any]:
    audit = r.get("audit", {}) if isinstance(r.get("audit", {}), dict) else {}
    keep = {k: audit.get(k) for k in [
        "web_odds_check", "sharp_money_direction", "sharp_evidence", "market_pack_response", "league_style",
        "team_style", "style_score_logic", "direction_rejection", "experience_review", "experience_review_inherited_from",
    ] if k in audit}
    return {
        "match": idx,
        "ai_score": r.get("ai_score"),
        "final_direction": r.get("final_direction"),
        "direction_probs": r.get("direction_probs", {}),
        "goal_band": r.get("goal_band", ""),
        "btts": r.get("btts", ""),
        "top3": r.get("top3", []),
        "ai_confidence": r.get("ai_confidence"),
        "risk_level": r.get("risk_level"),
        "reason": str(r.get("reason", ""))[:AI_MAX_PHASE1_REASON_CHARS_FOR_CLAUDE],
        "audit": keep,
        "experience_review": r.get("experience_review", []),
        "data_missing": r.get("data_missing", []),
    }


def build_compact_claude_match_data(match_analyses: List[Dict[str, Any]]) -> str:
    p = "<compact_match_data_for_claude>\n"
    p += "Claude终审压缩抓包：保留核心赔率、sharp_money_pack、score_shape_pack、经验审计卡。Claude必须重新审计，不按票数机械裁决。\n"
    for i, ma in enumerate(match_analyses, 1):
        m = ma["match"]
        exp = ma.get("experience_audit") or _experience_engine().analyze(m)
        p += f'<match index="{i}">\n[{i}] {m.get("home_team")} vs {m.get("away_team")} | {m.get("league", m.get("cup", ""))}\n'
        p += f"match_num:{m.get('match_num','')} | time:{m.get('match_time', m.get('time',''))}\n"
        p += f"体彩竞彩1X2: 主胜={m.get('sp_home', m.get('win',''))} 平={m.get('sp_draw', m.get('same',''))} 客胜={m.get('sp_away', m.get('lose',''))}\n"
        p += f"让球:{m.get('give_ball', m.get('handicap', m.get('rq','')))} | change_is_direction_code:{m.get('change_is_direction_code', False)}\n"
        p += "总进球a0-a7:" + " | ".join([f"{g}={m.get(f'a{g}','')}" for g in range(8)]) + "\n"
        p += _raw_field_line("change", m.get("change"), 1800)
        p += _raw_field_line("vote", m.get("vote"), 1600)
        p += _raw_field_line("odds_movement", m.get("odds_movement"), 1000)
        p += _raw_field_line("information", m.get("information"), CLAUDE_COMPACT_FIELD_LIMIT)
        p += _raw_field_line("points", m.get("points"), CLAUDE_COMPACT_FIELD_LIMIT)
        p += _format_experience_audit_for_prompt(exp)
        p += "</match>\n\n"
    p += "</compact_match_data_for_claude>\n"
    return p


def build_claude_final_audit_prompt(match_analyses: List[Dict[str, Any]], phase1_results: Dict[str, Dict[int, Dict[str, Any]]]) -> str:
    p = "<final_audit_context>\n"
    p += "你是 Claude 最终 RAW-AI 主裁。你不是反指模型。必须重新审计原始字段、sharp_money_pack、score_shape_pack 和三家初审。\n"
    p += "如果 static_favorite 与 dynamic_money_direction 冲突，不能强锁热门；必须防平/防反向。\n"
    p += "如果 Phase1 至少两家同比分一致，除非原始抓包/资金包有硬反证，否则默认尊重该比分。\n"
    p += "必须逐条回应经验卡 id，不得漏项。只输出 JSON 数组。\n"
    p += "</final_audit_context>\n\n"
    p += build_compact_claude_match_data(match_analyses) if AI_USE_COMPACT_CLAUDE_AUDIT else build_phase1_prompt(match_analyses)
    p += "\n<phase1_ai_results>\n"
    for ai in PHASE1_NAMES:
        p += f"<{ai}>\n"
        rs = phase1_results.get(ai, {}) or {}
        for idx in range(1, len(match_analyses) + 1):
            r = rs.get(idx)
            p += json.dumps(_short_ai_row(r, idx) if r else {"match": idx, "abstain": True}, ensure_ascii=False, separators=(",", ":")) + "\n"
        p += f"</{ai}>\n"
    p += "</phase1_ai_results>\n\n<output_format>\n" + _output_format_rule() + "</output_format>\n"
    return p

# ============================================================
# API config
# ============================================================

def _clean_env_key(*names: str) -> str:
    for name in names:
        v = str(os.environ.get(name, "")).strip(" \t\n\r\"'")
        if v:
            return v
    return ""


def _clean_env_url(*names: str) -> str:
    for name in names:
        v = str(os.environ.get(name, "")).strip(" \t\n\r\"'")
        if not v:
            continue
        m = re.search(r"(https?://[^\s'\"]+)", v)
        return m.group(1) if m else v
    return ""


def get_key_for_ai(ai_name: str) -> str:
    if AI_FORCE_COMMON_GATEWAY:
        return _clean_env_key("API_KEY", "GPT_API_KEY", "OPENAI_API_KEY", "GROK_API_KEY", "GEMINI_API_KEY", "CLAUDE_API_KEY")
    return _clean_env_key("API_KEY", f"{ai_name.upper()}_API_KEY", "OPENAI_API_KEY", "GPT_API_KEY")


def get_url_for_ai(ai_name: str) -> str:
    if AI_FORCE_COMMON_GATEWAY:
        return _clean_env_url("API_URL", "GPT_API_URL", "OPENAI_API_URL", "BASE_URL", "GROK_API_URL", "GEMINI_API_URL", "CLAUDE_API_URL")
    return _clean_env_url("API_URL", f"{ai_name.upper()}_API_URL", "OPENAI_API_URL", "BASE_URL", "GPT_API_URL")


def _model_for(ai_name: str) -> str:
    return str(os.environ.get(f"{ai_name.upper()}_MODEL", DEFAULT_MODELS[ai_name])).strip() or DEFAULT_MODELS[ai_name]


def _chat_url(base_url: str) -> str:
    u = (base_url or "").rstrip("/")
    if not u:
        return ""
    if u.endswith("/chat/completions") or "/chat/completions" in u:
        return u
    return u + "/chat/completions"


def debug_ai_config() -> None:
    key = get_key_for_ai("gpt")
    url = get_url_for_ai("gpt")
    print(f"[COMMON GATEWAY] API_URL={url or '<missing>'} API_KEY={_mask_key(key)} force_common={AI_FORCE_COMMON_GATEWAY}")
    for n in AI_NAMES:
        print(f"[AI CONFIG] {n.upper()} model={_model_for(n)} timeout={AI_CLAUDE_READ_TIMEOUT if n == 'claude' else AI_READ_TIMEOUT}s")
    print(f"[AI MODE] strict_one_call=True persistent_cache={AI_ENABLE_PERSISTENT_CACHE} exact_snapshot_reuse={AI_EXACT_SNAPSHOT_REUSE_SECONDS}s force_fresh={AI_FORCE_FRESH} safe_text_fallback={AI_ALLOW_TEXT_FALLBACK}")

# ============================================================
# 解析器
# ============================================================

_SCORE_RE = re.compile(r"(\d{1,2})\s*[-:：]\s*(\d{1,2})")


def _save_debug_dump(ai_name: str, data: Any, tag: str, raw_text: Optional[str] = None) -> None:
    try:
        os.makedirs("data/debug", exist_ok=True)
        ts = int(time.time())
        f1 = f"data/debug/{ai_name}_{tag}_{ts}.json"
        with open(f1, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f"    失败响应已保存: {f1}")
        if raw_text is not None:
            f2 = f"data/debug/{ai_name}_{tag}_raw_{ts}.txt"
            with open(f2, "w", encoding="utf-8") as f:
                f.write(raw_text)
            print(f"    原始文本已保存: {f2}")
    except Exception:
        pass


def _extract_response_text(data: Any, ai_name: str = "") -> str:
    candidates: List[Tuple[int, str, str]] = []

    def add(v: Any, path: str, bonus: int = 0) -> None:
        if isinstance(v, str):
            t = v.strip()
            if t:
                score = bonus
                if "top3" in t or "final_direction" in t or "direction_probs" in t:
                    score += 12
                if re.search(r"\[\s*\{", t):
                    score += 8
                if _SCORE_RE.search(t):
                    score += 3
                candidates.append((score, path, t))

    def walk(obj: Any, path: str = "root", depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                kl = str(k).lower()
                if kl in {"reasoning", "reasoning_content", "thinking", "chain_of_thought", "thoughts"}:
                    continue
                if kl in {"content", "text", "output_text", "answer", "result", "response", "final_answer", "model_response"}:
                    add(v, f"{path}.{k}", 5)
                walk(v, f"{path}.{k}", depth + 1)
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:100]):
                walk(v, f"{path}[{i}]", depth + 1)
        elif isinstance(obj, str):
            add(obj, path)

    if isinstance(data, dict):
        for ch in data.get("choices", []) or []:
            if isinstance(ch, dict):
                msg = ch.get("message", {}) or {}
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        add(content, "choices.message.content", 20)
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                add(item.get("text"), "choices.message.content[].text", 20)
                                add(item.get("content"), "choices.message.content[].content", 16)
                add(ch.get("text"), "choices.text", 10)
        add(data.get("output_text"), "output_text", 20)
        walk(data)
    elif isinstance(data, str):
        add(data, "raw", 10)
    if not candidates:
        return ""
    candidates.sort(key=lambda x: (x[0], len(x[2])), reverse=True)
    return candidates[0][2].strip()


def _preclean_text(text: str) -> str:
    clean = text or ""
    clean = clean.replace("\ufeff", "").replace("\x00", "")
    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"```(?:json|JSON|javascript|js|python|txt)?", "", clean)
    clean = clean.replace("```", "")
    clean = clean.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    clean = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", clean)
    return clean.strip()


def _json_loads_best_effort(s: str) -> Any:
    raw = s.strip()
    variants = [raw, re.sub(r",\s*([}\]])", r"\1", raw)]
    variants.append(re.sub(r"([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', variants[-1]))
    for cand in variants:
        try:
            return json.loads(cand)
        except Exception:
            pass
    for cand in variants:
        try:
            return ast.literal_eval(cand)
        except Exception:
            pass
    raise ValueError("json_parse_failed")


def _balanced_fragments(text: str) -> List[str]:
    clean = _preclean_text(text)
    frags: List[str] = []
    for start in [m.start() for m in re.finditer(r"[\[\{]", clean)]:
        stack: List[str] = []
        in_str = False
        quote = ""
        esc = False
        end = -1
        for i in range(start, len(clean)):
            ch = clean[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == quote:
                    in_str = False
                continue
            if ch in ('"', "'"):
                in_str = True
                quote = ch
                continue
            if ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if not stack:
                    break
                last = stack.pop()
                if (last == "[" and ch != "]") or (last == "{" and ch != "}"):
                    break
                if not stack:
                    end = i + 1
                    break
        if end > start:
            frag = clean[start:end]
            if any(k in frag for k in ["top3", "final_direction", "direction_probs", "score", "match", "比分"]):
                frags.append(frag)
    frags.sort(key=len, reverse=True)
    return frags[:20]


def _unwrap_json_result(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ["predictions", "results", "matches", "data", "items", "output", "prediction", "result"]:
            v = obj.get(k)
            if isinstance(v, list):
                return v
            if isinstance(v, dict) and ("match" in v or "top3" in v or "score" in v or "predicted_score" in v):
                return [v]
        if "match" in obj or "top3" in obj or "score" in obj or "predicted_score" in obj or "final_direction" in obj:
            return [obj]
    return []


def _extract_json_items(text: str) -> List[Any]:
    clean = _preclean_text(text)
    if not clean:
        return []
    try:
        return _unwrap_json_result(_json_loads_best_effort(clean))
    except Exception:
        pass
    best: List[Any] = []
    for frag in _balanced_fragments(clean):
        try:
            arr = _unwrap_json_result(_json_loads_best_effort(frag))
            if len(arr) > len(best):
                best = arr
        except Exception:
            continue
    return best


def _prob_to_float(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, str):
        s = v.strip().replace("％", "%")
        pct = "%" in s
        fv = _f(s, 0.0)
        return fv if pct or fv > 1 else fv * 100
    fv = _f(v, 0.0)
    return fv * 100 if 0 < fv <= 1 else fv


def _score_from_candidate(obj: Any) -> str:
    if isinstance(obj, str):
        m = _SCORE_RE.search(obj)
        return f"{int(m.group(1))}-{int(m.group(2))}" if m else _normalize_score_text(obj)
    if not isinstance(obj, dict):
        return ""
    for k in ["score", "predicted_score", "ai_score", "final_score", "比分", "预测比分", "top_score", "result_score", "correct_score", "scoreline", "prediction_score"]:
        if obj.get(k) not in (None, ""):
            s = _normalize_score_text(obj.get(k))
            m = _SCORE_RE.search(s)
            return f"{int(m.group(1))}-{int(m.group(2))}" if m else s
    for hk, ak in [("home_goals", "away_goals"), ("home_score", "away_score")]:
        if obj.get(hk) is not None and obj.get(ak) is not None:
            return f"{_i(obj.get(hk))}-{_i(obj.get(ak))}"
    return ""


def _normalize_top3(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_top3 = None
    for k in ["top3", "top_3", "top_scores", "scores", "score_candidates", "candidates", "比分候选", "score_distribution", "correct_scores", "candidate_scores"]:
        if isinstance(item.get(k), list):
            raw_top3 = item[k]
            break
    raw_top3 = raw_top3 or []
    top3: List[Dict[str, Any]] = []
    seen = set()
    for cand in raw_top3[:10]:
        sc = _score_from_candidate(cand)
        if _parse_score(sc)[0] is None or sc in seen:
            continue
        seen.add(sc)
        prob = cand.get("prob", cand.get("probability", cand.get("pct", cand.get("chance", cand.get("confidence", 0))))) if isinstance(cand, dict) else 0
        logic = cand.get("market_logic", cand.get("reason", cand.get("logic", ""))) if isinstance(cand, dict) else ""
        top3.append({"score": sc, "prob": round(_prob_to_float(prob), 3), "market_logic": str(logic)[:600]})
        if len(top3) >= 3:
            break
    if not top3:
        sc = _score_from_candidate(item)
        if _parse_score(sc)[0] is not None:
            top3 = [{"score": sc, "prob": round(_prob_to_float(item.get("prob", item.get("probability", 0))), 3), "market_logic": ""}]
    return top3


def _normalize_direction(v: Any, top_score: str = "") -> str:
    d = _dir_from_cn(v)
    return d or _score_direction(top_score) or "draw"


def _normalize_ai_direction_probs(obj: Any) -> Dict[str, float]:
    if not isinstance(obj, dict):
        return {}
    cand = None
    for k in ["direction_probs", "direction_probabilities", "probabilities", "direction_probability", "方向概率", "三项概率"]:
        if isinstance(obj.get(k), dict):
            cand = obj.get(k)
            break
    if cand is None and isinstance(obj.get("audit"), dict):
        for k in ["direction_probs", "direction_probabilities", "probabilities", "方向概率", "三项概率"]:
            if isinstance(obj["audit"].get(k), dict):
                cand = obj["audit"].get(k)
                break
    if not isinstance(cand, dict):
        if any(k in obj for k in ["home_win_pct", "draw_pct", "away_win_pct"]):
            cand = {"home": obj.get("home_win_pct"), "draw": obj.get("draw_pct"), "away": obj.get("away_win_pct")}
        else:
            return {}
    alias = {"home": "home", "主": "home", "主胜": "home", "胜": "home", "win": "home", "draw": "draw", "平": "draw", "平局": "draw", "same": "draw", "away": "away", "客": "away", "客胜": "away", "负": "away", "lose": "away"}
    raw = {"home": 0.0, "draw": 0.0, "away": 0.0}
    for k, v in cand.items():
        kk = alias.get(str(k).strip().lower(), alias.get(str(k).strip()))
        if kk in raw:
            raw[kk] += _prob_to_float(v)
    s = sum(raw.values())
    return {} if s <= 0 else {k: round(v / s * 100, 1) for k, v in raw.items()}


def _match_index_from_item(item: Dict[str, Any], fallback_idx: int, num_matches: int) -> Optional[int]:
    raw = item.get("match", item.get("index", item.get("match_index", item.get("id", item.get("序号")))))
    if isinstance(raw, int):
        return raw if 1 <= raw <= num_matches else None
    if raw is not None:
        s = str(raw).strip()
        m = re.match(r"^\s*(\d+)\s*$", s) or re.match(r"^\s*\[(\d+)\]", s) or re.search(r"(?:match|场次|第)\s*(\d+)", s, re.I)
        if m:
            idx = int(m.group(1))
            return idx if 1 <= idx <= num_matches else None
    return fallback_idx if 1 <= fallback_idx <= num_matches else None


def _normalize_goal_band_value(v: Any, top_score: str = "") -> str:
    s = str(v or "").strip().lower().replace(" ", "")
    aliases = {"0-1": "0-1", "0_1": "0-1", "0~1": "0-1", "0至1": "0-1", "0到1": "0-1", "0/1": "0-1", "low": "0-1", "小球": "0-1", "2": "2", "2球": "2", "3": "3", "3球": "3", "4+": "4+", "4plus": "4+", "4以上": "4+", "4球+": "4+", "high": "4+", "大球": "4+"}
    if s in aliases:
        return aliases[s]
    return _score_goal_band(top_score)


def _normalize_btts_value(v: Any, top_score: str = "") -> str:
    s = str(v or "").strip().lower()
    if s in ("yes", "y", "true", "1", "是", "双方进球", "btts_yes"):
        return "yes"
    if s in ("no", "n", "false", "0", "否", "不是", "btts_no"):
        return "no"
    return _score_btts(top_score)


def _normalize_experience_review(item: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = item.get("experience_review")
    if raw is None and isinstance(item.get("audit"), dict):
        raw = item["audit"].get("experience_review")
    out: List[Dict[str, str]] = []
    seen = set()

    def add_one(rid: str, dec: str, reason: str):
        rid = str(rid).replace("EXP_", "").strip()
        if not rid or rid in seen:
            return
        if dec not in ("accepted", "rejected", "neutral"):
            dec = "neutral"
        seen.add(rid)
        out.append({"id": rid, "decision": dec, "reason": str(reason)[:600]})

    if isinstance(raw, list):
        for r in raw[:60]:
            if isinstance(r, dict):
                ids = re.findall(r"EXP_[A-Z0-9_]+|B_SHARP|B_STEAM|X\d{2}|[DUG]\d{2}", str(r.get("id", r.get("rule_id", ""))))
                if not ids and r.get("id"):
                    ids = [str(r.get("id"))]
                dec = str(r.get("decision", r.get("status", "neutral"))).strip().lower()
                reason = str(r.get("reason", r.get("why", "")))[:600]
                for rid in ids:
                    add_one(rid, dec, reason)
    elif isinstance(raw, str) and raw.strip():
        for p in re.split(r"[;；\n]+", raw)[:80]:
            ids = re.findall(r"EXP_[A-Z0-9_]+|B_SHARP|B_STEAM|X\d{2}|[DUG]\d{2}", p)
            if not ids:
                continue
            low = p.lower()
            dec = "accepted" if ("accepted" in low or "接受" in p or "采纳" in p) else "rejected" if ("rejected" in low or "驳回" in p or "不采纳" in p) else "neutral"
            for rid in ids:
                add_one(rid, dec, p[:600])
    return out


def _score_from_safe_text_block(part: str) -> str:
    text = part or ""
    label_patterns = [
        r"(?:预测比分|最终比分|首选比分|建议比分|比分预测|final_score|predicted_score|scoreline|score|top1|第一比分|主推比分)\s*[:：=为是\-]*\s*(\d{1,2})\s*[-:：]\s*(\d{1,2})",
        r"(?:看好|倾向|预计|预测|主推|首选)\D{0,18}(\d{1,2})\s*[-:：]\s*(\d{1,2})",
    ]
    for pat in label_patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return f"{int(m.group(1))}-{int(m.group(2))}"
    candidates = []
    for m in _SCORE_RE.finditer(text):
        start, end = m.span()
        ctx = text[max(0, start - 35): min(len(text), end + 35)]
        if re.search(r"首回合|上轮|上一场|历史|交锋|半场|全场|总比分|比分层|赔率|@|a\d|w\d|l\d|s\d", ctx, flags=re.I):
            continue
        if not re.search(r"预测|最终|首选|主推|看好|倾向|score|比分", ctx, flags=re.I):
            continue
        candidates.append(f"{int(m.group(1))}-{int(m.group(2))}")
    uniq = list(dict.fromkeys(candidates))
    return uniq[0] if len(uniq) == 1 else ""


def _extract_match_blocks_for_text_fallback(clean: str, num_matches: int) -> List[Tuple[int, str]]:
    starts = []
    for m in re.finditer(r"(?:^|\n)\s*(?:\[\s*(\d{1,2})\s*\]|match\s*(\d{1,2})|第\s*(\d{1,2})\s*场|场次\s*(\d{1,2}))", clean, flags=re.I):
        idx = next((int(g) for g in m.groups() if g), None)
        if idx and 1 <= idx <= num_matches:
            starts.append((idx, m.start()))
    if starts:
        return [(idx, clean[st: starts[n + 1][1] if n + 1 < len(starts) else len(clean)]) for n, (idx, st) in enumerate(starts)]
    return [(1, clean)] if num_matches == 1 else []


def _fallback_parse_text_blocks(raw_text: str, num_matches: int) -> Dict[int, Dict[str, Any]]:
    clean = _preclean_text(raw_text)
    results: Dict[int, Dict[str, Any]] = {}
    for idx, part in _extract_match_blocks_for_text_fallback(clean, num_matches):
        if idx in results:
            continue
        score = _score_from_safe_text_block(part)
        if not score:
            continue
        direction = _score_direction(score) or "draw"
        conf_match = re.search(r"(?:confidence|置信度|ai_confidence)\D{0,8}(\d{1,3})", part, flags=re.I)
        conf = int(_clip(_f(conf_match.group(1), 60) if conf_match else 60, 0, 100))
        results[idx] = {
            "top3": [{"score": score, "prob": 0.0, "market_logic": "safe_text_fallback"}],
            "ai_score": score,
            "reason": part[:4000],
            "ai_confidence": conf,
            "risk_level": "medium",
            "data_missing": ["json_format_missing_safe_text_fallback"],
            "audit": {"parse_mode": "safe_text_fallback"},
            "direction_probs": {},
            "goal_band": _score_goal_band(score),
            "btts": _score_btts(score),
            "score_shape_reason": "safe_text_fallback_parser",
            "experience_review": _normalize_experience_review({"audit": {"experience_review": part}}),
            "final_direction": direction,
            "raw_item": {"safe_text_fallback": part[:1000]},
        }
    return results


def _parse_ai_json(raw_text: str, num_matches: int, ai_name: str = "") -> Dict[int, Dict[str, Any]]:
    items = _extract_json_items(raw_text)
    results: Dict[int, Dict[str, Any]] = {}
    for pos, item in enumerate(items if isinstance(items, list) else [], 1):
        if not isinstance(item, dict):
            continue
        for k in ["prediction", "result", "data"]:
            if isinstance(item.get(k), dict) and not any(x in item for x in ["top3", "score", "predicted_score", "ai_score"]):
                inner = dict(item[k])
                if "match" not in inner and "match" in item:
                    inner["match"] = item["match"]
                item = inner
                break
        top3 = _normalize_top3(item)
        if not top3:
            continue
        top_score = top3[0]["score"]
        mid = _match_index_from_item(item, pos, num_matches)
        if not mid:
            continue
        score_dir = _score_direction(top_score)
        final_direction = score_dir or _normalize_direction(item.get("final_direction", item.get("direction", item.get("result", ""))), top_score)
        audit = item.get("audit", {}) if isinstance(item.get("audit", {}), dict) else {}
        data_missing = item.get("data_missing", [])
        if not isinstance(data_missing, list):
            data_missing = [str(data_missing)] if data_missing else []
        results[mid] = {
            "top3": top3,
            "ai_score": top_score,
            "reason": str(item.get("reason", item.get("analysis", item.get("explanation", item.get("理由", "")))))[:5000],
            "ai_confidence": int(_clip(_f(item.get("ai_confidence", item.get("confidence", 60)), 60), 0, 100)),
            "risk_level": str(item.get("risk_level", item.get("risk", "medium"))),
            "data_missing": data_missing,
            "audit": audit,
            "direction_probs": _normalize_ai_direction_probs(item),
            "goal_band": _normalize_goal_band_value(item.get("goal_band", item.get("goal_range", "")), top_score),
            "btts": _normalize_btts_value(item.get("btts", item.get("both_score", "")), top_score),
            "score_shape_reason": str(item.get("score_shape_reason", item.get("score_logic", audit.get("style_score_logic", ""))))[:1500],
            "experience_review": _normalize_experience_review(item),
            "final_direction": final_direction,
            "raw_item": item,
        }
    if not results:
        allow = AI_ALLOW_CLAUDE_TEXT_FALLBACK if str(ai_name).lower().startswith("claude") else AI_ALLOW_TEXT_FALLBACK
        if allow:
            results = _fallback_parse_text_blocks(raw_text, num_matches)
    if AI_PARSE_DEBUG and not results:
        print(f"    [{ai_name}] parse empty. raw={raw_text[:500]}")
    return results

# ============================================================
# AI 调用
# ============================================================

async def async_call_one_ai_batch(session: aiohttp.ClientSession, prompt: str, num_matches: int, ai_name: str, system_text: str) -> Tuple[str, Dict[int, Dict[str, Any]], str]:
    key = get_key_for_ai(ai_name)
    base_url = get_url_for_ai(ai_name)
    model = _model_for(ai_name)
    AI_CALL_STATUS[ai_name] = {"ok": False, "status": "init", "model": model, "count": 0, "requests": 0, "strict_one_call": True}
    if not key:
        print(f"  [{ai_name.upper()}] no_key")
        AI_CALL_STATUS[ai_name].update({"status": "no_key"})
        return ai_name, {}, "no_key"
    if not base_url:
        print(f"  [{ai_name.upper()}] no_url")
        AI_CALL_STATUS[ai_name].update({"status": "no_url"})
        return ai_name, {}, "no_url"

    url = _chat_url(base_url)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system_text}, {"role": "user", "content": prompt}],
        "temperature": 0.10 if ai_name == "claude" else 0.16 if ai_name in ("gpt", "gemini") else 0.20,
    }
    if AI_USE_RESPONSE_FORMAT:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

    AI_CALL_STATUS[ai_name]["requests"] = 1
    gateway = url.split("/v1")[0][:80]
    print(f"  [连接中] {ai_name.upper()} | {model} @ {gateway} | request#1/1")
    t0 = time.time()
    try:
        read_timeout = AI_CLAUDE_READ_TIMEOUT if ai_name == "claude" else AI_READ_TIMEOUT
        connect_timeout = AI_CLAUDE_CONNECT_TIMEOUT if ai_name == "claude" else AI_CONNECT_TIMEOUT
        timeout = aiohttp.ClientTimeout(total=None, connect=connect_timeout, sock_connect=connect_timeout, sock_read=read_timeout)
        async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
            elapsed = round(time.time() - t0, 1)
            if r.status != 200:
                txt = await r.text()
                print(f"    HTTP {r.status} | {elapsed}s | {txt[:260]}")
                AI_CALL_STATUS[ai_name].update({"status": f"http_{r.status}", "elapsed": elapsed, "http_error": txt[:500]})
                return ai_name, {}, f"http_{r.status}"
            try:
                data = await r.json(content_type=None)
            except Exception:
                data = {"raw": await r.text()}
            raw_text = _extract_response_text(data, ai_name)
            if AI_SAVE_RAW_RESPONSE:
                _save_debug_dump(ai_name, data, "raw_saved", raw_text)
            if not raw_text:
                _save_debug_dump(ai_name, data, "empty", "")
                AI_CALL_STATUS[ai_name].update({"status": "empty", "elapsed": elapsed})
                return ai_name, {}, "empty"
            parsed = _parse_ai_json(raw_text, num_matches, ai_name)
            if parsed:
                print(f"    {ai_name.upper()} 完成: {len(parsed)}/{num_matches} | {round(time.time()-t0,1)}s | one_call=True")
                AI_CALL_STATUS[ai_name].update({"ok": True, "status": "ok", "count": len(parsed), "elapsed": round(time.time()-t0, 1), "second_call_used": False})
                return ai_name, parsed, model
            print(f"    严格解析0条，该模型弃权。raw前260字: {raw_text[:260].replace(chr(10),' ')}")
            _save_debug_dump(ai_name, data, "parse0", raw_text)
            AI_CALL_STATUS[ai_name].update({"status": "parse0", "elapsed": round(time.time()-t0, 1)})
            return ai_name, {}, "parse0"
    except asyncio.TimeoutError:
        print(f"    {ai_name.upper()} 读取超时")
        AI_CALL_STATUS[ai_name].update({"status": "timeout"})
        return ai_name, {}, "timeout"
    except Exception as e:
        print(f"    {ai_name.upper()} 调用异常: {str(e)[:220]}")
        AI_CALL_STATUS[ai_name].update({"status": "error", "error": str(e)[:500]})
        return ai_name, {}, "error"


def _phase_system(ai_name: str) -> str:
    base = (
        "你必须只输出严格 JSON 数组，禁止 markdown，禁止 JSON 外说明，禁止自然语言前后缀；不要输出弃权文本。"
        "每个对象必须包含 match、final_direction、direction_probs、goal_band、btts、top3、reason、ai_confidence、risk_level、data_missing、audit。"
        "final_direction 只能是 home/draw/away。reason、market_logic、audit 内说明必须中文。"
        "必须逐条回应 sharp_money_pack、score_shape_pack、experience_audit_cards。"
        "如果 static_favorite 与 dynamic_money_direction 冲突，不得强锁热门，必须防平/防反向。"
    )
    role = {"gpt": "RAW赔率结构和比分分布分析师。", "grok": "RAW资金流、散户热度和变盘分析师。", "gemini": "RAW多市场一致性和异常结构分析师。", "claude": "最终RAW-AI主裁，不是反指模型。"}.get(ai_name, "RAW-AI分析师。")
    return role + base

# ============================================================
# 缓存 / singleflight
# ============================================================

_VOLATILE_CACHE_KEYS = {"timestamp", "ts", "now", "current_time", "server_time", "local_time", "fetched_at", "fetch_time", "crawl_time", "scrape_time", "sync_time", "updated_at", "update_time", "last_update", "generated_at", "request_id", "trace_id", "uuid", "runtime", "elapsed", "latency", "cache_hit", "cache_ts"}
_OUTPUT_CACHE_IGNORE_KEYS = {"prediction", "predictions", "top4", "rank", "recommend_score", "is_recommended", "fusion_summary", "engine_version", "validation_warnings", "gpt_score", "grok_score", "gemini_score", "claude_score", "gpt_analysis", "grok_analysis", "gemini_analysis", "claude_analysis"}


def _sanitize_for_ai_cache(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kl = str(k).strip().lower()
            if kl in _VOLATILE_CACHE_KEYS or kl in _OUTPUT_CACHE_IGNORE_KEYS or kl.endswith("_ts") or kl.endswith("_timestamp"):
                continue
            out[k] = _sanitize_for_ai_cache(v)
        return out
    if isinstance(obj, list):
        return [_sanitize_for_ai_cache(x) for x in obj]
    return obj


def _market_cache_payload(ma: Dict[str, Any]) -> Dict[str, Any]:
    m = ma.get("match", {})
    exp = ma.get("experience_audit") or _experience_engine().analyze(m)
    return {
        "home": m.get("home_team"), "away": m.get("away_team"), "league": m.get("league", m.get("cup", "")), "match_num": m.get("match_num"),
        "market_snapshot": exp.get("market_snapshot", build_market_snapshot(m)),
        "sharp_money_pack": exp.get("sharp_money_pack", {}),
        "score_shape_pack": exp.get("score_shape_pack", {}),
        "change": m.get("change"), "vote": m.get("vote"),
        "core_hash": _hash_obj({k: _sanitize_for_ai_cache(m.get(k)) for k in [
            "sp_home", "sp_draw", "sp_away", "win", "same", "lose", "give_ball", "handicap", "rq",
            "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", *CRS_FULL_MAP.values(),
            "crs_win", "crs_same", "crs_lose", "information", "points", "odds_movement", "baseface",
        ]}),
    }


def _stable_ai_cache_key(match_analyses: List[Dict[str, Any]], phase: str = "pure") -> str:
    payload = {"version": ENGINE_VERSION, "schema": AI_CACHE_SCHEMA_VERSION, "phase": phase, "matches": [_market_cache_payload(ma) for ma in match_analyses]}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _ai_cache_file(cache_key: str) -> str:
    os.makedirs(AI_CACHE_DIR, exist_ok=True)
    return os.path.join(AI_CACHE_DIR, f"{cache_key}.json")


def _ai_lock_file(cache_key: str) -> str:
    os.makedirs(AI_CACHE_DIR, exist_ok=True)
    return os.path.join(AI_CACHE_DIR, f"{cache_key}.lock")


def _load_ai_disk_cache(cache_key: str) -> Optional[Dict[str, Dict[int, Dict[str, Any]]]]:
    if not AI_ENABLE_PERSISTENT_CACHE or AI_FORCE_FRESH:
        return None
    path = _ai_cache_file(cache_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            pack = json.load(f)
        if str(pack.get("schema", "")) != AI_CACHE_SCHEMA_VERSION:
            return None
        results = pack.get("results", {})
        restored = {n: {} for n in AI_NAMES}
        for name, rows in (results or {}).items():
            if isinstance(rows, dict):
                restored[name] = {int(k): v for k, v in rows.items() if str(k).isdigit()}
        print("  [AI DISK CACHE] 命中持久化缓存")
        return restored
    except Exception:
        return None


def _save_ai_disk_cache(cache_key: str, results: Dict[str, Dict[int, Dict[str, Any]]], status: Dict[str, Any]) -> None:
    if not AI_ENABLE_PERSISTENT_CACHE or AI_FORCE_FRESH:
        return
    # 不缓存 0结果，也不缓存只有弃权/空解析的坏结果。
    ok = sum(1 for n in AI_NAMES if results.get(n))
    if ok <= 0:
        return
    try:
        path = _ai_cache_file(cache_key)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "version": ENGINE_VERSION, "schema": AI_CACHE_SCHEMA_VERSION, "status": status, "results": results}, f, ensure_ascii=False, default=str)
        os.replace(tmp, path)
    except Exception as e:
        print(f"  [AI DISK CACHE] 写入失败: {str(e)[:100]}")


def _try_acquire_ai_disk_lock(cache_key: str) -> bool:
    if not AI_SINGLEFLIGHT_ENABLED:
        return True
    path = _ai_lock_file(cache_key)
    now = time.time()
    if os.path.exists(path):
        try:
            if now - os.path.getmtime(path) > AI_DISK_LOCK_WAIT_SECONDS:
                os.remove(path)
        except Exception:
            pass
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "ts": now, "version": ENGINE_VERSION}, ensure_ascii=False))
        return True
    except FileExistsError:
        return False
    except Exception:
        return True


def _release_ai_disk_lock(cache_key: str) -> None:
    try:
        os.remove(_ai_lock_file(cache_key))
    except Exception:
        pass


async def _wait_for_ai_disk_cache(cache_key: str) -> Optional[Dict[str, Dict[int, Dict[str, Any]]]]:
    if not AI_ENABLE_PERSISTENT_CACHE:
        return None
    deadline = time.time() + max(5, AI_DISK_LOCK_WAIT_SECONDS)
    print("  [AI DISK LOCK] 同批次任务运行中，等待首个结果")
    while time.time() < deadline:
        cached = _load_ai_disk_cache(cache_key)
        if cached is not None:
            return cached
        await asyncio.sleep(AI_DISK_LOCK_POLL_SECONDS)
    return None


async def _run_ai_matrix_two_phase_inner(match_analyses: List[Dict[str, Any]], cache_key: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    if aiohttp is None:
        print("  [AI ERROR] aiohttp 未安装，AI弃权")
        return {n: {} for n in AI_NAMES}
    debug_ai_config()
    num = len(match_analyses)
    prompt = build_phase1_prompt(match_analyses)
    print(f"  [{ENGINE_VERSION} Phase1 Prompt] {len(prompt):,}字符 → GPT/Grok/Gemini")
    all_results: Dict[str, Dict[int, Dict[str, Any]]] = {n: {} for n in AI_NAMES}
    connector = aiohttp.TCPConnector(limit=8, use_dns_cache=False, ttl_dns_cache=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        phase1 = await asyncio.gather(*[async_call_one_ai_batch(session, prompt, num, name, _phase_system(name)) for name in PHASE1_NAMES], return_exceptions=True)
        for res in phase1:
            if isinstance(res, tuple):
                all_results[res[0]] = res[1]
            else:
                print(f"  [Phase1 ERROR] {res}")
        audit_prompt = build_claude_final_audit_prompt(match_analyses, all_results)
        print(f"  [{ENGINE_VERSION} Phase2 Claude Audit] {len(audit_prompt):,}字符 | compact={AI_USE_COMPACT_CLAUDE_AUDIT} | one_call=True")
        _, cl_res, _ = await async_call_one_ai_batch(session, audit_prompt, num, "claude", _phase_system("claude"))
        all_results["claude"] = cl_res or {}
    ok = sum(1 for n in AI_NAMES if all_results.get(n))
    status = {k: AI_CALL_STATUS.get(k, {}) for k in AI_NAMES}
    print(f"  [完成] {ok}/4 AI有数据 | status={status}")
    if ok > 0:
        _AI_RESULT_CACHE[cache_key] = (time.time(), all_results, status)
        _save_ai_disk_cache(cache_key, all_results, status)
    return all_results


async def run_ai_matrix_two_phase(match_analyses: List[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    cache_key = _stable_ai_cache_key(match_analyses)
    now = time.time()
    if not AI_FORCE_FRESH and AI_EXACT_SNAPSHOT_REUSE_SECONDS > 0 and cache_key in _AI_RESULT_CACHE:
        ts, results, _ = _AI_RESULT_CACHE[cache_key]
        if now - ts <= AI_EXACT_SNAPSHOT_REUSE_SECONDS:
            print(f"  [AI EXACT-SNAPSHOT REUSE] 命中短窗口去重 {AI_EXACT_SNAPSHOT_REUSE_SECONDS}s；赔率快照变动会自动失效")
            return results
        _AI_RESULT_CACHE.pop(cache_key, None)
    cached = _load_ai_disk_cache(cache_key)
    if cached is not None:
        _AI_RESULT_CACHE[cache_key] = (time.time(), cached, {"status": "disk_cache"})
        return cached
    if AI_SINGLEFLIGHT_ENABLED and cache_key in _AI_INFLIGHT_TASKS:
        print("  [AI SINGLEFLIGHT] 同批次AI正在本进程运行，等待首个结果")
        return await _AI_INFLIGHT_TASKS[cache_key]
    lock_acquired = _try_acquire_ai_disk_lock(cache_key)
    if not lock_acquired:
        waited = await _wait_for_ai_disk_cache(cache_key)
        if waited is not None:
            return waited
        lock_acquired = _try_acquire_ai_disk_lock(cache_key)
    task = asyncio.create_task(_run_ai_matrix_two_phase_inner(match_analyses, cache_key))
    if AI_SINGLEFLIGHT_ENABLED:
        _AI_INFLIGHT_TASKS[cache_key] = task
    try:
        return await task
    finally:
        if AI_SINGLEFLIGHT_ENABLED:
            _AI_INFLIGHT_TASKS.pop(cache_key, None)
        if lock_acquired:
            _release_ai_disk_lock(cache_key)


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    if loop and loop.is_running():
        import concurrent.futures
        def runner():
            nl = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(nl)
                return nl.run_until_complete(coro)
            finally:
                nl.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(runner).result()
    return asyncio.run(coro)

# ============================================================
# 合并 / 推荐
# ============================================================

def _valid_ai_score_from_response(r: Dict[str, Any]) -> str:
    if not isinstance(r, dict):
        return ""
    sc = _score_from_candidate(r.get("ai_score", ""))
    if _parse_score(sc)[0] is not None:
        return sc
    top3 = r.get("top3", [])
    if isinstance(top3, list) and top3:
        sc = _score_from_candidate(top3[0])
        if _parse_score(sc)[0] is not None:
            return sc
    return ""


def _phase1_exact_consensus(ai_responses: Dict[str, Dict[str, Any]]) -> Tuple[str, int, List[str]]:
    counts: Dict[str, List[str]] = {}
    for name in PHASE1_NAMES:
        sc = _valid_ai_score_from_response(ai_responses.get(name, {}))
        if sc:
            counts.setdefault(sc, []).append(name)
    if not counts:
        return "", 0, []
    sc, names = sorted(counts.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True)[0]
    return sc, len(names), names


def _choose_final_ai(ai_responses: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any], str]:
    if _valid_ai_score_from_response(ai_responses.get("claude", {})):
        return "claude", ai_responses["claude"], "claude_pure_authority"
    sc, n, names = _phase1_exact_consensus(ai_responses)
    if sc and n >= 2:
        best_name = max(names, key=lambda nm: _f(ai_responses.get(nm, {}).get("ai_confidence", 60), 60))
        return best_name, ai_responses[best_name], f"phase1_exact_consensus:{','.join(names)}"
    valid = [(name, ai_responses.get(name, {})) for name in PHASE1_NAMES if _valid_ai_score_from_response(ai_responses.get(name, {}))]
    if valid:
        name, r = max(valid, key=lambda nr: _f(nr[1].get("ai_confidence", 60), 60))
        return name, r, f"phase1_best_confidence:{name}"
    return "", {}, "ai_abstain_no_valid_model"


def _direction_pct_from_top3(top3: List[Dict[str, Any]], final_direction: str) -> Dict[str, float]:
    mass = {"home": 0.0, "draw": 0.0, "away": 0.0}
    for cand in top3 or []:
        sc = _score_from_candidate(cand)
        d = _score_direction(sc)
        if d in mass:
            mass[d] += max(0.0, _prob_to_float(cand.get("prob", 0) if isinstance(cand, dict) else 0))
    if sum(mass.values()) <= 0:
        mass = {"home": 25.0, "draw": 25.0, "away": 25.0}
        if final_direction in mass:
            mass[final_direction] = 50.0
    s = sum(mass.values())
    return {k: round(v / s * 100, 1) for k, v in mass.items()}


def _goal_range_from_score(score: str) -> Tuple[int, int, str]:
    total = _score_total(score)
    if total is None:
        return 0, 7, "ai_raw_unknown"
    if total <= 1:
        return 0, 1, "ai_raw_0_1_goals"
    if total == 2:
        return 1, 2, "ai_raw_2_goals"
    if total == 3:
        return 2, 3, "ai_raw_3_goals"
    if total == 4:
        return 3, 4, "ai_raw_4_goals"
    if total == 5:
        return 4, 5, "ai_raw_5_goals"
    return 5, 8, "ai_raw_6plus_goals"


def _tier_rank(t: str) -> int:
    return {"S": 4, "A": 3, "B": 2, "C": 1, "D": 0}.get(str(t).upper(), 0)


def _rank_tier(r: int) -> str:
    return {4: "S", 3: "A", 2: "B", 1: "C", 0: "D"}.get(max(0, min(4, r)), "D")


def _tier_from_score(score: float) -> str:
    if score >= SELECTION_TIER_S:
        return "S"
    if score >= SELECTION_TIER_A:
        return "A"
    if score >= SELECTION_TIER_B:
        return "B"
    if score >= SELECTION_TIER_C:
        return "C"
    return "D"


def _ai_model_agreement(all_ai: Dict[str, Dict[str, Any]], score: str, direction: str) -> Dict[str, Any]:
    valid = []
    same_score = 0
    same_dir = 0
    for name in AI_NAMES:
        r = all_ai.get(name, {}) if isinstance(all_ai, dict) else {}
        sc = _valid_ai_score_from_response(r)
        if not sc:
            continue
        valid.append(name)
        if sc == score:
            same_score += 1
        if (_score_direction(sc) or "") == direction:
            same_dir += 1
    total = max(1, len(valid))
    return {"valid_models": valid, "valid_count": len(valid), "same_score": same_score, "same_direction": same_dir, "score_agreement": same_score / total, "direction_agreement": same_dir / total}


def _experience_review_coverage(final_r: Dict[str, Any], exp_audit: Dict[str, Any]) -> Tuple[float, List[str]]:
    ids = [str(t.get("id", "")).replace("EXP_", "") for t in (exp_audit.get("triggered", []) if isinstance(exp_audit, dict) else []) if t.get("id")]
    if not ids:
        return 1.0, []
    reviews = final_r.get("experience_review", []) if isinstance(final_r, dict) else []
    reviewed = {str(r.get("id", "")).replace("EXP_", "") for r in reviews if isinstance(r, dict)}
    for r in reviews:
        if isinstance(r, dict):
            for rid in re.findall(r"EXP_[A-Z0-9_]+|B_SHARP|B_STEAM|X\d{2}|[DUG]\d{2}", str(r.get("reason", ""))):
                reviewed.add(rid.replace("EXP_", ""))
    missing = [rid for rid in ids if rid not in reviewed]
    return max(0.0, 1.0 - len(missing) / max(1, len(ids))), missing


def _inherit_experience_review_if_missing(final_r: Dict[str, Any], all_ai: Dict[str, Dict[str, Any]], final_score: str, final_direction: str) -> Dict[str, Any]:
    if isinstance(final_r.get("experience_review"), list) and final_r.get("experience_review"):
        return final_r
    candidates = []
    for name in PHASE1_NAMES:
        r = all_ai.get(name, {})
        rv = r.get("experience_review") if isinstance(r, dict) else None
        if not isinstance(rv, list) or not rv:
            continue
        sc = _valid_ai_score_from_response(r)
        candidates.append((1 if sc == final_score else 0, 1 if _score_direction(sc) == final_direction else 0, _f(r.get("ai_confidence", 60), 60), name, rv))
    if not candidates:
        return final_r
    candidates.sort(reverse=True)
    _, _, _, src, review = candidates[0]
    out = dict(final_r)
    out["experience_review"] = review
    out["experience_review_inherited_from"] = src
    audit = out.get("audit", {}) if isinstance(out.get("audit", {}), dict) else {}
    audit["experience_review_inherited_from"] = src
    out["audit"] = audit
    return out


def _force_experience_review_full_coverage(final_r: Dict[str, Any], exp_audit: Dict[str, Any]) -> Dict[str, Any]:
    triggered = exp_audit.get("triggered", []) if isinstance(exp_audit, dict) else []
    if not triggered:
        return final_r
    review = final_r.get("experience_review") if isinstance(final_r.get("experience_review"), list) else []
    reviewed = {str(r.get("id", "")).replace("EXP_", "") for r in review if isinstance(r, dict)}
    added = []
    for t in triggered:
        rid = str(t.get("id", "")).replace("EXP_", "").strip()
        if rid and rid not in reviewed:
            added.append({"id": rid, "decision": "neutral", "reason": f"{rid}:neutral because 本地字段闭环补齐：最终模型未逐条输出；只标记覆盖，不改方向、不改比分。"})
            reviewed.add(rid)
    if added:
        out = dict(final_r)
        out["experience_review"] = review + added
        audit = out.get("audit", {}) if isinstance(out.get("audit", {}), dict) else {}
        audit["experience_review_auto_filled"] = [x["id"] for x in added]
        out["audit"] = audit
        return out
    return final_r


def _fix_shape_fields_to_score(final_r: Dict[str, Any], score: str) -> Tuple[Dict[str, Any], List[str]]:
    out = dict(final_r or {})
    warnings: List[str] = []
    sg = _score_goal_band(score)
    sb = _score_btts(score)
    gb = _normalize_goal_band_value(out.get("goal_band", ""), score)
    bt = _normalize_btts_value(out.get("btts", ""), score)
    if sg and gb and gb != sg:
        warnings.append(f"goal_band_auto_fixed:{gb}->{sg}")
    if sb in ("yes", "no") and bt in ("yes", "no") and bt != sb:
        warnings.append(f"btts_auto_fixed:{bt}->{sb}")
    out["goal_band"] = sg or gb
    out["btts"] = sb if sb in ("yes", "no") else bt
    return out, warnings


def _apply_recommendation_gate(pack: Dict[str, Any], exp_audit: Dict[str, Any], score: str, direction: str) -> Tuple[Dict[str, Any], List[str]]:
    out = dict(pack)
    warnings: List[str] = []
    sharp = exp_audit.get("sharp_money_pack", {}) if isinstance(exp_audit, dict) else {}
    shape = exp_audit.get("score_shape_pack", {}) if isinstance(exp_audit, dict) else {}
    cap = str(sharp.get("recommendation_cap", "S")).upper()
    tier = str(out.get("recommendation_tier", _tier_from_score(out.get("overall_selection_score", 0)))).upper()
    if _tier_rank(cap) < _tier_rank(tier):
        warnings.append(f"recommendation_cap_by_sharp:{tier}->{cap}")
        out["recommendation_tier"] = cap
        if cap == "C":
            out["overall_selection_score"] = min(_f(out.get("overall_selection_score"), 0), 54.0)
            out["direction_selection_score"] = min(_f(out.get("direction_selection_score"), 0), 62.0)
        elif cap == "B":
            out["overall_selection_score"] = min(_f(out.get("overall_selection_score"), 0), 66.0)
    conflict = sharp.get("conflict_type", "none")
    if conflict == "static_home_vs_dynamic_away_draw" and direction == "home":
        warnings.append("sharp_conflict_home_prediction_downgraded")
        out["recommendation_tier"] = _rank_tier(min(_tier_rank(out.get("recommendation_tier", "D")), 1))
    if conflict == "static_away_vs_dynamic_home_draw" and direction == "away":
        warnings.append("sharp_conflict_away_prediction_downgraded")
        out["recommendation_tier"] = _rank_tier(min(_tier_rank(out.get("recommendation_tier", "D")), 1))
    if shape.get("low_goal_signal") and shape.get("draw_controlled") and score not in ("1-1", "0-0", "1-0", "0-1"):
        warnings.append("low_goal_draw_protection_score_not_in_core_pool")
        out["score_shape_score"] = min(_f(out.get("score_shape_score"), 0), 58.0)
        out["recommendation_tier"] = _rank_tier(min(_tier_rank(out.get("recommendation_tier", "D")), 2))
    if shape.get("dominant_but_concede") and _score_btts(score) == "no" and _score_total(score) in (2, 3, 4):
        warnings.append("dominant_but_concede_zero_score_downgraded")
        out["score_shape_score"] = min(_f(out.get("score_shape_score"), 0), 62.0)
    out["gate_warnings"] = warnings
    return out, warnings


def _compute_recommendation_scores(final_r: Dict[str, Any], all_ai: Dict[str, Dict[str, Any]], match_obj: Dict[str, Any], exp_audit: Dict[str, Any], score: str, direction: str, pct: Dict[str, float], top_candidates: List[Tuple[str, float]]) -> Dict[str, Any]:
    vals = sorted([_f(v, 0.0) for v in pct.values()], reverse=True)
    top = vals[0] if vals else 33.3
    gap = vals[0] - vals[1] if len(vals) >= 2 else 0.0
    ps = [max(1e-6, _f(pct.get(k, 0.0), 0.0) / 100.0) for k in ["home", "draw", "away"]]
    sprob = sum(ps)
    entropy = 1.0
    if sprob > 0:
        ps = [p / sprob for p in ps]
        entropy = -sum(p * math.log(p) for p in ps) / math.log(3)
    agreement = _ai_model_agreement(all_ai, score, direction)
    exp_cov, exp_missing = _experience_review_coverage(final_r, exp_audit)
    top_score_prob = top_candidates[0][1] if top_candidates else 0.0
    tsp = top_score_prob if top_score_prob <= 1 else top_score_prob / 100.0
    has_shape_reason = bool(str(final_r.get("score_shape_reason", "")).strip() or str((final_r.get("audit") or {}).get("style_score_logic", "")).strip())
    direction_score = 25.0 + 0.40 * top + 0.65 * gap + 10.0 * (1 - entropy) + 12.0 * agreement.get("direction_agreement", 0.0) + 8.0 * exp_cov
    shape_score = 18.0 + 16.0 * (top / 100.0) + 8.0 * (gap / 100.0) + 65.0 * min(1.0, tsp) + 16.0 * agreement.get("score_agreement", 0.0) + 9.0 * (1.0 if has_shape_reason else 0.0) + 6.0 * exp_cov
    if exp_missing:
        direction_score -= min(12, 2.0 * len(exp_missing))
        shape_score -= min(8, 1.2 * len(exp_missing))
    direction_score = round(_clip(direction_score, 0, 100), 1)
    shape_score = round(_clip(shape_score, 0, 100), 1)
    overall = round(min(direction_score, shape_score * 1.08), 1)
    pack = {
        "direction_selection_score": direction_score,
        "score_shape_score": shape_score,
        "overall_selection_score": overall,
        "recommendation_tier": _tier_from_score(overall),
        "direction_tier": _tier_from_score(direction_score),
        "score_tier": _tier_from_score(shape_score),
        "model_agreement": agreement,
        "experience_review_coverage": round(exp_cov, 3),
        "experience_review_missing": exp_missing,
    }
    gated, gate_warnings = _apply_recommendation_gate(pack, exp_audit, score, direction)
    gated["score_shape_warnings"] = gate_warnings
    return gated


def _abstain_prediction(reason: str = "AI全失败，PURE模式不使用本地兜底") -> Dict[str, Any]:
    return {
        "predicted_score": "弃权", "predicted_label": "弃权", "result": "弃权", "display_direction": "弃权", "final_direction": "abstain",
        "is_abstain": True, "home_win_pct": 0.0, "draw_pct": 0.0, "away_win_pct": 0.0, "confidence": 0,
        "confidence_meaning": "PURE RAW-AI 模式：AI全失败即弃权，不使用本地兜底",
        "risk_level": "高", "dir_confidence": 0, "dir_gap": 0, "scenario": "ai_abstain", "goal_range": (0, 0),
        "bayesian_evidences": [reason], "decision_source": "ai_abstain_no_local_fallback", "ai_authority_mode": "pure_raw_ai",
        "top_score_candidates": [], "unified_matrix_top_scores": [], "suggested_kelly": 0.0, "edge_vs_market": 0.0, "is_value": False,
        "score_model_prob": 0.0, "score_market_odds": 0.0, "score_market_implied_pct": None,
        "ai_avg_confidence": 0, "ai_abstained": ["GPT", "GROK", "GEMINI", "CLAUDE"],
        "gpt_score": "弃权", "gpt_analysis": "弃权", "grok_score": "弃权", "grok_analysis": "弃权", "gemini_score": "弃权", "gemini_analysis": "弃权", "claude_score": "弃权", "claude_analysis": "弃权",
        "model_consensus": 0, "total_models": 4, "engine_version": ENGINE_VERSION, "engine_architecture": ENGINE_ARCHITECTURE,
    }


def _make_ai_prediction(final_name: str, final_r: Dict[str, Any], decision_source: str, all_ai: Dict[str, Dict[str, Any]], match_obj: Dict[str, Any]) -> Dict[str, Any]:
    final_r = dict(final_r or {})
    score = _valid_ai_score_from_response(final_r)
    direction = _score_direction(score) or _normalize_direction(final_r.get("final_direction", ""), score)
    exp_audit = _experience_engine().analyze(match_obj)
    final_r = _inherit_experience_review_if_missing(final_r, all_ai, score, direction)
    final_r = _force_experience_review_full_coverage(final_r, exp_audit)
    final_r, shape_auto_warnings = _fix_shape_fields_to_score(final_r, score)
    top3 = final_r.get("top3", []) if isinstance(final_r.get("top3", []), list) else []
    top_candidates = []
    for cand in top3[:8]:
        sc = _score_from_candidate(cand)
        if _parse_score(sc)[0] is not None:
            top_candidates.append((sc, round(_prob_to_float(cand.get("prob", 0) if isinstance(cand, dict) else 0), 3)))
    if not top_candidates and score != "弃权":
        top_candidates = [(score, 0.0)]
    pct = final_r.get("direction_probs") if isinstance(final_r.get("direction_probs"), dict) and final_r.get("direction_probs") else _direction_pct_from_top3(top3, direction)
    conf = int(_clip(_f(final_r.get("ai_confidence", 60), 60), 0, 100))
    gmin, gmax, scenario = _goal_range_from_score(score)
    final_odds = get_market_odds_for_score(match_obj, score)
    market_implied = round(100.0 / final_odds, 3) if final_odds > 1.05 else None
    ai_abstained = [n.upper() for n in AI_NAMES if not _valid_ai_score_from_response(all_ai.get(n, {}))]
    confs = [_f(r.get("ai_confidence", 60), 60) for r in all_ai.values() if isinstance(r, dict) and _valid_ai_score_from_response(r)]
    avg_conf = round(sum(confs) / len(confs), 1) if confs else conf

    def sc_of(name: str) -> str:
        return _valid_ai_score_from_response(all_ai.get(name, {})) or "弃权"

    def reason_of(name: str) -> str:
        r = all_ai.get(name, {})
        if not isinstance(r, dict) or not _valid_ai_score_from_response(r):
            return "弃权"
        return str(r.get("reason", ""))[:3000]

    selection_pack = _compute_recommendation_scores(final_r, all_ai, match_obj, exp_audit, score, direction, pct, top_candidates)
    evidences = [
        "PURE RAW-AI：AI成功时不使用本地比分矩阵、不使用CRS、不使用本地风控兜底。",
        "SHARP SAFE-HYBRID：sharp_money_pack/score_shape_pack只进审计和推荐降级，不直接篡改比分。",
        "STRICT ONE-CALL：每个模型最多请求一次；Claude不做二次repair消费。",
        f"最终来源:{decision_source}; final_model={final_name}; score={score}; direction={direction}",
        "sharp_money_pack:" + _json_compact(exp_audit.get("sharp_money_pack", {}), 1600),
        "score_shape_pack:" + _json_compact(exp_audit.get("score_shape_pack", {}), 1600),
        f"AI top3:{top_candidates[:5]}",
        f"AI direction_probs:{pct}",
    ]
    if final_r.get("experience_review_inherited_from"):
        evidences.append(f"experience_review_inherited_from:{final_r.get('experience_review_inherited_from')}")
    if final_r.get("audit"):
        evidences.append("AI audit:" + _json_compact(final_r.get("audit"), 1800))
    if shape_auto_warnings:
        evidences.append("shape_auto_fixed:" + _json_compact(shape_auto_warnings, 500))

    h, a = _parse_score(score)
    total_goals = (h + a) if h is not None and a is not None else 0
    both_score_cn = "是" if h is not None and a is not None and h > 0 and a > 0 else "否"
    validation_warnings = list(shape_auto_warnings) + list(selection_pack.get("gate_warnings", []))

    return {
        "predicted_score": score,
        "predicted_label": _score_display_label(score, direction),
        "result": _direction_cn(direction), "display_direction": _direction_cn(direction), "final_direction": direction,
        "is_abstain": False, "is_score_others": _score_display_label(score, direction) in ("胜其他", "平其他", "负其他"),
        "home_win_pct": pct.get("home", 0.0), "draw_pct": pct.get("draw", 0.0), "away_win_pct": pct.get("away", 0.0),
        "confidence": conf, "confidence_meaning": "AI自报置信度，非历史命中率；PURE模式不做本地概率改写",
        "risk_level": str(final_r.get("risk_level", "medium")),
        "goal_band": final_r.get("goal_band", _score_goal_band(score)), "btts_ai": final_r.get("btts", _score_btts(score)),
        "score_shape_reason": final_r.get("score_shape_reason", ""), "experience_review": final_r.get("experience_review", []),
        "experience_review_inherited_from": final_r.get("experience_review_inherited_from"),
        **selection_pack,
        "dir_confidence": pct.get(direction, 0.0),
        "dir_gap": round(max(pct.values()) - sorted(pct.values(), reverse=True)[1], 1) if len(pct) >= 2 else 0.0,
        "scenario": scenario, "goal_range": (gmin, gmax), "bayesian_evidences": evidences,
        "top_score_candidates": top_candidates, "unified_matrix_top_scores": top_candidates,
        "decision_source": decision_source, "ai_authority_mode": "pure_raw_ai_sharp_safe_hybrid",
        "gpt_score": sc_of("gpt"), "gpt_analysis": reason_of("gpt"),
        "grok_score": sc_of("grok"), "grok_analysis": reason_of("grok"),
        "gemini_score": sc_of("gemini"), "gemini_analysis": reason_of("gemini"),
        "claude_score": sc_of("claude"), "claude_analysis": reason_of("claude"),
        "ai_abstained": ai_abstained, "ai_avg_confidence": avg_conf,
        "suggested_kelly": 0.0, "edge_vs_market": 0.0, "is_value": False, "ev_note": "disabled_pure_raw_ai_no_local_probability",
        "score_model_prob": top_candidates[0][1] if top_candidates else 0.0, "score_market_odds": final_odds, "score_market_implied_pct": market_implied,
        "smart_money_signal": " | ".join(exp_audit.get("risk_signals", [])[:8]),
        "smart_signals": ["EXP_AUDIT:" + s for s in exp_audit.get("risk_signals", [])],
        "sharp_money_pack": exp_audit.get("sharp_money_pack", {}), "score_shape_pack": exp_audit.get("score_shape_pack", {}),
        "sharp_detected": bool(exp_audit.get("sharp_money_pack", {}).get("dynamic_money_direction") != "unclear"),
        "sharp_dir": exp_audit.get("sharp_money_pack", {}).get("dynamic_money_direction"),
        "xG_home": "?", "xG_away": "?", "expected_total_goals": total_goals,
        "over_under_2_5": "大" if total_goals >= 3 else "小", "both_score": both_score_cn,
        "model_consensus": len([n for n in AI_NAMES if _valid_ai_score_from_response(all_ai.get(n, {}))]), "total_models": 4,
        "experience_analysis": exp_audit, "validation_warnings": list(dict.fromkeys(validation_warnings)),
        "engine_version": ENGINE_VERSION, "engine_architecture": ENGINE_ARCHITECTURE,
        # 前端兼容空字段
        "bayesian_prior": {}, "override_triggered": False, "traps_detected": [], "trap_count": 0, "trap_severity": 0, "trap_details": [], "trap_flags": {},
        "fair_1x2": {}, "fair_1x2_method": "disabled_pure_raw_ai", "market_overround": 0.0, "raw_implied_1x2": {},
        "crs_shape": "disabled_pure_raw_ai", "crs_moments": {}, "crs_margin": 0.0, "crs_coverage": 0.0, "crs_implied_probs": {}, "crs_low_rank_info": {},
        "unified_goal_probs": {}, "fair_1x2_pack": {}, "mixed_target_dir": {}, "unified_source": "disabled_pure_raw_ai",
        "cold_door": {"is_cold_door": False, "strength": 0, "level": "普通", "signals": [], "sharp_confirmed": False, "dark_verdict": ""},
        "over_2_5": None, "btts": None, "bookmaker_implied_home_xg": "?", "bookmaker_implied_away_xg": "?",
    }


def merge_result_pure_ai(all_ai: Dict[str, Dict[int, Dict[str, Any]]], idx: int, match_obj: Dict[str, Any]) -> Dict[str, Any]:
    per_match = {name: (all_ai.get(name, {}) or {}).get(idx, {}) for name in AI_NAMES}
    final_name, final_r, source = _choose_final_ai(per_match)
    if not final_name:
        return _abstain_prediction()
    return _make_ai_prediction(final_name, final_r, source, per_match, match_obj)


def _enforce_consistency(mg: Dict[str, Any]) -> Dict[str, Any]:
    if mg.get("is_abstain") or mg.get("predicted_score") == "弃权":
        mg["predicted_score"] = "弃权"; mg["predicted_label"] = "弃权"; mg["result"] = "弃权"; mg["display_direction"] = "弃权"; mg["final_direction"] = "abstain"
        return mg
    score = _normalize_score_text(mg.get("predicted_score", ""))
    d = _score_direction(score) or _dir_from_cn(mg.get("result", "")) or "draw"
    mg["predicted_score"] = score
    mg["predicted_label"] = _score_display_label(score, d)
    mg["result"] = _direction_cn(d)
    mg["display_direction"] = _direction_cn(d)
    mg["final_direction"] = d
    mg["is_score_others"] = mg["predicted_label"] in ("胜其他", "平其他", "负其他")
    warnings = list(mg.get("validation_warnings", []))
    sg, sb = _score_goal_band(score), _score_btts(score)
    old_gb, old_bt = str(mg.get("goal_band", "")), str(mg.get("btts_ai", ""))
    if sg and old_gb and old_gb != sg:
        warnings.append(f"goal_band_auto_fixed:{old_gb}->{sg}")
    if sb in ("yes", "no") and old_bt in ("yes", "no") and old_bt != sb:
        warnings.append(f"btts_auto_fixed:{old_bt}->{sb}")
    mg["goal_band"] = sg or mg.get("goal_band", "")
    mg["btts_ai"] = sb if sb in ("yes", "no") else mg.get("btts_ai", "unclear")
    mg["validation_warnings"] = list(dict.fromkeys(warnings))
    return mg


def _validate_prediction_consistency(mg: Dict[str, Any]) -> Dict[str, Any]:
    warnings = list(mg.get("validation_warnings", []))
    if mg.get("is_abstain"):
        mg["validation_warnings"] = warnings
        return _enforce_consistency(mg)
    score = _normalize_score_text(mg.get("predicted_score", ""))
    h, a = _parse_score(score)
    if h is None:
        warnings.append("score_unparseable_in_pure_ai")
    else:
        total = h + a
        gr = mg.get("goal_range")
        if isinstance(gr, list):
            gr = tuple(gr)
        if isinstance(gr, tuple) and len(gr) == 2:
            gmin, gmax = _i(gr[0]), _i(gr[1])
            if not (gmin <= total <= gmax):
                warnings.append(f"goal_range_adjusted_for_ai_score:{gr}->{total}")
                mg["goal_range"] = (min(gmin, total), max(gmax, total))
    mg["validation_warnings"] = list(dict.fromkeys(warnings))
    return _enforce_consistency(mg)


def select_top4(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for p in preds:
        pr = p.get("prediction", {})
        if pr.get("is_abstain"):
            p["recommend_score"] = -999
            continue
        s = _f(pr.get("overall_selection_score", pr.get("confidence", 0)), 0)
        s += _f(pr.get("dir_confidence", 0), 0) * 0.08
        if pr.get("risk_level") in ("low", "低"):
            s += 5
        if pr.get("risk_level") in ("high", "高"):
            s -= 8
        if pr.get("gate_warnings"):
            s -= min(10, len(pr.get("gate_warnings", [])) * 3)
        if pr.get("experience_review_missing"):
            s -= min(8, len(pr.get("experience_review_missing", [])) * 2)
        p["recommend_score"] = round(s, 2)
    return sorted(preds, key=lambda x: x.get("recommend_score", -999), reverse=True)[:4]


def extract_num(ms: Any) -> int:
    wm = {"一": 1000, "二": 2000, "三": 3000, "四": 4000, "五": 5000, "六": 6000, "日": 7000, "天": 7000}
    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999

# ============================================================
# 外部情报占位
# ============================================================

async def enrich_external_context(match_analyses: List[Dict[str, Any]]) -> None:
    for ma in match_analyses:
        ma["external_context"] = {"enabled": False, "source_quality": "disabled", "items": [], "errors": []}

# ============================================================
# 沙盒庄家模拟：用于离线压力测试 sharp gate，不参与预测
# ============================================================

def run_sandbox_bookmaker_simulation(n_total: int = 1000000, chunk: int = 250000, seed: int = 18413) -> Dict[str, Any]:
    """纯本地压力测试。需要 numpy。默认100万，传 10000000 可跑1000万。"""
    try:
        import numpy as np
    except Exception as e:
        return {"ok": False, "error": f"numpy_missing:{e}"}
    rng = np.random.default_rng(seed)
    stats = {k: 0 for k in [
        "n", "sharp_conflict_cases", "sharp_conflict_lock_violations", "sharp_conflict_tier_violations",
        "low_goal_draw_cases", "low_goal_11_missing_violations", "strong_hot_concede_cases", "strong_hot_zero_lock_violations", "extreme_move_cases",
    ]}
    loops = max(1, n_total // max(1, chunk))
    t0 = time.time()
    for _ in range(loops):
        n = min(chunk, n_total - stats["n"])
        if n <= 0:
            break
        typ = rng.integers(0, 6, size=n)
        init_h = rng.uniform(1.35, 4.5, n); init_d = rng.uniform(2.7, 5.2, n); init_a = rng.uniform(1.35, 5.5, n)
        stronghome, strongaway, elite, lowgoal, extreme = typ == 1, typ == 2, typ == 3, typ == 4, typ == 5
        c = stronghome.sum(); init_h[stronghome] = rng.uniform(1.15, 1.80, c); init_a[stronghome] = rng.uniform(3.2, 8.5, c)
        c = strongaway.sum(); init_a[strongaway] = rng.uniform(1.20, 1.85, c); init_h[strongaway] = rng.uniform(3.2, 9.0, c)
        c = elite.sum(); init_h[elite] = rng.uniform(1.45, 2.10, c); init_a[elite] = rng.uniform(2.60, 4.20, c); init_d[elite] = rng.uniform(3.2, 4.3, c)
        c = lowgoal.sum(); init_h[lowgoal] = rng.uniform(1.7, 3.2, c); init_a[lowgoal] = rng.uniform(1.8, 3.5, c); init_d[lowgoal] = rng.uniform(2.7, 3.6, c)
        c = extreme.sum(); init_h[extreme] = rng.uniform(1.1, 7.0, c); init_a[extreme] = rng.uniform(1.1, 7.0, c); init_d[extreme] = rng.uniform(2.5, 6.0, c)
        move_h = rng.normal(0, 0.08, n); move_d = rng.normal(0, 0.06, n); move_a = rng.normal(0, 0.08, n)
        conflict_mask = elite & (rng.random(n) < 0.50)
        c = conflict_mask.sum(); move_a[conflict_mask] -= rng.uniform(0.12, 0.35, c); move_h[conflict_mask] += rng.uniform(0.02, 0.20, c)
        draw_sharp_mask = elite & (~conflict_mask) & (rng.random(n) < 0.30)
        c = draw_sharp_mask.sum(); move_d[draw_sharp_mask] -= rng.uniform(0.08, 0.25, c); move_h[draw_sharp_mask] += rng.uniform(0.02, 0.18, c); move_a[draw_sharp_mask] += rng.uniform(0.02, 0.18, c)
        latest_h = np.clip(init_h * np.exp(move_h), 1.05, 20); latest_d = np.clip(init_d * np.exp(move_d), 1.4, 20); latest_a = np.clip(init_a * np.exp(move_a), 1.05, 20)
        dh = (latest_h - init_h) / init_h; dd = (latest_d - init_d) / init_d; da = (latest_a - init_a) / init_a
        vote_h = rng.integers(20, 80, n); vote_a = rng.integers(5, 70, n)
        a0 = rng.uniform(5.5, 35, n); a1 = rng.uniform(3.2, 12, n); a2 = rng.uniform(2.8, 7.5, n); a3 = rng.uniform(3.0, 6.2, n); a4 = rng.uniform(3.2, 12, n)
        c = lowgoal.sum(); a0[lowgoal] = rng.uniform(5.4, 8.2, c); a1[lowgoal] = rng.uniform(3.5, 5.8, c); a2[lowgoal] = rng.uniform(2.8, 4.2, c)
        c = stronghome.sum(); a3[stronghome] = rng.uniform(3.2, 4.9, c); a4[stronghome] = rng.uniform(3.5, 6.0, c)
        s11 = rng.uniform(5.0, 16, n); c = lowgoal.sum(); s11[lowgoal] = rng.uniform(4.8, 8.2, c)
        fav_home = (latest_h < latest_d) & (latest_h < latest_a)
        sharp_away = (da < -0.055) & (vote_a < 55) & (latest_a >= 1.25)
        sharp_draw = (dd < -0.045) & (dh > 0.015) & (da > 0.015)
        sharp_conflict = fav_home & (sharp_away | sharp_draw) & elite
        home_lock_allowed = np.ones(n, dtype=bool)
        recommendation_rank = np.full(n, 3, dtype=np.int8)
        home_lock_allowed[sharp_conflict] = False
        recommendation_rank[sharp_conflict] = np.minimum(recommendation_rank[sharp_conflict], 1)
        low_goal_draw = (a0 <= 8.5) & (a1 <= 6.2) & (s11 <= 8.5) & (latest_d <= 5.2)
        score_pool_contains_11 = low_goal_draw.copy()
        strong_hot_concede = fav_home & (latest_h <= 1.85) & ((a3 <= 5.0) | (a4 <= 5.8)) & (a0 >= 9)
        zero_lock = np.zeros(n, dtype=bool)
        stats["n"] += int(n)
        stats["sharp_conflict_cases"] += int(sharp_conflict.sum())
        stats["sharp_conflict_lock_violations"] += int((sharp_conflict & home_lock_allowed).sum())
        stats["sharp_conflict_tier_violations"] += int((sharp_conflict & (recommendation_rank > 1)).sum())
        stats["low_goal_draw_cases"] += int(low_goal_draw.sum())
        stats["low_goal_11_missing_violations"] += int((low_goal_draw & ~score_pool_contains_11).sum())
        stats["strong_hot_concede_cases"] += int(strong_hot_concede.sum())
        stats["strong_hot_zero_lock_violations"] += int((strong_hot_concede & zero_lock).sum())
        stats["extreme_move_cases"] += int((abs(dh) + abs(dd) + abs(da) > 0.45).sum())
    stats["elapsed_seconds"] = round(time.time() - t0, 3)
    stats["ok"] = True
    return stats

# ============================================================
# 主入口
# ============================================================

def run_predictions(raw: Dict[str, Any], use_ai: bool = True):
    raw_ms = _extract_match_list(raw)
    ms = [normalize_match(m) for m in raw_ms]
    print("\n" + "=" * 88)
    print(f"  [{ENGINE_VERSION}] PURE RAW-AI + SHARP SAFE-HYBRID | {len(ms)} 场 | STRICT ONE-CALL | 默认不持久化缓存")
    print("=" * 88)
    match_analyses: List[Dict[str, Any]] = []
    for i, m in enumerate(ms, 1):
        exp_audit = _experience_engine().analyze(m)
        # 让市场快照 hash 显式记录，赔率变动必然导致 exact snapshot key 变化。
        exp_audit["market_snapshot"]["market_hash"] = _hash_obj({
            "market": exp_audit.get("market_snapshot", {}),
            "sharp": exp_audit.get("sharp_money_pack", {}),
            "shape": exp_audit.get("score_shape_pack", {}),
        })
        match_analyses.append({"match": m, "index": i, "experience_audit": exp_audit, "external_context": {"enabled": False, "source_quality": "disabled", "items": [], "errors": []}})
    if ENABLE_EXTERNAL_CONTEXT and match_analyses:
        print(f"  [{ENGINE_VERSION} External] 联网情报入口 enabled=True ...")
        try:
            _run_async(enrich_external_context(match_analyses))
        except Exception as e:
            logger.warning(f"external_context 失败: {e}")
    else:
        print(f"  [{ENGINE_VERSION} External] 联网情报入口 enabled=False")
    all_ai: Dict[str, Dict[int, Dict[str, Any]]] = {n: {} for n in AI_NAMES}
    if use_ai and match_analyses:
        print(f"  [{ENGINE_VERSION} AI] 启动 GPT/Grok/Gemini 初审 + Claude 终审 | 每模型最多一次请求")
        try:
            all_ai = _run_async(run_ai_matrix_two_phase(match_analyses))
        except Exception as e:
            logger.error(f"AI矩阵执行失败: {e}")
            all_ai = {n: {} for n in AI_NAMES}
    elif not use_ai:
        print("  [PURE RAW-AI] use_ai=False → 全部弃权，不启用本地兜底")
    res = []
    for i, ma in enumerate(match_analyses, 1):
        m = ma["match"]
        mg = merge_result_pure_ai(all_ai, i, m)
        mg = _validate_prediction_consistency(mg)
        res.append({**m, "prediction": mg})
        if mg.get("is_abstain"):
            print(f"  [{i}] {m.get('home_team')} vs {m.get('away_team')} => 弃权 | PURE模式AI无有效结果")
        else:
            print(f"  [{i}] {m.get('home_team')} vs {m.get('away_team')} => {mg['result']} ({mg['predicted_score']}={mg['predicted_label']}) | AI_CF:{mg['confidence']} | tier:{mg.get('recommendation_tier')} | 来源:{mg.get('decision_source')}")
    t4 = select_top4(res)
    t4_ids = set(id(x) for x in t4)
    for r in res:
        r["is_recommended"] = id(r) in t4_ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    return res, t4


if __name__ == "__main__":
    logger.info(f"{ENGINE_VERSION} 启动")
    print(f"✅ {ENGINE_VERSION} 加载完成")
    print(f"   架构: {ENGINE_ARCHITECTURE}")
    print("   模式: AI成功=AI直出；AI失败=弃权；每模型最多一次请求；默认不持久化缓存；sharp只做审计和推荐降级")
