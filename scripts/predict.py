# -*- coding: utf-8 -*-
"""
vMAX 18.4.10 — PURE RAW-AI + Claude JSON Repair + 字段闭环修复版
============================================================
设计边界：
1. 纯 AI 主审：GPT/Grok/Gemini 初审，Claude 终审；Claude 失败时使用 Phase1 AI 共识。
2. 不跑 CRS 矩阵、不跑贝叶斯后验、不跑本地比分矩阵、不跑本地风控裁决。
3. 本地只做：抓包格式化、AI 调用、JSON/文本解析、字段闭环、前端兼容、结构化审计。
4. Phase1 三家吃完整抓包；Claude 终审默认吃压缩抓包 + 三家结构化摘要。
5. Claude 若有返回但 JSON 解析失败，自动二次 JSON Repair，不让自然语言残片直接进入裁决。
6. goal_band / btts 先按最终比分闭环，再参与评分，避免自动修复后仍被扣分。
7. change 支持两种语义：真实赔率变动幅度 / -1降水 +1升水方向编码。
8. D15 杯赛保守卡只在首回合/淘汰赛/两回合等保守场景触发，不再对所有杯赛小组赛误触发。
9. URL/KEY 默认保持旧版统一中转逻辑：API_KEY/API_URL 优先，旧变量兼容兜底。

入口：
    run_predictions(raw, use_ai=True) -> (res, top4)
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

# ============================================================
# 版本常量
# ============================================================

ENGINE_VERSION = "vMAX 18.4.10"
ENGINE_ARCHITECTURE = (
    "PURE RAW-AI: GPT/Grok/Gemini完整抓包初审 + Claude终审 + Claude JSON Repair + "
    "STRICT JSON结构化裁决 + change语义识别 + goal_band先闭环后评分 + "
    "经验卡场景化触发；无CRS/无贝叶斯/无本地比分矩阵/AI失败即弃权"
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
    "1-0": "w10", "2-0": "w20", "2-1": "w21", "3-0": "w30", "3-1": "w31",
    "3-2": "w32", "4-0": "w40", "4-1": "w41", "4-2": "w42", "5-0": "w50",
    "5-1": "w51", "5-2": "w52",
    "0-0": "s00", "1-1": "s11", "2-2": "s22", "3-3": "s33",
    "0-1": "l01", "0-2": "l02", "1-2": "l12", "0-3": "l03", "1-3": "l13",
    "2-3": "l23", "0-4": "l04", "1-4": "l14", "2-4": "l24", "0-5": "l05",
    "1-5": "l15", "2-5": "l25",
}

SCORE_OTHERS_HOME = [
    "4-3", "5-3", "5-4", "6-0", "6-1", "6-2", "6-3", "6-4",
    "7-0", "7-1", "7-2", "胜其他", "9-0",
]
SCORE_OTHERS_DRAW = ["4-4", "5-5", "6-6", "平其他", "9-9"]
SCORE_OTHERS_AWAY = [
    "3-4", "0-6", "1-6", "2-6", "3-6", "0-7", "1-7", "2-7",
    "负其他", "0-9",
]
ALL_SCORE_OTHERS = SCORE_OTHERS_HOME + SCORE_OTHERS_DRAW + SCORE_OTHERS_AWAY

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


AI_DECISION_CACHE_TTL = _env_int("AI_DECISION_CACHE_TTL", 1800)
AI_DISABLE_CACHE = _env_bool("AI_DISABLE_CACHE", False)
AI_SINGLEFLIGHT_ENABLED = _env_bool("AI_SINGLEFLIGHT_ENABLED", True)
AI_MAX_REQUESTS_PER_AI = max(1, _env_int("AI_MAX_REQUESTS_PER_AI", 1))
AI_FORCE_COMMON_GATEWAY = _env_bool("FORCE_COMMON_GATEWAY_URL", True)

AI_READ_TIMEOUT = _env_int("AI_READ_TIMEOUT", 1000)
AI_CLAUDE_READ_TIMEOUT = _env_int("AI_CLAUDE_READ_TIMEOUT", 1500)
AI_CONNECT_TIMEOUT = _env_int("AI_CONNECT_TIMEOUT", 25)
AI_CLAUDE_CONNECT_TIMEOUT = _env_int("AI_CLAUDE_CONNECT_TIMEOUT", 45)

ENABLE_EXTERNAL_CONTEXT = _env_bool("ENABLE_EXTERNAL_CONTEXT", False)
AI_PARSE_DEBUG = _env_bool("AI_PARSE_DEBUG", False)
AI_SAVE_RAW_RESPONSE = _env_bool("AI_SAVE_RAW_RESPONSE", False)

AI_ALLOW_TEXT_FALLBACK = _env_bool("AI_ALLOW_TEXT_FALLBACK", True)
AI_ALLOW_CLAUDE_TEXT_FALLBACK = _env_bool("AI_ALLOW_CLAUDE_TEXT_FALLBACK", False)
AI_ENABLE_CLAUDE_JSON_REPAIR = _env_bool("AI_ENABLE_CLAUDE_JSON_REPAIR", True)
AI_ENABLE_ANY_MODEL_JSON_REPAIR = _env_bool("AI_ENABLE_ANY_MODEL_JSON_REPAIR", False)
AI_REPAIR_MAX_RAW_CHARS = _env_int("AI_REPAIR_MAX_RAW_CHARS", 16000)

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

AI_PERSISTENT_CACHE_ENABLED = _env_bool("AI_PERSISTENT_CACHE_ENABLED", True)
AI_CACHE_DIR = str(os.environ.get("AI_CACHE_DIR", "data/ai_cache")).strip() or "data/ai_cache"
AI_DISK_LOCK_WAIT_SECONDS = _env_int("AI_DISK_LOCK_WAIT_SECONDS", max(120, AI_CLAUDE_READ_TIMEOUT + 120))
AI_DISK_LOCK_POLL_SECONDS = max(1, _env_int("AI_DISK_LOCK_POLL_SECONDS", 3))
AI_CACHE_STRIP_VOLATILE_KEYS = _env_bool("AI_CACHE_STRIP_VOLATILE_KEYS", True)

ENABLE_EXPERIENCE_AUDIT_CARDS = _env_bool("ENABLE_EXPERIENCE_AUDIT_CARDS", True)
EXPERIENCE_AUDIT_MAX_CARDS = max(0, _env_int("EXPERIENCE_AUDIT_MAX_CARDS", 12))
EXPERIENCE_AUDIT_MIN_WEIGHT = _env_int("EXPERIENCE_AUDIT_MIN_WEIGHT", 4)
EXPERIENCE_AUDIT_INCLUDE_EXTENDED = _env_bool("EXPERIENCE_AUDIT_INCLUDE_EXTENDED", True)

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
        if s == "" or s.lower() in ("none", "nan", "null", "-", "n/a"):
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
    if max_len and max_len > 0:
        return s[:max_len]
    return s


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


def _dict_get_any(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d:
            return d.get(k)
    return default

# ============================================================
# 比分与方向
# ============================================================

def _normalize_score_text(s: Any) -> str:
    return str(s).strip().replace(" ", "").replace("：", "-").replace(":", "-").replace("–", "-").replace("—", "-")


def _parse_score(s: Any) -> Tuple[Optional[int], Optional[int]]:
    try:
        ss = _normalize_score_text(s)
        if not ss:
            return None, None
        if "胜" in ss and "其他" in ss:
            return 9, 0
        if "平" in ss and "其他" in ss:
            return 9, 9
        if "负" in ss and "其他" in ss:
            return 0, 9
        if ss.lower() in ("home", "draw", "away", "abstain") or ss in ("主胜", "客胜", "平局", "胜", "平", "负", "弃权"):
            return None, None
        m = re.search(r"(\d{1,2})\s*[-:：]\s*(\d{1,2})", ss)
        if not m:
            return None, None
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None, None


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
    if h is None:
        return None
    return h + a


def _direction_cn(direction: str) -> str:
    return {"home": "主胜", "draw": "平局", "away": "客胜", "abstain": "弃权"}.get(direction, "弃权")


def _dir_from_cn(v: Any) -> Optional[str]:
    s = str(v).strip().lower()
    if s in ("home", "主胜", "胜", "主", "主队", "home_win", "h"):
        return "home"
    if s in ("draw", "平局", "平", "和", "tie", "x", "d"):
        return "draw"
    if s in ("away", "客胜", "负", "客", "客队", "away_win", "a"):
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

# ============================================================
# change 语义识别
# ============================================================

def _is_change_direction_code(change: Any) -> bool:
    if not isinstance(change, dict) or not change:
        return False
    keys = {"win", "same", "draw", "lose", "home", "away"}
    usable = []
    for k, v in change.items():
        if str(k).strip().lower() not in keys:
            continue
        if v in (None, ""):
            continue
        try:
            fv = float(str(v).strip())
        except Exception:
            return False
        usable.append(fv)
    if not usable:
        return False
    return all(v in (-1.0, 0.0, 1.0) for v in usable)


def _change_value(change: Dict[str, Any], *keys: str) -> float:
    if not isinstance(change, dict):
        return 0.0
    for k in keys:
        if k in change:
            return _f(change.get(k), 0.0)
    return 0.0


def _change_pack(match_data: Dict[str, Any]) -> Dict[str, Any]:
    change = match_data.get("change", {}) or {}
    if not isinstance(change, dict):
        return {
            "raw": {},
            "is_direction_code": False,
            "win": 0.0,
            "same": 0.0,
            "lose": 0.0,
            "desc": "",
        }

    is_code = _is_change_direction_code(change)
    win = _change_value(change, "win", "home")
    same = _change_value(change, "same", "draw")
    lose = _change_value(change, "lose", "away")

    def code_desc(v: float) -> str:
        if v < 0:
            return "降水"
        if v > 0:
            return "升水"
        return "不动"

    desc = ""
    if is_code:
        desc = f"主胜{code_desc(win)} 平{code_desc(same)} 客胜{code_desc(lose)}"

    return {
        "raw": change,
        "is_direction_code": is_code,
        "win": win,
        "same": same,
        "lose": lose,
        "desc": desc,
    }

# ============================================================
# match 规范化
# ============================================================

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

    skip = {"vote", "change", "points", "information", "prediction", "stats", "smart_signals"}
    if m.get("sp_home") is None:
        v = _deep_find_value(m, ["win", "odds_win", "spf_sp3", "sp3", "胜"], skip)
        if v is not None:
            m["sp_home"] = v
            m["win"] = v
    if m.get("sp_draw") is None:
        v = _deep_find_value(m, ["draw", "same", "odds_draw", "spf_sp1", "sp1", "平"], skip)
        if v is not None:
            m["sp_draw"] = v
            m["same"] = v
    if m.get("sp_away") is None:
        v = _deep_find_value(m, ["lose", "away_win", "odds_lose", "spf_sp0", "sp0", "负"], skip)
        if v is not None:
            m["sp_away"] = v
            m["lose"] = v

    if "give_ball" not in m:
        m["give_ball"] = m.get("handicap") or m.get("rq") or m.get("let_ball") or "0"

    if not isinstance(m.get("change"), dict):
        ch = {}
        for src_key, dst_key in [
            ("change_win", "win"), ("cw", "win"),
            ("change_same", "same"), ("cs", "same"), ("change_draw", "same"),
            ("change_lose", "lose"), ("cl", "lose"), ("change_away", "lose"),
        ]:
            if src_key in m:
                ch[dst_key] = m.get(src_key)
        m["change"] = ch

    m["change_pack"] = _change_pack(m)
    return m


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
# 外部情报入口
# ============================================================

async def _fetch_json_or_text(session: aiohttp.ClientSession, method: str, url: str, **kwargs) -> Tuple[bool, Any, str]:
    try:
        async with session.request(method, url, timeout=aiohttp.ClientTimeout(total=30), **kwargs) as r:
            text = await r.text()
            if r.status < 200 or r.status >= 300:
                return False, {"status": r.status, "text": text[:1000]}, "http_error"
            try:
                return True, json.loads(text), "json"
            except Exception:
                return True, text, "text"
    except Exception as e:
        return False, {"error": str(e)[:300]}, "exception"


def _parse_external_endpoints() -> List[str]:
    raw = os.environ.get("EXTERNAL_CONTEXT_ENDPOINTS", "") or os.environ.get("EXTERNAL_CONTEXT_URLS", "")
    if not raw:
        one = os.environ.get("EXTERNAL_CONTEXT_URL", "").strip()
        return [one] if one else []
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in raw.split(",") if x.strip()]


async def build_external_context_for_match(session: aiohttp.ClientSession, match_obj: Dict[str, Any]) -> Dict[str, Any]:
    ctx = {
        "enabled": bool(ENABLE_EXTERNAL_CONTEXT),
        "source_quality": "disabled" if not ENABLE_EXTERNAL_CONTEXT else "missing",
        "items": [],
        "errors": [],
    }
    if not ENABLE_EXTERNAL_CONTEXT:
        return ctx

    payload = {
        "home_team": match_obj.get("home_team", ""),
        "away_team": match_obj.get("away_team", ""),
        "league": match_obj.get("league", match_obj.get("cup", "")),
        "match": match_obj,
    }
    token = os.environ.get("EXTERNAL_CONTEXT_TOKEN", "").strip()
    for url in _parse_external_endpoints()[:3]:
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        ok, data, kind = await _fetch_json_or_text(
            session,
            os.environ.get("EXTERNAL_CONTEXT_METHOD", "POST").upper(),
            url,
            headers=headers,
            json=payload,
        )
        if ok:
            ctx["items"].append({"source": url, "kind": kind, "data": data})
            ctx["source_quality"] = "provider"
        else:
            ctx["errors"].append({"source": url, "data": data})
    return ctx


async def enrich_external_context(match_analyses: List[Dict[str, Any]]) -> None:
    if not ENABLE_EXTERNAL_CONTEXT or aiohttp is None:
        for ma in match_analyses:
            ma["external_context"] = {"enabled": False, "source_quality": "disabled", "items": [], "errors": []}
        return
    async with aiohttp.ClientSession() as session:
        tasks = [build_external_context_for_match(session, ma["match"]) for ma in match_analyses]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for ma, r in zip(match_analyses, results):
            if isinstance(r, Exception):
                ma["external_context"] = {"enabled": True, "source_quality": "error", "items": [], "errors": [str(r)[:300]]}
            else:
                ma["external_context"] = r


def _format_external_context_for_prompt(ctx: Dict[str, Any], limit: int = 2500) -> str:
    if not isinstance(ctx, dict) or not ctx.get("enabled"):
        return "external_context: disabled\n"
    return _json_compact(ctx, limit) + "\n"

# ============================================================
# 经验审计卡：轻量兼容版，只进 prompt，不改结果
# ============================================================

_RULE_NAME = {
    "D01": "大热必死",
    "D08": "强强对话平局率高",
    "D10": "平手盘水位不动易平",
    "D13": "攻防数据接近必防平",
    "D15": "杯赛首轮/淘汰赛保守审计",
    "U04": "受注比例一边倒反向操作",
    "U09": "排名差大但盘口便宜",
    "U10": "赔率剧烈变动",
    "G08": "0球赔率极低信号",
    "G10": "CRS 0-0赔率极低",
    "G11": "双闷队0-0高危",
    "G12": "双攻队大球高危",
    "B_SHARP": "平局Sharp资金突进",
    "B_STEAM": "Steam资金方向",
    "X01": "强客低赔中热复核",
    "X02": "体彩让球强方向受热复核",
    "X03": "方向与总进球/BTTS一致性复核",
}

_RULE_QUESTION = {
    "D01": "热门低赔是否只是名气盘？是否存在防平/防冷？",
    "D08": "强强对话是否更像试探、消耗、保守？",
    "D10": "平手盘不动是否说明双方定价均衡？",
    "D13": "攻防接近是否支持平局或低比分？",
    "D15": "首回合/淘汰赛/两回合语境是否导致保守、平局或低比分？",
    "U04": "受注一边倒是真共识还是反向操作？",
    "U09": "排名差与盘口不匹配，是强队低估还是盘口便宜？",
    "U10": "剧烈变盘是信息/资金驱动还是诱导？",
    "G08": "0球赔率偏低是否提示极小球/0-0？",
    "G10": "0-0赔率低是否与总0球低位共振？",
    "G11": "双闷队是否支持0-0/1-0/0-1？",
    "G12": "双攻/双漏是否支持BTTS与大球？",
    "B_SHARP": "平赔独降且两边升，是真平局资金还是诱平？",
    "B_STEAM": "散户未跟的降水是否更像专业资金？",
    "X01": "强客低赔+客热中高+客赔降水，是真强客还是顺水诱买？",
    "X02": "强方向热度与降水同向，需验证是真资金还是顺水？",
    "X03": "最终比分必须与总进球和BTTS一致。",
}


class ExperienceAuditEngine:
    def _sf(self, val: Any, d: float = 0.0) -> float:
        return _f(val, d)

    def _si(self, val: Any, d: int = 0) -> int:
        return _i(val, d)

    def analyze(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        if not ENABLE_EXPERIENCE_AUDIT_CARDS:
            return {
                "enabled": False,
                "triggered": [],
                "risk_signals": [],
                "total_score": 0,
                "recommendation": "disabled",
                "change_is_direction_code": False,
            }

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

        sp_h = self._sf(match_data.get("sp_home", match_data.get("win")), 2.5)
        sp_d = self._sf(match_data.get("sp_draw", match_data.get("same")), 3.2)
        sp_a = self._sf(match_data.get("sp_away", match_data.get("lose")), 3.5)
        give_ball = self._sf(match_data.get("give_ball", match_data.get("handicap", match_data.get("rq", 0))), 0)
        hr = self._si(match_data.get("home_rank"), 10)
        ar = self._si(match_data.get("away_rank"), 10)
        vote = match_data.get("vote", {}) or {}

        ch_pack = match_data.get("change_pack") if isinstance(match_data.get("change_pack"), dict) else _change_pack(match_data)
        change_is_code = bool(ch_pack.get("is_direction_code"))
        wc = self._sf(ch_pack.get("win"), 0)
        sc = self._sf(ch_pack.get("same"), 0)
        lc = self._sf(ch_pack.get("lose"), 0)

        context_text = " ".join([
            str(match_data.get("league", "")),
            str(match_data.get("cup", "")),
            str(match_data.get("baseface", "")),
            str(match_data.get("expert_intro", "")),
            str(match_data.get("intelligence", "")),
            str(match_data.get("odds_movement", "")),
        ])

        vh = self._si(vote.get("win"), 33)
        va = self._si(vote.get("lose"), 33)
        a0 = self._sf(match_data.get("a0"), 99)
        a1 = self._sf(match_data.get("a1"), 99)
        a2 = self._sf(match_data.get("a2"), 99)
        s00 = self._sf(match_data.get("s00"), 99)

        if sp_h < 1.40 and vh >= 55:
            add("D01", "平局", 8, f"主赔{sp_h}极低+主胜受注{vh}%", "draw")
            risk_signals.append("EXP_D01 大热必死")
        if sp_a < 1.40 and va >= 55:
            add("D01", "平局", 8, f"客赔{sp_a}极低+客胜受注{va}%", "draw")
            risk_signals.append("EXP_D01 大热必死")

        if not change_is_code:
            if abs(give_ball) < 0.1 and abs(wc) < 0.02 and abs(lc) < 0.02:
                add("D10", "平局", 8, "平手盘临场水位几乎不变", "draw")
        else:
            if abs(give_ball) < 0.1 and wc == 0 and lc == 0:
                add("D10", "平局", 8, "平手盘且主客方向编码均不动", "draw")

        if abs(hr - ar) <= 3:
            add("D13", "平局", 5, f"排名接近{hr}vs{ar}", "draw")

        cup_like = any(k in context_text.lower() for k in [
            "杯", "cup", "欧冠", "欧联", "解放者", "南球", "淘汰", "半决赛", "决赛"
        ])
        group_like = any(k in context_text for k in [
            "小组赛", "分组赛", "小组第", "分组赛第"
        ])
        conservative_knockout_like = any(k in context_text for k in [
            "首回合", "首轮", "淘汰赛", "半决赛", "决赛", "两回合"
        ])

        if cup_like and conservative_knockout_like and not group_like:
            add("D15", "平局", 5, "杯赛/淘汰赛保守场景，需审计平局或低比分", "draw")

        if vh >= 65:
            add("U04", "冷门", 8, f"主胜受注{vh}%过热", "upset_away")
            risk_signals.append(f"EXP_U04 主胜超热{vh}%")
        if va >= 65:
            add("U04", "冷门", 8, f"客胜受注{va}%过热", "upset_home")
            risk_signals.append(f"EXP_U04 客胜超热{va}%")

        if abs(hr - ar) >= 8 and abs(give_ball) <= 0.5:
            add("U09", "冷门", 8, f"排名差{abs(hr-ar)}但让球仅{give_ball}", "upset")

        if not change_is_code:
            if max(abs(wc), abs(lc), abs(sc)) >= 0.15:
                add("U10", "盘口", 7, f"最大变动{max(abs(wc), abs(lc), abs(sc)):.2f}", "audit")

        if a0 < 8.5:
            add("G08", "大小球", 7, f"0球@{a0}", "under")
            risk_signals.append(f"EXP_G08 0球@{a0}")
        if s00 < 9.5:
            add("G10", "大小球", 7, f"0-0@{s00}", "zero_zero")

        if change_is_code:
            if sc < 0 and wc > 0 and lc > 0:
                add("B_SHARP", "盘口", 7, "平赔方向编码降水且胜负升水", "draw")
            if wc < 0 and vh < 50:
                add("B_STEAM", "盘口", 7, f"主赔降水方向信号，散户仅{vh}%", "steam_home")
            if lc < 0 and va < 50:
                add("B_STEAM", "盘口", 7, f"客赔降水方向信号，散户仅{va}%", "steam_away")
        else:
            if sc < -0.05 and wc > 0 and lc > 0:
                add("B_SHARP", "盘口", 7, f"平赔降{sc:.2f}且胜负升", "draw")
            if wc < -0.10 and vh < 50:
                add("B_STEAM", "盘口", 7, f"主赔降{wc:.2f}散户仅{vh}%", "steam_home")
            if lc < -0.10 and va < 50:
                add("B_STEAM", "盘口", 7, f"客赔降{lc:.2f}散户仅{va}%", "steam_away")

        if EXPERIENCE_AUDIT_INCLUDE_EXTENDED:
            if change_is_code:
                if sp_a <= 1.75 and 50 <= va < 65 and lc < 0:
                    add("X01", "二次升级", 8, f"客赔{sp_a}低位+客热{va}%+客赔方向降水", "audit_strong_away_heat")
                    risk_signals.append("EXP_X01 强客低赔中热复核")
                if (vh >= 55 and wc < 0) or (va >= 55 and lc < 0):
                    hot = "主" if vh >= 55 and wc < 0 else "客"
                    add("X02", "二次升级", 6, f"{hot}方向热度与降水方向信号同向", "audit_hot_drop")
            else:
                if sp_a <= 1.75 and 50 <= va < 65 and lc < 0:
                    add("X01", "二次升级", 8, f"客赔{sp_a}低位+客热{va}%+客赔降{lc:.2f}", "audit_strong_away_heat")
                    risk_signals.append("EXP_X01 强客低赔中热复核")
                if (vh >= 55 and wc < 0) or (va >= 55 and lc < 0):
                    hot = "主" if vh >= 55 and wc < 0 else "客"
                    add("X02", "二次升级", 6, f"{hot}方向热度与降水同向", "audit_hot_drop")

            if (0 < a1 <= 5.2) or (0 < a2 <= 3.7):
                add("X03", "二次升级", 6, f"1球@{a1}/2球@{a2}，需审计经济型赢球与BTTS", "audit_goal_shape")

        dedup: Dict[str, Dict[str, Any]] = {}
        for t in triggered:
            if t["id"] not in dedup or t["weight"] > dedup[t["id"]]["weight"]:
                dedup[t["id"]] = t

        out = sorted(dedup.values(), key=lambda x: (-int(x.get("weight", 0)), str(x.get("id", ""))))
        if EXPERIENCE_AUDIT_MAX_CARDS > 0:
            out = out[:EXPERIENCE_AUDIT_MAX_CARDS]

        total_score = sum(int(t.get("weight", 0)) for t in out)

        return {
            "enabled": True,
            "mode": "prompt_only_no_probability_change_no_score_change",
            "change_is_direction_code": change_is_code,
            "change_desc": ch_pack.get("desc", ""),
            "triggered": out,
            "triggered_count": len(out),
            "total_score": total_score,
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
    rows = exp.get("triggered", []) or []
    if not rows:
        return '<experience_audit_cards mode="prompt_only_no_decision">无明显历史经验审计卡。</experience_audit_cards>\n'
    p = '<experience_audit_cards mode="prompt_only_no_decision">\n'
    p += "这些卡片只作为审计问题；禁止直接改方向、禁止直接改比分。必须在 audit.experience_review 中说明 accepted/rejected/neutral。\n"
    p += f"change_is_direction_code={exp.get('change_is_direction_code', False)}"
    if exp.get("change_desc"):
        p += f" | change_desc={exp.get('change_desc')}"
    p += "\n"
    for t in rows:
        p += f"- {t.get('id')} {t.get('name')} | 权重={t.get('weight')} | 原因={t.get('reason')} | 问题={t.get('ai_question')}\n"
    p += "</experience_audit_cards>\n"
    return p

# ============================================================
# Prompt 构造
# ============================================================

def _raw_field_line(label: str, value: Any, limit: int = 1200) -> str:
    if value is None or value == "" or value == {} or value == []:
        return ""
    text = _json_compact(value, limit) if isinstance(value, (dict, list)) else str(value)
    if limit and limit > 0:
        text = text[:limit]
    return f"{label}:{text.replace(chr(10), ' ')}\n"


def _raw_full_packet_line(match_obj: Dict[str, Any]) -> str:
    if not INCLUDE_FULL_RAW_PACKET:
        return ""
    limit = RAW_PACKET_CHAR_LIMIT
    raw_json = _json_compact(match_obj, limit if limit and limit > 0 else 0)
    suffix = ""
    try:
        full_len = len(json.dumps(match_obj, ensure_ascii=False, default=str, separators=(",", ":")))
        if limit and limit > 0 and full_len > limit:
            suffix = f"...[TRUNCATED full_len={full_len} limit={limit}]"
    except Exception:
        pass
    return f"raw_match_full_json:{raw_json}{suffix}\n"


def _output_format_rule() -> str:
    return (
        "严格输出 JSON 数组。每场一个对象："
        '{"match":1,"final_direction":"home/draw/away","direction_probs":{"home":45,"draw":28,"away":27},'
        '"goal_band":"0-1/2/3/4+","btts":"yes/no/unclear",'
        '"top3":[{"score":"2-0","prob":18,"market_logic":"..."}],'
        '"reason":"...","ai_confidence":0-100,"risk_level":"low/medium/high","data_missing":[],'
        '"audit":{"odds_source":"体彩竞彩抓包赔率","web_odds_check":"searched/web_search_unavailable/european_odds_missing",'
        '"sharp_money_direction":"home/draw/away/home_or_draw/away_or_draw/unclear","sharp_evidence":"...",'
        '"league_style":"...","team_style":"...","style_score_logic":"...","direction_rejection":"...",'
        '"experience_review":"D01:neutral because ...; X01:accepted because ..."}}\n'
        "禁止 markdown，禁止 JSON 外文本。match 字段优先输出数字序号。\n"
    )


def build_phase1_prompt(match_analyses: List[Dict[str, Any]]) -> str:
    p = "<context>\n"
    p += "你是竞彩足球 RAW-AI 比分预测模型。match_data 中的赔率是中国体彩竞彩抓包赔率，不是欧洲公司均赔。\n"
    p += "你需要基于原始抓包、赔率变动、散户热度、总进球、半全场、经验审计卡、可用联网材料，独立判断方向和比分。\n"
    p += "禁止引用 CRS、贝叶斯、本地矩阵、本地陷阱裁决。经验卡只作为审计问题，不是裁决。\n"
    p += "若没有联网能力，audit.web_odds_check 写 web_search_unavailable，data_missing 加 external_european_odds，禁止假装联网。\n"
    p += "必须输出 direction_probs、goal_band、btts、top3比分；top3[0]必须与final_direction一致。\n"
    p += "change 字段若是 -1/0/1，代表方向编码：-1=降水，0=不动，1=升水；不是赔率变动幅度。\n"
    p += "</context>\n\n"
    p += "<output_format>\n" + _output_format_rule() + "</output_format>\n\n"
    p += "<match_data>\n"

    for i, ma in enumerate(match_analyses, 1):
        m = ma["match"]
        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        league = m.get("league", m.get("cup", ""))
        ch_pack = m.get("change_pack") if isinstance(m.get("change_pack"), dict) else _change_pack(m)

        p += f'<match index="{i}">\n'
        p += f"[{i}] {h} vs {a} | {league}\n"
        p += f"体彩竞彩1X2: 主胜={m.get('sp_home', m.get('win',''))} 平={m.get('sp_draw', m.get('same',''))} 客胜={m.get('sp_away', m.get('lose',''))}\n"
        p += f"让球:{m.get('give_ball', m.get('handicap', m.get('rq','')))}\n"
        p += f"change_semantics: is_direction_code={ch_pack.get('is_direction_code')} desc={ch_pack.get('desc','')}\n"
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
        p += _raw_field_line("odds_movement", m.get("odds_movement"), 1200)
        p += _raw_field_line("intelligence", m.get("intelligence"), FIELD_LIMIT_INFORMATION)
        p += _raw_field_line("expert_intro", m.get("expert_intro"), 3000)
        p += _raw_field_line("baseface", m.get("baseface"), FIELD_LIMIT_INFORMATION)
        p += _raw_field_line("information", m.get("information"), FIELD_LIMIT_INFORMATION)
        p += _raw_field_line("points", m.get("points"), FIELD_LIMIT_POINTS)

        style_pack = {k: v for k, v in m.items() if k in (
            "league_style", "league_profile", "team_style", "home_style", "away_style", "play_style",
            "tactical_style", "pace_rating", "tempo", "home_form", "away_form", "weather", "injury",
            "lineup", "news", "motivation", "schedule", "home_rank", "away_rank", "home_stats", "away_stats",
        )}
        p += _raw_field_line("style_and_team_core", style_pack, FIELD_LIMIT_STYLE_EXTRA)
        p += _raw_full_packet_line(m)

        exp = ma.get("experience_audit") or _experience_engine().analyze(m)
        p += _format_experience_audit_for_prompt(exp)
        p += "<external_context>\n" + _format_external_context_for_prompt(ma.get("external_context", {})) + "</external_context>\n"
        p += "</match>\n\n"

    p += "</match_data>\n"
    return p


def build_compact_claude_match_data(match_analyses: List[Dict[str, Any]]) -> str:
    p = "<compact_match_data_for_claude>\n"
    p += "Claude终审压缩抓包：Phase1三家已经看过完整抓包。这里保留核心原始字段、赔率结构、经验审计卡，避免prompt过大导致断流。\n"
    p += "这些字段仍然是体彩竞彩抓包赔率，不是欧洲欧赔。Claude必须重新审计，不按票数机械裁决。\n"
    p += "change 字段若是 -1/0/1，代表方向编码：-1=降水，0=不动，1=升水；不是赔率变动幅度。\n\n"

    for i, ma in enumerate(match_analyses, 1):
        m = ma["match"]
        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        league = m.get("league", m.get("cup", ""))
        ch_pack = m.get("change_pack") if isinstance(m.get("change_pack"), dict) else _change_pack(m)

        p += f'<match index="{i}">\n'
        p += f"[{i}] {h} vs {a} | {league}\n"
        p += f"match_num:{m.get('match_num','')} | time:{m.get('match_time', m.get('time',''))}\n"
        p += f"体彩竞彩1X2: 主胜={m.get('sp_home', m.get('win',''))} 平={m.get('sp_draw', m.get('same',''))} 客胜={m.get('sp_away', m.get('lose',''))}\n"
        p += f"让球:{m.get('give_ball', m.get('handicap', m.get('rq','')))}\n"
        p += f"change_semantics: is_direction_code={ch_pack.get('is_direction_code')} desc={ch_pack.get('desc','')}\n"
        p += "总进球a0-a7:" + " | ".join([f"{g}={m.get(f'a{g}','')}" for g in range(8)]) + "\n"

        hf_l = []
        for k, lb in HFTF_MAP.items():
            v = m.get(k, None)
            if v not in (None, "", 0, "0"):
                hf_l.append(f"{lb}={v}")
        if hf_l:
            p += "半全场:" + " | ".join(hf_l[:9]) + "\n"

        p += _raw_field_line("change", m.get("change"), 1800)
        p += _raw_field_line("vote", m.get("vote"), 1600)
        p += _raw_field_line("odds_movement", m.get("odds_movement"), 1200)
        p += _raw_field_line("intelligence", m.get("intelligence"), CLAUDE_COMPACT_FIELD_LIMIT)
        p += _raw_field_line("expert_intro", m.get("expert_intro"), CLAUDE_COMPACT_FIELD_LIMIT)
        p += _raw_field_line("baseface", m.get("baseface"), CLAUDE_COMPACT_FIELD_LIMIT)
        p += _raw_field_line("information", m.get("information"), CLAUDE_COMPACT_FIELD_LIMIT)
        p += _raw_field_line("points", m.get("points"), CLAUDE_COMPACT_FIELD_LIMIT)

        style_pack = {k: v for k, v in m.items() if k in (
            "league_style", "league_profile", "team_style", "home_style", "away_style", "play_style",
            "tactical_style", "pace_rating", "tempo", "home_form", "away_form", "weather", "injury",
            "lineup", "news", "motivation", "schedule", "home_rank", "away_rank", "home_stats", "away_stats",
        )}
        p += _raw_field_line("style_and_team_core", style_pack, CLAUDE_COMPACT_FIELD_LIMIT)

        exp = ma.get("experience_audit") or _experience_engine().analyze(m)
        p += _format_experience_audit_for_prompt(exp)
        p += "</match>\n\n"

    p += "</compact_match_data_for_claude>\n"
    return p


def _short_ai_row(r: Dict[str, Any], idx: int) -> Dict[str, Any]:
    audit = r.get("audit", {}) if isinstance(r.get("audit", {}), dict) else {}
    keep_audit = {}
    for k in [
        "web_odds_check", "european_odds", "market_divergence", "sharp_money_direction", "sharp_evidence",
        "league_style", "team_style", "style_score_logic", "direction_rejection", "total_goals", "money_flow",
        "experience_review",
    ]:
        if k in audit:
            keep_audit[k] = audit[k]

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
        "audit": keep_audit,
        "data_missing": r.get("data_missing", []),
    }


def build_claude_final_audit_prompt(match_analyses: List[Dict[str, Any]], phase1_results: Dict[str, Dict[int, Dict[str, Any]]]) -> str:
    p = "<final_audit_context>\n"
    p += "你是 Claude 最终 RAW-AI 主裁。你看到的是体彩竞彩抓包核心字段和 GPT/Grok/Gemini 初审。\n"
    p += "你不是反指模型，不需要为了审计而反对初审。选择证据最完整、最符合原始字段的一组。\n"
    p += "必须输出 JSON 数组；字段同 phase1；禁止 JSON 外文本。\n"
    p += "必须复核：方向、direction_probs、goal_band、btts、top3比分、experience_review。\n"
    p += "如果改动初审比分，必须指出硬依据；不能无证据从2-0改2-1或从1-1改0-0。\n"
    p += "若三家 Phase1 完全同比分一致，只有原始抓包存在硬反证时才允许改比分；否则应保留一致结论并补充审计原因。\n"
    p += "</final_audit_context>\n\n"

    if AI_USE_COMPACT_CLAUDE_AUDIT:
        p += build_compact_claude_match_data(match_analyses)
    else:
        p += build_phase1_prompt(match_analyses)

    p += "\n<phase1_ai_results>\n"
    for ai in PHASE1_NAMES:
        p += f"<{ai}>\n"
        rs = phase1_results.get(ai, {}) or {}
        for idx in range(1, len(match_analyses) + 1):
            r = rs.get(idx)
            if r:
                p += json.dumps(_short_ai_row(r, idx), ensure_ascii=False, separators=(",", ":")) + "\n"
            else:
                p += json.dumps({"match": idx, "abstain": True}, ensure_ascii=False) + "\n"
        p += f"</{ai}>\n"
    p += "</phase1_ai_results>\n\n"
    p += "<output_format>\n" + _output_format_rule() + "</output_format>\n"
    return p

# ============================================================
# URL / KEY
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
        return _clean_env_key(
            "API_KEY",
            "GPT_API_KEY",
            "OPENAI_API_KEY",
            "GROK_API_KEY",
            "GEMINI_API_KEY",
            "CLAUDE_API_KEY",
        )
    return _clean_env_key("API_KEY", f"{ai_name.upper()}_API_KEY", "OPENAI_API_KEY", "GPT_API_KEY")


def get_url_for_ai(ai_name: str) -> str:
    if AI_FORCE_COMMON_GATEWAY:
        return _clean_env_url(
            "API_URL",
            "GPT_API_URL",
            "OPENAI_API_URL",
            "BASE_URL",
            "GROK_API_URL",
            "GEMINI_API_URL",
            "CLAUDE_API_URL",
        )
    return _clean_env_url("API_URL", f"{ai_name.upper()}_API_URL", "OPENAI_API_URL", "BASE_URL", "GPT_API_URL")


def get_common_key() -> str:
    return get_key_for_ai("gpt")


def get_common_url() -> str:
    return get_url_for_ai("gpt")


def _model_for(ai_name: str) -> str:
    env_name = f"{ai_name.upper()}_MODEL"
    return str(os.environ.get(env_name, DEFAULT_MODELS[ai_name])).strip() or DEFAULT_MODELS[ai_name]


def _chat_url(base_url: str) -> str:
    u = (base_url or "").rstrip("/")
    if not u:
        return ""
    if u.endswith("/chat/completions"):
        return u
    if "/chat/completions" in u:
        return u
    return u + "/chat/completions"


def debug_ai_config() -> None:
    key = get_common_key()
    url = get_common_url()
    print(f"[COMMON GATEWAY] API_URL={url or '<missing>'} API_KEY={_mask_key(key)} force_common={AI_FORCE_COMMON_GATEWAY}")
    for n in AI_NAMES:
        print(f"[AI CONFIG] {n.upper()} model={_model_for(n)} timeout={AI_CLAUDE_READ_TIMEOUT if n == 'claude' else AI_READ_TIMEOUT}s")
    print(
        f"[AI MODE] compact_claude={AI_USE_COMPACT_CLAUDE_AUDIT} "
        f"cache_ttl={AI_DECISION_CACHE_TTL} max_req_per_ai={AI_MAX_REQUESTS_PER_AI} "
        f"claude_repair={AI_ENABLE_CLAUDE_JSON_REPAIR}"
    )

# ============================================================
# 顶级解析器
# ============================================================

_DIRECTION_RE = re.compile(
    r"(final_direction|direction|方向|赛果|结果)\s*[:：=]\s*['\"]?(home|draw|away|主胜|平局|客胜|胜|平|负)",
    re.I,
)
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
                if "top3" in t or "top_3" in t or "比分" in t:
                    score += 8
                if "final_direction" in t or "direction_probs" in t:
                    score += 7
                if re.search(r"\[\s*\{", t):
                    score += 6
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
                if kl in {"content", "text", "output_text", "answer", "result", "response", "model_response", "final_answer"}:
                    add(v, f"{path}.{k}", bonus=2)
                walk(v, f"{path}.{k}", depth + 1)
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:120]):
                walk(v, f"{path}[{i}]", depth + 1)
        elif isinstance(obj, str):
            s = obj.strip()
            if len(s) >= 2:
                add(s, path)

    try:
        if isinstance(data, dict):
            for ch in data.get("choices", []) or []:
                if not isinstance(ch, dict):
                    continue
                msg = ch.get("message", {}) or {}
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        add(content, "choices.message.content", bonus=10)
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                add(item.get("text"), "choices.message.content[].text", bonus=10)
                                add(item.get("content"), "choices.message.content[].content", bonus=8)
                    for k in ["text", "answer", "response", "output_text", "final_answer", "output", "result", "model_response"]:
                        add(msg.get(k), f"choices.message.{k}", bonus=5)
                add(ch.get("text"), "choices.text", bonus=5)

            add(data.get("output_text"), "output_text", bonus=10)
            add(data.get("text"), "text", bonus=5)
            add(data.get("answer"), "answer", bonus=5)
            add(data.get("result"), "result", bonus=5)

            if isinstance(data.get("candidates"), list):
                for cand in data["candidates"]:
                    if isinstance(cand, dict):
                        cont = cand.get("content", {})
                        if isinstance(cont, dict):
                            for part in cont.get("parts", []) or []:
                                if isinstance(part, dict):
                                    add(part.get("text"), "candidates.content.parts.text", bonus=10)

            walk(data)

        elif isinstance(data, str):
            add(data, "raw", bonus=5)

    except Exception as e:
        print(f"    响应文本提取异常:{str(e)[:120]}")

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


def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)


def _quote_unquoted_keys(s: str) -> str:
    return re.sub(r"([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', s)


def _json_loads_best_effort(s: str) -> Any:
    raw = s.strip()
    variants = [
        raw,
        _remove_trailing_commas(raw),
        _quote_unquoted_keys(_remove_trailing_commas(raw)),
    ]
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
    frags: List[str] = []
    clean = _preclean_text(text)
    for start, ch0 in [(m.start(), m.group(0)) for m in re.finditer(r"[\[\{]", clean)]:
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
            if any(k in frag for k in ["top3", "final_direction", "direction_probs", "score", "match", "比分", "predictions", "results"]):
                frags.append(frag)

    frags.sort(key=len, reverse=True)
    return frags[:30]


def _object_lines(text: str) -> List[Dict[str, Any]]:
    out = []
    clean = _preclean_text(text)
    for line in clean.splitlines():
        line = line.strip().rstrip(",")
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = _json_loads_best_effort(line)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def _looks_like_prediction_item(x: Any) -> bool:
    if not isinstance(x, dict):
        return False
    keys = set(str(k) for k in x.keys())
    return bool(keys & {
        "match", "index", "match_index", "id", "序号",
        "top3", "top_3", "top_scores", "scores", "score_candidates",
        "score", "predicted_score", "ai_score", "final_score",
        "final_direction", "direction", "direction_probs", "direction_probabilities",
        "goal_band", "btts", "audit",
    })


def _unwrap_json_result(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj

    if not isinstance(obj, dict):
        return []

    wrapper_keys = [
        "predictions",
        "prediction_results",
        "results",
        "matches",
        "match_results",
        "games",
        "items",
        "data",
        "output",
        "outputs",
        "response",
        "responses",
        "result",
        "prediction",
        "final",
        "final_answer",
    ]

    for k in wrapper_keys:
        v = obj.get(k)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            inner = _unwrap_json_result(v)
            if inner:
                return inner

    if _looks_like_prediction_item(obj):
        return [obj]

    numeric_items = []
    for k, v in obj.items():
        if str(k).strip().isdigit() and isinstance(v, dict):
            vv = dict(v)
            vv.setdefault("match", int(str(k).strip()))
            numeric_items.append(vv)
    if numeric_items:
        return numeric_items

    object_items = []
    for k, v in obj.items():
        if isinstance(v, dict) and _looks_like_prediction_item(v):
            vv = dict(v)
            vv.setdefault("match_label", str(k))
            object_items.append(vv)
    if object_items:
        return object_items

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
    if best:
        return best

    lines = _object_lines(clean)
    if lines:
        return lines

    obj_frags = [f for f in _balanced_fragments(clean) if f.startswith("{")]
    objs = []
    for f in obj_frags:
        try:
            o = _json_loads_best_effort(f)
            if isinstance(o, dict):
                objs.append(o)
        except Exception:
            pass

    if objs:
        uniq = []
        seen = set()
        for o in objs:
            h = _hash_obj(o)
            if h not in seen:
                seen.add(h)
                uniq.append(o)
        return uniq

    return []


def _prob_to_float(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, str):
        s = v.strip().replace("％", "%")
        pct = "%" in s
        fv = _f(s, 0.0)
        if pct:
            return fv
        if 0 < fv <= 1:
            return fv * 100
        return fv
    fv = _f(v, 0.0)
    if 0 < fv <= 1:
        return fv * 100
    return fv


def _score_from_candidate(obj: Any) -> str:
    if isinstance(obj, str):
        m = _SCORE_RE.search(obj)
        if m:
            return f"{int(m.group(1))}-{int(m.group(2))}"
        return _normalize_score_text(obj)

    if not isinstance(obj, dict):
        return ""

    for k in [
        "score", "predicted_score", "ai_score", "final_score", "比分", "预测比分", "top_score", "result_score",
        "correct_score", "scoreline", "prediction_score", "预测", "赛果比分",
    ]:
        if obj.get(k) not in (None, ""):
            s = _normalize_score_text(obj.get(k))
            m = _SCORE_RE.search(s)
            if m:
                return f"{int(m.group(1))}-{int(m.group(2))}"
            return s

    for hk, ak in [("home_goals", "away_goals"), ("home_score", "away_score"), ("主队进球", "客队进球")]:
        if obj.get(hk) is not None and obj.get(ak) is not None:
            return f"{_i(obj.get(hk))}-{_i(obj.get(ak))}"

    return ""


def _normalize_top3(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_top3 = None
    for k in [
        "top3", "top_3", "top_scores", "scores", "score_candidates", "candidates", "比分候选",
        "score_distribution", "correct_scores", "candidate_scores", "topScore", "top_three",
    ]:
        if isinstance(item.get(k), list):
            raw_top3 = item[k]
            break

    if raw_top3 is None:
        raw_top3 = []

    top3: List[Dict[str, Any]] = []
    seen = set()

    for cand in raw_top3[:10]:
        sc = _score_from_candidate(cand)
        if _parse_score(sc)[0] is None:
            continue
        if sc in seen:
            continue
        seen.add(sc)

        if isinstance(cand, dict):
            prob = cand.get("prob", cand.get("probability", cand.get("pct", cand.get("chance", cand.get("confidence", 0)))))
            logic = cand.get("market_logic", cand.get("reason", cand.get("logic", cand.get("explanation", ""))))
            top3.append({"score": sc, "prob": round(_prob_to_float(prob), 3), "market_logic": str(logic)[:600]})
        else:
            top3.append({"score": sc, "prob": 0.0, "market_logic": ""})

        if len(top3) >= 3:
            break

    if not top3:
        sc = _score_from_candidate(item)
        if _parse_score(sc)[0] is not None:
            prob = item.get("prob", item.get("probability", item.get("score_probability", 0)))
            top3 = [{"score": sc, "prob": round(_prob_to_float(prob), 3), "market_logic": ""}]

    return top3


def _normalize_direction(v: Any, top_score: str = "") -> str:
    d = _dir_from_cn(v)
    if d:
        return d
    sd = _score_direction(top_score)
    return sd or "draw"


def _normalize_ai_direction_probs(obj: Any) -> Dict[str, float]:
    if not isinstance(obj, dict):
        return {}

    cand = None
    for k in [
        "direction_probs", "direction_probabilities", "probabilities", "direction_probability",
        "方向概率", "三项概率", "win_draw_loss_probs",
    ]:
        if isinstance(obj.get(k), dict):
            cand = obj.get(k)
            break

    if cand is None and isinstance(obj.get("audit"), dict):
        for k in ["direction_probs", "direction_probabilities", "probabilities", "方向概率", "三项概率"]:
            if isinstance(obj["audit"].get(k), dict):
                cand = obj["audit"].get(k)
                break

    if not isinstance(cand, dict):
        keys = ["home_win_pct", "draw_pct", "away_win_pct"]
        if any(k in obj for k in keys):
            cand = {"home": obj.get("home_win_pct"), "draw": obj.get("draw_pct"), "away": obj.get("away_win_pct")}
        else:
            return {}

    raw = {"home": 0.0, "draw": 0.0, "away": 0.0}
    alias = {
        "home": "home", "主": "home", "主胜": "home", "胜": "home", "home_win": "home", "win": "home",
        "draw": "draw", "平": "draw", "平局": "draw", "和": "draw", "same": "draw", "tie": "draw",
        "away": "away", "客": "away", "客胜": "away", "负": "away", "away_win": "away", "lose": "away",
    }

    for k, v in cand.items():
        kk = alias.get(str(k).strip().lower(), alias.get(str(k).strip()))
        if kk in raw:
            raw[kk] += _prob_to_float(v)

    if sum(raw.values()) <= 0:
        return {}

    s = sum(raw.values())
    return {k: round(v / s * 100, 1) for k, v in raw.items()}


def _match_index_from_item(item: Dict[str, Any], fallback_idx: int, num_matches: int) -> Optional[int]:
    raw = item.get("match", item.get("index", item.get("match_index", item.get("id", item.get("序号")))))

    if isinstance(raw, int):
        return raw if 1 <= raw <= num_matches else None

    if raw is not None:
        s = str(raw).strip()
        m = (
            re.match(r"^\s*(\d+)\s*$", s)
            or re.match(r"^\s*\[(\d+)\]", s)
            or re.search(r"(?:match|场次|第)\s*(\d+)", s, re.I)
        )
        if m:
            idx = int(m.group(1))
            return idx if 1 <= idx <= num_matches else None

    return fallback_idx if 1 <= fallback_idx <= num_matches else None


def _normalize_goal_band_value(v: Any, top_score: str = "") -> str:
    s = str(v or "").strip().lower().replace(" ", "")
    aliases = {
        "0-1": "0-1", "0_1": "0-1", "0~1": "0-1", "0至1": "0-1", "0到1": "0-1",
        "0/1": "0-1", "low": "0-1", "小球": "0-1",
        "2": "2", "2球": "2", "two": "2",
        "3": "3", "3球": "3", "three": "3",
        "4+": "4+", "4plus": "4+", "4以上": "4+", "4球+": "4+", "high": "4+", "大球": "4+",
    }
    if s in aliases:
        return aliases[s]

    total = _score_total(top_score)
    if total is None:
        return ""
    if total <= 1:
        return "0-1"
    if total == 2:
        return "2"
    if total == 3:
        return "3"
    return "4+"


def _normalize_btts_value(v: Any, top_score: str = "") -> str:
    s = str(v or "").strip().lower()
    if s in ("yes", "y", "true", "1", "是", "双方进球", "btts_yes"):
        return "yes"
    if s in ("no", "n", "false", "0", "否", "不是", "btts_no"):
        return "no"

    h, a = _parse_score(top_score)
    if h is not None and a is not None:
        return "yes" if h > 0 and a > 0 else "no"

    return "unclear"


def _normalize_experience_review(item: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = item.get("experience_review")
    if raw is None and isinstance(item.get("audit"), dict):
        raw = item["audit"].get("experience_review")

    out: List[Dict[str, str]] = []

    if isinstance(raw, list):
        for r in raw[:30]:
            if isinstance(r, dict):
                rid = str(r.get("id", r.get("rule_id", ""))).strip()
                dec = str(r.get("decision", r.get("status", "neutral"))).strip().lower()
                if dec not in ("accepted", "rejected", "neutral"):
                    dec = "neutral"
                reason = str(r.get("reason", r.get("why", "")))[:600]
                if rid:
                    out.append({"id": rid.replace("EXP_", ""), "decision": dec, "reason": reason})

    elif isinstance(raw, str) and raw.strip():
        parts = re.split(r"[;；\n]+", raw)
        for p in parts[:30]:
            m = re.search(r"(EXP_[A-Z0-9_]+|[DUGBMLFX]\d{2}|B_SHARP|B_STEAM|X\d{2})", p)
            if not m:
                continue
            dec = "neutral"
            low = p.lower()
            if "accepted" in low or "接受" in p or "采纳" in p:
                dec = "accepted"
            elif "rejected" in low or "驳回" in p or "不采纳" in p:
                dec = "rejected"
            out.append({"id": m.group(1).replace("EXP_", ""), "decision": dec, "reason": p[:600]})

    return out


def _fallback_parse_text_blocks(raw_text: str, num_matches: int) -> Dict[int, Dict[str, Any]]:
    clean = _preclean_text(raw_text)
    results: Dict[int, Dict[str, Any]] = {}
    parts = re.split(r"(?=\n?\s*(?:\[?\d+\]?|match\s*\d+|第\s*\d+\s*场)[\].、:：\s])", clean, flags=re.I)
    if len(parts) <= 1:
        parts = [clean]

    for pos, part in enumerate(parts, 1):
        if not part.strip():
            continue

        idx = None
        m_idx = re.search(r"(?:match\s*|第\s*)?(\d{1,2})(?:\s*场)?", part, flags=re.I)
        if m_idx:
            cand_idx = int(m_idx.group(1))
            if 1 <= cand_idx <= num_matches:
                idx = cand_idx

        if idx is None:
            idx = pos if 1 <= pos <= num_matches else None

        if idx is None or idx in results:
            continue

        m_score = _SCORE_RE.search(part)
        if not m_score:
            continue

        score = f"{int(m_score.group(1))}-{int(m_score.group(2))}"
        direction = _score_direction(score) or "draw"
        conf_match = re.search(r"(?:confidence|置信度|ai_confidence)\D{0,8}(\d{1,3})", part, flags=re.I)
        conf = int(_clip(_f(conf_match.group(1), 60) if conf_match else 60, 0, 100))

        results[idx] = {
            "top3": [{"score": score, "prob": 0.0, "market_logic": "text_fallback"}],
            "ai_score": score,
            "reason": part[:4000],
            "ai_confidence": conf,
            "risk_level": "medium",
            "is_score_others": False,
            "detected_traps": [],
            "data_missing": ["json_format_missing_text_fallback"],
            "audit": {},
            "direction_probs": {},
            "goal_band": _normalize_goal_band_value("", score),
            "btts": _normalize_btts_value("", score),
            "score_shape_reason": "text_fallback_parser",
            "experience_review": [],
            "final_direction": direction,
            "raw_item": {"text_fallback": part[:1000]},
        }

    return results


def _parse_ai_json(raw_text: str, num_matches: int, ai_name: str = "") -> Dict[int, Dict[str, Any]]:
    items = _extract_json_items(raw_text)
    results: Dict[int, Dict[str, Any]] = {}

    if not isinstance(items, list):
        items = []

    for pos, item in enumerate(items, 1):
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

        final_direction = _normalize_direction(
            item.get("final_direction", item.get("direction", item.get("result", item.get("赛果", "")))),
            top_score,
        )
        score_dir = _score_direction(top_score)
        if score_dir in VALID_DIRS:
            final_direction = score_dir

        conf = item.get("ai_confidence", item.get("confidence", item.get("conf", item.get("置信度", 60))))
        traps = item.get("detected_traps", item.get("traps", item.get("risk_flags", [])))
        if not isinstance(traps, list):
            traps = [str(traps)] if traps else []

        data_missing = item.get("data_missing", [])
        if not isinstance(data_missing, list):
            data_missing = [str(data_missing)] if data_missing else []

        audit = item.get("audit", {}) if isinstance(item.get("audit", {}), dict) else {}

        results[mid] = {
            "top3": top3,
            "ai_score": top_score,
            "reason": str(item.get("reason", item.get("analysis", item.get("explanation", item.get("理由", "")))))[:5000],
            "ai_confidence": int(_clip(_f(conf, 60), 0, 100)),
            "risk_level": str(item.get("risk_level", item.get("risk", item.get("风险", "medium")))),
            "is_score_others": _score_display_label(top_score) in ("胜其他", "平其他", "负其他"),
            "detected_traps": traps,
            "data_missing": data_missing,
            "audit": audit,
            "direction_probs": _normalize_ai_direction_probs(item),
            "goal_band": _normalize_goal_band_value(
                item.get("goal_band", item.get("goal_range", item.get("total_goals_band", item.get("goal_interval", "")))),
                top_score,
            ),
            "btts": _normalize_btts_value(
                item.get("btts", item.get("both_score", item.get("both_teams_score", item.get("双方进球", "")))),
                top_score,
            ),
            "score_shape_reason": str(item.get(
                "score_shape_reason",
                item.get("score_logic", item.get("score_reason", audit.get("style_score_logic", ""))),
            ))[:1500],
            "experience_review": _normalize_experience_review(item),
            "final_direction": final_direction,
            "raw_item": item,
        }

    if not results:
        allow_text = AI_ALLOW_TEXT_FALLBACK
        if str(ai_name).lower().startswith("claude"):
            allow_text = AI_ALLOW_CLAUDE_TEXT_FALLBACK
        if allow_text:
            results = _fallback_parse_text_blocks(raw_text, num_matches)

    if AI_PARSE_DEBUG and not results:
        print(f"    [{ai_name}] parse empty. raw={raw_text[:500]}")

    return results

# ============================================================
# AI 调用
# ============================================================

async def _repair_ai_json_output(
    session: aiohttp.ClientSession,
    ai_name: str,
    raw_text: str,
    num_matches: int,
    url: str,
    headers: Dict[str, str],
    model: str,
) -> Dict[int, Dict[str, Any]]:
    if not raw_text:
        return {}

    if ai_name != "claude" and not AI_ENABLE_ANY_MODEL_JSON_REPAIR:
        return {}

    if ai_name == "claude" and not AI_ENABLE_CLAUDE_JSON_REPAIR:
        return {}

    raw_cut = raw_text[:AI_REPAIR_MAX_RAW_CHARS]

    repair_prompt = (
        "你只执行 JSON 修复任务。\n"
        "下面是一段足球预测模型输出，可能包含自然语言、markdown、外层字段、中文字段或格式错误。\n"
        "请把它转换为严格 JSON 数组，不允许输出 JSON 之外的任何文字。\n"
        "数组长度必须尽量等于比赛场次数。\n"
        "每个对象必须包含字段：\n"
        "match, final_direction, direction_probs, goal_band, btts, top3, reason, ai_confidence, risk_level, data_missing, audit。\n"
        "final_direction 只能是 home/draw/away。\n"
        "top3 必须是数组，元素格式为 {\"score\":\"2-0\",\"prob\":18,\"market_logic\":\"...\"}。\n"
        "如果原文没有某字段，请补空值；不得改变原文中的最终方向和比分结论。\n\n"
        f"比赛场次数：{num_matches}\n\n"
        "<raw_model_output>\n"
        f"{raw_cut}\n"
        "</raw_model_output>\n"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是 JSON 格式修复器，只输出严格 JSON 数组，禁止解释。"},
            {"role": "user", "content": repair_prompt},
        ],
        "temperature": 0.0,
    }

    try:
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=AI_CLAUDE_CONNECT_TIMEOUT if ai_name == "claude" else AI_CONNECT_TIMEOUT,
            sock_connect=AI_CLAUDE_CONNECT_TIMEOUT if ai_name == "claude" else AI_CONNECT_TIMEOUT,
            sock_read=AI_CLAUDE_READ_TIMEOUT if ai_name == "claude" else AI_READ_TIMEOUT,
        )

        print(f"    {ai_name.upper()} parse0 → 启动 JSON repair")

        async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
            if r.status != 200:
                text_for_error = await r.text()
                print(f"    {ai_name.upper()} JSON repair HTTP {r.status}: {text_for_error[:260]}")
                return {}

            try:
                data = await r.json(content_type=None)
            except Exception:
                text = await r.text()
                data = {"raw": text.strip()}

            repair_text = _extract_response_text(data, ai_name + "_repair")
            if not repair_text:
                _save_debug_dump(ai_name, data, "repair_empty", "")
                return {}

            parsed = _parse_ai_json(repair_text, num_matches, ai_name + "_repair")
            if parsed:
                print(f"    {ai_name.upper()} JSON repair 成功: {len(parsed)}/{num_matches}")
                return parsed

            print(f"    {ai_name.upper()} JSON repair 仍解析0条，raw前300字: {repair_text[:300].replace(chr(10), ' ')}")
            _save_debug_dump(ai_name, data, "repair_parse0", repair_text)
            return {}

    except Exception as e:
        print(f"    {ai_name.upper()} JSON repair 异常: {str(e)[:220]}")
        return {}


async def async_call_one_ai_batch(
    session: aiohttp.ClientSession,
    prompt: str,
    num_matches: int,
    ai_name: str,
    system_text: str,
) -> Tuple[str, Dict[int, Dict[str, Any]], str]:
    key = get_key_for_ai(ai_name)
    base_url = get_url_for_ai(ai_name)
    model = _model_for(ai_name)

    AI_CALL_STATUS[ai_name] = {
        "ok": False,
        "status": "init",
        "model": model,
        "count": 0,
        "requests": 0,
        "parse_mode": "",
    }

    if not key:
        print(f"  [{ai_name.upper()}] no_key: 检查 API_KEY / {ai_name.upper()}_API_KEY / GPT_API_KEY")
        AI_CALL_STATUS[ai_name].update({"status": "no_key"})
        return ai_name, {}, "no_key"

    if not base_url:
        print(f"  [{ai_name.upper()}] no_url: 检查 API_URL / {ai_name.upper()}_API_URL / GPT_API_URL")
        AI_CALL_STATUS[ai_name].update({"status": "no_url"})
        return ai_name, {}, "no_url"

    url = _chat_url(base_url)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.12 if ai_name == "claude" else 0.18 if ai_name in ("gpt", "gemini") else 0.22,
    }

    for req_no in range(1, AI_MAX_REQUESTS_PER_AI + 1):
        AI_CALL_STATUS[ai_name]["requests"] = req_no
        gateway = url.split("/v1")[0][:80]
        print(f"  [连接中] {ai_name.upper()} | {model} @ {gateway} | request#{req_no}")

        t0 = time.time()

        try:
            read_timeout = AI_CLAUDE_READ_TIMEOUT if ai_name == "claude" else AI_READ_TIMEOUT
            connect_timeout = AI_CLAUDE_CONNECT_TIMEOUT if ai_name == "claude" else AI_CONNECT_TIMEOUT
            timeout = aiohttp.ClientTimeout(
                total=None,
                connect=connect_timeout,
                sock_connect=connect_timeout,
                sock_read=read_timeout,
            )

            async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                elapsed = round(time.time() - t0, 1)

                if r.status != 200:
                    try:
                        text_for_error = await r.text()
                    except Exception:
                        text_for_error = ""
                    print(f"    HTTP {r.status} | {elapsed}s | {text_for_error[:260]}")
                    AI_CALL_STATUS[ai_name].update({
                        "status": f"http_{r.status}",
                        "http_error": text_for_error[:500],
                        "elapsed": elapsed,
                    })
                    continue

                try:
                    data = await r.json(content_type=None)
                except Exception:
                    text = await r.text()
                    data = {"raw": text.strip()}

                raw_text = _extract_response_text(data, ai_name)

                if AI_SAVE_RAW_RESPONSE:
                    _save_debug_dump(ai_name, data, "raw_saved", raw_text)

                if not raw_text:
                    print("    空文本响应")
                    _save_debug_dump(ai_name, data, "empty", "")
                    AI_CALL_STATUS[ai_name].update({
                        "status": "empty",
                        "elapsed": elapsed,
                    })
                    continue

                parsed = _parse_ai_json(raw_text, num_matches, ai_name)

                if parsed:
                    print(f"    {ai_name.upper()} 完成: {len(parsed)}/{num_matches} | {round(time.time()-t0, 1)}s | parse=json")
                    AI_CALL_STATUS[ai_name].update({
                        "ok": True,
                        "status": "ok",
                        "count": len(parsed),
                        "model": model,
                        "parse_mode": "json",
                        "elapsed": round(time.time() - t0, 1),
                    })
                    return ai_name, parsed, model

                repaired = await _repair_ai_json_output(
                    session=session,
                    ai_name=ai_name,
                    raw_text=raw_text,
                    num_matches=num_matches,
                    url=url,
                    headers=headers,
                    model=model,
                )

                if repaired:
                    print(f"    {ai_name.upper()} 完成: {len(repaired)}/{num_matches} | {round(time.time()-t0, 1)}s | parse=json_repair")
                    AI_CALL_STATUS[ai_name].update({
                        "ok": True,
                        "status": "ok_repaired",
                        "count": len(repaired),
                        "model": model,
                        "parse_mode": "json_repair",
                        "elapsed": round(time.time() - t0, 1),
                    })
                    return ai_name, repaired, model

                print(f"    {ai_name.upper()} 结构化解析0条，raw前500字: {raw_text[:500].replace(chr(10), ' ')}")

                if ai_name == "claude":
                    print("    CLAUDE 有返回但未解析成功：已保存 data/debug/claude_parse0_raw_*.txt")
                    print("    常见原因：自然语言审计、外层字段不兼容、JSON截断、match序号缺失、返回被中转包装。")

                _save_debug_dump(ai_name, data, "parse0", raw_text)
                AI_CALL_STATUS[ai_name].update({
                    "status": "parse0_strict_json",
                    "raw_preview": raw_text[:500],
                    "model": model,
                    "elapsed": round(time.time() - t0, 1),
                })

        except asyncio.TimeoutError:
            print(f"    {ai_name.upper()} 读取超时")
            AI_CALL_STATUS[ai_name].update({"status": "timeout"})
        except Exception as e:
            print(f"    {ai_name.upper()} 调用异常: {str(e)[:220]}")
            AI_CALL_STATUS[ai_name].update({"status": "error", "error": str(e)[:500]})

    return ai_name, {}, "all_failed"


def _phase_system(ai_name: str) -> str:
    base = "只输出 JSON 数组。禁止 markdown、禁止 JSON 外说明。不得引用 CRS、本地矩阵、贝叶斯或固定模板。"
    if ai_name == "gpt":
        return "你是 RAW 赔率结构和比分分布分析师。" + base
    if ai_name == "grok":
        return "你是 RAW 资金流、散户热度和变盘分析师。" + base
    if ai_name == "gemini":
        return "你是 RAW 多市场一致性和异常结构分析师。" + base
    if ai_name == "claude":
        return "你是最终 RAW-AI 主裁，不是反指模型。默认尊重证据最完整的初审，只有原始字段硬反证才改票。" + base
    return base

# ============================================================
# 缓存 / singleflight
# ============================================================

_VOLATILE_CACHE_KEYS = {
    "timestamp", "ts", "now", "current_time", "server_time", "local_time",
    "fetched_at", "fetch_time", "crawl_time", "scrape_time", "sync_time",
    "updated_at", "update_time", "last_update", "last_updated",
    "generated_at", "generated_time", "created_at", "request_id", "trace_id",
    "uuid", "uid", "runtime", "elapsed", "latency", "cache_hit", "cache_ts",
}


def _sanitize_for_ai_cache(obj: Any) -> Any:
    if not AI_CACHE_STRIP_VOLATILE_KEYS:
        return obj

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kl = str(k).strip().lower()
            if kl in _VOLATILE_CACHE_KEYS:
                continue
            if kl.endswith("_ts") or kl.endswith("_timestamp") or kl.endswith("_time_ms"):
                continue
            out[k] = _sanitize_for_ai_cache(v)
        return out

    if isinstance(obj, list):
        return [_sanitize_for_ai_cache(x) for x in obj]

    return obj


def _stable_ai_cache_key(match_analyses: List[Dict[str, Any]], phase: str = "pure") -> str:
    compact = []
    for ma in match_analyses:
        m = ma.get("match", {})
        stable_m = _sanitize_for_ai_cache(m)
        stable_ctx = _sanitize_for_ai_cache(ma.get("external_context", {}))
        compact.append({
            "home": m.get("home_team"),
            "away": m.get("away_team"),
            "league": m.get("league", m.get("cup", "")),
            "match_num": m.get("match_num"),
            "id": m.get("id"),
            "sp_home": m.get("sp_home", m.get("win")),
            "sp_draw": m.get("sp_draw", m.get("same")),
            "sp_away": m.get("sp_away", m.get("lose")),
            "give_ball": m.get("give_ball"),
            "a": [m.get(f"a{i}") for i in range(8)],
            "change": m.get("change"),
            "change_pack": m.get("change_pack"),
            "vote": m.get("vote"),
            "information_hash": _hash_obj(_sanitize_for_ai_cache(m.get("information"))),
            "points_hash": _hash_obj(_sanitize_for_ai_cache(m.get("points"))),
            "raw_match_hash": _hash_obj(stable_m),
            "external_context_hash": _hash_obj(stable_ctx),
        })

    raw = json.dumps({
        "version": ENGINE_VERSION,
        "phase": phase,
        "matches": compact,
    }, ensure_ascii=False, sort_keys=True, default=str)

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ai_cache_file(cache_key: str) -> str:
    os.makedirs(AI_CACHE_DIR, exist_ok=True)
    return os.path.join(AI_CACHE_DIR, f"{cache_key}.json")


def _ai_lock_file(cache_key: str) -> str:
    os.makedirs(AI_CACHE_DIR, exist_ok=True)
    return os.path.join(AI_CACHE_DIR, f"{cache_key}.lock")


def _load_ai_disk_cache(cache_key: str) -> Optional[Dict[str, Dict[int, Dict[str, Any]]]]:
    if AI_DISABLE_CACHE or not AI_PERSISTENT_CACHE_ENABLED:
        return None

    path = _ai_cache_file(cache_key)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            pack = json.load(f)

        ts = float(pack.get("ts", 0))
        if time.time() - ts > AI_DECISION_CACHE_TTL:
            try:
                os.remove(path)
            except Exception:
                pass
            return None

        results = pack.get("results", {})
        restored = {n: {} for n in AI_NAMES}

        for name, rows in (results or {}).items():
            if isinstance(rows, dict):
                restored[name] = {}
                for k, v in rows.items():
                    try:
                        restored[name][int(k)] = v
                    except Exception:
                        continue

        print(f"  [AI DISK CACHE] 命中持久化缓存 ttl={AI_DECISION_CACHE_TTL}s")
        return restored

    except Exception as e:
        print(f"  [AI DISK CACHE] 读取失败: {str(e)[:100]}")
        return None


def _save_ai_disk_cache(cache_key: str, results: Dict[str, Dict[int, Dict[str, Any]]], status: Dict[str, Any]) -> None:
    if AI_DISABLE_CACHE or not AI_PERSISTENT_CACHE_ENABLED:
        return

    try:
        ok = sum(1 for n in AI_NAMES if results.get(n))
        if ok <= 0:
            return

        path = _ai_cache_file(cache_key)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "ts": time.time(),
                "version": ENGINE_VERSION,
                "status": status,
                "results": results,
            }, f, ensure_ascii=False, default=str)
        os.replace(tmp, path)

    except Exception as e:
        print(f"  [AI DISK CACHE] 写入失败: {str(e)[:100]}")


def _try_acquire_ai_disk_lock(cache_key: str) -> bool:
    if not AI_SINGLEFLIGHT_ENABLED or not AI_PERSISTENT_CACHE_ENABLED:
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
    if not AI_PERSISTENT_CACHE_ENABLED:
        return
    try:
        os.remove(_ai_lock_file(cache_key))
    except Exception:
        pass


async def _wait_for_ai_disk_cache(cache_key: str) -> Optional[Dict[str, Dict[int, Dict[str, Any]]]]:
    deadline = time.time() + max(5, AI_DISK_LOCK_WAIT_SECONDS)
    print("  [AI DISK LOCK] 同批次任务正在其他触发中运行，等待首个结果")

    while time.time() < deadline:
        cached = _load_ai_disk_cache(cache_key)
        if cached is not None:
            return cached
        await asyncio.sleep(AI_DISK_LOCK_POLL_SECONDS)

    print("  [AI DISK LOCK] 等待超时，继续本轮请求")
    return None


async def _run_ai_matrix_two_phase_inner(
    match_analyses: List[Dict[str, Any]],
    cache_key: str,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    if aiohttp is None:
        print("  [AI ERROR] aiohttp 未安装，AI弃权")
        return {n: {} for n in AI_NAMES}

    debug_ai_config()

    num = len(match_analyses)
    prompt = build_phase1_prompt(match_analyses)
    print(f"  [v18.4.10 Phase1 Prompt] {len(prompt):,}字符 → GPT/Grok/Gemini")

    all_results: Dict[str, Dict[int, Dict[str, Any]]] = {n: {} for n in AI_NAMES}
    connector = aiohttp.TCPConnector(limit=8, use_dns_cache=False, ttl_dns_cache=0)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            async_call_one_ai_batch(session, prompt, num, name, _phase_system(name))
            for name in PHASE1_NAMES
        ]
        phase1 = await asyncio.gather(*tasks, return_exceptions=True)

        for res in phase1:
            if isinstance(res, tuple):
                all_results[res[0]] = res[1]
            else:
                print(f"  [Phase1 ERROR] {res}")

        valid_phase1 = sum(1 for n in PHASE1_NAMES if all_results.get(n))

        if _env_bool("AI_RUN_CLAUDE_ONLY_IF_PHASE1_VALID", False):
            min_valid = _env_int("AI_MIN_PHASE1_VALID_FOR_CLAUDE", 2)
            if valid_phase1 < min_valid:
                print(f"  [Claude Skip] Phase1有效模型不足 {valid_phase1}/{min_valid}")
                all_results["claude"] = {}
                AI_CALL_STATUS["claude"] = {
                    "ok": False,
                    "status": "skipped_phase1_not_enough",
                    "valid_phase1": valid_phase1,
                    "min_valid": min_valid,
                }
            else:
                audit_prompt = build_claude_final_audit_prompt(match_analyses, all_results)
                print(f"  [v18.4.10 Phase2 Claude Audit] {len(audit_prompt):,}字符 | compact={AI_USE_COMPACT_CLAUDE_AUDIT}")
                _, cl_res, _ = await async_call_one_ai_batch(session, audit_prompt, num, "claude", _phase_system("claude"))
                all_results["claude"] = cl_res or {}
        else:
            audit_prompt = build_claude_final_audit_prompt(match_analyses, all_results)
            print(f"  [v18.4.10 Phase2 Claude Audit] {len(audit_prompt):,}字符 | compact={AI_USE_COMPACT_CLAUDE_AUDIT}")
            _, cl_res, _ = await async_call_one_ai_batch(session, audit_prompt, num, "claude", _phase_system("claude"))
            all_results["claude"] = cl_res or {}

    ok = sum(1 for n in AI_NAMES if all_results.get(n))
    status = {k: AI_CALL_STATUS.get(k, {}) for k in AI_NAMES}
    print(f"  [完成] {ok}/4 AI有数据 | status={status}")

    if not AI_DISABLE_CACHE and ok > 0:
        _AI_RESULT_CACHE[cache_key] = (time.time(), all_results, status)
        _save_ai_disk_cache(cache_key, all_results, status)
    elif ok == 0:
        print("  [AI CACHE] 本轮 0/4 AI 有效，不写入缓存")

    return all_results


async def run_ai_matrix_two_phase(match_analyses: List[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    cache_key = _stable_ai_cache_key(match_analyses)
    now = time.time()

    if not AI_DISABLE_CACHE and cache_key in _AI_RESULT_CACHE:
        ts, results, _ = _AI_RESULT_CACHE[cache_key]
        if now - ts <= AI_DECISION_CACHE_TTL:
            print(f"  [AI CACHE] 命中内存缓存 ttl={AI_DECISION_CACHE_TTL}s")
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
            _AI_RESULT_CACHE[cache_key] = (time.time(), waited, {"status": "disk_cache_waited"})
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
# AI 结果选择 / 前端兼容
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

    valid = [
        (name, ai_responses.get(name, {}))
        for name in PHASE1_NAMES
        if _valid_ai_score_from_response(ai_responses.get(name, {}))
    ]
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
        if final_direction in mass:
            mass[final_direction] = 50.0
            for d in mass:
                if d != final_direction:
                    mass[d] = 25.0
        else:
            mass = {"home": 33.3, "draw": 33.3, "away": 33.4}

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


def _score_goal_band(score: str) -> str:
    return _normalize_goal_band_value("", score)


def _score_btts(score: str) -> str:
    return _normalize_btts_value("", score)


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
    return {
        "valid_models": valid,
        "valid_count": len(valid),
        "same_score": same_score,
        "same_direction": same_dir,
        "score_agreement": same_score / total,
        "direction_agreement": same_dir / total,
    }


def _experience_review_coverage(final_r: Dict[str, Any], exp_audit: Dict[str, Any]) -> Tuple[float, List[str]]:
    triggered = exp_audit.get("triggered", []) if isinstance(exp_audit, dict) else []
    ids = [str(t.get("id", "")).replace("EXP_", "") for t in triggered if t.get("id")]

    if not ids:
        return 1.0, []

    reviews = final_r.get("experience_review", []) if isinstance(final_r, dict) else []
    reviewed = {str(r.get("id", "")).replace("EXP_", "") for r in reviews if isinstance(r, dict)}
    missing = [rid for rid in ids if rid not in reviewed]

    return max(0.0, 1.0 - len(missing) / max(1, len(ids))), missing


def _compute_recommendation_scores(
    final_r: Dict[str, Any],
    all_ai: Dict[str, Dict[str, Any]],
    match_obj: Dict[str, Any],
    exp_audit: Dict[str, Any],
    score: str,
    direction: str,
    pct: Dict[str, float],
    top_candidates: List[Tuple[str, float]],
) -> Dict[str, Any]:
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
    claude_valid = _valid_ai_score_from_response(all_ai.get("claude", {})) != ""

    exp_cov, exp_missing = _experience_review_coverage(final_r, exp_audit)
    audit = final_r.get("audit", {}) if isinstance(final_r.get("audit", {}), dict) else {}
    data_missing = final_r.get("data_missing", []) if isinstance(final_r.get("data_missing", []), list) else []

    web_missing = (
        "external_european_odds" in data_missing
        or str(audit.get("web_odds_check", "")).lower() in ("web_search_unavailable", "european_odds_missing", "missing")
    )

    gb = final_r.get("goal_band", "") or _score_goal_band(score)
    bt = final_r.get("btts", "") or _score_btts(score)
    score_gb = _score_goal_band(score)
    score_bt = _score_btts(score)

    original_gb = final_r.get("goal_band_original", "")
    original_bt = final_r.get("btts_original", "")

    warnings = []

    if original_gb and score_gb and original_gb != score_gb:
        warnings.append(f"goal_band_auto_fixed:{original_gb}->{score_gb}")

    if original_bt in ("yes", "no") and score_bt in ("yes", "no") and original_bt != score_bt:
        warnings.append(f"btts_auto_fixed:{original_bt}->{score_bt}")

    has_shape_reason = bool(
        str(final_r.get("score_shape_reason", "")).strip()
        or str(audit.get("style_score_logic", "")).strip()
    )

    top_score_prob = top_candidates[0][1] if top_candidates else 0.0
    tsp = top_score_prob if top_score_prob <= 1 else top_score_prob / 100.0

    direction_score = (
        25.0
        + 0.40 * top
        + 0.65 * gap
        + 10.0 * (1 - entropy)
        + 12.0 * agreement.get("direction_agreement", 0.0)
        + 8.0 * exp_cov
    )

    shape_score = (
        18.0
        + 16.0 * (top / 100.0)
        + 8.0 * (gap / 100.0)
        + 65.0 * min(1.0, tsp)
        + 16.0 * agreement.get("score_agreement", 0.0)
        + 14.0 * 1.0
        + 9.0 * (1.0 if has_shape_reason else 0.0)
        + 6.0 * exp_cov
    )

    if web_missing:
        direction_score -= 5
        shape_score -= 3

    if exp_missing:
        direction_score -= min(12, 2.0 * len(exp_missing))
        shape_score -= min(8, 1.2 * len(exp_missing))

    hard_warnings = [
        w for w in warnings
        if not w.startswith("goal_band_auto_fixed:")
        and not w.startswith("btts_auto_fixed:")
    ]

    if hard_warnings:
        shape_score -= 18
        direction_score -= 3

    if not claude_valid:
        direction_score -= 4
        shape_score -= 4

    direction_score = round(_clip(direction_score, 0, 100), 1)
    shape_score = round(_clip(shape_score, 0, 100), 1)
    overall = round(min(direction_score, shape_score * 1.08), 1)

    return {
        "direction_selection_score": direction_score,
        "score_shape_score": shape_score,
        "overall_selection_score": overall,
        "recommendation_tier": _tier_from_score(overall),
        "direction_tier": _tier_from_score(direction_score),
        "score_tier": _tier_from_score(shape_score),
        "model_agreement": agreement,
        "experience_review_coverage": round(exp_cov, 3),
        "experience_review_missing": exp_missing,
        "score_shape_warnings": warnings,
        "web_odds_missing_penalty": bool(web_missing),
        "claude_missing_penalty": not claude_valid,
        "goal_band_closed_loop": gb,
        "btts_closed_loop": bt,
    }


def _abstain_prediction(reason: str = "AI全失败，PURE模式不使用本地兜底") -> Dict[str, Any]:
    return {
        "predicted_score": "弃权",
        "predicted_label": "弃权",
        "result": "弃权",
        "display_direction": "弃权",
        "final_direction": "abstain",
        "is_abstain": True,
        "is_score_others": False,
        "home_win_pct": 0.0,
        "draw_pct": 0.0,
        "away_win_pct": 0.0,
        "confidence": 0,
        "confidence_meaning": "PURE RAW-AI 模式：AI全失败即弃权，不使用本地兜底",
        "risk_level": "高",
        "dir_confidence": 0,
        "dir_gap": 0,
        "scenario": "ai_abstain",
        "goal_range": (0, 0),
        "bayesian_evidences": [reason],
        "bayesian_prior": {},
        "override_triggered": False,
        "traps_detected": [],
        "trap_count": 0,
        "trap_severity": 0,
        "trap_details": [],
        "trap_flags": {},
        "fair_1x2": {},
        "fair_1x2_method": "disabled_pure_raw_ai",
        "market_overround": 0.0,
        "raw_implied_1x2": {},
        "crs_shape": "disabled_pure_raw_ai",
        "crs_moments": {},
        "crs_margin": 0.0,
        "crs_coverage": 0.0,
        "crs_implied_probs": {},
        "crs_low_rank_info": {},
        "top_score_candidates": [],
        "unified_matrix_top_scores": [],
        "unified_goal_probs": {},
        "fair_1x2_pack": {},
        "mixed_target_dir": {},
        "unified_source": "disabled_pure_raw_ai",
        "decision_source": "ai_abstain_no_local_fallback",
        "ai_authority_mode": "pure_raw_ai",
        "suggested_kelly": 0.0,
        "edge_vs_market": 0.0,
        "is_value": False,
        "ev_note": "disabled_pure_raw_ai",
        "score_model_prob": 0.0,
        "score_market_odds": 0.0,
        "score_market_implied_pct": None,
        "smart_money_signal": "",
        "smart_signals": [],
        "cold_door": {
            "is_cold_door": False,
            "strength": 0,
            "level": "普通",
            "signals": [],
            "sharp_confirmed": False,
            "dark_verdict": "",
        },
        "xG_home": "?",
        "xG_away": "?",
        "expected_total_goals": 0,
        "over_under_2_5": "弃权",
        "both_score": "弃权",
        "ai_avg_confidence": 0,
        "ai_abstained": ["GPT", "GROK", "GEMINI", "CLAUDE"],
        "gpt_score": "弃权",
        "gpt_analysis": "弃权",
        "grok_score": "弃权",
        "grok_analysis": "弃权",
        "gemini_score": "弃权",
        "gemini_analysis": "弃权",
        "claude_score": "弃权",
        "claude_analysis": "弃权",
        "ai_call_status": {k: AI_CALL_STATUS.get(k, {}) for k in AI_NAMES},
        "claude_parse_status": AI_CALL_STATUS.get("claude", {}).get("status", ""),
        "engine_version": ENGINE_VERSION,
        "engine_architecture": ENGINE_ARCHITECTURE,
    }


def _make_ai_prediction(
    final_name: str,
    final_r: Dict[str, Any],
    decision_source: str,
    all_ai: Dict[str, Dict[str, Any]],
    match_obj: Dict[str, Any],
) -> Dict[str, Any]:
    final_r = dict(final_r or {})

    score = _valid_ai_score_from_response(final_r)
    direction = _score_direction(score) or _normalize_direction(final_r.get("final_direction", ""), score)

    top3 = final_r.get("top3", []) if isinstance(final_r.get("top3", []), list) else []
    top_candidates = []

    for cand in top3[:8]:
        sc = _score_from_candidate(cand)
        if _parse_score(sc)[0] is not None:
            prob = _prob_to_float(cand.get("prob", 0) if isinstance(cand, dict) else 0)
            top_candidates.append((sc, round(prob, 3)))

    if not top_candidates and score != "弃权":
        top_candidates = [(score, 0.0)]

    score_goal_band = _score_goal_band(score)
    score_btts = _score_btts(score)
    original_goal_band = final_r.get("goal_band", "")
    original_btts = final_r.get("btts", "")

    if score_goal_band:
        final_r["goal_band_original"] = original_goal_band
        final_r["goal_band"] = score_goal_band

    if score_btts in ("yes", "no"):
        final_r["btts_original"] = original_btts
        final_r["btts"] = score_btts

    pct = (
        final_r.get("direction_probs")
        if isinstance(final_r.get("direction_probs"), dict) and final_r.get("direction_probs")
        else _direction_pct_from_top3(top3, direction)
    )

    conf = int(_clip(_f(final_r.get("ai_confidence", 60), 60), 0, 100))
    gmin, gmax, scenario = _goal_range_from_score(score)
    total_goals = _score_total(score) or 0
    final_odds = get_market_odds_for_score(match_obj, score)
    market_implied = round(100.0 / final_odds, 3) if final_odds > 1.05 else None

    ai_abstained = [n.upper() for n in AI_NAMES if not _valid_ai_score_from_response(all_ai.get(n, {}))]
    confs = [
        _f(r.get("ai_confidence", 60), 60)
        for r in all_ai.values()
        if isinstance(r, dict) and _valid_ai_score_from_response(r)
    ]
    avg_conf = round(sum(confs) / len(confs), 1) if confs else conf

    def sc_of(name: str) -> str:
        return _valid_ai_score_from_response(all_ai.get(name, {})) or "弃权"

    def reason_of(name: str) -> str:
        r = all_ai.get(name, {})
        if not isinstance(r, dict) or not _valid_ai_score_from_response(r):
            return "弃权"
        return str(r.get("reason", ""))[:3000]

    exp_audit = _experience_engine().analyze(match_obj)
    selection_pack = _compute_recommendation_scores(
        final_r,
        all_ai,
        match_obj,
        exp_audit,
        score,
        direction,
        pct,
        top_candidates,
    )

    evidences = [
        "PURE RAW-AI：AI成功时不使用本地比分矩阵、不使用CRS、不使用本地风控兜底。",
        "STRICT JSON：Claude默认不允许自然语言残片直接进入最终裁决；parse0时启用JSON repair。",
        f"最终来源:{decision_source}; final_model={final_name}; score={score}; direction={direction}",
        f"AI top3:{top_candidates[:5]}",
        f"AI direction_probs:{pct} ({'AI原生direction_probs' if final_r.get('direction_probs') else 'top3_direction_share_fallback'})",
    ]

    if final_r.get("audit"):
        evidences.append("AI audit:" + _json_compact(final_r.get("audit"), 1800))

    if final_r.get("data_missing"):
        evidences.append("data_missing:" + _json_compact(final_r.get("data_missing"), 800))

    if exp_audit.get("triggered"):
        evidences.append(
            "experience_audit_cards(prompt_only):"
            + _json_compact([
                {k: t.get(k) for k in ("id", "name", "reason", "ai_question")}
                for t in exp_audit.get("triggered", [])
            ], 1800)
        )

    if selection_pack.get("score_shape_warnings"):
        evidences.append("score_shape_warnings:" + _json_compact(selection_pack.get("score_shape_warnings"), 500))

    if selection_pack.get("experience_review_missing"):
        evidences.append("experience_review_missing:" + _json_compact(selection_pack.get("experience_review_missing"), 800))

    if selection_pack.get("claude_missing_penalty"):
        evidences.append("claude_missing_penalty:true; 当前最终结果来自Phase1兜底，不是Claude终审")

    return {
        "predicted_score": score,
        "predicted_label": _score_display_label(score, direction),
        "result": _direction_cn(direction),
        "display_direction": _direction_cn(direction),
        "final_direction": direction,
        "is_abstain": False,
        "is_score_others": _score_display_label(score, direction) in ("胜其他", "平其他", "负其他"),
        "home_win_pct": pct.get("home", 0.0),
        "draw_pct": pct.get("draw", 0.0),
        "away_win_pct": pct.get("away", 0.0),
        "confidence": conf,
        "confidence_meaning": "AI自报置信度，非历史命中率；PURE模式不做本地概率改写",
        "risk_level": str(final_r.get("risk_level", "medium")),
        "goal_band": final_r.get("goal_band", _score_goal_band(score)),
        "goal_band_original": final_r.get("goal_band_original", ""),
        "btts_ai": final_r.get("btts", _score_btts(score)),
        "btts_original": final_r.get("btts_original", ""),
        "score_shape_reason": final_r.get("score_shape_reason", ""),
        "experience_review": final_r.get("experience_review", []),
        **selection_pack,
        "dir_confidence": pct.get(direction, 0.0),
        "dir_gap": round(max(pct.values()) - sorted(pct.values(), reverse=True)[1], 1) if len(pct) >= 2 else 0.0,
        "scenario": scenario,
        "goal_range": (gmin, gmax),
        "bayesian_evidences": evidences,
        "bayesian_prior": {},
        "override_triggered": False,
        "traps_detected": [],
        "trap_count": 0,
        "trap_severity": 0,
        "trap_details": [],
        "trap_flags": {},
        "fair_1x2": {},
        "fair_1x2_method": "disabled_pure_raw_ai",
        "market_overround": 0.0,
        "raw_implied_1x2": {},
        "crs_shape": "disabled_pure_raw_ai",
        "crs_moments": {},
        "crs_margin": 0.0,
        "crs_coverage": 0.0,
        "crs_implied_probs": {},
        "crs_low_rank_info": {},
        "top_score_candidates": top_candidates,
        "unified_matrix_top_scores": top_candidates,
        "unified_goal_probs": {},
        "fair_1x2_pack": {},
        "mixed_target_dir": {},
        "unified_source": "disabled_pure_raw_ai",
        "decision_source": decision_source,
        "ai_authority_mode": "pure_raw_ai",
        "gpt_score": sc_of("gpt"),
        "gpt_analysis": reason_of("gpt"),
        "grok_score": sc_of("grok"),
        "grok_analysis": reason_of("grok"),
        "gemini_score": sc_of("gemini"),
        "gemini_analysis": reason_of("gemini"),
        "claude_score": sc_of("claude"),
        "claude_analysis": reason_of("claude"),
        "ai_abstained": ai_abstained,
        "ai_avg_confidence": avg_conf,
        "ai_call_status": {k: AI_CALL_STATUS.get(k, {}) for k in AI_NAMES},
        "claude_parse_status": AI_CALL_STATUS.get("claude", {}).get("status", ""),
        "claude_parse_mode": AI_CALL_STATUS.get("claude", {}).get("parse_mode", ""),
        "value_kill_count": 0,
        "suggested_kelly": 0.0,
        "edge_vs_market": 0.0,
        "is_value": False,
        "ev_note": "disabled_pure_raw_ai_no_local_probability",
        "score_model_prob": top_candidates[0][1] if top_candidates else 0.0,
        "score_market_odds": final_odds,
        "score_market_implied_pct": market_implied,
        "smart_money_signal": " | ".join(exp_audit.get("risk_signals", [])[:8]),
        "smart_signals": ["EXP_AUDIT:" + s for s in exp_audit.get("risk_signals", [])],
        "cold_door": {
            "is_cold_door": False,
            "strength": 0,
            "level": "普通",
            "signals": [],
            "sharp_confirmed": False,
            "dark_verdict": "",
        },
        "xG_home": "?",
        "xG_away": "?",
        "over_under_2_5": "大" if total_goals >= 3 else "小",
        "both_score": "是" if (_parse_score(score)[0] or 0) > 0 and (_parse_score(score)[1] or 0) > 0 else "否",
        "expected_total_goals": total_goals,
        "over_2_5": None,
        "btts": None,
        "bookmaker_implied_home_xg": "?",
        "bookmaker_implied_away_xg": "?",
        "sharp_detected": False,
        "sharp_dir": None,
        "fair_dir": None,
        "shin_dir": None,
        "actual_handicap_signed": None,
        "theoretical_handicap_signed": None,
        "model_consensus": len([n for n in AI_NAMES if _valid_ai_score_from_response(all_ai.get(n, {}))]),
        "total_models": 4,
        "extreme_warning": "",
        "refined_poisson": {},
        "poisson": {},
        "elo": {},
        "random_forest": {},
        "gradient_boost": {},
        "neural_net": {},
        "logistic": {},
        "svm": {},
        "knn": {},
        "dixon_coles": {},
        "bradley_terry": {},
        "home_form": {},
        "away_form": {},
        "handicap_signal": "",
        "odds_movement": {},
        "vote_analysis": {},
        "h2h_blood": {},
        "crs_analysis": {},
        "ttg_analysis": {},
        "halftime": {},
        "pace_rating": "",
        "kelly_home": {},
        "kelly_away": {},
        "odds": {},
        "experience_analysis": {"mode": "prompt_only_no_decision", **exp_audit},
        "pro_odds": {},
        "asian_handicap_probs": {},
        "top_scores": [],
        "engine_version": ENGINE_VERSION,
        "engine_architecture": ENGINE_ARCHITECTURE,
    }


def merge_result_pure_ai(all_ai: Dict[str, Dict[int, Dict[str, Any]]], idx: int, match_obj: Dict[str, Any]) -> Dict[str, Any]:
    per_match = {name: (all_ai.get(name, {}) or {}).get(idx, {}) for name in AI_NAMES}
    final_name, final_r, source = _choose_final_ai(per_match)

    if not final_name:
        return _abstain_prediction()

    return _make_ai_prediction(final_name, final_r, source, per_match, match_obj)

# ============================================================
# 后处理 / 推荐
# ============================================================

def _enforce_consistency(mg: Dict[str, Any]) -> Dict[str, Any]:
    if mg.get("is_abstain") or mg.get("predicted_score") == "弃权":
        mg["predicted_score"] = "弃权"
        mg["predicted_label"] = "弃权"
        mg["result"] = "弃权"
        mg["display_direction"] = "弃权"
        mg["final_direction"] = "abstain"
        return mg

    score = _normalize_score_text(mg.get("predicted_score", ""))
    d = _score_direction(score) or _dir_from_cn(mg.get("result", "")) or "draw"

    mg["predicted_score"] = score
    mg["predicted_label"] = _score_display_label(score, d)
    mg["result"] = _direction_cn(d)
    mg["display_direction"] = _direction_cn(d)
    mg["final_direction"] = d
    mg["is_score_others"] = mg["predicted_label"] in ("胜其他", "平其他", "负其他")

    score_gb = _score_goal_band(score)
    score_bt = _score_btts(score)
    if score_gb:
        mg["goal_band"] = score_gb
    if score_bt in ("yes", "no"):
        mg["btts_ai"] = score_bt

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

    mg["validation_warnings"] = warnings
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

        if pr.get("claude_missing_penalty"):
            s -= 2.5

        if pr.get("experience_review_missing"):
            s -= min(6, len(pr.get("experience_review_missing", [])) * 1.5)

        p["recommend_score"] = round(s, 2)

    return sorted(preds, key=lambda x: x.get("recommend_score", -999), reverse=True)[:4]


def extract_num(ms: Any) -> int:
    wm = {"一": 1000, "二": 2000, "三": 3000, "四": 4000, "五": 5000, "六": 6000, "日": 7000, "天": 7000}
    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999

# ============================================================
# 主入口
# ============================================================

def run_predictions(raw: Dict[str, Any], use_ai: bool = True):
    ms = raw.get("matches", []) if isinstance(raw, dict) else []
    ms = [normalize_match(m) for m in ms]

    print("\n" + "=" * 88)
    print(f"  [{ENGINE_VERSION}] PURE RAW-AI 主审 | {len(ms)} 场 | Claude JSON Repair | AI失败即弃权")
    print("=" * 88)

    match_analyses: List[Dict[str, Any]] = []

    for i, m in enumerate(ms, 1):
        exp_audit = _experience_engine().analyze(m)
        match_analyses.append({
            "match": m,
            "index": i,
            "experience_audit": exp_audit,
            "external_context": {"enabled": False, "source_quality": "disabled", "items": [], "errors": []},
        })

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
        print(f"  [{ENGINE_VERSION} AI] 启动 GPT/Grok/Gemini 初审 + Claude 终审")
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
            print(
                f"  [{i}] {m.get('home_team')} vs {m.get('away_team')} => "
                f"{mg['result']} ({mg['predicted_score']}={mg['predicted_label']}) | "
                f"AI_CF:{mg['confidence']} | 来源:{mg.get('decision_source')} | "
                f"Claude:{mg.get('claude_parse_status', '')}/{mg.get('claude_parse_mode', '')}"
            )

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
    print("   模式: AI成功=AI直出；Claude parse0=JSON repair；AI失败=弃权；无本地兜底")