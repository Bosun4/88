# -*- coding: utf-8 -*-
"""
vMAX 20.2.1-FULL-ANCHOR — 3AI Gemini-Referee FULL-RESEARCH Web-Augmented Engine
=================================================================================

本版基于 vMAX 20.2.1-FULL 的 AI-native 架构升级：
1. 保留原版 API 调用链：get_key_for_ai / get_url_for_ai / _chat_url / async_call_ai_json / _run_one_chunk / run_ai_native_web / run_predictions。
2. 本地不做足球预测裁决：不判断 Sharp 真伪、不改比分、不改方向、不用本地经验卡覆盖 AI。
3. 新增 AI 锚点事实层：0-0、1-1、总进球赔率、4球/5球/6+尾部、让球盘比分形态、联赛风格提示。
4. Prompt 强制 AI 做 anchor_audit：每场必须解释为什么不是 0-0/1-1/1-0/2-1/3-1 等关键候选。
5. Gemini 终审仍是最终裁判；本地只做协议校验、JSON解析、字段闭环、落盘、前端兼容。
6. Mock 只用于工程闭环，不代表真实命中率。

入口：
    run_predictions(raw, use_ai=True) -> (res, top4)

关键环境变量：
    API_URL / API_KEY 或各模型单独 *_API_URL / *_API_KEY
    AI_MOCK_MODE=true
    AI_RESEARCH_MODE=research|enhanced|production
    AI_CHUNK_SIZE=6
    AI_ENABLE_CROSS_EXAM=true
    AI_ENABLE_CONSISTENCY_JUDGE=true
    AI_NATIVE_WEB=true
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
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None

try:
    import structlog
    logger = structlog.get_logger()
except Exception:  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)


# ============================================================
# 版本常量 / 基础配置
# ============================================================

ENGINE_VERSION = "vMAX 20.2.1-FULL-ANCHOR"
ENGINE_ARCHITECTURE = (
    "AI-NATIVE WEB-AUGMENTED 3AI FULL-RESEARCH + MARKET-ANCHOR PROMPT: "
    "本地只做协议层；GPT/Grok 初审 + 互审 + Gemini Web-aware 终审 + 一致性审计；"
    "新增0-0/1-1/总进球/让球盘/联赛风格锚点事实层；Top4/推荐等级由 Gemini 输出。"
)

VALID_DIRS = {"home", "draw", "away"}
AI_NAMES = ["gpt", "grok", "gemini"]
PHASE1_NAMES = ["gpt", "grok"]

DEFAULT_MODELS = {
    "gpt": "熊猫-特供-X-10-gpt-5.5",
    "grok": "熊猫-A-5-grok-4.2-fast-200w上下文",
    "gemini": "熊猫-8-AS正向-gemini-3.5-flash",
}

CRS_FULL_MAP = {
    "1-0": "w10", "2-0": "w20", "2-1": "w21", "3-0": "w30", "3-1": "w31",
    "3-2": "w32", "4-0": "w40", "4-1": "w41", "4-2": "w42", "5-0": "w50",
    "5-1": "w51", "5-2": "w52",
    "0-0": "s00", "1-1": "s11", "2-2": "s22", "3-3": "s33",
    "0-1": "l01", "0-2": "l02", "1-2": "l12", "0-3": "l03", "1-3": "l13",
    "2-3": "l23", "0-4": "l04", "1-4": "l14", "2-4": "l24",
    "0-5": "l05", "1-5": "l15", "2-5": "l25",
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

STANDARD_TOTAL_GOAL_ODDS = {
    0: 9.5,
    1: 5.5,
    2: 3.5,
    3: 4.0,
    4: 7.0,
    5: 14.0,
    6: 30.0,
    7: 70.0,
}

LEAGUE_STYLE_RESEARCH_HINTS = {
    "德甲": {
        "style_hint": "高节奏、转换快、BTTS 与 3球以上尾部常见；但仍需结合盘口与阵容验证。",
        "score_shapes_to_audit": ["2-1", "3-1", "3-2", "2-2", "1-2"],
        "risk_note": "不能机械大球；若总进球赔率不支持4+，需压回2-3球。",
    },
    "意甲": {
        "style_hint": "结构更谨慎，强队也经常在3球内解决；尤文类球队需重点审计1-0/2-0/1-1/0-0/2-1。",
        "score_shapes_to_audit": ["1-0", "2-0", "1-1", "0-0", "2-1", "3-0"],
        "risk_note": "若0-0/1-1赔率被压低，不能无脑给2-1。",
    },
    "英超": {
        "style_hint": "节奏高但分层明显；强强对决可能互相限制，保级/杯赛可能变成低比分肉搏。",
        "score_shapes_to_audit": ["1-1", "2-1", "1-2", "2-2", "1-0", "0-0"],
        "risk_note": "必须结合比赛性质：强强、保级、赛程密度、轮换、欧战前后。",
    },
    "英冠": {
        "style_hint": "身体对抗强、波动大，低比分与混战比分都要审计；不能只看低赔方向。",
        "score_shapes_to_audit": ["1-1", "1-0", "0-1", "2-1", "1-2", "0-0"],
        "risk_note": "若让球浅且1-1/0-0赔率低，平局必须被严肃审计。",
    },
    "法甲": {
        "style_hint": "整体偏谨慎，但强弱差和个别强队会拉高尾部；需结合球队风格。",
        "score_shapes_to_audit": ["1-0", "1-1", "2-0", "2-1", "0-0", "3-0"],
        "risk_note": "不能只按联赛标签判断，必须交叉总进球与正确比分赔率簇。",
    },
    "西甲": {
        "style_hint": "控球与节奏分化明显，低比分和2-1/1-1较常见；强队主场可扩展到2-0/3-0。",
        "score_shapes_to_audit": ["1-0", "2-0", "2-1", "1-1", "0-0", "3-0"],
        "risk_note": "若4球赔率偏高，优先审计0-3球比分带。",
    },
    "荷甲": {
        "style_hint": "开放程度较高，高比分尾部需要纳入，但仍需受总进球赔率约束。",
        "score_shapes_to_audit": ["2-1", "3-1", "3-2", "2-2", "4-1"],
        "risk_note": "如果4球/5球没有被压低，不应强推大比分。",
    },
}

PROMPT_STRIP_KEYS = {
    "prediction", "predictions", "top4", "rank", "recommend_score",
    "predicted_score", "predicted_label", "result", "display_direction", "final_direction",
    "raw_ai_direction", "score_implied_direction", "home_win_pct", "draw_pct", "away_win_pct",
    "confidence", "is_recommended", "is_strict_recommended", "is_top4_candidate",
    "gpt_score", "gpt_analysis", "grok_score", "grok_analysis", "gemini_score", "gemini_analysis",
    "claude_score", "claude_analysis", "final_referee_score", "final_referee_analysis", "bayesian_evidences", "score_market_evidence", "sharp_audit",
    "model_agreement", "experience_review", "recommendation", "ai_run_metadata", "ai_call_status",
}


# ============================================================
# 环境变量工具
# ============================================================

def _env_bool(name: str, default: bool = False) -> bool:
    v = str(os.environ.get(name, str(default))).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(str(os.environ.get(name, default)).strip()))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, default)).strip())
    except Exception:
        return default


AI_MOCK_MODE = _env_bool("AI_MOCK_MODE", False)
AI_FORCE_COMMON_GATEWAY = _env_bool("FORCE_COMMON_GATEWAY_URL", True)

AI_RESEARCH_MODE = str(os.environ.get("AI_RESEARCH_MODE", "research")).strip().lower()
if AI_RESEARCH_MODE not in {"production", "enhanced", "research"}:
    AI_RESEARCH_MODE = "research"

if AI_RESEARCH_MODE == "production":
    _default_native_web = False
    _default_cross_exam = False
    _default_consistency = False
    _default_chunk_size = 10
elif AI_RESEARCH_MODE == "enhanced":
    _default_native_web = False
    _default_cross_exam = False
    _default_consistency = True
    _default_chunk_size = 8
else:
    _default_native_web = True
    _default_cross_exam = True
    _default_consistency = True
    _default_chunk_size = 6

AI_NATIVE_WEB = _env_bool("AI_NATIVE_WEB", _default_native_web)
AI_REQUIRE_WEB_SOURCES = _env_bool("AI_REQUIRE_WEB_SOURCES", AI_NATIVE_WEB)
AI_WARN_MISSING_PUBLISHED_AT = _env_bool("AI_WARN_MISSING_PUBLISHED_AT", False)
AI_WEB_MAX_SOURCES_PER_MATCH = max(0, _env_int("AI_WEB_MAX_SOURCES_PER_MATCH", 8))
AI_CHUNK_SIZE = max(1, _env_int("AI_CHUNK_SIZE", _default_chunk_size))
# Keep the default prompt budget below the historical 3-4万 token range while
# still allowing explicit environment overrides for larger research runs.
AI_MAX_PROMPT_CHARS_PER_CHUNK = max(30000, _env_int("AI_MAX_PROMPT_CHARS_PER_CHUNK", 60000))
AI_ENABLE_CROSS_EXAM = _env_bool("AI_ENABLE_CROSS_EXAM", _default_cross_exam)
AI_ENABLE_CONSISTENCY_JUDGE = _env_bool("AI_ENABLE_CONSISTENCY_JUDGE", _default_consistency)
AI_ENABLE_FALLBACK_REFEREE = _env_bool("AI_ENABLE_FALLBACK_REFEREE", True)
AI_CONSISTENCY_JUDGE_MODEL = str(os.environ.get("AI_CONSISTENCY_JUDGE_MODEL", "gpt")).strip().lower()
AI_FINAL_REFEREE_MODEL = str(os.environ.get("AI_FINAL_REFEREE_MODEL", "gemini")).strip().lower() or "gemini"
AI_FALLBACK_REFEREE_MODEL = str(os.environ.get("AI_FALLBACK_REFEREE_MODEL", "gpt")).strip().lower()

AI_READ_TIMEOUT = _env_int("AI_READ_TIMEOUT", 5400)
AI_FINAL_READ_TIMEOUT = _env_int("AI_FINAL_READ_TIMEOUT", _env_int("AI_CLAUDE_READ_TIMEOUT", 7200))
AI_CONNECT_TIMEOUT = _env_int("AI_CONNECT_TIMEOUT", 120)
AI_HTTP_TOTAL_TIMEOUT = _env_int("AI_HTTP_TOTAL_TIMEOUT", 0)
AI_TEMPERATURE_PHASE1 = _env_float("AI_TEMPERATURE_PHASE1", 0.18)
AI_TEMPERATURE_CRITIC = _env_float("AI_TEMPERATURE_CRITIC", 0.10)
AI_TEMPERATURE_FINAL = _env_float("AI_TEMPERATURE_FINAL", 0.08)
AI_USE_RESPONSE_FORMAT = _env_bool("AI_USE_RESPONSE_FORMAT", True)
AI_SAVE_RAW_RESPONSE = _env_bool("AI_SAVE_RAW_RESPONSE", False)
AI_PARSE_DEBUG = _env_bool("AI_PARSE_DEBUG", False)

AI_PHASE_SNAPSHOT_ENABLED = _env_bool("AI_PHASE_SNAPSHOT_ENABLED", True)
AI_PHASE_RESULT_DIR = str(os.environ.get("AI_PHASE_RESULT_DIR", "data/ai_phase_results_v20")).strip() or "data/ai_phase_results_v20"

AI_FILL_TOP4_WITH_NON_RECOMMENDABLE = _env_bool("AI_FILL_TOP4_WITH_NON_RECOMMENDABLE", False)
MIN_AI_RECOMMEND_TIER = str(os.environ.get("MIN_AI_RECOMMEND_TIER", "B")).strip().upper()

AI_CALL_STATUS: Dict[str, Dict[str, Any]] = {n: {} for n in AI_NAMES}
AI_RESULT_FILES: Dict[str, str] = {}
_LAST_AI_RUN_METADATA: Dict[str, Any] = {}


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
        for token in ["↑", "↓", "↗", "↘", "最大", "最小", "红", "蓝", ",", "％", "%"]:
            s = s.replace(token, "")
        s = s.strip()
        if "/" in s:
            return default
        return float(s)
    except Exception:
        return default


def _i(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, _f(v, 0.0)))


def _exists(v: Any) -> bool:
    return v not in (None, "", "-", "N/A", "n/a", "None", "none", "null", {}, [])


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


def _short_hash(s: Any, n: int = 12) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:n]


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _mask_key(k: str) -> str:
    if not k:
        return ""
    if len(k) <= 8:
        return "***"
    return f"{k[:4]}...{k[-4:]}"


def _strip_output_fields(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_output_fields(v) for k, v in obj.items() if str(k) not in PROMPT_STRIP_KEYS}
    if isinstance(obj, list):
        return [_strip_output_fields(x) for x in obj]
    return obj


def _safe_json_line(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


# ============================================================
# 比分 / 方向
# ============================================================

_SCORE_RE = re.compile(r"(\d{1,2})\s*[-:：]\s*(\d{1,2})")


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
        if ss.lower() in ("home", "draw", "away", "abstain"):
            return None, None
        m = _SCORE_RE.search(ss)
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
    if h is None or a is None:
        return None
    return h + a


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


def _direction_cn(direction: str) -> str:
    return {"home": "主胜", "draw": "平局", "away": "客胜", "abstain": "弃权"}.get(direction, "弃权")


def _dir_from_any(v: Any) -> Optional[str]:
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


# ============================================================
# 输入抽取 / 标准化
# ============================================================

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


def _is_change_direction_code(change: Any) -> bool:
    if not isinstance(change, dict) or not change:
        return False
    vals = []
    for k in ("win", "same", "draw", "lose", "home", "away"):
        if k in change:
            vals.append(_f(change.get(k), 999))
    if not vals:
        return False
    return all(v in (-1, 0, 1) for v in vals)


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

    skip = {"vote", "change", "points", "information", "prediction", "stats", "smart_signals", "recommendation"}
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
        m["give_ball"] = m.get("handicap") or m.get("rq") or m.get("let_ball") or ""

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

    m["change_is_direction_code"] = _is_change_direction_code(m.get("change"))
    return m


def _extract_match_list(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if not isinstance(raw, dict):
        return []

    matches = raw.get("matches")
    if isinstance(matches, list):
        return [x for x in matches if isinstance(x, dict)]

    if isinstance(matches, dict):
        scope = str(raw.get("scope", raw.get("target_mode", "today_only"))).lower()
        preferred_keys = []
        if "today" in scope or scope in ("", "today_only"):
            preferred_keys.extend(["today", "today_only"])
        preferred_keys.extend(["today", "list", "data", "items", "matches"])
        for k in preferred_keys:
            if isinstance(matches.get(k), list):
                return [x for x in matches.get(k, []) if isinstance(x, dict)]
        out: List[Dict[str, Any]] = []
        for v in matches.values():
            if isinstance(v, list):
                out.extend([x for x in v if isinstance(x, dict)])
        if out:
            return out

    if isinstance(raw.get("today"), list):
        return [x for x in raw.get("today", []) if isinstance(x, dict)]

    if isinstance(raw.get("top4"), list) and not matches:
        return [x for x in raw.get("top4", []) if isinstance(x, dict)]

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
# Evidence Compiler：只产事实，不做足球判断
# ============================================================

def _score_market_facts(match_obj: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for sc, key in CRS_FULL_MAP.items():
        odd = _f(match_obj.get(key), 0.0)
        if odd > 1.01:
            rows.append({"score": sc, "odds": odd, "direction": _score_direction(sc), "total_goals": _score_total(sc)})
    rows = sorted(rows, key=lambda x: x["odds"])
    for i, r in enumerate(rows, 1):
        r["all_rank"] = i

    by_dir = {"home": [], "draw": [], "away": []}
    for r in rows:
        d = r.get("direction")
        if d in by_dir:
            by_dir[d].append(r)
    for d, arr in by_dir.items():
        for i, r in enumerate(arr, 1):
            r["direction_rank"] = i

    return {
        "available": bool(rows),
        "lowest_scores_overall": rows[:10],
        "lowest_scores_by_direction": {d: arr[:6] for d, arr in by_dir.items()},
        "coverage_count": len(rows),
    }


def _total_goal_market_facts(match_obj: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for i in range(8):
        odd = _f(match_obj.get(f"a{i}"), 0.0)
        if odd > 1.01:
            rows.append({"goals": i, "odds": odd})
    rows = sorted(rows, key=lambda x: x["odds"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return {
        "available": bool(rows),
        "lowest_total_goals": rows[:8],
        "mode_goals": rows[0]["goals"] if rows else None,
        "coverage_count": len(rows),
    }


def _half_full_facts(match_obj: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for k, lb in HFTF_MAP.items():
        odd = _f(match_obj.get(k), 0.0)
        if odd > 1.01:
            rows.append({"code": k, "label": lb, "odds": odd})
    rows.sort(key=lambda x: x["odds"])
    return {"available": bool(rows), "lowest": rows[:9], "coverage_count": len(rows)}


def _parse_handicap_value(raw: Any) -> Optional[float]:
    try:
        s = str(raw or "").strip()
        if not s:
            return None
        sign = 1.0
        if "受" in s or s.startswith("-"):
            sign = -1.0
        cn_map = [
            ("平手", 0.0), ("平/半", 0.25), ("半球", 0.5), ("半/一", 0.75),
            ("一球", 1.0), ("一/球半", 1.25), ("球半", 1.5), ("球半/两", 1.75),
            ("两球", 2.0), ("两/两半", 2.25), ("两半", 2.5), ("两半/三", 2.75), ("三球", 3.0),
        ]
        for k, v in cn_map:
            if k in s:
                return sign * v
        m = re.search(r"(-?\d+(?:\.\d+)?)(?:\s*/\s*(-?\d+(?:\.\d+)?))?", s)
        if not m:
            return None
        a = float(m.group(1))
        if m.group(2) is not None:
            b = float(m.group(2))
            val = (abs(a) + abs(b)) / 2.0
        else:
            val = abs(a)
        if s.startswith("-"):
            sign = -1.0
        return sign * val
    except Exception:
        return None


def _handicap_score_shape_templates(line_abs: float) -> Dict[str, Any]:
    if line_abs <= 0.25:
        return {"line_bucket": "level_or_quarter", "must_audit_scores": ["0-0", "1-1", "1-0", "0-1", "2-1", "1-2"], "logic": "平手/平半盘常见一球内分胜负或平局，不能机械推2-1。"}
    if line_abs <= 0.75:
        return {"line_bucket": "half_ball", "must_audit_scores": ["1-0", "2-1", "1-1", "0-0", "3-2", "2-0"], "logic": "半球附近通常对应一球边际，2-1/1-0/1-1都必须比较。"}
    if line_abs <= 1.25:
        return {"line_bucket": "one_ball", "must_audit_scores": ["1-0", "2-0", "2-1", "3-1", "1-1"], "logic": "一球盘常见赢一球或赢两球，需比较1-0/2-0/2-1。"}
    if line_abs <= 1.75:
        return {"line_bucket": "ball_and_half", "must_audit_scores": ["2-0", "3-0", "3-1", "2-1", "4-1"], "logic": "球半附近需要审计强队两球胜与三球胜，不得只给2-1。"}
    if line_abs <= 2.25:
        return {"line_bucket": "two_ball", "must_audit_scores": ["2-0", "3-0", "3-1", "4-1", "4-2"], "logic": "两球盘应审计2-0/3-0/3-1/4-2等强弱差比分。"}
    return {"line_bucket": "deep_handicap", "must_audit_scores": ["3-0", "4-0", "4-1", "5-0", "5-1", "胜其他"], "logic": "深盘需要审计穿盘、胜其他和尾部比分，但仍受总进球赔率约束。"}


def _league_style_anchor_facts(match_obj: Dict[str, Any]) -> Dict[str, Any]:
    league = str(match_obj.get("league", match_obj.get("cup", "")) or "")
    hits = []
    for key, info in LEAGUE_STYLE_RESEARCH_HINTS.items():
        if key in league:
            hits.append({"league_key": key, **info})
    competition_notes = []
    text = league + " " + str(match_obj.get("match_num", "")) + " " + str(match_obj.get("information", ""))[:300]
    if any(k in text for k in ["保级", "降级", "争冠", "争四", "欧战资格", "升级附加", "附加赛"]):
        competition_notes.append("存在战意/排名压力关键词，AI必须联网或结合上下文审计比赛性质。")
    if any(k in text for k in ["杯", "淘汰", "决赛", "半决赛", "欧冠", "欧联", "国王杯", "足总杯"]):
        competition_notes.append("杯赛/淘汰赛关键词出现，AI必须审计首回合/次回合/保守策略/轮换风险。")
    return {
        "league": league,
        "matched_style_hints": hits,
        "competition_notes": competition_notes,
        "ai_instruction": "这些是风格研究提示，不是本地预测结论；AI必须结合赔率、阵容、赛程、球队风格验证。",
    }


def _score_anchor_facts(match_obj: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for sc, key in CRS_FULL_MAP.items():
        odd = _f(match_obj.get(key), 0.0)
        if odd > 1.01:
            rows.append({"score": sc, "odds": odd, "direction": _score_direction(sc), "total_goals": _score_total(sc)})
    rows.sort(key=lambda x: x["odds"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    odds_by_score = {r["score"]: r["odds"] for r in rows}
    observations = []
    s00 = odds_by_score.get("0-0", 0.0)
    s11 = odds_by_score.get("1-1", 0.0)
    s10 = odds_by_score.get("1-0", 0.0)
    s01 = odds_by_score.get("0-1", 0.0)
    s21 = odds_by_score.get("2-1", 0.0)
    s12 = odds_by_score.get("1-2", 0.0)

    if 0 < s00 <= 11.0:
        observations.append({"anchor": "low_0_0_odds", "value": s00, "meaning_for_ai": "0-0赔率低于或接近11，必须严肃审计闷局/低比分；不能默认1-1或2-1。"})
    if 0 < s11 <= 7.5:
        observations.append({"anchor": "low_1_1_odds", "value": s11, "meaning_for_ai": "1-1赔率较低，双方进球但低总进球路径需要重点比较。"})
    if 0 < s10 <= 9.5 or 0 < s01 <= 9.5:
        observations.append({"anchor": "low_one_nil_path", "value": {"1-0": s10, "0-1": s01}, "meaning_for_ai": "1-0/0-1路径较低时，盘口可能指向一球边际而非开放大比分。"})
    if 0 < s21 <= 8.5 or 0 < s12 <= 8.5:
        observations.append({"anchor": "low_2_1_path", "value": {"2-1": s21, "1-2": s12}, "meaning_for_ai": "2-1/1-2路径低时，可以作为主流候选，但必须和0-0/1-1/1-0、总进球模态交叉。"})

    draw_rows = [r for r in rows if r["direction"] == "draw"][:4]
    low_draw_rank = [r for r in draw_rows if r.get("rank", 99) <= 8]
    if low_draw_rank:
        observations.append({"anchor": "draw_scores_in_low_rank", "value": low_draw_rank, "meaning_for_ai": "平局比分进入低赔簇，AI必须显式写出为什么选/不选平。"})

    high_tail = [r for r in rows if r.get("total_goals") is not None and r["total_goals"] >= 4 and r["odds"] <= 30][:8]
    if high_tail:
        observations.append({"anchor": "high_score_tail_compressed", "value": high_tail, "meaning_for_ai": "4球以上正确比分赔率存在压缩，AI必须检查3-2/4-1/4-2/胜其他尾部是否真实有支撑。"})

    return {
        "available": bool(rows),
        "lowest_scores_overall": rows[:12],
        "specific_odds": {"0-0": s00, "1-1": s11, "1-0": s10, "0-1": s01, "2-1": s21, "1-2": s12},
        "observations_for_ai": observations,
        "ai_instruction": "这些只是正确比分赔率锚点事实；AI必须和总进球、让球、联赛风格、新闻阵容交叉后再裁决。",
    }


def _total_goal_anchor_facts(match_obj: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    odds_by_goal = {}
    for g in range(8):
        odd = _f(match_obj.get(f"a{g}"), 0.0)
        if odd > 1.01:
            row = {"goals": g, "odds": odd, "standard_reference": STANDARD_TOTAL_GOAL_ODDS.get(g), "compression_ratio_std_div_actual": round(STANDARD_TOTAL_GOAL_ODDS.get(g, 50.0) / odd, 3) if odd > 0 else None}
            rows.append(row)
            odds_by_goal[g] = odd
    rows.sort(key=lambda x: x["odds"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    observations = []
    a0 = odds_by_goal.get(0, 0.0)
    a1 = odds_by_goal.get(1, 0.0)
    a2 = odds_by_goal.get(2, 0.0)
    a3 = odds_by_goal.get(3, 0.0)
    a4 = odds_by_goal.get(4, 0.0)
    a5 = odds_by_goal.get(5, 0.0)
    a6 = odds_by_goal.get(6, 0.0)
    a7 = odds_by_goal.get(7, 0.0)

    if a4 > 6.0:
        observations.append({"anchor": "four_goals_odds_above_6", "value": a4, "meaning_for_ai": "4球赔率高于6，4球不是强压缩主模态；AI必须优先审计0-3球比分带，不能机械推3-1/2-2。"})
    elif 0 < a4 <= 5.0:
        observations.append({"anchor": "four_goals_odds_compressed", "value": a4, "meaning_for_ai": "4球赔率被压低，3-1/2-2/3-2等尾部必须进入比较。"})

    if 0 < a5 <= 8.0:
        observations.append({"anchor": "five_goals_extreme_compressed", "value": a5, "meaning_for_ai": "5球赔率极低，必须审计3-2/4-1/4-2/胜其他路径。"})
    elif 0 < a5 <= 10.0:
        observations.append({"anchor": "five_goals_mild_compressed", "value": a5, "meaning_for_ai": "5球赔率偏低，高比分尾部不能忽略，但仍需看4球/6+是否共振。"})

    if 0 < a6 <= 16.0 or 0 < a7 <= 30.0:
        observations.append({"anchor": "six_or_seven_plus_tail_compressed", "value": {"6": a6, "7+": a7}, "meaning_for_ai": "6球或7+赔率压缩，必须检查胜其他/负其他，不得只停留在2-1。"})

    small_cluster = []
    for g in [0, 1, 2]:
        if g in odds_by_goal:
            small_cluster.append({"goals": g, "odds": odds_by_goal[g], "rank": next((r["rank"] for r in rows if r["goals"] == g), None)})
    high_cluster = []
    for g in [4, 5, 6, 7]:
        if g in odds_by_goal:
            high_cluster.append({"goals": g, "odds": odds_by_goal[g], "rank": next((r["rank"] for r in rows if r["goals"] == g), None)})

    if rows and rows[0]["goals"] in (0, 1, 2):
        observations.append({"anchor": "total_goals_mode_low", "value": rows[0], "meaning_for_ai": "总进球最低赔在0-2球，低比分候选必须前置审计。"})
    if rows and rows[0]["goals"] in (4, 5, 6, 7):
        observations.append({"anchor": "total_goals_mode_high", "value": rows[0], "meaning_for_ai": "总进球最低赔在4+，高比分候选必须前置审计。"})

    return {
        "available": bool(rows),
        "lowest_total_goals": rows[:8],
        "mode_goals": rows[0]["goals"] if rows else None,
        "specific_odds": {"0": a0, "1": a1, "2": a2, "3": a3, "4": a4, "5": a5, "6": a6, "7+": a7},
        "small_cluster": small_cluster,
        "high_cluster": high_cluster,
        "observations_for_ai": observations,
        "ai_instruction": "总进球赔率只作为形态锚点；AI必须与正确比分赔率、让球盘、联赛风格、阵容新闻交叉。",
    }


def _handicap_anchor_facts(match_obj: Dict[str, Any]) -> Dict[str, Any]:
    raw = match_obj.get("give_ball", match_obj.get("handicap", match_obj.get("rq", "")))
    val = _parse_handicap_value(raw)
    if val is None:
        return {"raw": raw, "parsed_line": None, "favorite_side_by_line_sign": "unclear", "score_shape_template": {}, "ai_instruction": "盘口无法解析，AI需自行从原始字段判断。"}
    fav = "home" if val > 0 else "away" if val < 0 else "unclear"
    tpl = _handicap_score_shape_templates(abs(val))
    return {"raw": raw, "parsed_line": val, "favorite_side_by_line_sign": fav, "score_shape_template": tpl, "ai_instruction": "让球盘只提供比分形态审计模板，不代表本地预测方向。AI必须结合1X2、正确比分、总进球和外部盘口验证。"}


def _cross_anchor_questions(match_obj: Dict[str, Any]) -> List[str]:
    qs = []
    score_f = _score_anchor_facts(match_obj)
    total_f = _total_goal_anchor_facts(match_obj)
    hand_f = _handicap_anchor_facts(match_obj)
    s00 = score_f.get("specific_odds", {}).get("0-0", 0)
    s11 = score_f.get("specific_odds", {}).get("1-1", 0)
    a4 = total_f.get("specific_odds", {}).get("4", 0)
    mode_g = total_f.get("mode_goals")
    if 0 < s00 <= 11:
        qs.append("0-0赔率≤11：为什么不是0-0？如果选其他比分，必须解释突破闷局的证据。")
    if 0 < s11 <= 7.5:
        qs.append("1-1赔率偏低：为什么不是1-1？必须比较BTTS与低总进球是否共振。")
    if a4 > 6:
        qs.append("4球赔率>6：若选择3-1/2-2/3-2等4球比分，必须解释为什么能突破4球高赔率阻力。")
    if mode_g in (0, 1, 2):
        qs.append("总进球主模态在0-2球：不能机械输出2-1，必须审计0-0/1-0/0-1/1-1/2-0/0-2。")
    if mode_g in (4, 5, 6, 7):
        qs.append("总进球主模态在4+：不能机械输出1-0/1-1，必须审计3-1/2-2/3-2/4-1/胜其他。")
    tpl = hand_f.get("score_shape_template", {})
    must = tpl.get("must_audit_scores", [])
    if must:
        qs.append(f"让球盘形态模板要求重点比较：{','.join(must)}。AI必须说明最终比分如何从这些候选中胜出。")
    if not qs:
        qs.append("必须同时比较：正确比分低赔簇、总进球最低赔、让球盘形态、联赛/球队风格，禁止默认1-1或2-1。")
    return qs


def build_evidence_packet(match_obj: Dict[str, Any], index: int) -> Dict[str, Any]:
    m = _strip_output_fields(match_obj)
    style_keys = [
        "league_style", "league_profile", "team_style", "home_style", "away_style", "play_style",
        "tactical_style", "pace_rating", "tempo", "home_form", "away_form", "weather", "injury",
        "lineup", "news", "motivation", "schedule", "home_rank", "away_rank", "home_stats", "away_stats",
    ]
    correct_score_odds = {sc: m.get(key) for sc, key in CRS_FULL_MAP.items() if m.get(key) not in (None, "", 0, "0")}
    total_goals_odds = {str(i): m.get(f"a{i}") for i in range(8) if m.get(f"a{i}") not in (None, "", 0, "0")}
    score_anchor = _score_anchor_facts(m)
    total_anchor = _total_goal_anchor_facts(m)
    handicap_anchor = _handicap_anchor_facts(m)
    league_anchor = _league_style_anchor_facts(m)
    cross_questions = _cross_anchor_questions(m)
    evidence = {
        "match": index,
        "identity": {
            "home_team": m.get("home_team", m.get("home", "Home")),
            "away_team": m.get("away_team", m.get("guest", "Away")),
            "league": m.get("league", m.get("cup", "")),
            "match_num": m.get("match_num", ""),
            "match_time": m.get("match_time", m.get("time", "")),
        },
        "lottery_market_1x2": {"home": m.get("sp_home", m.get("win")), "draw": m.get("sp_draw", m.get("same")), "away": m.get("sp_away", m.get("lose")), "note": "中国体彩竞彩抓包赔率，不是欧洲均赔。"},
        "handicap": {"raw": m.get("give_ball", m.get("handicap", m.get("rq", "")))},
        "movement": {"change": m.get("change", {}), "change_is_direction_code": bool(m.get("change_is_direction_code", False)), "coding_note": "change_is_direction_code=true 时，-1=降水/下降，0=不变，1=升水/上升。", "odds_movement": m.get("odds_movement", {})},
        "public_vote": m.get("vote", {}),
        "total_goals_odds": total_goals_odds,
        "correct_score_odds": correct_score_odds,
        "half_full_time_odds": {HFTF_MAP[k]: m.get(k) for k in HFTF_MAP if m.get(k) not in (None, "", 0, "0")},
        "derived_market_facts_no_judgement": {"score_market_facts": _score_market_facts(m), "total_goal_market_facts": _total_goal_market_facts(m), "half_full_market_facts": _half_full_facts(m)},
        "ai_anchor_facts_no_judgement": {
            "score_anchor_facts": score_anchor,
            "total_goal_anchor_facts": total_anchor,
            "handicap_anchor_facts": handicap_anchor,
            "league_style_anchor_facts": league_anchor,
            "mandatory_cross_anchor_questions": cross_questions,
            "important_note": "以上锚点只作为 AI 审计任务与市场事实，不是本地预测。AI 必须自己判断这些锚点是否成立，并在 anchor_audit 中逐项回答。",
        },
        "context_raw_fields": {
            "information": m.get("information", ""), "points": m.get("points", ""), "baseface": m.get("baseface", ""),
            "expert_intro": m.get("expert_intro", ""), "intelligence": m.get("intelligence", ""),
            "style_and_team_core": {k: m.get(k) for k in style_keys if m.get(k) not in (None, "", {}, [])},
        },
        "data_quality": {
            "has_1x2": all(m.get(k) not in (None, "", 0, "0") for k in ["sp_home", "sp_draw", "sp_away"]),
            "has_total_goals": bool(total_goals_odds),
            "has_correct_score": bool(correct_score_odds),
            "has_vote": isinstance(m.get("vote"), dict) and bool(m.get("vote")),
            "has_change": isinstance(m.get("change"), dict) and bool(m.get("change")),
            "has_context_news": any(_exists(m.get(k)) for k in ["information", "points", "injury", "lineup", "news"]),
        },
    }

    if evidence["data_quality"]["has_1x2"]:
        h = _f(m.get("sp_home"))
        d = _f(m.get("sp_draw"))
        a = _f(m.get("sp_away"))
        if h > 0 and d > 0 and a > 0:
            raw_h = 1 / h
            raw_d = 1 / d
            raw_a = 1 / a
            sum_raw = raw_h + raw_d + raw_a
            
            fair_h = raw_h / sum_raw
            fair_d = raw_d / sum_raw
            fair_a = raw_a / sum_raw

            evidence["market_implied"] = {
                "home_raw_prob": round(raw_h, 4),
                "draw_raw_prob": round(raw_d, 4),
                "away_raw_prob": round(raw_a, 4),
                "overround": round(sum_raw - 1, 4),
                "home_fair_prob": round(fair_h, 4),
                "draw_fair_prob": round(fair_d, 4),
                "away_fair_prob": round(fair_a, 4),
                "home_fair_odds": round(1 / fair_h, 4),
                "draw_fair_odds": round(1 / fair_d, 4),
                "away_fair_odds": round(1 / fair_a, 4)
            }

    return evidence


# ============================================================
# Prompt / Schema
# ============================================================

def _canonical_output_schema_text() -> str:
    return r'''
必须输出严格 JSON object，顶层格式：{"predictions":[...]}。不要输出 markdown，不要输出 JSON 外文本。
每个 prediction 必须包含：
{
  "match": 1,
  "final_direction": "home/draw/away",
  "predicted_score": "2-1",
  "direction_probs": {"home": 45, "draw": 28, "away": 27},
  "goal_band": "0-1/2/3/4+",
  "btts": "yes/no/unclear",
  "top3": [
    {"score":"2-1", "prob":16, "logic":"中文专业说明"}
  ],
  "anchor_audit": {
    "answered_cross_anchor_questions": [],
    "zero_zero_case": "为什么选/不选0-0；必须引用0-0赔率、总进球模态、比赛性质或阵容证据",
    "one_one_case": "为什么选/不选1-1；必须引用1-1赔率、BTTS、低进球共振或反证",
    "one_goal_margin_case": "为什么选/不选1-0/0-1/2-1/1-2",
    "four_plus_case": "若涉及3-1/2-2/3-2/4-1/胜其他，说明4球/5球/6+赔率是否支持；若不涉及，说明为什么压低尾部",
    "handicap_score_shape_case": "让球盘对应的比分形态如何约束最终比分",
    "league_style_case": "联赛/球队风格如何支持或反驳最终比分",
    "final_score_vs_anchor_summary": "最终比分如何从所有锚点中胜出"
  },
  "market_interpretation": {
    "one_x_two":"中文说明", "handicap":"中文说明", "correct_score":"中文说明",
    "total_goals":"中文说明", "half_full_time":"中文说明", "external_market":"中文说明"
  },
  "money_flow": {
    "public_money_direction":"home/draw/away/unclear",
    "sharp_money_direction":"home/draw/away/unclear",
    "sharp_confidence":0,
    "reverse_line_movement":false,
    "steam_move":"home/draw/away/none/unclear",
    "evidence":"中文说明"
  },
  "contextual_logic": {
    "league_style":"中文说明", "team_style":"中文说明", "tempo":"low/medium/high/unclear",
    "score_shape":"中文说明", "btts_likelihood":"yes/no/unclear", "rotation_risk":"low/medium/high/unclear"
  },
  "rejected_cases": {"home":"中文说明", "draw":"中文说明", "away":"中文说明"},
  "web_research": {
    "used": true,
    "failure_reason": "",
    "search_queries": [],
    "sources": [
      {"title":"", "url":"", "publisher":"", "published_at":"", "accessed_at":"", "source_type":"injury/lineup/odds/news/stats/tactical", "reliability":"high/medium/low", "claim":"", "impact":"direction/score/risk/no_impact"}
    ],
    "freshness_grade":"live/recent/stale/missing",
    "key_findings": [],
    "source_conflicts": []
  },
  "recommendation": {
    "tier":"S/A/B/C/D",
    "is_recommended":true,
    "top4_priority":1,
    "bet_confidence":0,
    "direction_stability":"strong/medium/weak",
    "score_stability":"strong/medium/weak",
    "risk_level":"low/medium/high",
    "risk_tags":[],
    "bet_action":"main/small/hedge/observe/no_bet",
    "why_this_can_fail":[],
    "minimum_evidence_needed":[],
    "why_recommended":"中文说明"
  },
  "data_quality": {"missing":[], "raw_packet_quality":"high/medium/low"},
  "reason":"中文综合理由"
}
硬约束：predicted_score 暗示的方向必须等于 final_direction；goal_band 与 predicted_score 总进球一致；btts 与 predicted_score 一致；top3[0].score 必须等于 predicted_score；anchor_audit 必须逐项回答本场 mandatory_cross_anchor_questions。若证据不足，请 recommendation.is_recommended=false、bet_action=observe/no_bet，不要硬推。
'''.strip()


def _web_research_instruction(role: str) -> str:
    if not AI_NATIVE_WEB:
        return "AI_NATIVE_WEB=false：如果无法联网，web_research.used=false，failure_reason 写 no_web_capability_or_disabled。"
    common = (
        "必须执行 Web-Augmented Match Research。若你的 API/模型具备联网能力，必须搜索并输出 sources；"
        "若无法联网，web_research.used=false 且 failure_reason 写 no_web_tool_available，禁止编造来源。"
        "所有影响方向/比分/推荐等级的联网信息必须进入 web_research.sources。来源必须有 title/url/published_at/claim；"
        "没有来源的所谓最新消息不能作为硬证据。"
    )
    if role == "gpt":
        return common + "你的联网重点：外部欧赔/亚盘/交易所、赔率时间点、竞彩与外部市场分歧。"
    if role == "grok":
        return common + "你的联网重点：最新伤停、预计首发、临场新闻、资金流/热度/市场异动。"
    if role == "gemini":
        return common + "你的联网重点：球队风格、赛程密度、杯赛赛制、战意、战术影响。"
    return common + "你的联网重点：审计三家来源质量、冲突来源、新鲜度，以及哪些外部信息真正改变判断。"


def _phase1_system(ai_name: str) -> str:
    role_intro = {
        "gpt": "你是 Probabilistic Market Structure Analyst，专攻 1X2、让球、正确比分赔率簇、总进球模态、外部赔率对照。",
        "grok": "你是 Money Flow / Sharp Movement Analyst，专攻 change、vote、热度、Sharp/Steam/Reverse Line Movement、临场新闻。",
        "gemini": "你是 Gemini Final Web-aware Referee，负责战术/来源质量审计、交叉证据仲裁和最终推荐。",
    }.get(ai_name, "你是足球量化 RAW-AI 分析师。")
    return (
        role_intro
        + "只能输出严格 JSON object，顶层必须是 predictions。所有解释字段必须中文。"
        + "禁止引用本地模型结论；允许使用原始体彩正确比分赔率 correct_score_odds 与 ai_anchor_facts_no_judgement。"
        + "不要机械投票；不要默认1-1或2-1；必须完成 anchor_audit，说明为什么不是关键反例比分。"
        + "必须给出 rejected_cases，说明为什么不选其他方向。"
    )


def build_phase1_prompt(evidence_batch: List[Dict[str, Any]], ai_name: str) -> str:
    n = len(evidence_batch)
    p = []
    p.append("<task>")
    p.append(f"本批次共有 {n} 场，match 编号必须完整覆盖：" + ",".join(str(e["match"]) for e in evidence_batch))
    p.append("你看到的是中国体彩竞彩抓包 Evidence，不是本地预测结论。本地没有 Sharp/比分/推荐裁决。")
    p.append(_web_research_instruction(ai_name))
    p.append("你的任务不是给空泛推荐，而是基于市场结构、资金流、比分赔率、总进球、半全场、联网情报形成可审计预测。")
    p.append("强制审计：每场必须读取 ai_anchor_facts_no_judgement，并在 anchor_audit 中逐项回答 mandatory_cross_anchor_questions。")
    p.append("禁止懒惰比分：不要因为常见就默认1-1或2-1；当0-0赔率≤11、1-1低赔、4球赔率>6或总进球主模态偏低时，必须严肃审计低比分。")
    p.append("</task>\n")
    p.append("<output_schema>")
    p.append(_canonical_output_schema_text())
    p.append("</output_schema>\n")
    p.append("<evidence_batch>")
    for e in evidence_batch:
        p.append(_safe_json_line(e))
    p.append("</evidence_batch>")
    return "\n".join(p)


def _short_prediction_for_prompt(r: Dict[str, Any]) -> Dict[str, Any]:
    keep = {}
    for k in [
        "match", "final_direction", "predicted_score", "direction_probs", "goal_band", "btts", "top3",
        "anchor_audit", "market_interpretation", "money_flow", "contextual_logic", "rejected_cases", "recommendation",
        "data_quality", "reason", "web_research", "final_web_audit", "validation_warnings",
    ]:
        if k in r:
            keep[k] = r[k]
    return keep


def build_critic_prompt(evidence_batch: List[Dict[str, Any]], reviewer_name: str, phase1_results: Dict[str, Dict[int, Dict[str, Any]]]) -> str:
    p = []
    p.append("<task>")
    p.append(f"你是 {reviewer_name.upper()} Critic。你不是重新预测主裁，只负责审查其他 AI 的漏洞。")
    p.append("必须指出：赔率误读、联网来源不可靠、Sharp 解释不充分、比分与总进球/BTTS/方向冲突、推荐等级过高、anchor_audit 没有回答0-0/1-1/4球/让球盘关键问题。")
    p.append("输出严格 JSON object，顶层格式 {\"critic_reports\":[...]}，不要 markdown。")
    p.append("每个 critic_report 格式：{\"match\":1,\"critic_model\":\"gpt\",\"target_findings\":[{\"target_model\":\"grok\",\"issue_type\":\"market/web/sharp/score/schema/risk/anchor\",\"severity\":\"low/medium/high\",\"comment\":\"中文\"}],\"own_revision_hint\":\"中文\"}")
    p.append("</task>\n<evidence_batch>")
    for e in evidence_batch:
        p.append(_safe_json_line(e))
    p.append("</evidence_batch>\n<phase1_results>")
    for model, rows in phase1_results.items():
        if model == reviewer_name:
            continue
        p.append(f"<{model}>")
        for e in evidence_batch:
            r = rows.get(e["match"], {})
            if r:
                p.append(_safe_json_line(_short_prediction_for_prompt(r)))
        p.append(f"</{model}>")
    p.append("</phase1_results>")
    return "\n".join(p)


def build_gemini_final_prompt(evidence_batch: List[Dict[str, Any]], phase1_results: Dict[str, Dict[int, Dict[str, Any]]], critic_reports: Dict[str, List[Dict[str, Any]]]) -> str:
    p = []
    p.append("<final_adjudication_protocol>")
    p.append("你是 Gemini 最终 Web-aware 裁判。你必须重新审计 raw evidence、GPT/Grok 初审、互审意见和联网来源质量。")
    p.append("证据优先级：raw market structure > correct-score cluster > total-goals mode > handicap score-shape > money-flow/sharp interpretation > tactical/web context > Phase1 consensus。")
    p.append("多数意见不自动成立；若 GPT/Grok 基于同一低赔/单边市场理由一致，这属于相关证据，不是独立证据。")
    p.append("S级必须至少有两个独立证据族同时支持：市场结构、正确比分赔率簇、总进球模态、让球盘形态、资金流/Sharp、联网阵容伤停、战术/赛程背景。仅赔率低赔或单边市场最多给A；无联网且依赖阵容/战意/实力碾压，最高给B。")
    p.append("低比分锚点审计：若0-0赔率≤11或1-1赔率偏低，必须显式比较0-0/1-1/1-0/0-1/2-0/0-2，不能机械给2-1。")
    p.append("4球锚点审计：若4球赔率>6，选择3-1/2-2/3-2必须有强证据；否则优先压回0-3球比分带。")
    p.append("高比分尾部审计：若5球≤8、6球≤16或7+≤30，必须检查3-2/4-1/4-2/胜其他。")
    p.append("强客低赔审计：若客胜<=1.50，必须比较0-0/0-1/1-1/0-2/1-2与总进球模态；不能机械给0-2或S级。")
    p.append("赛季末战意倒挂预警：如果强队已经提前夺冠或无欲无求（面临大面积轮换/伤停），而弱队处于降级区死磕保级，必须极度重视下盘（包括弱队主胜或平局）。即使强队硬实力碾压、机构依然给出客胜低赔，也应主动下调强队的推荐等级，并在比分上重点防范保级队爆冷（如1-1, 1-0, 2-1）。不可无视战意倒挂给客胜高信心。")
    p.append("超低赔豪门信任预警：当皇马等豪门胜赔极低（<=1.20），且自身丢冠后无欲无求大面积轮换，而对手处于降级状态时，虽然可能取胜，但极易输掉盘口甚至爆出冷平。如果预测净胜球>=2（如3-0, 2-0），必须在 recommendation 中明确防范赢球输盘风险，并将该比赛的 risk_level 直接锁定为 high，confidence 不得超过 65。")
    p.append("如果 sharp_money_direction 与 final_direction 冲突，必须解释为什么该信号是噪音，或者主动下调 recommendation.tier。")
    p.append("如果联网来源缺 URL/发布时间/claim，不能作为硬证据。必须输出 source_conflicts 和 final_web_audit。")
    p.append("最终推荐等级、是否进 Top4、bet_confidence 全部由你输出；本地只排序，不会改你的足球判断。")
    p.append("</final_adjudication_protocol>\n")
    p.append("<output_schema>")
    p.append(_canonical_output_schema_text())
    p.append("额外要求：每个 prediction 增加 final_web_audit 字段：{\"web_used_by_final\":true,\"decisive_web_evidence\":[],\"ignored_web_evidence\":[],\"web_confidence\":0}")
    p.append("</output_schema>\n")
    p.append("<evidence_batch>")
    for e in evidence_batch:
        p.append(_safe_json_line(e))
    p.append("</evidence_batch>\n<phase1_results>")
    for model in PHASE1_NAMES:
        p.append(f"<{model}>")
        rows = phase1_results.get(model, {})
        for e in evidence_batch:
            r = rows.get(e["match"], {})
            if r:
                p.append(_safe_json_line(_short_prediction_for_prompt(r)))
            else:
                p.append(_safe_json_line({"match": e["match"], "missing": True}))
        p.append(f"</{model}>")
    p.append("</phase1_results>\n<critic_reports>")
    p.append(_safe_json_line(critic_reports))
    p.append("</critic_reports>")
    return "\n".join(p)


def build_fallback_referee_prompt(evidence_batch: List[Dict[str, Any]], phase1_results: Dict[str, Dict[int, Dict[str, Any]]], critic_reports: Dict[str, List[Dict[str, Any]]]) -> str:
    p = []
    p.append("你是 Gemini 终审失败后的 AI fallback referee。不要使用本地规则。基于 raw evidence、ai_anchor_facts_no_judgement、Phase1 和 critic reports 输出最终 predictions。")
    p.append("输出 schema 与 Gemini final 完全一致。必须完成 anchor_audit。")
    p.append(_canonical_output_schema_text())
    p.append("<evidence_batch>")
    for e in evidence_batch:
        p.append(_safe_json_line(e))
    p.append("</evidence_batch><phase1_results>")
    p.append(_safe_json_line({m: {str(k): _short_prediction_for_prompt(v) for k, v in rows.items()} for m, rows in phase1_results.items()}))
    p.append("</phase1_results><critic_reports>")
    p.append(_safe_json_line(critic_reports))
    p.append("</critic_reports>")
    return "\n".join(p)


def build_consistency_judge_prompt(evidence_batch: List[Dict[str, Any]], final_predictions: Dict[int, Dict[str, Any]]) -> str:
    p = []
    p.append("你是 Consistency Judge，只检查结构一致性，不做足球判断，不改变预测方向/比分，除非存在字段自相矛盾时给出 repair 建议。")
    p.append("输出严格 JSON object：{\"repairs\":[{\"match\":1,\"valid\":true,\"warnings\":[],\"repair\":{...}}]}。")
    p.append("检查：predicted_score方向=final_direction；goal_band与比分总进球一致；btts与比分一致；top3[0]=predicted_score；web_research.used=true时必须有sources；anchor_audit必须存在。")
    p.append("不得根据足球观点改比分，只能修字段。")
    p.append("<evidence_batch>")
    for e in evidence_batch:
        p.append(_safe_json_line(e))
    p.append("</evidence_batch><final_predictions>")
    for idx, r in final_predictions.items():
        p.append(_safe_json_line(_short_prediction_for_prompt(r)))
    p.append("</final_predictions>")
    return "\n".join(p)


# ============================================================
# API / Mock 调用：保留原版调用链
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
        return _clean_env_key("API_KEY", "GPT_API_KEY", "OPENAI_API_KEY", "GROK_API_KEY", "GEMINI_API_KEY")
    return _clean_env_key(f"{ai_name.upper()}_API_KEY", "API_KEY", "OPENAI_API_KEY", "GPT_API_KEY")


def get_url_for_ai(ai_name: str) -> str:
    if AI_FORCE_COMMON_GATEWAY:
        return _clean_env_url("API_URL", "GPT_API_URL", "OPENAI_API_URL", "BASE_URL", "GROK_API_URL", "GEMINI_API_URL")
    return _clean_env_url(f"{ai_name.upper()}_API_URL", "API_URL", "OPENAI_API_URL", "BASE_URL", "GPT_API_URL")


def _model_for(ai_name: str) -> str:
    env_name = f"{ai_name.upper()}_MODEL"
    return str(os.environ.get(env_name, DEFAULT_MODELS.get(ai_name, "model"))).strip() or DEFAULT_MODELS.get(ai_name, "model")


def _chat_url(base_url: str) -> str:
    u = (base_url or "").rstrip("/")
    if not u:
        return ""
    if u.endswith("/chat/completions") or "/chat/completions" in u:
        return u
    return u + "/chat/completions"


def debug_ai_config() -> None:
    print(f"[AI CONFIG] mode={AI_RESEARCH_MODE} mock={AI_MOCK_MODE} native_web={AI_NATIVE_WEB} chunk_size={AI_CHUNK_SIZE} cross_exam={AI_ENABLE_CROSS_EXAM} consistency_judge={AI_ENABLE_CONSISTENCY_JUDGE}")
    for n in AI_NAMES:
        print(f"[AI CONFIG] {n.upper()} model={_model_for(n)} key={_mask_key(get_key_for_ai(n))} url={get_url_for_ai(n) or '<missing>'}")


async def async_call_ai_json(session: Optional[Any], ai_name: str, system_text: str, prompt: str, phase: str, expected_matches: List[int]) -> Tuple[str, Any, Dict[str, Any]]:
    t0 = time.time()
    model = _model_for(ai_name)
    status = {"ok": False, "ai_name": ai_name, "model": model, "phase": phase, "elapsed": 0.0}

    if AI_MOCK_MODE:
        raw_obj = _mock_ai_response(ai_name, phase, prompt, expected_matches)
        status.update({"ok": True, "status": "mock_ok", "elapsed": round(time.time() - t0, 3)})
        _update_call_status(ai_name, phase, status)
        return ai_name, raw_obj, status

    if aiohttp is None:
        status.update({"status": "aiohttp_missing"})
        _update_call_status(ai_name, phase, status)
        return ai_name, {}, status

    key = get_key_for_ai(ai_name)
    base_url = get_url_for_ai(ai_name)
    if not key:
        status.update({"status": "no_key"})
        _update_call_status(ai_name, phase, status)
        return ai_name, {}, status
    if not base_url:
        status.update({"status": "no_url"})
        _update_call_status(ai_name, phase, status)
        return ai_name, {}, status

    url = _chat_url(base_url)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    temperature = AI_TEMPERATURE_FINAL if phase in ("final", "fallback_referee") else AI_TEMPERATURE_CRITIC if phase == "critic" else AI_TEMPERATURE_PHASE1
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system_text}, {"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if AI_USE_RESPONSE_FORMAT:
        payload["response_format"] = {"type": "json_object"}

    try:
        read_timeout = AI_FINAL_READ_TIMEOUT if phase in ("final", "fallback_referee") else AI_READ_TIMEOUT
        total_timeout = None if AI_HTTP_TOTAL_TIMEOUT <= 0 else AI_HTTP_TOTAL_TIMEOUT
        timeout = aiohttp.ClientTimeout(
            total=total_timeout,
            connect=None if AI_CONNECT_TIMEOUT <= 0 else AI_CONNECT_TIMEOUT,
            sock_connect=None if AI_CONNECT_TIMEOUT <= 0 else AI_CONNECT_TIMEOUT,
            sock_read=None if read_timeout <= 0 else read_timeout,
        )
        assert session is not None
        async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
            text = await r.text()
            if r.status < 200 or r.status >= 300:
                status.update({"status": f"http_{r.status}", "http_error": text[:800], "elapsed": round(time.time() - t0, 1)})
                _update_call_status(ai_name, phase, status)
                return ai_name, {}, status
            try:
                data = json.loads(text)
            except Exception:
                data = {"raw": text}
            raw_text = _extract_response_text(data)
            if AI_SAVE_RAW_RESPONSE:
                _save_debug_dump(ai_name, phase, data, raw_text)
            obj = _json_loads_best_effort_object(raw_text)
            if not isinstance(obj, (dict, list)) or not obj:
                status.update({
                    "ok": False,
                    "status": "parse_failed",
                    "parse_error": "empty_or_invalid_json_object",
                    "raw_excerpt": raw_text[:300],
                    "elapsed": round(time.time() - t0, 1),
                })
                _update_call_status(ai_name, phase, status)
                return ai_name, {}, status
            status.update({"ok": True, "status": "ok", "elapsed": round(time.time() - t0, 1)})
            _update_call_status(ai_name, phase, status)
            return ai_name, obj, status
    except asyncio.TimeoutError:
        status.update({"status": "timeout", "elapsed": round(time.time() - t0, 1)})
    except Exception as e:
        status.update({"status": "error", "error": str(e)[:500], "elapsed": round(time.time() - t0, 1)})

    _update_call_status(ai_name, phase, status)
    return ai_name, {}, status


def _update_call_status(ai_name: str, phase: str, status: Dict[str, Any]) -> None:
    cur = AI_CALL_STATUS.setdefault(ai_name, {})
    cur[phase] = status
    cur["last_status"] = status.get("status")
    cur["model"] = status.get("model")


def _save_debug_dump(ai_name: str, phase: str, data: Any, raw_text: str = "") -> None:
    try:
        _ensure_dir("data/debug_v20")
        ts = int(time.time())
        with open(f"data/debug_v20/{ai_name}_{phase}_{ts}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        if raw_text:
            with open(f"data/debug_v20/{ai_name}_{phase}_{ts}.txt", "w", encoding="utf-8") as f:
                f.write(raw_text)
    except Exception:
        pass


def _extract_response_text(data: Any) -> str:
    candidates: List[Tuple[int, str]] = []

    def add(v: Any, bonus: int = 0) -> None:
        if isinstance(v, str):
            t = v.strip()
            if t:
                score = bonus
                if "predictions" in t or "critic_reports" in t or "repairs" in t:
                    score += 10
                if "final_direction" in t or "predicted_score" in t:
                    score += 5
                candidates.append((score, t))

    def walk(obj: Any, depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() in {"reasoning", "reasoning_content", "thinking", "chain_of_thought", "thoughts"}:
                    continue
                if str(k).lower() in {"content", "text", "output_text", "answer", "result", "response"}:
                    add(v, 3)
                walk(v, depth + 1)
        elif isinstance(obj, list):
            for x in obj[:100]:
                walk(x, depth + 1)
        else:
            add(obj, 0)

    if isinstance(data, dict):
        for ch in data.get("choices", []) or []:
            if isinstance(ch, dict):
                msg = ch.get("message", {})
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        add(content, 10)
                    elif isinstance(content, list):
                        for it in content:
                            if isinstance(it, dict):
                                add(it.get("text"), 10)
                                add(it.get("content"), 8)
                add(ch.get("text"), 5)
        add(data.get("output_text"), 10)
        add(data.get("text"), 5)
        walk(data)
    elif isinstance(data, str):
        add(data, 5)

    if not candidates:
        return ""
    candidates.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    return candidates[0][1]


def _preclean_text(text: str) -> str:
    clean = text or ""
    clean = clean.replace("\ufeff", "").replace("\x00", "")
    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"```(?:json|JSON|javascript|js|python|txt)?", "", clean)
    clean = clean.replace("```", "")
    clean = clean.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    clean = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", clean)
    return clean.strip()


def _json_loads_best_effort_object(text: str) -> Any:
    clean = _preclean_text(text)
    if not clean:
        return {}
    variants = [clean, re.sub(r",\s*([}\]])", r"\1", clean)]
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
    for frag in _balanced_fragments(clean):
        for cand in [frag, re.sub(r",\s*([}\]])", r"\1", frag)]:
            try:
                return json.loads(cand)
            except Exception:
                try:
                    return ast.literal_eval(cand)
                except Exception:
                    continue
    return {}


def _balanced_fragments(text: str) -> List[str]:
    frags: List[str] = []
    clean = _preclean_text(text)
    for start in [m.start() for m in re.finditer(r"[\[{]", clean)]:
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
            if any(k in frag for k in ["predictions", "critic_reports", "repairs", "final_direction", "predicted_score"]):
                frags.append(frag)
    frags.sort(key=len, reverse=True)
    return frags[:20]


# ============================================================
# 解析 / 标准化 AI 输出
# ============================================================

def _unwrap_predictions(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ["predictions", "results", "matches", "data", "items", "output"]:
            if isinstance(obj.get(k), list):
                return obj[k]
        if "match" in obj and ("predicted_score" in obj or "top3" in obj or "final_direction" in obj):
            return [obj]
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
            return fv * 100.0
        return fv
    fv = _f(v, 0.0)
    if 0 < fv <= 1:
        return fv * 100.0
    return fv


def _score_from_candidate(obj: Any) -> str:
    if isinstance(obj, str):
        m = _SCORE_RE.search(obj)
        if m:
            return f"{int(m.group(1))}-{int(m.group(2))}"
        return _normalize_score_text(obj)
    if not isinstance(obj, dict):
        return ""
    for k in ["score", "predicted_score", "ai_score", "final_score", "比分", "预测比分", "scoreline", "correct_score"]:
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


def _normalize_top3(item: Dict[str, Any], predicted_score: str = "") -> List[Dict[str, Any]]:
    raw = None
    for k in ["top3", "top_3", "top_scores", "scores", "score_candidates", "candidates", "correct_scores"]:
        if isinstance(item.get(k), list):
            raw = item[k]
            break
    if raw is None:
        raw = []
    out: List[Dict[str, Any]] = []
    seen = set()
    for cand in raw[:10]:
        sc = _score_from_candidate(cand)
        if _parse_score(sc)[0] is None or sc in seen:
            continue
        seen.add(sc)
        if isinstance(cand, dict):
            prob = cand.get("prob", cand.get("probability", cand.get("pct", cand.get("chance", 0))))
            logic = cand.get("logic", cand.get("market_logic", cand.get("reason", cand.get("explanation", ""))))
        else:
            prob, logic = 0, ""
        out.append({"score": sc, "prob": round(_prob_to_float(prob), 3), "logic": str(logic)[:900]})
        if len(out) >= 3:
            break
    if not out and predicted_score and _parse_score(predicted_score)[0] is not None:
        out = [{"score": predicted_score, "prob": 0.0, "logic": "top3_missing_but_predicted_score_present"}]
    if out and predicted_score and out[0]["score"] != predicted_score and _parse_score(predicted_score)[0] is not None:
        out = [{"score": predicted_score, "prob": out[0].get("prob", 0.0), "logic": "protocol_repair_top3_primary_score"}] + [x for x in out if x["score"] != predicted_score]
        out = out[:3]
    return out


def _normalize_direction_probs(item: Dict[str, Any]) -> Dict[str, float]:
    cand = None
    for k in ["direction_probs", "direction_probabilities", "probabilities", "方向概率", "三项概率"]:
        if isinstance(item.get(k), dict):
            cand = item.get(k)
            break
    if cand is None:
        cand = {}
    alias = {
        "home": "home", "主": "home", "主胜": "home", "胜": "home", "home_win": "home", "win": "home",
        "draw": "draw", "平": "draw", "平局": "draw", "和": "draw", "same": "draw", "tie": "draw",
        "away": "away", "客": "away", "客胜": "away", "负": "away", "away_win": "away", "lose": "away",
    }
    raw = {"home": 0.0, "draw": 0.0, "away": 0.0}
    if isinstance(cand, dict):
        for k, v in cand.items():
            kk = alias.get(str(k).strip().lower(), alias.get(str(k).strip()))
            if kk in raw:
                raw[kk] += _prob_to_float(v)
    s = sum(raw.values())
    if s <= 0:
        return {"home": 33.3, "draw": 33.3, "away": 33.4, "_synthetic_probs": True}
    out = {k: round(v / s * 100.0, 1) for k, v in raw.items()}
    out["_synthetic_probs"] = False
    return out


def _normalize_web_research(item: Dict[str, Any]) -> Dict[str, Any]:
    web = item.get("web_research")
    if not isinstance(web, dict):
        audit = item.get("audit") if isinstance(item.get("audit"), dict) else {}
        web = audit.get("web_research") if isinstance(audit.get("web_research"), dict) else {}
    sources = web.get("sources", []) if isinstance(web, dict) else []
    if not isinstance(sources, list):
        sources = []
    if AI_WEB_MAX_SOURCES_PER_MATCH > 0:
        sources = sources[:AI_WEB_MAX_SOURCES_PER_MATCH]
    warnings = []
    used = bool(web.get("used", False)) if isinstance(web, dict) else False
    if used and not sources:
        warnings.append("web_used_but_no_sources")
    for i, src in enumerate(sources):
        if not isinstance(src, dict):
            warnings.append(f"source_{i}_not_object")
            continue
        if not src.get("title") and not src.get("url"):
            warnings.append(f"source_{i}_missing_title_url")
        if not src.get("claim"):
            warnings.append(f"source_{i}_missing_claim")
        if not src.get("published_at") and AI_WARN_MISSING_PUBLISHED_AT:
            warnings.append(f"source_{i}_missing_published_at")
    return {
        "used": used,
        "failure_reason": str(web.get("failure_reason", "")) if isinstance(web, dict) else "web_research_missing",
        "search_queries": web.get("search_queries", []) if isinstance(web.get("search_queries", []), list) else [],
        "sources": sources,
        "freshness_grade": str(web.get("freshness_grade", "missing")) if isinstance(web, dict) else "missing",
        "key_findings": web.get("key_findings", []) if isinstance(web.get("key_findings", []), list) else [],
        "source_conflicts": web.get("source_conflicts", []) if isinstance(web.get("source_conflicts", []), list) else [],
        "validation_warnings": warnings,
    }


def _normalize_recommendation(item: Dict[str, Any]) -> Dict[str, Any]:
    rec = item.get("recommendation") if isinstance(item.get("recommendation"), dict) else {}
    tier = str(rec.get("tier", item.get("recommendation_tier", "D"))).strip().upper()
    if tier not in {"S", "A", "B", "C", "D"}:
        tier = "D"
    risk_tags = rec.get("risk_tags", [])
    if not isinstance(risk_tags, list):
        risk_tags = [str(risk_tags)] if risk_tags else []
    return {
        "tier": tier,
        "is_recommended": bool(rec.get("is_recommended", tier in {"S", "A", "B"})),
        "top4_priority": _i(rec.get("top4_priority", 99), 99),
        "bet_confidence": int(_clip(_f(rec.get("bet_confidence", item.get("ai_confidence", 0)), 0), 0, 100)),
        "direction_stability": str(rec.get("direction_stability", "unknown")),
        "score_stability": str(rec.get("score_stability", "unknown")),
        "risk_level": str(rec.get("risk_level", item.get("risk_level", "medium"))),
        "risk_tags": risk_tags[:12],
        "bet_action": str(rec.get("bet_action", "")),
        "why_this_can_fail": rec.get("why_this_can_fail", []) if isinstance(rec.get("why_this_can_fail", []), list) else [],
        "minimum_evidence_needed": rec.get("minimum_evidence_needed", []) if isinstance(rec.get("minimum_evidence_needed", []), list) else [],
        "why_recommended": str(rec.get("why_recommended", item.get("reason", "")))[:1500],
    }


def _match_index_from_item(item: Dict[str, Any], fallback_idx: int, expected_set: set) -> Optional[int]:
    raw = item.get("match", item.get("index", item.get("match_index", item.get("id", item.get("序号")))))
    if isinstance(raw, int):
        return raw if raw in expected_set else None
    if raw is not None:
        s = str(raw).strip()
        m = re.match(r"^\s*(\d+)\s*$", s) or re.match(r"^\s*\[(\d+)\]", s) or re.search(r"(?:match|场次|第)\s*(\d+)", s, re.I)
        if m:
            idx = int(m.group(1))
            return idx if idx in expected_set else None
    return fallback_idx if fallback_idx in expected_set else None


def normalize_ai_predictions(obj: Any, expected_matches: List[int], source_model: str, phase: str) -> Dict[int, Dict[str, Any]]:
    items = _unwrap_predictions(obj)
    expected_set = set(expected_matches)
    out: Dict[int, Dict[str, Any]] = {}
    for pos, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        idx = _match_index_from_item(item, pos if pos in expected_set else expected_matches[min(pos - 1, len(expected_matches) - 1)], expected_set)
        if idx is None:
            continue
        predicted_score = _score_from_candidate(item.get("predicted_score", item.get("score", item.get("ai_score", ""))))
        if _parse_score(predicted_score)[0] is None:
            top3_raw = item.get("top3")
            if isinstance(top3_raw, list) and top3_raw:
                predicted_score = _score_from_candidate(top3_raw[0])
        if _parse_score(predicted_score)[0] is None:
            continue
        score_dir = _score_direction(predicted_score)
        raw_dir = _dir_from_any(item.get("final_direction", item.get("direction", ""))) or score_dir or "draw"
        final_direction = score_dir or raw_dir
        direction_conflict = bool(score_dir in VALID_DIRS and raw_dir in VALID_DIRS and score_dir != raw_dir)
        top3 = _normalize_top3(item, predicted_score)
        direction_probs = _normalize_direction_probs(item)
        web = _normalize_web_research(item)
        rec = _normalize_recommendation(item)
        warnings = []
        if direction_conflict:
            warnings.append(f"dir_score_conflict_protocol_fixed:{raw_dir}->{score_dir}")
        if direction_probs.get("_synthetic_probs"):
            warnings.append("direction_probs_missing_synthetic")
        warnings.extend(web.get("validation_warnings", []))
        if not isinstance(item.get("anchor_audit"), dict):
            warnings.append("anchor_audit_missing_or_invalid")
        out[idx] = {
            "match": idx,
            "source_model": source_model,
            "source_phase": phase,
            "final_direction": final_direction,
            "raw_ai_direction": raw_dir,
            "predicted_score": predicted_score,
            "direction_probs": direction_probs,
            "goal_band": _score_goal_band(predicted_score),
            "btts": _score_btts(predicted_score),
            "top3": top3,
            "anchor_audit": item.get("anchor_audit", {}) if isinstance(item.get("anchor_audit"), dict) else {},
            "market_interpretation": item.get("market_interpretation", {}) if isinstance(item.get("market_interpretation"), dict) else {},
            "money_flow": item.get("money_flow", {}) if isinstance(item.get("money_flow"), dict) else {},
            "contextual_logic": item.get("contextual_logic", {}) if isinstance(item.get("contextual_logic"), dict) else {},
            "rejected_cases": item.get("rejected_cases", {}) if isinstance(item.get("rejected_cases"), dict) else {},
            "web_research": web,
            "final_web_audit": item.get("final_web_audit", {}) if isinstance(item.get("final_web_audit"), dict) else {},
            "recommendation": rec,
            "data_quality": item.get("data_quality", {}) if isinstance(item.get("data_quality"), dict) else {},
            "reason": str(item.get("reason", item.get("analysis", item.get("explanation", ""))))[:5000],
            "validation_warnings": list(dict.fromkeys(warnings)),
            "raw_item": item,
        }
    return out


def parse_critic_reports(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict):
        rows = obj.get("critic_reports", obj.get("reports", []))
    elif isinstance(obj, list):
        rows = obj
    else:
        rows = []
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def parse_repairs(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict):
        rows = obj.get("repairs", [])
    else:
        rows = []
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


# ============================================================
# Mock API：只验证工程闭环，不代表真实命中率
# ============================================================

def _extract_json_lines_from_prompt(prompt: str) -> List[Dict[str, Any]]:
    rows = []
    for line in prompt.splitlines():
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "match" in obj and "identity" in obj:
                rows.append(obj)
        except Exception:
            continue
    return rows


def _mock_score_from_anchors(e: Dict[str, Any], best: str, idx: int) -> str:
    anchor = e.get("ai_anchor_facts_no_judgement", {})
    score_f = anchor.get("score_anchor_facts", {}) if isinstance(anchor, dict) else {}
    total_f = anchor.get("total_goal_anchor_facts", {}) if isinstance(anchor, dict) else {}
    spec = score_f.get("specific_odds", {}) if isinstance(score_f, dict) else {}
    mode_g = total_f.get("mode_goals") if isinstance(total_f, dict) else None
    s00 = _f(spec.get("0-0"), 0)
    s11 = _f(spec.get("1-1"), 0)
    a4 = _f(total_f.get("specific_odds", {}).get("4"), 0) if isinstance(total_f, dict) else 0

    if 0 < s00 <= 11 and mode_g in (0, 1, 2):
        return "0-0" if idx % 2 else "1-0" if best == "home" else "0-1" if best == "away" else "0-0"
    if 0 < s11 <= 7.5 and mode_g in (1, 2, 3, None):
        return "1-1" if best == "draw" or idx % 2 == 0 else "2-1" if best == "home" else "1-2"
    if a4 > 6 and mode_g in (0, 1, 2, 3):
        if best == "home":
            return "1-0" if idx % 2 else "2-0"
        if best == "away":
            return "0-1" if idx % 2 else "0-2"
        return "0-0" if idx % 2 else "1-1"
    if mode_g in (4, 5, 6, 7):
        return "3-2" if best == "home" else "2-3" if best == "away" else "2-2"
    return {"home": "2-1" if idx % 3 == 0 else "1-0", "draw": "1-1" if idx % 2 == 0 else "0-0", "away": "1-2" if idx % 3 == 0 else "0-1"}.get(best, "1-1")


def _mock_ai_response(ai_name: str, phase: str, prompt: str, expected_matches: List[int]) -> Dict[str, Any]:
    if phase == "critic":
        return {"critic_reports": [{"match": idx, "critic_model": ai_name, "target_findings": [], "own_revision_hint": "mock critic: 未发现结构性硬冲突"} for idx in expected_matches]}
    if phase == "consistency":
        return {"repairs": [{"match": idx, "valid": True, "warnings": [], "repair": {}} for idx in expected_matches]}

    evs = _extract_json_lines_from_prompt(prompt)
    if not evs:
        evs = [{"match": idx, "identity": {"home_team": "Home", "away_team": "Away", "league": ""}, "lottery_market_1x2": {}, "derived_market_facts_no_judgement": {}, "ai_anchor_facts_no_judgement": {}} for idx in expected_matches]

    preds = []
    for e in evs:
        idx = int(e.get("match"))
        one = e.get("lottery_market_1x2", {})
        sp_h, sp_d, sp_a = _f(one.get("home"), 2.3), _f(one.get("draw"), 3.2), _f(one.get("away"), 3.1)
        implied = {"home": 1 / max(sp_h, 1.01), "draw": 1 / max(sp_d, 1.01), "away": 1 / max(sp_a, 1.01)}
        if ai_name == "grok" and idx % 5 == 0:
            best = "draw"
        elif ai_name == "gemini" and idx % 7 == 0:
            best = "away" if sp_a < sp_h else "home"
        else:
            best = max(implied.items(), key=lambda kv: kv[1])[0]
        if phase in ("final", "fallback_referee"):
            best = max(implied.items(), key=lambda kv: kv[1])[0]
        score = _mock_score_from_anchors(e, best, idx)
        probs_raw = implied.copy()
        s = sum(probs_raw.values()) or 1
        probs = {k: round(v / s * 100, 1) for k, v in probs_raw.items()}
        if probs[best] < max(probs.values()):
            probs[best] = max(probs.values())
        rec_tier = "A" if phase == "final" and max(probs.values()) >= 42 else "B" if max(probs.values()) >= 37 else "C"
        preds.append({
            "match": idx,
            "final_direction": _score_direction(score) or best,
            "predicted_score": score,
            "direction_probs": probs,
            "goal_band": _score_goal_band(score),
            "btts": _score_btts(score),
            "top3": [{"score": score, "prob": 15, "logic": f"mock {ai_name}/{phase}: 锚点闭环主比分"}, {"score": "1-1" if score != "1-1" else "1-0", "prob": 10, "logic": "mock secondary"}],
            "anchor_audit": {
                "answered_cross_anchor_questions": e.get("ai_anchor_facts_no_judgement", {}).get("mandatory_cross_anchor_questions", []),
                "zero_zero_case": "mock: 已审计0-0赔率与总进球模态",
                "one_one_case": "mock: 已审计1-1赔率与BTTS",
                "one_goal_margin_case": "mock: 已审计一球边际候选",
                "four_plus_case": "mock: 已审计4+尾部",
                "handicap_score_shape_case": "mock: 已审计让球盘比分形态",
                "league_style_case": "mock: 已审计联赛风格",
                "final_score_vs_anchor_summary": "mock: 仅用于工程测试，不代表真实命中率",
            },
            "market_interpretation": {"one_x_two": "mock", "handicap": "mock", "correct_score": "mock", "total_goals": "mock", "half_full_time": "mock", "external_market": "mock"},
            "money_flow": {"public_money_direction": "unclear", "sharp_money_direction": "unclear", "sharp_confidence": 0, "reverse_line_movement": False, "steam_move": "unclear", "evidence": "mock no real web"},
            "contextual_logic": {"league_style": "mock", "team_style": "mock", "tempo": "medium", "score_shape": "mock", "btts_likelihood": _score_btts(score), "rotation_risk": "unclear"},
            "rejected_cases": {"home": "mock", "draw": "mock", "away": "mock"},
            "web_research": {"used": False, "failure_reason": "AI_MOCK_MODE_no_real_web", "search_queries": [], "sources": [], "freshness_grade": "missing", "key_findings": [], "source_conflicts": []},
            "recommendation": {"tier": rec_tier, "is_recommended": rec_tier in {"S", "A", "B"}, "top4_priority": idx, "bet_confidence": int(max(probs.values())), "direction_stability": "medium", "score_stability": "medium", "risk_level": "medium", "risk_tags": ["mock_mode"], "why_recommended": "mock 用于工程测试，不代表真实命中率"},
            "data_quality": {"missing": ["real_web", "real_ai_reasoning"], "raw_packet_quality": "mock"},
            "reason": f"mock {ai_name}/{phase} response for engineering test",
            "final_web_audit": {"web_used_by_final": False, "decisive_web_evidence": [], "ignored_web_evidence": [], "web_confidence": 0},
        })
    return {"predictions": preds}


# ============================================================
# 落盘
# ============================================================

def _make_run_id(evidence_all: List[Dict[str, Any]]) -> str:
    first = ""
    if evidence_all:
        ident = evidence_all[0].get("identity", {})
        first = f"{ident.get('home_team','')}_{ident.get('away_team','')}"
    safe = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", first)[:32]
    return f"{_now_ts()}_{len(evidence_all)}m_{_short_hash(_hash_obj(evidence_all))}_{safe}".strip("_")


def _save_snapshot(run_id: str, tag: str, obj: Any) -> str:
    if not AI_PHASE_SNAPSHOT_ENABLED:
        return ""
    try:
        _ensure_dir(AI_PHASE_RESULT_DIR)
        path = os.path.join(AI_PHASE_RESULT_DIR, f"{run_id}_{tag}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
        return path
    except Exception as e:
        print(f"[SNAPSHOT ERROR] {tag}: {str(e)[:160]}")
        return ""


# ============================================================
# Orchestrator
# ============================================================

def _chunk_evidence(evidence_all: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    chunks: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_len = 0
    for e in evidence_all:
        s = len(_safe_json_line(e))
        if cur and (len(cur) >= AI_CHUNK_SIZE or cur_len + s > AI_MAX_PROMPT_CHARS_PER_CHUNK):
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(e)
        cur_len += s
    if cur:
        chunks.append(cur)
    return chunks


async def _run_one_chunk(session: Optional[Any], run_id: str, chunk_id: int, evidence_batch: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    expected = [int(e["match"]) for e in evidence_batch]
    print(f"  [Chunk {chunk_id}] matches={expected} size={len(evidence_batch)}")

    phase1: Dict[str, Dict[int, Dict[str, Any]]] = {n: {} for n in PHASE1_NAMES}
    tasks = []
    for ai in PHASE1_NAMES:
        prompt = build_phase1_prompt(evidence_batch, ai)
        tasks.append(async_call_ai_json(session, ai, _phase1_system(ai), prompt, "phase1", expected))
    phase1_returns = await asyncio.gather(*tasks, return_exceptions=True)
    for ret in phase1_returns:
        if isinstance(ret, Exception):
            print(f"  [Chunk {chunk_id}] Phase1 exception: {ret}")
            continue
        ai, obj, st = ret
        rows = normalize_ai_predictions(obj, expected, ai, "phase1")
        phase1[ai] = rows
        print(f"  [Chunk {chunk_id}] {ai.upper()} phase1 {len(rows)}/{len(expected)} status={st.get('status')}")
    _save_snapshot(run_id, f"chunk{chunk_id}_phase1", {"phase1": {m: {str(k): v for k, v in rows.items()} for m, rows in phase1.items()}, "status": AI_CALL_STATUS})

    critic_reports: Dict[str, List[Dict[str, Any]]] = {}
    if AI_ENABLE_CROSS_EXAM:
        critic_tasks = []
        for ai in PHASE1_NAMES:
            prompt = build_critic_prompt(evidence_batch, ai, phase1)
            critic_tasks.append(async_call_ai_json(session, ai, _phase1_system(ai), prompt, "critic", expected))
        critic_returns = await asyncio.gather(*critic_tasks, return_exceptions=True)
        for ret in critic_returns:
            if isinstance(ret, Exception):
                continue
            ai, obj, st = ret
            critic_reports[ai] = parse_critic_reports(obj)
            print(f"  [Chunk {chunk_id}] {ai.upper()} critic reports={len(critic_reports[ai])} status={st.get('status')}")
    _save_snapshot(run_id, f"chunk{chunk_id}_critics", {"critic_reports": critic_reports, "status": AI_CALL_STATUS})

    final_rows: Dict[int, Dict[str, Any]] = {}
    final_ai = AI_FINAL_REFEREE_MODEL if AI_FINAL_REFEREE_MODEL in AI_NAMES else "gemini"
    final_prompt = build_gemini_final_prompt(evidence_batch, phase1, critic_reports)
    _, final_obj, final_st = await async_call_ai_json(session, final_ai, _phase1_system("gemini"), final_prompt, "final", expected)
    final_rows = normalize_ai_predictions(final_obj, expected, final_ai, "final")
    print(f"  [Chunk {chunk_id}] {final_ai.upper()} final {len(final_rows)}/{len(expected)} status={final_st.get('status')}")

    missing = [idx for idx in expected if idx not in final_rows]
    if missing and AI_ENABLE_FALLBACK_REFEREE:
        fallback_ai = AI_FALLBACK_REFEREE_MODEL if AI_FALLBACK_REFEREE_MODEL in PHASE1_NAMES else "gpt"
        fb_prompt = build_fallback_referee_prompt([e for e in evidence_batch if e["match"] in missing], phase1, critic_reports)
        _, fb_obj, fb_st = await async_call_ai_json(session, fallback_ai, _phase1_system(fallback_ai), fb_prompt, "fallback_referee", missing)
        fb_rows = normalize_ai_predictions(fb_obj, missing, fallback_ai, "fallback_referee")
        final_rows.update(fb_rows)
        print(f"  [Chunk {chunk_id}] FALLBACK {fallback_ai.upper()} {len(fb_rows)}/{len(missing)} status={fb_st.get('status')}")

    missing = [idx for idx in expected if idx not in final_rows]
    if missing:
        for idx in missing:
            final_rows[idx] = _phase1_consensus_fallback(idx, phase1)
        print(f"  [Chunk {chunk_id}] Phase1 consensus fallback filled {len(missing)}")

    if AI_ENABLE_CONSISTENCY_JUDGE and final_rows:
        judge_ai = AI_CONSISTENCY_JUDGE_MODEL if AI_CONSISTENCY_JUDGE_MODEL in AI_NAMES else "gpt"
        judge_prompt = build_consistency_judge_prompt(evidence_batch, final_rows)
        _, judge_obj, judge_st = await async_call_ai_json(session, judge_ai, _phase1_system(judge_ai), judge_prompt, "consistency", expected)
        repairs = parse_repairs(judge_obj)
        _apply_consistency_repairs(final_rows, repairs)
        print(f"  [Chunk {chunk_id}] CONSISTENCY {judge_ai.upper()} repairs={len(repairs)} status={judge_st.get('status')}")

    for idx, row in final_rows.items():
        phase1_pack = {}
        for model_name in PHASE1_NAMES:
            pr = phase1.get(model_name, {}).get(idx, {})
            if pr:
                phase1_pack[model_name] = _short_prediction_for_prompt(pr)
        row["phase1_model_outputs"] = phase1_pack
        row["critic_reports_by_model"] = {model_name: [cr for cr in reports if _i(cr.get("match"), 0) == idx] for model_name, reports in critic_reports.items()}

    _save_snapshot(run_id, f"chunk{chunk_id}_final", {"final": {str(k): v for k, v in final_rows.items()}, "status": AI_CALL_STATUS})
    return final_rows


def _phase1_consensus_fallback(idx: int, phase1: Dict[str, Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
    candidates = []
    for ai in PHASE1_NAMES:
        r = phase1.get(ai, {}).get(idx)
        if not r:
            continue
        rec = r.get("recommendation", {})
        candidates.append((rec.get("bet_confidence", 0), ai, r))
    if not candidates:
        return _abstain_ai_prediction(idx, "all_ai_failed")
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, ai, r = candidates[0]
    out = dict(r)
    out["source_model"] = ai
    out["source_phase"] = "phase1_consensus_fallback"
    out.setdefault("validation_warnings", []).append("gemini_final_missing_used_phase1_ai_fallback")
    return out


def _apply_consistency_repairs(final_rows: Dict[int, Dict[str, Any]], repairs: List[Dict[str, Any]]) -> None:
    for row in repairs:
        idx = _i(row.get("match"), 0)
        if idx not in final_rows:
            continue
        warnings = row.get("warnings", [])
        if isinstance(warnings, list):
            final_rows[idx].setdefault("validation_warnings", []).extend([str(w) for w in warnings])
        repair = row.get("repair")
        if not isinstance(repair, dict):
            continue
        allowed = {"goal_band", "btts", "top3", "web_research", "data_quality", "anchor_audit"}
        for k, v in repair.items():
            if k in allowed:
                final_rows[idx][k] = v
        _protocol_enforce_prediction(final_rows[idx])


def _protocol_enforce_prediction(r: Dict[str, Any]) -> Dict[str, Any]:
    score = _score_from_candidate(r.get("predicted_score", ""))
    if _parse_score(score)[0] is None:
        return r
    score_dir = _score_direction(score)
    raw_dir = _dir_from_any(r.get("final_direction"))
    warnings = list(r.get("validation_warnings", []))
    if score_dir in VALID_DIRS and raw_dir in VALID_DIRS and score_dir != raw_dir:
        warnings.append(f"protocol_direction_fixed_to_score:{raw_dir}->{score_dir}")
    if score_dir in VALID_DIRS:
        r["final_direction"] = score_dir
    r["predicted_score"] = score
    r["goal_band"] = _score_goal_band(score)
    r["btts"] = _score_btts(score)
    r["top3"] = _normalize_top3(r, score)
    if not isinstance(r.get("anchor_audit"), dict):
        warnings.append("anchor_audit_missing_or_invalid")
        r["anchor_audit"] = {}
    r["validation_warnings"] = list(dict.fromkeys(warnings))
    return r


async def run_ai_native_web(evidence_all: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    global _LAST_AI_RUN_METADATA
    for _n in AI_NAMES:
        AI_CALL_STATUS[_n] = {}
    AI_RESULT_FILES.clear()
    _LAST_AI_RUN_METADATA = {}

    run_id = _make_run_id(evidence_all)
    chunks = _chunk_evidence(evidence_all)
    debug_ai_config()
    print(f"  [{ENGINE_VERSION}] run_id={run_id} chunks={len(chunks)}")

    all_final: Dict[int, Dict[str, Any]] = {}
    session = None
    try:
        if not AI_MOCK_MODE and aiohttp is not None:
            connector = aiohttp.TCPConnector(limit=8, use_dns_cache=False, ttl_dns_cache=0, force_close=False)
            session = aiohttp.ClientSession(connector=connector)
        for i, chunk in enumerate(chunks, 1):
            rows = await _run_one_chunk(session, run_id, i, chunk)
            all_final.update(rows)
    finally:
        if session is not None:
            await session.close()

    agg_path = _save_snapshot(run_id, "final_all_chunks", {"final": {str(k): v for k, v in all_final.items()}, "status": AI_CALL_STATUS})
    _LAST_AI_RUN_METADATA = {
        "run_id": run_id,
        "engine_version": ENGINE_VERSION,
        "chunk_count": len(chunks),
        "phase_result_dir": AI_PHASE_RESULT_DIR,
        "aggregate_file": agg_path,
        "ai_call_status": dict(AI_CALL_STATUS),
        "mock_mode": AI_MOCK_MODE,
        "research_mode": AI_RESEARCH_MODE,
        "native_web": AI_NATIVE_WEB,
        "cross_exam": AI_ENABLE_CROSS_EXAM,
        "consistency_judge": AI_ENABLE_CONSISTENCY_JUDGE,
    }
    return all_final


# ============================================================
# 前端兼容 Adapter / Top4
# ============================================================

def _abstain_ai_prediction(idx: int, reason: str) -> Dict[str, Any]:
    return {
        "match": idx,
        "source_model": "none",
        "source_phase": "abstain",
        "final_direction": "abstain",
        "predicted_score": "弃权",
        "direction_probs": {"home": 0.0, "draw": 0.0, "away": 0.0},
        "goal_band": "",
        "btts": "unclear",
        "top3": [],
        "anchor_audit": {},
        "recommendation": {"tier": "D", "is_recommended": False, "top4_priority": 999, "bet_confidence": 0, "risk_level": "high", "risk_tags": [reason], "why_recommended": "弃权"},
        "reason": reason,
        "validation_warnings": [reason],
    }


def _tier_rank(tier: str) -> int:
    return {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}.get(str(tier).upper(), 0)


def _min_tier_ok(tier: str) -> bool:
    return _tier_rank(tier) >= _tier_rank(MIN_AI_RECOMMEND_TIER)


def _direction_pct_display(pct: Dict[str, Any]) -> Dict[str, float]:
    return {k: round(_f(pct.get(k), 0.0), 1) for k in ["home", "draw", "away"]}


def _goal_range_from_score(score: str) -> Tuple[int, int, str]:
    total = _score_total(score)
    if total is None:
        return 0, 0, "ai_native_unknown"
    if total <= 1:
        return 0, 1, "ai_native_0_1_goals"
    if total == 2:
        return 1, 2, "ai_native_2_goals"
    if total == 3:
        return 2, 3, "ai_native_3_goals"
    if total == 4:
        return 3, 4, "ai_native_4_goals"
    if total == 5:
        return 4, 5, "ai_native_5_goals"
    return 5, 8, "ai_native_6plus_goals"


def _legacy_model_score(ai_r: Dict[str, Any], model_name: str, final_score: str) -> str:
    phase1 = ai_r.get("phase1_model_outputs", {}) if isinstance(ai_r.get("phase1_model_outputs"), dict) else {}
    row = phase1.get(model_name, {}) if isinstance(phase1.get(model_name), dict) else {}
    sc = _score_from_candidate(row.get("predicted_score", "")) if row else ""
    if _parse_score(sc)[0] is not None:
        return sc
    return final_score


def _legacy_model_analysis(ai_r: Dict[str, Any], model_name: str) -> str:
    phase1 = ai_r.get("phase1_model_outputs", {}) if isinstance(ai_r.get("phase1_model_outputs"), dict) else {}
    row = phase1.get(model_name, {}) if isinstance(phase1.get(model_name), dict) else {}
    if row:
        reason = str(row.get("reason", "")).strip()
        if reason:
            return reason[:3000]
        rec = row.get("recommendation", {}) if isinstance(row.get("recommendation"), dict) else {}
        why = str(rec.get("why_recommended", "")).strip()
        if why:
            return why[:3000]
    return "该模型本轮未返回可展示分析；最终结果以AI终审为准。"


def adapt_ai_to_frontend(ai_r: Dict[str, Any], match_obj: Dict[str, Any]) -> Dict[str, Any]:
    if not ai_r or ai_r.get("predicted_score") == "弃权" or ai_r.get("final_direction") == "abstain":
        return _abstain_prediction("AI全部失败或最终弃权")

    _protocol_enforce_prediction(ai_r)
    score = ai_r.get("predicted_score", "弃权")
    direction = ai_r.get("final_direction", _score_direction(score) or "draw")
    pct = _direction_pct_display(ai_r.get("direction_probs", {}))
    rec = ai_r.get("recommendation", {}) if isinstance(ai_r.get("recommendation"), dict) else {}
    tier = str(rec.get("tier", "D")).upper()
    is_ai_recommended = bool(rec.get("is_recommended", False)) and _min_tier_ok(tier)
    top_candidates = []
    for cand in ai_r.get("top3", [])[:8]:
        if isinstance(cand, dict):
            sc = _score_from_candidate(cand)
            if _parse_score(sc)[0] is not None:
                top_candidates.append((sc, round(_prob_to_float(cand.get("prob", 0)), 3)))
    gmin, gmax, scenario = _goal_range_from_score(score)
    h, a = _parse_score(score)
    total_goals = (h or 0) + (a or 0) if h is not None and a is not None else 0
    final_odds = get_market_odds_for_score(match_obj, score)
    market_implied = round(100.0 / final_odds, 3) if final_odds > 1.05 else None
    warnings = [w for w in list(ai_r.get("validation_warnings", [])) if "missing_published_at" not in str(w)]

    web_research = ai_r.get("web_research", {}) if isinstance(ai_r.get("web_research"), dict) else {}
    money_flow = ai_r.get("money_flow", {}) if isinstance(ai_r.get("money_flow"), dict) else {}
    market_interpretation = ai_r.get("market_interpretation", {}) if isinstance(ai_r.get("market_interpretation"), dict) else {}
    contextual_logic = ai_r.get("contextual_logic", {}) if isinstance(ai_r.get("contextual_logic"), dict) else {}
    anchor_audit = ai_r.get("anchor_audit", {}) if isinstance(ai_r.get("anchor_audit"), dict) else {}

    evidences = [
        "AI-NATIVE：本地不做足球预测判断；方向、比分、Top4等级均来自 AI 输出。",
        "ANCHOR-AUDIT：本地只提供0-0/1-1/总进球/让球盘/联赛风格事实锚点，AI必须在anchor_audit中解释。",
        "WEB-AUGMENTED：Prompt 要求 AI 联网并输出 sources；本地只校验来源字段完整性。",
        "LOCAL PROTOCOL ONLY：本地只修字段闭环，如 goal_band/btts 与比分一致，不改变足球观点。",
        f"final_model={ai_r.get('source_model')} phase={ai_r.get('source_phase')} score={score} direction={direction}",
        "anchor_audit:" + _json_compact(anchor_audit, 1500),
        "market_interpretation:" + _json_compact(market_interpretation, 1200),
        "money_flow:" + _json_compact(money_flow, 1200),
        "contextual_logic:" + _json_compact(contextual_logic, 1200),
        "web_research:" + _json_compact(web_research, 1200),
        "recommendation:" + _json_compact(rec, 1200),
    ]

    return {
        "predicted_score": score,
        "predicted_label": _score_display_label(score, direction),
        "result": _direction_cn(direction),
        "display_direction": _direction_cn(direction),
        "final_direction": direction,
        "raw_ai_direction": ai_r.get("raw_ai_direction", direction),
        "score_implied_direction": _score_direction(score),
        "dir_score_conflict": any("dir_score_conflict" in w or "protocol_direction_fixed" in w for w in warnings),
        "is_abstain": False,
        "is_score_others": _score_display_label(score, direction) in ("胜其他", "平其他", "负其他"),
        "home_win_pct": pct.get("home", 0.0),
        "draw_pct": pct.get("draw", 0.0),
        "away_win_pct": pct.get("away", 0.0),
        "confidence": int(_clip(_f(rec.get("bet_confidence", max(pct.values()) if pct else 0), 0), 0, 100)),
        "confidence_meaning": "AI recommendation.bet_confidence，非历史校准命中率；本地不改概率。",
        "risk_level": rec.get("risk_level", "medium"),
        "goal_band": ai_r.get("goal_band", _score_goal_band(score)),
        "btts": ai_r.get("btts", _score_btts(score)),
        "btts_ai": ai_r.get("btts", _score_btts(score)),
        "both_score": "是" if _score_btts(score) == "yes" else "否",
        "over_under_2_5": "大" if total_goals >= 3 else "小",
        "expected_total_goals": total_goals,
        "goal_range": (gmin, gmax),
        "scenario": scenario,
        "recommendation": rec,
        "recommendation_tier": tier,
        "recommend_gate_pass": is_ai_recommended,
        "recommend_gate_reasons": [] if is_ai_recommended else ["ai_not_recommended_or_below_min_tier"],
        "safe_top_gate_score": _f(rec.get("bet_confidence", 0), 0),
        "overall_selection_score": _f(rec.get("bet_confidence", 0), 0),
        "direction_selection_score": _f(rec.get("bet_confidence", 0), 0),
        "score_shape_score": _f(rec.get("bet_confidence", 0), 0),
        "direction_tier": tier,
        "score_tier": tier,
        "recommendation_downgrade_reasons": [],
        "top_score_candidates": top_candidates,
        "unified_matrix_top_scores": top_candidates,
        "score_model_prob": top_candidates[0][1] if top_candidates else 0.0,
        "score_market_odds": final_odds,
        "score_market_implied_pct": market_implied,
        "anchor_audit": anchor_audit,
        "market_interpretation": market_interpretation,
        "money_flow": money_flow,
        "contextual_logic": contextual_logic,
        "rejected_cases": ai_r.get("rejected_cases", {}),
        "web_research": web_research,
        "final_web_audit": ai_r.get("final_web_audit", {}),
        "data_quality": ai_r.get("data_quality", {}),
        "ai_native_reason": ai_r.get("reason", ""),
        "validation_warnings": list(dict.fromkeys(warnings)),
        "bayesian_evidences": evidences,
        "bayesian_prior": {},
        "override_triggered": False,
        "traps_detected": [],
        "trap_count": 0,
        "trap_severity": 0,
        "trap_details": [],
        "trap_flags": {},
        "fair_1x2": {},
        "fair_1x2_method": "disabled_ai_native_web",
        "market_overround": 0.0,
        "raw_implied_1x2": {},
        "crs_shape": "raw_correct_score_odds_only_no_local_crs_matrix",
        "crs_moments": {},
        "crs_margin": 0.0,
        "crs_coverage": 0.0,
        "crs_implied_probs": {},
        "crs_low_rank_info": {},
        "unified_goal_probs": {},
        "fair_1x2_pack": {},
        "mixed_target_dir": {},
        "unified_source": "ai_native_web_anchor_audit",
        "decision_source": f"ai_native:{ai_r.get('source_model')}:{ai_r.get('source_phase')}",
        "ai_authority_mode": "ai_native_web_no_local_football_judgement",
        "ev_note": "disabled_local_ev_no_local_probability",
        "suggested_kelly": 0.0,
        "edge_vs_market": 0.0,
        "is_value": False,
        "smart_money_signal": str(money_flow.get("evidence", ""))[:1000],
        "smart_signals": rec.get("risk_tags", []),
        "sharp_detected": str(money_flow.get("sharp_money_direction", "unclear")) not in ("", "unclear", "none"),
        "sharp_dir": money_flow.get("sharp_money_direction", "unclear"),
        "sharp_conflict": False,
        "sharp_unresolved": False,
        "sharp_audit": money_flow,
        "score_market_evidence": market_interpretation,
        "score_market_gate_pass": None,
        "score_market_alignment": None,
        "sharp_clearance_score": None,
        "cold_door": {"is_cold_door": False, "strength": 0, "level": "AI-native字段，不由本地判断冷门", "signals": rec.get("risk_tags", []), "sharp_confirmed": str(money_flow.get("sharp_money_direction", "unclear")) not in ("", "unclear", "none"), "dark_verdict": ""},
        "xG_home": "?",
        "xG_away": "?",
        "bookmaker_implied_home_xg": "?",
        "bookmaker_implied_away_xg": "?",
        "over_2_5": None,
        "fair_dir": None,
        "shin_dir": None,
        "actual_handicap_signed": None,
        "theoretical_handicap_signed": None,
        "gpt_score": _legacy_model_score(ai_r, "gpt", score),
        "gpt_analysis": _legacy_model_analysis(ai_r, "gpt"),
        "grok_score": _legacy_model_score(ai_r, "grok", score),
        "grok_analysis": _legacy_model_analysis(ai_r, "grok"),
        "gemini_score": score,
        "gemini_analysis": ai_r.get("reason", "")[:3000] or "Gemini终审已给出结构化预测",
        "final_referee_score": score,
        "final_referee_analysis": ai_r.get("reason", "")[:3000],
        "claude_score": "弃用",
        "claude_analysis": "Claude 已淘汰：当前版本由 Gemini 终审裁判。",
        "final_ai_score": score,
        "final_ai_analysis": ai_r.get("reason", "")[:3000],
        "ai_abstained": [],
        "ai_avg_confidence": _f(rec.get("bet_confidence", 0), 0),
        "ai_call_status": dict(AI_CALL_STATUS),
        "ai_result_files": dict(AI_RESULT_FILES),
        "ai_run_metadata": dict(_LAST_AI_RUN_METADATA),
        "model_consensus": None,
        "total_models": 3,
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
        "experience_analysis": {"mode": "removed_no_local_experience_cards"},
        "pro_odds": {},
        "asian_handicap_probs": {},
        "top_scores": [],
        "engine_version": ENGINE_VERSION,
        "engine_architecture": ENGINE_ARCHITECTURE,
    }


def _abstain_prediction(reason: str = "AI全失败，AI-native模式不使用本地足球兜底") -> Dict[str, Any]:
    return {
        "predicted_score": "弃权",
        "predicted_label": "弃权",
        "result": "弃权",
        "display_direction": "弃权",
        "final_direction": "abstain",
        "is_abstain": True,
        "home_win_pct": 0.0,
        "draw_pct": 0.0,
        "away_win_pct": 0.0,
        "confidence": 0,
        "confidence_meaning": "AI-native 模式：AI失败即弃权，不使用本地预测兜底",
        "risk_level": "高",
        "goal_range": (0, 0),
        "recommendation": {"tier": "D", "is_recommended": False, "top4_priority": 999, "bet_confidence": 0, "risk_level": "high", "risk_tags": [reason], "why_recommended": "弃权"},
        "recommendation_tier": "D",
        "recommend_gate_pass": False,
        "recommend_gate_reasons": [reason],
        "top_score_candidates": [],
        "bayesian_evidences": [reason],
        "decision_source": "ai_abstain_no_local_football_fallback",
        "ai_authority_mode": "ai_native_web_no_local_football_judgement",
        "ai_call_status": dict(AI_CALL_STATUS),
        "ai_run_metadata": dict(_LAST_AI_RUN_METADATA),
        "engine_version": ENGINE_VERSION,
        "engine_architecture": ENGINE_ARCHITECTURE,
    }


def _recommend_sort_key(p: Dict[str, Any]) -> Tuple[int, int, int, float]:
    pr = p.get("prediction", {})
    rec = pr.get("recommendation", {}) if isinstance(pr.get("recommendation"), dict) else {}
    tier = _tier_rank(rec.get("tier", "D"))
    is_rec = 1 if bool(rec.get("is_recommended", False)) and not pr.get("is_abstain") else 0
    priority = -_i(rec.get("top4_priority", 999), 999)
    conf = _f(rec.get("bet_confidence", pr.get("confidence", 0)), 0)
    return (is_rec, tier, priority, conf)


def select_top4(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    eligible = []
    for p in preds:
        pr = p.get("prediction", {})
        rec = pr.get("recommendation", {}) if isinstance(pr.get("recommendation"), dict) else {}
        if pr.get("is_abstain"):
            continue
        if bool(rec.get("is_recommended", False)) and _min_tier_ok(rec.get("tier", "D")):
            eligible.append(p)
    eligible = sorted(eligible, key=_recommend_sort_key, reverse=True)
    if len(eligible) >= 4 or not AI_FILL_TOP4_WITH_NON_RECOMMENDABLE:
        return eligible[:4]
    rest = [p for p in preds if p not in eligible]
    rest = sorted(rest, key=_recommend_sort_key, reverse=True)
    return (eligible + rest)[:4]


def extract_num(ms: Any) -> int:
    wm = {"一": 1000, "二": 2000, "三": 3000, "四": 4000, "五": 5000, "六": 6000, "日": 7000, "天": 7000}
    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999


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
# 主入口
# ============================================================

def run_predictions(raw: Dict[str, Any], use_ai: bool = True):
    raw_ms = _extract_match_list(raw)
    ms = [normalize_match(m) for m in raw_ms]

    print("\n" + "=" * 92)
    print(f"  [{ENGINE_VERSION}] AI-NATIVE WEB-AUGMENTED ANCHOR-AUDIT | {len(ms)} 场 | 本地只做协议层 | chunk={AI_CHUNK_SIZE} | mock={AI_MOCK_MODE}")
    print("=" * 92)

    evidence_all = [build_evidence_packet(m, i) for i, m in enumerate(ms, 1)]

    ai_final: Dict[int, Dict[str, Any]] = {}
    if use_ai and evidence_all:
        try:
            ai_final = _run_async(run_ai_native_web(evidence_all))
        except Exception as e:
            logger.error(f"AI-native矩阵执行失败: {e}")
            ai_final = {}
    elif not use_ai:
        print("  [AI-NATIVE] use_ai=False → 全部弃权，不启用本地足球兜底")

    res = []
    for i, m in enumerate(ms, 1):
        ai_r = ai_final.get(i) or _abstain_ai_prediction(i, "missing_final_ai_result")
        pred = adapt_ai_to_frontend(ai_r, m) if not ai_r.get("final_direction") == "abstain" else _abstain_prediction(ai_r.get("reason", "abstain"))
        res.append({**m, "prediction": pred})
        if pred.get("is_abstain"):
            print(f"  [{i}] {m.get('home_team')} vs {m.get('away_team')} => 弃权")
        else:
            rec = pred.get("recommendation", {})
            gate = "PASS" if pred.get("recommend_gate_pass") else "NO-GATE"
            print(
                f"  [{i}] {m.get('home_team')} vs {m.get('away_team')} => "
                f"{pred['result']} ({pred['predicted_score']}) | tier:{rec.get('tier')} | "
                f"AI_CF:{rec.get('bet_confidence')} | gate:{gate} | source:{pred.get('decision_source')}"
            )

    t4 = select_top4(res)
    t4_ids = set(id(x) for x in t4)
    for r in res:
        r["is_top4_candidate"] = id(r) in t4_ids
        r["is_recommended"] = bool(id(r) in t4_ids and r.get("prediction", {}).get("recommend_gate_pass"))
        r["is_strict_recommended"] = r["is_recommended"]

    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    return res, t4


# ============================================================
# vMAX 20.3.0 FULL-SHARP-CLUSTER 增强层
# ============================================================
# 说明：下面是完整内嵌版，不依赖外部 patch 文件。
# 保留 20.2.1-FULL-ANCHOR 的 API 调用链/分批/解析/前端兼容，只增强 Evidence 与 Prompt。

# 保存 20.2.1 基础函数，避免增强模块覆盖原来的比分/JSON/工具行为。
_BASE_BUILD_EVIDENCE_PACKET_V2021 = build_evidence_packet
_BASE_ADAPT_AI_TO_FRONTEND_V2021 = adapt_ai_to_frontend
_BASE_NORMALIZE_AI_PREDICTIONS_V2021 = normalize_ai_predictions
_BASE_SHORT_PREDICTION_FOR_PROMPT_V2021 = _short_prediction_for_prompt
_BASE_F = _f
_BASE_I = _i
_BASE_EXISTS = _exists
_BASE_JSON_COMPACT = _json_compact
_BASE_NORMALIZE_SCORE_TEXT = _normalize_score_text
_BASE_PARSE_SCORE = _parse_score
_BASE_SCORE_DIRECTION = _score_direction
_BASE_SCORE_TOTAL = _score_total
_BASE_SCORE_BTTS = _score_btts
_BASE_SCORE_GOAL_BAND = _score_goal_band

VALID_DIRS = {"home", "draw", "away"}

CRS_FULL_MAP = {
    "1-0": "w10", "2-0": "w20", "2-1": "w21", "3-0": "w30", "3-1": "w31",
    "3-2": "w32", "4-0": "w40", "4-1": "w41", "4-2": "w42", "5-0": "w50",
    "5-1": "w51", "5-2": "w52",
    "0-0": "s00", "1-1": "s11", "2-2": "s22", "3-3": "s33",
    "0-1": "l01", "0-2": "l02", "1-2": "l12", "0-3": "l03", "1-3": "l13",
    "2-3": "l23", "0-4": "l04", "1-4": "l14", "2-4": "l24",
    "0-5": "l05", "1-5": "l15", "2-5": "l25",
}

HFTF_MAP = {
    "ss": "主/主", "sp": "主/平", "sf": "主/负",
    "ps": "平/主", "pp": "平/平", "pf": "平/负",
    "fs": "负/主", "fp": "负/平", "ff": "负/负",
}

_SCORE_RE = re.compile(r"(\d{1,2})\s*[-:：]\s*(\d{1,2})")

# ------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------

def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "" or s.lower() in {"none", "nan", "null", "-", "n/a"}:
            return default
        return float(s.replace("%", "").replace("％", ""))
    except Exception:
        return default


def _i(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _exists(v: Any) -> bool:
    return v not in (None, "", "-", "N/A", "n/a", "None", "none", "null", {}, [])


def _as_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _as_list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []


def _json_compact(obj: Any, max_len: int = 12000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        s = str(obj)
    return s[:max_len] if max_len else s


def _normalize_score_text(s: Any) -> str:
    return str(s).strip().replace(" ", "").replace("：", "-").replace(":", "-").replace("–", "-").replace("—", "-")


def _parse_score(s: Any) -> Tuple[Optional[int], Optional[int]]:
    ss = _normalize_score_text(s)
    if not ss:
        return None, None
    m = _SCORE_RE.search(ss)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _score_direction(score: Any) -> Optional[str]:
    h, a = _parse_score(score)
    if h is None or a is None:
        return None
    if h > a:
        return "home"
    if h < a:
        return "away"
    return "draw"


def _score_total(score: Any) -> Optional[int]:
    h, a = _parse_score(score)
    if h is None or a is None:
        return None
    return h + a


def _score_btts(score: Any) -> str:
    h, a = _parse_score(score)
    if h is None or a is None:
        return "unclear"
    return "yes" if h > 0 and a > 0 else "no"


def _score_goal_band(score: Any) -> str:
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


def _safe_pct_from_odds(odds: float) -> float:
    return 100.0 / odds if odds and odds > 1.0001 else 0.0


def _devig_3way(odds: Dict[str, Any]) -> Dict[str, float]:
    raw = {k: _safe_pct_from_odds(_f(v, 0.0)) for k, v in odds.items()}
    s = sum(raw.values())
    if s <= 0:
        return {k: 0.0 for k in odds}
    return {k: round(v / s * 100.0, 2) for k, v in raw.items()}


def _rank_rows_by_odds(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = sorted([r for r in rows if _f(r.get("odds"), 0) > 1.01], key=lambda r: _f(r.get("odds"), 9999))
    for n, r in enumerate(out, 1):
        r["rank"] = n
    return out


def _movement_label(code: Any) -> str:
    """竞彩抓包常见 change 方向码：-1=赔率下降/压低，0=无变化，1=赔率上升/抬高。"""
    c = _i(code, 99)
    if c == -1:
        return "odds_down_market_support_up"
    if c == 1:
        return "odds_up_market_support_down"
    if c == 0:
        return "unchanged"
    return "unknown"

# ------------------------------------------------------------
# 抓包标准化：只补齐顶层，不做预测
# ------------------------------------------------------------

def normalize_match_v203(raw_m: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(raw_m or {})
    home = m.get("home_team") or m.get("home") or m.get("host") or m.get("team_home") or m.get("homeName") or "Home"
    away = m.get("away_team") or m.get("guest") or m.get("away") or m.get("team_away") or m.get("awayName") or "Away"
    m["home_team"] = home
    m["away_team"] = away
    m["home"] = home
    m["guest"] = away

    # 竞彩 1X2
    m["sp_home"] = m.get("sp_home", m.get("win"))
    m["sp_draw"] = m.get("sp_draw", m.get("same"))
    m["sp_away"] = m.get("sp_away", m.get("lose"))

    # 原代码常把 [] 当 dict 处理，这里统一包一层 safe dict。
    for k in ["change", "hhad_change", "ttg_change", "crs_change", "hafu_change", "vote"]:
        if not isinstance(m.get(k), dict):
            m[k] = {}
    return m

# ------------------------------------------------------------
# 市场事实编译：1X2 / HHAD / TTG / HFTF / CRS
# ------------------------------------------------------------

def compile_1x2_facts(m: Dict[str, Any]) -> Dict[str, Any]:
    odds = {"home": m.get("sp_home", m.get("win")), "draw": m.get("sp_draw", m.get("same")), "away": m.get("sp_away", m.get("lose"))}
    rows = _rank_rows_by_odds([{"direction": k, "odds": _f(v, 0.0)} for k, v in odds.items()])
    fair = _devig_3way(odds)
    return {
        "available": all(_f(v, 0.0) > 1.01 for v in odds.values()),
        "odds": odds,
        "fair_no_margin_pct": fair,
        "lowest_direction": rows[0]["direction"] if rows else "unknown",
        "ranked": rows,
        "market_gap": {
            "home_vs_away_odds_ratio": round(_f(odds.get("away"), 0.0) / max(_f(odds.get("home"), 0.0), 1e-9), 3) if _f(odds.get("home"), 0.0) else None,
            "favorite_fair_pct": max(fair.values()) if fair else 0.0,
        },
    }


def compile_hhad_facts(m: Dict[str, Any]) -> Dict[str, Any]:
    rq = _f(m.get("give_ball", m.get("handicap", m.get("rq", 0))), 0.0)
    odds = {
        "hhad_win": _f(m.get("hhad_win"), 0.0),
        "hhad_same": _f(m.get("hhad_same"), 0.0),
        "hhad_lose": _f(m.get("hhad_lose"), 0.0),
    }
    rows = _rank_rows_by_odds([{"code": k, "odds": v} for k, v in odds.items()])
    fair = _devig_3way(odds)

    # 竞彩让球语义。rq=-1 表示主让1；rq=+1 表示主受让1。
    if rq < 0:
        n = abs(int(rq)) if float(rq).is_integer() else abs(rq)
        semantic = {
            "hhad_win": f"home_cover_win_by_more_than_{n}",
            "hhad_same": f"home_exact_win_by_{n}",
            "hhad_lose": f"home_no_cover_draw_or_away_or_win_by_less_than_{n}",
        }
    elif rq > 0:
        n = int(rq) if float(rq).is_integer() else rq
        semantic = {
            "hhad_win": f"home_plus_{n}_wins_home_win_or_draw_or_lose_by_less_than_{n}",
            "hhad_same": f"home_exact_lose_by_{n}",
            "hhad_lose": f"home_plus_{n}_loses_by_more_than_{n}",
        }
    else:
        semantic = {"hhad_win": "home_win", "hhad_same": "draw", "hhad_lose": "away_win"}

    return {
        "available": all(v > 1.01 for v in odds.values()),
        "rq": rq,
        "odds": odds,
        "fair_no_margin_pct": fair,
        "ranked": rows,
        "lowest_hhad_code": rows[0]["code"] if rows else "unknown",
        "semantic_map": semantic,
        "interpretation_rule": "事实映射，不代表本地推荐。Gemini 必须结合 1X2/CRS/TTG 审计让球是否支持穿盘、赢一球或不胜。",
    }


def compile_ttg_facts(m: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for i in range(8):
        odd = _f(m.get(f"a{i}"), 0.0)
        if odd > 1.01:
            rows.append({"goals": i, "odds": odd, "change_code": _as_dict(m.get("ttg_change")).get(f"a{i}"), "change_label": _movement_label(_as_dict(m.get("ttg_change")).get(f"a{i}"))})
    rows = _rank_rows_by_odds(rows)
    # goal-band pressure by implied odds, not a prediction.
    band_map = {"0-1": [0, 1], "2": [2], "3": [3], "4+": [4, 5, 6, 7]}
    band_pressure = []
    for band, goals in band_map.items():
        vals = [r for r in rows if r["goals"] in goals]
        implied_sum = sum(_safe_pct_from_odds(_f(r.get("odds"), 0.0)) for r in vals)
        min_odd = min([_f(r.get("odds"), 9999) for r in vals], default=0.0)
        band_pressure.append({"goal_band": band, "goals": goals, "min_odds": round(min_odd, 3), "raw_implied_sum": round(implied_sum, 3)})
    band_pressure.sort(key=lambda r: (-r["raw_implied_sum"], r["min_odds"] if r["min_odds"] else 9999))
    return {
        "available": bool(rows),
        "lowest_total_goals": rows[:8],
        "mode_goals": rows[0]["goals"] if rows else None,
        "goal_band_pressure": band_pressure,
        "movement_summary": [r for r in rows if r.get("change_label") not in ("unknown", "unchanged")],
    }


def compile_hftf_facts(m: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for code, label in HFTF_MAP.items():
        odd = _f(m.get(code), 0.0)
        if odd > 1.01:
            rows.append({"code": code, "label": label, "odds": odd, "change_code": _as_dict(m.get("hafu_change")).get(code), "change_label": _movement_label(_as_dict(m.get("hafu_change")).get(code))})
    rows = _rank_rows_by_odds(rows)
    return {"available": bool(rows), "lowest": rows[:9], "movement_summary": [r for r in rows if r.get("change_label") not in ("unknown", "unchanged")]} 


def compile_crs_rows(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    chg = _as_dict(m.get("crs_change"))
    rows = []
    for score, key in CRS_FULL_MAP.items():
        odd = _f(m.get(key), 0.0)
        if odd > 1.01:
            rows.append({
                "score": score,
                "key": key,
                "odds": odd,
                "direction": _score_direction(score),
                "total_goals": _score_total(score),
                "goal_band": _score_goal_band(score),
                "btts": _score_btts(score),
                "change_code": chg.get(key),
                "change_label": _movement_label(chg.get(key)),
            })
    rows = _rank_rows_by_odds(rows)
    # direction ranks
    by_dir = {"home": [], "draw": [], "away": []}
    for r in rows:
        by_dir.setdefault(r.get("direction"), []).append(r)
    for arr in by_dir.values():
        for n, r in enumerate(arr, 1):
            r["direction_rank"] = n
    return rows

# ------------------------------------------------------------
# 比分簇诊断：核心升级
# ------------------------------------------------------------

SCORE_CLUSTERS = {
    "low_score_draw_cluster": ["0-0", "1-1", "1-0", "0-1"],
    "home_narrow_win_cluster": ["1-0", "2-0", "2-1"],
    "away_narrow_win_cluster": ["0-1", "0-2", "1-2"],
    "high_btts_tail_cluster": ["2-2", "3-1", "1-3", "3-2", "2-3"],
    "home_cover_cluster": ["2-0", "3-0", "3-1", "4-0", "4-1"],
    "away_cover_cluster": ["0-2", "0-3", "1-3", "0-4", "1-4"],
}

ADJACENT_AUDIT_MAP = {
    "2-1": ["1-1", "1-0", "2-0", "1-2", "2-2", "3-1"],
    "1-2": ["1-1", "0-1", "0-2", "2-1", "2-2", "1-3"],
    "2-0": ["1-0", "3-0", "2-1", "0-0", "1-1"],
    "0-2": ["0-1", "0-3", "1-2", "0-0", "1-1"],
    "1-0": ["0-0", "1-1", "2-0", "2-1", "0-1"],
    "0-1": ["0-0", "1-1", "0-2", "1-2", "1-0"],
    "1-1": ["0-0", "1-0", "0-1", "2-1", "1-2", "2-2"],
    "3-1": ["2-1", "2-0", "3-0", "2-2", "4-1"],
    "1-3": ["1-2", "0-2", "0-3", "2-2", "1-4"],
    "3-0": ["2-0", "3-1", "4-0", "2-1"],
    "0-3": ["0-2", "1-3", "0-4", "1-2"],
}


def compile_score_cluster_diagnostics(m: Dict[str, Any]) -> Dict[str, Any]:
    rows = compile_crs_rows(m)
    odds_by_score = {r["score"]: _f(r["odds"], 0.0) for r in rows}

    clusters = []
    for name, scores in SCORE_CLUSTERS.items():
        present = [s for s in scores if s in odds_by_score]
        implied_sum = sum(_safe_pct_from_odds(odds_by_score[s]) for s in present)
        min_odd = min([odds_by_score[s] for s in present], default=0.0)
        mean_odd = sum([odds_by_score[s] for s in present]) / max(len(present), 1) if present else 0.0
        lowest_scores = sorted([{"score": s, "odds": odds_by_score[s]} for s in present], key=lambda x: x["odds"])[:4]
        clusters.append({
            "cluster": name,
            "scores": present,
            "min_odds": round(min_odd, 3),
            "mean_odds": round(mean_odd, 3),
            "raw_implied_sum": round(implied_sum, 3),
            "lowest_scores": lowest_scores,
        })
    clusters.sort(key=lambda r: (-r["raw_implied_sum"], r["min_odds"] if r["min_odds"] else 9999))

    adjacent = {}
    for base, rivals in ADJACENT_AUDIT_MAP.items():
        if base not in odds_by_score:
            continue
        base_odd = odds_by_score[base]
        rival_rows = []
        for rv in rivals:
            if rv not in odds_by_score:
                continue
            rv_odd = odds_by_score[rv]
            rival_rows.append({
                "rival": rv,
                "rival_odds": rv_odd,
                "lower_than_base": rv_odd < base_odd,
                "near_base": abs(rv_odd - base_odd) / max(base_odd, 1e-9) <= 0.18,
            })
        adjacent[base] = {"base_odds": base_odd, "rivals": sorted(rival_rows, key=lambda x: x["rival_odds"])}

    by_dir = {"home": [], "draw": [], "away": []}
    for r in rows:
        by_dir[r["direction"]].append(r)

    return {
        "available": bool(rows),
        "lowest_scores_overall": rows[:12],
        "lowest_scores_by_direction": {d: arr[:6] for d, arr in by_dir.items()},
        "cluster_ranking": clusters,
        "adjacent_score_audit_table": adjacent,
        "movement_summary": [r for r in rows if r.get("change_label") not in ("unknown", "unchanged")],
        "required_gemini_rule": "Gemini 输出 predicted_score 前必须先读取 adjacent_score_audit_table；若 rival_odds 更低或接近但不选，必须解释。解释不足时 recommendation.tier 最高 B。",
    }

# ------------------------------------------------------------
# Public / Sharp facts：只产事实，不直接裁决
# ------------------------------------------------------------

def compile_public_vote_facts(m: Dict[str, Any]) -> Dict[str, Any]:
    vote = _as_dict(m.get("vote"))
    three = {"home": _f(vote.get("win"), 0.0), "draw": _f(vote.get("same"), 0.0), "away": _f(vote.get("lose"), 0.0)}
    hhad = {"hhad_win": _f(vote.get("hhad_win"), 0.0), "hhad_same": _f(vote.get("hhad_same"), 0.0), "hhad_lose": _f(vote.get("hhad_lose"), 0.0)}
    dom = max(three.items(), key=lambda kv: kv[1])[0] if any(three.values()) else "unknown"
    return {
        "available": bool(vote),
        "vote_pct_1x2": three,
        "dominant_public_1x2": dom,
        "public_skew_strength": max(three.values()) if any(three.values()) else 0.0,
        "vote_pct_hhad": hhad,
    }


def compile_sharp_money_facts(m: Dict[str, Any]) -> Dict[str, Any]:
    one = compile_1x2_facts(m)
    hhad = compile_hhad_facts(m)
    vote = compile_public_vote_facts(m)
    ttg = compile_ttg_facts(m)
    crs_diag = compile_score_cluster_diagnostics(m)

    warnings: List[str] = []
    candidates: List[Dict[str, Any]] = []

    public_side = vote.get("dominant_public_1x2", "unknown")
    public_strength = _f(vote.get("public_skew_strength"), 0.0)
    market_side = one.get("lowest_direction", "unknown")

    # 只有 1X2 change 缺失时，不准声称真实 RLM。只能标记 public-vs-market contradiction。
    had_change = _as_dict(m.get("change"))
    if not had_change:
        warnings.append("had_change_missing_cannot_confirm_true_rlm")

    if public_side in VALID_DIRS and market_side in VALID_DIRS:
        if public_strength >= 58 and public_side != market_side:
            candidates.append({
                "side": market_side,
                "reason": "public_vote_dominant_but_1x2_lowest_market_on_other_side",
                "strength": "medium" if public_strength >= 65 else "weak",
                "public_side": public_side,
                "market_side": market_side,
            })

    # CRS 降赔项：-1 是压低，说明该比分市场支持上升。
    crs_down = [r for r in crs_diag.get("movement_summary", []) if r.get("change_label") == "odds_down_market_support_up"]
    for r in crs_down[:8]:
        candidates.append({
            "side": r.get("direction"),
            "reason": f"crs_score_odds_down:{r.get('score')}",
            "strength": "medium" if r.get("rank", 99) <= 8 else "weak",
            "score": r.get("score"),
            "odds": r.get("odds"),
        })

    # TTG 降赔：只说明总进球形态支持增强。
    ttg_down = [r for r in ttg.get("movement_summary", []) if r.get("change_label") == "odds_down_market_support_up"]
    goal_shape_candidates = [{"goals": r.get("goals"), "odds": r.get("odds"), "reason": "ttg_odds_down"} for r in ttg_down[:5]]

    # HHAD 最低项可作为穿盘/不穿盘事实，但不是方向裁决。
    hhad_low = hhad.get("lowest_hhad_code", "unknown")
    if hhad_low != "unknown":
        candidates.append({
            "side": "handicap_structure",
            "reason": f"lowest_hhad={hhad_low}:{hhad.get('semantic_map', {}).get(hhad_low, '')}",
            "strength": "medium",
            "hhad_lowest_code": hhad_low,
        })

    strong_count = sum(1 for c in candidates if c.get("strength") == "medium")
    confidence = "strong" if strong_count >= 3 else "medium" if strong_count >= 1 else "weak"

    return {
        "public_side": public_side,
        "public_strength": public_strength,
        "market_lowest_side": market_side,
        "reverse_line_movement": {
            "detected": False if not had_change else None,
            "status": "unavailable_without_had_change" if not had_change else "requires_open_current_odds",
            "note": "当前抓包大多只有 change 方向码或空数组；没有开盘/即时赔率序列时不能硬判真实 RLM。",
        },
        "sharp_side_candidates": candidates[:12],
        "goal_shape_candidates": goal_shape_candidates,
        "confidence": confidence,
        "warnings": warnings,
        "hard_rule_for_ai": "AI 可以使用 sharp_side_candidates 作为资金/市场异动事实，但不得把 warnings 缺失的数据当成已确认聪明钱。",
    }

# ------------------------------------------------------------
# 情报源审计：抓包文字不是实时 web source
# ------------------------------------------------------------

def compile_packet_context_facts(m: Dict[str, Any]) -> Dict[str, Any]:
    analyse = _as_dict(m.get("analyse"))
    information = _as_dict(m.get("information"))
    points = _as_dict(m.get("points"))

    injury = {
        "home_injury": information.get("home_injury", ""),
        "guest_injury": information.get("guest_injury", ""),
    }
    news_blocks = {
        "intro": m.get("intro", ""),
        "analyse_intro": analyse.get("intro", ""),
        "analyse_baseface": analyse.get("baseface", ""),
        "information_home_good": information.get("home_good_news", ""),
        "information_guest_good": information.get("guest_good_news", ""),
        "information_home_bad": information.get("home_bad_news", ""),
        "information_guest_bad": information.get("guest_bad_news", ""),
        "points_deep": points.get("deep", ""),
        "points_match_points": points.get("match_points", ""),
    }
    nonempty_news = {k: v for k, v in news_blocks.items() if _exists(v)}

    third_party_picks = {
        "predict_array": _as_list(m.get("predict")),
        "analyse_had": analyse.get("had_analyse", []),
        "analyse_ttg": analyse.get("ttg_analyse", []),
        "analyse_hafu": analyse.get("hafu_analyse", []),
        "analyse_crs": analyse.get("crs_analyse", []),
        "analyse_odds_refs": {
            "had_odds": analyse.get("had_odds", ""),
            "ttg_odds": analyse.get("ttg_odds", ""),
            "hafu_odds": analyse.get("hafu_odds", ""),
            "crs_odds": analyse.get("crs_odds", ""),
        }
    }

    return {
        "available": bool(nonempty_news or injury or third_party_picks.get("predict_array")),
        "packet_news_blocks": nonempty_news,
        "injury_blocks": injury,
        "team_lists": {
            "home_first_team": information.get("home_first_team", ""),
            "guest_first_team": information.get("guest_first_team", ""),
        },
        "third_party_packet_picks": third_party_picks,
        "source_reliability_note": "这些是抓包内置情报/推荐，不等于实时外部 Web sources。若 AI_NATIVE_WEB=false 或 no_web_tool_available，不能把它们包装成联网来源。",
    }

# ------------------------------------------------------------
# 联赛波动 profile：风险先验，不做方向裁决
# ------------------------------------------------------------

def compile_league_volatility_profile(m: Dict[str, Any]) -> Dict[str, Any]:
    league = str(m.get("cup", m.get("league", "")))
    high_goal = any(x in league for x in ["德甲", "德乙", "荷甲", "挪超", "瑞典", "美职", "澳超"])
    low_goal = any(x in league for x in ["意甲", "西甲", "法甲", "芬超", "日职", "韩职"])
    drawish = any(x in league for x in ["法甲", "西甲", "意甲", "芬超", "韩职", "日职"])
    return {
        "league": league,
        "goal_volatility": "high" if high_goal else "low" if low_goal else "medium",
        "draw_baseline": "high" if drawish else "medium",
        "tail_risk": "high" if high_goal else "medium",
        "note": "联赛 profile 只作为 Gemini 风险审计提醒；不能直接覆盖 AI 比分。",
    }

# ------------------------------------------------------------
# Enhanced Evidence 汇总
# ------------------------------------------------------------

def build_enhanced_market_modules(match_obj: Dict[str, Any], index: int) -> Dict[str, Any]:
    m = normalize_match_v203(match_obj)
    return {
        "market_microstructure_v203": {
            "one_x_two": compile_1x2_facts(m),
            "handicap_hhad": compile_hhad_facts(m),
            "total_goals_ttg": compile_ttg_facts(m),
            "half_full_time": compile_hftf_facts(m),
        },
        "score_cluster_diagnostics_v203": compile_score_cluster_diagnostics(m),
        "sharp_money_facts_v203": compile_sharp_money_facts(m),
        "packet_context_facts_v203": compile_packet_context_facts(m),
        "league_volatility_profile_v203": compile_league_volatility_profile(m),
    }


def build_evidence_packet_v203(match_obj: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    可直接替换 v20.2.1 的 build_evidence_packet。
    注意：此函数只编译事实，不输出 predicted_score，不做本地预测判断。
    """
    m = normalize_match_v203(match_obj)

    correct_score_odds = {sc: m.get(key) for sc, key in CRS_FULL_MAP.items() if m.get(key) not in (None, "", 0, "0")}
    total_goals_odds = {str(i): m.get(f"a{i}") for i in range(8) if m.get(f"a{i}") not in (None, "", 0, "0")}
    hftf_odds = {HFTF_MAP[k]: m.get(k) for k in HFTF_MAP if m.get(k) not in (None, "", 0, "0")}

    evidence = {
        "match": index,
        "identity": {
            "home_team": m.get("home_team", "Home"),
            "away_team": m.get("away_team", "Away"),
            "league": m.get("cup", m.get("league", "")),
            "match_num": m.get("week_no", m.get("match_num", "")),
            "week": m.get("week", ""),
            "match_time_ts": m.get("stime", ""),
            "wtime_ts": m.get("wtime", ""),
        },
        "lottery_market_1x2": {
            "home": m.get("sp_home"),
            "draw": m.get("sp_draw"),
            "away": m.get("sp_away"),
            "note": "中国体彩竞彩 HAD 赔率，不是欧洲均赔。",
        },
        "handicap": {
            "raw": m.get("give_ball", m.get("handicap", m.get("rq", ""))),
            "hhad_win": m.get("hhad_win"),
            "hhad_same": m.get("hhad_same"),
            "hhad_lose": m.get("hhad_lose"),
            "note": "中国竞彩让球胜平负 HHAD；give_ball=-1 表示主让1，hhad_same 通常对应主队刚好赢1球。",
        },
        "movement": {
            "had_change": _as_dict(m.get("change")),
            "hhad_change": _as_dict(m.get("hhad_change")),
            "ttg_change": _as_dict(m.get("ttg_change")),
            "crs_change": _as_dict(m.get("crs_change")),
            "hafu_change": _as_dict(m.get("hafu_change")),
            "coding_note": "-1=赔率下降/压低，0=不变，1=赔率上升/抬高。空数组/空对象表示无可用变化数据，不能硬判 RLM。",
        },
        "public_vote": _as_dict(m.get("vote")),
        "total_goals_odds": total_goals_odds,
        "correct_score_odds": correct_score_odds,
        "half_full_time_odds": hftf_odds,
        "data_quality": {
            "has_1x2": all(_f(m.get(k), 0.0) > 1.01 for k in ["sp_home", "sp_draw", "sp_away"]),
            "has_hhad": all(_f(m.get(k), 0.0) > 1.01 for k in ["hhad_win", "hhad_same", "hhad_lose"]),
            "has_total_goals": bool(total_goals_odds),
            "has_correct_score": bool(correct_score_odds),
            "has_hftf": bool(hftf_odds),
            "has_vote": bool(_as_dict(m.get("vote"))),
            "has_market_change": any(bool(_as_dict(m.get(k))) for k in ["change", "hhad_change", "ttg_change", "crs_change", "hafu_change"]),
            "has_packet_context": any(_exists(m.get(k)) for k in ["intro", "analyse", "information", "points"]),
        },
    }
    evidence.update(build_enhanced_market_modules(m, index))
    return evidence

# ------------------------------------------------------------
# Prompt addendum：把 20.2.1 的自由发挥改成审计表
# ------------------------------------------------------------

PHASE1_ROLE_SPLIT_ADDENDUM = """
【v20.6 AI 主导分工】
你不是比分生成器，而是风险控制型赛前分析师；目标不是每场硬推，而是避免把低质量比赛包装成高信心推荐。
GPT：正方市场结构师，重点输出 market_audit / score_cluster_audit / goal_market_audit / candidate_scores / 可买条件；若证据不足必须说 observe/no_bet。
Grok：反方审判员，重点寻找 favorite_trap / draw_trap / away_win_overreach / rotation_or_motivation_risk / source_hallucination；不要顺着热门结论。
Gemini：终审裁判，必须综合正方、反方、raw evidence 和来源质量，允许 final_direction=abstain/no_bet；若证据不足，recommendation.is_recommended=false。
三方都必须列出 why_this_can_fail；不能把“名气强、低赔、常见比分模板”当成独立充分证据。
若 GPT/Grok 仍输出最终比分，Gemini 可以参考但必须重新审计，不得机械照抄。
""".strip()

GEMINI_FINAL_AUDIT_ADDENDUM = """
【v20.6 Gemini 终审硬约束：AI 自主、反幻觉、可放弃】
1. 必须读取 score_cluster_diagnostics_v203.cluster_ranking 与 adjacent_score_audit_table。
2. 输出 predicted_score 前必须完成相邻比分审计：
   - 2-1 必须比较 1-1/1-0/2-0/1-2/2-2/3-1。
   - 1-2 必须比较 1-1/0-1/0-2/2-1/2-2/1-3。
   - 2-0 必须比较 1-0/3-0/2-1/0-0/1-1。
   - 0-2 必须比较 0-1/0-3/1-2/0-0/1-1。
   - 1-0 必须比较 0-0/1-1/2-0/2-1/0-1。
   - 0-1 必须比较 0-0/1-1/0-2/1-2/1-0。
   - 1-1 必须比较 0-0/1-0/0-1/2-1/1-2/2-2。
3. 若相邻比分赔率更低或接近但最终不选，必须在 rejected_cases 或 reason 中解释；解释不足时 recommendation.tier 最高 B。
4. 必须读取 sharp_money_facts_v203。若 had_change 缺失，只能说 RLM 不可确认，不能声称“聪明钱已确认”。
5. 若 no_web_tool_available，不能把抓包里的 injury/news/points 包装成实时联网来源；这些只能算 packet_context，最高 reliability=medium。
6. 推荐等级必须拆分：direction_edge、score_cluster_strength、goal_band_strength、btts_alignment、sharp_alignment、web_source_quality、market_conflict_penalty。
7. 你必须输出 bet_action：main/small/hedge/observe/no_bet；本地不会替你改足球方向、比分或原始 confidence，只会展示风险提示。
8. 你必须输出 why_this_can_fail 与 minimum_evidence_needed；若关键证据缺失且无法覆盖失败路径，应主动 no_bet/observe，而不是硬给 A/B 推荐。
""".strip()

RECOMMENDATION_COMPONENT_SCHEMA = {
    "recommendation_components": {
        "direction_edge": "0-100，方向市场优势，不等于比分命中率",
        "score_cluster_strength": "0-100，正确比分簇支持强度",
        "goal_band_strength": "0-100，总进球带支持强度",
        "btts_alignment": "0-100，BTTS 与比分/总球/联赛形态的一致度",
        "sharp_alignment": "0-100，sharp_money_facts_v203 与最终方向的一致度；缺数据不能高分",
        "web_source_quality": "0-100，无真实联网来源时为0或很低",
        "market_conflict_penalty": "0到-40，冲突越大越负",
        "final_grade_reason": "中文说明为什么给 S/A/B/C/D",
    }
}

# ------------------------------------------------------------
# 可选：本地只做推荐降级，不改比分/方向
# ------------------------------------------------------------

def protocol_downgrade_recommendation_only(pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    v20.6: no longer locally caps AI recommendation tiers from component heuristics.
    This layer only records protocol risk hints and lets the AI final referee own
    tier / is_recommended / bet_confidence. Hard local football decisions were
    moved out of the main path to avoid becoming a fourth pseudo-referee.
    """
    if not isinstance(pred, dict):
        return pred
    rec = pred.setdefault("recommendation", {}) if isinstance(pred.get("recommendation"), dict) else {}
    comps = pred.get("recommendation_components", {}) if isinstance(pred.get("recommendation_components"), dict) else {}
    reasons: List[str] = []

    if _f(comps.get("score_cluster_strength"), 0) < 55:
        reasons.append("score_cluster_strength_below_55")
    if _f(comps.get("sharp_alignment"), 0) < 40 and _f(comps.get("direction_edge"), 0) < 70:
        reasons.append("sharp_alignment_low_and_direction_edge_not_strong")
    if _f(comps.get("web_source_quality"), 0) <= 0 and any(k in str(pred.get("reason", "")) for k in ["伤停", "首发", "战意", "体能"]):
        reasons.append("reason_uses_context_without_verified_web_sources")
    if _f(comps.get("direction_edge"), 0) < 55:
        reasons.append("direction_edge_below_55")

    if reasons:
        pred.setdefault("protocol_risk_hints", []).extend(reasons)
        pred["protocol_risk_hints"] = list(dict.fromkeys(str(x) for x in pred.get("protocol_risk_hints", []) if str(x).strip()))[:24]
        pred.setdefault("validation_warnings", []).extend([f"protocol_risk_hint:{r}" for r in reasons])
        pred["validation_warnings"] = list(dict.fromkeys(str(x) for x in pred.get("validation_warnings", []) if str(x).strip()))[:80]

    tier = str(rec.get("tier", pred.get("recommendation_tier", "D"))).upper()
    rec["tier"] = tier if tier in {"S", "A", "B", "C", "D"} else "D"
    pred["recommendation_tier"] = rec["tier"]
    pred["recommend_gate_pass"] = bool(rec.get("is_recommended", False)) and _min_tier_ok(rec["tier"])
    return pred


# 恢复基础工具函数，增强模块内部会使用这些已恢复的全局函数。
_f = _BASE_F
_i = _BASE_I
_exists = _BASE_EXISTS
_json_compact = _BASE_JSON_COMPACT
_normalize_score_text = _BASE_NORMALIZE_SCORE_TEXT
_parse_score = _BASE_PARSE_SCORE
_score_direction = _BASE_SCORE_DIRECTION
_score_total = _BASE_SCORE_TOTAL
_score_btts = _BASE_SCORE_BTTS
_score_goal_band = _BASE_SCORE_GOAL_BAND

ENGINE_VERSION = "vMAX 20.5.0-PREMATCH-STRUCTURED-COMPLETE"
ENGINE_ARCHITECTURE = (
    "AI-NATIVE WEB-AUGMENTED 3AI FULL-SHARP-CLUSTER: 保留20.2.1完整AI调用链；"
    "新增Sharp/聪明钱事实编译、HHAD让球语义、CRS比分簇、TTG/CRS change消费、相邻比分审计；"
    "新增赛前综合因子V2风控闸门：联赛DNA/战意轮换/杯赛跨洲/弱主胜防平/客胜复核/数据质量/资金冲突；新增结构化外部因子与临场确认升级；新增推荐分层：主推/小注/防平/观察/放弃；"
    "本地不改足球方向/比分，只做Evidence编译、协议校验、推荐风险展示。"
)


def build_evidence_packet(match_obj: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    v20.3 完整版 Evidence Compiler。
    基于 20.2.1-FULL-ANCHOR 原始证据 + v20.3 Sharp/Score-Cluster 模块合并。
    本函数只编译事实，不输出 predicted_score，不做本地足球裁决。
    """
    evidence = _BASE_BUILD_EVIDENCE_PACKET_V2021(match_obj, index)
    try:
        evidence.update(build_enhanced_market_modules(match_obj, index))
        evidence["evidence_compiler_version"] = "v20.3.0_sharp_cluster_full"
        evidence.setdefault("protocol_notes", []).extend([
            "v20.3: sharp_money_facts_v203 是事实编译，不是本地判断Sharp真伪。",
            "v20.3: score_cluster_diagnostics_v203 只描述赔率簇和相邻比分关系，不替AI选比分。",
            "v20.3: Gemini 必须基于相邻比分审计给出最终 predicted_score。",
        ])
    except Exception as e:
        evidence.setdefault("data_quality", {})["v203_enhancement_error"] = str(e)[:300]
    return evidence


def _canonical_output_schema_text() -> str:
    return r"""
必须输出严格 JSON object，顶层格式：{"predictions":[...]}。不要输出 markdown，不要输出 JSON 外文本。
每个 prediction 必须包含：
{
  "match": 1,
  "final_direction": "home/draw/away",
  "predicted_score": "2-1",
  "direction_probs": {"home": 45, "draw": 28, "away": 27},
  "goal_band": "0-1/2/3/4+",
  "btts": "yes/no/unclear",
  "risk_score_candidates": [
    {"score":"1-2", "risk_type":"away_fightback/high_btts_tail", "reason":"中文说明"}
  ],
  "tail_risk_flags": ["weak_home_favorite_btts_tail"],
  "confidence_downgrade_reason": "",
  "top3": [
    {"score":"2-1", "prob":16, "logic":"中文专业说明"}
  ],
  "market_interpretation": {
    "one_x_two":"中文说明", "handicap":"中文说明", "correct_score":"中文说明",
    "total_goals":"中文说明", "half_full_time":"中文说明", "external_market":"中文说明"
  },
  "money_flow": {
    "public_money_direction":"home/draw/away/unclear",
    "sharp_money_direction":"home/draw/away/unclear",
    "sharp_confidence":0,
    "reverse_line_movement":false,
    "steam_move":"home/draw/away/none/unclear",
    "evidence":"中文说明"
  },
  "score_cluster_audit": {
    "selected_cluster":"home_narrow/draw_low/away_narrow/home_cover/away_cover/high_btts/other",
    "why_selected_score":"中文说明",
    "adjacent_scores_checked":[{"score":"1-1","odds":0,"verdict":"reject/keep/near_miss","reason":"中文"}],
    "lowest_score_not_selected_reason":"中文说明"
  },
  "sharp_money_audit": {
    "available":true,
    "confirmed_sharp_direction":"home/draw/away/unclear",
    "rlm_confirmed":false,
    "vote_vs_odds_conflict":"中文说明",
    "crs_change_impact":"中文说明",
    "ttg_change_impact":"中文说明"
  },
  "anchor_audit": {
    "zero_zero":"中文说明", "one_one":"中文说明", "one_zero":"中文说明", "zero_one":"中文说明",
    "two_one":"中文说明", "one_two":"中文说明", "high_score_tail":"中文说明", "handicap_cover":"中文说明"
  },
  "contextual_logic": {
    "league_style":"中文说明", "team_style":"中文说明", "tempo":"low/medium/high/unclear",
    "score_shape":"中文说明", "btts_likelihood":"yes/no/unclear", "rotation_risk":"low/medium/high/unclear"
  },
  "rejected_cases": {"home":"中文说明", "draw":"中文说明", "away":"中文说明"},
  "web_research": {
    "used": true,
    "failure_reason": "",
    "search_queries": [],
    "sources": [
      {"title":"", "url":"", "publisher":"", "published_at":"", "accessed_at":"", "source_type":"injury/lineup/odds/news/stats/tactical", "reliability":"high/medium/low", "claim":"", "impact":"direction/score/risk/no_impact"}
    ],
    "freshness_grade":"live/recent/stale/missing",
    "key_findings": [],
    "source_conflicts": []
  },
  "recommendation_components": {
    "direction_edge":0,
    "score_cluster_strength":0,
    "goal_band_strength":0,
    "btts_alignment":0,
    "sharp_alignment":0,
    "web_source_quality":0,
    "market_conflict_penalty":0,
    "final_grade_reason":"中文说明"
  },
  "recommendation": {
    "tier":"S/A/B/C/D",
    "is_recommended":true,
    "top4_priority":1,
    "bet_confidence":0,
    "direction_stability":"strong/medium/weak",
    "score_stability":"strong/medium/weak",
    "risk_level":"low/medium/high",
    "risk_tags":[],
    "bet_action":"main/small/hedge/observe/no_bet",
    "why_this_can_fail":[],
    "minimum_evidence_needed":[],
    "why_recommended":"中文说明"
  },
  "data_quality": {"missing":[], "raw_packet_quality":"high/medium/low"},
  "reason":"中文综合理由"
}
硬约束：predicted_score 暗示的方向必须等于 final_direction；goal_band 与 predicted_score 总进球一致；btts 与 predicted_score 一致；top3[0].score 必须等于 predicted_score；必须完成 score_cluster_audit / sharp_money_audit / anchor_audit / recommendation_components。若主胜概率低于或等于52%、客胜概率不低于23%、且 BTTS=yes，不得把 1-2、2-2、2-3 视为无关尾部；如果最终仍选主胜 2-1，必须说明为什么排除客队反打与 4+尾部，并填充 risk_score_candidates / tail_risk_flags。若证据不足，请 recommendation.is_recommended=false、bet_action=observe/no_bet，不要硬推。
""".strip()


def _phase1_system(ai_name: str) -> str:
    role_intro = {
        "gpt": "你是 Probabilistic Market Structure Analyst，专攻 1X2、HHAD让球、正确比分赔率簇、总进球模态、外部赔率对照。你不是最终裁判。",
        "grok": "你是 Money Flow / Sharp Movement Analyst，专攻 vote、change、CRS/TTG压赔抬赔、热度、Sharp/Steam/RLM、临场新闻。你不是最终裁判。",
        "gemini": "你是 Gemini Final Web-aware Referee，负责战术/来源质量审计、相邻比分审计、交叉证据仲裁和最终推荐。",
    }.get(ai_name, "你是足球量化 RAW-AI 分析师。")
    return (
        role_intro
        + "只能输出严格 JSON object，顶层必须是 predictions。所有解释字段必须中文。"
        + "禁止引用本地模型结论；允许使用 raw evidence、ai_anchor_facts_no_judgement、market_microstructure_v203、score_cluster_diagnostics_v203、sharp_money_facts_v203。"
        + "不要机械投票；不要默认1-1/2-1/0-2；必须完成 score_cluster_audit、sharp_money_audit、anchor_audit。"
    )


def build_phase1_prompt(evidence_batch: List[Dict[str, Any]], ai_name: str) -> str:
    n = len(evidence_batch)
    p = []
    p.append("<task>")
    p.append(f"本批次共有 {n} 场，match 编号必须完整覆盖：" + ",".join(str(e["match"]) for e in evidence_batch))
    p.append("你看到的是中国体彩竞彩抓包 Evidence，不是本地预测结论。本地没有 Sharp 真伪裁决、没有比分裁决、没有推荐裁决。")
    p.append(_web_research_instruction(ai_name))
    p.append(PHASE1_ROLE_SPLIT_ADDENDUM)
    if ai_name == "gpt":
        p.append("GPT重点：读取 market_microstructure_v203 与 score_cluster_diagnostics_v203，输出市场结构审计、比分簇审计、相邻比分表。不要把资金流当主任务。")
    elif ai_name == "grok":
        p.append("Grok重点：读取 sharp_money_facts_v203、movement、vote、packet_context_facts_v203，判断热度/压赔/反向移动是否可确认；缺少change时必须写不可确认。")
    else:
        p.append("Gemini若参与初审，也必须按最终裁判标准完成相邻比分和来源审计。")
    p.append("强制：每场必须显式读取 score_cluster_diagnostics_v203.adjacent_score_audit_table；不能只看最低赔率。")
    p.append("弱主胜尾部约束：如果主胜概率低于或等于52%、客胜概率不低于23%、且 BTTS=yes，不得把 1-2、2-2、2-3 视为无关尾部。如果最终仍选主胜 2-1，必须说明为什么排除客队反打与 4+ 尾部，并在 risk_score_candidates 或 tail_risk_flags 中体现这些风险。")
    p.append("强制：若 web_research.used=false，不得把抓包 information/points 伪装成联网来源；只能作为 packet_context。")
    p.append("</task>\n")
    p.append("<output_schema>")
    p.append(_canonical_output_schema_text())
    p.append("</output_schema>\n")
    p.append("<evidence_batch>")
    for e in evidence_batch:
        p.append(_safe_json_line(e))
    p.append("</evidence_batch>")
    return "\n".join(p)


def _short_prediction_for_prompt(r: Dict[str, Any]) -> Dict[str, Any]:
    keep = {}
    for k in [
        "match", "final_direction", "predicted_score", "direction_probs", "goal_band", "btts", "top3",
        "score_cluster_audit", "sharp_money_audit", "anchor_audit", "market_interpretation", "money_flow",
        "contextual_logic", "rejected_cases", "recommendation_components", "recommendation",
        "data_quality", "reason", "web_research", "final_web_audit", "risk_score_candidates",
        "tail_risk_flags", "confidence_downgrade_reason", "validation_warnings",
    ]:
        if k in r:
            keep[k] = r[k]
    return keep


def build_critic_prompt(evidence_batch: List[Dict[str, Any]], reviewer_name: str, phase1_results: Dict[str, Dict[int, Dict[str, Any]]]) -> str:
    p = []
    p.append("<task>")
    p.append(f"你是 {reviewer_name.upper()} Critic。你不是重新预测主裁，只负责审查其他 AI 的漏洞。")
    p.append("必须指出：赔率误读、HHAD语义误读、联网来源不可靠、Sharp证据不足、CRS相邻比分未审、TTG/BTTS冲突、推荐等级过高。")
    p.append("尤其审查：是否把 no_web_tool_available 当成真实联网；是否把抓包文字当实时新闻；是否机械选2-1/1-2/0-2。")
    p.append("输出严格 JSON object，顶层格式 {\"critic_reports\":[...]}，不要 markdown。")
    p.append("每个 critic_report 格式：{\"match\":1,\"critic_model\":\"gpt\",\"target_findings\":[{\"target_model\":\"grok\",\"issue_type\":\"market/web/sharp/score/schema/risk/cluster/hhad\",\"severity\":\"low/medium/high\",\"comment\":\"中文\"}],\"own_revision_hint\":\"中文\"}")
    p.append("</task>\n<evidence_batch>")
    for e in evidence_batch:
        p.append(_safe_json_line(e))
    p.append("</evidence_batch>\n<phase1_results>")
    for model, rows in phase1_results.items():
        if model == reviewer_name:
            continue
        p.append(f"<{model}>")
        for e in evidence_batch:
            r = rows.get(e["match"], {})
            if r:
                p.append(_safe_json_line(_short_prediction_for_prompt(r)))
        p.append(f"</{model}>")
    p.append("</phase1_results>")
    return "\n".join(p)


def build_gemini_final_prompt(evidence_batch: List[Dict[str, Any]], phase1_results: Dict[str, Dict[int, Dict[str, Any]]], critic_reports: Dict[str, List[Dict[str, Any]]]) -> str:
    p = []
    p.append("<final_adjudication_protocol>")
    p.append("你是 Gemini 最终 Web-aware 裁判。你必须重新审计 raw evidence、GPT/Grok 初审、互审意见、v20.3市场簇和联网来源质量。")
    p.append("证据优先级：raw market structure > score_cluster_diagnostics_v203 > HHAD让球语义 > total-goals mode > sharp_money_facts_v203 > tactical/web context > Phase1 consensus。")
    p.append("多数意见不自动成立；若 GPT/Grok 基于同一低赔/单边市场理由一致，这属于相关证据，不是独立证据。")
    p.append(GEMINI_FINAL_AUDIT_ADDENDUM)
    p.append("弱主胜尾部约束：如果主胜概率低于或等于52%、客胜概率不低于23%、且 BTTS=yes，不得把 1-2、2-2、2-3 视为无关尾部。如果最终仍选主胜 2-1，必须说明为什么排除客队反打与 4+ 尾部，并在 risk_score_candidates 或 tail_risk_flags 中体现这些风险；若置信度不足以支撑单一比分，主动下调 recommendation.bet_confidence。")
    p.append("S级必须同时满足：方向边际强、比分簇强、总进球带强、相邻比分解释完整、Sharp/热度不冲突、推荐组件分数透明。仅赔率低赔最多A；无真实联网且依赖阵容/战意/伤停，最高B。")
    p.append("强制输出 recommendation_components，不能只给 tier 和 bet_confidence。")
    p.append("最终推荐等级、是否进 Top4、bet_confidence 全部由你输出；本地只排序，不会改你的足球判断。")
    p.append("</final_adjudication_protocol>\n")
    p.append("<output_schema>")
    p.append(_canonical_output_schema_text())
    p.append("额外要求：每个 prediction 增加 final_web_audit 字段：{\"web_used_by_final\":true,\"decisive_web_evidence\":[],\"ignored_web_evidence\":[],\"web_confidence\":0}")
    p.append("</output_schema>\n")
    p.append("<evidence_batch>")
    for e in evidence_batch:
        p.append(_safe_json_line(e))
    p.append("</evidence_batch>\n<phase1_results>")
    for model in PHASE1_NAMES:
        p.append(f"<{model}>")
        rows = phase1_results.get(model, {})
        for e in evidence_batch:
            r = rows.get(e["match"], {})
            if r:
                p.append(_safe_json_line(_short_prediction_for_prompt(r)))
            else:
                p.append(_safe_json_line({"match": e["match"], "missing": True}))
        p.append(f"</{model}>")
    p.append("</phase1_results>\n<critic_reports>")
    p.append(_safe_json_line(critic_reports))
    p.append("</critic_reports>")
    return "\n".join(p)


def build_fallback_referee_prompt(evidence_batch: List[Dict[str, Any]], phase1_results: Dict[str, Dict[int, Dict[str, Any]]], critic_reports: Dict[str, List[Dict[str, Any]]]) -> str:
    p = []
    p.append("你是 Gemini 终审失败后的 AI fallback referee。不要使用本地规则。基于 raw evidence、v20.3市场簇、Sharp事实、Phase1 和 critic reports 输出最终 predictions。")
    p.append("输出 schema 与 Gemini final 完全一致。必须完成 score_cluster_audit / sharp_money_audit / anchor_audit / recommendation_components。")
    p.append(_canonical_output_schema_text())
    p.append("<evidence_batch>")
    for e in evidence_batch:
        p.append(_safe_json_line(e))
    p.append("</evidence_batch><phase1_results>")
    p.append(_safe_json_line({m: {str(k): _short_prediction_for_prompt(v) for k, v in rows.items()} for m, rows in phase1_results.items()}))
    p.append("</phase1_results><critic_reports>")
    p.append(_safe_json_line(critic_reports))
    p.append("</critic_reports>")
    return "\n".join(p)


def build_consistency_judge_prompt(evidence_batch: List[Dict[str, Any]], final_predictions: Dict[int, Dict[str, Any]]) -> str:
    p = []
    p.append("你是 Consistency Judge，只检查结构一致性，不做足球判断，不改变预测方向/比分，除非存在字段自相矛盾时给出 repair 建议。")
    p.append("输出严格 JSON object：{\"repairs\":[{\"match\":1,\"valid\":true,\"warnings\":[],\"repair\":{...}}]}。")
    p.append("检查：predicted_score方向=final_direction；goal_band与比分总进球一致；btts与比分一致；top3[0]=predicted_score；web_research.used=true时必须有sources；score_cluster_audit/sharp_money_audit/anchor_audit/recommendation_components 必须存在；risk_score_candidates/tail_risk_flags/confidence_downgrade_reason 若存在必须保持数组/字符串结构。")
    p.append("不得根据足球观点改比分，只能修字段。")
    p.append("<evidence_batch>")
    for e in evidence_batch:
        p.append(_safe_json_line(e))
    p.append("</evidence_batch><final_predictions>")
    for idx, r in final_predictions.items():
        p.append(_safe_json_line(_short_prediction_for_prompt(r)))
    p.append("</final_predictions>")
    return "\n".join(p)


TAIL_RISK_PROTECTION_SCORES = ["1-2", "2-2", "2-3"]
TAIL_RISK_DOWNGRADE_REASON = "Weak home favorite with BTTS tail risk"
TWO_ONE_NO_BET_REASON = "2-1 home score blocked by draw/away/tail risk hard gate"


def _normalize_risk_score_candidates(value: Any) -> List[Dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    out: List[Dict[str, Any]] = []
    seen = set()
    for cand in rows[:12]:
        sc = _score_from_candidate(cand)
        if _parse_score(sc)[0] is None or sc in seen:
            continue
        seen.add(sc)
        if isinstance(cand, dict):
            risk_type = str(cand.get("risk_type", cand.get("type", "ai_supplied_tail_risk")))[:120]
            reason = str(cand.get("reason", cand.get("logic", cand.get("explanation", ""))))[:900]
            prob = cand.get("prob", cand.get("probability", cand.get("pct", None)))
        else:
            risk_type = "ai_supplied_tail_risk"
            reason = ""
            prob = None
        row = {"score": sc, "risk_type": risk_type, "reason": reason}
        if prob is not None:
            row["prob"] = round(_prob_to_float(prob), 3)
        out.append(row)
    return out


def _append_risk_candidate(candidates: List[Dict[str, Any]], score: str, risk_type: str, reason: str) -> None:
    if any(c.get("score") == score for c in candidates):
        return
    candidates.append({"score": score, "risk_type": risk_type, "reason": reason})


def _listify_strs(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v)[:160] for v in value if str(v).strip()]
    if value in (None, "", {}, []):
        return []
    return [str(value)[:160]]


def _raw_btts_yes_signal(row: Dict[str, Any], raw_item: Dict[str, Any]) -> bool:
    if str(row.get("btts", "")).strip().lower() == "yes":
        return True
    for v in [raw_item.get("btts"), raw_item.get("btts_likelihood")]:
        if str(v).strip().lower() in {"yes", "y", "true", "是", "双方进球"}:
            return True
    ctx = raw_item.get("contextual_logic") if isinstance(raw_item.get("contextual_logic"), dict) else {}
    if str(ctx.get("btts_likelihood", "")).strip().lower() in {"yes", "y", "true", "是", "双方进球"}:
        return True
    return False


def _extract_prob_map_0_100(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, float] = {}
    for key in ["home", "draw", "away"]:
        raw = value.get(key)
        pct = _f(raw, 0.0)
        if 0 < pct <= 1.0:
            pct *= 100.0
        out[key] = pct
    return out


def _candidate_score_prob(cand: Dict[str, Any]) -> float:
    if not isinstance(cand, dict):
        return 0.0
    for key in ["prob", "probability", "pct", "percent", "score_prob"]:
        if key in cand:
            val = _f(cand.get(key), 0.0)
            return val * 100.0 if 0 < val <= 1.0 else val
    return 0.0


def _collect_score_candidates_for_gate(pred: Dict[str, Any], raw_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source_name, container in [
        ("top3", pred.get("top3")),
        ("risk_score_candidates", pred.get("risk_score_candidates")),
        ("candidate_scores", pred.get("candidate_scores")),
        ("raw_top3", raw_item.get("top3")),
        ("raw_candidate_scores", raw_item.get("candidate_scores")),
        ("raw_risk_score_candidates", raw_item.get("risk_score_candidates")),
    ]:
        if not isinstance(container, list):
            continue
        for cand in container:
            if not isinstance(cand, dict):
                continue
            score = _score_from_candidate(cand)
            if not score:
                score = str(cand.get("score", "")).strip()
            if not score:
                continue
            rows.append({
                "score": score,
                "prob": _candidate_score_prob(cand),
                "source": source_name,
                "risk_type": cand.get("risk_type", cand.get("type", "")),
            })
    return rows


def apply_two_one_home_hard_no_bet_gate(pred: Dict[str, Any]) -> Dict[str, Any]:
    """Hard recommendation gate for fragile home 2-1 calls; never changes score/direction."""
    if not isinstance(pred, dict):
        return pred
    if pred.get("final_direction") != "home" or _score_from_candidate(pred.get("predicted_score")) != "2-1":
        return pred

    raw_item = pred.get("raw_item", {}) if isinstance(pred.get("raw_item"), dict) else {}
    probs = _extract_prob_map_0_100(pred.get("direction_probs")) or {
        "home": _f(pred.get("home_win_pct"), 0.0),
        "draw": _f(pred.get("draw_pct"), 0.0),
        "away": _f(pred.get("away_win_pct"), 0.0),
    }
    home_pct = _f(probs.get("home"), 0.0)
    draw_pct = _f(probs.get("draw"), 0.0)
    away_pct = _f(probs.get("away"), 0.0)
    non_home_pct = draw_pct + away_pct

    matrix_probs = _extract_prob_map_0_100(pred.get("matrix_direction_probs"))
    matrix_flags = pred.get("matrix_disagreement_flags", {}) if isinstance(pred.get("matrix_disagreement_flags"), dict) else {}
    matrix_draw = _f(matrix_probs.get("draw"), 0.0)
    matrix_away = _f(matrix_probs.get("away"), 0.0)

    candidates = _collect_score_candidates_for_gate(pred, raw_item)
    risk_scores = {c["score"] for c in candidates if c.get("score")}
    # Candidate tail probability is reserved for explicit AI risk/candidate lists.
    # Matrix top-score distributions are handled separately through matrix flags
    # and direction probabilities; summing all matrix away/draw top scores here
    # can incorrectly turn a clean strong-home 2-1 into NO_BET simply because a
    # calibrated probability distribution contains normal low-probability tails.
    away_tail_prob = sum(_f(c.get("prob"), 0.0) for c in candidates if _score_direction(c.get("score")) == "away")
    draw_tail_prob = sum(_f(c.get("prob"), 0.0) for c in candidates if _score_direction(c.get("score")) == "draw")
    tail_scores_present = bool({"1-2", "2-2", "2-3", "3-2"} & risk_scores)
    btts_or_tail = _raw_btts_yes_signal(pred, raw_item) or str(pred.get("btts", "")).lower() == "yes" or tail_scores_present

    reasons: List[str] = []
    if home_pct and home_pct <= 52.0 and away_pct >= 23.0 and btts_or_tail:
        reasons.append("weak_home_2_1_btts_away_pct_gate")
    if home_pct and non_home_pct >= 47.0 and btts_or_tail:
        reasons.append("non_home_mass_too_high_for_home_2_1")
    if draw_pct >= 29.0 or away_pct >= 26.0:
        reasons.append("draw_or_away_probability_too_high_for_home_2_1")
    if bool(matrix_flags.get("matrix_vs_final_direction_conflict")):
        reasons.append("matrix_direction_conflicts_with_home_2_1")
    if bool(matrix_flags.get("matrix_away_tail_warning")) or bool(matrix_flags.get("matrix_draw_risk_warning")):
        reasons.append("matrix_draw_or_away_tail_warning")
    if bool(matrix_flags.get("matrix_high_goal_tail_conflict")):
        reasons.append("matrix_high_goal_tail_conflict_with_2_1")
    if matrix_away >= 32.0 or matrix_draw >= 30.0:
        reasons.append("matrix_non_home_probability_too_high")
    if away_tail_prob >= 16.0 or draw_tail_prob >= 18.0:
        reasons.append("candidate_tail_probability_too_high")
    if tail_scores_present and (away_pct >= 22.0 or draw_pct >= 26.0):
        reasons.append("explicit_1_2_2_2_2_3_tail_present")

    if not reasons:
        return pred

    rec = pred.setdefault("recommendation", {}) if isinstance(pred.get("recommendation"), dict) else {}
    if not isinstance(pred.get("recommendation"), dict):
        pred["recommendation"] = rec
    rec["tier"] = "D"
    rec["is_recommended"] = False
    original_bet_confidence = int(_clip(_f(rec.get("bet_confidence", pred.get("confidence", 0)), 0), 0, 100))
    rec.setdefault("original_bet_confidence", original_bet_confidence)
    rec["risk_adjusted_bet_confidence"] = min(original_bet_confidence, 49)
    rec["display_bet_confidence"] = min(original_bet_confidence, 49)
    rec["risk_level"] = "high"
    tags = rec.get("risk_tags", [])
    if not isinstance(tags, list):
        tags = [str(tags)] if tags else []
    tags.extend(["two_one_home_hard_no_bet", *reasons])
    rec["risk_tags"] = list(dict.fromkeys(str(x) for x in tags if str(x).strip()))[:20]

    pred["recommendation_tier"] = "D"
    pred["recommend_gate_pass"] = False
    gate_reasons = pred.get("recommend_gate_reasons", [])
    if not isinstance(gate_reasons, list):
        gate_reasons = [str(gate_reasons)] if gate_reasons else []
    gate_reasons.extend([TWO_ONE_NO_BET_REASON, *reasons])
    pred["recommend_gate_reasons"] = list(dict.fromkeys(str(x) for x in gate_reasons if str(x).strip()))[:24]
    pred["direction_tier"] = "D"
    pred["score_tier"] = "D"
    original_confidence = int(_clip(_f(pred.get("confidence", rec.get("bet_confidence", 0)), 0), 0, 100))
    pred.setdefault("original_confidence", original_confidence)
    pred["risk_adjusted_confidence"] = min(original_confidence, 49)
    pred["display_confidence"] = min(original_confidence, 49)
    pred["confidence_downgrade_reason"] = TWO_ONE_NO_BET_REASON
    pred["no_bet_reason"] = TWO_ONE_NO_BET_REASON + ": " + ",".join(reasons)
    pred["sub50_tiebreaker_warning"] = bool(home_pct <= 52.0 or non_home_pct >= 47.0)
    pred.setdefault("recommendation_downgrade_reasons", []).extend(["two_one_home_hard_no_bet", *reasons])
    pred.setdefault("validation_warnings", []).extend(["two_one_home_hard_no_bet_gate_applied", *reasons])
    pred["validation_warnings"] = list(dict.fromkeys(pred.get("validation_warnings", [])))
    return pred


def _contains_tail_risk_signal(row: Dict[str, Any], raw_item: Dict[str, Any], risk_candidates: List[Dict[str, Any]]) -> bool:
    if str(row.get("goal_band", "")).strip() == "4+":
        return True
    scores = {c.get("score") for c in risk_candidates if isinstance(c, dict)}
    scores.update(c.get("score") for c in row.get("top3", []) if isinstance(c, dict))
    for key in ["candidate_scores", "top3", "risk_score_candidates"]:
        for cand in raw_item.get(key, []) if isinstance(raw_item.get(key), list) else []:
            sc = _score_from_candidate(cand)
            if sc:
                scores.add(sc)
    if any(sc in {"2-2", "2-3", "3-2"} for sc in scores):
        return True
    text = _json_compact({
        "tail_risk_flags": raw_item.get("tail_risk_flags"),
        "risk_tags": _as_dict(raw_item.get("recommendation")).get("risk_tags"),
        "reason": raw_item.get("reason"),
        "score_cluster_audit": raw_item.get("score_cluster_audit"),
    }, 3000).lower()
    return any(token in text for token in ["4+", "high_btts", "tail", "尾部", "高比分", "2-3"])


def apply_weak_home_tail_risk_protection(row: Dict[str, Any]) -> Dict[str, Any]:
    """Protocol-level risk display/downgrade; never changes final_direction or predicted_score."""
    if not isinstance(row, dict):
        return row
    raw_item = row.get("raw_item", {}) if isinstance(row.get("raw_item"), dict) else {}
    risk_candidates = _normalize_risk_score_candidates(raw_item.get("risk_score_candidates", row.get("risk_score_candidates", [])))
    tail_flags = _listify_strs(raw_item.get("tail_risk_flags", row.get("tail_risk_flags", [])))

    probs = row.get("direction_probs", {}) if isinstance(row.get("direction_probs"), dict) else {}
    home_pct = _f(probs.get("home"), 0.0)
    away_pct = _f(probs.get("away"), 0.0)
    btts_or_tail = _raw_btts_yes_signal(row, raw_item) or _contains_tail_risk_signal(row, raw_item, risk_candidates)
    weak_home_tail = (
        row.get("final_direction") == "home"
        and home_pct <= 52.0
        and away_pct >= 23.0
        and btts_or_tail
    )

    if weak_home_tail:
        for sc in TAIL_RISK_PROTECTION_SCORES:
            _append_risk_candidate(
                risk_candidates,
                sc,
                "weak_home_favorite_btts_tail",
                "home_win_pct<=52 and away_win_pct>=23 with BTTS/tail risk; keep away fight-back and 4+ draw/away tails visible",
            )
        tail_flags.extend([
            "weak_home_favorite_btts_tail",
            "away_win_not_negligible",
            "protect_1_2_2_2_2_3_tail",
        ])
        rec = row.setdefault("recommendation", {}) if isinstance(row.get("recommendation"), dict) else {}
        if not isinstance(row.get("recommendation"), dict):
            row["recommendation"] = rec
        old_conf = int(_clip(_f(rec.get("bet_confidence", 0), 0), 0, 100))
        if old_conf >= 70:
            rec.setdefault("original_bet_confidence", old_conf)
            rec["risk_adjusted_bet_confidence"] = min(old_conf, 60)
            rec["display_bet_confidence"] = min(old_conf, 60)
            row["confidence_downgrade_reason"] = TAIL_RISK_DOWNGRADE_REASON
        elif raw_item.get("confidence_downgrade_reason"):
            row["confidence_downgrade_reason"] = str(raw_item.get("confidence_downgrade_reason"))[:300]
        else:
            row.setdefault("confidence_downgrade_reason", "")
        risk_tags = rec.get("risk_tags", [])
        if not isinstance(risk_tags, list):
            risk_tags = [str(risk_tags)] if risk_tags else []
        risk_tags.extend(["weak_home_favorite_btts_tail", "away_fightback_tail"])
        rec["risk_tags"] = list(dict.fromkeys(str(x) for x in risk_tags if str(x).strip()))[:12]
        if home_pct <= 50.0 or away_pct >= 24.0:
            rec["tier"] = "D"
            rec["is_recommended"] = False
            rec.setdefault("original_bet_confidence", old_conf)
            risk_conf = min(int(_clip(_f(rec.get("bet_confidence", old_conf), 0), 0, 100)), 49)
            rec["risk_adjusted_bet_confidence"] = risk_conf
            rec["display_bet_confidence"] = risk_conf
            rec["risk_level"] = "high"
            row["confidence_downgrade_reason"] = TWO_ONE_NO_BET_REASON
            row.setdefault("recommendation_downgrade_reasons", []).append("weak_home_tail_forced_no_bet")
            row.setdefault("validation_warnings", []).append("weak_home_tail_forced_no_bet")
        elif rec.get("risk_level") in (None, "", "low"):
            rec["risk_level"] = "medium"
        row.setdefault("validation_warnings", []).append("weak_home_favorite_btts_tail_protection_applied")
    else:
        row["confidence_downgrade_reason"] = str(raw_item.get("confidence_downgrade_reason", row.get("confidence_downgrade_reason", "")))[:300]

    row["risk_score_candidates"] = risk_candidates[:12]
    row["tail_risk_flags"] = list(dict.fromkeys(tail_flags))[:20]
    row["validation_warnings"] = list(dict.fromkeys(row.get("validation_warnings", [])))
    return row


def normalize_ai_predictions(obj: Any, expected_matches: List[int], source_model: str, phase: str) -> Dict[int, Dict[str, Any]]:
    out = _BASE_NORMALIZE_AI_PREDICTIONS_V2021(obj, expected_matches, source_model, phase)
    for idx, row in out.items():
        raw_item = row.get("raw_item", {}) if isinstance(row.get("raw_item"), dict) else {}
        for k in [
            "score_cluster_audit", "sharp_money_audit", "recommendation_components",
            "risk_score_candidates", "tail_risk_flags", "confidence_downgrade_reason",
            "market_audit", "score_cluster_audit", "goal_market_audit", "market_conflicts", "candidate_scores",
            "public_heat_audit", "packet_news_risk_audit", "trap_candidates", "final_score_audit",
        ]:
            if isinstance(raw_item.get(k), (dict, list)):
                row[k] = raw_item.get(k)
        warnings = list(row.get("validation_warnings", []))
        if not isinstance(raw_item.get("score_cluster_audit"), dict):
            warnings.append("score_cluster_audit_missing_or_invalid")
        if not isinstance(raw_item.get("sharp_money_audit"), dict):
            warnings.append("sharp_money_audit_missing_or_invalid")
        if not isinstance(raw_item.get("recommendation_components"), dict):
            warnings.append("recommendation_components_missing_or_invalid")
        row["validation_warnings"] = list(dict.fromkeys(warnings))
        apply_weak_home_tail_risk_protection(row)
    return out


def _normalize_prob_dict_shadow(d: Dict[Any, float], floor: float = 0.0) -> Dict[Any, float]:
    vals = {k: max(floor, _f(v, 0.0)) for k, v in (d or {}).items()}
    total = sum(vals.values())
    if total <= 0:
        n = len(vals) or 1
        return {k: 1.0 / n for k in vals}
    return {k: v / total for k, v in vals.items()}


def fair_probs_from_1x2_shadow(sp_h: float, sp_d: float, sp_a: float, method: str = "power") -> Dict[str, Any]:
    """Shadow-only 1X2 fair probabilities in percent; never feeds final decision fields."""
    odds = {"home": _f(sp_h), "draw": _f(sp_d), "away": _f(sp_a)}
    if any(v <= 1.01 for v in odds.values()):
        return {
            "method": "fallback",
            "fair_probs": {"home": 33.3, "draw": 33.3, "away": 33.4},
            "raw_implied": {"home": 33.3, "draw": 33.3, "away": 33.4},
            "overround": 0.0,
        }
    q = {k: 1.0 / v for k, v in odds.items()}
    overround_sum = sum(q.values())
    raw_pct = {k: round(v * 100, 3) for k, v in q.items()}
    if method == "multiplicative":
        p = _normalize_prob_dict_shadow(q)
        used = "multiplicative"
    else:
        lo, hi = 0.01, 10.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            sm = sum(v ** mid for v in q.values())
            if sm > 1.0:
                lo = mid
            else:
                hi = mid
        k = (lo + hi) / 2.0
        p = _normalize_prob_dict_shadow({name: val ** k for name, val in q.items()})
        used = "power"
    fair = {k: round(v * 100, 3) for k, v in p.items()}
    drift = round(100.0 - sum(fair.values()), 3)
    if fair:
        last = list(fair.keys())[-1]
        fair[last] = round(fair[last] + drift, 3)
    return {"method": used, "fair_probs": fair, "raw_implied": raw_pct, "overround": round(overround_sum - 1.0, 5)}


def fair_probs_from_ttg_shadow(match_obj: Dict[str, Any], method: str = "power") -> Dict[int, float]:
    """Shadow-only total-goals distribution from a0-a7; returns 0..1 probabilities."""
    raw = {}
    for g in range(8):
        odd = _f((match_obj or {}).get(f"a{g}", 0))
        if odd > 1.01:
            raw[g] = 1.0 / odd
    if len(raw) < 3:
        return {}
    if method == "multiplicative":
        return _normalize_prob_dict_shadow(raw)
    lo, hi = 0.01, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        sm = sum(v ** mid for v in raw.values())
        if sm > 1.0:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2.0
    return _normalize_prob_dict_shadow({g: v ** k for g, v in raw.items()})


def crs_implied_probabilities_shadow(match_obj: Dict[str, Any]) -> Tuple[Dict[str, float], float, float]:
    """Shadow-only CRS implied probabilities in percent, including coarse other-score buckets."""
    raw_odds = {}
    for score, key in CRS_FULL_MAP.items():
        odds = _f((match_obj or {}).get(key, 0))
        if odds > 1.1:
            raw_odds[score] = odds
    extras = {}
    for key, scores_set in [
        ("crs_win", SCORE_OTHERS_HOME),
        ("crs_same", SCORE_OTHERS_DRAW),
        ("crs_lose", SCORE_OTHERS_AWAY),
    ]:
        odds = _f((match_obj or {}).get(key, 0))
        if odds > 1.1:
            extras[key] = {"odds": odds, "scores": scores_set}
    if len(raw_odds) < 8:
        return {}, 0.0, 0.0
    raw_sum = sum(1.0 / o for o in raw_odds.values()) + sum(1.0 / ex["odds"] for ex in extras.values())
    if raw_sum <= 0:
        return {}, 0.0, 0.0
    probs = {score: (1.0 / odds) / raw_sum * 100.0 for score, odds in raw_odds.items()}
    for ex in extras.values():
        total_prob = (1.0 / ex["odds"]) / raw_sum * 100.0
        if ex["scores"]:
            per = total_prob / len(ex["scores"])
            for sc in ex["scores"]:
                probs[sc] = probs.get(sc, 0.0) + per
    return probs, round(raw_sum - 1.0, 3), round(len(raw_odds) / max(1, len(CRS_FULL_MAP)), 2)


def _poisson_pmf_shadow(lam: float, k: int) -> float:
    lam = max(0.05, min(8.0, float(lam)))
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _matrix_moments_shadow(probs: Dict[str, float]) -> Dict[str, float]:
    regular = {}
    for sc, p in (probs or {}).items():
        h, a = _parse_score(sc)
        if h is None or a is None or h > 8 or a > 8:
            continue
        regular[(h, a)] = _f(p, 0.0)
    total = sum(regular.values())
    if total <= 0:
        return {}
    norm = {k: v / total for k, v in regular.items()}
    e_h = sum(h * p for (h, _a), p in norm.items())
    e_a = sum(a * p for (_h, a), p in norm.items())
    var_h = sum((h - e_h) ** 2 * p for (h, _a), p in norm.items())
    var_a = sum((a - e_a) ** 2 * p for (_h, a), p in norm.items())
    std_h = math.sqrt(var_h) if var_h > 0 else 0.01
    std_a = math.sqrt(var_a) if var_a > 0 else 0.01
    cov = sum((h - e_h) * (a - e_a) * p for (h, a), p in norm.items())
    corr = cov / (std_h * std_a) if std_h * std_a > 0 else 0.0
    return {"lambda_h": round(e_h, 3), "lambda_a": round(e_a, 3), "lambda_total": round(e_h + e_a, 3), "var_h": round(var_h, 3), "var_a": round(var_a, 3), "corr": round(corr, 3)}


def _matrix_shape_verdict_shadow(moments: Dict[str, float]) -> str:
    if not moments:
        return "unknown"
    lh, la, lt = moments.get("lambda_h", 1.3), moments.get("lambda_a", 1.2), moments.get("lambda_total", 2.5)
    corr = moments.get("corr", 0.0)
    if lt >= 3.0 and corr >= 0.15:
        return "shootout"
    if lt <= 2.2 and moments.get("var_h", 1.0) < 1.2 and moments.get("var_a", 1.0) < 1.2:
        return "grinder"
    if lh - la >= 1.2:
        return "lopsided_h"
    if la - lh >= 1.2:
        return "lopsided_a"
    if abs(lh - la) < 0.4:
        return "balanced"
    return "normal"


def _direction_probs_from_score_probs_shadow(probs: Dict[str, float]) -> Dict[str, float]:
    out = {"home": 0.0, "draw": 0.0, "away": 0.0}
    for sc, p in (probs or {}).items():
        d = _score_direction(sc)
        if d in out:
            out[d] += p
    total = sum(out.values())
    if total <= 0:
        return {"home": 33.3, "draw": 33.3, "away": 33.4}
    return {k: round(v / total * 100.0, 2) for k, v in out.items()}


def build_unified_score_matrix_shadow(match_obj: Dict[str, Any], max_goals: int = 8) -> Dict[str, Any]:
    """Build a v18.1-inspired score matrix as diagnostics only; no final decision writes."""
    max_goals = max(5, min(10, int(max_goals or 8)))
    sp_h = _f((match_obj or {}).get("sp_home", (match_obj or {}).get("win", 0)))
    sp_d = _f((match_obj or {}).get("sp_draw", (match_obj or {}).get("same", 0)))
    sp_a = _f((match_obj or {}).get("sp_away", (match_obj or {}).get("lose", 0)))
    fair_pack = fair_probs_from_1x2_shadow(sp_h, sp_d, sp_a, method="power")
    target_dir = {k: v / 100.0 for k, v in fair_pack.get("fair_probs", {}).items()}
    ttg = fair_probs_from_ttg_shadow(match_obj, method="power")
    crs_probs_pct, crs_margin, crs_coverage = crs_implied_probabilities_shadow(match_obj)
    crs_moments = _matrix_moments_shadow(crs_probs_pct)
    if crs_moments:
        lam_h = _f(crs_moments.get("lambda_h"), 1.25)
        lam_a = _f(crs_moments.get("lambda_a"), 1.15)
    else:
        exp_goals = sum(g * p for g, p in ttg.items()) if ttg else 2.5
        home_edge = (target_dir.get("home", 0.333) - target_dir.get("away", 0.333))
        lam_h = exp_goals * (0.50 + max(-0.25, min(0.25, home_edge * 0.65)))
        lam_a = max(0.05, exp_goals - lam_h)
    base = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            base[f"{h}-{a}"] = _poisson_pmf_shadow(lam_h, h) * _poisson_pmf_shadow(lam_a, a)
    matrix = _normalize_prob_dict_shadow(base)
    crs_grid = {sc: max(1e-9, crs_probs_pct.get(sc, 0.0) / 100.0) for sc in matrix}
    crs_mass = sum(crs_grid.values())
    if crs_mass > 0.05 and crs_coverage >= 0.45:
        crs_grid = _normalize_prob_dict_shadow(crs_grid, floor=1e-9)
        crs_weight = min(0.72, 0.35 + crs_coverage * 0.45)
        matrix = _normalize_prob_dict_shadow({sc: (matrix[sc] ** (1.0 - crs_weight)) * (crs_grid[sc] ** crs_weight) for sc in matrix})

    def direction_of_score(sc: str) -> str:
        d = _score_direction(sc)
        return d if d in VALID_DIRS else "draw"

    def total_of_score(sc: str) -> int:
        h, a = _parse_score(sc)
        return 99 if h is None else h + a

    for _ in range(12):
        cur_dir = {"home": 0.0, "draw": 0.0, "away": 0.0}
        for sc, p in matrix.items():
            cur_dir[direction_of_score(sc)] += p
        for sc in list(matrix.keys()):
            d = direction_of_score(sc)
            if cur_dir.get(d, 0.0) > 1e-9:
                ratio = target_dir.get(d, cur_dir[d]) / cur_dir[d]
                matrix[sc] *= max(0.35, min(2.85, ratio))
        matrix = _normalize_prob_dict_shadow(matrix)
        if ttg:
            cur_ttg = {}
            for sc, p in matrix.items():
                bucket = min(total_of_score(sc), 7)
                cur_ttg[bucket] = cur_ttg.get(bucket, 0.0) + p
            for sc in list(matrix.keys()):
                bucket = min(total_of_score(sc), 7)
                if bucket in ttg and cur_ttg.get(bucket, 0.0) > 1e-9:
                    ratio = ttg[bucket] / cur_ttg[bucket]
                    matrix[sc] *= max(0.35, min(2.85, ratio))
            matrix = _normalize_prob_dict_shadow(matrix)
    dir_probs = {"home": 0.0, "draw": 0.0, "away": 0.0}
    goal_probs = {}
    for sc, p in matrix.items():
        dir_probs[direction_of_score(sc)] += p
        bucket = min(total_of_score(sc), 7)
        goal_probs[bucket] = goal_probs.get(bucket, 0.0) + p
    top_scores = sorted(matrix.items(), key=lambda x: x[1], reverse=True)
    goal_pct = {str(k): round(v * 100.0, 2) for k, v in sorted(goal_probs.items())}
    drift = round(100.0 - sum(goal_pct.values()), 2)
    if goal_pct:
        last = list(goal_pct.keys())[-1]
        goal_pct[last] = round(goal_pct[last] + drift, 2)
    matrix_moments = _matrix_moments_shadow({sc: p * 100.0 for sc, p in matrix.items()})
    return {
        "matrix": matrix,
        "fair_1x2": fair_pack,
        "direction_probs": {k: round(v * 100.0, 2) for k, v in dir_probs.items()},
        "goal_probs": goal_pct,
        "top_scores": [{"score": sc, "prob": round(p * 100.0, 3)} for sc, p in top_scores[:20]],
        "lambda_h": round(lam_h, 3),
        "lambda_a": round(lam_a, 3),
        "shape_verdict": _matrix_shape_verdict_shadow(matrix_moments or crs_moments),
        "crs_margin": crs_margin,
        "crs_coverage": crs_coverage,
        "source": "unified_score_matrix_v18_1_shadow",
    }


def attach_matrix_shadow_fields(prediction: Dict[str, Any], match_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Attach matrix diagnostics without modifying final_direction, score, confidence, result, or display fields."""
    protected = {k: prediction.get(k) for k in ["predicted_score", "final_direction", "confidence", "result", "display_direction", "home_win_pct", "draw_pct", "away_win_pct"]}
    try:
        matrix_pack = build_unified_score_matrix_shadow(match_obj)
        top_scores = matrix_pack.get("top_scores", [])
        recommended_score = top_scores[0]["score"] if top_scores else ""
        recommended_direction = _score_direction(recommended_score) if recommended_score else None
        matrix_direction_probs = matrix_pack.get("direction_probs", {})
        final_direction = prediction.get("final_direction")
        predicted_score = prediction.get("predicted_score")
        confidence = _f(prediction.get("confidence", 0), 0)
        pred_total = _score_total(predicted_score)
        high_tail = sum(_f(v, 0.0) for k, v in matrix_pack.get("goal_probs", {}).items() if _i(k, 0) >= 4)
        away_tail = sum(_f(row.get("prob", 0.0), 0.0) for row in top_scores[:8] if _score_direction(row.get("score")) == "away")
        draw_prob = _f(matrix_direction_probs.get("draw", 0.0), 0.0)
        flags = {
            "matrix_vs_final_direction_conflict": bool(recommended_direction in VALID_DIRS and final_direction in VALID_DIRS and recommended_direction != final_direction),
            "matrix_high_goal_tail_conflict": bool(pred_total is not None and pred_total <= 2 and high_tail >= 38.0),
            "matrix_away_tail_warning": bool(final_direction != "away" and (matrix_direction_probs.get("away", 0.0) >= 34.0 or away_tail >= 18.0)),
            "matrix_draw_risk_warning": bool(final_direction != "draw" and draw_prob >= 30.0),
            "matrix_low_confidence_warning": bool(confidence < 55 or (top_scores and _f(top_scores[0].get("prob", 0.0), 0.0) < 8.0)),
        }
        prediction.update({
            "matrix_direction_probs": matrix_direction_probs,
            "matrix_top_scores": top_scores,
            "matrix_goal_probs": matrix_pack.get("goal_probs", {}),
            "matrix_lambda_home": matrix_pack.get("lambda_h"),
            "matrix_lambda_away": matrix_pack.get("lambda_a"),
            "matrix_shape_verdict": matrix_pack.get("shape_verdict", "unknown"),
            "matrix_recommended_score": recommended_score,
            "matrix_recommended_direction": recommended_direction,
            "matrix_disagreement_flags": flags,
            "matrix_shadow_error": "",
        })
    except Exception as exc:
        prediction["matrix_shadow_error"] = str(exc)[:300]
    for key, value in protected.items():
        prediction[key] = value
    return prediction



PREMATCH_V2_VERSION = "v20.4_pre_match_factor_gate"

LEAGUE_DNA_PROFILES = {
    "德甲": {"volatility": "high", "draw_risk": "medium", "btts": "high", "away_penalty": 1, "notes": "高节奏/高BTTS/末段波动，客胜与小胜比分需降权复核"},
    "瑞超": {"volatility": "medium", "draw_risk": "high", "btts": "medium", "away_penalty": 1, "notes": "北欧联赛平局与1球差较多，弱优势必须防1-1/2-2"},
    "芬超": {"volatility": "medium", "draw_risk": "high", "btts": "medium", "away_penalty": 1, "notes": "低比分和平局权重偏高，客胜需市场确认"},
    "挪超": {"volatility": "high", "draw_risk": "medium", "btts": "high", "away_penalty": 1, "notes": "节奏与大球尾部较强，比分需用簇而非单点"},
    "美职": {"volatility": "high", "draw_risk": "high", "btts": "high", "away_penalty": 2, "notes": "主客差/旅行/反转波动大，客胜默认高风险"},
    "葡超": {"volatility": "medium", "draw_risk": "medium", "btts": "medium", "away_penalty": 0, "notes": "强弱分化明显，但中下游/保级题材需防平"},
}


def _tier_cap_value(tier: str) -> int:
    return {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}.get(str(tier).upper(), 1)


def _cap_recommendation_tier(pred: Dict[str, Any], max_tier: str, reason: str, tag: str = "") -> None:
    rec = pred.setdefault("recommendation", {}) if isinstance(pred.get("recommendation"), dict) else {}
    if not isinstance(pred.get("recommendation"), dict):
        pred["recommendation"] = rec
    cur = str(rec.get("tier", pred.get("recommendation_tier", "D"))).upper()
    max_tier = str(max_tier).upper()
    if _tier_cap_value(cur) > _tier_cap_value(max_tier):
        rec["tier"] = max_tier
        pred["recommendation_tier"] = max_tier
        pred.setdefault("recommendation_downgrade_reasons", []).append(reason)
    else:
        rec["tier"] = cur if cur in {"S", "A", "B", "C", "D"} else "D"
        pred["recommendation_tier"] = rec["tier"]
    if tag:
        tags = rec.get("risk_tags", [])
        if not isinstance(tags, list):
            tags = [str(tags)] if tags else []
        tags.append(tag)
        rec["risk_tags"] = list(dict.fromkeys(str(x) for x in tags if str(x).strip()))[:24]
    if rec.get("tier") not in {"S", "A", "B"}:
        rec["is_recommended"] = False
        pred["recommend_gate_pass"] = False
        pred.setdefault("recommend_gate_reasons", []).append(reason)
    else:
        rec["is_recommended"] = bool(rec.get("is_recommended", True))


def _league_dna_profile(league: str) -> Dict[str, Any]:
    league_s = str(league or "")
    for key, profile in LEAGUE_DNA_PROFILES.items():
        if key in league_s:
            return {"key": key, **profile}
    if any(x in league_s for x in ["杯", "Cup", "欧冠", "欧联", "亚冠", "世俱杯"]):
        return {"key": "cup_or_cross_context", "volatility": "high", "draw_risk": "medium", "btts": "medium", "away_penalty": 1, "notes": "杯赛/跨赛制比赛轮换、战意、旅行和首发不确定性更高"}
    return {"key": "generic", "volatility": "medium", "draw_risk": "medium", "btts": "medium", "away_penalty": 0, "notes": "未配置专属联赛DNA，按通用风险处理"}


def _has_confirmed_web_or_lineup(pred: Dict[str, Any], match_obj: Dict[str, Any]) -> Tuple[bool, bool]:
    web = pred.get("web_research", {}) if isinstance(pred.get("web_research"), dict) else {}
    web_used = bool(web.get("used")) and bool(web.get("sources"))
    dq = pred.get("data_quality", {}) if isinstance(pred.get("data_quality"), dict) else {}
    text = _json_compact({"data_quality": dq, "information": match_obj.get("information"), "intelligence": match_obj.get("intelligence")}, 3000).lower()
    lineup_tokens = ["official_lineup", "confirmed_lineup", "lineup_confirmed", "首发已确认", "官方首发"]
    lineup_confirmed = any(t.lower() in text for t in lineup_tokens)
    return web_used, lineup_confirmed


def _draw_cluster_present(pred: Dict[str, Any], match_obj: Dict[str, Any]) -> bool:
    scores = set()
    for key in ["top_score_candidates", "unified_matrix_top_scores", "matrix_top_scores", "top3", "risk_score_candidates"]:
        rows = pred.get(key, [])
        if isinstance(rows, list):
            for x in rows[:12]:
                if isinstance(x, dict):
                    sc = _score_from_candidate(x.get("score", x))
                elif isinstance(x, (list, tuple)) and x:
                    sc = _score_from_candidate(x[0])
                else:
                    sc = _score_from_candidate(x)
                if sc:
                    scores.add(sc)
    s11 = _f(match_obj.get("s11", 0), 0)
    s22 = _f(match_obj.get("s22", 0), 0)
    return bool({"0-0", "1-1", "2-2", "3-3"} & scores) or (0 < s11 <= 7.2) or (0 < s22 <= 13.5)


def _money_flow_state(pred: Dict[str, Any]) -> Dict[str, Any]:
    mf = pred.get("money_flow", {}) if isinstance(pred.get("money_flow"), dict) else {}
    sharp = str(mf.get("sharp_money_direction", "unclear") or "unclear").lower()
    public = str(mf.get("public_money_direction", "unclear") or "unclear").lower()
    final_dir = str(pred.get("final_direction", ""))
    sharp_clear = sharp in VALID_DIRS
    public_clear = public in VALID_DIRS
    conflict = bool(sharp_clear and final_dir in VALID_DIRS and sharp != final_dir)
    reverse = bool(mf.get("reverse_line_movement"))
    return {"sharp": sharp, "public": public, "sharp_clear": sharp_clear, "public_clear": public_clear, "conflict": conflict, "reverse": reverse}






def extract_structured_prematch_factors(pred: Dict[str, Any], match_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort structured prematch factors from existing packet fields; no external calls."""
    intelligence = match_obj.get("intelligence", {}) if isinstance(match_obj.get("intelligence"), dict) else {}
    information = match_obj.get("information", {}) if isinstance(match_obj.get("information"), dict) else {}
    web = pred.get("web_research", {}) if isinstance(pred.get("web_research"), dict) else {}
    money = _money_flow_state(pred) if "_money_flow_state" in globals() else {}
    text = _json_compact({
        "league": match_obj.get("league"),
        "intelligence": intelligence,
        "information": information,
        "expert_intro": match_obj.get("expert_intro"),
        "baseface": match_obj.get("baseface"),
        "contextual_logic": pred.get("contextual_logic"),
        "reason": pred.get("ai_native_reason", pred.get("reason", "")),
        "web_key_findings": web.get("key_findings") if isinstance(web, dict) else [],
    }, 10000).lower()

    def has_any(words: List[str]) -> bool:
        return any(str(w).lower() in text for w in words)

    h_inj = str(intelligence.get("h_inj", ""))
    a_inj = str(intelligence.get("g_inj", intelligence.get("away_inj", "")))
    injury_text = f"{h_inj} {a_inj}".lower()
    critical_words = ["门将", "中卫", "后腰", "队长", "头号", "核心", "射手"]
    critical_absence = any(w in injury_text for w in critical_words) or (any(w in injury_text for w in ["停赛", "受伤"]) and any(w in injury_text for w in ["主力", "核心", "队长", "头号"]))

    lineup_confirmed = has_any(["official_lineup", "confirmed_lineup", "lineup_confirmed", "首发已确认", "官方首发"])
    lineup_predicted = has_any(["预计首发", "预测首发", "probable lineup", "expected lineup"])
    lineup_unclear = not lineup_confirmed and not lineup_predicted

    rotation_risk = has_any(["轮换", "替补阵容", "二队", "青年队", "大幅轮休", "主力轮休", "rotation", "rotate", "rested"])
    fatigue_risk = has_any(["一周双赛", "连续客场", "短休", "加时", "刚踢完", "欧战后", "travel", "疲劳", "远征", "时差"])
    motivation_clear = has_any(["争冠", "保级", "争四", "欧战资格", "升级", "晋级", "决赛", "半决赛", "必须", "主场告别", "战意明确", "motivation"])
    motivation_unclear = has_any(["无欲无求", "锁定", "提前", "战意不足", "排名已定"]) or not motivation_clear
    weather_risk = has_any(["大雨", "暴雨", "大风", "雪", "高温", "低温", "人工草", "草皮", "湿滑", "weather", "pitch"])
    referee_risk = has_any(["红牌", "黄牌", "点球", "裁判", "严哨", "referee", "penalty"])
    market_confirmed = bool(money.get("sharp_clear")) or has_any(["升盘", "降水", "盘口支持", "资金支持", "sharp", "steam"])
    market_conflict = bool(money.get("conflict") or money.get("reverse")) or has_any(["资金分歧", "盘口反向", "反向资金", "退盘", "热门不升盘", "热度过高但不升盘"])
    web_sources = web.get("sources", []) if isinstance(web, dict) else []
    web_source_count = len(web_sources) if isinstance(web_sources, list) else 0

    score = 100
    penalties = []
    if lineup_unclear:
        score -= 22; penalties.append("lineup_unclear")
    if rotation_risk:
        score -= 16; penalties.append("rotation_risk")
    if fatigue_risk:
        score -= 10; penalties.append("fatigue_or_travel")
    if critical_absence:
        score -= 12; penalties.append("critical_absence")
    if motivation_unclear:
        score -= 10; penalties.append("motivation_unclear")
    if market_conflict:
        score -= 22; penalties.append("market_conflict")
    elif not market_confirmed:
        score -= 12; penalties.append("market_unconfirmed")
    if web_source_count <= 0:
        score -= 14; penalties.append("no_web_sources")
    if weather_risk:
        score -= 6; penalties.append("weather_pitch_risk")
    if referee_risk:
        score -= 4; penalties.append("referee_risk")

    return {
        "version": "prematch_structured_v1",
        "lineup_confirmed": lineup_confirmed,
        "lineup_predicted": lineup_predicted,
        "lineup_unclear": lineup_unclear,
        "rotation_risk": rotation_risk,
        "fatigue_risk": fatigue_risk,
        "motivation_clear": motivation_clear,
        "motivation_unclear": motivation_unclear,
        "critical_absence": critical_absence,
        "weather_risk": weather_risk,
        "referee_risk": referee_risk,
        "market_confirmed": market_confirmed,
        "market_conflict": market_conflict,
        "web_source_count": web_source_count,
        "structured_quality_score": max(0, min(100, score)),
        "penalties": penalties,
        "injury_summary": {"home": h_inj[:300], "away": a_inj[:300]},
    }


def _structured_confirmation_bonus(factors: Dict[str, Any], final_dir: str) -> int:
    bonus = 0
    if factors.get("lineup_confirmed"):
        bonus += 18
    if factors.get("market_confirmed"):
        bonus += 16
    if factors.get("motivation_clear"):
        bonus += 10
    if factors.get("web_source_count", 0) >= 2:
        bonus += 8
    if final_dir == "away" and not factors.get("market_confirmed"):
        bonus -= 20
    return bonus
def _context_text_for_gate(pred: Dict[str, Any], match_obj: Dict[str, Any]) -> str:
    return _json_compact({
        "league": match_obj.get("league"),
        "match_num": match_obj.get("match_num"),
        "intelligence": match_obj.get("intelligence"),
        "information": match_obj.get("information"),
        "expert_intro": match_obj.get("expert_intro"),
        "baseface": match_obj.get("baseface"),
        "contextual_logic": pred.get("contextual_logic"),
        "reason": pred.get("ai_native_reason", pred.get("reason", "")),
    }, 6000).lower()


def _match_context_flags(pred: Dict[str, Any], match_obj: Dict[str, Any], dna: Dict[str, Any]) -> Dict[str, Any]:
    league = str(match_obj.get("league", ""))
    text = _context_text_for_gate(pred, match_obj)
    cup_like = dna.get("key") == "cup_or_cross_context" or any(x in league for x in ["杯", "亚冠", "欧冠", "欧联", "世俱杯", "Cup"])
    cross_region = any(x in league for x in ["亚冠", "世俱杯", "解放者", "南球杯"]) or any(t in text for t in ["跨洲", "中立场", "neutral", "远征", "长途", "旅行", "时差"])
    rotation_risk = any(t in text for t in ["轮休", "轮换", "替补", "二队", "青年", "休息", "rotation", "rotate", "rested"])
    importance_unclear = cup_like and not any(t in text for t in ["争冠", "保级", "必须", "晋级", "决赛", "半决赛", "主场告别", "战意", "motivation"])
    name_favorite = False
    sp_h = _f(match_obj.get("sp_home"), 0)
    sp_a = _f(match_obj.get("sp_away"), 0)
    if sp_h > 1.01 and sp_a > 1.01:
        name_favorite = min(sp_h, sp_a) <= 1.75
    return {
        "cup_like": cup_like,
        "cross_region": cross_region,
        "rotation_risk": rotation_risk,
        "importance_unclear": importance_unclear,
        "name_favorite": name_favorite,
    }


def _weak_home_win_context(pred: Dict[str, Any], match_obj: Dict[str, Any], draw_prob: float, edge_gap: float, draw_cluster: bool) -> bool:
    if pred.get("final_direction") != "home":
        return False
    score = _score_from_candidate(pred.get("predicted_score"))
    total = _score_total(score)
    home_pct = _f(pred.get("home_win_pct"), 0)
    sp_h = _f(match_obj.get("sp_home"), 0)
    return bool(
        (home_pct and home_pct <= 47.0)
        or edge_gap <= 9.0
        or (draw_prob >= 27.0 and draw_cluster)
        or score in {"1-0", "2-1"}
        or (total is not None and total <= 2 and 1.70 <= sp_h <= 2.35)
    )


def assign_selection_layer(pred: Dict[str, Any], match_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Assign user-facing selection layer after gates; never changes score/direction."""
    if not isinstance(pred, dict) or pred.get("is_abstain"):
        return pred
    rec = pred.get("recommendation", {}) if isinstance(pred.get("recommendation"), dict) else {}
    audit = pred.get("pre_match_factor_audit", {}) if isinstance(pred.get("pre_match_factor_audit"), dict) else {}
    rules = set(str(x) for x in audit.get("rules_applied", []) if str(x))
    tier = str(rec.get("tier", pred.get("recommendation_tier", "D"))).upper()
    final_dir = str(pred.get("final_direction", ""))
    conf = int(_clip(_f(rec.get("bet_confidence", pred.get("confidence", 0)), 0), 0, 100))
    draw_prob = _f(audit.get("draw_probability_watch", pred.get("draw_pct", 0)), 0)
    edge_gap = _f(audit.get("edge_gap", 0), 0)
    dq = _f(audit.get("data_quality_score", 0), 0)
    high_tail = _f(audit.get("high_goal_tail_pct", 0), 0)
    structured_factors = audit.get("structured_factors", {}) if isinstance(audit.get("structured_factors"), dict) else {}
    confirmation_bonus = _structured_confirmation_bonus(structured_factors, final_dir)
    gate_pass = bool(pred.get("recommend_gate_pass"))

    hard_no_bet_rules = {
        "prematch_v2_money_flow_conflict_no_bet",
        "prematch_v2_strong_draw_cluster_no_bet",
        "prematch_v2_structured_market_conflict_no_bet",
    }
    away_unconfirmed = "prematch_v2_away_win_without_external_market_confirmation" in rules
    cup_context = any(r in rules for r in [
        "prematch_v2_cup_cross_context_lineup_motivation_required",
        "prematch_v2_cross_region_requires_external_confirmation",
    ])
    weak_home_draw = "prematch_v2_weak_home_win_draw_guard" in rules
    draw_defense = "prematch_v2_draw_defense_gate" in rules or "prematch_v2_high_draw_league_non_draw_cap" in rules
    high_vol = "prematch_v2_high_volatility_league_requires_confirmation" in rules

    layer = "观察"
    stake = 0.0
    hedge = []
    reasons = []

    if hard_no_bet_rules & rules:
        layer, stake = "放弃", 0.0
        reasons.append("硬闸门触发")
    elif pred.get("recommendation_tier") == "D" and final_dir == "away" and confirmation_bonus >= 24 and away_unconfirmed and not (cup_context or high_vol):
        layer, stake = "观察", 0.0
        hedge.append("临场确认后客不败")
        reasons.append("客胜缺确认被硬降D，但结构化信息有一定支持，保留观察")
    elif pred.get("recommendation_tier") == "D":
        layer, stake = "放弃", 0.0
        reasons.append("等级为D")
    elif gate_pass and tier in {"S", "A"} and (conf + confirmation_bonus) >= 72 and dq >= 55 and not rules:
        layer, stake = "主推", 1.0
        reasons.append("推荐闸门通过且结构化因子确认充分")
    elif gate_pass and tier in {"A", "B"} and (conf + confirmation_bonus) >= 64 and not (away_unconfirmed or cup_context or weak_home_draw):
        layer, stake = "小注", 0.5
        reasons.append("推荐闸门通过但存在轻中度风险")
    elif (not gate_pass) and tier == "C" and final_dir == "away" and confirmation_bonus >= 22 and not (cup_context or hard_no_bet_rules & rules):
        layer, stake = "小注", 0.25
        hedge.append("客不败/临场水位确认")
        reasons.append("客胜原被降级，但结构化首发/市场/web确认足够，恢复为小注观察")
    elif (not gate_pass) and tier == "D" and final_dir == "away" and confirmation_bonus >= 24 and away_unconfirmed and not (cup_context or hard_no_bet_rules & rules or high_vol):
        layer, stake = "观察", 0.0
        hedge.append("临场确认后客不败")
        reasons.append("客胜缺确认被硬降D，但结构化信息有一定支持，保留观察")
    elif weak_home_draw or (draw_defense and final_dir != "draw"):
        layer, stake = "防平", 0.25
        hedge.append("平局")
        if draw_prob >= 29 or edge_gap <= 8:
            hedge.append("双选/不败优先")
        reasons.append("平局防线触发，胜负方向只能防平观察")
    elif away_unconfirmed and final_dir == "away":
        layer, stake = "观察", 0.0
        hedge.append("临场确认后小注客队/客不败")
        reasons.append("客胜缺少外部市场或sharp确认")
    elif cup_context:
        layer, stake = "观察", 0.0
        hedge.append("等待官方首发/战意/临场盘口")
        reasons.append("杯赛或跨洲场景需要临场确认")
    elif high_vol or high_tail >= 45:
        layer, stake = "观察", 0.0
        hedge.append("比分簇替代单点")
        reasons.append("高波动/高比分尾部，不适合单点重注")
    elif tier == "C":
        layer, stake = "观察", 0.0
        reasons.append("C级仅观察，不进入投注池")
    else:
        layer, stake = "观察", 0.0
        reasons.append("默认观察")

    if layer == "放弃":
        stake = 0.0
    pred["selection_layer"] = layer
    pred["selection_stake_unit"] = stake
    pred["selection_hedge_suggestions"] = list(dict.fromkeys(hedge))[:6]
    pred["selection_layer_reasons"] = list(dict.fromkeys(reasons))[:8]
    pred["selection_confirmation_bonus"] = confirmation_bonus
    # Backward-compatible Chinese summary for frontends that prefer one field.
    pred["final_action"] = layer
    return pred

def apply_pre_match_factor_v2_gate(pred: Dict[str, Any], match_obj: Dict[str, Any]) -> Dict[str, Any]:
    """综合赛前因子风控闸门：只降推荐/补风险标签，不改比分和方向。"""
    if not isinstance(pred, dict) or pred.get("is_abstain"):
        return pred
    protected = {k: pred.get(k) for k in ["predicted_score", "final_direction", "result", "display_direction", "home_win_pct", "draw_pct", "away_win_pct"]}
    rec = pred.setdefault("recommendation", {}) if isinstance(pred.get("recommendation"), dict) else {}
    if not isinstance(pred.get("recommendation"), dict):
        pred["recommendation"] = rec

    league = str(match_obj.get("league", ""))
    dna = _league_dna_profile(league)
    context_flags = _match_context_flags(pred, match_obj, dna)
    structured_factors = extract_structured_prematch_factors(pred, match_obj)
    web_used, lineup_confirmed = _has_confirmed_web_or_lineup(pred, match_obj)
    lineup_confirmed = bool(lineup_confirmed or structured_factors.get("lineup_confirmed"))
    mf = _money_flow_state(pred)
    probs = _extract_prob_map_0_100(pred.get("direction_probs")) or {
        "home": _f(pred.get("home_win_pct"), 0), "draw": _f(pred.get("draw_pct"), 0), "away": _f(pred.get("away_win_pct"), 0)
    }
    final_dir = str(pred.get("final_direction", ""))
    final_prob = _f(probs.get(final_dir), 0)
    draw_prob = max(_f(probs.get("draw"), 0), _f((pred.get("matrix_direction_probs") or {}).get("draw"), 0))
    away_prob = max(_f(probs.get("away"), 0), _f((pred.get("matrix_direction_probs") or {}).get("away"), 0))
    home_prob = max(_f(probs.get("home"), 0), _f((pred.get("matrix_direction_probs") or {}).get("home"), 0))
    sorted_probs = sorted([_f(v, 0) for k, v in probs.items() if k in VALID_DIRS], reverse=True)
    edge_gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else 0
    high_tail = sum(_f(v, 0.0) for k, v in (pred.get("matrix_goal_probs") or {}).items() if _i(k, 0) >= 4)
    draw_cluster = _draw_cluster_present(pred, match_obj)

    data_quality_score = 100
    if not web_used:
        data_quality_score -= 28
    if not lineup_confirmed:
        data_quality_score -= 22
    if not mf["sharp_clear"]:
        data_quality_score -= 18
    if not match_obj.get("change"):
        data_quality_score -= 8
    structured_score = _f(structured_factors.get("structured_quality_score"), data_quality_score)
    data_quality_score = round((max(0, min(100, data_quality_score)) * 0.55) + (structured_score * 0.45), 1)

    audit = {
        "version": PREMATCH_V2_VERSION,
        "league_dna": dna,
        "match_context_flags": context_flags,
        "structured_factors": structured_factors,
        "web_used_with_sources": web_used,
        "lineup_confirmed": lineup_confirmed,
        "money_flow": mf,
        "data_quality_score": data_quality_score,
        "edge_gap": round(edge_gap, 2),
        "draw_probability_watch": round(draw_prob, 2),
        "draw_cluster_present": draw_cluster,
        "high_goal_tail_pct": round(high_tail, 2),
        "rules_applied": [],
    }

    def apply(max_tier: str, reason: str, tag: str) -> None:
        audit["rules_applied"].append(reason)
        audit.setdefault("risk_hints", []).append({"max_tier": max_tier, "reason": reason, "tag": tag})
        # v20.6: prematch 本地层只记录 risk hints，避免成为第四个足球裁判。
        # 硬降级交给 AI 终审 prompt 的 bet_action / recommendation，或前端展示 final_action 风险。

    # P0 结构化因子硬风险：轮换/战意/关键伤停/市场冲突必须进入推荐层。
    if structured_factors.get("market_conflict"):
        apply("D", "prematch_v2_structured_market_conflict_no_bet", "prematch_v2_structured_market_conflict")
    if structured_factors.get("critical_absence") and not lineup_confirmed:
        apply("C", "prematch_v2_critical_absence_without_lineup_confirmation", "prematch_v2_critical_absence")
    if structured_factors.get("rotation_risk") and not lineup_confirmed:
        apply("C", "prematch_v2_rotation_risk_requires_lineup", "prematch_v2_rotation_risk")
    if structured_factors.get("fatigue_risk") and final_dir == "away":
        apply("C", "prematch_v2_away_fatigue_travel_risk", "prematch_v2_away_fatigue")

    # P0 数据质量：无外部来源/无首发时，不能把上下文判断包装成高置信精选。
    data_quality_hard_context = final_dir == "away" or (final_dir != "draw" and bool(draw_cluster)) or dna.get("volatility") == "high" or high_tail >= 45.0
    if data_quality_score < 35 and data_quality_hard_context and str(rec.get("tier", "D")).upper() in {"S", "A"}:
        apply("B", "prematch_v2_data_quality_below_35_no_high_grade", "prematch_v2_low_data_quality")
    elif data_quality_score < 55 and str(rec.get("tier", "D")).upper() in {"S", "A"} and data_quality_hard_context:
        apply("B", "prematch_v2_data_quality_below_55_caps_to_B", "prematch_v2_medium_data_quality")
    elif (not web_used or not lineup_confirmed) and str(rec.get("tier", "D")).upper() in {"S", "A"} and data_quality_hard_context:
        apply("B", "prematch_v2_missing_web_or_lineup_caps_to_B", "prematch_v2_missing_web_or_lineup")

    # P0 平局防线：弱优势/平赔簇/矩阵平局高时，胜负方向不进强推。
    if final_dir != "draw" and (draw_prob >= 31.0 or (draw_cluster and edge_gap <= 10.0) or edge_gap <= 6.0):
        apply("C", "prematch_v2_draw_defense_gate", "prematch_v2_draw_defense")
        if draw_prob >= 34.0 or (draw_cluster and edge_gap <= 5.0):
            apply("D", "prematch_v2_strong_draw_cluster_no_bet", "prematch_v2_strong_draw_cluster")

    # P0 弱主胜防平：主胜小比分/弱边际/高平局联赛，不能直接进主推。
    if _weak_home_win_context(pred, match_obj, draw_prob, edge_gap, draw_cluster):
        if dna.get("draw_risk") == "high" or draw_prob >= 30.0 or not web_used:
            apply("C", "prematch_v2_weak_home_win_draw_guard", "prematch_v2_weak_home_win_draw_guard")
        else:
            apply("B", "prematch_v2_weak_home_win_needs_confirmation", "prematch_v2_weak_home_win")

    # P0 杯赛/跨洲/名气强队：未确认首发和战意时，不追名气强队。
    if context_flags.get("cup_like") or context_flags.get("cross_region"):
        if context_flags.get("rotation_risk") or context_flags.get("importance_unclear") or not lineup_confirmed:
            if context_flags.get("name_favorite") or final_dir in {"home", "away"}:
                apply("C", "prematch_v2_cup_cross_context_lineup_motivation_required", "prematch_v2_cup_cross_context")
        if context_flags.get("cross_region") and not web_used:
            apply("C", "prematch_v2_cross_region_requires_external_confirmation", "prematch_v2_cross_region")

    # P0 客胜二次审核：无sharp/无web/高波动联赛下的客胜不作主推。
    if final_dir == "away":
        if dna.get("away_penalty", 0) >= 2 or (not web_used and not mf["sharp_clear"]):
            apply("D", "prematch_v2_away_win_without_external_market_confirmation", "prematch_v2_away_win_second_review")
        elif dna.get("away_penalty", 0) >= 1 or final_prob <= 46 or away_prob <= home_prob + 6:
            apply("C", "prematch_v2_away_win_context_risk", "prematch_v2_away_win_context_risk")

    # 联赛DNA：德甲/美职/北欧等高波动，尤其末段/无web时降级。
    if dna.get("volatility") == "high" and (not web_used or high_tail >= 34.0):
        if final_dir == "away":
            apply("C", "prematch_v2_high_volatility_league_requires_confirmation", "prematch_v2_high_volatility_league")
        else:
            apply("B", "prematch_v2_high_volatility_league_requires_confirmation", "prematch_v2_high_volatility_league")
    if dna.get("draw_risk") == "high" and final_dir != "draw" and draw_prob >= 27.0:
        apply("C", "prematch_v2_high_draw_league_non_draw_cap", "prematch_v2_high_draw_league")

    # 资金流：sharp反向/反向盘口未解释，直接放弃；sharp不明则限制高等级。
    if mf["conflict"] or mf["reverse"]:
        apply("D", "prematch_v2_money_flow_conflict_no_bet", "prematch_v2_money_flow_conflict")
    elif not mf["sharp_clear"] and str(rec.get("tier", "D")).upper() in {"S", "A"}:
        apply("B", "prematch_v2_sharp_unclear_no_A_or_S", "prematch_v2_sharp_unclear")

    # high_btts_tail / 高比分尾部：不是自动反向，但会压低单点胜负信心。
    if high_tail >= 45.0 and final_dir != "draw":
        if final_dir == "away" or edge_gap <= 8.0:
            apply("C", "prematch_v2_high_goal_tail_direction_instability", "prematch_v2_high_goal_tail")
        else:
            apply("B", "prematch_v2_high_goal_tail_direction_instability", "prematch_v2_high_goal_tail")

    pred["pre_match_factor_audit"] = audit
    pred.setdefault("validation_warnings", []).extend([f"prematch_v2:{r}" for r in audit["rules_applied"]])
    pred["validation_warnings"] = list(dict.fromkeys(str(x) for x in pred.get("validation_warnings", []) if str(x).strip()))[:80]
    pred["recommendation_downgrade_reasons"] = list(dict.fromkeys(str(x) for x in pred.get("recommendation_downgrade_reasons", []) if str(x).strip()))[:80]
    pred["recommend_gate_reasons"] = list(dict.fromkeys(str(x) for x in pred.get("recommend_gate_reasons", []) if str(x).strip()))[:80]

    rec = pred.get("recommendation", {}) if isinstance(pred.get("recommendation"), dict) else {}
    tier = str(rec.get("tier", pred.get("recommendation_tier", "D"))).upper()
    pred["recommendation_tier"] = tier
    pred["recommend_gate_pass"] = bool(rec.get("is_recommended", False)) and _min_tier_ok(tier)
    pred["direction_tier"] = tier
    pred["score_tier"] = tier
    if audit["rules_applied"] and not pred.get("recommend_gate_pass"):
        pred.setdefault("recommend_gate_reasons", []).append("prematch_v2_risk_hints_present")
    for key, value in protected.items():
        pred[key] = value
    return pred

def adapt_ai_to_frontend(ai_r: Dict[str, Any], match_obj: Dict[str, Any]) -> Dict[str, Any]:
    apply_weak_home_tail_risk_protection(ai_r)
    pred = _BASE_ADAPT_AI_TO_FRONTEND_V2021(ai_r, match_obj)
    if not isinstance(pred, dict) or pred.get("is_abstain"):
        return pred
    raw_item = ai_r.get("raw_item", {}) if isinstance(ai_r.get("raw_item"), dict) else {}
    for k in [
        "score_cluster_audit", "sharp_money_audit", "recommendation_components", "risk_score_candidates",
        "tail_risk_flags", "confidence_downgrade_reason", "market_audit",
        "goal_market_audit", "market_conflicts", "candidate_scores", "public_heat_audit",
        "packet_news_risk_audit", "trap_candidates", "final_score_audit",
    ]:
        v = ai_r.get(k, None)
        if v in (None, {}, []):
            v = raw_item.get(k, None)
        if v not in (None, {}, []):
            pred[k] = v
    # 本地只做推荐风控展示/降级，不改方向和比分。
    try:
        protocol_downgrade_recommendation_only(pred)
        rec = pred.get("recommendation", {}) if isinstance(pred.get("recommendation"), dict) else {}
        tier = str(rec.get("tier", "D")).upper()
        pred["recommendation_tier"] = tier
        pred["recommend_gate_pass"] = bool(rec.get("is_recommended", False)) and _min_tier_ok(tier)
        pred["recommend_gate_reasons"] = [] if pred["recommend_gate_pass"] else ["ai_not_recommended_or_below_min_tier_or_v203_component_downgrade"]
        pred["direction_tier"] = tier
        pred["score_tier"] = tier
    except Exception as e:
        pred.setdefault("validation_warnings", []).append(f"v203_recommendation_component_downgrade_error:{str(e)[:120]}")
    try:
        attach_matrix_shadow_fields(pred, match_obj)
    except Exception as e:
        pred["matrix_shadow_error"] = str(e)[:300]
    try:
        apply_two_one_home_hard_no_bet_gate(pred)
        rec = pred.get("recommendation", {}) if isinstance(pred.get("recommendation"), dict) else {}
        tier = str(rec.get("tier", "D")).upper()
        pred["recommendation_tier"] = tier
        pred["recommend_gate_pass"] = bool(rec.get("is_recommended", False)) and _min_tier_ok(tier)
        if not pred["recommend_gate_pass"]:
            pred.setdefault("recommend_gate_reasons", []).append("ai_not_recommended_or_two_one_tail_gate")
            pred["recommend_gate_reasons"] = list(dict.fromkeys(pred.get("recommend_gate_reasons", [])))
        pred["direction_tier"] = tier
        pred["score_tier"] = tier
    except Exception as e:
        pred.setdefault("validation_warnings", []).append(f"two_one_hard_gate_error:{str(e)[:120]}")
    try:
        apply_pre_match_factor_v2_gate(pred, match_obj)
    except Exception as e:
        pred.setdefault("validation_warnings", []).append(f"prematch_v2_gate_error:{str(e)[:120]}")
    try:
        assign_selection_layer(pred, match_obj)
    except Exception as e:
        pred.setdefault("validation_warnings", []).append(f"selection_layer_error:{str(e)[:120]}")
    pred["engine_version"] = ENGINE_VERSION
    pred["engine_architecture"] = ENGINE_ARCHITECTURE
    return pred


if __name__ == "__main__":
    logger.info(f"{ENGINE_VERSION} 启动")
    print(f"✅ {ENGINE_VERSION} 加载完成")
    print(f"   架构: {ENGINE_ARCHITECTURE}")
    print("   模式: 完整版；保留AI模块调用链；新增Sharp/CRS簇/HHAD/相邻比分审计；本地不改最终足球判断。")


def apply_deep_favorite_score_moderation(final_direction, predicted_score, probabilities, score_candidates, total_goal_probs, logic_reasoning):
    '''
    Moderates 3-0 or 0-3 predictions to 2-0 or 0-2 for deep favorites if the goal band or candidate proximity doesn't strongly support a blowout.
    '''
    if final_direction not in ["home", "away"]:
        return False, predicted_score, []

    is_home_fav = (final_direction == "home" and probabilities.get("home", 0) >= 0.58 and (probabilities.get("draw", 0) + probabilities.get("away", 0)) <= 0.42)
    is_away_fav = (final_direction == "away" and probabilities.get("away", 0) >= 0.58 and (probabilities.get("draw", 0) + probabilities.get("home", 0)) <= 0.42)

    if not (is_home_fav or is_away_fav):
        return False, predicted_score, []

    candidates_dict = {sc["score"]: sc.get("probability", 0) for sc in score_candidates} if isinstance(score_candidates, list) else {}

    moderated = False
    new_score = predicted_score
    cluster = []

    if predicted_score == "3-0" and is_home_fav:
        prob_3_0 = candidates_dict.get("3-0", 0)
        prob_2_0 = candidates_dict.get("2-0", 0)
        prob_tg_3 = total_goal_probs.get(3, 0)
        prob_tg_4 = total_goal_probs.get(4, 0)
        prob_tg_2 = total_goal_probs.get(2, 0)
        
        if abs(prob_3_0 - prob_2_0) <= 0.05 and prob_tg_4 < max(prob_tg_2, prob_tg_3):
            moderated = True
            new_score = "2-0"
            cluster = ["2-0", "3-0", "3-1"]

    elif predicted_score == "0-3" and is_away_fav:
        prob_0_3 = candidates_dict.get("0-3", 0)
        prob_0_2 = candidates_dict.get("0-2", 0)
        prob_tg_3 = total_goal_probs.get(3, 0)
        prob_tg_4 = total_goal_probs.get(4, 0)
        prob_tg_2 = total_goal_probs.get(2, 0)
        
        if abs(prob_0_3 - prob_0_2) <= 0.05 and prob_tg_4 < max(prob_tg_2, prob_tg_3):
            moderated = True
            new_score = "0-2"
            cluster = ["0-2", "0-3", "1-3"]

    return moderated, new_score, cluster
