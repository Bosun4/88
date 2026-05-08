# -*- coding: utf-8 -*-
"""
vMAX 20.1 — AI-Native Web-Augmented Multi-Agent Adjudication Engine
=====================================================================

设计边界：
1. 本地不做足球预测判断：不判断 Sharp 真伪、不判断比分是否应该推荐、不用经验规则改方向/比分。
2. 本地只做协议层：抓包整理、Evidence 编译、动态分批、AI 调用、JSON 解析、Schema 校验、落盘、前端字段兼容。
3. 四 AI 分工：
   - GPT：市场结构 + 外部赔率/盘口对照。
   - Grok：资金流/热度/临场新闻。
   - Gemini：战术/赛程/风格一致性。
   - Claude：最终 Web-aware 仲裁。
4. 支持 AI 原生联网：Prompt 强制输出 web_research / sources / freshness / source_conflicts。
5. 支持 AI 互审：Phase1 三家初审后，可进入 critic 阶段，互相指出漏洞。
6. 支持一致性审计：Final Judge 只检查 JSON、方向/比分/goal_band/BTTS/source 格式，不做足球判断。
7. Top4 / 推荐等级由 AI 输出；本地只根据 AI 的 recommendation 字段排序。
8. Claude 失败时可启用 AI fallback referee；若仍失败才使用 Phase1 AI 共识兜底。
9. 提供 AI_MOCK_MODE，可在无真实 API 时做闭环测试。Mock 只用于工程可行性，不代表真实命中率。

入口：
    run_predictions(raw, use_ai=True) -> (res, top4)

环境变量要点：
    API_URL / API_KEY 或各模型单独 *_API_URL / *_API_KEY
    AI_MOCK_MODE=true       # 沙盒/本地闭环测试
    AI_CHUNK_SIZE=8         # 小批次
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
from dataclasses import dataclass
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

ENGINE_VERSION = "vMAX 20.1.2"
ENGINE_ARCHITECTURE = (
    "AI-NATIVE WEB-AUGMENTED: 本地只做协议层；GPT/Grok/Gemini 分工初审 + AI互审 + "
    "Claude Web-aware 终审 + 一致性审计；Top4/推荐等级由 AI 输出；本地不做 Sharp/比分/推荐裁决。"
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

HFTF_MAP = {
    "ss": "主/主", "sp": "主/平", "sf": "主/负",
    "ps": "平/主", "pp": "平/平", "pf": "平/负",
    "fs": "负/主", "fp": "负/平", "ff": "负/负",
}

PROMPT_STRIP_KEYS = {
    "prediction", "predictions", "top4", "rank", "recommend_score",
    "predicted_score", "predicted_label", "result", "display_direction", "final_direction",
    "raw_ai_direction", "score_implied_direction", "home_win_pct", "draw_pct", "away_win_pct",
    "confidence", "is_recommended", "is_strict_recommended", "is_top4_candidate",
    "gpt_score", "gpt_analysis", "grok_score", "grok_analysis", "gemini_score", "gemini_analysis",
    "claude_score", "claude_analysis", "bayesian_evidences", "score_market_evidence", "sharp_audit",
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

# 成本模式：production 默认用于日常全量跑；enhanced 用于精选复核；research 才是满血互审。
# 关键原则：默认不能再用高成本 research 配置，避免 40+ 场批量时成本爆炸。
AI_RESEARCH_MODE = str(os.environ.get("AI_RESEARCH_MODE", "production")).strip().lower()
if AI_RESEARCH_MODE not in {"production", "enhanced", "research"}:
    AI_RESEARCH_MODE = "production"

if AI_RESEARCH_MODE == "production":
    _default_native_web = False
    _default_cross_exam = False
    _default_consistency = False
    _default_chunk_size = 10
elif AI_RESEARCH_MODE == "enhanced":
    _default_native_web = False  # 建议只在最终候选/冲突场通过环境变量打开。
    _default_cross_exam = False
    _default_consistency = True
    _default_chunk_size = 8
else:  # research
    _default_native_web = True
    _default_cross_exam = True
    _default_consistency = True
    _default_chunk_size = 6

AI_NATIVE_WEB = _env_bool("AI_NATIVE_WEB", _default_native_web)
AI_REQUIRE_WEB_SOURCES = _env_bool("AI_REQUIRE_WEB_SOURCES", AI_NATIVE_WEB)
AI_WARN_MISSING_PUBLISHED_AT = _env_bool("AI_WARN_MISSING_PUBLISHED_AT", False)
AI_WEB_MAX_SOURCES_PER_MATCH = max(0, _env_int("AI_WEB_MAX_SOURCES_PER_MATCH", 8))
AI_CHUNK_SIZE = max(1, _env_int("AI_CHUNK_SIZE", _default_chunk_size))
AI_MAX_PROMPT_CHARS_PER_CHUNK = max(30000, _env_int("AI_MAX_PROMPT_CHARS_PER_CHUNK", 140000))
AI_ENABLE_CROSS_EXAM = _env_bool("AI_ENABLE_CROSS_EXAM", _default_cross_exam)
AI_ENABLE_CONSISTENCY_JUDGE = _env_bool("AI_ENABLE_CONSISTENCY_JUDGE", _default_consistency)
AI_ENABLE_FALLBACK_REFEREE = _env_bool("AI_ENABLE_FALLBACK_REFEREE", True)
AI_CONSISTENCY_JUDGE_MODEL = str(os.environ.get("AI_CONSISTENCY_JUDGE_MODEL", "gpt")).strip().lower()
AI_FALLBACK_REFEREE_MODEL = str(os.environ.get("AI_FALLBACK_REFEREE_MODEL", "gpt")).strip().lower()

AI_READ_TIMEOUT = _env_int("AI_READ_TIMEOUT", 5400)
AI_CLAUDE_READ_TIMEOUT = _env_int("AI_CLAUDE_READ_TIMEOUT", 7200)
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


def build_evidence_packet(match_obj: Dict[str, Any], index: int) -> Dict[str, Any]:
    m = _strip_output_fields(match_obj)

    style_keys = [
        "league_style", "league_profile", "team_style", "home_style", "away_style", "play_style",
        "tactical_style", "pace_rating", "tempo", "home_form", "away_form", "weather", "injury",
        "lineup", "news", "motivation", "schedule", "home_rank", "away_rank", "home_stats", "away_stats",
    ]

    correct_score_odds = {
        sc: m.get(key)
        for sc, key in CRS_FULL_MAP.items()
        if m.get(key) not in (None, "", 0, "0")
    }

    total_goals_odds = {
        str(i): m.get(f"a{i}")
        for i in range(8)
        if m.get(f"a{i}") not in (None, "", 0, "0")
    }

    evidence = {
        "match": index,
        "identity": {
            "home_team": m.get("home_team", m.get("home", "Home")),
            "away_team": m.get("away_team", m.get("guest", "Away")),
            "league": m.get("league", m.get("cup", "")),
            "match_num": m.get("match_num", ""),
            "match_time": m.get("match_time", m.get("time", "")),
        },
        "lottery_market_1x2": {
            "home": m.get("sp_home", m.get("win")),
            "draw": m.get("sp_draw", m.get("same")),
            "away": m.get("sp_away", m.get("lose")),
            "note": "中国体彩竞彩抓包赔率，不是欧洲均赔。",
        },
        "handicap": {"raw": m.get("give_ball", m.get("handicap", m.get("rq", "")))},
        "movement": {
            "change": m.get("change", {}),
            "change_is_direction_code": bool(m.get("change_is_direction_code", False)),
            "coding_note": "change_is_direction_code=true 时，-1=降水/下降，0=不变，1=升水/上升。",
            "odds_movement": m.get("odds_movement", {}),
        },
        "public_vote": m.get("vote", {}),
        "total_goals_odds": total_goals_odds,
        "correct_score_odds": correct_score_odds,
        "half_full_time_odds": {
            HFTF_MAP[k]: m.get(k)
            for k in HFTF_MAP
            if m.get(k) not in (None, "", 0, "0")
        },
        "derived_market_facts_no_judgement": {
            "score_market_facts": _score_market_facts(m),
            "total_goal_market_facts": _total_goal_market_facts(m),
            "half_full_market_facts": _half_full_facts(m),
        },
        "context_raw_fields": {
            "information": m.get("information", ""),
            "points": m.get("points", ""),
            "baseface": m.get("baseface", ""),
            "expert_intro": m.get("expert_intro", ""),
            "intelligence": m.get("intelligence", ""),
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
    "why_recommended":"中文说明"
  },
  "data_quality": {"missing":[], "raw_packet_quality":"high/medium/low"},
  "reason":"中文综合理由"
}
硬约束：predicted_score 暗示的方向必须等于 final_direction；goal_band 与 predicted_score 总进球一致；btts 与 predicted_score 一致；top3[0].score 必须等于 predicted_score。
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
        "gemini": "你是 Tactical & Contextual Consistency Analyst，专攻战术风格、赛程、战意、伤停、比分形态。",
        "claude": "你是 Final Web-aware Adjudicator，负责源质量审计、交叉证据仲裁和最终推荐。",
    }.get(ai_name, "你是足球量化 RAW-AI 分析师。")
    return (
        role_intro
        + "只能输出严格 JSON object，顶层必须是 predictions。所有解释字段必须中文。"
        + "禁止引用本地模型结论；允许使用原始体彩正确比分赔率 correct_score_odds。"
        + "不要机械投票；必须给出 rejected_cases，说明为什么不选其他方向。"
    )


def build_phase1_prompt(evidence_batch: List[Dict[str, Any]], ai_name: str) -> str:
    n = len(evidence_batch)
    p = []
    p.append("<task>")
    p.append(f"本批次共有 {n} 场，match 编号必须完整覆盖：" + ",".join(str(e["match"]) for e in evidence_batch))
    p.append("你看到的是中国体彩竞彩抓包 Evidence，不是本地预测结论。本地没有 Sharp/比分/推荐裁决。")
    p.append(_web_research_instruction(ai_name))
    p.append("你的任务不是给空泛推荐，而是基于市场结构、资金流、比分赔率、总进球、半全场、联网情报形成可审计预测。")
    p.append("</task>\n")
    p.append("<output_schema>")
    p.append(_canonical_output_schema_text())
    p.append("</output_schema>\n")
    p.append("<evidence_batch>")
    for e in evidence_batch:
        p.append(_safe_json_line(e))
    p.append("</evidence_batch>")
    return "\n".join(p)


def build_critic_prompt(evidence_batch: List[Dict[str, Any]], reviewer_name: str, phase1_results: Dict[str, Dict[int, Dict[str, Any]]]) -> str:
    p = []
    p.append("<task>")
    p.append(f"你是 {reviewer_name.upper()} Critic。你不是重新预测主裁，只负责审查其他 AI 的漏洞。")
    p.append("必须指出：赔率误读、联网来源不可靠、Sharp 解释不充分、比分与总进球/BTTS/方向冲突、推荐等级过高。")
    p.append("输出严格 JSON object，顶层格式 {\"critic_reports\":[...]}，不要 markdown。")
    p.append("每个 critic_report 格式：{\"match\":1,\"critic_model\":\"gpt\",\"target_findings\":[{\"target_model\":\"grok\",\"issue_type\":\"market/web/sharp/score/schema/risk\",\"severity\":\"low/medium/high\",\"comment\":\"中文\"}],\"own_revision_hint\":\"中文\"}")
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


def build_claude_final_prompt(
    evidence_batch: List[Dict[str, Any]],
    phase1_results: Dict[str, Dict[int, Dict[str, Any]]],
    critic_reports: Dict[str, List[Dict[str, Any]]],
) -> str:
    p = []
    p.append("<final_adjudication_protocol>")
    p.append("你是 Claude 最终 Web-aware 主裁。你必须重新审计 raw evidence、三家初审、三家互审意见和联网来源质量。")
    p.append("证据优先级：raw market structure > correct-score cluster > total-goals mode > money-flow/sharp interpretation > tactical/web context > Phase1 consensus。")
    p.append("多数意见不自动成立；若多数共享同一个弱假设，你必须覆盖。")
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


def build_fallback_referee_prompt(
    evidence_batch: List[Dict[str, Any]],
    phase1_results: Dict[str, Dict[int, Dict[str, Any]]],
    critic_reports: Dict[str, List[Dict[str, Any]]],
) -> str:
    p = []
    p.append("你是 Claude 失败后的 AI fallback referee。不要使用本地规则。基于 raw evidence、Phase1 和 critic reports 输出最终 predictions。")
    p.append("输出 schema 与 Claude final 完全一致。")
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
    p.append("检查：predicted_score方向=final_direction；goal_band与比分总进球一致；btts与比分一致；top3[0]=predicted_score；web_research.used=true时必须有sources。")
    p.append("不得根据足球观点改比分，只能修字段。")
    p.append("<evidence_batch>")
    for e in evidence_batch:
        p.append(_safe_json_line(e))
    p.append("</evidence_batch><final_predictions>")
    for idx, r in final_predictions.items():
        p.append(_safe_json_line(_short_prediction_for_prompt(r)))
    p.append("</final_predictions>")
    return "\n".join(p)


def _short_prediction_for_prompt(r: Dict[str, Any]) -> Dict[str, Any]:
    keep = {}
    for k in [
        "match", "final_direction", "predicted_score", "direction_probs", "goal_band", "btts", "top3",
        "market_interpretation", "money_flow", "contextual_logic", "rejected_cases", "recommendation",
        "data_quality", "reason", "web_research", "final_web_audit", "validation_warnings",
    ]:
        if k in r:
            keep[k] = r[k]
    return keep


# ============================================================
# API / Mock 调用
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
    return _clean_env_key(f"{ai_name.upper()}_API_KEY", "API_KEY", "OPENAI_API_KEY", "GPT_API_KEY")


def get_url_for_ai(ai_name: str) -> str:
    if AI_FORCE_COMMON_GATEWAY:
        return _clean_env_url("API_URL", "GPT_API_URL", "OPENAI_API_URL", "BASE_URL", "GROK_API_URL", "GEMINI_API_URL", "CLAUDE_API_URL")
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


async def async_call_ai_json(
    session: Optional[Any],
    ai_name: str,
    system_text: str,
    prompt: str,
    phase: str,
    expected_matches: List[int],
) -> Tuple[str, Any, Dict[str, Any]]:
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
        read_timeout = AI_CLAUDE_READ_TIMEOUT if ai_name == "claude" else AI_READ_TIMEOUT
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

    # balanced fragment fallback
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
        # protocol repair: top3[0] must equal predicted_score, not football judgement
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


def _normalize_goal_band_value(v: Any, score: str) -> str:
    s = str(v or "").strip().lower().replace(" ", "")
    aliases = {
        "0-1": "0-1", "0_1": "0-1", "0~1": "0-1", "0至1": "0-1", "0到1": "0-1", "0/1": "0-1", "low": "0-1", "小球": "0-1",
        "2": "2", "2球": "2", "two": "2",
        "3": "3", "3球": "3", "three": "3",
        "4+": "4+", "4plus": "4+", "4以上": "4+", "4球+": "4+", "high": "4+", "大球": "4+",
    }
    if s in aliases:
        return aliases[s]
    return _score_goal_band(score)


def _normalize_btts_value(v: Any, score: str) -> str:
    s = str(v or "").strip().lower()
    if s in ("yes", "y", "true", "1", "是", "双方进球", "btts_yes"):
        return "yes"
    if s in ("no", "n", "false", "0", "否", "不是", "btts_no"):
        return "no"
    return _score_btts(score)


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
        goal_band = _score_goal_band(predicted_score)
        btts = _score_btts(predicted_score)
        web = _normalize_web_research(item)
        rec = _normalize_recommendation(item)
        warnings = []
        if direction_conflict:
            warnings.append(f"dir_score_conflict_protocol_fixed:{raw_dir}->{score_dir}")
        if direction_probs.get("_synthetic_probs"):
            warnings.append("direction_probs_missing_synthetic")
        warnings.extend(web.get("validation_warnings", []))
        out[idx] = {
            "match": idx,
            "source_model": source_model,
            "source_phase": phase,
            "final_direction": final_direction,
            "raw_ai_direction": raw_dir,
            "predicted_score": predicted_score,
            "direction_probs": direction_probs,
            "goal_band": goal_band,
            "btts": btts,
            "top3": top3,
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
    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append(r)
    return out


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


def _mock_ai_response(ai_name: str, phase: str, prompt: str, expected_matches: List[int]) -> Dict[str, Any]:
    if phase == "critic":
        return {"critic_reports": [{"match": idx, "critic_model": ai_name, "target_findings": [], "own_revision_hint": "mock critic: 未发现结构性硬冲突"} for idx in expected_matches]}
    if phase == "consistency":
        return {"repairs": [{"match": idx, "valid": True, "warnings": [], "repair": {}} for idx in expected_matches]}

    evs = _extract_json_lines_from_prompt(prompt)
    if not evs:
        evs = [{"match": idx, "identity": {"home_team": "Home", "away_team": "Away", "league": ""}, "lottery_market_1x2": {}, "derived_market_facts_no_judgement": {}} for idx in expected_matches]

    preds = []
    for e in evs:
        idx = int(e.get("match"))
        one = e.get("lottery_market_1x2", {})
        sp_h, sp_d, sp_a = _f(one.get("home"), 2.3), _f(one.get("draw"), 3.2), _f(one.get("away"), 3.1)
        implied = {"home": 1 / max(sp_h, 1.01), "draw": 1 / max(sp_d, 1.01), "away": 1 / max(sp_a, 1.01)}
        # Mock 轻微区分模型，但只用于测试 schema/流程。
        if ai_name == "grok" and idx % 5 == 0:
            best = "draw"
        elif ai_name == "gemini" and idx % 7 == 0:
            best = "away" if sp_a < sp_h else "home"
        else:
            best = max(implied.items(), key=lambda kv: kv[1])[0]
        if phase in ("final", "fallback_referee"):
            # Claude mock 倾向整合市场低赔 + 防过热，仍只是可运行模拟。
            best = max(implied.items(), key=lambda kv: kv[1])[0]
        score_map = {
            "home": "2-1" if idx % 3 == 0 else "1-0",
            "draw": "1-1" if idx % 2 == 0 else "0-0",
            "away": "1-2" if idx % 3 == 0 else "0-1",
        }
        score = score_map[best]
        probs_raw = implied.copy()
        s = sum(probs_raw.values()) or 1
        probs = {k: round(v / s * 100, 1) for k, v in probs_raw.items()}
        # ensure selected dir is argmax in mock final
        if probs[best] < max(probs.values()):
            probs[best] = max(probs.values())
        h, a = _parse_score(score)
        rec_tier = "A" if phase == "final" and max(probs.values()) >= 42 else "B" if max(probs.values()) >= 37 else "C"
        preds.append({
            "match": idx,
            "final_direction": best,
            "predicted_score": score,
            "direction_probs": probs,
            "goal_band": _score_goal_band(score),
            "btts": _score_btts(score),
            "top3": [
                {"score": score, "prob": 15, "logic": f"mock {ai_name}/{phase}: 主比分与方向闭环"},
                {"score": "2-0" if best == "home" else "1-1" if best == "draw" else "0-2", "prob": 10, "logic": "mock secondary"},
            ],
            "market_interpretation": {"one_x_two": "mock: 根据低赔方向生成", "handicap": "mock", "correct_score": "mock", "total_goals": "mock", "half_full_time": "mock", "external_market": "mock"},
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

    # Phase 1
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

    # Critic stage
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

    # Claude final
    final_rows: Dict[int, Dict[str, Any]] = {}
    claude_prompt = build_claude_final_prompt(evidence_batch, phase1, critic_reports)
    _, claude_obj, claude_st = await async_call_ai_json(session, "claude", _phase1_system("claude"), claude_prompt, "final", expected)
    final_rows = normalize_ai_predictions(claude_obj, expected, "claude", "final")
    print(f"  [Chunk {chunk_id}] CLAUDE final {len(final_rows)}/{len(expected)} status={claude_st.get('status')}")

    # fallback referee if Claude incomplete
    missing = [idx for idx in expected if idx not in final_rows]
    if missing and AI_ENABLE_FALLBACK_REFEREE:
        fallback_ai = AI_FALLBACK_REFEREE_MODEL if AI_FALLBACK_REFEREE_MODEL in PHASE1_NAMES else "gpt"
        fb_prompt = build_fallback_referee_prompt([e for e in evidence_batch if e["match"] in missing], phase1, critic_reports)
        _, fb_obj, fb_st = await async_call_ai_json(session, fallback_ai, _phase1_system(fallback_ai), fb_prompt, "fallback_referee", missing)
        fb_rows = normalize_ai_predictions(fb_obj, missing, fallback_ai, "fallback_referee")
        final_rows.update(fb_rows)
        print(f"  [Chunk {chunk_id}] FALLBACK {fallback_ai.upper()} {len(fb_rows)}/{len(missing)} status={fb_st.get('status')}")

    # last fallback: phase1 AI consensus, still AI output, not local football model
    missing = [idx for idx in expected if idx not in final_rows]
    if missing:
        for idx in missing:
            final_rows[idx] = _phase1_consensus_fallback(idx, phase1)
        print(f"  [Chunk {chunk_id}] Phase1 consensus fallback filled {len(missing)}")

    # Consistency judge: protocol repairs only
    if AI_ENABLE_CONSISTENCY_JUDGE and final_rows:
        judge_ai = AI_CONSISTENCY_JUDGE_MODEL if AI_CONSISTENCY_JUDGE_MODEL in AI_NAMES else "gpt"
        judge_prompt = build_consistency_judge_prompt(evidence_batch, final_rows)
        _, judge_obj, judge_st = await async_call_ai_json(session, judge_ai, _phase1_system(judge_ai), judge_prompt, "consistency", expected)
        repairs = parse_repairs(judge_obj)
        _apply_consistency_repairs(final_rows, repairs)
        print(f"  [Chunk {chunk_id}] CONSISTENCY {judge_ai.upper()} repairs={len(repairs)} status={judge_st.get('status')}")

    # Attach phase snapshots per match so legacy frontends can show GPT/Grok/Gemini/Claude panels
    # without leaking placeholders like “见阶段快照”. This is metadata only; it does not alter football judgement.
    for idx, row in final_rows.items():
        phase1_pack = {}
        for model_name in PHASE1_NAMES:
            pr = phase1.get(model_name, {}).get(idx, {})
            if pr:
                phase1_pack[model_name] = _short_prediction_for_prompt(pr)
        row["phase1_model_outputs"] = phase1_pack
        row["critic_reports_by_model"] = {
            model_name: [cr for cr in reports if _i(cr.get("match"), 0) == idx]
            for model_name, reports in critic_reports.items()
        }

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
    out.setdefault("validation_warnings", []).append("claude_missing_used_phase1_ai_fallback")
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
        # Only protocol fields can be repaired. No new football judgement.
        allowed = {"goal_band", "btts", "top3", "web_research", "data_quality"}
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
    r["validation_warnings"] = list(dict.fromkeys(warnings))
    return r


async def run_ai_native_web(evidence_all: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    global _LAST_AI_RUN_METADATA
    # 防止同一 Python 进程多次 run_predictions 时遗留上一轮 AI 状态。
    for _n in AI_NAMES:
        AI_CALL_STATUS[_n] = {}
    AI_RESULT_FILES.clear()
    _LAST_AI_RUN_METADATA = {}

    run_id = _make_run_id(evidence_all)
    chunks = _chunk_evidence(evidence_all)
    debug_ai_config()
    print(f"  [{ENGINE_VERSION}] run_id={run_id} chunks={len(chunks)}")

    all_final: Dict[int, Dict[str, Any]] = {}
    connector = None
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
    # If a legacy frontend uses this field as the score slot, final_score is safer than a placeholder.
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

    evidences = [
        "AI-NATIVE：本地不做足球预测判断；方向、比分、Top4等级均来自 AI 输出。",
        "WEB-AUGMENTED：Prompt 要求 AI 联网并输出 sources；本地只校验来源字段完整性。",
        "LOCAL PROTOCOL ONLY：本地只修字段闭环，如 goal_band/btts 与比分一致，不改变足球观点。",
        f"final_model={ai_r.get('source_model')} phase={ai_r.get('source_phase')} score={score} direction={direction}",
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
        # Frontend-compatible grading fields. These are AI recommendation display scores, not local football judgement.
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

        "market_interpretation": market_interpretation,
        "money_flow": money_flow,
        "contextual_logic": contextual_logic,
        "rejected_cases": ai_r.get("rejected_cases", {}),
        "web_research": web_research,
        "final_web_audit": ai_r.get("final_web_audit", {}),
        "data_quality": ai_r.get("data_quality", {}),
        "ai_native_reason": ai_r.get("reason", ""),
        "validation_warnings": list(dict.fromkeys(warnings)),

        # Legacy-compatible fields retained as disabled/empty.
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
        "unified_source": "ai_native_web",
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

        "cold_door": {
            "is_cold_door": False,
            "strength": 0,
            "level": "AI-native字段，不由本地判断冷门",
            "signals": rec.get("risk_tags", []),
            "sharp_confirmed": str(money_flow.get("sharp_money_direction", "unclear")) not in ("", "unclear", "none"),
            "dark_verdict": "",
        },

        "xG_home": "?",
        "xG_away": "?",
        "bookmaker_implied_home_xg": "?",
        "bookmaker_implied_away_xg": "?",
        "over_2_5": None,
        "fair_dir": None,
        "shin_dir": None,
        "actual_handicap_signed": None,
        "theoretical_handicap_signed": None,

        # Legacy panel fields. Never output placeholders as scores; old UI may use these as primary display.
        "gpt_score": _legacy_model_score(ai_r, "gpt", score),
        "gpt_analysis": _legacy_model_analysis(ai_r, "gpt"),
        "grok_score": _legacy_model_score(ai_r, "grok", score),
        "grok_analysis": _legacy_model_analysis(ai_r, "grok"),
        "gemini_score": _legacy_model_score(ai_r, "gemini", score),
        "gemini_analysis": _legacy_model_analysis(ai_r, "gemini"),
        "claude_score": score,
        "claude_analysis": ai_r.get("reason", "")[:3000] or "AI终审已给出结构化预测",
        "final_ai_score": score,
        "final_ai_analysis": ai_r.get("reason", "")[:3000],
        "ai_abstained": [],
        "ai_avg_confidence": _f(rec.get("bet_confidence", 0), 0),
        "ai_call_status": dict(AI_CALL_STATUS),
        "ai_result_files": dict(AI_RESULT_FILES),
        "ai_run_metadata": dict(_LAST_AI_RUN_METADATA),
        "model_consensus": None,
        "total_models": 4,

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
    print(f"  [{ENGINE_VERSION}] AI-NATIVE WEB-AUGMENTED | {len(ms)} 场 | 本地只做协议层 | chunk={AI_CHUNK_SIZE} | mock={AI_MOCK_MODE}")
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


if __name__ == "__main__":
    logger.info(f"{ENGINE_VERSION} 启动")
    print(f"✅ {ENGINE_VERSION} 加载完成")
    print(f"   架构: {ENGINE_ARCHITECTURE}")
    print("   模式: AI判断足球；本地只做协议/分批/解析/落盘/前端兼容。")
