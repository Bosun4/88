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
    GPT_API_URL / GPT_API_KEY
    GROK_API_URL / GROK_API_KEY
    GEMINI_API_URL / GEMINI_API_KEY
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
import contextvars
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
# 注: ENGINE_VERSION / ENGINE_ARCHITECTURE 的权威定义在下方读盘范式段
# (模块加载末段会覆盖, 此处旧赋值为僵尸已移除, 避免双真值源漂移)。

VALID_DIRS = {"home", "draw", "away"}
BETTABLE_ACTIONS = {"main", "small", "hedge"}
AI_NAMES = ["gpt", "grok", "gemini"]

# 【读盘范式 v2.1 大球判据阈值 / 175场回归校准 20260601】
# 单一真相来源：判大球的曲线塌缩斜率阈值与超大球尾部豁免阈值。
# 实证：阈值1.45是全表最差召回(28.6%)；放宽到1.70后召回75%/F1 53.8%。
# 尾部豁免(a6<=11 或 a7<=14)专治巴西6-2/大阪6-1此类被单点诱盘逻辑误杀的碾压大球。
BIG_GOAL_SLOPE_THRESHOLD = 1.70
BIG_GOAL_TAIL_A6_MAX = 11.0
BIG_GOAL_TAIL_A7_MAX = 14.0
PHASE1_NAMES = ["gpt", "grok"]

DEFAULT_MODELS = {
    "gpt": "gpt-5.5",
    "grok": "grok-4.3-c",
    "gemini": "熊猫-顶级特供-X-17-gemini-3.1-pro-preview-联网",
}

# 简单粗暴的 1-5 号接口池：URL/KEY 从后台环境变量读取，模型名在代码里写死。
# 你只需要在这里补 2/3/4/5 的模型名；对应后台变量为：
#   GEMINI_API_URL_2 / GEMINI_API_KEY_2  或  GEMINI_API_URL2 / GEMINI_API_KEY2
# GPT/GROK 同理。模型名留空的 slot 会被跳过，避免误用未配置模型。
AI_ENDPOINT_MODEL_SLOTS = {
    "gpt": {
        1: "gpt-5.5",
        2: "gpt-5.5",
        3: "gpt-5.5",
        4: "gpt-5.5",
        5: "熊猫-特供-X-10-gpt-5.5",
    },
    "grok": {
        1: "grok-4.3-c",
        2: "grok-4.3-c",
        3: "grok-4.3-c",
        4: "grok-4.3-c",
        5: "熊猫-A-5-grok-4.2-fast-200w上下文",
    },
    "gemini": {
        1: "熊猫-顶级特供-X-17-gemini-3.1-pro-preview-联网",
        2: "熊猫-顶级特供-X-17-gemini-3.1-pro-preview-联网",  # TODO: 填你的 GEMINI 2号模型名
        3: "熊猫-顶级特供-X-17-gemini-3.1-pro-preview-联网",  # TODO: 填你的 GEMINI 3号模型名
        4: "熊猫-顶级特供-X-17-gemini-3.1-pro-preview-联网",  # TODO: 填你的 GEMINI 4号模型名
        5: "熊猫-X-10-官逆-gemini-3.1-pro-联网",  # TODO: 填你的 GEMINI 5号模型名
    },
}

AI_ENDPOINT_RR_CURSOR: Dict[str, int] = {}
AI_ENDPOINT_SLOT_OVERRIDE: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("AI_ENDPOINT_SLOT_OVERRIDE", default=None)


def _resolve_endpoint_model_slot(ai_name: str, slot: int) -> str:
    """Resolve hard-coded endpoint model slots safely.

    Slot values are model strings, e.g. "gpt-5.5". They are NOT keys into
    DEFAULT_MODELS. This guard also tolerates accidental DEFAULT_MODELS-style
    aliases ("gpt"/"grok"/"gemini") without import-time KeyError.
    """
    name = str(ai_name or "").strip().lower()
    value = AI_ENDPOINT_MODEL_SLOTS.get(name, {}).get(slot, "")
    model = str(value or "").strip()
    if not model:
        return ""
    return DEFAULT_MODELS.get(model, model)


def _resolve_endpoint_model_slot(ai_name: str, slot: int) -> str:
    """Resolve hard-coded endpoint model slots safely.

    Slot values are model strings, e.g. "gpt-5.5". They are NOT keys into
    DEFAULT_MODELS. This guard also tolerates accidental DEFAULT_MODELS-style
    aliases ("gpt"/"grok"/"gemini") without import-time KeyError.
    """
    name = str(ai_name or "").strip().lower()
    value = AI_ENDPOINT_MODEL_SLOTS.get(name, {}).get(slot, "")
    model = str(value or "").strip()
    if not model:
        return ""
    return DEFAULT_MODELS.get(model, model)

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
    "世界杯": {
        "style_hint": "中立场锦标赛足球，不能照搬俱乐部联赛节奏；首轮最闷，第二轮最开放，第三轮不是机械小球，而是净胜球收窄、轮换与出线形势驱动的高波动分层场。",
        "score_shapes_to_audit": ["0-0", "1-1", "1-0", "0-1", "2-1", "1-2", "2-2", "3-1", "1-3"],
        "risk_note": "第三轮必须先审计出线形势、是否已出线、是否需净胜球、是否会轮换；已出线强队 vs 有动机方是诱盘高危，不能只因名气和低赔强推热门。实证(项目自对账)：后期轮强队被逼平是高频系统偏差(西班牙0-0/葡萄牙1-1/卡塔尔1-1/沙特1-1/厄瓜多尔0-0)，出线+轮换场默认下调热门胜信心上限≤60，优先防 0-0/1-1/1-0/0-1/被逼平。",
    },
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
# Legacy flag removed - each model now uses dedicated URL/KEY
# AI_FORCE_COMMON_GATEWAY = _env_bool("FORCE_COMMON_GATEWAY_URL", True)

AI_RESEARCH_MODE = str(os.environ.get("AI_RESEARCH_MODE", "research")).strip().lower()
if AI_RESEARCH_MODE not in {"production", "enhanced", "research"}:
    AI_RESEARCH_MODE = "research"

AI_RUN_MODE = str(os.environ.get("AI_RUN_MODE", "")).strip().lower()
if AI_RUN_MODE not in {"", "fast_batch", "deep_research", "post_review"}:
    AI_RUN_MODE = ""
if not AI_RUN_MODE:
    AI_RUN_MODE = {
        "production": "fast_batch",
        "enhanced": "deep_research",
        "research": "deep_research",
    }.get(AI_RESEARCH_MODE, "deep_research")

if AI_RUN_MODE == "fast_batch":
    _default_native_web = True
    _default_cross_exam = False
    _default_consistency = True
    _default_chunk_size = 12
elif AI_RUN_MODE == "post_review":
    _default_native_web = False
    _default_cross_exam = False
    _default_consistency = False
    _default_chunk_size = 30
elif AI_RESEARCH_MODE == "production":
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
AI_CHUNK_CONCURRENCY = max(1, _env_int("AI_CHUNK_CONCURRENCY", _env_int("AI_BATCH_SIZE", 1)))
AI_MODEL_CONCURRENCY = max(1, _env_int("AI_MODEL_CONCURRENCY", AI_CHUNK_CONCURRENCY))
AI_PHASE1_PARALLEL = _env_bool("AI_PHASE1_PARALLEL", True)
# Keep the default prompt budget below the historical 3-4万 token range while
# still allowing explicit environment overrides for larger research runs.
AI_MAX_PROMPT_CHARS_PER_CHUNK = max(30000, _env_int("AI_MAX_PROMPT_CHARS_PER_CHUNK", 60000))
AI_ENABLE_CROSS_EXAM = _env_bool("AI_ENABLE_CROSS_EXAM", _default_cross_exam)
AI_ENABLE_CONSISTENCY_JUDGE = _env_bool("AI_ENABLE_CONSISTENCY_JUDGE", _default_consistency)
AI_ENABLE_FALLBACK_REFEREE = _env_bool("AI_ENABLE_FALLBACK_REFEREE", True)
# Safety: phase1 models are analysts, not final referees. If Gemini/fallback final
# cannot produce a valid row, abstain instead of promoting phase1/Grok into final score.
AI_ALLOW_PHASE1_FINAL_FALLBACK = _env_bool("AI_ALLOW_PHASE1_FINAL_FALLBACK", False)
AI_CONSISTENCY_JUDGE_MODEL = str(os.environ.get("AI_CONSISTENCY_JUDGE_MODEL", "gpt")).strip().lower()
AI_FINAL_REFEREE_MODEL = str(os.environ.get("AI_FINAL_REFEREE_MODEL", "gemini")).strip().lower() or "gemini"
AI_FALLBACK_REFEREE_MODEL = str(os.environ.get("AI_FALLBACK_REFEREE_MODEL", "gpt")).strip().lower()

AI_READ_TIMEOUT = _env_int("AI_READ_TIMEOUT", 5400)
AI_FINAL_READ_TIMEOUT = _env_int("AI_FINAL_READ_TIMEOUT", _env_int("AI_CLAUDE_READ_TIMEOUT", 7200))
AI_CONNECT_TIMEOUT = _env_int("AI_CONNECT_TIMEOUT", 120)
# Plan B: bounded retry for the FINAL referee (and fallback referee) only.
# Pure transport/transient resilience; does NOT touch request URL/key/payload/stream.
AI_FINAL_RETRY_MAX = max(0, _env_int("AI_FINAL_RETRY_MAX", 2))
AI_FINAL_RETRY_BASE_DELAY = _env_int("AI_FINAL_RETRY_BASE_DELAY_MS", 1500)
AI_ENDPOINT_MAX_SLOTS = max(1, min(5, _env_int("AI_ENDPOINT_MAX_SLOTS", 5)))
AI_ENDPOINT_FAILOVER = _env_bool("AI_ENDPOINT_FAILOVER", True)
AI_ENDPOINT_ROUND_ROBIN = _env_bool("AI_ENDPOINT_ROUND_ROBIN", True)
AI_ENDPOINT_SLOT_QUEUE = _env_bool("AI_ENDPOINT_SLOT_QUEUE", True)
AI_ENDPOINT_SLOT_WORKERS = max(1, min(5, _env_int("AI_ENDPOINT_SLOT_WORKERS", AI_ENDPOINT_MAX_SLOTS)))
# Plan B+: when Gemini final referee fails, GPT runs a 16-role family debate to
# adjudicate the final score (acts as the referee, not a phase1 analyst).
AI_ENABLE_FAMILY_DEBATE_REFEREE = _env_bool("AI_ENABLE_FAMILY_DEBATE_REFEREE", True)
AI_FAMILY_DEBATE_MODEL = str(os.environ.get("AI_FAMILY_DEBATE_MODEL", "gpt")).strip().lower() or "gpt"
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
# P0-1 OU 独立判定头 (2026-07-02 十日审计升级)
# 背景: over_under_2_5 曾 = f(predicted_score) 机械耦合, 比分错则OU必错。
# 本头用 a0-a7 总进球赔率反推市场进球概率曲线, 独立判定大小球;
# 与比分冲突时只打 ou_score_conflict 标签 —— 不改方向、不改比分 (军规)。
# ============================================================

def derive_market_goal_curve(match_obj: Dict[str, Any]) -> Optional[Dict[int, float]]:
    """a0-a7 总进球赔率 -> 去水归一的市场进球概率曲线 {0: p0, ..., 7: p7+}。

    纯市场事实反推, 不做足球判断。赔率无效(≤1.01)的档位按0概率处理;
    有效档位不足4个视为数据不可用, 返回 None。
    """
    if not isinstance(match_obj, dict):
        return None
    raw: Dict[int, float] = {}
    valid = 0
    for g in range(8):
        odd = _f(match_obj.get(f"a{g}"), 0.0)
        if odd > 1.01:
            raw[g] = 1.0 / odd
            valid += 1
        else:
            raw[g] = 0.0
    if valid < 4:
        return None
    total = sum(raw.values())
    if total <= 0:
        return None
    return {g: p / total for g, p in raw.items()}


def derive_ou_head(
    match_obj: Dict[str, Any],
    predicted_score: str,
    goal_range: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """独立大小球判定。返回 {over_under_2_5, ou_market_prob_over, ou_score_conflict, ou_source}。

    判定规则:
      1. 市场曲线可用: P(总进球>=3) 明显偏离 0.5 (±0.05 外) → 直接按市场;
         临界区 (0.45~0.55) 用 AI goal_range 中点做 tiebreak, 无 goal_range 则按比分总球。
      2. 市场曲线不可用: 回退旧逻辑 (比分总球 >=3 为大) —— 向后兼容。
      3. 弃权比分: OU 输出 None。
    冲突标记: 独立判定与比分隐含 OU 不一致时 ou_score_conflict=True (仅标签, 不改比分)。
    """
    h, a = _parse_score(predicted_score)
    if h is None or a is None:
        return {
            "over_under_2_5": None,
            "ou_market_prob_over": None,
            "ou_score_conflict": False,
            "ou_source": "abstain",
        }
    score_total = h + a
    score_ou = "大" if score_total >= 3 else "小"

    curve = derive_market_goal_curve(match_obj)
    if curve is None:
        return {
            "over_under_2_5": score_ou,
            "ou_market_prob_over": None,
            "ou_score_conflict": False,
            "ou_source": "score_fallback",
        }

    p_over = round(sum(p for g, p in curve.items() if g >= 3), 4)
    # 阈值0.5支点(38场回放最优); 仅窄带[0.48,0.52]真五五开时才用AI倾向tiebreak
    if p_over > 0.52:
        ou = "大"
        source = "market"
    elif p_over < 0.48:
        ou = "小"
        source = "market"
    else:
        # 临界区: 用 AI goal_range 倾向 tiebreak, 缺失则退回比分口径
        if goal_range and len(goal_range) == 2:
            mid = (_f(goal_range[0]) + _f(goal_range[1])) / 2.0
            ou = "大" if mid >= 2.5 else "小"
            source = "goal_range_tiebreak"
        else:
            ou = score_ou
            source = "score_tiebreak"
    return {
        "over_under_2_5": ou,
        "ou_market_prob_over": p_over,
        "ou_score_conflict": ou != score_ou,
        "ou_source": source,
    }


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

    # 【读盘范式 v2.1 / 实证 175 场回归校准 20260601】判大球看整条曲线塌缩，不看 a4 单点。
    # a4>5.3 → 排除线（真实大球率仅 12.8%，按小球处理）；a4<=5.3 不再单点定大球。
    # 真信号 = 曲线斜率 a5/a4 + 超大球尾部共振（a6/a7 同步压低）。
    # 阈值校准: 1.45→1.70（175场: 召回28.6%→75%, F1 36.8%→53.8%；1.45是全表最差召回）。
    # 尾部共振豁免: a6<=11 或 a7<=14 时即使 slope 略高也释放（专治巴西6-2 a7=12 被单点诱盘误杀）。
    curve_slope = (a5 / a4) if (a4 and a5 and a4 > 0) else None
    _tail_resonance = (a6 > 0 and a6 <= BIG_GOAL_TAIL_A6_MAX) or (a7 > 0 and a7 <= BIG_GOAL_TAIL_A7_MAX)
    if a4 > 5.3:
        observations.append({"anchor": "four_goals_exclusion_line", "value": a4, "meaning_for_ai": "4球赔率>5.3（排除线）：实证真实4+大球率仅约13%，本场大球路径阻力大，AI应以0-3球小球带为主审，除非有联赛风格/资金流强反证，不要主推4+大比分。"})
    elif curve_slope is not None and curve_slope <= BIG_GOAL_SLOPE_THRESHOLD:
        observations.append({"anchor": "big_goal_curve_collapse", "value": {"a4": a4, "a5": a5, "a5_over_a4": round(curve_slope, 2)}, "meaning_for_ai": "大球曲线塌缩确认：5球赔率随4球一起被压（a5/a4<=1.70，175场实证此带真实大球率约42%，远高于32%基础率），这是整条大球簇上移的真信号，而非单点诱盘。应解除1-1/2-1小球锚定，主客对称释放大球带3-1/1-3/2-2/2-3/3-2/4-1/1-4，并结合联赛相对分位定档。"})
    elif (0 < a4 <= 5.3) and _tail_resonance:
        observations.append({"anchor": "big_goal_tail_resonance", "value": {"a4": a4, "a5": a5, "a6": a6, "a7": a7, "a5_over_a4": round(curve_slope, 2) if curve_slope else None}, "meaning_for_ai": "超大球尾部共振：a4偏低且6球/7+尾部同步被压（a6<=11 或 a7<=14），说明市场对整条超大球带防范极深，是碾压型大球的真信号（实证巴西6-2/大阪6-1此类被旧单点诱盘逻辑误杀）。应释放大球带3-1/1-3/2-2/2-3/3-2/4-1/1-4并向上审计4-2/5+尾部。"})
    elif 0 < a4 <= 5.3:
        observations.append({"anchor": "four_goals_single_point_low_caution", "value": {"a4": a4, "a5": a5}, "meaning_for_ai": "4球赔率偏低（<=5.3）但5球未同步压缩（a5/a4>1.70）且无超大球尾部共振：疑似单点诱盘，市场只压了4球这一点。不可仅凭a4低就判大球，应优先审计2-3球平衡带，只有出现联赛偏大球或资金推强队等额外共振时才谨慎释放大球。"})

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
    # 【大球曲线塌缩注入·读盘范式v2.1 / 175场回归校准 20260602】触发条件与观察层 _total_goal_anchor_facts 严格同源：
    #   ① 曲线塌缩: a5/a4 斜率 <= BIG_GOAL_SLOPE_THRESHOLD(1.70)  （5球跟4球一起被压=整簇上移真信号）
    #   ② 尾部共振豁免: a6<=11 或 a7<=14（市场对整条超大球带防范极深=碾压型大球真信号，专治巴西6-2/大阪6-1被单点诱盘误杀）
    # a4>5.3 是排除线不注入；不看 a4 单点。常量单一真相来源，杜绝执行层与观察层阈值撕裂。
    a4 = _f(match_obj.get("a4"), 0.0)
    a5 = _f(match_obj.get("a5"), 0.0)
    a6 = _f(match_obj.get("a6"), 0.0)
    a7 = _f(match_obj.get("a7"), 0.0)
    _slope = (a5 / a4) if (a4 > 0 and a5 > 0) else None
    _tail_resonance = (a6 > 0 and a6 <= BIG_GOAL_TAIL_A6_MAX) or (a7 > 0 and a7 <= BIG_GOAL_TAIL_A7_MAX)
    _slope_collapse = (_slope is not None and _slope <= BIG_GOAL_SLOPE_THRESHOLD)
    if (a4 > 0 and a4 <= 5.3) and (_slope_collapse or _tail_resonance):
        must = list(tpl.get("must_audit_scores", []))
        for sc in BIG_GOAL_PRIMARY:
            if sc not in must:
                must.append(sc)
        tpl = dict(tpl)
        tpl["must_audit_scores"] = must
        _trigger = "曲线塌缩(a5/a4<=%.2f)" % BIG_GOAL_SLOPE_THRESHOLD if _slope_collapse else "超大球尾部共振(a6<=%g或a7<=%g)" % (BIG_GOAL_TAIL_A6_MAX, BIG_GOAL_TAIL_A7_MAX)
        tpl["big_goal_injection"] = {
            "triggered": True,
            "a4": a4,
            "a5": a5,
            "a6": a6,
            "a7": a7,
            "trigger": _trigger,
            "injected_cluster": BIG_GOAL_PRIMARY,
            "reason": "大球%s=整簇上移真信号(非单点诱盘)，强制将对称大球带并入must_audit_scores；AI必须主客对称审计，不得机械压回2-1/1-1。" % _trigger,
        }
    return {"raw": raw, "parsed_line": val, "favorite_side_by_line_sign": fav, "score_shape_template": tpl, "ai_instruction": "让球盘只提供比分形态审计模板，不代表本地预测方向。AI必须结合1X2、正确比分、总进球和外部盘口验证。"}


def _cross_anchor_questions(match_obj: Dict[str, Any]) -> List[str]:
    qs = []
    score_f = _score_anchor_facts(match_obj)
    total_f = _total_goal_anchor_facts(match_obj)
    hand_f = _handicap_anchor_facts(match_obj)
    s00 = score_f.get("specific_odds", {}).get("0-0", 0)
    s11 = score_f.get("specific_odds", {}).get("1-1", 0)
    a4 = total_f.get("specific_odds", {}).get("4", 0)
    a5 = total_f.get("specific_odds", {}).get("5", 0)
    a6 = total_f.get("specific_odds", {}).get("6", 0)
    a7 = total_f.get("specific_odds", {}).get("7+", 0)
    _slope = (a5 / a4) if (a4 and a5 and a4 > 0) else None
    _tail_reson = (a6 and a6 <= BIG_GOAL_TAIL_A6_MAX) or (a7 and a7 <= BIG_GOAL_TAIL_A7_MAX)
    mode_g = total_f.get("mode_goals")
    if 0 < s00 <= 11:
        qs.append("0-0赔率≤11：为什么不是0-0？如果选其他比分，必须解释突破闷局的证据。注意：0-0属LOW档(0-1球)，1-1属MID档(2-3球)，两者不同档，禁止拿0-0与1-1直接比赔率；必须先定LOW/MID再档内选。")
    if 0 < s11 <= 7.5:
        qs.append("1-1赔率偏低：为什么不是1-1？1-1是MID档众数比分，选它不需额外理由；若要偏离到1-2/2-1等须写出可证伪的背离理由，否则回落1-1。")
    if a4 > 5.3:
        qs.append("4球赔率>5.3（排除线）：实证真实大球率仅约13%，若选择3-1/2-2/3-2等4+比分，必须解释为何能突破4球高赔阻力；否则压回0-3球带。")
    elif (a4 and a4 <= 5.3) and (_slope is not None and _slope <= BIG_GOAL_SLOPE_THRESHOLD):
        qs.append(f"⚠大球曲线塌缩确认（a4={a4}, a5={a5}, a5/a4={round(_slope,2)}<={BIG_GOAL_SLOPE_THRESHOLD}）：5球跟着4球一起被压，是整簇上移的真信号。定档HIGH(4+)，主客对称审计大球带：3-1/1-3/2-2/3-2/2-3/4-1/1-4。【HIGH档弃精确】HIGH档无任何比分超过16%概率(175场实证)，不要勉强押准某个4+精确比分，主推“大球方向+主/客大胜方向”即可。")
    elif (a4 and a4 <= 5.3) and _tail_reson:
        qs.append(f"⚠超大球尾部共振（a4={a4}, a6={a6}, a7={a7}：a6<={BIG_GOAL_TAIL_A6_MAX}或a7<={BIG_GOAL_TAIL_A7_MAX}）：市场对整条超大球带防范极深，是碾压型大球真信号（实证巴西6-2/大阪6-1被旧单点诱盘误杀）。定档HIGH(4+)，主客对称审计大球带并向上审计4-2/5+尾部；【HIGH档弃精确】不押精确比分，主推大球/大胜方向。")
    elif a4 and a4 <= 5.3:
        qs.append(f"4球赔率偏低(a4={a4})但5球未同步压(a5={a5}, a5/a4>{BIG_GOAL_SLOPE_THRESHOLD})且无超大球尾部共振：疑似单点诱盘，不可仅凭a4低就释放大球。优先审计MID(2-3球)平衡带，除非联赛风格偏大球或资金强推一方。")
    if mode_g in (0, 1, 2):
        qs.append("总进球主模态在0-2球：不能机械输出2-1，必须审计0-0/1-0/0-1/1-1/2-0/0-2。")
    if mode_g in (4, 5, 6, 7):
        qs.append("总进球主模态在4+：不能机械输出1-0/1-1，必须主客对称审计3-1/1-3/2-2/3-2/2-3/4-1/1-4/4-2/胜其他。")
    # ============================================================
    # 世界杯/国际赛比分形状强约束（intl_score_bias_audit_20260610 落地）
    # 数据支撑：X-0 零封预测 73% 失败；同方向比分低估:高估=5:1；判平默认1-1只中3/9。
    # 06-10 实证：奥地利无意走对路径(3-1命中)，挪威踩零封幻觉(0-3→实际1-4)。
    # 此三条只挂 world_cup/intl_friendly，不污染俱乐部联赛读盘。
    # ============================================================
    _lg = str(match_obj.get("league", ""))
    _is_intl = any(t in _lg for t in ["世界杯", "world cup", "worldcup", "fifa world", "国际赛", "国际友谊", "友谊赛", "热身赛", "friendly"])
    if _is_intl:
        qs.append("【零封税·条件化】国际赛/世界杯不能机械默认 X-0，也不能机械给负方安慰球。若弱方有独立破门证据（反击/xG/定位球/对手防线轮换/BTTS盘口共振），优先保留 X-1；若强弱悬殊、弱方破门证据不足且零封比分簇(2-0/3-0/4-0)低赔集中，必须允许并优先审计 N-0，不得因旧零封税强行上修 N-1。")
        qs.append("【进球带上修·条件化】国际赛/世界杯方向判定不变，但进球带上修只在总进球曲线整簇塌缩或强队火力证据充分时使用；深盘造强但胜赔不实压、1-1锚点未抬死、弱队可能极限收缩时，必须把0-0/1-1列入最终候选并说明为什么没有入选。")
        qs.append("【判平二段裁决】若最终判平局，禁止默认 1-1，必须在 0-0/1-1/2-2 三者间用进球曲线分位(a0/a2/a4)显式选形状并写出理由。数据支撑：9 场判平全押 1-1 仅中 3。")
        qs.append("【世界杯第三轮硬审计】若识别到小组赛第三轮/末轮，必须先回答五个问题：①双方是否都还有晋级/排名/净胜球动机；②是否存在已出线强队大轮换/主力轮休风险（联网核实预计首发与教练发布会口径）；③是否属于双方都无所求的闷平土壤；④是否属于一方需抢分、一方可接受小负/小胜的控分土壤；⑤是否两队同组并行开赛、存在‘默契球/算计净胜球’的控分共谋土壤。未联网回答这五点，不得把热门强队直接推成穿盘大胜，且 recommendation 最高 B。")
        qs.append("【世界杯第三轮爆冷高发·实证先验】项目实测(2026小组赛23场对账)：强弱悬殊却踢平的高信心翻车集中在第二/三轮（西班牙0-0、葡萄牙1-1、卡塔尔1-1、沙特1-1、厄瓜多尔0-0），根因是‘强队已出线→大面积轮换→控分/慢热→爆冷与闷平’被系统性低估。第三轮已出线强队对阵有动机弱队时，必须把【被逼平/小负/0进球】列为首要风险路径，默认下调该热门方向信心上限至 ≤60，并显式审计 0-0/1-1/1-0/0-1/1-2/2-1。")
        qs.append("【世界杯第三轮比分簇优先级】R3 控分不是机械小球，而是净胜球收窄与热门被掀风险上升。若无强证据支持刷净胜球或全主力强攻，必须优先审计 1-0/0-1/1-1/2-1/1-2 这些一球差与冷平方向；只有联网确认双方都需抢胜/抢净胜球且双方均派主力时，才释放 3-1/1-3/2-2/3-2/2-3 等开放比分。已出线强队大轮换证据成立时，禁止主推任何净胜≥2的穿盘大胜。")
        # [补丁A 2026-06-22] 单边碾压上修先验(市场版,不依赖pred)：深盘+某方零封簇低赔=单边碾压结构,
        # 第一轮21球缺口集中在此类8场。与曲线塌缩判据互补(碾压场总进球分布天然分散,a4常>5.3不触发)。
        try:
            _ub_line = _parse_handicap_value(match_obj.get("give_ball", match_obj.get("handicap", match_obj.get("rq", ""))))
            if _ub_line is not None and abs(_ub_line) >= 1.5:
                _h_cs = min([o for o in [get_market_odds_for_score(match_obj, s) for s in ("2-0", "3-0")] if o] or [99])
                _a_cs = min([o for o in [get_market_odds_for_score(match_obj, s) for s in ("0-2", "0-3")] if o] or [99])
                _fav, _cs_low = ("home", _h_cs) if _h_cs <= _a_cs else ("away", _a_cs)
                if _cs_low <= 6.0:
                    _up = "3-0/3-1/4-1" if _fav == "home" else "0-3/1-3/1-4"
                    qs.append(f"【单边碾压进球上修·强制】本场深盘(让球{abs(_ub_line):.2f})且{'主' if _fav=='home' else '客'}方零封簇低赔({_cs_low:.2f})=单边碾压结构。实证(世界杯小组赛)此类场进球量级被系统性低估一档(瑞典1-0→5-1/德国4-0→7-1/挪威2-0→4-1)。若方向判为{'主胜' if _fav=='home' else '客胜'}，必须把进球带从2-0/2-1上修审计到{_up}，并显式说明为何不选更高比分；不得机械回落2-1。此判据针对单边碾压,不要求总进球曲线塌缩。")
        except Exception:
            pass
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
            "team_stats_reliable": not (
                isinstance(m.get("home_stats"), dict) and m.get("home_stats", {}).get("data_available") is False
                or isinstance(m.get("away_stats"), dict) and m.get("away_stats", {}).get("data_available") is False
            ),
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

GROK_WEBMAX_EXTERNAL_INTELLIGENCE_ADDENDUM = (
    "你的联网重点：你是 Grok Web-Max 外部事实总参谋，不是比分裁判。必须最大化搜索并结构化输出可验证赛前事实："
    "伤停/停赛/复出、预计或官方首发、战意与出线条件、赛程密度/旅行疲劳、天气/场地/中立场、"
    "官方公告/主流媒体/跟队记者/数据站，以及 Bet365/William Hill/威廉希尔/Pinnacle/竞彩/百家均值等当前赔率、亚盘、大小球、正确比分快照。"
    "必须排查脏数据：无 URL、过期新闻、预测站搬运、社媒传闻、盘口快照伪装时间序列、Bet365/威廉/Pinnacle/竞彩方向冲突、赔率升水降水与大小球变化不一致。"
    "每条会影响方向、比分或推荐等级的 claim 必须进入 external_fact_table，并包含 category、claim、source_type、source_title、source_url、published_at、freshness、confidence、impact_direction、why_it_matters。"
    "必须输出 source_conflict_audit、evidence_quality_score(0-100)、minimum_evidence_needed、external_facts_decision_impact。"
    "source_type 优先级：official > mainstream_media > beat_reporter > data_site > prediction_site > social_rumor。"
    "无 URL、url 为 #、过旧新闻、单一预测站、社媒传闻均不得作为升权硬证据；只能降级或标记 risk_only。"
    "禁止编造盘口时间序列；当前赔率只能作为 market_snapshot，不能推导 T-60m/T-30m、临场回补、资金持续流入、诱盘闭环。"
    "若 sources 缺失或质量低，必须写 missing_external_confirmation，不得把伤停/首发/战意/赛程/天气当硬证据升推荐。"
)

EXTERNAL_FACT_FIELDS = [
    "external_fact_table",
    "source_conflict_audit",
    "evidence_quality_score",
    "minimum_evidence_needed",
    "external_facts_decision_impact",
]

FULL_SPECTRUM_AUDIT_FIELDS = [
    "gemini_independent_research",
    "bookmaker_cross_audit",
    "tempo_xg_tactical_audit",
    "worldcup_upset_audit",
    "score_elimination_audit",
    "dirty_work_checklist",
]

EXTERNAL_FACT_CONTEXT_TERMS = [
    "伤停", "停赛", "复出", "首发", "阵容", "战意", "出线", "轮换", "轮休", "体能", "旅行", "天气", "场地", "核心", "发布会",
    "injury", "injuries", "suspension", "lineup", "line-up", "starting xi", "motivation", "rotation", "travel", "weather", "pitch",
]

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
        return common + "你的联网重点：外部欧赔/亚盘/交易所、赔率时间点。注意：evidence 里的 dual_market_divergence_calibration 已经用 Shin 法算出国内竞彩 vs 国际清算盘的 skew 偏斜度与 z_gap，这是已成事实，你必须直接解读该结果（不要重复联网去算一遍分歧），联网仅用于验证/补充该偏斜是否有基本面支撑。"
    if role == "grok":
        return common + GROK_WEBMAX_EXTERNAL_INTELLIGENCE_ADDENDUM
    if role == "gemini":
        return common + "你的联网重点：Bet365/William Hill/威廉希尔/Pinnacle/竞彩/百家均值赔率差异、升水降水、亚盘、大小球、正确比分簇、球队节奏/xG/xGA、伤停首发、天气场地、世界杯轮次/出线/净胜球动机、战术与爆冷土壤。"
    return common + "你的联网重点：审计三家来源质量、冲突来源、新鲜度，以及哪些外部信息真正改变判断。"


def _short_prediction_for_prompt(r: Dict[str, Any]) -> Dict[str, Any]:
    keep = {}
    for k in [
        "match", "final_direction", "predicted_score", "direction_probs", "goal_band", "btts", "top3",
        "anchor_audit", "market_interpretation", "money_flow", "contextual_logic", "rejected_cases", "recommendation",
        "data_quality", "reason", "web_research", "final_web_audit", "validation_warnings",
        "external_fact_table", "source_conflict_audit", "evidence_quality_score",
        "minimum_evidence_needed", "external_facts_decision_impact",
        *FULL_SPECTRUM_AUDIT_FIELDS,
    ]:
        if k in r:
            keep[k] = r[k]
    return keep


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
    """Get model-specific API key. Each model now has its own dedicated key."""
    ep = _endpoint_candidates_for_ai(ai_name)
    return ep[0]["key"] if ep else ""


def get_url_for_ai(ai_name: str) -> str:
    """Get model-specific API URL. Each model now has its own dedicated URL."""
    ep = _endpoint_candidates_for_ai(ai_name)
    return ep[0]["url"] if ep else ""


def _model_for(ai_name: str) -> str:
    ep = _endpoint_candidates_for_ai(ai_name)
    return ep[0]["model"] if ep else DEFAULT_MODELS.get(ai_name, "model")


def _slot_env_names(prefix: str, kind: str, slot: int) -> List[str]:
    if slot <= 1:
        return [f"{prefix}_API_{kind}"]
    return [f"{prefix}_API_{kind}_{slot}", f"{prefix}_API_{kind}{slot}"]


def _endpoint_candidates_for_ai(ai_name: str) -> List[Dict[str, Any]]:
    """Read simple numbered endpoint slots: URL/KEY from env, model from code.

    Slot 1 uses GPT_API_URL/GPT_API_KEY. Slots 2-5 support both
    GPT_API_URL_2/GPT_API_KEY_2 and GPT_API_URL2/GPT_API_KEY2 aliases.
    Model names intentionally live in AI_ENDPOINT_MODEL_SLOTS above.
    """
    name = str(ai_name or "").strip().lower()
    prefix = name.upper()
    out: List[Dict[str, Any]] = []
    for slot in range(1, AI_ENDPOINT_MAX_SLOTS + 1):
        model = _resolve_endpoint_model_slot(name, slot)
        if not model:
            continue
        url = _clean_env_url(*_slot_env_names(prefix, "URL", slot))
        key = _clean_env_key(*_slot_env_names(prefix, "KEY", slot))
        if not url or not key:
            continue
        out.append({"name": f"{name}_{slot}", "ai_name": name, "slot": slot, "url": url, "key": key, "model": model})
    return out


def _ordered_endpoints_for_ai(ai_name: str) -> List[Dict[str, Any]]:
    eps = _endpoint_candidates_for_ai(ai_name)
    slot_override = AI_ENDPOINT_SLOT_OVERRIDE.get()
    if slot_override is not None:
        pinned = [ep for ep in eps if int(ep.get("slot", 0)) == int(slot_override)]
        if pinned:
            return pinned
        # If a model lacks this numbered slot, fall back to its normal ordered
        # endpoints instead of failing the whole match. This keeps mixed GPT/Grok/
        # Gemini deployments usable while still pinning configured slots.
    if not eps or not AI_ENDPOINT_ROUND_ROBIN:
        return eps
    key = str(ai_name or "").strip().lower()
    cur = AI_ENDPOINT_RR_CURSOR.get(key, 0) % len(eps)
    AI_ENDPOINT_RR_CURSOR[key] = cur + 1
    return eps[cur:] + eps[:cur]


def _chat_url(base_url: str) -> str:
    u = (base_url or "").rstrip("/")
    if not u:
        return ""
    if u.endswith("/chat/completions") or "/chat/completions" in u:
        return u
    return u + "/chat/completions"


def debug_ai_config() -> None:
    print(f"[AI CONFIG] mode={AI_RESEARCH_MODE} run_mode={AI_RUN_MODE} mock={AI_MOCK_MODE} native_web={AI_NATIVE_WEB} chunk_size={AI_CHUNK_SIZE} chunk_concurrency={AI_CHUNK_CONCURRENCY} model_concurrency={AI_MODEL_CONCURRENCY} phase1_parallel={AI_PHASE1_PARALLEL} cross_exam={AI_ENABLE_CROSS_EXAM} consistency_judge={AI_ENABLE_CONSISTENCY_JUDGE} endpoint_slots={AI_ENDPOINT_MAX_SLOTS} endpoint_failover={AI_ENDPOINT_FAILOVER} endpoint_round_robin={AI_ENDPOINT_ROUND_ROBIN} endpoint_slot_queue={AI_ENDPOINT_SLOT_QUEUE} endpoint_slot_workers={AI_ENDPOINT_SLOT_WORKERS}")
    for n in AI_NAMES:
        eps = _endpoint_candidates_for_ai(n)
        if not eps:
            print(f"[AI CONFIG] {n.upper()} endpoints=<missing>")
            continue
        for ep in eps:
            print(f"[AI CONFIG] {n.upper()} slot={ep['slot']} model={ep['model']} key={_mask_key(ep['key'])} url={ep['url'] or '<missing>'}")


def _is_retryable_ai_status(status: Dict[str, Any]) -> bool:
    """Transient failures worth retrying for the final referee.

    Retryable: HTTP 429/5xx (channel saturation), timeouts, transport errors,
    and empty/parse_failed completions (provider returned 0 completion tokens
    under load). Non-retryable: no_key/no_url/aiohttp_missing/http_4xx (except 429).
    """
    st = str(status.get("status", "")).lower()
    if st in {"timeout", "error", "parse_failed"}:
        return True
    if st.startswith("http_"):
        code = st.replace("http_", "")
        if code == "429":
            return True
        if code.startswith("5"):
            return True
    return False


async def async_call_ai_json_with_retry(session: Optional[Any], ai_name: str, system_text: str, prompt: str, phase: str, expected_matches: List[int], max_retries: int = 0) -> Tuple[str, Any, Dict[str, Any]]:
    """Wrap async_call_ai_json with bounded exponential-backoff retry on transient errors.

    Only used for final/fallback referee phases. The first OK result wins; on
    repeated transient failure the last status is returned so the caller can
    fall through to the fallback referee or abstain.
    """
    attempts = max(0, int(max_retries)) + 1
    last: Tuple[str, Any, Dict[str, Any]] = (ai_name, {}, {"ok": False, "status": "not_called"})
    for attempt in range(attempts):
        name, obj, st = await async_call_ai_json(session, ai_name, system_text, prompt, phase, expected_matches)
        last = (name, obj, st)
        if st.get("ok"):
            if attempt > 0:
                st["retry_succeeded_on_attempt"] = attempt + 1
            return last
        if attempt < attempts - 1 and _is_retryable_ai_status(st):
            delay = (AI_FINAL_RETRY_BASE_DELAY / 1000.0) * (2 ** attempt)
            print(f"  [RETRY] {ai_name.upper()} {phase} attempt {attempt + 1}/{attempts} failed status={st.get('status')}; retrying in {delay:.1f}s")
            try:
                await asyncio.sleep(delay)
            except Exception:
                pass
            continue
        break
    return last


async def async_call_ai_json(session: Optional[Any], ai_name: str, system_text: str, prompt: str, phase: str, expected_matches: List[int]) -> Tuple[str, Any, Dict[str, Any]]:
    t0 = time.time()
    endpoints = _ordered_endpoints_for_ai(ai_name)
    model = endpoints[0]["model"] if endpoints else _model_for(ai_name)
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

    if not endpoints:
        # Backward-compatible escape hatch for tests or callers that monkeypatch
        # the legacy get_key_for_ai/get_url_for_ai helpers directly.
        legacy_key = get_key_for_ai(ai_name)
        legacy_url = get_url_for_ai(ai_name)
        if legacy_key and legacy_url:
            endpoints = [{
                "name": f"{str(ai_name).lower()}_legacy",
                "ai_name": str(ai_name).lower(),
                "slot": 1,
                "url": legacy_url,
                "key": legacy_key,
                "model": _model_for(ai_name),
            }]
        else:
            status.update({"status": "no_key", "endpoint_count": 0})
            _update_call_status(ai_name, phase, status)
            return ai_name, {}, status

    temperature = AI_TEMPERATURE_FINAL if phase in ("final", "fallback_referee", "family_debate_referee") else AI_TEMPERATURE_CRITIC if phase == "critic" else AI_TEMPERATURE_PHASE1
    last_status = status
    tries = endpoints if AI_ENDPOINT_FAILOVER else endpoints[:1]
    for attempt, endpoint in enumerate(tries, start=1):
        ep_t0 = time.time()
        model = endpoint["model"]
        url = _chat_url(endpoint["url"])
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {endpoint['key']}"}
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system_text}, {"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if AI_USE_RESPONSE_FORMAT:
            payload["response_format"] = {"type": "json_object"}
        status = {
            "ok": False,
            "ai_name": ai_name,
            "model": model,
            "phase": phase,
            "elapsed": 0.0,
            "endpoint_name": endpoint["name"],
            "endpoint_slot": endpoint["slot"],
            "endpoint_attempt": attempt,
            "endpoint_total": len(tries),
        }
        try:
            read_timeout = AI_FINAL_READ_TIMEOUT if phase in ("final", "fallback_referee", "family_debate_referee") else AI_READ_TIMEOUT
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
                    status.update({"status": f"http_{r.status}", "http_error": text[:800], "elapsed": round(time.time() - ep_t0, 1)})
                    _update_call_status(ai_name, phase, status)
                    last_status = status
                    if AI_ENDPOINT_FAILOVER and attempt < len(tries) and _is_retryable_ai_status(status):
                        print(f"  [ENDPOINT FAILOVER] {ai_name.upper()} {phase} {endpoint['name']} status={status.get('status')} -> next slot")
                        continue
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
                        "elapsed": round(time.time() - ep_t0, 1),
                    })
                    _update_call_status(ai_name, phase, status)
                    last_status = status
                    if AI_ENDPOINT_FAILOVER and attempt < len(tries) and _is_retryable_ai_status(status):
                        print(f"  [ENDPOINT FAILOVER] {ai_name.upper()} {phase} {endpoint['name']} status=parse_failed -> next slot")
                        continue
                    return ai_name, {}, status
                status.update({"ok": True, "status": "ok", "elapsed": round(time.time() - ep_t0, 1), "total_elapsed": round(time.time() - t0, 1)})
                _update_call_status(ai_name, phase, status)
                return ai_name, obj, status
        except asyncio.TimeoutError:
            status.update({"status": "timeout", "elapsed": round(time.time() - ep_t0, 1)})
        except Exception as e:
            status.update({"status": "error", "error": str(e)[:500], "elapsed": round(time.time() - ep_t0, 1)})
        _update_call_status(ai_name, phase, status)
        last_status = status
        if AI_ENDPOINT_FAILOVER and attempt < len(tries) and _is_retryable_ai_status(status):
            print(f"  [ENDPOINT FAILOVER] {ai_name.upper()} {phase} {endpoint['name']} status={status.get('status')} -> next slot")
            continue
        return ai_name, {}, status

    last_status.update({"elapsed": round(time.time() - t0, 1)})
    _update_call_status(ai_name, phase, last_status)
    return ai_name, {}, last_status


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


def _extract_sse_response_text(raw: str) -> str:
    """Extract assistant text from OpenAI-compatible SSE streams.

    Some proxy providers return `text/event-stream` bodies even when the request
    did not ask for streaming. Treating the whole `data: {...}` stream as JSON
    makes every call look like `empty_or_invalid_json_object`; this then lets
    lower-authority phase1 rows leak into final predictions.
    """
    if not isinstance(raw, str) or "data:" not in raw[:2000]:
        return ""
    parts: List[str] = []
    best_objects: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        # Full JSON object may itself be carried as a text event.
        txt = _extract_response_text(obj) if not (isinstance(obj, dict) and "choices" in obj) else ""
        if txt and ("predictions" in txt or "critic_reports" in txt or "repairs" in txt):
            best_objects.append(txt)
        if isinstance(obj, dict):
            for ch in obj.get("choices", []) or []:
                if not isinstance(ch, dict):
                    continue
                delta = ch.get("delta") if isinstance(ch.get("delta"), dict) else {}
                msg = ch.get("message") if isinstance(ch.get("message"), dict) else {}
                for v in [delta.get("content"), delta.get("reasoning_content"), msg.get("content"), ch.get("text")]:
                    if isinstance(v, str) and v:
                        parts.append(v)
    joined = "".join(parts).strip()
    if joined:
        return joined
    if best_objects:
        best_objects.sort(key=len, reverse=True)
        return best_objects[0]
    return ""


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
        sse_text = _extract_sse_response_text(data)
        if sse_text:
            add(sse_text, 20)
        add(data, 5)

    # Provider fallback: json.loads(raw_text) failed and async_call_ai_json wrapped
    # the body as {"raw": text}. Parse SSE before considering the raw envelope.
    if isinstance(data, dict) and isinstance(data.get("raw"), str):
        sse_text = _extract_sse_response_text(data.get("raw", ""))
        if sse_text:
            add(sse_text, 20)

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


def _repair_truncated_json(text: str) -> str:
    """
    Attempts to repair a truncated JSON string by closing unclosed brackets, braces,
    and quotes. Handles trailing commas gracefully.
    """
    text = text.strip()
    if not text:
        return ""
    
    in_str = False
    quote_char = None
    esc = False
    
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == quote_char:
                in_str = False
                quote_char = None
        else:
            if ch in ('"', "'"):
                in_str = True
                quote_char = ch
    
    if in_str:
        text += '"' if quote_char == '"' else "'"
    
    stack = []
    in_str = False
    esc = False
    quote_char = None
    
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == quote_char:
                in_str = False
                quote_char = None
        else:
            if ch in ('"', "'"):
                in_str = True
                quote_char = ch
            elif ch in ('[', '{'):
                stack.append(ch)
            elif ch in (']', '}'):
                if stack:
                    last = stack[-1]
                    if (last == '[' and ch == ']') or (last == '{' and ch == '}'):
                        stack.pop()
    
    text = re.sub(r',\s*$', '', text)
    text = re.sub(r':\s*$', ': null', text)
    
    while stack:
        last = stack.pop()
        if last == '[':
            text += ']'
        elif last == '{':
            text += '}'
            
    return text


def _json_loads_best_effort_object(text: str) -> Any:
    clean = _preclean_text(text)
    if not clean:
        return {}
    variants = [
        clean, 
        re.sub(r",\s*([}\]])", r"\1", clean),
        _repair_truncated_json(clean),
        re.sub(r",\s*([}\]])", r"\1", _repair_truncated_json(clean))
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
    for frag in _balanced_fragments(clean):
        for cand in [frag, re.sub(r",\s*([}\]])", r"\1", frag), _repair_truncated_json(frag)]:
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
        if len(out) >= 5:
            break
    if not out and predicted_score and _parse_score(predicted_score)[0] is not None:
        out = [{"score": predicted_score, "prob": 0.0, "logic": "top3_missing_but_predicted_score_present"}]
    elif predicted_score and _parse_score(predicted_score)[0] is not None and predicted_score not in seen:
        out.append({"score": predicted_score, "prob": 0.0, "logic": "predicted_score_retained_without_reordering"})
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


def _valid_external_source_count(web_research: Dict[str, Any], external_fact_table: Any = None) -> int:
    count = 0
    sources = web_research.get("sources", []) if isinstance(web_research, dict) else []
    if isinstance(sources, list):
        for src in sources:
            if not isinstance(src, dict):
                continue
            url = str(src.get("url", "")).strip()
            title = str(src.get("title", src.get("source_title", ""))).strip()
            claim = str(src.get("claim", "")).strip()
            if url and url != "#" and (title or claim):
                count += 1
    if isinstance(external_fact_table, list):
        for fact in external_fact_table:
            if not isinstance(fact, dict):
                continue
            url = str(fact.get("source_url", fact.get("url", ""))).strip()
            claim = str(fact.get("claim", "")).strip()
            if url and url != "#" and claim:
                count += 1
    return count


def _copy_external_fact_fields_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in EXTERNAL_FACT_FIELDS:
        value = item.get(key)
        if value in (None, {}, []):
            audit = item.get("final_web_audit") if isinstance(item.get("final_web_audit"), dict) else {}
            value = audit.get(key)
        if key == "evidence_quality_score":
            if value not in (None, ""):
                out[key] = int(_clip(_f(value, 0), 0, 100))
        elif isinstance(value, (dict, list)):
            out[key] = value
    return out


def _copy_full_spectrum_audit_fields_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    final_audit = item.get("final_web_audit") if isinstance(item.get("final_web_audit"), dict) else {}
    for key in FULL_SPECTRUM_AUDIT_FIELDS:
        value = item.get(key)
        if value in (None, {}, []):
            value = final_audit.get(key)
        if isinstance(value, (dict, list)):
            out[key] = value
    return out


def _external_context_text(pred: Dict[str, Any]) -> str:
    return _json_compact({
        "reason": pred.get("reason"),
        "ai_native_reason": pred.get("ai_native_reason"),
        "ai_score_reason": pred.get("ai_score_reason"),
        "final_ai_analysis": pred.get("final_ai_analysis"),
        "final_referee_analysis": pred.get("final_referee_analysis"),
        "contextual_logic": pred.get("contextual_logic"),
        "recommendation": pred.get("recommendation"),
        "external_fact_table": pred.get("external_fact_table"),
        "external_facts_decision_impact": pred.get("external_facts_decision_impact"),
    }, 5000).lower()


def _apply_external_fact_source_gate(pred: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(pred, dict) or pred.get("is_abstain"):
        return pred
    web = pred.get("web_research", {}) if isinstance(pred.get("web_research"), dict) else {}
    valid_sources = _valid_external_source_count(web, pred.get("external_fact_table"))
    warnings = pred.setdefault("validation_warnings", [])
    if not isinstance(warnings, list):
        warnings = [str(warnings)] if warnings else []
        pred["validation_warnings"] = warnings
    sources = web.get("sources", []) if isinstance(web.get("sources", []), list) else []
    if sources and valid_sources < len(sources):
        warnings.append("external_source_url_missing_or_invalid")

    text = _external_context_text(pred)
    needs_external_source = any(term.lower() in text for term in EXTERNAL_FACT_CONTEXT_TERMS)
    evidence_quality = _normalize_external_evidence_quality(pred)

    def _cap_to_c_observe(reason: str, tag_list: List[str], downgrade_msg: str) -> None:
        """P1 硬闸统一降级动作: tier 封顶 C + 不推荐 + observe + gate 不过。"""
        rec = pred.setdefault("recommendation", {}) if isinstance(pred.get("recommendation"), dict) else {}
        if not isinstance(pred.get("recommendation"), dict):
            pred["recommendation"] = rec
        current_tier = str(rec.get("tier", pred.get("recommendation_tier", "D"))).upper()
        rec["tier"] = "C" if _tier_cap_value(current_tier) > _tier_cap_value("C") else current_tier
        rec["is_recommended"] = False
        rec["bet_action"] = "observe"
        tags = rec.get("risk_tags", [])
        if not isinstance(tags, list):
            tags = [str(tags)] if tags else []
        tags.extend(tag_list)
        rec["risk_tags"] = list(dict.fromkeys(str(t) for t in tags if str(t).strip()))[:24]
        pred["recommendation_tier"] = rec["tier"]
        pred["recommend_gate_pass"] = False
        pred.setdefault("recommend_gate_reasons", []).append(reason)
        if not pred.get("confidence_downgrade_reason"):
            pred["confidence_downgrade_reason"] = downgrade_msg
        warnings.extend(tag_list)

    if valid_sources == 0 and needs_external_source:
        _cap_to_c_observe(
            "external_fact_without_source",
            ["external_fact_without_source", "missing_external_confirmation"],
            "missing_external_confirmation: 外部事实论据缺少有效来源，禁止升权。",
        )

    # P1 硬闸 3: evidence_quality 低于 50 从单纯 warning 升级为硬降级。
    if 0 < evidence_quality < 50:
        warnings.append("external_evidence_quality_below_50")
        if needs_external_source:
            _cap_to_c_observe(
                "external_evidence_quality_below_50",
                ["external_evidence_quality_below_50"],
                "external_evidence_quality_below_50: 证据质量<50，禁止依赖外部事实升权。",
            )

    # P1 硬闸 2: source_conflict_audit 未解决冲突 -> 硬降级。
    if _has_unresolved_source_conflict(pred) and needs_external_source:
        _cap_to_c_observe(
            "unresolved_source_conflict",
            ["unresolved_source_conflict"],
            "unresolved_source_conflict: 来源冲突未解决，禁止升权。",
        )

    # P1 硬闸 1: external_fact 全部 stale/过旧 -> 硬降级。
    if _all_external_facts_stale(pred) and needs_external_source:
        _cap_to_c_observe(
            "external_facts_all_stale",
            ["external_facts_all_stale"],
            "external_facts_all_stale: 外部事实均过时，禁止依赖陈旧情报升权。",
        )

    pred["validation_warnings"] = list(dict.fromkeys(warnings))
    return pred


def _has_unresolved_source_conflict(pred: Dict[str, Any]) -> bool:
    """P1: source_conflict_audit.has_conflict=true 且 conflicts 非空 => 未解决冲突。"""
    audit = pred.get("source_conflict_audit")
    if not isinstance(audit, dict):
        return False
    if not audit.get("has_conflict"):
        return False
    conflicts = audit.get("conflicts")
    return isinstance(conflicts, list) and len(conflicts) > 0


# 新鲜度: same_day/recent_3d/recent/live/fresh 视为新鲜; stale/expired/old 视为过时。
_FRESH_TOKENS = ("same_day", "recent_3d", "recent", "live", "fresh", "today")
_STALE_TOKENS = ("stale", "expired", "old", "outdated")


def _all_external_facts_stale(pred: Dict[str, Any]) -> bool:
    """P1: external_fact_table 非空且所有条目 freshness 都是 stale/过旧 => 返回 True。
    谨慎便: 只要有任一条 fresh 或 freshness 未知(unknown/空)就不判 stale，避免误杀。"""
    facts = pred.get("external_fact_table")
    if not isinstance(facts, list) or not facts:
        return False
    saw_any = False
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        fr = str(fact.get("freshness", "")).strip().lower()
        if not fr or fr == "unknown":
            return False  # 未知新鲜度不当 stale，不误杀
        if any(tok in fr for tok in _FRESH_TOKENS):
            return False  # 有新鲜来源 -> 不是全 stale
        if any(tok in fr for tok in _STALE_TOKENS):
            saw_any = True
            continue
        return False  # 不识别的 freshness 值保守不判 stale
    return saw_any


def _source_quality_floor_and_cap(web_research: Dict[str, Any], external_fact_table: Any = None) -> Tuple[int, List[str]]:
    warnings: List[str] = []
    sources = web_research.get("sources", []) if isinstance(web_research, dict) else []
    total_sources = len(sources) if isinstance(sources, list) else 0
    valid_sources = _valid_external_source_count(web_research, external_fact_table)
    if valid_sources <= 0:
        if total_sources:
            warnings.append("external_source_url_missing_or_invalid")
            return 40, warnings
        warnings.append("missing_external_confirmation")
        return 30, warnings
    if valid_sources == 1:
        return 75, warnings
    return 100, warnings


def _normalize_external_evidence_quality(pred: Dict[str, Any]) -> int:
    web = pred.get("web_research", {}) if isinstance(pred.get("web_research"), dict) else {}
    raw_score = int(_clip(_f(pred.get("evidence_quality_score", 0), 0), 0, 100))
    cap, warnings = _source_quality_floor_and_cap(web, pred.get("external_fact_table"))
    normalized = min(raw_score, cap) if raw_score else 0
    existing = pred.setdefault("validation_warnings", [])
    if not isinstance(existing, list):
        existing = [str(existing)] if existing else []
    existing.extend(warnings)
    pred["validation_warnings"] = list(dict.fromkeys(existing))
    pred["evidence_quality_score"] = normalized
    return normalized


def _direction_probability_for(pred: Dict[str, Any], direction: str) -> float:
    probs = pred.get("direction_probs", {}) if isinstance(pred.get("direction_probs"), dict) else {}
    value = _f(probs.get(direction), 0.0)
    if value:
        return value
    pct_keys = {"home": "home_win_pct", "draw": "draw_pct", "away": "away_win_pct"}
    return _f(pred.get(pct_keys.get(direction, "")), 0.0)


def _max_direction_probability(pred: Dict[str, Any]) -> Tuple[str, float]:
    probs = pred.get("direction_probs", {}) if isinstance(pred.get("direction_probs"), dict) else {}
    rows = [(d, _direction_probability_for({**pred, "direction_probs": probs}, d)) for d in VALID_DIRS]
    return max(rows, key=lambda x: x[1]) if rows else ("", 0.0)


def _v2_candidate_score_prob(candidate: Any) -> Tuple[str, float]:
    if isinstance(candidate, dict):
        return _score_from_candidate(candidate.get("score")), _f(candidate.get("prob"), 0.0)
    if isinstance(candidate, (list, tuple)) and candidate:
        return _score_from_candidate(candidate[0]), _f(candidate[1] if len(candidate) > 1 else 0.0, 0.0)
    return _score_from_candidate(candidate), 0.0


def _cap_to_observe(pred: Dict[str, Any], reason: str, max_tier: str = "C") -> None:
    rec = pred.setdefault("recommendation", {}) if isinstance(pred.get("recommendation"), dict) else {}
    if not isinstance(pred.get("recommendation"), dict):
        pred["recommendation"] = rec
    current_tier = str(rec.get("tier", pred.get("recommendation_tier", "D"))).upper()
    rec["tier"] = max_tier if _tier_cap_value(current_tier) > _tier_cap_value(max_tier) else current_tier
    rec["is_recommended"] = False
    rec["bet_action"] = "observe"
    pred["recommendation_tier"] = rec["tier"]
    pred["recommend_gate_pass"] = False
    reasons = pred.setdefault("recommend_gate_reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)] if reasons else []
        pred["recommend_gate_reasons"] = reasons
    reasons.append(reason)
    warnings = pred.setdefault("validation_warnings", [])
    if not isinstance(warnings, list):
        warnings = [str(warnings)] if warnings else []
        pred["validation_warnings"] = warnings
    warnings.append(reason)
    pred["recommend_gate_reasons"] = list(dict.fromkeys(reasons))
    pred["validation_warnings"] = list(dict.fromkeys(warnings))


def _apply_direction_candidate_consistency_gate(pred: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(pred, dict) or pred.get("is_abstain"):
        return pred
    final_dir = str(pred.get("final_direction", ""))
    if final_dir in VALID_DIRS:
        max_dir, max_prob = _max_direction_probability(pred)
        final_prob = _direction_probability_for(pred, final_dir)
        if max_dir in VALID_DIRS and max_dir != final_dir and max_prob - final_prob >= 3.0:
            _cap_to_observe(pred, "direction_probability_not_supporting_final_direction", "C")
    top3 = pred.get("top3", []) if isinstance(pred.get("top3"), list) else []
    if not top3:
        top3 = pred.get("top_score_candidates", []) if isinstance(pred.get("top_score_candidates"), list) else []
    if len(top3) >= 2:
        first_score, first_prob = _v2_candidate_score_prob(top3[0])
        predicted_score = _score_from_candidate(pred.get("predicted_score"))
        higher_later = any(_v2_candidate_score_prob(x)[1] > first_prob + 0.01 for x in top3[1:])
        if first_score == predicted_score and higher_later:
            _cap_to_observe(pred, "top3_probability_order_conflict", "C")
    return pred


CONTRARIAN_MARKET_TERMS = ["反向steam", "steam", "造热", "大热必死", "诱盘", "反打", "聪明钱", "sharp", "rlm", "reverse line"]
MARKET_SOURCE_TERMS = ["market", "odds", "盘口", "赔率", "亚盘", "大小球", "market_snapshot", "pinnacle", "bet365", "william"]


def _has_valid_market_source(pred: Dict[str, Any]) -> bool:
    web = pred.get("web_research", {}) if isinstance(pred.get("web_research"), dict) else {}
    sources = web.get("sources", []) if isinstance(web.get("sources", []), list) else []
    facts = pred.get("external_fact_table", []) if isinstance(pred.get("external_fact_table", []), list) else []
    for row in sources + facts:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url", row.get("source_url", ""))).strip()
        if not url or url == "#":
            continue
        text = _json_compact(row, 1200).lower()
        if any(term.lower() in text for term in MARKET_SOURCE_TERMS):
            return True
    return False


def _apply_contrarian_market_claim_gate(pred: Dict[str, Any]) -> Dict[str, Any]:
    text = _external_context_text(pred)
    if any(term.lower() in text for term in CONTRARIAN_MARKET_TERMS) and not _has_valid_market_source(pred):
        _cap_to_observe(pred, "contrarian_market_claim_without_valid_market_source", "C")
    return pred


# === 下注推荐模块 bet_recommendation (2026-06-27) ===
# 本地确定性纯函数：基于真实盘口赔率算各玩法期望值，产出 激进/稳健 两套下注组合。
# 严格遵守本地闸门原则：只新增 pred["bet_recommendation"] 字段，
# 绝不修改 AI 终审的 final_direction / predicted_score / recommendation.tier / bet_action。
# 不调用 AI、不联网、无随机；同输入同输出，可单测可回归。

BET_RECO_BUDGET = 200          # 每场总预算(元)
BET_RECO_MIN_STAKE = 20        # 单注下限(元)
BET_RECO_STAKE_STEP = 5        # 金额取整步长(元)
BET_RECO_NO_BET_ACTIONS = {"no_bet", "observe"}
BET_RECO_DISCLAIMER = "算法推荐，非盈利保证；金额仅为基于期望值的分配建议，是否下注请自行决定。"


def _bet_score_zh(score: str) -> str:
    """正确比分中文标签，区分 0-0/1-1/2-2 平局三态。"""
    return str(score)


def _extract_market_odds(match_obj: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """把 match_obj 原始赔率规整为 {market: {selection: odds}}。无赔率的玩法不生成键。"""
    if not isinstance(match_obj, dict):
        match_obj = {}
    out: Dict[str, Dict[str, float]] = {}

    # 正确比分
    cs: Dict[str, float] = {}
    for sc, key in CRS_FULL_MAP.items():
        o = _f(match_obj.get(key), 0.0)
        if o > 1.01:
            cs[sc] = o
    if cs:
        out["correct_score"] = cs

    # 胜平负 1X2
    onex2: Dict[str, float] = {}
    h = _f(match_obj.get("sp_home", match_obj.get("win")), 0.0)
    d = _f(match_obj.get("sp_draw", match_obj.get("same")), 0.0)
    a = _f(match_obj.get("sp_away", match_obj.get("lose")), 0.0)
    if h > 1.01:
        onex2["home"] = h
    if d > 1.01:
        onex2["draw"] = d
    if a > 1.01:
        onex2["away"] = a
    if onex2:
        out["one_x_two"] = onex2

    # 总进球数 a0-a7
    tg: Dict[str, float] = {}
    for n in range(8):
        o = _f(match_obj.get(f"a{n}"), 0.0)
        if o > 1.01:
            tg[str(n)] = o
    if tg:
        out["total_goals"] = tg
        # 推导大小球 2.5：小=总进球0/1/2 赔率合成，大=3+ 合成（用最低赔近似，仅作展示参考）
        under = [tg[k] for k in ("0", "1", "2") if k in tg]
        over = [tg[k] for k in ("3", "4", "5", "6", "7") if k in tg]
        ou: Dict[str, float] = {}
        if under:
            # 合成赔率近似：1 / Σ(1/odds)
            inv = sum(1.0 / x for x in under if x > 1.0)
            if inv > 0:
                ou["under_2.5"] = round(1.0 / inv, 2)
        if over:
            inv = sum(1.0 / x for x in over if x > 1.0)
            if inv > 0:
                ou["over_2.5"] = round(1.0 / inv, 2)
        if ou:
            out["over_under"] = ou

    # 半全场 HFTF（缺字段跳过）
    hf: Dict[str, float] = {}
    for code, label in HFTF_MAP.items():
        o = _f(match_obj.get(code), 0.0)
        if o > 1.01:
            hf[label] = o
    if hf:
        out["half_full"] = hf

    return out


def _bet_p_model_for_score(pred: Dict[str, Any], score: str) -> float:
    """从终审 top3 取该比分的模型主观概率(0-1)；无则给保守值。"""
    top3 = pred.get("top3") if isinstance(pred.get("top3"), list) else []
    for t in top3:
        if isinstance(t, dict) and str(t.get("score")) == str(score):
            return _clip(_f(t.get("prob"), 0.0) / 100.0, 0.0, 0.95)
    # 不在 top3：给一个低保守概率
    return 0.08


def _bet_goal_band_main(pred: Dict[str, Any]) -> Optional[int]:
    gb = str(pred.get("goal_band", "")).strip()
    # goal_band 形如 "0-1/2/3/4+" 或单值；取 predicted_score 总进球更稳
    ps = str(pred.get("predicted_score", ""))
    tot = _score_total(ps)
    if tot is not None:
        return tot
    for tok in ("4", "3", "2", "1", "0"):
        if tok in gb:
            return int(tok)
    return None


def _build_bet_candidates(pred: Dict[str, Any], odds_map: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """生成候选腿 + p_model + ev_ratio。返回未分配金额的 leg 列表。"""
    legs: List[Dict[str, Any]] = []
    final_dir = str(pred.get("final_direction", "")).lower()
    dprobs = pred.get("direction_probs") if isinstance(pred.get("direction_probs"), dict) else {}
    predicted_score = str(pred.get("predicted_score", ""))

    def add(market: str, selection: str, odds: float, p_model: float, label: str, reason: str) -> None:
        if odds <= 1.01 or p_model <= 0:
            return
        ev = p_model * (odds - 1.0) - (1.0 - p_model)
        legs.append({
            "market": market,
            "selection": selection,
            "label": label,
            "odds": round(odds, 2),
            "p_model": round(p_model, 3),
            "ev_ratio": round(ev, 3),
            "reason": reason,
        })

    # 1) 正确比分：top3 + predicted_score + score_elimination_audit keep
    cs_odds = odds_map.get("correct_score", {})
    seen_scores = set()
    if cs_odds:
        elim = pred.get("score_elimination_audit") if isinstance(pred.get("score_elimination_audit"), dict) else {}
        cand_scores: List[str] = []
        if predicted_score:
            cand_scores.append(predicted_score)
        for t in (pred.get("top3") or []):
            if isinstance(t, dict) and t.get("score"):
                cand_scores.append(str(t.get("score")))
        for sc_key, verdict in elim.items():
            if isinstance(verdict, str) and verdict.strip().startswith("keep") and sc_key in cs_odds:
                cand_scores.append(sc_key)
        for sc in cand_scores:
            if sc in seen_scores or sc not in cs_odds:
                continue
            seen_scores.add(sc)
            p = _bet_p_model_for_score(pred, sc)
            tag = "主选比分" if sc == predicted_score else "候选比分"
            add("correct_score", sc, cs_odds[sc], p, f"比分 {_bet_score_zh(sc)}",
                f"{tag}，赔率{cs_odds[sc]:.2f}，模型命中估计{p*100:.0f}%")

    # 2) 胜平负
    onex2 = odds_map.get("one_x_two", {})
    if onex2 and final_dir in onex2:
        p = _clip(_f(dprobs.get(final_dir), 0.0) / 100.0, 0.0, 0.95)
        if p <= 0:
            p = 0.4
        zh = {"home": "主胜", "draw": "平局", "away": "客胜"}.get(final_dir, final_dir)
        add("one_x_two", final_dir, onex2[final_dir], p, zh,
            f"终审方向{zh}，胜平负赔率{onex2[final_dir]:.2f}，方向概率{p*100:.0f}%")

    # 3) 总进球数（主选 band）
    tg = odds_map.get("total_goals", {})
    main_band = _bet_goal_band_main(pred)
    if tg and main_band is not None and str(main_band) in tg:
        add("total_goals", str(main_band), tg[str(main_band)], 0.42, f"总进球{main_band}球",
            f"主选总进球{main_band}球，赔率{tg[str(main_band)]:.2f}")

    # 4) 大小球 2.5
    ou = odds_map.get("over_under", {})
    if ou and main_band is not None:
        if main_band >= 3 and "over_2.5" in ou:
            add("over_under", "over_2.5", ou["over_2.5"], 0.45, "大2.5",
                f"主选总进球{main_band}球→大2.5，合成赔率{ou['over_2.5']:.2f}")
        elif main_band <= 2 and "under_2.5" in ou:
            add("over_under", "under_2.5", ou["under_2.5"], 0.50, "小2.5",
                f"主选总进球{main_band}球→小2.5，合成赔率{ou['under_2.5']:.2f}")

    # 5) 半全场（保守：方向胜→平/主 或 主/主；缺字段已跳过）
    hf = odds_map.get("half_full", {})
    if hf and final_dir in ("home", "away"):
        tempo = str((pred.get("contextual_logic") or {}).get("tempo", "")).lower()
        slow = tempo in ("low", "medium") or "慢热" in str(pred.get("reason", ""))
        if final_dir == "home":
            pick = "平/主" if slow else "主/主"
        else:
            pick = "平/负" if slow else "负/负"
        if pick in hf:
            add("half_full", pick, hf[pick], 0.22, f"半全场{pick}",
                f"方向{final_dir}+{'慢热' if slow else '强势'}→{pick}，赔率{hf[pick]:.2f}")

    return legs


def _bet_round_stake(x: float) -> int:
    """向下取整到 BET_RECO_STAKE_STEP 的倍数。"""
    return int(x // BET_RECO_STAKE_STEP) * BET_RECO_STAKE_STEP


def _allocate_budget(legs: List[Dict[str, Any]], budget: int, min_stake: int, weight_key: str) -> List[Dict[str, Any]]:
    """预算分配：按 weight_key 加权→单注下限钳制→取整到步长→余额补给最高权重腿。
    会就地砍掉无法满足 min_stake 的最弱腿。返回带 stake/potential_payout 的腿列表。"""
    if not legs:
        return []
    work = [dict(l) for l in legs]
    # 权重(截断为正)
    def w_of(l: Dict[str, Any]) -> float:
        return max(_f(l.get(weight_key), 0.0), 0.01)

    # 最多容纳腿数
    max_legs = budget // min_stake
    # 按权重降序，超出容量先砍最弱
    work.sort(key=w_of, reverse=True)
    if len(work) > max_legs:
        work = work[:max_legs]

    while work:
        total_w = sum(w_of(l) for l in work)
        if total_w <= 0:
            break
        # 初始分配
        raw = [budget * w_of(l) / total_w for l in work]
        # 若最小分配 < min_stake，砍掉权重最弱腿后重算
        min_idx = min(range(len(work)), key=lambda i: raw[i])
        if raw[min_idx] < min_stake and len(work) > 1:
            del work[min_idx]
            continue
        # 取整到步长
        stakes = [max(_bet_round_stake(r), min_stake) for r in raw]
        # 确保不超预算：从最低权重腿往下削
        total = sum(stakes)
        order_low = sorted(range(len(work)), key=lambda i: w_of(work[i]))
        gi = 0
        while total > budget and gi < len(order_low):
            i = order_low[gi]
            if stakes[i] - BET_RECO_STAKE_STEP >= min_stake:
                stakes[i] -= BET_RECO_STAKE_STEP
                total -= BET_RECO_STAKE_STEP
            else:
                gi += 1
        if total > budget and len(work) > 1:
            # 仍超：砍最弱腿重算
            del work[order_low[0]]
            continue
        # 余额补给最高权重腿（取整步长）
        remain = budget - sum(stakes)
        if remain >= BET_RECO_STAKE_STEP:
            best_i = max(range(len(work)), key=lambda i: w_of(work[i]))
            stakes[best_i] += _bet_round_stake(remain)
        for l, s in zip(work, stakes):
            l["stake"] = int(s)
            l["potential_payout"] = round(s * _f(l.get("odds"), 0.0), 1)
        return work
    return []


def _bet_combo_expected_return(legs: List[Dict[str, Any]]) -> float:
    """组合期望返还 = Σ stake_i * odds_i * p_model_i（腿间独立近似）。"""
    return round(sum(_f(l.get("stake")) * _f(l.get("odds")) * _f(l.get("p_model")) for l in legs), 1)


def _apply_bet_recommendation_gate(pred: Dict[str, Any], match_obj: Dict[str, Any]) -> Dict[str, Any]:
    """主入口：组装 steady/aggressive 两套组合，写入 pred["bet_recommendation"]。
    只新增字段，不改任何终审判断。"""
    reco = pred.get("recommendation") if isinstance(pred.get("recommendation"), dict) else {}
    bet_action = str(reco.get("bet_action", "")).lower()

    base = {
        "available": True,
        "per_match_budget": BET_RECO_BUDGET,
        "min_stake": BET_RECO_MIN_STAKE,
        "disclaimer": BET_RECO_DISCLAIMER,
        "default_view": "aggressive",
        "no_bet": False,
        "no_bet_reason": "",
        "version": "bet_reco_v1",
    }

    # 终审不建议下注：两套都置空
    if bet_action in BET_RECO_NO_BET_ACTIONS or not bool(reco.get("is_recommended", True)):
        base["no_bet"] = True
        base["no_bet_reason"] = f"终审推荐为 {bet_action or '不推荐'}，本场不建议下注"
        base["steady"] = {"total_stake": 0, "expected_return": 0.0, "reason": base["no_bet_reason"], "legs": []}
        base["aggressive"] = {"total_stake": 0, "expected_return": 0.0, "reason": base["no_bet_reason"], "legs": []}
        pred["bet_recommendation"] = base
        return pred

    odds_map = _extract_market_odds(match_obj)
    if not odds_map:
        base["available"] = False
        base["no_bet"] = True
        base["no_bet_reason"] = "无可用盘口赔率字段，无法生成下注推荐"
        base["steady"] = {"total_stake": 0, "expected_return": 0.0, "reason": base["no_bet_reason"], "legs": []}
        base["aggressive"] = {"total_stake": 0, "expected_return": 0.0, "reason": base["no_bet_reason"], "legs": []}
        pred["bet_recommendation"] = base
        return pred

    all_legs = _build_bet_candidates(pred, odds_map)

    # 稳健：p_model≥0.35 且 ev_ratio≥-0.05，排除高赔长尾 odds>8
    steady_pool = [l for l in all_legs if _f(l.get("p_model")) >= 0.35 and _f(l.get("ev_ratio")) >= -0.05 and _f(l.get("odds")) <= 8.0]
    steady_pool.sort(key=lambda l: _f(l.get("p_model")), reverse=True)
    steady_pool = steady_pool[:3]
    steady_legs = _allocate_budget(steady_pool, BET_RECO_BUDGET, BET_RECO_MIN_STAKE, "p_model")

    # 激进：ev_ratio>0，允许高赔长尾
    agg_pool = [l for l in all_legs if _f(l.get("ev_ratio")) > 0]
    agg_pool.sort(key=lambda l: _f(l.get("ev_ratio")), reverse=True)
    agg_pool = agg_pool[:4]
    agg_legs = _allocate_budget(agg_pool, BET_RECO_BUDGET, BET_RECO_MIN_STAKE, "ev_ratio")

    if steady_legs:
        base["steady"] = {
            "total_stake": int(sum(_i(l.get("stake")) for l in steady_legs)),
            "expected_return": _bet_combo_expected_return(steady_legs),
            "reason": "稳健组合：命中率优先，选模型概率较高且非高赔长尾的标的，控制波动。",
            "legs": steady_legs,
        }
    else:
        base["steady"] = {"total_stake": 0, "expected_return": 0.0, "reason": "无满足稳健条件(命中率≥35%且非高赔长尾)的标的。", "legs": []}

    if agg_legs:
        base["aggressive"] = {
            "total_stake": int(sum(_i(l.get("stake")) for l in agg_legs)),
            "expected_return": _bet_combo_expected_return(agg_legs),
            "reason": "激进组合：期望值优先，纳入正期望的高赔标的(含大比分长尾)，追求收益最大化，命中率较低。",
            "legs": agg_legs,
        }
    else:
        base["aggressive"] = {"total_stake": 0, "expected_return": 0.0, "reason": "无正期望值标的，激进组合空仓。", "legs": []}

    # 若激进空但稳健有，默认视图回退稳健
    if not agg_legs and steady_legs:
        base["default_view"] = "steady"

    pred["bet_recommendation"] = base
    return pred


# === 审计修复 2026-06-21 (根据世界杯小组赛真实赛果对账) ===
# 背景：现网 6/20 完赛 4 场，方向 3/4 准但比分 0/4，错误全部是
# “给强弱悬殊场的负方多给了一个安慰球”(4-1→实3-0, 2-1→实2-0)，
# 以及唯一方向翻车落在低信心平局档(土耳其 conf=41 戦1-1→实0-1)。


def _apply_lopsided_consolation_goal_gate(pred: Dict[str, Any]) -> Dict[str, Any]:
    """Fix1 (确定性后处理): 强弱悬殊、高信心胜场，压掉负方的单个安慰球。

    仅在同时满足时生效(保守，避免 n=4 过拟合)：
      - final_direction 为 home/away 胜
      - 胜方信心 ≥ 70 (强热)
      - 负方进球 == 1 且胜方进球 ≥ 3 (净胜 ≥2 的“安慰球”形态)
    动作：负方 1→0 (4-1→4-0, 3-1→3-0)，保持 final_direction 不变，
    同步 goal_band/btts/top3[0]，并在 validation_warnings 记录溯源。
    不修改推荐等级(只改比分形态，不动闸门)。
    """
    if not isinstance(pred, dict) or pred.get("is_abstain"):
        return pred
    final_dir = str(pred.get("final_direction", ""))
    if final_dir not in ("home", "away"):
        return pred
    score = _score_from_candidate(pred.get("predicted_score"))
    h, a = _parse_score(score)
    if h is None or a is None:
        return pred
    winner, loser = (h, a) if final_dir == "home" else (a, h)
    if not (loser == 1 and winner >= 3):
        return pred
    _, conf = _max_direction_probability(pred)
    if conf < 70.0:
        return pred
    new_h, new_a = (h, a - 1) if final_dir == "home" else (h - 1, a)
    new_score = f"{new_h}-{new_a}"
    if _score_direction(new_score) != final_dir:
        return pred  # 安全网：绝不能改变方向
    old_score = score
    pred["predicted_score"] = new_score
    try:
        pred["goal_band"] = _score_goal_band(new_score)
        pred["btts"] = _score_btts(new_score)
    except Exception:
        pass
    top3 = pred.get("top3")
    if isinstance(top3, list) and top3 and isinstance(top3[0], dict):
        if _score_from_candidate(top3[0].get("score")) == old_score:
            top3[0]["score"] = new_score
            top3[0].setdefault("logic", "")
            top3[0]["logic"] = (str(top3[0]["logic"]) + " |audit_consolation_goal_compressed").strip()
    warnings = pred.setdefault("validation_warnings", [])
    if not isinstance(warnings, list):
        warnings = [str(warnings)] if warnings else []
        pred["validation_warnings"] = warnings
    warnings.append(f"lopsided_consolation_goal_compressed:{old_score}->{new_score}")
    pred["validation_warnings"] = list(dict.fromkeys(warnings))
    pred["score_shape_calibrated"] = True
    return pred





def _apply_low_confidence_draw_guard(pred: Dict[str, Any]) -> Dict[str, Any]:
    """Fix3 (复用硬闸): 低信心平局预测降为 observe，不计入实单。

    触发：final_direction==draw 且平局信心 < 45。
    依据：本次唯一方向翻车(土耳其 conf=41 戦1-1→实0-1)落在此档。
    与现有 _apply_direction_candidate_consistency_gate 同风格，调用 _cap_to_observe。
    """
    if not isinstance(pred, dict) or pred.get("is_abstain"):
        return pred
    if str(pred.get("final_direction", "")) != "draw":
        return pred
    draw_conf = _direction_probability_for(pred, "draw")
    if draw_conf < 45.0:
        _cap_to_observe(pred, f"low_confidence_draw_observe:conf={draw_conf:.0f}<45", "C")
    return pred


def _sync_gate_with_bet_action(pred: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(pred, dict) or pred.get("is_abstain"):
        return pred
    rec = pred.get("recommendation", {}) if isinstance(pred.get("recommendation"), dict) else {}
    action = str(rec.get("bet_action") or "").lower()
    if action and action not in BETTABLE_ACTIONS:
        rec["is_recommended"] = False
        pred["recommend_gate_pass"] = False
        reasons = pred.setdefault("recommend_gate_reasons", [])
        if not isinstance(reasons, list):
            reasons = [str(reasons)] if reasons else []
            pred["recommend_gate_reasons"] = reasons
        reasons.append("gate_action_not_bettable")
        pred["recommend_gate_reasons"] = list(dict.fromkeys(reasons))
    return pred


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
        raw_direction_value = item.get("final_direction", item.get("direction", ""))
        parsed_raw_dir = _dir_from_any(raw_direction_value)
        raw_dir = parsed_raw_dir or score_dir or "draw"
        final_direction = score_dir or raw_dir
        direction_conflict = bool(score_dir in VALID_DIRS and raw_dir in VALID_DIRS and score_dir != raw_dir)
        top3 = _normalize_top3(item, predicted_score)
        direction_probs = _normalize_direction_probs(item)
        web = _normalize_web_research(item)
        rec = _normalize_recommendation(item)
        external_facts = _copy_external_fact_fields_from_item(item)
        full_spectrum_audits = _copy_full_spectrum_audit_fields_from_item(item)
        warnings = []
        if direction_conflict:
            warnings.append(f"dir_score_conflict_protocol_fixed:{raw_dir}->{score_dir}")
        if raw_direction_value not in (None, "") and parsed_raw_dir is None:
            warnings.append(f"invalid_final_direction_protocol_fixed:{str(raw_direction_value)[:40]}->{final_direction}")
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
            **external_facts,
            **full_spectrum_audits,
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
    phase1_returns = []
    if AI_PHASE1_PARALLEL:
        tasks = []
        for ai in PHASE1_NAMES:
            prompt = build_phase1_prompt(evidence_batch, ai)
            tasks.append(async_call_ai_json(session, ai, _phase1_system(ai), prompt, "phase1", expected))
        phase1_returns = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        for ai in PHASE1_NAMES:
            prompt = build_phase1_prompt(evidence_batch, ai)
            try:
                phase1_returns.append(await async_call_ai_json(session, ai, _phase1_system(ai), prompt, "phase1", expected))
            except Exception as exc:
                phase1_returns.append(exc)
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
        if AI_PHASE1_PARALLEL:
            critic_tasks = []
            for ai in PHASE1_NAMES:
                prompt = build_critic_prompt(evidence_batch, ai, phase1)
                critic_tasks.append(async_call_ai_json(session, ai, _phase1_system(ai), prompt, "critic", expected))
            critic_returns = await asyncio.gather(*critic_tasks, return_exceptions=True)
        else:
            critic_returns = []
            for ai in PHASE1_NAMES:
                prompt = build_critic_prompt(evidence_batch, ai, phase1)
                try:
                    critic_returns.append(await async_call_ai_json(session, ai, _phase1_system(ai), prompt, "critic", expected))
                except Exception as exc:
                    critic_returns.append(exc)
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
    _, final_obj, final_st = await async_call_ai_json_with_retry(session, final_ai, _phase1_system("gemini"), final_prompt, "final", expected, AI_FINAL_RETRY_MAX)
    final_rows = normalize_ai_predictions(final_obj, expected, final_ai, "final")
    print(f"  [Chunk {chunk_id}] {final_ai.upper()} final {len(final_rows)}/{len(expected)} status={final_st.get('status')}")

    missing = [idx for idx in expected if idx not in final_rows]
    if missing and AI_ENABLE_FALLBACK_REFEREE:
        fallback_ai = AI_FALLBACK_REFEREE_MODEL if AI_FALLBACK_REFEREE_MODEL in PHASE1_NAMES else "gpt"
        fb_prompt = build_fallback_referee_prompt([e for e in evidence_batch if e["match"] in missing], phase1, critic_reports)
        _, fb_obj, fb_st = await async_call_ai_json_with_retry(session, fallback_ai, _phase1_system(fallback_ai), fb_prompt, "fallback_referee", missing, AI_FINAL_RETRY_MAX)
        fb_rows = normalize_ai_predictions(fb_obj, missing, fallback_ai, "fallback_referee")
        final_rows.update(fb_rows)
        print(f"  [Chunk {chunk_id}] FALLBACK {fallback_ai.upper()} {len(fb_rows)}/{len(missing)} status={fb_st.get('status')}")

    # Plan B+: Gemini final and standard fallback both failed -> GPT runs a 16-role
    # family debate as the final referee (decides the score), before any abstain.
    missing = [idx for idx in expected if idx not in final_rows]
    if missing and AI_ENABLE_FAMILY_DEBATE_REFEREE:
        debate_ai = AI_FAMILY_DEBATE_MODEL if AI_FAMILY_DEBATE_MODEL in AI_NAMES else "gpt"
        debate_prompt = build_family_debate_referee_prompt([e for e in evidence_batch if e["match"] in missing], phase1, critic_reports)
        _, db_obj, db_st = await async_call_ai_json_with_retry(session, debate_ai, _phase1_system(debate_ai), debate_prompt, "family_debate_referee", missing, AI_FINAL_RETRY_MAX)
        db_rows = normalize_ai_predictions(db_obj, missing, debate_ai, "family_debate_referee")
        for idx, row in db_rows.items():
            row.setdefault("validation_warnings", []).append("final_by_gpt_family_debate_referee")
        final_rows.update(db_rows)
        print(f"  [Chunk {chunk_id}] FAMILY-DEBATE {debate_ai.upper()} {len(db_rows)}/{len(missing)} status={db_st.get('status')}")

    missing = [idx for idx in expected if idx not in final_rows]
    if missing:
        if AI_ALLOW_PHASE1_FINAL_FALLBACK:
            for idx in missing:
                final_rows[idx] = _phase1_consensus_fallback(idx, phase1)
            print(f"  [Chunk {chunk_id}] Phase1 consensus fallback filled {len(missing)} (explicitly allowed)")
        else:
            for idx in missing:
                final_rows[idx] = _abstain_ai_prediction(idx, "final_referee_missing_no_phase1_fallback")
            print(f"  [Chunk {chunk_id}] Final referee missing; abstained {len(missing)} instead of promoting phase1")

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
        warnings.append(f"protocol_score_direction_conflict_preserved:{raw_dir}!={score_dir}")
        r["score_direction_conflict"] = True
    elif score_dir in VALID_DIRS and raw_dir not in VALID_DIRS:
        warnings.append(f"protocol_direction_filled_from_score:{score_dir}")
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


def _score_shape_selector(pred: Dict[str, Any], match_obj: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Re-rank score candidates without changing the final direction.

    This is the missing score-shape layer between AI single-score output and
    recommendation gates. It consumes the existing AI/market candidates and
    only promotes scores that preserve final_direction.
    """
    if not isinstance(pred, dict) or pred.get("is_abstain"):
        return pred
    final_dir = _dir_from_any(pred.get("final_direction"))
    score = _score_from_candidate(pred.get("predicted_score"))
    h, a = _parse_score(score)
    if final_dir not in VALID_DIRS or h is None or a is None:
        return pred

    raw_item = pred.get("raw_item", {}) if isinstance(pred.get("raw_item"), dict) else {}
    candidates = _collect_score_candidates_for_gate(pred, raw_item) if "_collect_score_candidates_for_gate" in globals() else []
    candidate_scores = []
    for c in candidates:
        sc = _score_from_candidate(c.get("score"))
        if _parse_score(sc)[0] is not None and sc not in candidate_scores:
            candidate_scores.append(sc)
    if score not in candidate_scores:
        candidate_scores.insert(0, score)

    goal_band = str(pred.get("goal_band", "")).strip().lower()
    btts_yes = _raw_btts_yes_signal(pred, raw_item) or str(pred.get("btts", "")).strip().lower() in {"yes", "y", "true", "是"}
    compact = _json_compact({
        "tail_risk_flags": pred.get("tail_risk_flags") or raw_item.get("tail_risk_flags"),
        "score_cluster_audit": pred.get("score_cluster_audit") or raw_item.get("score_cluster_audit"),
        "goal_market_audit": pred.get("goal_market_audit") or raw_item.get("goal_market_audit"),
    }, 2500).lower()
    high_goal_tail = goal_band in {"4+", "high", "4"} or any(t in compact for t in ["4+", "high_btts", "高比分", "tail", "尾部"])
    low_goal_signal = goal_band in {"0-1", "0", "1", "low"} or any(t in compact for t in ["0-0", "low", "闷局", "低比分"])

    promoted = ""
    reason = ""

    # 1) Clean-sheet over-anchor: promote same-direction BTTS candidate when
    # the AI/market evidence already surfaced BTTS or fight-back tails.
    if final_dir == "home" and a == 0 and btts_yes:
        for sc in [f"{h}-{1}", f"{max(h + 1, 2)}-1"]:
            if _score_direction(sc) == final_dir and sc in candidate_scores:
                promoted, reason = sc, "score_shape_selector_btts_clean_sheet_uplift"
                break
    elif final_dir == "away" and h == 0 and btts_yes:
        for sc in [f"1-{a}", f"1-{max(a + 1, 2)}"]:
            if _score_direction(sc) == final_dir and sc in candidate_scores:
                promoted, reason = sc, "score_shape_selector_btts_clean_sheet_uplift"
                break

    # 2) Draw band selector: keep draw direction but allow low/high draw bands
    # when existing evidence explicitly points away from the 1-1 default.
    if not promoted and final_dir == "draw":
        draw_order = [score]
        if high_goal_tail:
            draw_order = ["2-2", "3-3", score, "1-1", "0-0"]
        elif low_goal_signal:
            draw_order = ["0-0", score, "1-1", "2-2"]
        for sc in draw_order:
            if _score_direction(sc) == "draw" and (sc == score or sc in candidate_scores):
                promoted = sc
                if promoted != score:
                    reason = "score_shape_selector_draw_band_rerank"
                break

    if promoted and promoted != score and _score_direction(promoted) == final_dir:
        old_score = score
        pred["predicted_score"] = promoted
        pred["goal_band"] = _score_goal_band(promoted)
        pred["btts"] = _score_btts(promoted)
        pred["score_shape_calibrated"] = True
        pred.setdefault("score_shape_selector", {})
        pred["score_shape_selector"].update({"old_score": old_score, "new_score": promoted, "reason": reason})
        pred.setdefault("validation_warnings", []).append(f"{reason}:{old_score}->{promoted}")

    # Preserve a candidate distribution for consumers; do not force the old
    # predicted_score into rank #1.
    dist = []
    seen = set()
    for sc in [pred.get("predicted_score"), *candidate_scores]:
        sc = _score_from_candidate(sc)
        if _parse_score(sc)[0] is None or sc in seen:
            continue
        if _score_direction(sc) != final_dir:
            continue
        seen.add(sc)
        dist.append({"score": sc, "prob": 0.0, "logic": "score_shape_selector_candidate"})
        if len(dist) >= 5:
            break
    if dist:
        pred["score_distribution"] = dist
        pred["top3"] = dist[:3]
    pred["validation_warnings"] = list(dict.fromkeys(pred.get("validation_warnings", [])))
    return pred


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
            connector = aiohttp.TCPConnector(limit=max(8, AI_MODEL_CONCURRENCY * 4), use_dns_cache=False, ttl_dns_cache=0, force_close=False)
            session = aiohttp.ClientSession(connector=connector)
        if AI_ENDPOINT_SLOT_QUEUE and len(evidence_all) > 1:
            worker_count = min(AI_ENDPOINT_SLOT_WORKERS, AI_ENDPOINT_MAX_SLOTS, len(evidence_all))
            queue: asyncio.Queue[Tuple[int, Dict[str, Any]]] = asyncio.Queue()
            for i, evidence in enumerate(evidence_all, 1):
                queue.put_nowait((i, evidence))

            async def _slot_worker(slot: int) -> None:
                token = AI_ENDPOINT_SLOT_OVERRIDE.set(slot)
                try:
                    while True:
                        try:
                            item_id, evidence = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            return
                        match_id = evidence.get("match")
                        print(f"  [Slot {slot}] start match={match_id} queue_item={item_id}")
                        try:
                            rows = await _run_one_chunk(session, run_id, item_id, [evidence])
                            all_final.update(rows)
                            print(f"  [Slot {slot}] done match={match_id}")
                        except Exception as exc:
                            print(f"  [Slot {slot}] failed match={match_id}: {exc}")
                        finally:
                            queue.task_done()
                finally:
                    AI_ENDPOINT_SLOT_OVERRIDE.reset(token)

            await asyncio.gather(*[_slot_worker(slot) for slot in range(1, worker_count + 1)])
        elif AI_CHUNK_CONCURRENCY <= 1 or len(chunks) <= 1:
            for i, chunk in enumerate(chunks, 1):
                rows = await _run_one_chunk(session, run_id, i, chunk)
                all_final.update(rows)
        else:
            semaphore = asyncio.Semaphore(AI_CHUNK_CONCURRENCY)

            async def _run_limited_chunk(i: int, chunk: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
                async with semaphore:
                    return await _run_one_chunk(session, run_id, i, chunk)

            tasks = [_run_limited_chunk(i, chunk) for i, chunk in enumerate(chunks, 1)]
            chunk_returns = await asyncio.gather(*tasks, return_exceptions=True)
            for i, ret in enumerate(chunk_returns, 1):
                if isinstance(ret, Exception):
                    print(f"  [Chunk {i}] failed under chunk concurrency: {ret}")
                    continue
                all_final.update(ret)
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
        "run_mode": AI_RUN_MODE,
        "research_mode": AI_RESEARCH_MODE,
        "native_web": AI_NATIVE_WEB,
        "effective_chunk_size": AI_CHUNK_SIZE,
        "effective_chunk_concurrency": AI_CHUNK_CONCURRENCY,
        "endpoint_slot_queue": AI_ENDPOINT_SLOT_QUEUE,
        "endpoint_slot_workers": AI_ENDPOINT_SLOT_WORKERS,
        "effective_model_concurrency": AI_MODEL_CONCURRENCY,
        "effective_phase1_parallel": AI_PHASE1_PARALLEL,
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
    if not ai_r:
        return _abstain_prediction("AI全部失败或最终弃权")
    if ai_r.get("predicted_score") == "弃权" or ai_r.get("final_direction") == "abstain":
        reason = str((ai_r.get("validation_warnings") or ["AI全部失败或最终弃权"])[0])
        return _merge_abstain_analysis(_abstain_prediction(reason), ai_r, reason)

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
    # P0-1: 大小球独立判定头(用a0-a7市场进球曲线), 不再被主比分总球数机械绑架
    ou_head = derive_ou_head(match_obj, score, goal_range=(gmin, gmax))
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
        "over_under_2_5": ou_head["over_under_2_5"] if ou_head["over_under_2_5"] is not None else ("大" if total_goals >= 3 else "小"),
        "ou_market_prob_over": ou_head["ou_market_prob_over"],
        "ou_score_conflict": ou_head["ou_score_conflict"],
        "ou_source": ou_head["ou_source"],
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


def _analysis_from_phase1(ai_r: Dict[str, Any]) -> Dict[str, Any]:
    phase1 = ai_r.get("phase1_model_outputs", {}) if isinstance(ai_r.get("phase1_model_outputs"), dict) else {}
    # Prefer a successfully parsed analyst row for display only. This must never
    # decide final score/direction when final referee is missing.
    for name in ["grok", "gpt", "gemini"]:
        row = phase1.get(name) if isinstance(phase1.get(name), dict) else None
        if row:
            return row
    return {}


def _merge_abstain_analysis(pred: Dict[str, Any], ai_r: Dict[str, Any], reason: str) -> Dict[str, Any]:
    if not isinstance(pred, dict) or not isinstance(ai_r, dict):
        return pred
    phase1 = ai_r.get("phase1_model_outputs", {}) if isinstance(ai_r.get("phase1_model_outputs"), dict) else {}
    display_row = _analysis_from_phase1(ai_r)

    # Keep abstain/no-bet semantics immutable. Only restore analysis/display fields.
    pred.update({
        "predicted_score": "弃权",
        "predicted_label": "弃权",
        "result": "弃权",
        "display_direction": "弃权",
        "final_direction": "abstain",
        "is_abstain": True,
        "confidence": 0,
        "recommend_gate_pass": False,
        "decision_source": "ai_abstain_final_referee_missing_analysis_preserved",
        "ai_authority_mode": "ai_native_web_no_local_football_judgement",
    })
    rec = pred.get("recommendation", {}) if isinstance(pred.get("recommendation"), dict) else {}
    tags = list(rec.get("risk_tags", [])) if isinstance(rec.get("risk_tags"), list) else []
    tags.append(reason)
    pred["recommendation"] = {
        "tier": "D",
        "is_recommended": False,
        "top4_priority": 999,
        "bet_confidence": 0,
        "risk_level": "high",
        "risk_tags": list(dict.fromkeys(str(x) for x in tags if str(x).strip())),
        "why_recommended": "终审缺失，禁止输出比分；仅保留初审分析供审计。",
    }
    pred["recommendation_tier"] = "D"
    pred["recommend_gate_reasons"] = list(dict.fromkeys([reason] + list(pred.get("recommend_gate_reasons", []))))

    for k in [
        "phase1_model_outputs", "critic_reports_by_model", "validation_warnings",
        "score_cluster_audit", "sharp_money_audit", "recommendation_components",
        "risk_score_candidates", "tail_risk_flags", "confidence_downgrade_reason",
        "market_audit", "goal_market_audit", "market_conflicts", "candidate_scores",
        "public_heat_audit", "packet_news_risk_audit", "trap_candidates", "final_score_audit",
        "family_debate",
    ]:
        if ai_r.get(k) not in (None, {}, []):
            pred[k] = ai_r.get(k)

    for k in [
        "anchor_audit", "market_interpretation", "money_flow", "contextual_logic",
        "rejected_cases", "web_research", "final_web_audit", "data_quality",
    ]:
        v = ai_r.get(k)
        if v in (None, {}, []) and display_row:
            v = display_row.get(k)
        if v not in (None, {}, []):
            pred[k] = v

    # Legacy fields consumed by the current front-end cards. These are explicitly
    # labelled as analyst/initial-review output, not final referee predictions.
    final_reason = str(ai_r.get("reason", reason))[:3000]
    pred["final_referee_score"] = "弃权"
    pred["final_referee_analysis"] = final_reason or "终审缺失：禁止输出最终比分。"
    pred["gemini_score"] = "弃权"
    pred["gemini_analysis"] = final_reason or "Gemini终审缺失；本场仅展示初审审计，不给最终比分。"
    pred["final_ai_score"] = "弃权"
    pred["final_ai_analysis"] = pred["gemini_analysis"]
    for name in ["gpt", "grok"]:
        row = phase1.get(name) if isinstance(phase1.get(name), dict) else {}
        score = _score_from_candidate(row.get("predicted_score", "")) if row else ""
        pred[f"{name}_score"] = score if _parse_score(score)[0] is not None else "初审无有效比分"
        pred[f"{name}_analysis"] = _legacy_model_analysis(ai_r, name)
    pred["bayesian_evidences"] = [
        "终审缺失：本地按AI-native安全规则弃权，不恢复任何本地/phase1比分。",
        "以下分析来自可解析的 phase1 初审，仅用于赛前审计展示，不构成最终预测。",
        "final_referee_missing_no_phase1_fallback",
    ]
    if display_row:
        pred["bayesian_evidences"].extend([
            "phase1_display_source:" + str(display_row.get("source_model", "unknown")),
            "phase1_reason:" + str(display_row.get("reason", ""))[:1200],
        ])
    pred["ai_call_status"] = dict(AI_CALL_STATUS)
    pred["ai_run_metadata"] = dict(_LAST_AI_RUN_METADATA)
    pred["engine_version"] = ENGINE_VERSION
    pred["engine_architecture"] = ENGINE_ARCHITECTURE
    return pred


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
        action = str(rec.get("bet_action", "")).lower()
        action_ok = not action or action in BETTABLE_ACTIONS
        if bool(rec.get("is_recommended", False)) and bool(pr.get("recommend_gate_pass")) and _min_tier_ok(rec.get("tier", "D")) and action_ok:
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


def _shin_devig_3way(odds: Dict[str, Any], max_iter: int = 1000, tol: float = 1e-12) -> Tuple[Dict[str, float], float]:
    """
    基于 Jullien & Salanié (1994) 论文的 Shin Method 不动点迭代实现。
    精准剥离庄家抽水并求解 insider trading 暴露指数 z-value。
    """
    raw_odds = {k: _f(v, 0.0) for k, v in odds.items()}
    if any(v <= 1.01 for v in raw_odds.values()):
        return {k: 33.33 for k in odds}, 0.0
        
    pi = {k: 1.0 / v for k, v in raw_odds.items()}
    sum_pi = sum(pi.values())
    n = len(odds)
    
    zz_prev = 0.0
    zz_tmp = 0.0
    for _ in range(max_iter):
        zz_prev = zz_tmp
        s_terms = sum(math.sqrt(zz_prev**2 + 4.0 * (1.0 - zz_prev) * ((val**2) / sum_pi)) for val in pi.values())
        zz_tmp = (s_terms - 2.0) / (n - 2) if n > 2 else 0.0
        if abs(zz_tmp - zz_prev) <= tol:
            break
            
    z = zz_tmp
    probs = {}
    denom = 2 * (1.0 - z) if z < 1.0 else 1.0
    for k, p_val in pi.items():
        if z >= 1.0:
            probs[k] = 0.0
        else:
            term = z**2 + 4.0 * (1.0 - z) * ((p_val**2) / sum_pi)
            probs[k] = (math.sqrt(term) - z) / denom
            
    s_probs = sum(probs.values())
    if s_probs > 0:
        fair_probs = {k: round(v / s_probs * 100.0, 3) for k, v in probs.items()}
    else:
        fair_probs = {k: round(100.0 / n, 3) for k in odds}
        
    return fair_probs, round(z, 5)


def _devig_3way(odds: Dict[str, Any]) -> Dict[str, float]:
    fair, _ = _shin_devig_3way(odds)
    return fair



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
    
    # 采用高精度 Shin Method 剥离 1X2 抽水并求出 z-value
    fair, z_val = _shin_devig_3way(odds)
    
    return {
        "available": all(_f(v, 0.0) > 1.01 for v in odds.values()),
        "odds": odds,
        "fair_no_margin_pct": fair,
        "insider_trading_z_index": z_val, # 物理注入 z-value 事实
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

# ============================================================
# 比分簇枚举【唯一权威数据源 / Single Source of Truth】
# 任何 prompt 文字、审计表、一致性闸都必须引用以下常量，
# 严禁在别处手写残缺/不对称的比分列表。
# 大球带必须主客对称：3-1<->1-3, 4-1<->1-4, 3-2<->2-3, 4-2<->2-4, 4-0<->0-4。
# ============================================================
BIG_GOAL_CLUSTER = ["3-1", "1-3", "2-2", "3-2", "2-3", "4-1", "1-4", "4-2", "2-4", "4-0", "0-4"]
BIG_GOAL_CLUSTER_STR = "/".join(BIG_GOAL_CLUSTER)
# 中性"对攻大球"主推候选（最常见的 4+ 进球比分，主客对称，不含 5+ 长尾）
BIG_GOAL_PRIMARY = ["3-1", "1-3", "2-2", "3-2", "2-3", "4-1", "1-4", "4-2", "2-4"]
BIG_GOAL_PRIMARY_STR = "/".join(BIG_GOAL_PRIMARY)

ADJACENT_AUDIT_MAP = {
    "2-1": ["1-1", "1-0", "2-0", "1-2", "2-2", "3-1", "4-1"],
    "1-2": ["1-1", "0-1", "0-2", "2-1", "2-2", "1-3", "1-4"],
    "2-0": ["1-0", "3-0", "2-1", "0-0", "1-1", "3-1"],
    "0-2": ["0-1", "0-3", "1-2", "0-0", "1-1", "1-3"],
    "1-0": ["0-0", "1-1", "2-0", "2-1", "0-1"],
    "0-1": ["0-0", "1-1", "0-2", "1-2", "1-0"],
    "1-1": ["0-0", "1-0", "0-1", "2-1", "1-2", "2-2", "3-1", "1-3"],
    "3-1": ["2-1", "2-0", "3-0", "2-2", "4-1", "3-2", "4-2"],
    "1-3": ["1-2", "0-2", "0-3", "2-2", "1-4", "2-3", "2-4"],
    "2-2": ["2-1", "1-2", "3-1", "1-3", "3-2", "2-3", "3-3"],
    "3-2": ["2-2", "3-1", "2-3", "4-2", "2-1", "4-1"],
    "2-3": ["2-2", "1-3", "3-2", "2-4", "1-2", "1-4"],
    "4-1": ["3-1", "3-0", "4-0", "4-2", "2-1", "3-2"],
    "1-4": ["1-3", "0-3", "0-4", "2-4", "1-2", "2-3"],
    "3-0": ["2-0", "3-1", "4-0", "2-1", "4-1"],
    "0-3": ["0-2", "1-3", "0-4", "1-2", "1-4"],
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



# ------------------------------------------------------------
# Prompt addendum：把 20.2.1 的自由发挥改成审计表
# ------------------------------------------------------------

PHASE1_ROLE_SPLIT_ADDENDUM = """
【v20.6 AI 主导分工】
你不是比分生成器，而是风险控制型赛前分析师；目标不是每场硬推，而是避免把低质量比赛包装成高信心推荐。
GPT：正方市场结构师，重点输出 market_audit / score_cluster_audit / goal_market_audit / candidate_scores / 可买条件；若证据不足必须说 observe/no_bet。
Grok：反方审判员，重点寻找 favorite_trap / draw_trap / away_win_overreach / rotation_or_motivation_risk / source_hallucination；不要顺着热门结论。
Gemini：终审裁判，必须综合正方、反方、raw evidence 和来源质量；final_direction 只能是 home/draw/away，证据不足时用 recommendation.is_recommended=false 与 bet_action=observe/no_bet 表达不下注，不得把 no_bet/abstain 写成胜平负方向。
三方都必须列出 why_this_can_fail；不能把“名气强、低赔、常见比分模板”当成独立充分证据。
若 GPT/Grok 仍输出最终比分，Gemini 可以参考但必须重新审计，不得机械照抄。
【世界杯第三轮专项军规】若比赛属于世界杯/国际大赛小组赛第三轮/末轮，必须把“出线形势/净胜球需求/是否已出线/轮换强度”视为与盘口同级的主轴变量。第三轮不是默认小球，而是“净胜球收窄 + 热门被掀风险上升 + 场景分层”——必须先分为：双方都需分、双方都无所求、已出线强队vs有动机方 三类，再决定比分簇。
""".strip()

GEMINI_FINAL_AUDIT_ADDENDUM = """
【vMAX Gemini 终审最高裁判协议：逆向审计、反诱盘、背离终结者】
你是 88 系统的最高终审裁判。你拥有最高推翻权与最终裁定权，无需盲从 GPT 或 Grok 的错误保守结论！
0. 【低区分度强制弃权——最高优先级，凌驾以下所有条款】：本协议的所有“打破保守/强推比分/反打/提级”动作，只有在盘口对某一结果具备真实区分度时才允许执行。判据：读取三方隐含概率(去抽水后)，令 favorite_prob = max(home,draw,away 隐含概率)。
   - 若 favorite_prob < 0.45（三方势均、市场无定价优势）：这是一场事前不可预测的比赛。真实赛果对账证明此档 AI 方向命中仅 ~29%（低于随机），平局占比高且分散。你【必须】将 recommendation.is_recommended=false、bet_action=observe 或 no_bet，confidence【硬上限 40】，并在 final_referee_analysis 写明“低区分度盘口，无定价优势，主动弃权”。此时 final_direction 仍按你最优判断填写(系统要求非空)，但严禁包装成可买推荐、严禁给中高信心。
   - 严禁用“名气强/常见比分/大球对攻”等理由绕过本条把低区分度场硬抬成 main/A/S 级。
   - 数据依据(259场真实赛果)：低区分度命中29%、中等42%、较明确53%、深盘大热60%——信心必须随 favorite_prob 单调，不得在 <0.45 档给高信心。
1. 【中性看待平局，不得系统性反平局】：平局是与主胜/客胜完全对等的合法结论，不是需要被“打破”的保守包袱。真实赛果对账显示系统平局召回率仅 25%、主胜被高估(判52% vs 真实40%)，根因是历史 prompt 单向鼓励推大球、打压判平。纠正：仅当前置模块判定【大球对攻（总进球 4+）活跃】且有独立证据(xG/对攻战意/双方均需取胜)支撑时，才可将大球带 3-1/1-3/2-2/3-2/2-3/4-1/1-4/4-2 提级；无此类证据时不得为“显得专业”而强推大球、不得把本该判平/小球的场硬抬成大比分。【对称强制】凡考虑主队大胜比分（3-1/4-1/3-2），必须同时审计客队对称镜像（1-3/1-4/2-3）与平局镜像（1-1/2-2），不得只押单边、也不得默认排除平局。
2. 【终结聪明钱背离】：仔细阅读 Grok 报告的聪明钱资金变动。若散户看主/机构看客（或相反），产生明显的聪明钱背离，无论常规强弱实力对比多大，你必须直接选择【反打下盘】或将热门强队硬降级为 D（坚决放弃不推）。
3. 必须读取 local_quantitative_intelligence（战意/经验/聪明钱蒸汽）与 jingcai_market_facts（竞彩超额抽水率）作为客观事实；系统不提供任何静态数理/泊松比分基准，比分与方向完全由你像人一样读盘推理得出。
4. 强制执行“市场背离探测 (Divergence Detection)”——锚点是真实市场事实（大众投票热度、临场变盘方向、聪明钱资金流、国际盘偏斜），不是任何数理模型：
   - 如果大众极热某方向、但临场赔率反向大降水或不降反升，必须警惕“正诱造热/大热必死”陷阱，主动下调推荐等级；
   - 如果某方向临场庄家急剧降水保护，寻找被隐藏的资金入场。
5. 必须输出 predicted_score 前完成相邻比分审计（候选必须主客对称，禁止漏列镜像比分）：
   - 2-1 比较 1-1/1-0/2-0/1-2/2-2/3-1/4-1
   - 1-2 比较 1-1/0-1/0-2/2-1/2-2/1-3/1-4
   - 2-2 比较 2-1/1-2/3-1/1-3/3-2/2-3/3-3
   - 凡涉及任一大球带比分，必须整组对称审计：3-1/1-3/2-2/3-2/2-3/4-1/1-4/4-2/2-4/4-0/0-4
6. 你必须输出 bet_action：main/small/hedge/observe/no_bet。若证据冲突导致你无法覆盖失败路径，主动选择 no_bet，不要硬给高等级。
7. 【双轨国际盘背离审计——有限度降级，禁止一票否决】：若 dual_market_divergence_calibration.available=true，你必须在判决中显式引用 skew_metrics_pct 与 insider_z_gap，不得忽略：
   - 某方向 skew_pct > +5% 且 z_gap 显著正值：国内体彩将该方向赔率强行低开避险（诱盘/大热风险），需下调该方向 tier 与 bet_confidence。【降级上限】但双轨背离是单一风险信号，不得单凭此项将推荐直接归零：背离幅度 5%~10% 最多降一档/扣信心10分，>10% 最多降两档/扣信心20分；若盘口本体(1X2去水、比分簇、让球形态)对方向有强区分度支撑，不得仅因正 skew 就把本该可推的高区分度场压成 observe。
   - 某方向 skew_pct < -5% 且 z_gap 显著：国际清算盘已破防、国内滞后给出超额宽裕度（价值洼地），可适度上调该方向优先级。
   - 若 final_direction 恰好落在被强行低开(正 skew)的方向上，必须在 final_referee_analysis 解释为何仍坚持，否则视为追热失职。
   该国际盘 Shin 偏斜度为已成事实，与 sharp_money 信号互相印证时可提级，互相冲突时以偏斜度为准并说明理由。【场景提醒】世界杯/国家队赛事国内外盘差异天然偏大，正 skew 几乎场场出现，不得把“偏斜存在”等同于“必须弃权”，只有偏斜强度大且与盘口本体冲突时才降重档。
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

ENGINE_VERSION = "vMAX 20.6.0-READING-PARADIGM"
ENGINE_ARCHITECTURE = (
    "AI-NATIVE WEB-AUGMENTED 3AI FULL-SHARP-CLUSTER: 保留20.2.1完整AI调用链；"
    "新增Sharp/聪明钱事实编译、HHAD让球语义、CRS比分簇、TTG/CRS change消费、相邻比分审计；"
    "新增赛前综合因子V2风控闸门：联赛DNA/战意轮换/杯赛跨洲/弱主胜防平/客胜复核/数据质量/资金冲突；新增结构化外部因子与临场确认升级；新增推荐分层：主推/小注/防平/观察/放弃；"
    "本地不改足球方向/比分，只做Evidence编译、协议校验、推荐风险展示。"
)


def build_evidence_packet(match_obj: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    v20.6.0 升级版 Evidence Compiler — 融入本地量化影子事实与竞彩风控背离监测。
    【核心改动】：
    1. 整合本地 5 个闲置量化模块（战意、经验、泊松、蒸汽聪明钱），在编译事实阶段提前测算。
    2. 计算中国竞彩特有的 Overround（超额抽水）与区域风控倾斜。
    3. 将静态数理估值与本地经验结果作为 AI 事实参考，但从 prompt 协议上释放“AI 必须屈从数理”的过度施压，
       改而要求 AI 主动探测“数理估值（静态）与真实市场（变盘/诱盘/热度 skew）”之间的背离（Divergence），
       以此戳破庄家做盘陷阱，实现真正的逆向博弈思维。
    """
    evidence = _BASE_BUILD_EVIDENCE_PACKET_V2021(match_obj, index)
    try:
        # 1. 引入本地量化与基本面组件
        import league_intel
        import experience_rules
        import quant_edge

        league_key = league_intel.detect_league_key(match_obj.get("league", ""))
        
        # 战意基本面挖掘
        motivation_facts = league_intel.analyze_motivation(match_obj, league_key)

        # 世界杯/国际赛读盘先验注入（5届320场分轮实证+双窗口状态档；作evidence非裁判）
        world_cup_reading = None
        if league_key in ("world_cup", "intl_friendly"):
            try:
                world_cup_reading = league_intel.analyze_world_cup_context(match_obj)
            except Exception:
                world_cup_reading = None
        
        # 经验规则引擎
        prediction_shell = {"home_win_pct": 33, "draw_pct": 33, "away_win_pct": 34, "model_consensus": 2}
        experience_verdict = experience_rules.apply_experience_to_prediction(match_obj, prediction_shell)
        
        # v20.7 P0 去污: 移除 SteamMoveDetector 伪 steam 旁路。
        # 实证: 竞彩抓包 change 仅为方向码 -1/0/1(967场全样本零浮点增量),
        # SteamMoveDetector 把 -1 当作 -1.00 odds 暴跌, 对几乎每场"赔率下行"比赛喷射
        # 满格 strength=10 伪 steam 信号并注入 AI 终审, 污染读盘且违反"不造假"军规。
        # 方向码事实已由 compile_sharp_money_facts(_movement_label) 这条诚实链单一承担。

        # 2. 计算中国竞彩超额抽水 (Overround) —— 纯市场事实，非数理锚
        overround = 0.0
        has_1x2 = all(match_obj.get(k) not in (None, "", 0, "0") for k in ["sp_home", "sp_draw", "sp_away"])
        if has_1x2:
            try:
                h = float(match_obj.get("sp_home") or 0)
                d = float(match_obj.get("sp_draw") or 0)
                a = float(match_obj.get("sp_away") or 0)
                if h > 0 and d > 0 and a > 0:
                    overround = (1/h + 1/d + 1/a) - 1.0
            except:
                pass

        # 3. 注入竞彩超额抽水事实（纯市场事实，不含任何泊松/数理比分锚，避免污染 AI 读盘）
        evidence["jingcai_market_facts"] = {
            "jingcai_overround_pct": round(overround * 100, 2),
            "note": (
                "这是中国体彩竞彩 1X2 的超额抽水率（Overround），由三向赔率隐含概率之和减 1 得到，属客观市场事实。"
                "竞彩抽水通常 11% 左右且有极强区域风控倾向（为规避大众热门会过度降水）。"
                "AI 应将抽水率作为庄家做盘强度的背景参考，结合大众投票/临场变盘/资金流自行读盘探测诱盘陷阱，"
                "系统不提供任何静态数理比分基准，比分与方向判断完全由 AI 像人一样读盘推理得出。"
            )
        }
        
        # 5. 整合本地量化智能作为原始事实，交给 AI 决策
        evidence["local_quantitative_intelligence"] = {
            "motivation_scenarios": motivation_facts,
            "world_cup_reading_intel": world_cup_reading,
            "empirical_experience_triggered_rules": experience_verdict.get("experience_analysis", {}).get("rules", []),
            "empirical_over_2_5_pct": experience_verdict.get("over_2_5", 50.0),
            "steam_movement_signals": None,  # v20.7 P0 去污: 伪 steam 旁路已移除, 方向码事实归 sharp_money_facts_v203
            "compiler": "v20.7.1_poisson_purged_steam_fakesig_purged_pure_board_reading"
        }
        
        # 6. 增强原有的市场模块并合并
        evidence.update(build_enhanced_market_modules(match_obj, index))
        
        # ============================================================
        # 升级：双轨市场背离校准（Dual-Market Divergence Calibration）
        # ============================================================
        # 提取全球基准低抽水欧赔 (如 Pinnacle/Bet365 收盘欧赔，数据源已支持抓取)
        global_odds = {
            "home": match_obj.get("global_home", match_obj.get("b365_h", match_obj.get("pinnacle_h"))),
            "draw": match_obj.get("global_draw", match_obj.get("b365_d", match_obj.get("pinnacle_d"))),
            "away": match_obj.get("global_away", match_obj.get("b365_a", match_obj.get("pinnacle_a")))
        }
        
        has_global_1x2 = all(_f(global_odds[k]) > 1.01 for k in ["home", "draw", "away"])
        
        if has_global_1x2 and evidence["data_quality"]["has_1x2"]:
            # 1. 计算国内竞彩的去抽水 Shin 概率
            # 注意:base evidence 的 lottery_market_1x2 结构为 {home,draw,away,note},无 odds 子键,
            # 因此显式抽取三向赔率(含 note 会污染 devig 入参)。
            _lm = evidence.get("lottery_market_1x2", {}) or {}
            local_1x2_odds = {
                "home": _lm.get("home", match_obj.get("sp_home")),
                "draw": _lm.get("draw", match_obj.get("sp_draw")),
                "away": _lm.get("away", match_obj.get("sp_away")),
            }
            local_probs, local_z = _shin_devig_3way(local_1x2_odds)
            # 2. 计算国际低抽水的去抽水 Shin 概率
            global_probs, global_z = _shin_devig_3way(global_odds)

            # 3. 测算偏斜度: Skew = (Local Prob / Global Prob) - 1.0(守护除零)
            def _safe_skew(loc, glob):
                return (loc / glob - 1.0) if glob and glob > 0 else 0.0
            skew_home = _safe_skew(local_probs["home"], global_probs["home"])
            skew_draw = _safe_skew(local_probs["draw"], global_probs["draw"])
            skew_away = _safe_skew(local_probs["away"], global_probs["away"])
            
            evidence["dual_market_divergence_calibration"] = {
                "available": True,
                "local_shin_probabilities": local_probs,
                "global_shin_probabilities": global_probs,
                "skew_metrics_pct": {
                    "skew_home_pct": round(skew_home * 100, 2),
                    "skew_draw_pct": round(skew_draw * 100, 2),
                    "skew_away_pct": round(skew_away * 100, 2),
                },
                "insider_z_gap": round(local_z - global_z, 5),
                "interpretation": (
                    "skew_pct > 5.0% 且 z_gap 显著正值：代表国内体彩强行低开该方向赔率避险（诱盘/大热风险，赔率无博弈性价比）；"
                    "skew_pct < -5.0% 且 z_gap 显著：代表国内体彩对该方向赔率给出了超额宽裕度（国际清算已破防，国内滞后，属于核心价值洼地）。"
                )
            }
        else:
            evidence["dual_market_divergence_calibration"] = {
                "available": False,
                "note": "缺少国际欧赔基准或体彩赔率，无法计算防守偏斜度。自动退回单轨博弈。"
            }

        evidence["evidence_compiler_version"] = "v20.7.0_poisson_purged_pure_board_reading"
        evidence.setdefault("protocol_notes", []).extend([
            "v20.6: local_quantitative_intelligence 与 jingcai_market_facts（纯抽水事实）已经预先注入；系统不提供静态数理/泊松比分基准。",
            "v20.6: dual_market_divergence_calibration 已注入高精度 Shin 偏斜度与 z-value，AI 必须检测风控背离。",
            "v20.6: 竞彩高抽水及风控背离探测（Divergence Detection）已启用。AI 必须积极识破静态数理与变盘背离之间的庄家陷阱。",
            "v20.6: sharp_money_facts_v203 是事实编译，不是本地判断Sharp真伪。",
            "v20.6: score_cluster_diagnostics_v203 只描述赔率簇 and 相邻比分关系，不替AI选比分。",
            "v20.6: Gemini 必须基于相邻比分审计给出最终 predicted_score 并回答 mandatory_cross_anchor_questions。"
        ])
    except Exception as e:
        evidence.setdefault("data_quality", {})["v207_pre_inject_error"] = str(e)[:300]
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
    {"score":"1-2", "risk_type":"中文短语(如:客队反击尾部/高分平局等)", "reason":"中文说明"}
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
  "external_fact_table": [
    {"category":"injury/lineup/motivation/schedule/weather/venue/media/market_snapshot/referee", "claim":"中文事实", "source_type":"official/mainstream_media/beat_reporter/data_site/prediction_site/social_rumor", "source_title":"", "source_url":"", "published_at":"", "freshness":"same_day/recent_3d/stale/unknown", "confidence":"high/medium/low", "impact_direction":"upgrade_home/downgrade_home/upgrade_draw/upgrade_away/upgrade_over/downgrade_over/risk_only/no_clear_impact", "why_it_matters":"中文说明"}
  ],
  "source_conflict_audit": {"has_conflict":false, "conflicts":[]},
  "evidence_quality_score": 0,
  "minimum_evidence_needed": [],
  "external_facts_decision_impact": {"direction_impact":"supports_home/supports_draw/supports_away/mixed/unclear", "goal_impact":"supports_over/supports_under/mixed/unclear", "recommendation_impact":"can_upgrade/hold/downgrade/no_bet", "main_reason":"中文说明"},
  "gemini_independent_research": {"searched":true, "key_sources":[], "missing_sources":[], "cross_checked_against_grok":"中文说明"},
  "bookmaker_cross_audit": {"bet365":"中文说明", "william_hill":"中文说明", "pinnacle_or_low_margin":"中文说明", "jingcai":"中文说明", "average_market":"中文说明", "water_movement":"升水/降水/无可靠时间序列说明", "asian_handicap":"中文说明", "ou_total_goals":"中文说明", "correct_score_cluster":"中文说明", "bookmaker_intent":"中文庄家意图/诱盘/保护判断"},
  "tempo_xg_tactical_audit": {"tempo":"low/medium/high/unclear", "xg_signal":"中文说明", "xga_signal":"中文说明", "pressing_transition":"中文说明", "formation_matchup":"中文说明", "key_player_impact":"中文说明"},
  "worldcup_upset_audit": {"context":"group_stage/knockout/friendly/other", "japan_type_counter":"yes/no/unclear+中文说明", "morocco_type_low_block":"yes/no/unclear+中文说明", "croatia_type_resilience":"yes/no/unclear+中文说明", "favorite_slow_start_risk":"中文说明", "upset_path":"中文说明"},
  "score_elimination_audit": {"0-0":"keep/reject+原因", "1-1":"keep/reject+原因", "2-2":"keep/reject+原因", "1-2":"keep/reject+原因", "2-1":"keep/reject+原因", "selected_score_final_reason":"中文说明"},
  "dirty_work_checklist": {"lineup":false, "injury_suspension":false, "motivation_table":false, "weather_pitch":false, "referee":false, "travel_rest":false, "odds_sources":false, "xg_tempo":false, "worldcup_context":false},
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
硬约束：final_direction 只能是 home/draw/away；no_bet/observe 只能写在 recommendation.bet_action，不能写成 final_direction；系统级 abstain 只用于程序兜底，不允许 AI 主动输出。predicted_score 暗示的方向必须等于 final_direction；goal_band 与 predicted_score 总进球一致；btts 与 predicted_score 一致；top3[0].score 必须等于 predicted_score；必须完成 score_cluster_audit / sharp_money_audit / anchor_audit / recommendation_components / bookmaker_cross_audit / tempo_xg_tactical_audit / score_elimination_audit。若主胜概率低于或等于52%、客胜概率不低于23%、且 BTTS=yes，不得把 1-2、2-2、2-3 视为无关尾部；如果最终仍选主胜 2-1，必须说明为什么排除客队反打与 4+尾部，并填充 risk_score_candidates / tail_risk_flags。若强队让球达到球半/两球级别，但1X2胜赔仍在1.23-1.55、平赔/1-1锚点没有被抬死，必须按“深盘造强、胜赔不实压”审计，risk_score_candidates 至少保留 1-1 与一球小胜路径。若使用伤停/首发/战意/轮换/赛程/天气/xG/赔率源等外部事实影响推荐，必须提供 external_fact_table 与有效 source_url；无来源或来源冲突未解决时只能降级，不得升为 main。【区分身份，禁止无源一票否决】本条仅限制“依赖外部事实的升档”：无有效来源时，不得因伤停/首发/战意/赛程/天气等外部论据将推荐升为 main。但盘口结构本身（1X2去水概率、正确比分簇、让球形态、总进球模态、资金流/Sharp）是可独立验证的市场事实，不依赖外部新闻来源；若盘口结构对某方向有强区分度（favorite_prob 高、比分簇集中、Sharp 印证），允许基于盘口本体给出与区分度相称的 confidence 与推荐等级，无源不得成为压低盘口本体高信心的理由。若证据不足，请 recommendation.is_recommended=false、bet_action=observe/no_bet，不要硬推。【低区分度闸门】若三方隐含概率(去抽水)最大值<0.45（势均力敌、市场无定价优势），必须 recommendation.is_recommended=false、bet_action=observe/no_bet、confidence≤ 40；此类场次事前不可预测(真实赛果命中仅~29%)，不得给中高信心或 main 推荐。
""".strip()


def _phase1_system(ai_name: str) -> str:
    role_intro = {
        "gpt": "你是 Probabilistic Market Structure Analyst，专攻 1X2、HHAD让球、正确比分赔率簇、总进球模态、外部赔率对照。你不是最终裁判。",
        "grok": "你是 Grok Web-Max External Intelligence Analyst，专攻联网外部事实、来源质量、冲突审计、伤停首发战意赛程天气与市场快照。你不是最终裁判。",
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
        p.append("GPT重点【结构化清道夫与冷门探测器】：不要去盲目预测谁赢。你的唯一任务是扫描全盘的 score_cluster_diagnostics_v203。如果发现 0-0/1-1 的低分模式被严重压缩，或者强队客场让球却持续升水背离，你必须在 risk_score_candidates 里直接写入爆冷红灯，不需要给具体预测。若出现球半/两球深盘但强队1X2胜赔仍不实压、平赔/1-1未抬死，要按卡塔尔1-1瑞士型结构处理：深盘只是造强，不代表穿盘，必须保留1-1与一球小胜路径。不要被常规低赔迷惑，寻找深层陷阱！")
    elif ai_name == "grok":
        p.append("Grok重点【Web-Max 外部事实与资金背离审判员】：优先核实 external_fact_table/source_conflict_audit/evidence_quality_score，覆盖伤停、首发、战意、赛程、天气、权威新闻与市场快照；同时读取 sharp_money_facts_v203、movement、vote 审计公众热度与赔率背离。不得编造盘口时间序列，不得把无来源事实用于升 main。你不是终审裁判，但必须按统一 schema 给出基于外部事实和盘口背离的暂定 final_direction/predicted_score，供 Gemini 复核。")
    else:
        p.append("Gemini若参与初审，也必须按最终裁判标准完成相邻比分和来源审计。")
    p.append("强制：每场必须显式读取 score_cluster_diagnostics_v203.adjacent_score_audit_table；不能只看最低赔率。")
    p.append("")
    p.append("【联赛风格与战意动态锚定】：比分预测绝对不能一刀切！你必须首先评估【联赛进球生态】与【比赛重要程度】：")
    p.append("1. 进攻高波或高进球异动压缩（如德甲、荷甲、美职、挪超、解放者杯等大球联赛；判大球看曲线塌缩 a5/a4<=1.70 或超大球尾部共振(a6<=11/a7<=14) 而非 a4 单点低，因为 a4 被压但 a5 没跟是单点诱盘假信号）：防守往往让位于进攻，不得机械保守。不要机械拘泥于 2-1、1-1 等常规最低赔率。若联赛偏大球+曲线整簇塌缩+资金推强队共振，必须敢于将大球带（3-1、1-3、2-2、3-2、2-3、4-1、1-4、4-2 等主客对称比分）作为主推首选，不要仅仅把它们当做风险尾部藏起来！注意 a4>5.3 是排除线（真实大球仅约13%），按小球处理。【对称强制】考虑任一主队大胜比分时，必须同时列出客队镜像比分（3-1↔1-3、4-1↔1-4、3-2↔2-3）。")
    p.append("【负方安慰球审计·2026-06-21新增】：强弱悬殊+高信心的一边倒胜场，不要机械给负方留一个安慰球。真实赛果对账(世界杯小组赛)显示高热强队赢球时负方常被零封(美国2-0非2-1、巴西3-0非4-1)。规则：当胜方信心>=70且判定净胜>=2时，默认负方进球=0(优先 N-0 而非 N-1)；只有负方有独立破门证据(对攻战意/快反/定位球质量/客场不弃赛+xG支撑)时才可保留负方1球。【与大球带不冲突】本条只压负方安慰球，不阻止双方对攻的对称大比分(2-2/3-2/3-3)。")
    p.append("2. 防守绞肉联赛（如西甲、意甲、法乙、阿甲等及次级联赛）：天生小球属性，2-1已是双方发挥极好的天花板。在此类联赛中，无需强行防范 2-2 或 3-1，反而要极度警惕 0-0 闷平或 1-0 窄胜。")
    p.append("3. 特殊战意节点：杯赛附加赛/淘汰赛首回合极度保守（容错率极低，首选0-0/1-1）；无欲无求的谢幕战则防守松懈（极易出大球）。")
    p.append("4. 世界杯小组赛第三轮/末轮：先判断‘双方都需抢分’、‘双方都无所求’、‘已出线强队vs有动机方’这三类场景。第三轮真规律不是机械压总进球，而是净胜球收窄、1球差和冷平/小冷增多。实证先验(项目2026小组赛自对账)：已出线强队在后期轮被逼平是高频系统偏差(西班牙0-0、葡萄牙1-1、卡塔尔1-1、沙特1-1、厄瓜多尔0-0)。原因是出线后大面积轮换+控分慢热。因此若热门一方已出线、可轮换、平局或小负仍可接受，必须优先防 1-1/0-1/1-0/1-2/2-1 这类收窄比分，下调该热门胜的信心(上限≤60)，不得仅因名气与低赔强推穿盘大胜。联网核实热门是否已锁名次、是否轮换、是否同组并行开赛存在算计净胜球/默契球土壤。")
    p.append("5. 2026 新制（最佳第三名）会削弱纯躺平，但不会消灭诱盘：若一方需要刷净胜球、另一方也有抢分希望，比分会从‘单纯低分’转向‘开放但净胜差未必拉大’。这类场景要优先比较 2-1/1-2/2-2/3-2/2-3，而不是直接从大热低赔跳到3-0/0-3。")
    p.append("请结合真实的足球世界逻辑，为当前比赛选择最符合其土壤的比分，不要被单纯的赔率数字束缚想象力！")
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
        *EXTERNAL_FACT_FIELDS, *FULL_SPECTRUM_AUDIT_FIELDS,
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
    p.append("【独立联网裁判职责】：即使 Grok 无法联网、来源为空或数据混乱，你也不能只复述 Grok。若你的 API/模型具备联网能力，必须独立执行 Web-Augmented Match Research，交叉验证球队新闻、伤停首发、战意赛程、天气场地、权威赔率/盘口快照与主流数据源；若无法联网，必须在 web_research.used=false 与 final_web_audit.web_used_by_final=false 中写明 no_web_tool_available/no_web_capability_or_disabled，且任何依赖外部事实的推荐不得升为 main。")
    p.append("【全市场终审职责】：你是最终裁判，必须自己复核 1X2 欧赔/竞彩、HHAD/亚盘让球语义、总进球/大小球、正确比分赔率簇、相邻比分、BTTS、资金热度/Sharp、dual_market_divergence_calibration 与 score_cluster_diagnostics_v203；不得把赔率、亚盘、比分簇任务只交给 GPT/Grok。最终方向、比分、推荐等级必须由你完成全维读盘后裁定。")
    p.append("【庄家逆向读盘职责】：必须像庄家一样审计 Bet365、William Hill/威廉希尔、Pinnacle/低抽水基准、竞彩与百家均值之间的分歧；逐项解释升水/降水、亚盘升降盘、大小球水位、总进球赔率、正确比分簇是否在保护、诱导、分散或封顶。没有真实时间序列时只能写当前快照，不得编造临场故事。")
    p.append("【节奏/xG/战术脏活】：必须独立查并评估球队节奏、xG/xGA、射门质量、转换速度、压迫强度、阵型对位、核心球员缺阵/复出、天气场地、裁判尺度、旅行休息和赛程密度；这些只能作为读盘证据，不能无来源升 main。")
    p.append("【世界杯爆冷脑回路】：必须主动审计日本型反击爆冷、摩洛哥型低位铁桶、克罗地亚型韧性拖平/加时土壤、强队慢热/名气过热/弱队效率反杀路径；输出 worldcup_upset_audit，不得用强弱名气直接杀冷。")
    p.append("【世界杯第三轮三分法——强制】若比赛属于小组赛第三轮/末轮，你必须先把场景归入以下之一：A.双方都需抢分/抢净胜球；B.双方都无所求或都可接受特定结果；C.已出线强队/热门方 vs 有动机方。A 类可释放开放比分，但仍要防净胜球只收窄到1球差；B 类优先防 0-0/1-1/0-1/1-0；C 类把热门被掀、0进球、小负或被逼平列为主风险，不得把热门穿盘大胜当默认线。")
    p.append("【第三轮轮换·控分·爆冷实证先验——强制】项目自对账实测(2026小组赛多场)：已出线强队在第二/三轮被逼平是高频系统偏差(西班牙0-0、葡萄牙1-1、卡塔尔1-1、沙特1-1、厄瓜多尔0-0全部踢平)。原因链：出线后11人轮休→阵容生疏与慢热→不打净胜球→控分/闷平/被反杀。因此：R3 一旦识别到热门方已出线+可轮换，你对该热门方胜的信心上限硬限 60，除非有联网证据证明他们仍派主力抢名次/进球净胜球争首名。同时重点释放有动机弱队的抢分动机/定位球爆冷路径。")
    p.append("【世界杯第三轮联网硬要求】你具备联网能力，必须主动联网核实第三轮四件事：①两队实时出线形势与最佳第三名撑杆；②热门方是否已锁定出线/锁定小组名次(决定是否轮换)；③教练发布会/跟队记者披露的轮换/主力保留意图；④是否存在同组并行开赛下的默契球/算计净胜球争议。以上每条影响方向/比分/推荐的 claim 必须进入 external_fact_table / web_research.sources(含 url+published_at)。无来源时，允许保留盘口本体方向，但 recommendation 最高 B，且不得把第三轮轮换/出线战意推演包装成 main。")
    p.append("【比分淘汰协议】：最终选比分前必须逐一审计 0-0、1-1、2-2、1-2、2-1 以及相邻镜像比分，说明 keep/reject 原因；尤其 2-1 只有在 1-1/1-2/2-2/高分尾部被充分排除后才能主推。")
    p.append("【前置共识权重】：如果 GPT 和 Grok 在方向或比分上达成高度一致（例如均为 1-0），除非你有致命的反向硬证据（例如明显的伤停或极强的聪明钱背离），否则不得仅仅因为 1-1 或 0-0 是全场最低赔率，就强行推翻前置共识走向保守平局。")
    p.append("【联赛风格与战意动态锚定】：比分预测绝对不能一刀切！你必须首先评估【联赛进球生态】与【比赛重要程度】：")
    p.append("1. 进攻高波联赛（如德甲、荷甲、挪超、美职、澳超等）：防守往往让位于进攻，不要机械拘泥于 2-1 或防守平局。若双方战术开放且支持 BTTS，必须敢于将 3-2、3-1 甚至 4-2 这种极端高比分直接作为主推首选，不要仅仅把它们当做风险尾部藏起来！")
    p.append("【负方安慰球审计·2026-06-21新增】：强弱悬殊+高信心的一边倒胜场，不要机械给负方留一个安慰球。真实赛果对账(世界杯小组赛)显示高热强队赢球时负方常被零封(美国2-0非2-1、巴西3-0非4-1)。规则：当胜方信心>=70且判定净胜>=2时，默认负方进球=0(优先 N-0 而非 N-1)；只有负方有独立破门证据(对攻战意/快反/定位球质量/客场不弃赛+xG支撑)时才可保留负方1球。【与大球带不冲突】本条只压负方安慰球，不阻止双方对攻的对称大比分(2-2/3-2/3-3)。")
    p.append("2. 防守绞肉联赛（如西甲、意甲、法乙、阿甲等及次级联赛）：天生小球属性，2-1已是双方发挥极好的天花板。在此类联赛中，无需强行防范 2-2 或 3-1，反而要极度警惕 0-0 闷平或 1-0 窄胜。")
    p.append("3. 特殊战意节点：杯赛附加赛/淘汰赛首回合极度保守（容错率极低，首选0-0/1-1）；无欲无求的谢幕战则防守松懈（极易出大球）。")
    p.append("请结合真实的足球世界逻辑，为当前比赛选择最符合其土壤的比分，不要被单纯的赔率数字束缚想象力！")
    p.append("证据优先级：raw market structure > score_cluster_diagnostics_v203 > HHAD让球语义 > total-goals mode > sharp_money_facts_v203 > tactical/web context > Phase1 consensus。")
    p.append("多数意见不自动成立；若 GPT/Grok 基于同一低赔/单边市场理由一致，这属于相关证据，不是独立证据。")
    p.append(GEMINI_FINAL_AUDIT_ADDENDUM)
    p.append("")
    p.append("S级必须同时满足：方向边际强、比分簇强、总进球带强、相邻比分解释完整、Sharp/热度不冲突、推荐组件分数透明。仅赔率低赔最多A；无真实联网且依赖阵容/战意/伤停，最高B。")
    p.append("Grok Web-Max 证据规则：你必须审计 Grok 的 external_fact_table、source_conflict_audit、evidence_quality_score、minimum_evidence_needed。若外部事实来源为空、URL为空/#、来源冲突未解决或 evidence_quality_score<50，任何依赖伤停/首发/战意/轮换/赛程/天气的推荐不得升为 main；可以保留比分判断，但必须降级 recommendation。")
    p.append("禁止把当前赔率快照讲成盘口时间序列；没有本地多时间点数据时，不得引用 T-60m/T-30m、临场回补、资金持续流入或诱盘闭环。")
    p.append("强制输出 recommendation_components、gemini_independent_research、bookmaker_cross_audit、tempo_xg_tactical_audit、worldcup_upset_audit、score_elimination_audit、dirty_work_checklist，不能只给 tier 和 bet_confidence。")
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
    p.append("若 dual_market_divergence_calibration.available=true，同样必须应用双轨背离规则：skew_pct > +5% 且 z_gap 正=该方向被诱盘低开，下调 tier/confidence；skew_pct < -5%=价值洼地，可上调优先级。")
    p.append("同样必须应用 Grok Web-Max 证据规则：审计 external_fact_table/source_conflict_audit/evidence_quality_score；无有效来源或来源冲突未解决时，不得因伤停/首发/战意/赛程/天气升为 main。禁止编造盘口时间序列。")
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


# 16-role "family debate" cast used by the GPT fallback referee when Gemini final
# is unavailable. Each role is a reading-paradigm lens; GPT must internally let
# them argue, then a chair reconciles into ONE final score per match. This is a
# referee role (decides the final score), NOT a phase1 analyst promotion.
FAMILY_DEBATE_ROLES = [
    ("祖父·市场结构", "只信1x2/让球/大小球原生赔率结构与overround，先定方向边际强弱，反对凭名气下注。"),
    ("祖母·正确比分簇", "盯CRS正确比分赔率簇与相邻比分密度，指出市场最厚的比分形状。"),
    ("父亲·总进球档位", "先定goal_band(0-1/2-3/4+)，禁止跳过定档直接报比分。"),
    ("母亲·大小球线移动", "看ttg/盘口位移：线下行偏小球、线上行偏大球，按联赛赋权强弱。"),
    ("大伯·大球曲线塌缩", "判大球只看a5/a4斜率塌缩与a6/a7尾部共振，反对a4单点诱盘假信号。"),
    ("二叔·让球形态", "用让球盘深浅+资金定比分形状：对称(1-1/2-2)还是单边(2-0/0-2)。"),
    ("姑姑·资金流/Sharp", "读public vs sharp、reverse line movement、steam，揪庄家诱盘背离。"),
    ("舅舅·联赛风格", "前置联赛进球生态：大开大合联赛归大球，绞肉联赛归小球，不可一刀切。"),
    ("哥哥·强强高分簇", "强强对话审计4球≤5.3且5球≤7.8时的2-2/3-2高分簇倾向。"),
    ("姐姐·弱势反打", "审计客胜/弱队进球路径与BTTS尾部，防机械零封。"),
    ("表哥·逆向博弈", "低赔比分被异常压低=庄家入口，主张反向考虑或回避。"),
    ("表姐·战意与重要性", "赛季末战意倒挂、杯赛淘汰赛保守、谢幕战松懈，作信心放大器/缩小器。"),
    ("叔公·伤停轮换", "读intelligence伤停/轮换/试阵，友谊赛热身赛默认阵容脆弱。"),
    ("婶婶·联网与时效", "核验web_research来源URL/时效，缺来源不得当硬证据。"),
    ("小弟·风险与弃权", "证据不足/自相矛盾时主张降级或no_bet，宁可不推不可硬推。"),
    ("家长·主裁仲裁", "汇总全场争论，按证据优先级裁定唯一final比分与tier，输出可证伪理由。"),
]


def _family_debate_roster_text() -> str:
    return "\n".join(f"{i+1}. {name}：{stance}" for i, (name, stance) in enumerate(FAMILY_DEBATE_ROLES))


def build_family_debate_referee_prompt(evidence_batch: List[Dict[str, Any]], phase1_results: Dict[str, Dict[int, Dict[str, Any]]], critic_reports: Dict[str, List[Dict[str, Any]]]) -> str:
    p = []
    p.append("<gpt_family_debate_final_referee>")
    p.append("Gemini 终审已不可用。现在由你（GPT）独自扮演一个16人足球家庭的全部成员，对每场比赛展开内部辩论，最终仲裁出唯一最终比分。你是终审裁判，不是phase1初审；你的输出就是最终结果。")
    p.append("【16个家庭角色，必须全部在内心发言】")
    p.append(_family_debate_roster_text())
    p.append("【辩论规则】")
    p.append("1. 每场先让16角色各自从自己视角给出倾向（方向+比分+一句理由），允许互相反驳。")
    p.append("2. 证据优先级：raw market structure > 正确比分簇 > 总进球模态 > 让球盘形态 > 资金流/Sharp > 联赛风格/战意 > 联网背景 > Phase1 共识。多数票不自动成立，相同低赔理由算同一族证据。")
    p.append("3. 两段式读盘：先由父亲定goal_band档位+把握度，再在档内由家长仲裁唯一比分；选非众数比分必须写可证伪的背离理由，否则回落众数并下调confidence。")
    p.append("4. 家长（主裁）负责收敛分歧，输出每场唯一 predicted_score，并在 debate_summary 里浓缩关键分歧与裁定依据。")
    p.append("5. 证据不足或自相矛盾时，小弟有权要求 recommendation.is_recommended=false / bet_action=no_bet，但仍要给出最可能比分，不要整场留空。")
    p.append("6. 必须应用 Grok Web-Max 证据规则：婶婶审计 external_fact_table/source_conflict_audit/evidence_quality_score；没有有效来源的伤停/首发/战意/赛程/天气论据只能降级，不得升为 main；禁止把赔率快照编成 T-60m/T-30m 时间序列。")
    p.append("输出 schema 与 Gemini final 完全一致（同样的 predictions 数组与字段），额外要求：每个 prediction 增加 \"family_debate\":{\"debate_summary\":\"中文，浓缩16角色关键分歧与裁定\",\"chair_reasoning\":\"家长最终裁定逻辑\",\"dissent\":[\"被否决的少数派比分及理由\"]}。")
    p.append("必须完成 score_cluster_audit / sharp_money_audit / anchor_audit / recommendation_components。不要使用本地规则，不要 markdown，只输出严格 JSON object。")
    p.append("</gpt_family_debate_final_referee>")
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
    p.append("检查：final_direction 只能是 home/draw/away，no_bet/observe 只能存在于 recommendation.bet_action；predicted_score方向=final_direction；goal_band与比分总进球一致；btts与比分一致；top3[0]=predicted_score；web_research.used=true时必须有sources；external_fact_table 非空时每条必须有 source_url，且必须同步存在 source_conflict_audit/evidence_quality_score/external_facts_decision_impact；若外部事实无来源或冲突未解决，repair 只能降级 recommendation 为 observe/no_bet，不得改比分硬升；score_cluster_audit/sharp_money_audit/anchor_audit/recommendation_components/bookmaker_cross_audit/tempo_xg_tactical_audit/score_elimination_audit 必须存在；score_elimination_audit 必须覆盖 0-0/1-1/2-2/1-2/2-1；risk_score_candidates/tail_risk_flags/confidence_downgrade_reason 若存在必须保持数组/字符串结构。")
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
DEEP_FAVORITE_LOOSE_EURO_GUARD_REASON = "deep favorite handicap strengthened while 1x2 favorite price remains loose; protect 1-1/one-goal paths"


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


def _deep_favorite_guard_side(match_obj: Dict[str, Any]) -> Tuple[str, float, float]:
    sp_home = _f(match_obj.get("sp_home", match_obj.get("win")), 0.0)
    sp_away = _f(match_obj.get("sp_away", match_obj.get("lose")), 0.0)
    if sp_home > 1.01 and (sp_away <= 1.01 or sp_home < sp_away):
        return "home", sp_home, sp_away
    if sp_away > 1.01 and (sp_home <= 1.01 or sp_away < sp_home):
        return "away", sp_away, sp_home
    return "", 0.0, 0.0


def _deep_favorite_loose_euro_guard_trigger(match_obj: Dict[str, Any]) -> Dict[str, Any]:
    side, fav_odds, dog_odds = _deep_favorite_guard_side(match_obj)
    if side not in {"home", "away"}:
        return {"triggered": False}
    line = _parse_handicap_value(match_obj.get("give_ball", match_obj.get("handicap", match_obj.get("rq", ""))))
    if line is None or abs(line) < 1.5:
        return {"triggered": False}
    line_side = "home" if line > 0 else "away" if line < 0 else ""
    if line_side and line_side != side:
        return {"triggered": False}
    sp_draw = _f(match_obj.get("sp_draw", match_obj.get("same")), 0.0)
    s11 = _f(match_obj.get("s11"), 0.0)
    loose_favorite_price = 1.23 <= fav_odds <= 1.55
    draw_live = (3.60 <= sp_draw <= 5.30) or (0 < s11 <= 8.0)
    if not (loose_favorite_price and draw_live):
        return {"triggered": False}
    return {
        "triggered": True,
        "favorite_side": side,
        "favorite_odds": fav_odds,
        "underdog_odds": dog_odds,
        "draw_odds": sp_draw,
        "handicap_line": line,
        "one_one_odds": s11,
    }


def apply_deep_favorite_loose_euro_guard(pred: Dict[str, Any], match_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Surface Qatar-Switzerland style 1-1 upset risk without changing final score/direction."""
    if not isinstance(pred, dict) or pred.get("is_abstain"):
        return pred
    trigger = _deep_favorite_loose_euro_guard_trigger(match_obj)
    if not trigger.get("triggered"):
        return pred
    side = str(trigger.get("favorite_side", ""))
    if pred.get("final_direction") != side:
        return pred

    candidates = _normalize_risk_score_candidates(pred.get("risk_score_candidates", []))
    if side == "away":
        add_scores = ["1-1", "0-1", "1-2"]
    else:
        add_scores = ["1-1", "1-0", "2-1"]
    for sc in add_scores:
        _append_risk_candidate(candidates, sc, "deep_favorite_loose_euro_draw_guard", DEEP_FAVORITE_LOOSE_EURO_GUARD_REASON)
    pred["risk_score_candidates"] = candidates[:12]
    _add_unique_tail_flags(pred, "deep_favorite_loose_euro_draw_guard", "protect_1_1_and_one_goal_win_paths")

    pred.setdefault("deep_favorite_loose_euro_audit", {}).update(trigger)
    pred.setdefault("validation_warnings", []).append("deep_favorite_loose_euro_guard_applied")
    _cap_recommendation_tier(pred, "C", "deep_favorite_loose_euro_guard", "deep_favorite_loose_euro_draw_guard")
    rec = pred.setdefault("recommendation", {}) if isinstance(pred.get("recommendation"), dict) else {}
    if not isinstance(pred.get("recommendation"), dict):
        pred["recommendation"] = rec
    rec["risk_level"] = "high"
    old_conf = int(_clip(_f(rec.get("bet_confidence", pred.get("confidence", 0)), 0), 0, 100))
    rec.setdefault("original_bet_confidence", old_conf)
    rec["risk_adjusted_bet_confidence"] = min(old_conf, 54)
    rec["display_bet_confidence"] = min(old_conf, 54)
    pred["risk_adjusted_confidence"] = min(int(_clip(_f(pred.get("confidence", old_conf), 0), 0, 100)), 54)
    pred["display_confidence"] = pred["risk_adjusted_confidence"]
    pred["confidence_downgrade_reason"] = DEEP_FAVORITE_LOOSE_EURO_GUARD_REASON
    pred.setdefault("recommend_gate_reasons", []).append("deep_favorite_loose_euro_guard")
    pred.setdefault("recommendation_downgrade_reasons", []).append("deep_favorite_loose_euro_guard")
    pred["validation_warnings"] = list(dict.fromkeys(str(x) for x in pred.get("validation_warnings", []) if str(x).strip()))[:80]
    pred["recommend_gate_reasons"] = list(dict.fromkeys(str(x) for x in pred.get("recommend_gate_reasons", []) if str(x).strip()))[:80]
    return pred


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

    # v20.7 P0 去污: matrix_*(泊松出身)已删除,2-1 硬闸不再读任何泊松字段。

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
    # v20.7 P0 去污: 移除泊松 matrix_flags/draw/away 驱动的 NO_BET 判定,2-1 硬闸仅由 AI 概率与显式候选尾部驱动。
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
        and str(row.get("league", "")).lower() not in {"美职", "mls", "美职联"}
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
            "family_debate", *EXTERNAL_FACT_FIELDS, *FULL_SPECTRUM_AUDIT_FIELDS,
        ]:
            if k == "evidence_quality_score" and raw_item.get(k) not in (None, ""):
                row[k] = int(_clip(_f(raw_item.get(k), 0), 0, 100))
            elif isinstance(raw_item.get(k), (dict, list)):
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























PREMATCH_V2_VERSION = "v20.4_pre_match_factor_gate"

LEAGUE_DNA_PROFILES = {
    # 友谊赛/热身赛不是常规联赛：方向可读但比分极脆，低赔强队常带公众入口属性。
    # 本地只做读盘先验与风险展示，不改 AI 终审方向/比分。
    "国际友谊": {"volatility": "high", "draw_risk": "high", "btts": "high", "away_penalty": 1, "notes": "国际友谊赛/热身赛高方差练兵窗口：压低零封与大胜想象，保留1-1/2-2与客队进球路径"},
    "国际赛": {"volatility": "high", "draw_risk": "high", "btts": "high", "away_penalty": 1, "notes": "国际比赛日前后热身属性强，低赔/名气强队不是稳态实力确认，比分需防BTTS和平局尾部"},
    "友谊": {"volatility": "high", "draw_risk": "high", "btts": "high", "away_penalty": 1, "notes": "友谊赛高换人/试阵/保护主力，零封与大胜路径默认脆弱"},
    "热身": {"volatility": "high", "draw_risk": "high", "btts": "high", "away_penalty": 1, "notes": "热身赛高方差，强队低赔需当公众入口复核，不直接当精选确认"},
    "friendly": {"volatility": "high", "draw_risk": "high", "btts": "high", "away_penalty": 1, "notes": "Friendly context: high variance, rotation and BTTS/draw tails; cap clean-sheet and heavy-favorite imagination"},
    "德甲": {"volatility": "high", "draw_risk": "medium", "btts": "high", "away_penalty": 1, "notes": "高节奏/高BTTS/末段波动，客胜与小胜比分需降权复核"},
    "瑞超": {"volatility": "medium", "draw_risk": "high", "btts": "medium", "away_penalty": 1, "notes": "北欧联赛平局与1球差较多，弱优势必须防1-1/2-2"},
    "芬超": {"volatility": "medium", "draw_risk": "high", "btts": "medium", "away_penalty": 1, "notes": "低比分和平局权重偏高，客胜需市场确认"},
    "挪超": {"volatility": "high", "draw_risk": "medium", "btts": "high", "away_penalty": 1, "notes": "节奏与大球尾部较强，比分需用簇而非单点"},
    "美职": {"volatility": "high", "draw_risk": "high", "btts": "high", "away_penalty": 2, "notes": "主客差/旅行/反转波动大，客胜默认高风险"},
    "葡超": {"volatility": "medium", "draw_risk": "medium", "btts": "medium", "away_penalty": 0, "notes": "强弱分化明显，但中下游/保级题材需防平"},
    "世界杯": {"volatility": "medium", "draw_risk": "high", "btts": "low", "away_penalty": 0, "neutral_venue": True, "notes": "世界杯=中立场锦标赛足球(5届320场场均2.48偏低/U2.5约28%)：默认低分/防守反击，不照搬联赛大球；无主场优势，主客名义≠地利；分轮战意差异大(首轮最闷、末轮控分/净胜球收窄)，比分用簇不用单点，强强/强弱错配防0封脆弱与诱盘"},
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
    league_l = league_s.lower()
    friendly_tokens = ["国际友谊", "国际赛", "友谊", "热身", "friendly"]
    if any(t.lower() in league_l for t in friendly_tokens):
        profile = LEAGUE_DNA_PROFILES.get("国际友谊", {})
        return {"key": "friendly_context", **profile}
    for key, profile in LEAGUE_DNA_PROFILES.items():
        if key in league_s or key.lower() in league_l:
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
    for key in ["top_score_candidates", "unified_matrix_top_scores", "top3", "risk_score_candidates"]:
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






def _is_friendly_context(dna: Dict[str, Any], match_obj: Dict[str, Any]) -> bool:
    if dna.get("key") == "friendly_context":
        return True
    league = str(match_obj.get("league", "")).lower()
    return any(t in league for t in ["国际友谊", "国际赛", "友谊", "热身", "friendly"])


def _add_unique_tail_flags(pred: Dict[str, Any], *flags: str) -> None:
    cur = pred.get("tail_risk_flags", [])
    if not isinstance(cur, list):
        cur = [str(cur)] if cur else []
    cur.extend(str(f) for f in flags if str(f).strip())
    pred["tail_risk_flags"] = list(dict.fromkeys(cur))[:20]


def _apply_friendly_reading_prior_gate(
    pred: Dict[str, Any],
    match_obj: Dict[str, Any],
    audit: Dict[str, Any],
    web_used: bool,
    lineup_confirmed: bool,
    final_dir: str,
    draw_prob: float,
    draw_cluster: bool,
) -> None:
    """Friendlies: surface rotation/BTTS/draw-tail risk without changing score or direction."""
    dna = audit.get("league_dna", {}) if isinstance(audit.get("league_dna"), dict) else {}
    if not _is_friendly_context(dna, match_obj):
        return

    score = _score_from_candidate(pred.get("predicted_score"))
    candidates = pred.get("risk_score_candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    candidates = _normalize_risk_score_candidates(candidates)

    btts_no = str(pred.get("btts", "")).strip().lower() in {"no", "n", "false", "否", "both_teams_no"}
    no_confirm = not (web_used and lineup_confirmed)
    clean_sheet_trap = final_dir in {"home", "away"} and score in {"1-0", "2-0", "0-1", "0-2"} and (no_confirm or btts_no or draw_cluster or draw_prob >= 23.0)
    if clean_sheet_trap:
        audit["rules_applied"].append("prematch_v2_friendly_clean_sheet_trap_guard")
        audit.setdefault("risk_hints", []).append({"max_tier": "C", "reason": "prematch_v2_friendly_clean_sheet_trap_guard", "tag": "friendly_clean_sheet_fragility"})
        _add_unique_tail_flags(pred, "friendly_clean_sheet_fragility", "friendly_btts_late_goal_risk", "friendly_2_2_draw_tail")
        if final_dir == "home":
            add_scores = ["1-1", "2-1", "2-2"]
        else:
            add_scores = ["1-1", "1-2", "2-2"]
        for sc in add_scores:
            _append_risk_candidate(candidates, sc, "friendly_context_tail", "友谊赛/热身赛零封路径脆弱，需保留BTTS与2-2平局尾部")

    rec = pred.get("recommendation", {}) if isinstance(pred.get("recommendation"), dict) else {}
    tier = str(rec.get("tier", pred.get("recommendation_tier", "D"))).upper()
    sp_home = _f(match_obj.get("sp_home"), 0)
    sp_away = _f(match_obj.get("sp_away"), 0)
    vote = match_obj.get("vote", {}) if isinstance(match_obj.get("vote"), dict) else {}
    public_home = _f(vote.get("win"), 0)
    public_away = _f(vote.get("lose"), 0)
    low_price_home = final_dir == "home" and 1.01 < sp_home <= 1.35 and public_home >= 65.0
    low_price_away = final_dir == "away" and 1.01 < sp_away <= 1.45 and public_away >= 65.0
    if tier == "B" and (low_price_home or low_price_away) and no_confirm:
        audit["rules_applied"].append("prematch_v2_friendly_favorite_overheat_cap")
        audit.setdefault("risk_hints", []).append({"max_tier": "C", "reason": "prematch_v2_friendly_favorite_overheat_cap", "tag": "friendly_favorite_overheat"})
        _add_unique_tail_flags(pred, "friendly_favorite_overheat", "friendly_draw_tail")
        for sc in (["1-1", "2-1", "2-2"] if final_dir == "home" else ["1-1", "1-2", "2-2"]):
            _append_risk_candidate(candidates, sc, "friendly_favorite_overheat", "友谊赛低赔/名气强队易成为公众入口，B级无首发确认不作精选")

    pred["risk_score_candidates"] = candidates[:12]



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
    worldcup_r3 = ("世界杯" in league or "world cup" in league.lower() or "worldcup" in league.lower()) and any(t in text for t in ["第三轮", "第3轮", "末轮"])
    already_qualified_or_can_accept_less = any(t in text for t in ["已出线", "提前晋级", "锁定出线", "打平即可", "小负即可", "轮换", "轮休", "保留主力", "避强签", "头名", "第二名"])
    must_win_or_need_margin = any(t in text for t in ["必须取胜", "唯有取胜", "至少赢", "净胜球", "抢头名", "争最佳第三", "背水一战", "生死战", "抢分", "晋级希望"])
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
        "worldcup_r3": worldcup_r3,
        "already_qualified_or_can_accept_less": already_qualified_or_can_accept_less,
        "must_win_or_need_margin": must_win_or_need_margin,
        "name_favorite": name_favorite,
    }


def _weak_home_win_context(pred: Dict[str, Any], match_obj: Dict[str, Any], draw_prob: float, edge_gap: float, draw_cluster: bool) -> bool:
    if pred.get("final_direction") != "home":
        return False
        
    # [vMAX 重构]: 大球对攻模态或高海拔等非结构化特质，豁免弱主胜防平锁
    dna = _league_dna_profile(str(match_obj.get("league", "")).lower())
    expected_goals = _f(pred.get("expected_total_goals", 0))
    if expected_goals >= 2.8 or dna.get("goal_mode") == "high":
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
    friendly_clean_sheet = "prematch_v2_friendly_clean_sheet_trap_guard" in rules
    friendly_overheat = "prematch_v2_friendly_favorite_overheat_cap" in rules

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
        league_str = str(match_obj.get("league", "")).lower()
        is_nordic_hot = league_str in {"挪超", "瑞典超", "瑞超"} and _f(pred.get("home_win_pct"), 0) > 65.0
        
        if is_nordic_hot:
            layer, stake = "防平", 0.25
            reasons.append("北欧赛事主胜大热(>65%)，强制降级为防平观察")
            hedge.append("防冷")
        else:
            layer, stake = "主推", 1.0
            reasons.append("推荐闸门通过且结构化因子确认充分")
    elif friendly_clean_sheet or friendly_overheat:
        layer, stake = "防平", 0.25
        hedge.extend(["平局", "2-2尾部"] if friendly_clean_sheet else ["平局", "强队热度降温"])
        reasons.append("友谊赛/热身赛先验触发：低赔或零封路径只作防平观察")
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
    # v20.7 P0 去污: 方向概率不再与泊松 matrix_direction_probs 做 max 融合,仅用 AI 读盘。
    draw_prob = _f(probs.get("draw"), 0)
    away_prob = _f(probs.get("away"), 0)
    home_prob = _f(probs.get("home"), 0)
    sorted_probs = sorted([_f(v, 0) for k, v in probs.items() if k in VALID_DIRS], reverse=True)
    edge_gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else 0
    # v20.7 P0 去污: high_tail 原取自泊松 matrix_goal_probs,现泊松已删除。
    # 保持行为等价: matrix 缺失 → high_tail 退化为 0(该泊松高进球尾部信号不再参与),不引入新数据源。
    high_tail = 0.0
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

    # 友谊赛/热身赛读盘先验：只补风险与审计，不改 AI 终审方向/比分。
    _apply_friendly_reading_prior_gate(pred, match_obj, audit, web_used, lineup_confirmed, final_dir, draw_prob, draw_cluster)

    # P0 结构化因子硬风险：轮换/战意/关键伤停/市场冲突必须进入推荐层。
    if structured_factors.get("market_conflict"):
        apply("D", "prematch_v2_structured_market_conflict_no_bet", "prematch_v2_structured_market_conflict")
    if structured_factors.get("critical_absence") and not lineup_confirmed:
        apply("C", "prematch_v2_critical_absence_without_lineup_confirmation", "prematch_v2_critical_absence")
    if structured_factors.get("rotation_risk") and not lineup_confirmed:
        apply("C", "prematch_v2_rotation_risk_requires_lineup", "prematch_v2_rotation_risk")
    if structured_factors.get("fatigue_risk") and final_dir == "away":
        apply("C", "prematch_v2_away_fatigue_travel_risk", "prematch_v2_away_fatigue")

    # P0 世界杯第三轮：已出线/可接受平或小负/轮换方，不允许被包装成热门强推。
    if context_flags.get("worldcup_r3"):
        if context_flags.get("already_qualified_or_can_accept_less") and context_flags.get("rotation_risk") and final_dir != "draw":
            apply("C", "prematch_v2_worldcup_r3_rotation_or_qualification_cap", "prematch_v2_worldcup_r3_rotation")
        if context_flags.get("already_qualified_or_can_accept_less") and final_dir != "draw" and draw_cluster:
            apply("C", "prematch_v2_worldcup_r3_draw_or_small_loss_trap", "prematch_v2_worldcup_r3_draw_trap")
        if context_flags.get("must_win_or_need_margin") and not web_used:
            apply("C", "prematch_v2_worldcup_r3_motivation_without_external_confirmation", "prematch_v2_worldcup_r3_unverified_motivation")
        # [R3热门保平先验·实证20260626] 自对账: 强弱悬殊却踢平是R3/后期轮高频系统偏差
        # (西班牙0-0/葡萄牙1-1/卡塔尔1-1/沙特1-1/厄瓜多尔0-0)。根因=已出线大轮换+控分慢热。
        # 即使AI未显式写"已出线/轮换"，只要 R3+名气热门+判净胜>=2 的穿盘大胜，无联网证据证明仍派主力刷净胜球时，
        # 强制把热门大胜降为最高 B(不进主推)，并标记防平。专治高信心强队被逼平翻车。
        _r3_score = _score_from_candidate(pred.get("predicted_score"))
        _r3_h, _r3_a = _parse_score(_r3_score)
        _r3_margin = abs(_r3_h - _r3_a) if (_r3_h is not None and _r3_a is not None) else 0
        if (context_flags.get("name_favorite") and final_dir in {"home", "away"} and _r3_margin >= 2
                and not (context_flags.get("must_win_or_need_margin") and web_used)):
            apply("B", "prematch_v2_worldcup_r3_favorite_blowout_hold_prior_caps_to_B", "prematch_v2_worldcup_r3_favorite_hold")
            if not web_used or context_flags.get("already_qualified_or_can_accept_less") or context_flags.get("rotation_risk"):
                apply("C", "prematch_v2_worldcup_r3_favorite_blowout_unverified_draw_defense", "prematch_v2_worldcup_r3_favorite_hold_draw")

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
        league_str = str(match_obj.get("league", "")).lower()
        is_mls_or_nordic = league_str in {"美职", "mls", "美职联", "挪超", "瑞超", "瑞典超", "芬超"}
        # [vMAX重构]: 对波动高但分胜负概率高(BTTS大)的联赛，免除硬性的"C"级锁
        if (dna.get("draw_risk") == "high" or draw_prob >= 30.0 or not web_used) and not is_mls_or_nordic:
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
    if not isinstance(pred, dict):
        return pred
    if pred.get("is_abstain"):
        reason = str((ai_r.get("validation_warnings") or pred.get("recommend_gate_reasons") or ["AI全部失败或最终弃权"])[0]) if isinstance(ai_r, dict) else "AI全部失败或最终弃权"
        return _merge_abstain_analysis(pred, ai_r if isinstance(ai_r, dict) else {}, reason)
    raw_item = ai_r.get("raw_item", {}) if isinstance(ai_r.get("raw_item"), dict) else {}
    for k in [
        "score_cluster_audit", "sharp_money_audit", "recommendation_components", "risk_score_candidates",
        "tail_risk_flags", "confidence_downgrade_reason", "market_audit",
        "goal_market_audit", "market_conflicts", "candidate_scores", "public_heat_audit",
        "packet_news_risk_audit", "trap_candidates", "final_score_audit", "family_debate",
        *EXTERNAL_FACT_FIELDS, *FULL_SPECTRUM_AUDIT_FIELDS,
    ]:
        v = ai_r.get(k, None)
        if v in (None, {}, []):
            v = raw_item.get(k, None)
        if k == "evidence_quality_score" and v not in (None, ""):
            pred[k] = int(_clip(_f(v, 0), 0, 100))
        elif v not in (None, {}, []):
            pred[k] = v
    # 本地只做推荐风控展示/降级，不改方向和比分。
    try:
        # 兼容旧版本回归测试中，如果传入 apply_weak_home_tail_risk_protection 处理过的 row
        # 它的 row["recommendation"] 已经修改过，但 _BASE_ADAPT_AI_TO_FRONTEND_V2021 底层（即 adapt_ai_to_frontend）
        # 它是基于最原始的 raw_item 重新取 rec = ai_r.get("recommendation")。
        # 因此，我们需要确保这里的 pred (即 _BASE_ADAPT_AI_TO_FRONTEND_V2021 返回的 dict) 
        # 中的 recommendation、confidence、display_confidence 等指标能完美体现被 weak_home_tail 降级之后的值！
        if ai_r.get("recommendation", {}).get("original_bet_confidence") is not None:
            pred["recommendation"] = ai_r.get("recommendation")
            pred["confidence"] = int(_clip(_f(ai_r.get("recommendation", {}).get("original_bet_confidence", pred.get("confidence", 0)), 0), 0, 100))
            pred["display_confidence"] = ai_r.get("recommendation", {}).get("display_bet_confidence", pred.get("display_confidence", pred["confidence"]))
            pred["risk_adjusted_confidence"] = ai_r.get("recommendation", {}).get("risk_adjusted_bet_confidence", pred.get("risk_adjusted_confidence", pred["confidence"]))
            
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
    # v20.7 P0 去污: 泊松影子矩阵已删除,不再注入 matrix_*(泊松出身)字段。
    # 读盘比分/方向完全由 AI 读盘得出,严禁机械数理锚反向污染盘感。
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
        apply_deep_favorite_loose_euro_guard(pred, match_obj)
    except Exception as e:
        pred.setdefault("validation_warnings", []).append(f"deep_favorite_loose_euro_guard_error:{str(e)[:120]}")
    try:
        apply_pre_match_factor_v2_gate(pred, match_obj)
    except Exception as e:
        pred.setdefault("validation_warnings", []).append(f"prematch_v2_gate_error:{str(e)[:120]}")
    try:
        _apply_external_fact_source_gate(pred)
    except Exception as e:
        pred.setdefault("validation_warnings", []).append(f"external_fact_source_gate_error:{str(e)[:120]}")
    try:
        _apply_direction_candidate_consistency_gate(pred)
    except Exception as e:
        pred.setdefault("validation_warnings", []).append(f"direction_candidate_consistency_gate_error:{str(e)[:120]}")
    try:
        _apply_contrarian_market_claim_gate(pred)
    except Exception as e:
        pred.setdefault("validation_warnings", []).append(f"contrarian_market_claim_gate_error:{str(e)[:120]}")
    try:
        _score_shape_selector(pred, match_obj)
    except Exception as e:
        pred.setdefault("validation_warnings", []).append(f"score_shape_selector_error:{str(e)[:120]}")
    # [补丁A1 2026-06-22] 禁用 consolation gate: 第一轮27场回测净收益=0,
    # 且方向与赛果相反(大胜场x-1比x-0更高频),一旦触发只会打碎唯一可能命中的x-1大胜。
    # 保留函数代码以可追溯,仅摘除调用(回测确认禁用为零回归纯收益)。
    # try:
    #     _apply_lopsided_consolation_goal_gate(pred)
    # except Exception as e:
    #     pred.setdefault("validation_warnings", []).append(f"lopsided_consolation_goal_gate_error:{str(e)[:120]}")
    try:
        _apply_low_confidence_draw_guard(pred)
    except Exception as e:
        pred.setdefault("validation_warnings", []).append(f"low_confidence_draw_guard_error:{str(e)[:120]}")
    try:
        assign_selection_layer(pred, match_obj)
    except Exception as e:
        pred.setdefault("validation_warnings", []).append(f"selection_layer_error:{str(e)[:120]}")
    try:
        _sync_gate_with_bet_action(pred)
    except Exception as e:
        pred.setdefault("validation_warnings", []).append(f"gate_action_sync_error:{str(e)[:120]}")
    # 下注推荐：本地确定性，基于全部 gate 终态赔率算期望值，只新增字段。
    try:
        _apply_bet_recommendation_gate(pred, match_obj)
    except Exception as e:
        pred.setdefault("validation_warnings", []).append(f"bet_recommendation_gate_error:{str(e)[:120]}")
    pred["engine_version"] = ENGINE_VERSION
    pred["engine_architecture"] = ENGINE_ARCHITECTURE
    return pred


if __name__ == "__main__":
    logger.info(f"{ENGINE_VERSION} 启动")
    print(f"✅ {ENGINE_VERSION} 加载完成")
    print(f"   架构: {ENGINE_ARCHITECTURE}")
    print("   模式: 完整版；保留AI模块调用链；新增Sharp/CRS簇/HHAD/相邻比分审计；本地不改最终足球判断。")
