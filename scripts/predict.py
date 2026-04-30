# ====================================================================
# 🚀 vMAX 19.2 — FirstBase Raw Packet 四AI审计版
# --------------------------------------------------------------------
# 这版原则:
#   ✅ AI调用层回到第一版 v19.0 稳定底座
#   ✅ 不限制输出字数，不传 max_tokens / maxOutputTokens
#   ✅ 拉长 GPT/Grok/Gemini/Claude 等待时间
#   ✅ Phase 1: GPT / Grok / Gemini 三家并行独立分析
#   ✅ Phase 1 Repair: 缺失场次自动单场补跑
#   ✅ Phase 2: Claude 接收三家结论 + 原始抓包做最终审计
#   ✅ Claude 不按票数裁决，必须重新审计抓包
#   ✅ 程序只做 JSON 校验、字段兼容、方向一致，不篡改 Claude 最终比分
#   ✅ EV/Kelly 默认不使用 LLM 主观比分概率
# ====================================================================

import json
import os
import re
import time
import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Tuple, Optional


# ====================================================================
# 日志
# ====================================================================

try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)


# ====================================================================
# 安全导入
# ====================================================================

EnsemblePredictor = None
ExperienceEngine = None
predict_match = None
apply_experience_to_prediction = None
upgrade_ensemble_predict = None

try:
    from config import *
except Exception as e:
    logger.warning(f"config 导入异常: {e}")

try:
    from models import EnsemblePredictor
except Exception as e:
    logger.warning(f"models 导入异常: {e}")

try:
    from odds_engine import predict_match
except Exception as e:
    logger.warning(f"odds_engine 导入异常: {e}")

try:
    from league_intel import build_league_intelligence
except Exception as e:
    logger.warning(f"league_intel 导入异常: {e}")

try:
    from experience_rules import ExperienceEngine, apply_experience_to_prediction
except Exception as e:
    logger.warning(f"experience_rules 导入异常: {e}")

try:
    from advanced_models import upgrade_ensemble_predict
except Exception as e:
    logger.warning(f"advanced_models 导入异常: {e}")

try:
    from odds_history import apply_odds_history
except Exception:
    def apply_odds_history(m, mg):
        return mg

try:
    from quant_edge import apply_quant_edge
except Exception:
    def apply_quant_edge(m, mg):
        return mg

try:
    from wencai_intel import apply_wencai_intel
except Exception:
    def apply_wencai_intel(m, mg):
        return mg


try:
    ensemble = EnsemblePredictor() if EnsemblePredictor else None
except Exception as e:
    logger.warning(f"EnsemblePredictor 初始化失败: {e}")
    ensemble = None

try:
    exp_engine = ExperienceEngine() if ExperienceEngine else None
except Exception as e:
    logger.warning(f"ExperienceEngine 初始化失败: {e}")
    exp_engine = None


# ====================================================================
# 基础配置
# ====================================================================

ENGINE_VERSION = "vMAX 19.2"
ENGINE_ARCHITECTURE = "FirstBase Raw Packet + 3 Analysts + Claude Final Audit"

APPLY_LEGACY_ENHANCERS = False
ENABLE_LLM_VALUE_BET = False

AI_CALL_STATUS = {
    "gpt": "",
    "grok": "",
    "gemini": "",
    "claude": "",
}

LOCKED_CORE_FIELDS = {
    "predicted_score",
    "predicted_label",
    "predicted_direction",
    "result",
    "display_direction",
    "final_direction",
    "home_win_pct",
    "draw_pct",
    "away_win_pct",
    "confidence",
    "ai_confidence",
    "ai_confidence_pct",
    "confidence_score",
    "confidence_pct",
    "analysis_confidence",
    "goal_range",
    "goal_interval",
    "predicted_goal_range",
    "goal_range_label",
}

VALID_DIRS = {"home", "draw", "away"}

STANDARD_GOAL_ODDS = {
    0: 9.5,
    1: 5.5,
    2: 3.5,
    3: 4.0,
    4: 7.0,
    5: 14.0,
    6: 30.0,
    7: 70.0,
}

TTG_ANCHORS = {
    0: {"hard_low": 8.0, "std": 9.5},
    1: {"hard_low": 4.5, "std": 5.5},
    2: {"hard_low": 3.0, "std": 3.5},
    3: {"hard_low": 3.8, "std": 4.0},
    4: {"hard_low": 5.0, "std": 7.0},
    5: {"hard_low": 8.0, "std": 14.0},
    6: {"hard_low": 16.0, "std": 30.0},
    7: {"hard_low": 30.0, "std": 70.0},
}

CRS_FULL_MAP = {
    "1-0": "w10",
    "2-0": "w20",
    "2-1": "w21",
    "3-0": "w30",
    "3-1": "w31",
    "3-2": "w32",
    "4-0": "w40",
    "4-1": "w41",
    "4-2": "w42",
    "5-0": "w50",
    "5-1": "w51",
    "5-2": "w52",

    "0-0": "s00",
    "1-1": "s11",
    "2-2": "s22",
    "3-3": "s33",

    "0-1": "l01",
    "0-2": "l02",
    "1-2": "l12",
    "0-3": "l03",
    "1-3": "l13",
    "2-3": "l23",
    "0-4": "l04",
    "1-4": "l14",
    "2-4": "l24",
    "0-5": "l05",
    "1-5": "l15",
    "2-5": "l25",
}

HFTF_MAP = {
    "ss": "主/主",
    "sp": "主/平",
    "sf": "主/负",
    "ps": "平/主",
    "pp": "平/平",
    "pf": "平/负",
    "fs": "负/主",
    "fp": "负/平",
    "ff": "负/负",
}

CUP_KEYWORDS = [
    "杯",
    "淘汰",
    "决赛",
    "半决赛",
    "四分之一",
    "欧冠",
    "欧联",
    "国王杯",
    "足总杯",
    "联赛杯",
    "解放者杯",
    "南球杯",
]


# ====================================================================
# 通用工具
# ====================================================================

def _f(v, default=0.0):
    try:
        if v is None:
            return default
        if isinstance(v, str):
            v = v.replace("%", "").replace(",", "").strip()
            if v == "":
                return default
        return float(v)
    except Exception:
        return default


def _i(v, default=0):
    try:
        return int(round(_f(v, default)))
    except Exception:
        return default


def _safe_str(v, max_len=None) -> str:
    s = "" if v is None else str(v)
    s = s.replace("\r", " ").replace("\n", " ").strip()
    if max_len and len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _ttg_label(g: int) -> str:
    return "7+球" if int(g) >= 7 else f"{int(g)}球"


def _parse_score(s: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        s_str = str(s).strip().replace(" ", "").replace("：", "-").replace(":", "-")

        if "胜" in s_str and "其他" in s_str:
            return 9, 0
        if "平" in s_str and "其他" in s_str:
            return 9, 9
        if "负" in s_str and "其他" in s_str:
            return 0, 9

        if s_str in ["主胜", "客胜", "平局", "home", "away", "draw"]:
            return None, None

        p = s_str.split("-")
        if len(p) != 2:
            return None, None

        return int(p[0]), int(p[1])
    except Exception:
        return None, None


def _score_direction(score_str: str) -> Optional[str]:
    h, a = _parse_score(score_str)
    if h is None:
        return None

    s = str(score_str)

    if "胜其他" in s or score_str == "9-0":
        return "home"
    if "平其他" in s or score_str == "9-9":
        return "draw"
    if "负其他" in s or score_str == "0-9":
        return "away"

    if h > a:
        return "home"
    if h < a:
        return "away"
    return "draw"


def _direction_cn(direction: str) -> str:
    return {
        "home": "主胜",
        "draw": "平局",
        "away": "客胜",
    }.get(direction, "平局")


def _score_to_label(score: str) -> Tuple[str, bool]:
    s = str(score).strip()

    if "胜其他" in s or s == "9-0":
        return "胜其他", True
    if "平其他" in s or s == "9-9":
        return "平其他", True
    if "负其他" in s or s == "0-9":
        return "负其他", True

    return s, False


def _goal_range_from_score(score: str) -> str:
    h, a = _parse_score(score)
    if h is None:
        return "其他"

    total = h + a

    if total <= 1:
        return "0-1球"
    if total == 2:
        return "2球"
    if total == 3:
        return "3球"
    if total == 4:
        return "4球"
    if total == 5:
        return "5球"
    return "6+球"


def _normalize_goal_range_for_ui(goal_range: str, score: str = "") -> Tuple[str, str]:
    s = str(goal_range or "").strip().replace(" ", "")

    if not s or s in ["?", "None", "null", "其他"]:
        s = _goal_range_from_score(score)

    if s in ["0-1球", "0-1", "0~1球", "0到1球"]:
        return "0-1", "0-1球"

    if "6+" in s or "6＋" in s or "六+" in s:
        return "6+", "6+球"

    m = re.search(r"(\d+)", s)
    if m:
        n = int(m.group(1))
        if n >= 6:
            return "6+", "6+球"
        return str(n), f"{n}球"

    return "?", "未知"


def _pct_normalize(h, d, a) -> Tuple[float, float, float]:
    h = _f(h, 33.3)
    d = _f(d, 33.3)
    a = _f(a, 33.4)

    total = h + d + a

    if 0.8 <= total <= 1.2:
        h, d, a = h * 100, d * 100, a * 100
        total = h + d + a

    if total <= 0:
        return 33.3, 33.3, 33.4

    return (
        round(h / total * 100, 1),
        round(d / total * 100, 1),
        round(a / total * 100, 1),
    )


def _norm_key(k: str) -> str:
    return str(k).lower().replace(" ", "").replace("_", "").replace("-", "")


def _deep_find_value(obj, aliases: List[str], positive_only=False, default=0, skip_keys: Optional[List[str]] = None):
    alias_set = {_norm_key(a) for a in aliases}
    skip_set = {_norm_key(k) for k in (skip_keys or [])}

    def _ok(v):
        if positive_only:
            return 1.01 < _f(v, 0) < 100
        return v is not None and str(v).strip() != ""

    def _walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                nk = _norm_key(k)
                if nk in skip_set:
                    continue
                if nk in alias_set and _ok(v):
                    return v
                found = _walk(v)
                if found is not None:
                    return found
        elif isinstance(x, list):
            for item in x:
                found = _walk(item)
                if found is not None:
                    return found
        return None

    found = _walk(obj)
    return found if found is not None else default


def _extract_observation_codes(observation_signals: List[str]) -> List[str]:
    codes = []
    for s in observation_signals:
        m = re.search(r"\[(OBS\d+)[^\]]*\]", str(s))
        if m:
            codes.append(m.group(1))
    return codes


def _extract_observation_labels(observation_signals: List[str]) -> List[str]:
    labels = []

    for s in observation_signals:
        text = str(s)
        m = re.search(r"\[(OBS\d+)\s*([^\]]*)\]", text)

        if m:
            code = m.group(1)
            name = m.group(2).strip()
            labels.append(f"{code} {name}".strip())
        else:
            labels.append(text[:40])

    return labels


def _extract_observation_objects(observation_signals: List[str]) -> List[Dict[str, str]]:
    items = []

    for s in observation_signals:
        text = str(s)
        m = re.search(r"\[(OBS\d+)\s*([^\]]*)\]\s*(.*)", text)

        if m:
            items.append({
                "code": m.group(1),
                "name": m.group(2).strip(),
                "detail": m.group(3).strip(),
                "raw": text,
            })
        else:
            items.append({
                "code": "",
                "name": text[:30],
                "detail": text,
                "raw": text,
            })

    return items


# ====================================================================
# 环境变量 / 第一版 API 通道底座
# ====================================================================

FALLBACK_URLS = [
    None,
    "https://www.api522.pro/v1",
    "https://api522.pro/v1",
    "https://api521.pro/v1",
    "http://69.63.213.33:666/v1",
]

GPT_DEFAULT_URL = "https://ai.newapi.life/v1"
GPT_DEFAULT_KEY = globals().get("GPT_DEFAULT_KEY", "")

GPT_KEY_ALIASES = [
    "GPT_API_KEY","API_KEY"
]

GPT_URL_ALIASES = [
    "GPT_API_URL"
]


def get_clean_env_url(name, default=""):
    v = str(os.environ.get(name, globals().get(name, default))).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/?#=&%-]+)", v)
    return match.group(1) if match else v


def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")


def get_first_clean_env_key(names: List[str], default="") -> str:
    for name in names:
        v = str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")
        if v:
            return v
    return str(default or "").strip(" \t\n\r\"'")


def get_first_clean_env_url(names: List[str], default="") -> str:
    for name in names:
        v = str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")
        if v:
            match = re.search(r"(https?://[a-zA-Z0-9._:/?#=&%-]+)", v)
            return match.group(1) if match else v

    v = str(default or "").strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/?#=&%-]+)", v)
    return match.group(1) if match else v


def _mask_key(v: str) -> str:
    if not v:
        return "EMPTY"
    s = str(v)
    if len(s) >= 8:
        return s[:4] + "****" + s[-4:]
    return "SET"


def debug_ai_config():
    print("\n[AI配置检查]")
    print(f"GPT key    = {_mask_key(get_first_clean_env_key(GPT_KEY_ALIASES, GPT_DEFAULT_KEY))}")
    print(f"GPT url    = {get_first_clean_env_url(GPT_URL_ALIASES, GPT_DEFAULT_URL)}")
    print(f"GROK key   = {_mask_key(get_clean_env_key('GROK_API_KEY'))}")
    print(f"GROK url   = {get_clean_env_url('GROK_API_URL')}")
    print(f"GEMINI key = {_mask_key(get_clean_env_key('GEMINI_API_KEY'))}")
    print(f"GEMINI url = {get_clean_env_url('GEMINI_API_URL')}")
    print(f"CLAUDE key = {_mask_key(get_clean_env_key('CLAUDE_API_KEY'))}")
    print(f"CLAUDE url = {get_clean_env_url('CLAUDE_API_URL')}")


# ====================================================================
# Match 标准化
# ====================================================================

def normalize_match(raw_m: Dict) -> Dict:
    raw_m = raw_m or {}
    m = dict(raw_m)

    for nested_key in [
        "v2_odds_dict",
        "odds_dict",
        "odds",
        "v2",
        "odds_v2",
        "packet",
        "raw_odds",
        "data",
        "detail",
    ]:
        nested = m.get(nested_key)
        if isinstance(nested, dict):
            m.update(nested)

    m["home_team"] = (
        m.get("home_team")
        or m.get("home")
        or m.get("host")
        or m.get("team_home")
        or m.get("homeName")
        or "Home"
    )
    m["away_team"] = (
        m.get("away_team")
        or m.get("guest")
        or m.get("away")
        or m.get("team_away")
        or m.get("awayName")
        or "Away"
    )

    m["home"] = m.get("home") or m["home_team"]
    m["guest"] = m.get("guest") or m["away_team"]

    odds_skip = ["vote", "change", "points", "information", "prediction", "stats", "smart_signals"]

    sp_home = _deep_find_value(m, [
        "sp_home", "win", "odds_win", "home_win", "home_win_odds",
        "odds_home", "h_odds", "had_h", "had_win",
        "spf_sp3", "spf_3", "sp3", "homeOdds", "winOdds", "胜",
    ], positive_only=True, default=0, skip_keys=odds_skip)

    sp_draw = _deep_find_value(m, [
        "sp_draw", "same", "draw", "odds_draw", "draw_odds",
        "had_d", "had_draw", "spf_sp1", "spf_1", "sp1",
        "drawOdds", "sameOdds", "平",
    ], positive_only=True, default=0, skip_keys=odds_skip)

    sp_away = _deep_find_value(m, [
        "sp_away", "lose", "away_win", "odds_away", "away_win_odds",
        "guest_win", "guest_odds", "had_a", "had_lose",
        "spf_sp0", "spf_0", "sp0", "awayOdds", "loseOdds", "负",
    ], positive_only=True, default=0, skip_keys=odds_skip)

    m["sp_home"] = sp_home
    m["sp_draw"] = sp_draw
    m["sp_away"] = sp_away
    m["win"] = sp_home
    m["same"] = sp_draw
    m["lose"] = sp_away

    m["give_ball"] = (
        m.get("give_ball")
        if m.get("give_ball") not in [None, ""]
        else m.get("handicap", m.get("rq", m.get("let_ball", "0")))
    )

    change = m.get("change", {})
    if not isinstance(change, dict):
        change = {}

    change["win"] = _deep_find_value(m, [
        "change_win", "cw", "home_change", "win_change",
        "odds_change_home", "change_sp3", "sp3_change",
    ], positive_only=False, default=change.get("win", 0), skip_keys=["vote", "points", "information", "prediction"])

    change["same"] = _deep_find_value(m, [
        "change_same", "cs", "draw_change", "same_change",
        "odds_change_draw", "change_sp1", "sp1_change",
    ], positive_only=False, default=change.get("same", 0), skip_keys=["vote", "points", "information", "prediction"])

    change["lose"] = _deep_find_value(m, [
        "change_lose", "cl", "away_change", "lose_change",
        "odds_change_away", "change_sp0", "sp0_change",
    ], positive_only=False, default=change.get("lose", 0), skip_keys=["vote", "points", "information", "prediction"])

    m["change"] = change

    vote = m.get("vote", {})
    if not isinstance(vote, dict):
        vote = {}

    vote["win"] = _deep_find_value(m, [
        "vote_win", "hot_home", "public_home", "win_vote",
        "home_vote", "vote_sp3", "support_home",
    ], positive_only=False, default=vote.get("win", 0), skip_keys=["change", "points", "information", "prediction"])

    vote["same"] = _deep_find_value(m, [
        "vote_same", "hot_draw", "public_draw", "draw_vote",
        "same_vote", "vote_sp1", "support_draw",
    ], positive_only=False, default=vote.get("same", 0), skip_keys=["change", "points", "information", "prediction"])

    vote["lose"] = _deep_find_value(m, [
        "vote_lose", "hot_away", "public_away", "away_vote",
        "lose_vote", "vote_sp0", "support_away",
    ], positive_only=False, default=vote.get("lose", 0), skip_keys=["change", "points", "information", "prediction"])

    m["vote"] = vote

    return m


# ====================================================================
# 欧赔 / 盘口 / 基本面解析
# ====================================================================

def _compute_no_vig_probs(match_obj: Dict) -> Dict[str, float]:
    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_d = _f(match_obj.get("sp_draw", match_obj.get("same", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))

    if sp_h > 1 and sp_d > 1 and sp_a > 1:
        margin = 1 / sp_h + 1 / sp_d + 1 / sp_a
        if margin > 0:
            return {
                "home": (1 / sp_h) / margin * 100,
                "draw": (1 / sp_d) / margin * 100,
                "away": (1 / sp_a) / margin * 100,
                "margin": margin,
            }

    return {
        "home": 33.3,
        "draw": 33.3,
        "away": 33.4,
        "margin": 0.0,
    }


def _infer_theoretical_handicap(sp_h: float, sp_d: float, sp_a: float) -> float:
    if sp_h <= 1.01 or sp_d <= 1.01 or sp_a <= 1.01:
        return 0.0

    probs = _compute_no_vig_probs({
        "sp_home": sp_h,
        "sp_draw": sp_d,
        "sp_away": sp_a,
    })

    home_p = probs["home"] / 100.0
    draw_p = probs["draw"] / 100.0
    away_p = probs["away"] / 100.0

    edge = home_p - away_p
    draw_adjust = 1.0 - 0.50 * max(0.0, draw_p - 0.27)
    effective_edge = edge * draw_adjust

    if effective_edge >= 0.48:
        return 2.25
    if effective_edge >= 0.42:
        return 1.75
    if effective_edge >= 0.34:
        return 1.25
    if effective_edge >= 0.25:
        return 0.75
    if effective_edge >= 0.12:
        return 0.25
    if effective_edge > -0.12:
        return 0.0
    if effective_edge > -0.25:
        return -0.25
    if effective_edge > -0.34:
        return -0.75
    if effective_edge > -0.42:
        return -1.25
    if effective_edge > -0.48:
        return -1.75
    return -2.25


def _parse_actual_handicap(match_obj: Dict) -> float:
    raw = match_obj.get("give_ball", match_obj.get("handicap", "0"))
    s = str(raw).strip().replace(" ", "")

    if not s:
        return 0.0

    def _extract_mag(x: str) -> float:
        nums = re.findall(r"-?\d+\.?\d*", x)
        if not nums:
            return 0.0
        vals = [abs(_f(n)) for n in nums]
        return sum(vals) / len(vals)

    mag = _extract_mag(s)

    if mag == 0:
        return 0.0

    if "主受让" in s or "客让" in s:
        return -mag
    if "主让" in s or "客受让" in s:
        return +mag

    if "/" in s:
        nums = re.findall(r"-?\d+\.?\d*", s)
        if nums:
            vals = [_f(n) for n in nums]
            avg = sum(vals) / len(vals)
            return -avg

    numeric = _f(s, None)
    if numeric is not None:
        return -numeric

    return 0.0


def _extract_form_record(text: str) -> Tuple[int, int, int]:
    if not text:
        return 0, 0, 0

    patterns = [
        r"近\s*\d+\s*[主客场]*\s*(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
        r"(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
    ]

    for pat in patterns:
        m = re.search(pat, str(text))
        if m:
            try:
                return int(m.group(1)), int(m.group(2)), int(m.group(3))
            except Exception:
                pass

    return 0, 0, 0


def _extract_avg_goals(text: str) -> Tuple[float, float]:
    if not text:
        return 0.0, 0.0

    gf_match = re.search(r"场均进球[^0-9]*(\d+\.?\d*)", str(text))
    ga_match = re.search(r"场均失球[^0-9]*(\d+\.?\d*)", str(text))

    return (
        float(gf_match.group(1)) if gf_match else 0.0,
        float(ga_match.group(1)) if ga_match else 0.0,
    )


# ====================================================================
# Sharp / Steam 方向识别：仅作为观察，不做决策
# ====================================================================

def _detect_sharp_direction(smart_signals: List) -> Optional[str]:
    for s in smart_signals:
        s_str = str(s)
        if "Sharp" not in s_str and "sharp" not in s_str:
            continue

        if re.search(r"(主胜|主队|走主|→\s*主|流向\s*主|资金\s*主|Sharp主)", s_str):
            return "home"
        if re.search(r"(客胜|客队|走客|→\s*客|流向\s*客|资金\s*客|Sharp客)", s_str):
            return "away"
        if re.search(r"(平局|平赔|走平|→\s*平|流向\s*平|资金\s*平|Sharp平)", s_str):
            return "draw"

    return None


def _detect_steam_direction(smart_signals: List) -> Tuple[Optional[str], Optional[str]]:
    for s in smart_signals:
        s_str = str(s)
        if "Steam" not in s_str and "steam" not in s_str:
            continue

        is_reverse = "反向" in s_str or "未跟" in s_str or "不跟" in s_str
        steam_type = "reverse" if is_reverse else "normal"

        if re.search(r"(主胜.*Steam|Steam.*主胜|主胜.*降水|主.*Steam)", s_str):
            return "home", steam_type
        if re.search(r"(客胜.*Steam|Steam.*客胜|客胜.*降水|客.*Steam)", s_str):
            return "away", steam_type
        if re.search(r"(平.*Steam|Steam.*平)", s_str):
            return "draw", steam_type

    return None, None


# ====================================================================
# OBS 中性观察信号
# ====================================================================

def build_observation_signals(
    match_obj: Dict,
    engine_result: Dict,
    smart_signals: List,
    exp_goals: float
) -> List[str]:
    facts = []

    probs = _compute_no_vig_probs(match_obj)

    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_d = _f(match_obj.get("sp_draw", match_obj.get("same", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))

    change = match_obj.get("change", {}) or {}
    cw = _f(change.get("win", 0))
    cs = _f(change.get("same", 0))
    cl = _f(change.get("lose", 0))

    vote = match_obj.get("vote", {}) or {}
    vh = _f(vote.get("win", 0))
    vd = _f(vote.get("same", 0))
    va = _f(vote.get("lose", 0))

    total_move = abs(cw) + abs(cs) + abs(cl)

    if total_move > 0:
        if cs <= -0.04 and cs < cw and cs < cl:
            facts.append(
                f"[OBS01 平赔独降] 平赔变化 {cs:+.2f}, 主胜变化 {cw:+.2f}, 客胜变化 {cl:+.2f}。"
            )
        if cw <= -0.04 and cw < cs and cw < cl:
            facts.append(
                f"[OBS02 主胜降水] 主胜变化 {cw:+.2f}, 平赔变化 {cs:+.2f}, 客胜变化 {cl:+.2f}。"
            )
        if cl <= -0.04 and cl < cw and cl < cs:
            facts.append(
                f"[OBS03 客胜降水] 客胜变化 {cl:+.2f}, 主胜变化 {cw:+.2f}, 平赔变化 {cs:+.2f}。"
            )

    if sp_h > 1.05 and sp_d > 1.05 and sp_a > 1.05:
        theoretical = _infer_theoretical_handicap(sp_h, sp_d, sp_a)
        actual = _parse_actual_handicap(match_obj)
        diff = actual - theoretical

        if abs(diff) >= 0.5:
            facts.append(
                f"[OBS04 让球深度差异] 实际让球 {actual:+.2f}, 欧赔反推理论 {theoretical:+.2f}, "
                f"差异 {diff:+.2f} 球。"
            )

    max_dir = max(["home", "draw", "away"], key=lambda k: probs[k])
    if probs[max_dir] >= 45:
        facts.append(
            f"[OBS05 欧赔去水倾向] 主 {probs['home']:.1f}% / 平 {probs['draw']:.1f}% / "
            f"客 {probs['away']:.1f}%, 最高方向={max_dir}。"
        )

    if max(vh, vd, va) >= 58:
        hot_dir = "主胜" if vh >= max(vd, va) else ("平局" if vd >= max(vh, va) else "客胜")
        hot_val = max(vh, vd, va)
        facts.append(f"[OBS06 散户热度] {hot_dir}散户占比 {hot_val:.0f}%。")

    low_ttg = []
    for g in range(8):
        odds = _f(match_obj.get(f"a{g}", 0))
        anchor = TTG_ANCHORS.get(g, {})
        hard_low = anchor.get("hard_low", 0)
        if odds > 1 and hard_low > 0 and odds <= hard_low:
            low_ttg.append(f"{_ttg_label(g)}={odds:.2f}")

    if low_ttg:
        facts.append("[OBS07 总进球低赔点] " + " | ".join(low_ttg) + "。")

    if exp_goals > 0:
        if exp_goals >= 2.8:
            facts.append(f"[OBS08 期望总进球偏高] λ={exp_goals:.2f}。")
        elif exp_goals <= 2.2:
            facts.append(f"[OBS09 期望总进球偏低] λ={exp_goals:.2f}。")

    crs_items = []
    for sc, key in CRS_FULL_MAP.items():
        v = _f(match_obj.get(key, 0))
        if v > 1:
            crs_items.append((sc, key, v))

    if len(crs_items) < 8:
        facts.append(f"[OBS10 CRS数据不足] 当前可见CRS项 {len(crs_items)}/27。")
    else:
        lowest = sorted(crs_items, key=lambda x: x[2])[:5]
        facts.append(
            "[OBS11 CRS低赔Top5] " + " | ".join([f"{sc}={v:.1f}" for sc, _, v in lowest]) + "。"
        )

    hftf_items = []
    for k, label in HFTF_MAP.items():
        v = _f(match_obj.get(k, 0))
        if v > 1:
            hftf_items.append((label, v))

    if hftf_items:
        low_hftf = sorted(hftf_items, key=lambda x: x[1])[:3]
        facts.append(
            "[OBS12 半全场低赔Top3] " + " | ".join([f"{label}={v:.2f}" for label, v in low_hftf]) + "。"
        )

    points = match_obj.get("points", {}) or {}
    h_txt = str(points.get("home_strength", ""))
    a_txt = str(points.get("guest_strength", ""))

    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))

    if hxg > 0 and axg > 0:
        h_for, _ = _extract_avg_goals(h_txt)
        a_for, _ = _extract_avg_goals(a_txt)

        diffs = []
        if h_for > 0 and abs(hxg - h_for) >= 0.8:
            diffs.append(f"主xG={hxg:.2f} vs 主场均进球={h_for:.2f}")
        if a_for > 0 and abs(axg - a_for) >= 0.8:
            diffs.append(f"客xG={axg:.2f} vs 客场均进球={a_for:.2f}")

        if diffs:
            facts.append("[OBS13 xG与场均差异] " + "；".join(diffs) + "。")

    league = str(match_obj.get("league", match_obj.get("cup", "")))
    is_cup = any(kw in league for kw in CUP_KEYWORDS)
    if is_cup:
        facts.append(f"[OBS14 杯赛属性] 联赛字段={league}, 需要考虑淘汰赛/轮换/保守性。")

    sharp_dir = _detect_sharp_direction(smart_signals)
    steam_dir, steam_type = _detect_steam_direction(smart_signals)

    if sharp_dir:
        facts.append(f"[OBS15 Sharp方向] smart_signals中检测到 Sharp 方向={sharp_dir}。")
    if steam_dir:
        facts.append(f"[OBS16 Steam方向] smart_signals中检测到 Steam 方向={steam_dir}, 类型={steam_type}。")

    sigs_short = [str(s) for s in smart_signals[:8]]
    if sigs_short:
        facts.append("[OBS17 原始智能信号] " + " | ".join(sigs_short))

    return facts


# ====================================================================
# Ensemble 信号：仅参考
# ====================================================================

def _normalize_model_probs(h, d, a):
    h, d, a = _f(h), _f(d), _f(a)
    s = h + d + a

    if 0.8 <= s <= 1.2:
        return h * 100, d * 100, a * 100

    if 80 <= s <= 120:
        return h, d, a

    return None


def collect_ensemble_signals(stats: Dict) -> Dict[str, Any]:
    if not stats:
        return {
            "models": [],
            "consensus": None,
            "consensus_count": 0,
            "total": 0,
            "top_scores": [],
        }

    model_keys = [
        ("refined_poisson", "Refined Poisson"),
        ("elo", "Elo"),
        ("dixon_coles", "Dixon-Coles"),
        ("bradley_terry", "Bradley-Terry"),
        ("random_forest", "Random Forest"),
        ("gradient_boost", "Gradient Boost"),
        ("neural_net", "Neural Net"),
        ("logistic", "Logistic"),
        ("svm", "SVM"),
        ("knn", "KNN"),
    ]

    rows = []
    direction_counts = {"home": 0, "draw": 0, "away": 0}

    for key, name in model_keys:
        m = stats.get(key, {}) or {}
        if not isinstance(m, dict):
            continue

        h = m.get("home_win_pct", m.get("home", 0))
        d = m.get("draw_pct", m.get("draw", 0))
        a = m.get("away_win_pct", m.get("away", 0))

        norm = _normalize_model_probs(h, d, a)
        if not norm:
            continue

        h, d, a = norm

        if max(h, d, a) == h:
            direction = "home"
        elif max(h, d, a) == a:
            direction = "away"
        else:
            direction = "draw"

        direction_counts[direction] += 1

        rows.append({
            "name": name,
            "home": round(h, 1),
            "draw": round(d, 1),
            "away": round(a, 1),
            "direction": direction,
        })

    total = len(rows)
    consensus = max(direction_counts, key=direction_counts.get) if total > 0 else None
    consensus_count = direction_counts[consensus] if consensus else 0

    rp = stats.get("refined_poisson", {}) or {}
    top_scores = rp.get("top_scores", [])[:5] if isinstance(rp, dict) else []

    return {
        "models": rows,
        "consensus": consensus,
        "consensus_count": consensus_count,
        "total": total,
        "top_scores": top_scores,
    }


# ====================================================================
# 抓包格式化
# ====================================================================

def format_match_block(
    idx: int,
    match: Dict,
    engine_result: Dict,
    observation_signals: List[str],
    ensemble_signals: Dict,
    smart_signals: List
) -> str:
    home = match.get("home_team", match.get("home", "Home"))
    away = match.get("away_team", match.get("guest", "Away"))
    league = match.get("league", match.get("cup", ""))
    is_cup = any(kw in str(league) for kw in CUP_KEYWORDS)

    sp_h = _f(match.get("sp_home", match.get("win", 0)))
    sp_d = _f(match.get("sp_draw", match.get("same", 0)))
    sp_a = _f(match.get("sp_away", match.get("lose", 0)))

    probs = _compute_no_vig_probs(match)

    actual_hc = _parse_actual_handicap(match)
    theoretical_hc = _infer_theoretical_handicap(sp_h, sp_d, sp_a) if sp_h > 1 and sp_d > 1 and sp_a > 1 else 0.0
    hc_raw = match.get("give_ball", match.get("handicap", "0"))

    block = f"\n════════════════════════════════════\n"
    block += f"第 {idx} 场: {home} vs {away}\n"
    block += f"════════════════════════════════════\n"
    block += f"联赛/赛事: {league}{' [杯赛/淘汰赛属性]' if is_cup else ''}\n"
    block += f"比赛编号: {match.get('match_num', match.get('id', idx))}\n"

    block += "\n【1. 欧赔原始数据】\n"
    block += f"即时欧赔: 主胜 {sp_h:.2f} / 平局 {sp_d:.2f} / 客胜 {sp_a:.2f}\n"

    if probs.get("margin", 0) > 0:
        block += (
            f"欧赔去水概率: 主 {probs['home']:.1f}% / 平 {probs['draw']:.1f}% / "
            f"客 {probs['away']:.1f}% / 返还率约 {100 / probs['margin']:.1f}%\n"
        )
    else:
        block += "欧赔去水概率: 数据不足\n"

    block += "\n【2. 让球/盘口】\n"
    block += f"原始 give_ball/handicap: {hc_raw}\n"
    block += f"标准化实际让球: {actual_hc:+.2f}，内部约定：正数=主让，负数=客让/主受让\n"
    block += f"欧赔反推理论让球: {theoretical_hc:+.2f}\n"
    block += f"实际 - 理论 差异: {actual_hc - theoretical_hc:+.2f} 球\n"

    change = match.get("change", {}) or {}
    cw = _f(change.get("win", 0))
    cs = _f(change.get("same", 0))
    cl = _f(change.get("lose", 0))

    block += "\n【3. 赔率变动】\n"
    block += f"主胜变化 {cw:+.2f} / 平赔变化 {cs:+.2f} / 客胜变化 {cl:+.2f}\n"
    block += "说明: 负数=降水/赔率下调，正数=升水/赔率上调。\n"

    hxg = engine_result.get("bookmaker_implied_home_xg", None)
    axg = engine_result.get("bookmaker_implied_away_xg", None)
    exp_total = _f(engine_result.get("expected_total_goals", 0))
    if exp_total <= 0:
        exp_total = _f(hxg, 0) + _f(axg, 0)

    block += "\n【4. 庄家隐含 xG / 期望进球】\n"
    block += f"主 xG: {hxg if hxg is not None else 'N/A'} / 客 xG: {axg if axg is not None else 'N/A'}\n"
    block += f"期望总进球: {exp_total:.2f}\n"
    block += f"大2.5概率: {engine_result.get('over_25', 'N/A')} / 双方进球BTTS: {engine_result.get('btts', 'N/A')}\n"

    block += "\n【5. 总进球数赔率 a0~a7】\n"
    ttg_lines = []
    for g in range(8):
        v = _f(match.get(f"a{g}", 0))
        if v > 1:
            anchor = TTG_ANCHORS.get(g, {})
            hard_low = anchor.get("hard_low", 0)
            mark = ""
            if hard_low and v <= hard_low:
                mark = " [低赔点]"
            ttg_lines.append(f"{_ttg_label(g)}={v:.2f}{mark}")
        else:
            ttg_lines.append(f"{_ttg_label(g)}=N/A")
    block += " | ".join(ttg_lines) + "\n"

    block += "\n【6. 精确比分赔率 CRS】\n"

    crs_h = []
    crs_d = []
    crs_a = []

    for sc, key in CRS_FULL_MAP.items():
        v = _f(match.get(key, 0))
        if v <= 1:
            continue

        text = f"{sc}={v:.1f}"
        direction = _score_direction(sc)

        if direction == "home":
            crs_h.append(text)
        elif direction == "draw":
            crs_d.append(text)
        elif direction == "away":
            crs_a.append(text)

    block += "主胜系: " + (" | ".join(crs_h) if crs_h else "N/A") + "\n"
    block += "平局系: " + (" | ".join(crs_d) if crs_d else "N/A") + "\n"
    block += "客胜系: " + (" | ".join(crs_a) if crs_a else "N/A") + "\n"

    others = []
    for k, label in [
        ("crs_win", "胜其他"),
        ("crs_same", "平其他"),
        ("crs_lose", "负其他"),
    ]:
        v = _f(match.get(k, 0))
        if v > 1:
            others.append(f"{label}={v:.1f}")

    if others:
        block += "其他比分: " + " | ".join(others) + "\n"

    crs_items = []
    for sc, key in CRS_FULL_MAP.items():
        v = _f(match.get(key, 0))
        if v > 1:
            crs_items.append((sc, v))

    if crs_items:
        low = sorted(crs_items, key=lambda x: x[1])[:6]
        block += "CRS低赔Top6: " + " | ".join([f"{sc}={v:.1f}" for sc, v in low]) + "\n"

    hf_lines = []
    for k, label in HFTF_MAP.items():
        v = _f(match.get(k, 0))
        if v > 1:
            hf_lines.append(f"{label}={v:.2f}")

    if hf_lines:
        block += "\n【7. 半全场赔率】\n"
        block += " | ".join(hf_lines) + "\n"

    vote = match.get("vote", {}) or {}
    if vote:
        vh = vote.get("win", "?")
        vd = vote.get("same", "?")
        va = vote.get("lose", "?")
        block += "\n【8. 散户分布】\n"
        block += f"主胜 {vh}% / 平局 {vd}% / 客胜 {va}%\n"

    sharp_dir = _detect_sharp_direction(smart_signals)
    steam_dir, steam_type = _detect_steam_direction(smart_signals)

    if smart_signals or sharp_dir or steam_dir:
        block += "\n【9. 资金/智能信号原文】\n"
        if sharp_dir:
            block += f"识别到 Sharp 方向: {sharp_dir}\n"
        if steam_dir:
            block += f"识别到 Steam 方向: {steam_dir}, 类型={steam_type}\n"

        for s in smart_signals[:12]:
            block += f"- {_safe_str(s, 260)}\n"

    points = match.get("points", {}) or {}
    if isinstance(points, dict):
        block += "\n【10. 基本面情报】\n"

        h_text = _safe_str(points.get("home_strength", ""), 600)
        a_text = _safe_str(points.get("guest_strength", ""), 600)
        m_text = _safe_str(points.get("match_points", ""), 500)
        h2h_text = _safe_str(points.get("history", points.get("h2h", "")), 400)

        if h_text:
            block += f"主队: {h_text}\n"
        if a_text:
            block += f"客队: {a_text}\n"
        if h2h_text:
            block += f"交锋: {h2h_text}\n"
        if m_text:
            block += f"赛事要点: {m_text}\n"

    info = match.get("information", {}) or {}
    if isinstance(info, dict):
        info_lines = []

        for k, label in [
            ("home_injury", "主伤停"),
            ("guest_injury", "客伤停"),
            ("home_bad_news", "主利空"),
            ("guest_bad_news", "客利空"),
            ("home_good_news", "主利好"),
            ("guest_good_news", "客利好"),
        ]:
            v = info.get(k)
            if v:
                info_lines.append(f"{label}: {_safe_str(v, 300)}")

        if info_lines:
            block += "\n【11. 伤停/异动消息】\n"
            block += "\n".join(info_lines) + "\n"

    if ensemble_signals.get("total", 0) > 0:
        block += "\n【12. 统计模型矩阵，仅供参考】\n"

        for row in ensemble_signals["models"]:
            block += (
                f"{row['name']}: 主{row['home']:.1f}% / 平{row['draw']:.1f}% / "
                f"客{row['away']:.1f}% → {row['direction']}\n"
            )

        consensus = ensemble_signals.get("consensus")
        if consensus:
            ratio = ensemble_signals["consensus_count"] / max(1, ensemble_signals["total"])
            block += (
                f"模型共识: {ensemble_signals['consensus_count']}/{ensemble_signals['total']} "
                f"倾向 {consensus} ({ratio:.0%})\n"
            )

        ts = ensemble_signals.get("top_scores", [])
        if ts:
            ts_str = []
            for t in ts[:5]:
                if isinstance(t, dict):
                    ts_str.append(f"{t.get('score', '?')}({t.get('prob', '?')}%)")
            if ts_str:
                block += "Refined Poisson Top比分: " + " | ".join(ts_str) + "\n"

    if observation_signals:
        block += "\n【13. 系统 OBS 观察信号，仅供参考，可采纳也可否决】\n"
        for fact in observation_signals:
            block += f"- {fact}\n"
    else:
        block += "\n【13. 系统 OBS 观察信号】\n无明显 OBS。\n"

    return block


# ====================================================================
# Prompt 构建：Phase 1 三家分析
# ====================================================================

PHASE1_ROLES = {
    "gpt": {
        "name": "赔率结构分析师",
        "focus": "欧赔、让球、总进球、CRS、半全场的一致性与裂痕。重点看赔率结构是否支持方向和比分。",
        "temperature": 0.18,
    },
    "grok": {
        "name": "资金流分析师",
        "focus": "Sharp、Steam、赔率变动、散户冷热、赔率是否跟随资金。重点看资金信号与盘口是否共振或冲突。",
        "temperature": 0.28,
    },
    "gemini": {
        "name": "基本面分析师",
        "focus": "战绩、伤停、赛程、联赛属性、杯赛属性、主客场强弱。重点看真实实力和比赛动机。",
        "temperature": 0.20,
    },
}


def build_phase1_prompt(match_blocks: List[str], role_key: str) -> str:
    role = PHASE1_ROLES[role_key]

    p = ""
    p += "<context>\n"
    p += "你是竞彩足球分析团队中的一名独立分析师。下面是多场比赛的完整抓包数据。\n"
    p += "你需要基于原始抓包自主分析。系统 OBS 观察信号只是线索，可能有效，也可能是噪声。\n"
    p += "</context>\n\n"

    p += "<your_role>\n"
    p += f"角色: {role['name']}\n"
    p += f"专长: {role['focus']}\n"
    p += "</your_role>\n\n"

    p += "<rules>\n"
    p += "1. 必须先从你的专长角度分析，再检查其他维度是否反对你的结论。\n"
    p += "2. 不要机械服从 OBS 信号，也不要机械反打 OBS 信号。\n"
    p += "3. 必须给出 predicted_score、predicted_direction、goal_range、top3。\n"
    p += "4. predicted_score 的方向必须与 predicted_direction 一致。\n"
    p += "5. top3[0] 必须等于 predicted_score。\n"
    p += "6. 必须列出 doubts，即反对自己结论的证据。\n"
    p += "7. 如果数据不足，要降低 confidence。\n"
    p += "8. 推理不限制字数，但必须保持 JSON 可解析，不要输出 markdown。\n"
    p += "</rules>\n\n"

    p += "<match_data>\n"
    for block in match_blocks:
        p += block
    p += "\n</match_data>\n\n"

    p += "<output_format>\n"
    p += "严格输出 JSON 数组，不要 markdown，不要代码块，不要前缀后缀。\n"
    p += "[\n"
    p += "  {\n"
    p += "    \"match\": 1,\n"
    p += "    \"predicted_score\": \"2-1\",\n"
    p += "    \"predicted_direction\": \"home\",\n"
    p += "    \"goal_range\": \"3球\",\n"
    p += "    \"home_win_pct\": 52,\n"
    p += "    \"draw_pct\": 27,\n"
    p += "    \"away_win_pct\": 21,\n"
    p += "    \"confidence\": 68,\n"
    p += "    \"top3\": [\n"
    p += "      {\"score\":\"2-1\", \"prob\":17},\n"
    p += "      {\"score\":\"1-0\", \"prob\":14},\n"
    p += "      {\"score\":\"1-1\", \"prob\":13}\n"
    p += "    ],\n"
    p += "    \"accepted_observations\": [\"CRS主胜小比分低赔集中\"],\n"
    p += "    \"rejected_observations\": [\"散户热度不足以单独反打\"],\n"
    p += "    \"key_signals\": [\"主胜赔率结构更完整\", \"总进球更接近3球\"],\n"
    p += "    \"doubts\": [\"平局仍有保护\", \"xG差距不大\"],\n"
    p += "    \"data_quality\": {\"odds_complete\": true, \"crs_complete\": true, \"ttg_complete\": true, \"notes\": []},\n"
    p += "    \"reason\": \"完整中文推理，不限制字数，但必须保持 JSON 字符串合法\"\n"
    p += "  }\n"
    p += "]\n"
    p += "</output_format>\n"

    return p


def build_phase1_retry_prompt(match_block: str, role_key: str, target_match_id: int) -> str:
    role = PHASE1_ROLES[role_key]

    p = ""
    p += "<context>\n"
    p += "这是一次补分析。你只需要分析下面这一场比赛。\n"
    p += f"重要：输出 JSON 中 match 必须等于 {target_match_id}。\n"
    p += "系统 OBS 只是观察线索，可以采纳也可以否决。\n"
    p += "</context>\n\n"

    p += "<your_role>\n"
    p += f"角色: {role['name']}\n"
    p += f"专长: {role['focus']}\n"
    p += "</your_role>\n\n"

    p += "<rules>\n"
    p += "1. 只分析这一场，不要输出其他比赛。\n"
    p += "2. predicted_score 必须与 predicted_direction 一致。\n"
    p += "3. top3[0] 必须等于 predicted_score。\n"
    p += "4. 推理不限制字数，但必须保持 JSON 可解析。\n"
    p += "5. 严格 JSON 数组输出，不要 markdown。\n"
    p += "</rules>\n\n"

    p += "<match_data>\n"
    p += match_block
    p += "\n</match_data>\n\n"

    p += "<output_format>\n"
    p += "[\n"
    p += "  {\n"
    p += f"    \"match\": {target_match_id},\n"
    p += "    \"predicted_score\": \"2-1\",\n"
    p += "    \"predicted_direction\": \"home\",\n"
    p += "    \"goal_range\": \"3球\",\n"
    p += "    \"home_win_pct\": 52,\n"
    p += "    \"draw_pct\": 27,\n"
    p += "    \"away_win_pct\": 21,\n"
    p += "    \"confidence\": 68,\n"
    p += "    \"top3\": [\n"
    p += "      {\"score\":\"2-1\", \"prob\":17},\n"
    p += "      {\"score\":\"1-0\", \"prob\":14},\n"
    p += "      {\"score\":\"1-1\", \"prob\":13}\n"
    p += "    ],\n"
    p += "    \"accepted_observations\": [],\n"
    p += "    \"rejected_observations\": [],\n"
    p += "    \"key_signals\": [],\n"
    p += "    \"doubts\": [],\n"
    p += "    \"reason\": \"完整中文推理，不限制字数，但必须保持 JSON 字符串合法\"\n"
    p += "  }\n"
    p += "]\n"
    p += "</output_format>\n"

    return p


# ====================================================================
# Prompt 构建：Phase 2 Claude 终审
# ====================================================================

def _phase1_summary_line(r: Dict) -> str:
    if not isinstance(r, dict) or not r:
        return "无数据"

    top3 = r.get("top3", [])
    top3_str = ""

    if isinstance(top3, list):
        tmp = []
        for t in top3[:3]:
            if isinstance(t, dict):
                tmp.append(f"{t.get('score', '?')}({t.get('prob', '?')}%)")
        top3_str = ", ".join(tmp)

    return (
        f"方向={r.get('predicted_direction', '?')} | "
        f"比分={r.get('predicted_score', '?')} | "
        f"进球区间={r.get('goal_range_label', r.get('goal_range', '?'))} | "
        f"信心={r.get('confidence', '?')} | "
        f"top3=[{top3_str}]"
    )


def build_phase2_prompt(match_blocks: List[str], phase1_results: Dict[str, Dict[int, Dict]]) -> str:
    num_matches = len(match_blocks)

    p = ""
    p += "<context>\n"
    p += "你是 Claude 终审复盘者。GPT / Grok / Gemini 已经分别从赔率结构、资金流、基本面角度完成独立分析。\n"
    p += "你现在拿到三家完整结论 + 原始抓包，需要做最终审计并输出最终预测。\n"
    p += "</context>\n\n"

    p += "<critical_rules>\n"
    p += "1. 你不是投票裁判，不能简单按三家多数派裁决。\n"
    p += "2. 你必须重新审计原始抓包。三家结论只是待验证材料。\n"
    p += "3. 当三家一致但原始数据不支持时，你可以推翻三家。\n"
    p += "4. 当二对一时，按证据质量裁决，不按人数裁决。\n"
    p += "5. 证据优先级：CRS/总进球赔率结构 > 欧赔/让球一致性 > xG > Sharp/Steam/变盘 > 基本面 > 散户热度。\n"
    p += "6. OBS 只是观察，不是诱盘结论。\n"
    p += "7. 如果三家共同依赖同一个 OBS 信号，要降低共识权重。\n"
    p += "8. predicted_score 方向必须与 predicted_direction 一致。\n"
    p += "9. top3[0] 必须等于 predicted_score。\n"
    p += "10. 必须说明采纳/否决哪些分析师，以及为什么。\n"
    p += "11. 推理不限制字数，但必须保持 JSON 可解析。\n"
    p += "</critical_rules>\n\n"

    p += "<coverage_rules>\n"
    p += "如果某一家显示无数据，说明该分析师本场缺席，不得把它当作反对或支持票。\n"
    p += "你必须在输出中写 analysis_coverage，说明本场实际参与分析的 AI 数量。\n"
    p += "如果只有 1 家或 2 家 Phase1 分析有效，confidence 不得超过 70，除非原始抓包证据极强。\n"
    p += "</coverage_rules>\n\n"

    p += "<three_analysts_results>\n"

    for i in range(1, num_matches + 1):
        p += f"\n════════ 第 {i} 场三家结论 ════════\n"

        for ai_name in ["gpt", "grok", "gemini"]:
            r = phase1_results.get(ai_name, {}).get(i, {})
            p += f"\n【{ai_name.upper()}】{_phase1_summary_line(r)}\n"

            if not r:
                continue

            reason = _safe_str(r.get("reason", ""), 2000)
            key_signals = r.get("key_signals", [])
            accepted = r.get("accepted_observations", [])
            rejected = r.get("rejected_observations", [])
            doubts = r.get("doubts", [])

            if key_signals:
                p += "key_signals: " + " | ".join(str(x) for x in key_signals[:8]) + "\n"
            if accepted:
                p += "accepted_observations: " + " | ".join(str(x) for x in accepted[:8]) + "\n"
            if rejected:
                p += "rejected_observations: " + " | ".join(str(x) for x in rejected[:8]) + "\n"
            if doubts:
                p += "doubts: " + " | ".join(str(x) for x in doubts[:8]) + "\n"
            if reason:
                p += f"reason: {reason}\n"

    p += "\n</three_analysts_results>\n\n"

    p += "<raw_match_data>\n"
    for block in match_blocks:
        p += block
    p += "\n</raw_match_data>\n\n"

    p += "<output_format>\n"
    p += "严格输出 JSON 数组，不要 markdown，不要代码块，不要前缀后缀。\n"
    p += "[\n"
    p += "  {\n"
    p += "    \"match\": 1,\n"
    p += "    \"predicted_score\": \"2-1\",\n"
    p += "    \"predicted_direction\": \"home\",\n"
    p += "    \"goal_range\": \"3球\",\n"
    p += "    \"home_win_pct\": 55,\n"
    p += "    \"draw_pct\": 26,\n"
    p += "    \"away_win_pct\": 19,\n"
    p += "    \"confidence\": 72,\n"
    p += "    \"agreement_pattern\": \"GPT/Grok主胜，Gemini平局\",\n"
    p += "    \"analysis_coverage\": {\"gpt\": true, \"grok\": true, \"gemini\": true, \"valid_count\": 3},\n"
    p += "    \"adopted_analysts\": [\"gpt\", \"grok\"],\n"
    p += "    \"rejected_analysts\": [\"gemini\"],\n"
    p += "    \"audit_result\": \"原始CRS和盘口更支持主胜小胜，Gemini的平局保护证据不足\",\n"
    p += "    \"top3\": [\n"
    p += "      {\"score\":\"2-1\", \"prob\":17},\n"
    p += "      {\"score\":\"1-0\", \"prob\":14},\n"
    p += "      {\"score\":\"1-1\", \"prob\":13}\n"
    p += "    ],\n"
    p += "    \"accepted_observations\": [\"让球深度与CRS主胜小比分共振\"],\n"
    p += "    \"rejected_observations\": [\"散户热度不足以反推客队\"],\n"
    p += "    \"doubts\": [\"xG差距不大\", \"平局赔率仍有保护\"],\n"
    p += "    \"arbitration_reason\": \"完整中文终审理由，不限制字数，但必须保持 JSON 字符串合法\"\n"
    p += "  }\n"
    p += "]\n"
    p += "</output_format>\n"

    return p


# ====================================================================
# AI 调用配置
# ====================================================================

PHASE1_CONFIGS = [
    {
        "ai_name": "gpt",
        "url_env": "GPT_API_URL",
        "key_env": "GPT_API_KEY",
        "models": [
            "gpt-5.5"
        ],
        "role_key": "gpt",
    },
    {
        "ai_name": "grok",
        "url_env": "GROK_API_URL",
        "key_env": "GROK_API_KEY",
        "models": ["熊猫-A-5-grok-4.2-fast-200w上下文"],
        "role_key": "grok",
    },
    {
        "ai_name": "gemini",
        "url_env": "GEMINI_API_URL",
        "key_env": "GEMINI_API_KEY",
        "models": ["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking"],
        "role_key": "gemini",
    },
]

CLAUDE_CONFIG = {
    "ai_name": "claude",
    "url_env": "CLAUDE_API_URL",
    "key_env": "CLAUDE_API_KEY",
    "models": ["熊猫-69-满血openrouter-claude-opus-4.7-上下文1000k"],
    "temperature": 0.22,
}


def _build_urls_for_ai(ai_name: str, url_env: str) -> List[str]:
    if ai_name == "gpt":
        primary_url = get_first_clean_env_url(GPT_URL_ALIASES, GPT_DEFAULT_URL)

        urls = []
        if primary_url:
            urls.append(primary_url)

        if GPT_DEFAULT_URL and GPT_DEFAULT_URL not in urls:
            urls.append(GPT_DEFAULT_URL)

        return urls

    primary_url = get_clean_env_url(url_env)
    backup = [u for u in FALLBACK_URLS if u and u != primary_url][:1]
    return [primary_url] + backup


# ====================================================================
# AI 调用层：基于第一版，不限制输出字数
# ====================================================================

async def async_call_ai_batch(
    session,
    prompt: str,
    url_env: str,
    key_env: str,
    models_list: List[str],
    num_matches: int,
    ai_name: str,
    sys_prompt: str,
    temperature: float,
    phase: str
) -> Tuple[str, Dict[int, Dict], str]:
    if ai_name == "gpt":
        key = get_first_clean_env_key(GPT_KEY_ALIASES, GPT_DEFAULT_KEY)
    else:
        key = get_clean_env_key(key_env)

    if not key:
        status = f"no_key:{key_env}"
        print(f"  [跳过] {ai_name.upper()} 无可用 KEY: {key_env}")
        return ai_name, {}, status

    urls = _build_urls_for_ai(ai_name, url_env)

    CONNECT_TIMEOUT = 20
    READ_TIMEOUT_MAP = {
        "claude": 1800,
        "grok": 900,
        "gpt": 1200,
        "gemini": 1800,
    }
    READ_TIMEOUT = READ_TIMEOUT_MAP.get(ai_name, 900)

    for mn in models_list:
        connected = False

        for base_url in urls:
            if not base_url:
                continue

            is_gem = "generateContent" in base_url
            url = base_url.rstrip("/")

            if not is_gem and "chat/completions" not in url:
                url += "/chat/completions"

            headers = {"Content-Type": "application/json"}

            if is_gem:
                headers["x-goog-api-key"] = key
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": prompt}],
                        }
                    ],
                    "generationConfig": {
                        "temperature": temperature,
                    },
                    "systemInstruction": {
                        "parts": [{"text": sys_prompt}],
                    },
                }
            else:
                headers["Authorization"] = f"Bearer {key}"
                payload = {
                    "model": mn,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                }

            gw = url.split("/v1")[0][:48]
            print(f"  [🔌] {ai_name.upper()} | {mn[:32]} @ {gw}")

            t0 = time.time()

            try:
                timeout = aiohttp.ClientTimeout(
                    total=None,
                    connect=CONNECT_TIMEOUT,
                    sock_connect=CONNECT_TIMEOUT,
                    sock_read=READ_TIMEOUT,
                )

                async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                    elapsed = round(time.time() - t0, 1)

                    if r.status in (502, 504):
                        print(f"    💀 HTTP {r.status} | {elapsed}s → 换URL")
                        continue

                    if r.status == 400:
                        text = await r.text()
                        print(f"    💀 HTTP 400 | {elapsed}s | 模型={mn} | {text[:300]}")
                        break

                    if r.status == 401:
                        print(f"    💀 HTTP 401 | key无效 → 换通道")
                        break

                    if r.status == 429:
                        print(f"    ⚠️ HTTP 429 | 限流，稍后重试")
                        await asyncio.sleep(2.0)
                        continue

                    if r.status != 200:
                        text = await r.text()
                        print(f"    ⚠️ HTTP {r.status} | {elapsed}s | 模型={mn} | {text[:200]}")
                        continue

                    connected = True
                    print(f"    ✅ 已连上 {elapsed}s | 等待完整数据...")

                    try:
                        data = await r.json(content_type=None)
                    except Exception as e:
                        print(f"    ⚠️ JSON响应读取失败: {str(e)[:120]}")
                        break

                    elapsed = round(time.time() - t0, 1)
                    raw_text = _extract_response_text(data, is_gem)

                    if not raw_text or len(raw_text) < 10:
                        print("    ⚠️ 空响应 → 换模型")
                        break

                    results = _parse_ai_json(raw_text, num_matches, phase=phase)

                    if results:
                        print(f"    ✅ {ai_name.upper()} 完成: {len(results)}/{num_matches} | {elapsed}s")
                        return ai_name, results, mn

                    print("    ⚠️ JSON解析0条 → 换模型")
                    break

            except aiohttp.ClientConnectorError:
                continue
            except asyncio.TimeoutError:
                if not connected:
                    continue
                else:
                    print(f"    ⏱️ {ai_name.upper()} 读取超时: {READ_TIMEOUT}s")
                    return ai_name, {}, "read_timeout"
            except Exception as e:
                if not connected:
                    continue
                else:
                    print(f"    ⚠️ 调用异常: {str(e)[:120]}")
                    return ai_name, {}, "error"

            await asyncio.sleep(0.2)

    return ai_name, {}, "all_failed"


def _extract_response_text(data, is_gem=False) -> str:
    raw_text = ""

    try:
        if is_gem:
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return raw_text

        if data.get("choices"):
            msg = data["choices"][0].get("message", {})

            if isinstance(msg, dict):
                content_val = msg.get("content", "")

                if isinstance(content_val, str) and content_val.strip():
                    raw_text = content_val.strip()

                elif isinstance(content_val, list):
                    best = ""
                    for item in content_val:
                        if isinstance(item, dict):
                            t = item.get("text", item.get("content", ""))
                            if isinstance(t, str) and len(t) > len(best):
                                best = t.strip()
                    raw_text = best

                if not raw_text:
                    for field in [
                        "text",
                        "answer",
                        "response",
                        "output_text",
                        "final_answer",
                        "output",
                        "result",
                        "completion",
                    ]:
                        v = msg.get(field, "")
                        if isinstance(v, str) and v.strip():
                            raw_text = v.strip()
                            break

                if not raw_text:
                    skip = ("reasoning_content", "thinking", "reasoning", "thoughts")
                    best = ""
                    for k, v in msg.items():
                        if k in skip:
                            continue
                        if isinstance(v, str) and "[" in v and "match" in v:
                            if len(v) > len(best):
                                best = v.strip()
                    if best:
                        raw_text = best

        if not raw_text:
            full_str = json.dumps(data, ensure_ascii=False)
            m_match = re.search(r'\[\s*\{\s*\\?"match\\?"', full_str)
            if m_match:
                start_pos = m_match.start()
                depth = 0
                end_pos = start_pos

                for ci in range(start_pos, min(start_pos + 500000, len(full_str))):
                    if full_str[ci] == "[":
                        depth += 1
                    elif full_str[ci] == "]":
                        depth -= 1

                    if depth == 0:
                        end_pos = ci + 1
                        break

                if end_pos > start_pos:
                    extracted = full_str[start_pos:end_pos]
                    if '\\"' in extracted:
                        try:
                            extracted = json.loads('"' + extracted + '"')
                        except Exception:
                            extracted = extracted.replace('\\"', '"')
                    raw_text = extracted

    except Exception as ex:
        print(f"    ⚠️ 响应提取异常: {str(ex)[:100]}")

    return raw_text


# ====================================================================
# JSON 解析与校验
# ====================================================================

def _strip_thinking_blocks(text: str) -> str:
    clean = str(text or "")
    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"```(?:json|JSON|python|txt)?", "", clean)
    clean = clean.replace("```", "")
    return clean.strip()


def _extract_json_array(clean: str) -> str:
    m_re = re.search(r'\[\s*\{\s*"match"', clean)
    if not m_re:
        m_re = re.search(r"\[\s*\{\s*'match'", clean)

    if m_re:
        start_idx = m_re.start()
        depth = 0
        in_str = False
        escape = False
        end_idx = start_idx

        for i in range(start_idx, len(clean)):
            ch = clean[i]

            if escape:
                escape = False
                continue

            if ch == "\\":
                escape = True
                continue

            if ch == '"':
                in_str = not in_str
                continue

            if in_str:
                continue

            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break

        if end_idx > start_idx:
            return clean[start_idx:end_idx]

    start = clean.find("[")
    end = clean.rfind("]") + 1
    if start != -1 and end > start:
        return clean[start:end]

    return ""


def _parse_ai_json(raw_text: str, num_matches: int, phase: str) -> Dict[int, Dict]:
    clean = _strip_thinking_blocks(raw_text)
    json_str = _extract_json_array(clean)

    results = {}

    if not json_str:
        return results

    arr = []

    try:
        arr = json.loads(json_str)
    except json.JSONDecodeError:
        try:
            fixed = json_str.replace("'", '"')
            arr = json.loads(fixed)
        except Exception:
            try:
                last_brace = json_str.rfind("}")
                if last_brace != -1:
                    arr = json.loads(json_str[:last_brace + 1] + "]")
            except Exception:
                arr = []

    if isinstance(arr, dict):
        arr = [arr]

    if not isinstance(arr, list):
        return results

    for item in arr:
        if not isinstance(item, dict):
            continue

        valid, fixed, errors = validate_ai_item(item, phase=phase)

        if not fixed.get("match"):
            continue

        try:
            mid = int(fixed["match"])
        except Exception:
            continue

        fixed["ai_validation_errors"] = errors

        if 1 <= mid <= max(num_matches, 1):
            results[mid] = fixed

    return results


def validate_ai_item(item: Dict, phase: str) -> Tuple[bool, Dict, List[str]]:
    errors = []
    out = dict(item or {})

    try:
        out["match"] = int(out.get("match"))
    except Exception:
        errors.append("match_id_invalid")
        out["match"] = None
        return False, out, errors

    score = str(out.get("predicted_score", "")).strip()

    if not score:
        errors.append("score_missing")
        score = "1-1"

    score_label, is_others = _score_to_label(score)

    h, a = _parse_score(score_label)
    if h is None and score_label not in ["胜其他", "平其他", "负其他"]:
        errors.append("score_invalid")
        score_label = "1-1"

    expected_dir = _score_direction(score_label)

    if expected_dir:
        out["predicted_direction"] = expected_dir
    else:
        pred_dir = str(out.get("predicted_direction", "draw")).strip().lower()
        if pred_dir not in VALID_DIRS:
            errors.append("direction_invalid")
            pred_dir = "draw"
        out["predicted_direction"] = pred_dir

    out["predicted_score"] = score_label
    out["predicted_label"] = score_label
    out["is_score_others"] = is_others

    hp, dp, ap = _pct_normalize(
        out.get("home_win_pct", 33.3),
        out.get("draw_pct", 33.3),
        out.get("away_win_pct", 33.4),
    )

    out["home_win_pct"] = hp
    out["draw_pct"] = dp
    out["away_win_pct"] = ap

    conf = _i(out.get("confidence", 55), 55)
    conf = max(25, min(95, conf))
    out["confidence"] = conf

    raw_goal_range = out.get("goal_range") or _goal_range_from_score(score_label)
    bucket, label = _normalize_goal_range_for_ui(raw_goal_range, score_label)
    out["goal_range"] = bucket
    out["goal_range_label"] = label

    top3 = out.get("top3", [])
    if not isinstance(top3, list):
        top3 = []

    fixed_top3 = []
    top3_has_main = False

    for t in top3:
        if not isinstance(t, dict):
            continue

        sc = str(t.get("score", "")).strip()
        if not sc:
            continue

        prob = _f(t.get("prob", 0))
        if prob <= 1 and prob > 0:
            prob *= 100

        fixed_top3.append({
            "score": sc,
            "prob": round(prob, 1),
        })

        if sc == score_label:
            top3_has_main = True

    if not top3_has_main:
        fixed_top3.insert(0, {"score": score_label, "prob": 0})

    fixed_top3 = sorted(
        fixed_top3,
        key=lambda x: 0 if x.get("score") == score_label else 1
    )[:3]

    out["top3"] = fixed_top3

    for k in [
        "accepted_observations",
        "rejected_observations",
        "doubts",
        "key_signals",
        "adopted_analysts",
        "rejected_analysts",
    ]:
        if not isinstance(out.get(k), list):
            out[k] = []

    if not isinstance(out.get("data_quality"), dict):
        out["data_quality"] = {
            "odds_complete": None,
            "crs_complete": None,
            "ttg_complete": None,
            "notes": [],
        }

    if not isinstance(out.get("analysis_coverage"), dict):
        out["analysis_coverage"] = {}

    if phase == "claude":
        if not out.get("arbitration_reason"):
            out["arbitration_reason"] = out.get("reason", "")
        out["audit_result"] = _safe_str(out.get("audit_result", ""), 1200)
    else:
        out["reason"] = _safe_str(out.get("reason", ""), 5000)

    return len(errors) == 0, out, errors


# ====================================================================
# Phase 1 覆盖率 / 缺失补跑
# ====================================================================

def _is_valid_phase1_result(r: Dict) -> bool:
    if not isinstance(r, dict) or not r:
        return False

    score = str(r.get("predicted_score", "")).strip()
    direction = str(r.get("predicted_direction", "")).strip()

    if not score or score in ["-", "None", "null"]:
        return False

    if direction not in VALID_DIRS:
        return False

    return True


def _missing_phase1_ids(phase1_results: Dict[str, Dict[int, Dict]], ai_name: str, num_matches: int) -> List[int]:
    missing = []
    data = phase1_results.get(ai_name, {}) or {}

    for mid in range(1, num_matches + 1):
        if not _is_valid_phase1_result(data.get(mid, {})):
            missing.append(mid)

    return missing


def _phase1_coverage_for_match(phase1_results: Dict[str, Dict[int, Dict]], idx: int) -> Dict[str, Any]:
    gpt_ok = _is_valid_phase1_result(phase1_results.get("gpt", {}).get(idx, {}))
    grok_ok = _is_valid_phase1_result(phase1_results.get("grok", {}).get(idx, {}))
    gemini_ok = _is_valid_phase1_result(phase1_results.get("gemini", {}).get(idx, {}))

    valid_count = sum([gpt_ok, grok_ok, gemini_ok])

    return {
        "gpt": gpt_ok,
        "grok": grok_ok,
        "gemini": gemini_ok,
        "valid_count": valid_count,
        "coverage_ok": valid_count >= 2,
        "coverage_full": valid_count == 3,
    }


async def repair_missing_phase1_results(
    session,
    phase1_results: Dict[str, Dict[int, Dict]],
    match_blocks: List[str],
    num_matches: int,
    sys_prompts: Dict[str, str]
) -> Dict[str, Dict[int, Dict]]:
    print("\n  [Phase 1 Repair] 检查三家缺失场次...")

    for cfg in PHASE1_CONFIGS:
        ai_name = cfg["ai_name"]
        role_key = cfg["role_key"]

        missing_ids = _missing_phase1_ids(phase1_results, ai_name, num_matches)

        if not missing_ids:
            print(f"    ✅ {ai_name.upper()} 无缺失")
            continue

        if str(AI_CALL_STATUS.get(ai_name, "")).startswith("no_key"):
            print(f"    ⛔ {ai_name.upper()} 缺失 {missing_ids}，但状态={AI_CALL_STATUS.get(ai_name)}，跳过补跑")
            continue

        print(f"    🔁 {ai_name.upper()} 缺失场次: {missing_ids}，开始单场补跑")

        for mid in missing_ids:
            block = match_blocks[mid - 1]
            prompt = build_phase1_retry_prompt(block, role_key, mid)
            temp = PHASE1_ROLES[role_key]["temperature"]

            _, results, status = await async_call_ai_batch(
                session=session,
                prompt=prompt,
                url_env=cfg["url_env"],
                key_env=cfg["key_env"],
                models_list=cfg["models"],
                num_matches=num_matches,
                ai_name=ai_name,
                sys_prompt=sys_prompts[ai_name],
                temperature=temp,
                phase="phase1",
            )

            if mid not in results and 1 in results and mid != 1:
                fixed = dict(results[1])
                fixed["match"] = mid
                results[mid] = fixed

            if mid in results and _is_valid_phase1_result(results[mid]):
                phase1_results.setdefault(ai_name, {})[mid] = results[mid]
                print(f"      ✅ {ai_name.upper()} 第{mid}场补回: {results[mid].get('predicted_score')}")
            else:
                print(f"      ⚠️ {ai_name.upper()} 第{mid}场仍缺失 | 状态={status}")

    return phase1_results


# ====================================================================
# Phase 1 / Phase 2 执行
# ====================================================================

async def run_phase1_three(match_blocks: List[str], num_matches: int) -> Dict[str, Dict[int, Dict]]:
    print(f"\n  [Phase 1] GPT/Grok/Gemini 三家并行分析 ({num_matches} 场)...")

    sys_prompts = {
        "gpt": "<role>你是赔率结构分析师，专注欧赔、让球、总进球、CRS、半全场一致性。</role><instruction>严格输出 JSON 数组，不要 markdown。</instruction>",
        "grok": "<role>你是资金流分析师，专注 Sharp、Steam、变盘、散户冷热、资金与盘口背离。</role><instruction>严格输出 JSON 数组，不要 markdown。</instruction>",
        "gemini": "<role>你是基本面分析师，专注战绩、伤停、赛程、联赛属性、杯赛属性、主客场强弱。</role><instruction>严格输出 JSON 数组，不要 markdown。</instruction>",
    }

    connector = aiohttp.TCPConnector(limit=10, use_dns_cache=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []

        for cfg in PHASE1_CONFIGS:
            role_key = cfg["role_key"]
            prompt = build_phase1_prompt(match_blocks, role_key)
            temp = PHASE1_ROLES[role_key]["temperature"]

            tasks.append(async_call_ai_batch(
                session=session,
                prompt=prompt,
                url_env=cfg["url_env"],
                key_env=cfg["key_env"],
                models_list=cfg["models"],
                num_matches=num_matches,
                ai_name=cfg["ai_name"],
                sys_prompt=sys_prompts[cfg["ai_name"]],
                temperature=temp,
                phase="phase1",
            ))

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {
            "gpt": {},
            "grok": {},
            "gemini": {},
        }

        for res in raw_results:
            if isinstance(res, tuple):
                ai_name, results, status = res
                output[ai_name] = results
                AI_CALL_STATUS[ai_name] = status
                print(f"  [状态] {ai_name.upper()} => {status} | 返回 {len(results)}/{num_matches} 场")
            else:
                print(f"  [Phase1异常] {res}")

        output = await repair_missing_phase1_results(
            session=session,
            phase1_results=output,
            match_blocks=match_blocks,
            num_matches=num_matches,
            sys_prompts=sys_prompts,
        )

    ok = sum(1 for v in output.values() if v)
    print(f"  [Phase 1 完成] {ok}/3 家有数据")

    for ai_name in ["gpt", "grok", "gemini"]:
        missing = _missing_phase1_ids(output, ai_name, num_matches)
        if missing:
            print(f"  [Phase 1 覆盖警告] {ai_name.upper()} 仍缺失: {missing}")

    return output


async def run_phase2_claude_audit(
    match_blocks: List[str],
    phase1_results: Dict[str, Dict[int, Dict]],
    num_matches: int
) -> Tuple[Dict[int, Dict], str]:
    print(f"\n  [Phase 2] Claude 终审复盘 ({num_matches} 场)...")

    prompt = build_phase2_prompt(match_blocks, phase1_results)
    print(f"  [Claude Prompt] {len(prompt):,} 字符")

    sys_prompt = (
        "<role>你是 Claude 终审复盘者。你必须重新审计原始抓包，不得简单按三家多数派裁决。</role>\n"
        "<instruction>严格输出 JSON 数组，禁止 markdown，禁止前缀后缀。</instruction>"
    )

    connector = aiohttp.TCPConnector(limit=5, use_dns_cache=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        ai_name, results, model_name = await async_call_ai_batch(
            session=session,
            prompt=prompt,
            url_env=CLAUDE_CONFIG["url_env"],
            key_env=CLAUDE_CONFIG["key_env"],
            models_list=CLAUDE_CONFIG["models"],
            num_matches=num_matches,
            ai_name="claude",
            sys_prompt=sys_prompt,
            temperature=CLAUDE_CONFIG["temperature"],
            phase="claude",
        )

    AI_CALL_STATUS["claude"] = model_name

    print(f"  [Phase 2 完成] Claude 返回 {len(results)}/{num_matches} | 状态={model_name}")

    return results, f"claude:{model_name}"


# ====================================================================
# 输出包装 / EV / 兼容字段
# ====================================================================

def _enforce_direction_consistency(result: Dict) -> Dict:
    out = dict(result or {})

    score = out.get("predicted_score", "1-1")
    score_label, is_others = _score_to_label(score)

    expected_dir = _score_direction(score_label)
    if not expected_dir:
        expected_dir = out.get("predicted_direction", "draw")
        if expected_dir not in VALID_DIRS:
            expected_dir = "draw"

    result_cn = _direction_cn(expected_dir)

    raw_goal_range = out.get("goal_range") or _goal_range_from_score(score_label)
    goal_bucket, goal_label = _normalize_goal_range_for_ui(raw_goal_range, score_label)

    conf = _i(out.get("confidence", 55), 55)
    conf = max(25, min(95, conf))

    hp, dp, ap = _pct_normalize(
        out.get("home_win_pct", 33.3),
        out.get("draw_pct", 33.3),
        out.get("away_win_pct", 33.4),
    )

    out["predicted_score"] = score_label
    out["predicted_label"] = score_label
    out["predicted_direction"] = expected_dir
    out["final_direction"] = expected_dir
    out["result"] = result_cn
    out["display_direction"] = result_cn
    out["is_score_others"] = is_others

    out["goal_range"] = goal_bucket
    out["goal_interval"] = goal_bucket
    out["predicted_goal_range"] = goal_bucket
    out["goal_range_label"] = goal_label

    out["confidence"] = conf
    out["ai_confidence"] = conf
    out["ai_confidence_pct"] = conf
    out["ai_confidence_score"] = conf
    out["confidence_score"] = conf
    out["confidence_pct"] = conf
    out["analysis_confidence"] = conf
    out["dir_confidence"] = conf
    out["ai_confidence_value"] = conf
    out["aiConfidence"] = conf
    out["confidenceValue"] = conf
    out["final_confidence"] = conf
    out["prediction_confidence"] = conf
    out["finalConfidence"] = conf
    out["ai_conf"] = conf
    out["cf"] = conf
    out["ai_trust"] = conf
    out["ai_trust_pct"] = conf
    out["trust_score"] = conf
    out["model_confidence"] = conf
    out["model_confidence_pct"] = conf

    out["home_win_pct"] = hp
    out["draw_pct"] = dp
    out["away_win_pct"] = ap

    return out


def _extract_score_prob_from_ai(ai_result: Dict, score: str) -> float:
    candidates = []

    for key in ["top3", "score_probs", "scores"]:
        arr = ai_result.get(key, [])
        if isinstance(arr, list):
            candidates.extend(arr)

    alt = ai_result.get("alternative")
    if isinstance(alt, dict):
        candidates.append(alt)

    for item in candidates:
        if not isinstance(item, dict):
            continue

        if str(item.get("score", "")).strip() == str(score).strip():
            p = _f(item.get("prob", 0))
            if p > 1:
                p /= 100.0
            return max(0.0, min(1.0, p))

    return 0.0


def _calculate_score_ev(score_prob: float, odds: float) -> Tuple[float, float, bool]:
    if odds <= 1.05 or score_prob <= 0:
        return 0.0, 0.0, False

    ev = score_prob * odds - 1.0
    b = odds - 1.0
    q = 1.0 - score_prob

    if b > 0:
        kelly = ((b * score_prob) - q) / b
        kelly_pct = round(max(0.0, kelly * 0.5) * 100, 2)
    else:
        kelly_pct = 0.0

    ev_pct = round(ev * 100, 2)
    is_value = ev_pct > 5

    return ev_pct, kelly_pct, is_value


def _get_score_odds(match: Dict, score: str, direction: str, is_others: bool) -> float:
    if is_others:
        if direction == "home":
            return _f(match.get("crs_win", 0))
        if direction == "away":
            return _f(match.get("crs_lose", 0))
        return _f(match.get("crs_same", 0))

    key = CRS_FULL_MAP.get(score, "")
    if not key:
        return 0.0

    return _f(match.get(key, 0))


def _ai_score_summary(r: Dict) -> str:
    if not isinstance(r, dict) or not r:
        return "-"
    return str(r.get("predicted_score", "-"))


def _ai_reason_summary(r: Dict, empty_text: str) -> str:
    if not isinstance(r, dict) or not r:
        return empty_text
    return _safe_str(r.get("reason", ""), 2000)


def _apply_legacy_enhancer_readonly(enhancer, match: Dict, mg: Dict, *args):
    before = {k: mg.get(k) for k in LOCKED_CORE_FIELDS}

    try:
        mg2 = enhancer(match, mg, *args) if args else enhancer(match, mg)
    except Exception as e:
        logger.warning(f"{getattr(enhancer, '__name__', 'enhancer')} 失败: {e}")
        return mg

    if not isinstance(mg2, dict):
        return mg

    for k, v in before.items():
        mg2[k] = v

    return mg2


def assemble_final_prediction(
    match: Dict,
    engine_result: Dict,
    stats: Dict,
    phase1_results: Dict[str, Dict[int, Dict]],
    claude_result: Dict,
    observation_signals: List[str],
    ensemble_signals: Dict,
    idx: int,
    ai_provider: str
) -> Dict:
    cr = _enforce_direction_consistency(claude_result or {})

    predicted_score = cr.get("predicted_score", "1-1")
    predicted_label = cr.get("predicted_label", predicted_score)
    final_direction = cr.get("final_direction", "draw")
    result_cn = cr.get("result", "平局")
    is_others = bool(cr.get("is_score_others", False))

    h_score, a_score = _parse_score(predicted_score)
    goal_count = h_score + a_score if h_score is not None else None

    home_pct = _f(cr.get("home_win_pct", 33.3), 33.3)
    draw_pct = _f(cr.get("draw_pct", 33.3), 33.3)
    away_pct = _f(cr.get("away_win_pct", 33.4), 33.4)

    confidence = _i(cr.get("confidence", 55), 55)
    confidence = max(25, min(95, confidence))

    coverage = _phase1_coverage_for_match(phase1_results, idx)

    if coverage["valid_count"] <= 1:
        confidence = min(confidence, 62)
    elif coverage["valid_count"] == 2:
        confidence = min(confidence, 72)

    risk = "低" if confidence >= 75 else ("中" if confidence >= 55 else "高")

    final_odds = _get_score_odds(match, predicted_score, final_direction, is_others)
    raw_score_prob = _extract_score_prob_from_ai(cr, predicted_score)

    if ENABLE_LLM_VALUE_BET:
        score_prob = raw_score_prob
        ev_pct, kelly_pct, is_value = _calculate_score_ev(score_prob, final_odds)
        value_reason = "基于 Claude top3 精确比分概率计算，未做历史校准"
    else:
        score_prob = raw_score_prob
        ev_pct, kelly_pct, is_value = 0.0, 0.0, False
        value_reason = "v19.2 暂不使用 LLM 主观比分概率计算正式EV/Kelly"

    raw_goal_range = cr.get("goal_range") or _goal_range_from_score(predicted_score)
    goal_bucket, goal_label = _normalize_goal_range_for_ui(raw_goal_range, predicted_score)

    obs_codes = _extract_observation_codes(observation_signals)
    obs_labels = _extract_observation_labels(observation_signals)
    obs_objects = _extract_observation_objects(observation_signals)

    hp_list = [home_pct, draw_pct, away_pct]
    hp_sorted = sorted(hp_list)

    expected_total_goals = _f(engine_result.get("expected_total_goals", 0))
    if expected_total_goals <= 0:
        expected_total_goals = (
            _f(engine_result.get("bookmaker_implied_home_xg", 1.3)) +
            _f(engine_result.get("bookmaker_implied_away_xg", 0.9))
        )

    smart_signals_raw = stats.get("smart_signals", []) if stats else []

    p1_gpt = phase1_results.get("gpt", {}).get(idx, {})
    p1_grok = phase1_results.get("grok", {}).get(idx, {})
    p1_gemini = phase1_results.get("gemini", {}).get(idx, {})

    claude_reason = (
        cr.get("arbitration_reason")
        or cr.get("reason")
        or cr.get("audit_result")
        or ""
    )

    accepted_obs = cr.get("accepted_observations", [])
    rejected_obs = cr.get("rejected_observations", [])
    doubts = cr.get("doubts", [])

    return {
        "predicted_score": predicted_score,
        "predicted_label": predicted_label,
        "predicted_direction": final_direction,
        "result": result_cn,
        "display_direction": result_cn,
        "final_direction": final_direction,
        "is_score_others": is_others,

        "decision_title": "vMAX 19.2 决策剖析",
        "decision_engine_version": ENGINE_VERSION,
        "decision_architecture": ENGINE_ARCHITECTURE,
        "engine_version": ENGINE_VERSION,
        "engine_architecture": ENGINE_ARCHITECTURE,

        "goal_range": goal_bucket,
        "goal_interval": goal_bucket,
        "predicted_goal_range": goal_bucket,
        "goal_range_label": goal_label,
        "goal_bucket": goal_bucket,
        "goals_range": goal_bucket,
        "goal_range_text": goal_label,
        "total_goals_range": goal_bucket,
        "predicted_goals_range": goal_bucket,
        "predicted_goals_label": goal_label,
        "expected_goal_range": goal_bucket,
        "expected_goals_range": goal_bucket,
        "goal_zone": goal_bucket,
        "goals_zone": goal_bucket,
        "score_goal_range": goal_bucket,
        "goal_count": goal_count,
        "total_goal_count": goal_count,
        "goals_count": goal_count,
        "goal_count_label": goal_label,

        "home_win_pct": home_pct,
        "draw_pct": draw_pct,
        "away_win_pct": away_pct,

        "confidence": confidence,
        "ai_confidence": confidence,
        "ai_confidence_pct": confidence,
        "ai_confidence_score": confidence,
        "ai_confidence_value": confidence,
        "aiConfidence": confidence,
        "confidence_score": confidence,
        "confidenceValue": confidence,
        "confidence_pct": confidence,
        "analysis_confidence": confidence,
        "final_confidence": confidence,
        "prediction_confidence": confidence,
        "finalConfidence": confidence,
        "ai_conf": confidence,
        "cf": confidence,
        "ai_trust": confidence,
        "ai_trust_pct": confidence,
        "trust_score": confidence,
        "model_confidence": confidence,
        "model_confidence_pct": confidence,
        "dir_confidence": confidence,
        "risk_level": risk,
        "dir_gap": round(hp_sorted[-1] - hp_sorted[-2], 1) if len(hp_sorted) >= 2 else 0,

        "ai_call_status": dict(AI_CALL_STATUS),
        "gpt_status": AI_CALL_STATUS.get("gpt", ""),
        "grok_status": AI_CALL_STATUS.get("grok", ""),
        "gemini_status": AI_CALL_STATUS.get("gemini", ""),
        "claude_status": AI_CALL_STATUS.get("claude", ""),

        "ai_provider": ai_provider,
        "claude_score": predicted_score,
        "claude_analysis": claude_reason[:3000],
        "arbitration_reason": claude_reason,
        "audit_result": cr.get("audit_result", ""),
        "agreement_pattern": cr.get("agreement_pattern", "Claude终审复盘"),

        "phase1_coverage": coverage,
        "analysis_coverage": cr.get("analysis_coverage", coverage),
        "coverage_ok": coverage["coverage_ok"],
        "coverage_full": coverage["coverage_full"],

        "adopted_analysts": cr.get("adopted_analysts", []),
        "rejected_analysts": cr.get("rejected_analysts", []),
        "alternative_score": cr.get("alternative", {}),
        "top3": cr.get("top3", []),

        "gpt_score": _ai_score_summary(p1_gpt),
        "gpt_analysis": _ai_reason_summary(p1_gpt, "GPT 未返回有效分析。"),
        "gpt_doubts": p1_gpt.get("doubts", []) if p1_gpt else [],
        "gpt_key_signals": p1_gpt.get("key_signals", []) if p1_gpt else [],

        "grok_score": _ai_score_summary(p1_grok),
        "grok_analysis": _ai_reason_summary(p1_grok, "GROK 未返回有效分析。"),
        "grok_doubts": p1_grok.get("doubts", []) if p1_grok else [],
        "grok_key_signals": p1_grok.get("key_signals", []) if p1_grok else [],

        "gemini_score": _ai_score_summary(p1_gemini),
        "gemini_analysis": _ai_reason_summary(p1_gemini, "GEMINI 未返回有效分析。"),
        "gemini_doubts": p1_gemini.get("doubts", []) if p1_gemini else [],
        "gemini_key_signals": p1_gemini.get("key_signals", []) if p1_gemini else [],

        "ai_abstained": [
            n.upper()
            for n in ["gpt", "grok", "gemini"]
            if not phase1_results.get(n, {}).get(idx)
        ],

        "accepted_observations": accepted_obs,
        "rejected_observations": rejected_obs,
        "doubts": doubts,
        "data_quality": cr.get("data_quality", {}),
        "ai_validation_errors": cr.get("ai_validation_errors", []),

        "traps_detected": obs_labels,
        "trap_codes": obs_codes,
        "trap_items": obs_objects,
        "observation_items": obs_objects,
        "trap_count": len(observation_signals),
        "trap_facts": observation_signals,
        "observation_signals": observation_signals,
        "trap_matrix_title": "观察信号矩阵",
        "trap_matrix_subtitle": "仅为抓包观察，不代表固定诱盘结论",

        "score_odds": final_odds,
        "raw_llm_score_prob": round(raw_score_prob * 100, 2),
        "score_prob": round(score_prob * 100, 2),
        "suggested_kelly": kelly_pct,
        "edge_vs_market": ev_pct,
        "is_value": is_value,
        "value_reason": value_reason,

        "raw_smart_signals": smart_signals_raw,
        "smart_signals": observation_signals,
        "smart_money_signal": " | ".join([str(s) for s in observation_signals[:10]]),
        "sharp_detected": _detect_sharp_direction(smart_signals_raw) is not None,
        "sharp_dir": _detect_sharp_direction(smart_signals_raw),

        "xG_home": round(_f(engine_result.get("bookmaker_implied_home_xg", 1.3)), 2),
        "xG_away": round(_f(engine_result.get("bookmaker_implied_away_xg", 0.9)), 2),
        "expected_total_goals": round(expected_total_goals, 2),
        "over_2_5": engine_result.get("over_25", 50) if engine_result else 50,
        "btts": engine_result.get("btts", 45) if engine_result else 45,
        "bookmaker_implied_home_xg": engine_result.get("bookmaker_implied_home_xg", "?"),
        "bookmaker_implied_away_xg": engine_result.get("bookmaker_implied_away_xg", "?"),

        "model_consensus_dir": ensemble_signals.get("consensus", ""),
        "model_consensus_count": ensemble_signals.get("consensus_count", 0),
        "total_models": ensemble_signals.get("total", 0),
        "ensemble_reference": ensemble_signals,

        "refined_poisson": stats.get("refined_poisson", {}) if stats else {},
        "elo": stats.get("elo", {}) if stats else {},
        "random_forest": stats.get("random_forest", {}) if stats else {},
        "gradient_boost": stats.get("gradient_boost", {}) if stats else {},
        "neural_net": stats.get("neural_net", {}) if stats else {},
        "logistic": stats.get("logistic", {}) if stats else {},
        "svm": stats.get("svm", {}) if stats else {},
        "knn": stats.get("knn", {}) if stats else {},
        "dixon_coles": stats.get("dixon_coles", {}) if stats else {},
        "bradley_terry": stats.get("bradley_terry", {}) if stats else {},
        "experience_analysis": stats.get("experience_analysis", {}) if stats else {},
        "top_scores": stats.get("refined_poisson", {}).get("top_scores", []) if stats else [],
    }


# ====================================================================
# Top4 推荐
# ====================================================================

def select_top4(preds):
    def _score(x):
        p = x.get("prediction", {}) or {}

        confidence = _f(p.get("confidence", 0))
        score_prob = _f(p.get("score_prob", 0))
        risk_penalty = 0

        if p.get("risk_level") == "高":
            risk_penalty += 8
        if p.get("ai_validation_errors"):
            risk_penalty += 5
        if p.get("ai_abstained"):
            risk_penalty += len(p.get("ai_abstained", [])) * 2
        if not p.get("coverage_ok", True):
            risk_penalty += 8

        return confidence + score_prob * 0.2 - risk_penalty

    preds_sorted = sorted(preds, key=_score, reverse=True)
    return preds_sorted[:4]


def extract_num(ms):
    wm = {
        "一": 1000,
        "二": 2000,
        "三": 3000,
        "四": 4000,
        "五": 5000,
        "六": 6000,
        "日": 7000,
        "天": 7000,
    }

    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))

    return base + int(nums[0]) if nums else 9999


def _fallback_claude_result_from_phase1(phase1_results: Dict[str, Dict[int, Dict]], idx: int) -> Dict:
    for name in ["gpt", "grok", "gemini"]:
        r = phase1_results.get(name, {}).get(idx, {})
        if r and r.get("predicted_score"):
            out = dict(r)
            out["agreement_pattern"] = f"Claude失败，采用{name.upper()}兜底"
            out["adopted_analysts"] = [name]
            out["rejected_analysts"] = []
            out["audit_result"] = f"Claude 未返回有效终审，临时采用 {name.upper()} 输出。"
            out["arbitration_reason"] = f"Claude 未返回有效终审，临时采用 {name.upper()} 输出：{r.get('reason', '')}"
            out["confidence"] = max(25, int(_f(r.get("confidence", 45)) * 0.85))
            return out

    return {
        "match": idx,
        "predicted_score": "1-1",
        "predicted_direction": "draw",
        "goal_range": "2球",
        "home_win_pct": 33.3,
        "draw_pct": 34.0,
        "away_win_pct": 32.7,
        "confidence": 30,
        "top3": [
            {"score": "1-1", "prob": 0},
            {"score": "1-0", "prob": 0},
            {"score": "0-1", "prob": 0},
        ],
        "agreement_pattern": "全部AI失败",
        "adopted_analysts": [],
        "rejected_analysts": [],
        "analysis_coverage": {"gpt": False, "grok": False, "gemini": False, "valid_count": 0},
        "audit_result": "全部AI失败，兜底输出1-1。",
        "arbitration_reason": "全部AI失败，兜底输出1-1。此结果不可作为强判断。",
        "accepted_observations": [],
        "rejected_observations": [],
        "doubts": ["AI未返回有效结果，使用保守兜底"],
    }


# ====================================================================
# 主入口
# ====================================================================

def run_predictions(raw, use_ai=True):
    raw_matches = raw.get("matches", [])
    num = len(raw_matches)

    for k in AI_CALL_STATUS:
        AI_CALL_STATUS[k] = ""

    print("\n" + "=" * 80)
    print(f"  [{ENGINE_VERSION}] {ENGINE_ARCHITECTURE} | {num} 场")
    print("=" * 80)

    debug_ai_config()

    match_analyses = []
    match_blocks = []

    for i, raw_m in enumerate(raw_matches):
        m = normalize_match(raw_m)

        try:
            if predict_match:
                eng = predict_match(m)
            else:
                eng = {}
        except Exception as e:
            logger.warning(f"predict_match 失败: {e}")
            eng = {}

        try:
            sp = ensemble.predict(m, {}) if ensemble else {}
        except Exception as e:
            logger.warning(f"ensemble.predict 失败: {e}")
            sp = {}

        try:
            exp_result = exp_engine.analyze(m) if exp_engine else {}
        except Exception as e:
            logger.warning(f"exp_engine.analyze 失败: {e}")
            exp_result = {}

        exp_goals = _f(eng.get("expected_total_goals", 0))
        if exp_goals <= 0:
            hxg = _f(eng.get("bookmaker_implied_home_xg", 0))
            axg = _f(eng.get("bookmaker_implied_away_xg", 0))
            exp_goals = hxg + axg if (hxg and axg) else 2.5

        smart_signals = sp.get("smart_signals", []) if isinstance(sp, dict) else []

        observation_signals = build_observation_signals(
            match_obj=m,
            engine_result=eng,
            smart_signals=smart_signals,
            exp_goals=exp_goals,
        )

        ensemble_signals = collect_ensemble_signals(sp)

        block = format_match_block(
            idx=i + 1,
            match=m,
            engine_result=eng,
            observation_signals=observation_signals,
            ensemble_signals=ensemble_signals,
            smart_signals=smart_signals,
        )

        match_blocks.append(block)

        match_analyses.append({
            "raw_match": raw_m,
            "match": m,
            "engine": eng,
            "stats": sp,
            "experience": exp_result,
            "observation_signals": observation_signals,
            "ensemble_signals": ensemble_signals,
        })

    phase1_results = {
        "gpt": {},
        "grok": {},
        "gemini": {},
    }
    claude_results = {}
    ai_provider = "no_ai"

    if use_ai and match_blocks:
        async def _run_full_ai():
            p1 = await run_phase1_three(match_blocks, num)
            p2, provider = await run_phase2_claude_audit(match_blocks, p1, num)
            return p1, p2, provider

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            def _run_in_thread(coro):
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run_in_thread, _run_full_ai())
                try:
                    phase1_results, claude_results, ai_provider = future.result()
                except Exception as e:
                    logger.error(f"AI 四家矩阵执行崩溃: {e}")
                    phase1_results = {"gpt": {}, "grok": {}, "gemini": {}}
                    claude_results = {}
                    ai_provider = "ai_crashed"
        else:
            try:
                phase1_results, claude_results, ai_provider = asyncio.run(_run_full_ai())
            except Exception as e:
                logger.error(f"AI 四家矩阵执行崩溃: {e}")
                phase1_results = {"gpt": {}, "grok": {}, "gemini": {}}
                claude_results = {}
                ai_provider = "ai_crashed"

    res = []

    for i, ma in enumerate(match_analyses):
        idx = i + 1
        raw_m = ma["raw_match"]
        m = ma["match"]

        cr = claude_results.get(idx, {})
        if not cr:
            cr = _fallback_claude_result_from_phase1(phase1_results, idx)

        mg = assemble_final_prediction(
            match=m,
            engine_result=ma["engine"],
            stats=ma["stats"],
            phase1_results=phase1_results,
            claude_result=cr,
            observation_signals=ma["observation_signals"],
            ensemble_signals=ma["ensemble_signals"],
            idx=idx,
            ai_provider=ai_provider,
        )

        if APPLY_LEGACY_ENHANCERS:
            if exp_engine and apply_experience_to_prediction:
                mg = _apply_legacy_enhancer_readonly(
                    apply_experience_to_prediction,
                    m,
                    mg,
                    exp_engine,
                )

            mg = _apply_legacy_enhancer_readonly(apply_odds_history, m, mg)
            mg = _apply_legacy_enhancer_readonly(apply_quant_edge, m, mg)
            mg = _apply_legacy_enhancer_readonly(apply_wencai_intel, m, mg)

            if upgrade_ensemble_predict:
                mg = _apply_legacy_enhancer_readonly(upgrade_ensemble_predict, m, mg)

        mg = _enforce_direction_consistency(mg)

        combined = {**raw_m, **m, "prediction": mg}

        for k in [
            "predicted_score",
            "predicted_label",
            "predicted_direction",
            "result",
            "display_direction",
            "final_direction",

            "confidence",
            "ai_confidence",
            "ai_confidence_pct",
            "ai_confidence_score",
            "ai_confidence_value",
            "aiConfidence",
            "confidence_score",
            "confidenceValue",
            "confidence_pct",
            "analysis_confidence",
            "final_confidence",
            "prediction_confidence",
            "finalConfidence",
            "ai_conf",
            "cf",
            "ai_trust",
            "ai_trust_pct",
            "trust_score",
            "model_confidence",
            "model_confidence_pct",

            "goal_range",
            "goal_interval",
            "predicted_goal_range",
            "goal_range_label",
            "goal_bucket",
            "goals_range",
            "goal_range_text",
            "total_goals_range",
            "predicted_goals_range",
            "predicted_goals_label",
            "expected_goal_range",
            "expected_goals_range",
            "goal_zone",
            "goals_zone",
            "score_goal_range",
            "goal_count",
            "total_goal_count",
            "goals_count",
            "goal_count_label",

            "decision_title",
            "decision_engine_version",
            "decision_architecture",
            "engine_version",
            "engine_architecture",

            "ai_call_status",
            "gpt_status",
            "grok_status",
            "gemini_status",
            "claude_status",

            "phase1_coverage",
            "analysis_coverage",
            "coverage_ok",
            "coverage_full",

            "claude_score",
            "claude_analysis",
            "gpt_score",
            "gpt_analysis",
            "grok_score",
            "grok_analysis",
            "gemini_score",
            "gemini_analysis",
        ]:
            combined[k] = mg.get(k)

        combined["engine_version"] = ENGINE_VERSION
        combined["decision_title"] = "vMAX 19.2 决策剖析"
        combined["decision_engine_version"] = ENGINE_VERSION
        combined["decision_architecture"] = ENGINE_ARCHITECTURE

        for ck in [
            "confidence",
            "ai_confidence",
            "ai_confidence_pct",
            "ai_confidence_score",
            "ai_confidence_value",
            "aiConfidence",
            "confidence_score",
            "confidenceValue",
            "confidence_pct",
            "analysis_confidence",
            "final_confidence",
            "prediction_confidence",
            "finalConfidence",
            "ai_conf",
            "cf",
            "ai_trust",
            "ai_trust_pct",
            "trust_score",
            "model_confidence",
            "model_confidence_pct",
        ]:
            combined[ck] = mg.get("confidence", 0)

        for gk in [
            "goal_range",
            "goal_interval",
            "predicted_goal_range",
            "goal_bucket",
            "goals_range",
            "total_goals_range",
            "predicted_goals_range",
            "expected_goal_range",
            "expected_goals_range",
            "goal_zone",
            "goals_zone",
            "score_goal_range",
        ]:
            combined[gk] = mg.get("goal_range")

        for gk in [
            "goal_range_label",
            "goal_range_text",
            "predicted_goals_label",
            "goal_count_label",
        ]:
            combined[gk] = mg.get("goal_range_label")

        for gk in [
            "goal_count",
            "total_goal_count",
            "goals_count",
        ]:
            combined[gk] = mg.get("goal_count")

        res.append(combined)

        obs_tag = f" [OBS{len(ma['observation_signals'])}]" if ma.get("observation_signals") else ""
        err_tag = f" [校验{len(mg.get('ai_validation_errors', []))}]" if mg.get("ai_validation_errors") else ""
        abstain_tag = f" [缺席{','.join(mg.get('ai_abstained', []))}]" if mg.get("ai_abstained") else ""
        cov = mg.get("phase1_coverage", {})
        cov_tag = f" [覆盖{cov.get('valid_count', 0)}/3]"

        print(
            f"  [{idx}] {m.get('home_team', m.get('home', '?'))} vs "
            f"{m.get('away_team', m.get('guest', '?'))} => "
            f"{mg['result']} ({mg['predicted_score']}) | "
            f"CF: {mg['confidence']}% | {mg.get('agreement_pattern', 'Claude终审')}"
            f"{cov_tag}{obs_tag}{err_tag}{abstain_tag}"
        )

    t4 = select_top4(res)
    t4ids = [t.get("id") for t in t4]

    for r in res:
        r["is_recommended"] = r.get("id") in t4ids

    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    return res, t4


# ====================================================================
# 本地启动
# ====================================================================

if __name__ == "__main__":
    logger.info(f"{ENGINE_VERSION} 启动")
    print(f"✅ {ENGINE_VERSION} 加载完成")
    print(f"   架构: {ENGINE_ARCHITECTURE}")
    print("   Phase 1: GPT / Grok / Gemini 三家独立分析")
    print("   Phase 1 Repair: 缺失场次自动单场补跑")
    print("   Phase 2: Claude 接收三家结论 + 原始抓包做最终审计")
    print("   规则: Claude 必须重新审计抓包，不按票数机械裁决")
    print("   调用层: 回到第一版底座，不限制输出字数，长等待")
    print("   EV/Kelly: 默认不使用 LLM 主观比分概率")