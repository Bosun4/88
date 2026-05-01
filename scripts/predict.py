# ====================================================================
# 🚀 vMAX 19.3 — 体彩Sharp分级 + 外部市场比对 + 量化锁定链
# --------------------------------------------------------------------
# 核心原则:
#   ✅ 保留 v19.2 的 FirstBase Raw Packet 四AI审计结构
#   ✅ 恢复 v18 的 CRS矩阵几何 + 贝叶斯方向后验 + 决策锁定链
#   ✅ Sharp 是体彩核心变量，但必须先分级: S0/S1/S2/S3/S4/SD
#   ✅ Claude 不再直接决定最终比分，只做终审审计意见
#   ✅ 最终比分由程序锁定链决定: 方向 → 进球区间 → 比分
#   ✅ 新增外部欧指/市场交叉验证入口 external_market / external_urls
#   ✅ 体彩官方方向与外部市场冲突时标记 LOCAL_MARKET_TRAP
#   ✅ EV/Kelly 默认不使用 LLM 主观比分概率
# ====================================================================

import json
import os
import re
import time
import math
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
build_league_intelligence = None
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

ENGINE_VERSION = "vMAX 19.3"
ENGINE_ARCHITECTURE = "FirstBase Raw Packet + External Market Audit + Sharp Grading + Quant Lock Chain"

APPLY_LEGACY_ENHANCERS = False
ENABLE_LLM_VALUE_BET = False
ENABLE_EXTERNAL_FETCH = True

AI_CALL_STATUS = {
    "gpt": "",
    "grok": "",
    "gemini": "",
    "claude": "",
}

VALID_DIRS = {"home", "draw", "away"}

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

SCORE_OTHERS_HOME = [
    "4-3", "5-3", "5-4", "6-0", "6-1", "6-2", "6-3", "6-4",
    "7-0", "7-1", "7-2", "胜其他", "9-0",
]

SCORE_OTHERS_DRAW = [
    "4-4", "5-5", "6-6", "平其他", "9-9",
]

SCORE_OTHERS_AWAY = [
    "3-4", "0-6", "1-6", "2-6", "3-6", "0-7", "1-7", "2-7",
    "负其他", "0-9",
]

ALL_SCORE_OTHERS = SCORE_OTHERS_HOME + SCORE_OTHERS_DRAW + SCORE_OTHERS_AWAY

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
    "欧协联",
    "解放者杯",
    "南球杯",
    "国王杯",
    "足总杯",
    "联赛杯",
    "Cup",
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
    s = str(score).strip().replace(" ", "")

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


def _normalize_goal_range_for_ui(goal_range: Any, score: str = "") -> Tuple[str, str]:
    s = str(goal_range or "").strip().replace(" ", "")

    if isinstance(goal_range, (list, tuple)) and len(goal_range) >= 2:
        lo = int(goal_range[0])
        hi = int(goal_range[1])
        if lo == 0 and hi <= 1:
            return "0-1", "0-1球"
        if lo == hi:
            return str(lo), f"{lo}球"
        if hi >= 6:
            return "6+", "6+球"
        return f"{lo}-{hi}", f"{lo}-{hi}球"

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
            return 1.01 < _f(v, 0) < 1000
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


def _extract_form_record(text: str) -> Tuple[int, int, int]:
    if not text:
        return 0, 0, 0

    patterns = [
        r"近\s*\d+\s*[主客场]*\s*(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
        r"(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
        r"近\s*\d+[:：]\s*(\d+)W(\d+)D(\d+)L",
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


def _fundamental_strength(match_obj: Dict, side: str) -> Dict[str, Any]:
    info_src = match_obj.get("points", {})
    if not isinstance(info_src, dict):
        info_src = {}

    key_strength = "home_strength" if side == "home" else "guest_strength"
    txt = str(info_src.get(key_strength, ""))

    w, d, l = _extract_form_record(txt)
    total = w + d + l
    win_rate = (w / total) if total > 0 else 0.5

    goals_for, goals_against = _extract_avg_goals(txt)

    score = 0.0
    if total > 0:
        score += (win_rate - 0.5) * 80
    if goals_for > 0:
        score += (goals_for - 1.3) * 20
    if goals_against > 0:
        score -= (goals_against - 1.3) * 20

    return {
        "wins": w,
        "draws": d,
        "losses": l,
        "total": total,
        "win_rate": round(win_rate, 3),
        "goals_for": round(goals_for, 2),
        "goals_against": round(goals_against, 2),
        "strength_score": round(max(-100, min(100, score)), 1),
    }


def calculate_value_bet(prob_pct, odds):
    if not odds or odds <= 1.05:
        return {"ev": 0.0, "kelly": 0.0, "is_value": False}

    prob = prob_pct / 100.0
    ev = prob * odds - 1.0
    b = odds - 1.0
    q = 1.0 - prob

    if b <= 0:
        return {"ev": round(ev * 100, 2), "kelly": 0.0, "is_value": False}

    kelly = ((b * prob) - q) / b

    return {
        "ev": round(ev * 100, 2),
        "kelly": round(max(0.0, kelly * 0.5) * 100, 2),
        "is_value": ev > 0.05,
    }


# ====================================================================
# 环境变量 / AI通道
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
    "GPT_API_KEY",
    "API_KEY",
]

GPT_URL_ALIASES = [
    "GPT_API_URL",
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
# 欧赔 / 盘口
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


# ====================================================================
# 外部市场联网/比对层
# ====================================================================

def _extract_external_urls(match_obj: Dict) -> List[str]:
    urls = []

    for key in [
        "external_url",
        "external_urls",
        "odds_url",
        "odds_urls",
        "market_url",
        "market_urls",
        "reference_url",
        "reference_urls",
    ]:
        v = match_obj.get(key)
        if not v:
            continue

        if isinstance(v, str):
            urls.append(v)
        elif isinstance(v, list):
            urls.extend([str(x) for x in v if x])
        elif isinstance(v, dict):
            urls.extend([str(x) for x in v.values() if x])

    clean = []
    for u in urls:
        u = str(u).strip()
        if u.startswith("http://") or u.startswith("https://"):
            if u not in clean:
                clean.append(u)

    return clean[:5]


def _strip_html(text: str, max_len=5000) -> str:
    s = str(text or "")
    s = re.sub(r"<script.*?</script>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<style.*?</style>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]


async def _fetch_external_url(session, url: str) -> Dict[str, Any]:
    try:
        timeout = aiohttp.ClientTimeout(total=15, connect=8, sock_read=12)
        headers = {
            "User-Agent": "Mozilla/5.0 vMAX/19.3 MarketAudit",
            "Accept": "text/html,application/json,text/plain,*/*",
        }
        async with session.get(url, headers=headers, timeout=timeout) as r:
            text = await r.text(errors="ignore")
            return {
                "url": url,
                "status": r.status,
                "text": _strip_html(text, 5000),
            }
    except Exception as e:
        return {
            "url": url,
            "status": "error",
            "text": f"fetch_error: {str(e)[:120]}",
        }


async def enrich_external_contexts(match_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    connector = aiohttp.TCPConnector(limit=6, use_dns_cache=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        task_map = []

        for idx, ma in enumerate(match_analyses):
            urls = _extract_external_urls(ma.get("match", {}))
            for url in urls:
                tasks.append(_fetch_external_url(session, url))
                task_map.append(idx)

        if not tasks:
            return match_analyses

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for ma in match_analyses:
            ma["external_fetch"] = []

        for idx, res in zip(task_map, results):
            if isinstance(res, dict):
                match_analyses[idx].setdefault("external_fetch", []).append(res)
            else:
                match_analyses[idx].setdefault("external_fetch", []).append({
                    "url": "",
                    "status": "error",
                    "text": str(res)[:200],
                })

    return match_analyses


def _extract_external_market_from_match(match_obj: Dict, fetched: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    fetched = fetched or []

    ext = match_obj.get("external_market")
    if isinstance(ext, dict):
        odds_obj = ext.get("odds", ext)
        probs_obj = ext.get("probs", ext.get("probabilities", {}))

        h_odd = _f(odds_obj.get("home", odds_obj.get("win", odds_obj.get("sp_home", 0))))
        d_odd = _f(odds_obj.get("draw", odds_obj.get("same", odds_obj.get("sp_draw", 0))))
        a_odd = _f(odds_obj.get("away", odds_obj.get("lose", odds_obj.get("sp_away", 0))))

        hp = _f(probs_obj.get("home", probs_obj.get("win", 0)))
        dp = _f(probs_obj.get("draw", probs_obj.get("same", 0)))
        ap = _f(probs_obj.get("away", probs_obj.get("lose", 0)))

        out = {
            "available": False,
            "source": ext.get("source", "external_market"),
            "odds": {},
            "probs": {},
            "snippets": [],
        }

        if h_odd > 1 and d_odd > 1 and a_odd > 1:
            probs = _compute_no_vig_probs({
                "sp_home": h_odd,
                "sp_draw": d_odd,
                "sp_away": a_odd,
            })
            out["available"] = True
            out["odds"] = {"home": h_odd, "draw": d_odd, "away": a_odd}
            out["probs"] = {
                "home": round(probs["home"], 1),
                "draw": round(probs["draw"], 1),
                "away": round(probs["away"], 1),
            }
            return out

        if hp > 0 and dp > 0 and ap > 0:
            hp, dp, ap = _pct_normalize(hp, dp, ap)
            out["available"] = True
            out["probs"] = {"home": hp, "draw": dp, "away": ap}
            return out

    h_odd = _deep_find_value(match_obj, [
        "external_sp_home", "external_home", "external_win",
        "avg_sp_home", "avg_home", "avg_win",
        "market_sp_home", "market_home", "market_win",
    ], positive_only=True, default=0)

    d_odd = _deep_find_value(match_obj, [
        "external_sp_draw", "external_draw", "external_same",
        "avg_sp_draw", "avg_draw", "avg_same",
        "market_sp_draw", "market_draw", "market_same",
    ], positive_only=True, default=0)

    a_odd = _deep_find_value(match_obj, [
        "external_sp_away", "external_away", "external_lose",
        "avg_sp_away", "avg_away", "avg_lose",
        "market_sp_away", "market_away", "market_lose",
    ], positive_only=True, default=0)

    snippets = []
    for item in fetched[:3]:
        if isinstance(item, dict):
            txt = _safe_str(item.get("text", ""), 1500)
            if txt:
                snippets.append({
                    "url": item.get("url", ""),
                    "status": item.get("status", ""),
                    "text": txt,
                })

    if h_odd > 1 and d_odd > 1 and a_odd > 1:
        probs = _compute_no_vig_probs({
            "sp_home": h_odd,
            "sp_draw": d_odd,
            "sp_away": a_odd,
        })
        return {
            "available": True,
            "source": "external_odds_fields",
            "odds": {"home": h_odd, "draw": d_odd, "away": a_odd},
            "probs": {
                "home": round(probs["home"], 1),
                "draw": round(probs["draw"], 1),
                "away": round(probs["away"], 1),
            },
            "snippets": snippets,
        }

    return {
        "available": bool(snippets),
        "source": "external_fetch_snippets_only" if snippets else "missing",
        "odds": {},
        "probs": {},
        "snippets": snippets,
    }


def compute_external_market_audit(match_obj: Dict, external_market: Dict[str, Any]) -> Dict[str, Any]:
    local_probs = _compute_no_vig_probs(match_obj)
    local_core = {
        "home": local_probs["home"],
        "draw": local_probs["draw"],
        "away": local_probs["away"],
    }

    local_dir = max(local_core, key=local_core.get)
    local_sorted = sorted(local_core.values(), reverse=True)
    local_gap = local_sorted[0] - local_sorted[1] if len(local_sorted) >= 2 else 0

    if not external_market or not external_market.get("probs"):
        return {
            "available": False,
            "local_dir": local_dir,
            "external_dir": None,
            "local_gap": round(local_gap, 1),
            "external_gap": 0.0,
            "conflict": False,
            "risk_tag": "NO_EXTERNAL_MARKET",
            "notes": ["无可标准化外部市场赔率。若提供 external_market 或 external_urls，可启用交叉验证。"],
        }

    ext_probs = external_market.get("probs", {})
    hp, dp, ap = _pct_normalize(ext_probs.get("home", 0), ext_probs.get("draw", 0), ext_probs.get("away", 0))
    external_core = {"home": hp, "draw": dp, "away": ap}

    external_dir = max(external_core, key=external_core.get)
    ext_sorted = sorted(external_core.values(), reverse=True)
    external_gap = ext_sorted[0] - ext_sorted[1] if len(ext_sorted) >= 2 else 0

    conflict = local_dir != external_dir

    risk_tag = "normal"
    notes = []

    if conflict and local_gap >= 4:
        risk_tag = "LOCAL_MARKET_TRAP"
        notes.append("体彩/本地盘口最高方向与外部市场最高方向冲突，本地最低赔方向不得直接采纳。")
    elif conflict:
        risk_tag = "MARKET_DISAGREEMENT"
        notes.append("本地盘口与外部市场方向不一致，但本地优势差距不大。")
    else:
        notes.append("本地盘口与外部市场方向一致。")

    return {
        "available": True,
        "local_dir": local_dir,
        "external_dir": external_dir,
        "local_gap": round(local_gap, 1),
        "external_gap": round(external_gap, 1),
        "local_probs": {k: round(v, 1) for k, v in local_core.items()},
        "external_probs": {k: round(v, 1) for k, v in external_core.items()},
        "conflict": conflict,
        "risk_tag": risk_tag,
        "notes": notes,
    }


# ====================================================================
# CRS 矩阵几何分析
# ====================================================================

def crs_implied_probabilities(match_obj: Dict) -> Tuple[Dict[str, float], float, float]:
    raw_odds = {}

    for score, key in CRS_FULL_MAP.items():
        odds = _f(match_obj.get(key, 0))
        if odds > 1.1:
            raw_odds[score] = odds

    extras = {}

    for key, scores_set in [
        ("crs_win", SCORE_OTHERS_HOME),
        ("crs_same", SCORE_OTHERS_DRAW),
        ("crs_lose", SCORE_OTHERS_AWAY),
    ]:
        odds = _f(match_obj.get(key, 0))
        if odds > 1.1:
            extras[key] = {
                "odds": odds,
                "scores": scores_set,
            }

    if len(raw_odds) < 8:
        return {}, 0.0, 0.0

    raw_sum = sum(1 / o for o in raw_odds.values())

    for ex in extras.values():
        raw_sum += 1 / ex["odds"]

    if raw_sum <= 0:
        return {}, 0.0, 0.0

    margin = raw_sum - 1.0

    probs = {}

    for score, odds in raw_odds.items():
        probs[score] = (1 / odds) / raw_sum * 100

    for key, ex in extras.items():
        total_prob = (1 / ex["odds"]) / raw_sum * 100
        num = len(ex["scores"])
        if num > 0:
            per = total_prob / num
            for sc in ex["scores"]:
                probs[sc] = probs.get(sc, 0) + per

    coverage = len(raw_odds) / len(CRS_FULL_MAP)

    return probs, round(margin, 3), round(coverage, 2)


def compute_statistical_moments(probs: Dict[str, float]) -> Dict[str, float]:
    regular = {}

    for sc, p in probs.items():
        if sc in ALL_SCORE_OTHERS and not str(sc).replace("-", "").isdigit():
            continue

        try:
            h, a = str(sc).split("-")
            h, a = int(h), int(a)

            if h > 8 or a > 8:
                continue

            regular[(h, a)] = p
        except Exception:
            continue

    if not regular:
        return {}

    total = sum(regular.values())

    if total < 1:
        return {}

    reg_normalized = {k: v / total for k, v in regular.items()}

    e_h = sum(h * p for (h, a), p in reg_normalized.items())
    e_a = sum(a * p for (h, a), p in reg_normalized.items())

    var_h = sum((h - e_h) ** 2 * p for (h, a), p in reg_normalized.items())
    var_a = sum((a - e_a) ** 2 * p for (h, a), p in reg_normalized.items())

    std_h = math.sqrt(var_h) if var_h > 0 else 0.01
    std_a = math.sqrt(var_a) if var_a > 0 else 0.01

    cov = sum((h - e_h) * (a - e_a) * p for (h, a), p in reg_normalized.items())
    corr = cov / (std_h * std_a) if (std_h * std_a) > 0 else 0.0

    if std_h > 0.01:
        skew_h = sum(((h - e_h) / std_h) ** 3 * p for (h, a), p in reg_normalized.items())
    else:
        skew_h = 0.0

    if std_a > 0.01:
        skew_a = sum(((a - e_a) / std_a) ** 3 * p for (h, a), p in reg_normalized.items())
    else:
        skew_a = 0.0

    return {
        "lambda_h": round(e_h, 3),
        "lambda_a": round(e_a, 3),
        "var_h": round(var_h, 3),
        "var_a": round(var_a, 3),
        "std_h": round(std_h, 3),
        "std_a": round(std_a, 3),
        "cov": round(cov, 3),
        "corr": round(corr, 3),
        "skew_h": round(skew_h, 3),
        "skew_a": round(skew_a, 3),
        "lambda_total": round(e_h + e_a, 3),
    }


def classify_shape(moments: Dict[str, float]) -> Tuple[str, List[str]]:
    if not moments:
        return "unknown", ["CRS数据不足，无法分析形状"]

    lh = moments.get("lambda_h", 1.3)
    la = moments.get("lambda_a", 1.2)
    lt = moments.get("lambda_total", 2.5)
    corr = moments.get("corr", 0.0)
    var_h = moments.get("var_h", 1.0)
    var_a = moments.get("var_a", 1.0)
    skew_h = moments.get("skew_h", 0.0)
    skew_a = moments.get("skew_a", 0.0)

    anomalies = []
    verdict = "normal"

    if lt >= 3.0 and corr >= 0.15:
        verdict = "shootout"
        anomalies.append(f"互射局: λ总{lt:.2f}, 相关{corr:.2f}")
    elif lt <= 2.2 and var_h < 1.2 and var_a < 1.2:
        verdict = "grinder"
        anomalies.append(f"磨局: λ总{lt:.2f}, 方差低({var_h:.2f}/{var_a:.2f})")
    elif lh - la >= 1.2:
        verdict = "lopsided_h"
        anomalies.append(f"主队碾压: λ主{lh:.2f} vs 客{la:.2f}")
    elif la - lh >= 1.2:
        verdict = "lopsided_a"
        anomalies.append(f"客队碾压: λ客{la:.2f} vs 主{lh:.2f}")
    elif abs(lh - la) < 0.4:
        verdict = "balanced"
        anomalies.append(f"均势: λ主{lh:.2f} vs 客{la:.2f}")

    if abs(skew_h) > 1.8:
        anomalies.append(f"主队进球分布偏度异常: {skew_h:.2f}")
    if abs(skew_a) > 1.8:
        anomalies.append(f"客队进球分布偏度异常: {skew_a:.2f}")
    if corr < -0.15:
        anomalies.append(f"负相关{corr:.2f}: 单边场，一方得势另一方沉默")

    return verdict, anomalies


def compute_direction_from_crs(probs: Dict[str, float]) -> Dict[str, float]:
    home_p = 0.0
    draw_p = 0.0
    away_p = 0.0

    for sc, p in probs.items():
        if sc == "胜其他" or sc == "9-0":
            home_p += p
            continue
        if sc == "平其他" or sc == "9-9":
            draw_p += p
            continue
        if sc == "负其他" or sc == "0-9":
            away_p += p
            continue

        try:
            h, a = str(sc).split("-")
            h, a = int(h), int(a)

            if h > a:
                home_p += p
            elif h < a:
                away_p += p
            else:
                draw_p += p
        except Exception:
            pass

    total = home_p + draw_p + away_p

    if total > 0:
        return {
            "home": round(home_p / total * 100, 2),
            "draw": round(draw_p / total * 100, 2),
            "away": round(away_p / total * 100, 2),
        }

    return {
        "home": 33.3,
        "draw": 33.3,
        "away": 33.4,
    }


def analyze_crs_matrix(match_obj: Dict) -> Dict[str, Any]:
    probs, margin, coverage = crs_implied_probabilities(match_obj)

    if not probs:
        return {
            "implied_probs": {},
            "margin": 0.0,
            "coverage": 0.0,
            "moments": {},
            "shape_verdict": "unknown",
            "anomalies": ["CRS数据缺失"],
            "direction_probs": {"home": 33.3, "draw": 33.3, "away": 33.4},
            "top_scores": [],
        }

    moments = compute_statistical_moments(probs)
    verdict, anomalies = classify_shape(moments)
    direction_probs = compute_direction_from_crs(probs)

    sorted_scores = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    top_scores = [
        {
            "score": sc,
            "prob": round(p, 2),
            "direction": _score_direction(sc),
            "goals": sum(_parse_score(sc)) if _parse_score(sc)[0] is not None else None,
            "odds": round(100 / p, 2) if p > 0 else 0,
        }
        for sc, p in sorted_scores[:10]
    ]

    return {
        "implied_probs": {k: round(v, 2) for k, v in probs.items()},
        "margin": margin,
        "coverage": coverage,
        "moments": moments,
        "shape_verdict": verdict,
        "anomalies": anomalies,
        "direction_probs": direction_probs,
        "top_scores": top_scores,
    }


# ====================================================================
# TTG 总进球分析
# ====================================================================

def analyze_ttg(match_obj: Dict) -> Dict[str, Any]:
    odds = {}

    for g in range(8):
        v = _f(match_obj.get(f"a{g}", 0))
        if v > 1:
            odds[g] = v

    if not odds:
        return {
            "available": False,
            "mode": None,
            "top3": [],
            "mean": 2.5,
            "anchors": [],
            "low_points": [],
        }

    inv = {g: 1 / o for g, o in odds.items()}
    total = sum(inv.values())

    probs = {g: inv[g] / total * 100 for g in inv} if total > 0 else {}
    mode = max(probs, key=probs.get)
    mean = sum(g * (p / 100.0) for g, p in probs.items())

    top3 = sorted(
        [{"goals": g, "odds": odds[g], "prob": round(probs[g], 2)} for g in odds],
        key=lambda x: x["prob"],
        reverse=True,
    )[:3]

    anchors = []
    low_points = []

    for g, o in odds.items():
        anchor = TTG_ANCHORS.get(g, {})
        hard_low = anchor.get("hard_low", 0)
        std = anchor.get("std", 0)

        if hard_low and o <= hard_low:
            low_points.append(f"{_ttg_label(g)}={o:.2f}")

        if std and o > 0:
            ratio = std / o
            if ratio >= 1.45:
                anchors.append({
                    "goals": g,
                    "odds": o,
                    "std": std,
                    "compression": round(ratio, 2),
                })

    return {
        "available": True,
        "mode": mode,
        "top3": top3,
        "mean": round(mean, 2),
        "probs": {str(g): round(p, 2) for g, p in probs.items()},
        "odds": {str(g): odds[g] for g in odds},
        "anchors": anchors,
        "low_points": low_points,
    }


def _ttg_score_weight(score: str, ttg_analysis: Dict[str, Any]) -> float:
    h, a = _parse_score(score)
    if h is None:
        return 0.25

    total_goals = h + a
    mode = ttg_analysis.get("mode")

    if mode is None:
        return 1.0

    diff = abs(total_goals - int(mode))
    weight = math.exp(-0.45 * diff)

    typical_scores = {
        0: ["0-0"],
        1: ["1-0", "0-1"],
        2: ["1-1", "2-0", "0-2"],
        3: ["2-1", "1-2", "3-0", "0-3"],
        4: ["3-1", "1-3", "2-2"],
        5: ["3-2", "2-3", "4-1", "1-4"],
        6: ["4-2", "2-4", "3-3"],
        7: ["4-3", "3-4", "5-2", "2-5", "5-1", "1-5"],
    }

    if score in typical_scores.get(int(mode), []):
        weight *= 1.35

    for anchor in ttg_analysis.get("anchors", []):
        g = anchor.get("goals")
        if g == total_goals:
            weight *= 1.25
        elif abs(g - total_goals) >= 2:
            weight *= 0.75

    return max(0.15, min(1.8, weight))


# ====================================================================
# Sharp / Steam 识别与分级
# ====================================================================

def detect_sharp_direction(smart_signals: List) -> Dict[str, Any]:
    detected = False
    sharp_dir = None
    raw = []

    for s in smart_signals:
        s_str = str(s)
        if "Sharp" not in s_str and "sharp" not in s_str:
            continue

        detected = True
        raw.append(s_str)

        if re.search(r"(主胜|主队|走主|→\s*主|流向\s*主|资金\s*主|Sharp主|home)", s_str, flags=re.IGNORECASE):
            sharp_dir = "home"
            break

        if re.search(r"(客胜|客队|走客|→\s*客|流向\s*客|资金\s*客|Sharp客|away)", s_str, flags=re.IGNORECASE):
            sharp_dir = "away"
            break

        if re.search(r"(平局|平赔|走平|→\s*平|流向\s*平|资金\s*平|Sharp平|进平局|draw)", s_str, flags=re.IGNORECASE):
            sharp_dir = "draw"
            break

    return {
        "detected": detected,
        "sharp_dir": sharp_dir,
        "raw": raw,
    }


def detect_steam_direction(smart_signals: List) -> Dict[str, Any]:
    steam_dir = None
    steam_type = None
    raw = []

    for s in smart_signals:
        s_str = str(s)
        if "Steam" not in s_str and "steam" not in s_str:
            continue

        raw.append(s_str)

        is_reverse = "反向" in s_str or "未跟" in s_str or "不跟" in s_str
        steam_type = "reverse" if is_reverse else "normal"

        if re.search(r"(主胜.*Steam|Steam.*主胜|主胜.*降水|主.*Steam|home)", s_str, flags=re.IGNORECASE):
            steam_dir = "home"
            break

        if re.search(r"(客胜.*Steam|Steam.*客胜|客胜.*降水|客.*Steam|away)", s_str, flags=re.IGNORECASE):
            steam_dir = "away"
            break

        if re.search(r"(平.*Steam|Steam.*平|draw)", s_str, flags=re.IGNORECASE):
            steam_dir = "draw"
            break

    return {
        "steam_dir": steam_dir,
        "steam_type": steam_type,
        "raw": raw,
    }


def classify_sharp_signal(
    match_obj: Dict,
    smart_signals: List,
    crs_analysis: Dict,
    ttg_analysis: Dict,
    external_audit: Optional[Dict] = None,
) -> Dict[str, Any]:
    sharp_info = detect_sharp_direction(smart_signals)
    sharp_dir = sharp_info.get("sharp_dir")
    detected = sharp_info.get("detected", False)

    if not detected or not sharp_dir:
        return {
            "sharp_detected": False,
            "sharp_dir": None,
            "sharp_level": "S0",
            "sharp_trust": 0.0,
            "sharp_reason": ["无明确Sharp方向"],
            "raw": sharp_info.get("raw", []),
        }

    reason = []
    trust = 45.0

    local_probs = _compute_no_vig_probs(match_obj)
    local_core = {
        "home": local_probs["home"],
        "draw": local_probs["draw"],
        "away": local_probs["away"],
    }
    local_dir = max(local_core, key=local_core.get)

    change = match_obj.get("change", {}) or {}
    cw = _f(change.get("win", 0))
    cs = _f(change.get("same", 0))
    cl = _f(change.get("lose", 0))

    move_dir = None
    if cw < cs and cw < cl and cw <= -0.04:
        move_dir = "home"
    elif cs < cw and cs < cl and cs <= -0.04:
        move_dir = "draw"
    elif cl < cw and cl < cs and cl <= -0.04:
        move_dir = "away"

    if move_dir == sharp_dir:
        trust += 18
        reason.append(f"Sharp方向与降水方向一致: {sharp_dir}")
    elif move_dir:
        trust -= 12
        reason.append(f"Sharp方向{sharp_dir}与降水方向{move_dir}冲突")

    if local_dir == sharp_dir:
        trust += 10
        reason.append(f"体彩/本地去水最高方向支持Sharp: {sharp_dir}")
    else:
        trust -= 8
        reason.append(f"体彩/本地去水最高方向{local_dir}不支持Sharp{sharp_dir}")

    crs_direction_probs = crs_analysis.get("direction_probs", {}) if crs_analysis else {}
    crs_dir = None

    if crs_direction_probs:
        crs_dir = max(crs_direction_probs, key=crs_direction_probs.get)
        crs_vals = sorted(crs_direction_probs.values(), reverse=True)
        crs_gap = crs_vals[0] - crs_vals[1] if len(crs_vals) >= 2 else 0

        if crs_dir == sharp_dir:
            trust += 18 if crs_gap >= 5 else 10
            reason.append(f"CRS方向聚合支持Sharp: {crs_dir}, gap={crs_gap:.1f}")
        else:
            trust -= 18 if crs_gap >= 5 else 8
            reason.append(f"CRS方向聚合{crs_dir}反对Sharp{sharp_dir}, gap={crs_gap:.1f}")

    top_scores = crs_analysis.get("top_scores", []) if crs_analysis else []
    sharp_path_scores = []

    for item in top_scores[:8]:
        if isinstance(item, dict):
            sc = item.get("score")
        elif isinstance(item, tuple):
            sc = item[0]
        else:
            continue

        if _score_direction(sc) == sharp_dir:
            sharp_path_scores.append(sc)

    if sharp_path_scores:
        trust += 10
        reason.append(f"Sharp方向存在CRS低赔比分路径: {sharp_path_scores[:3]}")
    else:
        trust -= 15
        reason.append("Sharp方向缺少CRS低赔比分路径")

    vote = match_obj.get("vote", {}) or {}
    vh = _f(vote.get("win", 0))
    vd = _f(vote.get("same", 0))
    va = _f(vote.get("lose", 0))

    hot_dir = None
    if max(vh, vd, va) >= 58:
        if vh >= vd and vh >= va:
            hot_dir = "home"
        elif vd >= vh and vd >= va:
            hot_dir = "draw"
        else:
            hot_dir = "away"

    if hot_dir == sharp_dir:
        trust -= 10
        reason.append(f"Sharp方向与散户大热同向: {sharp_dir}，疑似顺势诱导")
    elif hot_dir and hot_dir != sharp_dir:
        trust += 12
        reason.append(f"Sharp方向反散户大热: Sharp={sharp_dir}, hot={hot_dir}")

    if external_audit and external_audit.get("available"):
        external_dir = external_audit.get("external_dir")
        local_external_conflict = external_audit.get("conflict", False)

        if external_dir == sharp_dir:
            trust += 15
            reason.append(f"外部市场支持Sharp: {external_dir}")
        elif local_external_conflict and local_dir == sharp_dir and external_dir != sharp_dir:
            trust -= 18
            reason.append(
                f"体彩方向与Sharp同向但外部市场反对，疑似体彩诱盘: local={local_dir}, external={external_dir}"
            )
        elif external_dir and external_dir != sharp_dir:
            trust -= 8
            reason.append(f"外部市场不支持Sharp: external={external_dir}, sharp={sharp_dir}")

    ttg_mode = ttg_analysis.get("mode") if ttg_analysis else None
    if ttg_mode is not None:
        if int(ttg_mode) <= 1 and sharp_dir != "draw":
            reason.append(f"总进球主模态{ttg_mode}球，Sharp方向必须落到1-0/0-1路径")
        elif int(ttg_mode) == 2:
            reason.append(f"总进球主模态2球，Sharp方向需匹配2-0/0-2/1-1路径")
        elif int(ttg_mode) == 3:
            reason.append(f"总进球主模态3球，Sharp方向需匹配2-1/1-2路径")

    trust = max(0, min(100, trust))

    if trust >= 82:
        level = "S4"
    elif trust >= 68:
        level = "S3"
    elif trust >= 52:
        level = "S2"
    elif trust >= 35:
        level = "S1"
    else:
        level = "SD"

    return {
        "sharp_detected": True,
        "sharp_dir": sharp_dir,
        "sharp_level": level,
        "sharp_trust": round(trust, 1),
        "sharp_reason": reason,
        "raw": sharp_info.get("raw", []),
    }


# ====================================================================
# 陷阱矩阵：v19.3 核心版
# ====================================================================

def detect_T1_draw_trap_with_crs(match_obj: Dict, crs_analysis: Dict, sharp_eval: Dict) -> Optional[Dict]:
    change = match_obj.get("change", {}) or {}
    cw = _f(change.get("win", 0))
    cs = _f(change.get("same", 0))
    cl = _f(change.get("lose", 0))

    if cs >= -0.04:
        return None

    if cs > cw or cs > cl:
        return None

    w10 = _f(match_obj.get("w10", 999), 999)
    w21 = _f(match_obj.get("w21", 999), 999)
    l01 = _f(match_obj.get("l01", 999), 999)
    l12 = _f(match_obj.get("l12", 999), 999)
    s11 = _f(match_obj.get("s11", 999), 999)

    if min(w10, w21, l01, l12, s11) >= 900:
        return None

    home_small_strong = min(w10, w21) <= s11 * 1.08
    away_small_strong = min(l01, l12) <= s11 * 1.08
    draw_really_strong = s11 < min(w10, w21, l01, l12) * 0.95

    crs_dir = max(crs_analysis.get("direction_probs", {"draw": 33.3}), key=crs_analysis.get("direction_probs", {"draw": 33.3}).get)

    if draw_really_strong and crs_dir == "draw":
        return {
            "trap": "T1_REAL_DRAW_PROTECTION",
            "description": f"平赔独降但CRS真平局保护: s11={s11:.1f} 低于主/客小胜路径",
            "severity": 2,
            "direction_adjust": {"draw": +1.0, "home": -0.2, "away": -0.2},
            "score_multipliers": {},
            "boost_scores": ["1-1", "0-0"],
        }

    if home_small_strong:
        return {
            "trap": "T1_DRAW_TRAP_HOME_PATH",
            "description": f"平赔独降但主胜小比分路径更强: 1-0={w10:.1f}, 2-1={w21:.1f}, 1-1={s11:.1f}",
            "severity": 3,
            "direction_adjust": {"home": +1.4, "draw": -1.2, "away": -0.3},
            "score_multipliers": {"1-1": 0.45, "0-0": 0.5, "2-2": 0.55},
            "boost_scores": ["1-0", "2-1"],
            "suppress_draw_sharp": True,
        }

    if away_small_strong:
        return {
            "trap": "T1_DRAW_TRAP_AWAY_PATH",
            "description": f"平赔独降但客胜小比分路径更强: 0-1={l01:.1f}, 1-2={l12:.1f}, 1-1={s11:.1f}",
            "severity": 3,
            "direction_adjust": {"away": +1.4, "draw": -1.2, "home": -0.3},
            "score_multipliers": {"1-1": 0.45, "0-0": 0.5, "2-2": 0.55},
            "boost_scores": ["0-1", "1-2"],
            "suppress_draw_sharp": True,
        }

    return None


def detect_T2_T3_handicap_trap(match_obj: Dict) -> Optional[Dict]:
    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_d = _f(match_obj.get("sp_draw", match_obj.get("same", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))

    if sp_h < 1.05 or sp_d < 1.05 or sp_a < 1.05:
        return None

    theoretical = _infer_theoretical_handicap(sp_h, sp_d, sp_a)
    actual = _parse_actual_handicap(match_obj)
    diff = actual - theoretical

    if abs(diff) < 0.5:
        return None

    fund_h = _fundamental_strength(match_obj, "home")
    fund_a = _fundamental_strength(match_obj, "away")

    odds_strong = "home" if sp_h < sp_a else "away"

    if fund_h["total"] >= 3 and fund_a["total"] >= 3:
        fund_diff = fund_h["strength_score"] - fund_a["strength_score"]

        if odds_strong == "home" and fund_diff >= 20:
            return None

        if odds_strong == "away" and fund_diff <= -20:
            return None

    if diff >= 0.5:
        severity = 2 if abs(diff) < 1.0 else 3
        return {
            "trap": "T2_HANDICAP_DEEPER",
            "description": f"让球偏深: 理论{theoretical:+.2f} vs 实际{actual:+.2f}, 差{diff:+.2f}球，主队真强路径",
            "severity": severity,
            "direction_adjust": {"home": +1.2 * min(2.0, abs(diff)), "away": -0.5, "draw": -0.3},
            "score_multipliers": {},
            "boost_scores": ["2-0", "3-0", "2-1", "3-1"] if abs(diff) >= 1.0 else ["2-1", "2-0"],
        }

    severity = 2 if abs(diff) < 1.0 else 3
    return {
        "trap": "T3_HANDICAP_SHALLOWER",
        "description": f"让球偏浅: 理论{theoretical:+.2f} vs 实际{actual:+.2f}, 差{diff:+.2f}球，主队偏弱/客队不败路径",
        "severity": severity,
        "direction_adjust": {"home": -1.0 * min(2.0, abs(diff)), "away": +1.2 * min(2.0, abs(diff)), "draw": +0.4},
        "score_multipliers": {},
        "boost_scores": ["0-1", "1-2", "0-2", "1-1"] if abs(diff) >= 1.0 else ["0-1", "1-1"],
    }


def detect_T6_T7_score_range_trap(match_obj: Dict, exp_goals: float, ttg_analysis: Dict) -> Optional[Dict]:
    a0 = _f(match_obj.get("a0", 999), 999)
    a1 = _f(match_obj.get("a1", 999), 999)
    a2 = _f(match_obj.get("a2", 999), 999)

    low_small = 0
    if 0 < a0 < 8.0:
        low_small += 1
    if 0 < a1 < 4.5:
        low_small += 1
    if 0 < a2 < 3.0:
        low_small += 1

    if low_small >= 2 and exp_goals >= 2.8:
        return {
            "trap": "T6_SMALL_SCORE_TRAP",
            "description": f"诱小比分: a0/a1/a2压低{low_small}项，但λ={exp_goals:.2f}偏高",
            "severity": 2,
            "direction_adjust": {},
            "score_multipliers": {"0-0": 0.35, "1-0": 0.65, "0-1": 0.65, "1-1": 0.7},
            "boost_scores": ["2-1", "1-2", "2-2", "3-1", "1-3"],
        }

    a5 = _f(match_obj.get("a5", 999), 999)
    a6 = _f(match_obj.get("a6", 999), 999)
    a7 = _f(match_obj.get("a7", 999), 999)

    low_large = 0
    if 0 < a5 < 10:
        low_large += 1
    if 0 < a6 < 16:
        low_large += 1
    if 0 < a7 < 30:
        low_large += 1

    if low_large >= 2 and exp_goals <= 2.3:
        return {
            "trap": "T7_LARGE_SCORE_TRAP",
            "description": f"诱大比分: a5/a6/a7压低{low_large}项，但λ={exp_goals:.2f}偏低",
            "severity": 2,
            "direction_adjust": {},
            "score_multipliers": {"3-2": 0.4, "4-2": 0.35, "3-3": 0.35},
            "boost_scores": ["1-0", "0-1", "1-1", "2-0", "0-2", "2-1", "1-2"],
        }

    return None


def detect_T13_grinder_draw(match_obj: Dict, engine_result: Dict, ttg_analysis: Dict, crs_analysis: Dict) -> Optional[Dict]:
    probs = _compute_no_vig_probs(match_obj)
    max_dir_prob = max(probs["home"], probs["draw"], probs["away"])

    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))

    if hxg <= 0 or axg <= 0:
        return None

    total_xg = hxg + axg
    xg_diff = abs(hxg - axg)

    mode = ttg_analysis.get("mode")
    s11 = _f(match_obj.get("s11", 999), 999)
    s00 = _f(match_obj.get("s00", 999), 999)

    if total_xg <= 2.25 and xg_diff <= 0.35 and max_dir_prob < 48 and mode in [0, 1, 2]:
        if min(s11, s00) < 9.5:
            return {
                "trap": "T13_GRINDER_DRAW",
                "description": f"闷局/低比分平保护: xG总{total_xg:.2f}, xG差{xg_diff:.2f}, TTG模式{mode}球",
                "severity": 2,
                "direction_adjust": {"draw": +1.0, "home": -0.25, "away": -0.25},
                "score_multipliers": {"2-1": 0.7, "1-2": 0.7, "3-1": 0.45, "1-3": 0.45},
                "boost_scores": ["0-0", "1-1", "1-0", "0-1"],
            }

    return None


def detect_T14_cup_context(match_obj: Dict, sharp_eval: Dict, crs_analysis: Dict) -> Optional[Dict]:
    league = str(match_obj.get("league", match_obj.get("cup", "")))
    is_cup = any(kw in league for kw in CUP_KEYWORDS)

    if not is_cup:
        return None

    probs = _compute_no_vig_probs(match_obj)
    max_dir = max(["home", "draw", "away"], key=lambda k: probs[k])
    max_prob = probs[max_dir]

    crs_dir = max(crs_analysis.get("direction_probs", {"draw": 33.3}), key=crs_analysis.get("direction_probs", {"draw": 33.3}).get)

    direction_adjust = {"home": 0.0, "draw": +0.25, "away": 0.0}
    boost_scores = ["1-1", "1-0", "0-1", "2-1", "1-2"]
    score_multipliers = {"3-0": 0.7, "0-3": 0.7, "4-1": 0.55, "1-4": 0.55}

    if max_prob >= 55 and crs_dir != max_dir:
        direction_adjust[max_dir] -= 0.45
        direction_adjust["draw"] += 0.55

    sharp_level = sharp_eval.get("sharp_level")
    sharp_dir = sharp_eval.get("sharp_dir")

    if sharp_level in ["S3", "S4"] and sharp_dir in VALID_DIRS:
        direction_adjust[sharp_dir] += 0.35
        direction_adjust["draw"] -= 0.15

    return {
        "trap": "T14_CUP_CONTEXT",
        "description": f"杯赛/淘汰赛语境: {league}，降低大比分与单边碾压，优先小胜/平局路径",
        "severity": 1,
        "direction_adjust": direction_adjust,
        "score_multipliers": score_multipliers,
        "boost_scores": boost_scores,
    }


def detect_external_market_trap(external_audit: Dict, sharp_eval: Dict) -> Optional[Dict]:
    if not external_audit or not external_audit.get("available"):
        return None

    if external_audit.get("risk_tag") != "LOCAL_MARKET_TRAP":
        return None

    local_dir = external_audit.get("local_dir")
    external_dir = external_audit.get("external_dir")
    sharp_dir = sharp_eval.get("sharp_dir")
    sharp_level = sharp_eval.get("sharp_level")

    direction_adjust = {}

    if local_dir in VALID_DIRS:
        direction_adjust[local_dir] = direction_adjust.get(local_dir, 0) - 0.85

    if external_dir in VALID_DIRS:
        direction_adjust[external_dir] = direction_adjust.get(external_dir, 0) + 0.85

    if sharp_dir == local_dir and sharp_level in ["S1", "S2", "SD"]:
        direction_adjust[sharp_dir] = direction_adjust.get(sharp_dir, 0) - 0.55

    return {
        "trap": "T17_LOCAL_MARKET_TRAP",
        "description": f"体彩方向({local_dir})与外部市场({external_dir})冲突，本地最低赔/Sharp同向需降权",
        "severity": 3,
        "direction_adjust": direction_adjust,
        "score_multipliers": {},
        "boost_scores": [],
    }


def detect_all_traps(
    match_obj: Dict,
    engine_result: Dict,
    smart_signals: List,
    exp_goals: float,
    crs_analysis: Dict,
    ttg_analysis: Dict,
    sharp_eval: Dict,
    external_audit: Dict,
) -> Dict[str, Any]:
    detectors = [
        lambda: detect_T1_draw_trap_with_crs(match_obj, crs_analysis, sharp_eval),
        lambda: detect_T2_T3_handicap_trap(match_obj),
        lambda: detect_T6_T7_score_range_trap(match_obj, exp_goals, ttg_analysis),
        lambda: detect_T13_grinder_draw(match_obj, engine_result, ttg_analysis, crs_analysis),
        lambda: detect_T14_cup_context(match_obj, sharp_eval, crs_analysis),
        lambda: detect_external_market_trap(external_audit, sharp_eval),
    ]

    traps = []

    for detector in detectors:
        try:
            r = detector()
            if r:
                traps.append(r)
        except Exception as e:
            logger.warning(f"trap detector failed: {e}")

    has_real_draw = any(t.get("trap") == "T1_REAL_DRAW_PROTECTION" for t in traps)
    if has_real_draw:
        traps = [t for t in traps if t.get("trap") not in ["T1_DRAW_TRAP_HOME_PATH", "T1_DRAW_TRAP_AWAY_PATH"]]

    has_t13 = any(t.get("trap") == "T13_GRINDER_DRAW" for t in traps)
    if has_t13:
        traps = [t for t in traps if t.get("trap") != "T6_SMALL_SCORE_TRAP"]

    direction_adjust = {"home": 0.0, "draw": 0.0, "away": 0.0}
    score_multipliers = {}
    boost_scores = []
    confidence_penalty = 0
    total_severity = 0
    suppress_draw_sharp = False

    for t in traps:
        total_severity += t.get("severity", 1)

        for k, v in t.get("direction_adjust", {}).items():
            if k in direction_adjust:
                direction_adjust[k] += v

        for k, v in t.get("score_multipliers", {}).items():
            if k in score_multipliers:
                score_multipliers[k] = min(score_multipliers[k], v)
            else:
                score_multipliers[k] = v

        boost_scores.extend(t.get("boost_scores", []))
        confidence_penalty += t.get("confidence_penalty", 0)

        if t.get("suppress_draw_sharp"):
            suppress_draw_sharp = True

    steam_info = detect_steam_direction(smart_signals)

    return {
        "traps_detected": traps,
        "trap_count": len(traps),
        "total_severity": total_severity,
        "direction_adjust": direction_adjust,
        "score_multipliers": score_multipliers,
        "boost_scores": list(dict.fromkeys(boost_scores)),
        "confidence_penalty": confidence_penalty,
        "suppress_draw_sharp": suppress_draw_sharp,
        "sharp_eval": sharp_eval,
        "sharp_detected": sharp_eval.get("sharp_detected", False),
        "sharp_dir": sharp_eval.get("sharp_dir"),
        "sharp_level": sharp_eval.get("sharp_level"),
        "sharp_trust": sharp_eval.get("sharp_trust"),
        "steam_dir": steam_info.get("steam_dir"),
        "steam_type": steam_info.get("steam_type"),
    }


# ====================================================================
# 量化审计摘要
# ====================================================================

def build_quant_audit(
    match_obj: Dict,
    crs_analysis: Dict,
    ttg_analysis: Dict,
    external_audit: Dict,
    sharp_eval: Dict,
    trap_report: Dict,
) -> Dict[str, Any]:
    probs = _compute_no_vig_probs(match_obj)
    market_core = {
        "home": probs["home"],
        "draw": probs["draw"],
        "away": probs["away"],
    }
    market_dir = max(market_core, key=market_core.get)
    vals = sorted(market_core.values(), reverse=True)
    market_gap = vals[0] - vals[1] if len(vals) >= 2 else 0

    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_d = _f(match_obj.get("sp_draw", match_obj.get("same", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))

    actual_hc = _parse_actual_handicap(match_obj)
    theoretical_hc = _infer_theoretical_handicap(sp_h, sp_d, sp_a) if sp_h > 1 and sp_d > 1 and sp_a > 1 else 0.0

    crs_dir_probs = crs_analysis.get("direction_probs", {})
    crs_dir = max(crs_dir_probs, key=crs_dir_probs.get) if crs_dir_probs else None

    crs_vals = sorted(crs_dir_probs.values(), reverse=True) if crs_dir_probs else []
    crs_gap = crs_vals[0] - crs_vals[1] if len(crs_vals) >= 2 else 0

    return {
        "market_dir": market_dir,
        "market_gap": round(market_gap, 1),
        "market_probs": {k: round(v, 1) for k, v in market_core.items()},
        "actual_handicap": actual_hc,
        "theoretical_handicap": theoretical_hc,
        "handicap_diff": round(actual_hc - theoretical_hc, 2),
        "crs_dir": crs_dir,
        "crs_gap": round(crs_gap, 1),
        "crs_direction_probs": crs_dir_probs,
        "crs_shape": crs_analysis.get("shape_verdict"),
        "crs_moments": crs_analysis.get("moments", {}),
        "crs_top_scores": crs_analysis.get("top_scores", [])[:6],
        "ttg_mode": ttg_analysis.get("mode"),
        "ttg_mean": ttg_analysis.get("mean"),
        "ttg_top3": ttg_analysis.get("top3", []),
        "ttg_anchors": ttg_analysis.get("anchors", []),
        "external_audit": external_audit,
        "sharp_eval": sharp_eval,
        "trap_count": trap_report.get("trap_count", 0),
        "trap_direction_adjust": trap_report.get("direction_adjust", {}),
    }


# ====================================================================
# 贝叶斯方向后验 + 锁定链
# ====================================================================

def _ai_response_to_score(r: Dict) -> Optional[str]:
    if not isinstance(r, dict):
        return None

    for key in ["predicted_score", "ai_score", "score", "claude_score"]:
        sc = str(r.get(key, "")).strip()
        if sc and _parse_score(sc)[0] is not None:
            return sc

    top3 = r.get("top3", [])
    if isinstance(top3, list) and top3:
        first = top3[0]
        if isinstance(first, dict):
            sc = str(first.get("score", "")).strip()
        else:
            sc = str(first).strip()
        if sc and _parse_score(sc)[0] is not None:
            return sc

    return None


def compute_direction_posterior(
    match_obj: Dict,
    engine_result: Dict,
    trap_report: Dict,
    crs_analysis: Dict,
    ttg_analysis: Dict,
    ai_responses: Dict[str, Dict],
    external_audit: Dict,
) -> Dict[str, Any]:
    probs = _compute_no_vig_probs(match_obj)

    prior = {
        "home": max(0.05, probs["home"] / 100.0),
        "draw": max(0.05, probs["draw"] / 100.0),
        "away": max(0.05, probs["away"] / 100.0),
    }

    total_prior = sum(prior.values())
    prior = {k: v / total_prior for k, v in prior.items()}

    log_odds = {
        "home": math.log(prior["home"]),
        "draw": math.log(prior["draw"]),
        "away": math.log(prior["away"]),
    }

    evidences = []

    crs_direction = crs_analysis.get("direction_probs", {})
    if crs_direction and sum(crs_direction.values()) > 50:
        crs_p = {
            "home": max(0.05, crs_direction.get("home", 33.3) / 100),
            "draw": max(0.05, crs_direction.get("draw", 33.3) / 100),
            "away": max(0.05, crs_direction.get("away", 33.4) / 100),
        }

        for d in log_odds:
            llr = math.log(crs_p[d] / prior[d]) * 1.15
            log_odds[d] += llr

        evidences.append(f"CRS方向聚合:{crs_direction}")

    if external_audit and external_audit.get("available"):
        ext_probs = external_audit.get("external_probs", {})
        if ext_probs:
            ext_p = {
                "home": max(0.05, ext_probs.get("home", 33.3) / 100),
                "draw": max(0.05, ext_probs.get("draw", 33.3) / 100),
                "away": max(0.05, ext_probs.get("away", 33.4) / 100),
            }
            for d in log_odds:
                llr = math.log(ext_p[d] / prior[d]) * 1.05
                log_odds[d] += llr
            evidences.append(f"外部市场方向:{external_audit.get('external_dir')} risk={external_audit.get('risk_tag')}")

    sharp_eval = trap_report.get("sharp_eval", {})
    sharp_dir = sharp_eval.get("sharp_dir")
    sharp_level = sharp_eval.get("sharp_level", "S0")
    sharp_trust = _f(sharp_eval.get("sharp_trust", 0)) / 100.0

    if sharp_dir in log_odds:
        if sharp_level == "S4":
            sharp_weight = 3.2
        elif sharp_level == "S3":
            sharp_weight = 2.4
        elif sharp_level == "S2":
            sharp_weight = 1.4
        elif sharp_level == "S1":
            sharp_weight = 0.5
        elif sharp_level == "SD":
            sharp_weight = -0.8
        else:
            sharp_weight = 0.0

        if sharp_dir == "draw" and trap_report.get("suppress_draw_sharp"):
            sharp_weight = min(sharp_weight, 0.2)

        effective = sharp_weight * max(0.2, sharp_trust)

        log_odds[sharp_dir] += effective
        for other in log_odds:
            if other != sharp_dir:
                log_odds[other] -= effective / 2

        evidences.append(f"Sharp分级:{sharp_level}→{sharp_dir}, trust={sharp_trust:.2f}, effective={effective:+.2f}")

    steam_dir = trap_report.get("steam_dir")
    steam_type = trap_report.get("steam_type")

    if steam_dir in log_odds:
        base = 1.2 if steam_type != "reverse" else 1.7
        if steam_dir == sharp_dir and sharp_level in ["S3", "S4"]:
            base += 0.4
        if steam_dir == "draw" and trap_report.get("suppress_draw_sharp"):
            base *= 0.25

        log_odds[steam_dir] += base
        for other in log_odds:
            if other != steam_dir:
                log_odds[other] -= base / 2

        evidences.append(f"Steam({steam_type})→{steam_dir}, +{base:.2f}")

    trap_adj = trap_report.get("direction_adjust", {})
    for d, v in trap_adj.items():
        if d in log_odds:
            log_odds[d] += v

    if trap_adj and any(abs(v) > 0.05 for v in trap_adj.values()):
        evidences.append(f"陷阱方向调整:{trap_adj}")

    ai_directions = {"home": 0.0, "draw": 0.0, "away": 0.0}
    ai_score_votes = {}

    ai_weights = {
        "gpt": 0.75,
        "grok": 0.75,
        "gemini": 0.75,
        "claude": 1.05,
    }

    for name, r in ai_responses.items():
        sc = _ai_response_to_score(r)
        if not sc:
            continue

        d = _score_direction(sc)
        if d not in VALID_DIRS:
            continue

        conf = _f(r.get("confidence", r.get("ai_confidence", 60)), 60)
        w = ai_weights.get(name, 0.7) * max(0.45, min(1.15, conf / 70))

        ai_directions[d] += w
        ai_score_votes[sc] = ai_score_votes.get(sc, 0) + w

        top3 = r.get("top3", [])
        if isinstance(top3, list):
            for rank, item in enumerate(top3[1:3], 2):
                if isinstance(item, dict):
                    sc2 = str(item.get("score", "")).strip()
                else:
                    sc2 = str(item).strip()

                if _parse_score(sc2)[0] is not None:
                    ai_score_votes[sc2] = ai_score_votes.get(sc2, 0) + w * (0.35 if rank == 2 else 0.2)

    total_ai = sum(ai_directions.values())
    if total_ai > 0:
        max_ai = max(ai_directions.values())
        consensus = max_ai / total_ai

        ai_weight = 0.65 if consensus >= 0.7 else (0.4 if consensus >= 0.5 else 0.2)

        if sharp_level in ["S3", "S4"]:
            ai_weight *= 0.75
        elif sharp_level == "SD":
            ai_weight *= 0.5

        for d in log_odds:
            if ai_directions.get(d, 0) > 0:
                share = ai_directions[d] / total_ai
                log_odds[d] += math.log(max(0.05, share) / (1 / 3)) * ai_weight

        evidences.append(f"AI方向共识:{ai_directions}, 权重={ai_weight:.2f}")

    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))

    if hxg > 0.3 and axg > 0.3:
        xg_diff = hxg - axg

        if xg_diff > 0.5:
            log_odds["home"] += min(1.0, xg_diff * 0.55)
            log_odds["away"] -= min(0.7, xg_diff * 0.35)
            evidences.append(f"xG主优{xg_diff:+.2f}")
        elif xg_diff < -0.5:
            log_odds["away"] += min(1.0, abs(xg_diff) * 0.55)
            log_odds["home"] -= min(0.7, abs(xg_diff) * 0.35)
            evidences.append(f"xG客优{xg_diff:+.2f}")

    change = match_obj.get("change", {}) or {}
    cw = _f(change.get("win", 0))
    cs = _f(change.get("same", 0))
    cl = _f(change.get("lose", 0))

    if cw < -0.05:
        log_odds["home"] += min(0.65, abs(cw) * 5)
    elif cw > 0.05:
        log_odds["home"] -= min(0.4, cw * 4)

    if cs < -0.05:
        if not trap_report.get("suppress_draw_sharp"):
            log_odds["draw"] += min(0.55, abs(cs) * 4.5)
    elif cs > 0.05:
        log_odds["draw"] -= min(0.3, cs * 3)

    if cl < -0.05:
        log_odds["away"] += min(0.65, abs(cl) * 5)
    elif cl > 0.05:
        log_odds["away"] -= min(0.4, cl * 4)

    mode = ttg_analysis.get("mode")
    if mode is not None:
        mode = int(mode)
        if mode <= 1:
            log_odds["draw"] += 0.25
            evidences.append(f"TTG主模态{mode}球→低比分/平保护")
        elif mode == 2:
            log_odds["draw"] += 0.15
            evidences.append("TTG主模态2球→1-1/2-0/0-2路径")
        elif mode >= 5:
            weak_side = "home" if probs["home"] < probs["away"] else "away"
            log_odds[weak_side] += 0.18
            evidences.append(f"TTG主模态{mode}球→弱方参与度提高")

    league_str = str(match_obj.get("league", match_obj.get("cup", "")))
    if any(kw in league_str for kw in CUP_KEYWORDS):
        log_odds["draw"] += 0.22
        evidences.append("杯赛/淘汰赛90分钟平局轻加成")

    temperature = 1.65
    scaled = {k: v / temperature for k, v in log_odds.items()}
    max_log = max(scaled.values())
    exp_vals = {k: math.exp(v - max_log) for k, v in scaled.items()}
    total_exp = sum(exp_vals.values())

    posterior = {k: v / total_exp for k, v in exp_vals.items()}

    posterior = {k: max(0.03, min(0.88, v)) for k, v in posterior.items()}
    total_adj = sum(posterior.values())
    posterior = {k: v / total_adj for k, v in posterior.items()}

    final_direction = max(posterior, key=posterior.get)
    sorted_p = sorted(posterior.values(), reverse=True)
    dir_gap = sorted_p[0] - sorted_p[1] if len(sorted_p) >= 2 else 0

    return {
        "posterior": {k: round(v * 100, 2) for k, v in posterior.items()},
        "final_direction": final_direction,
        "dir_confidence": round(posterior[final_direction] * 100, 1),
        "dir_gap": round(dir_gap * 100, 1),
        "evidences": evidences,
        "prior": {k: round(v * 100, 2) for k, v in prior.items()},
        "ai_score_votes": ai_score_votes,
    }


def check_sharp_override(
    posterior_result: Dict[str, Any],
    trap_report: Dict[str, Any],
    external_audit: Dict[str, Any],
) -> Tuple[bool, Optional[str], str]:
    sharp_eval = trap_report.get("sharp_eval", {})
    sharp_dir = sharp_eval.get("sharp_dir")
    sharp_level = sharp_eval.get("sharp_level")
    sharp_trust = _f(sharp_eval.get("sharp_trust", 0))

    if sharp_dir not in VALID_DIRS:
        return False, None, "无Sharp方向"

    if sharp_level != "S4":
        return False, None, f"Sharp等级{sharp_level}不足S4"

    if sharp_trust < 82:
        return False, None, f"Sharp trust {sharp_trust}不足82"

    if external_audit and external_audit.get("available"):
        if external_audit.get("conflict") and external_audit.get("external_dir") != sharp_dir:
            return False, None, "外部市场反对Sharp，禁用Override"

    posterior = posterior_result.get("posterior", {})
    if _f(posterior.get(sharp_dir, 0)) < 25:
        return False, None, "Sharp方向后验低于25"

    return True, sharp_dir, "S4 Sharp Override触发"


def determine_goal_range(
    direction: str,
    moments: Dict[str, float],
    exp_goals: float,
    trap_report: Dict[str, Any],
    match_obj: Dict[str, Any],
    engine_result: Dict[str, Any],
    ttg_analysis: Dict[str, Any],
) -> Tuple[int, int, str]:
    mode = ttg_analysis.get("mode")
    mean = _f(ttg_analysis.get("mean", 0))

    lt = moments.get("lambda_total", 0) if moments else 0

    if lt <= 0:
        lt = exp_goals

    blended = 0.45 * lt + 0.35 * exp_goals + 0.20 * (mean if mean > 0 else exp_goals)

    if mode is not None:
        mode = int(mode)

        if mode <= 1:
            return 0, 2, "grinder"
        if mode == 2:
            return 1, 3, "low_goals"
        if mode == 3:
            return 2, 4, "normal"
        if mode == 4:
            return 3, 5, "high_goals"
        if mode >= 5:
            return 3, 7, "shootout"

    if blended >= 3.5:
        return 3, 6, "shootout"
    if blended >= 2.9:
        return 2, 5, "high_goals"
    if blended >= 2.3:
        return 2, 4, "normal"
    if blended >= 1.8:
        return 1, 3, "low_goals"

    return 0, 2, "grinder"


def select_score(
    direction: str,
    goal_range: Tuple[int, int],
    scenario: str,
    crs_analysis: Dict[str, Any],
    ai_score_votes: Dict[str, float],
    trap_report: Dict[str, Any],
    ttg_analysis: Dict[str, Any],
    match_obj: Dict[str, Any],
) -> Tuple[str, List[Tuple[str, float]]]:
    g_min, g_max = goal_range
    crs_probs = crs_analysis.get("implied_probs", {})
    candidates = {}

    for sc, p in crs_probs.items():
        sc_dir = _score_direction(sc)
        if sc_dir != direction:
            continue

        h, a = _parse_score(sc)
        if h is None:
            continue

        total_goals = h + a

        if "其他" in sc:
            if scenario == "shootout":
                candidates[sc] = p * 0.75
            else:
                candidates[sc] = p * 0.25
            continue

        if not (g_min <= total_goals <= g_max):
            continue

        candidates[sc] = p

    for sc, vote_pts in ai_score_votes.items():
        if _score_direction(sc) != direction:
            continue

        h, a = _parse_score(sc)
        if h is None:
            continue

        total_goals = h + a
        if not (g_min <= total_goals <= g_max):
            continue

        candidates[sc] = candidates.get(sc, 0) + vote_pts * 1.2

    for sc, mult in trap_report.get("score_multipliers", {}).items():
        if sc in candidates:
            candidates[sc] *= mult

    for sc in trap_report.get("boost_scores", []):
        if sc in candidates:
            candidates[sc] *= 1.35

    for sc in list(candidates.keys()):
        candidates[sc] *= _ttg_score_weight(sc, ttg_analysis)

    if scenario == "shootout":
        for sc in ["0-0", "1-0", "0-1"]:
            if sc in candidates:
                candidates[sc] *= 0.35
        for sc in ["2-2", "3-2", "2-3", "3-3", "3-1", "1-3"]:
            if sc in candidates:
                candidates[sc] *= 1.18

    elif scenario == "grinder":
        for sc in ["2-2", "3-1", "1-3", "3-2", "2-3"]:
            if sc in candidates:
                candidates[sc] *= 0.4
        for sc in ["1-0", "0-1", "0-0", "1-1"]:
            if sc in candidates:
                candidates[sc] *= 1.18

    elif scenario == "low_goals":
        for sc in ["3-2", "2-3", "3-3", "4-2", "2-4"]:
            if sc in candidates:
                candidates[sc] *= 0.45
        for sc in ["1-0", "0-1", "2-1", "1-2", "1-1", "0-0", "2-0", "0-2"]:
            if sc in candidates:
                candidates[sc] *= 1.1

    elif scenario == "high_goals":
        for sc in ["0-0", "1-0", "0-1"]:
            if sc in candidates:
                candidates[sc] *= 0.55

    sharp_eval = trap_report.get("sharp_eval", {})
    sharp_dir = sharp_eval.get("sharp_dir")
    sharp_level = sharp_eval.get("sharp_level")

    if sharp_dir == direction and sharp_level in ["S3", "S4"]:
        for sc in list(candidates.keys()):
            if _score_direction(sc) == sharp_dir:
                candidates[sc] *= 1.08

    if not candidates:
        fallback_map = {
            "home": "1-0" if scenario in ["grinder", "low_goals"] else "2-1",
            "away": "0-1" if scenario in ["grinder", "low_goals"] else "1-2",
            "draw": "1-1" if scenario != "grinder" else "0-0",
        }
        return fallback_map[direction], [(fallback_map[direction], 1.0)]

    sorted_scores = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    best = sorted_scores[0][0]

    return best, [(sc, round(v, 3)) for sc, v in sorted_scores[:10]]


def decision_lock_chain(
    match_obj: Dict[str, Any],
    engine_result: Dict[str, Any],
    trap_report: Dict[str, Any],
    crs_analysis: Dict[str, Any],
    ttg_analysis: Dict[str, Any],
    ai_responses: Dict[str, Dict],
    external_audit: Dict[str, Any],
    exp_goals: float,
) -> Dict[str, Any]:
    posterior_result = compute_direction_posterior(
        match_obj=match_obj,
        engine_result=engine_result,
        trap_report=trap_report,
        crs_analysis=crs_analysis,
        ttg_analysis=ttg_analysis,
        ai_responses=ai_responses,
        external_audit=external_audit,
    )

    posterior = posterior_result["posterior"]
    final_direction = posterior_result["final_direction"]

    override_triggered, override_dir, override_reason = check_sharp_override(
        posterior_result=posterior_result,
        trap_report=trap_report,
        external_audit=external_audit,
    )

    if override_triggered and override_dir:
        final_direction = override_dir
        cur_max = max(posterior.values())

        if posterior.get(override_dir, 0) < cur_max:
            posterior[override_dir] = cur_max + 5
            total = sum(posterior.values())
            posterior = {k: round(v / total * 100, 2) for k, v in posterior.items()}

    goal_range_min, goal_range_max, scenario = determine_goal_range(
        direction=final_direction,
        moments=crs_analysis.get("moments", {}),
        exp_goals=exp_goals,
        trap_report=trap_report,
        match_obj=match_obj,
        engine_result=engine_result,
        ttg_analysis=ttg_analysis,
    )

    best_score, top_candidates = select_score(
        direction=final_direction,
        goal_range=(goal_range_min, goal_range_max),
        scenario=scenario,
        crs_analysis=crs_analysis,
        ai_score_votes=posterior_result.get("ai_score_votes", {}),
        trap_report=trap_report,
        ttg_analysis=ttg_analysis,
        match_obj=match_obj,
    )

    final_direction_lock = _score_direction(best_score) or final_direction

    if final_direction_lock != final_direction:
        aligned = None
        for sc, pts in top_candidates:
            if _score_direction(sc) == final_direction:
                aligned = sc
                break

        if aligned:
            best_score = aligned
            final_direction_lock = _score_direction(best_score) or final_direction

    is_score_others = best_score in ALL_SCORE_OTHERS or "其他" in str(best_score)
    display_label = best_score

    if is_score_others:
        if final_direction_lock == "home":
            display_label = "胜其他"
            best_score = "胜其他"
        elif final_direction_lock == "away":
            display_label = "负其他"
            best_score = "负其他"
        else:
            display_label = "平其他"
            best_score = "平其他"

    post_argmax = max(posterior, key=posterior.get)

    raw_probability_argmax = post_argmax
    probability_was_corrected = False

    if post_argmax != final_direction_lock:
        cur_max = posterior[post_argmax]
        posterior[final_direction_lock] = cur_max + 3
        total = sum(posterior.values())
        posterior = {k: round(v / total * 100, 2) for k, v in posterior.items()}
        probability_was_corrected = True

    result_cn = _direction_cn(final_direction_lock)

    return {
        "predicted_score": best_score,
        "predicted_label": display_label,
        "predicted_direction": final_direction_lock,
        "final_direction": final_direction_lock,
        "result": result_cn,
        "display_direction": result_cn,
        "is_score_others": is_score_others,

        "home_win_pct": posterior["home"],
        "draw_pct": posterior["draw"],
        "away_win_pct": posterior["away"],

        "scenario": scenario,
        "goal_range": (goal_range_min, goal_range_max),
        "goal_range_label": _normalize_goal_range_for_ui((goal_range_min, goal_range_max), best_score)[1],

        "dir_confidence": posterior_result["dir_confidence"],
        "dir_gap": posterior_result["dir_gap"],
        "bayesian_prior": posterior_result["prior"],
        "bayesian_evidences": posterior_result["evidences"],

        "override_triggered": override_triggered,
        "override_reason": override_reason,

        "top_score_candidates": top_candidates,
        "raw_probability_argmax": raw_probability_argmax,
        "probability_was_corrected": probability_was_corrected,
        "posterior_after_lock": posterior,
    }


# ====================================================================
# 观察信号与格式化
# ====================================================================

def _extract_observation_codes(observation_signals: List[str]) -> List[str]:
    codes = []
    for s in observation_signals:
        m = re.search(r"\[(OBS\d+|T\d+|S[0-4D]+)[^\]]*\]", str(s))
        if m:
            codes.append(m.group(1))
    return codes


def _extract_observation_labels(observation_signals: List[str]) -> List[str]:
    labels = []
    for s in observation_signals:
        text = str(s)
        m = re.search(r"\[(OBS\d+|T\d+|S[0-4D]+)\s*([^\]]*)\]", text)
        if m:
            labels.append(f"{m.group(1)} {m.group(2).strip()}".strip())
        else:
            labels.append(text[:40])
    return labels


def _extract_observation_objects(observation_signals: List[str]) -> List[Dict[str, str]]:
    items = []
    for s in observation_signals:
        text = str(s)
        m = re.search(r"\[(OBS\d+|T\d+|S[0-4D]+)\s*([^\]]*)\]\s*(.*)", text)
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


def build_observation_signals(
    match_obj: Dict,
    engine_result: Dict,
    crs_analysis: Dict,
    ttg_analysis: Dict,
    sharp_eval: Dict,
    external_audit: Dict,
    trap_report: Dict,
    exp_goals: float,
) -> List[str]:
    facts = []

    probs = _compute_no_vig_probs(match_obj)

    facts.append(
        f"[OBS01 欧赔去水] 主{probs['home']:.1f}% / 平{probs['draw']:.1f}% / 客{probs['away']:.1f}%，最高={max(['home','draw','away'], key=lambda k: probs[k])}。"
    )

    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_d = _f(match_obj.get("sp_draw", match_obj.get("same", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))

    if sp_h > 1 and sp_d > 1 and sp_a > 1:
        actual = _parse_actual_handicap(match_obj)
        theoretical = _infer_theoretical_handicap(sp_h, sp_d, sp_a)
        facts.append(
            f"[OBS02 让球深度] 实际{actual:+.2f} / 欧赔理论{theoretical:+.2f} / 差异{actual - theoretical:+.2f}。"
        )

    change = match_obj.get("change", {}) or {}
    cw = _f(change.get("win", 0))
    cs = _f(change.get("same", 0))
    cl = _f(change.get("lose", 0))

    if abs(cw) + abs(cs) + abs(cl) > 0:
        facts.append(f"[OBS03 赔率变动] 主{cw:+.2f} / 平{cs:+.2f} / 客{cl:+.2f}。")

    if crs_analysis.get("coverage", 0) > 0:
        facts.append(
            f"[OBS04 CRS矩阵] 方向={crs_analysis.get('direction_probs')}，形状={crs_analysis.get('shape_verdict')}，λ={crs_analysis.get('moments', {}).get('lambda_total', '?')}。"
        )
        top = crs_analysis.get("top_scores", [])[:5]
        if top:
            facts.append(
                "[OBS05 CRS低赔路径] " + " | ".join([f"{x.get('score')}({x.get('prob')}%)" for x in top]) + "。"
            )

    if ttg_analysis.get("available"):
        facts.append(
            f"[OBS06 TTG主模态] mode={ttg_analysis.get('mode')}球，均值={ttg_analysis.get('mean')}，Top3={ttg_analysis.get('top3')}。"
        )
        if ttg_analysis.get("anchors"):
            facts.append(f"[OBS07 TTG锚点] {ttg_analysis.get('anchors')}。")

    if sharp_eval.get("sharp_detected"):
        facts.append(
            f"[OBS08 Sharp分级] dir={sharp_eval.get('sharp_dir')} level={sharp_eval.get('sharp_level')} trust={sharp_eval.get('sharp_trust')}。"
        )
        for r in sharp_eval.get("sharp_reason", [])[:4]:
            facts.append(f"[OBS08 Sharp证据] {r}。")

    if external_audit:
        facts.append(
            f"[OBS09 外部市场] available={external_audit.get('available')} local={external_audit.get('local_dir')} external={external_audit.get('external_dir')} risk={external_audit.get('risk_tag')}。"
        )

    for t in trap_report.get("traps_detected", []):
        facts.append(f"[{t.get('trap', 'TRAP')}] {t.get('description', '')}")

    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))
    if hxg > 0 and axg > 0:
        facts.append(f"[OBS10 庄家xG] 主{xg:.2f} / 客{axg:.2f}。" if False else f"[OBS10 庄家xG] 主{hxg:.2f} / 客{axg:.2f} / 总{hxg+axg:.2f}。")

    league = str(match_obj.get("league", match_obj.get("cup", "")))
    if any(kw in league for kw in CUP_KEYWORDS):
        facts.append(f"[OBS11 杯赛属性] {league}，90分钟需考虑保守/轮换/首回合/淘汰赛情境。")

    return facts


def format_match_block(
    idx: int,
    match: Dict,
    engine_result: Dict,
    crs_analysis: Dict,
    ttg_analysis: Dict,
    sharp_eval: Dict,
    external_market: Dict,
    external_audit: Dict,
    trap_report: Dict,
    quant_audit: Dict,
    observation_signals: List[str],
    ensemble_signals: Dict,
    smart_signals: List,
) -> str:
    home = match.get("home_team", match.get("home", "Home"))
    away = match.get("away_team", match.get("guest", "Away"))
    league = match.get("league", match.get("cup", ""))
    is_cup = any(kw in str(league) for kw in CUP_KEYWORDS)

    sp_h = _f(match.get("sp_home", match.get("win", 0)))
    sp_d = _f(match.get("sp_draw", match.get("same", 0)))
    sp_a = _f(match.get("sp_away", match.get("lose", 0)))

    probs = _compute_no_vig_probs(match)

    block = "\n════════════════════════════════════\n"
    block += f"第 {idx} 场: {home} vs {away}\n"
    block += "════════════════════════════════════\n"
    block += f"联赛/赛事: {league}{' [杯赛/淘汰赛属性]' if is_cup else ''}\n"
    block += f"比赛编号: {match.get('match_num', match.get('id', idx))}\n"

    block += "\n【0. 程序量化审计摘要】\n"
    block += f"本地欧赔方向: {quant_audit.get('market_dir')} gap={quant_audit.get('market_gap')} | probs={quant_audit.get('market_probs')}\n"
    block += (
        f"让球: 实际{quant_audit.get('actual_handicap'):+.2f} / "
        f"理论{quant_audit.get('theoretical_handicap'):+.2f} / "
        f"差异{quant_audit.get('handicap_diff'):+.2f}\n"
    )
    block += f"CRS方向: {quant_audit.get('crs_dir')} gap={quant_audit.get('crs_gap')} | shape={quant_audit.get('crs_shape')}\n"
    block += f"CRS低赔路径: {quant_audit.get('crs_top_scores')}\n"
    block += f"TTG主模态: {quant_audit.get('ttg_mode')}球 | 均值={quant_audit.get('ttg_mean')} | Top3={quant_audit.get('ttg_top3')}\n"
    block += f"Sharp分级: {sharp_eval.get('sharp_level')} dir={sharp_eval.get('sharp_dir')} trust={sharp_eval.get('sharp_trust')}\n"
    block += f"外部市场审计: {external_audit}\n"
    block += f"陷阱方向调整: {trap_report.get('direction_adjust')}\n"

    block += "\n【1. 欧赔原始数据】\n"
    block += f"即时欧赔: 主胜 {sp_h:.2f} / 平局 {sp_d:.2f} / 客胜 {sp_a:.2f}\n"

    if probs.get("margin", 0) > 0:
        block += (
            f"欧赔去水概率: 主 {probs['home']:.1f}% / 平 {probs['draw']:.1f}% / "
            f"客 {probs['away']:.1f}% / 返还率约 {100 / probs['margin']:.1f}%\n"
        )
    else:
        block += "欧赔去水概率: 数据不足\n"

    block += "\n【2. 外部市场/联网比对】\n"
    if external_market.get("odds"):
        block += f"外部赔率: {external_market.get('odds')}\n"
    if external_market.get("probs"):
        block += f"外部去水概率: {external_market.get('probs')}\n"
    if external_market.get("snippets"):
        block += "外部抓取摘要:\n"
        for sn in external_market.get("snippets", [])[:2]:
            block += f"- {sn.get('url', '')} | status={sn.get('status')} | {sn.get('text', '')[:800]}\n"
    if not external_market.get("available"):
        block += "无可标准化外部市场数据。若模型通道具备联网能力，可以自行检索；若不能联网，严禁编造外部数据。\n"

    block += "\n【3. 让球/盘口】\n"
    block += f"原始 give_ball/handicap: {match.get('give_ball', match.get('handicap', '0'))}\n"
    block += f"标准化实际让球: {_parse_actual_handicap(match):+.2f}，内部约定：正数=主让，负数=客让/主受让\n"
    block += f"欧赔反推理论让球: {_infer_theoretical_handicap(sp_h, sp_d, sp_a):+.2f}\n"

    change = match.get("change", {}) or {}
    block += "\n【4. 赔率变动】\n"
    block += f"主胜变化 {_f(change.get('win', 0)):+.2f} / 平赔变化 {_f(change.get('same', 0)):+.2f} / 客胜变化 {_f(change.get('lose', 0)):+.2f}\n"

    block += "\n【5. 庄家隐含 xG / 期望进球】\n"
    hxg = engine_result.get("bookmaker_implied_home_xg", None)
    axg = engine_result.get("bookmaker_implied_away_xg", None)
    exp_total = _f(engine_result.get("expected_total_goals", 0))
    if exp_total <= 0:
        exp_total = _f(hxg, 0) + _f(axg, 0)
    block += f"主 xG: {hxg if hxg is not None else 'N/A'} / 客 xG: {axg if axg is not None else 'N/A'}\n"
    block += f"期望总进球: {exp_total:.2f}\n"
    block += f"大2.5概率: {engine_result.get('over_25', 'N/A')} / BTTS: {engine_result.get('btts', 'N/A')}\n"

    block += "\n【6. 总进球数赔率 a0~a7】\n"
    ttg_lines = []
    for g in range(8):
        v = _f(match.get(f"a{g}", 0))
        if v > 1:
            mark = ""
            anchor = TTG_ANCHORS.get(g, {})
            if anchor.get("hard_low") and v <= anchor["hard_low"]:
                mark = " [低赔点]"
            ttg_lines.append(f"{_ttg_label(g)}={v:.2f}{mark}")
        else:
            ttg_lines.append(f"{_ttg_label(g)}=N/A")
    block += " | ".join(ttg_lines) + "\n"
    block += f"TTG分析: {ttg_analysis}\n"

    block += "\n【7. CRS 精确比分赔率】\n"
    crs_h = []
    crs_d = []
    crs_a = []
    for sc, key in CRS_FULL_MAP.items():
        v = _f(match.get(key, 0))
        if v <= 1:
            continue
        text = f"{sc}={v:.1f}"
        d = _score_direction(sc)
        if d == "home":
            crs_h.append(text)
        elif d == "draw":
            crs_d.append(text)
        elif d == "away":
            crs_a.append(text)
    block += "主胜系: " + (" | ".join(crs_h) if crs_h else "N/A") + "\n"
    block += "平局系: " + (" | ".join(crs_d) if crs_d else "N/A") + "\n"
    block += "客胜系: " + (" | ".join(crs_a) if crs_a else "N/A") + "\n"
    block += f"CRS矩阵分析: direction={crs_analysis.get('direction_probs')} shape={crs_analysis.get('shape_verdict')} moments={crs_analysis.get('moments')}\n"
    block += f"CRS TopScores: {crs_analysis.get('top_scores', [])[:8]}\n"

    hf_lines = []
    for k, label in HFTF_MAP.items():
        v = _f(match.get(k, 0))
        if v > 1:
            hf_lines.append(f"{label}={v:.2f}")
    if hf_lines:
        block += "\n【8. 半全场赔率】\n"
        block += " | ".join(hf_lines) + "\n"

    vote = match.get("vote", {}) or {}
    if vote:
        block += "\n【9. 散户分布】\n"
        block += f"主胜 {vote.get('win', '?')}% / 平局 {vote.get('same', '?')}% / 客胜 {vote.get('lose', '?')}%\n"

    if smart_signals or sharp_eval.get("sharp_detected"):
        block += "\n【10. Sharp/Steam/智能信号】\n"
        block += f"Sharp分级: {sharp_eval}\n"
        for s in smart_signals[:12]:
            block += f"- {_safe_str(s, 260)}\n"

    points = match.get("points", {}) or {}
    if isinstance(points, dict):
        block += "\n【11. 基本面情报】\n"
        for k, label in [
            ("home_strength", "主队"),
            ("guest_strength", "客队"),
            ("history", "交锋"),
            ("h2h", "交锋"),
            ("match_points", "赛事要点"),
        ]:
            txt = _safe_str(points.get(k, ""), 700)
            if txt:
                block += f"{label}: {txt}\n"

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
                info_lines.append(f"{label}: {_safe_str(v, 400)}")
        if info_lines:
            block += "\n【12. 伤停/异动消息】\n"
            block += "\n".join(info_lines) + "\n"

    if ensemble_signals.get("total", 0) > 0:
        block += "\n【13. 统计模型矩阵，仅供参考】\n"
        for row in ensemble_signals["models"]:
            block += (
                f"{row['name']}: 主{row['home']:.1f}% / 平{row['draw']:.1f}% / "
                f"客{row['away']:.1f}% → {row['direction']}\n"
            )
        block += f"模型共识: {ensemble_signals.get('consensus')} {ensemble_signals.get('consensus_count')}/{ensemble_signals.get('total')}\n"

    if observation_signals:
        block += "\n【14. 系统观察信号/陷阱信号】\n"
        for fact in observation_signals:
            block += f"- {fact}\n"

    return block


# ====================================================================
# Ensemble 信号
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
# AI Prompt
# ====================================================================

PHASE1_ROLES = {
    "gpt": {
        "name": "赔率结构分析师",
        "focus": "欧赔、让球、CRS、TTG、半全场的一致性与裂痕。重点检查比分路径是否成立。",
        "temperature": 0.18,
    },
    "grok": {
        "name": "体彩Sharp资金流分析师",
        "focus": "Sharp/Steam、赔率变动、散户冷热、体彩诱盘、外部市场冲突。重点给Sharp分级。",
        "temperature": 0.28,
    },
    "gemini": {
        "name": "基本面与赛事情境分析师",
        "focus": "战绩、伤停、主客场、杯赛/淘汰赛属性、赛程动机。重点判断场景是否支持盘口方向。",
        "temperature": 0.20,
    },
}


def build_phase1_prompt(match_blocks: List[str], role_key: str) -> str:
    role = PHASE1_ROLES[role_key]

    p = ""
    p += "<context>\n"
    p += "你是竞彩足球体彩市场分析团队中的一名独立分析师。下面是完整原始抓包、程序量化审计、外部市场比对与Sharp分级。\n"
    p += "体彩场次经过筛选，可能存在诱盘。Sharp是重要核心变量，但必须先判断Sharp真假与等级。\n"
    p += "如果你的模型通道具备联网能力，可以主动检索赔率/赛前情报/主流市场，但必须和程序提供的外部市场审计交叉验证；如果不能联网，禁止编造外部数据。\n"
    p += "</context>\n\n"

    p += "<your_role>\n"
    p += f"角色: {role['name']}\n"
    p += f"专长: {role['focus']}\n"
    p += "</your_role>\n\n"

    p += "<sharp_grading_rules>\n"
    p += "S4: Sharp + 变盘 + CRS + 外部市场/盘口 至少3项共振，可覆盖方向。\n"
    p += "S3: Sharp + 至少2项结构共振，是主方向强证据。\n"
    p += "S2: Sharp + 1项结构共振，只能加权。\n"
    p += "S1: 只有Sharp文本，不得单独改方向。\n"
    p += "SD: Sharp与CRS/TTG/外部市场明显冲突，按诱饵或低可信处理。\n"
    p += "</sharp_grading_rules>\n\n"

    p += "<hard_rules>\n"
    p += "1. 不得把Sharp文本直接等价为最终方向，必须说明Sharp等级。\n"
    p += "2. 体彩/本地最低赔方向若与外部市场冲突，必须标记 LOCAL_MARKET_TRAP，不得直接采纳最低赔方向。\n"
    p += "3. 最终比分必须落在 CRS低赔路径 + TTG主模态附近。\n"
    p += "4. 平赔独降不等于真平局，必须比较1-1与1-0/2-1/0-1/1-2路径。\n"
    p += "5. 杯赛属性只降低大比分和单边碾压，不得单独把强方向改成平局。\n"
    p += "6. predicted_score 的方向必须与 predicted_direction 一致。\n"
    p += "7. top3[0] 必须等于 predicted_score。\n"
    p += "8. 必须输出 doubts，即反对自己结论的证据。\n"
    p += "9. 如果数据不足，要降低 confidence。\n"
    p += "10. 严格输出 JSON 数组，不要 markdown，不要前缀后缀。\n"
    p += "</hard_rules>\n\n"

    p += "<match_data>\n"
    for block in match_blocks:
        p += block
    p += "\n</match_data>\n\n"

    p += "<output_format>\n"
    p += "[\n"
    p += "  {\n"
    p += "    \"match\": 1,\n"
    p += "    \"surface_direction\": \"away\",\n"
    p += "    \"sharp_direction\": \"away\",\n"
    p += "    \"sharp_grade\": \"S2\",\n"
    p += "    \"external_market_direction\": \"home\",\n"
    p += "    \"local_market_trap\": true,\n"
    p += "    \"crs_direction\": \"home\",\n"
    p += "    \"ttg_mode\": 1,\n"
    p += "    \"main_conflict\": \"Sharp客胜与CRS/外部市场冲突\",\n"
    p += "    \"predicted_score\": \"1-0\",\n"
    p += "    \"predicted_direction\": \"home\",\n"
    p += "    \"goal_range\": \"1球\",\n"
    p += "    \"home_win_pct\": 44,\n"
    p += "    \"draw_pct\": 31,\n"
    p += "    \"away_win_pct\": 25,\n"
    p += "    \"confidence\": 66,\n"
    p += "    \"top3\": [\n"
    p += "      {\"score\":\"1-0\", \"prob\":16},\n"
    p += "      {\"score\":\"1-1\", \"prob\":14},\n"
    p += "      {\"score\":\"2-1\", \"prob\":12}\n"
    p += "    ],\n"
    p += "    \"accepted_observations\": [],\n"
    p += "    \"rejected_observations\": [],\n"
    p += "    \"key_signals\": [],\n"
    p += "    \"doubts\": [],\n"
    p += "    \"reason\": \"中文推理，说明Sharp等级、外部市场、CRS、TTG、最终比分路径\"\n"
    p += "  }\n"
    p += "]\n"
    p += "</output_format>\n"

    return p


def build_phase1_retry_prompt(match_block: str, role_key: str, target_match_id: int) -> str:
    role = PHASE1_ROLES[role_key]

    p = ""
    p += "<context>\n"
    p += "这是一次单场补分析。只分析下面这一场。\n"
    p += f"输出 JSON 中 match 必须等于 {target_match_id}。\n"
    p += "</context>\n\n"

    p += "<your_role>\n"
    p += f"角色: {role['name']}\n"
    p += f"专长: {role['focus']}\n"
    p += "</your_role>\n\n"

    p += "<rules>\n"
    p += "1. 必须判断Sharp等级。\n"
    p += "2. 必须检查外部市场冲突。\n"
    p += "3. predicted_score 必须与 predicted_direction 一致。\n"
    p += "4. top3[0] 必须等于 predicted_score。\n"
    p += "5. 严格输出JSON数组，不要markdown。\n"
    p += "</rules>\n\n"

    p += "<match_data>\n"
    p += match_block
    p += "\n</match_data>\n\n"

    p += "<output_format>\n"
    p += "[{\"match\":%d,\"predicted_score\":\"1-0\",\"predicted_direction\":\"home\",\"goal_range\":\"1球\",\"home_win_pct\":44,\"draw_pct\":31,\"away_win_pct\":25,\"confidence\":60,\"top3\":[{\"score\":\"1-0\",\"prob\":16},{\"score\":\"1-1\",\"prob\":14},{\"score\":\"2-1\",\"prob\":12}],\"sharp_grade\":\"S2\",\"local_market_trap\":false,\"key_signals\":[],\"doubts\":[],\"reason\":\"中文推理\"}]\n" % target_match_id
    p += "</output_format>\n"

    return p


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
        f"Sharp={r.get('sharp_grade', '?')}/{r.get('sharp_direction', '?')} | "
        f"外部方向={r.get('external_market_direction', '?')} | "
        f"CRS={r.get('crs_direction', '?')} | "
        f"信心={r.get('confidence', '?')} | "
        f"top3=[{top3_str}]"
    )


def build_phase2_prompt(match_blocks: List[str], phase1_results: Dict[str, Dict[int, Dict]]) -> str:
    num_matches = len(match_blocks)

    p = ""
    p += "<context>\n"
    p += "你是Claude终审审计者。GPT/Grok/Gemini已经从不同角度输出结论。\n"
    p += "你必须重新审计原始抓包、程序量化审计、Sharp分级、外部市场比对和三家结论。\n"
    p += "注意: 你的输出是审计意见，不是最终程序锁定结果。最终比分会经过程序决策锁定链。\n"
    p += "</context>\n\n"

    p += "<critical_rules>\n"
    p += "1. 不按票数裁决，必须按证据质量裁决。\n"
    p += "2. 三家如果共同依赖同一个OBS或同一个Sharp文本，只按1个证据源计。\n"
    p += "3. Sharp是体彩核心变量，但必须分级；S3/S4可强跟，SD必须降权。\n"
    p += "4. 本地最低赔方向若与外部市场冲突，必须标记LOCAL_MARKET_TRAP。\n"
    p += "5. 平赔独降不能直接等于平局。\n"
    p += "6. 杯赛属性不能单独改变方向，只能影响比分保守性。\n"
    p += "7. 最终建议比分必须落在CRS低赔路径和TTG主模态附近。\n"
    p += "8. predicted_score方向必须与predicted_direction一致。\n"
    p += "9. top3[0]必须等于predicted_score。\n"
    p += "10. 严格JSON数组输出，不要markdown。\n"
    p += "</critical_rules>\n\n"

    p += "<three_analysts_results>\n"

    for i in range(1, num_matches + 1):
        p += f"\n════════ 第 {i} 场三家结论 ════════\n"

        for ai_name in ["gpt", "grok", "gemini"]:
            r = phase1_results.get(ai_name, {}).get(i, {})
            p += f"\n【{ai_name.upper()}】{_phase1_summary_line(r)}\n"

            if not r:
                continue

            for field in ["main_conflict", "key_signals", "accepted_observations", "rejected_observations", "doubts"]:
                v = r.get(field)
                if v:
                    p += f"{field}: {v}\n"

            reason = _safe_str(r.get("reason", ""), 1800)
            if reason:
                p += f"reason: {reason}\n"

    p += "\n</three_analysts_results>\n\n"

    p += "<raw_match_data>\n"
    for block in match_blocks:
        p += block
    p += "\n</raw_match_data>\n\n"

    p += "<output_format>\n"
    p += "[\n"
    p += "  {\n"
    p += "    \"match\": 1,\n"
    p += "    \"predicted_score\": \"1-0\",\n"
    p += "    \"predicted_direction\": \"home\",\n"
    p += "    \"goal_range\": \"1球\",\n"
    p += "    \"home_win_pct\": 44,\n"
    p += "    \"draw_pct\": 31,\n"
    p += "    \"away_win_pct\": 25,\n"
    p += "    \"confidence\": 66,\n"
    p += "    \"sharp_grade_final\": \"S2\",\n"
    p += "    \"external_market_conflict\": true,\n"
    p += "    \"local_market_trap\": true,\n"
    p += "    \"crs_ttg_score_path\": \"1-0/1-1强于客胜路径\",\n"
    p += "    \"agreement_pattern\": \"GPT/Grok客胜但同源Sharp污染，Gemini主不败\",\n"
    p += "    \"analysis_coverage\": {\"gpt\": true, \"grok\": true, \"gemini\": true, \"valid_count\": 3},\n"
    p += "    \"adopted_analysts\": [\"gemini\"],\n"
    p += "    \"rejected_analysts\": [\"gpt\", \"grok\"],\n"
    p += "    \"audit_result\": \"Sharp客胜没有通过外部市场和CRS验证，主不败路径更合理\",\n"
    p += "    \"top3\": [\n"
    p += "      {\"score\":\"1-0\", \"prob\":16},\n"
    p += "      {\"score\":\"1-1\", \"prob\":14},\n"
    p += "      {\"score\":\"2-1\", \"prob\":12}\n"
    p += "    ],\n"
    p += "    \"accepted_observations\": [],\n"
    p += "    \"rejected_observations\": [],\n"
    p += "    \"doubts\": [],\n"
    p += "    \"arbitration_reason\": \"完整中文终审理由\"\n"
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
        "models": ["gpt-5.5"],
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
    "temperature": 0.20,
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
# AI 调用层
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
    phase: str,
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
                        print(f"    ⚠️ HTTP {r.status} | {elapsed}s | 模型={mn} | {text[:240]}")
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
                print(f"    ⏱️ {ai_name.upper()} 读取超时: {READ_TIMEOUT}s")
                return ai_name, {}, "read_timeout"
            except Exception as e:
                if not connected:
                    continue
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
                        "message_content",
                        "assistant_content",
                        "model_response",
                    ]:
                        v = msg.get(field, "")
                        if isinstance(v, str) and v.strip():
                            raw_text = v.strip()
                            break

                if not raw_text:
                    skip = ("reasoning_content", "thinking", "reasoning", "thoughts", "chain_of_thought", "cot")
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
        top3 = out.get("top3", [])
        if isinstance(top3, list) and top3:
            first = top3[0]
            if isinstance(first, dict):
                score = str(first.get("score", "")).strip()
            else:
                score = str(first).strip()

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

    conf = _i(out.get("confidence", out.get("ai_confidence", 55)), 55)
    conf = max(25, min(95, conf))
    out["confidence"] = conf
    out["ai_confidence"] = conf

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
        if isinstance(t, dict):
            sc = str(t.get("score", "")).strip()
            prob = _f(t.get("prob", 0))
        else:
            sc = str(t).strip()
            prob = 0

        if not sc:
            continue

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
        key=lambda x: 0 if x.get("score") == score_label else 1,
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
        out["audit_result"] = _safe_str(out.get("audit_result", ""), 1500)
    else:
        out["reason"] = _safe_str(out.get("reason", ""), 5000)

    return len(errors) == 0, out, errors


# ====================================================================
# Phase 1 覆盖率 / 补跑
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
    sys_prompts: Dict[str, str],
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
        "gpt": (
            "<role>你是赔率结构分析师，专注欧赔、让球、CRS、TTG、半全场一致性。</role>"
            "<instruction>严格输出JSON数组，不要markdown。若通道可联网，可检索赔率但必须说明外部市场冲突。</instruction>"
        ),
        "grok": (
            "<role>你是体彩Sharp资金流分析师，专注Sharp/Steam、体彩诱盘、外部市场冲突。</role>"
            "<instruction>必须给Sharp分级S0/S1/S2/S3/S4/SD。严格输出JSON数组。</instruction>"
        ),
        "gemini": (
            "<role>你是基本面与赛事情境分析师，专注战绩、伤停、赛程、杯赛属性、主客场强弱。</role>"
            "<instruction>严格输出JSON数组，不要markdown。</instruction>"
        ),
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
    num_matches: int,
) -> Tuple[Dict[int, Dict], str]:
    print(f"\n  [Phase 2] Claude 终审审计 ({num_matches} 场)...")

    prompt = build_phase2_prompt(match_blocks, phase1_results)
    print(f"  [Claude Prompt] {len(prompt):,} 字符")

    sys_prompt = (
        "<role>你是Claude终审审计者。你必须重新审计原始抓包、Sharp分级、外部市场冲突、CRS/TTG路径。</role>\n"
        "<instruction>你的输出是审计意见，不是最终程序锁定结果。严格JSON数组，禁止markdown。</instruction>"
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
# 输出包装
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

    raw_argmax = max({"home": hp, "draw": dp, "away": ap}, key={"home": hp, "draw": dp, "away": ap}.get)
    probability_was_corrected = raw_argmax != expected_dir

    if probability_was_corrected:
        pcts = {"home": hp, "draw": dp, "away": ap}
        cur_max = pcts[raw_argmax]
        pcts[expected_dir] = cur_max + 3
        total = sum(pcts.values())
        hp = round(pcts["home"] / total * 100, 1)
        dp = round(pcts["draw"] / total * 100, 1)
        ap = round(pcts["away"] / total * 100, 1)

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
    out["final_confidence"] = conf
    out["prediction_confidence"] = conf
    out["dir_confidence"] = out.get("dir_confidence", conf)

    out["home_win_pct"] = hp
    out["draw_pct"] = dp
    out["away_win_pct"] = ap

    out["raw_probability_argmax"] = out.get("raw_probability_argmax", raw_argmax)
    out["probability_was_corrected"] = out.get("probability_was_corrected", probability_was_corrected)

    return out


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
    return str(r.get("predicted_score", r.get("ai_score", "-")))


def _ai_reason_summary(r: Dict, empty_text: str) -> str:
    if not isinstance(r, dict) or not r:
        return empty_text

    return _safe_str(
        r.get("reason", r.get("arbitration_reason", r.get("audit_result", ""))),
        2500,
    )


def _ai_response_from_result(r: Dict) -> Dict:
    if not isinstance(r, dict):
        return {}

    return {
        "predicted_score": r.get("predicted_score", r.get("ai_score", "")),
        "predicted_direction": r.get("predicted_direction", r.get("final_direction", "")),
        "confidence": r.get("confidence", r.get("ai_confidence", 55)),
        "top3": r.get("top3", []),
        "reason": r.get("reason", r.get("arbitration_reason", r.get("audit_result", ""))),
        "sharp_grade": r.get("sharp_grade", r.get("sharp_grade_final", "")),
        "local_market_trap": r.get("local_market_trap", False),
    }


def assemble_final_prediction(
    match: Dict,
    engine_result: Dict,
    stats: Dict,
    phase1_results: Dict[str, Dict[int, Dict]],
    claude_result: Dict,
    observation_signals: List[str],
    ensemble_signals: Dict,
    crs_analysis: Dict,
    ttg_analysis: Dict,
    sharp_eval: Dict,
    external_market: Dict,
    external_audit: Dict,
    trap_report: Dict,
    quant_audit: Dict,
    exp_goals: float,
    idx: int,
    ai_provider: str,
) -> Dict:
    p1_gpt = phase1_results.get("gpt", {}).get(idx, {})
    p1_grok = phase1_results.get("grok", {}).get(idx, {})
    p1_gemini = phase1_results.get("gemini", {}).get(idx, {})

    ai_responses = {
        "gpt": _ai_response_from_result(p1_gpt),
        "grok": _ai_response_from_result(p1_grok),
        "gemini": _ai_response_from_result(p1_gemini),
        "claude": _ai_response_from_result(claude_result),
    }

    lock_result = decision_lock_chain(
        match_obj=match,
        engine_result=engine_result,
        trap_report=trap_report,
        crs_analysis=crs_analysis,
        ttg_analysis=ttg_analysis,
        ai_responses=ai_responses,
        external_audit=external_audit,
        exp_goals=exp_goals,
    )

    confidence_base = _f(lock_result.get("dir_confidence", 55), 55)

    confidence = confidence_base

    if lock_result.get("dir_gap", 0) >= 12:
        confidence += 6
    elif lock_result.get("dir_gap", 0) < 6:
        confidence -= 6

    if sharp_eval.get("sharp_level") in ["S3", "S4"]:
        confidence += 5
    elif sharp_eval.get("sharp_level") == "SD":
        confidence -= 5

    if external_audit.get("risk_tag") == "LOCAL_MARKET_TRAP":
        confidence -= 4

    confidence -= trap_report.get("confidence_penalty", 0)

    coverage = _phase1_coverage_for_match(phase1_results, idx)

    if coverage["valid_count"] <= 1:
        confidence = min(confidence, 62)
    elif coverage["valid_count"] == 2:
        confidence = min(confidence, 72)

    confidence = int(max(30, min(95, round(confidence))))

    lock_result["confidence"] = confidence

    cr = _enforce_direction_consistency(lock_result)

    predicted_score = cr.get("predicted_score", "1-1")
    predicted_label = cr.get("predicted_label", predicted_score)
    final_direction = cr.get("final_direction", "draw")
    result_cn = cr.get("result", "平局")
    is_others = bool(cr.get("is_score_others", False))

    h_score, a_score = _parse_score(predicted_score)
    goal_count = h_score + a_score if h_score is not None else None

    risk = "低" if confidence >= 75 else ("中" if confidence >= 55 else "高")

    final_odds = _get_score_odds(match, predicted_score, final_direction, is_others)

    crs_prob = crs_analysis.get("implied_probs", {}).get(predicted_score, 0)
    if ENABLE_LLM_VALUE_BET:
        ev_data = calculate_value_bet(crs_prob, final_odds)
        ev_pct, kelly_pct, is_value = ev_data["ev"], ev_data["kelly"], ev_data["is_value"]
        value_reason = "基于CRS矩阵去水概率计算"
    else:
        ev_pct, kelly_pct, is_value = 0.0, 0.0, False
        value_reason = "v19.3 默认不使用LLM主观比分概率计算正式EV/Kelly"

    goal_bucket, goal_label = _normalize_goal_range_for_ui(cr.get("goal_range"), predicted_score)

    obs_codes = _extract_observation_codes(observation_signals)
    obs_labels = _extract_observation_labels(observation_signals)
    obs_objects = _extract_observation_objects(observation_signals)

    claude_reason = (
        claude_result.get("arbitration_reason")
        or claude_result.get("reason")
        or claude_result.get("audit_result")
        or ""
    )

    accepted_obs = claude_result.get("accepted_observations", [])
    rejected_obs = claude_result.get("rejected_observations", [])
    doubts = claude_result.get("doubts", [])

    top3 = [{"score": sc, "prob": round(score_pts, 2)} for sc, score_pts in cr.get("top_score_candidates", [])[:3]]
    if not top3:
        top3 = [{"score": predicted_score, "prob": 0}]

    return {
        "predicted_score": predicted_score,
        "predicted_label": predicted_label,
        "predicted_direction": final_direction,
        "result": result_cn,
        "display_direction": result_cn,
        "final_direction": final_direction,
        "is_score_others": is_others,

        "decision_title": "vMAX 19.3 决策剖析",
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

        "home_win_pct": cr.get("home_win_pct"),
        "draw_pct": cr.get("draw_pct"),
        "away_win_pct": cr.get("away_win_pct"),

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
        "dir_confidence": cr.get("dir_confidence", confidence),
        "dir_gap": cr.get("dir_gap", 0),
        "risk_level": risk,

        "ai_call_status": dict(AI_CALL_STATUS),
        "gpt_status": AI_CALL_STATUS.get("gpt", ""),
        "grok_status": AI_CALL_STATUS.get("grok", ""),
        "gemini_status": AI_CALL_STATUS.get("gemini", ""),
        "claude_status": AI_CALL_STATUS.get("claude", ""),

        "ai_provider": ai_provider,
        "claude_score": _ai_score_summary(claude_result),
        "claude_analysis": claude_reason[:3500],
        "arbitration_reason": claude_reason,
        "audit_result": claude_result.get("audit_result", ""),
        "agreement_pattern": claude_result.get("agreement_pattern", "Claude终审审计"),

        "phase1_coverage": coverage,
        "analysis_coverage": claude_result.get("analysis_coverage", coverage),
        "coverage_ok": coverage["coverage_ok"],
        "coverage_full": coverage["coverage_full"],

        "adopted_analysts": claude_result.get("adopted_analysts", []),
        "rejected_analysts": claude_result.get("rejected_analysts", []),
        "top3": top3,
        "top_score_candidates": cr.get("top_score_candidates", []),

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
        "data_quality": claude_result.get("data_quality", {}),
        "ai_validation_errors": claude_result.get("ai_validation_errors", []),

        "bayesian_prior": cr.get("bayesian_prior", {}),
        "bayesian_evidences": cr.get("bayesian_evidences", []),
        "posterior_after_lock": cr.get("posterior_after_lock", {}),
        "override_triggered": cr.get("override_triggered", False),
        "override_reason": cr.get("override_reason", ""),
        "raw_probability_argmax": cr.get("raw_probability_argmax", ""),
        "probability_was_corrected": cr.get("probability_was_corrected", False),

        "traps_detected": obs_labels,
        "trap_codes": obs_codes,
        "trap_items": obs_objects,
        "observation_items": obs_objects,
        "trap_count": len(observation_signals),
        "trap_facts": observation_signals,
        "observation_signals": observation_signals,
        "trap_matrix_title": "观察信号矩阵",
        "trap_matrix_subtitle": "抓包观察 + Sharp分级 + 外部市场审计",

        "trap_report": trap_report,
        "trap_details": [{"trap": t.get("trap"), "desc": t.get("description")} for t in trap_report.get("traps_detected", [])],
        "trap_severity": trap_report.get("total_severity", 0),

        "sharp_detected": sharp_eval.get("sharp_detected", False),
        "sharp_dir": sharp_eval.get("sharp_dir"),
        "sharp_level": sharp_eval.get("sharp_level"),
        "sharp_trust": sharp_eval.get("sharp_trust"),
        "sharp_reason": sharp_eval.get("sharp_reason", []),

        "external_market": external_market,
        "external_audit": external_audit,
        "external_market_available": external_audit.get("available", False),
        "external_market_direction": external_audit.get("external_dir"),
        "local_market_trap": external_audit.get("risk_tag") == "LOCAL_MARKET_TRAP",
        "market_risk_tag": external_audit.get("risk_tag"),

        "score_odds": final_odds,
        "crs_score_prob": round(crs_prob, 2),
        "suggested_kelly": kelly_pct,
        "edge_vs_market": ev_pct,
        "is_value": is_value,
        "value_reason": value_reason,

        "raw_smart_signals": stats.get("smart_signals", []) if stats else [],
        "smart_signals": observation_signals,
        "smart_money_signal": " | ".join([str(s) for s in observation_signals[:10]]),

        "xG_home": round(_f(engine_result.get("bookmaker_implied_home_xg", 1.3)), 2),
        "xG_away": round(_f(engine_result.get("bookmaker_implied_away_xg", 0.9)), 2),
        "expected_total_goals": round(exp_goals, 2),
        "over_2_5": engine_result.get("over_25", 50) if engine_result else 50,
        "btts": engine_result.get("btts", 45) if engine_result else 45,
        "bookmaker_implied_home_xg": engine_result.get("bookmaker_implied_home_xg", "?"),
        "bookmaker_implied_away_xg": engine_result.get("bookmaker_implied_away_xg", "?"),

        "crs_shape": crs_analysis.get("shape_verdict", "unknown"),
        "crs_moments": crs_analysis.get("moments", {}),
        "crs_margin": crs_analysis.get("margin", 0.0),
        "crs_coverage": crs_analysis.get("coverage", 0.0),
        "crs_implied_probs": crs_analysis.get("implied_probs", {}),
        "crs_direction_probs": crs_analysis.get("direction_probs", {}),
        "ttg_analysis": ttg_analysis,
        "quant_audit": quant_audit,

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
# 后处理保护
# ====================================================================

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


# ====================================================================
# Top4 推荐
# ====================================================================

def select_top4(preds):
    def _score(x):
        p = x.get("prediction", {}) or {}

        confidence = _f(p.get("confidence", 0))
        dir_gap = _f(p.get("dir_gap", 0))
        crs_prob = _f(p.get("crs_score_prob", 0))
        risk_penalty = 0

        if p.get("risk_level") == "高":
            risk_penalty += 8

        if p.get("ai_validation_errors"):
            risk_penalty += 5

        if p.get("ai_abstained"):
            risk_penalty += len(p.get("ai_abstained", [])) * 2

        if not p.get("coverage_ok", True):
            risk_penalty += 8

        if p.get("market_risk_tag") == "LOCAL_MARKET_TRAP":
            risk_penalty += 3

        sharp_level = p.get("sharp_level")
        sharp_bonus = 0
        if sharp_level == "S4":
            sharp_bonus += 8
        elif sharp_level == "S3":
            sharp_bonus += 5
        elif sharp_level == "SD":
            sharp_bonus -= 5

        return confidence + dir_gap * 0.35 + crs_prob * 0.15 + sharp_bonus - risk_penalty

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


def _run_coro_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        def _run_in_thread(c):
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(c)
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_in_thread, coro)
            return future.result()

    return asyncio.run(coro)


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
        if not isinstance(smart_signals, list):
            smart_signals = [str(smart_signals)]

        crs_analysis = analyze_crs_matrix(m)
        ttg_analysis = analyze_ttg(m)
        ensemble_signals = collect_ensemble_signals(sp)

        match_analyses.append({
            "raw_match": raw_m,
            "match": m,
            "engine": eng,
            "stats": sp,
            "experience": exp_result,
            "smart_signals": smart_signals,
            "exp_goals": exp_goals,
            "crs_analysis": crs_analysis,
            "ttg_analysis": ttg_analysis,
            "ensemble_signals": ensemble_signals,
            "external_fetch": [],
        })

    if ENABLE_EXTERNAL_FETCH and match_analyses:
        try:
            print("\n  [External Fetch] 检查 external_urls / odds_urls ...")
            match_analyses = _run_coro_sync(enrich_external_contexts(match_analyses))
        except Exception as e:
            logger.warning(f"外部市场抓取失败: {e}")

    match_blocks = []

    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        eng = ma["engine"]
        sp = ma["stats"]
        smart_signals = ma["smart_signals"]
        exp_goals = ma["exp_goals"]
        crs_analysis = ma["crs_analysis"]
        ttg_analysis = ma["ttg_analysis"]

        external_market = _extract_external_market_from_match(
            m,
            ma.get("external_fetch", []),
        )

        external_audit = compute_external_market_audit(m, external_market)

        sharp_eval = classify_sharp_signal(
            match_obj=m,
            smart_signals=smart_signals,
            crs_analysis=crs_analysis,
            ttg_analysis=ttg_analysis,
            external_audit=external_audit,
        )

        trap_report = detect_all_traps(
            match_obj=m,
            engine_result=eng,
            smart_signals=smart_signals,
            exp_goals=exp_goals,
            crs_analysis=crs_analysis,
            ttg_analysis=ttg_analysis,
            sharp_eval=sharp_eval,
            external_audit=external_audit,
        )

        quant_audit = build_quant_audit(
            match_obj=m,
            crs_analysis=crs_analysis,
            ttg_analysis=ttg_analysis,
            external_audit=external_audit,
            sharp_eval=sharp_eval,
            trap_report=trap_report,
        )

        observation_signals = build_observation_signals(
            match_obj=m,
            engine_result=eng,
            crs_analysis=crs_analysis,
            ttg_analysis=ttg_analysis,
            sharp_eval=sharp_eval,
            external_audit=external_audit,
            trap_report=trap_report,
            exp_goals=exp_goals,
        )

        block = format_match_block(
            idx=i + 1,
            match=m,
            engine_result=eng,
            crs_analysis=crs_analysis,
            ttg_analysis=ttg_analysis,
            sharp_eval=sharp_eval,
            external_market=external_market,
            external_audit=external_audit,
            trap_report=trap_report,
            quant_audit=quant_audit,
            observation_signals=observation_signals,
            ensemble_signals=ma["ensemble_signals"],
            smart_signals=smart_signals,
        )

        ma["external_market"] = external_market
        ma["external_audit"] = external_audit
        ma["sharp_eval"] = sharp_eval
        ma["trap_report"] = trap_report
        ma["quant_audit"] = quant_audit
        ma["observation_signals"] = observation_signals

        match_blocks.append(block)

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
            phase1_results, claude_results, ai_provider = _run_coro_sync(_run_full_ai())
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

        mg = assemble_final_prediction(
            match=m,
            engine_result=ma["engine"],
            stats=ma["stats"],
            phase1_results=phase1_results,
            claude_result=cr,
            observation_signals=ma["observation_signals"],
            ensemble_signals=ma["ensemble_signals"],
            crs_analysis=ma["crs_analysis"],
            ttg_analysis=ma["ttg_analysis"],
            sharp_eval=ma["sharp_eval"],
            external_market=ma["external_market"],
            external_audit=ma["external_audit"],
            trap_report=ma["trap_report"],
            quant_audit=ma["quant_audit"],
            exp_goals=ma["exp_goals"],
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

        root_fields = [
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

            "sharp_detected",
            "sharp_dir",
            "sharp_level",
            "sharp_trust",

            "external_market_available",
            "external_market_direction",
            "local_market_trap",
            "market_risk_tag",

            "crs_shape",
            "crs_moments",
            "ttg_analysis",
            "quant_audit",
        ]

        for k in root_fields:
            combined[k] = mg.get(k)

        combined["engine_version"] = ENGINE_VERSION
        combined["decision_title"] = "vMAX 19.3 决策剖析"
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

        sharp_tag = f" [Sharp {mg.get('sharp_level')}:{mg.get('sharp_dir')} {mg.get('sharp_trust')}]" if mg.get("sharp_detected") else ""
        market_tag = f" [{mg.get('market_risk_tag')}]" if mg.get("market_risk_tag") else ""
        trap_tag = f" [陷阱{ma['trap_report'].get('trap_count', 0)}]" if ma.get("trap_report") else ""
        cov = mg.get("phase1_coverage", {})
        cov_tag = f" [覆盖{cov.get('valid_count', 0)}/3]"

        print(
            f"  [{idx}] {m.get('home_team', m.get('home', '?'))} vs "
            f"{m.get('away_team', m.get('guest', '?'))} => "
            f"{mg['result']} ({mg['predicted_score']}) | "
            f"CF: {mg['confidence']}% | "
            f"{mg.get('scenario', 'normal')}"
            f"{cov_tag}{sharp_tag}{market_tag}{trap_tag}"
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
    print("   Phase 0: 原始抓包标准化")
    print("   Phase 1: 外部市场比对 + CRS矩阵 + TTG主模态 + Sharp分级")
    print("   Phase 2: GPT/Grok/Gemini 三家独立分析")
    print("   Phase 3: Claude 终审审计")
    print("   Phase 4: 程序决策锁定链，方向→区间→比分")
    print("   规则: Sharp是体彩核心变量，但必须先分级；Claude不直接锁死最终比分")
    print("   EV/Kelly: 默认不使用 LLM 主观比分概率")