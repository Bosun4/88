# ====================================================================
# 🚀 vMAX 19.5 — AI-First Evidence Packet 四AI终审版
# --------------------------------------------------------------------
# 核心原则:
#   ✅ 本地不再当最终裁判，不再用矩阵强行改方向/比分
#   ✅ 本地只负责: 原始抓包标准化 + 市场证据包 + 风险提示 + JSON闭环校验
#   ✅ GPT / Grok / Gemini 三家独立初审，全部看完整 raw_packet + evidence_packet
#   ✅ Claude 接收完整 raw_packet + evidence_packet + 三家初审结论做最终审计
#   ✅ Claude 不按票数裁决，必须重新审计市场结构
#   ✅ 本地不再用 T标签/D标签直接修正方向，只把它们作为 risk_notes 提供给 AI
#   ✅ 最终采用 Claude 的 predicted_score / final_direction / top3
#   ✅ 本地只修复 JSON、方向-比分一致性、字段缺失，不篡改 AI 最终比分
#   ✅ 保留 EV/Kelly，但只作为展示字段，不参与改比分
#   ✅ 兼容旧字段: win/same/lose, sp_home/sp_draw/sp_away, give_ball/rq/handicap
# ====================================================================

import json
import os
import re
import time
import math
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

try:
    import aiohttp
except Exception:
    aiohttp = None

# ====================================================================
# 日志与外部模块兼容
# ====================================================================

try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

try:
    from config import *
except Exception as e:
    logger.warning(f"config 导入异常: {e}")

try:
    from models import EnsemblePredictor
except Exception as e:
    logger.warning(f"models.EnsemblePredictor 导入异常: {e}")

    class EnsemblePredictor:
        def predict(self, m, ctx=None):
            return {}

try:
    from odds_engine import predict_match
except Exception as e:
    logger.warning(f"odds_engine.predict_match 导入异常: {e}")

    def predict_match(m):
        return {}

try:
    from league_intel import build_league_intelligence
except Exception as e:
    logger.warning(f"league_intel.build_league_intelligence 导入异常: {e}")

    def build_league_intelligence(m):
        return {}, {}, {}, {}

try:
    from experience_rules import ExperienceEngine
except Exception as e:
    logger.warning(f"experience_rules.ExperienceEngine 导入异常: {e}")

    class ExperienceEngine:
        def analyze(self, m):
            return {}

try:
    ensemble = EnsemblePredictor()
except Exception as e:
    logger.warning(f"EnsemblePredictor 初始化失败: {e}")
    ensemble = None

try:
    exp_engine = ExperienceEngine()
except Exception as e:
    logger.warning(f"ExperienceEngine 初始化失败: {e}")
    exp_engine = None

# ====================================================================
# 版本常量
# ====================================================================

ENGINE_VERSION = "vMAX 19.5"
ENGINE_ARCHITECTURE = "AI-First Evidence Packet + GPT/Grok/Gemini 初审 + Claude 最终审计 + Local Validator"

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

CUP_KEYWORDS = [
    "杯", "淘汰", "决赛", "半决赛", "四分之一",
    "欧冠", "欧联", "国王杯", "足总杯", "联赛杯",
    "解放者杯", "南球杯", "Cup", "cup",
]

CRS_FULL_MAP = {
    "1-0": "w10", "2-0": "w20", "2-1": "w21", "3-0": "w30", "3-1": "w31",
    "3-2": "w32", "4-0": "w40", "4-1": "w41", "4-2": "w42", "5-0": "w50",
    "5-1": "w51", "5-2": "w52",
    "0-0": "s00", "1-1": "s11", "2-2": "s22", "3-3": "s33",
    "0-1": "l01", "0-2": "l02", "1-2": "l12", "0-3": "l03", "1-3": "l13",
    "2-3": "l23", "0-4": "l04", "1-4": "l14", "2-4": "l24", "0-5": "l05",
    "1-5": "l15", "2-5": "l25",
}

HFTF_MAP = {
    "ss": "主/主", "sp": "主/平", "sf": "主/负",
    "ps": "平/主", "pp": "平/平", "pf": "平/负",
    "fs": "负/主", "fp": "负/平", "ff": "负/负",
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

LOCKED_OUTPUT_FIELDS = [
    "predicted_score",
    "predicted_label",
    "result",
    "display_direction",
    "final_direction",
    "top3",
    "home_win_pct",
    "draw_pct",
    "away_win_pct",
    "confidence",
    "risk_level",
]

# ====================================================================
# 通用工具
# ====================================================================

def _f(v, default=0.0):
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "" or s.lower() in ("none", "nan", "null", "-", "n/a"):
            return default
        s = s.replace("%", "")
        return float(s)
    except Exception:
        return default

def _i(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, _f(v, 0.0)))

def _safe_str(v, default=""):
    if v is None:
        return default
    return str(v)

def _round_dict(d: Dict[str, float], n: int = 3) -> Dict[str, float]:
    return {k: round(_f(v), n) for k, v in (d or {}).items()}

def _normalize_prob_dict(d: Dict[Any, float], floor: float = 0.0) -> Dict[Any, float]:
    out = {}
    for k, v in (d or {}).items():
        fv = max(floor, _f(v, 0.0))
        out[k] = fv
    s = sum(out.values())
    if s <= 0:
        n = len(out) or 1
        return {k: 1.0 / n for k in out}
    return {k: v / s for k, v in out.items()}

def _deep_find_value(obj, aliases, skip_keys=None):
    skip_keys = set(skip_keys or [])
    aliases_low = {str(a).lower() for a in aliases}

    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in aliases_low:
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

def _normalize_score_text(s: Any) -> str:
    return str(s).strip().replace(" ", "").replace("：", "-").replace(":", "-")

def _parse_score(s: Any) -> Tuple[Optional[int], Optional[int]]:
    try:
        s_str = _normalize_score_text(s)
        if not s_str:
            return None, None

        if "胜" in s_str and "其他" in s_str:
            return 9, 0
        if "平" in s_str and "其他" in s_str:
            return 9, 9
        if "负" in s_str and "其他" in s_str:
            return 0, 9

        if s_str in ["主胜", "客胜", "平局", "胜", "平", "负", "home", "draw", "away"]:
            return None, None

        p = s_str.split("-")
        if len(p) != 2:
            return None, None

        h = int(float(p[0]))
        a = int(float(p[1]))

        if h < 0 or a < 0:
            return None, None

        return h, a
    except Exception:
        return None, None

def _score_direction(score_str: Any) -> Optional[str]:
    ss = str(score_str)
    if "胜其他" in ss or ss == "9-0":
        return "home"
    if "平其他" in ss or ss == "9-9":
        return "draw"
    if "负其他" in ss or ss == "0-9":
        return "away"

    h, a = _parse_score(score_str)
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
    return {"home": "主胜", "draw": "平局", "away": "客胜"}.get(direction, "平局")

def _dir_from_cn(v: Any) -> Optional[str]:
    s = str(v).strip()
    if s in ("home", "主胜", "胜"):
        return "home"
    if s in ("draw", "平局", "平"):
        return "draw"
    if s in ("away", "客胜", "负"):
        return "away"
    return None

def _score_label(score: str, direction: Optional[str] = None) -> str:
    d = direction or _score_direction(score) or "draw"
    if str(score) in ("9-0", "胜其他") or "胜其他" in str(score):
        return "胜其他"
    if str(score) in ("9-9", "平其他") or "平其他" in str(score):
        return "平其他"
    if str(score) in ("0-9", "负其他") or "负其他" in str(score):
        return "负其他"
    return _normalize_score_text(score)

def _json_safe(obj: Any, max_len: int = 200000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    if len(s) > max_len:
        return s[:max_len] + "...<TRUNCATED>"
    return s

# ====================================================================
# 数据标准化
# ====================================================================

def normalize_match(raw_m: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(raw_m or {})

    nested_keys = [
        "v2_odds_dict", "odds_dict", "odds", "v2",
        "odds_v2", "packet", "raw_odds", "data", "detail",
    ]

    for nk in nested_keys:
        if isinstance(m.get(nk), dict):
            tmp = dict(m[nk])
            tmp.update(m)
            m = tmp

    home = (
        m.get("home_team") or m.get("home") or m.get("host") or
        m.get("team_home") or m.get("homeName") or m.get("home_name") or "Home"
    )
    away = (
        m.get("away_team") or m.get("guest") or m.get("away") or
        m.get("team_away") or m.get("awayName") or m.get("away_name") or "Away"
    )

    m["home_team"] = home
    m["away_team"] = away
    m["home"] = home
    m["guest"] = away

    skip = {
        "vote", "change", "points", "information",
        "prediction", "stats", "smart_signals",
    }

    sp_home = m.get("sp_home")
    if sp_home is None:
        sp_home = _deep_find_value(m, ["win", "odds_win", "spf_sp3", "sp3", "胜"], skip)

    sp_draw = m.get("sp_draw")
    if sp_draw is None:
        sp_draw = _deep_find_value(m, ["draw", "same", "odds_draw", "spf_sp1", "sp1", "平"], skip)

    sp_away = m.get("sp_away")
    if sp_away is None:
        sp_away = _deep_find_value(m, ["lose", "away_win", "odds_lose", "spf_sp0", "sp0", "负"], skip)

    if sp_home is not None:
        m["sp_home"] = sp_home
        m["win"] = sp_home
    if sp_draw is not None:
        m["sp_draw"] = sp_draw
        m["same"] = sp_draw
    if sp_away is not None:
        m["sp_away"] = sp_away
        m["lose"] = sp_away

    if "give_ball" not in m:
        m["give_ball"] = (
            m.get("handicap") or m.get("rq") or m.get("let_ball") or
            m.get("asian_handicap") or "0"
        )

    if not isinstance(m.get("change"), dict):
        ch = {}
        for src_key, dst_key in [
            ("change_win", "win"), ("cw", "win"),
            ("change_same", "same"), ("cs", "same"),
            ("change_draw", "same"),
            ("change_lose", "lose"), ("cl", "lose"),
            ("change_away", "lose"),
        ]:
            if src_key in m:
                ch[dst_key] = m.get(src_key)
        m["change"] = ch

    return m

# ====================================================================
# 盘口、公平概率、CRS、TTG
# ====================================================================

def fair_probs_from_1x2(
    sp_h: float,
    sp_d: float,
    sp_a: float,
    method: str = "power",
) -> Dict[str, Any]:
    odds = {"home": _f(sp_h), "draw": _f(sp_d), "away": _f(sp_a)}

    if any(v <= 1.01 for v in odds.values()):
        return {
            "method": "fallback",
            "fair_probs": {"home": 33.3, "draw": 33.3, "away": 33.4},
            "raw_implied": {"home": 33.3, "draw": 33.3, "away": 33.4},
            "overround": 0.0,
            "warning": "invalid_1x2_odds",
        }

    q = {k: 1.0 / v for k, v in odds.items()}
    overround_sum = sum(q.values())
    overround = overround_sum - 1.0
    raw_pct = {k: round(v * 100, 3) for k, v in q.items()}

    if method == "multiplicative":
        p = _normalize_prob_dict(q)

    elif method == "additive":
        cut = overround / 3.0
        add = {k: max(0.001, v - cut) for k, v in q.items()}
        p = _normalize_prob_dict(add)

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
        p = {name: val ** k for name, val in q.items()}
        p = _normalize_prob_dict(p)
        method = "power"

    return {
        "method": method,
        "fair_probs": {k: round(v * 100, 3) for k, v in p.items()},
        "raw_implied": raw_pct,
        "overround": round(overround, 5),
    }

def fair_probs_from_ttg(match_obj: Dict[str, Any], method: str = "power") -> Dict[int, float]:
    raw = {}
    for g in range(8):
        odd = _f(match_obj.get(f"a{g}", 0))
        if odd > 1.01:
            raw[g] = 1.0 / odd

    if len(raw) < 3:
        return {}

    if method == "multiplicative":
        return _normalize_prob_dict(raw)

    lo, hi = 0.01, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        sm = sum(v ** mid for v in raw.values())
        if sm > 1.0:
            lo = mid
        else:
            hi = mid

    k = (lo + hi) / 2.0
    p = {g: v ** k for g, v in raw.items()}
    return _normalize_prob_dict(p)

def _strong_side_from_1x2(sp_h: float, sp_a: float) -> str:
    if sp_h <= 1.01 or sp_a <= 1.01:
        return "home"
    return "home" if sp_h < sp_a else "away"

def _infer_theoretical_handicap_depth(sp_h: float, sp_a: float) -> float:
    if sp_h <= 1.01 or sp_a <= 1.01:
        return 0.0

    ratio = max(sp_h, sp_a) / max(1.01, min(sp_h, sp_a))

    if ratio >= 8.0:
        return 2.75
    if ratio >= 5.5:
        return 2.25
    if ratio >= 4.0:
        return 1.75
    if ratio >= 3.0:
        return 1.25
    if ratio >= 2.2:
        return 0.75
    if ratio >= 1.6:
        return 0.25
    if ratio >= 1.15:
        return 0.25
    return 0.0

def _infer_theoretical_handicap_signed(sp_h: float, sp_a: float) -> float:
    depth = _infer_theoretical_handicap_depth(sp_h, sp_a)
    if depth <= 0:
        return 0.0
    strong = _strong_side_from_1x2(sp_h, sp_a)
    return -depth if strong == "home" else depth

def _parse_actual_handicap_signed(match_obj: Dict) -> float:
    """
    统一主队坐标:
      主让 = 负
      主受让 = 正
      客让 = 正
      客受让 = 负
    """
    raw = (
        match_obj.get("give_ball")
        if match_obj.get("give_ball") is not None
        else match_obj.get("handicap", match_obj.get("rq", match_obj.get("let_ball", "0")))
    )

    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))
    strong_side = _strong_side_from_1x2(sp_h, sp_a)

    s0 = str(raw).strip()
    if not s0:
        return 0.0

    s = s0.replace("（", "(").replace("）", ")").replace("球", "").replace(" ", "")
    nums = re.findall(r"[-+]?\d+\.?\d*", s)
    val = abs(_f(nums[0], 0.0)) if nums else 0.0

    if "/" in s:
        parts = re.findall(r"[-+]?\d+\.?\d*", s)
        if len(parts) >= 2:
            val = (abs(_f(parts[0])) + abs(_f(parts[1]))) / 2.0

    if val <= 0:
        return 0.0

    if "主受让" in s or "主受" in s:
        return val
    if "客受让" in s or "客受" in s:
        return -val
    if "主让" in s:
        return -val
    if "客让" in s:
        return val

    if "受让" in s or "受" in s:
        return val
    if "让" in s:
        return -val if strong_side == "home" else val

    if s.startswith("-") or ("(" in s and "-" in s):
        return -val
    if s.startswith("+") or ("(" in s and "+" in s):
        return val

    return -val if strong_side == "home" else val

def _handicap_diff_for_strong_side(actual_signed: float, theoretical_signed: float, strong_side: str) -> float:
    """
    正数 = 强方实际更深
    负数 = 强方实际更浅
    """
    if strong_side == "home":
        return theoretical_signed - actual_signed
    if strong_side == "away":
        return actual_signed - theoretical_signed
    return 0.0

def get_market_odds_for_score(match_obj: Dict[str, Any], score: str) -> float:
    score = _normalize_score_text(score)
    key = CRS_FULL_MAP.get(score)

    if key:
        return _f(match_obj.get(key, 0))

    if score in ("胜其他", "9-0"):
        return _f(match_obj.get("crs_win", 0))
    if score in ("平其他", "9-9"):
        return _f(match_obj.get("crs_same", 0))
    if score in ("负其他", "0-9"):
        return _f(match_obj.get("crs_lose", 0))

    return 0.0

def _crs_low_rank_info(match_obj: Dict[str, Any]) -> Dict[str, Any]:
    odds_list = []
    for sc, key in CRS_FULL_MAP.items():
        odd = _f(match_obj.get(key, 0))
        if odd > 1.05:
            odds_list.append((sc, odd, _score_direction(sc)))

    odds_list.sort(key=lambda x: x[1])

    rank = {}
    for idx, (sc, odd, d) in enumerate(odds_list, 1):
        rank[sc] = {"rank": idx, "odds": odd, "direction": d}

    draw_scores = ["0-0", "1-1", "2-2", "3-3"]
    draw_available = [(sc, rank[sc]["odds"], rank[sc]["rank"]) for sc in draw_scores if sc in rank]
    draw_available.sort(key=lambda x: x[1])

    min_draw = draw_available[0] if draw_available else None

    return {
        "all_sorted": odds_list,
        "rank": rank,
        "draw_available": draw_available,
        "draw_low_rank": min_draw[2] if min_draw else 999,
        "draw_low_score": min_draw[0] if min_draw else "",
        "draw_low_odds": min_draw[1] if min_draw else 999.0,
        "low_scores": [(sc, odd) for sc, odd, _ in odds_list[:10]],
    }

def _has_low_draw_crs(match_obj: Dict[str, Any], rank_cutoff: int = 5, odds_cutoff: float = 8.5) -> bool:
    info = _crs_low_rank_info(match_obj)
    return info["draw_low_rank"] <= rank_cutoff or info["draw_low_odds"] <= odds_cutoff

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
            extras[key] = {"odds": odds, "scores": scores_set}

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

def compute_crs_moments(probs: Dict[str, float]) -> Dict[str, float]:
    regular = {}

    for sc, p in probs.items():
        try:
            h, a = sc.split("-")
            h, a = int(h), int(a)
            if h > 8 or a > 8:
                continue
            regular[(h, a)] = p
        except Exception:
            continue

    if not regular:
        return {}

    total = sum(regular.values())
    if total <= 0:
        return {}

    reg = {k: v / total for k, v in regular.items()}

    e_h = sum(h * p for (h, a), p in reg.items())
    e_a = sum(a * p for (h, a), p in reg.items())
    var_h = sum((h - e_h) ** 2 * p for (h, a), p in reg.items())
    var_a = sum((a - e_a) ** 2 * p for (h, a), p in reg.items())
    std_h = math.sqrt(var_h) if var_h > 0 else 0.01
    std_a = math.sqrt(var_a) if var_a > 0 else 0.01
    cov = sum((h - e_h) * (a - e_a) * p for (h, a), p in reg.items())
    corr = cov / (std_h * std_a) if std_h * std_a > 0 else 0.0

    return {
        "lambda_h": round(e_h, 3),
        "lambda_a": round(e_a, 3),
        "lambda_total": round(e_h + e_a, 3),
        "var_h": round(var_h, 3),
        "var_a": round(var_a, 3),
        "corr": round(corr, 3),
    }

def classify_crs_shape(moments: Dict[str, float]) -> Tuple[str, List[str]]:
    if not moments:
        return "unknown", ["CRS数据不足"]

    lh = moments.get("lambda_h", 1.3)
    la = moments.get("lambda_a", 1.2)
    lt = moments.get("lambda_total", 2.5)
    corr = moments.get("corr", 0.0)
    var_h = moments.get("var_h", 1.0)
    var_a = moments.get("var_a", 1.0)

    notes = []
    shape = "normal"

    if lt >= 3.15 and corr >= 0.12:
        shape = "shootout"
        notes.append(f"互射/大比分结构: λ总{lt:.2f}, corr={corr:+.2f}")
    elif lt <= 2.15 and var_h < 1.2 and var_a < 1.2:
        shape = "grinder"
        notes.append(f"低比分磨局: λ总{lt:.2f}, 方差{var_h:.2f}/{var_a:.2f}")
    elif lh - la >= 1.15:
        shape = "lopsided_home"
        notes.append(f"主队单边: λ主{lh:.2f} vs λ客{la:.2f}")
    elif la - lh >= 1.15:
        shape = "lopsided_away"
        notes.append(f"客队单边: λ客{la:.2f} vs λ主{lh:.2f}")
    elif abs(lh - la) < 0.45:
        shape = "balanced"
        notes.append(f"均势结构: λ主{lh:.2f} vs λ客{la:.2f}")

    if corr < -0.15:
        notes.append(f"负相关单边倾向: corr={corr:+.2f}")

    return shape, notes

def compute_direction_from_crs(probs: Dict[str, float]) -> Dict[str, float]:
    out = {"home": 0.0, "draw": 0.0, "away": 0.0}

    for sc, p in probs.items():
        d = _score_direction(sc)
        if d in out:
            out[d] += p

    total = sum(out.values())
    if total <= 0:
        return {"home": 33.3, "draw": 33.3, "away": 33.4}

    return {k: round(v / total * 100, 2) for k, v in out.items()}

def analyze_crs_matrix(match_obj: Dict) -> Dict[str, Any]:
    probs, margin, coverage = crs_implied_probabilities(match_obj)

    if not probs:
        return {
            "implied_probs": {},
            "margin": 0.0,
            "coverage": 0.0,
            "moments": {},
            "shape": "unknown",
            "shape_notes": ["CRS数据缺失或不足"],
            "direction_probs": {"home": 33.3, "draw": 33.3, "away": 33.4},
            "top_scores": [],
            "low_rank_info": _crs_low_rank_info(match_obj),
        }

    moments = compute_crs_moments(probs)
    shape, shape_notes = classify_crs_shape(moments)
    direction_probs = compute_direction_from_crs(probs)
    sorted_scores = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    return {
        "implied_probs": {k: round(v, 3) for k, v in probs.items()},
        "margin": margin,
        "coverage": coverage,
        "moments": moments,
        "shape": shape,
        "shape_notes": shape_notes,
        "direction_probs": direction_probs,
        "top_scores": [(sc, round(p, 3)) for sc, p in sorted_scores[:16]],
        "low_rank_info": _crs_low_rank_info(match_obj),
    }

# ====================================================================
# 资金、风险、情报提示
# ====================================================================

def detect_sharp_direction(smart_signals: List) -> Dict[str, Any]:
    detected = False
    sharp_dir = None
    details = []

    for s in smart_signals or []:
        s_str = str(s)

        if "Sharp" in s_str or "sharp" in s_str or "聪明钱" in s_str or "专业资金" in s_str:
            detected = True
            details.append(s_str[:180])

            if re.search(r"(主胜|主队|走主|→\s*主|流向\s*主|资金\s*主|Sharp主|聪明钱主|向主)", s_str):
                sharp_dir = "home"
                break
            elif re.search(r"(客胜|客队|走客|→\s*客|流向\s*客|资金\s*客|Sharp客|聪明钱客|向客)", s_str):
                sharp_dir = "away"
                break
            elif re.search(r"(平局|平赔|走平|→\s*平|流向\s*平|资金\s*平|Sharp平|进平局|聪明钱平)", s_str):
                sharp_dir = "draw"
                break

    return {"detected": detected, "sharp_dir": sharp_dir, "details": details}

def detect_steam_direction(smart_signals: List) -> Dict[str, Any]:
    steam_dir = None
    steam_type = None
    details = []

    for s in smart_signals or []:
        s_str = str(s)

        if "Steam" not in s_str and "steam" not in s_str and "异动" not in s_str:
            continue

        details.append(s_str[:180])
        is_reverse = "反向" in s_str or "未跟" in s_str or "不跟" in s_str or "散户未跟" in s_str

        if re.search(r"(主胜.*Steam|Steam.*主胜|主胜.*降水|主.*Steam|主胜.*异动|主赔降)", s_str):
            steam_dir = "home"
            steam_type = "reverse" if is_reverse else "normal"
            break
        elif re.search(r"(客胜.*Steam|Steam.*客胜|客胜.*降水|客.*Steam|客胜.*异动|客赔降)", s_str):
            steam_dir = "away"
            steam_type = "reverse" if is_reverse else "normal"
            break
        elif re.search(r"(平.*Steam|Steam.*平|平赔.*异动|平赔降)", s_str):
            steam_dir = "draw"
            steam_type = "reverse" if is_reverse else "normal"
            break

    return {"steam_dir": steam_dir, "steam_type": steam_type, "details": details}

def _extract_form_record(text: str) -> Tuple[int, int, int]:
    if not text:
        return 0, 0, 0

    patterns = [
        r"近\s*\d+\s*[主客场]*\s*(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
        r"(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
        r"近\s*\d+[:：]\s*(\d+)W(\d+)D(\d+)L",
    ]

    for pat in patterns:
        m = re.search(pat, str(text), flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1)), int(m.group(2)), int(m.group(3))
            except Exception:
                pass

    return 0, 0, 0

def _extract_avg_goals(text: str) -> Tuple[float, float]:
    if not text:
        return 0.0, 0.0

    text = str(text)
    gf = re.search(r"场均进球[^0-9]*(\d+\.?\d*)", text)
    ga = re.search(r"场均失球[^0-9]*(\d+\.?\d*)", text)

    return (
        float(gf.group(1)) if gf else 0.0,
        float(ga.group(1)) if ga else 0.0,
    )

def _fundamental_strength(match_obj: Dict, side: str) -> Dict[str, Any]:
    info_src = match_obj.get("points", {})
    if not isinstance(info_src, dict):
        info_src = {}

    key = "home_strength" if side == "home" else "guest_strength"
    txt = str(info_src.get(key, ""))

    w, d, l = _extract_form_record(txt)
    total = w + d + l
    win_rate = (w / total) if total > 0 else 0.5

    gf, ga = _extract_avg_goals(txt)

    score = 0.0

    if total > 0:
        score += (win_rate - 0.5) * 80

    if gf > 0:
        score += (gf - 1.3) * 20

    if ga > 0:
        score -= (ga - 1.3) * 20

    return {
        "wins": w,
        "draws": d,
        "losses": l,
        "total": total,
        "win_rate": round(win_rate, 3),
        "goals_for": round(gf, 2),
        "goals_against": round(ga, 2),
        "strength_score": round(max(-100, min(100, score)), 1),
    }

def build_risk_notes(
    match_obj: Dict[str, Any],
    engine_result: Dict[str, Any],
    stats: Dict[str, Any],
    fair: Dict[str, float],
    crs_analysis: Dict[str, Any],
    exp_goals: float,
) -> List[Dict[str, Any]]:
    """
    注意:
    这里不做本地裁决，只生成提示。
    risk_notes 只能给 AI 参考，不直接改方向和比分。
    """
    notes = []

    change = match_obj.get("change", {}) or {}
    cw = _f(change.get("win", 0))
    cs = _f(change.get("same", change.get("draw", 0)))
    cl = _f(change.get("lose", 0))

    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_d = _f(match_obj.get("sp_draw", match_obj.get("same", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))

    actual_hc = _parse_actual_handicap_signed(match_obj)
    theory_hc = _infer_theoretical_handicap_signed(sp_h, sp_a)
    strong_side = _strong_side_from_1x2(sp_h, sp_a)
    weak_side = "away" if strong_side == "home" else "home"
    strong_diff = _handicap_diff_for_strong_side(actual_hc, theory_hc, strong_side)

    crs_low = crs_analysis.get("low_rank_info", {})
    low_draw = crs_low.get("draw_low_rank", 999) <= 5 or crs_low.get("draw_low_odds", 999) <= 8.5
    fair_gap = abs(fair.get("home", 33.3) - fair.get("away", 33.3))
    top_fair = max(fair.get("home", 33.3), fair.get("draw", 33.3), fair.get("away", 33.4))

    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0)) if isinstance(engine_result, dict) else 0
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0)) if isinstance(engine_result, dict) else 0
    total_xg = hxg + axg if hxg > 0 and axg > 0 else 0.0

    if cs < -0.04:
        if low_draw and (fair.get("draw", 0) >= 20 or fair_gap <= 16) and (total_xg <= 2.85 or total_xg <= 0):
            notes.append({
                "code": "R01_REAL_DRAW_SIGNAL",
                "level": "high",
                "side_hint": "draw",
                "note": f"平赔独降{cs:+.2f}，且CRS平局低位 {crs_low.get('draw_low_score')}@{crs_low.get('draw_low_odds')} rank={crs_low.get('draw_low_rank')}，需要真平保护。",
            })
        else:
            notes.append({
                "code": "R02_FAKE_DRAW_OR_DRAW_BAIT",
                "level": "medium",
                "side_hint": strong_side,
                "note": f"平赔独降{cs:+.2f}，但真平证据不足，需判断是假平诱盘还是资金提前进平。",
            })

    if strong_diff <= -0.50:
        notes.append({
            "code": "R03_STRONG_SIDE_SHALLOW_HANDICAP",
            "level": "high",
            "side_hint": "draw_or_weak",
            "note": f"强方{strong_side}理论盘{theory_hc:.2f}，实际盘{actual_hc:.2f}，强方深浅差{strong_diff:+.2f}，强方偏浅，防平/防弱方。",
        })

    if strong_diff >= 0.50:
        notes.append({
            "code": "R04_STRONG_SIDE_DEEP_HANDICAP",
            "level": "medium",
            "side_hint": strong_side,
            "note": f"强方{strong_side}理论盘{theory_hc:.2f}，实际盘{actual_hc:.2f}，强方深浅差{strong_diff:+.2f}，强方偏深，可能支持强方方向但需防造热。",
        })

    if top_fair < 43 or fair_gap < 6:
        notes.append({
            "code": "R05_BALANCED_DIRECTION",
            "level": "medium",
            "side_hint": "draw",
            "note": f"1X2公平概率均势，最高方向{top_fair:.1f}%，主客差{fair_gap:.1f}%，方向不宜强锁。",
        })

    vote = match_obj.get("vote", {}) or {}
    vh = int(_f(vote.get("win", 33), 33))
    vd = int(_f(vote.get("same", 33), 33))
    va = int(_f(vote.get("lose", 33), 33))
    max_vote = max(vh, vd, va)

    if max_vote >= 60:
        hot_dir = "home" if vh == max_vote else ("draw" if vd == max_vote else "away")
        notes.append({
            "code": "R06_PUBLIC_HEAT",
            "level": "medium",
            "side_hint": hot_dir,
            "note": f"散户热度集中 {hot_dir}={max_vote}%，需要检查赔率是否同向降水，避免机械反指。",
        })

    a0 = _f(match_obj.get("a0", 0))
    a1 = _f(match_obj.get("a1", 0))
    a2 = _f(match_obj.get("a2", 0))
    a3 = _f(match_obj.get("a3", 0))
    a4 = _f(match_obj.get("a4", 0))
    a5 = _f(match_obj.get("a5", 0))
    a6 = _f(match_obj.get("a6", 0))
    a7 = _f(match_obj.get("a7", 0))

    if 0 < a4 <= 4.85:
        notes.append({
            "code": "R07_TTG_4_GOAL_ANCHOR",
            "level": "medium",
            "side_hint": "score_total_4",
            "note": f"4球赔率低位 a4={a4:.2f}，需要检查 2-2 / 3-1 / 1-3。",
        })

    if 0 < a5 <= 8.20:
        notes.append({
            "code": "R08_TTG_5_GOAL_ANCHOR",
            "level": "medium",
            "side_hint": "score_total_5",
            "note": f"5球赔率低位 a5={a5:.2f}，需要检查 3-2 / 2-3 / 4-1 / 1-4。",
        })

    if 0 < a7 <= 18:
        notes.append({
            "code": "R09_TTG_7PLUS_GOAL_ANCHOR",
            "level": "high",
            "side_hint": "score_total_7plus",
            "note": f"7球赔率极低 a7={a7:.2f}，需要检查 5-2 / 2-5 / 4-3 / 3-4 / 胜其他/负其他。",
        })

    if exp_goals >= 2.8 and ((0 < a0 < 8.0) + (0 < a1 < 4.5) + (0 < a2 < 3.0) >= 2):
        notes.append({
            "code": "R10_SMALL_SCORE_BAIT",
            "level": "medium",
            "side_hint": "avoid_too_small",
            "note": f"低进球赔率被压，但期望进球{exp_goals:.2f}偏高，警惕小比分诱导。",
        })

    if exp_goals <= 2.3 and ((0 < a5 < 10) + (0 < a6 < 16) + (0 < a7 < 30) >= 2):
        notes.append({
            "code": "R11_BIG_SCORE_BAIT",
            "level": "medium",
            "side_hint": "avoid_too_large",
            "note": f"高进球赔率被压，但期望进球{exp_goals:.2f}偏低，警惕大比分诱导。",
        })

    league = str(match_obj.get("league", match_obj.get("cup", "")))
    if any(kw in league for kw in CUP_KEYWORDS) and max(fair.get("home", 0), fair.get("away", 0)) >= 55:
        notes.append({
            "code": "R12_CUP_FAVORITE_RISK",
            "level": "medium",
            "side_hint": "draw_or_low_margin",
            "note": "杯赛/淘汰赛强方热度较高，需要检查平局、弱方小胜、低净胜球，不可机械反热门。",
        })

    sharp = detect_sharp_direction(stats.get("smart_signals", []) if isinstance(stats, dict) else [])
    steam = detect_steam_direction(stats.get("smart_signals", []) if isinstance(stats, dict) else [])

    if sharp.get("detected"):
        notes.append({
            "code": "R13_SHARP_MONEY",
            "level": "medium",
            "side_hint": sharp.get("sharp_dir"),
            "note": f"检测到聪明钱/专业资金信号，方向={sharp.get('sharp_dir')}，细节={sharp.get('details')[:2]}",
        })

    if steam.get("steam_dir"):
        notes.append({
            "code": "R14_STEAM_MOVE",
            "level": "medium",
            "side_hint": steam.get("steam_dir"),
            "note": f"检测到Steam/异动，方向={steam.get('steam_dir')}，类型={steam.get('steam_type')}，细节={steam.get('details')[:2]}",
        })

    return notes

# ====================================================================
# Evidence Packet
# ====================================================================

def estimate_expected_goals(match_obj: Dict[str, Any], engine_result: Dict[str, Any], stats: Dict[str, Any]) -> float:
    for src in [engine_result, stats]:
        if not isinstance(src, dict):
            continue

        for k in [
            "expected_total_goals", "exp_goals", "total_goals",
            "expected_goals", "lambda_total", "total_xg",
        ]:
            v = src.get(k)
            fv = _f(v, 0)
            if fv > 0.5:
                return round(fv, 3)

    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0)) if isinstance(engine_result, dict) else 0
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0)) if isinstance(engine_result, dict) else 0

    if hxg > 0 and axg > 0:
        return round(hxg + axg, 3)

    try:
        gp = []
        for gi in range(8):
            v = _f(match_obj.get(f"a{gi}", 0))
            if v > 1:
                gp.append((gi, 1.0 / v))

        if gp:
            tp = sum(p for _, p in gp)
            exp = sum(g * (p / tp) for g, p in gp)
            if 0.8 <= exp <= 6.5:
                return round(exp, 3)
    except Exception:
        pass

    return 2.5

def build_ttg_evidence(match_obj: Dict[str, Any]) -> Dict[str, Any]:
    raw = {}
    implied = {}
    compressed = []
    anchors = []

    for g in range(8):
        odd = _f(match_obj.get(f"a{g}", 0))
        raw[g] = odd

        if odd > 1:
            implied[g] = round(100.0 / odd, 3)
            std = STANDARD_GOAL_ODDS.get(g, 50)
            ratio = std / odd if odd > 0 else 0
            if ratio >= 1.35:
                compressed.append({
                    "goals": g,
                    "odds": odd,
                    "std": std,
                    "compression_ratio": round(ratio, 2),
                })

    probs = fair_probs_from_ttg(match_obj, method="power")
    fair_pct = {str(k): round(v * 100, 3) for k, v in probs.items()} if probs else {}

    a3 = _f(match_obj.get("a3", 0))
    a4 = _f(match_obj.get("a4", 0))
    a5 = _f(match_obj.get("a5", 0))
    a6 = _f(match_obj.get("a6", 0))
    a7 = _f(match_obj.get("a7", 0))

    if 0 < a3 <= 4.15:
        anchors.append({"goals": 3, "odds": a3, "score_candidates": ["2-1", "1-2", "3-0", "0-3", "1-1"]})
    if 0 < a4 <= 4.85:
        anchors.append({"goals": 4, "odds": a4, "score_candidates": ["2-2", "3-1", "1-3", "4-0", "0-4"]})
    if 0 < a5 <= 8.20:
        anchors.append({"goals": 5, "odds": a5, "score_candidates": ["3-2", "2-3", "4-1", "1-4", "5-0", "0-5"]})
    if 0 < a6 <= 16.50:
        anchors.append({"goals": 6, "odds": a6, "score_candidates": ["3-3", "4-2", "2-4", "5-1", "1-5"]})
    if 0 < a7 <= 18.00:
        anchors.append({"goals": "7+", "odds": a7, "score_candidates": ["5-2", "2-5", "4-3", "3-4", "胜其他", "负其他"]})

    return {
        "raw_a0_a7": raw,
        "raw_implied_pct": implied,
        "fair_goal_probs_pct": fair_pct,
        "compressed_goals": compressed,
        "anchors": anchors,
    }

def build_hftf_evidence(match_obj: Dict[str, Any]) -> Dict[str, Any]:
    items = []
    for k, label in HFTF_MAP.items():
        odd = _f(match_obj.get(k, 0))
        if odd > 1:
            items.append({"code": k, "label": label, "odds": odd, "implied_pct": round(100 / odd, 3)})

    items.sort(key=lambda x: x["odds"])
    return {"available": items, "low_rank": items[:8]}

def build_information_evidence(match_obj: Dict[str, Any]) -> Dict[str, Any]:
    info = match_obj.get("information", {})
    points = match_obj.get("points", {})

    out = {
        "injuries": {},
        "bad_news": {},
        "points_text": {},
        "fundamental_strength": {
            "home": _fundamental_strength(match_obj, "home"),
            "away": _fundamental_strength(match_obj, "away"),
        },
    }

    if isinstance(info, dict):
        for k in ["home_injury", "guest_injury", "away_injury"]:
            if info.get(k):
                out["injuries"][k] = str(info.get(k))[:800]
        for k in ["home_bad_news", "guest_bad_news", "away_bad_news"]:
            if info.get(k):
                out["bad_news"][k] = str(info.get(k))[:800]

    if isinstance(points, dict):
        for k in ["home_strength", "guest_strength", "match_points", "h2h", "history", "weather"]:
            if points.get(k):
                out["points_text"][k] = str(points.get(k))[:1000]

    return out

def build_evidence_packet(
    match_obj: Dict[str, Any],
    engine_result: Dict[str, Any],
    stats: Dict[str, Any],
    league_info: Dict[str, Any] = None,
    experience: Dict[str, Any] = None,
) -> Dict[str, Any]:
    match_obj = normalize_match(match_obj)
    engine_result = engine_result or {}
    stats = stats or {}

    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_d = _f(match_obj.get("sp_draw", match_obj.get("same", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))

    fair_pack = fair_probs_from_1x2(sp_h, sp_d, sp_a, method="power")
    fair = fair_pack.get("fair_probs", {"home": 33.3, "draw": 33.3, "away": 33.4})

    actual_hc = _parse_actual_handicap_signed(match_obj)
    theory_hc = _infer_theoretical_handicap_signed(sp_h, sp_a)
    strong_side = _strong_side_from_1x2(sp_h, sp_a)
    strong_diff = _handicap_diff_for_strong_side(actual_hc, theory_hc, strong_side)

    crs = analyze_crs_matrix(match_obj)
    ttg = build_ttg_evidence(match_obj)
    hftf = build_hftf_evidence(match_obj)
    exp_goals = estimate_expected_goals(match_obj, engine_result, stats)
    info_ev = build_information_evidence(match_obj)

    sharp = detect_sharp_direction(stats.get("smart_signals", []))
    steam = detect_steam_direction(stats.get("smart_signals", []))

    vote = match_obj.get("vote", {}) if isinstance(match_obj.get("vote", {}), dict) else {}
    change = match_obj.get("change", {}) if isinstance(match_obj.get("change", {}), dict) else {}

    risk_notes = build_risk_notes(
        match_obj=match_obj,
        engine_result=engine_result,
        stats=stats,
        fair=fair,
        crs_analysis=crs,
        exp_goals=exp_goals,
    )

    data_quality = {
        "has_1x2": sp_h > 1 and sp_d > 1 and sp_a > 1,
        "has_crs": crs.get("coverage", 0) >= 0.30,
        "crs_coverage": crs.get("coverage", 0),
        "has_ttg": len([g for g in range(8) if _f(match_obj.get(f'a{g}', 0)) > 1]) >= 3,
        "has_hftf": len(hftf.get("available", [])) >= 3,
        "has_engine_xg": _f(engine_result.get("bookmaker_implied_home_xg", 0)) > 0 and _f(engine_result.get("bookmaker_implied_away_xg", 0)) > 0,
        "has_public_vote": bool(vote),
        "has_odds_change": bool(change),
        "has_smart_signals": bool(stats.get("smart_signals", [])),
        "has_information": bool(match_obj.get("information")) or bool(match_obj.get("points")),
    }

    packet = {
        "engine_version": ENGINE_VERSION,
        "principle": "local_evidence_only_not_final_decision",
        "match_identity": {
            "home": match_obj.get("home_team"),
            "away": match_obj.get("away_team"),
            "league": match_obj.get("league", match_obj.get("cup", "")),
            "match_num": match_obj.get("match_num", ""),
            "id": match_obj.get("id", ""),
            "kickoff_time": match_obj.get("match_time", match_obj.get("time", "")),
        },
        "one_x_two": {
            "raw_odds": {
                "home": sp_h,
                "draw": sp_d,
                "away": sp_a,
            },
            "fair_pack": fair_pack,
            "fair_dir": max(fair, key=fair.get) if fair else None,
        },
        "handicap": {
            "raw": match_obj.get("give_ball", "0"),
            "coordinate_rule": "主让为负；主受让为正；客让为正；客受让为负",
            "actual_signed": round(actual_hc, 3),
            "theoretical_signed": round(theory_hc, 3),
            "strong_side_from_1x2": strong_side,
            "strong_side_depth_diff": round(strong_diff, 3),
            "interpretation": "positive=strong_side_deeper, negative=strong_side_shallower",
        },
        "crs": crs,
        "total_goals": ttg,
        "half_full": hftf,
        "market_movement": {
            "change": change,
            "meaning": "负数=赔率下降/降水；正数=赔率上升",
        },
        "public_money": {
            "vote": vote,
            "hot_side": _hot_side_from_vote(vote),
        },
        "smart_money": {
            "sharp": sharp,
            "steam": steam,
            "raw_signals": stats.get("smart_signals", [])[:20],
        },
        "xg_and_engine": {
            "expected_total_goals": exp_goals,
            "bookmaker_implied_home_xg": engine_result.get("bookmaker_implied_home_xg", None),
            "bookmaker_implied_away_xg": engine_result.get("bookmaker_implied_away_xg", None),
            "over_25": engine_result.get("over_25", None),
            "btts": engine_result.get("btts", None),
            "raw_engine_summary": _shrink_dict(engine_result, 60),
        },
        "information": info_ev,
        "league_info": _shrink_dict(league_info or {}, 40),
        "experience": _shrink_dict(experience or {}, 40),
        "risk_notes": risk_notes,
        "data_quality": data_quality,
    }

    return packet

def _hot_side_from_vote(vote: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(vote, dict) or not vote:
        return {}

    vals = {
        "home": int(_f(vote.get("win", 33), 33)),
        "draw": int(_f(vote.get("same", vote.get("draw", 33)), 33)),
        "away": int(_f(vote.get("lose", 33), 33)),
    }

    side = max(vals, key=vals.get)
    return {"side": side, "pct": vals[side], "all": vals}

def _shrink_dict(d: Dict[str, Any], max_items: int = 50) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    out = {}
    for idx, (k, v) in enumerate(d.items()):
        if idx >= max_items:
            out["_truncated"] = True
            break
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict):
            out[k] = _shrink_dict(v, 20)
        elif isinstance(v, list):
            out[k] = v[:20]
        else:
            out[k] = str(v)[:500]
    return out

# ====================================================================
# EV / Kelly
# ====================================================================

def calculate_independent_ev(
    model_prob_pct: float,
    market_odds: float,
    market_implied_pct: Optional[float] = None,
) -> Dict[str, Any]:
    if market_odds <= 1.05 or model_prob_pct <= 0:
        return {"ev": 0.0, "kelly": 0.0, "is_value": False, "note": "invalid_odds_or_prob"}

    prob = model_prob_pct / 100.0
    ev = prob * market_odds - 1.0
    b = market_odds - 1.0
    q = 1.0 - prob
    kelly = ((b * prob) - q) / b if b > 0 else 0.0

    is_value = ev > 0.06
    if market_implied_pct is not None and abs(model_prob_pct - market_implied_pct) < 1.0:
        is_value = False

    return {
        "ev": round(ev * 100, 2),
        "kelly": round(max(0.0, kelly * 0.5) * 100, 2),
        "is_value": bool(is_value),
        "note": "independent_ai_prob_vs_market",
    }

# ====================================================================
# AI Prompt
# ====================================================================

def build_phase1_prompt(match_analyses: List[Dict[str, Any]]) -> str:
    p = ""
    p += "<role>\n"
    p += "你是中国体彩竞彩足球市场的 AI-First 比分审计员。你只做独立初审，不是最终裁判。\n"
    p += "你必须基于 raw_packet 与 evidence_packet 判断方向、比分、进球区间、风险路径。\n"
    p += "</role>\n\n"

    p += _common_ai_rules()

    p += "<phase1_task>\n"
    p += "请对每场比赛独立分析，输出你自己的 top3 比分、最终方向、置信度、风险说明。\n"
    p += "不要迎合本地 risk_notes；risk_notes 只是证据提醒，不是结论。\n"
    p += "必须说明你为什么选择 top1，以及你放弃了哪些路径，例如防平、防弱方、强方穿盘、大比分、小比分。\n"
    p += "</phase1_task>\n\n"

    p += _output_schema_text()

    p += "<matches>\n"
    for ma in match_analyses:
        idx = ma["index"]
        m = ma["match"]
        packet = ma["evidence_packet"]

        p += f"<match index=\"{idx}\">\n"
        p += "<raw_packet>\n"
        p += _json_safe(m, max_len=80000)
        p += "\n</raw_packet>\n"
        p += "<evidence_packet>\n"
        p += _json_safe(packet, max_len=120000)
        p += "\n</evidence_packet>\n"
        p += "</match>\n\n"

    p += "</matches>\n"
    return p

def build_claude_final_audit_prompt(
    match_analyses: List[Dict[str, Any]],
    phase1_results: Dict[str, Dict[int, Dict[str, Any]]],
) -> str:
    p = ""
    p += "<role>\n"
    p += "你是最终审计模型 Claude。你的结论将作为最终输出。\n"
    p += "你必须重新审计 raw_packet、evidence_packet、GPT/Grok/Gemini 三家初审，不得按票数机械裁决。\n"
    p += "</role>\n\n"

    p += _common_ai_rules()

    p += "<final_audit_rules>\n"
    p += "1. 三家一致时，仍要检查盘口深浅、CRS低赔、总进球锚点、赔率变动、散户热度、Sharp/Steam 是否支持。\n"
    p += "2. 三家分歧时，优先选择证据链闭合的一方，不选择文字最肯定的一方。\n"
    p += "3. 如果本地 risk_notes 出现 REAL_DRAW、SHALLOW_HANDICAP、BALANCED_DIRECTION、CUP_FAVORITE_RISK，你必须明确说明是否防平/防弱方。\n"
    p += "4. 你可以推翻三家初审，但必须写出市场结构反证。\n"
    p += "5. 最终 top3[0].score 必须等于 predicted_score，且方向必须与 final_direction 一致。\n"
    p += "</final_audit_rules>\n\n"

    p += _output_schema_text()

    p += "<matches_with_phase1>\n"

    for ma in match_analyses:
        idx = ma["index"]
        m = ma["match"]
        packet = ma["evidence_packet"]

        p += f"<match index=\"{idx}\">\n"
        p += "<raw_packet>\n"
        p += _json_safe(m, max_len=80000)
        p += "\n</raw_packet>\n"
        p += "<evidence_packet>\n"
        p += _json_safe(packet, max_len=120000)
        p += "\n</evidence_packet>\n"

        p += "<phase1_results>\n"
        for ai_name in ["gpt", "grok", "gemini"]:
            r = phase1_results.get(ai_name, {}).get(idx, {})
            p += f"<{ai_name}>\n"
            if r:
                p += _json_safe(r, max_len=50000)
            else:
                p += "弃权或无有效JSON"
            p += f"\n</{ai_name}>\n"
        p += "</phase1_results>\n"

        p += "</match>\n\n"

    p += "</matches_with_phase1>\n"
    return p

def _common_ai_rules() -> str:
    p = ""
    p += "<core_rules>\n"
    p += "1. 市场主轴优先级: 1X2公平概率 → 盘口深浅 → CRS比分矩阵 → a0-a7总进球 → 赔率变动 → 散户/Sharp/Steam → 情报。\n"
    p += "2. 不得编造不存在的伤停、天气、新闻、阵容。如果 packet 没有外部信息，就写 external_evidence=[]。\n"
    p += "3. 如果你具备联网能力，只能使用可验证的赛前公开信息；必须放入 external_evidence，并与 packet_evidence 分开。不能把猜测当事实。\n"
    p += "4. 0-1、0-2、0-3 是合法客胜比分，禁止误判无效。\n"
    p += "5. 1-1、0-0、2-2 在 CRS 低赔或平赔独降时必须认真检查，不能被热门方向压死。\n"
    p += "6. 4球低赔重点检查 2-2/3-1/1-3；5球低赔重点检查 3-2/2-3；7球低赔重点检查 5-2/2-5/4-3/3-4/其他比分。\n"
    p += "7. 强方浅盘时，不得直接锁强方穿盘比分；必须比较强方小胜、平局、弱方小胜。\n"
    p += "8. 强方深盘时，可以支持强方，但仍要检查是否散户过热、赔率诱导、CRS是否跟随。\n"
    p += "9. 杯赛/淘汰赛大热必须检查平局和弱方低比分，不可机械反热门。\n"
    p += "10. top3 必须是三个不同比分，prob 为 0-100 的相对概率，不要求总和等于100，但顺序必须从强到弱。\n"
    p += "11. predicted_score 必须等于 top3[0].score。\n"
    p += "12. final_direction 必须是 home/draw/away，且必须与 predicted_score 的方向一致。\n"
    p += "</core_rules>\n\n"

    p += "<professional_reasoning_framework>\n"
    p += "每场 reason 必须覆盖以下结构:\n"
    p += "A. 1X2去水后的真实强弱与表面赔率差异。\n"
    p += "B. 理论盘口 vs 实际盘口，判断强方深盘/浅盘/正常盘。\n"
    p += "C. CRS低赔排序、最低平局比分、CRS方向概率是否支持最终方向。\n"
    p += "D. a0-a7总进球锚点如何落到具体比分。\n"
    p += "E. 赔率变动、散户热度、Sharp/Steam 是否同向或背离。\n"
    p += "F. 最终选择 top1 的原因，以及拒绝另外两个主要路径的原因。\n"
    p += "</professional_reasoning_framework>\n\n"
    return p

def _output_schema_text() -> str:
    p = ""
    p += "<output_schema>\n"
    p += "严格输出 JSON 数组。禁止 markdown、禁止解释 JSON 外文本。\n"
    p += "每场对象必须包含以下字段:\n"
    p += "{\n"
    p += '  "match": 1,\n'
    p += '  "predicted_score": "2-1",\n'
    p += '  "final_direction": "home",\n'
    p += '  "top3": [\n'
    p += '    {"score": "2-1", "prob": 16.5, "path": "主胜小胜"},\n'
    p += '    {"score": "1-1", "prob": 12.0, "path": "防平"},\n'
    p += '    {"score": "3-1", "prob": 9.5, "path": "主胜扩大"}\n'
    p += "  ],\n"
    p += '  "direction_probs": {"home": 45, "draw": 28, "away": 27},\n'
    p += '  "ai_confidence": 0-100,\n'
    p += '  "risk_level": "low/medium/high",\n'
    p += '  "is_score_others": false,\n'
    p += '  "market_reading": "盘口/CRS/进球锚点的核心判断",\n'
    p += '  "key_evidence": ["证据1", "证据2"],\n'
    p += '  "rejected_paths": ["为什么不选平局", "为什么不选客胜"],\n'
    p += '  "external_evidence": [],\n'
    p += '  "reason": "完整审计说明"\n'
    p += "}\n"
    p += "</output_schema>\n\n"
    return p

# ====================================================================
# AI 调用配置
# ====================================================================

FALLBACK_URLS = [
    None,
    "https://www.api522.pro/v1",
    "https://api522.pro/v1",
    "https://api521.pro/v1",
    "http://69.63.213.33:666/v1",
]

GPT_DEFAULT_URL = "https://ai.newapi.life/v1"
GPT_DEFAULT_KEY = ""

GPT_KEY_ALIASES = ["GPT_API_KEY", "API_KEY"]
GPT_URL_ALIASES = ["GPT_API_URL"]
GROK_KEY_ALIASES = ["GROK_API_KEY"]
GROK_URL_ALIASES = ["GROK_API_URL"]
GEMINI_KEY_ALIASES = ["GEMINI_API_KEY"]
GEMINI_URL_ALIASES = ["GEMINI_API_URL"]
CLAUDE_KEY_ALIASES = ["CLAUDE_API_KEY"]
CLAUDE_URL_ALIASES = ["CLAUDE_API_URL"]

def get_clean_env_url(name, default=""):
    v = str(os.environ.get(name, globals().get(name, default))).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/?#=&%-]+)", v)
    return match.group(1) if match else v

def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")

def get_first_clean_env_key(names: List[str], default=""):
    for n in names:
        v = get_clean_env_key(n)
        if v:
            return v
    return default

def get_first_clean_env_url(names: List[str], default=""):
    for n in names:
        v = get_clean_env_url(n)
        if v:
            return v
    return default

def _mask_key(k: str) -> str:
    if not k:
        return ""
    if len(k) <= 8:
        return "***"
    return f"{k[:4]}...{k[-4:]}"

def debug_ai_config():
    cfgs = [
        ("GPT", GPT_URL_ALIASES, GPT_KEY_ALIASES),
        ("GROK", GROK_URL_ALIASES, GROK_KEY_ALIASES),
        ("GEMINI", GEMINI_URL_ALIASES, GEMINI_KEY_ALIASES),
        ("CLAUDE", CLAUDE_URL_ALIASES, CLAUDE_KEY_ALIASES),
    ]

    for name, url_aliases, key_aliases in cfgs:
        print(
            f"[AI CONFIG] {name}: "
            f"url={get_first_clean_env_url(url_aliases)} "
            f"key={_mask_key(get_first_clean_env_key(key_aliases))}"
        )

def _ai_profile(ai_name: str) -> Dict[str, Any]:
    profiles = {
        "gpt": {
            "system": (
                "你是足球衍生品定价、比分分布和赔率结构审计专家。"
                "你要从1X2公平概率、盘口深浅、CRS、总进球锚点、资金流中重构比分路径。"
                "只输出JSON数组。"
            ),
            "temperature": 0.18,
            "read_timeout": 520,
        },
        "grok": {
            "system": (
                "你是足球市场情绪、散户热度、聪明钱与反向资金识别专家。"
                "你重点检查Sharp/Steam、赔率变动、热门造热、浅盘陷阱，但不得编造外部信息。"
                "只输出JSON数组。"
            ),
            "temperature": 0.24,
            "read_timeout": 520,
        },
        "gemini": {
            "system": (
                "你是足球多市场共振和非线性结构识别专家。"
                "你重点检查欧赔、让球、CRS、a0-a7、半全场之间的定价裂缝。"
                "只输出JSON数组。"
            ),
            "temperature": 0.16,
            "read_timeout": 560,
        },
        "claude": {
            "system": (
                "你是最终审计模型。你必须重新审计原始抓包、证据包和三家初审。"
                "禁止按票数机械裁决。最终比分必须由市场结构证据闭环支持。"
                "只输出JSON数组。"
            ),
            "temperature": 0.12,
            "read_timeout": 680,
        },
    }
    return profiles.get(ai_name, profiles["gpt"])

async def async_call_one_ai_batch(
    session,
    prompt: str,
    ai_name: str,
    url_aliases: List[str],
    key_aliases: List[str],
    models_list: List[str],
    num_matches: int,
):
    key = get_first_clean_env_key(key_aliases)

    if not key and ai_name == "gpt":
        key = GPT_DEFAULT_KEY

    if not key:
        return ai_name, {}, "no_key"

    primary_url = get_first_clean_env_url(url_aliases, GPT_DEFAULT_URL if ai_name == "gpt" else "")

    if ai_name == "gpt":
        urls = [primary_url or GPT_DEFAULT_URL]
    else:
        backup = [u for u in FALLBACK_URLS if u and u != primary_url][:1]
        urls = [primary_url] + backup

    profile = _ai_profile(ai_name)

    connect_timeout = 25
    read_timeout = profile["read_timeout"]

    for model_name in models_list:
        for base_url in urls:
            if not base_url:
                continue

            is_gemini = "generateContent" in base_url
            url = base_url.rstrip("/")

            if not is_gemini and "chat/completions" not in url:
                url += "/chat/completions"

            headers = {"Content-Type": "application/json"}

            if is_gemini:
                headers["x-goog-api-key"] = key
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": profile["temperature"],
                    },
                    "systemInstruction": {
                        "parts": [{"text": profile["system"]}]
                    },
                }
            else:
                headers["Authorization"] = f"Bearer {key}"
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": profile["system"]},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": profile["temperature"],
                }

            gw = url.split("/v1")[0][:60]
            print(f"  [连接中] {ai_name.upper()} | {model_name[:42]} @ {gw}")

            t0 = time.time()
            connected = False

            try:
                timeout = aiohttp.ClientTimeout(
                    total=None,
                    connect=connect_timeout,
                    sock_connect=connect_timeout,
                    sock_read=read_timeout,
                )

                async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                    elapsed_connect = round(time.time() - t0, 1)

                    if r.status in (502, 504):
                        print(f"    HTTP {r.status} | {elapsed_connect}s → 换URL")
                        continue

                    if r.status == 400:
                        text = await r.text()
                        print(f"    HTTP 400 | {elapsed_connect}s → 换模型 | {text[:180]}")
                        break

                    if r.status == 401:
                        print(f"    HTTP 401 | key/url不正确")
                        return ai_name, {}, "unauthorized"

                    if r.status == 429:
                        print(f"    HTTP 429 | {elapsed_connect}s → 换URL")
                        await asyncio.sleep(1.2)
                        continue

                    if r.status != 200:
                        text = await r.text()
                        print(f"    HTTP {r.status} | {elapsed_connect}s → 换URL | {text[:120]}")
                        continue

                    connected = True
                    print(f"    已连上 {elapsed_connect}s | 等待模型输出...")

                    try:
                        data = await r.json(content_type=None)
                    except Exception:
                        text = await r.text()
                        _save_debug_dump(ai_name, {"raw": text}, "non_json")
                        print("    响应非JSON → 换模型")
                        break

                    elapsed = round(time.time() - t0, 1)
                    usage = data.get("usage", {})
                    req_tokens = usage.get("total_tokens", 0) or (
                        usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                    )

                    if not req_tokens:
                        um = data.get("usageMetadata", {})
                        req_tokens = um.get("totalTokenCount", 0)

                    if req_tokens:
                        print(f"    token={req_tokens:,} | {elapsed}s")

                    raw_text = _extract_response_text(data, is_gemini)

                    if not raw_text or len(raw_text) < 10:
                        print("    空数据 → 换模型")
                        _save_debug_dump(ai_name, data, "empty")
                        break

                    results = _parse_ai_json(raw_text, num_matches)

                    if len(results) > 0:
                        print(f"    {ai_name.upper()} 完成: {len(results)}/{num_matches} | {elapsed}s")
                        return ai_name, results, model_name

                    print("    解析0条 → 保存debug并换模型")
                    _save_debug_dump(ai_name, {"raw_text": raw_text, "data": data}, "parse0")
                    break

            except aiohttp.ClientConnectorError:
                print("    连接失败 → 换URL")
                continue

            except asyncio.TimeoutError:
                if not connected:
                    print("    连接超时 → 换URL")
                    continue
                print("    读取超时")
                return ai_name, {}, "read_timeout"

            except Exception as e:
                if not connected:
                    print(f"    {str(e)[:120]} → 换URL")
                    continue
                print(f"    调用异常: {str(e)[:160]}")
                return ai_name, {}, "error"

            await asyncio.sleep(0.2)

    return ai_name, {}, "all_failed"

def _extract_response_text(data, is_gemini=False):
    raw_text = ""

    try:
        if is_gemini:
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return raw_text

        if data.get("choices"):
            msg = data["choices"][0].get("message", {})

            if isinstance(msg, dict):
                content_val = msg.get("content", "")

                if isinstance(content_val, str) and content_val.strip():
                    return content_val.strip()

                if isinstance(content_val, list):
                    best = ""
                    for item in content_val:
                        if isinstance(item, dict):
                            if item.get("type") == "text" and item.get("text"):
                                t = str(item.get("text", "")).strip()
                                if len(t) > len(best):
                                    best = t
                            elif item.get("text"):
                                t = str(item.get("text", "")).strip()
                                if len(t) > len(best):
                                    best = t
                    if best:
                        return best

                for field in [
                    "text", "answer", "response", "output_text", "final_answer",
                    "output", "result", "completion", "message_content",
                    "assistant_content", "model_response",
                ]:
                    v = msg.get(field, "")
                    if isinstance(v, str) and v.strip():
                        return v.strip()

                skip = {
                    "reasoning_content", "thinking", "reasoning", "reasoning_text",
                    "thoughts", "thought_process", "internal_thinking",
                    "chain_of_thought", "cot", "deliberation", "analysis_process",
                }

                best_with_json = ""
                for k, v in msg.items():
                    if k in skip:
                        continue
                    if isinstance(v, str) and "[" in v and ("match" in v or "predicted_score" in v):
                        if len(v) > len(best_with_json):
                            best_with_json = v.strip()

                if best_with_json:
                    return best_with_json

        if data.get("output") and isinstance(data["output"], list):
            best = ""
            for out_item in data["output"]:
                if isinstance(out_item, dict):
                    if out_item.get("type") == "message":
                        for ct in out_item.get("content", []):
                            if isinstance(ct, dict):
                                t = ct.get("text") or ct.get("content") or ""
                                if isinstance(t, str) and len(t) > len(best):
                                    best = t.strip()
                    elif out_item.get("text"):
                        t = str(out_item.get("text", "")).strip()
                        if len(t) > len(best):
                            best = t
            if best:
                return best

        full_str = json.dumps(data, ensure_ascii=False)
        m = re.search(r'\[\s*\{\s*\\?"match\\?"', full_str)
        if m:
            start = m.start()
            depth = 0
            end = start

            for i in range(start, min(start + 500000, len(full_str))):
                ch = full_str[i]
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            if end > start:
                extracted = full_str[start:end]
                if '\\"' in extracted:
                    try:
                        extracted = json.loads('"' + extracted + '"')
                    except Exception:
                        extracted = extracted.replace('\\"', '"')
                return extracted

    except Exception as e:
        print(f"    响应解析异常: {str(e)[:120]}")

    return raw_text

def _parse_ai_json(raw_text, num_matches):
    clean = str(raw_text or "")
    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"```(?:json|JSON|python|PYTHON)?", "", clean).replace("```", "").strip()

    json_str = ""

    m_re = re.search(r'\[\s*\{\s*"match"', clean)
    if not m_re:
        m_re = re.search(r'\[\s*\{\s*\'match\'', clean)

    if m_re:
        start_idx = m_re.start()
        depth = 0
        in_str = False
        escape = False
        quote = None

        for i in range(start_idx, len(clean)):
            ch = clean[i]

            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_str = False
                continue

            if ch in ('"', "'"):
                in_str = True
                quote = ch
                continue

            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    json_str = clean[start_idx:i + 1]
                    break

    if not json_str:
        start = clean.find("[")
        end = clean.rfind("]") + 1
        if start != -1 and end > start:
            json_str = clean[start:end]

    if not json_str:
        obj_match = re.search(r'\{\s*"matches"\s*:', clean)
        if obj_match:
            start = obj_match.start()
            end = clean.rfind("}") + 1
            if end > start:
                json_str = clean[start:end]

    arr = []

    if json_str:
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and isinstance(parsed.get("matches"), list):
                arr = parsed["matches"]
            elif isinstance(parsed, list):
                arr = parsed
        except Exception:
            try:
                fixed = json_str.replace("'", '"')
                fixed = re.sub(r",\s*}", "}", fixed)
                fixed = re.sub(r",\s*]", "]", fixed)
                parsed = json.loads(fixed)
                if isinstance(parsed, dict) and isinstance(parsed.get("matches"), list):
                    arr = parsed["matches"]
                elif isinstance(parsed, list):
                    arr = parsed
            except Exception:
                try:
                    last_brace = json_str.rfind("}")
                    if last_brace != -1:
                        partial = json_str[:last_brace + 1] + "]"
                        arr = json.loads(partial)
                except Exception:
                    arr = []

    results = {}

    if isinstance(arr, list):
        for item in arr:
            if not isinstance(item, dict):
                continue

            mid = item.get("match", item.get("index", item.get("id")))
            try:
                mid = int(mid)
            except Exception:
                continue

            if mid < 1 or mid > max(num_matches, 9999):
                continue

            norm = _normalize_ai_item(item)
            if norm:
                results[mid] = norm

    return results

def _normalize_ai_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    top3 = item.get("top3", [])
    predicted_score = item.get("predicted_score", item.get("ai_score", item.get("score", "")))

    norm_top3 = []

    if isinstance(top3, list):
        for t in top3:
            if isinstance(t, dict):
                sc = _normalize_score_text(t.get("score", ""))
                prob = _clip(_f(t.get("prob", t.get("p", 0)), 0), 0, 100)
                path = str(t.get("path", t.get("reason", "")))[:200]
            elif isinstance(t, str):
                sc = _normalize_score_text(t)
                prob = 0.0
                path = ""
            else:
                continue

            if _parse_score(sc)[0] is not None:
                norm_top3.append({"score": sc, "prob": round(prob, 3), "path": path})

    if not predicted_score and norm_top3:
        predicted_score = norm_top3[0]["score"]

    predicted_score = _normalize_score_text(predicted_score)

    h, a = _parse_score(predicted_score)
    if h is None:
        if norm_top3:
            predicted_score = norm_top3[0]["score"]
            h, a = _parse_score(predicted_score)
        else:
            return None

    if not norm_top3:
        norm_top3 = [{"score": predicted_score, "prob": 0.0, "path": "ai_single_score"}]

    if norm_top3[0]["score"] != predicted_score:
        found = next((x for x in norm_top3 if x["score"] == predicted_score), None)
        if found:
            norm_top3 = [found] + [x for x in norm_top3 if x["score"] != predicted_score]
        else:
            norm_top3 = [{"score": predicted_score, "prob": 0.0, "path": "predicted_score_inserted"}] + norm_top3

    seen = set()
    unique_top3 = []
    for t in norm_top3:
        sc = t["score"]
        if sc in seen:
            continue
        seen.add(sc)
        unique_top3.append(t)
        if len(unique_top3) >= 3:
            break

    while len(unique_top3) < 3:
        direction = _score_direction(predicted_score) or "draw"
        fallback = _fallback_scores_for_direction(direction)
        for sc in fallback:
            if sc not in seen:
                unique_top3.append({"score": sc, "prob": 0.0, "path": "local_fill_missing_top3"})
                seen.add(sc)
                break
        else:
            break

    parsed_dir = _score_direction(predicted_score)
    final_direction = item.get("final_direction", item.get("direction", ""))
    final_direction = _dir_from_cn(final_direction) or final_direction

    if final_direction not in VALID_DIRS:
        final_direction = parsed_dir

    if parsed_dir and final_direction != parsed_dir:
        final_direction = parsed_dir

    direction_probs = item.get("direction_probs", {})
    direction_probs = _normalize_direction_probs(direction_probs, fallback_dir=final_direction)

    ai_confidence = int(_clip(_f(item.get("ai_confidence", item.get("confidence", 60)), 60), 0, 100))

    return {
        "match": item.get("match"),
        "predicted_score": predicted_score,
        "ai_score": predicted_score,
        "final_direction": final_direction or parsed_dir or "draw",
        "top3": unique_top3[:3],
        "direction_probs": direction_probs,
        "ai_confidence": ai_confidence,
        "risk_level": str(item.get("risk_level", "medium")),
        "is_score_others": bool(item.get("is_score_others", predicted_score in ALL_SCORE_OTHERS or "其他" in predicted_score)),
        "market_reading": str(item.get("market_reading", "")),
        "key_evidence": item.get("key_evidence", []),
        "rejected_paths": item.get("rejected_paths", []),
        "external_evidence": item.get("external_evidence", []),
        "reason": str(item.get("reason", item.get("analysis", ""))),
        "raw_ai_item": item,
    }

def _normalize_direction_probs(direction_probs: Any, fallback_dir: str = "draw") -> Dict[str, float]:
    if not isinstance(direction_probs, dict):
        base = {"home": 33.3, "draw": 33.3, "away": 33.4}
        if fallback_dir in base:
            base[fallback_dir] += 6
            other_sum = sum(v for k, v in base.items() if k != fallback_dir)
            remain = 100 - base[fallback_dir]
            for k in list(base.keys()):
                if k != fallback_dir:
                    base[k] = base[k] / other_sum * remain
        return {k: round(v, 2) for k, v in base.items()}

    vals = {
        "home": _f(direction_probs.get("home", direction_probs.get("主胜", 0))),
        "draw": _f(direction_probs.get("draw", direction_probs.get("平局", 0))),
        "away": _f(direction_probs.get("away", direction_probs.get("客胜", 0))),
    }

    s = sum(vals.values())

    if s <= 0:
        return _normalize_direction_probs({}, fallback_dir)

    if s <= 1.5:
        vals = {k: v * 100 for k, v in vals.items()}
        s = sum(vals.values())

    vals = {k: v / s * 100 for k, v in vals.items()}
    return {k: round(vals[k], 2) for k in ["home", "draw", "away"]}

def _fallback_scores_for_direction(direction: str) -> List[str]:
    if direction == "home":
        return ["1-0", "2-1", "2-0", "3-1", "3-0"]
    if direction == "away":
        return ["0-1", "1-2", "0-2", "1-3", "0-3"]
    return ["1-1", "0-0", "2-2", "3-3"]

def _save_debug_dump(ai_name, data, tag):
    try:
        os.makedirs("data/debug", exist_ok=True)
        dump_file = f"data/debug/{ai_name}_{tag}_{int(time.time())}.json"
        with open(dump_file, "w", encoding="utf-8") as df:
            json.dump(data, df, ensure_ascii=False, indent=2, default=str)
        print(f"    失败响应已保存: {dump_file}")
    except Exception:
        pass

async def run_ai_matrix_two_phase(match_analyses):
    if aiohttp is None:
        print("  [AI ERROR] aiohttp 未安装，无法调用AI")
        return {"gpt": {}, "grok": {}, "gemini": {}, "claude": {}}

    num = len(match_analyses)
    phase1_prompt = build_phase1_prompt(match_analyses)

    print(f"  [{ENGINE_VERSION} Phase1 Prompt] {len(phase1_prompt):,} 字符 → GPT/Grok/Gemini 并行初审...")

    phase1_configs = [
        ("grok", GROK_URL_ALIASES, GROK_KEY_ALIASES, ["熊猫-A-5-grok-4.2-fast-200w上下文"]),
        ("gpt", GPT_URL_ALIASES, GPT_KEY_ALIASES, ["gpt-5.4"]),
        ("gemini", GEMINI_URL_ALIASES, GEMINI_KEY_ALIASES, ["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking"]),
    ]

    all_results = {"gpt": {}, "grok": {}, "gemini": {}, "claude": {}}

    connector = aiohttp.TCPConnector(limit=8, use_dns_cache=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            async_call_one_ai_batch(session, phase1_prompt, n, u_alias, k_alias, models, num)
            for n, u_alias, k_alias, models in phase1_configs
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, tuple):
                all_results[res[0]] = res[1]
            else:
                print(f"  [Phase1 ERROR] {res}")

        ok1 = sum(1 for n in ["gpt", "grok", "gemini"] if all_results.get(n))
        print(f"  [Phase1完成] {ok1}/3 AI有数据")

        audit_prompt = build_claude_final_audit_prompt(match_analyses, all_results)
        print(f"  [{ENGINE_VERSION} Phase2 Claude Audit] {len(audit_prompt):,} 字符 → Claude终审...")

        claude_name, claude_result, claude_model = await async_call_one_ai_batch(
            session=session,
            prompt=audit_prompt,
            ai_name="claude",
            url_aliases=CLAUDE_URL_ALIASES,
            key_aliases=CLAUDE_KEY_ALIASES,
            models_list=["熊猫-69-满血openrouter-claude-opus-4.7-上下文1000k"],
            num_matches=num,
        )

        all_results["claude"] = claude_result or {}

    ok = sum(1 for v in all_results.values() if v)
    print(f"  [完成] {ok}/4 AI有数据 | 架构=AI-First + Claude终审")
    return all_results

# ====================================================================
# 最终选择与一致性校验
# ====================================================================

def choose_final_ai_result(
    idx: int,
    all_ai: Dict[str, Dict[int, Dict[str, Any]]],
    evidence_packet: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    claude_r = all_ai.get("claude", {}).get(idx, {})
    if _valid_ai_result(claude_r):
        return claude_r, "claude"

    candidates = []
    weights = {"gpt": 1.0, "grok": 1.0, "gemini": 1.0}

    for name in ["gpt", "grok", "gemini"]:
        r = all_ai.get(name, {}).get(idx, {})
        if _valid_ai_result(r):
            sc = r.get("predicted_score") or r.get("ai_score")
            d = _score_direction(sc)
            conf = _f(r.get("ai_confidence", 60), 60)
            candidates.append((name, r, sc, d, conf, weights.get(name, 1.0)))

    if candidates:
        score_weight = {}
        direction_weight = {}

        for name, r, sc, d, conf, w in candidates:
            score_weight[sc] = score_weight.get(sc, 0.0) + w * (0.8 + conf / 100.0)
            if d:
                direction_weight[d] = direction_weight.get(d, 0.0) + w * (0.8 + conf / 100.0)

        best_score = max(score_weight, key=score_weight.get)
        best_dir = _score_direction(best_score) or max(direction_weight, key=direction_weight.get)

        best_candidate = None
        best_score_val = -1

        for name, r, sc, d, conf, w in candidates:
            val = 0
            if sc == best_score:
                val += 10
            if d == best_dir:
                val += 4
            val += conf / 100.0
            if val > best_score_val:
                best_score_val = val
                best_candidate = (name, r)

        if best_candidate:
            r = dict(best_candidate[1])
            r["fallback_note"] = "claude_missing_used_phase1_consensus"
            return r, f"phase1_consensus_{best_candidate[0]}"

    fallback = local_evidence_fallback(evidence_packet)
    fallback["fallback_note"] = "all_ai_missing_used_local_evidence_fallback"
    return fallback, "local_evidence_fallback"

def _valid_ai_result(r: Dict[str, Any]) -> bool:
    if not isinstance(r, dict):
        return False
    sc = r.get("predicted_score") or r.get("ai_score")
    h, a = _parse_score(sc)
    return h is not None

def local_evidence_fallback(evidence_packet: Dict[str, Any]) -> Dict[str, Any]:
    """
    只有四AI全部失败时才用。
    这不是主策略，只是防崩溃兜底。
    """
    fair = evidence_packet.get("one_x_two", {}).get("fair_pack", {}).get("fair_probs", {})
    crs = evidence_packet.get("crs", {})
    crs_top = crs.get("top_scores", [])

    if crs_top:
        best_score = crs_top[0][0]
    else:
        d = max(fair, key=fair.get) if fair else "draw"
        best_score = {"home": "1-0", "draw": "1-1", "away": "0-1"}.get(d, "1-1")

    d = _score_direction(best_score) or "draw"

    top3 = [{"score": best_score, "prob": 0.0, "path": "local_fallback"}]
    for sc in _fallback_scores_for_direction(d):
        if sc != best_score:
            top3.append({"score": sc, "prob": 0.0, "path": "local_fallback_fill"})
        if len(top3) >= 3:
            break

    return {
        "predicted_score": best_score,
        "ai_score": best_score,
        "final_direction": d,
        "top3": top3,
        "direction_probs": _normalize_direction_probs(fair, fallback_dir=d),
        "ai_confidence": 35,
        "risk_level": "high",
        "is_score_others": best_score in ALL_SCORE_OTHERS or "其他" in best_score,
        "market_reading": "AI全部缺失，本地仅按CRS/1X2做兜底，不作为高置信推荐。",
        "key_evidence": ["AI全部失败，本地兜底"],
        "rejected_paths": [],
        "external_evidence": [],
        "reason": "AI全部失败，本地以最低CRS比分或1X2公平概率做兜底。",
    }

def enforce_final_consistency(mg: Dict[str, Any]) -> Dict[str, Any]:
    warnings = list(mg.get("validation_warnings", []))

    score = _normalize_score_text(mg.get("predicted_score", mg.get("ai_score", "1-1")))
    h, a = _parse_score(score)

    if h is None:
        top3 = mg.get("top3", [])
        if top3 and isinstance(top3, list):
            first = top3[0]
            if isinstance(first, dict):
                score = _normalize_score_text(first.get("score", "1-1"))
            elif isinstance(first, str):
                score = _normalize_score_text(first)
            h, a = _parse_score(score)

    if h is None:
        score = "1-1"
        h, a = 1, 1
        warnings.append("invalid_score_replaced_with_1-1")

    direction = _score_direction(score) or "draw"

    top3 = mg.get("top3", [])
    if not isinstance(top3, list):
        top3 = []

    norm_top3 = []

    for t in top3:
        if isinstance(t, dict):
            sc = _normalize_score_text(t.get("score", ""))
            prob = round(_clip(_f(t.get("prob", 0)), 0, 100), 3)
            path = str(t.get("path", ""))[:200]
        elif isinstance(t, str):
            sc = _normalize_score_text(t)
            prob = 0.0
            path = ""
        else:
            continue

        if _parse_score(sc)[0] is not None:
            norm_top3.append({"score": sc, "prob": prob, "path": path})

    if not norm_top3 or norm_top3[0]["score"] != score:
        norm_top3 = [{"score": score, "prob": 0.0, "path": "validator_inserted_top1"}] + [
            x for x in norm_top3 if x["score"] != score
        ]
        warnings.append("top3_top1_aligned_with_predicted_score")

    seen = set()
    unique = []

    for t in norm_top3:
        if t["score"] in seen:
            continue
        seen.add(t["score"])
        unique.append(t)
        if len(unique) >= 3:
            break

    for sc in _fallback_scores_for_direction(direction):
        if len(unique) >= 3:
            break
        if sc not in seen:
            unique.append({"score": sc, "prob": 0.0, "path": "validator_fill_top3"})
            seen.add(sc)

    direction_probs = _normalize_direction_probs(mg.get("direction_probs", {}), fallback_dir=direction)

    mg["predicted_score"] = _score_label(score, direction)
    mg["ai_score"] = mg["predicted_score"]
    mg["predicted_label"] = _score_label(score, direction)
    mg["final_direction"] = direction
    mg["result"] = _direction_cn(direction)
    mg["display_direction"] = _direction_cn(direction)
    mg["top3"] = unique[:3]

    mg["home_win_pct"] = round(direction_probs.get("home", 33.3), 2)
    mg["draw_pct"] = round(direction_probs.get("draw", 33.3), 2)
    mg["away_win_pct"] = round(direction_probs.get("away", 33.4), 2)

    mg["confidence"] = int(_clip(_f(mg.get("ai_confidence", mg.get("confidence", 60)), 60), 0, 100))
    mg["risk_level"] = normalize_risk_level(mg.get("risk_level", "medium"), mg["confidence"])
    mg["is_score_others"] = mg["predicted_score"] in ["胜其他", "平其他", "负其他"] or "其他" in str(mg["predicted_score"])

    mg["validation_warnings"] = warnings
    return mg

def normalize_risk_level(v: Any, confidence: int = 60) -> str:
    s = str(v).lower().strip()
    if s in ("low", "低", "低风险"):
        return "低"
    if s in ("high", "高", "高风险"):
        return "高"
    if s in ("medium", "mid", "中", "中风险"):
        return "中"

    if confidence >= 72:
        return "低"
    if confidence < 52:
        return "高"
    return "中"

def extract_ai_score_prob(ai_result: Dict[str, Any], predicted_score: str) -> float:
    for t in ai_result.get("top3", []):
        if isinstance(t, dict) and _normalize_score_text(t.get("score", "")) == _normalize_score_text(predicted_score):
            p = _f(t.get("prob", 0), 0)
            if p > 0:
                return p
    return 0.0

def merge_result_ai_first(
    match_obj: Dict[str, Any],
    evidence_packet: Dict[str, Any],
    ai_result: Dict[str, Any],
    ai_source: str,
    all_ai_for_match: Dict[str, Dict[str, Any]],
    stats: Dict[str, Any],
    engine_result: Dict[str, Any],
) -> Dict[str, Any]:
    mg = dict(ai_result or {})
    mg = enforce_final_consistency(mg)

    predicted_score = mg["predicted_score"]
    final_odds = get_market_odds_for_score(match_obj, predicted_score)

    model_prob_pct = extract_ai_score_prob(mg, predicted_score)

    if model_prob_pct <= 0:
        model_prob_pct = max(
            1.0,
            {
                "低": 9.0,
                "中": 7.0,
                "高": 5.0,
            }.get(mg.get("risk_level", "中"), 6.0)
        )

    market_implied_pct = None
    if final_odds > 1.05:
        market_implied_pct = round(100.0 / final_odds, 3)

    ev_data = calculate_independent_ev(
        model_prob_pct=model_prob_pct,
        market_odds=final_odds,
        market_implied_pct=market_implied_pct,
    )

    fair = evidence_packet.get("one_x_two", {}).get("fair_pack", {}).get("fair_probs", {})
    crs = evidence_packet.get("crs", {})
    risk_notes = evidence_packet.get("risk_notes", [])

    mg.update({
        "ai_final_source": ai_source,
        "engine_version": ENGINE_VERSION,
        "engine_architecture": ENGINE_ARCHITECTURE,

        "score_market_odds": final_odds,
        "score_market_implied_pct": market_implied_pct,
        "score_model_prob": round(model_prob_pct, 3),
        "edge_vs_market": ev_data["ev"],
        "suggested_kelly": ev_data["kelly"],
        "is_value": ev_data["is_value"],
        "ev_note": ev_data.get("note", ""),

        "fair_1x2": fair,
        "fair_1x2_method": evidence_packet.get("one_x_two", {}).get("fair_pack", {}).get("method"),
        "market_overround": evidence_packet.get("one_x_two", {}).get("fair_pack", {}).get("overround"),
        "raw_implied_1x2": evidence_packet.get("one_x_two", {}).get("fair_pack", {}).get("raw_implied"),

        "actual_handicap_signed": evidence_packet.get("handicap", {}).get("actual_signed"),
        "theoretical_handicap_signed": evidence_packet.get("handicap", {}).get("theoretical_signed"),
        "strong_side_from_1x2": evidence_packet.get("handicap", {}).get("strong_side_from_1x2"),
        "strong_side_depth_diff": evidence_packet.get("handicap", {}).get("strong_side_depth_diff"),

        "crs_shape": crs.get("shape"),
        "crs_moments": crs.get("moments", {}),
        "crs_margin": crs.get("margin", 0.0),
        "crs_coverage": crs.get("coverage", 0.0),
        "crs_low_rank_info": crs.get("low_rank_info", {}),
        "crs_top_scores": crs.get("top_scores", []),

        "total_goals_evidence": evidence_packet.get("total_goals", {}),
        "half_full_evidence": evidence_packet.get("half_full", {}),
        "risk_notes": risk_notes,
        "risk_note_codes": [x.get("code") for x in risk_notes if isinstance(x, dict)],

        "smart_money": evidence_packet.get("smart_money", {}),
        "smart_money_signal": " | ".join(
            str(x.get("note", ""))[:120] for x in risk_notes[:8] if isinstance(x, dict)
        ),
        "data_quality": evidence_packet.get("data_quality", {}),

        "xG_home": evidence_packet.get("xg_and_engine", {}).get("bookmaker_implied_home_xg"),
        "xG_away": evidence_packet.get("xg_and_engine", {}).get("bookmaker_implied_away_xg"),
        "expected_total_goals": evidence_packet.get("xg_and_engine", {}).get("expected_total_goals"),
        "over_2_5": evidence_packet.get("xg_and_engine", {}).get("over_25"),
        "btts": evidence_packet.get("xg_and_engine", {}).get("btts"),

        "gpt_score": _ai_score_display(all_ai_for_match.get("gpt")),
        "gpt_analysis": _ai_reason_display(all_ai_for_match.get("gpt")),
        "grok_score": _ai_score_display(all_ai_for_match.get("grok")),
        "grok_analysis": _ai_reason_display(all_ai_for_match.get("grok")),
        "gemini_score": _ai_score_display(all_ai_for_match.get("gemini")),
        "gemini_analysis": _ai_reason_display(all_ai_for_match.get("gemini")),
        "claude_score": _ai_score_display(all_ai_for_match.get("claude")),
        "claude_analysis": _ai_reason_display(all_ai_for_match.get("claude")),

        "ai_abstained": [
            n.upper() for n in ["gpt", "grok", "gemini", "claude"]
            if not _valid_ai_result(all_ai_for_match.get(n, {}))
        ],

        "market_reading": mg.get("market_reading", ""),
        "key_evidence": mg.get("key_evidence", []),
        "rejected_paths": mg.get("rejected_paths", []),
        "external_evidence": mg.get("external_evidence", []),
        "ai_reason": mg.get("reason", ""),

        "confidence_meaning": "AI审计置信度，不等于历史命中率",
    })

    mg = enforce_final_consistency(mg)
    return mg

def _ai_score_display(r: Optional[Dict[str, Any]]) -> str:
    if not _valid_ai_result(r or {}):
        return "弃权"
    return str((r or {}).get("predicted_score", (r or {}).get("ai_score", "弃权")))

def _ai_reason_display(r: Optional[Dict[str, Any]]) -> str:
    if not _valid_ai_result(r or {}):
        return "弃权 (AI失效或无有效JSON)"
    return str((r or {}).get("reason", (r or {}).get("market_reading", "")))

# ====================================================================
# Top4 精选
# ====================================================================

def select_top4(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for p in preds:
        pr = p.get("prediction", {})

        s = 0.0
        s += _f(pr.get("confidence", 0)) * 0.55

        dir_vals = [
            _f(pr.get("home_win_pct", 33.3)),
            _f(pr.get("draw_pct", 33.3)),
            _f(pr.get("away_win_pct", 33.4)),
        ]
        dir_vals = sorted(dir_vals, reverse=True)
        dir_gap = dir_vals[0] - dir_vals[1] if len(dir_vals) >= 2 else 0
        s += min(12, dir_gap * 0.65)

        dq = pr.get("data_quality", {})
        if dq.get("has_1x2"):
            s += 3
        if dq.get("has_crs"):
            s += 4
        if dq.get("has_ttg"):
            s += 3
        if dq.get("has_engine_xg"):
            s += 2
        if dq.get("has_odds_change"):
            s += 2

        ai_source = pr.get("ai_final_source", "")
        if ai_source == "claude":
            s += 6
        elif "phase1_consensus" in ai_source:
            s += 2
        elif ai_source == "local_evidence_fallback":
            s -= 18

        if pr.get("risk_level") == "低":
            s += 5
        elif pr.get("risk_level") == "高":
            s -= 10

        if pr.get("is_value"):
            ev = _f(pr.get("edge_vs_market", 0))
            if ev >= 25:
                s += 6
            elif ev >= 12:
                s += 3

        if pr.get("validation_warnings"):
            s -= 5

        risk_codes = pr.get("risk_note_codes", [])
        if any(x in risk_codes for x in ["R03_STRONG_SIDE_SHALLOW_HANDICAP", "R05_BALANCED_DIRECTION", "R12_CUP_FAVORITE_RISK"]):
            s -= 3

        if pr.get("is_score_others"):
            s += 2

        p["recommend_score"] = round(s, 2)

    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]

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

# ====================================================================
# 主入口
# ====================================================================

def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    ms = [normalize_match(m) for m in ms]

    print("\n" + "=" * 88)
    print(f"  [{ENGINE_VERSION}] AI-First Evidence Packet 四AI终审版 | {len(ms)} 场")
    print("=" * 88)

    match_analyses = []

    for i, m in enumerate(ms):
        try:
            eng = predict_match(m)
        except Exception as e:
            logger.warning(f"predict_match 失败: {e}")
            eng = {}

        try:
            league_info, _, _, _ = build_league_intelligence(m)
        except Exception:
            league_info = {}

        try:
            stats = ensemble.predict(m, {}) if ensemble else {}
        except Exception as e:
            logger.warning(f"ensemble.predict 失败: {e}")
            stats = {}

        try:
            experience = exp_engine.analyze(m) if exp_engine else {}
        except Exception:
            experience = {}

        try:
            evidence_packet = build_evidence_packet(
                match_obj=m,
                engine_result=eng,
                stats=stats,
                league_info=league_info,
                experience=experience,
            )
        except Exception as e:
            logger.warning(f"build_evidence_packet 失败: {e}")
            evidence_packet = {
                "engine_version": ENGINE_VERSION,
                "error": f"evidence_packet_failed:{e}",
                "match_identity": {
                    "home": m.get("home_team"),
                    "away": m.get("away_team"),
                },
            }

        match_analyses.append({
            "index": i + 1,
            "match": m,
            "engine": eng,
            "stats": stats,
            "league_info": league_info,
            "experience": experience,
            "evidence_packet": evidence_packet,
        })

        print(
            f"  [证据包] {i+1}. {m.get('home_team')} vs {m.get('away_team')} | "
            f"CRS覆盖:{evidence_packet.get('crs', {}).get('coverage', 0) * 100:.0f}% | "
            f"risk_notes:{len(evidence_packet.get('risk_notes', []))}"
        )

    all_ai = {"claude": {}, "gemini": {}, "gpt": {}, "grok": {}}

    if use_ai and match_analyses:
        print(f"  [{ENGINE_VERSION} AI] 启动 GPT/Grok/Gemini 初审 + Claude 终审...")
        start_t = time.time()

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
                future = pool.submit(_run_in_thread, run_ai_matrix_two_phase(match_analyses))
                try:
                    all_ai = future.result()
                except Exception as e:
                    logger.error(f"AI 矩阵并发执行崩溃: {e}")
                    all_ai = {"claude": {}, "gemini": {}, "gpt": {}, "grok": {}}
        else:
            try:
                all_ai = asyncio.run(run_ai_matrix_two_phase(match_analyses))
            except Exception as e:
                logger.error(f"AI 矩阵执行失败: {e}")
                all_ai = {"claude": {}, "gemini": {}, "gpt": {}, "grok": {}}

        print(f"  [AI完成] 耗时 {time.time() - start_t:.1f}s")

    res = []

    for i, ma in enumerate(match_analyses):
        idx = i + 1
        m = ma["match"]
        evidence_packet = ma["evidence_packet"]

        final_ai, ai_source = choose_final_ai_result(idx, all_ai, evidence_packet)

        all_ai_for_match = {
            "gpt": all_ai.get("gpt", {}).get(idx, {}),
            "grok": all_ai.get("grok", {}).get(idx, {}),
            "gemini": all_ai.get("gemini", {}).get(idx, {}),
            "claude": all_ai.get("claude", {}).get(idx, {}),
        }

        mg = merge_result_ai_first(
            match_obj=m,
            evidence_packet=evidence_packet,
            ai_result=final_ai,
            ai_source=ai_source,
            all_ai_for_match=all_ai_for_match,
            stats=ma.get("stats", {}),
            engine_result=ma.get("engine", {}),
        )

        res.append({**m, "prediction": mg})

        warn_tag = " [VALIDATE]" if mg.get("validation_warnings") else ""
        ai_tag = f" [{mg.get('ai_final_source')}]"
        risk_tag = f" [风险:{mg.get('risk_level')}]"
        value_tag = " [VALUE]" if mg.get("is_value") else ""

        print(
            f"  [{idx}] {m.get('home_team')} vs {m.get('away_team')} => "
            f"{mg['result']} ({mg['predicted_score']}) | "
            f"AI_CF:{mg['confidence']} | "
            f"主{mg['home_win_pct']:.0f}% 平{mg['draw_pct']:.0f}% 客{mg['away_win_pct']:.0f}%"
            f"{ai_tag}{risk_tag}{value_tag}{warn_tag}"
        )

    t4 = select_top4(res)

    t4_keys = set()
    for t in t4:
        tid = t.get("id")
        if tid is not None and str(tid).strip() != "":
            t4_keys.add(("id", str(tid)))
        else:
            t4_keys.add(("idx", id(t)))

    for r in res:
        rid = r.get("id")
        if rid is not None and str(rid).strip() != "":
            r["is_recommended"] = ("id", str(rid)) in t4_keys
        else:
            r["is_recommended"] = ("idx", id(r)) in t4_keys

    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    return res, t4

# ====================================================================
# 快速测试入口
# ====================================================================

if __name__ == "__main__":
    logger.info(f"{ENGINE_VERSION} 启动")
    print(f"✅ {ENGINE_VERSION} 加载完成")
    print(f"   架构: {ENGINE_ARCHITECTURE}")
    print("   本地定位: Evidence Builder + AI Caller + JSON Validator")
    print("   最终原则: Claude终审优先；本地不裁判方向/比分")