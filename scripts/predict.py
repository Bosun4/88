# ====================================================================
# 🚀 vMAX 18.2 — 统一比分矩阵 + 平局保护 + 让球不穿 + 字段校验 + Claude终审
# --------------------------------------------------------------------
# 核心升级:
#   ✅ 修复 0-1 / 0-2 / 0-3 被误判无效
#   ✅ 修复 select_top4 缺失 id 时批量误标
#   ✅ 取消 AI reason[:800] 截断
#   ✅ 伪 Shin 改成 fair_1x2，旧字段 shin 仅兼容
#   ✅ 新增统一比分矩阵 P(H=h,A=a)
#   ✅ 新增平局保护层：1-1低赔、平/平低赔、2球低赔、λ接近时防止 T1 误杀平局
#   ✅ 新增让球不穿层：主让深但让负低赔时，防平/小胜，不再机械加强主胜
#   ✅ 新增 SPF 字段校验：防止 sp_draw/same 被 s11 污染
#   ✅ T1-T16 降级为 residual 修正层
#   ✅ GPT/Grok/Gemini 初审，Claude 接收三家结论终审
#   ✅ EV 使用统一矩阵概率 vs 市场赔率，避免 CRS 自引用
# ====================================================================

import json
import os
import re
import time
import asyncio
import aiohttp
import logging
import math
from typing import Dict, List, Any, Tuple, Optional


# ====================================================================
# 日志与外部模块兼容
# ====================================================================

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
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
    from experience_rules import ExperienceEngine, apply_experience_to_prediction
except Exception as e:
    logger.warning(f"experience_rules 导入异常: {e}")

    class ExperienceEngine:
        def analyze(self, m):
            return {}

    def apply_experience_to_prediction(m, mg, exp_engine=None):
        return mg


try:
    from advanced_models import upgrade_ensemble_predict
except Exception as e:
    logger.warning(f"advanced_models.upgrade_ensemble_predict 导入异常: {e}")

    def upgrade_ensemble_predict(m, mg):
        return mg


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
# 常量
# ====================================================================

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

LEAGUE_LOW_GOALS = ["意甲", "西甲", "法甲"]
LEAGUE_HIGH_GOALS = ["德甲", "荷甲"]
LEAGUE_UPSET = ["英超"]
LEAGUE_DRAW_PREFERRED = ["意甲"]

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


# ====================================================================
# 通用工具
# ====================================================================

def _f(v, default=0.0):
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "" or s.lower() in ("none", "nan", "null", "-", "—"):
            return default
        return float(s)
    except Exception:
        return default


def _i(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def _normalize_prob_dict(d: Dict[Any, float], floor: float = 0.0) -> Dict[Any, float]:
    out = {}
    for k, v in d.items():
        out[k] = max(floor, _f(v, 0.0))
    s = sum(out.values())
    if s <= 0:
        n = len(out) or 1
        return {k: 1.0 / n for k in out}
    return {k: v / s for k, v in out.items()}


def _softmax_dict(logits: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    if not logits:
        return {}
    t = max(0.05, float(temperature or 1.0))
    mx = max(logits.values())
    ex = {k: math.exp((v - mx) / t) for k, v in logits.items()}
    return _normalize_prob_dict(ex)


def _deep_find_value(obj, aliases, skip_keys=None):
    skip_keys = set(skip_keys or [])
    aliases_low = {str(a).lower() for a in aliases}

    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in aliases_low:
                return v

        for k, v in obj.items():
            kl = str(k).lower()
            if kl in skip_keys:
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


def _parse_score(s: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        s_str = str(s).strip().replace(" ", "").replace("：", "-").replace(":", "-")
        if not s_str:
            return None, None
        if "胜" in s_str and "其他" in s_str:
            return 9, 0
        if "平" in s_str and "其他" in s_str:
            return 9, 9
        if "负" in s_str and "其他" in s_str:
            return 0, 9
        if s_str in ["主胜", "客胜", "平局"]:
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
    ss = str(score_str)
    if "胜其他" in ss or ss == "9-0":
        return "home"
    if "平其他" in ss or ss == "9-9":
        return "draw"
    if "负其他" in ss or ss == "0-9":
        return "away"
    if h > a:
        return "home"
    if h < a:
        return "away"
    return "draw"


def _direction_cn(direction: str) -> str:
    return {"home": "主胜", "draw": "平局", "away": "客胜"}.get(direction, "平局")


# ====================================================================
# 字段标准化与 SPF 校验
# ====================================================================

def _validate_spf_draw_field(m: Dict[str, Any]) -> Dict[str, Any]:
    """
    防止 sp_draw / same 被 CRS 1-1 字段 s11 污染。
    常见污染表现:
      sp_draw == same == s11 == 5.70
      实际 1X2 平赔可能在 draw / spf_sp1 / sp1 / had_draw 中。
    """
    sp_draw = _f(m.get("sp_draw", m.get("same", 0)))
    s11 = _f(m.get("s11", 0))

    if sp_draw > 1 and s11 > 1 and abs(sp_draw - s11) <= 0.03:
        alt_keys = [
            "spf_sp1", "sp1", "had_draw", "hhad_draw",
            "draw", "odds_draw", "spf_draw", "平赔", "平",
        ]
        for k in alt_keys:
            alt = _f(m.get(k, 0))
            if alt > 1.01 and abs(alt - s11) > 0.03:
                m["sp_draw"] = alt
                m["same"] = alt
                m["_odds_field_warning"] = f"sp_draw疑似被s11污染，已用{k}={alt}"
                return m

        m["_odds_field_warning"] = f"sp_draw={sp_draw}疑似等于s11={s11}，但未找到替代1X2平赔"

    return m


def normalize_match(raw_m: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(raw_m or {})

    nested_keys = [
        "v2_odds_dict", "odds_dict", "odds", "v2",
        "odds_v2", "packet", "raw_odds", "data", "detail",
    ]

    for nk in nested_keys:
        if isinstance(m.get(nk), dict):
            m.update(m[nk])

    home = (
        m.get("home_team") or m.get("home") or m.get("host") or
        m.get("team_home") or m.get("homeName") or "Home"
    )
    away = (
        m.get("away_team") or m.get("guest") or m.get("away") or
        m.get("team_away") or m.get("awayName") or "Away"
    )

    m["home_team"] = home
    m["away_team"] = away
    m["home"] = home
    m["guest"] = away

    skip = {
        "vote", "change", "points", "information",
        "prediction", "stats", "smart_signals",
        "crs", "score", "correct_score",
    }

    sp_home = m.get("sp_home")
    if sp_home is None:
        sp_home = _deep_find_value(
            m,
            ["spf_sp3", "sp3", "had_win", "home_win", "odds_win", "win", "胜"],
            skip,
        )

    sp_draw = m.get("sp_draw")
    if sp_draw is None:
        sp_draw = _deep_find_value(
            m,
            ["spf_sp1", "sp1", "had_draw", "odds_draw", "draw", "same", "平"],
            skip,
        )

    sp_away = m.get("sp_away")
    if sp_away is None:
        sp_away = _deep_find_value(
            m,
            ["spf_sp0", "sp0", "had_lose", "away_win", "odds_lose", "lose", "负"],
            skip,
        )

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
        m["give_ball"] = m.get("handicap") or m.get("rq") or m.get("let_ball") or "0"

    # 让球胜平负字段兼容
    rq_aliases = {
        "rq_win": ["rq_win", "let_win", "handicap_win", "hhad_win", "让胜"],
        "rq_draw": ["rq_draw", "let_draw", "handicap_draw", "hhad_draw", "让平"],
        "rq_lose": ["rq_lose", "let_lose", "handicap_lose", "hhad_lose", "让负"],
    }
    for std_key, aliases in rq_aliases.items():
        if m.get(std_key) is None:
            val = _deep_find_value(m, aliases, skip)
            if val is not None:
                m[std_key] = val

    if not isinstance(m.get("change"), dict):
        ch = {}
        for src_key, dst_key in [
            ("change_win", "win"), ("cw", "win"),
            ("change_same", "same"), ("change_draw", "same"), ("cs", "same"),
            ("change_lose", "lose"), ("change_away", "lose"), ("cl", "lose"),
        ]:
            if src_key in m:
                ch[dst_key] = m.get(src_key)
        m["change"] = ch

    m = _validate_spf_draw_field(m)

    if os.environ.get("ODDS_DEBUG", "0") == "1":
        print(
            "[ODDS_CHECK]",
            m.get("home_team"), m.get("away_team"),
            "sp_home=", m.get("sp_home"),
            "sp_draw=", m.get("sp_draw"),
            "sp_away=", m.get("sp_away"),
            "s11=", m.get("s11"),
            "same=", m.get("same"),
            "rq=", m.get("rq_win"), m.get("rq_draw"), m.get("rq_lose"),
            "warn=", m.get("_odds_field_warning", ""),
        )

    return m


# ====================================================================
# 公平概率与 EV
# ====================================================================

def fair_probs_from_1x2(sp_h: float, sp_d: float, sp_a: float, method: str = "power") -> Dict[str, Any]:
    odds = {"home": _f(sp_h), "draw": _f(sp_d), "away": _f(sp_a)}

    if any(v <= 1.01 for v in odds.values()):
        return {
            "method": "fallback",
            "fair_probs": {"home": 33.3, "draw": 33.3, "away": 33.4},
            "raw_implied": {"home": 33.3, "draw": 33.3, "away": 33.4},
            "overround": 0.0,
            "shin_z": None,
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

    elif method == "shin":
        try:
            def shin_probs(z: float) -> Dict[str, float]:
                z = min(max(z, 1e-9), 0.999999)
                out = {}
                for k, qi in q.items():
                    val = (
                        math.sqrt(z * z + 4.0 * (1.0 - z) * (qi * qi / overround_sum)) - z
                    ) / (2.0 * (1.0 - z))
                    out[k] = max(1e-9, val)
                return out

            lo, hi = 1e-9, 0.999999
            z_mid = None
            for _ in range(80):
                mid = (lo + hi) / 2.0
                sm = sum(shin_probs(mid).values())
                if sm > 1.0:
                    lo = mid
                else:
                    hi = mid
                z_mid = mid

            sp = shin_probs(z_mid or 0.0)
            if not all(math.isfinite(v) for v in sp.values()) or sum(sp.values()) <= 0:
                return fair_probs_from_1x2(sp_h, sp_d, sp_a, method="power")

            p = _normalize_prob_dict(sp)
            return {
                "method": "shin",
                "fair_probs": {k: round(v * 100, 3) for k, v in p.items()},
                "raw_implied": raw_pct,
                "overround": round(overround, 5),
                "shin_z": round(z_mid or 0.0, 6),
            }
        except Exception:
            return fair_probs_from_1x2(sp_h, sp_d, sp_a, method="power")

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
        "shin_z": None,
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


def calculate_independent_ev(model_prob_pct: float, market_odds: float, market_implied_pct: Optional[float] = None):
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
        "note": "independent_model_vs_market",
    }


# ====================================================================
# 基本面与盘口工具
# ====================================================================

def _extract_form_record(text: str) -> Tuple[int, int, int]:
    if not text:
        return 0, 0, 0
    text = str(text)
    patterns = [
        r"近\s*\d+\s*[主客场]*\s*(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
        r"(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
        r"近\s*\d+[:：]\s*(\d+)W(\d+)D(\d+)L",
    ]
    for pat in patterns:
        m = re.search(pat, text)
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
    h_match = re.search(r"场均进球[^0-9]*(\d+\.?\d*)", text)
    a_match = re.search(r"场均失球[^0-9]*(\d+\.?\d*)", text)
    return (
        float(h_match.group(1)) if h_match else 0.0,
        float(a_match.group(1)) if a_match else 0.0,
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


def _infer_theoretical_handicap(sp_h: float, sp_a: float) -> float:
    if sp_h <= 1.01 or sp_a <= 1.01:
        return 0.0
    ratio = sp_a / sp_h
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
    if ratio >= 0.85:
        return 0.0
    if ratio >= 0.63:
        return -0.25
    if ratio >= 0.46:
        return -0.75
    if ratio >= 0.33:
        return -1.25
    if ratio >= 0.25:
        return -1.75
    if ratio >= 0.18:
        return -2.25
    return -2.75


def _parse_actual_handicap(match_obj: Dict) -> float:
    raw = match_obj.get("give_ball", match_obj.get("handicap", "0"))
    s = str(raw).strip()
    s = s.replace("主", "").replace("客", "")
    s = s.replace("受让", "+").replace("让", "-")
    if "/" in s:
        parts = s.split("/")
        try:
            val = (_f(parts[0].strip()) + _f(parts[1].strip())) / 2.0
            return -val
        except Exception:
            pass
    val = _f(s, 0.0)
    return -val


def detect_handicap_no_cover(match_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    竞彩让球胜平负保护层。
    give_ball=-1 表示主让1球。
    主让1但让负低赔，通常意味着主队不容易穿盘，需防平/小胜。
    """
    rq_win = _f(match_obj.get("rq_win", 0))
    rq_draw = _f(match_obj.get("rq_draw", 0))
    rq_lose = _f(match_obj.get("rq_lose", 0))
    give_ball = _f(match_obj.get("give_ball", 0))

    if give_ball <= -1 and rq_lose > 1 and rq_lose <= 1.70:
        return {
            "detected": True,
            "type": "HOME_DEEP_BUT_NO_COVER",
            "description": f"主让{abs(give_ball):.0f}但让负低赔{rq_lose:.2f}，主队不穿盘，防平/小胜",
            "direction_adjust": {"home": -0.45, "draw": +0.65, "away": +0.15},
            "boost_scores": ["1-1", "1-0", "2-1", "0-0"],
        }

    if give_ball >= 1 and rq_win > 1 and rq_win <= 1.70:
        return {
            "detected": True,
            "type": "AWAY_DEEP_BUT_NO_COVER",
            "description": f"客让{abs(give_ball):.0f}但让胜低赔{rq_win:.2f}，客队不穿盘，防平/小负",
            "direction_adjust": {"away": -0.45, "draw": +0.65, "home": +0.15},
            "boost_scores": ["1-1", "0-1", "1-2", "0-0"],
        }

    return {"detected": False}


# ====================================================================
# Sharp / Steam
# ====================================================================

def detect_sharp_direction(smart_signals: List) -> Dict[str, Any]:
    detected = False
    sharp_dir = None
    for s in smart_signals or []:
        s_str = str(s)
        if "Sharp" in s_str or "sharp" in s_str or "聪明钱" in s_str or "专业资金" in s_str:
            detected = True
            if re.search(r"(主胜|主队|走主|→\s*主|流向\s*主|资金\s*主|Sharp主|聪明钱主)", s_str):
                sharp_dir = "home"
                break
            elif re.search(r"(客胜|客队|走客|→\s*客|流向\s*客|资金\s*客|Sharp客|聪明钱客)", s_str):
                sharp_dir = "away"
                break
            elif re.search(r"(平局|平赔|走平|→\s*平|流向\s*平|资金\s*平|Sharp平|聪明钱平)", s_str):
                sharp_dir = "draw"
                break
    return {"detected": detected, "sharp_dir": sharp_dir}


def detect_steam_direction(smart_signals: List) -> Dict[str, Any]:
    steam_dir = None
    steam_type = None
    for s in smart_signals or []:
        s_str = str(s)
        if "Steam" not in s_str and "steam" not in s_str and "异动" not in s_str:
            continue
        is_reverse = "反向" in s_str or "未跟" in s_str or "不跟" in s_str
        if re.search(r"(主胜.*Steam|Steam.*主胜|主胜.*降水|主.*Steam|主胜.*异动)", s_str):
            steam_dir = "home"
            steam_type = "reverse" if is_reverse else "normal"
            break
        elif re.search(r"(客胜.*Steam|Steam.*客胜|客胜.*降水|客.*Steam|客胜.*异动)", s_str):
            steam_dir = "away"
            steam_type = "reverse" if is_reverse else "normal"
            break
        elif re.search(r"(平.*Steam|Steam.*平|平赔.*异动)", s_str):
            steam_dir = "draw"
            steam_type = "reverse" if is_reverse else "normal"
            break
    return {"steam_dir": steam_dir, "steam_type": steam_type}


# ====================================================================
# CRS 分析
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
            extras[key] = {"odds": odds, "scores": scores_set}

    if len(raw_odds) < 8:
        return {}, 0.0, 0.0

    raw_sum = sum(1 / o for o in raw_odds.values())
    for ex in extras.values():
        raw_sum += 1 / ex["odds"]

    margin = raw_sum - 1.0
    probs = {}

    for score, odds in raw_odds.items():
        probs[score] = (1 / odds) / raw_sum * 100

    for key, ex in extras.items():
        total_prob = (1 / ex["odds"]) / raw_sum * 100
        per = total_prob / max(1, len(ex["scores"]))
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
    if total < 1:
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

    if std_h > 0.01:
        skew_h = sum(((h - e_h) / std_h) ** 3 * p for (h, a), p in reg.items())
    else:
        skew_h = 0.0
    if std_a > 0.01:
        skew_a = sum(((a - e_a) / std_a) ** 3 * p for (h, a), p in reg.items())
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
        return "unknown", ["CRS数据不足"]

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
        anomalies.append(f"互射局:λ总{lt:.2f},相关{corr:.2f}")
    elif lt <= 2.2 and var_h < 1.2 and var_a < 1.2:
        verdict = "grinder"
        anomalies.append(f"磨局:λ总{lt:.2f}")
    elif lh - la >= 1.2:
        verdict = "lopsided_h"
        anomalies.append(f"主队碾压:{lh:.2f} vs {la:.2f}")
    elif la - lh >= 1.2:
        verdict = "lopsided_a"
        anomalies.append(f"客队碾压:{la:.2f} vs {lh:.2f}")
    elif abs(lh - la) < 0.4:
        verdict = "balanced"
        anomalies.append(f"均势:λ主{lh:.2f} vs 客{la:.2f}")

    if abs(skew_h) > 1.8:
        anomalies.append(f"主偏度异常:{skew_h:.2f}")
    if abs(skew_a) > 1.8:
        anomalies.append(f"客偏度异常:{skew_a:.2f}")
    if corr < -0.15:
        anomalies.append(f"负相关{corr:.2f}")

    return verdict, anomalies


def compute_direction_from_crs(probs: Dict[str, float]) -> Dict[str, float]:
    home_p = draw_p = away_p = 0.0
    for sc, p in probs.items():
        d = _score_direction(sc)
        if d == "home":
            home_p += p
        elif d == "draw":
            draw_p += p
        elif d == "away":
            away_p += p

    total = home_p + draw_p + away_p
    if total > 0:
        return {
            "home": round(home_p / total * 100, 2),
            "draw": round(draw_p / total * 100, 2),
            "away": round(away_p / total * 100, 2),
        }
    return {"home": 33.3, "draw": 33.3, "away": 33.4}


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

    return {
        "implied_probs": {k: round(v, 2) for k, v in probs.items()},
        "margin": margin,
        "coverage": coverage,
        "moments": moments,
        "shape_verdict": verdict,
        "anomalies": anomalies,
        "direction_probs": direction_probs,
        "top_scores": [(sc, round(p, 2)) for sc, p in sorted_scores[:10]],
    }


# ====================================================================
# 平局保护层
# ====================================================================

def draw_protection_score(match_obj: Dict[str, Any], crs_analysis: Dict[str, Any]) -> Dict[str, Any]:
    s11 = _f(match_obj.get("s11", 999), 999)
    s00 = _f(match_obj.get("s00", 999), 999)
    pp = _f(match_obj.get("pp", 999), 999)
    a2 = _f(match_obj.get("a2", 999), 999)
    a0 = _f(match_obj.get("a0", 999), 999)

    top_scores = crs_analysis.get("top_scores", []) if isinstance(crs_analysis, dict) else []
    top_names = [sc for sc, _ in top_scores[:4]]
    moments = crs_analysis.get("moments", {}) if isinstance(crs_analysis, dict) else {}

    lh = _f(moments.get("lambda_h", 0))
    la = _f(moments.get("lambda_a", 0))
    lt = _f(moments.get("lambda_total", 0))

    score = 0
    reasons = []

    if s11 > 1 and s11 <= 6.2:
        score += 2
        reasons.append(f"1-1低赔{s11:.2f}")
    if s00 > 1 and s00 <= 8.5:
        score += 1
        reasons.append(f"0-0低赔{s00:.2f}")
    if "1-1" in top_names:
        score += 2
        reasons.append("1-1在CRS前列")
    if pp > 1 and pp <= 5.0:
        score += 2
        reasons.append(f"半全场平/平{pp:.2f}")
    if a2 > 1 and a2 <= 3.4:
        score += 1
        reasons.append(f"2球低赔{a2:.2f}")
    if a0 > 1 and a0 <= 8.5:
        score += 1
        reasons.append(f"0球低赔{a0:.2f}")
    if lh > 0 and la > 0 and abs(lh - la) <= 0.45 and 1.75 <= lt <= 2.65:
        score += 2
        reasons.append(f"λ接近({lh:.2f}-{la:.2f},总{lt:.2f})")

    return {
        "score": score,
        "detected": score >= 4,
        "reasons": reasons,
    }


def has_strong_draw_evidence(match_obj: Dict[str, Any], crs_analysis: Dict[str, Any]) -> bool:
    return draw_protection_score(match_obj, crs_analysis).get("detected", False)


# ====================================================================
# T1-T16 陷阱
# ====================================================================

def detect_T1_draw_trap(match_obj: Dict, engine_result: Dict, smart_signals: List, fair: Dict, crs_analysis: Dict) -> Optional[Dict]:
    dp = draw_protection_score(match_obj, crs_analysis)
    if dp["detected"]:
        return {
            "trap": "T1_DRAW_TRAP_CANCELLED_BY_DRAW_GUARD",
            "description": "T1被平局保护层拦截:" + " + ".join(dp["reasons"][:5]),
            "severity": 1,
            "direction_adjust": {"draw": +0.85, "home": -0.25, "away": -0.05},
            "score_multipliers": {"1-1": 1.35, "0-0": 1.15, "2-2": 1.12},
            "boost_scores": ["1-1", "0-0", "2-2"],
        }

    change = match_obj.get("change", {}) or {}
    cw = _f(change.get("win", 0))
    cs = _f(change.get("same", 0))
    cl = _f(change.get("lose", 0))

    if cs >= -0.04:
        return None
    if cs > cw or cs > cl:
        return None

    evidence_score = 2
    evidence_detail = [f"平赔独降{cs:.2f}"]

    fair_h, fair_a = fair["home"], fair["away"]
    strong_fair = max(fair_h, fair_a)

    if strong_fair < 34:
        return None

    strong_side = "home" if fair_h > fair_a else "away"
    weak_side = "away" if strong_side == "home" else "home"
    strong_cn = "主" if strong_side == "home" else "客"

    if strong_fair >= 42:
        evidence_score += 2
    elif strong_fair >= 38:
        evidence_score += 1

    evidence_detail.append(f"{strong_cn}公平概率{strong_fair:.1f}%")

    strong_fund = _fundamental_strength(match_obj, strong_side)
    weak_fund = _fundamental_strength(match_obj, weak_side)

    if strong_fund["total"] >= 3:
        if strong_fund["win_rate"] >= 0.55 or strong_fund["strength_score"] >= 15:
            evidence_score += 2
            evidence_detail.append(f"{strong_cn}基本面强")
        elif strong_fund["win_rate"] < 0.30 and strong_fund["strength_score"] < -15:
            evidence_score -= 1

    if weak_fund["total"] >= 3:
        if weak_fund["win_rate"] <= 0.30 or weak_fund["strength_score"] <= -15:
            evidence_score += 1
            evidence_detail.append("弱方基本面差")

    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))

    if hxg > 0 and axg > 0:
        xg_diff = hxg - axg
        expected_sign = 1 if strong_side == "home" else -1
        signed_xg = xg_diff * expected_sign
        if signed_xg > 0.15:
            evidence_score += 1
            evidence_detail.append(f"xG同向{signed_xg:+.2f}")
        elif signed_xg < -0.3:
            evidence_score -= 2

    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))

    if sp_h > 1 and sp_a > 1:
        theoretical = _infer_theoretical_handicap(sp_h, sp_a)
        actual = _parse_actual_handicap(match_obj)
        hc_diff = actual - theoretical
        if (strong_side == "home" and hc_diff >= 0.3) or (strong_side == "away" and hc_diff <= -0.3):
            evidence_score += 1
            evidence_detail.append(f"让球偏深{hc_diff:+.2f}")

    if evidence_score < 5:
        return None

    severity = 3 if evidence_score < 6 else 4

    return {
        "trap": "T1_DRAW_TRAP",
        "description": f"诱平赔陷阱(得分{evidence_score}):" + " + ".join(evidence_detail),
        "severity": severity,
        "direction_adjust": {strong_side: +1.15, "draw": -0.85, weak_side: -0.35},
        "score_multipliers": {"1-1": 0.75, "2-2": 0.75, "0-0": 0.8},
        "suppress_draw_sharp": True,
    }


def detect_T2_T3_handicap_trap(match_obj: Dict, fair: Dict) -> Optional[Dict]:
    no_cover = detect_handicap_no_cover(match_obj)
    if no_cover.get("detected"):
        return {
            "trap": "T2_HANDICAP_NO_COVER",
            "description": no_cover["description"],
            "severity": 3,
            "direction_adjust": no_cover["direction_adjust"],
            "score_multipliers": {},
            "boost_scores": no_cover["boost_scores"],
        }

    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))

    if sp_h < 1.05 or sp_a < 1.05:
        return None

    theoretical = _infer_theoretical_handicap(sp_h, sp_a)
    actual = _parse_actual_handicap(match_obj)

    if abs(actual) < 0.1 and abs(theoretical) < 0.4:
        return None

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
            "description": f"让球偏深:理论{theoretical:.2f} vs 实际{actual:.2f},差{diff:+.2f}球",
            "severity": severity,
            "direction_adjust": {"home": +0.55 * min(2.0, abs(diff)), "away": -0.25, "draw": -0.15},
            "score_multipliers": {},
            "boost_scores": ["2-0", "3-0", "2-1", "3-1"] if abs(diff) >= 1.0 else ["2-1", "2-0"],
        }

    severity = 2 if abs(diff) < 1.0 else 3
    return {
        "trap": "T3_HANDICAP_SHALLOWER",
        "description": f"让球偏浅:理论{theoretical:.2f} vs 实际{actual:.2f},差{diff:+.2f}球",
        "severity": severity,
        "direction_adjust": {"home": -0.55 * min(2.0, abs(diff)), "away": +0.65 * min(2.0, abs(diff)), "draw": +0.3},
        "score_multipliers": {},
        "boost_scores": ["0-1", "1-2", "0-2", "1-1"] if abs(diff) >= 1.0 else ["0-1", "1-1"],
    }


def detect_T4_T5_fake_favorite(match_obj: Dict, engine_result: Dict, fair: Dict) -> Optional[Dict]:
    fair_h, fair_a = fair["home"], fair["away"]

    if fair_h > 48:
        fund_h = _fundamental_strength(match_obj, "home")
        fund_a = _fundamental_strength(match_obj, "away")
        if fund_h["total"] >= 3 and fund_a["total"] >= 3:
            if fund_h["strength_score"] < -5 and fund_a["strength_score"] > 15:
                return {
                    "trap": "T4_FAKE_HOME_FAVORITE",
                    "description": f"诱主胜:主公平概率{fair_h:.1f}%但主基本面{fund_h['strength_score']} vs 客{fund_a['strength_score']}",
                    "severity": 3,
                    "direction_adjust": {"home": -1.2, "away": +1.0, "draw": +0.35},
                    "score_multipliers": {"1-0": 0.65, "2-0": 0.55, "2-1": 0.75},
                }

    if fair_a > 48:
        fund_h = _fundamental_strength(match_obj, "home")
        fund_a = _fundamental_strength(match_obj, "away")
        if fund_h["total"] >= 3 and fund_a["total"] >= 3:
            if fund_a["strength_score"] < -5 and fund_h["strength_score"] > 15:
                return {
                    "trap": "T5_FAKE_AWAY_FAVORITE",
                    "description": f"诱客胜:客公平概率{fair_a:.1f}%但客基本面{fund_a['strength_score']} vs 主{fund_h['strength_score']}",
                    "severity": 3,
                    "direction_adjust": {"away": -1.2, "home": +1.0, "draw": +0.35},
                    "score_multipliers": {"0-1": 0.65, "0-2": 0.55, "1-2": 0.75},
                }
    return None


def detect_T6_T7_score_range_trap(match_obj: Dict, engine_result: Dict, exp_goals: float) -> Optional[Dict]:
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
            "description": f"诱小比分陷阱:a0/1/2压低{low_small}项但λ={exp_goals:.2f}",
            "severity": 2,
            "direction_adjust": {},
            "score_multipliers": {"0-0": 0.55, "1-0": 0.7, "0-1": 0.7, "1-1": 0.75},
            "boost_scores": ["2-1", "2-2", "3-1", "1-3", "3-2"],
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
            "description": f"诱大比分陷阱:a5/6/7压低{low_large}项但λ={exp_goals:.2f}",
            "severity": 2,
            "direction_adjust": {},
            "score_multipliers": {"3-2": 0.6, "4-2": 0.5, "3-3": 0.5},
            "boost_scores": ["1-0", "0-1", "1-1", "2-1", "1-2"],
        }

    return None


def detect_T8_false_cold(match_obj: Dict, smart_signals: List, fair: Dict) -> Optional[Dict]:
    sigs_str = " ".join(str(s) for s in smart_signals or [])
    cold_triggers = sum(1 for kw in ["坏消息", "崩盘", "造热", "背离", "盘口太便宜"] if kw in sigs_str)
    if cold_triggers < 2:
        return None

    vote = match_obj.get("vote", {}) or {}
    vh = int(_f(vote.get("win", 33), 33))
    va = int(_f(vote.get("lose", 33), 33))

    hot_dir = None
    if vh >= 58:
        hot_dir = "home"
    elif va >= 58:
        hot_dir = "away"

    if not hot_dir:
        return None

    fund = _fundamental_strength(match_obj, hot_dir)
    if fund["total"] >= 3 and fund["strength_score"] > 20 and fund["win_rate"] > 0.55:
        return {
            "trap": "T8_FALSE_COLD",
            "description": f"假冷门:{hot_dir}散户热但基本面真强",
            "severity": 2,
            "direction_adjust": {hot_dir: +0.9, "home" if hot_dir == "away" else "away": -0.6},
            "score_multipliers": {},
            "suppress_contrarian": True,
        }
    return None


def detect_T9_fake_contrarian(match_obj: Dict, fair: Dict, smart_signals: List) -> Optional[Dict]:
    vote = match_obj.get("vote", {}) or {}
    vh = int(_f(vote.get("win", 33), 33))
    va = int(_f(vote.get("lose", 33), 33))

    hot_dir = None
    hot_pct = 0
    if vh >= 60:
        hot_dir = "home"
        hot_pct = vh
    elif va >= 60:
        hot_dir = "away"
        hot_pct = va

    if not hot_dir:
        return None

    change = match_obj.get("change", {}) or {}
    cw = _f(change.get("win", 0))
    cl = _f(change.get("lose", 0))

    follow = False
    if hot_dir == "home" and cw < -0.04:
        follow = True
    elif hot_dir == "away" and cl < -0.04:
        follow = True

    if follow:
        return {
            "trap": "T9_FAKE_CONTRARIAN",
            "description": f"诱反指:{hot_dir}散户{hot_pct}%+赔率同向降水",
            "severity": 2,
            "direction_adjust": {hot_dir: +0.7},
            "score_multipliers": {},
            "suppress_contrarian": True,
        }
    return None


def detect_T10_silent_market(match_obj: Dict) -> Optional[Dict]:
    change = match_obj.get("change", {}) or {}
    total_move = abs(_f(change.get("win", 0))) + abs(_f(change.get("same", 0))) + abs(_f(change.get("lose", 0)))
    from_crs = sum(1 for k in ["w10", "w20", "w21", "s00", "s11", "l01", "l02", "l12"] if _f(match_obj.get(k, 0)) > 1)

    if total_move < 0.03 and from_crs < 6:
        return {
            "trap": "T10_SILENT_MARKET",
            "description": f"沉默盘:赔率变动{total_move:.3f}+CRS覆盖{from_crs}/8",
            "severity": 1,
            "direction_adjust": {},
            "score_multipliers": {},
            "confidence_penalty": 8,
        }
    return None


def detect_T11_xg_divergence(match_obj: Dict, engine_result: Dict) -> Optional[Dict]:
    hxg_book = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg_book = _f(engine_result.get("bookmaker_implied_away_xg", 0))
    if hxg_book <= 0 or axg_book <= 0:
        return None

    info_src = match_obj.get("points", {}) or {}
    h_for, _ = _extract_avg_goals(str(info_src.get("home_strength", "")))
    a_for, _ = _extract_avg_goals(str(info_src.get("guest_strength", "")))

    divergences = []
    if h_for > 0 and abs(hxg_book - h_for) > 0.8:
        divergences.append(f"主xG书{hxg_book:.2f}vs场均{h_for:.2f}")
    if a_for > 0 and abs(axg_book - a_for) > 0.8:
        divergences.append(f"客xG书{axg_book:.2f}vs场均{a_for:.2f}")

    if len(divergences) >= 2:
        return {
            "trap": "T11_XG_DIVERGENCE",
            "description": "xG背离:" + "; ".join(divergences),
            "severity": 1,
            "direction_adjust": {},
            "score_multipliers": {},
            "xg_override": {
                "home_xg": h_for if h_for > 0 else hxg_book,
                "away_xg": a_for if a_for > 0 else axg_book,
            },
        }
    return None


def detect_T12_missing_handicap(match_obj: Dict) -> Optional[Dict]:
    actual = _parse_actual_handicap(match_obj)
    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))

    if abs(actual) > 0.1:
        return None
    if sp_h < 1.01 or sp_a < 1.01:
        return None

    theoretical = _infer_theoretical_handicap(sp_h, sp_a)
    if abs(theoretical) < 0.4:
        return None

    return {
        "trap": "T12_MISSING_HANDICAP",
        "description": f"让球未开但理论让{theoretical:.2f}球",
        "severity": 1,
        "direction_adjust": {},
        "score_multipliers": {},
        "confidence_penalty": 5,
    }


def detect_T13_goalless_draw(match_obj: Dict, engine_result: Dict, fair: Dict, exp_goals: float) -> Optional[Dict]:
    if abs(fair["home"] - fair["away"]) > 10:
        return None

    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))
    if hxg <= 0 or axg <= 0:
        return None

    total_xg = hxg + axg
    if total_xg >= 2.3:
        return None

    info = match_obj.get("points", {}) or {}
    h_for, h_against = _extract_avg_goals(str(info.get("home_strength", "")))
    a_for, a_against = _extract_avg_goals(str(info.get("guest_strength", "")))

    weak_attack = 0
    if 0 < h_for < 1.4:
        weak_attack += 1
    if 0 < a_for < 1.4:
        weak_attack += 1

    strong_def = 0
    if 0 < h_against < 1.2:
        strong_def += 1
    if 0 < a_against < 1.2:
        strong_def += 1

    if weak_attack + strong_def < 2:
        return None

    a0 = _f(match_obj.get("a0", 999), 999)
    a1 = _f(match_obj.get("a1", 999), 999)
    a2 = _f(match_obj.get("a2", 999), 999)

    small_compressed = 0
    if 0 < a0 <= 10:
        small_compressed += 1
    if 0 < a1 <= 5:
        small_compressed += 1
    if 0 < a2 <= 3.5:
        small_compressed += 1

    if small_compressed < 1:
        return None

    vote = match_obj.get("vote", {}) or {}
    max_vote = max(
        int(_f(vote.get("win", 33), 33)),
        int(_f(vote.get("same", 33), 33)),
        int(_f(vote.get("lose", 33), 33)),
    )

    if max_vote >= 55:
        return None

    severity = 2
    if total_xg < 2.0 and small_compressed >= 2:
        severity = 3

    return {
        "trap": "T13_GOALLESS_DRAW",
        "description": f"闷平:xG总{total_xg:.2f}+弱攻{weak_attack}/强防{strong_def}+小球压低{small_compressed}",
        "severity": severity,
        "direction_adjust": {"draw": +0.95, "home": -0.25, "away": -0.25},
        "score_multipliers": {"2-1": 0.75, "1-2": 0.75, "2-2": 0.7, "3-1": 0.5, "1-3": 0.5},
        "boost_scores": ["0-0", "1-1", "1-0", "0-1"],
    }


def detect_T14_cup_favorite_trap(match_obj: Dict, fair: Dict) -> Optional[Dict]:
    league = str(match_obj.get("league", match_obj.get("cup", "")))
    if not any(kw in league for kw in CUP_KEYWORDS):
        return None

    fair_h, fair_a = fair["home"], fair["away"]
    strong_fair = max(fair_h, fair_a)
    if strong_fair < 55:
        return None

    strong_side = "home" if fair_h > fair_a else "away"
    weak_side = "away" if strong_side == "home" else "home"
    vote = match_obj.get("vote", {}) or {}
    vh = int(_f(vote.get("win", 33), 33))
    va = int(_f(vote.get("lose", 33), 33))
    strong_vote = vh if strong_side == "home" else va

    if strong_vote < 50:
        return None

    weak_fund = _fundamental_strength(match_obj, weak_side)
    if weak_fund["total"] >= 3:
        reasonable_weak = (
            weak_fund["win_rate"] >= 0.35 or
            weak_fund["goals_for"] >= 1.2 or
            weak_fund["strength_score"] > -10
        )
        if not reasonable_weak:
            return None

    strong_cn = "主" if strong_side == "home" else "客"
    return {
        "trap": "T14_CUP_FAVORITE",
        "description": f"杯赛大热:{strong_cn}公平概率{strong_fair:.1f}%+散户{strong_vote}%",
        "severity": 3,
        "direction_adjust": {strong_side: -0.75, weak_side: -0.15, "draw": +1.35},
        "score_multipliers": {"3-0": 0.5, "3-1": 0.6, "0-3": 0.5, "1-3": 0.6, "2-0": 0.85, "0-2": 0.85},
        "boost_scores": ["1-1", "0-0", "1-0", "0-1", "2-1", "1-2"],
    }


def detect_T15_historical_deadlock(match_obj: Dict, fair: Dict) -> Optional[Dict]:
    if abs(fair["home"] - fair["away"]) > 18:
        return None

    info = match_obj.get("points", {}) or {}
    text = " ".join(str(v) for v in info.values() if v)

    patterns = [
        r"对阵[^0-9]{0,20}(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
        r"历史交锋[^0-9]*(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
    ]

    best_w = best_d = best_l = 0
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                best_w = int(m.group(1))
                best_d = int(m.group(2))
                best_l = int(m.group(3))
                break
            except Exception:
                continue

    total = best_w + best_d + best_l
    if total < 3:
        return None

    draw_rate = best_d / total
    if draw_rate < 0.40 and best_d < 3:
        return None

    s11 = _f(match_obj.get("s11", 999), 999)
    if not (0 < s11 < 9.0):
        return None

    severity = 2 if draw_rate >= 0.50 else 1
    return {
        "trap": "T15_HISTORICAL_DEADLOCK",
        "description": f"历史僵局:{best_w}胜{best_d}平{best_l}负，平率{draw_rate:.0%}+s11={s11:.1f}",
        "severity": severity,
        "direction_adjust": {"draw": +0.65, "home": -0.2, "away": -0.2},
        "score_multipliers": {"3-1": 0.75, "1-3": 0.75, "3-0": 0.7, "0-3": 0.7},
        "boost_scores": ["1-1", "0-0", "1-0", "0-1"],
    }


def detect_T16_sharp_badnews_conflict(match_obj: Dict, smart_signals: List, fair: Dict) -> Optional[Dict]:
    sigs_str = " ".join(str(s) for s in smart_signals or [])
    sharp_info = detect_sharp_direction(smart_signals)
    if not sharp_info["detected"]:
        return None

    sharp_dir = sharp_info["sharp_dir"]
    if not sharp_dir or sharp_dir == "draw":
        return None

    has_home_bad = "主队坏消息" in sigs_str or "主坏消息" in sigs_str or "主利空" in sigs_str
    has_away_bad = "客队坏消息" in sigs_str or "客坏消息" in sigs_str or "客利空" in sigs_str

    has_bad = (sharp_dir == "home" and has_home_bad) or (sharp_dir == "away" and has_away_bad)
    if not has_bad:
        return None

    if fair.get(sharp_dir, 33) >= 55:
        return None

    return {
        "trap": "T16_SHARP_BADNEWS_CONFLICT",
        "description": f"Sharp({sharp_dir})+该方坏消息，对冲信号",
        "severity": 2,
        "direction_adjust": {sharp_dir: -0.45, "draw": +0.8, "home" if sharp_dir == "away" else "away": +0.2},
        "score_multipliers": {},
        "boost_scores": ["1-1", "0-0"],
        "downgrade_sharp_trust": 0.35,
    }


def detect_all_traps(match_obj: Dict, engine_result: Dict, ai_responses: Dict, smart_signals: List, exp_goals: float) -> Dict[str, Any]:
    match_obj = normalize_match(match_obj)

    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_d = _f(match_obj.get("sp_draw", match_obj.get("same", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))

    fair_pack = fair_probs_from_1x2(sp_h, sp_d, sp_a, method="power")
    fair = {
        "home": fair_pack["fair_probs"].get("home", 33.3),
        "draw": fair_pack["fair_probs"].get("draw", 33.3),
        "away": fair_pack["fair_probs"].get("away", 33.4),
    }

    crs_for_trap = analyze_crs_matrix(match_obj)

    all_detectors = [
        lambda: detect_T1_draw_trap(match_obj, engine_result, smart_signals, fair, crs_for_trap),
        lambda: detect_T2_T3_handicap_trap(match_obj, fair),
        lambda: detect_T4_T5_fake_favorite(match_obj, engine_result, fair),
        lambda: detect_T6_T7_score_range_trap(match_obj, engine_result, exp_goals),
        lambda: detect_T8_false_cold(match_obj, smart_signals, fair),
        lambda: detect_T9_fake_contrarian(match_obj, fair, smart_signals),
        lambda: detect_T10_silent_market(match_obj),
        lambda: detect_T11_xg_divergence(match_obj, engine_result),
        lambda: detect_T12_missing_handicap(match_obj),
        lambda: detect_T13_goalless_draw(match_obj, engine_result, fair, exp_goals),
        lambda: detect_T14_cup_favorite_trap(match_obj, fair),
        lambda: detect_T15_historical_deadlock(match_obj, fair),
        lambda: detect_T16_sharp_badnews_conflict(match_obj, smart_signals, fair),
    ]

    traps = []
    for detector in all_detectors:
        try:
            r = detector()
            if r:
                traps.append(r)
        except Exception as e:
            logger.warning(f"trap detector 异常: {str(e)[:120]}")

    # 互斥
    has_t14 = any(t.get("trap") == "T14_CUP_FAVORITE" for t in traps)
    if has_t14:
        traps = [t for t in traps if t.get("trap") != "T1_DRAW_TRAP"]

    has_draw_guard = any(t.get("trap") == "T1_DRAW_TRAP_CANCELLED_BY_DRAW_GUARD" for t in traps)
    if has_draw_guard:
        traps = [t for t in traps if t.get("trap") != "T1_DRAW_TRAP"]

    t2 = next((t for t in traps if t.get("trap") == "T2_HANDICAP_DEEPER"), None)
    t_nc = next((t for t in traps if t.get("trap") == "T2_HANDICAP_NO_COVER"), None)
    if t2 and t_nc:
        traps = [t for t in traps if t.get("trap") != "T2_HANDICAP_DEEPER"]

    has_t13 = any(t.get("trap") == "T13_GOALLESS_DRAW" for t in traps)
    if has_t13:
        traps = [t for t in traps if t.get("trap") != "T6_SMALL_SCORE_TRAP"]

    direction_adjust = {"home": 0.0, "draw": 0.0, "away": 0.0}
    score_multipliers = {}
    boost_scores = []
    suppress_contrarian = False
    xg_override = None
    confidence_penalty = 0
    total_severity = 0
    sharp_trust_override = 1.0

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

        if t.get("suppress_contrarian"):
            suppress_contrarian = True
        if t.get("xg_override"):
            xg_override = t["xg_override"]

        confidence_penalty += t.get("confidence_penalty", 0)

        if "downgrade_sharp_trust" in t:
            sharp_trust_override = min(sharp_trust_override, t["downgrade_sharp_trust"])

    sharp_info = detect_sharp_direction(smart_signals)
    steam_info = detect_steam_direction(smart_signals)
    dp = draw_protection_score(match_obj, crs_for_trap)
    nc = detect_handicap_no_cover(match_obj)

    return {
        "traps_detected": traps,
        "trap_count": len(traps),
        "total_severity": total_severity,
        "direction_adjust": direction_adjust,
        "score_multipliers": score_multipliers,
        "boost_scores": list(set(boost_scores)),
        "suppress_contrarian": suppress_contrarian,
        "xg_override": xg_override,
        "confidence_penalty": confidence_penalty,
        "sharp_trust_override": sharp_trust_override,
        "steam_trust_override": sharp_trust_override,

        "shin": fair,
        "fair_1x2": fair,
        "fair_1x2_method": fair_pack.get("method", "power"),
        "market_overround": fair_pack.get("overround", 0.0),
        "raw_implied_1x2": fair_pack.get("raw_implied", {}),
        "odds_field_warning": match_obj.get("_odds_field_warning", ""),

        "draw_protection": dp,
        "handicap_no_cover": nc,

        "sharp_detected": sharp_info["detected"],
        "sharp_dir": sharp_info["sharp_dir"],
        "steam_dir": steam_info["steam_dir"],
        "steam_type": steam_info["steam_type"],
    }


# ====================================================================
# 统一比分矩阵
# ====================================================================

def _poisson_pmf(lam: float, k: int) -> float:
    lam = max(0.05, min(8.0, float(lam)))
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _estimate_base_lambdas(match_obj: Dict[str, Any], engine_result: Dict[str, Any], crs_analysis: Dict[str, Any], exp_goals: float) -> Tuple[float, float]:
    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))

    moments = crs_analysis.get("moments", {}) if crs_analysis else {}
    ch = _f(moments.get("lambda_h", 0))
    ca = _f(moments.get("lambda_a", 0))

    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_d = _f(match_obj.get("sp_draw", match_obj.get("same", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))
    fair = fair_probs_from_1x2(sp_h, sp_d, sp_a, method="power")["fair_probs"]

    if hxg > 0.1 and axg > 0.1:
        lam_h, lam_a = hxg, axg
        if ch > 0.1 and ca > 0.1:
            lam_h = 0.65 * hxg + 0.35 * ch
            lam_a = 0.65 * axg + 0.35 * ca
        return max(0.05, lam_h), max(0.05, lam_a)

    if ch > 0.1 and ca > 0.1:
        return max(0.05, ch), max(0.05, ca)

    total = exp_goals if 1.0 <= exp_goals <= 6.0 else 2.5
    home_edge = (fair.get("home", 33.3) - fair.get("away", 33.3)) / 100.0
    lam_h = total * (0.50 + max(-0.25, min(0.25, home_edge * 0.65)))
    lam_a = max(0.05, total - lam_h)
    return max(0.05, lam_h), max(0.05, lam_a)


def apply_trap_residual_to_matrix(matrix: Dict[str, float], trap_report: Dict[str, Any]) -> Dict[str, float]:
    if not matrix:
        return matrix

    adjusted = dict(matrix)

    direction_adjust = trap_report.get("direction_adjust", {}) if trap_report else {}
    for sc in list(adjusted.keys()):
        d = _score_direction(sc)
        if d in direction_adjust:
            raw = _f(direction_adjust.get(d, 0.0))
            mult = math.exp(max(-0.65, min(0.65, raw * 0.16)))
            adjusted[sc] *= mult

    score_mults = trap_report.get("score_multipliers", {}) if trap_report else {}
    for sc, mult in score_mults.items():
        if sc in adjusted:
            adjusted[sc] *= max(0.15, min(2.2, _f(mult, 1.0)))

    boost_scores = trap_report.get("boost_scores", []) if trap_report else []
    for sc in boost_scores:
        if sc in adjusted:
            adjusted[sc] *= 1.30

    return _normalize_prob_dict(adjusted)


def apply_draw_guard_to_matrix(matrix: Dict[str, float], match_obj: Dict[str, Any], crs_analysis: Dict[str, Any], trap_report: Dict[str, Any]) -> Dict[str, float]:
    dp = trap_report.get("draw_protection") or draw_protection_score(match_obj, crs_analysis)
    nc = trap_report.get("handicap_no_cover") or detect_handicap_no_cover(match_obj)

    if not dp.get("detected") and not nc.get("detected"):
        return matrix

    adjusted = dict(matrix)
    draw_score = int(dp.get("score", 0))
    draw_mult = 1.0

    if dp.get("detected"):
        draw_mult *= min(2.35, 1.0 + draw_score * 0.18)

    if nc.get("detected"):
        draw_mult *= 1.35

    for sc in list(adjusted.keys()):
        d = _score_direction(sc)
        h, a = _parse_score(sc)
        tg = h + a if h is not None else 9

        if d == "draw":
            adjusted[sc] *= draw_mult

        if sc == "1-1":
            adjusted[sc] *= 1.45 if dp.get("detected") else 1.18
        if sc == "0-0":
            adjusted[sc] *= 1.20 if dp.get("detected") else 1.08
        if sc == "2-2" and tg == 4:
            adjusted[sc] *= 1.12

        if nc.get("detected"):
            if sc in nc.get("boost_scores", []):
                adjusted[sc] *= 1.28
            if d == "home" and tg >= 3 and match_obj.get("give_ball") is not None and _f(match_obj.get("give_ball")) <= -1:
                adjusted[sc] *= 0.88

    return _normalize_prob_dict(adjusted)


def build_unified_score_matrix(
    match_obj: Dict[str, Any],
    engine_result: Dict[str, Any],
    crs_analysis: Dict[str, Any],
    trap_report: Dict[str, Any],
    exp_goals: float,
    max_goals: int = 8,
) -> Dict[str, Any]:
    max_goals = max(5, min(10, int(max_goals or 8)))

    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_d = _f(match_obj.get("sp_draw", match_obj.get("same", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))

    fair_pack = fair_probs_from_1x2(sp_h, sp_d, sp_a, method="power")
    fair_1x2_pct = fair_pack["fair_probs"]
    target_dir = {k: v / 100.0 for k, v in fair_1x2_pct.items()}
    ttg = fair_probs_from_ttg(match_obj, method="power")
    lam_h, lam_a = _estimate_base_lambdas(match_obj, engine_result or {}, crs_analysis or {}, exp_goals)

    base = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            sc = f"{h}-{a}"
            base[sc] = _poisson_pmf(lam_h, h) * _poisson_pmf(lam_a, a)
    base = _normalize_prob_dict(base)

    crs_probs_pct = crs_analysis.get("implied_probs", {}) if crs_analysis else {}
    crs_grid = {sc: max(1e-9, crs_probs_pct.get(sc, 0.0) / 100.0) for sc in base}
    crs_mass = sum(crs_grid.values())
    crs_coverage = _f(crs_analysis.get("coverage", 0.0)) if crs_analysis else 0.0

    if crs_mass > 0.05 and crs_coverage >= 0.45:
        crs_grid = _normalize_prob_dict(crs_grid, floor=1e-9)
        crs_weight = min(0.72, 0.35 + crs_coverage * 0.45)
        fused = {}
        for sc in base:
            fused[sc] = (base[sc] ** (1.0 - crs_weight)) * (crs_grid[sc] ** crs_weight)
        matrix = _normalize_prob_dict(fused)
    else:
        matrix = dict(base)

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
            if cur_dir.get(d, 0) > 1e-9:
                ratio = target_dir.get(d, cur_dir[d]) / cur_dir[d]
                matrix[sc] *= max(0.35, min(2.85, ratio))

        matrix = _normalize_prob_dict(matrix)

        if ttg:
            cur_ttg = {}
            for sc, p in matrix.items():
                tg = total_of_score(sc)
                bucket = tg if tg <= 7 else 7
                cur_ttg[bucket] = cur_ttg.get(bucket, 0.0) + p

            for sc in list(matrix.keys()):
                tg = total_of_score(sc)
                bucket = tg if tg <= 7 else 7
                if bucket in ttg and cur_ttg.get(bucket, 0) > 1e-9:
                    ratio = ttg[bucket] / cur_ttg[bucket]
                    matrix[sc] *= max(0.35, min(2.85, ratio))

            matrix = _normalize_prob_dict(matrix)

    matrix = apply_trap_residual_to_matrix(matrix, trap_report)
    matrix = apply_draw_guard_to_matrix(matrix, match_obj, crs_analysis, trap_report)

    dir_probs = {"home": 0.0, "draw": 0.0, "away": 0.0}
    goal_probs = {}

    for sc, p in matrix.items():
        d = direction_of_score(sc)
        tg = total_of_score(sc)
        bucket = tg if tg <= 7 else 7
        dir_probs[d] += p
        goal_probs[bucket] = goal_probs.get(bucket, 0.0) + p

    top_scores = sorted(matrix.items(), key=lambda x: x[1], reverse=True)

    return {
        "matrix": matrix,
        "fair_1x2": fair_pack,
        "direction_probs": {k: round(v * 100, 2) for k, v in dir_probs.items()},
        "goal_probs": {k: round(v * 100, 2) for k, v in sorted(goal_probs.items())},
        "top_scores": [(sc, round(p * 100, 3)) for sc, p in top_scores[:20]],
        "lambda_h": round(lam_h, 3),
        "lambda_a": round(lam_a, 3),
        "source": "unified_score_matrix_v18_2",
    }


def matrix_direction(score_matrix: Dict[str, float]) -> Dict[str, float]:
    out = {"home": 0.0, "draw": 0.0, "away": 0.0}
    for sc, p in score_matrix.items():
        d = _score_direction(sc)
        if d in out:
            out[d] += p
    return {k: round(v * 100, 2) for k, v in out.items()}


def select_score_from_matrix(matrix: Dict[str, float], direction: str, goal_range: Tuple[int, int], ai_votes: Dict[str, float] = None) -> Tuple[str, List[Tuple[str, float]]]:
    ai_votes = ai_votes or {}
    g_min, g_max = goal_range
    candidates = {}

    for sc, p in matrix.items():
        d = _score_direction(sc)
        if d != direction:
            continue
        h, a = _parse_score(sc)
        if h is None:
            continue
        tg = h + a
        if not (g_min <= tg <= g_max):
            continue

        score = p * 100.0
        if sc in ai_votes:
            score *= (1.0 + min(0.20, ai_votes[sc] * 0.03))
        candidates[sc] = score

    if not candidates:
        fallback = {"home": "1-0", "draw": "1-1", "away": "0-1"}.get(direction, "1-1")
        return fallback, [(fallback, 1.0)]

    sorted_scores = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[0][0], [(sc, round(v, 3)) for sc, v in sorted_scores[:10]]


def get_market_odds_for_score(match_obj: Dict[str, Any], score: str) -> float:
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


def confidence_rank_score(dir_probs_pct: Dict[str, float], top_score_candidates: List[Tuple[str, float]], trap_report: Dict[str, Any], ai_valid_count: int) -> Tuple[int, str]:
    vals = sorted([_f(v) for v in dir_probs_pct.values()], reverse=True)
    top = vals[0] if vals else 33.3
    gap = (vals[0] - vals[1]) if len(vals) >= 2 else 0.0

    score = 42.0
    score += min(28.0, (top - 33.3) * 0.75)
    score += min(16.0, gap * 0.85)

    if top_score_candidates:
        top_score_prob = _f(top_score_candidates[0][1])
        score += min(10.0, top_score_prob * 0.9)

    severity = trap_report.get("total_severity", 0) if trap_report else 0
    if severity >= 8:
        score -= 8
    elif severity >= 5:
        score -= 4

    if trap_report.get("draw_protection", {}).get("detected") and gap < 18:
        score -= 3

    if ai_valid_count <= 1:
        score -= 5
    elif ai_valid_count >= 3:
        score += 3

    score = int(max(25, min(92, round(score))))
    risk = "低" if score >= 72 else ("中" if score >= 53 else "高")
    return score, risk


# ====================================================================
# 决策链
# ====================================================================

def determine_goal_range(direction: str, moments: Dict[str, float], exp_goals: float, trap_report: Dict[str, Any], match_obj: Dict[str, Any], engine_result: Dict[str, Any]) -> Tuple[int, int, str]:
    dp = trap_report.get("draw_protection", {})
    if dp.get("detected") and direction == "draw":
        return 0, 4, "draw_guard"

    actual_hc = _f(match_obj.get("give_ball", 0))
    a7 = _f(match_obj.get("a7", 999), 999)
    a6 = _f(match_obj.get("a6", 999), 999)
    a5 = _f(match_obj.get("a5", 999), 999)

    extreme_score = 0
    if 0 < a7 <= 25:
        extreme_score += 2
    elif 0 < a7 <= 35:
        extreme_score += 1
    if 0 < a6 <= 15:
        extreme_score += 2
    elif 0 < a6 <= 20:
        extreme_score += 1
    if 0 < a5 <= 8:
        extreme_score += 2
    elif 0 < a5 <= 12:
        extreme_score += 1

    if direction == "home" and actual_hc <= -1.5:
        extreme_score += 2
    elif direction == "away" and -actual_hc <= -1.5:
        extreme_score += 2

    if direction == "home":
        hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
        axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))
        if hxg - axg > 1.5:
            extreme_score += 1

    if extreme_score >= 5 and exp_goals >= 2.8:
        return 5, 12, "extreme_blowout"

    lt = moments.get("lambda_total", exp_goals) if moments else exp_goals
    lt_avg = lt * 0.6 + exp_goals * 0.4

    league_str = str(match_obj.get("league", match_obj.get("cup", "")))
    if any(kw in league_str for kw in LEAGUE_LOW_GOALS):
        lt_avg -= 0.2
    if any(kw in league_str for kw in LEAGUE_HIGH_GOALS):
        lt_avg += 0.2

    if lt_avg >= 3.5:
        return 3, 6, "shootout"
    if lt_avg >= 2.9:
        return 2, 5, "high_goals"
    if lt_avg >= 2.3:
        return 1, 4, "normal"
    if lt_avg >= 1.8:
        return 1, 3, "low_goals"
    return 0, 2, "grinder"


def decision_lock_chain(match_obj: Dict[str, Any], engine_result: Dict[str, Any], trap_report: Dict[str, Any], crs_analysis: Dict[str, Any], ai_responses: Dict[str, Dict], smart_signals: List[str], exp_goals: float) -> Dict[str, Any]:
    ai_directions = {"home": 0.0, "draw": 0.0, "away": 0.0}
    ai_votes = {}
    ai_weights = {"claude": 1.25, "gemini": 0.85, "grok": 0.85, "gpt": 0.85}

    for name, r in ai_responses.items():
        if not isinstance(r, dict):
            continue

        sc_raw = str(r.get("ai_score", "")).strip()
        top3 = r.get("top3", [])
        sc = sc_raw
        h0, a0 = _parse_score(sc)

        if h0 is None:
            if top3:
                if isinstance(top3[0], dict):
                    sc = str(top3[0].get("score", "")).strip()
                elif isinstance(top3[0], str):
                    sc = top3[0].strip()

        h, a = _parse_score(sc)
        if h is None:
            continue

        weight = ai_weights.get(name, 0.8)

        if h > a:
            ai_directions["home"] += weight
        elif h < a:
            ai_directions["away"] += weight
        else:
            ai_directions["draw"] += weight

        sc_clean = sc.replace(" ", "").strip()
        ai_votes[sc_clean] = ai_votes.get(sc_clean, 0.0) + weight

        for rank, t in enumerate(top3[1:3], 2):
            if isinstance(t, dict):
                sc2 = str(t.get("score", "")).replace(" ", "").strip()
            elif isinstance(t, str):
                sc2 = t.replace(" ", "").strip()
            else:
                continue

            h2, a2 = _parse_score(sc2)
            if h2 is not None:
                w2 = 0.25 if rank == 2 else 0.12
                ai_votes[sc2] = ai_votes.get(sc2, 0.0) + weight * w2

    unified = build_unified_score_matrix(
        match_obj=match_obj,
        engine_result=engine_result or {},
        crs_analysis=crs_analysis or {},
        trap_report=trap_report or {},
        exp_goals=exp_goals,
        max_goals=8,
    )

    matrix = unified["matrix"]
    posterior = unified["direction_probs"]

    ai_total = sum(ai_directions.values())
    if ai_total > 0:
        ai_share = {k: ai_directions.get(k, 0.0) / ai_total for k in VALID_DIRS}
        mat_logits = {k: math.log(max(0.01, posterior.get(k, 33.3) / 100.0)) for k in VALID_DIRS}

        for d in VALID_DIRS:
            delta = math.log(max(0.05, ai_share.get(d, 0.0)) / (1.0 / 3.0))
            mat_logits[d] += max(-0.16, min(0.16, delta * 0.16))

        posterior_prob = _softmax_dict(mat_logits, temperature=1.05)
        posterior = {k: round(v * 100, 2) for k, v in posterior_prob.items()}

        cur_dir = matrix_direction(matrix)
        for sc in list(matrix.keys()):
            d = _score_direction(sc)
            if d in VALID_DIRS and cur_dir.get(d, 0) > 0:
                ratio = (posterior[d] / 100.0) / (cur_dir[d] / 100.0)
                matrix[sc] *= max(0.70, min(1.45, ratio))

        matrix = _normalize_prob_dict(matrix)
        matrix = apply_draw_guard_to_matrix(matrix, match_obj, crs_analysis, trap_report)
        posterior = matrix_direction(matrix)

    final_direction = max(posterior, key=posterior.get)

    # 平局保护二次门槛：如果主胜领先但平局保护强，且平局已达到合理阈值，允许切平。
    dp = trap_report.get("draw_protection", {})
    if dp.get("detected"):
        if final_direction == "home" and posterior.get("draw", 0) >= 26 and posterior.get("home", 0) - posterior.get("draw", 0) <= 22:
            final_direction = "draw"
        elif final_direction == "away" and posterior.get("draw", 0) >= 26 and posterior.get("away", 0) - posterior.get("draw", 0) <= 22:
            final_direction = "draw"

    sorted_p = sorted(posterior.values(), reverse=True)
    dir_confidence = round(posterior.get(final_direction, sorted_p[0]), 1)
    dir_gap = round(sorted_p[0] - sorted_p[1], 1) if len(sorted_p) >= 2 else 0.0

    goal_range_min, goal_range_max, scenario = determine_goal_range(
        direction=final_direction,
        moments=crs_analysis.get("moments", {}),
        exp_goals=exp_goals,
        trap_report=trap_report,
        match_obj=match_obj,
        engine_result=engine_result,
    )

    best_score, top_candidates = select_score_from_matrix(
        matrix=matrix,
        direction=final_direction,
        goal_range=(goal_range_min, goal_range_max),
        ai_votes=ai_votes,
    )

    if top_candidates and _f(top_candidates[0][1]) < 0.5:
        for sc, p in unified["top_scores"]:
            if _score_direction(sc) == final_direction:
                best_score = sc
                break

    h, a = _parse_score(best_score)
    if h is None:
        final_direction_lock = final_direction
    elif h > a:
        final_direction_lock = "home"
    elif h < a:
        final_direction_lock = "away"
    else:
        final_direction_lock = "draw"

    if final_direction_lock != final_direction:
        for sc, p in sorted(matrix.items(), key=lambda x: x[1], reverse=True):
            if _score_direction(sc) == final_direction:
                best_score = sc
                final_direction_lock = final_direction
                break

    result_cn = _direction_cn(final_direction_lock)
    is_score_others = best_score in ALL_SCORE_OTHERS or "其他" in str(best_score)
    display_label = best_score

    if is_score_others:
        display_label = {"home": "胜其他", "draw": "平其他", "away": "负其他"}[final_direction_lock]

    evidences = []
    fair_pack = unified.get("fair_1x2", {})
    evidences.append(
        f"公平1X2({fair_pack.get('method','power')}): 主{posterior.get('home',0):.1f}% 平{posterior.get('draw',0):.1f}% 客{posterior.get('away',0):.1f}%"
    )
    evidences.append(
        f"统一比分矩阵 λ主{unified.get('lambda_h')} / λ客{unified.get('lambda_a')}，top={best_score}"
    )

    if trap_report.get("draw_protection", {}).get("detected"):
        evidences.append("平局保护:" + " + ".join(trap_report["draw_protection"].get("reasons", [])[:5]))

    if trap_report.get("handicap_no_cover", {}).get("detected"):
        evidences.append("让球不穿:" + trap_report["handicap_no_cover"].get("description", ""))

    if trap_report.get("odds_field_warning"):
        evidences.append("字段警告:" + trap_report.get("odds_field_warning", ""))

    if trap_report.get("trap_count", 0):
        evidences.append(f"陷阱残差修正:{trap_report.get('trap_count')}个，严重度{trap_report.get('total_severity',0)}")

    if ai_total > 0:
        evidences.append(f"AI residual:{ai_directions}")

    return {
        "predicted_score": best_score,
        "predicted_label": display_label,
        "result": result_cn,
        "display_direction": result_cn,
        "final_direction": final_direction_lock,
        "is_score_others": is_score_others,

        "home_win_pct": round(posterior["home"], 2),
        "draw_pct": round(posterior["draw"], 2),
        "away_win_pct": round(posterior["away"], 2),

        "scenario": scenario,
        "goal_range": (goal_range_min, goal_range_max),
        "dir_confidence": dir_confidence,
        "dir_gap": dir_gap,
        "evidences": evidences,
        "override_triggered": False,
        "top_score_candidates": top_candidates,
        "bayesian_prior": trap_report.get("fair_1x2", trap_report.get("shin", {})),

        "unified_matrix_top_scores": unified.get("top_scores", []),
        "unified_goal_probs": unified.get("goal_probs", {}),
        "unified_source": unified.get("source"),
        "fair_1x2_pack": unified.get("fair_1x2", {}),
    }


# ====================================================================
# AI Prompt
# ====================================================================

def load_ai_diary():
    return {"yesterday_win_rate": "N/A", "reflection": "", "kill_history": []}


def save_ai_diary(diary):
    return


def build_v18_prompt(match_analyses):
    p = "<context>\n"
    p += "你正在中国体彩竞彩足球市场进行量化比分预测。\n"
    p += "任务是识别赔率结构、资金流、CRS、总进球赔率之间的定价裂缝。\n"
    p += "你是初审模型，最终会由Claude接收三家结论再审计。\n"
    p += "</context>\n\n"

    p += "<iron_rules>\n"
    p += "铁律1: top3[0].score 必须与 final_direction 一致。\n"
    p += "铁律2: 0-1、0-2、0-3 是合法客胜比分，禁止当成无效比分。\n"
    p += "铁律3: CRS 1-1低赔、平/平低赔、2球低赔、λ接近时，必须防平。\n"
    p += "铁律4: 主让深但让负低赔时，代表不穿盘风险，不能机械加强主胜。\n"
    p += "铁律5: 没有外部伤停/新闻字段时，不得编造信息。\n"
    p += "</iron_rules>\n\n"

    p += "<output_format>\n"
    p += "严格 JSON 数组。每场必须包含 match, top3, reason, ai_confidence, is_score_others, detected_traps, final_direction。\n"
    p += "禁止 JSON 之外任何文本。\n"
    p += "</output_format>\n\n"

    p += "<match_data>\n"

    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        eng = ma.get("engine", {})
        stats = ma.get("stats", {})
        trap_preview = ma.get("trap_preview", {})
        crs_preview = ma.get("crs_preview", {})

        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        league = m.get("league", m.get("cup", ""))
        hc = m.get("give_ball", "0")

        sp_h = _f(m.get("sp_home", m.get("win", 0)))
        sp_d = _f(m.get("sp_draw", m.get("same", 0)))
        sp_a = _f(m.get("sp_away", m.get("lose", 0)))

        p += f"<match index=\"{i+1}\">\n"
        p += f"[{i+1}] {h} vs {a} | {league}\n"
        p += f"欧赔: {sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f} | 让球: {hc}\n"

        fair = trap_preview.get("fair_1x2", trap_preview.get("shin", {}))
        if fair:
            p += f"公平1X2: 主{fair.get('home',0):.1f}% 平{fair.get('draw',0):.1f}% 客{fair.get('away',0):.1f}% | method={trap_preview.get('fair_1x2_method','power')}\n"

        if trap_preview.get("odds_field_warning"):
            p += f"字段警告: {trap_preview.get('odds_field_warning')}\n"

        hxg = eng.get("bookmaker_implied_home_xg", "?")
        axg = eng.get("bookmaker_implied_away_xg", "?")
        p += f"庄家隐含xG: 主{hxg} vs 客{axg}\n"

        moments = crs_preview.get("moments", {})
        if moments:
            p += (
                f"CRS矩: λ主{moments.get('lambda_h',0):.2f}/客{moments.get('lambda_a',0):.2f} "
                f"总{moments.get('lambda_total',0):.2f} corr{moments.get('corr',0):+.2f} "
                f"形状={crs_preview.get('shape_verdict','?')}\n"
            )

        dp = trap_preview.get("draw_protection", {})
        if dp.get("detected"):
            p += "平局保护触发: " + " + ".join(dp.get("reasons", [])[:6]) + "\n"

        nc = trap_preview.get("handicap_no_cover", {})
        if nc.get("detected"):
            p += "让球不穿触发: " + nc.get("description", "") + "\n"

        traps = trap_preview.get("traps_detected", [])
        if traps:
            p += f"系统识别陷阱({len(traps)}个,严重度{trap_preview.get('total_severity',0)}):\n"
            for t in traps:
                p += f"  - {t.get('trap','?')}: {t.get('description','')[:160]}\n"

        a_list = []
        compressed = []
        for g in range(8):
            v = m.get(f"a{g}", "")
            a_list.append(f"{g}={v}")
            actual = _f(v)
            if actual > 1:
                std = STANDARD_GOAL_ODDS.get(g, 50)
                ratio = std / actual
                if ratio > 1.5:
                    compressed.append(f"{g}球(压低{ratio:.1f}x)")
        p += f"总进球: {' | '.join(a_list)}\n"
        if compressed:
            p += f"进球数压低: {', '.join(compressed)}\n"

        rq_win = m.get("rq_win", "")
        rq_draw = m.get("rq_draw", "")
        rq_lose = m.get("rq_lose", "")
        if rq_win or rq_draw or rq_lose:
            p += f"让球胜平负: 让胜{rq_win} 让平{rq_draw} 让负{rq_lose}\n"

        crs_lines = []
        for sc, key in CRS_FULL_MAP.items():
            odds = _f(m.get(key, 0))
            if odds > 1:
                crs_lines.append(f"{sc}={odds:.1f}")
        if crs_lines:
            p += f"CRS: {' | '.join(crs_lines)}\n"

        hf_l = []
        for k, lb in HFTF_MAP.items():
            v = _f(m.get(k, 0))
            if v > 1:
                hf_l.append(f"{lb}={v:.2f}")
        if hf_l:
            p += f"半全场: {' | '.join(hf_l)}\n"

        vote = m.get("vote", {})
        if vote:
            p += f"散户: 胜{vote.get('win','?')}% 平{vote.get('same','?')}% 负{vote.get('lose','?')}%\n"

        change = m.get("change", {})
        if change and isinstance(change, dict):
            cw = change.get("win", 0)
            cs = change.get("same", 0)
            cl = change.get("lose", 0)
            if cw or cs or cl:
                p += f"赔率变动: 胜{cw} 平{cs} 负{cl}，负数=赔率下降\n"

        points = m.get("points", {})
        if isinstance(points, dict):
            for k in ["home_strength", "guest_strength", "match_points"]:
                txt = str(points.get(k, ""))[:700].replace("\n", " ")
                if txt:
                    p += f"情报: {txt}\n"

        smart_sigs = stats.get("smart_signals", []) if isinstance(stats, dict) else []
        if smart_sigs:
            p += f"信号: {', '.join(str(s) for s in smart_sigs[:10])}\n"

        p += "</match>\n\n"

    p += "</match_data>\n"
    return p


def build_claude_final_audit_prompt(match_analyses, phase1_results):
    base_prompt = build_v18_prompt(match_analyses)
    p = "<final_audit_context>\n"
    p += "你是最终审计模型。你将看到原始数据和 GPT/Grok/Gemini 初审结论。\n"
    p += "不能按票数裁决，必须重新审计赔率结构、CRS、总进球、让球不穿、平局保护。\n"
    p += "当 CRS 1-1低赔、半全场平/平低赔、2球低赔、λ接近时，必须认真防平。\n"
    p += "主让深但让负低赔时，不可机械判主胜大穿。\n"
    p += "</final_audit_context>\n\n"
    p += base_prompt
    p += "\n\n<phase1_ai_results>\n"

    for ai_name in ["gpt", "grok", "gemini"]:
        p += f"<{ai_name}>\n"
        results = phase1_results.get(ai_name, {})
        if not results:
            p += "无有效结果\n"
        else:
            for idx in range(1, len(match_analyses) + 1):
                r = results.get(idx, {})
                if not r:
                    p += f"[{idx}] 弃权\n"
                    continue
                p += json.dumps({
                    "match": idx,
                    "ai_score": r.get("ai_score"),
                    "top3": r.get("top3", []),
                    "final_direction": r.get("final_direction", ""),
                    "ai_confidence": r.get("ai_confidence", 60),
                    "detected_traps": r.get("detected_traps", []),
                    "reason": r.get("reason", ""),
                }, ensure_ascii=False)
                p += "\n"
        p += f"</{ai_name}>\n\n"

    p += "</phase1_ai_results>\n"
    p += "<final_output_rule>严格输出 JSON 数组，禁止 JSON 外文本。</final_output_rule>\n"
    return p


# ====================================================================
# AI 调用
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


def get_clean_env_url(name, default=""):
    v = str(os.environ.get(name, globals().get(name, default))).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/?#=&%-]+)", v)
    return match.group(1) if match else v


def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")


async def async_call_one_ai_batch(session, prompt, url_env, key_env, models_list, num_matches, ai_name):
    key = get_clean_env_key(key_env)
    if not key and ai_name == "gpt":
        key = GPT_DEFAULT_KEY
    if not key:
        return ai_name, {}, "no_key"

    primary_url = get_clean_env_url(url_env, GPT_DEFAULT_URL if ai_name == "gpt" else "")
    if ai_name == "gpt":
        urls = [primary_url or GPT_DEFAULT_URL]
    else:
        backup = [u for u in FALLBACK_URLS if u and u != primary_url][:1]
        urls = [primary_url] + backup

    CONNECT_TIMEOUT = 20
    READ_TIMEOUT_MAP = {"claude": 480, "grok": 380, "gpt": 380, "gemini": 380}
    READ_TIMEOUT = READ_TIMEOUT_MAP.get(ai_name, 300)

    AI_PROFILES = {
        "claude": {
            "sys": "<role>你是最终审计模型。</role><instruction>只输出JSON数组，禁止前缀后缀。</instruction>",
            "temp": 0.18,
        },
        "gpt": {
            "sys": "<role>你是比分分布量化策略师。</role><instruction>严格输出JSON数组。</instruction>",
            "temp": 0.18,
        },
        "grok": {
            "sys": "<role>你是市场情绪和资金流分析师。</role><instruction>严格输出JSON数组。</instruction>",
            "temp": 0.25,
        },
        "gemini": {
            "sys": "<role>你是多市场共振识别模型。</role><instruction>严格输出JSON数组。</instruction>",
            "temp": 0.15,
        },
    }

    profile = AI_PROFILES.get(ai_name, AI_PROFILES["gpt"])

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
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": profile["temp"]},
                    "systemInstruction": {"parts": [{"text": profile["sys"]}]},
                }
            else:
                headers["Authorization"] = f"Bearer {key}"
                payload = {
                    "model": mn,
                    "messages": [
                        {"role": "system", "content": profile["sys"]},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": profile["temp"],
                }

            gw = url.split("/v1")[0][:45]
            print(f"  [连接中] {ai_name.upper()} | {mn[:28]} @ {gw}")
            t0 = time.time()

            try:
                timeout = aiohttp.ClientTimeout(
                    total=None,
                    connect=CONNECT_TIMEOUT,
                    sock_connect=CONNECT_TIMEOUT,
                    sock_read=READ_TIMEOUT,
                )
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                    elapsed_connect = round(time.time() - t0, 1)

                    if r.status in (502, 504):
                        print(f"    HTTP {r.status} | {elapsed_connect}s → 换URL")
                        continue
                    if r.status == 400:
                        print(f"    HTTP 400 | {elapsed_connect}s → 换模型")
                        break
                    if r.status == 429:
                        print(f"    HTTP 429 | {elapsed_connect}s → 换URL")
                        await asyncio.sleep(1)
                        continue
                    if r.status != 200:
                        print(f"    HTTP {r.status} | {elapsed_connect}s → 换URL")
                        continue

                    connected = True
                    print(f"    已连上 {elapsed_connect}s | 等待数据...")

                    try:
                        data = await r.json(content_type=None)
                    except Exception:
                        text = await r.text()
                        _save_debug_dump(ai_name, {"raw": text}, "non_json")
                        break

                    raw_text = _extract_response_text(data, is_gem, ai_name)
                    if not raw_text or len(raw_text) < 10:
                        _save_debug_dump(ai_name, data, "empty")
                        break

                    results = _parse_ai_json(raw_text, num_matches)
                    elapsed = round(time.time() - t0, 1)

                    if len(results) > 0:
                        print(f"    {ai_name.upper()} 完成: {len(results)}/{num_matches} | {elapsed}s")
                        return ai_name, results, mn

                    _save_debug_dump(ai_name, data, "parse0")
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
                    print(f"    {str(e)[:80]} → 换URL")
                    continue
                print(f"    调用异常: {str(e)[:120]}")
                return ai_name, {}, "error"

            await asyncio.sleep(0.2)

    return ai_name, {}, "all_failed"


def _extract_response_text(data, is_gem, ai_name):
    raw_text = ""
    try:
        if is_gem:
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            if data.get("choices"):
                msg = data["choices"][0].get("message", {})
                if isinstance(msg, dict):
                    content_val = msg.get("content", "")
                    if content_val:
                        if isinstance(content_val, str) and content_val.strip():
                            raw_text = content_val.strip()
                        elif isinstance(content_val, list):
                            for item in content_val:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    t = item.get("text", "").strip()
                                    if t and len(t) > len(raw_text):
                                        raw_text = t

                    if not raw_text:
                        for field in [
                            "text", "answer", "response", "output_text", "final_answer",
                            "output", "result", "completion", "message_content",
                            "assistant_content", "model_response",
                        ]:
                            v = msg.get(field, "")
                            if v and isinstance(v, str) and v.strip():
                                raw_text = v.strip()
                                break

                    if not raw_text:
                        skip = (
                            "reasoning_content", "thinking", "reasoning", "reasoning_text",
                            "thoughts", "thought_process", "internal_thinking",
                            "chain_of_thought", "cot", "deliberation", "analysis_process",
                        )
                        best = ""
                        for k, v in msg.items():
                            if k in skip:
                                continue
                            if isinstance(v, str) and '"match"' in v and "[" in v and len(v) > len(best):
                                best = v.strip()
                        if best:
                            raw_text = best

            if not raw_text and data.get("output") and isinstance(data["output"], list):
                for out_item in data["output"]:
                    if isinstance(out_item, dict) and out_item.get("type") == "message":
                        for ct in out_item.get("content", []):
                            if isinstance(ct, dict) and ct.get("text"):
                                t = ct["text"].strip()
                                if len(t) > len(raw_text):
                                    raw_text = t

            if not raw_text:
                full_str = json.dumps(data, ensure_ascii=False)
                m_match = re.search(r'\[\s*\{\s*\\?"match\\?"', full_str)
                if m_match:
                    start_pos = m_match.start()
                    depth = 0
                    end_pos = start_pos
                    for ci in range(start_pos, min(start_pos + 200000, len(full_str))):
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
        print(f"    响应解析异常: {str(ex)[:100]}")
    return raw_text


def _parse_ai_json(raw_text, num_matches):
    clean = raw_text
    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"```[\w]*", "", clean).strip()

    json_str = ""
    m_re = re.search(r'\[\s*\{\s*"match"', clean)

    if m_re:
        start_idx = m_re.start()
        depth = 0
        end_idx = start_idx
        for i in range(start_idx, len(clean)):
            if clean[i] == "[":
                depth += 1
            elif clean[i] == "]":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        if end_idx > start_idx:
            json_str = clean[start_idx:end_idx]

    if not json_str:
        start = clean.find("[")
        end = clean.rfind("]") + 1
        if start != -1 and end > start:
            json_str = clean[start:end]

    results = {}
    if json_str:
        try:
            arr = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                last_brace = json_str.rfind("}")
                arr = json.loads(json_str[:last_brace + 1] + "]") if last_brace != -1 else []
            except Exception:
                arr = []

        if isinstance(arr, list):
            for item in arr:
                if not isinstance(item, dict) or not item.get("match"):
                    continue
                try:
                    mid = int(item["match"])
                except Exception:
                    continue

                if mid < 1 or mid > max(num_matches, 9999):
                    continue

                top3 = item.get("top3", [])
                if top3 and isinstance(top3, list):
                    first = top3[0]
                    if isinstance(first, dict):
                        t1 = str(first.get("score", "1-1")).replace(" ", "").strip()
                    elif isinstance(first, str):
                        t1 = first.replace(" ", "").strip()
                    else:
                        t1 = "1-1"

                    results[mid] = {
                        "top3": top3,
                        "ai_score": t1,
                        "reason": str(item.get("reason", "")),
                        "ai_confidence": int(_f(item.get("ai_confidence", 60), 60)),
                        "is_score_others": bool(item.get("is_score_others", False)),
                        "detected_traps": item.get("detected_traps", []),
                        "final_direction": item.get("final_direction", ""),
                    }
                elif item.get("score"):
                    results[mid] = {
                        "top3": [{"score": str(item["score"]).replace(" ", "").strip(), "prob": 0}],
                        "ai_score": str(item["score"]).replace(" ", "").strip(),
                        "reason": str(item.get("reason", "")),
                        "ai_confidence": int(_f(item.get("ai_confidence", 60), 60)),
                        "is_score_others": bool(item.get("is_score_others", False)),
                        "detected_traps": item.get("detected_traps", []),
                        "final_direction": item.get("final_direction", ""),
                    }

    return results


def _save_debug_dump(ai_name, data, tag):
    try:
        os.makedirs("data/debug", exist_ok=True)
        dump_file = f"data/debug/{ai_name}_{tag}_{int(time.time())}.json"
        with open(dump_file, "w", encoding="utf-8") as df:
            json.dump(data, df, ensure_ascii=False, indent=2)
        print(f"    失败响应已保存: {dump_file}")
    except Exception:
        pass


async def run_ai_matrix_two_phase(match_analyses):
    num = len(match_analyses)
    prompt = build_v18_prompt(match_analyses)

    print(f"  [v18.2 Phase1 Prompt] {len(prompt):,} 字符 → GPT/Grok/Gemini 初审...")

    phase1_configs = [
        ("grok", "GROK_API_URL", "GROK_API_KEY", ["熊猫-A-5-grok-4.2-fast-200w上下文"]),
        ("gpt", "GPT_API_URL", "GPT_API_KEY", ["gpt-5.4"]),
        ("gemini", "GEMINI_API_URL", "GEMINI_API_KEY", ["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking"]),
    ]

    all_results = {"gpt": {}, "grok": {}, "gemini": {}, "claude": {}}

    connector = aiohttp.TCPConnector(limit=8, use_dns_cache=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            async_call_one_ai_batch(session, prompt, u, k, m, num, n)
            for n, u, k, m in phase1_configs
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
        print(f"  [v18.2 Claude Audit] {len(audit_prompt):,} 字符 → Claude终审...")

        _, claude_result, _ = await async_call_one_ai_batch(
            session=session,
            prompt=audit_prompt,
            url_env="CLAUDE_API_URL",
            key_env="CLAUDE_API_KEY",
            models_list=["熊猫-69-满血openrouter-claude-opus-4.7-上下文1000k"],
            num_matches=num,
            ai_name="claude",
        )

        all_results["claude"] = claude_result or {}

    ok = sum(1 for v in all_results.values() if v)
    print(f"  [完成] {ok}/4 AI有数据 | 架构=3初审+Claude终审")
    return all_results


# ====================================================================
# merge_result
# ====================================================================

def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    match_obj = normalize_match(match_obj)

    def _is_valid_ai(r):
        if not isinstance(r, dict):
            return False
        score = r.get("ai_score", "")
        if not score or score in ("-", "N/A", ""):
            return False
        h, a = _parse_score(score)
        return h is not None

    ai_valid = {
        "gpt": _is_valid_ai(gpt_r),
        "grok": _is_valid_ai(grok_r),
        "gemini": _is_valid_ai(gemini_r),
        "claude": _is_valid_ai(claude_r),
    }

    abstained = [n.upper() for n, v in ai_valid.items() if not v]
    if abstained:
        print(f"    弃权AI: {', '.join(abstained)}")

    ai_responses = {}
    if ai_valid["claude"]:
        ai_responses["claude"] = claude_r
    if ai_valid["gpt"]:
        ai_responses["gpt"] = gpt_r
    if ai_valid["grok"]:
        ai_responses["grok"] = grok_r
    if ai_valid["gemini"]:
        ai_responses["gemini"] = gemini_r

    exp_goals = 0.0
    for src in [engine_result, stats]:
        if not isinstance(src, dict):
            continue
        for k in ["expected_total_goals", "exp_goals", "total_goals", "expected_goals", "lambda_total", "total_xg"]:
            v = src.get(k)
            if v is not None:
                fv = _f(v)
                if fv > 0.5:
                    exp_goals = fv
                    break
        if exp_goals > 0:
            break

    if exp_goals <= 0:
        hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0)) if isinstance(engine_result, dict) else 0
        axg = _f(engine_result.get("bookmaker_implied_away_xg", 0)) if isinstance(engine_result, dict) else 0
        if hxg > 0 and axg > 0:
            exp_goals = hxg + axg
            print(f"    期望进球用xG总和: {hxg:.2f}+{axg:.2f}={exp_goals:.2f}")

    if exp_goals <= 0:
        try:
            gp = []
            for gi in range(8):
                v = _f(match_obj.get(f"a{gi}", 0))
                if v > 1:
                    gp.append((gi, 1 / v))
            if gp:
                tp = sum(p for _, p in gp)
                exp_goals = sum(g * (p / tp) for g, p in gp)
                print(f"    期望进球用a0-a7反推: {exp_goals:.2f}")
        except Exception:
            pass

    if exp_goals < 1.0 or exp_goals > 6.0:
        print(f"    期望进球异常({exp_goals:.2f}),使用默认2.5")
        exp_goals = 2.5

    smart_signals = stats.get("smart_signals", []) if isinstance(stats, dict) else []

    trap_report = detect_all_traps(match_obj, engine_result or {}, ai_responses, smart_signals, exp_goals)

    if trap_report.get("odds_field_warning"):
        print(f"    字段警告: {trap_report['odds_field_warning']}")

    if trap_report.get("draw_protection", {}).get("detected"):
        print(f"    平局保护: {trap_report['draw_protection']['score']}分 | {' + '.join(trap_report['draw_protection']['reasons'][:5])}")

    if trap_report.get("handicap_no_cover", {}).get("detected"):
        print(f"    让球不穿: {trap_report['handicap_no_cover']['description']}")

    if trap_report["trap_count"] > 0:
        print(f"    陷阱: {trap_report['trap_count']}个 严重度{trap_report['total_severity']}")
        for t in trap_report["traps_detected"][:5]:
            print(f"       [{t['trap']}] {t['description'][:100]}")

    crs_analysis = analyze_crs_matrix(match_obj)
    if crs_analysis["coverage"] > 0:
        print(f"    CRS: 覆盖{crs_analysis['coverage'] * 100:.0f}% 形状={crs_analysis['shape_verdict']}")

    lock_result = decision_lock_chain(
        match_obj=match_obj,
        engine_result=engine_result or {},
        trap_report=trap_report,
        crs_analysis=crs_analysis,
        ai_responses=ai_responses,
        smart_signals=smart_signals,
        exp_goals=exp_goals,
    )

    print(
        f"    方向: 主{lock_result['home_win_pct']:.0f}% "
        f"平{lock_result['draw_pct']:.0f}% "
        f"客{lock_result['away_win_pct']:.0f}%"
    )
    for ev in lock_result["evidences"][:5]:
        print(f"       - {ev}")

    predicted_score = lock_result["predicted_score"]
    predicted_label = lock_result["predicted_label"]
    result_cn = lock_result["result"]
    display_direction = lock_result["display_direction"]
    final_direction = lock_result["final_direction"]

    home_win_pct = lock_result["home_win_pct"]
    draw_pct = lock_result["draw_pct"]
    away_win_pct = lock_result["away_win_pct"]

    final_odds = get_market_odds_for_score(match_obj, predicted_score)

    model_prob_pct = 0.0
    for sc, p in lock_result.get("top_score_candidates", []):
        if sc == predicted_score:
            model_prob_pct = _f(p)
            break

    if model_prob_pct <= 0:
        for sc, p in lock_result.get("unified_matrix_top_scores", []):
            if sc == predicted_score:
                model_prob_pct = _f(p)
                break

    if model_prob_pct <= 0:
        model_prob_pct = 1.0

    market_implied_pct = round((1.0 / final_odds) * 100.0, 3) if final_odds > 1.05 else None

    ev_data = calculate_independent_ev(
        model_prob_pct=model_prob_pct,
        market_odds=final_odds,
        market_implied_pct=market_implied_pct,
    )

    weights = {"claude": 1.15, "gemini": 1.0, "grok": 1.0, "gpt": 1.0}
    ai_conf_sum = 0.0
    ai_conf_count = 0.0
    value_kills = 0

    for name, r in ai_responses.items():
        conf = _f(r.get("ai_confidence", 60), 60)
        ai_conf_sum += conf * weights.get(name, 1.0)
        ai_conf_count += weights.get(name, 1.0)
        if r.get("value_kill"):
            value_kills += 1

    avg_ai_conf = (ai_conf_sum / ai_conf_count) if ai_conf_count > 0 else 60.0

    cf, risk = confidence_rank_score(
        dir_probs_pct={"home": home_win_pct, "draw": draw_pct, "away": away_win_pct},
        top_score_candidates=lock_result.get("top_score_candidates", []),
        trap_report=trap_report,
        ai_valid_count=len(ai_responses),
    )

    cold_strength = 0
    cold_level = None
    cold_signals_arr = []

    for t in trap_report["traps_detected"]:
        if t["trap"] in ["T8_FALSE_COLD", "T4_FAKE_HOME_FAVORITE", "T5_FAKE_AWAY_FAVORITE", "T14_CUP_FAVORITE"]:
            cold_strength += t["severity"] * 3
            cold_signals_arr.append(t["description"])

    if cold_strength >= 12:
        cold_level = "顶级"
    elif cold_strength >= 7:
        cold_level = "高危"
    elif cold_strength >= 4:
        cold_level = "中等"

    cold_door = {
        "is_cold_door": cold_level is not None,
        "strength": cold_strength,
        "level": cold_level or "普通",
        "signals": cold_signals_arr,
        "sharp_confirmed": trap_report.get("sharp_detected", False),
        "dark_verdict": f"{cold_level}冷门!{len(cold_signals_arr)}条触发" if cold_level else "",
    }

    sigs = list(smart_signals)
    for t in trap_report["traps_detected"]:
        sigs.append(f"{t['trap']}:{t['description'][:70]}")
    if trap_report.get("draw_protection", {}).get("detected"):
        sigs.append("平局保护:" + " + ".join(trap_report["draw_protection"].get("reasons", [])[:4]))
    if trap_report.get("handicap_no_cover", {}).get("detected"):
        sigs.append("让球不穿:" + trap_report["handicap_no_cover"].get("description", ""))

    cl_raw = claude_r.get("ai_score", "") if isinstance(claude_r, dict) else ""
    cl_h, cl_a = _parse_score(cl_raw)
    cl_sc = cl_raw if cl_h is not None else predicted_score

    return {
        "predicted_score": predicted_score,
        "predicted_label": predicted_label,
        "result": result_cn,
        "display_direction": display_direction,
        "final_direction": final_direction,
        "is_score_others": lock_result["is_score_others"],

        "home_win_pct": round(home_win_pct, 1),
        "draw_pct": round(draw_pct, 1),
        "away_win_pct": round(away_win_pct, 1),

        "confidence": cf,
        "confidence_meaning": "排序置信度，不等于历史命中率",
        "risk_level": risk,
        "dir_confidence": lock_result["dir_confidence"],
        "dir_gap": lock_result["dir_gap"],

        "scenario": lock_result["scenario"],
        "goal_range": lock_result["goal_range"],

        "bayesian_evidences": lock_result["evidences"],
        "bayesian_prior": lock_result["bayesian_prior"],
        "override_triggered": lock_result["override_triggered"],

        "traps_detected": [t["trap"] for t in trap_report["traps_detected"]],
        "trap_count": trap_report["trap_count"],
        "trap_severity": trap_report["total_severity"],
        "trap_details": [{"trap": t["trap"], "desc": t["description"]} for t in trap_report["traps_detected"]],

        "draw_protection": trap_report.get("draw_protection", {}),
        "handicap_no_cover": trap_report.get("handicap_no_cover", {}),
        "odds_field_warning": trap_report.get("odds_field_warning", ""),

        "fair_1x2": trap_report.get("fair_1x2", {}),
        "fair_1x2_method": trap_report.get("fair_1x2_method", "power"),
        "market_overround": trap_report.get("market_overround", 0.0),
        "raw_implied_1x2": trap_report.get("raw_implied_1x2", {}),

        "crs_shape": crs_analysis.get("shape_verdict", "unknown"),
        "crs_moments": crs_analysis.get("moments", {}),
        "crs_margin": crs_analysis.get("margin", 0.0),
        "crs_coverage": crs_analysis.get("coverage", 0.0),
        "crs_implied_probs": crs_analysis.get("implied_probs", {}),
        "top_score_candidates": lock_result["top_score_candidates"],

        "unified_matrix_top_scores": lock_result.get("unified_matrix_top_scores", []),
        "unified_goal_probs": lock_result.get("unified_goal_probs", {}),
        "fair_1x2_pack": lock_result.get("fair_1x2_pack", {}),
        "unified_source": lock_result.get("unified_source"),

        "gpt_score": gpt_r.get("ai_score", "弃权") if ai_valid["gpt"] else "弃权",
        "gpt_analysis": gpt_r.get("reason", gpt_r.get("analysis", "弃权")) if ai_valid["gpt"] else "弃权 (AI失效)",
        "grok_score": grok_r.get("ai_score", "弃权") if ai_valid["grok"] else "弃权",
        "grok_analysis": grok_r.get("reason", grok_r.get("analysis", "弃权")) if ai_valid["grok"] else "弃权 (AI失效)",
        "gemini_score": gemini_r.get("ai_score", "弃权") if ai_valid["gemini"] else "弃权",
        "gemini_analysis": gemini_r.get("reason", gemini_r.get("analysis", "弃权")) if ai_valid["gemini"] else "弃权 (AI失效)",
        "claude_score": cl_sc if ai_valid["claude"] else "弃权",
        "claude_analysis": claude_r.get("reason", claude_r.get("analysis", "弃权")) if ai_valid["claude"] else "弃权 (AI失效)",

        "ai_abstained": [n.upper() for n, v in ai_valid.items() if not v],
        "ai_avg_confidence": round(avg_ai_conf, 1),
        "value_kill_count": value_kills,

        "suggested_kelly": ev_data["kelly"],
        "edge_vs_market": ev_data["ev"],
        "is_value": ev_data["is_value"],
        "ev_note": ev_data.get("note", ""),
        "score_model_prob": round(model_prob_pct, 3),
        "score_market_odds": final_odds,
        "score_market_implied_pct": market_implied_pct,

        "smart_money_signal": " | ".join(sigs[:14]),
        "smart_signals": sigs,

        "cold_door": cold_door,

        "xG_home": round(_f(engine_result.get("bookmaker_implied_home_xg", 1.3)) if isinstance(engine_result, dict) else 1.3, 2),
        "xG_away": round(_f(engine_result.get("bookmaker_implied_away_xg", 0.9)) if isinstance(engine_result, dict) else 0.9, 2),
        "over_under_2_5": "大" if (engine_result.get("over_25", 50) if isinstance(engine_result, dict) else 50) > 55 else "小",
        "both_score": "是" if (engine_result.get("btts", 45) if isinstance(engine_result, dict) else 45) > 50 else "否",
        "expected_total_goals": round(exp_goals, 2),
        "over_2_5": engine_result.get("over_25", 50) if isinstance(engine_result, dict) else 50,
        "btts": engine_result.get("btts", 45) if isinstance(engine_result, dict) else 45,
        "bookmaker_implied_home_xg": engine_result.get("bookmaker_implied_home_xg", "?") if isinstance(engine_result, dict) else "?",
        "bookmaker_implied_away_xg": engine_result.get("bookmaker_implied_away_xg", "?") if isinstance(engine_result, dict) else "?",

        "sharp_detected": trap_report.get("sharp_detected", False),
        "sharp_dir": trap_report.get("sharp_dir"),
        "fair_dir": max(trap_report["fair_1x2"], key=trap_report["fair_1x2"].get),
        "shin_dir": max(trap_report["shin"], key=trap_report["shin"].get),

        "model_consensus": stats.get("model_consensus", 0) if isinstance(stats, dict) else 0,
        "total_models": stats.get("total_models", 11) if isinstance(stats, dict) else 11,
        "extreme_warning": engine_result.get("scissors_gap_signal", "") if isinstance(engine_result, dict) else "",

        "refined_poisson": stats.get("refined_poisson", {}) if isinstance(stats, dict) else {},
        "poisson": {},
        "elo": stats.get("elo", {}) if isinstance(stats, dict) else {},
        "random_forest": stats.get("random_forest", {}) if isinstance(stats, dict) else {},
        "gradient_boost": stats.get("gradient_boost", {}) if isinstance(stats, dict) else {},
        "neural_net": stats.get("neural_net", {}) if isinstance(stats, dict) else {},
        "logistic": stats.get("logistic", {}) if isinstance(stats, dict) else {},
        "svm": stats.get("svm", {}) if isinstance(stats, dict) else {},
        "knn": stats.get("knn", {}) if isinstance(stats, dict) else {},
        "dixon_coles": stats.get("dixon_coles", {}) if isinstance(stats, dict) else {},
        "bradley_terry": stats.get("bradley_terry", {}) if isinstance(stats, dict) else {},
        "home_form": stats.get("home_form", {}) if isinstance(stats, dict) else {},
        "away_form": stats.get("away_form", {}) if isinstance(stats, dict) else {},
        "handicap_signal": stats.get("handicap_signal", "") if isinstance(stats, dict) else "",
        "odds_movement": stats.get("odds_movement", {}) if isinstance(stats, dict) else {},
        "vote_analysis": stats.get("vote_analysis", {}) if isinstance(stats, dict) else {},
        "h2h_blood": stats.get("h2h_blood", {}) if isinstance(stats, dict) else {},
        "crs_analysis": stats.get("crs_analysis", {}) if isinstance(stats, dict) else {},
        "ttg_analysis": stats.get("ttg_analysis", {}) if isinstance(stats, dict) else {},
        "halftime": stats.get("halftime", {}) if isinstance(stats, dict) else {},
        "pace_rating": stats.get("pace_rating", "") if isinstance(stats, dict) else "",
        "kelly_home": stats.get("kelly_home", {}) if isinstance(stats, dict) else {},
        "kelly_away": stats.get("kelly_away", {}) if isinstance(stats, dict) else {},
        "odds": stats.get("odds", {}) if isinstance(stats, dict) else {},
        "experience_analysis": stats.get("experience_analysis", {}) if isinstance(stats, dict) else {},
        "pro_odds": stats.get("pro_odds", {}) if isinstance(stats, dict) else {},
        "asian_handicap_probs": stats.get("asian_handicap_probs", {}) if isinstance(stats, dict) else {},
        "top_scores": stats.get("refined_poisson", {}).get("top_scores", []) if isinstance(stats, dict) else [],

        "engine_version": "vMAX 18.2",
        "engine_architecture": "统一比分矩阵 + 平局保护 + 让球不穿 + 字段校验 + 三初审Claude终审",
    }


# ====================================================================
# Top4
# ====================================================================

def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})

        s = pr.get("confidence", 0) * 0.4
        s += pr.get("dir_confidence", 50) * 0.15

        trap_count = pr.get("trap_count", 0)
        if trap_count >= 2:
            s += 5
        elif trap_count >= 1:
            s += 2.5

        ev = pr.get("edge_vs_market", 0)
        if pr.get("is_value"):
            if ev >= 30:
                s += 10
            elif ev >= 15:
                s += 5

        if pr.get("draw_protection", {}).get("detected") and pr.get("result") == "平局":
            s += 7

        if pr.get("handicap_no_cover", {}).get("detected"):
            s += 3

        cold = pr.get("cold_door", {})
        if cold.get("is_cold_door") and pr.get("confidence", 0) >= 60:
            s += 4

        if pr.get("risk_level") == "高":
            s -= 10
        elif pr.get("risk_level") == "低":
            s += 7

        if pr.get("is_score_others"):
            s += 6

        if pr.get("dir_gap", 0) < 8:
            s -= 5

        exp_info = pr.get("experience_analysis", {})
        exp_score = exp_info.get("total_score", 0)
        if exp_score >= 15 and pr.get("result") == "平局" and exp_info.get("draw_rules", 0) >= 3:
            s += 8
        elif exp_score >= 10:
            s += 4

        p["recommend_score"] = round(s, 2)

    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]


def extract_num(ms):
    wm = {"一": 1000, "二": 2000, "三": 3000, "四": 4000, "五": 5000, "六": 6000, "日": 7000, "天": 7000}
    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999


# ====================================================================
# 一致性校验
# ====================================================================

def _enforce_consistency(mg):
    score_str = mg.get("predicted_score", "1-1")

    if "胜其他" in score_str or score_str == "9-0":
        expected_dir = "主胜"
        expected_code = "home"
    elif "平其他" in score_str or score_str == "9-9":
        expected_dir = "平局"
        expected_code = "draw"
    elif "负其他" in score_str or score_str == "0-9":
        expected_dir = "客胜"
        expected_code = "away"
    else:
        h, a = _parse_score(score_str)
        if h is None:
            expected_dir = mg.get("result", "平局")
            expected_code = {"主胜": "home", "平局": "draw", "客胜": "away"}.get(expected_dir, "draw")
        else:
            if h > a:
                expected_dir = "主胜"
                expected_code = "home"
            elif h < a:
                expected_dir = "客胜"
                expected_code = "away"
            else:
                expected_dir = "平局"
                expected_code = "draw"

    mg["result"] = expected_dir
    mg["display_direction"] = expected_dir
    mg["final_direction"] = expected_code

    if "胜其他" in score_str or score_str == "9-0":
        mg["predicted_label"] = "胜其他"
        mg["predicted_score"] = "胜其他"
    elif "平其他" in score_str or score_str == "9-9":
        mg["predicted_label"] = "平其他"
        mg["predicted_score"] = "平其他"
    elif "负其他" in score_str or score_str == "0-9":
        mg["predicted_label"] = "负其他"
        mg["predicted_score"] = "负其他"
    else:
        mg["predicted_label"] = score_str

    return mg


# ====================================================================
# 主入口
# ====================================================================

def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    ms = [normalize_match(m) for m in ms]

    print("\n" + "=" * 80)
    print(f"  [vMAX 18.2] 统一比分矩阵+平局保护+让球不穿+字段校验 | {len(ms)} 场")
    print("=" * 80)

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
            sp = ensemble.predict(m, {}) if ensemble else {}
        except Exception as e:
            logger.warning(f"ensemble.predict 失败: {e}")
            sp = {}

        anchor_sigs = []
        try:
            a4_val = _f(m.get("a4", 0))
            a5_val = _f(m.get("a5", 0))
            a7_val = _f(m.get("a7", 0))
            if a4_val > 0 and a4_val < 5:
                anchor_sigs.append(f"4球锚点({a4_val:.2f})→典型2-2/3-1/1-3")
            if a5_val > 0 and a5_val < 8:
                anchor_sigs.append(f"5球锚点({a5_val:.2f})→典型3-2/2-3")
            if a7_val > 0 and a7_val < 18:
                anchor_sigs.append(f"7球锚点({a7_val:.2f})→典型5-2/5-1/2-5/1-5")
        except Exception:
            pass

        if anchor_sigs:
            if isinstance(sp, dict):
                existing = sp.get("smart_signals", [])
                if not isinstance(existing, list):
                    existing = [str(existing)]
                sp["smart_signals"] = existing + anchor_sigs
            else:
                sp = {"smart_signals": anchor_sigs}

        try:
            exp_result = exp_engine.analyze(m) if exp_engine else {}
        except Exception:
            exp_result = {}

        exp_goals_prev = _f(eng.get("expected_total_goals", 0)) if isinstance(eng, dict) else 0
        if exp_goals_prev <= 0:
            hxg = _f(eng.get("bookmaker_implied_home_xg", 0)) if isinstance(eng, dict) else 0
            axg = _f(eng.get("bookmaker_implied_away_xg", 0)) if isinstance(eng, dict) else 0
            exp_goals_prev = hxg + axg if (hxg and axg) else 2.5

        trap_preview = detect_all_traps(
            m,
            eng,
            {},
            sp.get("smart_signals", []) if isinstance(sp, dict) else [],
            exp_goals_prev,
        )

        crs_preview = analyze_crs_matrix(m)

        match_analyses.append({
            "match": m,
            "engine": eng,
            "league_info": league_info,
            "stats": sp,
            "index": i + 1,
            "experience": exp_result,
            "trap_preview": trap_preview,
            "crs_preview": crs_preview,
        })

    all_ai = {"claude": {}, "gemini": {}, "gpt": {}, "grok": {}}

    if use_ai and match_analyses:
        print("  [v18.2 AI] 启动 GPT/Grok/Gemini 初审 + Claude 终审...")
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

        print(f"  [完成] 耗时 {time.time() - start_t:.1f}s")

    res = []

    for i, ma in enumerate(match_analyses):
        m = ma["match"]

        mg = merge_result(
            ma["engine"],
            all_ai["gpt"].get(i + 1, {}),
            all_ai["grok"].get(i + 1, {}),
            all_ai["gemini"].get(i + 1, {}),
            all_ai["claude"].get(i + 1, {}),
            ma["stats"],
            m,
        )

        try:
            if exp_engine:
                mg = apply_experience_to_prediction(m, mg, exp_engine)
        except Exception as e:
            logger.warning(f"apply_experience_to_prediction 失败: {e}")

        try:
            mg = apply_odds_history(m, mg)
        except Exception as e:
            logger.warning(f"apply_odds_history 失败: {e}")

        try:
            mg = apply_quant_edge(m, mg)
        except Exception as e:
            logger.warning(f"apply_quant_edge 失败: {e}")

        try:
            mg = apply_wencai_intel(m, mg)
        except Exception as e:
            logger.warning(f"apply_wencai_intel 失败: {e}")

        try:
            mg = upgrade_ensemble_predict(m, mg)
        except Exception as e:
            logger.warning(f"upgrade_ensemble_predict 失败: {e}")

        mg = _enforce_consistency(mg)

        res.append({**m, "prediction": mg})

        trap_tag = f" [T{mg['trap_count']}]" if mg.get("trap_count", 0) > 0 else ""
        draw_tag = " [防平]" if mg.get("draw_protection", {}).get("detected") else ""
        nc_tag = " [不穿]" if mg.get("handicap_no_cover", {}).get("detected") else ""
        others_tag = " [其他比分]" if mg.get("is_score_others") else ""
        sharp_tag = " [Sharp]" if mg.get("sharp_detected") else ""
        scenario_tag = f" [{mg.get('scenario', 'normal')}]"

        print(
            f"  [{i+1}] {m.get('home_team')} vs {m.get('away_team')} => "
            f"{mg['result']} ({mg['predicted_score']}={mg['predicted_label']}) | "
            f"CF:{mg['confidence']} | 方向:{mg['dir_confidence']:.0f}%"
            f"{trap_tag}{draw_tag}{nc_tag}{others_tag}{sharp_tag}{scenario_tag}"
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


if __name__ == "__main__":
    logger.info("vMAX 18.2 启动")
    print("✅ vMAX 18.2 加载完成")
    print("   架构: 统一比分矩阵 + 平局保护 + 让球不穿 + 字段校验 + 三初审Claude终审")
    print("   一致性: predicted_score ↔ result ↔ display_direction ↔ final_direction")