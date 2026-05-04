
# -*- coding: utf-8 -*-
"""
vMAX 18.2 — 盘口坐标硬修复 + 其他比分闭环 + 联网情报入口 + 证据账本AI审计 + 最终IPF校准

完整替换版核心修复：
1. 修复客让/客受让盘口解析反向。
2. 修复 4-3/3-4/5-2/2-5 等其他比分赔率映射。
3. EV/Kelly 使用 calibrated_score_prob_pct，不再使用 AI 排序分。
4. TTG 锚点与陷阱残差先改矩阵形状，最后再做 IPF 校准。
5. 增加 external_context 联网情报入口，可接 Bing Search API 或自定义情报服务。
6. AI prompt 改为证据账本审计：fair_1x2 / handicap / CRS / TTG / external_context / risk / rejection。
7. 外部模块后处理不得篡改核心预测字段。
"""

import os
import re
import json
import time
import math
import asyncio
import logging
from typing import Dict, List, Any, Tuple, Optional

try:
    import aiohttp
except Exception:
    aiohttp = None

try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

try:
    from config import *  # noqa
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
except Exception:
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


ENGINE_VERSION = "vMAX 18.2.1"
ENGINE_ARCHITECTURE = "盘口坐标硬修复 + 其他比分闭环 + T1-T16/D17-D19完整风控 + 外部情报入口 + 证据账本AI审计 + 最终IPF校准"
VALID_DIRS = {"home", "draw", "away"}

STANDARD_GOAL_ODDS = {0: 9.5, 1: 5.5, 2: 3.5, 3: 4.0, 4: 7.0, 5: 14.0, 6: 30.0, 7: 70.0}

LEAGUE_LOW_GOALS = ["意甲", "西甲", "法甲", "希腊", "塞浦", "罗甲", "Serie A", "La Liga", "Ligue 1"]
LEAGUE_HIGH_GOALS = ["德甲", "荷甲", "挪超", "葡超", "英超", "Bundesliga", "Eredivisie", "Premier League"]
CUP_KEYWORDS = ["杯", "淘汰", "决赛", "半决赛", "四分之一", "欧冠", "欧联", "国王杯", "足总杯", "联赛杯", "解放者杯", "南球杯", "Cup", "cup"]

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

SCORE_OTHERS_HOME = ["4-3", "5-3", "5-4", "6-0", "6-1", "6-2", "6-3", "6-4", "7-0", "7-1", "7-2", "胜其他", "9-0"]
SCORE_OTHERS_DRAW = ["4-4", "5-5", "6-6", "平其他", "9-9"]
SCORE_OTHERS_AWAY = ["3-4", "0-6", "1-6", "2-6", "3-6", "0-7", "1-7", "2-7", "负其他", "0-9"]
ALL_SCORE_OTHERS = SCORE_OTHERS_HOME + SCORE_OTHERS_DRAW + SCORE_OTHERS_AWAY


def _f(v, default=0.0):
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "" or s.lower() in ("none", "nan", "null", "-", "n/a"):
            return default
        return float(s.replace("%", "").replace(",", ""))
    except Exception:
        return default

def _i(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

def _clip(v, lo, hi):
    return max(lo, min(hi, _f(v, 0.0)))

def _normalize_prob_dict(d: Dict[Any, float], floor: float = 0.0) -> Dict[Any, float]:
    out = {k: max(floor, _f(v, 0.0)) for k, v in (d or {}).items()}
    s = sum(out.values())
    if s <= 0:
        n = len(out) or 1
        return {k: 1.0 / n for k in out}
    return {k: v / s for k, v in out.items()}

def _softmax_dict(logits: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    if not logits:
        return {}
    t = max(0.05, temperature or 1.0)
    mx = max(logits.values())
    ex = {k: math.exp((v - mx) / t) for k, v in logits.items()}
    return _normalize_prob_dict(ex)

def _round_dict(d, n=3):
    return {k: round(_f(v), n) for k, v in (d or {}).items()}

def _deep_find_value(obj, aliases, skip_keys=None):
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

def _normalize_score_text(s):
    return str(s).strip().replace(" ", "").replace("：", "-").replace(":", "-").replace("—", "-").replace("–", "-")

def _parse_score(s: str) -> Tuple[Optional[int], Optional[int]]:
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
        if ss in ("主胜", "客胜", "平局", "胜", "平", "负"):
            return None, None
        p = ss.split("-")
        if len(p) != 2:
            return None, None
        return int(p[0]), int(p[1])
    except Exception:
        return None, None

def _score_direction(score_str: str) -> Optional[str]:
    ss = _normalize_score_text(score_str)
    if ss in SCORE_OTHERS_HOME or ss == "9-0" or "胜其他" in ss:
        return "home"
    if ss in SCORE_OTHERS_DRAW or ss == "9-9" or "平其他" in ss:
        return "draw"
    if ss in SCORE_OTHERS_AWAY or ss == "0-9" or "负其他" in ss:
        return "away"
    h, a = _parse_score(ss)
    if h is None:
        return None
    if h > a:
        return "home"
    if h < a:
        return "away"
    return "draw"

def _score_total(score_str: str) -> Optional[int]:
    h, a = _parse_score(score_str)
    return None if h is None else h + a

def _direction_cn(direction: str) -> str:
    return {"home": "主胜", "draw": "平局", "away": "客胜"}.get(direction, "平局")

def _dir_from_cn(v):
    s = str(v).strip()
    if s in ("home", "主胜", "胜"):
        return "home"
    if s in ("draw", "平局", "平"):
        return "draw"
    if s in ("away", "客胜", "负"):
        return "away"
    return None

def _score_display_label(score_str: str, direction_code: Optional[str] = None) -> str:
    ss = _normalize_score_text(score_str)
    if ss in SCORE_OTHERS_HOME or ss in ("胜其他", "9-0"):
        return "胜其他"
    if ss in SCORE_OTHERS_DRAW or ss in ("平其他", "9-9"):
        return "平其他"
    if ss in SCORE_OTHERS_AWAY or ss in ("负其他", "0-9"):
        return "负其他"
    return ss

def _parse_bool_env(name, default=False):
    v = str(os.environ.get(name, "")).strip().lower()
    if not v:
        return bool(default)
    return v in ("1", "true", "yes", "y", "on", "enable", "enabled")

def _json_compact(obj, max_len=2000):
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(obj)
    return s if len(s) <= max_len else s[:max_len] + "...<truncated>"


def normalize_match(raw_m: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(raw_m or {})
    for nk in ["v2_odds_dict", "odds_dict", "odds", "v2", "odds_v2", "packet", "raw_odds", "data", "detail"]:
        if isinstance(m.get(nk), dict):
            for k, v in m[nk].items():
                if k not in m or m.get(k) in (None, ""):
                    m[k] = v

    home = m.get("home_team") or m.get("home") or m.get("host") or m.get("team_home") or m.get("homeName") or "Home"
    away = m.get("away_team") or m.get("guest") or m.get("away") or m.get("team_away") or m.get("awayName") or "Away"
    m["home_team"] = m["home"] = home
    m["away_team"] = away
    m["guest"] = away

    skip = {"vote", "change", "points", "information", "prediction", "stats", "smart_signals"}
    sp_home = m.get("sp_home") or _deep_find_value(m, ["win", "odds_win", "spf_sp3", "sp3", "胜"], skip)
    sp_draw = m.get("sp_draw") or _deep_find_value(m, ["draw", "same", "odds_draw", "spf_sp1", "sp1", "平"], skip)
    sp_away = m.get("sp_away") or _deep_find_value(m, ["lose", "away_win", "odds_lose", "spf_sp0", "sp0", "负"], skip)

    if sp_home is not None:
        m["sp_home"] = m["win"] = sp_home
    if sp_draw is not None:
        m["sp_draw"] = m["same"] = sp_draw
    if sp_away is not None:
        m["sp_away"] = m["lose"] = sp_away

    if "give_ball" not in m:
        m["give_ball"] = m.get("handicap") or m.get("rq") or m.get("let_ball") or "0"

    if not isinstance(m.get("change"), dict):
        ch = {}
        for src_key, dst_key in [
            ("change_win", "win"), ("cw", "win"), ("change_same", "same"), ("cs", "same"),
            ("change_draw", "same"), ("change_lose", "lose"), ("cl", "lose"), ("change_away", "lose"),
        ]:
            if src_key in m:
                ch[dst_key] = m.get(src_key)
        m["change"] = ch
    return m


def calculate_independent_ev(model_prob_pct: float, market_odds: float, market_implied_pct: Optional[float] = None) -> Dict[str, Any]:
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
        "note": "calibrated_matrix_prob_vs_market",
    }


def fair_probs_from_1x2(sp_h: float, sp_d: float, sp_a: float, method: str = "power") -> Dict[str, Any]:
    odds = {"home": _f(sp_h), "draw": _f(sp_d), "away": _f(sp_a)}
    if any(v <= 1.01 for v in odds.values()):
        return {"method": "fallback", "fair_probs": {"home": 33.3, "draw": 33.3, "away": 33.4}, "raw_implied": {}, "overround": 0.0, "shin_z": None}
    q = {k: 1.0 / v for k, v in odds.items()}
    overround_sum = sum(q.values())
    overround = overround_sum - 1.0
    raw_pct = {k: round(v * 100, 3) for k, v in q.items()}

    if method == "multiplicative":
        p = _normalize_prob_dict(q)
    elif method == "additive":
        cut = overround / 3.0
        p = _normalize_prob_dict({k: max(0.001, v - cut) for k, v in q.items()})
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
        p = _normalize_prob_dict({name: val ** k for name, val in q.items()})
        method = "power"
    return {"method": method, "fair_probs": {k: round(v * 100, 3) for k, v in p.items()}, "raw_implied": raw_pct, "overround": round(overround, 5), "shin_z": None}

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
    return _normalize_prob_dict({g: v ** k for g, v in raw.items()})

def _poisson_pmf(lam: float, k: int) -> float:
    lam = max(0.05, min(8.0, float(lam)))
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _extract_form_record(text: str) -> Tuple[int, int, int]:
    if not text:
        return 0, 0, 0
    patterns = [
        r"近\s*\d+\s*[主客场]*\s*(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
        r"(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
        r"(\d+)W\s*(\d+)D\s*(\d+)L",
    ]
    for pat in patterns:
        m = re.search(pat, str(text), flags=re.I)
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return 0, 0, 0

def _extract_avg_goals(text: str) -> Tuple[float, float]:
    if not text:
        return 0.0, 0.0
    text = str(text)
    gf = re.search(r"场均进球[^0-9]*(\d+\.?\d*)", text)
    ga = re.search(r"场均失球[^0-9]*(\d+\.?\d*)", text)
    return float(gf.group(1)) if gf else 0.0, float(ga.group(1)) if ga else 0.0

def _fundamental_strength(match_obj: Dict, side: str) -> Dict[str, Any]:
    points = match_obj.get("points", {}) if isinstance(match_obj.get("points"), dict) else {}
    key = "home_strength" if side == "home" else "guest_strength"
    txt = str(points.get(key, ""))
    w, d, l = _extract_form_record(txt)
    total = w + d + l
    win_rate = w / total if total else 0.5
    gf, ga = _extract_avg_goals(txt)
    score = 0.0
    if total:
        score += (win_rate - 0.5) * 80
    if gf > 0:
        score += (gf - 1.3) * 20
    if ga > 0:
        score -= (ga - 1.3) * 20
    return {"wins": w, "draws": d, "losses": l, "total": total, "win_rate": round(win_rate, 3), "goals_for": round(gf, 2), "goals_against": round(ga, 2), "strength_score": round(max(-100, min(100, score)), 1)}


# 盘口坐标：主让=-，主受让=+，客让=+，客受让=-
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
    if ratio >= 1.15:
        return 0.25
    return 0.0

def _infer_theoretical_handicap_signed(sp_h: float, sp_a: float) -> float:
    depth = _infer_theoretical_handicap_depth(sp_h, sp_a)
    if depth <= 0:
        return 0.0
    return -depth if _strong_side_from_1x2(sp_h, sp_a) == "home" else depth

def _parse_actual_handicap(match_obj: Dict) -> float:
    return _parse_actual_handicap_signed(match_obj)

def _parse_actual_handicap_signed(match_obj: Dict) -> float:
    raw = match_obj.get("give_ball") if match_obj.get("give_ball") is not None else match_obj.get("handicap", match_obj.get("rq", match_obj.get("let_ball", "0")))
    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))
    strong_side = _strong_side_from_1x2(sp_h, sp_a)
    s0 = str(raw).strip()
    if not s0:
        return 0.0
    s = s0.replace("（", "(").replace("）", ")").replace("球", "").replace(" ", "")
    nums = re.findall(r"[-+]?\d+\.?\d*", s)
    if not nums:
        return 0.0
    vals = [abs(_f(x, 0.0)) for x in nums]
    val_abs = sum(vals[:2]) / 2.0 if "/" in s and len(vals) >= 2 else vals[0]

    has_home = "主" in s0
    has_away = "客" in s0
    has_receive = "受让" in s0 or "受" in s0
    has_let = "让" in s0

    if has_home and has_receive:
        return +val_abs
    if has_home and has_let:
        return -val_abs
    if has_away and has_receive:
        return -val_abs
    if has_away and has_let:
        return +val_abs
    if str(s).startswith("-") or ("(" in s and "-" in s):
        return -val_abs
    if str(s).startswith("+") or ("(" in s and "+" in s):
        return +val_abs
    if val_abs > 0:
        return -val_abs if strong_side == "home" else +val_abs
    return 0.0

def _handicap_depth_for_side(actual_signed: float, side: str) -> float:
    if side == "home":
        return max(0.0, -actual_signed)
    if side == "away":
        return max(0.0, actual_signed)
    return 0.0

def _handicap_diff_for_strong_side(actual_signed: float, theoretical_signed: float, strong_side: str) -> float:
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
    if score in SCORE_OTHERS_HOME or score in ("胜其他", "9-0"):
        return _f(match_obj.get("crs_win", 0))
    if score in SCORE_OTHERS_DRAW or score in ("平其他", "9-9"):
        return _f(match_obj.get("crs_same", 0))
    if score in SCORE_OTHERS_AWAY or score in ("负其他", "0-9"):
        return _f(match_obj.get("crs_lose", 0))
    return 0.0

def _crs_low_rank_info(match_obj: Dict[str, Any]) -> Dict[str, Any]:
    odds_list = []
    for sc, key in CRS_FULL_MAP.items():
        odd = _f(match_obj.get(key, 0))
        if odd > 1.05:
            odds_list.append((sc, odd, _score_direction(sc)))
    odds_list.sort(key=lambda x: x[1])
    rank = {sc: {"rank": i, "odds": odd, "direction": d} for i, (sc, odd, d) in enumerate(odds_list, 1)}
    draw_av = [(sc, rank[sc]["odds"], rank[sc]["rank"]) for sc in ["0-0", "1-1", "2-2", "3-3"] if sc in rank]
    draw_av.sort(key=lambda x: x[1])
    mn = draw_av[0] if draw_av else None
    return {
        "all_sorted": odds_list,
        "rank": rank,
        "draw_available": draw_av,
        "draw_low_rank": mn[2] if mn else 999,
        "draw_low_score": mn[0] if mn else "",
        "draw_low_odds": mn[1] if mn else 999.0,
        "low_scores": [(sc, odd) for sc, odd, _ in odds_list[:10]],
    }

def _has_low_draw_crs(match_obj, rank_cutoff=5, odds_cutoff=8.5):
    info = _crs_low_rank_info(match_obj)
    return info["draw_low_rank"] <= rank_cutoff or info["draw_low_odds"] <= odds_cutoff

def _has_low_score(match_obj, score, rank_cutoff=6, odds_cutoff=8.5):
    info = _crs_low_rank_info(match_obj)
    it = info["rank"].get(score)
    return bool(it and (it["rank"] <= rank_cutoff or it["odds"] <= odds_cutoff))


# ==============================
# 联网情报入口
# ==============================

def _external_context_enabled():
    return _parse_bool_env("ENABLE_EXTERNAL_CONTEXT", True)

def _parse_external_endpoints():
    raw = os.environ.get("EXTERNAL_CONTEXT_ENDPOINTS", "") or os.environ.get("EXTERNAL_CONTEXT_URLS", "")
    raw = raw.strip()
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

def _extract_builtin_external_context(match_obj):
    info = match_obj.get("information", {}) if isinstance(match_obj.get("information"), dict) else {}
    points = match_obj.get("points", {}) if isinstance(match_obj.get("points"), dict) else {}
    ext = match_obj.get("external_context", {}) if isinstance(match_obj.get("external_context"), dict) else {}
    out = {
        "source_quality": "missing", "injuries": [], "suspensions": [], "lineup_news": [],
        "weather": {}, "motivation": [], "schedule_pressure": [], "odds_history": [],
        "sharp_market_notes": [], "news_snippets": [], "data_missing": [], "raw": {},
    }
    for k in ["home_injury", "guest_injury", "injury", "injuries"]:
        if info.get(k):
            v = info.get(k)
            out["injuries"].extend(v if isinstance(v, list) else [str(v)])
    for k in ["suspension", "suspensions", "home_suspension", "guest_suspension"]:
        if info.get(k):
            v = info.get(k)
            out["suspensions"].extend(v if isinstance(v, list) else [str(v)])
    for k in ["lineup", "lineups", "probable_lineup", "starting"]:
        if info.get(k):
            v = info.get(k)
            out["lineup_news"].extend(v if isinstance(v, list) else [str(v)])
    if isinstance(info.get("weather"), dict):
        out["weather"] = info.get("weather")
    elif info.get("weather"):
        out["weather"] = {"note": str(info.get("weather"))}
    if points.get("match_points"):
        out["motivation"].append(str(points.get("match_points")))
    if ext:
        for k, v in ext.items():
            if k in out and isinstance(out[k], list):
                out[k].extend(v if isinstance(v, list) else [str(v)])
            elif k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k].update(v)
            else:
                out["raw"][k] = v
        out["source_quality"] = ext.get("source_quality", "provided")
    if any(out[k] for k in ["injuries", "suspensions", "lineup_news", "weather", "motivation"]):
        out["source_quality"] = "provided"
    out["data_missing"] = [k for k in ["injuries", "suspensions", "lineup_news", "weather", "motivation", "odds_history"] if not out.get(k)]
    return out

def _build_search_queries(match_obj):
    h = match_obj.get("home_team", match_obj.get("home", ""))
    a = match_obj.get("away_team", match_obj.get("guest", ""))
    lg = match_obj.get("league", match_obj.get("cup", ""))
    base = f"{h} vs {a} {lg}".strip()
    return [
        f"{base} injuries suspensions probable lineups preview",
        f"{base} team news weather schedule pressure",
        f"{h} {a} 伤停 首发 天气 赛前 情报",
    ]

async def _fetch_json_or_text(session, method, url, **kwargs):
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with session.request(method, url, timeout=timeout, **kwargs) as r:
            text = await r.text()
            if r.status < 200 or r.status >= 300:
                return False, None, f"http_{r.status}"
            try:
                return True, json.loads(text), "json"
            except Exception:
                return True, text[:4000], "text"
    except Exception as e:
        return False, None, str(e)[:120]

async def _search_bing_snippets(session, match_obj):
    key = os.environ.get("BING_SEARCH_API_KEY", "").strip()
    endpoint = os.environ.get("BING_SEARCH_API_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search").strip()
    if not key or not endpoint:
        return []
    headers = {"Ocp-Apim-Subscription-Key": key}
    out = []
    for q in _build_search_queries(match_obj):
        ok, data, _ = await _fetch_json_or_text(session, "GET", endpoint, headers=headers, params={"q": q, "count": 5, "mkt": os.environ.get("BING_SEARCH_MARKET", "en-US")})
        if ok and isinstance(data, dict):
            for item in data.get("webPages", {}).get("value", [])[:5]:
                out.append({"query": q, "name": item.get("name", ""), "url": item.get("url", ""), "snippet": item.get("snippet", ""), "dateLastCrawled": item.get("dateLastCrawled", "")})
    return out[:12]

async def build_external_context_for_match(session, match_obj):
    ctx = _extract_builtin_external_context(match_obj)
    if not _external_context_enabled() or aiohttp is None:
        return ctx
    payload = {
        "home_team": match_obj.get("home_team"),
        "away_team": match_obj.get("away_team"),
        "league": match_obj.get("league", match_obj.get("cup", "")),
        "match_time": match_obj.get("match_time", match_obj.get("time", "")),
        "queries": _build_search_queries(match_obj),
    }
    for url in _parse_external_endpoints()[:4]:
        headers = {"Content-Type": "application/json"}
        token = os.environ.get("EXTERNAL_CONTEXT_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        ok, data, kind = await _fetch_json_or_text(session, os.environ.get("EXTERNAL_CONTEXT_METHOD", "POST").upper(), url, headers=headers, json=payload)
        if not ok:
            continue
        if isinstance(data, dict):
            for key in ["injuries", "suspensions", "lineup_news", "motivation", "schedule_pressure", "odds_history", "sharp_market_notes", "news_snippets"]:
                val = data.get(key)
                if isinstance(val, list):
                    ctx[key].extend(val)
                elif val:
                    ctx[key].append(str(val))
            if isinstance(data.get("weather"), dict):
                ctx["weather"].update(data["weather"])
            ctx["raw"].setdefault("provider_results", []).append(data)
            ctx["source_quality"] = data.get("source_quality", "provider")
        elif isinstance(data, str):
            ctx["news_snippets"].append({"source": url, "snippet": data[:1000]})
            ctx["source_quality"] = "provider_text"
    snippets = await _search_bing_snippets(session, match_obj)
    if snippets:
        ctx["news_snippets"].extend(snippets)
        if ctx["source_quality"] == "missing":
            ctx["source_quality"] = "search_snippets"
    for key in ["injuries", "suspensions", "lineup_news", "motivation", "schedule_pressure", "odds_history", "sharp_market_notes", "news_snippets"]:
        if isinstance(ctx.get(key), list):
            seen, arr = set(), []
            for x in ctx[key]:
                sx = _json_compact(x, 800) if isinstance(x, dict) else str(x)
                if sx not in seen:
                    seen.add(sx)
                    arr.append(x)
            ctx[key] = arr[:12]
    ctx["data_missing"] = [k for k in ["injuries", "suspensions", "lineup_news", "weather", "motivation", "odds_history"] if not ctx.get(k)]
    return ctx

async def attach_external_contexts(match_analyses):
    if aiohttp is None:
        for ma in match_analyses:
            ma["external_context"] = _extract_builtin_external_context(ma["match"])
        return match_analyses
    connector = aiohttp.TCPConnector(limit=5, use_dns_cache=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        rs = await asyncio.gather(*[build_external_context_for_match(session, ma["match"]) for ma in match_analyses], return_exceptions=True)
    for ma, r in zip(match_analyses, rs):
        ma["external_context"] = r if isinstance(r, dict) else _extract_builtin_external_context(ma["match"])
    return match_analyses

def _run_coro_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        def runner(c):
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(c)
            finally:
                new_loop.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(runner, coro).result()
    return asyncio.run(coro)


# ==============================
# 市场信号和陷阱
# ==============================

def detect_sharp_direction(smart_signals):
    detected, sharp_dir = False, None
    for s in smart_signals or []:
        s = str(s)
        if "Sharp" in s or "sharp" in s or "聪明钱" in s or "专业资金" in s:
            detected = True
            if re.search(r"(主胜|主队|走主|流向\s*主|资金\s*主|Sharp主|聪明钱主|向主)", s):
                sharp_dir = "home"; break
            if re.search(r"(客胜|客队|走客|流向\s*客|资金\s*客|Sharp客|聪明钱客|向客)", s):
                sharp_dir = "away"; break
            if re.search(r"(平局|平赔|走平|流向\s*平|资金\s*平|Sharp平|聪明钱平)", s):
                sharp_dir = "draw"; break
    return {"detected": detected, "sharp_dir": sharp_dir}

def detect_steam_direction(smart_signals):
    steam_dir, steam_type = None, None
    for s in smart_signals or []:
        s = str(s)
        if "Steam" not in s and "steam" not in s and "异动" not in s:
            continue
        is_reverse = "反向" in s or "未跟" in s or "不跟" in s
        if "主" in s:
            steam_dir = "home"; steam_type = "reverse" if is_reverse else "normal"; break
        if "客" in s:
            steam_dir = "away"; steam_type = "reverse" if is_reverse else "normal"; break
        if "平" in s:
            steam_dir = "draw"; steam_type = "reverse" if is_reverse else "normal"; break
    return {"steam_dir": steam_dir, "steam_type": steam_type}

def detect_T1_draw_trap(match_obj, engine_result, smart_signals, fair):
    ch = match_obj.get("change", {}) or {}
    cs = _f(ch.get("same", ch.get("draw", 0)))
    if cs >= -0.04:
        return None
    fair_h, fair_d, fair_a = fair["home"], fair["draw"], fair["away"]
    strong_side = "home" if fair_h >= fair_a else "away"
    weak_side = "away" if strong_side == "home" else "home"
    crs = _crs_low_rank_info(match_obj)
    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))
    total_xg = hxg + axg if hxg > 0 and axg > 0 else 0.0
    fair_gap = abs(fair_h - fair_a)
    draw_low = crs["draw_low_rank"] <= 4 or crs["draw_low_odds"] <= 8.0
    if draw_low and (fair_d >= 20 or fair_gap <= 16) and (total_xg <= 2.85 or total_xg <= 0):
        return {"trap": "T1_REAL_DRAW_SIGNAL", "description": f"真平信号:平赔独降{cs:.2f}+CRS平局低位{crs['draw_low_score']}@{crs['draw_low_odds']:.1f}/rank{crs['draw_low_rank']}", "severity": 3, "direction_adjust": {"draw": 1.15, strong_side: -0.35, weak_side: -0.15}, "score_multipliers": {"0-0": 1.22, "1-1": 1.55, "2-2": 1.38}, "boost_scores": ["1-1", "0-0", "2-2"], "draw_guard": True}
    strong_fair = max(fair_h, fair_a)
    evidence_score = 2 + (2 if strong_fair >= 42 else 1 if strong_fair >= 38 else 0)
    if evidence_score < 4:
        return None
    return {"trap": "T1_FAKE_DRAW_TRAP", "description": f"假平诱盘:平赔独降{cs:.2f}+强方{strong_side}公平{strong_fair:.1f}%", "severity": 2, "direction_adjust": {strong_side: 0.75, "draw": -0.9, weak_side: -0.2}, "score_multipliers": {"1-1": 0.65, "2-2": 0.72, "0-0": 0.76}}

def detect_T2_T3_handicap_trap(match_obj, fair):
    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))
    if sp_h < 1.05 or sp_a < 1.05:
        return None
    theoretical = _infer_theoretical_handicap_signed(sp_h, sp_a)
    actual = _parse_actual_handicap_signed(match_obj)
    strong = _strong_side_from_1x2(sp_h, sp_a)
    weak = "away" if strong == "home" else "home"
    diff = _handicap_diff_for_strong_side(actual, theoretical, strong)
    if abs(diff) < 0.5:
        return None
    if diff >= 0.5:
        boost = ["2-0", "2-1", "3-0", "3-1"] if strong == "home" else ["0-2", "1-2", "0-3", "1-3"]
        return {"trap": "T2_HANDICAP_DEEPER", "description": f"强方让球偏深:{strong} 理论{theoretical:.2f} vs 实际{actual:.2f},强方差{diff:+.2f}", "severity": 2 if diff < 1.0 else 3, "direction_adjust": {strong: 0.65 * min(2, abs(diff)), weak: -0.3, "draw": -0.2}, "score_multipliers": {}, "boost_scores": boost}
    boost = ["1-1", "0-0", "2-2", "0-1", "1-2"] if strong == "home" else ["1-1", "0-0", "2-2", "1-0", "2-1"]
    return {"trap": "T3_HANDICAP_SHALLOWER", "description": f"强方让球偏浅:{strong} 理论{theoretical:.2f} vs 实际{actual:.2f},强方差{diff:+.2f}", "severity": 2 if abs(diff) < 1.0 else 3, "direction_adjust": {strong: -0.72 * min(2, abs(diff)), "draw": 0.58, weak: 0.32}, "score_multipliers": {"1-1": 1.22, "0-0": 1.12, "2-2": 1.18}, "boost_scores": boost, "shallow_guard": True}

def detect_T6_T7_score_range_trap(match_obj, engine_result, exp_goals):
    a0, a1, a2 = _f(match_obj.get("a0", 999), 999), _f(match_obj.get("a1", 999), 999), _f(match_obj.get("a2", 999), 999)
    low_small = int(0 < a0 < 8.0) + int(0 < a1 < 4.5) + int(0 < a2 < 3.0)
    if low_small >= 2 and exp_goals >= 2.8:
        return {"trap": "T6_SMALL_SCORE_TRAP", "description": f"诱小比分:a0/1/2压低{low_small}项但λ={exp_goals:.2f}", "severity": 2, "direction_adjust": {}, "score_multipliers": {"0-0": 0.55, "1-0": 0.72, "0-1": 0.72, "1-1": 0.78}, "boost_scores": ["2-1", "2-2", "3-1", "1-3", "3-2", "2-3"]}
    a5, a6, a7 = _f(match_obj.get("a5", 999), 999), _f(match_obj.get("a6", 999), 999), _f(match_obj.get("a7", 999), 999)
    low_large = int(0 < a5 < 10) + int(0 < a6 < 16) + int(0 < a7 < 30)
    if low_large >= 2 and exp_goals <= 2.3:
        return {"trap": "T7_LARGE_SCORE_TRAP", "description": f"诱大比分:a5/6/7压低{low_large}项但λ={exp_goals:.2f}", "severity": 2, "direction_adjust": {}, "score_multipliers": {"3-2": 0.62, "4-2": 0.50, "3-3": 0.58, "2-3": 0.62}, "boost_scores": ["1-0", "0-1", "1-1", "2-1", "1-2"]}
    return None

def detect_D17_balanced_draw_guard(match_obj, engine_result, fair):
    vals = sorted([fair.get("home", 33.3), fair.get("draw", 33.3), fair.get("away", 33.4)], reverse=True)
    top, gap, draw_p = vals[0], vals[0] - vals[1], fair.get("draw", 33.3)
    low_draw = _has_low_draw_crs(match_obj, 5, 8.5)
    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))
    total_xg = hxg + axg if hxg > 0 and axg > 0 else 0
    balanced_signal = top < 43.0 or gap < 6.0
    draw_evidence = draw_p >= 25.5 or low_draw or (0 < total_xg <= 2.55)
    if not (balanced_signal and draw_evidence):
        return None
    return {"trap": "D17_BALANCED_DRAW_GUARD", "description": f"均势平局保护:top{top:.1f}/gap{gap:.1f}/draw{draw_p:.1f}/low_draw={low_draw}/xg={total_xg:.2f}", "severity": 2, "direction_adjust": {"draw": 0.72, "home": -0.22, "away": -0.22}, "score_multipliers": {"0-0": 1.12, "1-1": 1.38, "2-2": 1.22}, "boost_scores": ["1-1", "0-0", "2-2"], "draw_guard": True}

def detect_D18_favorite_heat_guard(match_obj, fair):
    fair_h, fair_a = fair.get("home", 33.3), fair.get("away", 33.3)
    strong = "home" if fair_h >= fair_a else "away"
    weak = "away" if strong == "home" else "home"
    strong_fair = fair_h if strong == "home" else fair_a
    if strong_fair < 58:
        return None
    vote = match_obj.get("vote", {}) or {}
    strong_vote = int(_f(vote.get("win" if strong == "home" else "lose", 33), 33))
    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))
    diff = _handicap_diff_for_strong_side(_parse_actual_handicap_signed(match_obj), _infer_theoretical_handicap_signed(sp_h, sp_a), strong)
    low_draw = _has_low_draw_crs(match_obj, 6, 8.5)
    if strong_vote < 58 and not low_draw:
        return None
    if diff > -0.35 and not low_draw:
        return None
    return {"trap": "D18_FAVORITE_HEAT_GUARD", "description": f"大热浅盘防平/防弱:强方{strong}公平{strong_fair:.1f}% 散户{strong_vote}% diff{diff:+.2f}", "severity": 3, "direction_adjust": {strong: -0.70, "draw": 0.78, weak: 0.22}, "score_multipliers": {"1-1": 1.45, "0-0": 1.15, "2-2": 1.25}, "boost_scores": ["1-1", "0-0", "2-2", "1-0", "0-1"], "favorite_guard": True}

def detect_D19_long_away_guard(match_obj, fair):
    fair_a, fair_h = fair.get("away", 33.3), fair.get("home", 33.3)
    if fair_a < 40 or fair_a - fair_h < 4:
        return None
    sp_h, sp_a = _f(match_obj.get("sp_home", match_obj.get("win", 0))), _f(match_obj.get("sp_away", match_obj.get("lose", 0)))
    if sp_h <= sp_a:
        return None
    diff = _handicap_diff_for_strong_side(_parse_actual_handicap_signed(match_obj), _infer_theoretical_handicap_signed(sp_h, sp_a), "away")
    vote = match_obj.get("vote", {}) or {}
    away_vote = int(_f(vote.get("lose", 33), 33))
    low_home_or_draw = _has_low_score(match_obj, "1-1", 6, 8.5) or _has_low_score(match_obj, "1-0", 7, 10.0) or _has_low_score(match_obj, "2-1", 8, 11.0)
    if diff > -0.35 and away_vote < 55 and not low_home_or_draw:
        return None
    return {"trap": "D19_LONG_AWAY_GUARD", "description": f"强客浅盘保护:客公平{fair_a:.1f}% diff{diff:+.2f} 客热{away_vote}%", "severity": 2, "direction_adjust": {"away": -0.55, "draw": 0.35, "home": 0.35}, "score_multipliers": {"1-0": 1.28, "2-1": 1.20, "1-1": 1.30}, "boost_scores": ["1-0", "2-1", "1-1"], "long_away_guard": True}



def detect_T4_T5_fake_favorite(match_obj: Dict, engine_result: Dict, fair: Dict) -> Optional[Dict]:
    """伪热门/伪强方：1X2 给出高公平概率，但基本面与强方方向明显不匹配。"""
    fair_h, fair_a = fair.get("home", 33.3), fair.get("away", 33.3)
    fund_h = _fundamental_strength(match_obj, "home")
    fund_a = _fundamental_strength(match_obj, "away")

    if fair_h > 48 and fund_h["total"] >= 3 and fund_a["total"] >= 3:
        if fund_h["strength_score"] < -5 and fund_a["strength_score"] > 15:
            return {
                "trap": "T4_FAKE_HOME_FAVORITE",
                "description": f"诱主胜:主公平概率{fair_h:.1f}%但主基本面{fund_h['strength_score']} vs 客{fund_a['strength_score']}",
                "severity": 3,
                "direction_adjust": {"home": -1.25, "away": 0.95, "draw": 0.45},
                "score_multipliers": {"1-0": 0.65, "2-0": 0.58, "2-1": 0.72},
                "boost_scores": ["0-1", "1-1", "1-2"],
            }

    if fair_a > 48 and fund_h["total"] >= 3 and fund_a["total"] >= 3:
        if fund_a["strength_score"] < -5 and fund_h["strength_score"] > 15:
            return {
                "trap": "T5_FAKE_AWAY_FAVORITE",
                "description": f"诱客胜:客公平概率{fair_a:.1f}%但客基本面{fund_a['strength_score']} vs 主{fund_h['strength_score']}",
                "severity": 3,
                "direction_adjust": {"away": -1.25, "home": 0.95, "draw": 0.45},
                "score_multipliers": {"0-1": 0.65, "0-2": 0.58, "1-2": 0.72},
                "boost_scores": ["1-0", "1-1", "2-1"],
            }

    return None


def detect_T8_false_cold(match_obj: Dict, smart_signals: List, fair: Dict) -> Optional[Dict]:
    """假冷：散户热方向被坏消息/崩盘叙事压制，但基本面仍支持热门方向。"""
    sigs_str = " ".join(str(s) for s in smart_signals or [])
    cold_triggers = sum(1 for kw in ["坏消息", "崩盘", "造热", "背离", "盘口太便宜", "利空"] if kw in sigs_str)
    if cold_triggers < 2:
        return None

    vote = match_obj.get("vote", {}) or {}
    vh = int(_f(vote.get("win", 33), 33))
    va = int(_f(vote.get("lose", 33), 33))
    hot_dir = "home" if vh >= 58 else "away" if va >= 58 else None
    hot_pct = vh if hot_dir == "home" else va if hot_dir == "away" else 0
    if not hot_dir:
        return None

    fund = _fundamental_strength(match_obj, hot_dir)
    if fund["total"] >= 3 and fund["strength_score"] > 20 and fund["win_rate"] > 0.55:
        return {
            "trap": "T8_FALSE_COLD",
            "description": f"假冷门:{hot_dir}散户热{hot_pct}%但基本面真强({fund['strength_score']}分,胜率{fund['win_rate']:.2f})",
            "severity": 2,
            "direction_adjust": {hot_dir: 0.80, "home" if hot_dir == "away" else "away": -0.55},
            "score_multipliers": {},
            "suppress_contrarian": True,
        }
    return None


def detect_T9_fake_contrarian(match_obj: Dict, fair: Dict, smart_signals: List) -> Optional[Dict]:
    """伪反指：散户热方向同时赔率降水，不能机械反热门。"""
    vote = match_obj.get("vote", {}) or {}
    vh = int(_f(vote.get("win", 33), 33))
    va = int(_f(vote.get("lose", 33), 33))
    hot_dir, hot_pct = ("home", vh) if vh >= 60 else (("away", va) if va >= 60 else (None, 0))
    if not hot_dir:
        return None

    change = match_obj.get("change", {}) or {}
    cw, cl = _f(change.get("win", 0)), _f(change.get("lose", 0))
    follow = (hot_dir == "home" and cw < -0.04) or (hot_dir == "away" and cl < -0.04)
    if follow:
        return {
            "trap": "T9_FAKE_CONTRARIAN",
            "description": f"诱反指陷阱:{hot_dir}散户{hot_pct}%+赔率同向降水,反指风险高",
            "severity": 2,
            "direction_adjust": {hot_dir: 0.45},
            "score_multipliers": {},
            "suppress_contrarian": True,
        }
    return None


def detect_T10_silent_market(match_obj: Dict) -> Optional[Dict]:
    """沉默盘：变动少且 CRS 覆盖不足，降低信号置信度。"""
    change = match_obj.get("change", {}) or {}
    total_move = abs(_f(change.get("win", 0))) + abs(_f(change.get("same", change.get("draw", 0)))) + abs(_f(change.get("lose", 0)))
    from_crs = sum(1 for k in ["w10", "w20", "w21", "s00", "s11", "l01", "l02", "l12"] if _f(match_obj.get(k, 0)) > 1)
    if total_move < 0.03 and from_crs < 6:
        return {
            "trap": "T10_SILENT_MARKET",
            "description": f"沉默盘:赔率变动{total_move:.3f}+CRS覆盖{from_crs}/8 → 市场定价薄弱",
            "severity": 1,
            "direction_adjust": {},
            "score_multipliers": {},
            "confidence_penalty": 8,
        }
    return None


def detect_T11_xg_divergence(match_obj: Dict, engine_result: Dict) -> Optional[Dict]:
    """xG 与基本面场均进失球明显背离时，给矩阵一个低权重修正提示。"""
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
            "description": f"xG背离:{'; '.join(divergences)}",
            "severity": 1,
            "direction_adjust": {},
            "score_multipliers": {},
            "xg_override": {"home_xg": h_for if h_for > 0 else hxg_book, "away_xg": a_for if a_for > 0 else axg_book},
        }
    return None


def detect_T12_missing_handicap(match_obj: Dict) -> Optional[Dict]:
    """理论应让但实际未开，让球缺失或隐藏真实预期。"""
    actual = _parse_actual_handicap_signed(match_obj)
    if abs(actual) > 0.1:
        return None
    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))
    if sp_h < 1.01 or sp_a < 1.01:
        return None
    theoretical = _infer_theoretical_handicap_signed(sp_h, sp_a)
    if abs(theoretical) < 0.4:
        return None
    strong = _strong_side_from_1x2(sp_h, sp_a)
    weak = "away" if strong == "home" else "home"
    return {
        "trap": "T12_MISSING_HANDICAP",
        "description": f"让球未开但理论{theoretical:.2f}球 → 庄家隐藏真实预期",
        "severity": 1,
        "direction_adjust": {strong: -0.15, "draw": 0.25, weak: 0.05},
        "score_multipliers": {"1-1": 1.08},
        "confidence_penalty": 5,
    }


def detect_T13_goalless_draw(match_obj: Dict, engine_result: Dict, fair: Dict, exp_goals: float) -> Optional[Dict]:
    """闷平/低比分保护。"""
    if abs(fair.get("home", 33.3) - fair.get("away", 33.3)) > 10:
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
    weak_attack = int(0 < h_for < 1.4) + int(0 < a_for < 1.4)
    strong_def = int(0 < h_against < 1.2) + int(0 < a_against < 1.2)
    if weak_attack + strong_def < 2:
        return None

    a0, a1, a2 = _f(match_obj.get("a0", 999), 999), _f(match_obj.get("a1", 999), 999), _f(match_obj.get("a2", 999), 999)
    small_compressed = int(0 < a0 <= 10) + int(0 < a1 <= 5) + int(0 < a2 <= 3.5)
    if small_compressed < 1:
        return None

    vote = match_obj.get("vote", {}) or {}
    max_vote = max(int(_f(vote.get("win", 33), 33)), int(_f(vote.get("same", 33), 33)), int(_f(vote.get("lose", 33), 33)))
    if max_vote >= 58:
        return None

    sev = 3 if total_xg < 2.0 and small_compressed >= 2 else 2
    return {
        "trap": "T13_GOALLESS_DRAW",
        "description": f"闷平场景:xG总{total_xg:.2f}+弱攻{weak_attack}/强防{strong_def}+小球压低{small_compressed}项",
        "severity": sev,
        "direction_adjust": {"draw": 0.95, "home": -0.25, "away": -0.25},
        "score_multipliers": {"2-1": 0.78, "1-2": 0.78, "2-2": 0.75, "3-1": 0.55, "1-3": 0.55, "3-2": 0.45},
        "boost_scores": ["0-0", "1-1", "1-0", "0-1"],
    }


def detect_T14_cup_favorite_trap(match_obj: Dict, fair: Dict) -> Optional[Dict]:
    """杯赛/淘汰赛大热保护。"""
    league = str(match_obj.get("league", match_obj.get("cup", "")))
    if not any(kw in league for kw in CUP_KEYWORDS):
        return None
    fair_h, fair_a = fair.get("home", 33.3), fair.get("away", 33.3)
    strong_fair = max(fair_h, fair_a)
    if strong_fair < 55:
        return None
    strong = "home" if fair_h > fair_a else "away"
    weak = "away" if strong == "home" else "home"
    vote = match_obj.get("vote", {}) or {}
    strong_vote = int(_f(vote.get("win" if strong == "home" else "lose", 33), 33))
    if strong_vote < 50:
        return None
    weak_fund = _fundamental_strength(match_obj, weak)
    if weak_fund["total"] >= 3:
        reasonable_weak = weak_fund["win_rate"] >= 0.35 or weak_fund["goals_for"] >= 1.2 or weak_fund["strength_score"] > -10
        if not reasonable_weak:
            return None
    return {
        "trap": "T14_CUP_FAVORITE",
        "description": f"杯赛大热保护:{strong}公平概率{strong_fair:.1f}%+散户{strong_vote}%,淘汰赛弱队反扑",
        "severity": 3,
        "direction_adjust": {strong: -0.65, weak: 0.10, "draw": 1.05},
        "score_multipliers": {"3-0": 0.58, "3-1": 0.68, "0-3": 0.58, "1-3": 0.68},
        "boost_scores": ["1-1", "0-0", "1-0", "0-1", "2-1", "1-2"],
    }


def detect_T15_historical_deadlock(match_obj: Dict, fair: Dict) -> Optional[Dict]:
    """历史交锋僵局和平局保护。"""
    if abs(fair.get("home", 33.3) - fair.get("away", 33.3)) > 18:
        return None
    info = match_obj.get("points", {}) or {}
    text = " ".join(str(v) for v in info.values() if v)
    patterns = [
        r"对阵[^0-9]{0,20}(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
        r"历史交锋[^0-9]*(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
        r"交锋[^0-9]*(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
    ]
    best_w, best_d, best_l = 0, 0, 0
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            best_w, best_d, best_l = int(m.group(1)), int(m.group(2)), int(m.group(3))
            break
    total = best_w + best_d + best_l
    if total < 3:
        return None
    draw_rate = best_d / total
    if draw_rate < 0.40 and best_d < 3:
        return None
    s11 = _f(match_obj.get("s11", 999), 999)
    if not (0 < s11 < 9.0):
        return None
    return {
        "trap": "T15_HISTORICAL_DEADLOCK",
        "description": f"历史僵局:交锋{best_w}胜{best_d}平{best_l}负(平率{draw_rate:.0%})+s11赔{s11:.1f}",
        "severity": 2 if draw_rate >= 0.50 else 1,
        "direction_adjust": {"draw": 0.65, "home": -0.20, "away": -0.20},
        "score_multipliers": {"3-1": 0.75, "1-3": 0.75, "3-0": 0.70, "0-3": 0.70},
        "boost_scores": ["1-1", "0-0", "1-0", "0-1"],
    }


def detect_T16_sharp_badnews_conflict(match_obj: Dict, smart_signals: List, fair: Dict) -> Optional[Dict]:
    """Sharp 与该方坏消息冲突时，降低 sharp 方向，增加平局/对手保护。"""
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
    if fair.get(sharp_dir, 33.3) >= 55:
        return None
    other = "home" if sharp_dir == "away" else "away"
    return {
        "trap": "T16_SHARP_BADNEWS_CONFLICT",
        "description": f"Sharp({sharp_dir})+该方坏消息 → 对冲信号,平局优先",
        "severity": 2,
        "direction_adjust": {sharp_dir: -0.45, "draw": 0.80, other: 0.20},
        "score_multipliers": {},
        "boost_scores": ["1-1", "0-0"],
        "downgrade_sharp_trust": 0.35,
    }

def detect_all_traps(match_obj, engine_result, ai_responses, smart_signals, exp_goals):
    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_d = _f(match_obj.get("sp_draw", match_obj.get("same", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))
    fair_pack = fair_probs_from_1x2(sp_h, sp_d, sp_a, "power")
    fair = {k: fair_pack["fair_probs"].get(k, 33.3) for k in ["home", "draw", "away"]}

    detectors = [
        lambda: detect_T1_draw_trap(match_obj, engine_result, smart_signals, fair),
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
        lambda: detect_D17_balanced_draw_guard(match_obj, engine_result, fair),
        lambda: detect_D18_favorite_heat_guard(match_obj, fair),
        lambda: detect_D19_long_away_guard(match_obj, fair),
    ]

    traps = []
    for d in detectors:
        try:
            r = d()
            if r:
                traps.append(r)
        except Exception as e:
            logger.warning(f"trap detector 异常: {str(e)[:120]}")

    # 互斥与降噪：真平优先于假平；杯赛保护优先于假平；T2/T3 不应同时存在；闷平优先于诱小比分。
    if any(t.get("trap") == "T1_REAL_DRAW_SIGNAL" for t in traps):
        traps = [t for t in traps if t.get("trap") != "T1_FAKE_DRAW_TRAP"]
    if any(t.get("trap") == "T14_CUP_FAVORITE" for t in traps):
        traps = [t for t in traps if t.get("trap") != "T1_FAKE_DRAW_TRAP"]
    t2 = next((t for t in traps if t.get("trap") == "T2_HANDICAP_DEEPER"), None)
    t3 = next((t for t in traps if t.get("trap") == "T3_HANDICAP_SHALLOWER"), None)
    if t2 and t3:
        traps = [t for t in traps if t.get("trap") != ("T3_HANDICAP_SHALLOWER" if t2.get("severity", 1) >= t3.get("severity", 1) else "T2_HANDICAP_DEEPER")]
    if any(t.get("trap") == "T13_GOALLESS_DRAW" for t in traps):
        traps = [t for t in traps if t.get("trap") != "T6_SMALL_SCORE_TRAP"]
    t4 = next((t for t in traps if t.get("trap") in ["T4_FAKE_HOME_FAVORITE", "T5_FAKE_AWAY_FAVORITE"]), None)
    t8 = next((t for t in traps if t.get("trap") == "T8_FALSE_COLD"), None)
    if t4 and t8:
        if t4.get("severity", 1) >= t8.get("severity", 1):
            traps = [t for t in traps if t.get("trap") != "T8_FALSE_COLD"]
        else:
            traps = [t for t in traps if t.get("trap") not in ["T4_FAKE_HOME_FAVORITE", "T5_FAKE_AWAY_FAVORITE"]]

    direction_adjust = {"home": 0.0, "draw": 0.0, "away": 0.0}
    score_multipliers, boost_scores, flags = {}, [], {}
    total_severity, confidence_penalty = 0, 0
    suppress_contrarian = False
    xg_override = None
    sharp_trust_override = 1.0

    for t in traps:
        total_severity += t.get("severity", 1)
        for k, v in t.get("direction_adjust", {}).items():
            if k in direction_adjust:
                direction_adjust[k] += _f(v)
        for k, v in t.get("score_multipliers", {}).items():
            fv = _f(v, 1.0)
            if k in score_multipliers:
                score_multipliers[k] = min(score_multipliers[k], fv) if fv < 1 else max(score_multipliers[k], fv)
            else:
                score_multipliers[k] = fv
        boost_scores.extend(t.get("boost_scores", []))
        confidence_penalty += t.get("confidence_penalty", 0)
        if t.get("suppress_contrarian"):
            suppress_contrarian = True
        if t.get("xg_override"):
            xg_override = t.get("xg_override")
        if "downgrade_sharp_trust" in t:
            sharp_trust_override = min(sharp_trust_override, _f(t.get("downgrade_sharp_trust"), 1.0))
        for fk in ["draw_guard", "favorite_guard", "long_away_guard", "shallow_guard"]:
            if t.get(fk):
                flags[fk] = True

    direction_adjust = {k: round(_clip(v, -2.6, 2.6), 4) for k, v in direction_adjust.items()}
    sharp = detect_sharp_direction(smart_signals)
    steam = detect_steam_direction(smart_signals)

    return {
        "traps_detected": traps,
        "trap_count": len(traps),
        "total_severity": total_severity,
        "direction_adjust": direction_adjust,
        "score_multipliers": score_multipliers,
        "boost_scores": list(dict.fromkeys(boost_scores)),
        "confidence_penalty": confidence_penalty,
        "suppress_contrarian": suppress_contrarian,
        "xg_override": xg_override,
        "sharp_trust_override": sharp_trust_override,
        "steam_trust_override": sharp_trust_override,
        "flags": flags,
        "shin": fair,
        "fair_1x2": fair,
        "fair_1x2_method": fair_pack.get("method", "power"),
        "market_overround": fair_pack.get("overround", 0.0),
        "raw_implied_1x2": fair_pack.get("raw_implied", {}),
        "sharp_detected": sharp["detected"],
        "sharp_dir": sharp["sharp_dir"],
        "steam_dir": steam["steam_dir"],
        "steam_type": steam["steam_type"],
    }



# ==============================
# CRS 分析
# ==============================

def crs_implied_probabilities(match_obj):
    raw = {}
    for sc, key in CRS_FULL_MAP.items():
        odds = _f(match_obj.get(key, 0))
        if odds > 1.1:
            raw[sc] = odds
    extras = {}
    for key, scores in [("crs_win", SCORE_OTHERS_HOME), ("crs_same", SCORE_OTHERS_DRAW), ("crs_lose", SCORE_OTHERS_AWAY)]:
        odds = _f(match_obj.get(key, 0))
        if odds > 1.1:
            extras[key] = {"odds": odds, "scores": scores}
    if len(raw) < 8:
        return {}, 0.0, 0.0
    raw_sum = sum(1 / o for o in raw.values()) + sum(1 / ex["odds"] for ex in extras.values())
    if raw_sum <= 0:
        return {}, 0.0, 0.0
    probs = {sc: (1 / odds) / raw_sum * 100 for sc, odds in raw.items()}
    for ex in extras.values():
        total_prob = (1 / ex["odds"]) / raw_sum * 100
        per = total_prob / max(1, len(ex["scores"]))
        for sc in ex["scores"]:
            probs[sc] = probs.get(sc, 0) + per
    return probs, round(raw_sum - 1.0, 3), round(len(raw) / len(CRS_FULL_MAP), 2)

def compute_statistical_moments(probs):
    regular = {}
    for sc, p in probs.items():
        try:
            h, a = map(int, sc.split("-"))
            if h <= 8 and a <= 8:
                regular[(h, a)] = p
        except Exception:
            continue
    if not regular or sum(regular.values()) < 1:
        return {}
    total = sum(regular.values())
    norm = {k: v / total for k, v in regular.items()}
    eh = sum(h * p for (h, a), p in norm.items())
    ea = sum(a * p for (h, a), p in norm.items())
    vh = sum((h - eh) ** 2 * p for (h, a), p in norm.items())
    va = sum((a - ea) ** 2 * p for (h, a), p in norm.items())
    sh, sa = math.sqrt(max(vh, 1e-6)), math.sqrt(max(va, 1e-6))
    cov = sum((h - eh) * (a - ea) * p for (h, a), p in norm.items())
    corr = cov / (sh * sa) if sh * sa > 0 else 0
    return {"lambda_h": round(eh, 3), "lambda_a": round(ea, 3), "lambda_total": round(eh + ea, 3), "var_h": round(vh, 3), "var_a": round(va, 3), "corr": round(corr, 3)}

def compute_direction_from_crs(probs):
    out = {"home": 0.0, "draw": 0.0, "away": 0.0}
    for sc, p in probs.items():
        d = _score_direction(sc)
        if d in out:
            out[d] += p
    total = sum(out.values())
    return {k: round(v / total * 100, 2) for k, v in out.items()} if total > 0 else {"home": 33.3, "draw": 33.3, "away": 33.4}

def classify_shape(moments):
    if not moments:
        return "unknown", ["CRS数据不足"]
    lh, la, lt, corr = moments.get("lambda_h", 1.3), moments.get("lambda_a", 1.2), moments.get("lambda_total", 2.5), moments.get("corr", 0)
    if lt >= 3.0 and corr >= 0.15:
        return "shootout", [f"互射局:λ总{lt:.2f},corr{corr:.2f}"]
    if lt <= 2.2:
        return "grinder", [f"磨局:λ总{lt:.2f}"]
    if lh - la >= 1.2:
        return "lopsided_h", [f"主队碾压:λ主{lh:.2f} vs 客{la:.2f}"]
    if la - lh >= 1.2:
        return "lopsided_a", [f"客队碾压:λ客{la:.2f} vs 主{lh:.2f}"]
    if abs(lh - la) < 0.45:
        return "balanced", [f"均势:λ主{lh:.2f} vs 客{la:.2f}"]
    return "normal", []

def analyze_crs_matrix(match_obj):
    probs, margin, coverage = crs_implied_probabilities(match_obj)
    if not probs:
        return {"implied_probs": {}, "margin": 0.0, "coverage": 0.0, "moments": {}, "shape_verdict": "unknown", "anomalies": ["CRS数据缺失"], "direction_probs": {"home": 33.3, "draw": 33.3, "away": 33.4}, "top_scores": [], "low_rank_info": _crs_low_rank_info(match_obj)}
    moments = compute_statistical_moments(probs)
    verdict, anomalies = classify_shape(moments)
    top_scores = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:12]
    return {"implied_probs": {k: round(v, 2) for k, v in probs.items()}, "margin": margin, "coverage": coverage, "moments": moments, "shape_verdict": verdict, "anomalies": anomalies, "direction_probs": compute_direction_from_crs(probs), "top_scores": [(sc, round(p, 2)) for sc, p in top_scores], "low_rank_info": _crs_low_rank_info(match_obj)}


# ==============================
# 比分矩阵
# ==============================

def _estimate_base_lambdas(match_obj, engine_result, crs_analysis, exp_goals):
    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))
    moments = crs_analysis.get("moments", {}) if crs_analysis else {}
    ch, ca = _f(moments.get("lambda_h", 0)), _f(moments.get("lambda_a", 0))
    if hxg > 0.1 and axg > 0.1:
        if ch > 0.1 and ca > 0.1:
            return max(0.05, 0.58 * hxg + 0.42 * ch), max(0.05, 0.58 * axg + 0.42 * ca)
        return hxg, axg
    if ch > 0.1 and ca > 0.1:
        return ch, ca
    sp_h, sp_d, sp_a = _f(match_obj.get("sp_home", match_obj.get("win", 0))), _f(match_obj.get("sp_draw", match_obj.get("same", 0))), _f(match_obj.get("sp_away", match_obj.get("lose", 0)))
    fair = fair_probs_from_1x2(sp_h, sp_d, sp_a)["fair_probs"]
    total = exp_goals if 1.0 <= exp_goals <= 6.0 else 2.5
    edge = (fair.get("home", 33.3) - fair.get("away", 33.3)) / 100
    lh = total * (0.50 + max(-0.25, min(0.25, edge * 0.65)))
    return max(0.05, lh), max(0.05, total - lh)

def _btts_pct_from_sources(engine_result, match_obj):
    for k in ["btts", "both_score", "both_teams_score", "双方进球"]:
        v = engine_result.get(k) if isinstance(engine_result, dict) else None
        fv = _f(v, -1)
        if fv >= 0:
            return fv * 100 if fv <= 1 else fv
    if min(_f(match_obj.get("s11", 999), 999), _f(match_obj.get("w21", 999), 999), _f(match_obj.get("l12", 999), 999)) <= 8:
        return 53.0
    return 48.0

def apply_ttg_anchor_boost_to_matrix(matrix, match_obj, engine_result):
    if not matrix:
        return matrix
    m = dict(matrix)
    def mult(sc, f):
        if sc in m:
            m[sc] *= f
    a3, a4, a5, a6, a7 = [_f(match_obj.get(f"a{g}", 999), 999) for g in [3,4,5,6,7]]
    btts = _btts_pct_from_sources(engine_result, match_obj)
    if 0 < a3 <= 4.15:
        for sc, f in {"2-1":1.18, "1-2":1.18, "3-0":1.10, "0-3":1.10, "1-1":1.06}.items(): mult(sc, f)
    if 0 < a4 <= 4.85:
        for sc, f in {"2-2":1.62, "3-1":1.38, "1-3":1.38, "4-0":1.12, "0-4":1.12}.items(): mult(sc, f)
    if 0 < a5 <= 8.20:
        core = 1.78 if btts >= 52 else 1.42
        for sc, f in {"3-2":core, "2-3":core, "4-1":1.24, "1-4":1.24, "5-0":1.08, "0-5":1.08}.items(): mult(sc, f)
    if 0 < a6 <= 16.5:
        for sc, f in {"3-3":1.28, "4-2":1.22, "2-4":1.22, "5-1":1.12, "1-5":1.12}.items(): mult(sc, f)
    if 0 < a7 <= 18.0:
        for sc, f in {"5-2":1.45, "2-5":1.45, "4-3":1.42, "3-4":1.42, "6-1":1.18, "1-6":1.18}.items(): mult(sc, f)
    return _normalize_prob_dict(m)

def _mixed_direction_target(fair_1x2_pct, crs_analysis, trap_report, match_obj):
    fair = _normalize_prob_dict({k: _f(fair_1x2_pct.get(k, 33.3))/100 for k in VALID_DIRS})
    crs_pct = (crs_analysis or {}).get("direction_probs", {})
    cov = _f((crs_analysis or {}).get("coverage", 0))
    if crs_pct and cov >= 0.45:
        crs = _normalize_prob_dict({k: _f(crs_pct.get(k, 33.3))/100 for k in VALID_DIRS})
        fw, cw = 0.56, 0.34
    else:
        crs, fw, cw = dict(fair), 0.82, 0.08
    logits = {k: 0.0 for k in VALID_DIRS}
    for k, v in (trap_report or {}).get("direction_adjust", {}).items():
        if k in logits:
            logits[k] += _f(v) * 0.18
    if _has_low_draw_crs(match_obj, 5, 8.2):
        logits["draw"] += 0.18
    risk = _softmax_dict(logits, 1.2) or {k: 1/3 for k in VALID_DIRS}
    rw = 1 - fw - cw
    return _normalize_prob_dict({k: fw*fair[k] + cw*crs[k] + rw*risk[k] for k in VALID_DIRS})

def apply_trap_residual_to_matrix(matrix, trap_report):
    if not matrix:
        return matrix
    m = dict(matrix)
    adj = trap_report.get("direction_adjust", {}) if trap_report else {}
    for sc in list(m.keys()):
        d = _score_direction(sc)
        if d in adj:
            m[sc] *= math.exp(max(-0.60, min(0.60, _f(adj[d]) * 0.14)))
    for sc, mult in (trap_report.get("score_multipliers", {}) if trap_report else {}).items():
        if sc in m:
            m[sc] *= max(0.20, min(2.20, _f(mult, 1.0)))
    for sc in trap_report.get("boost_scores", []) if trap_report else []:
        if sc in m:
            m[sc] *= 1.22
    return _normalize_prob_dict(m)

def _direction_of_score(sc):
    return _score_direction(sc) or "draw"

def _total_of_score(sc):
    h, a = _parse_score(sc)
    return 99 if h is None else h + a

def _ipf_fit_direction_and_ttg(matrix, target_dir, ttg, loops=16):
    matrix = _normalize_prob_dict(matrix)
    for _ in range(loops):
        cur = {"home":0, "draw":0, "away":0}
        for sc, p in matrix.items():
            cur[_direction_of_score(sc)] += p
        for sc in list(matrix.keys()):
            d = _direction_of_score(sc)
            if cur.get(d, 0) > 1e-9:
                matrix[sc] *= max(0.42, min(2.35, target_dir.get(d, cur[d]) / cur[d]))
        matrix = _normalize_prob_dict(matrix)
        if ttg:
            cur_ttg = {}
            for sc, p in matrix.items():
                b = _total_of_score(sc)
                b = b if b <= 7 else 7
                cur_ttg[b] = cur_ttg.get(b, 0) + p
            for sc in list(matrix.keys()):
                b = _total_of_score(sc)
                b = b if b <= 7 else 7
                if b in ttg and cur_ttg.get(b, 0) > 1e-9:
                    matrix[sc] *= max(0.42, min(2.35, ttg[b] / cur_ttg[b]))
            matrix = _normalize_prob_dict(matrix)
    return matrix

def build_unified_score_matrix(match_obj, engine_result, crs_analysis, trap_report, exp_goals, max_goals=8):
    max_goals = max(5, min(10, int(max_goals or 8)))
    sp_h, sp_d, sp_a = _f(match_obj.get("sp_home", match_obj.get("win", 0))), _f(match_obj.get("sp_draw", match_obj.get("same", 0))), _f(match_obj.get("sp_away", match_obj.get("lose", 0)))
    fair_pack = fair_probs_from_1x2(sp_h, sp_d, sp_a, "power")
    target_dir = _mixed_direction_target(fair_pack["fair_probs"], crs_analysis or {}, trap_report or {}, match_obj)
    ttg = fair_probs_from_ttg(match_obj, "power")
    lh, la = _estimate_base_lambdas(match_obj, engine_result or {}, crs_analysis or {}, exp_goals)
    base = {f"{h}-{a}": _poisson_pmf(lh, h) * _poisson_pmf(la, a) for h in range(max_goals+1) for a in range(max_goals+1)}
    base = _normalize_prob_dict(base)
    crs_pct = crs_analysis.get("implied_probs", {}) if crs_analysis else {}
    cov = _f(crs_analysis.get("coverage", 0)) if crs_analysis else 0
    crs_grid = {sc: max(1e-9, crs_pct.get(sc, 0)/100) for sc in base}
    if sum(crs_grid.values()) > 0.05 and cov >= 0.45:
        crs_grid = _normalize_prob_dict(crs_grid, floor=1e-9)
        w = min(0.68, 0.32 + cov * 0.42)
        matrix = _normalize_prob_dict({sc: (base[sc] ** (1-w)) * (crs_grid[sc] ** w) for sc in base})
    else:
        matrix = dict(base)
    matrix = apply_ttg_anchor_boost_to_matrix(matrix, match_obj, engine_result or {})
    matrix = apply_trap_residual_to_matrix(matrix, trap_report)
    matrix = _ipf_fit_direction_and_ttg(matrix, target_dir, ttg, loops=16)
    dir_probs = {"home":0, "draw":0, "away":0}
    goal_probs = {}
    for sc, p in matrix.items():
        d = _direction_of_score(sc)
        b = _total_of_score(sc); b = b if b <= 7 else 7
        dir_probs[d] += p
        goal_probs[b] = goal_probs.get(b, 0) + p
    top_scores = sorted(matrix.items(), key=lambda x: x[1], reverse=True)
    return {"matrix": matrix, "fair_1x2": fair_pack, "mixed_target_dir": {k: round(v*100,2) for k,v in target_dir.items()}, "direction_probs": {k: round(v*100,2) for k,v in dir_probs.items()}, "goal_probs": {k: round(v*100,2) for k,v in sorted(goal_probs.items())}, "top_scores": [(sc, round(p*100,3)) for sc,p in top_scores[:30]], "lambda_h": round(lh,3), "lambda_a": round(la,3), "source": "unified_score_matrix_v18_2_final_ipf"}

def matrix_direction(matrix):
    out = {"home":0, "draw":0, "away":0}
    for sc, p in matrix.items():
        d = _score_direction(sc)
        if d in out:
            out[d] += p
    return {k: round(v*100,2) for k,v in out.items()}

def select_score_from_matrix(matrix, direction, goal_range, ai_votes=None):
    ai_votes = ai_votes or {}
    gmin, gmax = goal_range
    candidates, raw_probs = {}, {}
    for sc, p in matrix.items():
        if _score_direction(sc) != direction:
            continue
        h, a = _parse_score(sc)
        if h is None:
            continue
        total = h + a
        if gmin <= total <= gmax:
            raw = p * 100
            rank = raw * (1 + min(0.18, ai_votes.get(sc, 0) * 0.025))
            candidates[sc] = rank
            raw_probs[sc] = raw
    if not candidates:
        fallback = {"home":"1-0", "draw":"1-1", "away":"0-1"}.get(direction, "1-1")
        return fallback, [(fallback, 1.0)], matrix.get(fallback, 0.01) * 100
    sorted_scores = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    best = sorted_scores[0][0]
    return best, [(sc, round(v,3)) for sc,v in sorted_scores[:12]], round(raw_probs.get(best, 0), 3)

def confidence_rank_score(dir_probs_pct, top_score_candidates, trap_report, ai_valid_count):
    vals = sorted([_f(v) for v in dir_probs_pct.values()], reverse=True)
    top = vals[0] if vals else 33.3
    gap = vals[0] - vals[1] if len(vals) >= 2 else 0
    score = 40 + min(28, (top - 33.3)*0.70) + min(15, gap*0.78)
    if top_score_candidates:
        score += min(9, _f(top_score_candidates[0][1]) * 0.75)
    flags = trap_report.get("flags", {}) if trap_report else {}
    if flags.get("draw_guard") or flags.get("favorite_guard") or flags.get("long_away_guard"):
        score -= 5
    if trap_report and trap_report.get("total_severity", 0) >= 7:
        score -= 7
    elif trap_report and trap_report.get("trap_count", 0) >= 2:
        score -= 3
    if ai_valid_count <= 1:
        score -= 5
    elif ai_valid_count >= 3:
        score += 2
    if gap < 6:
        score = min(score, 58)
    if top < 43:
        score = min(score, 60)
    score = int(max(25, min(90, round(score))))
    return score, "低" if score >= 72 else ("中" if score >= 53 else "高")


def determine_goal_range(direction, moments, exp_goals, trap_report, match_obj, engine_result):
    actual = _parse_actual_handicap_signed(match_obj)
    a4, a5, a6, a7 = _f(match_obj.get("a4",999),999), _f(match_obj.get("a5",999),999), _f(match_obj.get("a6",999),999), _f(match_obj.get("a7",999),999)
    extreme = 0
    extreme += 2 if 0 < a7 <= 18 else 1 if 0 < a7 <= 26 else 0
    extreme += 1 if 0 < a6 <= 15 else 0
    extreme += 1 if 0 < a5 <= 8 else 0
    if direction == "home" and _handicap_depth_for_side(actual, "home") >= 1.75:
        extreme += 1
    if direction == "away" and _handicap_depth_for_side(actual, "away") >= 1.75:
        extreme += 1
    lt = moments.get("lambda_total", exp_goals) if moments else exp_goals
    lt_avg = lt*0.6 + exp_goals*0.4
    lg = str(match_obj.get("league", match_obj.get("cup", "")))
    if any(k in lg for k in LEAGUE_LOW_GOALS):
        lt_avg -= 0.15
    if any(k in lg for k in LEAGUE_HIGH_GOALS):
        lt_avg += 0.18
    flags = trap_report.get("flags", {}) if trap_report else {}
    if flags.get("draw_guard"):
        return (2,4,"draw_guard") if 0 < a4 <= 4.85 else (0,3,"draw_guard")
    if 0 < a5 <= 8.2 and lt_avg >= 2.65:
        return 2,5,"five_goal_anchor"
    if 0 < a4 <= 4.85 and lt_avg >= 2.35:
        return 2,4,"four_goal_anchor"
    if extreme >= 5 and lt_avg >= 3.35:
        return 4,8,"extreme_blowout"
    if lt_avg >= 3.35:
        return 3,6,"shootout"
    if lt_avg >= 2.85:
        return 2,5,"high_goals"
    if lt_avg >= 2.25:
        return 2,4,"normal"
    if lt_avg >= 1.75:
        return 1,3,"low_goals"
    return 0,2,"grinder"


# ==============================
# AI Prompt / API
# ==============================

def _format_external_context_for_prompt(ctx):
    if not isinstance(ctx, dict):
        return "source_quality=missing\n"
    lines = [f"source_quality={ctx.get('source_quality','missing')}; data_missing={ctx.get('data_missing', [])}"]
    for k, label in [("injuries","伤停"),("suspensions","停赛"),("lineup_news","阵容"),("weather","天气"),("motivation","战意"),("schedule_pressure","赛程"),("odds_history","赔率历史"),("sharp_market_notes","资金"),("news_snippets","联网摘要")]:
        if ctx.get(k):
            lines.append(f"{label}: {_json_compact(ctx.get(k), 2500)}")
    return "\n".join(lines) + "\n"

def build_v18_prompt(match_analyses):
    p = "<context>\n你正在中国体彩竞彩足球市场进行量化比分预测。核心任务是识别1X2、让球、CRS、总进球、资金/情报之间的定价裂缝。若external_context没有给出某项联网材料，禁止编造伤停、天气、新闻、首发。\n</context>\n\n"
    p += "<iron_rules>\n1 top3[0].score必须与final_direction一致。\n2 0-1/0-2/0-3是合法客胜比分。\n3 盘口坐标:主让负，主受让正，客让正，客受让负。\n4 强方浅盘必须防平或弱方。\n5 4球锚点落2-2/3-1/1-3；5球锚点落3-2/2-3；7球锚点落5-2/2-5/4-3/3-4。\n6 其他比分按胜其他/平其他/负其他理解。\n</iron_rules>\n\n"
    p += "<output_format>\n严格JSON数组。字段: match, final_direction, top3, reason, ai_confidence, risk_level, detected_traps, must_review_flags, data_missing, audit。audit包含fair_1x2,handicap,crs,ttg,external_context,risk,rejection。禁止JSON之外文本。\n</output_format>\n\n<match_data>\n"
    for i, ma in enumerate(match_analyses):
        m, eng, stats = ma["match"], ma.get("engine", {}), ma.get("stats", {})
        trap, crs, ctx = ma.get("trap_preview", {}), ma.get("crs_preview", {}), ma.get("external_context", {})
        h, a, lg = m.get("home_team", "Home"), m.get("away_team", "Away"), m.get("league", m.get("cup", ""))
        sp_h, sp_d, sp_a = _f(m.get("sp_home", m.get("win",0))), _f(m.get("sp_draw", m.get("same",0))), _f(m.get("sp_away", m.get("lose",0)))
        actual = _parse_actual_handicap_signed(m)
        theory = _infer_theoretical_handicap_signed(sp_h, sp_a)
        strong = _strong_side_from_1x2(sp_h, sp_a)
        diff = _handicap_diff_for_strong_side(actual, theory, strong)
        p += f'<match index="{i+1}">\n[{i+1}] {h} vs {a} | {lg}\n'
        p += f"欧赔:{sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f}; 原始让球:{m.get('give_ball','0')}; 理论盘口{theory:.2f}; 实际盘口{actual:.2f}; 强方={strong}; 强方深浅差={diff:+.2f}\n"
        fair = trap.get("fair_1x2", {})
        if fair:
            p += f"公平1X2: 主{fair.get('home',0):.1f}% 平{fair.get('draw',0):.1f}% 客{fair.get('away',0):.1f}%\n"
        p += f"庄家隐含xG: 主{eng.get('bookmaker_implied_home_xg','?')} 客{eng.get('bookmaker_implied_away_xg','?')}\n"
        p += f"CRS矩:{crs.get('moments',{})}; 形状={crs.get('shape_verdict','unknown')}; CRS低赔={crs.get('low_rank_info',{}).get('low_scores',[])}\n"
        p += f"系统陷阱:{[(t.get('trap'), t.get('description','')[:120]) for t in trap.get('traps_detected', [])]}\n"
        p += "总进球:" + " | ".join([f"{g}={m.get(f'a{g}','')}" for g in range(8)]) + "\n"
        crs_lines = [f"{sc}={_f(m.get(key,0)):.1f}" for sc,key in CRS_FULL_MAP.items() if _f(m.get(key,0))>1]
        if crs_lines:
            p += "CRS:" + " | ".join(crs_lines) + "\n"
        others = [f"{label}={m.get(k)}" for k,label in [("crs_win","胜其他"),("crs_same","平其他"),("crs_lose","负其他")] if m.get(k)]
        if others:
            p += "其他比分:" + " | ".join(others) + "\n"
        if m.get("vote"):
            p += f"散户:{m.get('vote')}\n"
        if m.get("change"):
            p += f"赔率变动:{m.get('change')}，负数=降水\n"
        points = m.get("points", {})
        if isinstance(points, dict):
            for k in ["home_strength", "guest_strength", "match_points"]:
                if points.get(k):
                    p += f"情报:{str(points.get(k))[:800].replace(chr(10),' ')}\n"
        sigs = stats.get("smart_signals", []) if isinstance(stats, dict) else []
        if sigs:
            p += f"信号:{', '.join(str(x) for x in sigs[:16])}\n"
        p += "<external_context>\n" + _format_external_context_for_prompt(ctx) + "</external_context>\n</match>\n\n"
    p += "</match_data>\n"
    return p

def build_claude_final_audit_prompt(match_analyses, phase1_results):
    p = "<final_audit_context>\n你是最终审计模型。不能按票数裁决，必须重新审计1X2、盘口深浅、CRS、总进球、外部情报和风控。若出现T1_REAL_DRAW_SIGNAL/D17/D18/D19/T3，即使三家一致也必须复审。若top1不在CRS低赔前8且不符合a0-a7锚点，ai_confidence不得高于68。\n</final_audit_context>\n\n"
    p += build_v18_prompt(match_analyses)
    p += "\n<phase1_ai_results>\n"
    for ai in ["gpt", "grok", "gemini"]:
        p += f"<{ai}>\n"
        rs = phase1_results.get(ai, {})
        for idx in range(1, len(match_analyses)+1):
            r = rs.get(idx, {})
            p += json.dumps({"match": idx, "ai_score": r.get("ai_score"), "top3": r.get("top3", []), "final_direction": r.get("final_direction",""), "ai_confidence": r.get("ai_confidence",60), "detected_traps": r.get("detected_traps", []), "reason": r.get("reason",""), "audit": r.get("audit", {})}, ensure_ascii=False) + "\n" if r else f"[{idx}] 弃权\n"
        p += f"</{ai}>\n"
    p += "</phase1_ai_results>\n<final_output_rule>\n严格输出JSON数组。字段: match, final_direction, top3, reason, ai_confidence, risk_level, detected_traps, must_review_flags, data_missing, audit。禁止JSON之外内容。\n</final_output_rule>"
    return p


FALLBACK_URLS = [None, "https://www.api522.pro/v1", "https://api522.pro/v1", "https://api521.pro/v1", "http://69.63.213.33:666/v1"]
GPT_DEFAULT_URL = "https://ai.newapi.life/v1"
GPT_DEFAULT_KEY = ""

def get_clean_env_url(name, default=""):
    v = str(os.environ.get(name, globals().get(name, default))).strip(" \t\n\r\"'")
    m = re.search(r"(https?://[a-zA-Z0-9._:/?#=&%-]+)", v)
    return m.group(1) if m else v

def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")

def _mask_key(k):
    return "" if not k else "***" if len(k) <= 8 else f"{k[:4]}...{k[-4:]}"

def debug_ai_config():
    for name, url_env, key_env in [("GPT","GPT_API_URL","GPT_API_KEY"),("GROK","GROK_API_URL","GROK_API_KEY"),("GEMINI","GEMINI_API_URL","GEMINI_API_KEY"),("CLAUDE","CLAUDE_API_URL","CLAUDE_API_KEY")]:
        print(f"[AI CONFIG] {name}: url={get_clean_env_url(url_env)} key={_mask_key(get_clean_env_key(key_env))}")
    print(f"[EXTERNAL] enabled={_external_context_enabled()} endpoints={_parse_external_endpoints()} bing={bool(os.environ.get('BING_SEARCH_API_KEY'))}")

async def async_call_one_ai_batch(session, prompt, url_env, key_env, models_list, num_matches, ai_name):
    key = get_clean_env_key(key_env)
    if not key and ai_name == "gpt":
        key = GPT_DEFAULT_KEY
    if not key:
        return ai_name, {}, "no_key"
    primary = get_clean_env_url(url_env, GPT_DEFAULT_URL if ai_name == "gpt" else "")
    urls = [primary or GPT_DEFAULT_URL] if ai_name == "gpt" else [primary] + [u for u in FALLBACK_URLS if u and u != primary][:1]
    profiles = {
        "claude": {"temp":0.14, "sys":"你是最终审计模型。只输出JSON数组。必须复查1X2、盘口、CRS、总进球、外部情报、资金和风控。"},
        "gpt": {"temp":0.18, "sys":"你是衍生品定价+比分分布量化策略师。只输出JSON数组。"},
        "grok": {"temp":0.24, "sys":"你是另类数据和市场情绪分析师。只输出JSON数组，不得编造未给数据。"},
        "gemini": {"temp":0.15, "sys":"你是非线性特征和多市场共振识别模型。只输出JSON数组。"},
    }
    profile = profiles.get(ai_name, profiles["gpt"])
    for mn in models_list:
        connected = False
        for base in urls:
            if not base:
                continue
            is_gem = "generateContent" in base
            url = base.rstrip("/")
            if not is_gem and "chat/completions" not in url:
                url += "/chat/completions"
            headers = {"Content-Type": "application/json"}
            if is_gem:
                headers["x-goog-api-key"] = key
                payload = {"contents":[{"role":"user","parts":[{"text":prompt}]}],"generationConfig":{"temperature":profile["temp"]},"systemInstruction":{"parts":[{"text":profile["sys"]}]}}
            else:
                headers["Authorization"] = f"Bearer {key}"
                payload = {"model": mn, "messages":[{"role":"system","content":profile["sys"]},{"role":"user","content":prompt}], "temperature": profile["temp"]}
            print(f"  [连接中] {ai_name.upper()} | {mn[:36]} @ {url.split('/v1')[0][:55]}")
            t0 = time.time()
            try:
                timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_connect=20, sock_read={"claude":620,"grok":500,"gpt":500,"gemini":500}.get(ai_name,420))
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                    elapsed = round(time.time()-t0,1)
                    if r.status in (502,504,429):
                        print(f"    HTTP {r.status} | {elapsed}s → 换URL")
                        continue
                    if r.status == 400:
                        print(f"    HTTP 400 | {elapsed}s → 换模型")
                        break
                    if r.status != 200:
                        txt = await r.text()
                        print(f"    HTTP {r.status} | {elapsed}s → 换URL | {txt[:120]}")
                        continue
                    connected = True
                    data = await r.json(content_type=None)
                    raw = _extract_response_text(data, is_gem, ai_name)
                    if not raw:
                        _save_debug_dump(ai_name, data, "empty")
                        break
                    results = _parse_ai_json(raw, num_matches)
                    if results:
                        print(f"    {ai_name.upper()} 完成: {len(results)}/{num_matches} | {round(time.time()-t0,1)}s")
                        return ai_name, results, mn
                    _save_debug_dump(ai_name, data, "parse0")
                    _save_debug_text(ai_name, raw, "parse0_raw")
                    print(f"    解析0条，raw前200字: {raw[:200].replace(chr(10), chr(32))}")
                    break
            except asyncio.TimeoutError:
                return ai_name, {}, "read_timeout" if connected else "connect_timeout"
            except Exception as e:
                if not connected:
                    print(f"    {str(e)[:80]} → 换URL")
                    continue
                return ai_name, {}, "error"
    return ai_name, {}, "all_failed"


def _extract_response_text(data, is_gem, ai_name):
    """
    v18.2.2 修复点:
    1) 兼容 OpenAI chat/completions、Responses API、Gemini generateContent、部分中转站魔改字段。
    2) 不只读取 message.content；当供应商把正文塞进 output_text/text/result/answer 等字段时也能提取。
    3) 如果常规字段为空，会递归扫描包含 JSON 痕迹的字符串，避免 API 有输出但解析层拿不到正文。
    """
    candidates = []

    def add_text(v, source=""):
        if isinstance(v, str):
            t = v.strip()
            if t:
                candidates.append((source, t))
        elif isinstance(v, list):
            for i, it in enumerate(v):
                add_text(it, f"{source}[{i}]")
        elif isinstance(v, dict):
            # 常见文本字段优先
            for k in [
                "content", "text", "output_text", "answer", "response", "result",
                "completion", "final_answer", "message_content", "assistant_content",
                "model_response", "generated_text",
            ]:
                if k in v:
                    add_text(v.get(k), f"{source}.{k}")

            # content parts
            if isinstance(v.get("parts"), list):
                for j, part in enumerate(v["parts"]):
                    if isinstance(part, dict):
                        add_text(part.get("text"), f"{source}.parts[{j}].text")

            # output content blocks
            if isinstance(v.get("content"), list):
                for j, ct in enumerate(v["content"]):
                    if isinstance(ct, dict):
                        add_text(ct.get("text"), f"{source}.content[{j}].text")
                        add_text(ct.get("output_text"), f"{source}.content[{j}].output_text")

    try:
        if is_gem:
            for cand in data.get("candidates", []) if isinstance(data, dict) else []:
                content = cand.get("content", {}) if isinstance(cand, dict) else {}
                for part in content.get("parts", []) if isinstance(content, dict) else []:
                    if isinstance(part, dict):
                        add_text(part.get("text"), "gemini.candidates.content.parts.text")
            if candidates:
                return max((t for _, t in candidates), key=len).strip()

        if isinstance(data, dict) and data.get("choices"):
            for ci, choice in enumerate(data.get("choices", [])):
                if not isinstance(choice, dict):
                    continue
                msg = choice.get("message", {})
                if isinstance(msg, dict):
                    add_text(msg.get("content"), f"choices[{ci}].message.content")
                    # 某些中转会把最终答案塞到这些字段
                    for field in [
                        "text", "answer", "response", "output_text", "final_answer",
                        "output", "result", "completion", "message_content",
                        "assistant_content", "model_response",
                    ]:
                        add_text(msg.get(field), f"choices[{ci}].message.{field}")

                    # reasoning_content 不用于展示，但如果供应商错误地把 JSON 放在这里，允许作为结构化解析兜底
                    for field in ["reasoning_content", "reasoning", "thinking", "reasoning_text"]:
                        val = msg.get(field)
                        if isinstance(val, str) and ('"match"' in val or "'match'" in val or "top3" in val):
                            add_text(val, f"choices[{ci}].message.{field}")
                add_text(choice.get("text"), f"choices[{ci}].text")

        # OpenAI Responses API / 其他 output 数组
        if isinstance(data, dict) and isinstance(data.get("output"), list):
            for oi, out_item in enumerate(data["output"]):
                add_text(out_item, f"output[{oi}]")

        # 常见顶层字段
        if isinstance(data, dict):
            for field in ["output_text", "text", "answer", "response", "result", "content"]:
                add_text(data.get(field), f"top.{field}")

        # 递归兜底：只收集看起来包含 JSON 结果的字符串，避免拿到无关字段
        def walk(obj, path="root"):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    walk(v, f"{path}[{i}]")
            elif isinstance(obj, str):
                s = obj.strip()
                if len(s) >= 20 and ("match" in s or "top3" in s or "final_direction" in s) and ("[" in s or "{" in s):
                    candidates.append((path, s))

        walk(data)

        if candidates:
            # 优先选择包含 JSON 数组和 match/top3 的文本，其次最长文本
            def score_item(item):
                src, t = item
                score = len(t)
                if re.search(r'\[\s*\{', t):
                    score += 50000
                if '"match"' in t or "'match'" in t:
                    score += 30000
                if "top3" in t:
                    score += 20000
                if "final_direction" in t:
                    score += 10000
                return score
            src, best = max(candidates, key=score_item)
            if os.environ.get("AI_PARSE_DEBUG", "").lower() in ("1", "true", "yes"):
                print(f"    提取响应字段: {src} | {len(best)}字")
            return best.strip()

    except Exception as ex:
        print(f"    响应文本提取异常: {str(ex)[:100]}")

    try:
        s = json.dumps(data, ensure_ascii=False)
        m = re.search(r'\[\s*\{\s*\\?"match\\?"', s)
        if m:
            return s[m.start():]
    except Exception:
        pass
    return ""


def _balanced_json_slice(text: str, start_idx: int) -> str:
    """从 start_idx 位置截取完整 JSON array/object，处理字符串中的括号。"""
    if start_idx < 0 or start_idx >= len(text):
        return ""
    opener = text[start_idx]
    closer = "]" if opener == "[" else "}"
    depth = 0
    in_str = False
    esc = False
    quote = ""
    for i in range(start_idx, len(text)):
        ch = text[i]
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
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start_idx:i + 1]
    return ""


def _try_load_json_like(s: str):
    if not s:
        return None
    raw = s.strip()
    # 去掉代码围栏残留
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.I).strip()
    raw = re.sub(r"```$", "", raw).strip()
    # 修复常见尾逗号
    raw2 = re.sub(r",\s*([}\]])", r"\1", raw)
    for cand in [raw, raw2]:
        try:
            return json.loads(cand)
        except Exception:
            pass
    # 某些模型会输出 Python 单引号风格
    try:
        import ast
        return ast.literal_eval(raw2)
    except Exception:
        return None


def _extract_result_array(raw_text: str):
    clean = raw_text or ""
    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", clean, flags=re.S | re.I)
    clean = re.sub(r"```[a-zA-Z0-9_-]*", "", clean).strip()

    # 如果是被 JSON 字符串转义过的一整段，先尝试反转义
    if '\\"match\\"' in clean or '\\n' in clean:
        try:
            maybe = json.loads('"' + clean.replace('"', '\\"') + '"')
            if isinstance(maybe, str) and len(maybe) > len(clean) * 0.5:
                clean = maybe
        except Exception:
            clean = clean.replace('\\"', '"').replace('\\n', '\n')

    # 直接整体解析
    obj = _try_load_json_like(clean)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ["predictions", "results", "matches", "data", "output", "items"]:
            if isinstance(obj.get(k), list):
                return obj[k]

    # 优先找包含 match 的数组
    starts = [m.start() for m in re.finditer(r"\[", clean)]
    candidate_arrays = []
    for st in starts:
        frag = _balanced_json_slice(clean, st)
        if not frag:
            continue
        if "match" not in frag and "top3" not in frag and "score" not in frag:
            continue
        parsed = _try_load_json_like(frag)
        if isinstance(parsed, list):
            candidate_arrays.append(parsed)

    if candidate_arrays:
        # 选择包含 dict 数量最多的数组
        return max(candidate_arrays, key=lambda arr: sum(isinstance(x, dict) for x in arr))

    # 可能是对象包数组，例如 {"predictions":[...]}
    for st in [m.start() for m in re.finditer(r"\{", clean)]:
        frag = _balanced_json_slice(clean, st)
        if not frag:
            continue
        parsed = _try_load_json_like(frag)
        if isinstance(parsed, dict):
            for k in ["predictions", "results", "matches", "data", "output", "items"]:
                if isinstance(parsed.get(k), list):
                    return parsed[k]

    return []


def _score_from_candidate(obj) -> str:
    if isinstance(obj, str):
        return _normalize_score_text(obj)
    if not isinstance(obj, dict):
        return ""
    for k in ["score", "predicted_score", "ai_score", "final_score", "比分", "预测比分", "top_score"]:
        if obj.get(k) is not None:
            sc = _normalize_score_text(obj.get(k))
            if _score_direction(sc):
                return sc
    # 有些模型输出 {"home":2,"away":1}
    hv = obj.get("home_goals", obj.get("home", obj.get("主", None)))
    av = obj.get("away_goals", obj.get("away", obj.get("客", None)))
    if hv is not None and av is not None:
        try:
            return f"{int(float(hv))}-{int(float(av))}"
        except Exception:
            pass
    return ""


def _normalize_top3(item: Dict[str, Any]) -> List[Any]:
    raw_top3 = None
    for k in ["top3", "top_3", "top_scores", "scores", "score_candidates", "candidates", "比分候选"]:
        if isinstance(item.get(k), list):
            raw_top3 = item.get(k)
            break
    if raw_top3 is None:
        raw_top3 = []

    top3 = []
    for cand in raw_top3[:5]:
        sc = _score_from_candidate(cand)
        if not sc:
            continue
        if isinstance(cand, dict):
            out = dict(cand)
            out["score"] = sc
            if "prob" not in out:
                out["prob"] = out.get("probability", out.get("pct", out.get("概率", 0)))
            top3.append(out)
        else:
            top3.append({"score": sc, "prob": 0})
        if len(top3) >= 3:
            break

    if not top3:
        sc = _score_from_candidate(item)
        if sc:
            top3 = [{"score": sc, "prob": item.get("prob", item.get("probability", 0))}]
    return top3


def _parse_ai_json(raw_text, num_matches):
    arr = _extract_result_array(raw_text)
    results = {}
    if not isinstance(arr, list):
        return results

    for pos, item in enumerate(arr, 1):
        if not isinstance(item, dict):
            continue

        mid_raw = None
        for k in ["match", "match_id", "match_index", "index", "idx", "场次", "序号"]:
            if item.get(k) is not None:
                mid_raw = item.get(k)
                break
        if mid_raw is None:
            # 如果数组长度等于比赛数，允许按顺序兜底
            mid = pos if len(arr) == num_matches else None
        else:
            try:
                # 支持 "1" / "001" / "match 1"
                mm = re.search(r"\d+", str(mid_raw))
                mid = int(mm.group(0)) if mm else int(mid_raw)
            except Exception:
                mid = None
        if mid is None or mid < 1 or mid > max(num_matches, 9999):
            continue

        top3 = _normalize_top3(item)
        if not top3:
            continue

        sc = _score_from_candidate(top3[0])
        if not sc or not _score_direction(sc):
            continue

        final_dir = item.get("final_direction", item.get("direction", item.get("result", "")))
        final_dir = _dir_from_cn(final_dir) or final_dir
        parsed = _score_direction(sc)
        if parsed and final_dir not in VALID_DIRS:
            final_dir = parsed

        conf = item.get("ai_confidence", item.get("confidence", item.get("置信度", 60)))
        reason = (
            item.get("reason") or item.get("analysis") or item.get("rationale") or
            item.get("market_logic") or item.get("结论理由") or ""
        )
        if not reason and item.get("audit"):
            reason = _json_compact(item.get("audit", {}), 2500)

        traps = item.get("detected_traps", item.get("traps", item.get("risk_flags", [])))
        if not isinstance(traps, list):
            traps = [str(traps)] if traps else []

        results[mid] = {
            "top3": top3,
            "ai_score": sc,
            "reason": str(reason),
            "ai_confidence": int(_clip(_f(conf, 60), 0, 100)),
            "is_score_others": _score_display_label(sc) in ("胜其他", "平其他", "负其他"),
            "detected_traps": traps,
            "final_direction": final_dir,
            "audit": item.get("audit", {}),
            "risk_level": item.get("risk_level", item.get("risk", "")),
            "must_review_flags": item.get("must_review_flags", []),
            "data_missing": item.get("data_missing", []),
        }

    if not results and os.environ.get("AI_PARSE_DEBUG", "").lower() in ("1", "true", "yes"):
        print(f"    JSON解析为空，raw前500字: {(raw_text or '')[:500]}")
    return results

def _save_debug_dump(ai_name, data, tag):
    try:
        os.makedirs("data/debug", exist_ok=True)
        fn = f"data/debug/{ai_name}_{tag}_{int(time.time())}.json"
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"    失败响应已保存: {fn}")
    except Exception:
        pass

def _save_debug_text(ai_name, text, tag):
    try:
        os.makedirs("data/debug", exist_ok=True)
        fn = f"data/debug/{ai_name}_{tag}_{int(time.time())}.txt"
        with open(fn, "w", encoding="utf-8") as f:
            f.write(str(text or ""))
        print(f"    原始文本已保存: {fn}")
    except Exception:
        pass

async def run_ai_matrix_two_phase(match_analyses):
    if aiohttp is None:
        return {"gpt": {}, "grok": {}, "gemini": {}, "claude": {}}
    num = len(match_analyses)
    prompt = build_v18_prompt(match_analyses)
    print(f"  [v18.2 Phase1 Prompt] {len(prompt):,}字符 → GPT/Grok/Gemini")
    configs = [
        ("grok","GROK_API_URL","GROK_API_KEY",["熊猫-A-5-grok-4.2-fast-200w上下文"]),
        ("gpt","GPT_API_URL","GPT_API_KEY",["gpt-5.4"]),
        ("gemini","GEMINI_API_URL","GEMINI_API_KEY",["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking"]),
    ]
    all_results = {"gpt": {}, "grok": {}, "gemini": {}, "claude": {}}
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=8, use_dns_cache=False)) as session:
        rs = await asyncio.gather(*[async_call_one_ai_batch(session, prompt, u, k, m, num, n) for n,u,k,m in configs], return_exceptions=True)
        for r in rs:
            if isinstance(r, tuple):
                all_results[r[0]] = r[1]
        audit_prompt = build_claude_final_audit_prompt(match_analyses, all_results)
        print(f"  [v18.2 Phase2 Claude Audit] {len(audit_prompt):,}字符")
        _, cr, _ = await async_call_one_ai_batch(session, audit_prompt, "CLAUDE_API_URL", "CLAUDE_API_KEY", ["熊猫-69-满血openrouter-claude-opus-4.7-上下文1000k"], num, "claude")
        all_results["claude"] = cr or {}
    print(f"  [完成] {sum(1 for v in all_results.values() if v)}/4 AI有数据")
    return all_results


# ==============================
# 决策链 / 合并
# ==============================

def decision_lock_chain(match_obj, engine_result, trap_report, crs_analysis, ai_responses, smart_signals, exp_goals):
    ai_dirs = {"home":0.0, "draw":0.0, "away":0.0}
    ai_votes = {}
    weights = {"claude":1.18, "gpt":0.88, "grok":0.88, "gemini":0.88}
    for name, r in ai_responses.items():
        if not isinstance(r, dict):
            continue
        sc = str(r.get("ai_score", "")).strip()
        if _parse_score(sc)[0] is None and r.get("top3"):
            first = r["top3"][0]
            sc = str(first.get("score", "") if isinstance(first, dict) else first)
        h, a = _parse_score(sc)
        if h is None:
            continue
        w = weights.get(name, 0.8)
        d = "home" if h > a else "away" if h < a else "draw"
        ai_dirs[d] += w
        sc = _normalize_score_text(sc)
        ai_votes[sc] = ai_votes.get(sc, 0) + w
        for rank, t in enumerate(r.get("top3", [])[1:3], 2):
            sc2 = _normalize_score_text(t.get("score","") if isinstance(t, dict) else t)
            if _parse_score(sc2)[0] is not None:
                ai_votes[sc2] = ai_votes.get(sc2,0) + w * (0.22 if rank == 2 else 0.10)
    unified = build_unified_score_matrix(match_obj, engine_result or {}, crs_analysis or {}, trap_report or {}, exp_goals, 8)
    matrix = unified["matrix"]
    posterior = unified["direction_probs"]
    ai_total = sum(ai_dirs.values())
    flags = trap_report.get("flags", {}) if trap_report else {}
    scale = 0.10 if flags else 0.16
    if ai_total > 0:
        ai_share = {k: ai_dirs[k]/ai_total for k in VALID_DIRS}
        logits = {k: math.log(max(0.01, posterior.get(k,33.3)/100)) for k in VALID_DIRS}
        for d in VALID_DIRS:
            delta = math.log(max(0.05, ai_share.get(d,0)) / (1/3))
            logits[d] += max(-0.12, min(0.12, delta * scale))
        post_prob = _softmax_dict(logits, 1.05)
        posterior = {k: round(v*100,2) for k,v in post_prob.items()}
        cur = matrix_direction(matrix)
        for sc in list(matrix.keys()):
            d = _score_direction(sc)
            if d in VALID_DIRS and cur.get(d,0)>0:
                ratio = (posterior[d]/100) / (cur[d]/100)
                matrix[sc] *= max(0.76, min(1.32, ratio))
        matrix = _normalize_prob_dict(matrix)
        posterior = matrix_direction(matrix)
    final_dir = max(posterior, key=posterior.get)
    sorted_dirs = sorted(posterior.items(), key=lambda x: x[1], reverse=True)
    if flags.get("draw_guard") and posterior.get("draw",0) >= sorted_dirs[0][1] - 6:
        final_dir = "draw"
    sorted_p = sorted(posterior.values(), reverse=True)
    dir_conf = round(sorted_p[0],1)
    dir_gap = round(sorted_p[0]-sorted_p[1],1) if len(sorted_p)>=2 else 0
    gmin,gmax,scenario = determine_goal_range(final_dir, crs_analysis.get("moments", {}), exp_goals, trap_report, match_obj, engine_result)
    best, top_candidates, cal_prob = select_score_from_matrix(matrix, final_dir, (gmin,gmax), ai_votes)
    lock_dir = _score_direction(best) or final_dir
    if lock_dir != final_dir:
        for sc, p in sorted(matrix.items(), key=lambda x:x[1], reverse=True):
            total = _score_total(sc)
            if _score_direction(sc) == final_dir and total is not None and gmin <= total <= gmax:
                best, cal_prob, lock_dir = sc, p*100, final_dir
                break
    label = _score_display_label(best, lock_dir)
    evid = [
        f"公平1X2({unified['fair_1x2'].get('method','power')}):{unified['fair_1x2'].get('fair_probs',{})}",
        f"混合方向目标:{unified.get('mixed_target_dir',{})}",
        f"后验方向:{posterior}",
        f"矩阵λ主{unified.get('lambda_h')}/λ客{unified.get('lambda_a')} top={best} 校准概率{cal_prob:.3f}%",
    ]
    if trap_report.get("trap_count", 0):
        evid.append(f"陷阱/保护:{trap_report.get('trap_count')}个 严重度{trap_report.get('total_severity')}")
    if ai_total > 0:
        evid.append(f"AI residual:{_round_dict(ai_dirs,2)}")
    return {"predicted_score": best, "predicted_label": label, "result": _direction_cn(lock_dir), "display_direction": _direction_cn(lock_dir), "final_direction": lock_dir, "is_score_others": label in ("胜其他","平其他","负其他"), "home_win_pct": posterior["home"], "draw_pct": posterior["draw"], "away_win_pct": posterior["away"], "scenario": scenario, "goal_range": (gmin,gmax), "dir_confidence": dir_conf, "dir_gap": dir_gap, "evidences": evid, "override_triggered": False, "top_score_candidates": top_candidates, "calibrated_score_prob_pct": round(cal_prob,3), "bayesian_prior": trap_report.get("fair_1x2", trap_report.get("shin", {})), "unified_matrix_top_scores": unified.get("top_scores", []), "unified_goal_probs": unified.get("goal_probs", {}), "unified_source": unified.get("source"), "fair_1x2_pack": unified.get("fair_1x2", {}), "mixed_target_dir": unified.get("mixed_target_dir", {})}

def _scenario_from_score_total(total, old_scenario=""):
    if total is None: return old_scenario or "normal"
    if total <= 1: return "grinder"
    if total == 2: return "low_goals"
    if total in (3,4): return "normal"
    if total == 5: return "high_goals"
    return "shootout"

def _enforce_consistency(mg):
    score = _normalize_score_text(mg.get("predicted_score", "1-1"))
    code = _score_direction(score) or _dir_from_cn(mg.get("result","")) or mg.get("final_direction","draw")
    mg["result"] = _direction_cn(code)
    mg["display_direction"] = _direction_cn(code)
    mg["final_direction"] = code
    mg["predicted_score"] = score
    mg["predicted_label"] = _score_display_label(score, code)
    mg["is_score_others"] = mg["predicted_label"] in ("胜其他","平其他","负其他")
    return mg

def _validate_prediction_consistency(mg):
    warnings = list(mg.get("validation_warnings", []))
    score = _normalize_score_text(mg.get("predicted_score",""))
    h,a = _parse_score(score)
    if h is None:
        mg["validation_warnings"] = warnings
        return _enforce_consistency(mg)
    total = h + a
    gr = mg.get("goal_range")
    if isinstance(gr, list): gr = tuple(gr)
    if isinstance(gr, tuple) and len(gr) == 2:
        gmin,gmax = _i(gr[0]), _i(gr[1])
        if not (gmin <= total <= gmax):
            warnings.append(f"goal_range_adjusted_for_score:{gr}->{total}")
            mg["goal_range"] = (min(gmin,total), max(gmax,total))
    scenario = mg.get("scenario","normal")
    if total <= 2 and scenario in ("extreme_blowout","shootout","high_goals","five_goal_anchor"):
        warnings.append(f"scenario_conflict:{scenario}->score_total_{total}")
        mg["scenario"] = _scenario_from_score_total(total, scenario)
    if total >= 5 and scenario in ("grinder","low_goals"):
        warnings.append(f"scenario_conflict:{scenario}->score_total_{total}")
        mg["scenario"] = _scenario_from_score_total(total, scenario)
    mg["validation_warnings"] = warnings
    return _enforce_consistency(mg)

def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    match_obj = normalize_match(match_obj)
    def valid(r):
        if not isinstance(r, dict):
            return False
        if _parse_score(r.get("ai_score", ""))[0] is not None:
            return True
        top3 = r.get("top3", [])
        if isinstance(top3, list) and top3:
            first = top3[0]
            sc = first.get("score", "") if isinstance(first, dict) else first
            return _parse_score(sc)[0] is not None
        return False
    ai_valid = {"gpt":valid(gpt_r), "grok":valid(grok_r), "gemini":valid(gemini_r), "claude":valid(claude_r)}
    if any(not v for v in ai_valid.values()):
        print("    弃权AI:", ", ".join(k.upper() for k,v in ai_valid.items() if not v))
    ai_responses = {}
    for k,r in [("claude",claude_r),("gpt",gpt_r),("grok",grok_r),("gemini",gemini_r)]:
        if ai_valid[k]:
            ai_responses[k] = r
    exp_goals = 0.0
    for src in [engine_result, stats]:
        if isinstance(src, dict):
            for k in ["expected_total_goals","exp_goals","total_goals","expected_goals","lambda_total","total_xg"]:
                fv = _f(src.get(k), 0)
                if fv > 0.5:
                    exp_goals = fv; break
        if exp_goals > 0: break
    if exp_goals <= 0 and isinstance(engine_result, dict):
        hxg, axg = _f(engine_result.get("bookmaker_implied_home_xg",0)), _f(engine_result.get("bookmaker_implied_away_xg",0))
        if hxg > 0 and axg > 0:
            exp_goals = hxg + axg
    if exp_goals <= 0:
        gp = [(g, 1/_f(match_obj.get(f"a{g}",0))) for g in range(8) if _f(match_obj.get(f"a{g}",0)) > 1]
        if gp:
            s = sum(p for _,p in gp)
            exp_goals = sum(g*(p/s) for g,p in gp)
    if exp_goals < 1.0 or exp_goals > 6.0:
        exp_goals = 2.5
    smart = stats.get("smart_signals", []) if isinstance(stats, dict) else []
    trap = detect_all_traps(match_obj, engine_result or {}, ai_responses, smart, exp_goals)
    if trap["trap_count"]:
        print(f"    陷阱/保护:{trap['trap_count']}个 严重度{trap['total_severity']}")
    crs = analyze_crs_matrix(match_obj)
    if crs["coverage"] > 0:
        print(f"    CRS:覆盖{crs['coverage']*100:.0f}% 形状={crs['shape_verdict']}")
    lock = decision_lock_chain(match_obj, engine_result or {}, trap, crs, ai_responses, smart, exp_goals)
    print(f"    方向: 主{lock['home_win_pct']:.0f}% 平{lock['draw_pct']:.0f}% 客{lock['away_win_pct']:.0f}%")
    pred = lock["predicted_score"]
    odds = get_market_odds_for_score(match_obj, pred)
    model_prob = _f(lock.get("calibrated_score_prob_pct",0))
    if model_prob <= 0: model_prob = 1.0
    implied = round(1/odds*100,3) if odds > 1.05 else None
    ev = calculate_independent_ev(model_prob, odds, implied)
    cf, risk = confidence_rank_score({"home":lock["home_win_pct"],"draw":lock["draw_pct"],"away":lock["away_win_pct"]}, lock.get("top_score_candidates", []), trap, len(ai_responses))
    mg = {
        **lock,
        "home_win_pct": round(lock["home_win_pct"],1), "draw_pct": round(lock["draw_pct"],1), "away_win_pct": round(lock["away_win_pct"],1),
        "confidence": cf, "confidence_meaning": "排序置信度，不等于历史命中率", "risk_level": risk,
        "bayesian_evidences": lock["evidences"], "traps_detected": [t["trap"] for t in trap["traps_detected"]],
        "trap_count": trap["trap_count"], "trap_severity": trap["total_severity"], "trap_details": [{"trap":t["trap"],"desc":t["description"]} for t in trap["traps_detected"]], "trap_flags": trap.get("flags", {}),
        "fair_1x2": trap.get("fair_1x2", {}), "fair_1x2_method": trap.get("fair_1x2_method","power"), "market_overround": trap.get("market_overround",0), "raw_implied_1x2": trap.get("raw_implied_1x2", {}),
        "crs_shape": crs.get("shape_verdict","unknown"), "crs_moments": crs.get("moments", {}), "crs_margin": crs.get("margin",0), "crs_coverage": crs.get("coverage",0), "crs_implied_probs": crs.get("implied_probs", {}), "crs_low_rank_info": crs.get("low_rank_info", {}),
        "suggested_kelly": ev["kelly"], "edge_vs_market": ev["ev"], "is_value": ev["is_value"], "ev_note": ev.get("note",""), "score_model_prob": round(model_prob,3), "score_market_odds": odds, "score_market_implied_pct": implied,
        "gpt_score": gpt_r.get("ai_score","弃权") if ai_valid["gpt"] else "弃权", "gpt_analysis": gpt_r.get("reason","弃权") if ai_valid["gpt"] else "弃权",
        "grok_score": grok_r.get("ai_score","弃权") if ai_valid["grok"] else "弃权", "grok_analysis": grok_r.get("reason","弃权") if ai_valid["grok"] else "弃权",
        "gemini_score": gemini_r.get("ai_score","弃权") if ai_valid["gemini"] else "弃权", "gemini_analysis": gemini_r.get("reason","弃权") if ai_valid["gemini"] else "弃权",
        "claude_score": claude_r.get("ai_score", pred) if ai_valid["claude"] else "弃权", "claude_analysis": claude_r.get("reason","弃权") if ai_valid["claude"] else "弃权",
        "ai_abstained": [k.upper() for k,v in ai_valid.items() if not v], "ai_avg_confidence": round(sum(_f(r.get("ai_confidence",60)) for r in ai_responses.values()) / max(1,len(ai_responses)),1),
        "smart_money_signal": " | ".join(str(x) for x in (list(smart)+[f"{t['trap']}:{t['description'][:80]}" for t in trap["traps_detected"]])[:16]),
        "smart_signals": list(smart), "sharp_detected": trap.get("sharp_detected",False), "sharp_dir": trap.get("sharp_dir"),
        "xG_home": round(_f(engine_result.get("bookmaker_implied_home_xg",1.3)) if isinstance(engine_result,dict) else 1.3,2),
        "xG_away": round(_f(engine_result.get("bookmaker_implied_away_xg",0.9)) if isinstance(engine_result,dict) else 0.9,2),
        "expected_total_goals": round(exp_goals,2), "actual_handicap_signed": _parse_actual_handicap_signed(match_obj), "theoretical_handicap_signed": _infer_theoretical_handicap_signed(_f(match_obj.get("sp_home",match_obj.get("win",0))), _f(match_obj.get("sp_away",match_obj.get("lose",0)))),
        "external_context": match_obj.get("external_context", {}),
        "refined_poisson": stats.get("refined_poisson", {}) if isinstance(stats,dict) else {}, "elo": stats.get("elo", {}) if isinstance(stats,dict) else {}, "experience_analysis": stats.get("experience_analysis", {}) if isinstance(stats,dict) else {},
        "engine_version": ENGINE_VERSION, "engine_architecture": ENGINE_ARCHITECTURE,
    }
    return _validate_prediction_consistency(_enforce_consistency(mg))


# ==============================
# 后处理 / Top4 / 主入口
# ==============================

LOCKED_CORE_FIELDS = {"predicted_score","predicted_label","result","display_direction","final_direction","home_win_pct","draw_pct","away_win_pct","confidence","dir_confidence","dir_gap","scenario","goal_range","top_score_candidates","unified_matrix_top_scores","calibrated_score_prob_pct"}

def _preserve_core_prediction(before, after, stage):
    if not isinstance(before, dict) or not isinstance(after, dict):
        return after
    changed = []
    for k in LOCKED_CORE_FIELDS:
        if k in before and after.get(k) != before.get(k):
            after[k] = before[k]; changed.append(k)
    if changed:
        warns = list(after.get("postprocess_warnings", []))
        warns.append(f"{stage}_core_fields_restored:{','.join(sorted(changed))}")
        after["postprocess_warnings"] = warns
    return after

def _evidence_alignment_score(pr):
    s = 0
    traps = set(pr.get("traps_detected", [])); fd = pr.get("final_direction"); pred = pr.get("predicted_score","")
    if "T1_REAL_DRAW_SIGNAL" in traps and fd == "draw": s += 5
    if "T2_HANDICAP_DEEPER" in traps and fd in ("home","away"): s += 3
    if pr.get("scenario") == "five_goal_anchor" and pred in ("3-2","2-3"): s += 5
    if "D17_BALANCED_DRAW_GUARD" in traps and fd != "draw": s -= 4
    if "D19_LONG_AWAY_GUARD" in traps and fd == "away": s -= 4
    if "T3_HANDICAP_SHALLOWER" in traps:
        fair = pr.get("fair_1x2", {})
        strong = max(["home","away"], key=lambda k: fair.get(k,33.3)) if fair else None
        if fd == strong: s -= 4
    return s

def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})
        s = pr.get("confidence",0)*0.38 + pr.get("dir_confidence",50)*0.13 + _evidence_alignment_score(pr)
        if pr.get("trap_count",0): s += min(3, pr.get("trap_count",0))
        flags = pr.get("trap_flags", {})
        if flags.get("draw_guard") or flags.get("favorite_guard") or flags.get("long_away_guard"): s -= 4
        ev = pr.get("edge_vs_market",0)
        if pr.get("is_value"):
            s += 8 if ev >= 30 else 4 if ev >= 15 else 0
        if pr.get("risk_level") == "高": s -= 10
        elif pr.get("risk_level") == "低": s += 6
        if pr.get("is_score_others"): s += 3
        if pr.get("dir_gap",0) < 8: s -= 5
        if pr.get("validation_warnings"): s -= 4
        p["recommend_score"] = round(s,2)
    preds.sort(key=lambda x:x.get("recommend_score",0), reverse=True)
    return preds[:4]

def extract_num(ms):
    wm = {"一":1000,"二":2000,"三":3000,"四":4000,"五":5000,"六":6000,"日":7000,"天":7000}
    base = next((v for k,v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999

def run_predictions(raw, use_ai=True):
    ms = [normalize_match(m) for m in raw.get("matches", [])]
    print("\n" + "="*88)
    print(f"  [{ENGINE_VERSION}] {ENGINE_ARCHITECTURE} | {len(ms)} 场")
    print("="*88)
    match_analyses = []
    for i, m in enumerate(ms):
        try: eng = predict_match(m)
        except Exception as e: logger.warning(f"predict_match失败:{e}"); eng = {}
        try: league_info, _, _, _ = build_league_intelligence(m)
        except Exception: league_info = {}
        try: sp = ensemble.predict(m, {}) if ensemble else {}
        except Exception as e: logger.warning(f"ensemble失败:{e}"); sp = {}
        anchor_sigs = []
        for g, lim, txt in [(3,4.15,"3球锚点→2-1/1-2/3-0/0-3"),(4,4.85,"4球锚点→2-2/3-1/1-3"),(5,8.2,"5球锚点→3-2/2-3"),(7,18,"7球锚点→5-2/2-5/4-3/3-4")]:
            val = _f(m.get(f"a{g}",0))
            if 0 < val <= lim: anchor_sigs.append(f"{txt}({val:.2f})")
        if anchor_sigs:
            if not isinstance(sp, dict): sp = {}
            existing = sp.get("smart_signals", [])
            if not isinstance(existing, list): existing = [str(existing)]
            sp["smart_signals"] = existing + anchor_sigs
        try: exp_result = exp_engine.analyze(m) if exp_engine else {}
        except Exception: exp_result = {}
        exp_goals = _f(eng.get("expected_total_goals",0)) if isinstance(eng,dict) else 0
        if exp_goals <= 0:
            hxg, axg = (_f(eng.get("bookmaker_implied_home_xg",0)), _f(eng.get("bookmaker_implied_away_xg",0))) if isinstance(eng,dict) else (0,0)
            exp_goals = hxg + axg if hxg and axg else 2.5
        trap = detect_all_traps(m, eng, {}, sp.get("smart_signals", []) if isinstance(sp,dict) else [], exp_goals)
        crs = analyze_crs_matrix(m)
        match_analyses.append({"match": m, "engine": eng, "league_info": league_info, "stats": sp, "index": i+1, "experience": exp_result, "trap_preview": trap, "crs_preview": crs})
    if match_analyses:
        print(f"  [{ENGINE_VERSION} External] 联网情报入口 enabled={_external_context_enabled()} ...")
        try:
            match_analyses = _run_coro_sync(attach_external_contexts(match_analyses))
        except Exception as e:
            logger.warning(f"外部情报加载失败:{e}")
            for ma in match_analyses:
                ma["external_context"] = _extract_builtin_external_context(ma["match"])
        for ma in match_analyses:
            ma["match"]["external_context"] = ma.get("external_context", {})
    all_ai = {"gpt":{},"grok":{},"gemini":{},"claude":{}}
    if use_ai and match_analyses:
        print(f"  [{ENGINE_VERSION} AI] 启动 GPT/Grok/Gemini 初审 + Claude 终审")
        try: all_ai = _run_coro_sync(run_ai_matrix_two_phase(match_analyses))
        except Exception as e: logger.error(f"AI矩阵失败:{e}")
    res = []
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        mg = merge_result(ma["engine"], all_ai["gpt"].get(i+1,{}), all_ai["grok"].get(i+1,{}), all_ai["gemini"].get(i+1,{}), all_ai["claude"].get(i+1,{}), ma["stats"], m)
        for stage, func in [("experience", lambda mm,gg: apply_experience_to_prediction(mm,gg,exp_engine) if exp_engine else gg), ("odds_history", apply_odds_history), ("quant_edge", apply_quant_edge), ("wencai", apply_wencai_intel), ("advanced_models", upgrade_ensemble_predict)]:
            try:
                before = dict(mg); mg2 = func(m, mg); mg = _preserve_core_prediction(before, mg2, stage)
            except Exception as e:
                logger.warning(f"{stage}后处理失败:{e}")
        mg = _validate_prediction_consistency(_enforce_consistency(mg))
        res.append({**m, "prediction": mg})
        tags = (f" [T{mg['trap_count']}]" if mg.get("trap_count",0) else "") + (" [其他比分]" if mg.get("is_score_others") else "") + (f" [{mg.get('scenario','normal')}]")
        print(f"  [{i+1}] {m.get('home_team')} vs {m.get('away_team')} => {mg['result']} ({mg['predicted_score']}={mg['predicted_label']}) | CF:{mg['confidence']} | 方向:{mg['dir_confidence']:.0f}% | P:{mg.get('score_model_prob',0)}%{tags}")
    t4 = select_top4(res)
    t4_keys = {("id", str(t.get("id"))) if t.get("id") not in (None,"") else ("idx", id(t)) for t in t4}
    for r in res:
        rid = r.get("id")
        r["is_recommended"] = (("id", str(rid)) in t4_keys) if rid not in (None,"") else (("idx", id(r)) in t4_keys)
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    return res, t4


def self_check_handicap_parser():
    cases = [
        ({"give_ball":"主让1球","sp_home":1.6,"sp_away":5.0}, -1.0),
        ({"give_ball":"主受让1球","sp_home":5.0,"sp_away":1.6}, +1.0),
        ({"give_ball":"客让1球","sp_home":5.0,"sp_away":1.6}, +1.0),
        ({"give_ball":"客受让1球","sp_home":1.6,"sp_away":5.0}, -1.0),
        ({"give_ball":"-0.5/1","sp_home":1.8,"sp_away":4.2}, -0.75),
        ({"give_ball":"+0.5/1","sp_home":4.2,"sp_away":1.8}, +0.75),
    ]
    ok = True
    for m, exp in cases:
        got = _parse_actual_handicap_signed(m)
        if abs(got - exp) > 1e-6:
            print(f"[SELF_CHECK_FAIL] {m['give_ball']} expected={exp} got={got}")
            ok = False
    if ok: print("[SELF_CHECK_OK] handicap parser")
    return ok

if __name__ == "__main__":
    logger.info(f"{ENGINE_VERSION} 启动")
    print(f"✅ {ENGINE_VERSION} 加载完成")
    print(f"   架构: {ENGINE_ARCHITECTURE}")
    print("   一致性: predicted_score ↔ predicted_label ↔ result ↔ direction ↔ goal_range ↔ scenario ↔ EV")
    self_check_handicap_parser()
