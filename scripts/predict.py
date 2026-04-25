import json
import os
import re
import time
import asyncio
import aiohttp
import logging
import math
import hashlib
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple, Optional


# ====================================================================
# 🚀 vMAX 18.7 CLEAN FULL
# --------------------------------------------------------------------
# 核心原则：
# 1. 不再叠补丁，不再 v18.6 / v18.6.1 / v18.6.2 层层覆盖。
# 2. 只保留一条清晰决策链：
#    原始数据 → 基础模型 → AI矩阵 → T1-T16陷阱 → CRS矩阵 → 贝叶斯方向 → 区间 → 比分。
# 3. final_direction 是主变量，predicted_score 必须投影到 final_direction + goal_range。
# 4. CRS 胜其他/平其他/负其他只作为聚合标签，不拆成假比分污染 λ。
# 5. EV/Kelly 使用模型比分概率，不用市场 implied probability 自证价值。
# ====================================================================

ENGINE_VERSION = "vMAX 18.7-clean-full"
ENGINE_ARCHITECTURE = "贝叶斯后验 + T1-T16陷阱矩阵 + CRS聚合修复 + 方向优先锁分 + EV校准"


try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)


try:
    from config import *
    from models import EnsemblePredictor
    from odds_engine import predict_match
    from league_intel import build_league_intelligence
    from experience_rules import ExperienceEngine, apply_experience_to_prediction
    from advanced_models import upgrade_ensemble_predict
except ImportError as e:
    logger.warning(f"基础核心模块导入异常: {e}")


try:
    from odds_history import apply_odds_history
except ImportError:
    def apply_odds_history(m, mg):
        return mg


try:
    from quant_edge import apply_quant_edge
except ImportError:
    def apply_quant_edge(m, mg):
        return mg


try:
    from wencai_intel import apply_wencai_intel
except ImportError:
    def apply_wencai_intel(m, mg):
        return mg


try:
    ensemble = EnsemblePredictor()
except Exception:
    ensemble = None


try:
    exp_engine = ExperienceEngine()
except Exception:
    exp_engine = None


# ====================================================================
# 常量
# ====================================================================

DIRECTION_CN = {"home": "主胜", "draw": "平局", "away": "客胜"}
CN_DIRECTION = {"主胜": "home", "平局": "draw", "客胜": "away"}

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

CRS_FULL_MAP = {
    "1-0": "w10", "2-0": "w20", "2-1": "w21", "3-0": "w30", "3-1": "w31",
    "3-2": "w32", "4-0": "w40", "4-1": "w41", "4-2": "w42", "5-0": "w50",
    "5-1": "w51", "5-2": "w52",

    "0-0": "s00", "1-1": "s11", "2-2": "s22", "3-3": "s33",

    "0-1": "l01", "0-2": "l02", "1-2": "l12", "0-3": "l03", "1-3": "l13",
    "2-3": "l23", "0-4": "l04", "1-4": "l14", "2-4": "l24", "0-5": "l05",
    "1-5": "l15", "2-5": "l25",
}

OTHER_SCORE_LABELS = {"胜其他", "平其他", "负其他"}

FIELD_ALIASES = {
    "home": ("home_team", "home", "host", "主队"),
    "away": ("away_team", "guest", "away", "客队"),
    "league": ("league", "cup", "competition", "赛事"),
    "sp_home": ("sp_home", "win", "hwin", "odds_home"),
    "sp_draw": ("sp_draw", "same", "draw", "odds_draw"),
    "sp_away": ("sp_away", "lose", "away_win", "odds_away"),
    "handicap": ("give_ball", "handicap", "asian_handicap", "AHh", "让球"),
}

_HANDICAP_WORDS = {
    "平手": 0.0,
    "平/半": 0.25,
    "平半": 0.25,
    "半球": 0.5,
    "半/一": 0.75,
    "半一": 0.75,
    "一球": 1.0,
    "一/球半": 1.25,
    "一球/球半": 1.25,
    "球半": 1.5,
    "球半/两": 1.75,
    "球半/两球": 1.75,
    "两球": 2.0,
    "两/两半": 2.25,
    "两球/两球半": 2.25,
    "两半": 2.5,
    "三球": 3.0,
}

_MATCH_INDEX_CACHE = {}


# ====================================================================
# 通用工具
# ====================================================================

def _f(v, default=0.0) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "" or s.lower() in {"nan", "none", "null", "n/a", "-"}:
            return default
        return float(s)
    except Exception:
        return default


def _i(v, default=0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _stable_json_hash(obj) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        s = str(obj)
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def _first_existing(d: Dict[str, Any], aliases, default=None):
    for k in aliases:
        if k in d and d.get(k) not in (None, "", "N/A", "-", "nan"):
            return d.get(k)
    return default


def _parse_score(s: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    安全比分解析。
    关键：0 是合法比分，不能用 if not h 判断。
    """
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

        if s_str in {"主胜", "客胜", "平局", "home", "draw", "away"}:
            return None, None

        parts = s_str.split("-")
        if len(parts) != 2:
            return None, None

        h = int(parts[0])
        a = int(parts[1])

        if h < 0 or a < 0 or h > 20 or a > 20:
            return None, None

        return h, a
    except Exception:
        return None, None


def _score_direction(score_str: Any) -> Optional[str]:
    raw = str(score_str).strip().replace(" ", "")
    h, a = _parse_score(raw)

    if h is None or a is None:
        return None

    if "胜其他" in raw or raw == "9-0":
        return "home"
    if "平其他" in raw or raw == "9-9":
        return "draw"
    if "负其他" in raw or raw == "0-9":
        return "away"

    if h > a:
        return "home"
    if h < a:
        return "away"
    return "draw"


def _score_total_goals(score_str: Any) -> Optional[int]:
    h, a = _parse_score(score_str)
    if h is None or a is None:
        return None
    if "其他" in str(score_str):
        return 9
    return h + a


def _score_inside_goal_range(score_str: Any, goal_range: Any) -> bool:
    try:
        if not goal_range or len(goal_range) != 2:
            return True

        g_min, g_max = int(goal_range[0]), int(goal_range[1])

        if "其他" in str(score_str):
            return g_min <= 9 <= g_max

        tg = _score_total_goals(score_str)
        if tg is None:
            return False

        return g_min <= tg <= g_max
    except Exception:
        return True


# ====================================================================
# 盘口解析
# ====================================================================

def _odds_strong_side(match_obj: Dict[str, Any]) -> Optional[str]:
    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))

    if sp_h <= 1.01 or sp_a <= 1.01:
        return None

    if sp_h < sp_a:
        return "home"
    if sp_a < sp_h:
        return "away"

    return None


def _parse_actual_handicap(match_obj: Dict) -> float:
    """
    内部约定：主让为正，客让为负。

    例：
    give_ball = "-1"   -> 竞彩主让1   -> +1.0
    give_ball = "+1"   -> 竞彩主受1   -> -1.0
    "主让1"            -> +1.0
    "主受让1"          -> -1.0
    "客让1"            -> -1.0
    "客受让1"          -> +1.0
    "1" 且主队欧赔低   -> +1.0
    "1" 且客队欧赔低   -> -1.0
    """
    raw = match_obj.get("give_ball", match_obj.get("handicap", match_obj.get("AHh", "0")))
    s = str(raw).strip().replace(" ", "")

    if not s or s.lower() in {"nan", "none", "null", "n/a", "-"}:
        return 0.0

    side = None
    if "主" in s:
        side = "home"
    elif "客" in s:
        side = "away"

    is_shou = "受" in s

    val = None

    for word, word_val in sorted(_HANDICAP_WORDS.items(), key=lambda kv: len(kv[0]), reverse=True):
        if word in s:
            val = word_val
            break

    if val is None:
        frac = re.search(r"([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)", s)
        if frac:
            val = (abs(_f(frac.group(1))) + abs(_f(frac.group(2)))) / 2.0
        else:
            num = re.search(r"[+-]?\d+(?:\.\d+)?", s)
            val = abs(_f(num.group(0), 0.0)) if num else 0.0

    val = float(val or 0.0)

    if side == "home":
        return -val if is_shou else val

    if side == "away":
        return val if is_shou else -val

    if re.match(r"^[+-]", s):
        signed_val = _f(s, 0.0)
        return -signed_val

    strong_side = _odds_strong_side(match_obj)

    if strong_side == "home":
        return val
    if strong_side == "away":
        return -val

    return 0.0


def _infer_theoretical_handicap(sp_h: float, sp_a: float) -> float:
    """
    从欧赔粗略反推理论让球。
    返回：主让为正，客让为负。
    """
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


# ====================================================================
# 基本面解析
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
        m = re.search(pat, text, flags=re.I)
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

    gf = 0.0
    ga = 0.0

    m1 = re.search(r"场均进球[^0-9]*(\d+\.?\d*)", text)
    m2 = re.search(r"场均失球[^0-9]*(\d+\.?\d*)", text)

    if m1:
        gf = _f(m1.group(1), 0.0)
    if m2:
        ga = _f(m2.group(1), 0.0)

    return gf, ga


def _fundamental_strength(match_obj: Dict, side: str) -> Dict[str, Any]:
    info_src = match_obj.get("points", {})
    if not isinstance(info_src, dict):
        info_src = {}

    key = "home_strength" if side == "home" else "guest_strength"
    txt = str(info_src.get(key, ""))

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
        "strength_score": round(_clamp(score, -100, 100), 1),
    }


# ====================================================================
# 信号解析
# ====================================================================

def detect_sharp_direction(smart_signals: List) -> Dict[str, Any]:
    detected = False
    sharp_dir = None

    for s in smart_signals or []:
        s_str = str(s)

        if "Sharp" not in s_str and "sharp" not in s_str:
            continue

        detected = True

        if re.search(r"(主胜|主队|走主|→\s*主|流向\s*主|资金\s*主|Sharp主|主赔)", s_str):
            sharp_dir = "home"
            break

        if re.search(r"(客胜|客队|走客|→\s*客|流向\s*客|资金\s*客|Sharp客|客赔)", s_str):
            sharp_dir = "away"
            break

        if re.search(r"(平局|平赔|走平|→\s*平|流向\s*平|资金\s*平|Sharp平|进平局)", s_str):
            sharp_dir = "draw"
            break

    return {"detected": detected, "sharp_dir": sharp_dir}


def detect_steam_direction(smart_signals: List) -> Dict[str, Any]:
    steam_dir = None
    steam_type = None

    for s in smart_signals or []:
        s_str = str(s)

        if "Steam" not in s_str:
            continue

        is_reverse = "反向" in s_str or "未跟" in s_str or "不跟" in s_str

        if re.search(r"(主胜.*Steam|Steam.*主胜|主胜.*降水|主.*Steam)", s_str):
            steam_dir = "home"
            steam_type = "reverse" if is_reverse else "normal"
            break

        if re.search(r"(客胜.*Steam|Steam.*客胜|客胜.*降水|客.*Steam)", s_str):
            steam_dir = "away"
            steam_type = "reverse" if is_reverse else "normal"
            break

        if re.search(r"(平.*Steam|Steam.*平)", s_str):
            steam_dir = "draw"
            steam_type = "reverse" if is_reverse else "normal"
            break

    return {"steam_dir": steam_dir, "steam_type": steam_type}


# ====================================================================
# 索引层
# ====================================================================

def _build_text_blob(match_obj: Dict[str, Any], stats: Optional[Dict[str, Any]] = None) -> str:
    chunks = []

    for root in (match_obj, stats or {}):
        if not isinstance(root, dict):
            continue

        for key in ("home_team", "away_team", "home", "guest", "league", "cup"):
            if root.get(key):
                chunks.append(str(root.get(key)))

        for key in ("points", "information", "vote", "change", "smart_signals"):
            v = root.get(key)

            if isinstance(v, dict):
                chunks.extend(str(x) for x in v.values() if x)
            elif isinstance(v, list):
                chunks.extend(str(x) for x in v if x)
            elif v:
                chunks.append(str(v))

    return " | ".join(chunks)


def build_match_index(match_obj: Dict[str, Any], stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    fp = _stable_json_hash({
        "m": match_obj,
        "smart": (stats or {}).get("smart_signals", []),
        "v": ENGINE_VERSION,
    })

    if fp in _MATCH_INDEX_CACHE:
        return _MATCH_INDEX_CACHE[fp]

    field = {name: _first_existing(match_obj, aliases) for name, aliases in FIELD_ALIASES.items()}

    crs_scores = []
    by_direction = defaultdict(list)
    by_total_goals = defaultdict(list)
    by_field_key = {}

    for score, key in CRS_FULL_MAP.items():
        odds = _f(match_obj.get(key, 0))
        if odds <= 1.1:
            continue

        h, a = _parse_score(score)
        if h is None:
            continue

        direction = "home" if h > a else ("away" if h < a else "draw")
        total_goals = h + a

        node = {
            "score": score,
            "key": key,
            "odds": odds,
            "direction": direction,
            "total_goals": total_goals,
            "raw_implied": 1.0 / odds,
        }

        crs_scores.append(node)
        by_direction[direction].append(node)
        by_total_goals[total_goals].append(node)
        by_field_key[key] = node

    others = {}
    for key, direction, label in [
        ("crs_win", "home", "胜其他"),
        ("crs_same", "draw", "平其他"),
        ("crs_lose", "away", "负其他"),
    ]:
        odds = _f(match_obj.get(key, 0))
        if odds > 1.1:
            others[label] = {
                "score": label,
                "key": key,
                "odds": odds,
                "direction": direction,
                "raw_implied": 1.0 / odds,
            }

    ttg = []
    compressed_goals = []

    for g in range(8):
        odds = _f(match_obj.get(f"a{g}", 0))
        if odds <= 1.1:
            continue

        std = STANDARD_GOAL_ODDS.get(g, 50.0)
        compression = std / odds if odds else 0.0

        item = {
            "goals": g,
            "key": f"a{g}",
            "odds": odds,
            "raw_implied": 1.0 / odds,
            "compression": compression,
        }

        ttg.append(item)

        if compression >= 1.5:
            compressed_goals.append(item)

    text_blob = _build_text_blob(match_obj, stats)
    tokens = re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", text_blob.lower())

    idx = {
        "fingerprint": fp,
        "field": field,
        "crs_scores": crs_scores,
        "crs_by_direction": dict(by_direction),
        "crs_by_total_goals": dict(by_total_goals),
        "crs_by_field_key": by_field_key,
        "crs_others": others,
        "ttg": ttg,
        "compressed_goals": compressed_goals,
        "text_blob": text_blob,
        "token_counter": Counter(tokens),
        "team_names": {
            "home": str(field.get("home") or ""),
            "away": str(field.get("away") or ""),
            "league": str(field.get("league") or ""),
        },
    }

    _MATCH_INDEX_CACHE[fp] = idx
    return idx


# ====================================================================
# T1-T16 陷阱矩阵
# ====================================================================

def detect_T1_draw_trap(match_obj: Dict, engine_result: Dict, smart_signals: List, shin: Dict) -> Optional[Dict]:
    change = match_obj.get("change", {}) or {}
    cw = _f(change.get("win", 0))
    cs = _f(change.get("same", 0))
    cl = _f(change.get("lose", 0))

    if cs >= -0.04:
        return None

    if cs > cw or cs > cl:
        return None

    shin_h = shin["home"]
    shin_a = shin["away"]
    strong_shin = max(shin_h, shin_a)

    if strong_shin < 34:
        return None

    strong_side = "home" if shin_h > shin_a else "away"
    weak_side = "away" if strong_side == "home" else "home"
    strong_cn = "主" if strong_side == "home" else "客"

    evidence_score = 2
    detail = [f"平赔独降{cs:.2f}", f"{strong_cn}Shin{strong_shin:.1f}%"]

    if strong_shin >= 42:
        evidence_score += 2
    elif strong_shin >= 38:
        evidence_score += 1

    strong_fund = _fundamental_strength(match_obj, strong_side)
    weak_fund = _fundamental_strength(match_obj, weak_side)

    if strong_fund["total"] >= 3:
        if strong_fund["win_rate"] >= 0.55 or strong_fund["strength_score"] >= 15:
            evidence_score += 2
            detail.append(f"{strong_cn}基本面强")
        elif strong_fund["win_rate"] >= 0.45:
            evidence_score += 1
        elif strong_fund["win_rate"] < 0.30 and strong_fund["strength_score"] < -15:
            evidence_score -= 1

    if weak_fund["total"] >= 3:
        if weak_fund["win_rate"] <= 0.30 or weak_fund["strength_score"] <= -15:
            evidence_score += 1
            detail.append("弱方基本面差")

    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))

    if hxg > 0 and axg > 0:
        xg_diff = hxg - axg
        signed_xg = xg_diff if strong_side == "home" else -xg_diff

        if signed_xg > 0.15:
            evidence_score += 1
            detail.append(f"xG同向{signed_xg:+.2f}")
        elif signed_xg < -0.30:
            evidence_score -= 2

    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))

    if sp_h > 1 and sp_a > 1:
        theoretical = _infer_theoretical_handicap(sp_h, sp_a)
        actual = _parse_actual_handicap(match_obj)
        hc_diff = actual - theoretical

        if (strong_side == "home" and hc_diff >= 0.3) or (strong_side == "away" and hc_diff <= -0.3):
            evidence_score += 1
            detail.append(f"让球偏深{hc_diff:+.2f}")

    if evidence_score < 4:
        return None

    severity = 3 if evidence_score < 6 else 4

    return {
        "trap": "T1_DRAW_TRAP",
        "description": f"诱平赔陷阱(得分{evidence_score}):" + " + ".join(detail),
        "severity": severity,
        "direction_adjust": {
            strong_side: +2.3,
            "draw": -2.6,
            weak_side: -1.0,
        },
        "score_multipliers": {
            "0-0": 0.35,
            "1-1": 0.35,
            "2-2": 0.40,
        },
        "boost_scores": ["2-1", "1-2", "2-0", "0-2"],
        "suppress_draw_sharp": True,
    }


def detect_T2_T3_handicap_trap(match_obj: Dict, shin: Dict) -> Optional[Dict]:
    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))

    if sp_h < 1.05 or sp_a < 1.05:
        return None

    theoretical = _infer_theoretical_handicap(sp_h, sp_a)
    actual = _parse_actual_handicap(match_obj)
    diff = actual - theoretical

    if abs(diff) < 0.5:
        return None

    fund_h = _fundamental_strength(match_obj, "home")
    fund_a = _fundamental_strength(match_obj, "away")

    if fund_h["total"] >= 3 and fund_a["total"] >= 3:
        fund_diff = fund_h["strength_score"] - fund_a["strength_score"]
        odds_strong = "home" if sp_h < sp_a else "away"

        if odds_strong == "home" and fund_diff >= 25:
            return None

        if odds_strong == "away" and fund_diff <= -25:
            return None

    if diff >= 0.5:
        severity = 2 if abs(diff) < 1.0 else 3

        return {
            "trap": "T2_HANDICAP_DEEPER",
            "description": f"让球偏深:理论{theoretical:.2f} vs 实际{actual:.2f},差{diff:+.2f}球(主队真实偏强)",
            "severity": severity,
            "direction_adjust": {
                "home": +1.1 * min(2.0, abs(diff)),
                "draw": -0.3,
                "away": -0.5,
            },
            "score_multipliers": {},
            "boost_scores": ["1-0", "2-0", "2-1", "3-1"],
        }

    severity = 2 if abs(diff) < 1.0 else 3

    return {
        "trap": "T3_HANDICAP_SHALLOWER",
        "description": f"让球偏浅:理论{theoretical:.2f} vs 实际{actual:.2f},差{diff:+.2f}球(主队真实偏弱)",
        "severity": severity,
        "direction_adjust": {
            "home": -1.0 * min(2.0, abs(diff)),
            "draw": +0.4,
            "away": +1.1 * min(2.0, abs(diff)),
        },
        "score_multipliers": {},
        "boost_scores": ["0-1", "1-1", "0-2", "1-2"],
    }


def detect_T4_T5_fake_favorite(match_obj: Dict, engine_result: Dict, shin: Dict) -> Optional[Dict]:
    shin_h = shin["home"]
    shin_a = shin["away"]

    fund_h = _fundamental_strength(match_obj, "home")
    fund_a = _fundamental_strength(match_obj, "away")

    if fund_h["total"] < 3 or fund_a["total"] < 3:
        return None

    if shin_h > 48 and fund_h["strength_score"] < -5 and fund_a["strength_score"] > 15:
        return {
            "trap": "T4_FAKE_HOME_FAVORITE",
            "description": f"诱主胜:Shin主{shin_h:.1f}%但主基本面{fund_h['strength_score']} vs 客{fund_a['strength_score']}",
            "severity": 3,
            "direction_adjust": {"home": -2.2, "draw": +0.5, "away": +1.8},
            "score_multipliers": {"1-0": 0.45, "2-0": 0.35, "2-1": 0.55},
            "boost_scores": ["0-1", "1-1", "1-2"],
        }

    if shin_a > 48 and fund_a["strength_score"] < -5 and fund_h["strength_score"] > 15:
        return {
            "trap": "T5_FAKE_AWAY_FAVORITE",
            "description": f"诱客胜:Shin客{shin_a:.1f}%但客基本面{fund_a['strength_score']} vs 主{fund_h['strength_score']}",
            "severity": 3,
            "direction_adjust": {"away": -2.2, "draw": +0.5, "home": +1.8},
            "score_multipliers": {"0-1": 0.45, "0-2": 0.35, "1-2": 0.55},
            "boost_scores": ["1-0", "1-1", "2-1"],
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

    if low_small >= 2 and exp_goals >= 2.85:
        return {
            "trap": "T6_SMALL_SCORE_TRAP",
            "description": f"诱小比分:a0/a1/a2压低{low_small}项，但λ={exp_goals:.2f}偏高",
            "severity": 2,
            "direction_adjust": {},
            "score_multipliers": {"0-0": 0.35, "1-0": 0.55, "0-1": 0.55, "1-1": 0.65},
            "boost_scores": ["2-1", "1-2", "3-1", "1-3", "2-2"],
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

    if low_large >= 2 and exp_goals <= 2.30:
        return {
            "trap": "T7_LARGE_SCORE_TRAP",
            "description": f"诱大比分:a5/a6/a7压低{low_large}项，但λ={exp_goals:.2f}偏低",
            "severity": 2,
            "direction_adjust": {},
            "score_multipliers": {"3-2": 0.45, "2-3": 0.45, "3-3": 0.40, "4-2": 0.30, "2-4": 0.30},
            "boost_scores": ["0-0", "1-0", "0-1", "1-1", "2-1", "1-2"],
        }

    return None


def detect_T8_false_cold(match_obj: Dict, smart_signals: List, shin: Dict) -> Optional[Dict]:
    sigs = " ".join(str(s) for s in smart_signals or [])

    cold_triggers = 0
    for kw in ["坏消息", "崩盘", "造热", "背离", "盘口太便宜"]:
        if kw in sigs:
            cold_triggers += 1

    if cold_triggers < 2:
        return None

    vote = match_obj.get("vote", {}) or {}
    vh = _i(vote.get("win", 33), 33)
    va = _i(vote.get("lose", 33), 33)

    hot_dir = None
    hot_pct = 0

    if vh >= 58:
        hot_dir = "home"
        hot_pct = vh
    elif va >= 58:
        hot_dir = "away"
        hot_pct = va

    if not hot_dir:
        return None

    fund = _fundamental_strength(match_obj, hot_dir)

    if fund["total"] >= 3 and fund["strength_score"] > 20 and fund["win_rate"] > 0.55:
        other = "away" if hot_dir == "home" else "home"

        return {
            "trap": "T8_FALSE_COLD",
            "description": f"假冷门:{hot_dir}散户热{hot_pct}%但基本面真强({fund['strength_score']}分)",
            "severity": 2,
            "direction_adjust": {hot_dir: +1.6, other: -1.1},
            "score_multipliers": {},
            "boost_scores": ["1-0", "2-0", "2-1"] if hot_dir == "home" else ["0-1", "0-2", "1-2"],
            "suppress_contrarian": True,
        }

    return None


def detect_T9_fake_contrarian(match_obj: Dict, shin: Dict, smart_signals: List) -> Optional[Dict]:
    vote = match_obj.get("vote", {}) or {}
    vh = _i(vote.get("win", 33), 33)
    va = _i(vote.get("lose", 33), 33)

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

    if not follow:
        return None

    return {
        "trap": "T9_FAKE_CONTRARIAN",
        "description": f"诱反指:{hot_dir}散户{hot_pct}%且赔率同向降水，反指降权",
        "severity": 2,
        "direction_adjust": {hot_dir: +1.2},
        "score_multipliers": {},
        "boost_scores": ["1-0", "2-1"] if hot_dir == "home" else ["0-1", "1-2"],
        "suppress_contrarian": True,
    }


def detect_T10_silent_market(match_obj: Dict) -> Optional[Dict]:
    change = match_obj.get("change", {}) or {}
    total_move = abs(_f(change.get("win", 0))) + abs(_f(change.get("same", 0))) + abs(_f(change.get("lose", 0)))

    idx = build_match_index(match_obj)
    crs_count = len(idx.get("crs_scores", []))

    if total_move < 0.03 and crs_count < 8:
        return {
            "trap": "T10_SILENT_MARKET",
            "description": f"沉默盘:赔率变动{total_move:.3f}+CRS覆盖{crs_count}项，市场定价薄弱",
            "severity": 1,
            "direction_adjust": {},
            "score_multipliers": {},
            "boost_scores": [],
            "confidence_penalty": 10,
        }

    return None


def detect_T11_xg_divergence(match_obj: Dict, engine_result: Dict) -> Optional[Dict]:
    hxg_book = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg_book = _f(engine_result.get("bookmaker_implied_away_xg", 0))

    if hxg_book <= 0 or axg_book <= 0:
        return None

    info = match_obj.get("points", {}) or {}
    h_txt = str(info.get("home_strength", ""))
    a_txt = str(info.get("guest_strength", ""))

    h_for, _ = _extract_avg_goals(h_txt)
    a_for, _ = _extract_avg_goals(a_txt)

    divergences = []

    if h_for > 0 and abs(hxg_book - h_for) > 0.8:
        divergences.append(f"主xG书{hxg_book:.2f}vs场均{h_for:.2f}")

    if a_for > 0 and abs(axg_book - a_for) > 0.8:
        divergences.append(f"客xG书{axg_book:.2f}vs场均{a_for:.2f}")

    if len(divergences) < 2:
        return None

    return {
        "trap": "T11_XG_DIVERGENCE",
        "description": "xG背离:" + "; ".join(divergences),
        "severity": 1,
        "direction_adjust": {},
        "score_multipliers": {},
        "boost_scores": [],
        "xg_override": {
            "home_xg": h_for if h_for > 0 else hxg_book,
            "away_xg": a_for if a_for > 0 else axg_book,
        },
    }


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
        "description": f"让球未开但理论让{theoretical:.2f}球，盘口隐藏真实预期",
        "severity": 1,
        "direction_adjust": {},
        "score_multipliers": {},
        "boost_scores": [],
        "confidence_penalty": 8,
    }


def detect_T13_goalless_draw(match_obj: Dict, engine_result: Dict, shin: Dict, exp_goals: float) -> Optional[Dict]:
    shin_h = shin["home"]
    shin_a = shin["away"]

    if abs(shin_h - shin_a) > 12:
        return None

    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))

    if hxg <= 0 or axg <= 0:
        return None

    total_xg = hxg + axg

    if total_xg >= 2.35:
        return None

    info = match_obj.get("points", {}) or {}
    h_txt = str(info.get("home_strength", ""))
    a_txt = str(info.get("guest_strength", ""))

    h_for, h_against = _extract_avg_goals(h_txt)
    a_for, a_against = _extract_avg_goals(a_txt)

    weak_attack = 0
    strong_def = 0

    if 0 < h_for < 1.4:
        weak_attack += 1
    if 0 < a_for < 1.4:
        weak_attack += 1
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
        _i(vote.get("win", 33), 33),
        _i(vote.get("same", 33), 33),
        _i(vote.get("lose", 33), 33),
    )

    if max_vote >= 58:
        return None

    severity = 3 if total_xg < 2.0 and small_compressed >= 2 else 2

    return {
        "trap": "T13_GOALLESS_DRAW",
        "description": f"闷平场景:xG总{total_xg:.2f}+弱攻{weak_attack}/强防{strong_def}+小球压低{small_compressed}项",
        "severity": severity,
        "direction_adjust": {"draw": +1.5, "home": -0.5, "away": -0.5},
        "score_multipliers": {
            "2-1": 0.60,
            "1-2": 0.60,
            "2-2": 0.55,
            "3-1": 0.35,
            "1-3": 0.35,
            "3-2": 0.25,
            "2-3": 0.25,
        },
        "boost_scores": ["0-0", "1-1", "1-0", "0-1"],
    }


def detect_T14_cup_favorite_trap(match_obj: Dict, shin: Dict) -> Optional[Dict]:
    league = str(match_obj.get("league", match_obj.get("cup", "")))
    cup_keywords = ["杯", "淘汰", "决赛", "半决赛", "四分之一", "欧冠", "欧联", "国王杯", "足总杯", "联赛杯"]

    if not any(kw in league for kw in cup_keywords):
        return None

    shin_h = shin["home"]
    shin_a = shin["away"]
    strong_shin = max(shin_h, shin_a)

    if strong_shin < 58:
        return None

    strong_side = "home" if shin_h > shin_a else "away"
    weak_side = "away" if strong_side == "home" else "home"

    vote = match_obj.get("vote", {}) or {}
    vh = _i(vote.get("win", 33), 33)
    va = _i(vote.get("lose", 33), 33)

    strong_vote = vh if strong_side == "home" else va

    if strong_vote < 55:
        return None

    weak_fund = _fundamental_strength(match_obj, weak_side)

    if weak_fund["total"] >= 3:
        reasonable_weak = (
            weak_fund["win_rate"] >= 0.35
            or weak_fund["goals_for"] >= 1.2
            or weak_fund["strength_score"] > -10
        )

        if not reasonable_weak:
            return None

    strong_cn = "主" if strong_side == "home" else "客"

    return {
        "trap": "T14_CUP_FAVORITE",
        "description": f"杯赛大热必死:{strong_cn}Shin{strong_shin:.1f}%+散户{strong_vote}%，弱队存在反扑空间",
        "severity": 3,
        "direction_adjust": {strong_side: -0.8, "draw": +2.2, weak_side: +0.2},
        "score_multipliers": {
            "3-0": 0.45,
            "0-3": 0.45,
            "4-0": 0.30,
            "0-4": 0.30,
            "3-1": 0.55,
            "1-3": 0.55,
        },
        "boost_scores": ["0-0", "1-1", "1-0", "0-1", "2-1", "1-2"],
    }


def detect_T15_historical_deadlock(match_obj: Dict, shin: Dict) -> Optional[Dict]:
    if abs(shin["home"] - shin["away"]) > 18:
        return None

    info = match_obj.get("points", {}) or {}
    text = " ".join(str(v) for v in info.values() if v)

    patterns = [
        r"对阵[^0-9]{0,20}(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
        r"历史交锋[^0-9]*(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
        r"近\s*\d+\s*次[^0-9]*(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
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

    total_h2h = best_w + best_d + best_l

    if total_h2h < 3:
        return None

    draw_rate = best_d / total_h2h if total_h2h > 0 else 0

    if draw_rate < 0.40 and best_d < 3:
        return None

    s11 = _f(match_obj.get("s11", 999), 999)

    if 0 < s11 > 9.0:
        return None

    severity = 2 if draw_rate >= 0.50 else 1

    return {
        "trap": "T15_HISTORICAL_DEADLOCK",
        "description": f"历史僵局:交锋{best_w}胜{best_d}平{best_l}负，平率{draw_rate:.0%}",
        "severity": severity,
        "direction_adjust": {"draw": +1.1, "home": -0.35, "away": -0.35},
        "score_multipliers": {
            "3-0": 0.50,
            "0-3": 0.50,
            "3-1": 0.60,
            "1-3": 0.60,
        },
        "boost_scores": ["0-0", "1-1", "1-0", "0-1"],
    }


def detect_T16_sharp_badnews_conflict(match_obj: Dict, smart_signals: List, shin: Dict) -> Optional[Dict]:
    sigs = " ".join(str(s) for s in smart_signals or [])
    sharp_info = detect_sharp_direction(smart_signals)

    if not sharp_info["detected"]:
        return None

    sharp_dir = sharp_info["sharp_dir"]

    if sharp_dir not in {"home", "away"}:
        return None

    has_home_bad = "主队坏消息" in sigs or "主坏消息" in sigs or "主队利空" in sigs
    has_away_bad = "客队坏消息" in sigs or "客坏消息" in sigs or "客队利空" in sigs

    has_bad = (sharp_dir == "home" and has_home_bad) or (sharp_dir == "away" and has_away_bad)

    if not has_bad:
        return None

    sharp_shin = shin.get(sharp_dir, 33.3)

    if sharp_shin >= 55:
        return None

    other = "away" if sharp_dir == "home" else "home"

    return {
        "trap": "T16_SHARP_BADNEWS_CONFLICT",
        "description": f"Sharp({sharp_dir})+该方坏消息爆炸，资金方向需要降权",
        "severity": 2,
        "direction_adjust": {sharp_dir: -0.8, "draw": +1.3, other: +0.3},
        "score_multipliers": {},
        "boost_scores": ["0-0", "1-1"],
        "downgrade_sharp_trust": 0.30,
    }


def detect_all_traps(
    match_obj: Dict,
    engine_result: Dict,
    ai_responses: Dict,
    smart_signals: List,
    exp_goals: float,
) -> Dict[str, Any]:
    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_d = _f(match_obj.get("sp_draw", match_obj.get("same", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))

    if sp_h > 1 and sp_d > 1 and sp_a > 1:
        margin = 1 / sp_h + 1 / sp_d + 1 / sp_a
        shin = {
            "home": (1 / sp_h) / margin * 100,
            "draw": (1 / sp_d) / margin * 100,
            "away": (1 / sp_a) / margin * 100,
        }
    else:
        shin = {"home": 33.3, "draw": 33.3, "away": 33.3}

    detectors = [
        lambda: detect_T1_draw_trap(match_obj, engine_result, smart_signals, shin),
        lambda: detect_T2_T3_handicap_trap(match_obj, shin),
        lambda: detect_T4_T5_fake_favorite(match_obj, engine_result, shin),
        lambda: detect_T6_T7_score_range_trap(match_obj, engine_result, exp_goals),
        lambda: detect_T8_false_cold(match_obj, smart_signals, shin),
        lambda: detect_T9_fake_contrarian(match_obj, shin, smart_signals),
        lambda: detect_T10_silent_market(match_obj),
        lambda: detect_T11_xg_divergence(match_obj, engine_result),
        lambda: detect_T12_missing_handicap(match_obj),
        lambda: detect_T13_goalless_draw(match_obj, engine_result, shin, exp_goals),
        lambda: detect_T14_cup_favorite_trap(match_obj, shin),
        lambda: detect_T15_historical_deadlock(match_obj, shin),
        lambda: detect_T16_sharp_badnews_conflict(match_obj, smart_signals, shin),
    ]

    traps = []

    for detector in detectors:
        try:
            t = detector()
            if t:
                traps.append(t)
        except Exception as e:
            logger.debug(f"陷阱检测异常: {e}")

    # 互斥规则
    has_t14 = any(t.get("trap") == "T14_CUP_FAVORITE" for t in traps)
    if has_t14:
        traps = [t for t in traps if t.get("trap") != "T1_DRAW_TRAP"]

    has_t13 = any(t.get("trap") == "T13_GOALLESS_DRAW" for t in traps)
    if has_t13:
        traps = [t for t in traps if t.get("trap") != "T6_SMALL_SCORE_TRAP"]

    t4_or_t5 = next((t for t in traps if t.get("trap") in {"T4_FAKE_HOME_FAVORITE", "T5_FAKE_AWAY_FAVORITE"}), None)
    t8 = next((t for t in traps if t.get("trap") == "T8_FALSE_COLD"), None)

    if t4_or_t5 and t8:
        if t4_or_t5.get("severity", 0) >= t8.get("severity", 0):
            traps = [t for t in traps if t.get("trap") != "T8_FALSE_COLD"]
        else:
            traps = [t for t in traps if t.get("trap") not in {"T4_FAKE_HOME_FAVORITE", "T5_FAKE_AWAY_FAVORITE"}]

    direction_adjust = {"home": 0.0, "draw": 0.0, "away": 0.0}
    score_multipliers = {}
    boost_scores = []
    suppress_contrarian = False
    xg_override = None
    confidence_penalty = 0
    total_severity = 0
    sharp_trust_override = 1.0

    for t in traps:
        total_severity += int(_f(t.get("severity", 1), 1))

        for k, v in t.get("direction_adjust", {}).items():
            if k in direction_adjust:
                direction_adjust[k] += _f(v, 0.0)

        for sc, mult in t.get("score_multipliers", {}).items():
            if sc in score_multipliers:
                score_multipliers[sc] *= _f(mult, 1.0)
            else:
                score_multipliers[sc] = _f(mult, 1.0)

        for sc in t.get("boost_scores", []):
            if sc not in boost_scores:
                boost_scores.append(sc)

        if t.get("suppress_contrarian"):
            suppress_contrarian = True

        if t.get("xg_override"):
            xg_override = t["xg_override"]

        confidence_penalty += int(_f(t.get("confidence_penalty", 0), 0))

        if "downgrade_sharp_trust" in t:
            sharp_trust_override = min(sharp_trust_override, _f(t["downgrade_sharp_trust"], 1.0))

    sharp_info = detect_sharp_direction(smart_signals)
    steam_info = detect_steam_direction(smart_signals)

    return {
        "traps_detected": traps,
        "trap_count": len(traps),
        "total_severity": total_severity,
        "direction_adjust": direction_adjust,
        "score_multipliers": score_multipliers,
        "boost_scores": boost_scores,
        "suppress_contrarian": suppress_contrarian,
        "xg_override": xg_override,
        "confidence_penalty": confidence_penalty,
        "sharp_trust_override": sharp_trust_override,
        "steam_trust_override": sharp_trust_override,
        "shin": shin,
        "sharp_detected": sharp_info["detected"],
        "sharp_dir": sharp_info["sharp_dir"],
        "steam_dir": steam_info["steam_dir"],
        "steam_type": steam_info["steam_type"],
    }


# ====================================================================
# CRS 矩阵
# ====================================================================

def crs_implied_probabilities(match_obj: Dict) -> Tuple[Dict[str, float], float, float]:
    """
    CRS 赔率反推概率。
    注意：胜其他/平其他/负其他只作为聚合标签保留，不拆成 4-3/6-0 这种虚拟比分。
    """
    idx = build_match_index(match_obj)
    raw_nodes = list(idx.get("crs_scores", []))

    if len(raw_nodes) < 8:
        return {}, 0.0, 0.0

    raw_sum = sum(n["raw_implied"] for n in raw_nodes)

    for ex in idx.get("crs_others", {}).values():
        raw_sum += ex["raw_implied"]

    if raw_sum <= 0:
        return {}, 0.0, 0.0

    probs = {}

    for n in raw_nodes:
        probs[n["score"]] = n["raw_implied"] / raw_sum * 100

    for label, ex in idx.get("crs_others", {}).items():
        probs[label] = ex["raw_implied"] / raw_sum * 100

    margin = raw_sum - 1.0
    coverage = len(raw_nodes) / len(CRS_FULL_MAP)

    return probs, round(margin, 3), round(coverage, 2)


def compute_statistical_moments(probs: Dict[str, float]) -> Dict[str, float]:
    """
    只让 CRS_FULL_MAP 里的常规比分参与 λ 统计。
    """
    regular = {}

    for sc, p in probs.items():
        raw = str(sc).strip()

        if raw not in CRS_FULL_MAP:
            continue

        h, a = _parse_score(raw)
        if h is None:
            continue

        regular[(h, a)] = _f(p, 0.0)

    total = sum(regular.values())
    if total <= 0.0001:
        return {}

    normalized = {k: v / total for k, v in regular.items()}

    e_h = sum(h * p for (h, a), p in normalized.items())
    e_a = sum(a * p for (h, a), p in normalized.items())

    var_h = sum((h - e_h) ** 2 * p for (h, a), p in normalized.items())
    var_a = sum((a - e_a) ** 2 * p for (h, a), p in normalized.items())

    std_h = math.sqrt(var_h) if var_h > 0 else 0.01
    std_a = math.sqrt(var_a) if var_a > 0 else 0.01

    cov = sum((h - e_h) * (a - e_a) * p for (h, a), p in normalized.items())
    corr = cov / (std_h * std_a) if std_h * std_a > 0 else 0.0

    skew_h = sum(((h - e_h) / std_h) ** 3 * p for (h, a), p in normalized.items()) if std_h > 0.01 else 0.0
    skew_a = sum(((a - e_a) / std_a) ** 3 * p for (h, a), p in normalized.items()) if std_a > 0.01 else 0.0

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

    lh = _f(moments.get("lambda_h", 1.3), 1.3)
    la = _f(moments.get("lambda_a", 1.2), 1.2)
    lt = _f(moments.get("lambda_total", 2.5), 2.5)
    corr = _f(moments.get("corr", 0.0), 0.0)
    var_h = _f(moments.get("var_h", 1.0), 1.0)
    var_a = _f(moments.get("var_a", 1.0), 1.0)
    skew_h = _f(moments.get("skew_h", 0.0), 0.0)
    skew_a = _f(moments.get("skew_a", 0.0), 0.0)

    anomalies = []
    verdict = "normal"

    if lt >= 3.0 and corr >= 0.15:
        verdict = "shootout"
        anomalies.append(f"互射局:λ总{lt:.2f},相关{corr:.2f}")

    elif lt <= 2.2 and var_h < 1.2 and var_a < 1.2:
        verdict = "grinder"
        anomalies.append(f"磨局:λ总{lt:.2f},方差低")

    elif lh - la >= 1.2:
        verdict = "lopsided_h"
        anomalies.append(f"主队碾压:λ主{lh:.2f} vs 客{la:.2f}")

    elif la - lh >= 1.2:
        verdict = "lopsided_a"
        anomalies.append(f"客队碾压:λ客{la:.2f} vs 主{lh:.2f}")

    elif abs(lh - la) < 0.4:
        verdict = "balanced"
        anomalies.append(f"均势:λ主{lh:.2f} vs 客{la:.2f}")

    if abs(skew_h) > 1.8:
        anomalies.append(f"主偏度异常{skew_h:.2f}")

    if abs(skew_a) > 1.8:
        anomalies.append(f"客偏度异常{skew_a:.2f}")

    if corr < -0.15:
        anomalies.append(f"负相关{corr:.2f}:单边场")

    return verdict, anomalies


def compute_direction_from_crs(probs: Dict[str, float]) -> Dict[str, float]:
    home_p = 0.0
    draw_p = 0.0
    away_p = 0.0

    for sc, p in probs.items():
        sc = str(sc).strip()

        if sc == "胜其他":
            home_p += p
            continue

        if sc == "平其他":
            draw_p += p
            continue

        if sc == "负其他":
            away_p += p
            continue

        d = _score_direction(sc)

        if d == "home":
            home_p += p
        elif d == "draw":
            draw_p += p
        elif d == "away":
            away_p += p

    total = home_p + draw_p + away_p

    if total <= 0:
        return {"home": 33.3, "draw": 33.3, "away": 33.3}

    return {
        "home": round(home_p / total * 100, 2),
        "draw": round(draw_p / total * 100, 2),
        "away": round(away_p / total * 100, 2),
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
            "direction_probs": {"home": 33.3, "draw": 33.3, "away": 33.3},
            "top_scores": [],
        }

    moments = compute_statistical_moments(probs)
    verdict, anomalies = classify_shape(moments)
    direction_probs = compute_direction_from_crs(probs)

    sorted_scores = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_scores = [(sc, round(p, 2)) for sc, p in sorted_scores[:10]]

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
# EV / Kelly
# ====================================================================

def calculate_value_bet(prob_pct, odds):
    """
    Kelly + EV。
    prob_pct 必须是模型概率，不是同一市场赔率反推概率。
    """
    odds = _f(odds, 0.0)
    prob_pct = _f(prob_pct, 0.0)

    if odds <= 1.05 or prob_pct <= 0:
        return {"ev": 0.0, "kelly": 0.0, "is_value": False}

    prob = _clamp(prob_pct / 100.0, 0.0001, 0.9999)
    ev = prob * odds - 1.0

    b = odds - 1.0
    q = 1.0 - prob

    if b <= 0:
        return {"ev": round(ev * 100, 2), "kelly": 0.0, "is_value": False}

    kelly = ((b * prob) - q) / b
    half_kelly = max(0.0, min(0.25, kelly * 0.5))

    return {
        "ev": round(ev * 100, 2),
        "kelly": round(half_kelly * 100, 2),
        "is_value": ev > 0.05,
    }


def _estimate_score_model_prob(
    predicted_score: str,
    final_direction: str,
    posterior_pct: Dict[str, float],
    top_candidates: List[Tuple[str, float]],
    crs_probs: Dict[str, float],
) -> float:
    dir_prob = _clamp(posterior_pct.get(final_direction, 33.3) / 100.0, 0.01, 0.99)

    same_dir_candidates = []

    for sc, pts in top_candidates or []:
        if _score_direction(sc) == final_direction:
            same_dir_candidates.append((sc, max(0.01, _f(pts, 0.01))))

    if same_dir_candidates:
        total_pts = sum(p for _, p in same_dir_candidates)
        target_pts = next((p for sc, p in same_dir_candidates if sc == predicted_score), None)

        if target_pts is None:
            target_pts = min(p for _, p in same_dir_candidates) * 0.5

        cond = target_pts / total_pts if total_pts > 0 else 0.10

    else:
        same_dir_market = {
            sc: p for sc, p in crs_probs.items()
            if _score_direction(sc) == final_direction
        }

        if same_dir_market:
            total_market = sum(same_dir_market.values())
            target_market = same_dir_market.get(predicted_score, min(same_dir_market.values()) * 0.5)
            cond = target_market / total_market if total_market > 0 else 0.10
        else:
            cond = 0.10

    raw_prob = dir_prob * cond * 100

    if "其他" in str(predicted_score):
        cap = 8.0
    else:
        tg = _score_total_goals(predicted_score)
        if tg is not None and tg >= 5:
            cap = 10.0
        else:
            cap = 18.0

    return round(_clamp(raw_prob, 0.4, cap), 2)


# ====================================================================
# 贝叶斯方向决策
# ====================================================================

def compute_direction_posterior(
    shin: Dict[str, float],
    trap_report: Dict[str, Any],
    crs_direction: Dict[str, float],
    ai_directions: Dict[str, float],
    engine_result: Dict[str, Any],
    match_obj: Dict[str, Any],
    smart_signals: List[str],
) -> Dict[str, Any]:
    prior = {
        "home": max(0.05, shin.get("home", 33.3) / 100.0),
        "draw": max(0.05, shin.get("draw", 33.3) / 100.0),
        "away": max(0.05, shin.get("away", 33.3) / 100.0),
    }

    s = sum(prior.values())
    prior = {k: v / s for k, v in prior.items()}

    log_odds = {k: math.log(v) for k, v in prior.items()}
    evidences = []

    # CRS 方向证据
    if crs_direction and sum(crs_direction.values()) > 50:
        crs_p = {
            "home": max(0.05, crs_direction.get("home", 33.3) / 100.0),
            "draw": max(0.05, crs_direction.get("draw", 33.3) / 100.0),
            "away": max(0.05, crs_direction.get("away", 33.3) / 100.0),
        }

        for d in log_odds:
            llr = math.log(crs_p[d] / prior[d]) * 1.10
            log_odds[d] += llr

        evidences.append(f"CRS方向:{crs_direction}")

    sharp_trust = trap_report.get("sharp_trust_override", 1.0)
    steam_trust = trap_report.get("steam_trust_override", 1.0)

    suppress_draw_signals = any(t.get("suppress_draw_sharp") for t in trap_report.get("traps_detected", []))

    trap_adj = trap_report.get("direction_adjust", {}) or {}
    max_trap_dir = None
    max_trap_val = 0.0

    for d, v in trap_adj.items():
        if d in {"home", "draw", "away"} and v > max_trap_val:
            max_trap_val = v
            max_trap_dir = d

    sharp_dir = trap_report.get("sharp_dir")

    if max_trap_dir and max_trap_val >= 2.0 and sharp_dir and sharp_dir != max_trap_dir:
        sharp_trust = min(sharp_trust, 0.25)
        steam_trust = min(steam_trust, 0.35)
        evidences.append(f"Sharp({sharp_dir})vs陷阱指向({max_trap_dir})，Sharp降权")

    # Sharp
    if trap_report.get("sharp_detected") and sharp_dir in log_odds:
        if sharp_dir == "draw" and suppress_draw_signals:
            sharp_trust = min(sharp_trust, 0.15)

        effective = 2.0 * sharp_trust
        log_odds[sharp_dir] += effective

        for d in log_odds:
            if d != sharp_dir:
                log_odds[d] -= effective / 2.0

        evidences.append(f"Sharp→{sharp_dir}(权重×{sharp_trust:.2f})")

    # Steam
    steam_dir = trap_report.get("steam_dir")
    steam_type = trap_report.get("steam_type")

    if steam_dir in log_odds:
        if steam_dir == "draw" and suppress_draw_signals:
            steam_trust = min(steam_trust, 0.15)

        base = 1.8 if steam_type == "reverse" else 1.1
        effective = base * steam_trust

        log_odds[steam_dir] += effective

        for d in log_odds:
            if d != steam_dir:
                log_odds[d] -= effective / 2.0

        evidences.append(f"Steam({steam_type or 'normal'})→{steam_dir}(权重×{steam_trust:.2f})")

    # 陷阱方向
    for d, v in trap_adj.items():
        if d in log_odds:
            log_odds[d] += _f(v, 0.0)

    if trap_adj and any(abs(_f(v, 0.0)) > 0.1 for v in trap_adj.values()):
        evidences.append(f"陷阱方向调整:{trap_adj}")

    # AI方向
    total_ai = sum(ai_directions.values())

    if total_ai > 0:
        max_ai = max(ai_directions.values())
        consensus = max_ai / total_ai if total_ai > 0 else 0

        ai_weight = 0.75 if consensus >= 0.70 else (0.45 if consensus >= 0.50 else 0.20)

        if trap_report.get("sharp_detected"):
            ai_weight *= 0.55

        for d in log_odds:
            if ai_directions.get(d, 0) > 0:
                share = ai_directions[d] / total_ai
                log_odds[d] += math.log(max(0.05, share) / (1 / 3)) * ai_weight

        evidences.append(f"AI共识{consensus:.0%},权重{ai_weight:.2f}")

    # xG
    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))

    xg_ov = trap_report.get("xg_override")
    if xg_ov:
        hxg = _f(xg_ov.get("home_xg", hxg), hxg)
        axg = _f(xg_ov.get("away_xg", axg), axg)

    if hxg > 0.3 and axg > 0.3:
        xg_diff = hxg - axg

        if xg_diff > 0.5:
            log_odds["home"] += min(1.2, xg_diff * 0.75)
            log_odds["away"] -= min(0.8, xg_diff * 0.45)
            evidences.append(f"xG主优{xg_diff:+.2f}")

        elif xg_diff < -0.5:
            log_odds["away"] += min(1.2, abs(xg_diff) * 0.75)
            log_odds["home"] -= min(0.8, abs(xg_diff) * 0.45)
            evidences.append(f"xG客优{xg_diff:+.2f}")

    # 赔率变动
    change = match_obj.get("change", {}) or {}
    cw = _f(change.get("win", 0))
    cs = _f(change.get("same", 0))
    cl = _f(change.get("lose", 0))

    if cw < -0.05:
        log_odds["home"] += min(0.8, abs(cw) * 6)
    elif cw > 0.05:
        log_odds["home"] -= min(0.4, cw * 4)

    if cs < -0.05 and trap_adj.get("draw", 0) > -1.0:
        log_odds["draw"] += min(0.6, abs(cs) * 5)

    if cl < -0.05:
        log_odds["away"] += min(0.8, abs(cl) * 6)
    elif cl > 0.05:
        log_odds["away"] -= min(0.4, cl * 4)

    # 散户反指
    if not trap_report.get("suppress_contrarian"):
        vote = match_obj.get("vote", {}) or {}
        vh = _i(vote.get("win", 33), 33)
        va = _i(vote.get("lose", 33), 33)
        max_vote = max(vh, va)

        if max_vote >= 62:
            hot_dir = "home" if vh == max_vote else "away"
            contra_power = min(1.0, (max_vote - 57) / 22)
            log_odds[hot_dir] -= contra_power

            for d in log_odds:
                if d != hot_dir:
                    log_odds[d] += contra_power / 2

            evidences.append(f"散户热{hot_dir}{max_vote}%,反指{contra_power:.2f}")

    # 平局保护
    if hxg > 0 and axg > 0:
        xg_total = hxg + axg
        xg_diff_abs = abs(hxg - axg)
        shin_max = max(shin.values())

        if xg_diff_abs < 0.35 and xg_total < 2.55 and shin_max < 55:
            log_odds["draw"] += 0.65
            evidences.append(f"低差平局保护:xG差{xg_diff_abs:.2f},总{xg_total:.2f}")

    # 杯赛平局轻微加成
    league = str(match_obj.get("league", match_obj.get("cup", "")))
    if any(kw in league for kw in ["杯", "淘汰", "欧冠", "欧联", "Cup"]):
        if shin.get("draw", 0) >= 20:
            log_odds["draw"] += 0.25
            evidences.append("杯赛平局轻加成")

    # Softmax
    temperature = 1.85
    scaled = {k: v / temperature for k, v in log_odds.items()}
    max_log = max(scaled.values())
    exp_vals = {k: math.exp(v - max_log) for k, v in scaled.items()}
    total_exp = sum(exp_vals.values())

    posterior = {k: v / total_exp for k, v in exp_vals.items()}

    posterior = {k: _clamp(v, 0.03, 0.88) for k, v in posterior.items()}
    total_adj = sum(posterior.values())
    posterior = {k: v / total_adj for k, v in posterior.items()}

    final_direction = max(posterior, key=posterior.get)
    sorted_p = sorted(posterior.values(), reverse=True)

    return {
        "posterior": {k: round(v * 100, 2) for k, v in posterior.items()},
        "final_direction": final_direction,
        "dir_confidence": round(posterior[final_direction] * 100, 1),
        "dir_gap": round((sorted_p[0] - sorted_p[1]) * 100, 1),
        "evidences": evidences,
        "prior": {k: round(v * 100, 2) for k, v in prior.items()},
    }


def check_sharp_override(
    shin: Dict[str, float],
    trap_report: Dict[str, Any],
    posterior: Dict[str, float],
    trap_score: int,
) -> Tuple[bool, Optional[str], int]:
    if not trap_report.get("sharp_detected"):
        return False, None, 0

    sharp_dir = trap_report.get("sharp_dir")

    if sharp_dir not in {"home", "draw", "away"}:
        return False, None, 0

    shin_argmax = max(shin, key=shin.get)

    if sharp_dir == shin_argmax:
        return False, None, 0

    if trap_score < 5:
        return False, None, 0

    if posterior.get(sharp_dir, 0) < 25:
        return False, None, 0

    for t in trap_report.get("traps_detected", []):
        if t.get("trap") == "T14_CUP_FAVORITE":
            return False, None, 0

    trap_adj_for_sharp = trap_report.get("direction_adjust", {}).get(sharp_dir, 0)

    if trap_adj_for_sharp < 0.5:
        return False, None, 0

    return True, sharp_dir, trap_score


# ====================================================================
# 进球区间
# ====================================================================

def determine_goal_range(
    direction: str,
    moments: Dict[str, float],
    exp_goals: float,
    trap_report: Dict[str, Any],
    match_obj: Dict[str, Any],
    engine_result: Dict[str, Any],
) -> Tuple[int, int, str]:
    actual_hc = _parse_actual_handicap(match_obj)

    lh = _f(moments.get("lambda_h", 0), 0.0) if moments else 0.0
    la = _f(moments.get("lambda_a", 0), 0.0) if moments else 0.0
    lt = _f(moments.get("lambda_total", exp_goals), exp_goals) if moments else exp_goals

    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))

    if hxg > 0 and axg > 0:
        xg_total = hxg + axg
        xg_diff = hxg - axg
    else:
        xg_total = exp_goals
        xg_diff = lh - la

    shape = classify_shape(moments)[0] if moments else "unknown"

    a5 = _f(match_obj.get("a5", 999), 999)
    a6 = _f(match_obj.get("a6", 999), 999)
    a7 = _f(match_obj.get("a7", 999), 999)

    tail_pressure = 0

    if 0 < a5 <= 8:
        tail_pressure += 2
    elif 0 < a5 <= 12:
        tail_pressure += 1

    if 0 < a6 <= 15:
        tail_pressure += 2
    elif 0 < a6 <= 22:
        tail_pressure += 1

    if 0 < a7 <= 28:
        tail_pressure += 2
    elif 0 < a7 <= 40:
        tail_pressure += 1

    if direction == "home":
        handicap_support = actual_hc >= 1.75
        xg_support = xg_diff >= 1.35
        lambda_support = lh - la >= 1.35
    elif direction == "away":
        handicap_support = actual_hc <= -1.75
        xg_support = xg_diff <= -1.35
        lambda_support = la - lh >= 1.35
    else:
        handicap_support = False
        xg_support = False
        lambda_support = False

    extreme_score = 0

    if handicap_support:
        extreme_score += 2
    if xg_support:
        extreme_score += 2
    if lambda_support:
        extreme_score += 1
    if tail_pressure >= 4:
        extreme_score += 2
    elif tail_pressure >= 2:
        extreme_score += 1
    if max(exp_goals, xg_total, lt) >= 3.3:
        extreme_score += 1
    if shape in {"lopsided_h", "lopsided_a"}:
        extreme_score += 1

    if (
        direction in {"home", "away"}
        and extreme_score >= 5
        and (handicap_support or xg_support)
        and max(exp_goals, xg_total, lt) >= 3.0
    ):
        return 5, 12, "extreme_blowout"

    lt_avg = lt * 0.5 + exp_goals * 0.3 + xg_total * 0.2

    try:
        crs_dir = analyze_crs_matrix(match_obj).get("direction_probs", {})
    except Exception:
        crs_dir = {}

    draw_alive = crs_dir.get("draw", 0) >= 25

    if lt_avg <= 1.75:
        return 0, 2, "grinder"

    if lt_avg <= 2.25:
        if direction == "draw" or draw_alive:
            return 0, 2, "low_draw"
        return 1, 3, "low_goals"

    if lt_avg <= 2.85:
        return 2, 4, "normal"

    if lt_avg <= 3.45:
        return 2, 5, "high_goals"

    return 3, 6, "shootout"


# ====================================================================
# 比分选择
# ====================================================================

def _extract_top_score(top_item: Any, default="1-1") -> str:
    if isinstance(top_item, dict):
        return str(top_item.get("score", default)).replace(" ", "").strip()
    if isinstance(top_item, str):
        return top_item.replace(" ", "").strip()
    return default


def _fallback_score_for_lock(direction: str, goal_range: Any, scenario: str = "normal") -> str:
    pools = {
        "home": {
            "grinder": ["1-0", "2-0"],
            "low_draw": ["1-0", "2-0"],
            "low_goals": ["1-0", "2-0", "2-1"],
            "normal": ["2-1", "2-0", "1-0", "3-1"],
            "high_goals": ["2-1", "3-1", "3-2", "2-0"],
            "shootout": ["3-2", "3-1", "4-2"],
            "extreme_blowout": ["胜其他", "4-1", "5-0"],
        },
        "draw": {
            "grinder": ["0-0", "1-1"],
            "low_draw": ["0-0", "1-1"],
            "low_goals": ["1-1", "0-0"],
            "normal": ["1-1", "2-2"],
            "high_goals": ["2-2", "1-1"],
            "shootout": ["2-2", "3-3"],
            "extreme_blowout": ["平其他", "3-3"],
        },
        "away": {
            "grinder": ["0-1", "0-2"],
            "low_draw": ["0-1", "0-2"],
            "low_goals": ["0-1", "0-2", "1-2"],
            "normal": ["1-2", "0-2", "0-1", "1-3"],
            "high_goals": ["1-2", "1-3", "2-3", "0-2"],
            "shootout": ["2-3", "1-3", "2-4"],
            "extreme_blowout": ["负其他", "1-4", "0-5"],
        },
    }

    pool = pools.get(direction, {}).get(scenario) or pools.get(direction, {}).get("normal") or ["1-1"]

    for sc in pool:
        if _score_inside_goal_range(sc, goal_range):
            return sc

    return {"home": "1-0", "draw": "1-1", "away": "0-1"}.get(direction, "1-1")


def select_score(
    direction: str,
    goal_range: Tuple[int, int],
    scenario: str,
    crs_probs: Dict[str, float],
    ai_votes: Dict[str, float],
    trap_report: Dict[str, Any],
    shin: Dict[str, float],
    moments: Dict[str, float],
    match_obj: Dict[str, Any],
) -> Tuple[str, List[Tuple[str, float]]]:
    g_min, g_max = goal_range
    candidates = {}

    if scenario == "extreme_blowout":
        label = {"home": "胜其他", "draw": "平其他", "away": "负其他"}[direction]
        odds_key = {"home": "crs_win", "draw": "crs_same", "away": "crs_lose"}[direction]
        odds = _f(match_obj.get(odds_key, 0))

        if 1.5 < odds < 80:
            return label, [(label, 100.0)]

    # CRS 概率
    for sc, p in crs_probs.items():
        sc = str(sc).strip()
        sc_dir = _score_direction(sc)

        if sc_dir != direction:
            continue

        is_other = sc in OTHER_SCORE_LABELS

        if is_other:
            if scenario == "extreme_blowout":
                candidates[sc] = candidates.get(sc, 0.0) + _f(p, 0.0)
            continue

        total_g = _score_total_goals(sc)
        if total_g is None:
            continue

        if not (g_min <= total_g <= g_max):
            continue

        candidates[sc] = candidates.get(sc, 0.0) + _f(p, 0.0)

    # AI 投票，只能同方向同区间
    for sc, vote_pts in ai_votes.items():
        sc = str(sc).replace(" ", "").strip()
        sc_dir = _score_direction(sc)

        if sc_dir != direction:
            continue

        total_g = _score_total_goals(sc)
        if total_g is None:
            continue

        if "其他" not in sc and not (g_min <= total_g <= g_max):
            continue

        candidates[sc] = candidates.get(sc, 0.0) + _f(vote_pts, 0.0) * 1.2

    # 陷阱乘数
    for sc, mult in trap_report.get("score_multipliers", {}).items():
        if sc in candidates:
            safe_mult = _clamp(_f(mult, 1.0), 0.15, 2.0)
            candidates[sc] *= safe_mult

    # boost 降低力度，避免过拟合
    for sc in trap_report.get("boost_scores", []):
        if sc in candidates:
            candidates[sc] *= 1.25

    # 场景修正
    if scenario in {"grinder", "low_draw"}:
        for sc in ["0-0", "1-1"]:
            if sc in candidates:
                candidates[sc] *= 1.35

        for sc in ["3-1", "1-3", "3-2", "2-3", "3-3"]:
            if sc in candidates:
                candidates[sc] *= 0.35

    elif scenario == "low_goals":
        for sc in ["1-0", "0-1", "2-1", "1-2", "1-1", "0-0"]:
            if sc in candidates:
                candidates[sc] *= 1.18

        for sc in ["3-2", "2-3", "3-3", "4-2", "2-4"]:
            if sc in candidates:
                candidates[sc] *= 0.35

    elif scenario == "normal":
        for sc in ["2-1", "1-2", "2-0", "0-2", "1-1"]:
            if sc in candidates:
                candidates[sc] *= 1.08

    elif scenario == "high_goals":
        for sc in ["2-1", "1-2", "3-1", "1-3", "2-2"]:
            if sc in candidates:
                candidates[sc] *= 1.12

        for sc in ["0-0", "1-0", "0-1"]:
            if sc in candidates:
                candidates[sc] *= 0.45

    elif scenario == "shootout":
        for sc in ["2-2", "3-2", "2-3", "3-3", "3-1", "1-3"]:
            if sc in candidates:
                candidates[sc] *= 1.22

        for sc in ["0-0", "1-0", "0-1"]:
            if sc in candidates:
                candidates[sc] *= 0.25

    # 强势方小幅加成
    if direction == "home" and shin.get("home", 33) > 58:
        for sc in ["2-0", "2-1", "3-1"]:
            if sc in candidates:
                candidates[sc] *= 1.05

    if direction == "away" and shin.get("away", 33) > 58:
        for sc in ["0-2", "1-2", "1-3"]:
            if sc in candidates:
                candidates[sc] *= 1.05

    if not candidates:
        fb = _fallback_score_for_lock(direction, goal_range, scenario)
        return fb, [(fb, 1.0)]

    sorted_scores = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[0][0], [(sc, round(p, 2)) for sc, p in sorted_scores[:10]]


# ====================================================================
# 终极一致性锁
# ====================================================================

def _normalize_top_candidates(candidates: Any) -> List[Tuple[str, float]]:
    out = []

    if not candidates:
        return out

    for item in candidates:
        score = None
        pts = 0.0

        if isinstance(item, dict):
            score = item.get("score") or item.get("predicted_score") or item.get("label")
            pts = _f(item.get("pts", item.get("prob", item.get("weight", 0.0))), 0.0)

        elif isinstance(item, (tuple, list)):
            if len(item) >= 1:
                score = item[0]
            if len(item) >= 2:
                pts = _f(item[1], 0.0)

        elif isinstance(item, str):
            score = item
            pts = 1.0

        score = str(score or "").replace(" ", "").strip()

        if not score:
            continue

        if _score_direction(score) not in {"home", "draw", "away"}:
            continue

        out.append((score, pts))

    out.sort(key=lambda x: x[1], reverse=True)
    return out


def _pick_candidate_for_lock(direction: str, goal_range: Any, top_candidates: Any) -> Optional[str]:
    for sc, pts in _normalize_top_candidates(top_candidates):
        if _score_direction(sc) != direction:
            continue

        if not _score_inside_goal_range(sc, goal_range):
            continue

        return sc

    return None


def _normalize_score_label_and_direction(score: Any) -> Tuple[str, str, str, bool]:
    s = str(score or "").replace(" ", "").strip()

    if "胜其他" in s or s == "9-0":
        return "胜其他", "胜其他", "home", True

    if "平其他" in s or s == "9-9":
        return "平其他", "平其他", "draw", True

    if "负其他" in s or s == "0-9":
        return "负其他", "负其他", "away", True

    d = _score_direction(s)

    if d not in {"home", "draw", "away"}:
        return "1-1", "1-1", "draw", False

    return s, s, d, False


def _normalize_probability_argmax(mg: Dict[str, Any], final_direction: str) -> Dict[str, Any]:
    pcts = {
        "home": _f(mg.get("home_win_pct", 33.3), 33.3),
        "draw": _f(mg.get("draw_pct", 33.3), 33.3),
        "away": _f(mg.get("away_win_pct", 33.3), 33.3),
    }

    current_argmax = max(pcts, key=pcts.get)

    if current_argmax != final_direction:
        pcts[final_direction] = max(pcts.values()) + 5.0

    total = sum(pcts.values())

    if total <= 0:
        pcts = {"home": 33.3, "draw": 33.3, "away": 33.3}
        total = sum(pcts.values())

    mg["home_win_pct"] = round(pcts["home"] / total * 100, 1)
    mg["draw_pct"] = round(pcts["draw"] / total * 100, 1)
    mg["away_win_pct"] = round(pcts["away"] / total * 100, 1)

    ordered = sorted(
        [mg["home_win_pct"], mg["draw_pct"], mg["away_win_pct"]],
        reverse=True,
    )

    mg["dir_confidence"] = round(ordered[0], 1)
    mg["dir_gap"] = round(ordered[0] - ordered[1], 1)

    return mg


def _enforce_consistency(mg: Dict[str, Any]) -> Dict[str, Any]:
    """
    终极锁：
    final_direction 是主变量。
    predicted_score 必须投影到 final_direction + goal_range。
    """
    if not isinstance(mg, dict):
        return mg

    audit = []

    pcts = {
        "home": _f(mg.get("home_win_pct", 33.3), 33.3),
        "draw": _f(mg.get("draw_pct", 33.3), 33.3),
        "away": _f(mg.get("away_win_pct", 33.3), 33.3),
    }

    final_direction = mg.get("final_direction")

    if final_direction not in {"home", "draw", "away"}:
        final_direction = max(pcts, key=pcts.get)
        audit.append(f"final_direction缺失→概率argmax:{final_direction}")

    current_score = str(mg.get("predicted_score", "")).replace(" ", "").strip()
    current_dir = _score_direction(current_score)
    goal_range = mg.get("goal_range")
    scenario = mg.get("scenario", "normal")
    top_candidates = mg.get("top_score_candidates", [])

    need_replace = False

    if current_dir != final_direction:
        need_replace = True
        audit.append(f"比分方向冲突:{current_score or '空'}({current_dir})≠{final_direction}")

    if not _score_inside_goal_range(current_score, goal_range):
        need_replace = True
        audit.append(f"比分不在区间:{current_score} not in {goal_range}")

    if need_replace:
        replacement = _pick_candidate_for_lock(final_direction, goal_range, top_candidates)

        if not replacement:
            replacement = _fallback_score_for_lock(final_direction, goal_range, scenario)
            audit.append(f"候选池无合法比分→fallback:{replacement}")
        else:
            audit.append(f"候选池重锁:{current_score or '空'}→{replacement}")

        current_score = replacement

    score, label, score_dir, is_others = _normalize_score_label_and_direction(current_score)

    if score_dir != final_direction:
        score = _fallback_score_for_lock(final_direction, goal_range, scenario)
        score, label, score_dir, is_others = _normalize_score_label_and_direction(score)
        audit.append(f"二次兜底方向修正→{score}")

    mg["predicted_score"] = score
    mg["predicted_label"] = label
    mg["final_direction"] = final_direction
    mg["result"] = DIRECTION_CN[final_direction]
    mg["display_direction"] = DIRECTION_CN[final_direction]
    mg["is_score_others"] = bool(is_others)

    mg = _normalize_probability_argmax(mg, final_direction)

    if audit:
        old = list(mg.get("lock_audit", []) or [])
        mg["lock_audit"] = old + audit

        sigs = list(mg.get("smart_signals", []) or [])
        sigs.insert(0, "🔒 CLEAN_LOCK:" + "；".join(audit[:3]))
        mg["smart_signals"] = sigs
        mg["smart_money_signal"] = " | ".join(sigs[:10])

    mg["engine_version"] = ENGINE_VERSION
    mg["engine_architecture"] = ENGINE_ARCHITECTURE

    return mg


# ====================================================================
# 决策锁定链
# ====================================================================

def decision_lock_chain(
    match_obj: Dict[str, Any],
    engine_result: Dict[str, Any],
    trap_report: Dict[str, Any],
    crs_analysis: Dict[str, Any],
    ai_responses: Dict[str, Dict],
    smart_signals: List[str],
    exp_goals: float,
) -> Dict[str, Any]:
    ai_directions = {"home": 0.0, "draw": 0.0, "away": 0.0}
    ai_votes = {}

    ai_weights = {
        "claude": 1.50,
        "gemini": 1.40,
        "grok": 1.35,
        "gpt": 1.00,
    }

    for name, r in ai_responses.items():
        if not isinstance(r, dict):
            continue

        sc_raw = str(r.get("ai_score", "")).replace(" ", "").strip()
        top3 = r.get("top3", [])

        sc = sc_raw
        h, a = _parse_score(sc)

        if h is None and top3:
            sc = _extract_top_score(top3[0], "")
            h, a = _parse_score(sc)

        if h is None:
            continue

        weight = ai_weights.get(name, 1.0)

        if h > a:
            d = "home"
        elif h < a:
            d = "away"
        else:
            d = "draw"

        ai_directions[d] += weight
        ai_votes[sc] = ai_votes.get(sc, 0.0) + weight

        for rank, t in enumerate(top3[1:3], 2):
            sc2 = _extract_top_score(t, "")
            h2, a2 = _parse_score(sc2)

            if h2 is None:
                continue

            w2 = 0.4 if rank == 2 else 0.2
            ai_votes[sc2] = ai_votes.get(sc2, 0.0) + weight * w2

    # Claude 高置信但不破坏硬多数，只加比分候选权重
    claude_r = ai_responses.get("claude", {})

    if isinstance(claude_r, dict) and _f(claude_r.get("ai_confidence", 0)) >= 75:
        cl_sc = str(claude_r.get("ai_score", "")).replace(" ", "").strip()
        cl_h, cl_a = _parse_score(cl_sc)

        if cl_h is not None:
            cl_dir = "home" if cl_h > cl_a else ("away" if cl_h < cl_a else "draw")

            other_dirs = {}
            other_confs = []
            valid_others = 0

            for n in ["gpt", "grok", "gemini"]:
                rr = ai_responses.get(n, {})
                if not isinstance(rr, dict):
                    continue

                h3, a3 = _parse_score(rr.get("ai_score", ""))

                if h3 is None:
                    continue

                valid_others += 1
                od = "home" if h3 > a3 else ("away" if h3 < a3 else "draw")
                other_dirs[od] = other_dirs.get(od, 0) + 1
                other_confs.append(_f(rr.get("ai_confidence", 60), 60))

            if other_dirs and valid_others >= 2:
                majority_dir = max(other_dirs, key=other_dirs.get)
                majority_count = other_dirs[majority_dir]
                hard_majority = majority_count >= max(2, int(valid_others * 0.67))
                avg_other_conf = sum(other_confs) / len(other_confs) if other_confs else 60
                claude_conf = _f(claude_r.get("ai_confidence", 60), 60)

                if cl_dir != majority_dir and hard_majority and claude_conf > avg_other_conf:
                    if cl_sc in ai_votes:
                        ai_votes[cl_sc] *= 2.0

    shin = trap_report.get("shin", {"home": 33.3, "draw": 33.3, "away": 33.3})
    crs_direction = crs_analysis.get("direction_probs", {})

    posterior_result = compute_direction_posterior(
        shin=shin,
        trap_report=trap_report,
        crs_direction=crs_direction,
        ai_directions=ai_directions,
        engine_result=engine_result,
        match_obj=match_obj,
        smart_signals=smart_signals,
    )

    posterior = posterior_result["posterior"]
    final_direction = posterior_result["final_direction"]

    trap_score = trap_report.get("total_severity", 0)
    override_triggered, override_dir, _ = check_sharp_override(shin, trap_report, posterior, trap_score)

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
    )

    best_score, top_candidates = select_score(
        direction=final_direction,
        goal_range=(goal_range_min, goal_range_max),
        scenario=scenario,
        crs_probs=crs_analysis.get("implied_probs", {}),
        ai_votes=ai_votes,
        trap_report=trap_report,
        shin=shin,
        moments=crs_analysis.get("moments", {}),
        match_obj=match_obj,
    )

    score, label, score_dir, is_others = _normalize_score_label_and_direction(best_score)

    if score_dir != final_direction:
        replacement = _pick_candidate_for_lock(final_direction, (goal_range_min, goal_range_max), top_candidates)
        if not replacement:
            replacement = _fallback_score_for_lock(final_direction, (goal_range_min, goal_range_max), scenario)

        score, label, score_dir, is_others = _normalize_score_label_and_direction(replacement)

    result_cn = DIRECTION_CN[final_direction]

    out = {
        "predicted_score": score,
        "predicted_label": label,
        "result": result_cn,
        "display_direction": result_cn,
        "final_direction": final_direction,
        "is_score_others": is_others,

        "home_win_pct": posterior["home"],
        "draw_pct": posterior["draw"],
        "away_win_pct": posterior["away"],

        "scenario": scenario,
        "goal_range": (goal_range_min, goal_range_max),
        "dir_confidence": posterior_result["dir_confidence"],
        "dir_gap": posterior_result["dir_gap"],
        "evidences": posterior_result["evidences"],
        "override_triggered": override_triggered,
        "top_score_candidates": top_candidates,
        "bayesian_prior": posterior_result["prior"],
    }

    return _enforce_consistency(out)


# ====================================================================
# AI Diary
# ====================================================================

def load_ai_diary():
    diary_file = "data/ai_diary.json"

    if os.path.exists(diary_file):
        try:
            with open(diary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "yesterday_win_rate": "N/A",
        "reflection": "v18.7 clean full 启动",
        "kill_history": [],
    }


def save_ai_diary(diary):
    os.makedirs("data", exist_ok=True)

    with open("data/ai_diary.json", "w", encoding="utf-8") as f:
        json.dump(diary, f, ensure_ascii=False, indent=2)


# ====================================================================
# AI Prompt
# ====================================================================

def build_v18_prompt(match_analyses):
    diary = load_ai_diary()

    p = "<context>\n"
    p += "你正在中国体彩竞彩足球市场进行量化比分预测。\n"
    p += "核心任务是识别庄家定价、赔率变动、Sharp/Steam、散户热度和CRS正确比分矩阵之间的冲突。\n"

    if diary.get("reflection"):
        p += f"[系统记忆] 昨日:{diary.get('yesterday_win_rate', 'N/A')} | 反思:{diary.get('reflection', '')}\n"

    p += "</context>\n\n"

    p += "<iron_rules>\n"
    p += "铁律1: top3[0].score 的方向必须与 final_direction 一致。\n"
    p += "铁律2: reason 结论方向必须与 top3[0].score 一致。\n"
    p += "铁律3: 0-0/0-1/0-2 都是合法比分，不得误判为空。\n"
    p += "铁律4: 若输出胜其他/平其他/负其他，is_score_others 必须为 true。\n"
    p += "铁律5: 不要为了追大赔率强行输出极端比分，除非 a5/a6/a7、让球、xG 三者共振。\n"
    p += "</iron_rules>\n\n"

    p += "<framework>\n"
    p += "每场 reason 按 Step1-Step5 写：\n"
    p += "Step1 方向：Shin/CRS/Sharp/Steam/散户冲突。\n"
    p += "Step2 陷阱：判断T1-T16是否成立。\n"
    p += "Step3 总进球：a0-a7、xG、CRS λ 判断区间。\n"
    p += "Step4 比分：只在方向×区间内选比分。\n"
    p += "Step5 EV：优先选方向、区间、赔率三者相容的比分。\n"
    p += "</framework>\n\n"

    p += "<output_format>\n"
    p += "严格输出 JSON 数组，每场字段：\n"
    p += "{\n"
    p += '  "match": 1,\n'
    p += '  "top3": [{"score":"2-1","prob":15},{"score":"1-0","prob":12},{"score":"2-0","prob":10}],\n'
    p += '  "reason": "Step1... Step2... Step3... Step4... Step5...",\n'
    p += '  "ai_confidence": 70,\n'
    p += '  "is_score_others": false,\n'
    p += '  "detected_traps": ["T1"],\n'
    p += '  "final_direction": "home"\n'
    p += "}\n"
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
        hc = m.get("give_ball", m.get("handicap", "0"))

        sp_h = _f(m.get("sp_home", m.get("win", 0)))
        sp_d = _f(m.get("sp_draw", m.get("same", 0)))
        sp_a = _f(m.get("sp_away", m.get("lose", 0)))

        p += f'<match index="{i + 1}">\n'
        p += f"[{i + 1}] {h} vs {a} | {league}\n"
        p += f"欧赔:{sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f} | 让球:{hc} | 内部盘口:{_parse_actual_handicap(m):+.2f}\n"

        shin = trap_preview.get("shin", {})
        if shin:
            p += f"Shin: 主{shin.get('home',0):.1f}% 平{shin.get('draw',0):.1f}% 客{shin.get('away',0):.1f}%\n"

        hxg = eng.get("bookmaker_implied_home_xg", "?")
        axg = eng.get("bookmaker_implied_away_xg", "?")
        p += f"xG: 主{hxg} vs 客{axg}\n"

        moments = crs_preview.get("moments", {})
        if moments:
            p += (
                f"CRS矩: λ主{moments.get('lambda_h',0):.2f}/客{moments.get('lambda_a',0):.2f} "
                f"总{moments.get('lambda_total',0):.2f} corr{moments.get('corr',0):+.2f} "
                f"形状={crs_preview.get('shape_verdict','?')}\n"
            )

        traps = trap_preview.get("traps_detected", [])
        if traps:
            p += f"陷阱预扫({len(traps)}个,严重度{trap_preview.get('total_severity',0)}):\n"
            for t in traps:
                p += f"- {t.get('trap','?')}: {t.get('description','')[:120]}\n"
        else:
            p += "陷阱预扫: 无明显触发\n"

        a_list = []
        compressed = []

        for g in range(8):
            v = m.get(f"a{g}", "")
            if v != "":
                a_list.append(f"{g}={v}")

                actual = _f(v, 0)
                if actual > 1:
                    std = STANDARD_GOAL_ODDS.get(g, 50)
                    ratio = std / actual

                    if ratio > 1.5:
                        compressed.append(f"{g}球压低{ratio:.1f}x")

        if a_list:
            p += f"总进球赔率: {' | '.join(a_list)}\n"

        if compressed:
            p += f"进球数压低: {', '.join(compressed)}\n"

        crs_lines = []
        for sc, key in CRS_FULL_MAP.items():
            odds = _f(m.get(key, 0))
            if odds > 1:
                crs_lines.append(f"{sc}={odds:.1f}")

        if crs_lines:
            p += f"CRS: {' | '.join(crs_lines)}\n"

        other_lines = []
        for k, label in [("crs_win", "胜其他"), ("crs_same", "平其他"), ("crs_lose", "负其他")]:
            v = m.get(k, "")
            if v:
                other_lines.append(f"{label}={v}")

        if other_lines:
            p += f"其他比分: {' | '.join(other_lines)}\n"

        vote = m.get("vote", {})
        if isinstance(vote, dict) and vote:
            p += f"散户: 胜{vote.get('win','?')}% 平{vote.get('same','?')}% 负{vote.get('lose','?')}%\n"

        change = m.get("change", {})
        if isinstance(change, dict):
            cw = change.get("win", 0)
            cs = change.get("same", 0)
            cl = change.get("lose", 0)
            if cw or cs or cl:
                p += f"赔率变动: 胜{cw} 平{cs} 负{cl}；负数=降水\n"

        info = m.get("information", {})
        if isinstance(info, dict):
            for k, label in [
                ("home_injury", "主伤停"),
                ("guest_injury", "客伤停"),
                ("home_bad_news", "主利空"),
                ("guest_bad_news", "客利空"),
            ]:
                if info.get(k):
                    p += f"{label}: {str(info[k])[:400].replace(chr(10), ' ')}\n"

        points = m.get("points", {})
        if isinstance(points, dict):
            for k in ["home_strength", "guest_strength", "match_points"]:
                txt = str(points.get(k, ""))[:500].replace("\n", " ")
                if txt:
                    p += f"情报: {txt}\n"

        smart_sigs = stats.get("smart_signals", []) if stats else []
        if smart_sigs:
            p += f"信号: {', '.join(str(s) for s in smart_sigs[:8])}\n"

        p += "</match>\n\n"

    p += "</match_data>\n"
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

GPT_DEFAULT_URL = "https://api.newapi.life/v1"
GPT_DEFAULT_KEY = ""


def get_clean_env_url(name, default=""):
    v = str(os.environ.get(name, globals().get(name, default))).strip(" \t\n\r\"'")
    m = re.search(r"(https?://[a-zA-Z0-9._:/-]+)", v)
    return m.group(1) if m else v


def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")


async def async_call_one_ai_batch(
    session,
    prompt,
    url_env,
    key_env,
    models_list,
    num_matches,
    ai_name,
):
    key = get_clean_env_key(key_env)

    if not key and ai_name == "gpt":
        key = GPT_DEFAULT_KEY

    if not key:
        return ai_name, {}, "no_key"

    if ai_name == "gpt":
        primary_url = get_clean_env_url(url_env, GPT_DEFAULT_URL)
        if not primary_url:
            primary_url = GPT_DEFAULT_URL
        urls = [primary_url]
        print(f"    🔌 [GPT] 使用通道: {primary_url}")
    else:
        primary_url = get_clean_env_url(url_env)
        backup = [u for u in FALLBACK_URLS if u and u != primary_url][:1]
        urls = [primary_url] + backup

    CONNECT_TIMEOUT = 20
    READ_TIMEOUT_MAP = {
        "claude": 380,
        "grok": 300,
        "gpt": 300,
        "gemini": 250,
    }
    READ_TIMEOUT = READ_TIMEOUT_MAP.get(ai_name, 220)

    AI_PROFILES = {
        "claude": {
            "sys": (
                "<role>你是足球市场微观结构和庄家定价分析师。</role>\n"
                "<priority>严格遵守用户消息中的 iron_rules，方向和比分必须一致。</priority>\n"
                "<instruction>只输出 JSON 数组，禁止前缀后缀。</instruction>"
            ),
            "temp": 0.22,
        },
        "gpt": {
            "sys": (
                "<role>你是足球概率分布和正确比分定价策略师。</role>\n"
                "<priority>严格遵守 iron_rules，尤其 0-0/0-1/0-2 合法。</priority>\n"
                "<instruction>只输出 JSON 数组。</instruction>"
            ),
            "temp": 0.18,
        },
        "grok": {
            "sys": (
                "<role>你是足球另类数据和情绪背离分析师。</role>\n"
                "<priority>识别散户过热、Sharp/Steam背离和赔率假动作。</priority>\n"
                "<instruction>只输出 JSON 数组。</instruction>"
            ),
            "temp": 0.28,
        },
        "gemini": {
            "sys": (
                "<role>你是非线性特征和多源共振识别模型。</role>\n"
                "<priority>必须保证 top1 比分方向与 final_direction 一致。</priority>\n"
                "<instruction>只输出 JSON 数组。</instruction>"
            ),
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

            gw = url.split("/v1")[0][:35]
            print(f"  [🔌连接中] {ai_name.upper()} | {mn[:22]} @ {gw}")
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
                        print(f"    💀 HTTP {r.status} | {elapsed_connect}s → 换URL")
                        continue

                    if r.status == 400:
                        print(f"    💀 400 | {elapsed_connect}s → 换模型")
                        break

                    if r.status == 429:
                        print(f"    🔥 429 | {elapsed_connect}s → 换URL")
                        await asyncio.sleep(1)
                        continue

                    if r.status != 200:
                        print(f"    ⚠️ HTTP {r.status} | {elapsed_connect}s → 换URL")
                        continue

                    connected = True
                    print(f"    ✅ 已连上! {elapsed_connect}s | 等待数据...")

                    try:
                        data = await r.json(content_type=None)
                    except Exception:
                        print("    ⚠️ 响应非JSON → 换模型")
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
                        print(f"    📊 {req_tokens:,} token | {elapsed}s")

                    raw_text = _extract_response_text(data, is_gem, ai_name)

                    if not raw_text or len(raw_text) < 10:
                        print("    ⚠️ 空数据 → 换模型")
                        _save_debug_dump(ai_name, data, "empty")
                        break

                    results = _parse_ai_json(raw_text, num_matches)

                    if len(results) > 0:
                        print(f"    ✅ {ai_name.upper()} 完成: {len(results)}/{num_matches} | {elapsed}s")
                        return ai_name, results, mn

                    print("    ⚠️ 解析0条 → 换模型")
                    _save_debug_dump(ai_name, data, "parse0")
                    break

            except aiohttp.ClientConnectorError:
                print("    🔌 连接失败 → 换URL")
                continue

            except asyncio.TimeoutError:
                if not connected:
                    print("    🔌 连接超时 → 换URL")
                    continue

                print("    ⏰ 读取超时")
                return ai_name, {}, "read_timeout"

            except Exception as e:
                if not connected:
                    print(f"    ⚠️ {str(e)[:60]} → 换URL")
                    continue

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

                            if v and isinstance(v, str) and v.strip():
                                raw_text = v.strip()
                                break

                    if not raw_text:
                        skip = {
                            "reasoning_content",
                            "thinking",
                            "reasoning",
                            "reasoning_text",
                            "thoughts",
                            "thought_process",
                            "internal_thinking",
                            "chain_of_thought",
                            "cot",
                            "deliberation",
                            "analysis_process",
                        }

                        best = ""

                        for k, v in msg.items():
                            if k in skip:
                                continue

                            if isinstance(v, str) and '"match"' in v and "[" in v:
                                if len(v) > len(best):
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
                m = re.search(r'\[\s*\{\s*\\?"match\\?"', full_str)

                if m:
                    start = m.start()
                    depth = 0
                    end = start

                    for i in range(start, min(start + 120000, len(full_str))):
                        if full_str[i] == "[":
                            depth += 1
                        elif full_str[i] == "]":
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

                        raw_text = extracted
                        print("    🆘 终极兜底: 从response dump中提取JSON")

    except Exception as ex:
        print(f"    ⚠️ 响应解析异常: {str(ex)[:80]}")

    return raw_text


def _parse_ai_json(raw_text, num_matches):
    clean = str(raw_text or "")
    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"```[\w]*", "", clean).strip()

    json_str = ""

    m = re.search(r'\[\s*\{\s*"match"', clean)

    if m:
        start = m.start()
        depth = 0
        end = start

        for i in range(start, len(clean)):
            if clean[i] == "[":
                depth += 1
            elif clean[i] == "]":
                depth -= 1

                if depth == 0:
                    end = i + 1
                    break

        if end > start:
            json_str = clean[start:end]
            print(f"    🎯 精确匹配JSON: {len(json_str)}字")

    if not json_str:
        start = clean.find("[")
        end = clean.rfind("]") + 1

        if start != -1 and end > start:
            json_str = clean[start:end]
            print(f"    🔍 兜底匹配JSON: {len(json_str)}字")

    results = {}

    if not json_str:
        return results

    try:
        arr = json.loads(json_str)
    except json.JSONDecodeError:
        try:
            last_brace = json_str.rfind("}")
            arr = json.loads(json_str[:last_brace + 1] + "]") if last_brace != -1 else []

            if arr:
                print(f"    🩹 断肢重生: {len(arr)}条")
        except Exception:
            arr = []

    if not isinstance(arr, list):
        return results

    for item in arr:
        if not isinstance(item, dict) or not item.get("match"):
            continue

        try:
            mid = int(item["match"])
        except Exception:
            continue

        top3 = item.get("top3", [])

        if top3:
            t1 = _extract_top_score(top3[0], "1-1")

            results[mid] = {
                "top3": top3,
                "ai_score": t1,
                "reason": str(item.get("reason", ""))[:1000],
                "ai_confidence": int(_f(item.get("ai_confidence", 60), 60)),
                "is_score_others": bool(item.get("is_score_others", False)),
                "detected_traps": item.get("detected_traps", []),
                "final_direction": item.get("final_direction", ""),
            }

        elif item.get("score"):
            results[mid] = {
                "ai_score": str(item["score"]).replace(" ", "").strip(),
                "reason": str(item.get("reason", ""))[:1000],
                "ai_confidence": int(_f(item.get("ai_confidence", 60), 60)),
                "is_score_others": bool(item.get("is_score_others", False)),
                "detected_traps": item.get("detected_traps", []),
            }

    return results


def _save_debug_dump(ai_name, data, tag):
    try:
        os.makedirs("data/debug", exist_ok=True)
        dump_file = f"data/debug/{ai_name}_{tag}_{int(time.time())}.json"

        with open(dump_file, "w", encoding="utf-8") as df:
            json.dump(data, df, ensure_ascii=False, indent=2)

        print(f"    📁 失败响应已保存: {dump_file}")
    except Exception:
        pass


async def run_ai_matrix_two_phase(match_analyses):
    num = len(match_analyses)
    prompt = build_v18_prompt(match_analyses)

    print(f"  [v18.7 Prompt] {len(prompt):,} 字符 → 4AI并行...")

    ai_configs = [
        ("grok", "GROK_API_URL", "GROK_API_KEY", ["熊猫-A-5-grok-4.2-fast-200w上下文"]),
        ("gpt", "GPT_API_URL", "GPT_API_KEY", ["gpt-5.4"]),
        ("gemini", "GEMINI_API_URL", "GEMINI_API_KEY", ["熊猫-特供-X-12-gemini-3.1-pro-preview-thinking"]),
        ("claude", "CLAUDE_API_URL", "CLAUDE_API_KEY", ["熊猫-69-满血openrouter-claude-opus-4.7-上下文1000k"]),
    ]

    all_results = {"gpt": {}, "grok": {}, "gemini": {}, "claude": {}}

    connector = aiohttp.TCPConnector(limit=10, use_dns_cache=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            async_call_one_ai_batch(session, prompt, u, k, m, num, n)
            for n, u, k, m in ai_configs
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, tuple):
                all_results[res[0]] = res[1]
            else:
                print(f"  [ERROR] {res}")

    ok = sum(1 for v in all_results.values() if v)
    print(f"  [完成] {ok}/4 AI有数据")

    return all_results


def _run_coro_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        def _runner():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)

            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_runner).result()

    return asyncio.run(coro)


# ====================================================================
# 合并结果
# ====================================================================

def _estimate_exp_goals(match_obj: Dict[str, Any], engine_result: Dict[str, Any], stats: Dict[str, Any]) -> float:
    exp_goals = 0.0

    for src in [engine_result, stats]:
        if not isinstance(src, dict):
            continue

        for k in [
            "expected_total_goals",
            "exp_goals",
            "total_goals",
            "expected_goals",
            "lambda_total",
            "total_xg",
        ]:
            v = src.get(k)

            if v is not None:
                fv = _f(v, 0.0)

                if fv > 0.5:
                    exp_goals = fv
                    break

        if exp_goals > 0:
            break

    if exp_goals <= 0:
        hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0)) if engine_result else 0
        axg = _f(engine_result.get("bookmaker_implied_away_xg", 0)) if engine_result else 0

        if hxg > 0 and axg > 0:
            exp_goals = hxg + axg
            print(f"    📐 期望进球用xG总和: {hxg:.2f}+{axg:.2f}={exp_goals:.2f}")

    if exp_goals <= 0:
        gp = []

        for gi in range(8):
            v = _f(match_obj.get(f"a{gi}", 0))
            if v > 1:
                gp.append((gi, 1 / v))

        if gp:
            total_prob = sum(p for _, p in gp)
            exp_goals = sum(g * (p / total_prob) for g, p in gp)
            print(f"    📐 期望进球用a0-a7反推: {exp_goals:.2f}")

    if exp_goals < 1.0 or exp_goals > 6.0:
        print(f"    ⚠️ 期望进球异常({exp_goals:.2f})，使用默认2.5")
        exp_goals = 2.5

    return exp_goals


def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    if isinstance(match_obj.get("v2_odds_dict"), dict):
        v2 = match_obj["v2_odds_dict"]
        match_obj = {**match_obj, **v2}
        print(f"    🔧 [字段兼容] v2_odds_dict→顶层({len(v2)}字段)")

    build_match_index(match_obj, stats)

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
        print(f"    🚫 弃权AI: {', '.join(abstained)}")

    ai_responses = {}

    if ai_valid["claude"]:
        ai_responses["claude"] = claude_r
    if ai_valid["gpt"]:
        ai_responses["gpt"] = gpt_r
    if ai_valid["grok"]:
        ai_responses["grok"] = grok_r
    if ai_valid["gemini"]:
        ai_responses["gemini"] = gemini_r

    engine_result = engine_result or {}
    stats = stats or {}

    exp_goals = _estimate_exp_goals(match_obj, engine_result, stats)
    smart_signals = stats.get("smart_signals", []) if stats else []

    trap_report = detect_all_traps(
        match_obj,
        engine_result,
        ai_responses,
        smart_signals,
        exp_goals,
    )

    if trap_report["trap_count"] > 0:
        print(f"    🎭 陷阱: {trap_report['trap_count']}个 严重度{trap_report['total_severity']}")
        for t in trap_report["traps_detected"][:4]:
            print(f"       [{t['trap']}] {t['description'][:80]}")

    crs_analysis = analyze_crs_matrix(match_obj)

    if crs_analysis["coverage"] > 0:
        print(f"    📊 CRS: 覆盖{crs_analysis['coverage'] * 100:.0f}% 形状={crs_analysis['shape_verdict']}")

    lock_result = decision_lock_chain(
        match_obj=match_obj,
        engine_result=engine_result,
        trap_report=trap_report,
        crs_analysis=crs_analysis,
        ai_responses=ai_responses,
        smart_signals=smart_signals,
        exp_goals=exp_goals,
    )

    print(
        f"    🎯 方向: 主{lock_result['home_win_pct']:.0f}% "
        f"平{lock_result['draw_pct']:.0f}% "
        f"客{lock_result['away_win_pct']:.0f}%"
    )

    for ev in lock_result.get("evidences", [])[:3]:
        print(f"       - {ev}")

    predicted_score = lock_result["predicted_score"]
    predicted_label = lock_result["predicted_label"]
    result_cn = lock_result["result"]
    display_direction = lock_result["display_direction"]
    final_direction = lock_result["final_direction"]
    is_score_others = lock_result["is_score_others"]

    posterior_pct = {
        "home": lock_result["home_win_pct"],
        "draw": lock_result["draw_pct"],
        "away": lock_result["away_win_pct"],
    }

    target_crs_key = CRS_FULL_MAP.get(predicted_score, "")
    final_odds = _f(match_obj.get(target_crs_key, 0))

    if not final_odds and is_score_others:
        if final_direction == "home":
            final_odds = _f(match_obj.get("crs_win", 0))
        elif final_direction == "draw":
            final_odds = _f(match_obj.get("crs_same", 0))
        else:
            final_odds = _f(match_obj.get("crs_lose", 0))

    model_score_prob = _estimate_score_model_prob(
        predicted_score=predicted_score,
        final_direction=final_direction,
        posterior_pct=posterior_pct,
        top_candidates=lock_result.get("top_score_candidates", []),
        crs_probs=crs_analysis.get("implied_probs", {}),
    )

    ev_data = calculate_value_bet(model_score_prob, final_odds)

    engine_conf = _f(engine_result.get("confidence", 50), 50)

    weights = {
        "claude": 1.40,
        "gemini": 1.35,
        "grok": 1.30,
        "gpt": 1.10,
    }

    ai_conf_sum = 0.0
    ai_conf_count = 0.0
    value_kills = 0

    for name, r in ai_responses.items():
        conf = _f(r.get("ai_confidence", 60), 60)
        ai_conf_sum += conf * weights.get(name, 1.0)
        ai_conf_count += weights.get(name, 1.0)

        if r.get("value_kill"):
            value_kills += 1

    avg_ai_conf = ai_conf_sum / ai_conf_count if ai_conf_count > 0 else 60

    cf = min(95, engine_conf + int((avg_ai_conf - 60) * 0.35)) + value_kills * 5
    cf -= trap_report.get("confidence_penalty", 0)

    dir_confidence = _f(lock_result.get("dir_confidence", 50), 50)
    dir_gap = _f(lock_result.get("dir_gap", 0), 0)

    if dir_confidence >= 70:
        cf += min(8, int((dir_confidence - 70) // 4))
    elif dir_confidence < 50:
        cf -= 8

    if dir_gap < 10:
        cf -= 5

    cf = int(_clamp(cf, 30, 95))
    risk = "低" if cf >= 75 else ("中" if cf >= 55 else "高")

    cold_strength = 0
    cold_level = None
    cold_signals = []

    for t in trap_report["traps_detected"]:
        if t["trap"] in {
            "T8_FALSE_COLD",
            "T4_FAKE_HOME_FAVORITE",
            "T5_FAKE_AWAY_FAVORITE",
            "T14_CUP_FAVORITE",
        }:
            cold_strength += t["severity"] * 3
            cold_signals.append(t["description"])

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
        "signals": cold_signals,
        "sharp_confirmed": trap_report.get("sharp_detected", False),
        "dark_verdict": f"❄️ {cold_level}冷门!{len(cold_signals)}条触发" if cold_level else "",
    }

    sigs = list(smart_signals or [])

    calibrated_ev_signal = f"🧮 校准EV:{ev_data['ev']:+.1f}% Kelly:{ev_data['kelly']:.2f}%"
    sigs.insert(0, calibrated_ev_signal)

    for t in trap_report["traps_detected"]:
        sigs.append(f"🎭 {t['trap']}:{t['description'][:60]}")

    if is_score_others:
        sigs.append("🔥 其他比分聚合触发")

    if lock_result.get("override_triggered"):
        sigs.append("⚡ Sharp Override 触发")

    cl_raw = claude_r.get("ai_score", "") if isinstance(claude_r, dict) else ""
    cl_sc = cl_raw if _parse_score(cl_raw)[0] is not None else predicted_score

    mg = {
        "predicted_score": predicted_score,
        "predicted_label": predicted_label,
        "result": result_cn,
        "display_direction": display_direction,
        "final_direction": final_direction,
        "is_score_others": is_score_others,

        "home_win_pct": round(lock_result["home_win_pct"], 1),
        "draw_pct": round(lock_result["draw_pct"], 1),
        "away_win_pct": round(lock_result["away_win_pct"], 1),

        "confidence": cf,
        "risk_level": risk,
        "dir_confidence": dir_confidence,
        "dir_gap": dir_gap,

        "scenario": lock_result["scenario"],
        "goal_range": lock_result["goal_range"],

        "bayesian_evidences": lock_result["evidences"],
        "bayesian_prior": lock_result["bayesian_prior"],
        "override_triggered": lock_result["override_triggered"],

        "traps_detected": [t["trap"] for t in trap_report["traps_detected"]],
        "trap_count": trap_report["trap_count"],
        "trap_severity": trap_report["total_severity"],
        "trap_details": [{"trap": t["trap"], "desc": t["description"]} for t in trap_report["traps_detected"]],

        "crs_shape": crs_analysis.get("shape_verdict", "unknown"),
        "crs_moments": crs_analysis.get("moments", {}),
        "crs_margin": crs_analysis.get("margin", 0.0),
        "crs_coverage": crs_analysis.get("coverage", 0.0),
        "crs_implied_probs": crs_analysis.get("implied_probs", {}),
        "top_score_candidates": lock_result["top_score_candidates"],

        "gpt_score": gpt_r.get("ai_score", "弃权") if ai_valid["gpt"] else "弃权",
        "gpt_analysis": gpt_r.get("reason", gpt_r.get("analysis", "弃权")) if ai_valid["gpt"] else "弃权 (AI失效,本场不参与决策)",

        "grok_score": grok_r.get("ai_score", "弃权") if ai_valid["grok"] else "弃权",
        "grok_analysis": grok_r.get("reason", grok_r.get("analysis", "弃权")) if ai_valid["grok"] else "弃权 (AI失效,本场不参与决策)",

        "gemini_score": gemini_r.get("ai_score", "弃权") if ai_valid["gemini"] else "弃权",
        "gemini_analysis": gemini_r.get("reason", gemini_r.get("analysis", "弃权")) if ai_valid["gemini"] else "弃权 (AI失效,本场不参与决策)",

        "claude_score": cl_sc if ai_valid["claude"] else "弃权",
        "claude_analysis": claude_r.get("reason", claude_r.get("analysis", "弃权")) if ai_valid["claude"] else "弃权 (AI失效,本场不参与决策)",

        "ai_abstained": [n.upper() for n, v in ai_valid.items() if not v],
        "ai_avg_confidence": round(avg_ai_conf, 1),
        "value_kill_count": value_kills,

        "suggested_kelly": ev_data["kelly"],
        "edge_vs_market": ev_data["ev"],
        "is_value": ev_data["is_value"],
        "model_score_prob": model_score_prob,
        "final_odds": final_odds,

        "smart_money_signal": " | ".join(sigs[:10]),
        "smart_signals": sigs,

        "cold_door": cold_door,

        "xG_home": round(_f(engine_result.get("bookmaker_implied_home_xg", 1.3)), 2),
        "xG_away": round(_f(engine_result.get("bookmaker_implied_away_xg", 0.9)), 2),
        "over_under_2_5": "大" if _f(engine_result.get("over_25", 50), 50) > 55 else "小",
        "both_score": "是" if _f(engine_result.get("btts", 45), 45) > 50 else "否",
        "expected_total_goals": round(exp_goals, 2),
        "over_2_5": engine_result.get("over_25", 50),
        "btts": engine_result.get("btts", 45),
        "bookmaker_implied_home_xg": engine_result.get("bookmaker_implied_home_xg", "?"),
        "bookmaker_implied_away_xg": engine_result.get("bookmaker_implied_away_xg", "?"),

        "sharp_detected": trap_report.get("sharp_detected", False),
        "sharp_dir": trap_report.get("sharp_dir"),
        "shin_dir": max(trap_report["shin"], key=trap_report["shin"].get),

        "model_consensus": stats.get("model_consensus", 0),
        "total_models": stats.get("total_models", 11),
        "extreme_warning": engine_result.get("scissors_gap_signal", ""),

        "refined_poisson": stats.get("refined_poisson", {}),
        "poisson": {},
        "elo": stats.get("elo", {}),
        "random_forest": stats.get("random_forest", {}),
        "gradient_boost": stats.get("gradient_boost", {}),
        "neural_net": stats.get("neural_net", {}),
        "logistic": stats.get("logistic", {}),
        "svm": stats.get("svm", {}),
        "knn": stats.get("knn", {}),
        "dixon_coles": stats.get("dixon_coles", {}),
        "bradley_terry": stats.get("bradley_terry", {}),
        "home_form": stats.get("home_form", {}),
        "away_form": stats.get("away_form", {}),
        "handicap_signal": stats.get("handicap_signal", ""),
        "odds_movement": stats.get("odds_movement", {}),
        "vote_analysis": stats.get("vote_analysis", {}),
        "h2h_blood": stats.get("h2h_blood", {}),
        "crs_analysis": stats.get("crs_analysis", {}),
        "ttg_analysis": stats.get("ttg_analysis", {}),
        "halftime": stats.get("halftime", {}),
        "pace_rating": stats.get("pace_rating", ""),
        "kelly_home": stats.get("kelly_home", {}),
        "kelly_away": stats.get("kelly_away", {}),
        "odds": stats.get("odds", {}),
        "experience_analysis": stats.get("experience_analysis", {}),
        "pro_odds": stats.get("pro_odds", {}),
        "asian_handicap_probs": stats.get("asian_handicap_probs", {}),
        "top_scores": stats.get("refined_poisson", {}).get("top_scores", []),

        "engine_version": ENGINE_VERSION,
        "engine_architecture": ENGINE_ARCHITECTURE,
    }

    return _enforce_consistency(mg)


# ====================================================================
# Top4 精选
# ====================================================================

def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})

        s = pr.get("confidence", 0) * 0.40
        s += pr.get("dir_confidence", 50) * 0.15

        trap_count = pr.get("trap_count", 0)

        if trap_count >= 2:
            s += 6
        elif trap_count >= 1:
            s += 3

        ev = pr.get("edge_vs_market", 0)

        if ev >= 30:
            s += 8
        elif ev >= 15:
            s += 4

        cold = pr.get("cold_door", {})

        if cold.get("is_cold_door") and pr.get("confidence", 0) >= 60:
            s += 3

        if pr.get("risk_level") == "高":
            s -= 10
        elif pr.get("risk_level") == "低":
            s += 6

        if pr.get("is_score_others"):
            s -= 2

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

def run_predictions(raw, use_ai=True, use_web=False):
    ms = raw.get("matches", [])

    print("\n" + "=" * 80)
    print(f"  [{ENGINE_VERSION}] CLEAN FULL | {len(ms)} 场")
    print("=" * 80)

    if use_web:
        print("  [WebIntel] 当前完整整合版不启用外部网页抓取，继续本地/AI/赔率模式。")

    match_analyses = []

    for i, m in enumerate(ms):
        try:
            build_match_index(m)
        except Exception as e:
            logger.debug(f"build_match_index异常: {e}")

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
        except Exception:
            sp = {}

        try:
            exp_result = exp_engine.analyze(m) if exp_engine else {}
        except Exception:
            exp_result = {}

        exp_goals_prev = _f(eng.get("expected_total_goals", 0))

        if exp_goals_prev <= 0:
            hxg = _f(eng.get("bookmaker_implied_home_xg", 0))
            axg = _f(eng.get("bookmaker_implied_away_xg", 0))
            exp_goals_prev = hxg + axg if hxg and axg else 2.5

        trap_preview = detect_all_traps(
            m,
            eng,
            {},
            sp.get("smart_signals", []) if sp else [],
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
        print("  [v18.7 AI] 启动4AI并行...")
        start_t = time.time()

        try:
            all_ai = _run_coro_sync(run_ai_matrix_two_phase(match_analyses))
        except Exception as e:
            logger.error(f"AI 矩阵执行崩溃: {e}")
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

        # 后处理允许保留，但最后必须再次一致性锁
        try:
            if exp_engine:
                mg = apply_experience_to_prediction(m, mg, exp_engine)
        except Exception:
            pass

        try:
            mg = apply_odds_history(m, mg)
        except Exception:
            pass

        try:
            mg = apply_quant_edge(m, mg)
        except Exception:
            pass

        try:
            mg = apply_wencai_intel(m, mg)
        except Exception:
            pass

        try:
            mg = upgrade_ensemble_predict(m, mg)
        except Exception:
            pass

        mg = _enforce_consistency(mg)

        res.append({**m, "prediction": mg})

        trap_tag = f" [🎭{mg['trap_count']}陷阱]" if mg.get("trap_count", 0) > 0 else ""
        others_tag = " [其他]" if mg.get("is_score_others") else ""
        sharp_tag = " [Sharp]" if mg.get("sharp_detected") else ""
        override_tag = " [Override]" if mg.get("override_triggered") else ""
        scenario_tag = f" [{mg.get('scenario', 'normal')}]"

        print(
            f"  [{i + 1}] {m.get('home_team', m.get('home'))} vs {m.get('away_team', m.get('guest'))} => "
            f"{mg['result']} ({mg['predicted_score']}={mg['predicted_label']}) | "
            f"CF:{mg['confidence']}% | 方向:{mg['dir_confidence']:.0f}%"
            f"{trap_tag}{others_tag}{sharp_tag}{override_tag}{scenario_tag}"
        )

    t4 = select_top4(res)
    t4ids = [t.get("id") for t in t4]

    for r in res:
        r["is_recommended"] = r.get("id") in t4ids

    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    diary = load_ai_diary()
    total_traps = sum(r.get("prediction", {}).get("trap_count", 0) for r in res)
    others_count = sum(1 for r in res if r.get("prediction", {}).get("is_score_others"))
    sharp_count = sum(1 for r in res if r.get("prediction", {}).get("sharp_detected"))
    high_conf = sum(1 for r in res if r.get("prediction", {}).get("confidence", 0) > 70)

    diary["yesterday_win_rate"] = f"{high_conf}/{max(1, len(res))}"
    diary["reflection"] = (
        f"{ENGINE_VERSION} | {total_traps}陷阱 {others_count}其他比分 "
        f"{sharp_count}Sharp | CRS聚合修复+方向优先锁分"
    )

    save_ai_diary(diary)

    return res, t4


if __name__ == "__main__":
    logger.info(f"{ENGINE_VERSION} 启动")
    print(f"✅ {ENGINE_VERSION} 加载完成")
    print(f"   架构: {ENGINE_ARCHITECTURE}")
    print("   约束: predicted_score ↔ result ↔ display_direction ↔ final_direction ↔ 概率argmax")