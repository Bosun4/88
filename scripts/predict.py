# ====================================================================
# 🚀 vMAX 19.1 — Raw Packet AI Analyst
# --------------------------------------------------------------------
# 核心逻辑:
#   ✅ 引擎职责: 抓包标准化 + 原始字段展开 + 中性观察信号 + JSON校验
#   ✅ AI职责: 自主分析欧赔/亚盘/总进球/CRS/半全场/散户/基本面/信号
#   ✅ 信号只作为观察事实,不强迫AI采纳
#   ✅ 不再三家投票,不再Claude按多数派裁判
#   ✅ 最终预测来自单个主AI,程序只校验字段一致性
#   ✅ 旧增强模块默认不允许改核心结果
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

ENGINE_VERSION = "vMAX 19.1"
ENGINE_ARCHITECTURE = "Raw Packet AI Analyst"

# 默认不让旧模块修改核心预测
APPLY_LEGACY_ENHANCERS = False

LOCKED_CORE_FIELDS = {
    "predicted_score",
    "predicted_label",
    "result",
    "display_direction",
    "final_direction",
    "predicted_direction",
    "home_win_pct",
    "draw_pct",
    "away_win_pct",
    "confidence",
    "ai_confidence",
    "ai_confidence_pct",
    "confidence_score",
    "goal_range",
    "goal_interval",
    "predicted_goal_range",
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

VALID_DIRS = {"home", "draw", "away"}


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


def _pct_normalize(h, d, a) -> Tuple[float, float, float]:
    h = _f(h, 33.3)
    d = _f(d, 33.3)
    a = _f(a, 33.4)

    total = h + d + a
    if total <= 0:
        return 33.3, 33.3, 33.4

    # 兼容 0~1 概率
    if 0.8 <= total <= 1.2:
        h, d, a = h * 100, d * 100, a * 100
        total = h + d + a

    return (
        round(h / total * 100, 1),
        round(d / total * 100, 1),
        round(a / total * 100, 1),
    )


# ====================================================================
# Match 标准化
# ====================================================================

def normalize_match(raw_m: Dict) -> Dict:
    """
    把抓包里的嵌套赔率展开。
    这是 v19.1 的关键：所有后续分析都用标准化后的 m。
    """
    m = dict(raw_m or {})

    for nested_key in [
        "v2_odds_dict",
        "odds_dict",
        "odds",
        "v2",
        "odds_v2",
        "packet",
        "raw_odds",
    ]:
        nested = m.get(nested_key)
        if isinstance(nested, dict):
            m.update(nested)

    # 主客队字段
    if "home_team" not in m:
        m["home_team"] = m.get("home", m.get("host", m.get("team_home", "Home")))
    if "away_team" not in m:
        m["away_team"] = m.get("guest", m.get("away", m.get("team_away", "Away")))

    if "home" not in m:
        m["home"] = m.get("home_team", "Home")
    if "guest" not in m:
        m["guest"] = m.get("away_team", "Away")

    # 欧赔字段
    if "sp_home" not in m:
        m["sp_home"] = m.get("win", m.get("home_win", m.get("odds_home", 0)))
    if "sp_draw" not in m:
        m["sp_draw"] = m.get("same", m.get("draw", m.get("odds_draw", 0)))
    if "sp_away" not in m:
        m["sp_away"] = m.get("lose", m.get("away_win", m.get("odds_away", 0)))

    if "win" not in m:
        m["win"] = m.get("sp_home", 0)
    if "same" not in m:
        m["same"] = m.get("sp_draw", 0)
    if "lose" not in m:
        m["lose"] = m.get("sp_away", 0)

    # 让球字段
    if "give_ball" not in m:
        m["give_ball"] = m.get("handicap", m.get("rq", "0"))

    # 变化字段
    change = m.get("change", {})
    if not isinstance(change, dict):
        change = {}

    for key, aliases in {
        "win": ["change_win", "cw", "home_change"],
        "same": ["change_same", "cs", "draw_change"],
        "lose": ["change_lose", "cl", "away_change"],
    }.items():
        if key not in change:
            for alias in aliases:
                if alias in m:
                    change[key] = m.get(alias)
                    break

    m["change"] = change

    # 散户字段
    vote = m.get("vote", {})
    if not isinstance(vote, dict):
        vote = {}

    for key, aliases in {
        "win": ["vote_win", "hot_home", "public_home"],
        "same": ["vote_same", "hot_draw", "public_draw"],
        "lose": ["vote_lose", "hot_away", "public_away"],
    }.items():
        if key not in vote:
            for alias in aliases:
                if alias in m:
                    vote[key] = m.get(alias)
                    break

    m["vote"] = vote

    return m


# ====================================================================
# 欧赔 / 让球 / 基本面解析
# ====================================================================

def _compute_no_vig_probs(match_obj: Dict) -> Dict[str, float]:
    """
    欧赔去水概率。
    注意：这不是严格 Shin 模型，不再命名为 Shin，避免误导 AI。
    """
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
    """
    从三项欧赔去水概率粗略反推理论让球。
    内部约定:
    + 表示主队让球
    - 表示客队让球 / 主队受让
    """
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

    # 平局概率越高，理论让球应越浅
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
    """
    解析实际让球值。
    内部约定:
    + 表示主队让球
    - 表示客队让球 / 主队受让

    常见竞彩字段:
    give_ball = -1 通常表示主队让1球，因此内部返回 +1
    give_ball = +1 通常表示主队受让1球，因此内部返回 -1
    """
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

    # 中文语义优先
    if "主受让" in s or "客让" in s:
        return -mag
    if "主让" in s or "客受让" in s:
        return +mag

    # 盘口斜杠: 0.5/1, -0.5/-1
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
# 资金 / Steam 方向识别，仅作为文本观察，不做决策
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
# 中性观察信号：不写结论、不强迫AI采纳
# ====================================================================

def build_observation_signals(
    match_obj: Dict,
    engine_result: Dict,
    smart_signals: List,
    exp_goals: float
) -> List[str]:
    """
    输出中性观察事实。
    不使用“必死、真强、诱主、反指”等强结论词。
    """
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

    # 赔率变动观察
    total_move = abs(cw) + abs(cs) + abs(cl)
    if total_move > 0:
        if cs < cw and cs < cl and cs <= -0.04:
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

    # 让球深度观察
    if sp_h > 1.05 and sp_d > 1.05 and sp_a > 1.05:
        theoretical = _infer_theoretical_handicap(sp_h, sp_d, sp_a)
        actual = _parse_actual_handicap(match_obj)
        diff = actual - theoretical

        if abs(diff) >= 0.5:
            facts.append(
                f"[OBS04 让球深度差异] 实际让球 {actual:+.2f}, 欧赔反推理论 {theoretical:+.2f}, "
                f"差异 {diff:+.2f} 球。"
            )

    # 欧赔去水概率
    max_dir = max(["home", "draw", "away"], key=lambda k: probs[k])
    if probs[max_dir] >= 45:
        facts.append(
            f"[OBS05 欧赔去水倾向] 主 {probs['home']:.1f}% / 平 {probs['draw']:.1f}% / "
            f"客 {probs['away']:.1f}%, 最高方向={max_dir}。"
        )

    # 散户热度
    if max(vh, vd, va) >= 58:
        hot_dir = "主胜" if vh >= max(vd, va) else ("平局" if vd >= max(vh, va) else "客胜")
        hot_val = max(vh, vd, va)
        facts.append(
            f"[OBS06 散户热度] {hot_dir}散户占比 {hot_val:.0f}%。"
        )

    # 总进球赔率压低
    low_ttg = []
    for g in range(8):
        odds = _f(match_obj.get(f"a{g}", 0))
        anchor = TTG_ANCHORS.get(g, {})
        hard_low = anchor.get("hard_low", 0)
        if odds > 1 and hard_low > 0 and odds <= hard_low:
            low_ttg.append(f"{g}球={odds:.2f}")

    if low_ttg:
        facts.append(
            f"[OBS07 总进球低赔点] {' | '.join(low_ttg)}。"
        )

    # 期望进球与赔率区间
    if exp_goals > 0:
        if exp_goals >= 2.8:
            facts.append(f"[OBS08 期望总进球偏高] λ={exp_goals:.2f}。")
        elif exp_goals <= 2.2:
            facts.append(f"[OBS09 期望总进球偏低] λ={exp_goals:.2f}。")

    # CRS 覆盖与低赔集中
    crs_items = []
    for sc, key in CRS_FULL_MAP.items():
        v = _f(match_obj.get(key, 0))
        if v > 1:
            crs_items.append((sc, key, v))

    if len(crs_items) < 8:
        facts.append(
            f"[OBS10 CRS数据不足] 当前可见CRS项 {len(crs_items)}/27。"
        )
    else:
        lowest = sorted(crs_items, key=lambda x: x[2])[:5]
        facts.append(
            "[OBS11 CRS低赔Top5] " + " | ".join([f"{sc}={v:.1f}" for sc, _, v in lowest]) + "。"
        )

    # 半全场低赔
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

    # xG与基本面场均背离
    info_src = match_obj.get("points", {}) or {}
    h_txt = str(info_src.get("home_strength", ""))
    a_txt = str(info_src.get("guest_strength", ""))

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

    # 杯赛属性
    league = str(match_obj.get("league", match_obj.get("cup", "")))
    is_cup = any(kw in league for kw in CUP_KEYWORDS)
    if is_cup:
        facts.append(f"[OBS14 杯赛属性] 联赛字段={league}, 需要考虑淘汰赛/轮换/保守性。")

    # Sharp / Steam 观察
    sharp_dir = _detect_sharp_direction(smart_signals)
    steam_dir, steam_type = _detect_steam_direction(smart_signals)

    if sharp_dir:
        facts.append(f"[OBS15 Sharp方向] smart_signals中检测到 Sharp 方向={sharp_dir}。")
    if steam_dir:
        facts.append(f"[OBS16 Steam方向] smart_signals中检测到 Steam 方向={steam_dir}, 类型={steam_type}。")

    # 原始 smart_signals 保留
    sigs_short = [str(s) for s in smart_signals[:8]]
    if sigs_short:
        facts.append("[OBS17 原始智能信号] " + " | ".join(sigs_short))

    return facts


# ====================================================================
# Ensemble 信号：只作为参考，不做决策
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
# 抓包格式化：Raw Packet First
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

    # 欧赔
    block += "\n【1. 欧赔原始数据】\n"
    block += f"即时欧赔: 主胜 {sp_h:.2f} / 平局 {sp_d:.2f} / 客胜 {sp_a:.2f}\n"
    block += (
        f"欧赔去水概率: 主 {probs['home']:.1f}% / 平 {probs['draw']:.1f}% / "
        f"客 {probs['away']:.1f}% / 返还率约 {100 / probs['margin']:.1f}%\n"
        if probs.get("margin", 0) > 0
        else "欧赔去水概率: 数据不足\n"
    )

    # 让球
    block += "\n【2. 让球/盘口】\n"
    block += f"原始 give_ball/handicap: {hc_raw}\n"
    block += f"标准化实际让球: {actual_hc:+.2f}，内部约定：正数=主让，负数=客让/主受让\n"
    block += f"欧赔反推理论让球: {theoretical_hc:+.2f}\n"
    block += f"实际 - 理论 差异: {actual_hc - theoretical_hc:+.2f} 球\n"

    # 赔率变动
    change = match.get("change", {}) or {}
    cw = _f(change.get("win", 0))
    cs = _f(change.get("same", 0))
    cl = _f(change.get("lose", 0))

    block += "\n【3. 赔率变动】\n"
    block += f"主胜变化 {cw:+.2f} / 平赔变化 {cs:+.2f} / 客胜变化 {cl:+.2f}\n"
    block += "说明: 负数=降水/赔率下调，正数=升水/赔率上调。\n"

    # 庄家隐含 xG
    hxg = engine_result.get("bookmaker_implied_home_xg", None)
    axg = engine_result.get("bookmaker_implied_away_xg", None)
    exp_total = _f(engine_result.get("expected_total_goals", 0))
    if exp_total <= 0:
        exp_total = _f(hxg, 0) + _f(axg, 0)

    block += "\n【4. 庄家隐含 xG / 期望进球】\n"
    block += f"主 xG: {hxg if hxg is not None else 'N/A'} / 客 xG: {axg if axg is not None else 'N/A'}\n"
    block += f"期望总进球: {exp_total:.2f}\n"
    block += f"大2.5概率: {engine_result.get('over_25', 'N/A')} / 双方进球BTTS: {engine_result.get('btts', 'N/A')}\n"

    # 总进球赔率
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
            ttg_lines.append(f"{g}球={v:.2f}{mark}")
        else:
            ttg_lines.append(f"{g}球=N/A")
    block += " | ".join(ttg_lines) + "\n"

    # CRS 比分赔率
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

    if crs_h:
        block += "主胜系: " + " | ".join(crs_h) + "\n"
    else:
        block += "主胜系: N/A\n"

    if crs_d:
        block += "平局系: " + " | ".join(crs_d) + "\n"
    else:
        block += "平局系: N/A\n"

    if crs_a:
        block += "客胜系: " + " | ".join(crs_a) + "\n"
    else:
        block += "客胜系: N/A\n"

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

    # CRS 低赔提示，只是事实
    crs_items = []
    for sc, key in CRS_FULL_MAP.items():
        v = _f(match.get(key, 0))
        if v > 1:
            crs_items.append((sc, v))

    if crs_items:
        low = sorted(crs_items, key=lambda x: x[1])[:6]
        block += "CRS低赔Top6: " + " | ".join([f"{sc}={v:.1f}" for sc, v in low]) + "\n"

    # 半全场
    hf_lines = []
    for k, label in HFTF_MAP.items():
        v = _f(match.get(k, 0))
        if v > 1:
            hf_lines.append(f"{label}={v:.2f}")

    if hf_lines:
        block += "\n【7. 半全场赔率】\n"
        block += " | ".join(hf_lines) + "\n"

    # 散户分布
    vote = match.get("vote", {}) or {}
    if vote:
        vh = vote.get("win", "?")
        vd = vote.get("same", "?")
        va = vote.get("lose", "?")
        block += "\n【8. 散户分布】\n"
        block += f"主胜 {vh}% / 平局 {vd}% / 客胜 {va}%\n"

    # 资金/智能信号
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

    # 基本面
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

    # 异动消息
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

    # 模型矩阵，仅参考
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

    # 中性观察信号
    if observation_signals:
        block += "\n【13. 系统观察信号，仅供参考，可采纳也可否决】\n"
        for fact in observation_signals:
            block += f"- {fact}\n"
    else:
        block += "\n【13. 系统观察信号】\n无明显观察信号。\n"

    return block


# ====================================================================
# Prompt 构建：单 AI 自主分析
# ====================================================================

def build_raw_ai_prompt(match_blocks: List[str]) -> str:
    p = ""

    p += "<context>\n"
    p += "你是竞彩足球/比分预测分析师。下面是多场比赛的完整抓包数据。你的任务是基于原始抓包自主分析，不要机械服从系统观察信号。\n"
    p += "系统观察信号只是事实摘要，可能有效，也可能是噪声。你需要自己判断哪些信号采纳，哪些信号否决。\n"
    p += "</context>\n\n"

    p += "<core_rules>\n"
    p += "1. 原始抓包数据优先，包括欧赔、让球、总进球、CRS比分赔率、半全场、散户、资金、基本面、伤停。\n"
    p += "2. 系统观察信号不能直接当结论，只能当待验证线索。\n"
    p += "3. 不要因为出现“平赔独降、让球深度差异、Sharp、Steam”等信号就固定反打或固定跟随。\n"
    p += "4. 必须同时给出支持自己结论的证据和反对自己结论的疑点。\n"
    p += "5. predicted_score 的方向必须与 predicted_direction 一致。\n"
    p += "6. top3[0] 必须与 predicted_score 一致。\n"
    p += "7. 比分判断要结合总进球赔率 a0~a7、CRS低赔结构、xG、双方进攻防守和盘口深度。\n"
    p += "8. 如果 CRS 数据不足，要降低精确比分信心，不要凭空大幅提高 confidence。\n"
    p += "9. 如果信号互相冲突，优先输出更保守的比分，并在 doubts 中说明冲突点。\n"
    p += "</core_rules>\n\n"

    p += "<analysis_steps>\n"
    p += "每一场请按以下顺序思考，但最终只输出 JSON：\n"
    p += "A. 检查数据完整性：欧赔/让球/总进球/CRS/半全场/基本面是否缺失。\n"
    p += "B. 判断主方向：home/draw/away。\n"
    p += "C. 判断进球区间：0-1球、2球、3球、4球、5球、6+球。\n"
    p += "D. 判断最可能比分 top3。\n"
    p += "E. 判断系统观察信号中哪些有效、哪些无效。\n"
    p += "F. 写出反证 doubts。\n"
    p += "</analysis_steps>\n\n"

    p += "<match_data>\n"
    for block in match_blocks:
        p += block
    p += "\n</match_data>\n\n"

    p += "<output_format>\n"
    p += "严格输出 JSON 数组。不要 markdown，不要代码块，不要前缀后缀。\n"
    p += "数组内每场一个对象，字段如下：\n"
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
    p += "      {\"score\": \"2-1\", \"prob\": 17},\n"
    p += "      {\"score\": \"1-0\", \"prob\": 14},\n"
    p += "      {\"score\": \"1-1\", \"prob\": 13}\n"
    p += "    ],\n"
    p += "    \"accepted_observations\": [\"让球深度与主胜方向一致\", \"CRS主胜小比分低赔集中\"],\n"
    p += "    \"rejected_observations\": [\"平赔下降不足以单独支持平局\"],\n"
    p += "    \"doubts\": [\"xG差距不大\", \"CRS平局项仍有保护\"],\n"
    p += "    \"data_quality\": {\n"
    p += "      \"odds_complete\": true,\n"
    p += "      \"crs_complete\": true,\n"
    p += "      \"ttg_complete\": true,\n"
    p += "      \"notes\": []\n"
    p += "    },\n"
    p += "    \"reason\": \"300-600字中文推理，说明为什么选这个方向和比分，以及为什么否决其他方向。\"\n"
    p += "  }\n"
    p += "]\n"
    p += "</output_format>\n"

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

DEFAULT_MAIN_MODELS = [
    # 优先使用你现有 Claude 通道；没有 key 会自动跳过
    {
        "ai_name": "claude",
        "url_env": "CLAUDE_API_URL",
        "key_env": "CLAUDE_API_KEY",
        "models": ["熊猫-69-满血openrouter-claude-opus-4.7-上下文1000k"],
        "temperature": 0.22,
    },
    # 其次 GPT
    {
        "ai_name": "gpt",
        "url_env": "GPT_API_URL",
        "key_env": "GPT_API_KEY",
        "models": ["gpt-5.5"],
        "temperature": 0.20,
    },
    # 再其次 Grok
    {
        "ai_name": "grok",
        "url_env": "GROK_API_URL",
        "key_env": "GROK_API_KEY",
        "models": ["熊猫-A-5-grok-4.2-fast-200w上下文"],
        "temperature": 0.28,
    },
    # 最后 Gemini
    {
        "ai_name": "gemini",
        "url_env": "GEMINI_API_URL",
        "key_env": "GEMINI_API_KEY",
        "models": ["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking"],
        "temperature": 0.20,
    },
]


def get_clean_env_url(name, default=""):
    v = str(os.environ.get(name, globals().get(name, default))).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/?#=&%-]+)", v)
    return match.group(1) if match else v


def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")


def _build_urls_for_ai(ai_name: str, url_env: str) -> List[str]:
    if ai_name == "gpt":
        primary_url = get_clean_env_url(url_env, GPT_DEFAULT_URL)
        if not primary_url:
            primary_url = GPT_DEFAULT_URL
        return [primary_url]

    primary_url = get_clean_env_url(url_env)
    backup = [u for u in FALLBACK_URLS if u and u != primary_url][:1]
    return [primary_url] + backup


async def async_call_ai_batch(
    session,
    prompt: str,
    url_env: str,
    key_env: str,
    models_list: List[str],
    num_matches: int,
    ai_name: str,
    sys_prompt: str,
    temperature: float
) -> Tuple[str, Dict[int, Dict], str]:
    """
    调用一个 AI，让它一次性批量分析所有比赛。
    """
    key = get_clean_env_key(key_env)

    if not key and ai_name == "gpt":
        key = GPT_DEFAULT_KEY

    if not key:
        return ai_name, {}, "no_key"

    urls = _build_urls_for_ai(ai_name, url_env)

    connect_timeout = 20
    read_timeout_map = {
        "claude": 420,
        "grok": 320,
        "gpt": 320,
        "gemini": 320,
    }
    read_timeout = read_timeout_map.get(ai_name, 260)

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
                    connect=connect_timeout,
                    sock_connect=connect_timeout,
                    sock_read=read_timeout,
                )

                async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                    elapsed = round(time.time() - t0, 1)

                    if r.status in (502, 504):
                        print(f"    💀 HTTP {r.status} | {elapsed}s → 换URL")
                        continue

                    if r.status == 400:
                        text = await r.text()
                        print(f"    💀 HTTP 400 | {elapsed}s → 换模型 | {text[:120]}")
                        break

                    if r.status == 401:
                        print(f"    💀 HTTP 401 | key无效 → 换通道")
                        break

                    if r.status == 429:
                        print(f"    ⚠️ HTTP 429 | 限流，稍后重试")
                        await asyncio.sleep(1.2)
                        continue

                    if r.status != 200:
                        text = await r.text()
                        print(f"    ⚠️ HTTP {r.status} | {elapsed}s | {text[:120]}")
                        continue

                    connected = True
                    data = await r.json(content_type=None)
                    raw_text = _extract_response_text(data, is_gem)

                    if not raw_text or len(raw_text) < 10:
                        print(f"    ⚠️ 空响应 → 换模型")
                        break

                    results = _parse_ai_json(raw_text, num_matches, phase="main")

                    elapsed = round(time.time() - t0, 1)

                    if results:
                        print(f"    ✅ {ai_name.upper()} 完成: {len(results)}/{num_matches} | {elapsed}s")
                        return ai_name, results, mn

                    print(f"    ⚠️ JSON解析0条 → 换模型")
                    break

            except aiohttp.ClientConnectorError:
                continue
            except asyncio.TimeoutError:
                if not connected:
                    continue
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

                for ci in range(start_pos, min(start_pos + 300000, len(full_str))):
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
# AI JSON 解析与校验
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


def _parse_ai_json(raw_text: str, num_matches: int, phase: str = "main") -> Dict[int, Dict]:
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


def validate_ai_item(item: Dict, phase: str = "main") -> Tuple[bool, Dict, List[str]]:
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

    # 概率规范化
    hp, dp, ap = _pct_normalize(
        out.get("home_win_pct", 33.3),
        out.get("draw_pct", 33.3),
        out.get("away_win_pct", 33.4),
    )

    out["home_win_pct"] = hp
    out["draw_pct"] = dp
    out["away_win_pct"] = ap

    # 置信度
    conf = _i(out.get("confidence", 55), 55)
    conf = max(25, min(95, conf))
    out["confidence"] = conf

    # 进球区间
    goal_range = str(out.get("goal_range", "")).strip()
    if not goal_range:
        goal_range = _goal_range_from_score(score_label)
    out["goal_range"] = goal_range

    # top3
    top3 = out.get("top3", [])
    if not isinstance(top3, list):
        top3 = []

    fixed_top3 = []

    # 强制 top3[0] = predicted_score
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

    # predicted_score 必须在第一位
    fixed_top3 = sorted(
        fixed_top3,
        key=lambda x: 0 if x.get("score") == score_label else 1
    )[:3]

    out["top3"] = fixed_top3

    # 列表字段兜底
    for k in ["accepted_observations", "rejected_observations", "doubts"]:
        if not isinstance(out.get(k), list):
            out[k] = []

    if not isinstance(out.get("data_quality"), dict):
        out["data_quality"] = {
            "odds_complete": None,
            "crs_complete": None,
            "ttg_complete": None,
            "notes": [],
        }

    out["reason"] = _safe_str(out.get("reason", ""), 1000)

    return len(errors) == 0, out, errors


# ====================================================================
# 主 AI 分析
# ====================================================================

async def run_main_ai_analysis(match_blocks: List[str], num_matches: int) -> Tuple[Dict[int, Dict], str]:
    print(f"\n  [AI] Raw Packet 单AI自主分析 ({num_matches} 场)...")

    prompt = build_raw_ai_prompt(match_blocks)

    print(f"  [Prompt] {len(prompt):,} 字符")

    sys_prompt = (
        "<role>你是竞彩足球原始抓包分析师。你需要根据完整抓包自主判断方向、比分和风险。</role>\n"
        "<rules>系统观察信号只是参考，不是结论。不要机械跟随任何单一信号。严格输出JSON数组。</rules>"
    )

    connector = aiohttp.TCPConnector(limit=5, use_dns_cache=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        for cfg in DEFAULT_MAIN_MODELS:
            ai_name, results, model_name = await async_call_ai_batch(
                session=session,
                prompt=prompt,
                url_env=cfg["url_env"],
                key_env=cfg["key_env"],
                models_list=cfg["models"],
                num_matches=num_matches,
                ai_name=cfg["ai_name"],
                sys_prompt=sys_prompt,
                temperature=cfg["temperature"],
            )

            if results:
                print(f"  [AI完成] 使用 {ai_name.upper()} / {model_name}，返回 {len(results)}/{num_matches}")
                return results, f"{ai_name}:{model_name}"

    print("  [AI失败] 所有主AI通道均失败")
    return {}, "all_failed"


# ====================================================================
# 方向一致性与输出包装
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

    out["predicted_score"] = score_label
    out["predicted_label"] = score_label
    out["predicted_direction"] = expected_dir
    out["final_direction"] = expected_dir
    out["result"] = result_cn
    out["display_direction"] = result_cn
    out["is_score_others"] = is_others

    goal_range = out.get("goal_range") or _goal_range_from_score(score_label)
    out["goal_range"] = goal_range
    out["goal_interval"] = goal_range
    out["predicted_goal_range"] = goal_range

    conf = _i(out.get("confidence", 55), 55)
    conf = max(25, min(95, conf))

    out["confidence"] = conf
    out["ai_confidence"] = conf
    out["ai_confidence_pct"] = conf
    out["confidence_score"] = conf
    out["dir_confidence"] = conf

    hp, dp, ap = _pct_normalize(
        out.get("home_win_pct", 33.3),
        out.get("draw_pct", 33.3),
        out.get("away_win_pct", 33.4),
    )

    out["home_win_pct"] = hp
    out["draw_pct"] = dp
    out["away_win_pct"] = ap

    return out


def _extract_score_prob_from_ai(ai_result: Dict, score: str) -> float:
    """
    只提取精确比分概率。
    禁止用方向概率替代比分概率计算 EV。
    """
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
    ai_result: Dict,
    observation_signals: List[str],
    ensemble_signals: Dict,
    idx: int,
    ai_provider: str
) -> Dict:
    cr = _enforce_direction_consistency(ai_result or {})

    predicted_score = cr.get("predicted_score", "1-1")
    predicted_label = cr.get("predicted_label", predicted_score)
    final_direction = cr.get("final_direction", "draw")
    result_cn = cr.get("result", "平局")
    is_others = bool(cr.get("is_score_others", False))

    home_pct = _f(cr.get("home_win_pct", 33.3), 33.3)
    draw_pct = _f(cr.get("draw_pct", 33.3), 33.3)
    away_pct = _f(cr.get("away_win_pct", 33.4), 33.4)

    confidence = _i(cr.get("confidence", 55), 55)
    confidence = max(25, min(95, confidence))

    risk = "低" if confidence >= 75 else ("中" if confidence >= 55 else "高")

    final_odds = _get_score_odds(match, predicted_score, final_direction, is_others)
    score_prob = _extract_score_prob_from_ai(cr, predicted_score)
    ev_pct, kelly_pct, is_value = _calculate_score_ev(score_prob, final_odds)

    # 没有精确比分概率时，不计算价值注
    value_reason = ""
    if final_odds > 1.05 and score_prob <= 0:
        value_reason = "缺少精确比分概率，禁止用方向概率计算比分EV"

    smart_signals = stats.get("smart_signals", []) if stats else []
    sigs = list(smart_signals)
    sigs.extend(observation_signals)

    accepted_obs = cr.get("accepted_observations", [])
    rejected_obs = cr.get("rejected_observations", [])
    doubts = cr.get("doubts", [])

    goal_range = cr.get("goal_range") or _goal_range_from_score(predicted_score)

    hp_list = [home_pct, draw_pct, away_pct]
    hp_sorted = sorted(hp_list)

    expected_total_goals = _f(engine_result.get("expected_total_goals", 0))
    if expected_total_goals <= 0:
        expected_total_goals = (
            _f(engine_result.get("bookmaker_implied_home_xg", 1.3)) +
            _f(engine_result.get("bookmaker_implied_away_xg", 0.9))
        )

    return {
        # 核心预测
        "predicted_score": predicted_score,
        "predicted_label": predicted_label,
        "predicted_direction": final_direction,
        "result": result_cn,
        "display_direction": result_cn,
        "final_direction": final_direction,
        "is_score_others": is_others,

        # UI兼容字段
        "decision_title": "vMAX 19.1 决策剖析",
        "decision_engine_version": ENGINE_VERSION,
        "decision_architecture": ENGINE_ARCHITECTURE,

        # 进球区间
        "goal_range": goal_range,
        "goal_interval": goal_range,
        "predicted_goal_range": goal_range,

        # 概率
        "home_win_pct": home_pct,
        "draw_pct": draw_pct,
        "away_win_pct": away_pct,

        # 置信度
        "confidence": confidence,
        "ai_confidence": confidence,
        "ai_confidence_pct": confidence,
        "confidence_score": confidence,
        "dir_confidence": confidence,
        "risk_level": risk,
        "dir_gap": round(hp_sorted[-1] - hp_sorted[-2], 1) if len(hp_sorted) >= 2 else 0,

        # AI分析
        "ai_provider": ai_provider,
        "main_ai_score": predicted_score,
        "main_ai_analysis": cr.get("reason", "")[:1000],
        "claude_score": predicted_score if "claude" in ai_provider else "",
        "claude_analysis": cr.get("reason", "")[:1000] if "claude" in ai_provider else "",
        "gpt_score": predicted_score if "gpt" in ai_provider else "",
        "gpt_analysis": cr.get("reason", "")[:800] if "gpt" in ai_provider else "",
        "grok_score": predicted_score if "grok" in ai_provider else "",
        "grok_analysis": cr.get("reason", "")[:800] if "grok" in ai_provider else "",
        "gemini_score": predicted_score if "gemini" in ai_provider else "",
        "gemini_analysis": cr.get("reason", "")[:800] if "gemini" in ai_provider else "",

        "arbitration_reason": cr.get("reason", ""),
        "agreement_pattern": "单AI自主分析",
        "alternative_score": cr.get("alternative", {}),
        "top3": cr.get("top3", []),

        # 观察信号
        "accepted_observations": accepted_obs,
        "rejected_observations": rejected_obs,
        "doubts": doubts,
        "data_quality": cr.get("data_quality", {}),
        "ai_validation_errors": cr.get("ai_validation_errors", []),

        # EV / Kelly：只用精确比分概率
        "score_odds": final_odds,
        "score_prob": round(score_prob * 100, 2),
        "suggested_kelly": kelly_pct,
        "edge_vs_market": ev_pct,
        "is_value": is_value,
        "value_reason": value_reason,

        # 中性观察信号
        "traps_detected": [],
        "trap_count": len(observation_signals),
        "trap_facts": observation_signals,
        "observation_signals": observation_signals,

        # 市场信号
        "smart_money_signal": " | ".join([str(s) for s in sigs[:10]]),
        "smart_signals": sigs,
        "sharp_detected": _detect_sharp_direction(smart_signals) is not None,
        "sharp_dir": _detect_sharp_direction(smart_signals),

        # xG / 进球
        "xG_home": round(_f(engine_result.get("bookmaker_implied_home_xg", 1.3)), 2),
        "xG_away": round(_f(engine_result.get("bookmaker_implied_away_xg", 0.9)), 2),
        "expected_total_goals": round(expected_total_goals, 2),
        "over_2_5": engine_result.get("over_25", 50) if engine_result else 50,
        "btts": engine_result.get("btts", 45) if engine_result else 45,
        "bookmaker_implied_home_xg": engine_result.get("bookmaker_implied_home_xg", "?"),
        "bookmaker_implied_away_xg": engine_result.get("bookmaker_implied_away_xg", "?"),

        # 模型参考
        "model_consensus_dir": ensemble_signals.get("consensus", ""),
        "model_consensus_count": ensemble_signals.get("consensus_count", 0),
        "total_models": ensemble_signals.get("total", 0),
        "ensemble_reference": ensemble_signals,

        # 老字段兼容
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

        "engine_version": ENGINE_VERSION,
        "engine_architecture": ENGINE_ARCHITECTURE,
    }


# ====================================================================
# Top4 推荐
# ====================================================================

def select_top4(preds):
    """
    推荐排序：
    1. 置信度
    2. 数据质量
    3. 精确比分概率
    4. 风险较低
    """
    def _score(x):
        p = x.get("prediction", {}) or {}
        confidence = _f(p.get("confidence", 0))
        score_prob = _f(p.get("score_prob", 0))
        risk_penalty = 0

        if p.get("risk_level") == "高":
            risk_penalty += 8
        if p.get("value_reason"):
            risk_penalty += 3
        if p.get("ai_validation_errors"):
            risk_penalty += 5

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


# ====================================================================
# 主入口
# ====================================================================

def run_predictions(raw, use_ai=True):
    raw_matches = raw.get("matches", [])
    num = len(raw_matches)

    print("\n" + "=" * 80)
    print(f"  [{ENGINE_VERSION}] {ENGINE_ARCHITECTURE} | {num} 场")
    print("=" * 80)

    match_analyses = []
    match_blocks = []

    # ------------------------------------------------------------
    # 预处理：只做数据准备，不做主观决策
    # ------------------------------------------------------------
    for i, raw_m in enumerate(raw_matches):
        m = normalize_match(raw_m)

        # 基础引擎结果，仅用于 xG / over / btts 等辅助字段
        try:
            if predict_match:
                eng = predict_match(m)
            else:
                eng = {}
        except Exception as e:
            logger.warning(f"predict_match 失败: {e}")
            eng = {}

        # ensemble 只作为参考，不参与后处理决策
        try:
            sp = ensemble.predict(m, {}) if ensemble else {}
        except Exception as e:
            logger.warning(f"ensemble.predict 失败: {e}")
            sp = {}

        # 经验分析只作为附加参考
        try:
            exp_result = exp_engine.analyze(m) if exp_engine else {}
        except Exception as e:
            logger.warning(f"exp_engine.analyze 失败: {e}")
            exp_result = {}

        # 期望进球
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

    # ------------------------------------------------------------
    # 单 AI 自主分析
    # ------------------------------------------------------------
    ai_results = {}
    ai_provider = "no_ai"

    if use_ai and match_blocks:
        async def _run_ai():
            return await run_main_ai_analysis(match_blocks, num)

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
                future = pool.submit(_run_in_thread, _run_ai())
                try:
                    ai_results, ai_provider = future.result()
                except Exception as e:
                    logger.error(f"AI 执行崩溃: {e}")
                    ai_results = {}
                    ai_provider = "ai_crashed"
        else:
            try:
                ai_results, ai_provider = asyncio.run(_run_ai())
            except Exception as e:
                logger.error(f"AI 执行崩溃: {e}")
                ai_results = {}
                ai_provider = "ai_crashed"

    # ------------------------------------------------------------
    # 整合最终预测
    # ------------------------------------------------------------
    res = []

    for i, ma in enumerate(match_analyses):
        idx = i + 1
        raw_m = ma["raw_match"]
        m = ma["match"]

        cr = ai_results.get(idx, {})

        if not cr:
            cr = {
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
                "accepted_observations": [],
                "rejected_observations": [],
                "doubts": ["AI未返回有效结果，使用保守兜底"],
                "data_quality": {
                    "odds_complete": None,
                    "crs_complete": None,
                    "ttg_complete": None,
                    "notes": ["AI failed"],
                },
                "reason": "AI未返回有效结果，兜底输出1-1。此结果不可作为强判断。",
            }

        mg = assemble_final_prediction(
            match=m,
            engine_result=ma["engine"],
            stats=ma["stats"],
            ai_result=cr,
            observation_signals=ma["observation_signals"],
            ensemble_signals=ma["ensemble_signals"],
            idx=idx,
            ai_provider=ai_provider,
        )

        # 旧增强模块默认关闭；打开后也只允许追加非核心字段
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

        # 最后只做字段一致性校验，不改变AI比分方向逻辑
        mg = _enforce_direction_consistency(mg)

        combined = {**raw_m, **m, "prediction": mg}
        res.append(combined)

        obs_tag = f" [OBS{len(ma['observation_signals'])}]" if ma.get("observation_signals") else ""
        val_tag = " [EV禁算]" if mg.get("value_reason") else ""
        err_tag = f" [校验{len(mg.get('ai_validation_errors', []))}]" if mg.get("ai_validation_errors") else ""

        print(
            f"  [{idx}] {m.get('home_team', m.get('home', '?'))} vs "
            f"{m.get('away_team', m.get('guest', '?'))} => "
            f"{mg['result']} ({mg['predicted_score']}) | "
            f"CF: {mg['confidence']}% | AI={ai_provider}{obs_tag}{val_tag}{err_tag}"
        )

    # Top4 推荐
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
    print("   引擎职责: 抓包标准化 + 原始字段展开 + 中性观察信号 + JSON校验")
    print("   AI职责: 自主分析，不按三家投票，不由Claude当多数派裁判")
    print("   注意: EV/Kelly 只使用精确比分概率，禁止用方向概率代替比分概率")