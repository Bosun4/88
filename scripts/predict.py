import json
import os
import re
import time
import asyncio
import aiohttp
import logging
import math
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# ====================================================================
# 🚀 vMAX 18.2 — 贝叶斯后验 + 16维陷阱矩阵 + 防幻觉AI + 决策锁定链
# --------------------------------------------------------------------
# 核心约束:
#   1. 不拆批：一次 prompt 覆盖全部比赛
#   2. 不切换模型：每个 AI 通道只调用指定模型一次
#   3. 不因解析失败重试模型：避免重复消耗 token
#   4. 强化 JSON 解析：content=null / proxy bug / markdown / 字符串 top3 全兜底
#   5. 四字段强一致：predicted_score=result=display_direction=概率argmax
# ====================================================================

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
    exp_engine = ExperienceEngine()
except Exception:
    ensemble = None
    exp_engine = None


# ====================================================================
# 通用工具 & 常量
# ====================================================================

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


def _f(v, default=0.0):
    """安全 float 转换"""
    try:
        return float(v) if v is not None and str(v).strip() != "" else default
    except Exception:
        return default


def calculate_value_bet(prob_pct, odds):
    """Kelly + EV 计算"""
    if not odds or odds <= 1.05:
        return {"ev": 0.0, "kelly": 0.0, "is_value": False}

    prob = prob_pct / 100.0
    ev = (prob * odds) - 1.0
    b = odds - 1.0
    q = 1.0 - prob

    if b <= 0:
        return {
            "ev": round(ev * 100, 2),
            "kelly": 0.0,
            "is_value": False
        }

    kelly = ((b * prob) - q) / b
    return {
        "ev": round(ev * 100, 2),
        "kelly": round(max(0.0, kelly * 0.5) * 100, 2),
        "is_value": ev > 0.05
    }


# ============================================================================
# 🎭 16维庄家陷阱识别引擎
# ============================================================================

def _extract_form_record(text: str) -> Tuple[int, int, int]:
    """从文本里抽取 '近5主场3胜1平1负' 这种信息,返回 (胜, 平, 负)"""
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
    """抽取 '场均进球X.X 场均失球X.X'"""
    if not text:
        return 0.0, 0.0

    h_match = re.search(r"场均进球[^0-9]*(\d+\.?\d*)", str(text))
    a_match = re.search(r"场均失球[^0-9]*(\d+\.?\d*)", str(text))

    return (
        float(h_match.group(1)) if h_match else 0.0,
        float(a_match.group(1)) if a_match else 0.0,
    )


def _fundamental_strength(match_obj: Dict, side: str) -> Dict[str, Any]:
    """基本面综合强度评估 (-100 ~ +100)"""
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
    """
    从欧赔反推理论亚盘让球深度
    返回: 理论让球(>0=主让,<0=客让)
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
    if ratio >= 1.15:
        return 0.0
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
    """
    解析实际让球值,约定:主让为正,客让为负

    竞彩格式:
      give_ball = "-1" 表示主队让1球 → 内部返回 +1.0
      give_ball = "+1" 表示主队受让1球 → 内部返回 -1.0
    """
    raw = match_obj.get("give_ball", match_obj.get("handicap", "0"))
    s = str(raw).strip()

    s = s.replace("主", "").replace("客", "").replace("受让", "+").replace("让", "-")

    if "/" in s:
        parts = s.split("/")
        try:
            val = (_f(parts[0].strip()) + _f(parts[1].strip())) / 2.0
            return -val
        except Exception:
            pass

    val = _f(s, 0.0)
    return -val


# ==========================================================
# T1: 诱平赔陷阱
# ==========================================================
def detect_T1_draw_trap(match_obj: Dict, engine_result: Dict,
                        smart_signals: List, shin: Dict) -> Optional[Dict]:
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

    shin_h, shin_a = shin["home"], shin["away"]
    strong_shin = max(shin_h, shin_a)

    if strong_shin < 34:
        return None

    strong_side = "home" if shin_h > shin_a else "away"
    strong_cn = "主" if strong_side == "home" else "客"
    weak_side = "away" if strong_side == "home" else "home"

    if strong_shin >= 42:
        evidence_score += 2
    elif strong_shin >= 38:
        evidence_score += 1

    evidence_detail.append(f"{strong_cn}Shin{strong_shin:.1f}%")

    strong_fund = _fundamental_strength(match_obj, strong_side)
    weak_fund = _fundamental_strength(match_obj, weak_side)

    if strong_fund["total"] >= 3:
        if strong_fund["win_rate"] >= 0.55 or strong_fund["strength_score"] >= 15:
            evidence_score += 2
            evidence_detail.append(f"{strong_cn}基本面强({strong_fund['win_rate']:.2f})")
        elif strong_fund["win_rate"] >= 0.45:
            evidence_score += 1
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

        if (strong_side == "home" and hc_diff >= 0.3) or \
           (strong_side == "away" and hc_diff <= -0.3):
            evidence_score += 1
            evidence_detail.append(f"让球偏深{hc_diff:+.2f}")

    if evidence_score < 4:
        return None

    severity = 3 if evidence_score < 6 else 4

    return {
        "trap": "T1_DRAW_TRAP",
        "description": f"诱平赔陷阱(得分{evidence_score}):{' + '.join(evidence_detail)}",
        "severity": severity,
        "direction_adjust": {
            strong_side: +2.5,
            "draw": -3.0,
            weak_side: -1.2
        },
        "score_multipliers": {
            "1-1": 0.2,
            "2-2": 0.25,
            "0-0": 0.3
        },
        "suppress_draw_sharp": True,
    }


# ==========================================================
# T2 & T3: 让球深度陷阱
# ==========================================================
def detect_T2_T3_handicap_trap(match_obj: Dict, shin: Dict) -> Optional[Dict]:
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
            "description": f"让球偏深:理论{theoretical:.2f} vs 实际{actual:.2f},差{diff:+.2f}球 (主队真强)",
            "severity": severity,
            "direction_adjust": {
                "home": +1.2 * min(2.0, abs(diff)),
                "away": -0.5,
                "draw": -0.4
            },
            "score_multipliers": {},
            "boost_scores": ["2-0", "3-0", "2-1", "3-1"] if abs(diff) >= 1.0 else ["2-1", "2-0"],
        }

    severity = 2 if abs(diff) < 1.0 else 3
    return {
        "trap": "T3_HANDICAP_SHALLOWER",
        "description": f"让球偏浅:理论{theoretical:.2f} vs 实际{actual:.2f},差{diff:+.2f}球 (主队真弱)",
        "severity": severity,
        "direction_adjust": {
            "home": -1.0 * min(2.0, abs(diff)),
            "away": +1.2 * min(2.0, abs(diff)),
            "draw": +0.5
        },
        "score_multipliers": {},
        "boost_scores": ["0-1", "1-2", "0-2", "1-1"] if abs(diff) >= 1.0 else ["0-1", "1-1"],
    }


# ==========================================================
# T4 & T5: 虚假强势陷阱
# ==========================================================
def detect_T4_T5_fake_favorite(match_obj: Dict, engine_result: Dict,
                               shin: Dict) -> Optional[Dict]:
    shin_h, shin_a = shin["home"], shin["away"]

    if shin_h > 48:
        fund_h = _fundamental_strength(match_obj, "home")
        fund_a = _fundamental_strength(match_obj, "away")

        if fund_h["total"] >= 3 and fund_a["total"] >= 3:
            if fund_h["strength_score"] < -5 and fund_a["strength_score"] > 15:
                return {
                    "trap": "T4_FAKE_HOME_FAVORITE",
                    "description": f"诱主胜:Shin主{shin_h:.1f}%但主队基本面{fund_h['strength_score']}分 vs 客{fund_a['strength_score']}分",
                    "severity": 3,
                    "direction_adjust": {
                        "home": -2.5,
                        "away": +2.0,
                        "draw": +0.5
                    },
                    "score_multipliers": {
                        "1-0": 0.4,
                        "2-0": 0.3,
                        "2-1": 0.5
                    },
                }

    if shin_a > 48:
        fund_h = _fundamental_strength(match_obj, "home")
        fund_a = _fundamental_strength(match_obj, "away")

        if fund_h["total"] >= 3 and fund_a["total"] >= 3:
            if fund_a["strength_score"] < -5 and fund_h["strength_score"] > 15:
                return {
                    "trap": "T5_FAKE_AWAY_FAVORITE",
                    "description": f"诱客胜:Shin客{shin_a:.1f}%但客队基本面{fund_a['strength_score']}分 vs 主{fund_h['strength_score']}分",
                    "severity": 3,
                    "direction_adjust": {
                        "away": -2.5,
                        "home": +2.0,
                        "draw": +0.5
                    },
                    "score_multipliers": {
                        "0-1": 0.4,
                        "0-2": 0.3,
                        "1-2": 0.5
                    },
                }

    return None


# ==========================================================
# T6 & T7: 比分区间陷阱
# ==========================================================
def detect_T6_T7_score_range_trap(match_obj: Dict, engine_result: Dict,
                                  exp_goals: float) -> Optional[Dict]:
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
            "description": f"诱小比分陷阱:a0/1/2赔率压低{low_small}项 但 λ={exp_goals:.2f}>=2.8",
            "severity": 2,
            "direction_adjust": {},
            "score_multipliers": {
                "0-0": 0.3,
                "1-0": 0.5,
                "0-1": 0.5,
                "1-1": 0.6
            },
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
            "description": f"诱大比分陷阱:a5/6/7压低{low_large}项 但 λ={exp_goals:.2f}<=2.3",
            "severity": 2,
            "direction_adjust": {},
            "score_multipliers": {
                "3-2": 0.4,
                "4-2": 0.3,
                "3-3": 0.3
            },
            "boost_scores": ["1-0", "0-1", "1-1", "2-1", "1-2"],
        }

    return None


# ==========================================================
# T8: 假冷门陷阱
# ==========================================================
def detect_T8_false_cold(match_obj: Dict, smart_signals: List,
                         shin: Dict) -> Optional[Dict]:
    sigs_str = " ".join(str(s) for s in smart_signals)

    cold_triggers = 0
    if "坏消息" in sigs_str:
        cold_triggers += 1
    if "崩盘" in sigs_str:
        cold_triggers += 1
    if "造热" in sigs_str:
        cold_triggers += 1
    if "背离" in sigs_str:
        cold_triggers += 1
    if "盘口太便宜" in sigs_str:
        cold_triggers += 1

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

    if fund["total"] >= 3:
        if fund["strength_score"] > 20 and fund["win_rate"] > 0.55:
            return {
                "trap": "T8_FALSE_COLD",
                "description": f"假冷门:{hot_dir}散户热但基本面真强({fund['strength_score']}分,胜率{fund['win_rate']:.2f})",
                "severity": 2,
                "direction_adjust": {
                    hot_dir: +2.0,
                    "home" if hot_dir == "away" else "away": -1.5
                },
                "score_multipliers": {},
                "suppress_contrarian": True,
            }

    return None


# ==========================================================
# T9: 诱反指陷阱
# ==========================================================
def detect_T9_fake_contrarian(match_obj: Dict, shin: Dict,
                              smart_signals: List) -> Optional[Dict]:
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
            "description": f"诱反指陷阱:{hot_dir}散户{hot_pct}%+赔率同向降水,反指=低效",
            "severity": 2,
            "direction_adjust": {
                hot_dir: +1.5
            },
            "score_multipliers": {},
            "suppress_contrarian": True,
        }

    return None


# ==========================================================
# T10: 沉默盘陷阱
# ==========================================================
def detect_T10_silent_market(match_obj: Dict) -> Optional[Dict]:
    change = match_obj.get("change", {}) or {}
    cw = abs(_f(change.get("win", 0)))
    cs = abs(_f(change.get("same", 0)))
    cl = abs(_f(change.get("lose", 0)))

    total_move = cw + cs + cl

    from_crs = 0
    for k in ["w10", "w20", "w21", "s00", "s11", "l01", "l02", "l12"]:
        if _f(match_obj.get(k, 0)) > 1:
            from_crs += 1

    if total_move < 0.03 and from_crs < 6:
        return {
            "trap": "T10_SILENT_MARKET",
            "description": f"沉默盘:赔率变动{total_move:.3f}+CRS覆盖{from_crs}/8 → 市场定价薄弱",
            "severity": 1,
            "direction_adjust": {},
            "score_multipliers": {},
            "confidence_penalty": 15,
        }

    return None


# ==========================================================
# T11: xG 背离陷阱
# ==========================================================
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
            "description": f"xG背离:{'; '.join(divergences)}",
            "severity": 1,
            "direction_adjust": {},
            "score_multipliers": {},
            "xg_override": {
                "home_xg": h_for if h_for > 0 else hxg_book,
                "away_xg": a_for if a_for > 0 else axg_book
            },
        }

    return None


# ==========================================================
# T12: 让球未开陷阱
# ==========================================================
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
        "description": f"让球未开但理论让{theoretical:.2f}球 → 庄家隐藏真实预期",
        "severity": 1,
        "direction_adjust": {},
        "score_multipliers": {},
        "confidence_penalty": 8,
    }


# ==========================================================
# T13: 闷平识别器
# ==========================================================
def detect_T13_goalless_draw(match_obj: Dict, engine_result: Dict,
                             shin: Dict, exp_goals: float) -> Optional[Dict]:
    shin_h = shin["home"]
    shin_a = shin["away"]

    if abs(shin_h - shin_a) > 10:
        return None

    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))

    if hxg <= 0 or axg <= 0:
        return None

    total_xg = hxg + axg

    if total_xg >= 2.3:
        return None

    info = match_obj.get("points", {}) or {}
    h_txt = str(info.get("home_strength", ""))
    a_txt = str(info.get("guest_strength", ""))

    h_for, h_against = _extract_avg_goals(h_txt)
    a_for, a_against = _extract_avg_goals(a_txt)

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
        int(_f(vote.get("lose", 33), 33))
    )

    if max_vote >= 55:
        return None

    severity = 2
    if total_xg < 2.0 and small_compressed >= 2:
        severity = 3

    return {
        "trap": "T13_GOALLESS_DRAW",
        "description": f"闷平场景:xG总{total_xg:.2f}+弱攻{weak_attack}/强防{strong_def}+小球压低{small_compressed}项",
        "severity": severity,
        "direction_adjust": {
            "draw": +1.5,
            "home": -0.5,
            "away": -0.5
        },
        "score_multipliers": {
            "2-1": 0.6,
            "1-2": 0.6,
            "2-2": 0.5,
            "3-1": 0.3,
            "1-3": 0.3,
            "3-2": 0.2
        },
        "boost_scores": ["0-0", "1-1", "1-0", "0-1"],
    }


# ==========================================================
# T14: 淘汰赛/杯赛 大热必死
# ==========================================================
def detect_T14_cup_favorite_trap(match_obj: Dict, shin: Dict) -> Optional[Dict]:
    league = str(match_obj.get("league", match_obj.get("cup", "")))
    cup_keywords = [
        "杯", "淘汰", "决赛", "半决赛", "四分之一",
        "欧冠", "欧联", "国王杯", "足总杯", "联赛杯", "Cup"
    ]

    is_cup = any(kw in league for kw in cup_keywords)

    if not is_cup:
        return None

    shin_h, shin_a = shin["home"], shin["away"]
    strong_shin = max(shin_h, shin_a)

    if strong_shin < 55:
        return None

    strong_side = "home" if shin_h > shin_a else "away"
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
        "description": f"杯赛大热必死:{strong_cn}Shin{strong_shin:.1f}%+散户{strong_vote}%,淘汰赛弱队反扑",
        "severity": 3,
        "direction_adjust": {
            strong_side: -1.0,
            weak_side: -0.3,
            "draw": +3.0
        },
        "score_multipliers": {
            "3-0": 0.3,
            "3-1": 0.4,
            "0-3": 0.3,
            "1-3": 0.4,
            "2-0": 0.7,
            "0-2": 0.7
        },
        "boost_scores": ["1-1", "0-0", "1-0", "0-1", "2-1", "1-2"],
    }


# ==========================================================
# T15: 历史僵局识别
# ==========================================================
def detect_T15_historical_deadlock(match_obj: Dict, shin: Dict) -> Optional[Dict]:
    shin_diff = abs(shin["home"] - shin["away"])

    if shin_diff > 18:
        return None

    info = match_obj.get("points", {}) or {}
    text = " ".join(str(v) for v in info.values() if v)

    h2h_patterns = [
        r"对阵[^0-9]{0,20}(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
        r"对[^0-9]{0,10}[^0-9]{2,20}(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
        r"历史交锋[^0-9]*(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]",
        r"(\d+)\s*平\s*(\d+)\s*[负败]",
    ]

    best_w, best_d, best_l = 0, 0, 0

    for pat in h2h_patterns:
        m = re.search(pat, text)
        if m:
            try:
                if len(m.groups()) == 3:
                    best_w = int(m.group(1))
                    best_d = int(m.group(2))
                    best_l = int(m.group(3))
                else:
                    best_d = int(m.group(1))
                    best_l = int(m.group(2))
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
        "description": f"历史僵局:交锋{best_w}胜{best_d}平{best_l}负(平率{draw_rate:.0%})+s11赔{s11:.1f}",
        "severity": severity,
        "direction_adjust": {
            "draw": +1.2,
            "home": -0.4,
            "away": -0.4
        },
        "score_multipliers": {
            "3-1": 0.6,
            "1-3": 0.6,
            "3-0": 0.5,
            "0-3": 0.5
        },
        "boost_scores": ["1-1", "0-0", "1-0", "0-1"],
    }


# ==========================================================
# T16: Sharp 与坏消息对冲识别
# ==========================================================
def detect_T16_sharp_badnews_conflict(match_obj: Dict, smart_signals: List,
                                      shin: Dict) -> Optional[Dict]:
    sigs_str = " ".join(str(s) for s in smart_signals)

    sharp_info = detect_sharp_direction(smart_signals)

    if not sharp_info["detected"]:
        return None

    sharp_dir = sharp_info["sharp_dir"]

    if not sharp_dir or sharp_dir == "draw":
        return None

    has_home_bad = "主队坏消息" in sigs_str or "主坏消息" in sigs_str or "主利空" in sigs_str
    has_away_bad = "客队坏消息" in sigs_str or "客坏消息" in sigs_str or "客利空" in sigs_str

    has_bad_news = (
        (sharp_dir == "home" and has_home_bad) or
        (sharp_dir == "away" and has_away_bad)
    )

    if not has_bad_news:
        return None

    sharp_shin = shin.get(sharp_dir, 33)

    if sharp_shin >= 55:
        return None

    return {
        "trap": "T16_SHARP_BADNEWS_CONFLICT",
        "description": f"Sharp({sharp_dir})+该方坏消息爆炸 → 对冲信号,平局优先",
        "severity": 2,
        "direction_adjust": {
            sharp_dir: -0.8,
            "draw": +1.5,
            "home" if sharp_dir == "away" else "away": +0.3
        },
        "score_multipliers": {},
        "boost_scores": ["1-1", "0-0"],
        "downgrade_sharp_trust": 0.3,
    }


def detect_sharp_direction(smart_signals: List) -> Dict[str, Any]:
    """从 smart_signals 提取 Sharp 方向"""
    detected = False
    sharp_dir = None

    for s in smart_signals:
        s_str = str(s)
        if "Sharp" in s_str or "sharp" in s_str:
            detected = True

            if re.search(r"(主胜|主队|走主|→\s*主|流向\s*主|资金\s*主|Sharp主)", s_str):
                sharp_dir = "home"
                break

            if re.search(r"(客胜|客队|走客|→\s*客|流向\s*客|资金\s*客|Sharp客)", s_str):
                sharp_dir = "away"
                break

            if re.search(r"(平局|平赔|走平|→\s*平|流向\s*平|资金\s*平|Sharp平|进平局)", s_str):
                sharp_dir = "draw"
                break

    return {
        "detected": detected,
        "sharp_dir": sharp_dir
    }


def detect_steam_direction(smart_signals: List) -> Dict[str, Any]:
    """从 smart_signals 提取 Steam 方向"""
    steam_dir = None
    steam_type = None

    for s in smart_signals:
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

    return {
        "steam_dir": steam_dir,
        "steam_type": steam_type
    }


# ==========================================================
# 主入口: 陷阱综合识别
# ==========================================================
def detect_all_traps(match_obj: Dict, engine_result: Dict,
                     ai_responses: Dict, smart_signals: List,
                     exp_goals: float) -> Dict[str, Any]:
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
        shin = {
            "home": 33.3,
            "draw": 33.3,
            "away": 33.3
        }

    all_detectors = [
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
    for detector in all_detectors:
        try:
            result = detector()
            if result:
                traps.append(result)
        except Exception:
            pass

    has_t14 = any(t.get("trap") == "T14_CUP_FAVORITE" for t in traps)
    if has_t14:
        traps = [t for t in traps if t.get("trap") != "T1_DRAW_TRAP"]

    t2 = next((t for t in traps if t.get("trap") == "T2_HANDICAP_DEEPER"), None)
    t3 = next((t for t in traps if t.get("trap") == "T3_HANDICAP_SHALLOWER"), None)
    if t2 and t3:
        if t2["severity"] >= t3["severity"]:
            traps = [t for t in traps if t.get("trap") != "T3_HANDICAP_SHALLOWER"]
        else:
            traps = [t for t in traps if t.get("trap") != "T2_HANDICAP_DEEPER"]

    has_t13 = any(t.get("trap") == "T13_GOALLESS_DRAW" for t in traps)
    if has_t13:
        traps = [t for t in traps if t.get("trap") != "T6_SMALL_SCORE_TRAP"]

    t4 = next((t for t in traps if t.get("trap") in ["T4_FAKE_HOME_FAVORITE", "T5_FAKE_AWAY_FAVORITE"]), None)
    t8 = next((t for t in traps if t.get("trap") == "T8_FALSE_COLD"), None)
    if t4 and t8:
        if t4["severity"] >= t8["severity"]:
            traps = [t for t in traps if t.get("trap") != "T8_FALSE_COLD"]
        else:
            traps = [t for t in traps if t.get("trap") not in ["T4_FAKE_HOME_FAVORITE", "T5_FAKE_AWAY_FAVORITE"]]

    direction_adjust = {
        "home": 0.0,
        "draw": 0.0,
        "away": 0.0
    }
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
        "shin": shin,
        "sharp_detected": sharp_info["detected"],
        "sharp_dir": sharp_info["sharp_dir"],
        "steam_dir": steam_info["steam_dir"],
        "steam_type": steam_info["steam_type"],
    }


# ============================================================================
# 📊 CRS 矩阵几何形状分析器
# ============================================================================

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
    "7-0", "7-1", "7-2", "胜其他", "9-0"
]
SCORE_OTHERS_DRAW = ["4-4", "5-5", "6-6", "平其他", "9-9"]
SCORE_OTHERS_AWAY = [
    "3-4", "0-6", "1-6", "2-6", "3-6", "0-7", "1-7", "2-7",
    "负其他", "0-9"
]
ALL_SCORE_OTHERS = SCORE_OTHERS_HOME + SCORE_OTHERS_DRAW + SCORE_OTHERS_AWAY


def crs_implied_probabilities(match_obj: Dict) -> Tuple[Dict[str, float], float, float]:
    """
    从CRS赔率反推概率分布(去庄家抽水)
    返回:(probs_dict, margin, coverage)
    """
    raw_odds = {}

    for score, key in CRS_FULL_MAP.items():
        try:
            odds = _f(match_obj.get(key, 0))
            if odds > 1.1:
                raw_odds[score] = odds
        except Exception:
            pass

    extras = {}
    for key, scores_set in [
        ("crs_win", SCORE_OTHERS_HOME),
        ("crs_same", SCORE_OTHERS_DRAW),
        ("crs_lose", SCORE_OTHERS_AWAY),
    ]:
        try:
            odds = _f(match_obj.get(key, 0))
            if odds > 1.1:
                extras[key] = {
                    "odds": odds,
                    "scores": scores_set
                }
        except Exception:
            pass

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

    reg_normalized = {
        k: v / total
        for k, v in regular.items()
    }

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
        return "unknown", ["CRS数据不足,无法分析形状"]

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
        anomalies.append(f"磨局:λ总{lt:.2f},方差低({var_h:.2f}/{var_a:.2f})")

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
        anomalies.append(f"主队进球分布偏度异常:{skew_h:.2f}")

    if abs(skew_a) > 1.8:
        anomalies.append(f"客队进球分布偏度异常:{skew_a:.2f}")

    if corr < -0.15:
        anomalies.append(f"负相关{corr:.2f}:单边场,一方得势另一方沉默")

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
        "away": 33.3
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
            "direction_probs": {
                "home": 33.3,
                "draw": 33.3,
                "away": 33.3
            },
            "top_scores": [],
        }

    moments = compute_statistical_moments(probs)
    verdict, anomalies = classify_shape(moments)
    direction_probs = compute_direction_from_crs(probs)
    sorted_scores = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_scores = [(sc, round(p, 2)) for sc, p in sorted_scores[:10]]

    return {
        "implied_probs": {
            k: round(v, 2)
            for k, v in probs.items()
        },
        "margin": margin,
        "coverage": coverage,
        "moments": moments,
        "shape_verdict": verdict,
        "anomalies": anomalies,
        "direction_probs": direction_probs,
        "top_scores": top_scores,
    }


def select_scores_in_direction_and_range(
    probs: Dict[str, float],
    direction: str,
    goal_range_min: int,
    goal_range_max: int,
    boost_scores: List[str] = None,
    suppress_scores: Dict[str, float] = None,
) -> List[Tuple[str, float]]:
    boost_scores = boost_scores or []
    suppress_scores = suppress_scores or {}

    results = []

    for sc, p in probs.items():
        if sc == "胜其他" or sc == "9-0":
            sc_dir = "home"
            total_goals = 9
        elif sc == "平其他" or sc == "9-9":
            sc_dir = "draw"
            total_goals = 9
        elif sc == "负其他" or sc == "0-9":
            sc_dir = "away"
            total_goals = 9
        else:
            try:
                h, a = str(sc).split("-")
                h, a = int(h), int(a)
                total_goals = h + a

                if h > a:
                    sc_dir = "home"
                elif h < a:
                    sc_dir = "away"
                else:
                    sc_dir = "draw"
            except Exception:
                continue

        if sc_dir != direction:
            continue

        if total_goals < 9:
            if not (goal_range_min <= total_goals <= goal_range_max):
                continue

        adjusted = p

        if sc in boost_scores:
            adjusted *= 1.4

        if sc in suppress_scores:
            adjusted *= suppress_scores[sc]

        results.append((sc, adjusted))

    results.sort(key=lambda x: x[1], reverse=True)

    return results


# ============================================================================
# 🧠 贝叶斯决策引擎 + 决策锁定链
# ============================================================================

def _normalize_score_text(s: Any) -> str:
    if s is None:
        return ""

    s = str(s).strip()
    s = s.replace("：", "-").replace(":", "-")
    s = s.replace(" ", "")
    s = s.replace("比", "-")
    s = s.replace("主胜其他", "胜其他")
    s = s.replace("客胜其他", "负其他")

    return s


def _parse_score(s: str) -> Tuple[Optional[int], Optional[int]]:
    """安全比分解析,支持虚拟比分"""
    try:
        s_str = _normalize_score_text(s)

        if "胜其他" in s_str:
            return 9, 0

        if "平其他" in s_str:
            return 9, 9

        if "负其他" in s_str:
            return 0, 9

        if s_str in ["主胜", "客胜", "平局", "胜", "平", "负"]:
            return None, None

        p = s_str.split("-")
        if len(p) != 2:
            return None, None

        return int(p[0]), int(p[1])
    except Exception:
        return None, None


def _score_direction(score_str: str) -> Optional[str]:
    score_str = _normalize_score_text(score_str)
    h, a = _parse_score(score_str)

    if h is None:
        return None

    if "胜其他" in score_str or score_str == "9-0":
        return "home"

    if "平其他" in score_str or score_str == "9-9":
        return "draw"

    if "负其他" in score_str or score_str == "0-9":
        return "away"

    if h > a:
        return "home"

    if h < a:
        return "away"

    return "draw"


def _extract_score_from_top3_item(item: Any) -> str:
    if isinstance(item, dict):
        return _normalize_score_text(item.get("score", ""))

    if isinstance(item, str):
        return _normalize_score_text(item)

    return ""


def _direction_from_cn(result_cn: str) -> Optional[str]:
    result_cn = str(result_cn or "")
    if "主" in result_cn:
        return "home"
    if "客" in result_cn or "负" == result_cn:
        return "away"
    if "平" in result_cn:
        return "draw"
    return None


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
        "home": max(0.05, shin.get("home", 33.3) / 100),
        "draw": max(0.05, shin.get("draw", 33.3) / 100),
        "away": max(0.05, shin.get("away", 33.3) / 100),
    }

    total = sum(prior.values())
    prior = {
        k: v / total
        for k, v in prior.items()
    }

    log_odds = {
        "home": math.log(prior["home"]),
        "draw": math.log(prior["draw"]),
        "away": math.log(prior["away"]),
    }

    evidences = []

    if crs_direction and sum(crs_direction.values()) > 50:
        crs_p = {
            "home": max(0.05, crs_direction["home"] / 100),
            "draw": max(0.05, crs_direction["draw"] / 100),
            "away": max(0.05, crs_direction["away"] / 100),
        }

        for d in log_odds:
            llr = math.log(crs_p[d] / prior[d]) * 1.2
            log_odds[d] += llr

        evidences.append(f"CRS方向:{crs_direction}")

    sharp_trust = trap_report.get("sharp_trust_override", 1.0)
    steam_trust = trap_report.get("steam_trust_override", 1.0)

    suppress_draw_signals = False
    cup_favorite_detected = False

    for t in trap_report.get("traps_detected", []):
        if t.get("suppress_draw_sharp"):
            suppress_draw_signals = True
        if t.get("trap") == "T14_CUP_FAVORITE":
            cup_favorite_detected = True

    trap_adj = trap_report.get("direction_adjust", {})
    max_trap_dir = None
    max_trap_val = 0

    for d, v in trap_adj.items():
        if v > max_trap_val:
            max_trap_val = v
            max_trap_dir = d

    sharp_dir = trap_report.get("sharp_dir")

    if max_trap_dir and max_trap_val >= 2.0 and sharp_dir and sharp_dir != max_trap_dir:
        sharp_trust = 0.15
        steam_trust = 0.25
        evidences.append(f"⚠️ Sharp({sharp_dir})vs陷阱指向({max_trap_dir}) → 降权Sharp×0.15")

    if cup_favorite_detected and sharp_dir and sharp_dir != "draw":
        sharp_trust = min(sharp_trust, 0.15)
        steam_trust = min(steam_trust, 0.15)
        log_odds["draw"] += 1.5
        evidences.append(f"⚠️ 杯赛大热必死+Sharp({sharp_dir}) → Sharp/Steam降权×0.15+平局加成")

    sharp_detected = trap_report.get("sharp_detected", False)

    if sharp_detected and sharp_dir and sharp_dir in log_odds:
        if sharp_dir == "draw" and suppress_draw_signals:
            sharp_trust = min(sharp_trust, 0.1)

        effective_sharp = 2.2 * sharp_trust
        log_odds[sharp_dir] += effective_sharp

        for other in log_odds:
            if other != sharp_dir:
                log_odds[other] -= effective_sharp / 2

        evidences.append(f"Sharp→{sharp_dir}(权重×{sharp_trust:.2f},净+{effective_sharp:.2f})")

    steam_dir = trap_report.get("steam_dir")
    steam_type = trap_report.get("steam_type")

    if steam_dir and steam_dir in log_odds:
        if steam_dir == "draw" and suppress_draw_signals:
            steam_trust = min(steam_trust, 0.1)

        base = 2.0 if steam_type == "reverse" else 1.2
        effective_steam = base * steam_trust
        log_odds[steam_dir] += effective_steam

        for other in log_odds:
            if other != steam_dir:
                log_odds[other] -= effective_steam / 2

        evidences.append(f"Steam({steam_type or 'normal'})→{steam_dir}(权重×{steam_trust:.2f},净+{effective_steam:.2f})")

    for d, v in trap_adj.items():
        if d in log_odds:
            log_odds[d] += v

    if trap_adj and any(abs(v) > 0.1 for v in trap_adj.values()):
        evidences.append(f"陷阱方向调整:{trap_adj}")

    total_ai = sum(ai_directions.values())

    if total_ai > 0:
        max_ai = max(ai_directions.values())
        consensus = max_ai / total_ai if total_ai > 0 else 0

        ai_weight = 0.8 if consensus >= 0.7 else (0.5 if consensus >= 0.5 else 0.2)

        if sharp_detected:
            ai_weight *= 0.5

        for d in log_odds:
            if ai_directions.get(d, 0) > 0:
                share = ai_directions[d] / total_ai
                log_odds[d] += math.log(max(0.05, share) / (1 / 3)) * ai_weight

        evidences.append(f"AI共识{consensus:.0%},权重{ai_weight:.1f}")

    hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))

    xg_ov = trap_report.get("xg_override")
    if xg_ov:
        hxg = xg_ov.get("home_xg", hxg)
        axg = xg_ov.get("away_xg", axg)

    if hxg > 0.3 and axg > 0.3:
        xg_diff = hxg - axg

        if xg_diff > 0.5:
            log_odds["home"] += min(1.5, xg_diff * 0.8)
            log_odds["away"] -= min(1.0, xg_diff * 0.5)
            evidences.append(f"xG主优{xg_diff:+.2f}")

        elif xg_diff < -0.5:
            log_odds["away"] += min(1.5, abs(xg_diff) * 0.8)
            log_odds["home"] -= min(1.0, abs(xg_diff) * 0.5)
            evidences.append(f"xG客优{xg_diff:+.2f}")

    change = match_obj.get("change", {}) or {}
    cw = _f(change.get("win", 0))
    cs = _f(change.get("same", 0))
    cl = _f(change.get("lose", 0))

    if cw < -0.05:
        log_odds["home"] += min(1.0, abs(cw) * 8)
    elif cw > 0.05:
        log_odds["home"] -= min(0.5, cw * 5)

    if cs < -0.05 and not (trap_report.get("direction_adjust", {}).get("draw", 0) < -1):
        log_odds["draw"] += min(0.8, abs(cs) * 6)

    if cl < -0.05:
        log_odds["away"] += min(1.0, abs(cl) * 8)
    elif cl > 0.05:
        log_odds["away"] -= min(0.5, cl * 5)

    if not trap_report.get("suppress_contrarian"):
        vote = match_obj.get("vote", {}) or {}
        vh = int(_f(vote.get("win", 33), 33))
        va = int(_f(vote.get("lose", 33), 33))
        max_vote = max(vh, va)

        if max_vote >= 60:
            hot_dir = "home" if vh == max_vote else "away"
            contra_power = min(1.2, (max_vote - 55) / 20)

            log_odds[hot_dir] -= contra_power

            for other in log_odds:
                if other != hot_dir:
                    log_odds[other] += contra_power / 2

            evidences.append(f"散户热{hot_dir}{max_vote}%,反指{contra_power:.2f}")

    hxg_safe = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg_safe = _f(engine_result.get("bookmaker_implied_away_xg", 0))

    if hxg_safe > 0 and axg_safe > 0:
        xg_total = hxg_safe + axg_safe
        xg_diff_abs = abs(hxg_safe - axg_safe)
        shin_max = max(shin.values())

        if xg_diff_abs < 0.4 and xg_total < 2.8 and shin_max < 55:
            if log_odds["draw"] < -1.0:
                recovery = min(1.2, abs(log_odds["draw"]) * 0.5)
                log_odds["draw"] += recovery
                evidences.append(f"平局保护:均势场(xG差{xg_diff_abs:.2f},总{xg_total:.2f})+{recovery:.2f}")

    league_str = str(match_obj.get("league", match_obj.get("cup", "")))
    if any(kw in league_str for kw in ["杯", "淘汰", "欧冠", "欧联", "Cup"]):
        if shin.get("draw", 0) >= 20:
            log_odds["draw"] += 0.4
            evidences.append("杯赛/欧战平局加成+0.4")

    temperature = 1.8
    scaled_log_odds = {
        k: v / temperature
        for k, v in log_odds.items()
    }

    max_log = max(scaled_log_odds.values())
    exp_vals = {
        k: math.exp(v - max_log)
        for k, v in scaled_log_odds.items()
    }

    total_exp = sum(exp_vals.values())
    posterior = {
        k: v / total_exp
        for k, v in exp_vals.items()
    }

    posterior = {
        k: max(0.03, min(0.88, v))
        for k, v in posterior.items()
    }

    total_adj = sum(posterior.values())
    posterior = {
        k: v / total_adj
        for k, v in posterior.items()
    }

    final_direction = max(posterior, key=posterior.get)
    max_p = posterior[final_direction]
    sorted_p = sorted(posterior.values(), reverse=True)
    dir_gap = sorted_p[0] - sorted_p[1]

    return {
        "posterior": {
            k: round(v * 100, 2)
            for k, v in posterior.items()
        },
        "final_direction": final_direction,
        "dir_confidence": round(max_p * 100, 1),
        "dir_gap": round(dir_gap * 100, 1),
        "evidences": evidences,
        "prior": {
            k: round(v * 100, 2)
            for k, v in prior.items()
        },
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

    if not sharp_dir:
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


def determine_goal_range(
    direction: str,
    moments: Dict[str, float],
    exp_goals: float,
    trap_report: Dict[str, Any],
    match_obj: Dict[str, Any],
    engine_result: Dict[str, Any],
) -> Tuple[int, int, str]:
    actual_hc_raw = _f(match_obj.get("give_ball", 0))

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

    if direction == "home":
        if actual_hc_raw <= -1.5:
            extreme_score += 2
    elif direction == "away":
        if actual_hc_raw >= 1.5:
            extreme_score += 2

    if direction == "home":
        hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
        axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))
        if hxg - axg > 1.5:
            extreme_score += 1

    if direction == "away":
        hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
        axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))
        if axg - hxg > 1.5:
            extreme_score += 1

    if extreme_score >= 5 and exp_goals >= 2.8:
        return 5, 12, "extreme_blowout"

    if not moments:
        lt = exp_goals
    else:
        lt = moments.get("lambda_total", exp_goals)

    lt_avg = (lt * 0.6 + exp_goals * 0.4)

    if lt_avg >= 3.5:
        return 3, 6, "shootout"

    if lt_avg >= 2.9:
        return 2, 5, "high_goals"

    if lt_avg >= 2.3:
        return 2, 4, "normal"

    if lt_avg >= 1.8:
        return 1, 3, "low_goals"

    return 0, 2, "grinder"


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

    if scenario == "extreme_blowout":
        if direction == "home":
            label = "胜其他"
        elif direction == "away":
            label = "负其他"
        else:
            label = "平其他"

        others_key = {
            "home": "crs_win",
            "away": "crs_lose",
            "draw": "crs_same"
        }[direction]

        others_odds = _f(match_obj.get(others_key, 0))

        if 1.5 < others_odds < 80:
            return label, [(label, 100.0)]

    candidates = {}

    for sc, p in crs_probs.items():
        sc_norm = _normalize_score_text(sc)
        sc_dir = _score_direction(sc_norm)

        if sc_dir != direction:
            continue

        h, a = _parse_score(sc_norm)

        if h is None:
            continue

        total_g = h + a

        if "其他" in sc_norm:
            if scenario == "extreme_blowout":
                candidates[sc_norm] = p * 1.5
            else:
                candidates[sc_norm] = p * 0.3
            continue

        if not (g_min <= total_g <= g_max):
            continue

        candidates[sc_norm] = p

    for sc, vote_pts in ai_votes.items():
        sc_norm = _normalize_score_text(sc)
        sc_dir = _score_direction(sc_norm)

        if sc_dir != direction:
            continue

        h, a = _parse_score(sc_norm)

        if h is None:
            continue

        total_g = h + a

        if "其他" not in sc_norm and not (g_min <= total_g <= g_max):
            continue

        ai_bonus = vote_pts * 1.8
        candidates[sc_norm] = candidates.get(sc_norm, 0) + ai_bonus

    score_mults = trap_report.get("score_multipliers", {})
    for sc, mult in score_mults.items():
        sc_norm = _normalize_score_text(sc)
        if sc_norm in candidates:
            candidates[sc_norm] *= mult

    boost_list = trap_report.get("boost_scores", [])
    for sc in boost_list:
        sc_norm = _normalize_score_text(sc)
        if sc_norm in candidates:
            candidates[sc_norm] *= 1.5

    if scenario == "shootout":
        for sc in ["0-0", "1-0", "0-1"]:
            if sc in candidates:
                candidates[sc] *= 0.2

        for sc in ["2-2", "3-2", "2-3", "3-3", "3-1", "1-3", "4-2", "2-4"]:
            if sc in candidates:
                candidates[sc] *= 1.4

    elif scenario == "grinder":
        for sc in ["2-2", "3-1", "1-3", "3-2", "2-3"]:
            if sc in candidates:
                candidates[sc] *= 0.3

        for sc in ["1-0", "0-1", "0-0", "1-1"]:
            if sc in candidates:
                candidates[sc] *= 1.3

    elif scenario == "low_goals":
        for sc in ["3-2", "2-3", "3-3", "4-2"]:
            if sc in candidates:
                candidates[sc] *= 0.3

        for sc in ["1-0", "0-1", "2-1", "1-2", "1-1", "0-0"]:
            if sc in candidates:
                candidates[sc] *= 1.2

    elif scenario == "high_goals":
        for sc in ["0-0", "1-0", "0-1"]:
            if sc in candidates:
                candidates[sc] *= 0.4

    if direction == "home":
        if shin.get("home", 33) > 55:
            for sc in ["2-0", "3-0", "2-1", "3-1"]:
                if sc in candidates:
                    candidates[sc] *= 1.15

    elif direction == "away":
        if shin.get("away", 33) > 55:
            for sc in ["0-2", "0-3", "1-2", "1-3"]:
                if sc in candidates:
                    candidates[sc] *= 1.15

    if not candidates:
        fallback_map = {
            "home": "1-0" if scenario in ["grinder", "low_goals"] else "2-1",
            "away": "0-1" if scenario in ["grinder", "low_goals"] else "1-2",
            "draw": "1-1" if scenario != "grinder" else "0-0",
        }
        return fallback_map[direction], [(fallback_map[direction], 1.0)]

    sorted_scores = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    best = sorted_scores[0][0]

    return best, [(sc, round(p, 2)) for sc, p in sorted_scores[:10]]


def decision_lock_chain(
    match_obj: Dict[str, Any],
    engine_result: Dict[str, Any],
    trap_report: Dict[str, Any],
    crs_analysis: Dict[str, Any],
    ai_responses: Dict[str, Dict],
    smart_signals: List[str],
    exp_goals: float,
) -> Dict[str, Any]:
    ai_directions = {
        "home": 0.0,
        "draw": 0.0,
        "away": 0.0
    }

    ai_votes = {}
    ai_weights = {
        "claude": 1.5,
        "gemini": 1.4,
        "grok": 1.35,
        "gpt": 1.0
    }

    for name, r in ai_responses.items():
        if not isinstance(r, dict):
            continue

        sc_raw = _normalize_score_text(r.get("ai_score", ""))
        top3 = r.get("top3", [])

        sc = sc_raw
        h0, a0 = _parse_score(sc)

        if h0 is None and top3:
            sc = _extract_score_from_top3_item(top3[0])

        h, a = _parse_score(sc)
        if h is None:
            continue

        weight = ai_weights.get(name, 1.0)

        if h > a:
            ai_directions["home"] += weight
        elif h < a:
            ai_directions["away"] += weight
        else:
            ai_directions["draw"] += weight

        sc_clean = _normalize_score_text(sc)
        ai_votes[sc_clean] = ai_votes.get(sc_clean, 0) + weight

        if isinstance(top3, list):
            for rank, t in enumerate(top3[1:3], 2):
                sc2 = _extract_score_from_top3_item(t)
                h2, a2 = _parse_score(sc2)

                if h2 is not None:
                    w2 = 0.4 if rank == 2 else 0.2
                    ai_votes[sc2] = ai_votes.get(sc2, 0) + weight * w2

    claude_r = ai_responses.get("claude", {})
    if isinstance(claude_r, dict) and _f(claude_r.get("ai_confidence", 0)) >= 75:
        cl_sc = _normalize_score_text(claude_r.get("ai_score", ""))
        cl_h, cl_a = _parse_score(cl_sc)

        if cl_h is not None:
            cl_dir = "home" if cl_h > cl_a else ("away" if cl_h < cl_a else "draw")
            other_dirs = {}
            other_confs = []
            valid_others = 0

            for n in ["gpt", "grok", "gemini"]:
                r = ai_responses.get(n, {})
                if not isinstance(r, dict):
                    continue

                sc = _normalize_score_text(r.get("ai_score", ""))
                h, a = _parse_score(sc)

                if h is None:
                    continue

                valid_others += 1
                d = "home" if h > a else ("away" if h < a else "draw")
                other_dirs[d] = other_dirs.get(d, 0) + 1
                other_confs.append(_f(r.get("ai_confidence", 60), 60))

            if other_dirs and valid_others >= 2:
                majority_dir = max(other_dirs, key=other_dirs.get)
                majority_count = other_dirs[majority_dir]
                is_hard_majority = majority_count >= max(2, int(valid_others * 0.67))
                avg_other_conf = sum(other_confs) / len(other_confs) if other_confs else 60
                claude_conf = _f(claude_r.get("ai_confidence", 60), 60)

                if cl_dir != majority_dir and is_hard_majority and claude_conf > avg_other_conf:
                    cl_clean = _normalize_score_text(cl_sc)
                    if cl_clean in ai_votes:
                        ai_votes[cl_clean] *= 2.5

    shin = trap_report.get("shin", {
        "home": 33.3,
        "draw": 33.3,
        "away": 33.3
    })

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
    override_triggered, override_dir, _ = check_sharp_override(
        shin,
        trap_report,
        posterior,
        trap_score
    )

    if override_triggered and override_dir:
        final_direction = override_dir
        cur_max = max(posterior.values())

        if posterior.get(override_dir, 0) < cur_max:
            posterior[override_dir] = cur_max + 5
            total = sum(posterior.values())
            if total > 0:
                posterior = {
                    k: round(v / total * 100, 2)
                    for k, v in posterior.items()
                }

    goal_range_min, goal_range_max, scenario = determine_goal_range(
        direction=final_direction,
        moments=crs_analysis.get("moments", {}),
        exp_goals=exp_goals,
        trap_report=trap_report,
        match_obj=match_obj,
        engine_result=engine_result,
    )

    crs_probs = crs_analysis.get("implied_probs", {})
    best_score, top_candidates = select_score(
        direction=final_direction,
        goal_range=(goal_range_min, goal_range_max),
        scenario=scenario,
        crs_probs=crs_probs,
        ai_votes=ai_votes,
        trap_report=trap_report,
        shin=shin,
        moments=crs_analysis.get("moments", {}),
        match_obj=match_obj,
    )

    best_score = _normalize_score_text(best_score)
    is_score_others = best_score in ALL_SCORE_OTHERS or "其他" in best_score

    if is_score_others:
        if best_score in SCORE_OTHERS_HOME or best_score == "胜其他":
            display_label = "胜其他"
            final_direction_lock = "home"
        elif best_score in SCORE_OTHERS_DRAW or best_score == "平其他":
            display_label = "平其他"
            final_direction_lock = "draw"
        else:
            display_label = "负其他"
            final_direction_lock = "away"
    else:
        display_label = best_score
        h, a = _parse_score(best_score)

        if h is None:
            final_direction_lock = final_direction
        else:
            final_direction_lock = "home" if h > a else ("away" if h < a else "draw")

    if final_direction_lock != final_direction:
        aligned = None

        for sc, pts in top_candidates:
            d = _score_direction(sc)
            if d == final_direction:
                aligned = sc
                break

        if aligned:
            best_score = _normalize_score_text(aligned)
            h, a = _parse_score(best_score)

            if h is None:
                if "胜其他" in best_score:
                    final_direction_lock = "home"
                elif "平其他" in best_score:
                    final_direction_lock = "draw"
                elif "负其他" in best_score:
                    final_direction_lock = "away"
            else:
                final_direction_lock = "home" if h > a else ("away" if h < a else "draw")

            is_score_others = "其他" in best_score

            if is_score_others:
                display_label = {
                    "home": "胜其他",
                    "draw": "平其他",
                    "away": "负其他"
                }[final_direction_lock]
            else:
                display_label = best_score

    direction_cn_map = {
        "home": "主胜",
        "draw": "平局",
        "away": "客胜"
    }

    result_cn = direction_cn_map[final_direction_lock]

    post_argmax = max(posterior, key=posterior.get)
    if post_argmax != final_direction_lock:
        cur_max = posterior[post_argmax]
        posterior[final_direction_lock] = cur_max + 3

        total = sum(posterior.values())
        if total > 0:
            posterior = {
                k: round(v / total * 100, 2)
                for k, v in posterior.items()
            }

    return {
        "predicted_score": best_score,
        "predicted_label": display_label,
        "result": result_cn,
        "display_direction": result_cn,
        "final_direction": final_direction_lock,
        "is_score_others": is_score_others,
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


# ====================================================================
# 🤖 v18.2 AI Prompt — 防幻觉 + 联赛弱先验 + 5步思维锚
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
        "reflection": "v18.2 贝叶斯+16维陷阱+防幻觉AI启动",
        "kill_history": []
    }


def save_ai_diary(diary):
    os.makedirs("data", exist_ok=True)

    with open("data/ai_diary.json", "w", encoding="utf-8") as f:
        json.dump(diary, f, ensure_ascii=False, indent=2)


def build_v18_prompt(match_analyses):
    diary = load_ai_diary()

    p = "<context>\n"
    p += "你正在中国体彩竞彩足球市场做量化比分预测。只允许使用 match_data 中出现的数据，不允许编造伤停、赛程、新闻、阵容或历史事实。\n"
    p += "如果某项信息缺失，必须写“该项缺失”，不得自行补全。\n"

    if diary.get("reflection"):
        p += f"[系统记忆] 昨日:{diary.get('yesterday_win_rate','N/A')} | 反思:{diary['reflection']}\n"

    p += "</context>\n\n"

    p += "<anti_hallucination_rules>\n"
    p += "1. 不得引用 match_data 未提供的球员名、伤停、新闻、排名、战意、教练、赛程密度。\n"
    p += "2. 不得把联赛风格当成硬结论。德甲大比分、英超爆冷、意甲/法甲平局、西甲小球只能作为弱先验，必须被赔率、CRS、xG、a0-a7共同验证。\n"
    p += "3. 如果数据冲突，按优先级处理: CRS矩阵/进球赔率 > Shin/欧赔 > Sharp/Steam > xG > 基本面文本 > 联赛弱先验。\n"
    p += "4. 输出 top3 必须是明确比分或胜其他/平其他/负其他，不允许输出方向词替代比分。\n"
    p += "5. top3[0].score 的方向必须等于 final_direction。\n"
    p += "</anti_hallucination_rules>\n\n"

    p += "<league_weak_priors>\n"
    p += "德甲/荷甲/挪超/美职: 允许倾向高节奏，但必须满足 λ_total>=2.8 或 a5/a6/a7至少一项压低，才可选3-2/4-1/胜其他。\n"
    p += "英超: 爆冷只在强队散户热度>=60%、赔率未继续保护强队、CRS方向不支持强队时成立；否则不得强行爆冷。\n"
    p += "意甲/法甲: 平局只在双方Shin差<12%、λ_total<=2.4、s11赔率不高或T13/T15触发时优先。\n"
    p += "西甲: 小球只在λ_total<=2.5、a0-a2压低或CRS形状grinder/balanced时优先；强队碾压场不得机械小球。\n"
    p += "杯赛/淘汰赛: 90分钟平局概率上修，但必须结合Shin、散户热度和CRS平局概率。\n"
    p += "</league_weak_priors>\n\n"

    p += "<iron_rules>\n"
    p += "铁律1 [方向-比分一致性]: top3[0].score 的比分方向必须与 final_direction 完全一致。\n"
    p += "铁律2 [胜其他标记]: is_score_others=true 时, top3 至少包含一个胜其他/负其他/平其他。\n"
    p += "铁律3 [资金优先但防诱饵]: 基本面与资金面冲突时优先看Sharp/Steam/赔率变动；若系统提示Sharp被降权，则不得盲从Sharp。\n"
    p += "铁律4 [诱平识别]: 平赔降水+强势方Shin>=40%+xG同向/让球偏深，多数是诱平，优先压制1-1/0-0。\n"
    p += "铁律5 [杯赛反指]: 杯/淘汰/决赛+强势方Shin>=55%+散户跟风，优先考虑平局或弱方小胜。\n"
    p += "</iron_rules>\n\n"

    p += "<analytical_framework>\n"
    p += "reason 必须按以下5步写，每步只基于 match_data:\n"
    p += "Step1 [市场底盘]: 说明Shin/欧赔/CRS方向谁占优。\n"
    p += "Step2 [陷阱矩阵]: 说明T1-T16是否触发及如何影响方向。\n"
    p += "Step3 [进球区间]: 用λ_total、a0-a7、CRS形状判断小球/普通/大球/胜其他。\n"
    p += "Step4 [联赛弱先验]: 仅在数据支持时引用联赛风格，否则写“不启用联赛先验”。\n"
    p += "Step5 [比分落点]: 在最终方向与进球区间内选 top3，禁止方向和比分冲突。\n"
    p += "</analytical_framework>\n\n"

    p += "<output_format>\n"
    p += "严格输出 JSON 数组。不要 Markdown，不要解释，不要前缀后缀。\n"
    p += "每场结构:\n"
    p += "{\n"
    p += "  \"match\": 1,\n"
    p += "  \"top3\": [{\"score\":\"2-1\",\"prob\":15},{\"score\":\"1-0\",\"prob\":12},{\"score\":\"2-0\",\"prob\":10}],\n"
    p += "  \"reason\": \"Step1... Step2... Step3... Step4... Step5...\",\n"
    p += "  \"ai_confidence\": 0,\n"
    p += "  \"is_score_others\": false,\n"
    p += "  \"detected_traps\": [\"T1\"],\n"
    p += "  \"final_direction\": \"home\"\n"
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
        hc = m.get("give_ball", "0")
        sp_h = _f(m.get("sp_home", m.get("win", 0)))
        sp_d = _f(m.get("sp_draw", m.get("same", 0)))
        sp_a = _f(m.get("sp_away", m.get("lose", 0)))

        p += f"<match index=\"{i+1}\">\n"
        p += f"[{i+1}] {h} vs {a} | {league}\n"
        p += f"欧赔: {sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f} | 让球: {hc}\n"

        shin = trap_preview.get("shin", {})
        if shin:
            p += f"Shin概率: 主{shin.get('home',0):.1f}% 平{shin.get('draw',0):.1f}% 客{shin.get('away',0):.1f}%\n"

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

        traps = trap_preview.get("traps_detected", [])
        if traps:
            p += f"🎭 系统识别陷阱({len(traps)}个,严重度{trap_preview.get('total_severity',0)}):\n"
            for t in traps:
                p += f"  - {t.get('trap','?')}: {t.get('description','')[:100]}\n"
        else:
            p += "🎭 系统未识别明显陷阱\n"

        a_list = []
        compressed = []
        for g in range(8):
            v = m.get(f"a{g}", "")
            a_list.append(f"{g}={v}")

            try:
                actual = _f(v)
                if actual > 1:
                    std = STANDARD_GOAL_ODDS.get(g, 50)
                    ratio = std / actual
                    if ratio > 1.5:
                        compressed.append(f"{g}球(压低{ratio:.1f}x)")
            except Exception:
                pass

        if a_list:
            p += f"总进球: {' | '.join(a_list)}\n"

        if compressed:
            p += f"⚠️ 进球数压低: {', '.join(compressed)}\n"

        crs_lines = []
        for sc, key in CRS_FULL_MAP.items():
            try:
                odds = _f(m.get(key, 0))
                if odds > 1:
                    crs_lines.append(f"{sc}={odds:.1f}")
            except Exception:
                pass

        if crs_lines:
            p += f"CRS: {' | '.join(crs_lines)}\n"

        crs_others = []
        for k, label in [
            ("crs_win", "胜其他"),
            ("crs_same", "平其他"),
            ("crs_lose", "负其他")
        ]:
            v = m.get(k, "")
            if v:
                crs_others.append(f"{label}={v}")

        if crs_others:
            p += f"📌 {' | '.join(crs_others)}\n"

        hf_l = []
        for k, lb in {
            "ss": "主/主",
            "sp": "主/平",
            "sf": "主/负",
            "ps": "平/主",
            "pp": "平/平",
            "pf": "平/负",
            "fs": "负/主",
            "fp": "负/平",
            "ff": "负/负"
        }.items():
            try:
                v = _f(m.get(k, 0))
                if v > 1:
                    hf_l.append(f"{lb}={v:.2f}")
            except Exception:
                pass

        if hf_l:
            p += f"半全场: {' | '.join(hf_l)}\n"

        vote = m.get("vote", {})
        if vote:
            p += f"散户: 胜{vote.get('win','?')}% 平{vote.get('same','?')}% 负{vote.get('lose','?')}%"

            try:
                max_v = max(
                    int(_f(vote.get("win", 33))),
                    int(_f(vote.get("lose", 33)))
                )
                if max_v >= 58:
                    p += f" ⚠️大热({max_v}%需检验是否反指)"
            except Exception:
                pass

            p += "\n"

        change = m.get("change", {})
        if change and isinstance(change, dict):
            cw = change.get("win", 0)
            cs = change.get("same", 0)
            cl = change.get("lose", 0)

            if cw or cs or cl:
                p += f"赔率变动: 胜{cw} 平{cs} 负{cl} (负=降水=资金流入)\n"

        info = m.get("information", {})
        if isinstance(info, dict):
            for k, label in [
                ("home_injury", "主伤停"),
                ("guest_injury", "客伤停"),
                ("home_bad_news", "主利空"),
                ("guest_bad_news", "客利空")
            ]:
                if info.get(k):
                    p += f"{label}: {str(info[k])[:500].replace(chr(10), ' ')}\n"

        points = m.get("points", {})
        if isinstance(points, dict):
            for k in ["home_strength", "guest_strength", "match_points"]:
                txt = str(points.get(k, ""))[:500].replace("\n", " ")
                if "场均" in txt or "主场" in txt or "客场" in txt:
                    p += f"情报: {txt}\n"
                    break

        smart_sigs = stats.get("smart_signals", []) if stats else []
        if smart_sigs:
            p += f"🔥 信号: {', '.join(str(s) for s in smart_sigs[:6])}\n"

        p += "</match>\n\n"

    p += "</match_data>\n"

    return p


# ====================================================================
# AI 调用引擎：不拆批、不切换模型、解析失败不重试
# ====================================================================

FALLBACK_URLS = [
    None,
    "https://www.api522.pro/v1",
    "https://api522.pro/v1",
    "https://api521.pro/v1",
    "http://69.63.213.33:666/v1"
]

GPT_DEFAULT_URL = "https://poloai.top/v1"
GPT_DEFAULT_KEY = ""


def get_clean_env_url(name, default=""):
    v = str(os.environ.get(name, globals().get(name, default))).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/-]+)", v)
    return match.group(1) if match else v


def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")


def _debug_short(v, max_len=500):
    try:
        s = json.dumps(v, ensure_ascii=False)
    except Exception:
        s = str(v)

    if len(s) > max_len:
        return s[:max_len] + "..."

    return s


def _save_debug_dump(ai_name, data, tag):
    try:
        os.makedirs("data/debug", exist_ok=True)
        dump_file = f"data/debug/{ai_name}_{tag}_{int(time.time())}.json"

        with open(dump_file, "w", encoding="utf-8") as df:
            json.dump(data, df, ensure_ascii=False, indent=2)

        print(f"    📁 失败响应已保存: {dump_file}")
    except Exception:
        pass


def _recursive_find_json_text(obj, max_items=50):
    """
    从任意响应结构里递归寻找包含 JSON 数组的字符串。
    专门兜底 proxy content=null 但其他字段存在完整输出的情况。
    """
    found = []

    def walk(x):
        if len(found) >= max_items:
            return

        if isinstance(x, str):
            s = x.strip()
            if '"match"' in s and "[" in s and "]" in s:
                found.append(s)
            elif '\\"match\\"' in s and "[" in s and "]" in s:
                try:
                    unescaped = json.loads(f'"{s}"')
                    found.append(unescaped)
                except Exception:
                    found.append(s.replace('\\"', '"'))
            return

        if isinstance(x, list):
            for item in x:
                walk(item)
            return

        if isinstance(x, dict):
            priority_keys = [
                "content", "text", "output_text", "answer", "response",
                "result", "completion", "message_content", "assistant_content",
                "model_response", "reasoning_content", "reasoning", "thinking"
            ]

            for k in priority_keys:
                if k in x:
                    walk(x[k])

            for k, v in x.items():
                if k not in priority_keys:
                    walk(v)

    walk(obj)

    if not found:
        return ""

    found.sort(key=len, reverse=True)
    return found[0]


def _extract_response_text(data, is_gem, ai_name):
    """多种响应结构中提取文本，增强 content=null 兜底"""
    raw_text = ""

    try:
        if is_gem:
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                texts = []

                for part in parts:
                    if isinstance(part, dict) and part.get("text"):
                        texts.append(str(part.get("text")).strip())

                raw_text = "\n".join(t for t in texts if t).strip()

        else:
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})

                if isinstance(msg, dict):
                    if msg.get("content") is None and data.get("usage", {}).get("completion_tokens", 0) > 100:
                        print(f"    🚨 [proxy bug] {ai_name.upper()} content=null 但消耗了 {data['usage'].get('completion_tokens', 0)} token")

                    content_val = msg.get("content", "")

                    if content_val:
                        if isinstance(content_val, str):
                            raw_text = content_val.strip()

                        elif isinstance(content_val, list):
                            texts = []
                            for item in content_val:
                                if isinstance(item, dict):
                                    if item.get("type") == "text" and item.get("text"):
                                        texts.append(str(item.get("text")).strip())
                                    elif item.get("text"):
                                        texts.append(str(item.get("text")).strip())
                                elif isinstance(item, str):
                                    texts.append(item.strip())

                            raw_text = "\n".join(t for t in texts if t).strip()

                    if not raw_text:
                        for field in [
                            "text", "answer", "response", "output_text", "final_answer",
                            "output", "result", "completion", "message_content",
                            "assistant_content", "model_response"
                        ]:
                            v = msg.get(field, "")
                            if v and isinstance(v, str) and v.strip():
                                raw_text = v.strip()
                                break

                    if not raw_text:
                        json_text = _recursive_find_json_text(msg)
                        if json_text:
                            raw_text = json_text
                            print(f"    🆘 从message递归字段提取JSON文本")

            if not raw_text and data.get("output") and isinstance(data["output"], list):
                texts = []
                for out_item in data["output"]:
                    if isinstance(out_item, dict):
                        if out_item.get("type") == "message":
                            for ct in out_item.get("content", []):
                                if isinstance(ct, dict):
                                    if ct.get("text"):
                                        texts.append(str(ct["text"]).strip())
                                    elif ct.get("content"):
                                        texts.append(str(ct["content"]).strip())
                        elif out_item.get("text"):
                            texts.append(str(out_item.get("text")).strip())

                raw_text = "\n".join(t for t in texts if t).strip()

        if not raw_text:
            json_text = _recursive_find_json_text(data)
            if json_text:
                raw_text = json_text
                print("    🆘 从完整response递归提取JSON文本")

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
                    print("    🆘 终极兜底: 从response dump中提取JSON")

    except Exception as ex:
        print(f"    ⚠️ 响应提取异常: {str(ex)[:120]}")

    return raw_text or ""


def _remove_thinking_blocks(text):
    if not text:
        return ""

    clean = str(text)
    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"```(?:json|JSON|javascript|js|python)?", "", clean)
    clean = clean.replace("```", "")
    return clean.strip()


def _extract_candidate_json_arrays(text):
    """
    从文本中提取所有平衡的 JSON 数组片段。
    """
    clean = _remove_thinking_blocks(text)

    arrays = []
    starts = [i for i, ch in enumerate(clean) if ch == "["]

    for start in starts:
        depth = 0
        in_str = False
        escape = False

        for i in range(start, len(clean)):
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
                    frag = clean[start:i + 1]
                    if '"match"' in frag or "'match'" in frag or "match" in frag:
                        arrays.append(frag)
                    break

    arrays.sort(key=len, reverse=True)
    return arrays


def _try_json_loads_relaxed(s):
    if not s:
        return None

    attempts = [s]

    if '\\"' in s:
        attempts.append(s.replace('\\"', '"'))

    for x in attempts:
        x = x.strip()
        try:
            return json.loads(x)
        except Exception:
            pass

        try:
            last_brace = x.rfind("}")
            if last_brace != -1:
                repaired = x[:last_brace + 1] + "]"
                return json.loads(repaired)
        except Exception:
            pass

    return None


def _normalize_top3(top3):
    out = []

    if not isinstance(top3, list):
        return out

    for item in top3:
        if isinstance(item, dict):
            score = _normalize_score_text(item.get("score", ""))
            prob = _f(item.get("prob", item.get("probability", 0)), 0)

            if _parse_score(score)[0] is not None:
                out.append({
                    "score": score,
                    "prob": prob
                })

        elif isinstance(item, str):
            score = _normalize_score_text(item)
            if _parse_score(score)[0] is not None:
                out.append({
                    "score": score,
                    "prob": 0
                })

    return out


def _validate_and_fix_ai_item(item):
    if not isinstance(item, dict):
        return None

    try:
        mid = int(item.get("match"))
    except Exception:
        return None

    top3 = _normalize_top3(item.get("top3", []))

    if not top3 and item.get("score"):
        score = _normalize_score_text(item.get("score"))
        if _parse_score(score)[0] is not None:
            top3 = [{
                "score": score,
                "prob": _f(item.get("prob", 0))
            }]

    if not top3:
        return None

    final_dir = item.get("final_direction", "")
    if final_dir not in ["home", "draw", "away"]:
        final_dir = _score_direction(top3[0]["score"]) or ""

    if final_dir in ["home", "draw", "away"]:
        top1_dir = _score_direction(top3[0]["score"])

        if top1_dir != final_dir:
            aligned = None
            for cand in top3:
                if _score_direction(cand["score"]) == final_dir:
                    aligned = cand
                    break

            if aligned:
                top3 = [aligned] + [x for x in top3 if x is not aligned]
            else:
                final_dir = top1_dir or final_dir

    top1 = top3[0]["score"]

    return mid, {
        "top3": top3[:3],
        "ai_score": top1,
        "reason": str(item.get("reason", item.get("analysis", "")))[:1000],
        "ai_confidence": int(_f(item.get("ai_confidence", item.get("confidence", 60)), 60)),
        "is_score_others": bool(item.get("is_score_others", any("其他" in x["score"] for x in top3))),
        "detected_traps": item.get("detected_traps", []),
        "final_direction": final_dir,
    }


def _parse_ai_json(raw_text, num_matches):
    """解析 AI 返回的 JSON,返回 {match_id: result_dict}"""
    clean = _remove_thinking_blocks(raw_text)
    results = {}

    candidates = _extract_candidate_json_arrays(clean)

    if not candidates:
        start = clean.find("[")
        end = clean.rfind("]") + 1
        if start != -1 and end > start:
            candidates = [clean[start:end]]

    for json_str in candidates:
        arr = _try_json_loads_relaxed(json_str)

        if not isinstance(arr, list):
            continue

        temp_results = {}

        for item in arr:
            parsed = _validate_and_fix_ai_item(item)
            if parsed:
                mid, obj = parsed
                if 1 <= mid <= max(num_matches, 1):
                    temp_results[mid] = obj

        if temp_results:
            print(f"    🎯 JSON解析成功: {len(temp_results)}/{num_matches}")
            results = temp_results
            break

    if not results:
        print(f"    ⚠️ JSON解析失败。响应片段: {_debug_short(clean, 800)}")

    return results


async def async_call_one_ai_batch(session, prompt, url_env, key_env, model_name,
                                  num_matches, ai_name):
    """
    单AI调用。
    约束:
      - 不拆批
      - 不切换模型
      - 已经产生200响应后，不因解析失败再次请求，避免重复烧钱
      - 只允许连接失败/HTTP 5xx/429 时换备用URL，不换模型
    """
    key = get_clean_env_key(key_env)

    if not key and ai_name == "gpt":
        key = GPT_DEFAULT_KEY

    if not key:
        return ai_name, {}, "no_key"

    model_name = str(model_name).strip()
    if not model_name:
        return ai_name, {}, "no_model"

    if ai_name == "gpt":
        primary_url = get_clean_env_url(url_env, GPT_DEFAULT_URL)
        if not primary_url or "poloai" not in primary_url:
            primary_url = GPT_DEFAULT_URL
        urls = [primary_url]
        print(f"    🔌 [GPT] 使用固定通道: {primary_url}")
    else:
        primary_url = get_clean_env_url(url_env)
        backup = [u for u in FALLBACK_URLS if u and u != primary_url][:1]
        urls = [primary_url] + backup

    CONNECT_TIMEOUT = 20
    READ_TIMEOUT_MAP = {
        "claude": 400,
        "grok": 300,
        "gpt": 300,
        "gemini": 300
    }
    READ_TIMEOUT = READ_TIMEOUT_MAP.get(ai_name, 300)

    AI_PROFILES = {
        "claude": {
            "sys": (
                "<role>你是顶级对冲基金的博弈论+市场微观结构首席分析师。</role>\n"
                "<priority>严格执行用户消息中的 anti_hallucination_rules 与 iron_rules。</priority>\n"
                "<style>逆向思维优先，但不得编造任何 match_data 未提供的信息。</style>\n"
                "<instruction>按5步思维锚推理，最终仅输出 JSON 数组。禁止前缀后缀。</instruction>"
            ),
            "temp": 0.18
        },
        "gpt": {
            "sys": (
                "<role>你是衍生品定价+概率分布偏差量化策略师。</role>\n"
                "<priority>严格遵守 anti_hallucination_rules，尤其不得编造伤停、新闻、排名、赛程。</priority>\n"
                "<style>从a0-a7和CRS矩阵反推真实λ。只信输入数据，不信叙事。</style>\n"
                "<instruction>严格输出JSON数组，禁止任何前缀后缀。</instruction>"
            ),
            "temp": 0.15
        },
        "grok": {
            "sys": (
                "<role>你是另类数据与情绪背离分析师。</role>\n"
                "<priority>严格遵守 anti_hallucination_rules。没有给出的信息一律视为未知。</priority>\n"
                "<style>捕捉散户热度、资金流与赔率背离，但不得无数据爆冷。</style>\n"
                "<instruction>只输出JSON数组。</instruction>"
            ),
            "temp": 0.22
        },
        "gemini": {
            "sys": (
                "<role>你是精通非线性特征的深度学习模式识别引擎。</role>\n"
                "<priority>严格遵守 anti_hallucination_rules 与 iron_rules。</priority>\n"
                "<style>检测欧赔/亚盘/CRS/进球数之间的定价裂痕。只有多指标同向才增强结论。</style>\n"
                "<instruction>综合输出最稳健预测，仅JSON数组。</instruction>"
            ),
            "temp": 0.12
        },
    }

    profile = AI_PROFILES.get(ai_name, AI_PROFILES["gpt"])

    for base_url in urls:
        if not base_url:
            continue

        is_gem = "generateContent" in base_url
        url = base_url.rstrip("/")

        if not is_gem and "chat/completions" not in url:
            url += "/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        if is_gem:
            headers["x-goog-api-key"] = key
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": profile["temp"]
                },
                "systemInstruction": {
                    "parts": [
                        {
                            "text": profile["sys"]
                        }
                    ]
                }
            }
        else:
            headers["Authorization"] = f"Bearer {key}"
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": profile["sys"]
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": profile["temp"]
            }

        gw = url.split("/v1")[0][:45]
        print(f"  [🔌连接中] {ai_name.upper()} | 固定模型={model_name[:36]} @ {gw}")
        t0 = time.time()
        connected = False

        try:
            timeout = aiohttp.ClientTimeout(
                total=None,
                connect=CONNECT_TIMEOUT,
                sock_connect=CONNECT_TIMEOUT,
                sock_read=READ_TIMEOUT
            )

            async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                elapsed_connect = round(time.time() - t0, 1)

                if r.status in (500, 502, 503, 504):
                    print(f"    ⚠️ HTTP {r.status} | {elapsed_connect}s → 换备用URL，不换模型")
                    continue

                if r.status == 429:
                    print(f"    🔥 429 | {elapsed_connect}s → 换备用URL，不换模型")
                    await asyncio.sleep(1)
                    continue

                if r.status != 200:
                    try:
                        err_text = await r.text()
                    except Exception:
                        err_text = ""
                    print(f"    ❌ HTTP {r.status} | 不重试模型 | {err_text[:200]}")
                    return ai_name, {}, f"http_{r.status}"

                connected = True
                print(f"    ✅ 已连上!{elapsed_connect}s | 等待数据...")

                try:
                    data = await r.json(content_type=None)
                except Exception:
                    txt = await r.text()
                    data = {
                        "_raw_text": txt
                    }

                elapsed = round(time.time() - t0, 1)

                usage = data.get("usage", {}) if isinstance(data, dict) else {}
                req_tokens = usage.get("total_tokens", 0) or (
                    usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                )

                if not req_tokens and isinstance(data, dict):
                    um = data.get("usageMetadata", {})
                    req_tokens = um.get("totalTokenCount", 0)

                if req_tokens:
                    print(f"    📊 {req_tokens:,} token | {elapsed}s")

                raw_text = _extract_response_text(data, is_gem, ai_name)

                if not raw_text or len(raw_text) < 10:
                    print("    ⚠️ 空数据 → 不切换模型，不重复请求")
                    _save_debug_dump(ai_name, data, "empty")
                    return ai_name, {}, "empty"

                results = _parse_ai_json(raw_text, num_matches)

                if len(results) > 0:
                    print(f"    ✅ {ai_name.upper()} 完成: {len(results)}/{num_matches} | {elapsed}s")
                    return ai_name, results, model_name

                print("    ⚠️ 解析0条 → 不切换模型，不重复请求")
                _save_debug_dump(ai_name, data, "parse0")
                return ai_name, {}, "parse0"

        except aiohttp.ClientConnectorError:
            print("    🔌 连接失败 → 换备用URL，不换模型")
            continue

        except asyncio.TimeoutError:
            if not connected:
                print("    🔌 连接超时 → 换备用URL，不换模型")
                continue

            print("    ⏰ 读取超时 | 已产生请求，不切换模型")
            return ai_name, {}, "read_timeout"

        except Exception as e:
            print(f"    ⚠️ {type(e).__name__}: {str(e)[:120]} | 不切换模型")
            return ai_name, {}, "error"

    return ai_name, {}, "all_urls_failed"


async def run_ai_matrix_two_phase(match_analyses):
    """
    AI矩阵入口。
    约束:
      - 不拆批：所有比赛一次 prompt
      - 不切换模型：每个AI只用一个固定模型
    """
    num = len(match_analyses)
    prompt = build_v18_prompt(match_analyses)

    print(f"  [v18.2 Prompt] {len(prompt):,} 字符 → 4AI并行 | 不拆批 | 固定模型")

    ai_configs = [
        (
            "grok",
            "GROK_API_URL",
            "GROK_API_KEY",
            os.environ.get("GROK_MODEL", "熊猫-A-6-grok-4.2-thinking")
        ),
        (
            "gpt",
            "GPT_API_URL",
            "GPT_API_KEY",
            os.environ.get("GPT_MODEL", "gpt-5.5")
        ),
        (
            "gemini",
            "GEMINI_API_URL",
            "GEMINI_API_KEY",
            os.environ.get("GEMINI_MODEL", "熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking")
        ),
        (
            "claude",
            "CLAUDE_API_URL",
            "CLAUDE_API_KEY",
            os.environ.get("CLAUDE_MODEL", "熊猫-69-满血openrouter-claude-opus-4.7")
        ),
    ]

    all_results = {
        "gpt": {},
        "grok": {},
        "gemini": {},
        "claude": {}
    }

    connector = aiohttp.TCPConnector(limit=10, use_dns_cache=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            async_call_one_ai_batch(session, prompt, u, k, model_name, num, n)
            for n, u, k, model_name in ai_configs
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


# ====================================================================
# 🌟 merge_result_v18
# ====================================================================

def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r,
                 stats, match_obj):
    if isinstance(match_obj.get("v2_odds_dict"), dict):
        v2 = match_obj["v2_odds_dict"]
        match_obj = {
            **match_obj,
            **v2
        }
        print(f"    🔧 [字段兼容] v2_odds_dict→顶层 ({len(v2)}个字段)")

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

    abstained = [
        n.upper()
        for n, v in ai_valid.items()
        if not v
    ]

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

    exp_goals = 0.0

    for src in [engine_result, stats]:
        if not src:
            continue

        for k in [
            "expected_total_goals",
            "exp_goals",
            "total_goals",
            "expected_goals",
            "lambda_total",
            "total_xg"
        ]:
            v = src.get(k)

            if v is not None:
                try:
                    fv = float(v)
                    if fv > 0.5:
                        exp_goals = fv
                        break
                except Exception:
                    pass

        if exp_goals > 0:
            break

    if exp_goals <= 0:
        hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0)) if engine_result else 0
        axg = _f(engine_result.get("bookmaker_implied_away_xg", 0)) if engine_result else 0

        if hxg > 0 and axg > 0:
            exp_goals = hxg + axg
            print(f"    📐 期望进球用xG总和: {hxg:.2f}+{axg:.2f}={exp_goals:.2f}")

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
                print(f"    📐 期望进球用a0-a7反推: {exp_goals:.2f}")
        except Exception:
            pass

    if exp_goals < 1.0 or exp_goals > 6.0:
        print(f"    ⚠️ 期望进球异常({exp_goals:.2f}),使用默认2.5")
        exp_goals = 2.5

    smart_signals = stats.get("smart_signals", []) if stats else []

    trap_report = detect_all_traps(
        match_obj,
        engine_result or {},
        ai_responses,
        smart_signals,
        exp_goals
    )

    if trap_report["trap_count"] > 0:
        print(f"    🎭 陷阱: {trap_report['trap_count']}个 严重度{trap_report['total_severity']}")
        for t in trap_report["traps_detected"][:4]:
            print(f"       [{t['trap']}] {t['description'][:70]}")

    crs_analysis = analyze_crs_matrix(match_obj)

    if crs_analysis["coverage"] > 0:
        print(f"    📊 CRS: 覆盖{crs_analysis['coverage']*100:.0f}% 形状={crs_analysis['shape_verdict']}")

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
        f"    🎯 方向: 主{lock_result['home_win_pct']:.0f}% "
        f"平{lock_result['draw_pct']:.0f}% 客{lock_result['away_win_pct']:.0f}%"
    )

    for ev in lock_result["evidences"][:3]:
        print(f"       - {ev}")

    predicted_score = lock_result["predicted_score"]
    predicted_label = lock_result["predicted_label"]
    result_cn = lock_result["result"]
    display_direction = lock_result["display_direction"]
    final_direction = lock_result["final_direction"]

    home_win_pct = lock_result["home_win_pct"]
    draw_pct = lock_result["draw_pct"]
    away_win_pct = lock_result["away_win_pct"]

    is_score_others = lock_result["is_score_others"]
    scenario = lock_result["scenario"]
    dir_confidence = lock_result["dir_confidence"]
    dir_gap = lock_result["dir_gap"]

    target_crs = CRS_FULL_MAP.get(predicted_score, "")
    final_odds = _f(match_obj.get(target_crs, 0))

    if not final_odds and is_score_others:
        if final_direction == "home":
            final_odds = _f(match_obj.get("crs_win", 0))
        elif final_direction == "away":
            final_odds = _f(match_obj.get("crs_lose", 0))
        else:
            final_odds = _f(match_obj.get("crs_same", 0))

    crs_prob = crs_analysis.get("implied_probs", {}).get(predicted_score, 5)
    ev_data = calculate_value_bet(crs_prob, final_odds)

    engine_conf = engine_result.get("confidence", 50) if engine_result else 50

    weights = {
        "claude": 1.4,
        "gemini": 1.35,
        "grok": 1.30,
        "gpt": 1.1
    }

    ai_conf_sum = 0
    ai_conf_count = 0
    value_kills = 0

    for name, r in ai_responses.items():
        conf = _f(r.get("ai_confidence", 60), 60)
        ai_conf_sum += conf * weights.get(name, 1.0)
        ai_conf_count += weights.get(name, 1.0)

        if r.get("value_kill"):
            value_kills += 1

    avg_ai_conf = (ai_conf_sum / ai_conf_count) if ai_conf_count > 0 else 60

    cf = min(95, engine_conf + int((avg_ai_conf - 60) * 0.4)) + value_kills * 6
    cf -= trap_report.get("confidence_penalty", 0)

    if dir_confidence >= 70:
        cf += min(10, int((dir_confidence - 70) // 3))
    elif dir_confidence < 50:
        cf -= 8

    if dir_gap < 10:
        cf -= 5

    cf = max(30, min(95, cf))
    risk = "低" if cf >= 75 else ("中" if cf >= 55 else "高")

    cold_strength = 0
    cold_level = None
    cold_signals_arr = []

    for t in trap_report["traps_detected"]:
        if t["trap"] in [
            "T8_FALSE_COLD",
            "T4_FAKE_HOME_FAVORITE",
            "T5_FAKE_AWAY_FAVORITE",
            "T14_CUP_FAVORITE"
        ]:
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
        "dark_verdict": f"❄️ {cold_level}冷门!{len(cold_signals_arr)}条触发" if cold_level else ""
    }

    sigs = list(smart_signals)

    for t in trap_report["traps_detected"]:
        sigs.append(f"🎭 {t['trap']}:{t['description'][:50]}")

    if is_score_others:
        sigs.append("🔥 胜其他场触发")

    if lock_result.get("override_triggered"):
        sigs.append("⚡ Sharp Override 触发")

    cl_raw = claude_r.get("ai_score", "") if isinstance(claude_r, dict) else ""
    cl_sc = cl_raw if _parse_score(cl_raw)[0] is not None else predicted_score

    return {
        "predicted_score": predicted_score,
        "predicted_label": predicted_label,
        "result": result_cn,
        "display_direction": display_direction,
        "final_direction": final_direction,
        "is_score_others": is_score_others,

        "home_win_pct": round(home_win_pct, 1),
        "draw_pct": round(draw_pct, 1),
        "away_win_pct": round(away_win_pct, 1),

        "confidence": cf,
        "risk_level": risk,
        "dir_confidence": dir_confidence,
        "dir_gap": dir_gap,

        "scenario": scenario,
        "goal_range": lock_result["goal_range"],

        "bayesian_evidences": lock_result["evidences"],
        "bayesian_prior": lock_result["bayesian_prior"],
        "override_triggered": lock_result["override_triggered"],

        "traps_detected": [t["trap"] for t in trap_report["traps_detected"]],
        "trap_count": trap_report["trap_count"],
        "trap_severity": trap_report["total_severity"],
        "trap_details": [
            {
                "trap": t["trap"],
                "desc": t["description"]
            }
            for t in trap_report["traps_detected"]
        ],

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

        "smart_money_signal": " | ".join(sigs[:10]),
        "smart_signals": sigs,

        "cold_door": cold_door,

        "xG_home": round(_f(engine_result.get("bookmaker_implied_home_xg", 1.3)) if engine_result else 1.3, 2),
        "xG_away": round(_f(engine_result.get("bookmaker_implied_away_xg", 0.9)) if engine_result else 0.9, 2),
        "over_under_2_5": "大" if (engine_result.get("over_25", 50) if engine_result else 50) > 55 else "小",
        "both_score": "是" if (engine_result.get("btts", 45) if engine_result else 45) > 50 else "否",
        "expected_total_goals": round(exp_goals, 2),
        "over_2_5": engine_result.get("over_25", 50) if engine_result else 50,
        "btts": engine_result.get("btts", 45) if engine_result else 45,
        "bookmaker_implied_home_xg": engine_result.get("bookmaker_implied_home_xg", "?") if engine_result else "?",
        "bookmaker_implied_away_xg": engine_result.get("bookmaker_implied_away_xg", "?") if engine_result else "?",
        "sharp_detected": trap_report.get("sharp_detected", False),
        "sharp_dir": trap_report.get("sharp_dir"),
        "shin_dir": max(trap_report["shin"], key=trap_report["shin"].get),
        "model_consensus": stats.get("model_consensus", 0) if stats else 0,
        "total_models": stats.get("total_models", 11) if stats else 11,
        "extreme_warning": engine_result.get("scissors_gap_signal", "") if engine_result else "",

        "refined_poisson": stats.get("refined_poisson", {}) if stats else {},
        "poisson": {},
        "elo": stats.get("elo", {}) if stats else {},
        "random_forest": stats.get("random_forest", {}) if stats else {},
        "gradient_boost": stats.get("gradient_boost", {}) if stats else {},
        "neural_net": stats.get("neural_net", {}) if stats else {},
        "logistic": stats.get("logistic", {}) if stats else {},
        "svm": stats.get("svm", {}) if stats else {},
        "knn": stats.get("knn", {}) if stats else {},
        "dixon_coles": stats.get("dixon_coles", {}) if stats else {},
        "bradley_terry": stats.get("bradley_terry", {}) if stats else {},
        "home_form": stats.get("home_form", {}) if stats else {},
        "away_form": stats.get("away_form", {}) if stats else {},
        "handicap_signal": stats.get("handicap_signal", "") if stats else "",
        "odds_movement": stats.get("odds_movement", {}) if stats else {},
        "vote_analysis": stats.get("vote_analysis", {}) if stats else {},
        "h2h_blood": stats.get("h2h_blood", {}) if stats else {},
        "crs_analysis": stats.get("crs_analysis", {}) if stats else {},
        "ttg_analysis": stats.get("ttg_analysis", {}) if stats else {},
        "halftime": stats.get("halftime", {}) if stats else {},
        "pace_rating": stats.get("pace_rating", "") if stats else "",
        "kelly_home": stats.get("kelly_home", {}) if stats else {},
        "kelly_away": stats.get("kelly_away", {}) if stats else {},
        "odds": stats.get("odds", {}) if stats else {},
        "experience_analysis": stats.get("experience_analysis", {}) if stats else {},
        "pro_odds": stats.get("pro_odds", {}) if stats else {},
        "asian_handicap_probs": stats.get("asian_handicap_probs", {}) if stats else {},
        "top_scores": stats.get("refined_poisson", {}).get("top_scores", []) if stats else [],

        "engine_version": "vMAX 18.2",
        "engine_architecture": "贝叶斯后验+16维陷阱矩阵+防幻觉AI+固定模型调用+决策锁定链",
    }


# ====================================================================
# Top4 精选
# ====================================================================

def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})

        s = pr.get("confidence", 0) * 0.4
        s += pr.get("dir_confidence", 50) * 0.15

        trap_count = pr.get("trap_count", 0)
        if trap_count >= 2:
            s += 8
        elif trap_count >= 1:
            s += 4

        ev = pr.get("edge_vs_market", 0)
        if ev >= 30:
            s += 12
        elif ev >= 15:
            s += 6

        cold = pr.get("cold_door", {})
        if cold.get("is_cold_door") and pr.get("confidence", 0) >= 60:
            s += 5

        if pr.get("risk_level") == "高":
            s -= 10
        elif pr.get("risk_level") == "低":
            s += 8

        if pr.get("is_score_others"):
            s += 10

        if pr.get("dir_gap", 0) < 8:
            s -= 5

        exp_info = pr.get("experience_analysis", {})
        exp_score = exp_info.get("total_score", 0)

        if exp_score >= 15 and pr.get("result") == "平局" and exp_info.get("draw_rules", 0) >= 3:
            s += 12
        elif exp_score >= 10:
            s += 5

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
        "天": 7000
    }

    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))

    return base + int(nums[0]) if nums else 9999


# ====================================================================
# 🔒 四字段强一致性终极校验
# ====================================================================

def _enforce_consistency(mg):
    score_str = _normalize_score_text(mg.get("predicted_score", "1-1"))

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
            expected_code = {
                "主胜": "home",
                "平局": "draw",
                "客胜": "away"
            }.get(expected_dir, "draw")
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

    pcts = {
        "home": _f(mg.get("home_win_pct", 33.3), 33.3),
        "draw": _f(mg.get("draw_pct", 33.3), 33.3),
        "away": _f(mg.get("away_win_pct", 33.3), 33.3)
    }

    pct_argmax = max(pcts, key=pcts.get)

    if pct_argmax != expected_code:
        cur_max = pcts[pct_argmax]
        pcts[expected_code] = cur_max + 5
        total = sum(pcts.values())

        if total > 0:
            mg["home_win_pct"] = round(pcts["home"] / total * 100, 1)
            mg["draw_pct"] = round(pcts["draw"] / total * 100, 1)
            mg["away_win_pct"] = round(pcts["away"] / total * 100, 1)

    if "胜其他" in score_str or score_str == "9-0":
        mg["predicted_label"] = "胜其他"
        mg["predicted_score"] = "胜其他"
        mg["is_score_others"] = True

    elif "平其他" in score_str or score_str == "9-9":
        mg["predicted_label"] = "平其他"
        mg["predicted_score"] = "平其他"
        mg["is_score_others"] = True

    elif "负其他" in score_str or score_str == "0-9":
        mg["predicted_label"] = "负其他"
        mg["predicted_score"] = "负其他"
        mg["is_score_others"] = True

    else:
        mg["predicted_score"] = score_str
        mg["predicted_label"] = score_str

    return mg


# ====================================================================
# 🚀 主入口 run_predictions
# ====================================================================

def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])

    print("\n" + "=" * 80)
    print(f"  [vMAX 18.2] 贝叶斯后验+16维陷阱矩阵+防幻觉AI+固定模型+决策锁定链 | {len(ms)} 场")
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
            exp_goals_prev = hxg + axg if (hxg and axg) else 2.5

        trap_preview = detect_all_traps(
            m,
            eng,
            {},
            sp.get("smart_signals", []) if sp else [],
            exp_goals_prev
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

    all_ai = {
        "claude": {},
        "gemini": {},
        "gpt": {},
        "grok": {}
    }

    if use_ai and match_analyses:
        print("  [v18.2 AI] 启动4AI并行 | 不拆批 | 不切换模型")
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
                    logger.error(f"AI矩阵并发执行崩溃: {e}")
                    all_ai = {
                        "claude": {},
                        "gemini": {},
                        "gpt": {},
                        "grok": {}
                    }
        else:
            all_ai = asyncio.run(run_ai_matrix_two_phase(match_analyses))

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
            m
        )

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

        res.append({
            **m,
            "prediction": mg
        })

        trap_tag = f" [🎭{mg['trap_count']}陷阱]" if mg.get("trap_count", 0) > 0 else ""
        others_tag = " [🔥胜其他]" if mg.get("is_score_others") else ""
        sharp_tag = " [💰Sharp]" if mg.get("sharp_detected") else ""
        override_tag = " [⚡Override]" if mg.get("override_triggered") else ""
        scenario_tag = f" [{mg.get('scenario', 'normal')}]"

        print(
            f"  [{i+1}] {m.get('home_team')} vs {m.get('away_team')} => "
            f"{mg['result']} ({mg['predicted_score']}={mg['predicted_label']}) | "
            f"CF: {mg['confidence']}% | 方向: {mg['dir_confidence']:.0f}%"
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
        f"vMAX18.2 | {total_traps}陷阱 {others_count}胜其他 "
        f"{sharp_count}Sharp | 贝叶斯+16维+固定模型+防幻觉"
    )

    save_ai_diary(diary)

    return res, t4


if __name__ == "__main__":
    logger.info("vMAX 18.2 启动")
    print("✅ vMAX 18.2 贝叶斯后验+16维陷阱矩阵+防幻觉AI+固定模型调用 加载完成")
    print("   架构: 16维陷阱矩阵 + CRS矩阵几何 + 贝叶斯后验 + 决策锁定")
    print("   AI: 不拆批，不切换模型，解析失败不重复请求")
    print("   一致性: predicted_score ↔ result ↔ display_direction ↔ 概率argmax")