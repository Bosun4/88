# ====================================================================
# 🚀 vMAX 19.0 — AI-First 架构(三家分析+Claude仲裁)
# --------------------------------------------------------------------
# 与 v18.x 的根本差异:
#   ❌ 删除 贝叶斯后验/决策锁定链/概率篡改/CRS矩阵几何/Shin/TTG锚点权重
#   ❌ 删除 引擎自己做决策的所有逻辑
#   ✅ 引擎只负责: 抓包格式化 + 陷阱事实标记 + ensemble信号注入
#   ✅ Phase 1: GPT/Grok/Gemini 三家批量并行,各自独立分析
#   ✅ Phase 2: Claude 拿到三家结论 + 原始数据 做最终仲裁
#   ✅ 最终输出直接采用 Claude,引擎不后处理
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
    def apply_odds_history(m, mg): return mg

try:
    from quant_edge import apply_quant_edge
except ImportError:
    def apply_quant_edge(m, mg): return mg

try:
    from wencai_intel import apply_wencai_intel
except ImportError:
    def apply_wencai_intel(m, mg): return mg

try:
    ensemble = EnsemblePredictor()
    exp_engine = ExperienceEngine()
except ImportError:
    ensemble = None
    exp_engine = None


# ====================================================================
# 通用工具
# ====================================================================

STANDARD_GOAL_ODDS = {
    0: 9.5, 1: 5.5, 2: 3.5, 3: 4.0,
    4: 7.0, 5: 14.0, 6: 30.0, 7: 70.0,
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

HFTF_MAP = {
    "ss": "主/主", "sp": "主/平", "sf": "主/负",
    "ps": "平/主", "pp": "平/平", "pf": "平/负",
    "fs": "负/主", "fp": "负/平", "ff": "负/负",
}

CUP_KEYWORDS = ["杯", "淘汰", "决赛", "半决赛", "四分之一", "欧冠", "欧联", "国王杯", "足总杯", "联赛杯"]


def _f(v, default=0.0):
    try:
        return float(v) if v is not None and str(v).strip() != "" else default
    except:
        return default


def _parse_score(s: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        s_str = str(s).strip().replace(" ", "").replace("：", "-").replace(":", "-")
        if "胜" in s_str and "其他" in s_str: return 9, 0
        if "平" in s_str and "其他" in s_str: return 9, 9
        if "负" in s_str and "其他" in s_str: return 0, 9
        if s_str in ["主胜", "客胜", "平局"]:
            return None, None
        p = s_str.split("-")
        if len(p) != 2:
            return None, None
        return int(p[0]), int(p[1])
    except:
        return None, None


def _score_direction(score_str: str) -> Optional[str]:
    h, a = _parse_score(score_str)
    if h is None:
        return None
    if "胜其他" in score_str or score_str == "9-0":
        return "home"
    if "平其他" in score_str or score_str == "9-9":
        return "draw"
    if "负其他" in score_str or score_str == "0-9":
        return "away"
    if h > a: return "home"
    if h < a: return "away"
    return "draw"


def _infer_theoretical_handicap(sp_h: float, sp_a: float) -> float:
    """从欧赔反推理论亚盘让球深度,返回主队让球(>0=主让,<0=客让)"""
    if sp_h <= 1.01 or sp_a <= 1.01:
        return 0.0
    ratio = sp_a / sp_h
    if ratio >= 8.0: return 2.75
    if ratio >= 5.5: return 2.25
    if ratio >= 4.0: return 1.75
    if ratio >= 3.0: return 1.25
    if ratio >= 2.2: return 0.75
    if ratio >= 1.6: return 0.25
    if ratio >= 1.15: return 0.25
    if ratio >= 0.85: return 0.0
    if ratio >= 0.63: return -0.25
    if ratio >= 0.46: return -0.75
    if ratio >= 0.33: return -1.25
    if ratio >= 0.25: return -1.75
    if ratio >= 0.18: return -2.25
    return -2.75


def _parse_actual_handicap(match_obj: Dict) -> float:
    """解析实际让球值,主让为正,客让为负"""
    raw = match_obj.get("give_ball", match_obj.get("handicap", "0"))
    s = str(raw).strip()
    s = s.replace("主", "").replace("客", "").replace("受让", "+").replace("让", "-")
    if "/" in s:
        parts = s.split("/")
        try:
            val = (_f(parts[0].strip()) + _f(parts[1].strip())) / 2.0
            return -val
        except:
            pass
    return -_f(s, 0.0)


def _compute_shin(match_obj: Dict) -> Dict[str, float]:
    """从欧赔计算 Shin 概率(去抽水)"""
    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_d = _f(match_obj.get("sp_draw", match_obj.get("same", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))
    if sp_h > 1 and sp_d > 1 and sp_a > 1:
        margin = 1/sp_h + 1/sp_d + 1/sp_a
        return {
            "home": (1/sp_h) / margin * 100,
            "draw": (1/sp_d) / margin * 100,
            "away": (1/sp_a) / margin * 100,
        }
    return {"home": 33.3, "draw": 33.3, "away": 33.3}


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
            except:
                pass
    return 0, 0, 0


def _extract_avg_goals(text: str) -> Tuple[float, float]:
    if not text:
        return 0.0, 0.0
    h_match = re.search(r"场均进球[^0-9]*(\d+\.?\d*)", str(text))
    a_match = re.search(r"场均失球[^0-9]*(\d+\.?\d*)", str(text))
    return (
        float(h_match.group(1)) if h_match else 0.0,
        float(a_match.group(1)) if a_match else 0.0,
    )


# ====================================================================
# 🎭 16 维陷阱事实标记(只输出文字,不计算权重)
# ====================================================================

def detect_traps_as_facts(match_obj: Dict, engine_result: Dict,
                          smart_signals: List, exp_goals: float) -> List[str]:
    """
    返回陷阱触发的客观事实文字列表,供 AI 自由参考。
    每条陷阱只描述事实,不输出 direction_adjust 或 score_multipliers。
    """
    facts = []
    shin = _compute_shin(match_obj)
    
    # ---------- T1: 平赔独降 ----------
    change = match_obj.get("change", {}) or {}
    cw = _f(change.get("win", 0))
    cs = _f(change.get("same", 0))
    cl = _f(change.get("lose", 0))
    
    if cs <= -0.04 and cs < cw and cs < cl:
        strong_shin = max(shin["home"], shin["away"])
        if strong_shin >= 34:
            strong_cn = "主" if shin["home"] > shin["away"] else "客"
            facts.append(
                f"[T1 诱平赔] 平赔独降 {cs:.2f},{strong_cn}方 Shin={strong_shin:.1f}%。"
                f"庄家可能在诱散户买平,真实立场偏{strong_cn}。"
            )
    
    # ---------- T2/T3: 让球深度 vs 理论让球 ----------
    sp_h = _f(match_obj.get("sp_home", match_obj.get("win", 0)))
    sp_a = _f(match_obj.get("sp_away", match_obj.get("lose", 0)))
    if sp_h > 1.05 and sp_a > 1.05:
        theoretical = _infer_theoretical_handicap(sp_h, sp_a)
        actual = _parse_actual_handicap(match_obj)
        diff = actual - theoretical
        if abs(diff) >= 0.5:
            if diff >= 0.5:
                facts.append(
                    f"[T2 让球偏深] 实际让球 {actual:+.2f}, 理论应让 {theoretical:+.2f}, "
                    f"庄家比欧赔显示的更看好主队 {abs(diff):.2f} 球。"
                )
            else:
                facts.append(
                    f"[T3 让球偏浅] 实际让球 {actual:+.2f}, 理论应让 {theoretical:+.2f}, "
                    f"庄家比欧赔显示的更不看好主队 {abs(diff):.2f} 球。"
                )
    
    # ---------- T4/T5: 虚假强势(Shin 强但基本面弱) ----------
    info_src = match_obj.get("points", {}) or {}
    h_txt = str(info_src.get("home_strength", ""))
    a_txt = str(info_src.get("guest_strength", ""))
    h_w, h_d, h_l = _extract_form_record(h_txt)
    a_w, a_d, a_l = _extract_form_record(a_txt)
    h_total = h_w + h_d + h_l
    a_total = a_w + a_d + a_l
    
    if shin["home"] > 48 and h_total >= 3 and a_total >= 3:
        h_wr = h_w / h_total
        a_wr = a_w / a_total
        if h_wr < 0.40 and a_wr > 0.50:
            facts.append(
                f"[T4 诱主胜] 主队 Shin {shin['home']:.1f}% 但近期胜率仅 {h_wr:.0%}({h_w}胜{h_d}平{h_l}负),"
                f"客队胜率 {a_wr:.0%}。庄家可能在诱主胜。"
            )
    
    if shin["away"] > 48 and h_total >= 3 and a_total >= 3:
        h_wr = h_w / h_total
        a_wr = a_w / a_total
        if a_wr < 0.40 and h_wr > 0.50:
            facts.append(
                f"[T5 诱客胜] 客队 Shin {shin['away']:.1f}% 但近期胜率仅 {a_wr:.0%}({a_w}胜{a_d}平{a_l}负),"
                f"主队胜率 {h_wr:.0%}。庄家可能在诱客胜。"
            )
    
    # ---------- T6/T7: 进球区间陷阱 ----------
    a0 = _f(match_obj.get("a0", 999), 999)
    a1 = _f(match_obj.get("a1", 999), 999)
    a2 = _f(match_obj.get("a2", 999), 999)
    a5 = _f(match_obj.get("a5", 999), 999)
    a6 = _f(match_obj.get("a6", 999), 999)
    a7 = _f(match_obj.get("a7", 999), 999)
    
    low_small = sum([0 < a0 < 8.0, 0 < a1 < 4.5, 0 < a2 < 3.0])
    if low_small >= 2 and exp_goals >= 2.8:
        facts.append(
            f"[T6 诱小球] 0/1/2 球赔率压低 {low_small}/3,但 λ={exp_goals:.2f} 偏高。"
            f"庄家压小球但实际可能是中大比分。"
        )
    
    low_large = sum([0 < a5 < 10, 0 < a6 < 16, 0 < a7 < 30])
    if low_large >= 2 and exp_goals <= 2.3:
        facts.append(
            f"[T7 诱大球] 5/6/7 球赔率压低 {low_large}/3,但 λ={exp_goals:.2f} 偏低。"
            f"庄家压大球但实际可能是中小比分。"
        )
    
    # ---------- T8: 假冷门 ----------
    sigs_str = " ".join(str(s) for s in smart_signals)
    cold_triggers = sum([
        "坏消息" in sigs_str, "崩盘" in sigs_str, "造热" in sigs_str,
        "背离" in sigs_str, "盘口太便宜" in sigs_str
    ])
    
    vote = match_obj.get("vote", {}) or {}
    vh = int(_f(vote.get("win", 33), 33))
    va = int(_f(vote.get("lose", 33), 33))
    if cold_triggers >= 2:
        hot_dir = "home" if vh >= 58 else ("away" if va >= 58 else None)
        if hot_dir == "home" and h_total >= 3 and (h_w/h_total) > 0.55:
            facts.append(
                f"[T8 假冷门] 散户热推主胜({vh}%) + 主队基本面真强(胜率{h_w/h_total:.0%}),"
                f"看似冷门信号但主胜可能就是真相。"
            )
        elif hot_dir == "away" and a_total >= 3 and (a_w/a_total) > 0.55:
            facts.append(
                f"[T8 假冷门] 散户热推客胜({va}%) + 客队基本面真强(胜率{a_w/a_total:.0%}),"
                f"看似冷门信号但客胜可能就是真相。"
            )
    
    # ---------- T9: 诱反指 ----------
    if vh >= 60 and cw < -0.04:
        facts.append(
            f"[T9 诱反指] 散户主胜 {vh}% + 主胜赔率降水 {cw:.2f}(钱跟散户),反指即自杀。"
        )
    elif va >= 60 and cl < -0.04:
        facts.append(
            f"[T9 诱反指] 散户客胜 {va}% + 客胜赔率降水 {cl:.2f}(钱跟散户),反指即自杀。"
        )
    
    # ---------- T10: 沉默盘 ----------
    total_move = abs(cw) + abs(cs) + abs(cl)
    crs_count = sum(1 for k in ["w10", "w20", "w21", "s00", "s11", "l01", "l02", "l12"]
                    if _f(match_obj.get(k, 0)) > 1)
    if total_move < 0.03 and crs_count < 6:
        facts.append(
            f"[T10 沉默盘] 总变盘 {total_move:.3f} + CRS 覆盖 {crs_count}/8,市场定价薄弱,信心降低。"
        )
    
    # ---------- T11: xG 背离 ----------
    hxg_book = _f(engine_result.get("bookmaker_implied_home_xg", 0))
    axg_book = _f(engine_result.get("bookmaker_implied_away_xg", 0))
    if hxg_book > 0 and axg_book > 0:
        h_for, _ = _extract_avg_goals(h_txt)
        a_for, _ = _extract_avg_goals(a_txt)
        divergences = []
        if h_for > 0 and abs(hxg_book - h_for) > 0.8:
            divergences.append(f"主xG 书={hxg_book:.2f} vs 场均={h_for:.2f}")
        if a_for > 0 and abs(axg_book - a_for) > 0.8:
            divergences.append(f"客xG 书={axg_book:.2f} vs 场均={a_for:.2f}")
        if len(divergences) >= 2:
            facts.append(f"[T11 xG背离] 庄家xG与场均严重背离: {'; '.join(divergences)}")
    
    # ---------- T12: 让球未开 ----------
    actual = _parse_actual_handicap(match_obj)
    if abs(actual) < 0.1 and sp_h > 1.01 and sp_a > 1.01:
        theoretical = _infer_theoretical_handicap(sp_h, sp_a)
        if abs(theoretical) >= 0.4:
            facts.append(
                f"[T12 让球未开] 让球字段为 0 但欧赔反推理论让 {theoretical:+.2f} 球,"
                f"庄家可能在隐藏真实预期。"
            )
    
    # ---------- T13: 闷平场景 ----------
    if hxg_book > 0 and axg_book > 0:
        total_xg = hxg_book + axg_book
        if total_xg < 2.3 and abs(shin["home"] - shin["away"]) <= 10:
            h_for, h_against = _extract_avg_goals(h_txt)
            a_for, a_against = _extract_avg_goals(a_txt)
            weak_attack = sum([0 < h_for < 1.4, 0 < a_for < 1.4])
            strong_def = sum([0 < h_against < 1.2, 0 < a_against < 1.2])
            small_compressed = sum([0 < a0 <= 10, 0 < a1 <= 5, 0 < a2 <= 3.5])
            if weak_attack + strong_def >= 2 and small_compressed >= 1:
                facts.append(
                    f"[T13 闷平] xG总{total_xg:.2f} + 弱攻{weak_attack}/强防{strong_def} + "
                    f"小球压低{small_compressed}/3,典型闷平场景。"
                )
    
    # ---------- T14: 杯赛大热必死 ----------
    league = str(match_obj.get("league", match_obj.get("cup", "")))
    is_cup = any(kw in league for kw in CUP_KEYWORDS)
    if is_cup:
        strong_shin = max(shin["home"], shin["away"])
        if strong_shin >= 55:
            strong_side = "主" if shin["home"] > shin["away"] else "客"
            strong_vote = vh if strong_side == "主" else va
            if strong_vote >= 50:
                facts.append(
                    f"[T14 杯赛大热] {league},{strong_side}方 Shin {strong_shin:.1f}% + 散户{strong_vote}%,"
                    f"淘汰赛大热必死规律下弱队反扑概率提升。"
                )
    
    # ---------- T15: 历史僵局 ----------
    info = match_obj.get("points", {}) or {}
    text_all = " ".join(str(v) for v in info.values() if v)
    h2h_match = re.search(r"对阵[^0-9]{0,20}(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]", text_all)
    if not h2h_match:
        h2h_match = re.search(r"历史交锋[^0-9]*(\d+)\s*胜\s*(\d+)\s*平\s*(\d+)\s*[负败]", text_all)
    if h2h_match:
        try:
            hw = int(h2h_match.group(1))
            hd = int(h2h_match.group(2))
            hl = int(h2h_match.group(3))
            total_h2h = hw + hd + hl
            if total_h2h >= 3:
                draw_rate = hd / total_h2h
                s11 = _f(match_obj.get("s11", 999), 999)
                if draw_rate >= 0.40 and 0 < s11 < 9.0:
                    facts.append(
                        f"[T15 历史僵局] 历史交锋 {hw}胜{hd}平{hl}负(平率 {draw_rate:.0%}) + "
                        f"1-1 赔率 {s11:.1f},容易再现平局。"
                    )
        except:
            pass
    
    # ---------- T16: Sharp 与坏消息冲突 ----------
    sharp_dir = _detect_sharp_direction(smart_signals)
    if sharp_dir and sharp_dir != "draw":
        has_bad = (
            (sharp_dir == "home" and ("主队坏消息" in sigs_str or "主坏消息" in sigs_str)) or
            (sharp_dir == "away" and ("客队坏消息" in sigs_str or "客坏消息" in sigs_str))
        )
        if has_bad and shin.get(sharp_dir, 33) < 55:
            facts.append(
                f"[T16 Sharp与坏消息冲突] Sharp 走 {sharp_dir} 但该方有坏消息爆出,"
                f"对冲信号,可能转为平局。"
            )
    
    return facts


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
        if "Steam" not in s_str:
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
# 📊 Ensemble 信号采集器(从 stats dict 提取 11 个模型 direction)
# ====================================================================

def collect_ensemble_signals(stats: Dict) -> Dict[str, Any]:
    """从 stats dict 提取 11 个模型的方向倾向 + 共识度"""
    if not stats:
        return {"models": [], "consensus": None, "consensus_count": 0, "total": 0}
    
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
        h = _f(m.get("home_win_pct", m.get("home", 0)))
        d = _f(m.get("draw_pct", m.get("draw", 0)))
        a = _f(m.get("away_win_pct", m.get("away", 0)))
        if h + d + a < 50:
            continue
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
    top_scores = rp.get("top_scores", [])[:3] if isinstance(rp, dict) else []
    
    return {
        "models": rows,
        "consensus": consensus,
        "consensus_count": consensus_count,
        "total": total,
        "top_scores": top_scores,
    }


# ====================================================================
# 📦 抓包数据 → Prompt 文本格式化
# ====================================================================

def format_match_block(idx: int, match: Dict, engine_result: Dict,
                       trap_facts: List[str], ensemble_signals: Dict,
                       smart_signals: List) -> str:
    """把一场比赛的全部抓包字段格式化为 prompt 块"""
    home = match.get("home_team", match.get("home", "Home"))
    away = match.get("away_team", match.get("guest", "Away"))
    league = match.get("league", match.get("cup", ""))
    is_cup = any(kw in str(league) for kw in CUP_KEYWORDS)
    hc = match.get("give_ball", "0")
    
    sp_h = _f(match.get("sp_home", match.get("win", 0)))
    sp_d = _f(match.get("sp_draw", match.get("same", 0)))
    sp_a = _f(match.get("sp_away", match.get("lose", 0)))
    
    shin = _compute_shin(match)
    
    # 让球理论 vs 实际
    theoretical_hc = _infer_theoretical_handicap(sp_h, sp_a) if sp_h > 1 and sp_a > 1 else 0
    actual_hc = _parse_actual_handicap(match)
    
    block = f"\n═══ 第 {idx} 场: {home} vs {away} ═══\n"
    block += f"联赛: {league}{' [杯赛/淘汰赛]' if is_cup else ''}\n"
    
    # 欧赔
    block += f"\n▼ 欧赔\n"
    block += f"即时: 主 {sp_h:.2f} / 平 {sp_d:.2f} / 客 {sp_a:.2f}\n"
    block += f"Shin概率(去抽水): 主 {shin['home']:.1f}% / 平 {shin['draw']:.1f}% / 客 {shin['away']:.1f}%\n"
    
    # 让球
    block += f"\n▼ 亚盘\n"
    block += f"实际让球: {actual_hc:+.2f} (give_ball={hc}, 主让正/客让负)\n"
    if theoretical_hc != 0:
        block += f"欧赔反推理论让球: {theoretical_hc:+.2f} (差异 {actual_hc - theoretical_hc:+.2f} 球)\n"
    
    # 赔率变动
    change = match.get("change", {}) or {}
    cw = _f(change.get("win", 0))
    cs = _f(change.get("same", 0))
    cl = _f(change.get("lose", 0))
    block += f"\n▼ 赔率变动(开盘到现在)\n"
    block += f"主 {cw:+.2f} / 平 {cs:+.2f} / 客 {cl:+.2f} (负=降水/钱流入,正=升水/钱流出)\n"
    
    # 庄家隐含 xG
    hxg = engine_result.get("bookmaker_implied_home_xg", "?")
    axg = engine_result.get("bookmaker_implied_away_xg", "?")
    block += f"\n▼ 庄家隐含 xG\n"
    block += f"主 xG: {hxg} / 客 xG: {axg}\n"
    
    # 总进球数赔率(标记压低项)
    block += f"\n▼ 总进球数赔率(a0~a7)\n"
    a_lines = []
    for g in range(8):
        v = _f(match.get(f"a{g}", 0))
        std = STANDARD_GOAL_ODDS.get(g, 50)
        if v > 1:
            ratio = std / v
            mark = f" ⚠️压低{ratio:.1f}x" if ratio > 1.5 else ""
            a_lines.append(f"{g}球={v:.2f}{mark}")
        else:
            a_lines.append(f"{g}球=N/A")
    block += " | ".join(a_lines) + "\n"
    
    # 锚点提示
    a4 = _f(match.get("a4", 0))
    a5 = _f(match.get("a5", 0))
    a7 = _f(match.get("a7", 0))
    anchors = []
    if 0 < a4 < 5: anchors.append(f"4球赔率={a4:.2f}<5,典型 2-2/3-1/1-3")
    if 0 < a5 < 8: anchors.append(f"5球赔率={a5:.2f}<8,典型 3-2/2-3")
    if 0 < a7 < 18: anchors.append(f"7+球赔率={a7:.2f}<18,典型 5-2/5-1 等大胜")
    if anchors:
        block += "⚓ 锚点: " + " | ".join(anchors) + "\n"
    
    # 27 项 CRS
    block += f"\n▼ 比分赔率 (CRS,27项)\n"
    crs_h = []
    crs_d = []
    crs_a = []
    for sc, key in CRS_FULL_MAP.items():
        v = _f(match.get(key, 0))
        if v > 1:
            text = f"{sc}={v:.1f}"
            if sc in ["1-0", "2-0", "2-1", "3-0", "3-1", "3-2", "4-0", "4-1", "4-2", "5-0", "5-1", "5-2"]:
                crs_h.append(text)
            elif sc in ["0-0", "1-1", "2-2", "3-3"]:
                crs_d.append(text)
            else:
                crs_a.append(text)
    if crs_h:
        block += f"主胜系: {' | '.join(crs_h)}\n"
    if crs_d:
        block += f"平局系: {' | '.join(crs_d)}\n"
    if crs_a:
        block += f"客胜系: {' | '.join(crs_a)}\n"
    
    others = []
    for k, label in [("crs_win", "胜其他"), ("crs_same", "平其他"), ("crs_lose", "负其他")]:
        v = _f(match.get(k, 0))
        if v > 1:
            others.append(f"{label}={v:.1f}")
    if others:
        block += f"📌 其他比分: {' | '.join(others)}\n"
    
    # 半全场
    hf_lines = []
    for k, label in HFTF_MAP.items():
        v = _f(match.get(k, 0))
        if v > 1:
            hf_lines.append(f"{label}={v:.2f}")
    if hf_lines:
        block += f"\n▼ 半全场 (9项)\n"
        block += " | ".join(hf_lines) + "\n"
    
    # 散户
    vote = match.get("vote", {}) or {}
    if vote:
        vh = vote.get("win", "?")
        vs_ = vote.get("same", "?")
        va = vote.get("lose", "?")
        block += f"\n▼ 散户分布\n"
        block += f"主 {vh}% / 平 {vs_}% / 客 {va}%"
        try:
            max_v = max(int(_f(vh, 33)), int(_f(va, 33)))
            if max_v >= 60:
                block += f" ⚠️大热({max_v}%)"
        except:
            pass
        block += "\n"
    
    # 资金信号 + 智能信号
    sharp_dir = _detect_sharp_direction(smart_signals)
    steam_dir, steam_type = _detect_steam_direction(smart_signals)
    if sharp_dir or steam_dir or smart_signals:
        block += f"\n▼ 资金/智能信号\n"
        if sharp_dir:
            block += f"Sharp 流向: {sharp_dir}\n"
        if steam_dir:
            block += f"Steam: {steam_dir} ({steam_type})\n"
        if smart_signals:
            sigs_short = [str(s) for s in smart_signals if "🎭" not in str(s)][:6]
            if sigs_short:
                block += f"其他信号: {' | '.join(sigs_short)}\n"
    
    # 基本面
    points = match.get("points", {}) or {}
    if isinstance(points, dict):
        block += f"\n▼ 基本面情报\n"
        h_text = str(points.get("home_strength", ""))[:400].replace("\n", " ")
        a_text = str(points.get("guest_strength", ""))[:400].replace("\n", " ")
        m_text = str(points.get("match_points", ""))[:300].replace("\n", " ")
        if h_text:
            block += f"主队: {h_text}\n"
        if a_text:
            block += f"客队: {a_text}\n"
        if m_text:
            block += f"赛事要点: {m_text}\n"
    
    # 异动
    info = match.get("information", {}) or {}
    if isinstance(info, dict):
        info_lines = []
        for k, label in [("home_injury", "主伤停"), ("guest_injury", "客伤停"),
                        ("home_bad_news", "主利空"), ("guest_bad_news", "客利空")]:
            v = info.get(k)
            if v:
                info_lines.append(f"{label}: {str(v)[:200].replace(chr(10), ' ')}")
        if info_lines:
            block += f"\n▼ 异动消息\n"
            block += "\n".join(info_lines) + "\n"
    
    # 陷阱事实标记
    if trap_facts:
        block += f"\n▼ 系统识别的可疑信号 (仅供参考,你可自由采纳或否决)\n"
        for fact in trap_facts:
            block += f"  {fact}\n"
    else:
        block += f"\n▼ 系统未识别明显陷阱\n"
    
    # ensemble 模型矩阵
    if ensemble_signals.get("total", 0) > 0:
        block += f"\n▼ 统计模型矩阵 ({ensemble_signals['total']} 个模型)\n"
        for row in ensemble_signals["models"]:
            block += f"  {row['name']:18s}: 主{row['home']:.0f}% / 平{row['draw']:.0f}% / 客{row['away']:.0f}% → {row['direction']}\n"
        if ensemble_signals.get("consensus"):
            ratio = ensemble_signals["consensus_count"] / max(1, ensemble_signals["total"])
            block += f"  → 共识: {ensemble_signals['consensus_count']}/{ensemble_signals['total']} 倾向 {ensemble_signals['consensus']} ({ratio:.0%})\n"
        ts = ensemble_signals.get("top_scores", [])
        if ts:
            ts_str = ", ".join([f"{t.get('score','?')}({t.get('prob','?')}%)" for t in ts if isinstance(t, dict)])
            if ts_str:
                block += f"  → Refined Poisson Top3: {ts_str}\n"
    
    return block


# ====================================================================
# Phase 1 Prompt 构建(给 GPT/Grok/Gemini)
# ====================================================================

PHASE1_ROLES = {
    "gpt": {
        "name": "量化分析师",
        "expertise": "你最擅长从欧赔/亚盘/总进球/CRS/半全场的数学一致性中发现定价裂痕。"
                     "重点检查 让球深度 vs 欧赔反推理论 是否匹配,a0-a7 进球数赔率 与 CRS 主流比分 是否互相印证。",
        "temp": 0.18,
    },
    "grok": {
        "name": "资金嗅觉师",
        "expertise": "你最擅长追踪 Sharp 资金流、Steam 信号、赔率变动方向 与 散户分布 的背离。"
                     "重点判断 散户>60% + 赔率不跟 = 诱盘 反指; 散户>60% + 赔率同向降水 = 诱反指。",
        "temp": 0.30,
    },
    "gemini": {
        "name": "基本面专家",
        "expertise": "你最擅长结合 战绩/历史交锋/伤停/联赛特点 判断真实强弱。"
                     "重点关注 主队主场 vs 客队客场 的近期战绩,以及联赛/杯赛属性对比赛走向的影响。",
        "temp": 0.20,
    },
}


def build_phase1_prompt(match_blocks: List[str], role_key: str) -> str:
    role = PHASE1_ROLES[role_key]
    p = "<context>\n"
    p += "你是顶级竞彩足球分析师团队的一员。下面是一批比赛的完整原始抓包数据,你需要对每一场都给出独立判断。\n"
    p += "</context>\n\n"
    
    p += "<your_role>\n"
    p += f"角色: {role['name']}\n"
    p += f"专长: {role['expertise']}\n"
    p += "</your_role>\n\n"
    
    p += "<analysis_method>\n"
    p += "对每一场:\n"
    p += "1. 先从你最擅长的角度独立分析\n"
    p += "2. 扫描其他维度看有没有反对你的证据\n"
    p += "3. 系统已预扫陷阱,你可以采纳或否决,要写明理由\n"
    p += "4. 综合统计模型矩阵的共识(参考但不盲从)\n"
    p += "5. 给出 top3 比分 + 主方向 + 信心度 + 推理\n"
    p += "6. 必须列出 doubts(让你自己也犹豫的反对证据,至少 1-2 条)\n"
    p += "</analysis_method>\n\n"
    
    p += "<iron_principles>\n"
    p += "- 资金面与基本面冲突时,优先资金面(Sharp/Steam/变盘),除非基本面差距极端\n"
    p += "- 散户>60% 同向 + 赔率不跟 = 诱盘,反指\n"
    p += "- 散户>60% 同向 + 赔率同向降水 = 诱反指,跟随\n"
    p += "- 杯赛/淘汰赛大热(Shin>55%) + 散户跟风 = 大热必死\n"
    p += "- 让球深度与欧赔反推差异>0.5球 = 庄家有真实立场\n"
    p += "- xG 总和<2.0 + 双方场均进球低 + a0/a1 压低 = 闷平\n"
    p += "- 陷阱标记只是事实,你自己判断要不要反指\n"
    p += "- top3[0] 比分方向必须与 main_direction 一致\n"
    p += "</iron_principles>\n\n"
    
    p += "<match_data>\n"
    for block in match_blocks:
        p += block
    p += "</match_data>\n\n"
    
    p += "<output_format>\n"
    p += "严格 JSON 数组,每场一个对象,不要前缀后缀,不要 markdown 代码块:\n"
    p += "[\n"
    p += "  {\n"
    p += "    \"match\": 1,\n"
    p += "    \"top3\": [{\"score\":\"2-1\",\"prob\":18}, {\"score\":\"1-0\",\"prob\":15}, {\"score\":\"2-0\",\"prob\":12}],\n"
    p += "    \"main_direction\": \"home\",\n"
    p += "    \"confidence\": 72,\n"
    p += "    \"reason\": \"300字推理,说清核心依据\",\n"
    p += "    \"key_signals\": [\"让球-1.5偏深vs理论-0.75\", \"主队场均2.1进\", \"Sharp走主\"],\n"
    p += "    \"doubts\": [\"客队近5主战4胜1平\", \"xG差距只有0.4\"],\n"
    p += "    \"is_score_others\": false,\n"
    p += "    \"trap_采纳\": [\"T3\"],\n"
    p += "    \"trap_否决\": [\"T1\"]\n"
    p += "  }\n"
    p += "]\n"
    p += "</output_format>\n"
    return p


# ====================================================================
# Phase 2 Prompt 构建(Claude 仲裁)
# ====================================================================

def build_phase2_prompt(match_blocks: List[str],
                        phase1_results: Dict[str, Dict[int, Dict]]) -> str:
    p = "<context>\n"
    p += "你是最终仲裁者。三个分析师(GPT 量化分析师 / Grok 资金嗅觉师 / Gemini 基本面专家)\n"
    p += "已经独立完成了对每一场比赛的分析。你拿到他们的全部结论 + 完整原始数据,做最终预测。\n"
    p += "</context>\n\n"
    
    p += "<arbitration_rules>\n"
    p += "1. 三家方向一致 → 你也选这个方向,只在比分细节上微调\n"
    p += "2. 二对一 → 默认跟多数,除非少数派 reason 明显更扎实(必须在 arbitration_reason 解释为什么少数派更对)\n"
    p += "3. 三家全分歧 → 你独立判断,必须解释为什么三家都不对\n"
    p += "4. 你不能凭空创造一个三家 top3 之外的比分(除非有强理由,且必须在 reason 写明)\n"
    p += "5. 如果某分析师列出了强 doubts(自我犹豫),他的判断要打折\n"
    p += "6. 批量分析时,如果发现某场你和三家都犹豫,优先选保守比分(1-1/2-1/0-1/1-0)\n"
    p += "7. predicted_score 方向必须与 predicted_direction 一致\n"
    p += "</arbitration_rules>\n\n"
    
    # 分析师结论汇总
    p += "<分析师结论>\n"
    num_matches = len(match_blocks)
    for i in range(1, num_matches + 1):
        p += f"\n═══ 第 {i} 场 ═══\n"
        for ai_name in ["gpt", "grok", "gemini"]:
            r = phase1_results.get(ai_name, {}).get(i, {})
            if not r:
                p += f"  {ai_name.upper()}: 无数据(弃权)\n"
                continue
            top3 = r.get("top3", [])
            top3_str = ", ".join([
                f"{t.get('score','?')}({t.get('prob','?')}%)"
                for t in top3 if isinstance(t, dict)
            ])
            p += f"  {ai_name.upper()}: 方向={r.get('main_direction','?')} | "
            p += f"top3=[{top3_str}] | conf={r.get('confidence','?')}\n"
            reason = str(r.get("reason", ""))[:250].replace("\n", " ")
            p += f"    reason: {reason}\n"
            ks = r.get("key_signals", [])
            if ks:
                p += f"    key_signals: {', '.join(str(k) for k in ks[:5])}\n"
            doubts = r.get("doubts", [])
            if doubts:
                p += f"    doubts: {', '.join(str(d) for d in doubts[:3])}\n"
    p += "\n</分析师结论>\n\n"
    
    # 原始数据(精简版,保留关键字段)
    p += "<原始数据>\n"
    for block in match_blocks:
        p += block
    p += "</原始数据>\n\n"
    
    p += "<output_format>\n"
    p += "严格 JSON 数组,每场一个对象,不要前缀后缀,不要 markdown 代码块:\n"
    p += "[\n"
    p += "  {\n"
    p += "    \"match\": 1,\n"
    p += "    \"predicted_score\": \"2-1\",\n"
    p += "    \"predicted_direction\": \"home\",\n"
    p += "    \"home_win_pct\": 55,\n"
    p += "    \"draw_pct\": 25,\n"
    p += "    \"away_win_pct\": 20,\n"
    p += "    \"confidence\": 75,\n"
    p += "    \"agreement_pattern\": \"三家一致(home)\",\n"
    p += "    \"arbitration_reason\": \"500字仲裁理由,说清为什么选这个,为什么否决其他\",\n"
    p += "    \"alternative\": {\"score\":\"1-1\",\"prob\":25},\n"
    p += "    \"is_score_others\": false,\n"
    p += "    \"kelly_suggest\": 0.05\n"
    p += "  }\n"
    p += "]\n"
    p += "</output_format>\n"
    return p


# ====================================================================
# AI 调用引擎(沿用 v18 稳定通道)
# ====================================================================

FALLBACK_URLS = [None, "https://www.api522.pro/v1", "https://api522.pro/v1",
                 "https://api521.pro/v1", "http://69.63.213.33:666/v1"]
GPT_DEFAULT_URL = "https://ai.newapi.life/v1"
GPT_DEFAULT_KEY = ""


def get_clean_env_url(name, default=""):
    v = str(os.environ.get(name, globals().get(name, default))).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/-]+)", v)
    return match.group(1) if match else v


def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")


async def async_call_one_ai(session, prompt, url_env, key_env, models_list,
                            num_matches, ai_name, sys_prompt, temperature):
    """调用单个 AI 批量返回 num_matches 条结果"""
    key = get_clean_env_key(key_env)
    if not key and ai_name == "gpt":
        key = GPT_DEFAULT_KEY
    if not key:
        return ai_name, {}, "no_key"
    
    if ai_name == "gpt":
        primary_url = get_clean_env_url(url_env, GPT_DEFAULT_URL)
        if not primary_url or "poloai" not in primary_url:
            primary_url = GPT_DEFAULT_URL
        urls = [primary_url]
    else:
        primary_url = get_clean_env_url(url_env)
        backup = [u for u in FALLBACK_URLS if u and u != primary_url][:1]
        urls = [primary_url] + backup
    
    CONNECT_TIMEOUT = 20
    READ_TIMEOUT_MAP = {"claude": 420, "grok": 320, "gpt": 320, "gemini": 280}
    READ_TIMEOUT = READ_TIMEOUT_MAP.get(ai_name, 200)
    
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
                    "generationConfig": {"temperature": temperature},
                    "systemInstruction": {"parts": [{"text": sys_prompt}]}
                }
            else:
                headers["Authorization"] = f"Bearer {key}"
                payload = {
                    "model": mn,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature
                }
            
            gw = url.split("/v1")[0][:35]
            print(f"  [🔌] {ai_name.upper()} | {mn[:22]} @ {gw}")
            t0 = time.time()
            
            try:
                timeout = aiohttp.ClientTimeout(
                    total=None, connect=CONNECT_TIMEOUT,
                    sock_connect=CONNECT_TIMEOUT, sock_read=READ_TIMEOUT
                )
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                    elapsed = round(time.time()-t0, 1)
                    if r.status in (502, 504):
                        print(f"    💀 HTTP {r.status} | {elapsed}s → 换URL")
                        continue
                    if r.status == 400:
                        print(f"    💀 400 | {elapsed}s → 换模型")
                        break
                    if r.status == 429:
                        await asyncio.sleep(1)
                        continue
                    if r.status != 200:
                        continue
                    
                    connected = True
                    print(f"    ✅ 已连上 {elapsed}s | 等待数据...")
                    
                    try:
                        data = await r.json(content_type=None)
                    except:
                        break
                    
                    elapsed = round(time.time()-t0, 1)
                    raw_text = _extract_response_text(data, is_gem)
                    
                    if not raw_text or len(raw_text) < 10:
                        print(f"    ⚠️ 空数据 → 换模型")
                        break
                    
                    results = _parse_ai_json(raw_text, num_matches)
                    
                    if len(results) > 0:
                        print(f"    ✅ {ai_name.upper()} 完成: {len(results)}/{num_matches} | {elapsed}s")
                        return ai_name, results, mn
                    else:
                        print(f"    ⚠️ 解析0条 → 换模型")
                        break
            
            except aiohttp.ClientConnectorError:
                continue
            except asyncio.TimeoutError:
                if not connected:
                    continue
                else:
                    return ai_name, {}, "read_timeout"
            except Exception as e:
                if not connected:
                    continue
                else:
                    return ai_name, {}, "error"
            
            await asyncio.sleep(0.2)
    
    return ai_name, {}, "all_failed"


def _extract_response_text(data, is_gem):
    """从 API 响应中提取文本(沿用 v18 多通道兜底)"""
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
                        for field in ["text", "answer", "response", "output_text",
                                     "final_answer", "output", "result", "completion"]:
                            v = msg.get(field, "")
                            if v and isinstance(v, str) and v.strip():
                                raw_text = v.strip()
                                break
                    
                    if not raw_text:
                        skip = ("reasoning_content", "thinking", "reasoning", "thoughts")
                        best = ""
                        for k in msg:
                            if k in skip: continue
                            v = msg[k]
                            if isinstance(v, str) and '"match"' in v and "[" in v:
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
                    for ci in range(start_pos, min(start_pos + 200000, len(full_str))):
                        if full_str[ci] == '[': depth += 1
                        elif full_str[ci] == ']': depth -= 1
                        if depth == 0:
                            end_pos = ci + 1
                            break
                    if end_pos > start_pos:
                        extracted = full_str[start_pos:end_pos]
                        if '\\"' in extracted:
                            try:
                                extracted = json.loads('"' + extracted + '"')
                            except:
                                extracted = extracted.replace('\\"', '"')
                        raw_text = extracted
    except Exception as ex:
        print(f"    ⚠️ 解析异常: {str(ex)[:80]}")
    return raw_text


def _parse_ai_json(raw_text, num_matches):
    """解析 AI 返回的 JSON"""
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
            if clean[i] == '[': depth += 1
            elif clean[i] == ']':
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
                last_brace = json_str.rfind('}')
                arr = json.loads(json_str[:last_brace+1] + "]") if last_brace != -1 else []
            except:
                arr = []
        
        if isinstance(arr, list):
            for item in arr:
                if not isinstance(item, dict) or not item.get("match"):
                    continue
                try:
                    mid = int(item["match"])
                except:
                    continue
                results[mid] = item
    
    return results


# ====================================================================
# Phase 1: 三家批量并行
# ====================================================================

async def run_phase1(match_blocks: List[str], num_matches: int) -> Dict[str, Dict]:
    print(f"\n  [Phase 1] 三家批量并行分析 ({num_matches} 场)...")
    
    configs = [
        ("gpt", "GPT_API_URL", "GPT_API_KEY", ["gpt-5.5"], "gpt"),
        ("grok", "GROK_API_URL", "GROK_API_KEY",
         ["熊猫-A-5-grok-4.2-fast-200w上下文"], "grok"),
        ("gemini", "GEMINI_API_URL", "GEMINI_API_KEY",
         ["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking"], "gemini"),
    ]
    
    sys_prompts = {
        "gpt": "<role>你是衍生品定价+概率分布偏差量化策略师。从赔率数学一致性发现定价裂痕。</role>\n"
               "<instruction>严格输出 JSON 数组,禁止前缀后缀。</instruction>",
        "grok": "<role>你是拥有全网实时数据嗅觉的另类数据分析师。专注于资金流和散户分布的背离。</role>\n"
                "<instruction>只输出 JSON 数组。</instruction>",
        "gemini": "<role>你是结合战绩、伤停、联赛特点的基本面专家。</role>\n"
                  "<instruction>仅 JSON 数组。</instruction>",
    }
    
    connector = aiohttp.TCPConnector(limit=10, use_dns_cache=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for ai_name, u, k, models, role_key in configs:
            prompt = build_phase1_prompt(match_blocks, role_key)
            temp = PHASE1_ROLES[role_key]["temp"]
            tasks.append(async_call_one_ai(
                session, prompt, u, k, models, num_matches,
                ai_name, sys_prompts[ai_name], temp
            ))
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    output = {"gpt": {}, "grok": {}, "gemini": {}}
    for res in results:
        if isinstance(res, tuple):
            output[res[0]] = res[1]
    
    ok = sum(1 for v in output.values() if v)
    print(f"  [Phase 1 完成] {ok}/3 AI 有数据")
    return output


# ====================================================================
# Phase 2: Claude 仲裁
# ====================================================================

async def run_phase2(match_blocks: List[str], num_matches: int,
                     phase1_results: Dict[str, Dict]) -> Dict[int, Dict]:
    print(f"\n  [Phase 2] Claude 批量仲裁 ({num_matches} 场)...")
    
    prompt = build_phase2_prompt(match_blocks, phase1_results)
    print(f"  [Phase 2 Prompt] {len(prompt):,} 字符")
    
    sys_prompt = (
        "<role>你是最终仲裁者。综合三家分析师结论 + 原始数据,做最终预测。</role>\n"
        "<priority>遵守 arbitration_rules,尤其是三家共识时跟随、二对一时默认多数。</priority>\n"
        "<instruction>仅输出 JSON 数组,禁止前缀后缀和 markdown 代码块。</instruction>"
    )
    
    connector = aiohttp.TCPConnector(limit=5, use_dns_cache=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        ai_name, results, mn = await async_call_one_ai(
            session, prompt, "CLAUDE_API_URL", "CLAUDE_API_KEY",
            ["熊猫-69-满血openrouter-claude-opus-4.7-上下文1000k"],
            num_matches, "claude", sys_prompt, 0.25
        )
    
    print(f"  [Phase 2 完成] Claude 返回 {len(results)}/{num_matches} 场")
    return results


# ====================================================================
# 结果整理 + 一致性补丁
# ====================================================================

def _enforce_direction_consistency(claude_result: Dict) -> Dict:
    """
    只对齐方向字段,不篡改概率。
    保证 predicted_score / predicted_direction / result 显示文字 一致。
    """
    score = claude_result.get("predicted_score", "1-1")
    
    if "胜其他" in str(score) or score == "9-0":
        expected_dir = "home"
        expected_cn = "主胜"
        score_label = "胜其他"
        is_others = True
    elif "平其他" in str(score) or score == "9-9":
        expected_dir = "draw"
        expected_cn = "平局"
        score_label = "平其他"
        is_others = True
    elif "负其他" in str(score) or score == "0-9":
        expected_dir = "away"
        expected_cn = "客胜"
        score_label = "负其他"
        is_others = True
    else:
        h, a = _parse_score(score)
        if h is None:
            expected_dir = claude_result.get("predicted_direction", "draw")
            expected_cn = {"home": "主胜", "draw": "平局", "away": "客胜"}.get(expected_dir, "平局")
            score_label = score
            is_others = False
        else:
            if h > a:
                expected_dir = "home"; expected_cn = "主胜"
            elif h < a:
                expected_dir = "away"; expected_cn = "客胜"
            else:
                expected_dir = "draw"; expected_cn = "平局"
            score_label = score
            is_others = False
    
    claude_result["predicted_direction"] = expected_dir
    claude_result["result"] = expected_cn
    claude_result["display_direction"] = expected_cn
    claude_result["final_direction"] = expected_dir
    claude_result["predicted_label"] = score_label
    claude_result["is_score_others"] = is_others
    if is_others:
        claude_result["predicted_score"] = score_label
    
    return claude_result


def assemble_final_prediction(match: Dict, engine_result: Dict, stats: Dict,
                              phase1_results: Dict[str, Dict],
                              claude_result: Dict, trap_facts: List[str],
                              ensemble_signals: Dict, idx: int) -> Dict:
    """
    把 Claude 仲裁结果包装成 v18 风格的输出 dict,保证下游 UI/落库兼容。
    """
    if isinstance(match.get("v2_odds_dict"), dict):
        match = {**match, **match["v2_odds_dict"]}
    
    # Claude 主输出
    cr = claude_result or {}
    cr = _enforce_direction_consistency(cr)
    
    predicted_score = cr.get("predicted_score", "1-1")
    predicted_label = cr.get("predicted_label", predicted_score)
    final_direction = cr.get("final_direction", "draw")
    result_cn = cr.get("result", "平局")
    
    home_pct = round(_f(cr.get("home_win_pct", 33)), 1)
    draw_pct = round(_f(cr.get("draw_pct", 33)), 1)
    away_pct = round(_f(cr.get("away_win_pct", 33)), 1)
    total_pct = home_pct + draw_pct + away_pct
    if total_pct > 0:
        home_pct = round(home_pct / total_pct * 100, 1)
        draw_pct = round(draw_pct / total_pct * 100, 1)
        away_pct = round(away_pct / total_pct * 100, 1)
    
    confidence = int(_f(cr.get("confidence", 60)))
    confidence = max(30, min(95, confidence))
    
    risk = "低" if confidence >= 75 else ("中" if confidence >= 55 else "高")
    
    # CRS 赔率与 EV
    target_crs = CRS_FULL_MAP.get(predicted_score, "")
    final_odds = _f(match.get(target_crs, 0))
    if not final_odds and cr.get("is_score_others"):
        if final_direction == "home":
            final_odds = _f(match.get("crs_win", 0))
        elif final_direction == "away":
            final_odds = _f(match.get("crs_lose", 0))
        else:
            final_odds = _f(match.get("crs_same", 0))
    
    direction_pct_map = {"home": home_pct, "draw": draw_pct, "away": away_pct}
    direction_prob = direction_pct_map.get(final_direction, 33)
    if final_odds > 1.05 and direction_prob > 0:
        prob = direction_prob / 100.0
        ev = (prob * final_odds) - 1.0
        b = final_odds - 1.0
        q = 1.0 - prob
        if b > 0:
            kelly = ((b * prob) - q) / b
            kelly_pct = round(max(0.0, kelly * 0.5) * 100, 2)
        else:
            kelly_pct = 0
        ev_pct = round(ev * 100, 2)
    else:
        ev_pct = 0
        kelly_pct = 0
    
    # smart signals 汇总
    smart_signals = stats.get("smart_signals", []) if stats else []
    sigs = list(smart_signals)
    sigs.extend(trap_facts)
    if cr.get("is_score_others"):
        sigs.append(f"🔥 其他比分场触发")
    
    # 三家分析师结论摘要
    p1_gpt = phase1_results.get("gpt", {}).get(idx, {})
    p1_grok = phase1_results.get("grok", {}).get(idx, {})
    p1_gemini = phase1_results.get("gemini", {}).get(idx, {})
    
    def _ai_summary(r):
        if not isinstance(r, dict) or not r:
            return "弃权"
        top3 = r.get("top3", [])
        if top3 and isinstance(top3[0], dict):
            return top3[0].get("score", "?")
        return "?"
    
    gpt_score = _ai_summary(p1_gpt)
    grok_score = _ai_summary(p1_grok)
    gemini_score = _ai_summary(p1_gemini)
    
    # 杯赛/联赛属性
    league = str(match.get("league", match.get("cup", "")))
    
    return {
        # 核心字段
        "predicted_score": predicted_score,
        "predicted_label": predicted_label,
        "result": result_cn,
        "display_direction": result_cn,
        "final_direction": final_direction,
        "is_score_others": cr.get("is_score_others", False),
        
        # 概率(直接来自 Claude)
        "home_win_pct": home_pct,
        "draw_pct": draw_pct,
        "away_win_pct": away_pct,
        
        "confidence": confidence,
        "risk_level": risk,
        "dir_confidence": confidence,
        "dir_gap": round(max(home_pct, draw_pct, away_pct) -
                         sorted([home_pct, draw_pct, away_pct])[1], 1),
        
        # Claude 仲裁元数据
        "agreement_pattern": cr.get("agreement_pattern", "未知"),
        "arbitration_reason": cr.get("arbitration_reason", ""),
        "alternative_score": cr.get("alternative", {}),
        
        # 三家分析师结论
        "gpt_score": gpt_score,
        "gpt_analysis": str(p1_gpt.get("reason", "弃权"))[:600] if p1_gpt else "弃权",
        "gpt_doubts": p1_gpt.get("doubts", []) if p1_gpt else [],
        "grok_score": grok_score,
        "grok_analysis": str(p1_grok.get("reason", "弃权"))[:600] if p1_grok else "弃权",
        "grok_doubts": p1_grok.get("doubts", []) if p1_grok else [],
        "gemini_score": gemini_score,
        "gemini_analysis": str(p1_gemini.get("reason", "弃权"))[:600] if p1_gemini else "弃权",
        "gemini_doubts": p1_gemini.get("doubts", []) if p1_gemini else [],
        "claude_score": predicted_score,
        "claude_analysis": cr.get("arbitration_reason", "")[:800],
        
        "ai_abstained": [n.upper() for n in ["gpt", "grok", "gemini"]
                        if not phase1_results.get(n, {}).get(idx)],
        
        # EV / Kelly
        "suggested_kelly": kelly_pct,
        "edge_vs_market": ev_pct,
        "is_value": ev_pct > 5,
        
        # 陷阱事实
        "traps_detected": [f.split("]")[0].lstrip("[") for f in trap_facts],
        "trap_count": len(trap_facts),
        "trap_facts": trap_facts,
        
        # 市场基础数据
        "smart_money_signal": " | ".join([str(s) for s in sigs[:8]]),
        "smart_signals": sigs,
        "sharp_detected": _detect_sharp_direction(smart_signals) is not None,
        "sharp_dir": _detect_sharp_direction(smart_signals),
        
        # 兜底
        "xG_home": round(_f(engine_result.get("bookmaker_implied_home_xg", 1.3)), 2),
        "xG_away": round(_f(engine_result.get("bookmaker_implied_away_xg", 0.9)), 2),
        "expected_total_goals": round(
            _f(engine_result.get("bookmaker_implied_home_xg", 1.3)) +
            _f(engine_result.get("bookmaker_implied_away_xg", 0.9)), 2),
        "over_2_5": engine_result.get("over_25", 50) if engine_result else 50,
        "btts": engine_result.get("btts", 45) if engine_result else 45,
        "bookmaker_implied_home_xg": engine_result.get("bookmaker_implied_home_xg", "?"),
        "bookmaker_implied_away_xg": engine_result.get("bookmaker_implied_away_xg", "?"),
        
        # ensemble 共识
        "model_consensus_dir": ensemble_signals.get("consensus", ""),
        "model_consensus_count": ensemble_signals.get("consensus_count", 0),
        "total_models": ensemble_signals.get("total", 0),
        
        # 兜底字段(供老 UI/落库使用)
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
        
        "engine_version": "vMAX 19.0",
        "engine_architecture": "AI-First (3家分析+Claude仲裁)",
    }


# ====================================================================
# Top4 推荐(新版只看 Claude 给的 confidence)
# ====================================================================

def select_top4(preds):
    """新版排序: 只用 Claude 的 confidence 作为唯一依据,简单直接。"""
    preds_sorted = sorted(
        preds,
        key=lambda x: _f(x.get("prediction", {}).get("confidence", 0)),
        reverse=True
    )
    return preds_sorted[:4]


def extract_num(ms):
    wm = {"一":1000, "二":2000, "三":3000, "四":4000, "五":5000, "六":6000, "日":7000, "天":7000}
    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999


# ====================================================================
# 主入口
# ====================================================================

def run_predictions(raw, use_ai=True):
    matches = raw.get("matches", [])
    num = len(matches)
    print("\n" + "=" * 80)
    print(f"  [vMAX 19.0] AI-First (3 家分析 + Claude 仲裁) | {num} 场")
    print("=" * 80)
    
    # ---- 预处理: 引擎只做数据准备 ----
    match_analyses = []
    match_blocks = []
    
    for i, m in enumerate(matches):
        try:
            eng = predict_match(m)
        except Exception as e:
            logger.warning(f"predict_match 失败: {e}")
            eng = {}
        
        try:
            sp = ensemble.predict(m, {}) if ensemble else {}
        except Exception:
            sp = {}
        
        try:
            exp_result = exp_engine.analyze(m) if exp_engine else {}
        except Exception:
            exp_result = {}
        
        # 期望进球
        exp_goals = _f(eng.get("expected_total_goals", 0))
        if exp_goals <= 0:
            hxg = _f(eng.get("bookmaker_implied_home_xg", 0))
            axg = _f(eng.get("bookmaker_implied_away_xg", 0))
            exp_goals = hxg + axg if (hxg and axg) else 2.5
        
        smart_signals = sp.get("smart_signals", []) if sp else []
        
        # 陷阱事实标记
        trap_facts = detect_traps_as_facts(m, eng, smart_signals, exp_goals)
        
        # ensemble 信号
        ensemble_signals = collect_ensemble_signals(sp)
        
        # 格式化为 prompt 块
        block = format_match_block(i+1, m, eng, trap_facts, ensemble_signals, smart_signals)
        match_blocks.append(block)
        
        match_analyses.append({
            "match": m,
            "engine": eng,
            "stats": sp,
            "experience": exp_result,
            "trap_facts": trap_facts,
            "ensemble_signals": ensemble_signals,
        })
    
    # ---- Phase 1 + Phase 2 ----
    phase1_results = {"gpt": {}, "grok": {}, "gemini": {}}
    claude_results = {}
    
    if use_ai and match_blocks:
        async def _run_full():
            p1 = await run_phase1(match_blocks, num)
            p2 = await run_phase2(match_blocks, num, p1)
            return p1, p2
        
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
                future = pool.submit(_run_in_thread, _run_full())
                try:
                    phase1_results, claude_results = future.result()
                except Exception as e:
                    logger.error(f"AI 矩阵执行崩溃: {e}")
        else:
            try:
                phase1_results, claude_results = asyncio.run(_run_full())
            except Exception as e:
                logger.error(f"AI 矩阵执行崩溃: {e}")
    
    # ---- 整合最终预测 ----
    res = []
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        idx = i + 1
        
        cr = claude_results.get(idx, {})
        if not cr:
            # Claude 失败 → 用 GPT 兜底
            for fallback in ["gpt", "gemini", "grok"]:
                fb = phase1_results.get(fallback, {}).get(idx, {})
                if fb and fb.get("top3"):
                    top1 = fb["top3"][0] if fb["top3"] else {}
                    cr = {
                        "predicted_score": top1.get("score", "1-1"),
                        "predicted_direction": fb.get("main_direction", "draw"),
                        "confidence": int(_f(fb.get("confidence", 50)) * 0.85),
                        "arbitration_reason": f"Claude 失败,采用 {fallback.upper()} 结论兜底: {fb.get('reason','')[:300]}",
                        "agreement_pattern": "Claude 失败兜底",
                    }
                    break
            if not cr:
                cr = {
                    "predicted_score": "1-1",
                    "predicted_direction": "draw",
                    "confidence": 35,
                    "arbitration_reason": "全部 AI 失败,兜底输出 1-1",
                    "agreement_pattern": "全部失败",
                }
        
        mg = assemble_final_prediction(
            m, ma["engine"], ma["stats"],
            phase1_results, cr,
            ma["trap_facts"], ma["ensemble_signals"], idx
        )
        
        # 应用其他增强模块(保持兼容性)
        try:
            if exp_engine:
                mg = apply_experience_to_prediction(m, mg, exp_engine)
        except Exception as e:
            logger.warning(f"apply_experience 失败: {e}")
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
            logger.warning(f"upgrade_ensemble 失败: {e}")
        
        # 最后再次确保方向一致
        mg = _enforce_direction_consistency(mg)
        
        res.append({**m, "prediction": mg})
        
        agree_tag = f" [{mg.get('agreement_pattern','?')}]"
        trap_tag = f" [🎭{mg['trap_count']}陷阱]" if mg.get('trap_count', 0) > 0 else ""
        others_tag = f" [🔥其他]" if mg.get("is_score_others") else ""
        
        print(f"  [{idx}] {m.get('home_team', m.get('home', '?'))} vs "
              f"{m.get('away_team', m.get('guest', '?'))} => "
              f"{mg['result']} ({mg['predicted_score']}) | "
              f"CF: {mg['confidence']}%{agree_tag}{trap_tag}{others_tag}")
    
    # Top4 推荐
    t4 = select_top4(res)
    t4ids = [t.get("id") for t in t4]
    for r in res:
        r["is_recommended"] = r.get("id") in t4ids
    
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    
    return res, t4


if __name__ == "__main__":
    logger.info("vMAX 19.0 AI-First 启动")
    print("✅ vMAX 19.0 加载完成")
    print("   架构: 3 家批量分析(GPT/Grok/Gemini) + Claude 批量仲裁")
    print("   引擎职责: 抓包格式化 + 陷阱事实标记 + ensemble 信号注入")
    print("   决策权: 100% AI,引擎不做加权计算")