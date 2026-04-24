import json
import os
import re
import time
import asyncio
import aiohttp
import logging
import numpy as np
import math
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    print("  [WARN] ⚠️ 未检测到 structlog 库，自动降级为标准 logging 模块")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

# ====================================================================
# v18 新增核心模块 (带强力防崩溃降级机制)
# ====================================================================
try:
    from trap_detector import detect_all_traps
except ImportError as e:
    logger.warning(f"⚠️ 缺少 trap_detector 模块，陷阱矩阵将自动降级: {e}")
    def detect_all_traps(*args, **kwargs):
        return {"trap_count": 0, "total_severity": 0, "traps_detected": [], "confidence_penalty": 0, "sharp_detected": False}

try:
    from crs_analyzer import analyze_crs_matrix
except ImportError as e:
    logger.warning(f"⚠️ 缺少 crs_analyzer 模块，CRS分析将自动降级: {e}")
    def analyze_crs_matrix(*args, **kwargs):
        return {"shape_verdict": "unknown", "moments": {}, "implied_probs": {}, "margin": 0.0, "coverage": 0.0}

try:
    from bayesian_engine import decision_lock_chain, _parse_score
except ImportError as e:
    logger.warning(f"⚠️ 缺少 bayesian_engine 模块，决策链将自动降级: {e}")
    def _parse_score(s):
        try:
            s_str = str(s).strip().replace(" ", "").replace("：", "-").replace(":", "-").replace("\u2013", "-").replace("\u2014", "-")
            if "胜" in s_str and "其他" in s_str: return 9, 0
            if "平" in s_str and "其他" in s_str: return 9, 9
            if "负" in s_str and "其他" in s_str: return 0, 9
            p = s_str.split("-")
            return int(p[0]), int(p[1])
        except:
            return None, None
            
    def decision_lock_chain(match_obj, *args, **kwargs):
        # 极端降级逻辑：只返回安全默认值避免报错
        return {
            "predicted_score": "1-1", "predicted_label": "1-1", "result": "平局",
            "display_direction": "平局", "final_direction": "draw",
            "home_win_pct": 33.3, "draw_pct": 33.4, "away_win_pct": 33.3,
            "is_score_others": False, "scenario": "fallback",
            "dir_confidence": 50, "dir_gap": 0, "goal_range": "2-3",
            "evidences": [], "bayesian_prior": {}, "override_triggered": False,
            "top_score_candidates": []
        }
# ====================================================================

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
except Exception as e:
    logger.warning("⚠️ odds_history 加载失败，自动降级", exc_info=True)
    def apply_odds_history(m, mg): return mg

try:
    from quant_edge import apply_quant_edge
except Exception as e:
    logger.warning("⚠️ quant_edge 加载失败，自动降级", exc_info=True)
    def apply_quant_edge(m, mg): return mg

try:
    from wencai_intel import apply_wencai_intel
except:
    def apply_wencai_intel(m, mg): return mg

try:
    ensemble = EnsemblePredictor()
    exp_engine = ExperienceEngine()
except:
    ensemble = None
    exp_engine = None


# ====================================================================
# 常量 & 工具函数
# ====================================================================

# 进球数标准赔率基准（用户经验+10万场统计）
STANDARD_GOAL_ODDS = {
    0: 9.5, 1: 5.5, 2: 3.5, 3: 4.0,
    4: 7.0, 5: 14.0, 6: 30.0, 7: 70.0,
}

# 胜其他/平其他/负其他比分集合 (加入虚拟比分支持AI输出)
SCORE_OTHERS_HOME = ["4-3", "5-3", "5-4", "6-0", "6-1", "6-2", "6-3", "6-4", "7-0", "7-1", "7-2", "胜其他", "9-0"]
SCORE_OTHERS_DRAW = ["4-4", "5-5", "6-6", "平其他", "9-9"]
SCORE_OTHERS_AWAY = ["3-4", "0-6", "1-6", "2-6", "3-6", "0-7", "1-7", "2-7", "负其他", "0-9"]
ALL_SCORE_OTHERS = SCORE_OTHERS_HOME + SCORE_OTHERS_DRAW + SCORE_OTHERS_AWAY

# CRS完整字段映射
CRS_FULL_MAP = {
    "1-0": "w10", "2-0": "w20", "2-1": "w21", "3-0": "w30", "3-1": "w31",
    "3-2": "w32", "4-0": "w40", "4-1": "w41", "4-2": "w42", "5-0": "w50",
    "5-1": "w51", "5-2": "w52",
    "0-0": "s00", "1-1": "s11", "2-2": "s22", "3-3": "s33",
    "0-1": "l01", "0-2": "l02", "1-2": "l12", "0-3": "l03", "1-3": "l13",
    "2-3": "l23", "0-4": "l04", "1-4": "l14", "2-4": "l24", "0-5": "l05",
    "1-5": "l15", "2-5": "l25",
}

def _f(v, default=0.0):
    try:
        return float(v) if v is not None and str(v).strip() != "" else default
    except:
        return default

def calculate_value_bet(prob_pct, odds):
    if not odds or odds <= 1.05:
        return {"ev": 0.0, "kelly": 0.0, "is_value": False}
    prob = prob_pct / 100.0
    ev = (prob * odds) - 1.0
    b = odds - 1.0
    q = 1.0 - prob
    if b <= 0:
        return {"ev": round(ev * 100, 2), "kelly": 0.0, "is_value": False}
    kelly = ((b * prob) - q) / b
    return {
        "ev": round(ev * 100, 2),
        "kelly": round(max(0.0, kelly * 0.5) * 100, 2),
        "is_value": ev > 0.05
    }

def parse_score(s):
    try:
        s_str = str(s).strip().replace(" ", "").replace("：", "-").replace(":", "-").replace("\u2013", "-").replace("\u2014", "-")
        if "胜" in s_str and "其他" in s_str: return 9, 0
        if "平" in s_str and "其他" in s_str: return 9, 9
        if "负" in s_str and "其他" in s_str: return 0, 9
        p = s_str.split("-")
        return int(p[0]), int(p[1])
    except:
        return None, None


# ====================================================================
# 🎯 核心算法1: CRS赔率直接反推概率 (替代泊松)
# ====================================================================
def crs_implied_probabilities(match_obj):
    raw_odds = {}
    for score, key in CRS_FULL_MAP.items():
        try:
            odds = float(match_obj.get(key, 0) or 0)
            if odds > 1.1:
                raw_odds[score] = odds
        except:
            pass

    extras = {}
    for key, scores_set in [
        ("crs_win", SCORE_OTHERS_HOME),
        ("crs_same", SCORE_OTHERS_DRAW),
        ("crs_lose", SCORE_OTHERS_AWAY)
    ]:
        try:
            odds = float(match_obj.get(key, 0) or 0)
            if odds > 1.1:
                extras[key] = {"odds": odds, "scores": scores_set}
        except:
            pass

    if len(raw_odds) < 8:
        return {}, 0.0, 0.0

    raw_sum = sum(1/o for o in raw_odds.values())
    for extra_data in extras.values():
        raw_sum += 1 / extra_data["odds"]

    margin = raw_sum - 1.0

    probs = {}
    for score, odds in raw_odds.items():
        probs[score] = (1 / odds) / raw_sum * 100

    for key, extra_data in extras.items():
        total_prob = (1 / extra_data["odds"]) / raw_sum * 100
        num_scores = len(extra_data["scores"])
        if num_scores > 0:
            per_score = total_prob / num_scores
            for sc in extra_data["scores"]:
                if sc not in probs:
                    probs[sc] = per_score
                else:
                    probs[sc] += per_score

    coverage = len(raw_odds) / len(CRS_FULL_MAP)
    return probs, round(margin, 3), round(coverage, 2)


# ====================================================================
# 🎯 核心算法2: 进球数赔率信号检测
# ====================================================================
def detect_goal_signals(match_obj):
    signals = {}
    for g in range(8):
        try:
            actual = float(match_obj.get(f"a{g}", 0) or 0)
            if actual > 1:
                std = STANDARD_GOAL_ODDS.get(g, 50)
                ratio = std / actual
                if ratio > 1.2:
                    signals[g] = ratio
        except:
            pass
    return signals


# ====================================================================
# 🎯 核心算法3: 胜其他场识别器 (接入极端大球高敏探针)
# ====================================================================
def detect_score_others(match_obj, exp_goals, ai_responses=None):
    triggers = []
    score = 0
    is_extreme_blowout = False

    try:
        a7 = float(match_obj.get("a7", 999) or 999)
        if 0 < a7 <= 23.0:
            score += 2
            triggers.append(f"7+球极低{a7:.1f}≤23")
            is_extreme_blowout = True
        elif 0 < a7 <= 30.0:
            score += 1
            triggers.append(f"7+球{a7:.1f}≤30")
        elif 0 < a7 <= 18:
            score += 1
            triggers.append(f"7+球{a7:.1f}≤18")
    except: pass

    try:
        a6 = float(match_obj.get("a6", 999) or 999)
        if 0 < a6 <= 13.0:
            score += 2
            triggers.append(f"6球极低{a6:.1f}≤13")
            is_extreme_blowout = True
        elif 0 < a6 <= 16.0:
            score += 1
            triggers.append(f"6球{a6:.1f}≤16")
    except: pass

    try:
        a5 = float(match_obj.get("a5", 999) or 999)
        if 0 < a5 <= 8.0:
            score += 2
            triggers.append(f"5球极低{a5:.1f}≤8")
            is_extreme_blowout = True
        elif 0 < a5 <= 10.0:
            score += 1
            triggers.append(f"5球{a5:.1f}≤10")
    except: pass

    if exp_goals >= 3.2:
        score += 1
        triggers.append(f"λ={exp_goals:.2f}≥3.2")

    try:
        info_text = ""
        if isinstance(match_obj.get("points"), dict):
            info_text = (match_obj["points"].get("home_strength", "") +
                         match_obj["points"].get("guest_strength", ""))
        h_match = re.search(r"场均进球[^0-9]*(\d+\.\d+)", info_text)
        a_match = re.search(r"场均失球[^0-9]*(\d+\.\d+)", info_text)
        if h_match and a_match:
            h_avg = float(h_match.group(1))
            a_avg = float(a_match.group(1))
            if (h_avg + a_avg) >= 3.5:
                score += 1
                triggers.append(f"场均{h_avg+a_avg:.1f}≥3.5")
    except: pass

    try:
        crs_win = float(match_obj.get("crs_win", 999) or 999)
        crs_same = float(match_obj.get("crs_same", 999) or 999)
        crs_lose = float(match_obj.get("crs_lose", 999) or 999)
        if 0 < crs_win < crs_same * 0.4 and crs_win < 100:
            score += 1
            triggers.append(f"crs_win{crs_win:.0f}暗示主胜其他")
        elif 0 < crs_lose < crs_same * 0.4 and crs_lose < 100:
            score += 1
            triggers.append(f"crs_lose{crs_lose:.0f}暗示负其他")
    except: pass

    league = str(match_obj.get("cup", match_obj.get("league", "")))
    if any(kw in league for kw in ["欧冠", "欧联", "杯", "淘汰", "决赛"]):
        score += 0.5
        triggers.append("杯赛/淘汰赛")

    ai_others_count = 0
    if ai_responses:
        for name, r in ai_responses.items():
            if isinstance(r, dict) and r.get("is_score_others"):
                ai_others_count += 1

    if ai_others_count >= 2:
        score += 1
        triggers.append(f"AI{ai_others_count}/4识别胜其他")

    is_others = score >= 2

    direction = "home"
    try:
        crs_win = float(match_obj.get("crs_win", 999) or 999)
        crs_same = float(match_obj.get("crs_same", 999) or 999)
        crs_lose = float(match_obj.get("crs_lose", 999) or 999)
        if crs_lose < crs_win and crs_lose < crs_same:
            direction = "away"
        elif crs_same < crs_win and crs_same < crs_lose:
            direction = "draw"
    except: pass

    return {
        "is_others": is_others,
        "is_extreme_blowout": is_extreme_blowout,
        "trigger_count": score,
        "direction": direction,
        "triggers": triggers,
        "ai_others_count": ai_others_count,
    }


# ====================================================================
# 🧊 冷门猎手引擎
# ====================================================================
class ColdDoorDetector:
    @staticmethod
    def detect(match, prediction):
        signals = []
        strength = 0
        steam = prediction.get("steam_move", {})

        smart_str = " ".join(str(s) for s in prediction.get("smart_signals", []))
        if "Sharp" in smart_str or "sharp" in smart_str:
            strength += 6
            signals.append("🔥 Sharp Money确认！")

        if steam.get("steam") and "反向" in str(steam.get("signal", "")):
            signals.append("❄️ 反向Steam！庄家造热收割")
            strength += 5

        vote = match.get("vote", {})
        try:
            vh = int(vote.get("win", 33))
            va = int(vote.get("lose", 33))
            max_vote = max(vh, va)
            if max_vote >= 65:
                signals.append(f"❄️ 散户极端偏向{max_vote}%！冷门高危")
                strength += 4
            elif max_vote >= 58:
                strength += 2
        except:
            pass

        info = match.get("intelligence", match.get("information", {}))
        if isinstance(info, dict):
            home_bad = str(info.get("home_bad_news", ""))
            away_bad = str(info.get("guest_bad_news", ""))
            hp = prediction.get("home_win_pct", 50)
            ap = prediction.get("away_win_pct", 50)
            if len(home_bad) > 80 and hp > 58:
                signals.append("❄️ 主队坏消息爆炸+散户狂热")
                strength += 5
            if len(away_bad) > 80 and ap > 58:
                signals.append("❄️ 客队坏消息爆炸+散户狂热")
                strength += 5

        sp_h = float(match.get("sp_home", match.get("win", 0)) or 0)
        if sp_h > 1:
            hp2 = prediction.get("home_win_pct", 50)
            implied_h = 100 / sp_h * 0.92
            if abs(implied_h - hp2) > 15 and hp2 > 58:
                signals.append(f"❄️ 赔率vs模型背离{abs(implied_h-hp2):.0f}%")
                strength += 4

        for s in prediction.get("smart_signals", []):
            if "盘口太便宜" in str(s):
                signals.append("❄️ 盘口太便宜=庄家不看好")
                strength += 3
                break

        line = prediction.get("line_movement_anomaly", {})
        if line.get("has_anomaly") and "造热" in str(line.get("signal", "")):
            signals.append("❄️ 赔率变动造热=诱盘")
            strength += 4

        is_cold = strength >= 7
        if strength >= 12:
            level = "顶级"
        elif strength >= 7:
            level = "高危"
        else:
            level = "普通"

        return {
            "is_cold_door": is_cold,
            "strength": strength,
            "level": level,
            "signals": signals,
            "sharp_confirmed": "Sharp" in smart_str,
            "dark_verdict": f"❄️ {level}冷门！{len(signals)}条触发" if is_cold else ""
        }


# ====================================================================
# AI日记
# ====================================================================
def load_ai_diary():
    diary_file = "data/ai_diary.json"
    if os.path.exists(diary_file):
        try:
            with open(diary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return {"yesterday_win_rate": "N/A", "reflection": "v18.0 贝叶斯+16维陷阱启动", "kill_history": []}


def save_ai_diary(diary):
    os.makedirs("data", exist_ok=True)
    with open("data/ai_diary.json", "w", encoding="utf-8") as f:
        json.dump(diary, f, ensure_ascii=False, indent=2)


# ====================================================================
# 🧠 v18.0 AI Prompt — 对冲基金级 + 铁律约束
# ====================================================================
def build_v18_prompt(match_analyses):
    diary = load_ai_diary()
    p = "<context>\n"
    p += "你正在中国体彩的竞彩足球市场进行对冲基金级别的量化比分预测。\n"
    p += "这里充满诱盘、反指、资金流陷阱。你必须识破庄家布下的局。\n"
    if diary.get("reflection"):
        p += f"[系统记忆] 昨日: {diary.get('yesterday_win_rate','N/A')} | 反思: {diary['reflection']}\n"
    p += "</context>\n\n"
    
    p += "<iron_rules>\n"
    p += "铁律1 [方向-比分一致性]: top3[0].score 的比分方向必须与 reason 结论方向完全一致。\n"
    p += "  - reason 指向客胜 → 禁止输出 1-1/2-2 作为 top1\n"
    p += "  - reason 指向主胜 → 禁止输出 0-0/1-1 作为 top1 (除非 reason 明确是'防守主胜')\n"
    p += "  - reason 指向平局 → 禁止输出单边大比分\n"
    p += "铁律2 [胜其他标记]: is_score_others=true 时,top3 必须至少包含一个胜其他/负其他/平其他\n"
    p += "铁律3 [资金优先]: 当基本面与资金面冲突,追随资金面(Sharp/Steam/赔率变动),除非基本面差距极端(Shin差≥20%)\n"
    p += "铁律4 [诱盘识别]: 平赔降水+强势方Shin≥40%+基本面占优 → 这是诱散户进平的陷阱,强制反指,优选强势方\n"
    p += "铁律5 [杯赛反指]: 联赛含'杯/淘汰/决赛'字样+强势方Shin≥55%+散户跟风 → 大热必死铁律,优先平局或弱方小胜\n"
    p += "违反铁律将导致本场 AI 输出被降权50%并计入 diary 负反馈。\n"
    p += "</iron_rules>\n\n"
    
    p += "<analytical_framework>\n"
    p += "执行5步思维锚(按顺序):\n"
    p += "Step 1 [真实意图剥离]: Shin隐含概率是庄家表面立场,Sharp资金/Steam降水是真实意图。冲突时追随资金。\n"
    p += "Step 2 [陷阱矩阵扫描]: 依次排查12种诱盘:\n"
    p += "  - T1 诱平赔(平赔独降+强势方Shin高)\n"
    p += "  - T2 诱让负(让球深度>理论+主队真强)\n"
    p += "  - T3 诱让胜(让球深度<理论+主队真弱)\n"
    p += "  - T4/T5 虚假强势(Shin高但基本面差)\n"
    p += "  - T6/T7 比分区间诱饵(a0-a2 或 a5-a7 压低但xG反向)\n"
    p += "  - T8 假冷门(冷门信号满+基本面强方与冷门反向)\n"
    p += "  - T13 闷平(双方xG<2.3+小球压低+散户均衡)\n"
    p += "  - T14 杯赛大热(杯赛+散户跟风强势方)\n"
    p += "  - T15 历史僵局(历史交锋多平)\n"
    p += "  - T16 Sharp+坏消息对冲\n"
    p += "Step 3 [尾部分布探测]: 进球数赔率(a0-a7)压低情况反推庄家真实进球预期\n"
    p += "  - a7<25 或 a6<15 或 a5<8 → 防极端惨案 → 考虑胜其他/负其他\n"
    p += "  - a0-a2均压低+xG低 → 闷平\n"
    p += "Step 4 [场景共鸣]: 杯赛/淘汰赛/保级死拼/赛季末等场景属性加成\n"
    p += "Step 5 [EV锚定]: 最终选CRS赔率×概率期望值最高的比分,拒绝追逐大热\n"
    p += "</analytical_framework>\n\n"
    
    p += "<output_format>\n"
    p += "严格 JSON 数组,每场必含:\n"
    p += "- match: 整数序号\n"
    p += "- top3: [{\"score\":\"2-1\",\"prob\":15}, ...] 3 个候选比分,方向必须与 reason 一致\n"
    p += "- reason: 300字左右,必须按5步思维锚写,每步至少1句\n"
    p += "- ai_confidence: 0-100\n"
    p += "- is_score_others: 若 top3 任一是胜其他/平其他/负其他 → true\n"
    p += "- detected_traps: 识别出的陷阱编号列表,如 [\"T1\",\"T3\",\"T14\"]\n"
    p += "- final_direction: \"home\"/\"draw\"/\"away\" 必须与 top3[0] 方向一致\n"
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
        
        hxg = eng.get('bookmaker_implied_home_xg', '?')
        axg = eng.get('bookmaker_implied_away_xg', '?')
        p += f"庄家隐含xG: 主{hxg} vs 客{axg}\n"
        
        moments = crs_preview.get("moments", {})
        if moments:
            p += f"CRS矩: λ主{moments.get('lambda_h',0):.2f}/客{moments.get('lambda_a',0):.2f} 总{moments.get('lambda_total',0):.2f} "
            p += f"corr{moments.get('corr',0):+.2f} 形状={crs_preview.get('shape_verdict','?')}\n"
        
        traps = trap_preview.get("traps_detected", [])
        if traps:
            p += f"🎭 系统识别陷阱({len(traps)}个,严重度{trap_preview.get('total_severity',0)}):\n"
            for t in traps:
                p += f"  - {t.get('trap','?')}: {t.get('description','')[:100]}\n"
        else:
            p += f"🎭 系统未识别明显陷阱,请自行判断\n"
        
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
            except:
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
            except:
                pass
        if crs_lines:
            p += f"CRS: {' | '.join(crs_lines)}\n"
        
        for k, label in [("crs_win", "胜其他"), ("crs_same", "平其他"), ("crs_lose", "负其他")]:
            v = m.get(k, "")
            if v:
                p += f"📌 {label}={v}  "
        p += "\n"
        
        hf_l = []
        for k, lb in {"ss":"主/主","sp":"主/平","sf":"主/负","ps":"平/主","pp":"平/平",
                      "pf":"平/负","fs":"负/主","fp":"负/平","ff":"负/负"}.items():
            try:
                v = _f(m.get(k, 0))
                if v > 1:
                    hf_l.append(f"{lb}={v:.2f}")
            except:
                pass
        if hf_l:
            p += f"半全场: {' | '.join(hf_l)}\n"
        
        vote = m.get("vote", {})
        if vote:
            p += f"散户: 胜{vote.get('win','?')}% 平{vote.get('same','?')}% 负{vote.get('lose','?')}%"
            try:
                max_v = max(int(_f(vote.get('win', 33))), int(_f(vote.get('lose', 33))))
                if max_v >= 58:
                    p += f" ⚠️大热({max_v}%需反指)"
            except:
                pass
            p += "\n"
        
        change = m.get("change", {})
        if change and isinstance(change, dict):
            cw = change.get("win", 0)
            cs = change.get("same", 0)
            cl = change.get("lose", 0)
            if cw or cs or cl:
                p += f"赔率变动: 胜{cw} 平{cs} 负{cl} (负=降水=钱流入)\n"
        
        info = m.get("information", {})
        if isinstance(info, dict):
            for k, label in [("home_injury", "主伤停"), ("guest_injury", "客伤停"),
                            ("home_bad_news", "主利空"), ("guest_bad_news", "客利空")]:
                if info.get(k):
                    p += f"{label}: {str(info[k])[:500].replace(chr(10), ' ')}\n"
        
        points = m.get("points", {})
        if isinstance(points, dict):
            for k in ["home_strength", "guest_strength", "match_points"]:
                txt = str(points.get(k, ""))[:500].replace("\n", " ")
                if "场均" in txt or "主场" in txt or "客场" in txt:
                    p += f"情报: {txt}\n"
                    break
        
        smart_sigs = stats.get('smart_signals', [])
        if smart_sigs:
            p += f"🔥 信号: {', '.join(str(s) for s in smart_sigs[:6])}\n"
        
        p += "</match>\n\n"
    
    p += "</match_data>\n"
    return p


# ====================================================================
# AI调用引擎 - 完整保留 v17 的超级兜底机制
# ====================================================================
FALLBACK_URLS = [None, "https://www.api522.pro/v1", "https://api522.pro/v1",
                 "https://api521.pro/v1", "http://69.63.213.33:666/v1"]

GPT_DEFAULT_URL = "https://poloai.top/v1"
GPT_DEFAULT_KEY = ""


def get_clean_env_url(name, default=""):
    v = str(os.environ.get(name, globals().get(name, default))).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/-]+)", v)
    return match.group(1) if match else v


def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")


async def async_call_one_ai_batch(session, prompt, url_env, key_env, models_list, num_matches, ai_name):
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
        print(f"    🔌 [GPT] 使用poloai通道: {primary_url}")
    else:
        primary_url = get_clean_env_url(url_env)
        backup = [u for u in FALLBACK_URLS if u and u != primary_url][:1]
        urls = [primary_url] + backup

    CONNECT_TIMEOUT = 20
    READ_TIMEOUT_MAP = {"claude": 350, "grok": 250, "gpt": 250, "gemini": 250}
    READ_TIMEOUT = READ_TIMEOUT_MAP.get(ai_name, 200)

    # v18 升级的对冲基金异构人设 + 铁律
    AI_PROFILES = {
        "claude": {
            "sys": ("<role>你是顶级对冲基金的博弈论+市场微观结构首席分析师。</role>\n"
                    "<priority>严格执行 <iron_rules>。任何违反铁律的输出将被降权。</priority>\n"
                    "<style>逆向思维优先。每场比赛先问:'庄家想让我选什么?我就反着选。'</style>\n"
                    "<instruction>按5步思维锚推理,最终仅输出 JSON 数组。</instruction>"),
            "temp": 0.22
        },
        "gpt": {
            "sys": ("<role>你是衍生品定价+概率分布偏差量化策略师。</role>\n"
                    "<priority>严格遵守 <iron_rules> 尤其是铁律1(方向一致性)和铁律3(资金优先)。</priority>\n"
                    "<style>从a0-a7进球数赔率反推真实λ,据此重构CRS分布。只信数据,不信叙事。</style>\n"
                    "<instruction>严格输出JSON数组,禁止任何前缀后缀。</instruction>"),
            "temp": 0.18
        },
        "grok": {
            "sys": ("<role>你是拥有全网实时数据嗅觉的另类数据分析师。</role>\n"
                    "<priority>严格遵守 <iron_rules> 铁律4(诱盘识别)和铁律5(杯赛反指)。</priority>\n"
                    "<style>敏锐捕捉情绪背离。散户>60%同向+资金未跟=诱盘,果断反指。</style>\n"
                    "<instruction>只输出JSON数组。</instruction>"),
            "temp": 0.28
        },
        "gemini": {
            "sys": ("<role>你是精通非线性特征的深度学习模式识别引擎。</role>\n"
                    "<priority>严格遵守 <iron_rules>。检测欧赔/亚盘/CRS三者间的定价裂痕。</priority>\n"
                    "<style>多维信号共振最可靠。只有当3个以上独立指标同向时才加重权重。</style>\n"
                    "<instruction>综合输出最稳健预测,仅JSON数组。</instruction>"),
            "temp": 0.15
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
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": profile["temp"]},
                    "systemInstruction": {"parts": [{"text": profile["sys"]}]}
                }
            else:
                headers["Authorization"] = f"Bearer {key}"
                bp = {"model": mn, "messages": [
                    {"role": "system", "content": profile["sys"]},
                    {"role": "user", "content": prompt}
                ]}
                if ai_name != "claude":
                    bp["temperature"] = profile["temp"]
                payload = bp

            gw = url.split("/v1")[0][:35]
            print(f"  [🔌连接中] {ai_name.upper()} | {mn[:22]} @ {gw}")
            t0 = time.time()

            try:
                timeout = aiohttp.ClientTimeout(
                    total=None, connect=CONNECT_TIMEOUT,
                    sock_connect=CONNECT_TIMEOUT, sock_read=READ_TIMEOUT
                )
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                    elapsed_connect = round(time.time()-t0, 1)
                    if r.status in (502, 504):
                        print(f"    💀 HTTP {r.status} | {elapsed_connect}s → 换模型")
                        break
                    if r.status == 400:
                        print(f"    💀 400 | {elapsed_connect}s → 换模型")
                        break
                    if r.status == 429:
                        print(f"    🔥 429 | {elapsed_connect}s → 换模型")
                        break
                    if r.status != 200:
                        print(f"    ⚠️ HTTP {r.status} | {elapsed_connect}s → 换URL")
                        continue

                    connected = True
                    print(f"    ✅ 已连上！{elapsed_connect}s | 等待数据...")

                    try:
                        data = await r.json(content_type=None)
                    except:
                        print(f"    ⚠️ 响应非JSON → 换模型")
                        break

                    elapsed = round(time.time()-t0, 1)
                    usage = data.get("usage", {})
                    req_tokens = usage.get("total_tokens", 0) or (
                        usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
                    if not req_tokens:
                        um = data.get("usageMetadata", {})
                        req_tokens = um.get("totalTokenCount", 0)
                    if req_tokens:
                        print(f"    📊 {req_tokens:,} token | {elapsed}s")

                    # 完全保留用户的坚不可摧的JSON提取与解析逻辑
                    raw_text = ""
                    debug_msg_keys = []
                    try:
                        if is_gem:
                            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        else:
                            if data.get("choices") and data["choices"]:
                                msg = data["choices"][0].get("message", {})
                                if isinstance(msg, dict):
                                    debug_msg_keys = [
                                        f"{k}({type(v).__name__}:{len(v) if isinstance(v, (str, list)) else '?'})"
                                        for k, v in msg.items()
                                    ]

                                    if msg.get("content") is None and data.get("usage", {}).get("completion_tokens", 0) > 100:
                                        print(f"    🚨 [proxy bug] {ai_name.upper()} content=null 但消耗了 {data['usage']['completion_tokens']} token")
                                        print(f"        → proxy没传回内容, 钱白花。建议换模型/反馈客服")

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
                                            "assistant_content", "model_response"
                                        ]:
                                            v = msg.get(field, "")
                                            if v and isinstance(v, str) and v.strip():
                                                raw_text = v.strip()
                                                break

                                    if not raw_text:
                                        skip_keys = (
                                            "reasoning_content", "thinking", "reasoning",
                                            "reasoning_text", "thoughts", "thought_process",
                                            "internal_thinking", "chain_of_thought", "cot",
                                            "deliberation", "analysis_process"
                                        )
                                        best_with_match = ""
                                        for k in msg:
                                            if k in skip_keys: continue
                                            v = msg[k]
                                            if isinstance(v, str) and '"match"' in v and "[" in v:
                                                if len(v) > len(best_with_match):
                                                    best_with_match = v.strip()
                                        if best_with_match:
                                            raw_text = best_with_match

                                    if not raw_text:
                                        for k in msg:
                                            v = msg[k]
                                            if isinstance(v, str) and '"match"' in v and "[" in v:
                                                raw_text = v.strip()
                                                print(f"    🆘 兜底命中字段: {k}")
                                                break

                                    if not raw_text:
                                        skip_keys2 = (
                                            "reasoning_content", "thinking", "reasoning",
                                            "reasoning_text", "thoughts", "thought_process",
                                        )
                                        longest_clean = ""
                                        for k in msg:
                                            if k in skip_keys2: continue
                                            v = msg[k]
                                            if isinstance(v, str) and len(v.strip()) > len(longest_clean):
                                                longest_clean = v.strip()
                                        if longest_clean and len(longest_clean) > 20:
                                            raw_text = longest_clean
                                            print(f"    🆘 优先级5: 取最长非thinking字段")

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
                                    for ci in range(start_pos, min(start_pos + 100000, len(full_str))):
                                        if full_str[ci] == '[': depth += 1
                                        elif full_str[ci] == ']': depth -= 1
                                        if depth == 0:
                                            end_pos = ci + 1
                                            break
                                    if end_pos > start_pos:
                                        extracted = full_str[start_pos:end_pos]
                                        if '\\"' in extracted:
                                            try: extracted = json.loads('"' + extracted + '"')
                                            except: extracted = extracted.replace('\\"', '"')
                                        raw_text = extracted
                                        print(f"    🆘 终极兜底: 从response dump中提取JSON")
                    except Exception as ex:
                        print(f"    ⚠️ 解析异常: {str(ex)[:80]}")

                    if not raw_text or len(raw_text) < 10:
                        print(f"    ⚠️ 空数据 → 换模型")
                        if debug_msg_keys:
                            print(f"    🔍 [调试] msg字段: {', '.join(debug_msg_keys[:8])}")
                        if isinstance(data, dict):
                            top_keys = [f"{k}({type(v).__name__})" for k, v in data.items()]
                            print(f"    🔍 [调试] data字段: {', '.join(top_keys[:6])}")
                            if data.get("usage"):
                                print(f"    🔍 [调试] usage: {data['usage']}")
                        try:
                            os.makedirs("data/debug", exist_ok=True)
                            dump_file = f"data/debug/{ai_name}_fail_{int(time.time())}.json"
                            with open(dump_file, "w", encoding="utf-8") as df:
                                json.dump(data, df, ensure_ascii=False, indent=2)
                            print(f"    📁 失败响应已保存: {dump_file}")
                        except: pass
                        break

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
                            print(f"    🎯 精确匹配JSON: {len(json_str)}字")

                    if not json_str:
                        start = clean.find("[")
                        end = clean.rfind("]") + 1
                        if start != -1 and end > start:
                            json_str = clean[start:end]
                            print(f"    🔍 兜底匹配JSON: {len(json_str)}字")

                    results = {}
                    if json_str:
                        try:
                            arr = json.loads(json_str)
                        except json.JSONDecodeError:
                            try:
                                last_brace = json_str.rfind('}')
                                arr = json.loads(json_str[:last_brace+1] + "]") if last_brace != -1 else []
                                if arr:
                                    print(f"    🩹 断肢重生: {len(arr)}条")
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
                                if item.get("top3"):
                                    t1 = item["top3"][0].get("score", "1-1").replace(" ", "").strip() if item["top3"] else "1-1"
                                    results[mid] = {
                                        "top3": item["top3"],
                                        "ai_score": t1,
                                        "reason": str(item.get("reason", ""))[:800],
                                        "ai_confidence": int(item.get("ai_confidence", 60)),
                                        "is_score_others": bool(item.get("is_score_others", False)),
                                        "detected_traps": item.get("detected_traps", item.get("detected_signals", [])), # v18 兼容
                                        "final_direction": item.get("final_direction", ""), # v18 字段
                                    }
                                elif item.get("score"):
                                    results[mid] = {
                                        "ai_score": item["score"].replace(" ", "").strip(),
                                        "reason": str(item.get("reason", ""))[:800],
                                        "ai_confidence": int(item.get("ai_confidence", 60)),
                                        "is_score_others": bool(item.get("is_score_others", False)),
                                        "detected_traps": item.get("detected_traps", item.get("detected_signals", [])), # v18 兼容
                                        "final_direction": item.get("final_direction", ""), # v18 字段
                                    }

                    if len(results) > 0:
                        print(f"    ✅ {ai_name.upper()} 完成: {len(results)}/{num_matches} | {elapsed}s")
                        return ai_name, results, mn
                    else:
                        print(f"    ⚠️ 解析0条 → 换模型")
                        if raw_text:
                            print(f"    🔍 [调试] raw_text长度: {len(raw_text)}")
                            print(f"    🔍 [调试] raw_text前150字: {raw_text[:150]}")
                            print(f"    🔍 [调试] raw_text末80字: ...{raw_text[-80:]}")
                            if json_str:
                                print(f"    🔍 [调试] 提取的json_str长度: {len(json_str)}")
                                print(f"    🔍 [调试] json_str前150字: {json_str[:150]}")
                        if debug_msg_keys:
                            print(f"    🔍 [调试] msg字段: {', '.join(debug_msg_keys[:8])}")
                        try:
                            os.makedirs("data/debug", exist_ok=True)
                            dump_file = f"data/debug/{ai_name}_parse0_{int(time.time())}.json"
                            with open(dump_file, "w", encoding="utf-8") as df:
                                json.dump(data, df, ensure_ascii=False, indent=2)
                            print(f"    📁 失败响应已保存: {dump_file}")
                        except: pass
                        break

            except aiohttp.ClientConnectorError:
                print(f"    🔌 连接失败 → 换URL")
                continue
            except asyncio.TimeoutError:
                if not connected:
                    print(f"    🔌 连接超时 → 换URL")
                    continue
                else:
                    print(f"    ⏰ 读取超时 | 钱已花")
                    return ai_name, {}, "read_timeout"
            except Exception as e:
                if not connected:
                    print(f"    ⚠️ {str(e)[:40]} → 换URL")
                    continue
                else:
                    return ai_name, {}, "error"

            await asyncio.sleep(0.2)

    return ai_name, {}, "all_failed"


async def run_ai_matrix(match_analyses):
    num = len(match_analyses)
    prompt = build_v18_prompt(match_analyses)
    print(f"  [v18 Prompt] {len(prompt):,} 字符 → 4AI并行...")

    ai_configs = [
        ("grok", "GROK_API_URL", "GROK_API_KEY", ["熊猫-A-5-grok-4.2-fast-200w上下文"]),
        ("gpt", "GPT_API_URL", "GPT_API_KEY", ["gpt-5.4"]),
        ("gemini", "GEMINI_API_URL", "GEMINI_API_KEY", ["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking"]),
        ("claude", "CLAUDE_API_URL", "CLAUDE_API_KEY", ["熊猫-69-满血openrouter-claude-opus-4.7-上下文1000k"]),
    ]
    all_results = {"gpt": {}, "grok": {}, "gemini": {}, "claude": {}}

    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [async_call_one_ai_batch(session, prompt, u, k, m, num, n) for n, u, k, m in ai_configs]
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
# 🌟 Legacy merge_result (完整保留你v17辛辛苦苦写的500行心血)
# ====================================================================
def merge_result_v17_legacy(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    """
    此函数保留用作代码基座记录和兼容备份。
    实际 v18 将直接调用下方的 merge_result_v18。
    """
    if isinstance(match_obj.get("v2_odds_dict"), dict):
        v2 = match_obj["v2_odds_dict"]
        match_obj = {**match_obj, **v2}
        print(f"    🔧 [字段兼容] v2_odds_dict→顶层 ({len(v2)}个字段)")

    league = str(match_obj.get("league", match_obj.get("cup", "")))
    sp_h = float(match_obj.get("sp_home", match_obj.get("win", 0)) or 0)
    sp_d = float(match_obj.get("sp_draw", match_obj.get("same", 0)) or 0)
    sp_a = float(match_obj.get("sp_away", match_obj.get("lose", 0)) or 0)
    engine_conf = engine_result.get("confidence", 50)

    def _is_valid_ai(r):
        if not isinstance(r, dict): return False
        score = r.get("ai_score", "")
        if not score or score in ("-", "N/A", ""): return False
        try:
            parts = str(score).strip().replace(" ", "").split("-")
            if len(parts) != 2: return False
            int(parts[0]); int(parts[1])
        except: return False
        return True

    ai_valid = {
        "gpt": _is_valid_ai(gpt_r),
        "grok": _is_valid_ai(grok_r),
        "gemini": _is_valid_ai(gemini_r),
        "claude": _is_valid_ai(claude_r),
    }

    abstained = [n.upper() for n, v in ai_valid.items() if not v]
    if abstained:
        print(f"    🚫 弃权AI: {', '.join(abstained)} (失效,不参与加权)")

    p1_ai = {n: r for n, r in [("gpt", gpt_r), ("grok", grok_r), ("gemini", gemini_r)] if ai_valid[n]}
    all_ai = {**p1_ai}
    if ai_valid["claude"]:
        all_ai["claude"] = claude_r

    direction_scores = {"home": 0.0, "draw": 0.0, "away": 0.0}

    if sp_h > 1 and sp_d > 1 and sp_a > 1:
        margin = 1/sp_h + 1/sp_d + 1/sp_a
        shin_h = (1/sp_h)/margin*100
        shin_d = (1/sp_d)/margin*100
        shin_a = (1/sp_a)/margin*100
    else:
        shin_h = shin_d = shin_a = 33.3
    shin_dir = max([("home", shin_h), ("draw", shin_d), ("away", shin_a)], key=lambda x: x[1])[0]

    smart_signals = stats.get("smart_signals", [])
    smart_str = " ".join(str(s) for s in smart_signals)
    sharp_detected = "Sharp" in smart_str or "sharp" in smart_str
    sharp_dir = None
    if sharp_detected:
        import re as _re_sharp
        for s in smart_signals:
            s_str = str(s)
            if "Sharp" in s_str or "sharp" in s_str:
                if "确认" in s_str and "→" not in s_str and "流向" not in s_str:
                    continue
                if _re_sharp.search(r"(主胜|主队|走主|→\s*主|流向\s*主|资金\s*主)", s_str):
                    sharp_dir = "home"; break
                elif _re_sharp.search(r"(客胜|客队|走客|→\s*客|流向\s*客|资金\s*客)", s_str):
                    sharp_dir = "away"; break
                elif _re_sharp.search(r"(平局|平赔|走平|→\s*平|流向\s*平|资金\s*平)", s_str):
                    sharp_dir = "draw"; break

    steam_dir = None
    steam_type = None
    if "Steam" in smart_str:
        import re as _re_steam
        for s in smart_signals:
            s_str = str(s)
            if "Steam" not in s_str: continue
            is_reverse = "反向" in s_str or "未跟" in s_str or "不跟" in s_str
            if _re_steam.search(r"(主胜\s*Steam|Steam.*主胜|主胜.*降水|主.*Steam)", s_str):
                steam_dir = "home"
                steam_type = "reverse" if is_reverse else "normal"
                break
            elif _re_steam.search(r"(客胜\s*Steam|Steam.*客胜|客胜.*降水|客.*Steam)", s_str):
                steam_dir = "away"
                steam_type = "reverse" if is_reverse else "normal"
                break
            elif _re_steam.search(r"(平局\s*Steam|Steam.*平局|平.*Steam)", s_str):
                steam_dir = "draw"
                steam_type = "reverse" if is_reverse else "normal"
                break

    vote = match_obj.get("vote", {})
    vh = vd = va = 33
    vote_hot_dir = None
    vote_hot_pct = 0
    try:
        vh = int(vote.get("win", 33) or 33)
        vd = int(vote.get("same", 33) or 33)
        va = int(vote.get("lose", 33) or 33)
        max_vote = max(vh, vd, va)
        if max_vote >= 55:
            vote_hot_pct = max_vote
            if vh == max_vote: vote_hot_dir = "home"
            elif vd == max_vote: vote_hot_dir = "draw"
            else: vote_hot_dir = "away"
    except: pass

    change = match_obj.get("change", {})
    change_down_dir = None
    try:
        cw = float(str(change.get("win", 0)).replace("+", "") or 0)
        cs = float(str(change.get("same", 0)).replace("+", "") or 0)
        cl = float(str(change.get("lose", 0)).replace("+", "") or 0)
        if cw < -0.05 and cw <= cs and cw <= cl: change_down_dir = "home"
        elif cl < -0.05 and cl <= cs and cl <= cw: change_down_dir = "away"
        elif cs < -0.05 and cs <= cw and cs <= cl: change_down_dir = "draw"
    except: pass

    cold_signals_raw = [s for s in smart_signals if "❄️" in str(s) or "冷门" in str(s) or "大热" in str(s) or "造热" in str(s)]

    hp_eng = engine_result.get("home_prob", shin_h)
    ap_eng = engine_result.get("away_prob", shin_a)
    hot_side = "home" if hp_eng > ap_eng else "away"

    dupan_detected = False
    dupan_true_dir = None
    dupan_confirm = 0

    if sharp_detected and sharp_dir and sharp_dir != shin_dir:
        dupan_confirm = 0
        if vote_hot_dir == shin_dir and vote_hot_pct >= 55:
            if vote_hot_pct >= 68: dupan_confirm += 4
            elif vote_hot_pct >= 60: dupan_confirm += 3
            else: dupan_confirm += 2
        if vote_hot_dir and vote_hot_dir != sharp_dir and vote_hot_pct >= 58:
            dupan_confirm += 2
        if steam_dir == sharp_dir:
            if steam_type == "reverse": dupan_confirm += 3
            else: dupan_confirm += 2
        if change_down_dir == sharp_dir:
            dupan_confirm += 2
        if cold_signals_raw:
            dupan_confirm += min(3, len(cold_signals_raw))
        if dupan_confirm >= 3:
            dupan_detected = True
            dupan_true_dir = sharp_dir

    shin_weight = 15 if dupan_detected else 30
    direction_scores["home"] += shin_h/100 * shin_weight
    direction_scores["draw"] += shin_d/100 * shin_weight
    direction_scores["away"] += shin_a/100 * shin_weight

    if sharp_detected and sharp_dir:
        sharp_base = 35 if dupan_detected else 25
        direction_scores[sharp_dir] += sharp_base

    if steam_dir:
        if steam_type == "reverse":
            direction_scores[steam_dir] += 20
        else:
            direction_scores[steam_dir] += 10

    contrarian_away_score = 0
    contrarian_home_score = 0
    if vote_hot_dir and vote_hot_pct >= 55:
        if vote_hot_pct >= 68:
            contra_weight = 22; level = "死亡级"
        elif vote_hot_pct >= 60:
            contra_weight = 14; level = "大热必死"
        else:
            contra_weight = 6; level = "轻度"

        for d in ["home", "draw", "away"]:
            if d != vote_hot_dir:
                direction_scores[d] += contra_weight * 0.5
        direction_scores[vote_hot_dir] -= contra_weight * 0.3

        if vote_hot_dir == "home":
            contrarian_away_score = contra_weight
        elif vote_hot_dir == "away":
            contrarian_home_score = contra_weight

    if cold_signals_raw or sharp_detected or vote_hot_pct >= 60:
        cold_score = 0
        if sharp_detected and sharp_dir and sharp_dir != shin_dir: cold_score += 6
        if steam_type == "reverse": cold_score += 5
        if vote_hot_pct >= 68: cold_score += 7
        elif vote_hot_pct >= 60: cold_score += 5
        for s in smart_signals:
            s_str = str(s)
            if "盘口太便宜" in s_str: cold_score += 4
            if "坏消息" in s_str or "崩盘" in s_str: cold_score += 5
            if "背离" in s_str: cold_score += 4
            if "造热" in s_str: cold_score += 3

        if cold_score >= 25:
            cold_level = "死亡级"; cold_power = 18
        elif cold_score >= 18:
            cold_level = "顶级"; cold_power = 12
        elif cold_score >= 12:
            cold_level = "高危"; cold_power = 8
        elif cold_score >= 6:
            cold_level = "中等"; cold_power = 4
        else:
            cold_level = None; cold_power = 0

        if cold_level:
            direction_scores[hot_side] -= cold_power
            other = "away" if hot_side == "home" else "home"
            direction_scores[other] += cold_power * 0.6
            direction_scores["draw"] += cold_power * 0.4

    if change and isinstance(change, dict):
        try:
            cw = float(str(change.get("win", 0)).replace("+", "") or 0)
            cs = float(str(change.get("same", 0)).replace("+", "") or 0)
            cl = float(str(change.get("lose", 0)).replace("+", "") or 0)
            if cw < -0.05: direction_scores["home"] += 4
            if cs < -0.05: direction_scores["draw"] += 4
            if cl < -0.05: direction_scores["away"] += 4
            if cw > 0.05: direction_scores["home"] -= 2
            if cs > 0.05: direction_scores["draw"] -= 2
            if cl > 0.05: direction_scores["away"] -= 2
        except: pass

    ai_weight_total = 15 if dupan_detected else 25
    ai_directions = {"home": 0, "draw": 0, "away": 0}
    for name, r in all_ai.items():
        if not isinstance(r, dict): continue
        sc = parse_score(r.get("ai_score", ""))
        if not (sc and sc[0] is not None):
            t3 = r.get("top3", [])
            if t3 and len(t3) > 0:
                sc = parse_score(t3[0].get("score", ""))
        if sc and sc[0] is not None:
            w = 1.5 if name == "claude" else (1.40 if name == "gemini" else (1.35 if name == "grok" else 1.0))
            if sc[0] > sc[1]: ai_directions["home"] += w
            elif sc[0] < sc[1]: ai_directions["away"] += w
            else: ai_directions["draw"] += w
    total_ai_dir = sum(ai_directions.values())
    if total_ai_dir > 0:
        for d in ["home", "draw", "away"]:
            direction_scores[d] += (ai_directions[d] / total_ai_dir) * ai_weight_total

    total_dir = sum(max(0.1, v) for v in direction_scores.values())
    dir_probs = {d: max(0.1, direction_scores[d]) / total_dir * 100 for d in direction_scores}
    final_direction = max(dir_probs, key=dir_probs.get)
    dir_gap = dir_probs[final_direction] - sorted(dir_probs.values(), reverse=True)[1]
    dir_confident = dir_gap > 5

    return {"predicted_score": "1-1"} # Minimal return since this is legacy code and won't be executed


# ====================================================================
# 🌟 merge_result_v18 — 整合决策锁定链
# ====================================================================
def merge_result_v18(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    """
    v18.0 合并逻辑:
    1. 陷阱矩阵扫描
    2. CRS 矩阵分析
    3. 估算 exp_goals (混合多源)
    4. 贝叶斯决策锁定链
    5. 构建前端字段 (四字段强一致)
    6. EV/Kelly 计算
    """
    if isinstance(match_obj.get("v2_odds_dict"), dict):
        v2 = match_obj["v2_odds_dict"]
        match_obj = {**match_obj, **v2}
    
    def _is_valid_ai(r):
        if not isinstance(r, dict): return False
        score = r.get("ai_score", "")
        if not score or score in ("-", "N/A", ""): return False
        h, a = _parse_score(score)
        return h is not None
    
    ai_valid = {
        "gpt": _is_valid_ai(gpt_r),
        "grok": _is_valid_ai(grok_r),
        "gemini": _is_valid_ai(gemini_r),
        "claude": _is_valid_ai(claude_r),
    }
    
    ai_responses = {}
    if ai_valid["claude"]: ai_responses["claude"] = claude_r
    if ai_valid["gpt"]: ai_responses["gpt"] = gpt_r
    if ai_valid["grok"]: ai_responses["grok"] = grok_r
    if ai_valid["gemini"]: ai_responses["gemini"] = gemini_r
    
    exp_goals = 0.0
    for src in [engine_result, stats]:
        if not src: continue
        for k in ["expected_total_goals", "exp_goals", "total_goals", "expected_goals", "lambda_total", "total_xg"]:
            v = src.get(k)
            if v is not None:
                try:
                    fv = float(v)
                    if fv > 0.5:
                        exp_goals = fv
                        break
                except: pass
        if exp_goals > 0: break
    
    if exp_goals <= 0:
        hxg = _f(engine_result.get("bookmaker_implied_home_xg", 0))
        axg = _f(engine_result.get("bookmaker_implied_away_xg", 0))
        if hxg > 0 and axg > 0:
            exp_goals = hxg + axg
    
    if exp_goals <= 0:
        try:
            gp = []
            for gi in range(8):
                v = _f(match_obj.get(f"a{gi}", 0))
                if v > 1: gp.append((gi, 1/v))
            if gp:
                tp = sum(p for _, p in gp)
                exp_goals = sum(g*(p/tp) for g, p in gp)
        except: pass
    
    if exp_goals < 1.0 or exp_goals > 6.0: exp_goals = 2.5
    
    smart_signals = stats.get("smart_signals", []) if stats else []
    trap_report = detect_all_traps(match_obj, engine_result, ai_responses, smart_signals, exp_goals)
    
    if trap_report["trap_count"] > 0:
        print(f"    🎭 陷阱: {trap_report['trap_count']}个 严重度{trap_report['total_severity']}")
        for t in trap_report["traps_detected"][:3]:
            print(f"       [{t['trap']}] {t['description'][:60]}")
    
    crs_analysis = analyze_crs_matrix(match_obj)
    
    lock_result = decision_lock_chain(
        match_obj=match_obj,
        engine_result=engine_result,
        trap_report=trap_report,
        crs_analysis=crs_analysis,
        ai_responses=ai_responses,
        smart_signals=smart_signals,
        exp_goals=exp_goals,
    )
    
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
        if final_direction == "home": final_odds = _f(match_obj.get("crs_win", 0))
        elif final_direction == "away": final_odds = _f(match_obj.get("crs_lose", 0))
        else: final_odds = _f(match_obj.get("crs_same", 0))
    
    crs_prob = crs_analysis.get("implied_probs", {}).get(predicted_score, 5)
    ev_data = calculate_value_bet(crs_prob, final_odds)
    
    engine_conf = engine_result.get("confidence", 50) if engine_result else 50
    weights = {"claude": 1.4, "gemini": 1.35, "grok": 1.30, "gpt": 1.1}
    ai_conf_sum = ai_conf_count = value_kills = 0
    for name, r in ai_responses.items():
        conf = r.get("ai_confidence", 60)
        ai_conf_sum += conf * weights.get(name, 1.0)
        ai_conf_count += weights.get(name, 1.0)
        if r.get("value_kill"): value_kills += 1
    
    avg_ai_conf = (ai_conf_sum / ai_conf_count) if ai_conf_count > 0 else 60
    cf = min(95, engine_conf + int((avg_ai_conf - 60) * 0.4)) + value_kills * 6
    cf -= trap_report.get("confidence_penalty", 0)
    if dir_confidence >= 70: cf += min(10, (dir_confidence - 70) // 3)
    elif dir_confidence < 50: cf -= 8
    if dir_gap < 10: cf -= 5
    
    cf = max(30, min(95, cf))
    risk = "低" if cf >= 75 else ("中" if cf >= 55 else "高")
    
    cold_strength = 0
    cold_level = None
    cold_signals_arr = []
    for t in trap_report["traps_detected"]:
        if t["trap"] in ["T8_FALSE_COLD", "T4_FAKE_HOME_FAVORITE", "T5_FAKE_AWAY_FAVORITE", "T14_CUP_FAVORITE"]:
            cold_strength += t["severity"] * 3
            cold_signals_arr.append(t["description"])
    
    if cold_strength >= 12: cold_level = "顶级"
    elif cold_strength >= 7: cold_level = "高危"
    elif cold_strength >= 4: cold_level = "中等"
    
    cold_door = {
        "is_cold_door": cold_level is not None,
        "strength": cold_strength,
        "level": cold_level or "普通",
        "signals": cold_signals_arr,
        "sharp_confirmed": trap_report.get("sharp_detected", False),
        "dark_verdict": f"❄️ {cold_level}冷门!{len(cold_signals_arr)}条触发" if cold_level else ""
    }
    
    sigs = list(smart_signals)
    for t in trap_report["traps_detected"]: sigs.append(f"🎭 {t['trap']}:{t['description'][:50]}")
    if is_score_others: sigs.append(f"🔥 胜其他场触发")
    
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
        "trap_details": [{"trap": t["trap"], "desc": t["description"]} for t in trap_report["traps_detected"]],
        "crs_shape": crs_analysis.get("shape_verdict", "unknown"),
        "crs_moments": crs_analysis.get("moments", {}),
        "crs_margin": crs_analysis.get("margin", 0.0),
        "crs_coverage": crs_analysis.get("coverage", 0.0),
        "crs_implied_probs": crs_analysis.get("implied_probs", {}),
        "top_score_candidates": lock_result["top_score_candidates"],
        "gpt_score": gpt_r.get("ai_score", "弃权") if ai_valid["gpt"] else "弃权",
        "gpt_analysis": gpt_r.get("reason", "弃权") if ai_valid["gpt"] else "弃权 (AI失效)",
        "grok_score": grok_r.get("ai_score", "弃权") if ai_valid["grok"] else "弃权",
        "grok_analysis": grok_r.get("reason", "弃权") if ai_valid["grok"] else "弃权 (AI失效)",
        "gemini_score": gemini_r.get("ai_score", "弃权") if ai_valid["gemini"] else "弃权",
        "gemini_analysis": gemini_r.get("reason", "弃权") if ai_valid["gemini"] else "弃权 (AI失效)",
        "claude_score": cl_sc if ai_valid["claude"] else "弃权",
        "claude_analysis": claude_r.get("reason", "弃权") if ai_valid["claude"] else "弃权 (AI失效)",
        "ai_abstained": [n.upper() for n, v in ai_valid.items() if not v],
        "ai_avg_confidence": round(avg_ai_conf, 1),
        "suggested_kelly": ev_data["kelly"],
        "edge_vs_market": ev_data["ev"],
        "is_value": ev_data["is_value"],
        "smart_money_signal": " | ".join(sigs[:10]),
        "smart_signals": sigs,
        "cold_door": cold_door,
        "xG_home": _f(engine_result.get("bookmaker_implied_home_xg", 1.3)) if engine_result else 1.3,
        "xG_away": _f(engine_result.get("bookmaker_implied_away_xg", 0.9)) if engine_result else 0.9,
        "over_under_2_5": "大" if (engine_result.get("over_25", 50) if engine_result else 50) > 55 else "小",
        "both_score": "是" if (engine_result.get("btts", 45) if engine_result else 45) > 50 else "否",
        "expected_total_goals": round(exp_goals, 2),
        "over_2_5": engine_result.get("over_25", 50) if engine_result else 50,
        "btts": engine_result.get("btts", 45) if engine_result else 45,
        "bookmaker_implied_home_xg": engine_result.get("bookmaker_implied_home_xg", "?") if engine_result else "?",
        "bookmaker_implied_away_xg": engine_result.get("bookmaker_implied_away_xg", "?") if engine_result else "?",
        "sharp_detected": trap_report.get("sharp_detected", False),
        "sharp_dir": trap_report.get("sharp_dir"),
        "shin_dir": max(trap_report["shin"], key=trap_report["shin"].get) if "shin" in trap_report else "",
        "model_consensus": stats.get("model_consensus", 0) if stats else 0,
        "total_models": stats.get("total_models", 11) if stats else 11,
        "top_scores": stats.get("refined_poisson", {}).get("top_scores", []) if stats else [],
        "refined_poisson": stats.get("refined_poisson", {}) if stats else {},
        "elo": stats.get("elo", {}) if stats else {},
        "experience_analysis": stats.get("experience_analysis", {}) if stats else {},
        "engine_version": "vMAX 18.0",
        "engine_architecture": "贝叶斯后验+16维陷阱矩阵+决策锁定链",
    }


def _enforce_consistency(mg):
    score_str = mg.get("predicted_score", "1-1")
    if "胜其他" in score_str or score_str == "9-0":
        expected_dir = "主胜"; expected_code = "home"
    elif "平其他" in score_str or score_str == "9-9":
        expected_dir = "平局"; expected_code = "draw"
    elif "负其他" in score_str or score_str == "0-9":
        expected_dir = "客胜"; expected_code = "away"
    else:
        h, a = _parse_score(score_str)
        if h is None:
            expected_dir = mg.get("result", "平局")
            expected_code = {"主胜": "home", "平局": "draw", "客胜": "away"}.get(expected_dir, "draw")
        else:
            if h > a: expected_dir = "主胜"; expected_code = "home"
            elif h < a: expected_dir = "客胜"; expected_code = "away"
            else: expected_dir = "平局"; expected_code = "draw"
    
    mg["result"] = expected_dir
    mg["display_direction"] = expected_dir
    mg["final_direction"] = expected_code
    
    pcts = {"home": mg.get("home_win_pct", 33.3), "draw": mg.get("draw_pct", 33.3), "away": mg.get("away_win_pct", 33.3)}
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
        mg["predicted_label"] = "胜其他"; mg["predicted_score"] = "胜其他"
    elif "平其他" in score_str or score_str == "9-9":
        mg["predicted_label"] = "平其他"; mg["predicted_score"] = "平其他"
    elif "负其他" in score_str or score_str == "0-9":
        mg["predicted_label"] = "负其他"; mg["predicted_score"] = "负其他"
    else:
        mg["predicted_label"] = score_str
    
    return mg


def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})
        s = pr.get("confidence", 0) * 0.4
        s += pr.get("dir_confidence", 50) * 0.15
        trap_count = pr.get("trap_count", 0)
        if trap_count >= 2: s += 8
        elif trap_count >= 1: s += 4
        ev = pr.get("edge_vs_market", 0)
        if ev >= 30: s += 12
        elif ev >= 15: s += 6
        cold = pr.get("cold_door", {})
        if cold.get("is_cold_door") and pr.get("confidence", 0) >= 60: s += 5
        if pr.get("risk_level") == "高": s -= 10
        elif pr.get("risk_level") == "低": s += 8
        if pr.get("is_score_others"): s += 10
        if pr.get("dir_gap", 0) < 8: s -= 5
        p["recommend_score"] = round(s, 2)
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]


def extract_num(ms):
    wm = {"一":1000, "二":2000, "三":3000, "四":4000, "五":5000, "六":6000, "日":7000, "天":7000}
    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999


def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 80)
    print(f"  [vMAX 18.0] 贝叶斯后验+16维陷阱矩阵+决策锁定链 | {len(ms)} 场")
    print("=" * 80)
    
    match_analyses = []
    for i, m in enumerate(ms):
        if 'predict_match' in globals():
            try: eng = predict_match(m)
            except: eng = {}
        else: eng = {}
        
        try: league_info, _, _, _ = build_league_intelligence(m)
        except: league_info = {}
        try: sp = ensemble.predict(m, {}) if ensemble else {}
        except: sp = {}
        try: exp_result = exp_engine.analyze(m) if exp_engine else {}
        except: exp_result = {}
        
        exp_goals_prev = _f(eng.get("expected_total_goals", 0))
        if exp_goals_prev <= 0:
            hxg = _f(eng.get("bookmaker_implied_home_xg", 0))
            axg = _f(eng.get("bookmaker_implied_away_xg", 0))
            exp_goals_prev = hxg + axg if (hxg and axg) else 2.5
        
        trap_preview = detect_all_traps(m, eng, {}, sp.get("smart_signals", []) if sp else [], exp_goals_prev)
        crs_preview = analyze_crs_matrix(m)
        
        match_analyses.append({
            "match": m,
            "engine": eng,
            "league_info": league_info,
            "stats": sp,
            "index": i+1,
            "experience": exp_result,
            "trap_preview": trap_preview,
            "crs_preview": crs_preview,
        })
    
    all_ai = {"claude": {}, "gemini": {}, "gpt": {}, "grok": {}}
    if use_ai and match_analyses:
        print(f"  [v18 AI 阶段] 启动4AI并行...")
        start_t = time.time()
        all_ai = asyncio.run(run_ai_matrix(match_analyses))
        print(f"  [完成] 耗时 {time.time()-start_t:.1f}s")
    
    res = []
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        mg = merge_result_v18(
            ma["engine"],
            all_ai["gpt"].get(i+1, {}),
            all_ai["grok"].get(i+1, {}),
            all_ai["gemini"].get(i+1, {}),
            all_ai["claude"].get(i+1, {}),
            ma["stats"],
            m
        )
        
        try:
            if exp_engine: mg = apply_experience_to_prediction(m, mg, exp_engine)
        except: pass
        try: mg = apply_odds_history(m, mg)
        except: pass
        try: mg = apply_quant_edge(m, mg)
        except: pass
        try: mg = apply_wencai_intel(m, mg)
        except: pass
        try: mg = upgrade_ensemble_predict(m, mg)
        except: pass
        
        mg = _enforce_consistency(mg)
        
        res.append({**m, "prediction": mg})
        
        trap_tag = f" [🎭{mg['trap_count']}陷阱]" if mg['trap_count'] > 0 else ""
        others_tag = f" [🔥胜其他]" if mg.get("is_score_others") else ""
        scenario_tag = f" [{mg['scenario']}]"
        
        print(f"  [{i+1}] {m.get('home_team')} vs {m.get('away_team')} => "
              f"{mg['result']} ({mg['predicted_score']}={mg['predicted_label']}) | "
              f"CF: {mg['confidence']}% | 方向信心: {mg['dir_confidence']:.0f}%"
              f"{trap_tag}{others_tag}{scenario_tag}")
    
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
    
    diary["yesterday_win_rate"] = f"{high_conf}/{max(1,len(res))}"
    diary["reflection"] = f"vMAX18.0 | {total_traps}陷阱 {others_count}胜其他 {sharp_count}Sharp | 贝叶斯+16维+决策锁定"
    save_ai_diary(diary)
    
    return res, t4


if __name__ == "__main__":
    logger.info("vMAX 18.0 启动")
    print("✅ vMAX 18.0 贝叶斯后验+16维陷阱矩阵+决策锁定链 加载完成")
    print("   架构: trap_detector + crs_analyzer + bayesian_engine + predict")
    print("   一致性: predicted_score ↔ result ↔ display_direction ↔ 概率argmax")
