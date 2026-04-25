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

# ====================================================================
# 🛡️ vMAX 18.3 — 固定模型 · 不拆批 · 不切换 · 强解析 · CRS+AI+信号融合
#
# 核心约束:
#   ✅ 不拆批: 每个比赛日只构建一个完整 Prompt，一次性发给每个 AI
#   ✅ 不切换模型: 每个 AI 只使用一个固定模型，不进行模型轮换
#   ✅ 默认不换 URL: 默认每个 AI 只请求一次，避免重复成本
#   ✅ 强解析: 恢复 v17 成功版本的多字段、多格式、多兜底 JSON 解析
#   ✅ 模型可由环境变量覆盖: GPT_MODEL / GROK_MODEL / GEMINI_MODEL / CLAUDE_MODEL
#   ✅ Koudai 情报源已彻底移除: 本文件不引用、不导入、不调用 Koudai
#
# 默认模型为旧版已验证可四 AI 出数的模型名:
#   GROK   = 熊猫-A-5-grok-4.2-fast-200w上下文
#   GPT    = gpt-5.4
#   GEMINI = 熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking
#   CLAUDE = 熊猫-69-满血openrouter-claude-opus-4.7-上下文1000k
# ====================================================================

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    print("  [WARN] ⚠️ 未检测到 structlog 库，自动降级为标准 logging 模块")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)


# ====================================================================
# 基础模块导入 + 安全降级
# ====================================================================

try:
    from config import *
except ImportError as e:
    logger.warning(f"config 导入异常: {e}")

try:
    from models import EnsemblePredictor
except ImportError as e:
    logger.warning(f"models.EnsemblePredictor 导入异常: {e}")

    class EnsemblePredictor:
        def predict(self, match_obj, extra=None):
            return {
                "model_consensus": 0,
                "total_models": 0,
                "smart_signals": [],
                "refined_poisson": {},
                "elo": {},
                "random_forest": {},
                "gradient_boost": {},
                "neural_net": {},
                "logistic": {},
                "svm": {},
                "knn": {},
                "dixon_coles": {},
                "bradley_terry": {},
                "home_form": {},
                "away_form": {},
                "handicap_signal": "",
                "odds_movement": {},
                "vote_analysis": {},
                "h2h_blood": {},
                "crs_analysis": {},
                "ttg_analysis": {},
                "halftime": {},
                "pace_rating": "",
                "kelly_home": {},
                "kelly_away": {},
                "odds": {},
                "experience_analysis": {},
                "pro_odds": {},
                "asian_handicap_probs": {},
            }

try:
    from odds_engine import predict_match
except ImportError as e:
    logger.warning(f"odds_engine.predict_match 导入异常: {e}")

    def predict_match(match_obj):
        return {
            "confidence": 50,
            "home_prob": 33.3,
            "draw_prob": 33.3,
            "away_prob": 33.3,
            "bookmaker_implied_home_xg": 1.25,
            "bookmaker_implied_away_xg": 1.10,
            "over_25": 50,
            "btts": 45,
        }

try:
    from league_intel import build_league_intelligence
except ImportError as e:
    logger.warning(f"league_intel.build_league_intelligence 导入异常: {e}")

    def build_league_intelligence(match_obj):
        return {}, None, None, None

try:
    from experience_rules import ExperienceEngine, apply_experience_to_prediction
except ImportError as e:
    logger.warning(f"experience_rules 导入异常: {e}")

    class ExperienceEngine:
        def analyze(self, match_obj):
            return {}

    def apply_experience_to_prediction(match_obj, prediction, exp_engine):
        return prediction

try:
    from advanced_models import upgrade_ensemble_predict
except ImportError as e:
    logger.warning(f"advanced_models.upgrade_ensemble_predict 导入异常: {e}")

    def upgrade_ensemble_predict(match_obj, prediction):
        return prediction

try:
    from odds_history import apply_odds_history
except Exception as e:
    logger.warning("⚠️ odds_history 加载失败，自动降级", exc_info=True)

    def apply_odds_history(match_obj, prediction):
        return prediction

try:
    from quant_edge import apply_quant_edge
except Exception as e:
    logger.warning("⚠️ quant_edge 加载失败，自动降级", exc_info=True)

    def apply_quant_edge(match_obj, prediction):
        return prediction

try:
    from wencai_intel import apply_wencai_intel
except Exception:
    def apply_wencai_intel(match_obj, prediction):
        return prediction


try:
    ensemble = EnsemblePredictor()
except Exception as e:
    logger.warning(f"EnsemblePredictor 初始化失败，使用降级模型: {e}")
    ensemble = EnsemblePredictor()

try:
    exp_engine = ExperienceEngine()
except Exception as e:
    logger.warning(f"ExperienceEngine 初始化失败，使用降级规则: {e}")
    exp_engine = ExperienceEngine()


# ====================================================================
# 常量
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

SCORE_OTHERS_HOME = [
    "4-3", "5-3", "5-4", "6-0", "6-1", "6-2", "6-3", "6-4",
    "7-0", "7-1", "7-2", "胜其他", "9-0"
]
SCORE_OTHERS_DRAW = ["4-4", "5-5", "6-6", "平其他", "9-9"]
SCORE_OTHERS_AWAY = [
    "3-4", "0-6", "1-6", "2-6", "3-6",
    "0-7", "1-7", "2-7", "负其他", "0-9"
]
ALL_SCORE_OTHERS = SCORE_OTHERS_HOME + SCORE_OTHERS_DRAW + SCORE_OTHERS_AWAY

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

DEFAULT_FIXED_MODELS = {
    "grok": "熊猫-A-5-grok-4.2-fast-200w上下文",
    "gpt": "gpt-5.5",
    "gemini": "熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking",
    "claude": "熊猫-69-满血openrouter-claude-opus-4.7-上下文1000k",
}

GPT_DEFAULT_URL = "https://poloai.top/v1"

FALLBACK_URLS = [
    "https://www.api522.pro/v1",
    "https://api522.pro/v1",
    "https://api521.pro/v1",
    "http://69.63.213.33:666/v1",
]

# 默认关闭备用 URL，确保每个 AI 默认只请求一次。
# 如需同模型备用通道，设置环境变量 ENABLE_SAME_MODEL_URL_FALLBACK=true。
ENABLE_SAME_MODEL_URL_FALLBACK = str(
    os.environ.get("ENABLE_SAME_MODEL_URL_FALLBACK", "false")
).strip().lower() in ("1", "true", "yes", "y", "on")


# ====================================================================
# 基础工具函数
# ====================================================================

def get_clean_env_url(name, default=""):
    raw = os.environ.get(name, globals().get(name, default))
    v = str(raw or "").strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/#?=&%-]+)", v)
    return match.group(1) if match else v


def get_clean_env_key(name):
    raw = os.environ.get(name, globals().get(name, ""))
    return str(raw or "").strip(" \t\n\r\"'")


def get_fixed_model(ai_name, model_env_name):
    env_model = str(os.environ.get(model_env_name, "") or "").strip(" \t\n\r\"'")
    if env_model:
        return env_model

    cfg_model = str(globals().get(model_env_name, "") or "").strip(" \t\n\r\"'")
    if cfg_model:
        return cfg_model

    return DEFAULT_FIXED_MODELS.get(ai_name, "")


def calculate_value_bet(prob_pct, odds):
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
            "is_value": False,
        }

    kelly = ((b * prob) - q) / b
    return {
        "ev": round(ev * 100, 2),
        "kelly": round(max(0.0, kelly * 0.5) * 100, 2),
        "is_value": ev > 0.05,
    }


def normalize_score(score):
    s = str(score or "").strip()
    s = s.replace(" ", "")
    s = s.replace("：", "-").replace(":", "-")
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("—", "-")
    s = s.replace("其它", "其他")

    if "主胜其他" in s or "胜其他" in s:
        return "胜其他"
    if "平局其他" in s or "平其他" in s:
        return "平其他"
    if "客胜其他" in s or "负其他" in s:
        return "负其他"

    m = re.search(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        return f"{int(m.group(1))}-{int(m.group(2))}"

    return s


def parse_score(s):
    try:
        s_str = normalize_score(s)
        if "胜" in s_str and "其他" in s_str:
            return 9, 0
        if "平" in s_str and "其他" in s_str:
            return 9, 9
        if "负" in s_str and "其他" in s_str:
            return 0, 9
        p = s_str.split("-")
        if len(p) != 2:
            return None, None
        return int(p[0]), int(p[1])
    except Exception:
        return None, None


def score_to_direction(score_str):
    h, a = parse_score(score_str)
    if h is None:
        return None
    if h > a:
        return "home"
    if h < a:
        return "away"
    return "draw"


def score_to_result_cn(score_str):
    d = score_to_direction(score_str)
    if d == "home":
        return "主胜"
    if d == "away":
        return "客胜"
    if d == "draw":
        return "平局"
    return None


def safe_float(v, default=0.0):
    try:
        if v is None:
            return default
        return float(str(v).replace("+", "").strip())
    except Exception:
        return default


def dump_debug_payload(ai_name, tag, payload):
    try:
        os.makedirs("data/debug", exist_ok=True)
        dump_file = f"data/debug/{ai_name}_{tag}_{int(time.time())}.json"
        with open(dump_file, "w", encoding="utf-8") as df:
            json.dump(payload, df, ensure_ascii=False, indent=2, default=str)
        print(f"    📁 调试响应已保存: {dump_file}")
    except Exception:
        pass


# ====================================================================
# CRS赔率直接反推概率
# ====================================================================

def crs_implied_probabilities(match_obj):
    raw_odds = {}

    for score, key in CRS_FULL_MAP.items():
        try:
            odds = float(match_obj.get(key, 0) or 0)
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
            odds = float(match_obj.get(key, 0) or 0)
            if odds > 1.1:
                extras[key] = {"odds": odds, "scores": scores_set}
        except Exception:
            pass

    if len(raw_odds) < 8:
        return {}, 0.0, 0.0

    raw_sum = sum(1 / o for o in raw_odds.values())
    for extra_data in extras.values():
        raw_sum += 1 / extra_data["odds"]

    if raw_sum <= 0:
        return {}, 0.0, 0.0

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
                probs[sc] = probs.get(sc, 0.0) + per_score

    coverage = len(raw_odds) / len(CRS_FULL_MAP)
    return probs, round(margin, 3), round(coverage, 2)


# ====================================================================
# 进球数赔率信号检测
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
        except Exception:
            pass
    return signals


# ====================================================================
# 胜其他识别器
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
    except Exception:
        pass

    try:
        a6 = float(match_obj.get("a6", 999) or 999)
        if 0 < a6 <= 13.0:
            score += 2
            triggers.append(f"6球极低{a6:.1f}≤13")
            is_extreme_blowout = True
        elif 0 < a6 <= 16.0:
            score += 1
            triggers.append(f"6球{a6:.1f}≤16")
    except Exception:
        pass

    try:
        a5 = float(match_obj.get("a5", 999) or 999)
        if 0 < a5 <= 8.0:
            score += 2
            triggers.append(f"5球极低{a5:.1f}≤8")
            is_extreme_blowout = True
        elif 0 < a5 <= 10.0:
            score += 1
            triggers.append(f"5球{a5:.1f}≤10")
    except Exception:
        pass

    if exp_goals >= 3.2:
        score += 1
        triggers.append(f"λ={exp_goals:.2f}≥3.2")

    try:
        info_text = ""
        if isinstance(match_obj.get("points"), dict):
            info_text = (
                str(match_obj["points"].get("home_strength", ""))
                + str(match_obj["points"].get("guest_strength", ""))
            )
        h_match = re.search(r"场均进球[^0-9]*(\d+\.\d+)", info_text)
        a_match = re.search(r"场均失球[^0-9]*(\d+\.\d+)", info_text)
        if h_match and a_match:
            h_avg = float(h_match.group(1))
            a_avg = float(a_match.group(1))
            if (h_avg + a_avg) >= 3.5:
                score += 1
                triggers.append(f"场均{h_avg + a_avg:.1f}≥3.5")
    except Exception:
        pass

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
    except Exception:
        pass

    league = str(match_obj.get("cup", match_obj.get("league", "")))
    if any(kw in league for kw in ["欧冠", "欧联", "杯", "淘汰", "决赛"]):
        score += 0.5
        triggers.append("杯赛/淘汰赛")

    ai_others_count = 0
    if ai_responses:
        for _, r in ai_responses.items():
            if isinstance(r, dict) and r.get("is_score_others"):
                ai_others_count += 1

    if ai_others_count >= 2:
        score += 1
        triggers.append(f"AI{ai_others_count}/4识别胜其他")

    direction = "home"
    try:
        crs_win = float(match_obj.get("crs_win", 999) or 999)
        crs_same = float(match_obj.get("crs_same", 999) or 999)
        crs_lose = float(match_obj.get("crs_lose", 999) or 999)
        if crs_lose < crs_win and crs_lose < crs_same:
            direction = "away"
        elif crs_same < crs_win and crs_same < crs_lose:
            direction = "draw"
    except Exception:
        pass

    return {
        "is_others": score >= 2,
        "is_extreme_blowout": is_extreme_blowout,
        "trigger_count": score,
        "direction": direction,
        "triggers": triggers,
        "ai_others_count": ai_others_count,
    }


# ====================================================================
# 冷门猎手引擎
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
            signals.append("🔥 Sharp Money确认")

        if steam.get("steam") and "反向" in str(steam.get("signal", "")):
            signals.append("❄️ 反向Steam")
            strength += 5

        vote = match.get("vote", {})
        try:
            vh = int(vote.get("win", 33))
            va = int(vote.get("lose", 33))
            max_vote = max(vh, va)
            if max_vote >= 65:
                signals.append(f"❄️ 散户极端偏向{max_vote}%")
                strength += 4
            elif max_vote >= 58:
                strength += 2
        except Exception:
            pass

        info = match.get("intelligence", match.get("information", {}))
        if isinstance(info, dict):
            home_bad = str(info.get("home_bad_news", ""))
            away_bad = str(info.get("guest_bad_news", ""))
            hp = prediction.get("home_win_pct", 50)
            ap = prediction.get("away_win_pct", 50)

            if len(home_bad) > 80 and hp > 58:
                signals.append("❄️ 主队坏消息密集+散户狂热")
                strength += 5
            if len(away_bad) > 80 and ap > 58:
                signals.append("❄️ 客队坏消息密集+散户狂热")
                strength += 5

        sp_h = float(match.get("sp_home", match.get("win", 0)) or 0)
        if sp_h > 1:
            hp2 = prediction.get("home_win_pct", 50)
            implied_h = 100 / sp_h * 0.92
            if abs(implied_h - hp2) > 15 and hp2 > 58:
                signals.append(f"❄️ 赔率vs模型背离{abs(implied_h - hp2):.0f}%")
                strength += 4

        for s in prediction.get("smart_signals", []):
            if "盘口太便宜" in str(s):
                signals.append("❄️ 盘口太便宜")
                strength += 3
                break

        line = prediction.get("line_movement_anomaly", {})
        if line.get("has_anomaly") and "造热" in str(line.get("signal", "")):
            signals.append("❄️ 赔率变动造热")
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
            "dark_verdict": f"❄️ {level}冷门，{len(signals)}条触发" if is_cold else "",
        }


# ====================================================================
# AI 日记
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
        "reflection": "持续进化中",
        "kill_history": [],
    }


def save_ai_diary(diary):
    os.makedirs("data", exist_ok=True)
    with open("data/ai_diary.json", "w", encoding="utf-8") as f:
        json.dump(diary, f, ensure_ascii=False, indent=2)


# ====================================================================
# Prompt 构建
# ====================================================================

def build_phase1_prompt(match_analyses):
    diary = load_ai_diary()

    p = (
        "你是一个敏感高智商的顶尖足球量化分析师。"
        "根据原始数据，独立分析每场比赛，结合盘口、赔率矩阵、资金流、进球数赔率、球队风格、联赛风格，"
        "识别庄家真实意图，给出 top3 候选比分。"
        "要求逆向思维、盘口验证、赔率验证、CRS验证、进球数验证同时存在。\n\n"
    )

    if diary.get("reflection"):
        p += f"【进化】胜率:{diary.get('yesterday_win_rate', 'N/A')} | {diary['reflection']}\n\n"

    p += "【关键指导原则】\n"
    p += "你必须同时参考: Sharp资金方向 + 散户反指 + 进球数压低信号 + CRS赔率矩阵。\n"
    p += "不要因为保守就固定选择1-0/1-1/2-0；真实比赛的进球分布需要结合联赛风格、球队风格、盘口信号判断。\n\n"

    p += "【进球数赔率解码】\n"
    p += "体彩进球数标准赔率基准:\n"
    p += "  0球~11.5倍 1球~5.5倍 2球~3.4倍 3球~3.75倍 4球~6倍 5球~8倍 6球~13倍 7+球~22倍\n"
    p += "如果实际赔率 < 标准，说明庄家防范该进球数。\n"
    p += "  - 4球开<5.5倍 → 庄家预期4球概率高\n"
    p += "  - 5球开<8倍 → 庄家预期猛攻或大比分\n"
    p += "  - 7+球开<18倍 → 互射局或惨案信号，必须考虑大球比分或胜其他/负其他\n\n"

    p += "【Sharp资金与散户反指】\n"
    p += "- Sharp信号显示主胜/客胜/平 → 优先相信该方向。\n"
    p += "- 散户>58%押某方向 → 反指，该方向小比分降权。\n"
    p += "- Sharp走主 + 散户也>58%主 → 双重确认，可选2-1/3-1/2-0等。\n"
    p += "- Sharp走客 + 散户>58%主 → 大热必死，必须优先客胜。若7+球<18倍，可考虑1-3/2-3/负其他。\n\n"

    p += "【极端大球/胜其他 强制规则】\n"
    p += "如果赔率显示 5球<7.8倍 或 6球<13倍，说明庄家极度防范惨案或大球比分。\n"
    p += "此时禁止被散户大热或反指误导去选择1-1/1-2/2-2这类低强度比分。\n"
    p += "必须顺应庄家真实进球防范，结合联赛风格、球队风格，输出大比分或胜其他/负其他。\n\n"

    p += "【体彩场次诱盘识别】\n"
    p += "你预测的是中国体彩精选场次，诱盘率高，必须警惕Shin欺骗。\n"
    p += "决策优先级:\n"
    p += "1) Sharp方向 ≠ Shin高概率方向 + 散户大热Shin方向 + Steam同Sharp方向 → 判定诱盘，按Sharp方向选比分。\n"
    p += "2) Sharp=Shin同方向 + 散户也同方向 → 方向明确，可选2+球差。\n"
    p += "3) 反向Steam，即钱进但散户没跟，是强信号。\n"
    p += "4) 散户>68%为死亡级热度，必须强反指。\n"
    p += "5) 散户60-68%为大热必死区间，显著反指。\n"
    p += "严禁看到 Shin主68% 就无脑主胜，必须交叉验证Sharp/散户/Steam。\n\n"

    p += "【胜其他识别】满足2条触发:\n"
    p += "1) 7+球赔率≤15倍  2) 5球赔率≤7倍  3) 期望λ≥3.2\n"
    p += "4) 双方场均≥3.5球  5) 胜其他赔率<平其他×0.4  6) 杯赛/淘汰赛\n"
    p += "触发后 top3 必须包含至少1个大比分或其他比分。\n\n"

    p += "【强主胜 vs 平局识别】\n"
    p += "- Shin主胜>60% + xG差>1.0 + 无冷门信号 → 考虑2-1/3-1/2-0。\n"
    p += "- Shin平局>40% + 双方xG接近 + 保级/长客/杯赛 → 优先1-1/0-0/2-2。\n"
    p += "- 客队盘口太便宜+排名悬殊 → 考虑冷门客胜或低比分客胜。\n\n"

    p += "【输出格式】只输出JSON数组，不要输出解释文字，不要Markdown。\n"
    p += "每场必须包含:\n"
    p += "  match(整数), top3([{score,prob}]), reason(不少于120字), ai_confidence(0-100), is_score_others(true/false), detected_signals([...]), final_direction(home/draw/away)\n"
    p += '示例: [{"match":1,"top3":[{"score":"2-1","prob":15},{"score":"3-1","prob":12},{"score":"2-0","prob":10}],"reason":"Sharp主胜与CRS主胜比分低赔一致，散户客热构成反指，进球数4球被压低。","ai_confidence":75,"is_score_others":false,"detected_signals":["Sharp主","4球压低"],"final_direction":"home"}]\n\n'

    p += "【原始数据】\n"

    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        eng = ma.get("engine", {})
        stats = ma.get("stats", {})

        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        league = m.get("league", m.get("cup", ""))
        hc = m.get("give_ball", "0")

        sp_h = float(m.get("sp_home", m.get("win", 0)) or 0)
        sp_d = float(m.get("sp_draw", m.get("same", 0)) or 0)
        sp_a = float(m.get("sp_away", m.get("lose", 0)) or 0)

        p += f"{'=' * 50}\n[{i + 1}] {h} vs {a} | {league}\n"
        p += f"欧赔: {sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f} | 让球: {hc}\n"

        if sp_h > 1 and sp_d > 1 and sp_a > 1:
            margin = 1 / sp_h + 1 / sp_d + 1 / sp_a
            if margin > 0:
                p += (
                    f"Shin概率: 主{(1 / sp_h) / margin * 100:.1f}% "
                    f"平{(1 / sp_d) / margin * 100:.1f}% "
                    f"客{(1 / sp_a) / margin * 100:.1f}%\n"
                )

        if m.get("hhad_win"):
            p += f"让球胜平负: {m['hhad_win']}/{m.get('hhad_same', '')}/{m.get('hhad_lose', '')}\n"

        hxg = eng.get("bookmaker_implied_home_xg", "?")
        axg = eng.get("bookmaker_implied_away_xg", "?")
        p += f"庄家隐含xG: 主{hxg} vs 客{axg}\n"

        a_list = []
        compressed = []
        for g in range(8):
            v = m.get(f"a{g}", "")
            a_list.append(f"{g}={v}")
            try:
                actual = float(v or 0)
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

        try:
            gp = []
            for gi in range(8):
                v = float(m.get(f"a{gi}", 0) or 0)
                if v > 1:
                    gp.append((gi, 1 / v))
            if gp:
                tp = sum(p2 for _, p2 in gp)
                eg = sum(g * (p2 / tp) for g, p2 in gp)
                p += f"→ 进球数赔率反推λ={eg:.2f}\n"
        except Exception:
            pass

        crs_lines = []
        for sc, key in CRS_FULL_MAP.items():
            try:
                odds = float(m.get(key, 0) or 0)
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
            ("crs_lose", "负其他"),
        ]:
            v = m.get(k, "")
            if v:
                crs_others.append(f"{label}={v}")
        if crs_others:
            p += f"📌 {' | '.join(crs_others)}\n"

        hf_l = []
        hf_map = {
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
        for k, lb in hf_map.items():
            try:
                v = float(m.get(k, 0) or 0)
                if v > 1:
                    hf_l.append(f"{lb}={v:.2f}")
            except Exception:
                pass
        if hf_l:
            p += f"半全场: {' | '.join(hf_l)}\n"

        vote = m.get("vote", {})
        if vote:
            p += (
                f"散户: 胜{vote.get('win', '?')}% "
                f"平{vote.get('same', '?')}% "
                f"负{vote.get('lose', '?')}%"
            )
            try:
                max_v = max(
                    int(vote.get("win", 33)),
                    int(vote.get("same", 33)),
                    int(vote.get("lose", 33)),
                )
                if max_v >= 58:
                    p += f" ⚠️大热({max_v}%反指)"
            except Exception:
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
            for k, label in [
                ("home_injury", "主伤停"),
                ("guest_injury", "客伤停"),
                ("home_bad_news", "主利空"),
                ("guest_bad_news", "客利空"),
            ]:
                if info.get(k):
                    p += f"{label}: {str(info[k])[:600].replace(chr(10), ' ')}\n"

        points = m.get("points", {})
        if isinstance(points, dict):
            for k in ["home_strength", "guest_strength", "match_points"]:
                txt = str(points.get(k, ""))[:600].replace("\n", " ")
                if "场均" in txt or len(txt) > 20:
                    p += f"情报: {txt}\n"
                    break

        smart_sigs = stats.get("smart_signals", [])
        if smart_sigs:
            p += f"🔥盘口信号: {', '.join(str(s) for s in smart_sigs[:8])}\n"

        for field in ["analyse", "baseface", "intro"]:
            txt = str(m.get(field, "")).replace("\n", " ")[:200]
            if len(txt) > 10:
                p += f"分析: {txt}\n"
                break

        p += "\n"

    p += f"【输出{len(match_analyses)}场JSON数组，只输出数组】\n"
    return p


# ====================================================================
# AI 响应强解析
# ====================================================================

def strip_thinking_and_markdown(text):
    clean = str(text or "")
    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"```(?:json|JSON|python|text)?", "", clean)
    clean = clean.replace("```", "")
    return clean.strip()


def extract_raw_text_from_response(data, is_gem=False):
    raw_text = ""
    debug_msg_keys = []

    try:
        if is_gem:
            if isinstance(data, dict):
                candidates = data.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    texts = []
                    for part in parts:
                        if isinstance(part, dict) and part.get("text"):
                            texts.append(str(part.get("text", "")))
                    raw_text = "\n".join(texts).strip()
            return raw_text, debug_msg_keys

        if isinstance(data, dict) and data.get("choices"):
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                if isinstance(msg, dict):
                    debug_msg_keys = [
                        f"{k}({type(v).__name__}:{len(v) if isinstance(v, (str, list)) else '?'})"
                        for k, v in msg.items()
                    ]

                    content_val = msg.get("content", "")
                    if content_val:
                        if isinstance(content_val, str) and content_val.strip():
                            raw_text = content_val.strip()
                        elif isinstance(content_val, list):
                            parts = []
                            for item in content_val:
                                if isinstance(item, dict):
                                    if item.get("type") == "text" and item.get("text"):
                                        parts.append(str(item.get("text", "")))
                                    elif item.get("text"):
                                        parts.append(str(item.get("text", "")))
                            raw_text = "\n".join(parts).strip()

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
                        skip_keys = {
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
                        best_with_match = ""
                        for k, v in msg.items():
                            if k in skip_keys:
                                continue
                            if isinstance(v, str) and '"match"' in v and "[" in v:
                                if len(v) > len(best_with_match):
                                    best_with_match = v.strip()
                        if best_with_match:
                            raw_text = best_with_match

                    if not raw_text:
                        for k, v in msg.items():
                            if isinstance(v, str) and '"match"' in v and "[" in v:
                                raw_text = v.strip()
                                print(f"    🆘 兜底命中字段: {k}")
                                break

                    if not raw_text:
                        skip_keys2 = {
                            "reasoning_content",
                            "thinking",
                            "reasoning",
                            "reasoning_text",
                            "thoughts",
                            "thought_process",
                        }
                        longest_clean = ""
                        for k, v in msg.items():
                            if k in skip_keys2:
                                continue
                            if isinstance(v, str) and len(v.strip()) > len(longest_clean):
                                longest_clean = v.strip()
                        if longest_clean and len(longest_clean) > 20:
                            raw_text = longest_clean
                            print("    🆘 取最长非thinking字段")

        if not raw_text and isinstance(data, dict) and data.get("output") and isinstance(data["output"], list):
            chunks = []
            for out_item in data["output"]:
                if not isinstance(out_item, dict):
                    continue
                if out_item.get("type") == "message":
                    for ct in out_item.get("content", []):
                        if isinstance(ct, dict):
                            if ct.get("text"):
                                chunks.append(str(ct["text"]))
                            elif ct.get("type") == "output_text" and ct.get("text"):
                                chunks.append(str(ct["text"]))
            if chunks:
                raw_text = "\n".join(chunks).strip()

        if not raw_text and isinstance(data, dict):
            full_str = json.dumps(data, ensure_ascii=False)
            m_match = re.search(r'\[\s*\{\s*\\?"match\\?"', full_str)
            if m_match:
                raw_text = full_str[m_match.start():]
                print("    🆘 从response dump中尝试提取JSON")

    except Exception as ex:
        print(f"    ⚠️ 响应文本提取异常: {str(ex)[:120]}")

    return raw_text, debug_msg_keys


def find_balanced_json_arrays(text):
    clean = strip_thinking_and_markdown(text)
    candidates = []
    starts = []

    for pat in [
        r'\[\s*\{\s*"match"',
        r"\[\s*\{\s*'match'",
        r'\[\s*\{',
    ]:
        for m in re.finditer(pat, clean):
            starts.append(m.start())

    starts = sorted(set(starts))

    for start in starts:
        depth = 0
        in_str = False
        quote = ""
        esc = False
        end = None

        for i in range(start, len(clean)):
            ch = clean[i]

            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == quote:
                    in_str = False
                    quote = ""
                continue

            if ch in ("'", '"'):
                in_str = True
                quote = ch
                continue

            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end:
            candidates.append(clean[start:end])
        else:
            candidates.append(clean[start:])

    if not candidates:
        start = clean.find("[")
        end = clean.rfind("]")
        if start != -1 and end > start:
            candidates.append(clean[start:end + 1])

    return candidates


def extract_balanced_json_objects(text):
    objects = []
    starts = [m.start() for m in re.finditer(r'\{\s*["\']?match["\']?', text)]
    for start in starts:
        depth = 0
        in_str = False
        quote = ""
        esc = False
        end = None

        for i in range(start, len(text)):
            ch = text[i]

            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == quote:
                    in_str = False
                    quote = ""
                continue

            if ch in ("'", '"'):
                in_str = True
                quote = ch
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end:
            objects.append(text[start:end])

    return objects


def repair_json_string(s):
    repaired = str(s or "").strip()
    repaired = repaired.replace("\ufeff", "")
    repaired = repaired.replace("“", '"').replace("”", '"')
    repaired = repaired.replace("‘", "'").replace("’", "'")
    repaired = re.sub(r",\s*([\]}])", r"\1", repaired)
    repaired = repaired.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")
    repaired = repaired.replace(":True", ":true").replace(":False", ":false").replace(":None", ":null")
    return repaired


def try_load_json_any(s):
    repaired = repair_json_string(s)

    try:
        return json.loads(repaired)
    except Exception:
        pass

    if '\\"' in repaired:
        try:
            decoded = json.loads(f'"{repaired}"')
            return json.loads(decoded)
        except Exception:
            pass

    try:
        last_brace = repaired.rfind("}")
        if last_brace != -1:
            cut = repaired[:last_brace + 1]
            if cut.strip().startswith("["):
                return json.loads(cut + "]")
    except Exception:
        pass

    return None


def robust_parse_ai_json(raw_text, ai_name, num_matches):
    clean = strip_thinking_and_markdown(raw_text)
    arrays = find_balanced_json_arrays(clean)

    for arr_text in arrays:
        parsed = try_load_json_any(arr_text)
        if isinstance(parsed, list):
            return parsed, arr_text

        objs = extract_balanced_json_objects(arr_text)
        parsed_objs = []
        for obj_text in objs:
            obj = try_load_json_any(obj_text)
            if isinstance(obj, dict):
                parsed_objs.append(obj)

        if parsed_objs:
            return parsed_objs, arr_text

    parsed = try_load_json_any(clean)
    if isinstance(parsed, list):
        return parsed, clean

    if isinstance(parsed, dict):
        for key in ["data", "result", "results", "matches", "output"]:
            val = parsed.get(key)
            if isinstance(val, list):
                return val, clean

    objs = extract_balanced_json_objects(clean)
    parsed_objs = []
    for obj_text in objs:
        obj = try_load_json_any(obj_text)
        if isinstance(obj, dict):
            parsed_objs.append(obj)

    if parsed_objs:
        return parsed_objs, clean

    return [], ""


def normalize_ai_item(item):
    if not isinstance(item, dict):
        return None

    try:
        mid = int(item.get("match"))
    except Exception:
        return None

    top3_raw = item.get("top3", [])
    top3 = []

    if isinstance(top3_raw, list):
        for t in top3_raw[:3]:
            if not isinstance(t, dict):
                continue
            score = normalize_score(t.get("score", ""))
            h, a = parse_score(score)
            if h is None:
                continue
            prob = t.get("prob", t.get("probability", 0))
            try:
                prob = float(prob)
            except Exception:
                prob = 0
            top3.append({"score": score, "prob": prob})

    score = normalize_score(item.get("score", item.get("ai_score", "")))
    if not score and top3:
        score = top3[0].get("score", "")

    h, a = parse_score(score)
    if h is None and top3:
        score = top3[0].get("score", "")
        h, a = parse_score(score)

    if h is None:
        return None

    if not top3:
        top3 = [{"score": score, "prob": 10}]

    detected_signals = item.get("detected_signals", item.get("detected_traps", []))
    if isinstance(detected_signals, str):
        detected_signals = [detected_signals]
    if not isinstance(detected_signals, list):
        detected_signals = []

    final_direction = item.get("final_direction", score_to_direction(score))
    if final_direction not in ("home", "draw", "away"):
        final_direction = score_to_direction(score)

    try:
        ai_confidence = int(float(item.get("ai_confidence", item.get("confidence", 60))))
    except Exception:
        ai_confidence = 60

    ai_confidence = max(0, min(100, ai_confidence))

    return mid, {
        "top3": top3,
        "ai_score": score,
        "reason": str(item.get("reason", item.get("analysis", "")))[:1200],
        "ai_confidence": ai_confidence,
        "is_score_others": bool(item.get("is_score_others", score in ALL_SCORE_OTHERS or "其他" in score)),
        "detected_signals": detected_signals,
        "final_direction": final_direction,
    }


def parse_ai_results(raw_text, ai_name, num_matches):
    arr, json_str = robust_parse_ai_json(raw_text, ai_name, num_matches)
    results = {}

    if not isinstance(arr, list):
        return results, json_str

    for item in arr:
        normalized = normalize_ai_item(item)
        if not normalized:
            continue
        mid, payload = normalized
        results[mid] = payload

    return results, json_str


# ====================================================================
# AI 调用引擎: 不拆批、不切模型、默认不换 URL
# ====================================================================

def get_ai_profile(ai_name):
    AI_PROFILES = {
        "claude": {
            "sys": (
                "你是足球量化分析师。必须按以下顺序分析:\n"
                "1) Sharp资金与Shin概率确认真实方向。\n"
                "2) 进球数a0-a7找压低信号。\n"
                "3) 散户投票>58%视为反指。\n"
                "4) CRS赔率矩阵交叉验证。\n"
                "5) 方向+进球数+反指确定top3比分。\n"
                "只输出JSON数组，不要输出Markdown。"
            ),
            "temp": 0.18,
        },
        "grok": {
            "sys": (
                "你是Grok足球盘口分析引擎。必须结合实时盘口、伤停、Sharp资金、散户反指、进球数赔率与CRS矩阵。\n"
                "重点识别体彩诱盘、大热必死、反向Steam、极端大球与胜其他。\n"
                "只输出JSON数组，不要输出Markdown。"
            ),
            "temp": 0.22,
        },
        "gpt": {
            "sys": (
                "你是量化足球分析师。必须使用CRS赔率矩阵、进球数赔率、Sharp资金、散户投票、欧赔Shin概率、盘口变动综合判断。\n"
                "输出每场top3比分、信心、方向与信号。只输出JSON数组。"
            ),
            "temp": 0.18,
        },
        "gemini": {
            "sys": (
                "你是模式敏感识别引擎。必须识别Shin与散户偏差、赔率变动、进球数隐含分布、CRS概率偏差。\n"
                "输出top3比分。只输出JSON数组，不要额外解释。"
            ),
            "temp": 0.15,
        },
    }

    return AI_PROFILES.get(ai_name, AI_PROFILES["gpt"])


async def async_call_one_ai_batch(session, prompt, url_env, key_env, model_env, num_matches, ai_name):
    key = get_clean_env_key(key_env)
    if not key:
        print(f"  [跳过] {ai_name.upper()} | 未配置 {key_env}")
        return ai_name, {}, "no_key"

    model_name = get_fixed_model(ai_name, model_env)
    if not model_name:
        print(f"  [跳过] {ai_name.upper()} | 未配置固定模型 {model_env}")
        return ai_name, {}, "no_model"

    if ai_name == "gpt":
        primary_url = get_clean_env_url(url_env, GPT_DEFAULT_URL)
        if not primary_url:
            primary_url = GPT_DEFAULT_URL
        urls = [primary_url]
        print(f"    🔌 [GPT] 使用固定通道: {primary_url}")
    else:
        primary_url = get_clean_env_url(url_env)
        if not primary_url:
            print(f"  [跳过] {ai_name.upper()} | 未配置 {url_env}")
            return ai_name, {}, "no_url"

        urls = [primary_url]

        if ENABLE_SAME_MODEL_URL_FALLBACK:
            for u in FALLBACK_URLS:
                if u and u != primary_url and u not in urls:
                    urls.append(u)
                    break

    CONNECT_TIMEOUT = 20
    READ_TIMEOUT_MAP = {
        "claude": 350,
        "grok": 300,
        "gpt": 300,
        "gemini": 300,
    }
    READ_TIMEOUT = READ_TIMEOUT_MAP.get(ai_name, 260)

    profile = get_ai_profile(ai_name)

    for url_idx, base_url in enumerate(urls):
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
                "systemInstruction": {"parts": [{"text": profile["sys"]}]},
            }
        else:
            headers["Authorization"] = f"Bearer {key}"
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": profile["sys"]},
                    {"role": "user", "content": prompt},
                ],
            }
            if ai_name != "claude":
                payload["temperature"] = profile["temp"]

        gw = url.split("/v1")[0][:45]
        if url_idx == 0:
            print(f"  [🔌连接中] {ai_name.upper()} | 固定模型={model_name} @ {gw}")
        else:
            print(f"  [🔌连接中] {ai_name.upper()} | 同模型备用通道={model_name} @ {gw}")

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

                if r.status != 200:
                    print(f"    ⚠️ HTTP {r.status} | {elapsed_connect}s | 不切模型")
                    try:
                        err_text = await r.text()
                        if err_text:
                            dump_debug_payload(ai_name, f"http_{r.status}", {"text": err_text[:3000]})
                    except Exception:
                        pass

                    if ENABLE_SAME_MODEL_URL_FALLBACK and url_idx + 1 < len(urls):
                        print("    ↪ 同模型换备用URL，不换模型")
                        continue

                    return ai_name, {}, f"http_{r.status}"

                print(f"    ✅ 已连上! {elapsed_connect}s | 等待数据...")

                try:
                    data = await r.json(content_type=None)
                except Exception:
                    text_data = await r.text()
                    data = {"raw_text": text_data}

                elapsed = round(time.time() - t0, 1)

                usage = {}
                if isinstance(data, dict):
                    usage = data.get("usage", {}) or {}
                req_tokens = usage.get("total_tokens", 0) or (
                    usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                )

                if not req_tokens and isinstance(data, dict):
                    um = data.get("usageMetadata", {})
                    req_tokens = um.get("totalTokenCount", 0)

                if req_tokens:
                    print(f"    📊 {req_tokens:,} token | {elapsed}s")

                if isinstance(data, dict) and data.get("raw_text"):
                    raw_text = data.get("raw_text", "")
                    debug_msg_keys = []
                else:
                    raw_text, debug_msg_keys = extract_raw_text_from_response(data, is_gem=is_gem)

                if not raw_text or len(raw_text.strip()) < 10:
                    print("    ⚠️ 空数据 | 不切模型，不重复请求")
                    if debug_msg_keys:
                        print(f"    🔍 msg字段: {', '.join(debug_msg_keys[:10])}")
                    if isinstance(data, dict):
                        top_keys = [f"{k}({type(v).__name__})" for k, v in data.items()]
                        print(f"    🔍 data字段: {', '.join(top_keys[:8])}")
                    dump_debug_payload(ai_name, "empty", data)
                    return ai_name, {}, "empty"

                results, json_str = parse_ai_results(raw_text, ai_name, num_matches)

                if results:
                    print(f"    🎯 JSON解析成功: {len(results)}/{num_matches}")
                    print(f"    ✅ {ai_name.upper()} 完成: {len(results)}/{num_matches} | {elapsed}s")
                    return ai_name, results, model_name

                print(f"    ⚠️ JSON解析失败。响应片段: {repr(raw_text[:1200])}")
                print("    ⚠️ 解析0条 → 不切换模型，不重复请求")
                dump_debug_payload(ai_name, "parse0", {
                    "raw_text": raw_text[:12000],
                    "json_candidate": json_str[:12000] if json_str else "",
                    "response": data,
                })
                return ai_name, {}, "parse0"

        except aiohttp.ClientConnectorError as e:
            print(f"    🔌 连接失败: {str(e)[:120]} | 不切模型")
            if ENABLE_SAME_MODEL_URL_FALLBACK and url_idx + 1 < len(urls):
                print("    ↪ 同模型换备用URL，不换模型")
                continue
            return ai_name, {}, "connect_error"

        except asyncio.TimeoutError:
            print(f"    ⏰ 超时 | 不切模型，不重复请求")
            return ai_name, {}, "timeout"

        except aiohttp.ClientOSError as e:
            print(f"    ⚠️ ClientOSError: {str(e)[:160]} | 不切模型")
            if ENABLE_SAME_MODEL_URL_FALLBACK and url_idx + 1 < len(urls):
                print("    ↪ 同模型换备用URL，不换模型")
                continue
            return ai_name, {}, "client_os_error"

        except Exception as e:
            print(f"    ⚠️ {type(e).__name__}: {str(e)[:160]} | 不切模型")
            return ai_name, {}, "error"

    return ai_name, {}, "all_failed"


async def run_ai_matrix_two_phase(match_analyses):
    num = len(match_analyses)
    prompt = build_phase1_prompt(match_analyses)

    print(f"  [v18.3 AI] 启动4AI并行 | 不拆批 | 不切换模型 | 默认单次请求")
    print(f"  [v18.3 Prompt] {len(prompt):,} 字符 → 4AI并行 | 固定模型")

    ai_configs = [
        ("grok", "GROK_API_URL", "GROK_API_KEY", "GROK_MODEL"),
        ("gpt", "GPT_API_URL", "GPT_API_KEY", "GPT_MODEL"),
        ("gemini", "GEMINI_API_URL", "GEMINI_API_KEY", "GEMINI_MODEL"),
        ("claude", "CLAUDE_API_URL", "CLAUDE_API_KEY", "CLAUDE_MODEL"),
    ]

    all_results = {
        "gpt": {},
        "grok": {},
        "gemini": {},
        "claude": {},
    }

    connector = aiohttp.TCPConnector(limit=8, ttl_dns_cache=300)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            async_call_one_ai_batch(session, prompt, url_env, key_env, model_env, num, ai_name)
            for ai_name, url_env, key_env, model_env in ai_configs
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, tuple):
                ai_name, ai_data, _status = res
                all_results[ai_name] = ai_data
            else:
                print(f"  [ERROR] AI任务异常: {res}")

    ok = sum(1 for v in all_results.values() if v)
    print(f"  [完成] {ok}/4 AI有数据")
    return all_results


# ====================================================================
# Merge 核心
# ====================================================================

def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
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
        if not isinstance(r, dict):
            return False
        score = r.get("ai_score", "")
        if not score and isinstance(r.get("top3"), list) and r["top3"]:
            score = r["top3"][0].get("score", "")
        h, a = parse_score(score)
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

    all_ai = {}
    for name, r in [
        ("gpt", gpt_r),
        ("grok", grok_r),
        ("gemini", gemini_r),
        ("claude", claude_r),
    ]:
        if ai_valid[name]:
            all_ai[name] = r

    direction_scores = {"home": 0.0, "draw": 0.0, "away": 0.0}

    if sp_h > 1 and sp_d > 1 and sp_a > 1:
        margin = 1 / sp_h + 1 / sp_d + 1 / sp_a
        shin_h = (1 / sp_h) / margin * 100
        shin_d = (1 / sp_d) / margin * 100
        shin_a = (1 / sp_a) / margin * 100
    else:
        shin_h = shin_d = shin_a = 33.3

    shin_dir = max(
        [("home", shin_h), ("draw", shin_d), ("away", shin_a)],
        key=lambda x: x[1],
    )[0]

    smart_signals = stats.get("smart_signals", [])
    smart_str = " ".join(str(s) for s in smart_signals)

    sharp_detected = "Sharp" in smart_str or "sharp" in smart_str
    sharp_dir = None

    if sharp_detected:
        for s in smart_signals:
            s_str = str(s)
            if "Sharp" not in s_str and "sharp" not in s_str:
                continue

            if "确认" in s_str and "→" not in s_str and "流向" not in s_str and "走" not in s_str:
                continue

            if re.search(r"(主胜|主队|走主|→\s*主|流向\s*主|资金\s*主)", s_str):
                sharp_dir = "home"
                break
            if re.search(r"(客胜|客队|走客|→\s*客|流向\s*客|资金\s*客)", s_str):
                sharp_dir = "away"
                break
            if re.search(r"(平局|平赔|走平|→\s*平|流向\s*平|资金\s*平)", s_str):
                sharp_dir = "draw"
                break

    steam_dir = None
    steam_type = None

    if "Steam" in smart_str:
        for s in smart_signals:
            s_str = str(s)
            if "Steam" not in s_str:
                continue

            is_reverse = "反向" in s_str or "未跟" in s_str or "不跟" in s_str

            if re.search(r"(主胜\s*Steam|Steam.*主胜|主胜.*降水|主.*Steam)", s_str):
                steam_dir = "home"
                steam_type = "reverse" if is_reverse else "normal"
                break
            if re.search(r"(客胜\s*Steam|Steam.*客胜|客胜.*降水|客.*Steam)", s_str):
                steam_dir = "away"
                steam_type = "reverse" if is_reverse else "normal"
                break
            if re.search(r"(平局\s*Steam|Steam.*平局|平.*Steam)", s_str):
                steam_dir = "draw"
                steam_type = "reverse" if is_reverse else "normal"
                break

    vote = match_obj.get("vote", {})
    vote_hot_dir = None
    vote_hot_pct = 0

    try:
        vh = int(vote.get("win", 33) or 33)
        vd = int(vote.get("same", 33) or 33)
        va = int(vote.get("lose", 33) or 33)
        max_vote = max(vh, vd, va)
        if max_vote >= 55:
            vote_hot_pct = max_vote
            if vh == max_vote:
                vote_hot_dir = "home"
            elif vd == max_vote:
                vote_hot_dir = "draw"
            else:
                vote_hot_dir = "away"
    except Exception:
        vh = vd = va = 33

    change = match_obj.get("change", {})
    change_down_dir = None
    try:
        cw = safe_float(change.get("win", 0))
        cs = safe_float(change.get("same", 0))
        cl = safe_float(change.get("lose", 0))

        if cw < -0.05 and cw <= cs and cw <= cl:
            change_down_dir = "home"
        elif cl < -0.05 and cl <= cs and cl <= cw:
            change_down_dir = "away"
        elif cs < -0.05 and cs <= cw and cs <= cl:
            change_down_dir = "draw"
    except Exception:
        pass

    cold_signals_raw = [
        s for s in smart_signals
        if "❄️" in str(s) or "冷门" in str(s) or "大热" in str(s) or "造热" in str(s)
    ]

    hp_eng = engine_result.get("home_prob", shin_h)
    ap_eng = engine_result.get("away_prob", shin_a)
    hot_side = "home" if hp_eng > ap_eng else "away"

    dupan_detected = False
    dupan_true_dir = None
    dupan_confirm = 0

    if sharp_detected and sharp_dir and sharp_dir != shin_dir:
        if vote_hot_dir == shin_dir and vote_hot_pct >= 55:
            if vote_hot_pct >= 68:
                dupan_confirm += 4
            elif vote_hot_pct >= 60:
                dupan_confirm += 3
            else:
                dupan_confirm += 2

        if vote_hot_dir and vote_hot_dir != sharp_dir and vote_hot_pct >= 58:
            dupan_confirm += 2

        if steam_dir == sharp_dir:
            dupan_confirm += 3 if steam_type == "reverse" else 2

        if change_down_dir == sharp_dir:
            dupan_confirm += 2

        if cold_signals_raw:
            dupan_confirm += min(3, len(cold_signals_raw))

        if dupan_confirm >= 3:
            dupan_detected = True
            dupan_true_dir = sharp_dir
            print(
                f"    🚨 [诱盘识别] Sharp({sharp_dir}) ≠ Shin({shin_dir}) "
                f"| 证据{dupan_confirm}分 → 真实方向={sharp_dir}"
            )

    shin_weight = 15 if dupan_detected else 30
    direction_scores["home"] += shin_h / 100 * shin_weight
    direction_scores["draw"] += shin_d / 100 * shin_weight
    direction_scores["away"] += shin_a / 100 * shin_weight

    if dupan_detected:
        print("    📉 诱盘模式: Shin权重30→15")

    if sharp_detected and sharp_dir:
        sharp_base = 35 if dupan_detected else 25
        direction_scores[sharp_dir] += sharp_base
        dir_cn = {"home": "主胜", "away": "客胜", "draw": "平局"}[sharp_dir]
        print(f"    💰 Sharp→{dir_cn} +{sharp_base}")

    if steam_dir:
        if steam_type == "reverse":
            direction_scores[steam_dir] += 20
            dir_cn = {"home": "主胜", "away": "客胜", "draw": "平局"}[steam_dir]
            print(f"    🚀🚀 反向Steam→{dir_cn} +20")
        else:
            direction_scores[steam_dir] += 10
            dir_cn = {"home": "主胜", "away": "客胜", "draw": "平局"}[steam_dir]
            print(f"    🚀 Steam→{dir_cn} +10")

    contrarian_away_score = 0
    contrarian_home_score = 0

    if vote_hot_dir and vote_hot_pct >= 55:
        if vote_hot_pct >= 68:
            contra_weight = 22
            level = "死亡级"
        elif vote_hot_pct >= 60:
            contra_weight = 14
            level = "大热必死"
        else:
            contra_weight = 6
            level = "轻度"

        for d in ["home", "draw", "away"]:
            if d != vote_hot_dir:
                direction_scores[d] += contra_weight * 0.5

        direction_scores[vote_hot_dir] -= contra_weight * 0.3

        dir_cn = {"home": "主胜", "away": "客胜", "draw": "平局"}[vote_hot_dir]
        print(f"    🎭 散户热{dir_cn}{vote_hot_pct}% [{level}] → 反指 权重{contra_weight}")

        if vote_hot_dir == "home":
            contrarian_away_score = contra_weight
        elif vote_hot_dir == "away":
            contrarian_home_score = contra_weight

    if cold_signals_raw or sharp_detected or vote_hot_pct >= 60:
        cold_score = 0

        if sharp_detected and sharp_dir and sharp_dir != shin_dir:
            cold_score += 6
        if steam_type == "reverse":
            cold_score += 5
        if vote_hot_pct >= 68:
            cold_score += 7
        elif vote_hot_pct >= 60:
            cold_score += 5

        for s in smart_signals:
            s_str = str(s)
            if "盘口太便宜" in s_str:
                cold_score += 4
            if "坏消息" in s_str or "崩盘" in s_str:
                cold_score += 5
            if "背离" in s_str:
                cold_score += 4
            if "造热" in s_str:
                cold_score += 3

        if cold_score >= 25:
            cold_level = "死亡级"
            cold_power = 18
        elif cold_score >= 18:
            cold_level = "顶级"
            cold_power = 12
        elif cold_score >= 12:
            cold_level = "高危"
            cold_power = 8
        elif cold_score >= 6:
            cold_level = "中等"
            cold_power = 4
        else:
            cold_level = None
            cold_power = 0

        if cold_level:
            direction_scores[hot_side] -= cold_power
            other = "away" if hot_side == "home" else "home"
            direction_scores[other] += cold_power * 0.6
            direction_scores["draw"] += cold_power * 0.4
            print(f"    ❄️ 冷门[{cold_level}] 分数{cold_score} → 降{hot_side} -{cold_power}")

    if change and isinstance(change, dict):
        try:
            cw = safe_float(change.get("win", 0))
            cs = safe_float(change.get("same", 0))
            cl = safe_float(change.get("lose", 0))

            move_log = []
            if cw < -0.05:
                direction_scores["home"] += 4
                move_log.append("主降")
            if cs < -0.05:
                direction_scores["draw"] += 4
                move_log.append("平降")
            if cl < -0.05:
                direction_scores["away"] += 4
                move_log.append("客降")

            if cw > 0.05:
                direction_scores["home"] -= 2
            if cs > 0.05:
                direction_scores["draw"] -= 2
            if cl > 0.05:
                direction_scores["away"] -= 2

            if move_log:
                print(f"    📊 赔率变动: {' '.join(move_log)}")
        except Exception:
            pass

    ai_weight_total = 15 if dupan_detected else 25
    ai_directions = {"home": 0.0, "draw": 0.0, "away": 0.0}

    for name, r in all_ai.items():
        if not isinstance(r, dict):
            continue

        score = r.get("ai_score", "")
        d = score_to_direction(score)

        if not d and isinstance(r.get("top3"), list) and r["top3"]:
            d = score_to_direction(r["top3"][0].get("score", ""))

        if d:
            w = 1.5 if name == "claude" else (1.40 if name == "gemini" else (1.35 if name == "grok" else 1.0))
            ai_directions[d] += w

    total_ai_dir = sum(ai_directions.values())
    if total_ai_dir > 0:
        for d in ["home", "draw", "away"]:
            direction_scores[d] += (ai_directions[d] / total_ai_dir) * ai_weight_total

    total_dir = sum(max(0.1, v) for v in direction_scores.values())
    dir_probs = {
        d: max(0.1, direction_scores[d]) / total_dir * 100
        for d in direction_scores
    }

    final_direction = max(dir_probs, key=dir_probs.get)
    dir_gap = dir_probs[final_direction] - sorted(dir_probs.values(), reverse=True)[1]
    dir_confident = dir_gap > 5

    print(
        f"    🎯 方向: 主{dir_probs['home']:.0f}% "
        f"平{dir_probs['draw']:.0f}% 客{dir_probs['away']:.0f}%"
    )

    pre_pred = {
        "home_win_pct": dir_probs["home"],
        "draw_pct": dir_probs["draw"],
        "away_win_pct": dir_probs["away"],
        "steam_move": stats.get("steam_move", {}),
        "smart_signals": smart_signals,
        "line_movement_anomaly": stats.get("line_movement_anomaly", {}),
    }
    cold_door = ColdDoorDetector.detect(match_obj, pre_pred)

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
            "total_xg",
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
        try:
            hxg = float(engine_result.get("bookmaker_implied_home_xg", 0) or 0)
            axg = float(engine_result.get("bookmaker_implied_away_xg", 0) or 0)
            if hxg > 0 and axg > 0:
                exp_goals = hxg + axg
                print(f"    📐 期望进球用xG总和: {hxg:.2f}+{axg:.2f}={exp_goals:.2f}")
        except Exception:
            pass

    if exp_goals <= 0:
        try:
            gp = []
            for gi in range(8):
                v = float(match_obj.get(f"a{gi}", 0) or 0)
                if v > 1:
                    gp.append((gi, 1 / v))
            if gp:
                tp = sum(p for _, p in gp)
                exp_goals = sum(g * (p / tp) for g, p in gp)
                print(f"    📐 期望进球用a0-a7反推: {exp_goals:.2f}")
        except Exception:
            pass

    if exp_goals <= 0:
        try:
            over25 = float(engine_result.get("over_25", 50) or 50)
            exp_goals = 2.0 + (over25 - 40) * 0.015
            print(f"    📐 期望进球用over25估算: {exp_goals:.2f}")
        except Exception:
            pass

    if exp_goals < 1.0 or exp_goals > 6.0:
        print(f"    ⚠️ 期望进球异常({exp_goals:.2f}), 使用默认2.5")
        exp_goals = 2.5

    goal_signals = detect_goal_signals(match_obj)
    strongest_goal = -1
    strongest_ratio = 1.0

    if goal_signals:
        strongest_goal = max(goal_signals, key=goal_signals.get)
        strongest_ratio = goal_signals[strongest_goal]
        sig_str = ", ".join(
            f"{g}球(x{r:.1f})"
            for g, r in sorted(goal_signals.items(), key=lambda x: -x[1])[:3]
        )
        print(f"    📈 进球信号: {sig_str}")

    others_info = detect_score_others(match_obj, exp_goals, all_ai)
    if others_info["is_others"]:
        print(
            f"    🔥 胜其他({others_info['trigger_count']:.1f}条): "
            f"{' | '.join(others_info['triggers'][:3])}"
        )

    crs_probs, crs_margin, crs_coverage = crs_implied_probabilities(match_obj)

    if crs_probs:
        print(f"    📋 CRS概率: 覆盖{crs_coverage * 100:.0f}% margin{crs_margin:.3f}")
    else:
        print("    ⚠️ CRS数据不足, 使用简化backup")

    home_xg = float(engine_result.get("bookmaker_implied_home_xg", 1.3) or 1.3)
    away_xg = float(engine_result.get("bookmaker_implied_away_xg", 0.9) or 0.9)

    xg_adj_log = []

    if sharp_detected and sharp_dir == "away":
        home_xg *= 0.85
        away_xg *= 1.20
        xg_adj_log.append("Sharp客")
    elif sharp_detected and sharp_dir == "home":
        home_xg *= 1.15
        away_xg *= 0.85
        xg_adj_log.append("Sharp主")

    if cold_door["is_cold_door"] and not sharp_detected:
        if hot_side == "home":
            home_xg *= 0.75
            away_xg *= 1.25
            xg_adj_log.append("冷主")
        else:
            away_xg *= 0.75
            home_xg *= 1.25
            xg_adj_log.append("冷客")

    home_xg = max(0.3, min(4.0, home_xg))
    away_xg = max(0.2, min(3.5, away_xg))

    if xg_adj_log:
        print(f"    ⚽ xG调整: 主{home_xg:.2f}/客{away_xg:.2f} ({' | '.join(xg_adj_log)})")

    backup_probs = {}
    if not crs_probs or crs_coverage < 0.5:
        for h_g in range(6):
            for a_g in range(6):
                p_h = math.exp(-home_xg) * (home_xg ** h_g) / math.factorial(h_g)
                p_a = math.exp(-away_xg) * (away_xg ** a_g) / math.factorial(a_g)
                backup_probs[f"{h_g}-{a_g}"] = round(p_h * p_a * 100, 2)

    ai_voted = {}

    for name, r in all_ai.items():
        if not isinstance(r, dict):
            continue

        score = normalize_score(r.get("ai_score", ""))
        h, a = parse_score(score)

        if h is None and isinstance(r.get("top3"), list) and r["top3"]:
            score = normalize_score(r["top3"][0].get("score", ""))
            h, a = parse_score(score)

        if h is not None:
            w = 1.5 if name == "claude" else (1.40 if name == "gemini" else (1.35 if name == "grok" else 1.0))
            ai_voted[score] = ai_voted.get(score, 0.0) + w

        t3 = r.get("top3", [])
        if isinstance(t3, list):
            for rank, t in enumerate(t3[1:3], 2):
                score2 = normalize_score(t.get("score", ""))
                h2, a2 = parse_score(score2)
                if h2 is not None:
                    w2 = 0.4 if rank == 2 else 0.2
                    ai_voted[score2] = ai_voted.get(score2, 0.0) + w2

    if ai_valid["claude"] and claude_r.get("ai_confidence", 0) >= 75:
        cl_score = normalize_score(claude_r.get("ai_score", ""))
        cl_dir = score_to_direction(cl_score)

        if cl_score and cl_dir:
            other_ai_dirs = {}
            valid_others = 0
            other_confidences = []

            for name in ["gpt", "grok", "gemini"]:
                if not ai_valid[name]:
                    continue

                valid_others += 1
                r = all_ai.get(name, {})

                if isinstance(r, dict):
                    d = score_to_direction(r.get("ai_score", ""))
                    if d:
                        other_ai_dirs[d] = other_ai_dirs.get(d, 0) + 1
                        other_confidences.append(r.get("ai_confidence", 60))

            if other_ai_dirs and valid_others >= 2:
                majority_dir, majority_count = max(other_ai_dirs.items(), key=lambda x: x[1])
                is_hard_majority = majority_count >= 3
                avg_other_conf = sum(other_confidences) / len(other_confidences) if other_confidences else 60
                claude_conf = claude_r.get("ai_confidence", 60)

                direction_independent = cl_dir != majority_dir
                confidence_dominant = claude_conf > avg_other_conf

                if direction_independent and is_hard_majority and confidence_dominant:
                    if cl_score in ai_voted:
                        ai_voted[cl_score] *= 2.0
                        print(
                            f"    👑 Claude独立裁决({cl_dir}信心{claude_conf}) "
                            f"vs 多数{majority_dir}({majority_count}/{valid_others}) → 权重×2"
                        )
                elif direction_independent and is_hard_majority and not confidence_dominant:
                    if cl_score in ai_voted:
                        ai_voted[cl_score] *= 1.3
                        print("    ✋ Claude方向独立但信心非主导 → 权重×1.3")
                elif not direction_independent:
                    if cl_score in ai_voted:
                        ai_voted[cl_score] *= 1.3
                        print(f"    🤝 Claude({cl_dir})与多数同向({majority_dir}) → 权重×1.3")

    ai_consensus_strength = 0
    if ai_voted:
        max_vote = max(ai_voted.values())
        total_vote = sum(ai_voted.values())
        ai_consensus_strength = max_vote / total_vote if total_vote > 0 else 0

    if dupan_detected and dupan_true_dir:
        if dupan_true_dir == "home":
            for sc in ["2-1", "2-0", "3-1"]:
                ai_voted[sc] = ai_voted.get(sc, 0) + 3.0
            print("    🎯 诱盘覆盖: 强加主胜比分 2-1/2-0/3-1")
        elif dupan_true_dir == "away":
            for sc in ["1-2", "0-2", "1-3"]:
                ai_voted[sc] = ai_voted.get(sc, 0) + 3.0
            print("    🎯 诱盘覆盖: 强加客胜比分 1-2/0-2/1-3")
        elif dupan_true_dir == "draw":
            for sc in ["1-1", "2-2", "0-0"]:
                ai_voted[sc] = ai_voted.get(sc, 0) + 2.5
            print("    🎯 诱盘覆盖: 强加平局比分 1-1/2-2/0-0")

    away_zero_prob = 50
    away_xg_for_zero = float(engine_result.get("bookmaker_implied_away_xg", 1.2) or 1.2)

    if away_xg_for_zero <= 0.8:
        away_zero_prob += 25
    elif away_xg_for_zero <= 1.0:
        away_zero_prob += 15
    elif away_xg_for_zero <= 1.2:
        away_zero_prob += 8

    away_stats_obj = match_obj.get("away_stats", {})
    if isinstance(away_stats_obj, dict):
        form = str(away_stats_obj.get("form", ""))
        recent5 = form[:5] if form else ""
        recent_L = recent5.count("L")
        if recent_L >= 4:
            away_zero_prob += 15
        elif recent_L >= 3:
            away_zero_prob += 8

        try:
            avg_for = float(away_stats_obj.get("avg_goals_for", 2) or 2)
            if avg_for < 0.8:
                away_zero_prob += 15
            elif avg_for < 1.2:
                away_zero_prob += 8
        except Exception:
            pass

    if shin_h > 65:
        away_zero_prob += 10
    elif shin_h > 55:
        away_zero_prob += 5

    if sharp_detected and sharp_dir == "away":
        away_zero_prob -= 30

    home_zero_prob = 50
    home_xg_for_zero = float(engine_result.get("bookmaker_implied_home_xg", 1.2) or 1.2)

    if home_xg_for_zero <= 0.8:
        home_zero_prob += 25
    elif home_xg_for_zero <= 1.0:
        home_zero_prob += 15
    elif home_xg_for_zero <= 1.2:
        home_zero_prob += 8

    home_stats_obj = match_obj.get("home_stats", {})
    if isinstance(home_stats_obj, dict):
        form = str(home_stats_obj.get("form", ""))
        recent5 = form[:5] if form else ""
        recent_L = recent5.count("L")

        if recent_L >= 4:
            home_zero_prob += 15
        elif recent_L >= 3:
            home_zero_prob += 8

        try:
            avg_for = float(home_stats_obj.get("avg_goals_for", 2) or 2)
            if avg_for < 0.8:
                home_zero_prob += 15
            elif avg_for < 1.2:
                home_zero_prob += 8
        except Exception:
            pass

    if shin_a > 65:
        home_zero_prob += 10
    elif shin_a > 55:
        home_zero_prob += 5

    if sharp_detected and sharp_dir == "home":
        home_zero_prob -= 30

    if exp_goals >= 3.0 or strongest_goal >= 3:
        away_zero_prob = min(away_zero_prob, 50)
        home_zero_prob = min(home_zero_prob, 50)
        print(f"    ⚠️ 大球信号(λ={exp_goals:.2f}, 最强进球={strongest_goal})拦截零封机制")

    zero_boost_applied = False

    if away_zero_prob >= 70 and shin_h > shin_a:
        for sc in ["1-0", "2-0", "3-0"]:
            ai_voted[sc] = ai_voted.get(sc, 0) + 2.0
        for sc in ["1-1", "2-1", "1-2", "3-1", "2-2"]:
            if sc in ai_voted:
                ai_voted[sc] *= 0.65
        print(f"    🧱 客队零封识别({away_zero_prob}分): 强加1-0/2-0/3-0")
        zero_boost_applied = True

    if home_zero_prob >= 70 and shin_a > shin_h:
        for sc in ["0-1", "0-2", "0-3"]:
            ai_voted[sc] = ai_voted.get(sc, 0) + 2.0
        for sc in ["1-1", "1-2", "2-1", "1-3", "2-2"]:
            if sc in ai_voted:
                ai_voted[sc] *= 0.65
        print(f"    🧱 主队零封识别({home_zero_prob}分): 强加0-1/0-2/0-3")
        zero_boost_applied = True

    btts_pct = float(engine_result.get("btts", 50) or 50)
    over25_pct = float(engine_result.get("over_25", engine_result.get("over_2_5", 50)) or 50)

    scenario = "normal"

    if others_info.get("is_extreme_blowout"):
        scenario = "extreme_blowout"
    elif dupan_detected:
        scenario = "sharp_reversal"
    elif exp_goals >= 3.5:
        scenario = "shootout"
    elif exp_goals >= 3.0 and (btts_pct >= 60 or over25_pct >= 60):
        scenario = "high_goals"
    elif exp_goals <= 2.0 and btts_pct <= 40:
        scenario = "low_goals"
    elif btts_pct >= 65:
        scenario = "btts_strong"
    elif btts_pct <= 30:
        scenario = "single_side"

    EXCLUDE_HIGH = {"0-0", "1-0", "0-1"}
    BOOST_HIGH = {"2-1", "1-2", "2-2", "3-1", "1-3"}
    BOOST_SHOOTOUT = {"3-2", "2-3", "3-3", "4-2", "2-4", "4-3", "3-4"}
    EXCLUDE_LOW = {"3-1", "1-3", "2-2", "3-2", "2-3", "3-3", "4-2", "2-4"}
    BOOST_LOW = {"0-0", "1-0", "0-1", "1-1"}
    BTTS_STRONG_BOOST = {"1-1", "2-1", "1-2", "2-2"}
    BTTS_STRONG_EXCLUDE = {"1-0", "2-0", "3-0", "0-1", "0-2", "0-3", "0-0"}
    SINGLE_SIDE_BOOST = {"1-0", "2-0", "0-1", "0-2", "3-0", "0-3"}
    SINGLE_SIDE_EXCLUDE = {"1-1", "2-2"}
    SHARP_REV_HOME_BOOST = {"2-1", "2-0", "3-1", "3-0", "3-2", "4-1", "4-2"}
    SHARP_REV_AWAY_BOOST = {"1-2", "0-2", "1-3", "0-3", "2-3", "1-4", "2-4"}
    SHARP_REV_DRAW_BOOST = {"1-1", "2-2", "0-0"}

    if scenario != "normal":
        scenario_desc = {
            "shootout": f"互射局(λ={exp_goals:.2f})",
            "high_goals": f"高进球场(λ={exp_goals:.2f},BTTS{btts_pct:.0f}%)",
            "low_goals": f"闷场(λ={exp_goals:.2f},BTTS{btts_pct:.0f}%)",
            "btts_strong": f"双方必进(BTTS{btts_pct:.0f}%)",
            "single_side": f"单边场(BTTS{btts_pct:.0f}%)",
            "sharp_reversal": f"诱盘反转(Shin骗局→Sharp真相={dupan_true_dir})",
            "extreme_blowout": "极端惨案防范",
        }
        print(f"    🎬 场景: {scenario_desc.get(scenario, scenario)}")

    all_candidates = set()

    if crs_probs:
        all_candidates.update(crs_probs.keys())
    if backup_probs:
        all_candidates.update(backup_probs.keys())

    all_candidates.update(ai_voted.keys())
    all_candidates.update(ALL_SCORE_OTHERS)

    score_ratings = {}

    for score_str in all_candidates:
        score_str = normalize_score(score_str)
        h_g, a_g = parse_score(score_str)

        if h_g is None:
            continue

        total_g = h_g + a_g
        goal_margin = h_g - a_g
        s = 0.0

        if crs_probs and score_str in crs_probs:
            s += min(35, crs_probs[score_str] * 2.0)
        elif score_str in backup_probs:
            s += min(15, backup_probs[score_str] * 1.2)

        if score_str in ai_voted:
            s += min(40, ai_voted[score_str] * 8)

        if total_g in goal_signals:
            ratio = goal_signals[total_g]
            if zero_boost_applied and total_g >= 5:
                s += min(5, (ratio - 1) * 4)
            else:
                s += min(15, (ratio - 1) * 12)

        if others_info["is_others"]:
            others_boost = 5 if zero_boost_applied else 15
            if score_str in SCORE_OTHERS_HOME and others_info["direction"] == "home":
                s += others_boost
            elif score_str in SCORE_OTHERS_AWAY and others_info["direction"] == "away":
                s += others_boost
            elif score_str in SCORE_OTHERS_DRAW and others_info["direction"] == "draw":
                s += others_boost
            elif score_str in ALL_SCORE_OTHERS:
                s += 2 if zero_boost_applied else 5

        if final_direction == "home" and goal_margin > 0:
            s += 10 * (dir_probs["home"] / 100)
        elif final_direction == "away" and goal_margin < 0:
            s += 10 * (dir_probs["away"] / 100)
        elif final_direction == "draw" and goal_margin == 0:
            s += 10 * (dir_probs["draw"] / 100)
        else:
            s -= 5

        if contrarian_away_score > 3 and goal_margin == 1 and h_g <= 2:
            s -= contrarian_away_score

        if contrarian_home_score > 3 and goal_margin == -1 and a_g <= 2:
            s -= contrarian_home_score

        if strongest_ratio > 2.0 and strongest_goal >= 0:
            if abs(total_g - strongest_goal) > 1:
                s -= 25

        if others_info["ai_others_count"] >= 2 and total_g <= 3:
            s -= 10

        if shin_h > 60 and (home_xg - away_xg) > 1.0 and goal_margin >= 1 and h_g >= 2:
            s += 10

        if scenario == "extreme_blowout":
            if total_g <= 3:
                s *= 0.10
            if score_str in ALL_SCORE_OTHERS or total_g >= 5:
                s *= 3.00
                if others_info["direction"] == "home" and score_str in SCORE_OTHERS_HOME:
                    s += 40
                elif others_info["direction"] == "away" and score_str in SCORE_OTHERS_AWAY:
                    s += 40
                elif others_info["direction"] == "draw" and score_str in SCORE_OTHERS_DRAW:
                    s += 40

        elif scenario == "sharp_reversal":
            if dupan_true_dir == "home":
                if score_str in SHARP_REV_HOME_BOOST:
                    s *= 1.70
                elif goal_margin <= 0:
                    s *= 0.20
            elif dupan_true_dir == "away":
                if score_str in SHARP_REV_AWAY_BOOST:
                    s *= 1.70
                elif goal_margin >= 0:
                    s *= 0.20
            elif dupan_true_dir == "draw":
                if score_str in SHARP_REV_DRAW_BOOST:
                    s *= 1.50
                elif goal_margin != 0:
                    s *= 0.40

        elif scenario == "shootout":
            if score_str in BOOST_SHOOTOUT:
                s *= 1.50
            elif score_str in EXCLUDE_HIGH:
                s *= 0.10
            elif total_g <= 2:
                s *= 0.40

        elif scenario == "high_goals":
            if score_str in EXCLUDE_HIGH:
                s *= 0.10
            elif score_str in BOOST_HIGH:
                s *= 1.30

        elif scenario == "low_goals":
            if score_str in EXCLUDE_LOW:
                s *= 0.10
            elif score_str in BOOST_LOW:
                s *= 1.40

        elif scenario == "btts_strong":
            if score_str in BTTS_STRONG_EXCLUDE:
                s *= 0.20
            elif score_str in BTTS_STRONG_BOOST:
                s *= 1.30

        elif scenario == "single_side":
            if score_str in SINGLE_SIDE_EXCLUDE:
                s *= 0.30
            elif score_str in SINGLE_SIDE_BOOST:
                s *= 1.25

        if scenario == "normal":
            if exp_goals >= 2.7 and score_str in EXCLUDE_HIGH:
                strength = min(1.0, (exp_goals - 2.4) / 1.1)
                s *= max(0.3, 1.0 - strength * 0.5)
            elif exp_goals <= 2.3 and score_str in {"3-1", "1-3", "3-2", "2-3", "3-3"}:
                s *= 0.5

        if s > 0:
            score_ratings[score_str] = round(s, 2)

    ranked = sorted(score_ratings.items(), key=lambda x: x[1], reverse=True)
    final_score = ranked[0][0] if ranked else "1-1"

    if final_score == "9-0":
        final_score = "胜其他"
    if final_score == "9-9":
        final_score = "平其他"
    if final_score == "0-9":
        final_score = "负其他"

    top_score_dir = score_to_direction(final_score)

    if top_score_dir != final_direction and dir_gap > 8:
        aligned_score = None
        for sc, pts in ranked:
            if score_to_direction(sc) == final_direction:
                aligned_score = sc
                break

        if aligned_score:
            top_pts = ranked[0][1]
            aligned_pts = score_ratings.get(aligned_score, 0)

            if aligned_pts >= top_pts * 0.70:
                print(
                    f"    🛡️ [方向一致性] 方向({final_direction})与比分top1({final_score}->{top_score_dir})不符 "
                    f"→ 改用{aligned_score}({aligned_pts:.0f})"
                )
                final_score = aligned_score

    is_score_others_final = final_score in ALL_SCORE_OTHERS or "其他" in final_score

    if is_score_others_final:
        if final_score in SCORE_OTHERS_HOME or final_score == "胜其他":
            display_label = "胜其他"
        elif final_score in SCORE_OTHERS_DRAW or final_score == "平其他":
            display_label = "平其他"
        else:
            display_label = "负其他"
    else:
        display_label = final_score

    print(f"    📊 TOP5: {' > '.join(f'{sc}({pts:.0f})' for sc, pts in ranked[:5])}")

    if is_score_others_final:
        print(f"    🏆 {final_score} → 「{display_label}」")

    target_crs = CRS_FULL_MAP.get(final_score, "")
    final_odds = float(match_obj.get(target_crs, 0) or 0)

    if not final_odds and is_score_others_final:
        if final_score in SCORE_OTHERS_HOME or final_score == "胜其他":
            final_odds = float(match_obj.get("crs_win", 0) or 0)
        elif final_score in SCORE_OTHERS_DRAW or final_score == "平其他":
            final_odds = float(match_obj.get("crs_same", 0) or 0)
        else:
            final_odds = float(match_obj.get("crs_lose", 0) or 0)

    final_prob = crs_probs.get(final_score, backup_probs.get(final_score, 5))
    ev_data = calculate_value_bet(final_prob, final_odds)

    weights = {
        "claude": 1.4,
        "gemini": 1.35,
        "grok": 1.30,
        "gpt": 1.1,
    }

    ai_conf_sum = 0
    ai_conf_count = 0
    value_kills = 0

    for name, r in all_ai.items():
        if not isinstance(r, dict):
            continue

        conf = r.get("ai_confidence", 60)
        ai_conf_sum += conf * weights.get(name, 1.0)
        ai_conf_count += weights.get(name, 1.0)

        if r.get("value_kill"):
            value_kills += 1

    avg_ai_conf = (ai_conf_sum / ai_conf_count) if ai_conf_count > 0 else 60

    cf = min(95, engine_conf + int((avg_ai_conf - 60) * 0.4)) + value_kills * 6

    if not dir_confident:
        cf = max(40, cf - 10)

    if any("🚨" in str(s) for s in smart_signals):
        cf = max(35, cf - 12)

    if cold_door["is_cold_door"]:
        cf = max(30, cf - 5)

    risk = "低" if cf >= 75 else ("中" if cf >= 55 else "高")

    sigs = list(smart_signals)

    if cold_door["is_cold_door"]:
        sigs.extend(cold_door["signals"])

    if others_info["is_others"]:
        sigs.append(f"🔥 胜其他场({others_info['trigger_count']:.1f}条)")

    cl_raw = claude_r.get("ai_score", "") if isinstance(claude_r, dict) else ""
    cl_parsed = parse_score(cl_raw)
    cl_sc = cl_raw if cl_parsed[0] is not None else final_score

    return {
        "predicted_score": final_score,
        "predicted_label": display_label,
        "is_score_others": is_score_others_final,
        "home_win_pct": round(dir_probs["home"], 1),
        "draw_pct": round(dir_probs["draw"], 1),
        "away_win_pct": round(dir_probs["away"], 1),
        "confidence": cf,
        "risk_level": risk,
        "over_under_2_5": "大" if engine_result.get("over_25", 50) > 55 else "小",
        "both_score": "是" if engine_result.get("btts", 45) > 50 else "否",

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
        "ai_consensus_strength": round(ai_consensus_strength, 2),
        "model_agreement": ai_consensus_strength > 0.5,

        "xG_home": round(home_xg, 2),
        "xG_away": round(away_xg, 2),

        "crs_implied_probs": {k: round(v, 2) for k, v in crs_probs.items()} if crs_probs else {},
        "crs_coverage": crs_coverage,
        "crs_margin": crs_margin,

        "scenario": scenario,
        "btts_pct_used": round(btts_pct, 1),
        "over25_pct_used": round(over25_pct, 1),

        "dupan_detected": dupan_detected,
        "dupan_true_dir": dupan_true_dir,
        "dupan_confirm_score": dupan_confirm,
        "shin_dir": shin_dir,
        "sharp_dir": sharp_dir,
        "away_zero_prob": away_zero_prob,
        "home_zero_prob": home_zero_prob,
        "vote_hot_dir": vote_hot_dir,
        "vote_hot_pct": vote_hot_pct,
        "steam_type": steam_type,

        "goal_signals": {str(k): round(v, 2) for k, v in goal_signals.items()},
        "strongest_goal_count": strongest_goal,
        "strongest_goal_ratio": round(strongest_ratio, 2),
        "score_others_info": others_info,

        "sharp_detected": sharp_detected,
        "cold_signals_count": len(cold_signals_raw),
        "contrarian_vote_away": round(contrarian_away_score, 1),
        "contrarian_vote_home": round(contrarian_home_score, 1),

        "suggested_kelly": ev_data["kelly"],
        "edge_vs_market": ev_data["ev"],

        "refined_poisson": stats.get("refined_poisson", {}),
        "poisson": backup_probs,
        "extreme_warning": engine_result.get("scissors_gap_signal", ""),
        "smart_money_signal": " | ".join(sigs),
        "smart_signals": sigs,
        "model_consensus": stats.get("model_consensus", 0),
        "total_models": stats.get("total_models", 11),
        "expected_total_goals": round(exp_goals, 2),
        "over_2_5": engine_result.get("over_25", 50),
        "btts": engine_result.get("btts", 45),

        "top_scores": stats.get("refined_poisson", {}).get("top_scores", []),
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
        "bookmaker_implied_home_xg": engine_result.get("bookmaker_implied_home_xg", "?"),
        "bookmaker_implied_away_xg": engine_result.get("bookmaker_implied_away_xg", "?"),
        "cold_door": cold_door,
    }


# ====================================================================
# 推荐 Top4
# ====================================================================

def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})
        s = pr.get("confidence", 0) * 0.4

        mx = max(
            pr.get("home_win_pct", 33),
            pr.get("away_win_pct", 33),
            pr.get("draw_pct", 33),
        )

        s += (mx - 33) * 0.2 + pr.get("model_consensus", 0) * 2

        if pr.get("risk_level") == "低":
            s += 12
        elif pr.get("risk_level") == "高":
            s -= 5

        if pr.get("model_agreement"):
            s += 10

        exp_info = pr.get("experience_analysis", {})
        exp_score = exp_info.get("total_score", 0)

        if exp_score >= 15 and pr.get("result") == "平局" and exp_info.get("draw_rules", 0) >= 3:
            s += 12
        elif exp_score >= 10:
            s += 5

        if exp_info.get("recommendation", "").startswith("⚠️"):
            s -= 3

        smart_money = str(pr.get("smart_money_signal", ""))
        direction = pr.get("result", "")

        if "Sharp" in smart_money:
            if ("客胜" in smart_money and direction == "主胜") or ("主胜" in smart_money and direction == "客胜"):
                s -= 30

        cold = pr.get("cold_door", {})
        if cold.get("is_cold_door"):
            s -= 8

        if pr.get("is_score_others"):
            s += 8

        if pr.get("dupan_detected"):
            s += 6

        if pr.get("ai_abstained"):
            s -= min(12, len(pr.get("ai_abstained", [])) * 3)

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

    print("\n" + "=" * 80)
    print(f"  [vMAX 18.3] CRS直接概率 + 全信号 + 固定模型 + 不拆批 + 强解析 | {len(ms)} 场")
    print("=" * 80)

    match_analyses = []

    for i, m in enumerate(ms):
        try:
            eng = predict_match(m)
        except Exception as e:
            print(f"    ⚠️ odds_engine失败: {type(e).__name__}: {str(e)[:100]}")
            eng = {
                "confidence": 50,
                "home_prob": 33.3,
                "draw_prob": 33.3,
                "away_prob": 33.3,
                "bookmaker_implied_home_xg": 1.25,
                "bookmaker_implied_away_xg": 1.10,
                "over_25": 50,
                "btts": 45,
            }

        try:
            league_info, _, _, _ = build_league_intelligence(m)
        except Exception:
            league_info = {}

        try:
            sp = ensemble.predict(m, {})
        except Exception as e:
            print(f"    ⚠️ ensemble失败: {type(e).__name__}: {str(e)[:100]}")
            sp = {
                "model_consensus": 0,
                "total_models": 0,
                "smart_signals": [],
                "refined_poisson": {},
            }

        try:
            exp_result = exp_engine.analyze(m)
        except Exception:
            exp_result = {}

        match_analyses.append({
            "match": m,
            "engine": eng,
            "league_info": league_info,
            "stats": sp,
            "index": i + 1,
            "experience": exp_result,
        })

    all_ai = {
        "claude": {},
        "gemini": {},
        "gpt": {},
        "grok": {},
    }

    if use_ai and match_analyses:
        start_t = time.time()
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
            m,
        )

        try:
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

        score_str = normalize_score(mg.get("predicted_score", "1-1"))
        expected_result = score_to_result_cn(score_str)

        if expected_result:
            mg["result"] = expected_result
        else:
            pcts = {
                "主胜": mg.get("home_win_pct", 33),
                "平局": mg.get("draw_pct", 33),
                "客胜": mg.get("away_win_pct", 33),
            }
            mg["result"] = max(pcts, key=pcts.get)

        try:
            expected_dir = score_to_result_cn(score_str)
            if expected_dir and mg.get("result") != expected_dir:
                print(f"    ⚠️ [一致性修复] {score_str}方向应为{expected_dir},覆盖旧result={mg.get('result')}")
                mg["result"] = expected_dir

            pl = mg.get("predicted_label", "")
            if pl and pl not in (score_str, "胜其他", "平其他", "负其他"):
                print(f"    ⚠️ [一致性修复] predicted_label={pl}与比分{score_str}不匹配,覆盖")
                mg["predicted_label"] = score_str
        except Exception:
            pass

        res.append({**m, "prediction": mg})

        cold = mg.get("cold_door", {})
        cold_tag = f" [❄️{cold.get('level', '')}冷门]" if cold.get("is_cold_door") else ""
        others_tag = " [🔥胜其他]" if mg.get("is_score_others") else ""
        sharp_tag = " [💰Sharp]" if mg.get("sharp_detected") else ""
        dupan_tag = " [⚡诱盘]" if mg.get("dupan_detected") else ""
        ai_bad = mg.get("ai_abstained", [])
        ai_tag = f" [🚫{','.join(ai_bad)}]" if ai_bad else ""

        print(
            f"  [{i + 1}] {m.get('home_team')} vs {m.get('away_team')} "
            f"=> {mg['result']} ({mg['predicted_score']}={mg['predicted_label']}) "
            f"| CF: {mg['confidence']}% | 方向: "
            f"{max(mg.get('home_win_pct', 0), mg.get('draw_pct', 0), mg.get('away_win_pct', 0)):.0f}%"
            f"{cold_tag}{others_tag}{sharp_tag}{dupan_tag}{ai_tag} [{mg.get('scenario', 'normal')}]"
        )

    t4 = select_top4(res)
    t4ids = [t.get("id") for t in t4]

    for r in res:
        r["is_recommended"] = r.get("id") in t4ids

    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    diary = load_ai_diary()

    cold_count = len([
        r for r in res
        if r.get("prediction", {}).get("cold_door", {}).get("is_cold_door")
    ])
    others_count = len([
        r for r in res
        if r.get("prediction", {}).get("is_score_others")
    ])
    sharp_count = len([
        r for r in res
        if r.get("prediction", {}).get("sharp_detected")
    ])
    dupan_count = len([
        r for r in res
        if r.get("prediction", {}).get("dupan_detected")
    ])

    diary["yesterday_win_rate"] = f"{len([r for r in res if r['prediction']['confidence'] > 70])}/{max(1, len(res))}"
    diary["reflection"] = (
        f"vMAX18.3 | {cold_count}冷门 {others_count}胜其他 {sharp_count}Sharp {dupan_count}诱盘 "
        f"| 固定模型·不拆批·强解析·CRS全信号"
    )

    save_ai_diary(diary)

    return res, t4


if __name__ == "__main__":
    logger.info("vMAX 18.3 启动")
    print("✅ vMAX 18.3 已加载 — 固定模型·不拆批·不切换·强解析·CRS全信号")