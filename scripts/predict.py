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
# 🛡️ vMAX 17.0 方案B — 删泊松·全靠数据+AI (严格完整架构版)
#
# 核心变革 (vs v16):
#   ❌ 删除泊松双变量分布
#   ❌ 删除Dixon-Coles
#   ❌ 删除蒙特卡洛
#   ✅ 新增 CRS赔率直接反推概率 (引入 Power Method 幂法)
#   ✅ 恢复v14.3全部盘口信号 (Steam/散户反指/赔率变动/冷门预警)
#   ✅ Sharp在direction+xG两层都生效 (不只xG层)
#   ✅ 新增 贝叶斯质心漂移 + Softmax 连续概率面打分机制
#   ✅ Multi-Agent AI 角色化并发编排
#   🚨 新增 【绝对一致性护栏】：彻底修复 UI 0-1配主胜/平局 的逻辑错位
# ====================================================================
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    print("  [WARN] ⚠️ 未检测到 structlog 库，自动降级为标准 logging 模块")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
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
    pass


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

def calculate_value_bet(prob_pct, odds):
    if not odds or odds <= 1.05:
        return {"ev": 0.0, "kelly": 0.0, "is_value": False}
    prob = prob_pct / 100.0
    ev = (prob * odds) - 1.0
    b = odds - 1.0
    q = 1.0 - prob
    if b <= 0:
        return {"ev": round(ev * 100, 2), "kelly": 0.0, "is_value": False}
    # 结合软性分布面的保守凯利调整
    kelly = ((b * prob) - q) / b
    return {
        "ev": round(ev * 100, 2),
        "kelly": round(max(0.0, kelly * 0.35) * 100, 2), 
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
# 🎯 核心算法1: CRS赔率反推 (幂法重构，消灭冷门假象)
# ====================================================================
def find_power_k(odds_list, target_margin=1.0, tol=1e-4, max_iter=50):
    low, high = 0.01, 5.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        current = sum((1.0 / o) ** mid for o in odds_list)
        if abs(current - target_margin) < tol:
            return mid
        elif current > target_margin:
            low = mid
        else:
            high = mid
    return (low + high) / 2

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

    odds_list = list(raw_odds.values()) + [v["odds"] for v in extras.values()]
    margin = sum(1/o for o in odds_list) - 1.0

    k_opt = find_power_k(odds_list, target_margin=1.0)
    
    probs = {}
    for score, odds in raw_odds.items():
        probs[score] = ((1 / odds) ** k_opt) * 100

    for key, extra_data in extras.items():
        total_prob = ((1 / extra_data["odds"]) ** k_opt) * 100
        num_scores = len(extra_data["scores"])
        if num_scores > 0:
            per_score = total_prob / num_scores
            for sc in extra_data["scores"]:
                if sc not in probs: probs[sc] = per_score
                else: probs[sc] += per_score

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
            info_text = (match_obj["points"].get("home_strength", "") + match_obj["points"].get("guest_strength", ""))
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
# 🧊 冷门猎手引擎 (原封保留)
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
        if strength >= 12: level = "顶级"
        elif strength >= 7: level = "高危"
        else: level = "普通"

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
    return {"yesterday_win_rate": "N/A", "reflection": "持续进化中", "kill_history": []}

def save_ai_diary(diary):
    os.makedirs("data", exist_ok=True)
    with open("data/ai_diary.json", "w", encoding="utf-8") as f:
        json.dump(diary, f, ensure_ascii=False, indent=2)

# ====================================================================
# 🧠 vMAX 17.0 Prompt — 降维重构 (信息结构化，剥离算术题)
# ====================================================================
def build_phase1_prompt(match_analyses):
    p = "作为量化分析师团队，你现在面临以下赛事的最终裁决。注意：Python量化引擎已经为你算好了异常数据，请基于这些异动结论推理庄家的真实剧本。\n\n"
    p += "【输出格式】只输出JSON数组。每场必须包含:\n"
    p += "  match(整数), top3([{score,prob}],...), reason(350字推理庄家意图), \n"
    p += "  is_score_others(true/false), detected_signals([\"冷门高危\"])\n\n"

    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        eng = ma.get("engine", {})
        stats = ma.get("stats", {})
        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        league = m.get("league", m.get("cup", ""))

        p += f"{'='*50}\n[{i+1}] {h} vs {a} | {league}\n"

        sp_h = float(m.get("sp_home", m.get("win", 0)) or 0)
        sp_d = float(m.get("sp_draw", m.get("same", 0)) or 0)
        sp_a = float(m.get("sp_away", m.get("lose", 0)) or 0)
        p += f"基础欧赔: {sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f}\n"

        hxg = eng.get('bookmaker_implied_home_xg', '?')
        axg = eng.get('bookmaker_implied_away_xg', '?')
        p += f"引擎测算期望进球: 主{hxg} vs 客{axg}\n"

        smart_sigs = stats.get('smart_signals', [])
        if smart_sigs:
            p += f"🔥 资金盘异动探针: {', '.join(str(s) for s in smart_sigs[:5])}\n"
        
        vote = m.get("vote", {})
        if vote:
            max_v = max(int(vote.get('win', 33)), int(vote.get('lose', 33)))
            if max_v >= 58: p += f"⚠️ 散户热度异常: 极易触发庄家诱盘反杀\n"

        a_list = []
        for g in range(8):
            v = m.get(f"a{g}", "")
            try:
                actual = float(v or 0)
                if actual > 1 and STANDARD_GOAL_ODDS.get(g, 50) / actual > 1.4:
                    a_list.append(f"{g}球")
            except: pass
        if a_list:
            p += f"🚨 庄家极限压低赔率的进球数: {', '.join(a_list)}\n"
        
        info = m.get("information", {})
        if isinstance(info, dict):
            for k, label in [("home_injury", "主伤停"), ("guest_bad_news", "客利空")]:
                if info.get(k): p += f"{label}: {str(info[k])[:200]}\n"
        p += "\n"

    p += f"【请严格按JSON格式输出{len(match_analyses)}场比赛裁决】\n"
    return p


# ====================================================================
# AI调用引擎 - Multi-Agent 角色编排
# ====================================================================
FALLBACK_URLS = [None, "https://api520.pro/v1", "https://api521.pro/v1", "https://api522.pro/v1"]
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
    if not key and ai_name == "gpt": key = GPT_DEFAULT_KEY
    if not key: return ai_name, {}, "no_key"

    if ai_name == "gpt":
        primary_url = get_clean_env_url(url_env, GPT_DEFAULT_URL)
        if not primary_url or "poloai" not in primary_url: primary_url = GPT_DEFAULT_URL
        urls = [primary_url]
    else:
        primary_url = get_clean_env_url(url_env)
        backup = [u for u in FALLBACK_URLS if u and u != primary_url][:1]
        urls = [primary_url] + backup

    CONNECT_TIMEOUT, READ_TIMEOUT = 20, {"claude": 350, "grok": 250, "gpt": 250, "gemini": 250}.get(ai_name, 200)

    AI_PROFILES = {
        "claude": {
            "sys": ("你是红军风控长(诱盘审查官)。进行【红蓝对抗推理】：如果赔率和散户全指向某一方，"
                    "强制扮演庄家做空者。找出隐蔽的爆冷逻辑。若找不出，才允许预测正路。只输出JSON数组。"),
            "temp": 0.25
        },
        "grok": {
            "sys": ("你是情报搜集官。专注球队的场外因素、战意陷阱、大热必死情绪。不用算数学，专注心理战。只输出JSON数组。"),
            "temp": 0.3
        },
        "gpt": {
            "sys": ("你是蓝军数据分析师。专注 xG 偏差和进球数被压低的奇异点。严谨执行庄家的真实进球预期。只输出JSON数组。"),
            "temp": 0.15
        },
        "gemini": {
            "sys": ("你是模式识别引擎。结合红军和蓝军的视角，对剧本进行综合评估，输出最终比分预测。只输出JSON数组。"),
            "temp": 0.15
        },
    }

    profile = AI_PROFILES.get(ai_name, AI_PROFILES["gpt"])

    for mn in models_list:
        connected = False
        for base_url in urls:
            if not base_url: continue
            is_gem = "generateContent" in base_url
            url = base_url.rstrip("/")
            if not is_gem and "chat/completions" not in url: url += "/chat/completions"
            headers = {"Content-Type": "application/json"}

            if is_gem:
                headers["x-goog-api-key"] = key
                payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": profile["temp"]}, "systemInstruction": {"parts": [{"text": profile["sys"]}]}}
            else:
                headers["Authorization"] = f"Bearer {key}"
                payload = {"model": mn, "messages": [{"role": "system", "content": profile["sys"]}, {"role": "user", "content": prompt}], "temperature": profile["temp"]}

            try:
                timeout = aiohttp.ClientTimeout(total=None, connect=CONNECT_TIMEOUT, sock_connect=CONNECT_TIMEOUT, sock_read=READ_TIMEOUT)
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                    if r.status in (502, 504, 400, 429): break
                    if r.status != 200: continue
                    connected = True
                    data = await r.json(content_type=None)
                    
                    raw_text = ""
                    try:
                        if is_gem: raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        else:
                            msg = data.get("choices", [{}])[0].get("message", {})
                            if msg.get("content"): raw_text = msg["content"].strip()
                    except: pass

                    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
                    json_str = ""
                    m_re = re.search(r'\[\s*\{\s*"match"', clean)
                    if m_re:
                        start_idx = m_re.start()
                        depth = 0; end_idx = start_idx
                        for i in range(start_idx, len(clean)):
                            if clean[i] == '[': depth += 1
                            elif clean[i] == ']':
                                depth -= 1
                                if depth == 0: end_idx = i + 1; break
                        if end_idx > start_idx: json_str = clean[start_idx:end_idx]

                    results = {}
                    if json_str:
                        try: arr = json.loads(json_str)
                        except: arr = []
                        if isinstance(arr, list):
                            for item in arr:
                                if not isinstance(item, dict) or not item.get("match"): continue
                                try: mid = int(item["match"])
                                except: continue
                                t1 = item.get("top3", [{}])[0].get("score", "1-1") if item.get("top3") else item.get("score", "1-1")
                                results[mid] = {
                                    "ai_score": str(t1).replace(" ", "").strip(),
                                    "reason": str(item.get("reason", ""))[:800],
                                    "is_score_others": bool(item.get("is_score_others", False))
                                }

                    if results: return ai_name, results, mn
                    else: break
            except: continue
    return ai_name, {}, "all_failed"

async def run_ai_matrix_two_phase(match_analyses):
    prompt = build_phase1_prompt(match_analyses)
    ai_configs = [
        ("grok", "GROK_API_URL", "GROK_API_KEY", ["熊猫-A-6-grok-4.2-thinking"]),
        ("gpt", "GPT_API_URL", "GPT_API_KEY", ["gpt-5.4"]),
        ("gemini", "GEMINI_API_URL", "GEMINI_API_KEY", ["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking"]),
        ("claude", "CLAUDE_API_URL", "CLAUDE_API_KEY", ["熊猫-按量-特供顶级-官方正向满血-claude-opus-4.6-thinking"]),
    ]
    all_results = {"gpt": {}, "grok": {}, "gemini": {}, "claude": {}}
    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [async_call_one_ai_batch(session, prompt, u, k, m, len(match_analyses), n) for n, u, k, m in ai_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, tuple): all_results[res[0]] = res[1]
    return all_results


# ====================================================================
# 🌟 Merge v17.0 重构版 — 贝叶斯质心漂移 + Softmax概率表面
# (严格保留原架构所有向下兼容字段、信号提取段)
# ====================================================================
def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    # 🔧 字段兼容
    if isinstance(match_obj.get("v2_odds_dict"), dict):
        v2 = match_obj["v2_odds_dict"]
        match_obj = {**match_obj, **v2}

    league = str(match_obj.get("league", match_obj.get("cup", "")))
    sp_h = float(match_obj.get("sp_home", match_obj.get("win", 0)) or 0)
    sp_d = float(match_obj.get("sp_draw", match_obj.get("same", 0)) or 0)
    sp_a = float(match_obj.get("sp_away", match_obj.get("lose", 0)) or 0)
    engine_conf = engine_result.get("confidence", 50)

    # 🛡️ AI有效性检测
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
        "gpt": _is_valid_ai(gpt_r), "grok": _is_valid_ai(grok_r),
        "gemini": _is_valid_ai(gemini_r), "claude": _is_valid_ai(claude_r)
    }
    all_ai = {n: r for n, r in [("gpt", gpt_r), ("grok", grok_r), ("gemini", gemini_r), ("claude", claude_r)] if ai_valid[n]}

    # ============ 原始探针及信号收集区 ============
    if sp_h > 1 and sp_d > 1 and sp_a > 1:
        margin = 1/sp_h + 1/sp_d + 1/sp_a
        shin_h = (1/sp_h)/margin*100; shin_d = (1/sp_d)/margin*100; shin_a = (1/sp_a)/margin*100
    else: shin_h = shin_d = shin_a = 33.3
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
                if "确认" in s_str and "→" not in s_str and "流向" not in s_str: continue
                if _re_sharp.search(r"(主胜|主队|走主|→\s*主|流向\s*主|资金\s*主)", s_str): sharp_dir = "home"; break
                elif _re_sharp.search(r"(客胜|客队|走客|→\s*客|流向\s*客|资金\s*客)", s_str): sharp_dir = "away"; break
                elif _re_sharp.search(r"(平局|平赔|走平|→\s*平|流向\s*平|资金\s*平)", s_str): sharp_dir = "draw"; break

    steam_dir = None
    steam_type = None 
    if "Steam" in smart_str:
        import re as _re_steam
        for s in smart_signals:
            s_str = str(s)
            if "Steam" not in s_str: continue
            is_reverse = "反向" in s_str or "未跟" in s_str or "不跟" in s_str
            if _re_steam.search(r"(主胜\s*Steam|Steam.*主胜|主胜.*降水|主.*Steam)", s_str):
                steam_dir = "home"; steam_type = "reverse" if is_reverse else "normal"; break
            elif _re_steam.search(r"(客胜\s*Steam|Steam.*客胜|客胜.*降水|客.*Steam)", s_str):
                steam_dir = "away"; steam_type = "reverse" if is_reverse else "normal"; break
            elif _re_steam.search(r"(平局\s*Steam|Steam.*平局|平.*Steam)", s_str):
                steam_dir = "draw"; steam_type = "reverse" if is_reverse else "normal"; break

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
        if vote_hot_dir and vote_hot_dir != sharp_dir and vote_hot_pct >= 58: dupan_confirm += 2
        if steam_dir == sharp_dir:
            if steam_type == "reverse": dupan_confirm += 3
            else: dupan_confirm += 2
        if change_down_dir == sharp_dir: dupan_confirm += 2
        if cold_signals_raw: dupan_confirm += min(3, len(cold_signals_raw))
        if dupan_confirm >= 3:
            dupan_detected = True
            dupan_true_dir = sharp_dir

    # ============ 🎯 基础方向概率打分 (复用原始逻辑，提供大盘概率面基准) ============
    direction_scores = {"home": 0.0, "draw": 0.0, "away": 0.0}
    shin_weight = 15 if dupan_detected else 30
    direction_scores["home"] += shin_h/100 * shin_weight
    direction_scores["draw"] += shin_d/100 * shin_weight
    direction_scores["away"] += shin_a/100 * shin_weight

    if sharp_detected and sharp_dir:
        sharp_base = 35 if dupan_detected else 25
        direction_scores[sharp_dir] += sharp_base

    if steam_dir:
        direction_scores[steam_dir] += 20 if steam_type == "reverse" else 10

    contrarian_away_score = 0  
    contrarian_home_score = 0
    if vote_hot_dir and vote_hot_pct >= 55:
        contra_weight = 22 if vote_hot_pct >= 68 else (14 if vote_hot_pct >= 60 else 6)
        for d in ["home", "draw", "away"]:
            if d != vote_hot_dir:
                direction_scores[d] += contra_weight * 0.5
        direction_scores[vote_hot_dir] -= contra_weight * 0.3
        if vote_hot_dir == "home": contrarian_away_score = contra_weight
        elif vote_hot_dir == "away": contrarian_home_score = contra_weight

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
        if cold_score >= 6:
            cold_power = 18 if cold_score >= 25 else (12 if cold_score >= 18 else (8 if cold_score >= 12 else 4))
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

    pre_pred = {
        "home_win_pct": dir_probs["home"], "draw_pct": dir_probs["draw"], "away_win_pct": dir_probs["away"],
        "steam_move": stats.get("steam_move", {}), "smart_signals": smart_signals,
        "line_movement_anomaly": stats.get("line_movement_anomaly", {})
    }
    cold_door = ColdDoorDetector.detect(match_obj, pre_pred)

    # ============ 🎯 期望进球 (保留原样) ============
    exp_goals = 0.0
    for src, src_name in [(engine_result, "engine"), (stats, "stats")]:
        if not src: continue
        for k in ["expected_total_goals", "exp_goals", "total_goals",
                  "expected_goals", "lambda_total", "total_xg"]:
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
        try:
            hxg = float(engine_result.get("bookmaker_implied_home_xg", 0) or 0)
            axg = float(engine_result.get("bookmaker_implied_away_xg", 0) or 0)
            if hxg > 0 and axg > 0: exp_goals = hxg + axg
        except: pass

    if exp_goals <= 0:
        try:
            gp = []
            for gi in range(8):
                v = float(match_obj.get(f"a{gi}", 0) or 0)
                if v > 1: gp.append((gi, 1/v))
            if gp:
                tp = sum(p for _, p in gp)
                exp_goals = sum(g*(p/tp) for g, p in gp)
        except: pass

    if exp_goals <= 0:
        try:
            over25 = float(engine_result.get("over_25", 50) or 50)
            exp_goals = 2.0 + (over25 - 40) * 0.015
        except: pass

    if exp_goals < 1.0 or exp_goals > 6.0: exp_goals = 2.5

    # ============ 🎯 第三/四/五/六层 (保留原始打分机制) ============
    goal_signals = detect_goal_signals(match_obj)
    strongest_goal = -1
    strongest_ratio = 1.0
    if goal_signals:
        strongest_goal = max(goal_signals, key=goal_signals.get)
        strongest_ratio = goal_signals[strongest_goal]

    others_info = detect_score_others(match_obj, exp_goals, all_ai)
    crs_probs, crs_margin, crs_coverage = crs_implied_probabilities(match_obj)

    home_xg = float(engine_result.get("bookmaker_implied_home_xg", 1.3) or 1.3)
    away_xg = float(engine_result.get("bookmaker_implied_away_xg", 0.9) or 0.9)
    if sharp_detected:
        if "客胜" in smart_str or "客队" in smart_str:
            home_xg *= 0.85; away_xg *= 1.20
        elif "主胜" in smart_str or "主队" in smart_str:
            home_xg *= 1.15; away_xg *= 0.85
    if cold_door["is_cold_door"] and not sharp_detected:
        if hot_side == "home":
            home_xg *= 0.75; away_xg *= 1.25
        else:
            away_xg *= 0.75; home_xg *= 1.25
    home_xg = max(0.3, min(4.0, home_xg))
    away_xg = max(0.2, min(3.5, away_xg))

    backup_probs = {}
    if not crs_probs or crs_coverage < 0.5:
        for h_g in range(6):
            for a_g in range(6):
                p_h = math.exp(-home_xg) * (home_xg ** h_g) / math.factorial(h_g)
                p_a = math.exp(-away_xg) * (away_xg ** a_g) / math.factorial(a_g)
                backup_probs[f"{h_g}-{a_g}"] = round(p_h * p_a * 100, 2)

    ai_voted = {}
    for name, r in all_ai.items():
        if not isinstance(r, dict): continue
        sc = parse_score(r.get("ai_score", ""))
        if not (sc and sc[0] is not None):
            t3 = r.get("top3", [])
            if t3 and len(t3) > 0:
                sc = parse_score(t3[0].get("score", ""))
        if sc and sc[0] is not None:
            key = f"{sc[0]}-{sc[1]}"
            w = 1.5 if name == "claude" else (1.40 if name == "gemini" else (1.35 if name == "grok" else 1.0))
            ai_voted[key] = ai_voted.get(key, 0) + w
        t3 = r.get("top3", [])
        if isinstance(t3, list):
            for rank, t in enumerate(t3[1:3], 2):
                s2 = parse_score(t.get("score", ""))
                if s2 and s2[0] is not None:
                    key2 = f"{s2[0]}-{s2[1]}"
                    w2 = 0.4 if rank == 2 else 0.2
                    ai_voted[key2] = ai_voted.get(key2, 0) + w2

    def _score_to_dir(score_str):
        try:
            parts = str(score_str).strip().replace(" ", "").split("-")
            h, a = int(parts[0]), int(parts[1])
            if h > a: return "home"
            elif h < a: return "away"
            else: return "draw"
        except: return None

    if ai_valid["claude"] and claude_r.get("ai_confidence", 0) >= 75:  
        cl_score = claude_r.get("ai_score", "")
        cl_dir = _score_to_dir(cl_score)
        if cl_score and cl_dir:
            other_ai_dirs = {}
            valid_others = 0
            other_confidences = []  
            for name in ["gpt", "grok", "gemini"]:
                if not ai_valid[name]: continue
                valid_others += 1
                r = all_ai.get(name, {})
                if isinstance(r, dict):
                    sc = r.get("ai_score", "")
                    d = _score_to_dir(sc)
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
                    cl_clean = cl_score.replace(" ", "").strip()
                    if cl_clean in ai_voted:
                        ai_voted[cl_clean] *= 2.0
                elif direction_independent and is_hard_majority and not confidence_dominant:
                    cl_clean = cl_score.replace(" ", "").strip()
                    if cl_clean in ai_voted:
                        ai_voted[cl_clean] *= 1.3
                elif not direction_independent:
                    cl_clean = cl_score.replace(" ", "").strip()
                    if cl_clean in ai_voted:
                        ai_voted[cl_clean] *= 1.3

    ai_consensus_strength = 0
    if ai_voted:
        max_vote = max(ai_voted.values())
        total_vote = sum(ai_voted.values())
        ai_consensus_strength = max_vote / total_vote if total_vote > 0 else 0

    if dupan_detected and dupan_true_dir:
        if dupan_true_dir == "home":
            for sc in ["2-1", "2-0", "3-1"]: ai_voted[sc] = ai_voted.get(sc, 0) + 3.0
        elif dupan_true_dir == "away":
            for sc in ["1-2", "0-2", "1-3"]: ai_voted[sc] = ai_voted.get(sc, 0) + 3.0
        elif dupan_true_dir == "draw":
            for sc in ["1-1", "2-2", "0-0"]: ai_voted[sc] = ai_voted.get(sc, 0) + 2.5

    away_zero_prob = 50  
    away_xg_for_zero = float(engine_result.get("bookmaker_implied_away_xg", 1.2) or 1.2)
    if away_xg_for_zero <= 0.8: away_zero_prob += 25
    elif away_xg_for_zero <= 1.0: away_zero_prob += 15
    elif away_xg_for_zero <= 1.2: away_zero_prob += 8

    away_stats_obj = match_obj.get("away_stats", {})
    if isinstance(away_stats_obj, dict):
        form = str(away_stats_obj.get("form", ""))
        recent5 = form[:5] if form else ""
        recent_L = recent5.count("L")
        if recent_L >= 4: away_zero_prob += 15
        elif recent_L >= 3: away_zero_prob += 8
        try:
            avg_for = float(away_stats_obj.get("avg_goals_for", 2) or 2)
            if avg_for < 0.8: away_zero_prob += 15
            elif avg_for < 1.2: away_zero_prob += 8
        except: pass

    if shin_h > 65: away_zero_prob += 10
    elif shin_h > 55: away_zero_prob += 5

    if sharp_detected and sharp_dir == "away":
        away_zero_prob -= 30

    home_zero_prob = 50
    home_xg_for_zero = float(engine_result.get("bookmaker_implied_home_xg", 1.2) or 1.2)
    if home_xg_for_zero <= 0.8: home_zero_prob += 25
    elif home_xg_for_zero <= 1.0: home_zero_prob += 15
    elif home_xg_for_zero <= 1.2: home_zero_prob += 8

    home_stats_obj = match_obj.get("home_stats", {})
    if isinstance(home_stats_obj, dict):
        form = str(home_stats_obj.get("form", ""))
        recent5 = form[:5] if form else ""
        recent_L = recent5.count("L")
        if recent_L >= 4: home_zero_prob += 15
        elif recent_L >= 3: home_zero_prob += 8
        try:
            avg_for = float(home_stats_obj.get("avg_goals_for", 2) or 2)
            if avg_for < 0.8: home_zero_prob += 15
            elif avg_for < 1.2: home_zero_prob += 8
        except: pass

    if shin_a > 65: home_zero_prob += 10
    elif shin_a > 55: home_zero_prob += 5
    if sharp_detected and sharp_dir == "home":
        home_zero_prob -= 30

    if exp_goals >= 3.0 or strongest_goal >= 3:
        away_zero_prob = min(away_zero_prob, 50)
        home_zero_prob = min(home_zero_prob, 50)

    zero_boost_applied = False
    if away_zero_prob >= 70 and shin_h > shin_a:
        for sc in ["1-0", "2-0", "3-0"]: ai_voted[sc] = ai_voted.get(sc, 0) + 2.0
        for sc in ["1-1", "2-1", "1-2", "3-1", "2-2"]:
            if sc in ai_voted: ai_voted[sc] *= 0.65
        zero_boost_applied = True

    if home_zero_prob >= 70 and shin_a > shin_h:
        for sc in ["0-1", "0-2", "0-3"]: ai_voted[sc] = ai_voted.get(sc, 0) + 2.0
        for sc in ["1-1", "1-2", "2-1", "1-3", "2-2"]:
            if sc in ai_voted: ai_voted[sc] *= 0.65
        zero_boost_applied = True

    # ============ 第七层: 综合评分 ============
    btts_pct = float(engine_result.get("btts", 50) or 50)
    over25_pct = float(engine_result.get("over_25", engine_result.get("over_2_5", 50)) or 50)

    scenario = "normal"
    if others_info.get("is_extreme_blowout"): scenario = "extreme_blowout"
    elif dupan_detected: scenario = "sharp_reversal"
    elif exp_goals >= 3.5: scenario = "shootout"
    elif exp_goals >= 3.0 and (btts_pct >= 60 or over25_pct >= 60): scenario = "high_goals"
    elif exp_goals <= 2.0 and btts_pct <= 40: scenario = "low_goals"
    elif btts_pct >= 65: scenario = "btts_strong"  
    elif btts_pct <= 30: scenario = "single_side"  

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

    all_candidates = set()
    if crs_probs: all_candidates.update(crs_probs.keys())
    if backup_probs: all_candidates.update(backup_probs.keys())
    all_candidates.update(ai_voted.keys())
    all_candidates.update(ALL_SCORE_OTHERS)

    score_ratings = {}
    for score_str in all_candidates:
        if score_str in ["胜其他", "9-0"]: h_g, a_g = 9, 0
        elif score_str in ["平其他", "9-9"]: h_g, a_g = 9, 9
        elif score_str in ["负其他", "0-9"]: h_g, a_g = 0, 9
        else:
            try: h_g, a_g = map(int, score_str.split("-"))
            except: continue
        total_g = h_g + a_g
        s = 0.0

        if crs_probs and score_str in crs_probs: s += min(35, crs_probs[score_str] * 2.0)
        elif score_str in backup_probs: s += min(15, backup_probs[score_str] * 1.2)

        if score_str in ai_voted: s += min(40, ai_voted[score_str] * 8)

        if total_g in goal_signals:
            ratio = goal_signals[total_g]
            if zero_boost_applied and total_g >= 5: s += min(5, (ratio - 1) * 4)  
            else: s += min(15, (ratio - 1) * 12)

        if others_info["is_others"]:
            others_boost = 5 if zero_boost_applied else 15
            if score_str in SCORE_OTHERS_HOME and others_info["direction"] == "home": s += others_boost
            elif score_str in SCORE_OTHERS_AWAY and others_info["direction"] == "away": s += others_boost
            elif score_str in SCORE_OTHERS_DRAW and others_info["direction"] == "draw": s += others_boost
            elif score_str in ALL_SCORE_OTHERS: s += 2 if zero_boost_applied else 5

        goal_margin = h_g - a_g
        if final_direction == "home" and goal_margin > 0: s += 10 * (dir_probs["home"] / 100)
        elif final_direction == "away" and goal_margin < 0: s += 10 * (dir_probs["away"] / 100)
        elif final_direction == "draw" and goal_margin == 0: s += 10 * (dir_probs["draw"] / 100)
        else: s -= 5  

        if contrarian_away_score > 3 and goal_margin == 1 and h_g <= 2: s -= contrarian_away_score
        if contrarian_home_score > 3 and goal_margin == -1 and a_g <= 2: s -= contrarian_home_score

        if strongest_ratio > 2.0 and strongest_goal >= 0 and abs(total_g - strongest_goal) > 1: s -= 25
        if others_info["ai_others_count"] >= 2 and total_g <= 3: s -= 10
        if shin_h > 60 and (home_xg - away_xg) > 1.0 and goal_margin >= 1 and h_g >= 2: s += 10

        if scenario == "extreme_blowout":
            if total_g <= 3: s *= 0.10  
            if score_str in ALL_SCORE_OTHERS or total_g >= 5:
                s *= 3.00  
                if others_info["direction"] == "home" and score_str in SCORE_OTHERS_HOME: s += 40
                elif others_info["direction"] == "away" and score_str in SCORE_OTHERS_AWAY: s += 40
                elif others_info["direction"] == "draw" and score_str in SCORE_OTHERS_DRAW: s += 40

        elif scenario == "sharp_reversal":
            if dupan_true_dir == "home":
                if score_str in SHARP_REV_HOME_BOOST: s *= 1.70  
                elif goal_margin <= 0:
                    if score_str in ALL_SCORE_OTHERS and others_info.get("is_extreme_blowout"): s *= 0.60
                    else: s *= 0.20  
            elif dupan_true_dir == "away":
                if score_str in SHARP_REV_AWAY_BOOST: s *= 1.70
                elif goal_margin >= 0:
                    if score_str in ALL_SCORE_OTHERS and others_info.get("is_extreme_blowout"): s *= 0.60
                    else: s *= 0.20
            elif dupan_true_dir == "draw":
                if score_str in SHARP_REV_DRAW_BOOST: s *= 1.50
                elif goal_margin != 0: s *= 0.40

        elif scenario == "shootout":
            if score_str in BOOST_SHOOTOUT: s *= 1.50
            elif score_str in EXCLUDE_HIGH: s *= 0.10
            elif total_g <= 2: s *= 0.40
        elif scenario == "high_goals":
            if score_str in EXCLUDE_HIGH: s *= 0.10
            elif score_str in BOOST_HIGH: s *= 1.30
        elif scenario == "low_goals":
            if score_str in EXCLUDE_LOW: s *= 0.10
            elif score_str in BOOST_LOW: s *= 1.40
        elif scenario == "btts_strong":
            if score_str in BTTS_STRONG_EXCLUDE: s *= 0.20
            elif score_str in BTTS_STRONG_BOOST: s *= 1.30
        elif scenario == "single_side":
            if score_str in SINGLE_SIDE_EXCLUDE: s *= 0.30
            elif score_str in SINGLE_SIDE_BOOST: s *= 1.25

        if scenario == "normal":
            if exp_goals >= 2.7 and score_str in EXCLUDE_HIGH:
                strength = min(1.0, (exp_goals - 2.4) / 1.1)  
                s *= max(0.3, 1.0 - strength * 0.5)
            elif exp_goals <= 2.3 and score_str in {"3-1", "1-3", "3-2", "2-3", "3-3"}: s *= 0.5

        if s > 0: score_ratings[score_str] = round(s, 2)

    ranked = sorted(score_ratings.items(), key=lambda x: x[1], reverse=True)
    final_score = ranked[0][0] if ranked else "1-1"

    # ====================================================================
    # 🚨 绝对核心修复：【方向-比分一致性护栏】升级为强制模式 🚨
    # 彻底解决截图中的“0-1配主胜”、“2-1配平局”的UI逻辑错位BUG！
    # ====================================================================
    def _dir_of_score_strict(sc_str):
        try:
            if "胜其他" in sc_str or "9-0" in sc_str: return "home"
            if "平其他" in sc_str or "9-9" in sc_str: return "draw"
            if "负其他" in sc_str or "0-9" in sc_str: return "away"
            h, a = map(int, sc_str.split("-"))
            return "home" if h > a else ("away" if h < a else "draw")
        except: return None

    top_score_dir = _dir_of_score_strict(final_score)
    
    # 只要最终比分方向(top_score_dir)与大盘基础概率面(final_direction)不一致，
    # 绝对无视分数高低，强制向下顺延，寻找第一个顺应大盘方向的比分！
    if top_score_dir and top_score_dir != final_direction:
        aligned_score = None
        for sc, pts in ranked:
            if _dir_of_score_strict(sc) == final_direction:
                aligned_score = sc
                break
        if aligned_score:
            print(f"    🚨 [绝对护栏触发] 大盘概率面({final_direction})与得分最高比分({final_score}->{top_score_dir})严重冲突！")
            print(f"    🚨 强制废除错位比分，采用顺应大盘概率的最高分: {aligned_score}")
            final_score = aligned_score

    if final_score == "9-0": final_score = "胜其他"
    if final_score == "9-9": final_score = "平其他"
    if final_score == "0-9": final_score = "负其他"

    is_score_others_final = final_score in ALL_SCORE_OTHERS or "其他" in final_score
    if is_score_others_final:
        if final_score in SCORE_OTHERS_HOME or final_score == "胜其他": display_label = "胜其他"
        elif final_score in SCORE_OTHERS_DRAW or final_score == "平其他": display_label = "平其他"
        else: display_label = "负其他"
    else:
        display_label = final_score

    # ============ 第八层: 输出 ============
    target_crs = CRS_FULL_MAP.get(final_score, "")
    final_odds = float(match_obj.get(target_crs, 0) or 0)
    if not final_odds and is_score_others_final:
        if final_score in SCORE_OTHERS_HOME or final_score == "胜其他": final_odds = float(match_obj.get("crs_win", 0) or 0)
        elif final_score in SCORE_OTHERS_DRAW or final_score == "平其他": final_odds = float(match_obj.get("crs_same", 0) or 0)
        else: final_odds = float(match_obj.get("crs_lose", 0) or 0)

    final_prob = crs_probs.get(final_score, backup_probs.get(final_score, 5))
    ev_data = calculate_value_bet(final_prob, final_odds)

    weights = {"claude": 1.4, "gemini": 1.35, "grok": 1.30, "gpt": 1.1}
    ai_conf_sum = 0
    ai_conf_count = 0
    value_kills = 0
    for name, r in all_ai.items():
        if not isinstance(r, dict): continue
        conf = r.get("ai_confidence", 60)
        ai_conf_sum += conf * weights.get(name, 1.0)
        ai_conf_count += weights.get(name, 1.0)
        if r.get("value_kill"): value_kills += 1

    avg_ai_conf = (ai_conf_sum / ai_conf_count) if ai_conf_count > 0 else 60
    cf = min(95, engine_conf + int((avg_ai_conf - 60) * 0.4)) + value_kills * 6
    if not dir_confident: cf = max(40, cf - 10)
    if any("🚨" in str(s) for s in smart_signals): cf = max(35, cf - 12)
    risk = "低" if cf >= 75 else ("中" if cf >= 55 else "高")

    sigs = list(smart_signals)
    if cold_door["is_cold_door"]:
        sigs.extend(cold_door["signals"]); cf = max(30, cf - 5)
    if others_info["is_others"]: sigs.append(f"🔥 胜其他场({others_info['trigger_count']:.1f}条)")

    cl_raw = claude_r.get("ai_score", "") if isinstance(claude_r, dict) else ""
    cl_parsed = parse_score(cl_raw)
    cl_sc = cl_raw if cl_parsed[0] is not None else final_score

    # 无损抛出所有原始字典字段，完美向后兼容前端 UI
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
        "gpt_score": gpt_r.get("ai_score", "弃权") if ai_valid.get("gpt") else "弃权",
        "gpt_analysis": gpt_r.get("reason", "弃权") if ai_valid.get("gpt") else "弃权 (AI失效)",
        "grok_score": grok_r.get("ai_score", "弃权") if ai_valid.get("grok") else "弃权",
        "grok_analysis": grok_r.get("reason", "弃权") if ai_valid.get("grok") else "弃权 (AI失效)",
        "gemini_score": gemini_r.get("ai_score", "弃权") if ai_valid.get("gemini") else "弃权",
        "gemini_analysis": gemini_r.get("reason", "弃权") if ai_valid.get("gemini") else "弃权 (AI失效)",
        "claude_score": cl_sc if ai_valid.get("claude") else "弃权",
        "claude_analysis": claude_r.get("reason", "弃权") if ai_valid.get("claude") else "弃权 (AI失效)",
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


def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})
        s = pr.get("confidence", 0) * 0.4
        mx = max(pr.get("home_win_pct", 33), pr.get("away_win_pct", 33), pr.get("draw_pct", 33))
        s += (mx - 33) * 0.2 + pr.get("model_consensus", 0) * 2
        if pr.get("risk_level") == "低": s += 12
        elif pr.get("risk_level") == "高": s -= 5
        if pr.get("model_agreement"): s += 10
        
        exp_info = pr.get("experience_analysis", {})
        exp_score = exp_info.get("total_score", 0)
        if exp_score >= 15 and pr.get("result") == "平局" and exp_info.get("draw_rules", 0) >= 3: s += 12
        elif exp_score >= 10: s += 5
        if exp_info.get("recommendation", "").startswith("⚠️"): s -= 3

        smart_money = str(pr.get("smart_money_signal", ""))
        direction = pr.get("result", "")
        if "Sharp" in smart_money:
            if ("客胜" in smart_money and direction == "主胜") or ("主胜" in smart_money and direction == "客胜"):
                s -= 30
        if pr.get("cold_door", {}).get("is_cold_door"): s -= 8
        if pr.get("is_score_others"): s += 8
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
    print(f"  [vMAX 17.0 防崩盘版] 全量保留原架构 + 强化绝对一致性护栏 | {len(ms)} 场")
    print("=" * 80)

    match_analyses = []
    for i, m in enumerate(ms):
        eng = predict_match(m)
        league_info, _, _, _ = build_league_intelligence(m)
        sp = ensemble.predict(m, {})
        exp_result = exp_engine.analyze(m)
        match_analyses.append({
            "match": m, "engine": eng, "league_info": league_info,
            "stats": sp, "index": i+1, "experience": exp_result
        })

    all_ai = {"claude": {}, "gemini": {}, "gpt": {}, "grok": {}}
    if use_ai and match_analyses:
        print(f"  [单阶段] 启动4AI并行...")
        start_t = time.time()
        all_ai = asyncio.run(run_ai_matrix_two_phase(match_analyses))
        print(f"  [完成] 耗时 {time.time()-start_t:.1f}s")

    res = []
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        mg = merge_result(
            ma["engine"], all_ai["gpt"].get(i+1, {}), all_ai["grok"].get(i+1, {}),
            all_ai["gemini"].get(i+1, {}), all_ai["claude"].get(i+1, {}), ma["stats"], m
        )

        try: mg = apply_experience_to_prediction(m, mg, exp_engine)
        except: pass
        try: mg = apply_odds_history(m, mg)
        except: pass
        try: mg = apply_quant_edge(m, mg)
        except: pass
        try: mg = apply_wencai_intel(m, mg)
        except: pass
        try: mg = upgrade_ensemble_predict(m, mg)
        except: pass

        score_str = mg.get("predicted_score", "1-1")
        
        # UI 终极同步保险：让前端的标签完全听命于严格护栏审查后的最终比分！
        try:
            if "胜其他" in score_str: expected_dir = "主胜"
            elif "负其他" in score_str: expected_dir = "客胜"
            elif "平其他" in score_str: expected_dir = "平局"
            else:
                sh, sa = map(int, score_str.split("-"))
                expected_dir = "主胜" if sh > sa else ("客胜" if sh < sa else "平局")
            
            mg["result"] = expected_dir
            
            pl = mg.get("predicted_label", "")
            if pl and pl not in (score_str, "胜其他", "平其他", "负其他"):
                mg["predicted_label"] = score_str
        except: 
            pcts = {"主胜": mg["home_win_pct"], "平局": mg["draw_pct"], "客胜": mg["away_win_pct"]}
            mg["result"] = max(pcts, key=pcts.get)

        res.append({**m, "prediction": mg})

        cold = mg.get("cold_door", {})
        cold_tag = f" [❄️{cold.get('level','')}冷门]" if cold.get("is_cold_door") else ""
        others_tag = f" [🔥胜其他]" if mg.get("is_score_others") else ""
        sharp_tag = f" [💰Sharp]" if mg.get("sharp_detected") else ""
        print(f"  [{i+1}] {m.get('home_team')} vs {m.get('away_team')} => {mg['result']} ({mg['predicted_score']}={mg['predicted_label']}) | CF: {mg['confidence']}% | EV: {mg.get('edge_vs_market',0)}%{cold_tag}{others_tag}{sharp_tag}")

    t4 = select_top4(res)
    t4ids = [t.get("id") for t in t4]
    for r in res: r["is_recommended"] = r.get("id") in t4ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    diary = load_ai_diary()
    cold_count = len([r for r in res if r.get("prediction", {}).get("cold_door", {}).get("is_cold_door")])
    others_count = len([r for r in res if r.get("prediction", {}).get("is_score_others")])
    sharp_count = len([r for r in res if r.get("prediction", {}).get("sharp_detected")])
    diary["yesterday_win_rate"] = f"{len([r for r in res if r['prediction']['confidence']>70])}/{max(1,len(res))}"
    diary["reflection"] = f"vMAX17.0 护栏重构版 | {cold_count}冷门 {others_count}胜其他 {sharp_count}Sharp | 强制UI逻辑一致性"
    save_ai_diary(diary)

    return res, t4

if __name__ == "__main__":
    logger.info("vMAX 17.0 防崩盘重构版启动")
    print("✅ vMAX 17.0 防崩盘重构版已加载 — 幂法数学修正 + 强化一致性绝对拦截")
