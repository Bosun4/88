import json
import requests
import time
import re
import os
import numpy as np
from collections import Counter
from config import *
from models import EnsemblePredictor

ensemble = EnsemblePredictor()

# ==========================================
# 辅助与清洗函数
# ==========================================
def detect_league_context(m):
    hr = int(m.get("home_rank", 10) or 10)
    ar = int(m.get("away_rank", 10) or 10)
    baseface = str(m.get("baseface", ""))
    ctx = []
    if hr <= 3 or ar <= 3:
        ctx.append("TITLE RACE: top-3 team involved, high motivation")
    elif hr >= 16 or ar >= 16:
        ctx.append("RELEGATION BATTLE: bottom-5 team fighting for survival, expect desperate play")
    if abs(hr - ar) <= 3 and hr <= 10 and ar <= 10:
        ctx.append("MID-TABLE DERBY: similar strength, draw probability higher than normal")
    if "保级" in baseface or "降级" in baseface:
        ctx.append("RELEGATION CONTEXT detected in analysis")
    if "争冠" in baseface or "第一" in baseface:
        ctx.append("TITLE CONTEXT detected in analysis")
    return ctx

def extract_clean_json(text):
    text = str(text or "").strip()
    fallback = {
        "home_attack_mod": 1.0, "home_defense_mod": 1.0,
        "away_attack_mod": 1.0, "away_defense_mod": 1.0,
        "pace_factor": 1.0, "ai_score": "?", "intelligence_summary": "format error"
    }
    
    # 尝试直接解析
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            parsed = json.loads(text[start:end+1])
            if "ai_score" in parsed:
                return parsed
        except Exception:
            pass
            
    # 正则硬解后备方案
    s_match = re.search(r'"ai_score"\s*:\s*"([^"]+)"', text)
    if s_match: fallback["ai_score"] = s_match.group(1)
    
    a_match = re.search(r'"intelligence_summary"\s*:\s*"(.*?)"', text, re.DOTALL)
    if a_match: fallback["intelligence_summary"] = a_match.group(1).replace('"', "'").replace("\n", " ").strip()[:150]
    
    for key in ["home_attack_mod", "home_defense_mod", "away_attack_mod", "away_defense_mod", "pace_factor"]:
        num_match = re.search(r'"' + key + r'"\s*:\s*([0-9.]+)', text)
        if num_match: fallback[key] = float(num_match.group(1))
        
    if fallback["ai_score"] != "?":
        return fallback
    return None

def get_clean_env_url(name, default=""):
    v = os.environ.get(name, globals().get(name, default))
    v = str(v).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/-]+)", v)
    return match.group(1) if match else v

def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")

# ==========================================
# AI API 路由与调用
# ==========================================
FALLBACK_URLS = [
    None,
    "https://api520.pro/v1",
    "https://www.api520.pro/v1",
    "https://api521.pro/v1",
    "https://www.api521.pro/v1",
    "https://api522.pro/v1",
    "https://www.api522.pro/v1",
    "https://69.63.213.33:666/v1",
]

def call_ai_model(prompt, url, key, model_name):
    if not url or not key: return {}
    is_native_gemini = "generateContent" in url
    if not is_native_gemini and "chat/completions" not in url:
        url = url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if is_native_gemini:
        headers["x-goog-api-key"] = key
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}
    else:
        headers["Authorization"] = "Bearer " + key
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "Pure JSON output only. No markdown."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 500
        }
    gw = url.split("/v1")[0] if "/v1" in url else url[:35]
    print("    > %s @ %s" % (model_name, gw))
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            if is_native_gemini:
                t = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                t = r.json()["choices"][0]["message"]["content"].strip()
            parsed = extract_clean_json(t)
            if parsed:
                print("    OK %s: %s (H_Atk: %s)" % (model_name, parsed.get("ai_score", "?"), parsed.get("home_attack_mod", 1.0)))
                return parsed
            else:
                print("    WARN %s: parse fail" % model_name)
        else:
            print("    ERR %s: HTTP %d" % (model_name, r.status_code))
    except Exception as e:
        print("    ERR %s: %s" % (model_name, str(e)[:50]))
    return {}

def call_with_fallback(prompt, url_env, key_env, models_list):
    key = get_clean_env_key(key_env)
    primary_url = get_clean_env_url(url_env, globals().get(url_env, ""))
    if not key:
        print("    X %s: no key" % key_env)
        return {}
    urls = [primary_url] + [u for u in FALLBACK_URLS if u and u != primary_url]
    for model_name in models_list:
        for url in urls:
            if not url: continue
            result = call_ai_model(prompt, url, key, model_name)
            if result and result.get("ai_score") and result["ai_score"] not in ["?", ""]:
                return result
            time.sleep(0.2)
    print("    X all models exhausted")
    return {}

def call_gpt(prompt):
    return call_with_fallback(prompt, "GPT_API_URL", "GPT_API_KEY", [
        "熊猫-A-7-gpt-5.4", "熊猫-按量-gpt-5.3-codex-满血", "熊猫-A-1-gpt-5.2",
    ])

def call_grok(prompt):
    return call_with_fallback(prompt, "GROK_API_URL", "GROK_API_KEY", [
        "熊猫-A-7-grok-4.2-多智能体讨论", "熊猫-A-4-grok-4.2-fast",
    ])

def call_gemini(prompt):
    return call_with_fallback(prompt, "GEMINI_API_URL", "GEMINI_API_KEY", [
        "熊猫特供S-按量-gemini-3-flash-preview", "熊猫特供-按量-SSS-gemini-3.1-pro-preview",
    ])

def call_claude(prompt):
    return call_with_fallback(prompt, "CLAUDE_API_URL", "CLAUDE_API_KEY", [
        "熊猫-按量-顶级特供-官max-claude-opus-4.6", "熊猫-按量-特供顶级-官方正向满血-claude-opus-4.6",
    ])

# ==========================================
# 核心 V6 引擎：Prompt、基础 xG 与 蒙特卡洛
# ==========================================
def build_intelligence_prompt(m):
    h, a = m.get("home_team", "?"), m.get("away_team", "?")
    lg = m.get("league", "?")
    intel = m.get("intelligence", {})
    hs = m.get("home_stats", {})
    ast = m.get("away_stats", {})
    
    p = "[SYSTEM] Supreme Football Intelligence Agent.\n"
    p += "[TASK] Analyze match context, output quantitative modifiers, AND predict EXACT SCORE for [%s] vs [%s].\n" % (h, a)
    p += "[CRITICAL RULES]\n"
    p += "- Focus ONLY on injuries, form, motivation, and tactical matchups. Ignore betting odds.\n"
    p += "- Modifiers default to 1.0. Range is 0.70 (severely weakened) to 1.30 (massively boosted).\n"
    p += "- Pure JSON output only. No markdown.\n\n"
    
    p += "[MATCH] %s | %s\n" % (lg, m.get("match_num", "?"))
    league_ctx = detect_league_context(m)
    if league_ctx:
        p += "[LEAGUE CONTEXT]\n"
        for c in league_ctx: p += "- %s\n" % c
        
    p += "\n[HOME %s] Rank#%s\n" % (h, m.get("home_rank", "?"))
    p += "Form: %s | Injuries: %s\n" % (hs.get("form", "?"), intel.get("h_inj", "None"))
    p += "Avg GF: %s | Avg GA: %s\n" % (hs.get("avg_goals_for", "1.2"), hs.get("avg_goals_against", "1.2"))
    
    p += "\n[AWAY %s] Rank#%s\n" % (a, m.get("away_rank", "?"))
    p += "Form: %s | Injuries: %s\n" % (ast.get("form", "?"), intel.get("g_inj", "None"))
    p += "Avg GF: %s | Avg GA: %s\n" % (ast.get("avg_goals_for", "1.2"), ast.get("avg_goals_against", "1.2"))
    
    baseface = str(m.get("baseface", "")).strip()
    if baseface:
        p += "\n[EXPERT ANALYSIS]\n%s\n" % baseface[:350]
        
    p += "\n[OUTPUT FORMAT] EXACTLY like this:\n"
    p += '{\n'
    p += '  "home_attack_mod": 1.05,\n'
    p += '  "home_defense_mod": 0.95,\n'
    p += '  "away_attack_mod": 0.90,\n'
    p += '  "away_defense_mod": 1.10,\n'
    p += '  "pace_factor": 1.05,\n'
    p += '  "ai_score": "2-1",\n'
    p += '  "intelligence_summary": "Short explanation of the key factors."\n'
    p += '}'
    return p

def calculate_base_xg(stats):
    home_scored = float(stats.get("home_form", {}).get("goals_scored_avg", 1.2))
    home_conceded = float(stats.get("home_form", {}).get("goals_conceded_avg", 1.0))
    away_scored = float(stats.get("away_form", {}).get("goals_scored_avg", 1.0))
    away_conceded = float(stats.get("away_form", {}).get("goals_conceded_avg", 1.3))
    
    if home_scored == 0: home_scored = 1.2
    if home_conceded == 0: home_conceded = 1.0
    if away_scored == 0: away_scored = 1.0
    if away_conceded == 0: away_conceded = 1.3

    base_home_xg = ((home_scored + away_conceded) / 2.0) * 1.08 
    base_away_xg = (away_scored + home_conceded) / 2.0
    return base_home_xg, base_away_xg

def run_monte_carlo_simulation(base_home_xg, base_away_xg, avg_modifiers, simulations=10000):
    final_home_xg = base_home_xg * avg_modifiers.get("home_attack_mod", 1.0) * avg_modifiers.get("away_defense_mod", 1.0)
    final_away_xg = base_away_xg * avg_modifiers.get("away_attack_mod", 1.0) * avg_modifiers.get("home_defense_mod", 1.0)
    
    pace = avg_modifiers.get("pace_factor", 1.0)
    final_home_xg = max(0.1, final_home_xg * pace)
    final_away_xg = max(0.1, final_away_xg * pace)
    
    home_goals_sim = np.random.poisson(lam=final_home_xg, size=simulations)
    away_goals_sim = np.random.poisson(lam=final_away_xg, size=simulations)
    
    home_wins = np.sum(home_goals_sim > away_goals_sim)
    draws = np.sum(home_goals_sim == away_goals_sim)
    away_wins = np.sum(home_goals_sim < away_goals_sim)
    
    scores = [f"{h}-{a}" for h, a in zip(home_goals_sim, away_goals_sim)]
    score_counts = Counter(scores)
    top_scores = score_counts.most_common(5)
    
    return {
        "final_home_xg": round(final_home_xg, 2),
        "final_away_xg": round(final_away_xg, 2),
        "prob_home_win": round((home_wins / simulations) * 100, 1),
        "prob_draw": round((draws / simulations) * 100, 1),
        "prob_away_win": round((away_wins / simulations) * 100, 1),
        "top_predicted_score": top_scores[0][0],
        "top_scores_distribution": {score: round((count/simulations)*100, 1) for score, count in top_scores}
    }

# ==========================================
# 融合引擎 (替代原本的 merge_all)
# ==========================================
def aggregate_and_merge(gpt, grok, gemini, claude, stats, match_obj):
    ai_results = [res for res in [gpt, grok, gemini, claude] if isinstance(res, dict) and "ai_score" in res]
    
    if not ai_results:
        # 兜底：如果 API 全挂，用原系统的旧逻辑值
        return {"predicted_score": stats.get("predicted_score", "1-1"), "confidence": 50, "result": "未知", "risk_level": "高"}

    # 1. 提取所有 AI 吐出的修正因子并取均值
    avg_mods = {
        "home_attack_mod": np.mean([r.get("home_attack_mod", 1.0) for r in ai_results]),
        "home_defense_mod": np.mean([r.get("home_defense_mod", 1.0) for r in ai_results]),
        "away_attack_mod": np.mean([r.get("away_attack_mod", 1.0) for r in ai_results]),
        "away_defense_mod": np.mean([r.get("away_defense_mod", 1.0) for r in ai_results]),
        "pace_factor": np.mean([r.get("pace_factor", 1.0) for r in ai_results]),
    }
    
    # 2. 统计 AI 直觉比分 (多数派原则)
    ai_scores = [r.get("ai_score") for r in ai_results]
    most_common_ai_score = Counter(ai_scores).most_common(1)[0][0]
    
    # 3. 计算基础 xG 并跑蒙特卡洛
    b_home_xg, b_away_xg = calculate_base_xg(stats)
    sim_result = run_monte_carlo_simulation(b_home_xg, b_away_xg, avg_mods)
    
    math_score = sim_result["top_predicted_score"]
    prob_h, prob_d, prob_a = sim_result["prob_home_win"], sim_result["prob_draw"], sim_result["prob_away_win"]
    
    # 4. 双擎对撞逻辑
    ai_h, ai_a = map(int, most_common_ai_score.split('-'))
    math_h, math_a = map(int, math_score.split('-'))
    ai_trend = "主胜" if ai_h > ai_a else ("客胜" if ai_a > ai_h else "平局")
    math_trend = "主胜" if math_h > math_a else ("客胜" if math_a > math_h else "平局")
    
    final_score = math_score if ai_trend != math_trend else most_common_ai_score
    result_trend = "主胜" if prob_h > prob_a and prob_h > prob_d else ("客胜" if prob_a > prob_h and prob_a > prob_d else "平局")
    
    # 动态置信度计算
    sys_cf = int(max(prob_h, prob_d, prob_a))
    if most_common_ai_score == math_score: sys_cf = min(98, sys_cf + 15)  # 完美共振加分
    elif ai_trend == math_trend: sys_cf = min(90, sys_cf + 8)
    else: sys_cf = max(40, sys_cf - 15)  # 发生分歧减分
    
    risk_str = "低" if sys_cf >= 70 else ("中" if sys_cf >= 55 else "高")
    smart = stats.get("smart_signals", [])
    extreme = stats.get("extreme_warning", "")
    
    # 完美适配你原有的字典输出结构，防止前端崩溃
    return {
        "predicted_score": final_score,
        "home_win_pct": prob_h,
        "draw_pct": prob_d,
        "away_win_pct": prob_a,
        "confidence": sys_cf,
        "result": result_trend,
        "risk_level": risk_str,
        "over_under_2_5": "大" if (sim_result["final_home_xg"] + sim_result["final_away_xg"]) > 2.5 else "小",
        "both_score": "是" if sim_result["final_home_xg"] > 0.8 and sim_result["final_away_xg"] > 0.8 else "否",
        
        "gpt_score": gpt.get("ai_score", "-"),
        "gpt_analysis": gpt.get("intelligence_summary", "N/A"),
        "grok_score": grok.get("ai_score", "-"),
        "grok_analysis": grok.get("intelligence_summary", "N/A"),
        "gemini_score": gemini.get("ai_score", "-"),
        "gemini_analysis": gemini.get("intelligence_summary", "N/A"),
        "claude_score": claude.get("ai_score", "-"),
        "claude_analysis": claude.get("intelligence_summary", "N/A"),
        
        "model_agreement": most_common_ai_score == math_score,
        "poisson": {"home_xg": sim_result["final_home_xg"], "away_xg": sim_result["final_away_xg"]},
        "expected_total_goals": round(sim_result["final_home_xg"] + sim_result["final_away_xg"], 2),
        "extreme_warning": extreme if extreme else "无",
        "smart_signals": smart,
        
        # 塞入你的历史遗留结构
        "elo": stats.get("elo", {}), "random_forest": stats.get("random_forest", {}),
        "top_scores": [{"score": k, "prob": v} for k, v in sim_result["top_scores_distribution"].items()]
    }

def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})
        s = pr.get("confidence", 0) * 0.4
        if pr.get("model_agreement"): s += 20
        if pr.get("risk_level") == "低": s += 15
        elif pr.get("risk_level") == "高": s -= 10
        smart = pr.get("smart_signals", [])
        if any("🚨" in str(sig) for sig in smart): s += 8
        p["recommend_score"] = round(s, 2)
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]

def extract_num(match_str):
    week_map = {"一": 1000, "二": 2000, "三": 3000, "四": 4000, "五": 5000, "六": 6000, "日": 7000, "天": 7000}
    base = next((v for k, v in week_map.items() if k in str(match_str)), 0)
    nums = re.findall(r"\d+", str(match_str))
    return base + int(nums[0]) if nums else 9999

# ==========================================
# 主运行入口
# ==========================================
def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 60)
    print("  ENGINE v6.0 (双擎完全体) | %d matches" % len(ms))
    print("  AI 修正因子 + 蒙特卡洛万次推演 | 纯享基本面")
    print("=" * 60)
    res = []
    
    for i, m in enumerate(ms):
        h, a = m.get("home_team", "?"), m.get("away_team", "?")
        print("\n" + "-" * 50)
        print("  [%d/%d] %s %s vs %s" % (i+1, len(ms), m.get("league",""), h, a))
        print("-" * 50)
        
        print("  Phase-0: 提取统计基底...")
        sp = ensemble.predict(m, {}) 
        
        if use_ai:
            print("  Phase-1: AI 情报特工出动 (提取战力系数)...")
            prompt = build_intelligence_prompt(m)
            gpt_res = call_gpt(prompt)
            time.sleep(0.3)
            grok_res = call_grok(prompt)
            time.sleep(0.3)
            gemini_res = call_gemini(prompt)
            time.sleep(0.3)
            claude_res = call_claude(prompt)
            time.sleep(0.3)
        else:
            blocked = {"ai_score": "-", "intelligence_summary": "AI blocked"}
            gpt_res = grok_res = gemini_res = claude_res = blocked
            
        print("  Phase-2: 启动万次对撞推演...")
        mg = aggregate_and_merge(gpt_res or {}, grok_res or {}, gemini_res or {}, claude_res or {}, sp, m)
        
        print("  => 【最终结论】: %s (%s) 置信度: %d%%" % (mg["result"], mg["predicted_score"], mg["confidence"]))
        print("  => 【核心动态】: xG主 %.2f | xG客 %.2f" % (mg["poisson"]["home_xg"], mg["poisson"]["away_xg"]))
        res.append({**m, "prediction": mg})
        
    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res:
        r["is_recommended"] = r["id"] in t4ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    
    print("\n" + "=" * 60)
    print("  🏆 核心 TOP4 推荐:")
    for i, t in enumerate(t4):
        pr = t.get("prediction", {})
        print("    %d. %s vs %s => %s (%s) %d%%" % (
            i+1, t.get("home_team"), t.get("away_team"),
            pr.get("result"), pr.get("predicted_score"), pr.get("confidence",0)))
    print("=" * 60)
    
    return res, t4
