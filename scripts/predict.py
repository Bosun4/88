import json
import os
import re
import time
import requests
import asyncio
import aiohttp
import numpy as np
from config import *
from models import EnsemblePredictor
from odds_engine import predict_match
from league_intel import build_league_intelligence
from experience_rules import ExperienceEngine, apply_experience_to_prediction
from advanced_models import upgrade_ensemble_predict

ensemble = EnsemblePredictor()
exp_engine = ExperienceEngine()

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
    return {"ev": round(ev * 100, 2), "kelly": round(max(0.0, kelly * 0.25) * 100, 2), "is_value": ev > 0.05}

def parse_score(s):
    try:
        p = str(s).split("-")
        return int(p[0]), int(p[1])
    except:
        return None, None

def load_ai_diary():
    diary_file = "data/ai_diary.json"
    if os.path.exists(diary_file):
        try:
            with open(diary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return None

def build_batch_prompt(match_analyses):
    p = "[ROLE] Elite Football Quant AI. Analyze matches and compute EXACT SCORE probabilities.\n"
    
    diary = load_ai_diary()
    if diary and diary.get("reflection"):
        p += f"\n[SYSTEM DIRECTIVE] Prev Win Rate: {diary.get('yesterday_win_rate', 'N/A')}. \n"
        p += f"EVOLUTION LOG: {diary['reflection']}. APPLY {diary.get('risk_adjustment', 'STRICT')} RISK FILTER.\n\n"
        
    # 核心修复 1：强制要求使用专业中文，并限制字数防止废话
    p += "[TASK] Output ONLY raw JSON array. NO markdown. The 'reason' field MUST BE in professional Chinese (中文) and strictly under 40 words.\n\n"
    
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        eng = ma["engine"]
        lg_info = ma["league_info"]
        exp = ma.get("experience", {})
        h, a = m.get("home_team", "Home"), m.get("away_team", "Away")
        
        p += f"[{i+1}] {h} vs {a} ({m.get('league', 'UNK')})\n"
        p += f"Implied Prob: H {eng.get('home_prob', 33):.1f}% | D {eng.get('draw_prob', 33):.1f}% | A {eng.get('away_prob', 34):.1f}%\n"
        p += f"Bookmaker xG: {eng.get('expected_goals', 2.5):.2f} | Gap Signal: {eng.get('scissors_gap_signal', 'None')}\n"
        p += f"Context: {lg_info}\n"

        if exp.get("triggered_count", 0) > 0:
            exp_names = ",".join([t["name"] for t in exp.get("triggered", [])[:4]])
            p += f"EXP Engine: {exp_names}\n"
            if exp.get("risk_signals"):
                p += f"ALERT: {', '.join(exp['risk_signals'][:3])}\n"
                
        p += f"SCORE CANDIDATES: {', '.join(eng.get('top3_scores', ['1-1', '0-0', '1-0']))}\n\n"

    p += "[OUTPUT STRUCTURE]\n"
    p += f"Produce EXACTLY {len(match_analyses)} JSON objects in this format:\n"
    # 核心修复 2：用中文示例引导大模型
    p += '[\n  {"match": 1, "score": "2-1", "reason": "庄家真实xG支持主队，且防冷平规则触发，看好主胜穿盘。"}\n]\n'
    return p

# ====================================================================
# 终极高可用 AI 矩阵轮询 + 异步并发引擎
# ====================================================================

FALLBACK_URLS = [
    None, 
    "https://api520.pro/v1", "https://www.api520.pro/v1",
    "https://api521.pro/v1", "https://www.api521.pro/v1",
    "https://api522.pro/v1", "https://www.api522.pro/v1", 
    "https://69.63.213.33:666/v1"
]

def get_clean_env_url(name, default=""):
    v = os.environ.get(name, globals().get(name, default))
    v = str(v).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/-]+)", v)
    return match.group(1) if match else v

def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")

async def async_call_one_ai_batch(session, prompt, url_env, key_env, models_list, num_matches, ai_name):
    """异步单路 AI 轮询"""
    key = get_clean_env_key(key_env)
    primary_url = get_clean_env_url(url_env)
    if not key: 
        return ai_name, {}, "no_key"
        
    urls = [primary_url] + [u for u in FALLBACK_URLS if u and u != primary_url]
    
    for mn in models_list:
        for base_url in urls:
            if not base_url: continue
            
            is_gem = "generateContent" in base_url
            url = base_url
            if not is_gem and "chat/completions" not in base_url:
                url = base_url.rstrip("/") + "/chat/completions"
                
            headers = {"Content-Type": "application/json"}
            if is_gem:
                headers["x-goog-api-key"] = key
                payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1}}
            else:
                headers["Authorization"] = f"Bearer {key}"
                payload = {
                    "model": mn, 
                    "messages": [
                        {"role": "system", "content": "Output ONLY valid JSON array. No markdown."},
                        {"role": "user", "content": prompt}
                    ], 
                    "temperature": 0.1
                }
                
            gw = url.split("/v1")[0] if "/v1" in url else url[:35]
            print(f"  [AI 寻优] {ai_name.upper()}: 尝试 {mn[:20]} @ {gw}")
            
            try:
                async with session.post(url, headers=headers, json=payload, timeout=45) as r:
                    if r.status == 200:
                        data = await r.json()
                        if is_gem: 
                            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        else: 
                            raw_text = data["choices"][0]["message"]["content"].strip()
                            
                        clean = re.sub(r"```\w*", "", raw_text).strip()
                        start, end = clean.find("["), clean.rfind("]")
                        results = {}
                        if start != -1 and end != -1:
                            try:
                                arr = json.loads(clean[start:end+1])
                                if isinstance(arr, list):
                                    for item in arr:
                                        if item.get("match") and item.get("score"):
                                            # 核心修复 3：解除 [:100] 的暴力截断，保留 AI 完整的句子
                                            analysis_text = str(item.get("reason", "")).strip()
                                            results[item["match"]] = {
                                                "ai_score": item["score"], 
                                                "analysis": analysis_text
                                            }
                            except: pass
                            
                        if len(results) >= max(1, num_matches * 0.4):
                            print(f"    ✅ {ai_name.upper()} 成功: {len(results)}/{num_matches} 已解析 (模型: {mn[:20]})")
                            return ai_name, results, mn
                    else:
                        print(f"    ⚠️ HTTP {r.status} - 切换线路...")
            except Exception as e:
                print(f"    ⚠️ 超时或异常 ({str(e)[:30]}) - 切换线路...")
            
            await asyncio.sleep(0.3)
            
    print(f"    ❌ {ai_name.upper()} 所有备用模型与线路均已失效！")
    return ai_name, {}, "failed"

async def run_ai_matrix(prompt, num_matches):
    ai_configs = [
        ("gpt", "GPT_API_URL", "GPT_API_KEY", [
            "熊猫-A-7-gpt-5.4",
            "熊猫-按量-gpt-5.3-codex-满血",
            "熊猫-A-10-gpt-5.3-codex",
            "熊猫-A-1-gpt-5.2",
        ]),
        ("grok", "GROK_API_URL", "GROK_API_KEY", [
            "熊猫-A-7-grok-4.2-多智能体讨论",
            "熊猫-A-6-grok-4.2-thinking",
        ]),
        ("claude", "CLAUDE_API_URL", "CLAUDE_API_KEY", [
            "熊猫-按量-顶级特供-官max-claude-opus-4.6-thinking",
            "熊猫-按量-满血copilot-claude-opus-4.6-thinking",
            "熊猫-按量-特供顶级-官方正向满血-claude-opus-4.6-thinking",
            "熊猫-按量-顶级特供-官max-claude-opus-4.6",
            "熊猫-特供-按量-Q-claude-opus-4.6",
            "熊猫-按量-特供顶级-官方正向满血-claude-sonnet-4.6-thinking",
            "熊猫-按量-满血copilot-claude-sonnet-4.6-thinking",
            "熊猫-按量-顶级特供-官max-claude-sonnet-4.6",
            "熊猫-特供-按量-Q-claude-sonnet-4.6",
        ]),
        ("gemini", "GEMINI_API_URL", "GEMINI_API_KEY", [
            "熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking",
            "熊猫-特供-X-12-gemini-3.1-pro-preview-thinking",
            "熊猫-顶级特供-X-17-gemini-3.1-pro-preview",
            "熊猫特供-按量-SSS-gemini-3.1-pro-preview",
            "熊猫-特供-X-12-gemini-3.1-pro-preview",
        ]),
    ]
    
    all_results = {"gpt": {}, "grok": {}, "claude": {}, "gemini": {}}
    tasks = []
    
    async with aiohttp.ClientSession() as session:
        for ai_name, url_env, key_env, models in ai_configs:
            tasks.append(async_call_one_ai_batch(session, prompt, url_env, key_env, models, num_matches, ai_name))
            
        results = await asyncio.gather(*tasks)
        
    for ai_name, parsed_data, model_used in results:
        all_results[ai_name] = parsed_data
        
    return all_results

def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    sp_h = float(match_obj.get("sp_home", 0) or 0)
    sp_d = float(match_obj.get("sp_draw", 0) or 0)
    sp_a = float(match_obj.get("sp_away", 0) or 0)
    mc = stats.get("model_consensus", 0)
    tm = stats.get("total_models", 11)
    smart = stats.get("smart_signals", [])
    extreme = stats.get("extreme_warning", "")

    engine_score = engine_result.get("primary_score", "1-1")
    engine_conf = engine_result.get("confidence", 50)
    candidates = engine_result.get("top3_scores", [engine_score])
    o25 = engine_result.get("over_25", 50)
    bt = engine_result.get("btts", 45)

    ai_all = {"gpt": gpt_r, "grok": grok_r, "gemini": gemini_r, "claude": claude_r}
    ai_scores = []
    for name, r in ai_all.items():
        sc = r.get("ai_score", "-") if isinstance(r, dict) else "-"
        if sc and sc not in ["-", "?", ""]:
            ai_scores.append(sc)

    vote_count = {}
    for sc in ai_scores:
        if sc in candidates:
            vote_count[sc] = vote_count.get(sc, 0) + 2
        else:
            vote_count[sc] = vote_count.get(sc, 0) + 1

    final_score = engine_score
    if vote_count:
        best_voted = max(vote_count, key=vote_count.get)
        if best_voted in candidates and vote_count[best_voted] >= 3:
            final_score = best_voted

    hp = engine_result.get("home_prob", 33)
    dp = engine_result.get("draw_prob", 33)
    ap = engine_result.get("away_prob", 34)
    shp = stats.get("home_win_pct", 33)
    sdp = stats.get("draw_pct", 33)
    sap = stats.get("away_win_pct", 34)
    fhp = hp * 0.75 + shp * 0.25
    fdp = dp * 0.75 + sdp * 0.25
    fap = ap * 0.75 + sap * 0.25
    fhp = max(3, fhp); fdp = max(3, fdp); fap = max(3, fap)
    ft = fhp + fdp + fap
    if ft > 0:
        fhp = round(fhp/ft*100, 1); fdp = round(fdp/ft*100, 1); fap = round(max(3, 100-fhp-fdp), 1)
        ft2 = fhp + fdp + fap
        if abs(ft2 - 100) > 0.5:
            fhp = round(fhp/ft2*100, 1); fdp = round(fdp/ft2*100, 1); fap = round(100-fhp-fdp, 1)

    cf = engine_conf
    agree_count = sum(1 for s in ai_scores if s == engine_score)
    cf = min(90, cf + agree_count * 4)
    has_warn = any("🚨" in str(s) for s in smart)
    if has_warn: cf = max(35, cf - 10)
    risk = "低" if cf >= 70 else ("中" if cf >= 50 else "高")

    vh = calculate_value_bet(fhp, sp_h)
    vd = calculate_value_bet(fdp, sp_d)
    va = calculate_value_bet(fap, sp_a)
    vt = []
    for k, v in zip(["主胜","平局","客胜"], [vh,vd,va]):
        if v and v.get("is_value"): vt.append("%s EV:+%s%% Kelly:%s%%" % (k, v["ev"], v["kelly"]))

    pcts = {"主胜": fhp, "平局": fdp, "客胜": fap}
    result = max(pcts, key=pcts.get)
    seen = set()
    us = [s for s in smart if s not in seen and not seen.add(s)]

    gpt_sc = gpt_r.get("ai_score", "-") if isinstance(gpt_r, dict) else "-"
    gpt_an = gpt_r.get("analysis", "N/A") if isinstance(gpt_r, dict) else "N/A"
    grok_sc = grok_r.get("ai_score", "-") if isinstance(grok_r, dict) else "-"
    grok_an = grok_r.get("analysis", "N/A") if isinstance(grok_r, dict) else "N/A"
    gem_sc = gemini_r.get("ai_score", "-") if isinstance(gemini_r, dict) else "-"
    gem_an = gemini_r.get("analysis", "N/A") if isinstance(gemini_r, dict) else "N/A"
    cl_sc = claude_r.get("ai_score", "-") if isinstance(claude_r, dict) else "-"
    cl_an = claude_r.get("analysis", "N/A") if isinstance(claude_r, dict) else "N/A"
    if cl_sc == "-":
        cl_sc = engine_score
        cl_an = engine_result.get("reason", "odds engine")

    return {
        "predicted_score": final_score,
        "home_win_pct": fhp, "draw_pct": fdp, "away_win_pct": fap,
        "confidence": cf, "result": result, "risk_level": risk,
        "over_under_2_5": "大" if o25 > 55 else "小",
        "both_score": "是" if bt > 50 else "否",
        "gpt_score": gpt_sc, "gpt_analysis": gpt_an,
        "grok_score": grok_sc, "grok_analysis": grok_an,
        "gemini_score": gem_sc, "gemini_analysis": gem_an,
        "claude_score": cl_sc, "claude_analysis": cl_an,
        "model_agreement": len(set(ai_scores)) <= 1 and len(ai_scores) >= 2,
        "candidate_scores": {sc: round(1.0 if sc == final_score else 0.5, 2) for sc in candidates},
        "poisson": stats.get("poisson", {}),
        "refined_poisson": stats.get("refined_poisson", {}),
        "value_bets_summary": vt,
        "extreme_warning": engine_result.get("scissors_gap_signal", extreme if extreme else "无"),
        "smart_money_signal": " | ".join(us) if us else "正常",
        "smart_signals": us, "model_consensus": mc, "total_models": tm,
        "expected_total_goals": engine_result.get("expected_goals", 2.5),
        "over_2_5": o25, "btts": bt,
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
        "bivariate_poisson": stats.get("bivariate_poisson", {}),
        "asian_handicap_probs": stats.get("asian_handicap_probs", {}),
        "bookmaker_implied_home_xg": engine_result.get("bookmaker_implied_home_xg", "?"),
        "bookmaker_implied_away_xg": engine_result.get("bookmaker_implied_away_xg", "?")
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
        if pr.get("value_bets_summary"): s += 8

        exp_info = pr.get("experience_analysis", {})
        exp_score = exp_info.get("total_score", 0)
        exp_draw_rules = exp_info.get("draw_rules", 0)
        
        if exp_score >= 15 and pr.get("result") == "平局" and exp_draw_rules >= 3:
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
                
        p["recommend_score"] = round(s, 2)
        
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]

def extract_num(ms):
    wm = {"一":1000,"二":2000,"三":3000,"四":4000,"五":5000,"六":6000,"日":7000,"天":7000}
    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999

def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 60)
    print(f"  [QUANT ENGINE vMAX] Executing Batch | {len(ms)} Data Points")
    print("=" * 60)

    match_analyses = []
    for i, m in enumerate(ms):
        eng = predict_match(m)
        league_info, _, _, _ = build_league_intelligence(m)
        sp = ensemble.predict(m, {})
        exp_result = exp_engine.analyze(m)
        match_analyses.append({
            "match": m, "engine": eng, "league_info": league_info,
            "stats": sp, "index": i, "experience": exp_result,
        })

    all_ai = {"claude": {}, "gemini": {}, "gpt": {}, "grok": {}}
    if use_ai and match_analyses:
        prompt = build_batch_prompt(match_analyses)
        print(f"  [INFO] Prompt Built: {len(prompt)} chars. Calling AI Matrix with Fallbacks...")
        all_ai = asyncio.run(run_ai_matrix(prompt, len(match_analyses)))

    res = []
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        idx = i + 1
        
        mg = merge_result(
            ma["engine"], 
            all_ai["gpt"].get(idx, {}), 
            all_ai["grok"].get(idx, {}), 
            all_ai["gemini"].get(idx, {}), 
            all_ai["claude"].get(idx, {}), 
            ma["stats"], m
        )
        
        mg = apply_experience_to_prediction(m, mg, exp_engine)
        mg = upgrade_ensemble_predict(m, mg)
        
        res.append({**m, "prediction": mg})
        print(f"  [{idx}] {m.get('home_team')} vs {m.get('away_team')} => {mg['result']} ({mg['predicted_score']}) | CF: {mg['confidence']}%")

    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res: r["is_recommended"] = r["id"] in t4ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    return res, t4


