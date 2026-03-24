import json
import os
import re
import time
import requests
import asyncio
import aiohttp
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
    p = "[ROLE] Pro Football Quant Analyst. Analyze matches and pick the EXACT SCORE.\n"
    
    diary = load_ai_diary()
    if diary and diary.get("reflection"):
        p += f"\n[CRITICAL SYSTEM MEMORY] Yesterday's Win Rate: {diary.get('yesterday_win_rate', 'N/A')}\n"
        p += f"LESSON LEARNED: {diary['reflection']}\n"
        p += f"ADJUSTMENT STRATEGY TODAY: {diary.get('risk_adjustment', 'Standard')}\n"
        p += "APPLY THIS STRATEGY to today's predictions.\n\n"
        
    p += "[TASK] You MUST pick from the provided CANDIDATES list. Focus on Data, EXP and Risk Signals.\n"
    p += "[FORMAT] Output ONLY a raw JSON array. NO markdown, NO extra text.\n\n"
    
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        eng = ma["engine"]
        lg_info = ma["league_info"]
        exp = ma.get("experience", {})
        h, a = m.get("home_team", "?"), m.get("away_team", "?")
        
        p += f"[{i+1}] {h} vs {a} ({m.get('league', '?')})\n"
        p += f"Base Probs: H:{eng.get('home_prob', 33):.1f}% D:{eng.get('draw_prob', 33):.1f}% A:{eng.get('away_prob', 34):.1f}% | xG: {eng.get('expected_goals', 2.5):.1f}\n"
        p += f"League Context: {lg_info}\n"

        if exp.get("triggered_count", 0) > 0:
            exp_names = ",".join([t["name"] for t in exp.get("triggered", [])[:3]])
            p += f"EXP Rules: {exp_names}\n"
            if exp.get("risk_signals"):
                p += f"RISK SIGNALS: {', '.join(exp['risk_signals'][:3])}\n"
                
        p += f"CANDIDATES: {', '.join(eng.get('top3_scores', ['1-1', '0-0', '1-0']))}\n\n"

    p += "[OUTPUT STRUCTURE]\n"
    p += f"Produce exactly {len(match_analyses)} objects in this exact format:\n"
    p += '[\n  {"match": 1, "score": "2-1", "reason": "Home adv and xG favor home"}\n]\n'
    return p

async def async_fetch_ai(session, model_name, api_url, api_key, payload, timeout=25):
    headers = {"Content-Type": "application/json"}
    is_gemini = "generateContent" in api_url
    if is_gemini:
        headers["x-goog-api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
        
    try:
        async with session.post(api_url, headers=headers, json=payload, timeout=timeout) as resp:
            if resp.status == 200:
                data = await resp.json()
                if is_gemini:
                    return model_name, data["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    return model_name, data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [WARN] {model_name} failed: {str(e)[:40]}")
    return model_name, None

async def run_ai_matrix(prompt, num_matches):
    payloads = {
        "claude": {"model": CLAUDE_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
        "gemini": {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}},
        "gpt": {"model": GPT_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
        "grok": {"model": GROK_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    }
    
    tasks = []
    async with aiohttp.ClientSession() as session:
        if CLAUDE_API_KEY: tasks.append(async_fetch_ai(session, "claude", CLAUDE_API_URL, CLAUDE_API_KEY, payloads["claude"]))
        if GEMINI_API_KEY: tasks.append(async_fetch_ai(session, "gemini", GEMINI_API_URL, GEMINI_API_KEY, payloads["gemini"]))
        if GPT_API_KEY: tasks.append(async_fetch_ai(session, "gpt", GPT_API_URL, GPT_API_KEY, payloads["gpt"]))
        if GROK_API_KEY: tasks.append(async_fetch_ai(session, "grok", GROK_API_URL, GROK_API_KEY, payloads["grok"]))
        
        results = await asyncio.gather(*tasks)
        
    parsed_results = {"claude": {}, "gemini": {}, "gpt": {}, "grok": {}}
    for name, content in results:
        if content:
            clean = re.sub(r"```\w*", "", content).strip()
            start, end = clean.find("["), clean.rfind("]")
            if start != -1 and end != -1:
                try:
                    arr = json.loads(clean[start:end+1])
                    for item in arr:
                        if item.get("match") and item.get("score"):
                            parsed_results[name][item["match"]] = {"ai_score": item["score"], "analysis": str(item.get("reason", ""))[:100]}
                except: pass
    return parsed_results

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
    has_warn = any("\U0001f6a8" in str(s) for s in smart)
    if has_warn: cf = max(35, cf - 10)
    risk = "低" if cf >= 70 else ("中" if cf >= 50 else "高")

    vh = calculate_value_bet(fhp, sp_h)
    vd = calculate_value_bet(fdp, sp_d)
    va = calculate_value_bet(fap, sp_a)
    vt = []
    for k, v in zip(["主胜","平局","客胜"], [vh,vd,va]):
        if v and v.get("is_value"): vt.append(f"{k} EV:+{v['ev']}% Kelly:{v['kelly']}%")

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
        "extreme_warning": extreme if extreme else "无",
        "smart_money_signal": " | ".join(us) if us else "正常",
        "smart_signals": us, "model_consensus": mc, "total_models": tm,
        "expected_total_goals": engine_result.get("expected_goals", 2.5),
        "over_2_5": o25, "btts": bt,
        "top_scores": stats.get("refined_poisson", {}).get("top_scores", []),
        "elo": stats.get("elo", {}), "random_forest": stats.get("random_forest", {}),
        "gradient_boost": stats.get("gradient_boost", {}), "neural_net": stats.get("neural_net", {}),
        "logistic": stats.get("logistic", {}), "svm": stats.get("svm", {}), "knn": stats.get("knn", {}),
        "dixon_coles": stats.get("dixon_coles", {}), "bradley_terry": stats.get("bradley_terry", {}),
        "home_form": stats.get("home_form", {}), "away_form": stats.get("away_form", {}),
        "handicap_signal": stats.get("handicap_signal", ""),
        "odds_movement": stats.get("odds_movement", {}), "vote_analysis": stats.get("vote_analysis", {}),
        "h2h_blood": stats.get("h2h_blood", {}), "crs_analysis": stats.get("crs_analysis", {}),
        "ttg_analysis": stats.get("ttg_analysis", {}), "halftime": stats.get("halftime", {}),
        "pace_rating": stats.get("pace_rating", ""),
        "kelly_home": stats.get("kelly_home", {}), "kelly_away": stats.get("kelly_away", {}),
        "odds": stats.get("odds", {}),
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
    print(f"  ENGINE v6.0 BATCH | {len(ms)} matches | Async AI + Self-Learn")
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

    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res: r["is_recommended"] = r["id"] in t4ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    return res, t4
