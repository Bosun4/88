import json, requests, time, re, os
from config import *
from models import EnsemblePredictor
from odds_engine import predict_match, build_ai_context
from league_intel import build_league_intelligence
from experience_rules import ExperienceEngine, apply_experience_to_prediction

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


# ====================================================================
# BATCH AI: One prompt for ALL matches
# ====================================================================

def build_batch_prompt(match_analyses):
    p = "[ROLE] Football analyst. For each match below, pick the best score from the candidates.\n"
    p += "[RULES] You MUST pick from CANDIDATES only. Output JSON array.\n\n"
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        eng = ma["engine"]
        league_info = ma["league_info"]
        exp_result = ma.get("experience", {})
        h, a = m.get("home_team", "?"), m.get("away_team", "?")
        lg = m.get("league", "?")
        sp_h, sp_d, sp_a = m.get("sp_home", 0), m.get("sp_draw", 0), m.get("sp_away", 0)
        top3 = eng.get("top3_scores", ["1-1"])
        intel = m.get("intelligence", {})
        p += "— MATCH %d: %s vs %s (%s) —\n" % (i+1, h, a, lg)
        p += "Odds: H=%.2f D=%.2f A=%.2f | Dir:%s(%s) | ExpGoals:%.1f\n" % (
            sp_h, sp_d, sp_a, eng["direction"], eng["direction_confidence"], eng["expected_goals"])
        p += "O2.5:%.0f%% BTTS:%.0f%% | 0g@%.1f\n" % (eng["over_25"], eng["btts"], eng.get("zero_odds", 99))
        p += "%s\n" % league_info

        # === 经验规则注入 AI prompt ===
        if exp_result.get("triggered_count", 0) > 0:
            exp_lines = []
            for t in exp_result.get("triggered", [])[:5]:
                exp_lines.append("%s(w%d)" % (t["name"], t["weight"]))
            p += "[EXP] %s | draw_boost:%.1f | rec:%s\n" % (
                ", ".join(exp_lines), exp_result.get("draw_boost", 0), exp_result.get("recommendation", ""))
            if exp_result.get("risk_signals"):
                p += "[RISK] %s\n" % " | ".join(exp_result["risk_signals"][:3])

        baseface = str(m.get("baseface", "")).strip()
        if baseface:
            p += "%s\n" % baseface[:250]
        intro = str(m.get("expert_intro", "")).strip()
        if intro:
            p += "Expert: %s\n" % intro[:150]
        h_bad = str(intel.get("home_bad_news", "")).strip()
        g_bad = str(intel.get("guest_bad_news", "")).strip()
        if h_bad: p += "H-: %s\n" % h_bad[:150]
        if g_bad: p += "A-: %s\n" % g_bad[:150]
        inj_h = str(intel.get("h_inj", intel.get("home_injury", "")))[:80]
        inj_a = str(intel.get("g_inj", intel.get("guest_injury", "")))[:80]
        if inj_h: p += "Inj H: %s\n" % inj_h
        if inj_a: p += "Inj A: %s\n" % inj_a
        had = m.get("had_analyse", [])
        if had: p += "Official: %s\n" % ",".join(str(x) for x in had)
        top5 = eng.get("top5_detail", [])
        if top5:
            p += "TOP5: %s\n" % ", ".join(["%s(dev%+.0f%%)" % (s[0], s[3]) for s in top5])
        p += "CANDIDATES: %s\n" % ", ".join(top3)
        p += "\n"
    p += "[OUTPUT] Return ONLY a JSON array with %d objects:\n" % len(match_analyses)
    p += '[{"match":1,"score":"X-X","reason":"30chars"},{"match":2,"score":"X-X","reason":"30chars"},…]\n'
    return p


def parse_batch_response(text, num_matches):
    text = str(text or "").strip()
    text = re.sub(r"```\w*", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            arr = json.loads(text[start:end+1])
            if isinstance(arr, list):
                results = {}
                for item in arr:
                    idx = item.get("match", 0)
                    sc = item.get("score", "")
                    reason = str(item.get("reason", ""))[:100]
                    if idx and sc:
                        results[idx] = {"ai_score": sc, "analysis": reason}
                return results
        except:
            pass
    results = {}
    for i in range(1, num_matches + 1):
        pattern = r'"match"\s*:\s*%d\s*,\s*"score"\s*:\s*"([^"]+)"' % i
        sm = re.search(pattern, text)
        if sm:
            results[i] = {"ai_score": sm.group(1), "analysis": "parsed"}
    return results


def get_clean_env_url(name, default=""):
    v = os.environ.get(name, globals().get(name, default))
    v = str(v).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/-]+)", v)
    return match.group(1) if match else v


def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")


def call_ai_model(prompt, url, key, model_name):
    if not url or not key: return None
    is_gem = "generateContent" in url
    if not is_gem and "chat/completions" not in url:
        url = url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if is_gem:
        headers["x-goog-api-key"] = key
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1}}
    else:
        headers["Authorization"] = "Bearer " + key
        payload = {"model": model_name, "messages": [
            {"role": "system", "content": "Output ONLY valid JSON array. No markdown."},
            {"role": "user", "content": prompt}
        ], "temperature": 0.1, "max_tokens": 2000}
    gw = url.split("/v1")[0] if "/v1" in url else url[:35]
    print("  AI: %s @ %s" % (model_name, gw))
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        if r.status_code == 200:
            if is_gem: return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            else: return r.json()["choices"][0]["message"]["content"].strip()
        else: print("  ERR: HTTP %d" % r.status_code)
    except Exception as e: print("  ERR: %s" % str(e)[:60])
    return None


# 🔥 针对截图新增的核心网关容灾列表
FALLBACK_URLS = [
    None, 
    "[https://api521.pro/v1](https://api521.pro/v1)",
    "[https://api.gemai.cc/v1](https://api.gemai.cc/v1)", 
    "[https://www.api520.pro/v1](https://www.api520.pro/v1)",
    "[https://api520.pro/v1](https://api520.pro/v1)",
    "[https://api522.pro/v1](https://api522.pro/v1)",
    "[https://69.63.213.33:666/v1](https://69.63.213.33:666/v1)"
]


def call_one_ai_batch(prompt, url_env, key_env, models_list, num_matches):
    """Call one AI provider with batch prompt, return parsed results"""
    key = get_clean_env_key(key_env)
    primary_url = get_clean_env_url(url_env)
    if not key: return {}, "no_key"
    urls = [primary_url] + [u for u in FALLBACK_URLS if u and u != primary_url]
    for mn in models_list:
        for url in urls:
            if not url: continue
            raw = call_ai_model(prompt, url, key, mn)
            if raw:
                results = parse_batch_response(raw, num_matches)
                if len(results) >= max(1, num_matches * 0.4):
                    print("    %s OK: %d/%d parsed" % (mn[:20], len(results), num_matches))
                    return results, mn
            time.sleep(0.3)
    return {}, "failed"


def call_all_ai_batch(prompt, num_matches):
    """Call ALL 4 AIs each once in batch mode. Returns dict of {ai_name: {match_idx: result}}"""
    # 🔥 核心更新：完美匹配截图模型，严格按照“最聪明/思考优先”的逻辑降级
    ai_configs = [
        ("gpt", "GPT_API_URL", "GPT_API_KEY", [
            "熊猫-A-7-gpt-5.4",                                    # 你指定的首选
            "熊猫-按量-gpt-5.3-codex-满血",
            "熊猫-A-10-gpt-5.3-codex",
            "gpt-4o"                                              # 终极兜底
        ]),
        ("grok", "GROK_API_URL", "GROK_API_KEY", [
            "熊猫-A-6-grok-4.2-thinking",                          # 优先思考模型 (截图1)
            "熊猫-A-7-grok-4.2-多智能体讨论",                      # 次选 (截图2)
            "grok-2"                                              # 终极兜底
        ]),
        ("claude", "CLAUDE_API_URL", "CLAUDE_API_KEY", [
            "熊猫-按量-顶级特供-官max-claude-opus-4.6-thinking",    # 最顶级 Opus + Thinking
            "熊猫-按量-特供顶级-官方正向满血-claude-sonnet-4.6-thinking", # 顶级 Sonnet + Thinking
            "熊猫-按量-满血copilot-claude-opus-4.6-thinking",      # 满血 Copilot Opus + Thinking
            "熊猫-按量-满血copilot-claude-sonnet-4.6-thinking",    # 满血 Copilot Sonnet + Thinking
            "熊猫-按量-顶级特供-官max-claude-opus-4.6",             # 降级：无Thinking的 Opus
            "claude-3-5-sonnet-latest"                            # 终极兜底
        ]),
        ("gemini", "GEMINI_API_URL", "GEMINI_API_KEY", [
            "熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking",    # 优先 SSS按量 + 3.1 + thinking
            "熊猫-特供-X-12-gemini-3.1-pro-preview-thinking",      # 备选 X-12 + 3.1 + thinking
            "熊猫-顶级特供-X-17-gemini-3.1-pro-preview",           # 备选 X-17 + 3.1
            "熊猫特供-按量-SSS-gemini-3.1-pro-preview",            # 备选 SSS按量无thinking
            "gemini-1.5-pro"                                      # 终极兜底
        ]),
    ]
    all_results = {}
    for ai_name, url_env, key_env, models in ai_configs:
        print("  Calling %s batch..." % ai_name.upper())
        results, model_used = call_one_ai_batch(prompt, url_env, key_env, models, num_matches)
        all_results[ai_name] = results
        if results:
            print("    %s: %d results via %s" % (ai_name, len(results), model_used[:25]))
        else:
            print("    %s: FAILED" % ai_name)
        time.sleep(0.5)
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

    # Collect all AI scores
    ai_all = {"gpt": gpt_r, "grok": grok_r, "gemini": gemini_r, "claude": claude_r}
    ai_scores = []
    for name, r in ai_all.items():
        sc = r.get("ai_score", "-") if isinstance(r, dict) else "-"
        if sc and sc not in ["-", "?", ""]:
            ai_scores.append(sc)

    # Vote: count which candidate gets most AI votes
    vote_count = {}
    for sc in ai_scores:
        if sc in candidates:
            vote_count[sc] = vote_count.get(sc, 0) + 2  # in-candidate vote = 2
        else:
            vote_count[sc] = vote_count.get(sc, 0) + 1  # out-candidate vote = 1

    # Final score: engine default, but AI consensus can override if in candidates
    final_score = engine_score
    if vote_count:
        best_voted = max(vote_count, key=vote_count.get)
        if best_voted in candidates and vote_count[best_voted] >= 3:
            final_score = best_voted  # 2+ AIs agree on a candidate -> use it

    # Probabilities from odds (75%) + stats (25%)
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

    # Confidence: engine base + AI agreement bonus
    cf = engine_conf
    agree_count = sum(1 for s in ai_scores if s == engine_score)
    cf = min(90, cf + agree_count * 4)
    has_warn = any("\U0001f6a8" in str(s) for s in smart)
    if has_warn: cf = max(35, cf - 10)
    risk = "\u4f4e" if cf >= 70 else ("\u4e2d" if cf >= 50 else "\u9ad8")

    vh = calculate_value_bet(fhp, sp_h)
    vd = calculate_value_bet(fdp, sp_d)
    va = calculate_value_bet(fap, sp_a)
    vt = []
    for k, v in zip(["\u4e3b\u80dc","\u5e73\u5c40","\u5ba2\u80dc"], [vh,vd,va]):
        if v and v.get("is_value"): vt.append("%s EV:+%s%% Kelly:%s%%" % (k, v["ev"], v["kelly"]))

    pcts = {"\u4e3b\u80dc": fhp, "\u5e73\u5c40": fdp, "\u5ba2\u80dc": fap}
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
    # Claude display: show engine reason if claude didn't respond
    if cl_sc == "-":
        cl_sc = engine_score
        cl_an = engine_result.get("reason", "odds engine")

    return {
        "predicted_score": final_score,
        "home_win_pct": fhp, "draw_pct": fdp, "away_win_pct": fap,
        "confidence": cf, "result": result, "risk_level": risk,
        "over_under_2_5": "\u5927" if o25 > 55 else "\u5c0f",
        "both_score": "\u662f" if bt > 50 else "\u5426",
        "gpt_score": gpt_sc, "gpt_analysis": gpt_an,
        "grok_score": grok_sc, "grok_analysis": grok_an,
        "gemini_score": gem_sc, "gemini_analysis": gem_an,
        "claude_score": cl_sc, "claude_analysis": cl_an,
        "model_agreement": len(set(ai_scores)) <= 1 and len(ai_scores) >= 2,
        "candidate_scores": {sc: round(1.0 if sc == final_score else 0.5, 2) for sc in candidates},
        "poisson": stats.get("poisson", {}),
        "refined_poisson": stats.get("refined_poisson", {}),
        "value_bets_summary": vt,
        "extreme_warning": extreme if extreme else "\u65e0",
        "smart_money_signal": " | ".join(us) if us else "\u6b63\u5e38",
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
        if pr.get("risk_level") == "\u4f4e": s += 12
        elif pr.get("risk_level") == "\u9ad8": s -= 5
        if pr.get("model_agreement"): s += 10
        if pr.get("value_bets_summary"): s += 8

        # === 经验规则加成 ===
        exp_info = pr.get("experience_analysis", {})
        exp_score = exp_info.get("total_score", 0)
        exp_draw_rules = exp_info.get("draw_rules", 0)
        # 经验规则得分高的比赛：如果预测方向与经验一致则加分
        if exp_score >= 15 and pr.get("result") == "\u5e73\u5c40" and exp_draw_rules >= 3:
            s += 12  # 经验强烈看平且模型也看平 -> 大幅加分
        elif exp_score >= 10:
            s += 5   # 经验信号较强
        # 经验风险信号扣分
        if exp_info.get("recommendation", "").startswith("\u26a0"):
            s -= 3  # 有高度警惕信号则保守

        p["recommend_score"] = round(s, 2)
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]


def extract_num(ms):
    wm = {"\u4e00":1000,"\u4e8c":2000,"\u4e09":3000,"\u56db":4000,"\u4e94":5000,"\u516d":6000,"\u65e5":7000,"\u5929":7000}
    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999


def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 60)
    print("  ENGINE v11 BATCH | %d matches | + ExperienceEngine v3.0" % len(ms))
    print("  Odds Engine 80%% + 4x AI batch 20%% + 58 Experience Rules")
    print("=" * 60)

    # Phase 1: Odds Engine + Experience Rules for ALL matches
    match_analyses = []
    for i, m in enumerate(ms):
        h, a = m.get("home_team", "?"), m.get("away_team", "?")
        eng = predict_match(m)
        league_info, _, _, _ = build_league_intelligence(m)
        sp = ensemble.predict(m, {})

        # === 经验规则分析 ===
        exp_result = exp_engine.analyze(m)
        exp_count = exp_result.get("triggered_count", 0)
        exp_tag = ""
        if exp_count >= 3:
            exp_tag = " [EXP:%d rules, score=%d]" % (exp_count, exp_result["total_score"])

        match_analyses.append({
            "match": m, "engine": eng, "league_info": league_info,
            "stats": sp, "index": i, "experience": exp_result,
        })
        print("  [%d] %s vs %s -> %s (%s) conf=%d%%%s" % (
            i+1, h, a, eng["primary_score"], eng["reason"], eng["confidence"], exp_tag))

    # Phase 2: ONE batch prompt -> call ALL 4 AIs (4 API calls total)
    all_ai = {"gpt": {}, "grok": {}, "gemini": {}, "claude": {}}
    if use_ai and match_analyses:
        print("\n  Building batch prompt for %d matches (+ experience context)..." % len(match_analyses))
        prompt = build_batch_prompt(match_analyses)
        print("  Prompt: %d chars | Calling 4 AIs..." % len(prompt))
        all_ai = call_all_ai_batch(prompt, len(match_analyses))
        for name, results in all_ai.items():
            print("  %s: %d results" % (name.upper(), len(results)))

    # Phase 3: Merge engine + 4 AI results + Experience Rules
    res = []
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        eng = ma["engine"]
        sp = ma["stats"]
        exp_result = ma["experience"]
        idx = i + 1
        gpt_r = all_ai.get("gpt", {}).get(idx, {})
        grok_r = all_ai.get("grok", {}).get(idx, {})
        gem_r = all_ai.get("gemini", {}).get(idx, {})
        cl_r = all_ai.get("claude", {}).get(idx, {})
        mg = merge_result(eng, gpt_r, grok_r, gem_r, cl_r, sp, m)

        # === 经验规则融合（核心新增） ===
        mg = apply_experience_to_prediction(m, mg, exp_engine)

        h, a = m.get("home_team", "?"), m.get("away_team", "?")
        exp_info = mg.get("experience_analysis", {})
        exp_tag = ""
        if exp_info.get("triggered_count", 0) > 0:
            exp_tag = " EXP:%d/%d" % (exp_info["triggered_count"], exp_info["total_score"])
        print("  [%d] %s vs %s => %s (%s) %d%% | GPT:%s Grok:%s Gem:%s Cl:%s%s" % (
            i+1, h, a, mg["result"], mg["predicted_score"], mg["confidence"],
            mg["gpt_score"], mg["grok_score"], mg["gemini_score"], mg["claude_score"], exp_tag))
        res.append({**m, "prediction": mg})

    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res: r["is_recommended"] = r["id"] in t4ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    print("\n  TOP4:")
    for i, t in enumerate(t4):
        pr = t.get("prediction", {})
        exp_info = pr.get("experience_analysis", {})
        exp_str = ""
        if exp_info.get("triggered_count", 0) > 0:
            exp_str = " | EXP:%d rules(%s)" % (exp_info["triggered_count"], exp_info.get("recommendation", "")[:15])
        print("    %d. %s vs %s => %s (%s) %d%%%s" % (
            i+1, t.get("home_team"), t.get("away_team"),
            pr.get("result"), pr.get("predicted_score"), pr.get("confidence", 0), exp_str))
    return res, t4

