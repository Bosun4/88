import json, requests, time, re, os
from config import *
from models import EnsemblePredictor
from odds_engine import predict_match, build_ai_context, calc_ttg, calc_had
from league_intel import build_league_intelligence

ensemble = EnsemblePredictor()

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


def build_ai_prompt(m, odds_analysis, league_info, candidates):
    h, a = m.get("home_team", "?"), m.get("away_team", "?")
    lg = m.get("league", "?")
    intel = m.get("intelligence", {})
    p = "[TASK] Choose the best exact score for %s vs %s (%s).\n" % (h, a, lg)
    p += "[IMPORTANT] You must choose from the CANDIDATES below. Do NOT invent other scores.\n\n"
    p += "%s\n\n" % odds_analysis
    p += "%s\n\n" % league_info
    baseface = str(m.get("baseface", "")).strip()
    if baseface:
        p += "[MATCH ANALYSIS] %s\n\n" % baseface[:400]
    intro = str(m.get("expert_intro", "")).strip()
    if intro:
        p += "[EXPERT] %s\n" % intro[:200]
    h_bad = str(intel.get("home_bad_news", "")).strip()
    g_bad = str(intel.get("guest_bad_news", "")).strip()
    h_good = str(intel.get("home_good_news", "")).strip()
    g_good = str(intel.get("guest_good_news", "")).strip()
    if h_bad: p += "[HOME-] %s\n" % h_bad[:200]
    if g_bad: p += "[AWAY-] %s\n" % g_bad[:200]
    if h_good: p += "[HOME+] %s\n" % h_good[:150]
    if g_good: p += "[AWAY+] %s\n" % g_good[:150]
    p += "\n[INJURIES] H: %s\n" % str(intel.get("h_inj", intel.get("home_injury", "?")))[:120]
    p += "A: %s\n" % str(intel.get("g_inj", intel.get("guest_injury", "?")))[:120]
    vote = m.get("vote", {})
    if vote:
        p += "[PUBLIC] W=%s%% D=%s%% L=%s%%\n" % (vote.get("win","?"), vote.get("same","?"), vote.get("lose","?"))
    had = m.get("had_analyse", [])
    if had:
        p += "[OFFICIAL TIP] %s\n" % ",".join(str(x) for x in had)
    p += "\n[CANDIDATES] You MUST pick one of these:\n"
    for i, c in enumerate(candidates):
        p += "  %d. %s\n" % (i+1, c)
    p += "\n[OUTPUT] Pure JSON: {\"ai_score\":\"X-X\",\"analysis\":\"50 char reason\"}\n"
    return p


def extract_clean_json(text):
    text = str(text or "").strip()
    fs, fa = "?", "format error"
    sm = re.search(r'"ai_score"\s*:\s*"([^"]+)"', text)
    if sm: fs = sm.group(1)
    am = re.search(r'"analysis"\s*:\s*"(.*?)"', text, re.DOTALL)
    if am: fa = am.group(1).replace('"',"'").replace("\n"," ").strip()[:150]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try: return json.loads(text[start:end+1])
        except: pass
    cleaned = re.sub(r"```\w*", "", text).strip()
    s2, e2 = cleaned.find("{"), cleaned.rfind("}")
    if s2 != -1 and e2 != -1:
        try: return json.loads(cleaned[s2:e2+1])
        except: pass
    if fs != "?": return {"ai_score": fs, "analysis": fa}
    return None

def get_clean_env_url(name, default=""):
    v = os.environ.get(name, globals().get(name, default))
    v = str(v).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/-]+)", v)
    return match.group(1) if match else v

def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")

def call_ai_model(prompt, url, key, model_name):
    if not url or not key: return {}
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
            {"role": "system", "content": "Output ONLY valid JSON. No markdown."},
            {"role": "user", "content": prompt}
        ], "temperature": 0.1, "max_tokens": 300}
    gw = url.split("/v1")[0] if "/v1" in url else url[:35]
    print("    > %s @ %s" % (model_name, gw))
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            if is_gem: t = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            else: t = r.json()["choices"][0]["message"]["content"].strip()
            parsed = extract_clean_json(t)
            if parsed:
                parsed["analysis"] = str(parsed.get("analysis","")).replace("```json","").replace("```","").strip()
                print("    OK %s: %s" % (model_name, parsed.get("ai_score","?")))
                return parsed
        else: print("    ERR %s: HTTP %d" % (model_name, r.status_code))
    except Exception as e: print("    ERR %s: %s" % (model_name, str(e)[:50]))
    return {}

FALLBACK_URLS = [None, "https://api520.pro/v1", "https://www.api520.pro/v1",
    "https://api521.pro/v1", "https://www.api521.pro/v1",
    "https://api522.pro/v1", "https://www.api522.pro/v1", "https://69.63.213.33:666/v1"]

def call_with_fallback(prompt, url_env, key_env, models_list):
    key = get_clean_env_key(key_env)
    primary_url = get_clean_env_url(url_env, globals().get(url_env, ""))
    if not key: return {}
    urls = [primary_url] + [u for u in FALLBACK_URLS if u and u != primary_url]
    for mn in models_list:
        for url in urls:
            if not url: continue
            r = call_ai_model(prompt, url, key, mn)
            if r and r.get("ai_score") and r["ai_score"] not in ["?",""]: return r
            time.sleep(0.2)
    return {}

def call_gpt(p):
    return call_with_fallback(p, "GPT_API_URL", "GPT_API_KEY", [
        "\u718a\u732b-A-7-gpt-5.4","\u718a\u732b-\u6309\u91cf-gpt-5.3-codex-\u6ee1\u8840",
        "\u718a\u732b-A-10-gpt-5.3-codex","\u718a\u732b-A-1-gpt-5.2",
        "\u718a\u732b-A-5-gpt-5.2","\u718a\u732b-A-8-deepseek-v3.2"])
def call_grok(p):
    return call_with_fallback(p, "GROK_API_URL", "GROK_API_KEY", [
        "\u718a\u732b-A-7-grok-4.2-\u591a\u667a\u80fd\u4f53\u8ba8\u8bba",
        "\u718a\u732b-A-4-grok-4.2-fast"])
def call_gemini(p):
    return call_with_fallback(p, "GEMINI_API_URL", "GEMINI_API_KEY", [
        "\u718a\u732b\u7279\u4f9bS-\u6309\u91cf-gemini-3-flash-preview",
        "\u718a\u732b\u7279\u4f9b-\u6309\u91cf-SSS-gemini-3.1-pro-preview",
        "\u718a\u732b-2-gemini-3.1-flash-lite-preview"])
def call_claude(p):
    return call_with_fallback(p, "CLAUDE_API_URL", "CLAUDE_API_KEY", [
        "\u718a\u732b-\u6309\u91cf-\u9876\u7ea7\u7279\u4f9b-\u5b98max-claude-opus-4.6",
        "\u718a\u732b-\u6309\u91cf-\u9876\u7ea7\u7279\u4f9b-\u5b98max-claude-opus-4.6-thinking",
        "\u718a\u732b-\u6309\u91cf-\u7279\u4f9b\u9876\u7ea7-\u5b98\u65b9\u6b63\u5411\u6ee1\u8840-claude-opus-4.6",
        "\u718a\u732b-\u7279\u4f9b-\u6309\u91cf-Q-claude-opus-4.6",
        "\u718a\u732b-\u7279\u4f9b-\u6309\u91cf-Q-claude-opus-4.6-thinking",
        "\u718a\u732b-\u7279\u4f9b-\u6309\u91cf-Q-claude-sonnet-4.6",
        "\u718a\u732b-\u7279\u4ef7\u9006-19-claude-opus-4-6",
        "\u718a\u732b-\u7279\u4ef7\u9006-15-claude-sonnet-4-6"])


def merge_all(engine_result, gpt, grok, gemini, claude, stats, match_obj):
    sp_h = float(match_obj.get("sp_home", 0) or 0)
    sp_d = float(match_obj.get("sp_draw", 0) or 0)
    sp_a = float(match_obj.get("sp_away", 0) or 0)
    v2 = match_obj.get("v2_odds_dict", {})
    if not v2: v2 = match_obj
    mc = stats.get("model_consensus", 0)
    tm = stats.get("total_models", 11)
    smart = stats.get("smart_signals", [])
    extreme = stats.get("extreme_warning", "")

    engine_score = engine_result.get("primary_score", engine_result.get("score", "1-1"))
    engine_conf = engine_result["confidence"]
    candidates = engine_result.get("top3_scores", [engine_score])

    o25 = engine_result.get("over_25", 50)
    bt = engine_result.get("btts", 45)

    cands = {engine_score: {"weight": 0.70 * engine_conf / 100, "sources": ["engine"]}}

    ai_all = [("gpt",gpt),("grok",grok),("gemini",gemini),("claude",claude)]
    ai_w = {"gpt": 0.10, "grok": 0.10, "gemini": 0.05, "claude": 0.05}
    ai_scores = []
    for name, res in ai_all:
        sc = res.get("ai_score") if isinstance(res, dict) else None
        if sc and sc not in ["-","?",""]:
            if sc not in cands: cands[sc] = {"weight": 0.0, "sources": []}
            w = ai_w.get(name, 0.05)
            if sc in candidates:
                w *= 1.5
            else:
                w *= 0.3
            cands[sc]["weight"] += w
            cands[sc]["sources"].append(name)
            ai_scores.append(sc)

    final_score = max(cands.items(), key=lambda x: x[1]["weight"])[0] if cands else engine_score
    fh, fa = parse_score(final_score)
    if fh is None: fh, fa = 1, 1

    if sp_h > 1 and sp_d > 1 and sp_a > 1:
        ih, id2, ia = 1/sp_h, 1/sp_d, 1/sp_a
        t = ih + id2 + ia
        ohp, odp, oap = ih/t*100, id2/t*100, ia/t*100
    else:
        ohp, odp, oap = 33, 33, 34

    ai_hp = ai_dp = ai_ap = 0.0
    ai_tw = 0.0
    for name, res in ai_all:
        sc = res.get("ai_score") if isinstance(res, dict) else None
        if not sc or sc in ["-","?",""]: continue
        hg, ag = parse_score(sc)
        if hg is None: continue
        w = ai_w.get(name, 0.05)
        ai_tw += w
        if hg > ag: ai_hp += w
        elif hg == ag: ai_dp += w
        else: ai_ap += w
    if ai_tw > 0:
        ai_hp = ai_hp/ai_tw*100; ai_dp = ai_dp/ai_tw*100; ai_ap = ai_ap/ai_tw*100
    else:
        ai_hp, ai_dp, ai_ap = ohp, odp, oap

    fhp = ohp * 0.75 + ai_hp * 0.10 + stats.get("home_win_pct", 33) * 0.15
    fdp = odp * 0.75 + ai_dp * 0.10 + stats.get("draw_pct", 33) * 0.15
    fap = oap * 0.75 + ai_ap * 0.10 + stats.get("away_win_pct", 33) * 0.15

    fhp = max(3, fhp); fdp = max(3, fdp); fap = max(3, fap)
    ft = fhp + fdp + fap
    if ft > 0:
        fhp = round(fhp/ft*100, 1); fdp = round(fdp/ft*100, 1); fap = round(max(3, 100-fhp-fdp), 1)
        ft2 = fhp + fdp + fap
        if abs(ft2 - 100) > 0.5:
            fhp = round(fhp/ft2*100, 1); fdp = round(fdp/ft2*100, 1); fap = round(100-fhp-fdp, 1)

    cf = engine_conf
    has_warn = any("🚨" in str(s) for s in smart)
    if has_warn: cf = max(35, cf - 10)
    ai_agree = sum(1 for s in ai_scores if s == engine_score)
    cf = min(95, cf + ai_agree * 5)

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

    return {
        "predicted_score": final_score,
        "home_win_pct": fhp, "draw_pct": fdp, "away_win_pct": fap,
        "confidence": cf, "result": result, "risk_level": risk,
        "over_under_2_5": "\u5927" if o25 > 55 else "\u5c0f",
        "both_score": "\u662f" if bt > 50 else "\u5426",
        "gpt_score": gpt.get("ai_score","-"), "gpt_analysis": gpt.get("analysis","N/A"),
        "grok_score": grok.get("ai_score","-"), "grok_analysis": grok.get("analysis","N/A"),
        "gemini_score": gemini.get("ai_score","-"), "gemini_analysis": gemini.get("analysis","N/A"),
        "claude_score": claude.get("ai_score","-"), "claude_analysis": claude.get("analysis","N/A"),
        "model_agreement": len(set(ai_scores)) <= 1 if ai_scores else False,
        "candidate_scores": {sc: round(c["weight"], 2) for sc, c in cands.items()},
        "poisson": {**stats.get("poisson",{}), "home_expected_goals": stats.get("poisson",{}).get("home_xg","?")},
        "refined_poisson": stats.get("refined_poisson",{}),
        "value_bets_summary": vt,
        "extreme_warning": extreme if extreme else "\u65e0",
        "smart_money_signal": " | ".join(us) if us else "\u6b63\u5e38",
        "smart_signals": us, "model_consensus": mc, "total_models": tm,
        "expected_total_goals": engine_result.get("expected_goals", 2.5),
        "over_2_5": o25, "btts": bt,
        "top_scores": stats.get("refined_poisson",{}).get("top_scores",[]),
        "elo": stats.get("elo",{}), "random_forest": stats.get("random_forest",{}),
        "gradient_boost": stats.get("gradient_boost",{}), "neural_net": stats.get("neural_net",{}),
        "logistic": stats.get("logistic",{}), "svm": stats.get("svm",{}), "knn": stats.get("knn",{}),
        "dixon_coles": stats.get("dixon_coles",{}), "bradley_terry": stats.get("bradley_terry",{}),
        "home_form": stats.get("home_form",{}), "away_form": stats.get("away_form",{}),
        "handicap_signal": stats.get("handicap_signal",""),
        "odds_movement": stats.get("odds_movement",{}), "vote_analysis": stats.get("vote_analysis",{}),
        "h2h_blood": stats.get("h2h_blood",{}), "crs_analysis": stats.get("crs_analysis",{}),
        "ttg_analysis": stats.get("ttg_analysis",{}), "halftime": stats.get("halftime",{}),
        "pace_rating": stats.get("pace_rating",""),
        "kelly_home": stats.get("kelly_home",{}), "kelly_away": stats.get("kelly_away",{}),
        "odds": stats.get("odds",{}),
    }


def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})
        s = pr.get("confidence", 0) * 0.4
        mx = max(pr.get("home_win_pct",33), pr.get("away_win_pct",33), pr.get("draw_pct",33))
        s += (mx - 33) * 0.2 + pr.get("model_consensus", 0) * 2
        if pr.get("risk_level") == "\u4f4e": s += 12
        elif pr.get("risk_level") == "\u9ad8": s -= 5
        if pr.get("model_agreement"): s += 10
        if pr.get("value_bets_summary"): s += 8
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
    print("  ENGINE v10.0 | %d matches" % len(ms))
    print("  Odds Engine 70%% + AI Advisor 30%%")
    print("=" * 60)
    res = []
    for i, m in enumerate(ms):
        h, a = m.get("home_team","?"), m.get("away_team","?")
        print("\n  [%d/%d] %s %s vs %s" % (i+1, len(ms), m.get("league",""), h, a))

        engine_result = predict_match(m)
        candidates = engine_result.get("top3_scores", [engine_result["primary_score"]])
        print("    Engine: %s (%s) conf=%d%% TOP3=%s" % (
            engine_result["primary_score"], engine_result["reason"],
            engine_result["confidence"], ",".join(candidates)))

        sp = ensemble.predict(m, {})
        odds_analysis = build_ai_context(m, engine_result)
        league_info, _, _, _ = build_league_intelligence(m)

        if use_ai:
            prompt = build_ai_prompt(m, odds_analysis, league_info, candidates)
            gpt_r = call_gpt(prompt); time.sleep(0.3)
            grok_r = call_grok(prompt); time.sleep(0.3)
            gem_r = call_gemini(prompt); time.sleep(0.3)
            cl_r = call_claude(prompt); time.sleep(0.3)
        else:
            b = {"ai_score": "-", "analysis": "blocked"}
            gpt_r = grok_r = gem_r = cl_r = b

        mg = merge_all(engine_result, gpt_r or {}, grok_r or {}, gem_r or {}, cl_r or {}, sp, m)
        print("  => %s (%s) %d%% | GPT:%s Grok:%s Gem:%s Claude:%s" % (
            mg["result"], mg["predicted_score"], mg["confidence"],
            mg["gpt_score"], mg["grok_score"], mg["gemini_score"], mg["claude_score"]))
        res.append({**m, "prediction": mg})

    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res: r["is_recommended"] = r["id"] in t4ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    return res, t4