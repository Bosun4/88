import json, requests, time, re, os
from config import *
from models import EnsemblePredictor

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


def detect_match_context(m):
    try: hr = int(m.get("home_rank", 10) or 10)
    except: hr = 10
    try: ar = int(m.get("away_rank", 10) or 10)
    except: ar = 10
    lg = str(m.get("league", ""))
    baseface = str(m.get("baseface", ""))
    ctx = []
    is_knockout = False
    if any(k in lg for k in ["\u6b27\u7f57\u5df4","\u6b27\u51a0","\u6b27\u534f\u8054","Europa","Champions","Conference"]):
        is_knockout = True
        ctx.append("KNOCKOUT LEG: Elimination match! Teams MUST attack. Goals typically 2.5-4+ per game. Do NOT predict low-scoring draws unless both legs are 0-0.")
    if "\u6b21\u56de\u5408" in baseface or "\u7b2c\u4e8c\u56de\u5408" in baseface or "2nd leg" in baseface.lower():
        ctx.append("2ND LEG: Trailing team will go all-out attack, expect MORE goals than 1st leg. Consider aggregate score pressure.")
    if hr <= 3 or ar <= 3:
        ctx.append("TITLE RACE: top-3 team, high motivation")
    if hr >= 16 or ar >= 16:
        ctx.append("RELEGATION: desperate play, chaotic")
    if abs(hr - ar) <= 3 and hr <= 10:
        ctx.append("SIMILAR STRENGTH: competitive match")
    if hr <= 5 and ar >= 14:
        ctx.append("MISMATCH: strong home favorite, 2+ goal win likely")
    return ctx, is_knockout


def get_crs_top5(m):
    v2 = m.get("v2_odds_dict", {})
    if not v2: v2 = m
    crs_map = {"w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","w32":"3-2",
               "s00":"0-0","s11":"1-1","s22":"2-2",
               "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3"}
    crs = []
    for k, sc in crs_map.items():
        ov = float(v2.get(k, 0) or 0)
        if ov > 1:
            crs.append((sc, ov, round(1/ov*100, 1)))
    crs.sort(key=lambda x: x[2], reverse=True)
    return crs[:7]


def get_ttg_summary(m):
    v2 = m.get("v2_odds_dict", {})
    if not v2: v2 = m
    ttg = {"a0":0,"a1":1,"a2":2,"a3":3,"a4":4,"a5":5,"a6":6,"a7":7}
    probs = {}
    for k, g in ttg.items():
        ov = float(v2.get(k, 0) or 0)
        if ov > 1:
            probs[g] = round(1/ov*100, 1)
    if not probs: return ""
    ti = sum(probs.values())
    if ti > 0: probs = {k: round(v/ti*100, 1) for k, v in probs.items()}
    exp = sum(g * p/100 for g, p in probs.items())
    parts = []
    for g in sorted(probs.keys()):
        parts.append("%dg=%.0f%%" % (g, probs[g]))
    return "Expected total: %.1f goals | %s" % (exp, " ".join(parts))


def parse_score(s):
    try:
        p = str(s).split("-")
        return int(p[0]), int(p[1])
    except:
        return None, None


# ====================================================================
# AI 1: GPT - Tactical Logic + Score Prediction
# ====================================================================
def build_gpt_prompt(m):
    h, a = m.get("home_team", "?"), m.get("away_team", "?")
    lg = m.get("league", "?")
    sp_h, sp_d, sp_a = m.get("sp_home", 0), m.get("sp_draw", 0), m.get("sp_away", 0)
    intel = m.get("intelligence", {})
    hs = m.get("home_stats", {})
    ast = m.get("away_stats", {})
    ctx, is_ko = detect_match_context(m)
    p = "[ROLE] Tactical analyst. Predict exact score using logic and data.\n"
    p += "[MATCH] %s | %s vs %s\n" % (lg, h, a)
    p += "[ODDS] H=%s D=%s A=%s HC=%s\n" % (sp_h, sp_d, sp_a, m.get("give_ball", "?"))
    if ctx:
        p += "[CONTEXT] %s\n" % " | ".join(ctx)
    ttg = get_ttg_summary(m)
    if ttg:
        p += "[GOALS MARKET] %s\n" % ttg
    baseface = str(m.get("baseface", "")).strip()
    if baseface:
        p += "\n[ANALYSIS]\n%s\n" % baseface[:500]
    p += "\n[HOME %s] Rank#%s | %sP %sW %sD %sL | GF=%s GA=%s | AvgGF=%s | Form:%s\n" % (
        h, m.get("home_rank","?"), hs.get("played","?"), hs.get("wins","?"),
        hs.get("draws","?"), hs.get("losses","?"), hs.get("goals_for","?"),
        hs.get("goals_against","?"), hs.get("avg_goals_for","?"), hs.get("form","?"))
    p += "[AWAY %s] Rank#%s | %sP %sW %sD %sL | GF=%s GA=%s | AvgGF=%s | Form:%s\n" % (
        a, m.get("away_rank","?"), ast.get("played","?"), ast.get("wins","?"),
        ast.get("draws","?"), ast.get("losses","?"), ast.get("goals_for","?"),
        ast.get("goals_against","?"), ast.get("avg_goals_for","?"), ast.get("form","?"))
    p += "Injuries H: %s\n" % str(intel.get("h_inj","?"))[:120]
    p += "Injuries A: %s\n" % str(intel.get("g_inj","?"))[:120]
    crs = get_crs_top5(m)
    if crs:
        p += "\n[CRS TOP7]\n"
        for sc, od, prob in crs:
            p += "  %s @%.2f (%.1f%%)\n" % (sc, od, prob)
    had = m.get("had_analyse", [])
    if had:
        p += "[OFFICIAL] %s\n" % ",".join(str(x) for x in had)
    p += "\n[IMPORTANT] Use the GOALS MARKET data to calibrate total goals. "
    if is_ko:
        p += "This is a KNOCKOUT match - goals are typically HIGHER than league games. Do NOT default to 1-0 or 1-1.\n"
    p += "Consider both teams AvgGF to estimate realistic total.\n"
    p += "[OUTPUT] Pure JSON: {\"ai_score\":\"2-1\",\"analysis\":\"80 char reasoning\"}\n"
    return p


# ====================================================================
# AI 2: Grok - Skeptical Analyst (NOT blind contrarian)
# ====================================================================
def build_grok_prompt(m):
    h, a = m.get("home_team", "?"), m.get("away_team", "?")
    lg = m.get("league", "?")
    sp_h, sp_d, sp_a = m.get("sp_home", 0), m.get("sp_draw", 0), m.get("sp_away", 0)
    intel = m.get("intelligence", {})
    ctx, is_ko = detect_match_context(m)
    p = "[ROLE] Skeptical analyst. You question assumptions but follow evidence.\n"
    p += "[IMPORTANT] You are NOT a blind contrarian. If data strongly supports the favorite, say so.\n"
    p += "Only challenge consensus when you find SPECIFIC evidence of a trap or mispricing.\n\n"
    p += "[MATCH] %s | %s vs %s\n" % (lg, h, a)
    p += "[ODDS] H=%s D=%s A=%s HC=%s\n" % (sp_h, sp_d, sp_a, m.get("give_ball", "?"))
    if ctx:
        p += "[CONTEXT] %s\n" % " | ".join(ctx)
    vote = m.get("vote", {})
    if vote:
        p += "[PUBLIC] W=%s%% D=%s%% L=%s%%\n" % (vote.get("win","?"), vote.get("same","?"), vote.get("lose","?"))
    change = m.get("change", {})
    if change and isinstance(change, dict):
        p += "[MOVEMENT] win=%s draw=%s lose=%s\n" % (change.get("win","0"), change.get("same","0"), change.get("lose","0"))
    ttg = get_ttg_summary(m)
    if ttg:
        p += "[GOALS MARKET] %s\n" % ttg
    h_good = str(intel.get("home_good_news", "")).strip()
    h_bad = str(intel.get("home_bad_news", "")).strip()
    g_good = str(intel.get("guest_good_news", "")).strip()
    g_bad = str(intel.get("guest_bad_news", "")).strip()
    if h_bad: p += "\n[HOME WEAKNESS] %s\n" % h_bad[:250]
    if g_bad: p += "[AWAY WEAKNESS] %s\n" % g_bad[:250]
    if g_good: p += "[AWAY STRENGTH] %s\n" % g_good[:200]
    p += "\n[INJURIES] H: %s\n" % str(intel.get("h_inj", intel.get("home_injury", "?")))[:150]
    p += "A: %s\n" % str(intel.get("g_inj", intel.get("guest_injury", "?")))[:150]
    crs = get_crs_top5(m)
    if crs:
        p += "\n[CRS TOP7]\n"
        for sc, od, prob in crs:
            p += "  %s @%.2f (%.1f%%)\n" % (sc, od, prob)
    p += "\n[ANALYSIS STEPS]\n"
    p += "1. Is the favorite genuinely strong or just popular? Check injuries and form.\n"
    p += "2. Public >60%% on one side + odds stable = possible trap. But only call it if you have evidence.\n"
    p += "3. If favorite has clear quality advantage, DO predict them to win.\n"
    if is_ko:
        p += "4. KNOCKOUT: expect more goals. Both teams have incentive to score.\n"
    p += "\n[OUTPUT] Pure JSON: {\"ai_score\":\"1-2\",\"analysis\":\"80 char reasoning\"}\n"
    return p


# ====================================================================
# AI 3: Gemini - Data Scorer (simplified for better parsing)
# ====================================================================
def build_gemini_prompt(m):
    h, a = m.get("home_team", "?"), m.get("away_team", "?")
    lg = m.get("league", "?")
    sp_h, sp_d, sp_a = m.get("sp_home", 0), m.get("sp_draw", 0), m.get("sp_away", 0)
    hs = m.get("home_stats", {})
    ast = m.get("away_stats", {})
    intel = m.get("intelligence", {})
    ctx, is_ko = detect_match_context(m)
    p = "[TASK] Predict exact score for %s vs %s (%s).\n" % (h, a, lg)
    p += "Odds: H=%s D=%s A=%s\n" % (sp_h, sp_d, sp_a)
    if ctx:
        p += "Context: %s\n" % "; ".join(ctx)
    ttg = get_ttg_summary(m)
    if ttg:
        p += "%s\n" % ttg
    p += "%s: %sW%sD%sL GF=%s GA=%s Form=%s\n" % (
        h, hs.get("wins","?"), hs.get("draws","?"), hs.get("losses","?"),
        hs.get("goals_for","?"), hs.get("goals_against","?"), hs.get("form","?"))
    p += "%s: %sW%sD%sL GF=%s GA=%s Form=%s\n" % (
        a, ast.get("wins","?"), ast.get("draws","?"), ast.get("losses","?"),
        ast.get("goals_for","?"), ast.get("goals_against","?"), ast.get("form","?"))
    p += "Injuries: H=%s A=%s\n" % (str(intel.get("h_inj","?"))[:80], str(intel.get("g_inj","?"))[:80])
    crs = get_crs_top5(m)
    if crs:
        p += "Market scores: %s\n" % ", ".join(["%s(%.1f%%)" % (sc, prob) for sc, od, prob in crs[:5]])
    p += "\nRespond ONLY with JSON: {\"ai_score\":\"2-1\",\"analysis\":\"short reason\"}\n"
    return p


# ====================================================================
# AI 4: Claude Analyst - Psychology + Narrative
# ====================================================================
def build_claude_analyst_prompt(m):
    h, a = m.get("home_team", "?"), m.get("away_team", "?")
    lg = m.get("league", "?")
    sp_h, sp_d, sp_a = m.get("sp_home", 0), m.get("sp_draw", 0), m.get("sp_away", 0)
    intel = m.get("intelligence", {})
    ctx, is_ko = detect_match_context(m)
    p = "[ROLE] Football psychology expert. Predict through human factors.\n"
    p += "[MATCH] %s | %s vs %s\n" % (lg, h, a)
    p += "[ODDS] H=%s D=%s A=%s\n" % (sp_h, sp_d, sp_a)
    if ctx:
        p += "[CONTEXT] %s\n" % " | ".join(ctx)
    ttg = get_ttg_summary(m)
    if ttg:
        p += "[GOALS] %s\n" % ttg
    baseface = str(m.get("baseface", "")).strip()
    if baseface:
        p += "\n[SITUATION] %s\n" % baseface[:400]
    intro = str(m.get("expert_intro", "")).strip()
    if intro:
        p += "[EXPERT] %s\n" % intro[:200]
    h_good = str(intel.get("home_good_news", "")).strip()
    h_bad = str(intel.get("home_bad_news", "")).strip()
    g_good = str(intel.get("guest_good_news", "")).strip()
    g_bad = str(intel.get("guest_bad_news", "")).strip()
    if h_good: p += "\n[HOME+] %s\n" % h_good[:200]
    if h_bad: p += "[HOME-] %s\n" % h_bad[:200]
    if g_good: p += "[AWAY+] %s\n" % g_good[:200]
    if g_bad: p += "[AWAY-] %s\n" % g_bad[:200]
    p += "\n[THINK] Who is mentally stronger? Who has more to lose? What happens if home goes behind?\n"
    if is_ko:
        p += "[KNOCKOUT] Elimination pressure = more goals, more drama. Trailing team goes kamikaze.\n"
    p += "[OUTPUT] Pure JSON: {\"ai_score\":\"1-0\",\"analysis\":\"80 char psychological verdict\"}\n"
    return p


# ====================================================================
# AI 5: Claude Arbiter - DECISIVE judge
# ====================================================================
def build_arbiter_prompt(m, gpt_res, grok_res, gemini_res, claude_p1_res, stats_pred):
    h, a = m.get("home_team", "?"), m.get("away_team", "?")
    sp_h, sp_d, sp_a = m.get("sp_home", 0), m.get("sp_draw", 0), m.get("sp_away", 0)
    ctx, is_ko = detect_match_context(m)
    p = "[ROLE] Supreme judge. You have 4 expert reports. Make a DECISIVE final call.\n"
    p += "[CRITICAL RULES]\n"
    p += "- Do NOT default to draws. Only predict draw if STRONG evidence supports it.\n"
    p += "- If 3+ experts agree on a direction (home/away win), follow that direction.\n"
    p += "- Use GOALS MARKET data to pick realistic total goals.\n"
    if is_ko:
        p += "- KNOCKOUT MATCH: Goals are typically HIGH (2.5-4+). Avoid 0-0 and 1-0 predictions.\n"
    p += "\n[4 EXPERT REPORTS] Weight: Grok40%% GPT35%% Gemini15%% Psychology10%%\n"
    p += "GPT(35%%): %s | %s\n" % (gpt_res.get("ai_score","?"), str(gpt_res.get("analysis","?"))[:100])
    p += "Grok(40%%): %s | %s\n" % (grok_res.get("ai_score","?"), str(grok_res.get("analysis","?"))[:100])
    p += "Gemini(15%%): %s | %s\n" % (gemini_res.get("ai_score","?"), str(gemini_res.get("analysis","?"))[:100])
    p += "Psychology(10%%): %s | %s\n\n" % (claude_p1_res.get("ai_score","?"), str(claude_p1_res.get("analysis","?"))[:100])
    scores = [gpt_res.get("ai_score",""), grok_res.get("ai_score",""), gemini_res.get("ai_score",""), claude_p1_res.get("ai_score","")]
    scores = [s for s in scores if s and s not in ["?","-",""]]
    if scores:
        total_goals = []
        for s in scores:
            hg, ag = parse_score(s)
            if hg is not None:
                total_goals.append(hg + ag)
        if total_goals:
            avg_g = sum(total_goals) / len(total_goals)
            p += "[AI AVG GOALS] %.1f (if your pick is far below this, reconsider)\n" % avg_g
    ttg = get_ttg_summary(m)
    if ttg:
        p += "[GOALS MARKET] %s\n" % ttg
    if ctx:
        p += "[CONTEXT] %s\n" % " | ".join(ctx)
    crs = stats_pred.get("crs_analysis", {}).get("top_scores", [])[:5]
    if crs:
        p += "[MARKET] %s\n" % ", ".join(["%s(%.1f%%)" % (s.get("score","?"), s.get("prob",0)) for s in crs])
    p += "[ODDS] H=%s D=%s A=%s HC=%s\n" % (sp_h, sp_d, sp_a, m.get("give_ball","?"))
    smart = stats_pred.get("smart_signals", [])
    if smart:
        p += "[SIGNALS] %s\n" % " | ".join(smart)
    p += "\n[VERDICT] Pick ONE exact score. Be DECISIVE. Explain in 100 chars.\n"
    p += "[OUTPUT] Pure JSON: {\"ai_score\":\"2-1\",\"analysis\":\"100 char verdict\"}\n"
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
    if fs != "?":
        return {"ai_score": fs, "analysis": fa}
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
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}
    else:
        headers["Authorization"] = "Bearer " + key
        payload = {"model": model_name, "messages": [
            {"role": "system", "content": "Output ONLY valid JSON. No markdown, no explanation."},
            {"role": "user", "content": prompt}
        ], "temperature": 0.2, "max_tokens": 400}
    gw = url.split("/v1")[0] if "/v1" in url else url[:35]
    print("    > %s @ %s" % (model_name, gw))
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            if is_gem:
                t = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                t = r.json()["choices"][0]["message"]["content"].strip()
            parsed = extract_clean_json(t)
            if parsed:
                parsed["analysis"] = str(parsed.get("analysis","")).replace("```json","").replace("```","").strip()
                print("    OK %s: %s" % (model_name, parsed.get("ai_score","?")))
                return parsed
            else:
                print("    WARN %s: parse fail" % model_name)
        else:
            print("    ERR %s: HTTP %d" % (model_name, r.status_code))
    except Exception as e:
        print("    ERR %s: %s" % (model_name, str(e)[:50]))
    return {}


FALLBACK_URLS = [
    None,
    "https://api520.pro/v1", "https://www.api520.pro/v1",
    "https://api521.pro/v1", "https://www.api521.pro/v1",
    "https://api522.pro/v1", "https://www.api522.pro/v1",
    "https://69.63.213.33:666/v1",
]

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


def merge_all(gpt, grok, gemini, claude_p1, claude_arb, stats, match_obj):
    sp_h = float(match_obj.get("sp_home", 0) or 0)
    sp_d = float(match_obj.get("sp_draw", 0) or 0)
    sp_a = float(match_obj.get("sp_away", 0) or 0)
    sys_score = stats.get("predicted_score", "1-1")
    o25 = stats.get("over_2_5", 50)
    bt = stats.get("btts", 50)
    mc = stats.get("model_consensus", 0)
    tm = stats.get("total_models", 11)
    top_crs = stats.get("crs_analysis", {}).get("top_scores", [])[:5]
    smart = stats.get("smart_signals", [])
    extreme = stats.get("extreme_warning", "")

    if sp_h > 1 and sp_d > 1 and sp_a > 1:
        ih, id2, ia = 1/sp_h, 1/sp_d, 1/sp_a
        t = ih + id2 + ia
        ohp, odp, oap = ih/t*100, id2/t*100, ia/t*100
    else:
        ohp, odp, oap = 33, 33, 34

    cands = {}
    ai_scores = []
    ai_w = {"grok": 0.40, "gpt": 0.35, "gemini": 0.15, "claude_p1": 0.10, "arbiter": 0.35}
    all_ai = [("gpt",gpt),("grok",grok),("gemini",gemini),("claude_p1",claude_p1),("arbiter",claude_arb)]
    for name, res in all_ai:
        sc = res.get("ai_score") if isinstance(res, dict) else None
        if sc and sc not in ["-","?",""]:
            if sc not in cands: cands[sc] = {"weight": 0.0, "sources": []}
            cands[sc]["weight"] += ai_w.get(name, 0.15)
            cands[sc]["sources"].append(name)
            ai_scores.append(sc)

    for s in top_crs[:3]:
        sc = s.get("score")
        if sc:
            if sc not in cands: cands[sc] = {"weight": 0.0, "sources": []}
            cands[sc]["weight"] += 0.22
            cands[sc]["sources"].append("market")

    has_warn = any("🚨" in str(s) for s in smart) or (extreme and extreme not in ["","\u65e0"])
    mkt_top = top_crs[0].get("score") if top_crs else None

    for sc in cands:
        if sc == mkt_top: cands[sc]["weight"] += 0.20
        cnt = ai_scores.count(sc)
        if cnt >= 3: cands[sc]["weight"] += 0.55
        elif cnt >= 2: cands[sc]["weight"] += 0.30
        grok_sc = grok.get("ai_score") if isinstance(grok, dict) else None
        gpt_sc = gpt.get("ai_score") if isinstance(gpt, dict) else None
        if sc == grok_sc and sc == gpt_sc: cands[sc]["weight"] += 0.25
        if has_warn and sc in [s.get("score") for s in top_crs]: cands[sc]["weight"] += 0.10

    final_score = max(cands.items(), key=lambda x: x[1]["weight"])[0] if cands else sys_score

    ai_hp = ai_dp = ai_ap = 0.0
    ai_tw = 0.0
    for name, res in all_ai:
        sc = res.get("ai_score") if isinstance(res, dict) else None
        if not sc or sc in ["-","?",""]: continue
        hg, ag = parse_score(sc)
        if hg is None: continue
        w = ai_w.get(name, 0.15)
        ai_tw += w
        if hg > ag: ai_hp += w
        elif hg == ag: ai_dp += w
        else: ai_ap += w
    if ai_tw > 0:
        ai_hp = ai_hp/ai_tw*100; ai_dp = ai_dp/ai_tw*100; ai_ap = ai_ap/ai_tw*100
    else:
        ai_hp, ai_dp, ai_ap = ohp, odp, oap

    fhp = ai_hp*0.55 + ohp*0.30 + stats.get("home_win_pct",33)*0.15
    fdp = ai_dp*0.55 + odp*0.30 + stats.get("draw_pct",33)*0.15
    fap = ai_ap*0.55 + oap*0.30 + stats.get("away_win_pct",33)*0.15
    if fdp < 10:
        b = 10 - fdp; fdp = 10; fhp -= b*0.6; fap -= b*0.4
    ft = fhp + fdp + fap
    if ft > 0:
        fhp = round(fhp/ft*100, 1); fdp = round(fdp/ft*100, 1); fap = round(100-fhp-fdp, 1)

    uai = len(set(ai_scores))
    ab = 25 if uai <= 1 and len(ai_scores) >= 3 else (12 if uai <= 2 else 0)
    mm = 18 if final_score == mkt_top else 0
    wp = -10 if has_warn else 0
    cf = min(95, max(35, round(48 + ab + mm + wp + (max(fhp, fdp, fap) - 33) * 0.3)))
    risk = "\u4f4e" if cf >= 75 else ("\u4e2d" if cf >= 55 else "\u9ad8")

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
        "claude_score": claude_arb.get("ai_score","-"), "claude_analysis": claude_arb.get("analysis","N/A"),
        "model_agreement": uai <= 1 if ai_scores else False,
        "candidate_scores": {sc: round(c["weight"], 2) for sc, c in cands.items()},
        "poisson": {**stats.get("poisson",{}), "home_expected_goals": stats.get("poisson",{}).get("home_xg","?")},
        "refined_poisson": stats.get("refined_poisson",{}),
        "value_bets_summary": vt,
        "extreme_warning": extreme if extreme else "\u65e0",
        "smart_money_signal": " | ".join(us) if us else "\u6b63\u5e38",
        "smart_signals": us, "model_consensus": mc, "total_models": tm,
        "expected_total_goals": stats.get("expected_total_goals", 2.5),
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
        "over_2_5": o25, "btts": bt, "pace_rating": stats.get("pace_rating",""),
        "kelly_home": stats.get("kelly_home",{}), "kelly_away": stats.get("kelly_away",{}),
        "odds": stats.get("odds",{}),
    }


def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})
        s = pr.get("confidence", 0) * 0.35
        mx = max(pr.get("home_win_pct",33), pr.get("away_win_pct",33), pr.get("draw_pct",33))
        s += (mx - 33) * 0.25 + pr.get("model_consensus", 0) * 2.5
        if pr.get("risk_level") == "\u4f4e": s += 10
        elif pr.get("risk_level") == "\u9ad8": s -= 5
        if pr.get("model_agreement"): s += 15
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
    print("  ENGINE v8.0 | %d matches" % len(ms))
    print("  5-AI | Goals calibrated | Knockout aware")
    print("=" * 60)
    res = []
    for i, m in enumerate(ms):
        h, a = m.get("home_team","?"), m.get("away_team","?")
        print("\n  [%d/%d] %s %s vs %s" % (i+1, len(ms), m.get("league",""), h, a))
        sp = ensemble.predict(m, {})
        if use_ai:
            gpt_r = call_gpt(build_gpt_prompt(m)); time.sleep(0.3)
            grok_r = call_grok(build_grok_prompt(m)); time.sleep(0.3)
            gem_r = call_gemini(build_gemini_prompt(m)); time.sleep(0.3)
            cl_p1 = call_claude(build_claude_analyst_prompt(m)); time.sleep(0.3)
            arb_p = build_arbiter_prompt(m, gpt_r or {}, grok_r or {}, gem_r or {}, cl_p1 or {}, sp)
            cl_arb = call_claude(arb_p); time.sleep(0.3)
        else:
            b = {"ai_score": "-", "analysis": "blocked"}
            gpt_r = grok_r = gem_r = cl_p1 = cl_arb = b
        mg = merge_all(gpt_r or {}, grok_r or {}, gem_r or {}, cl_p1 or {}, cl_arb or {}, sp, m)
        print("  => %s (%s) %d%% | GPT:%s Grok:%s Gem:%s Arb:%s" % (
            mg["result"], mg["predicted_score"], mg["confidence"],
            mg["gpt_score"], mg["grok_score"], mg["gemini_score"], mg["claude_score"]))
        res.append({**m, "prediction": mg})
    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res: r["is_recommended"] = r["id"] in t4ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    return res, t4