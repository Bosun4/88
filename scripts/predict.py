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


def detect_league_context(m):
    try: hr = int(m.get("home_rank", 10) or 10)
    except: hr = 10
    try: ar = int(m.get("away_rank", 10) or 10)
    except: ar = 10
    baseface = str(m.get("baseface", ""))
    ctx = []
    if hr <= 3 or ar <= 3:
        ctx.append("TITLE RACE: top-3 team involved")
    if hr >= 16 or ar >= 16:
        ctx.append("RELEGATION BATTLE: bottom-5 team")
    if hr <= 3 and ar <= 3:
        ctx.append("TITLE CLASH: both top-3, expect tight game")
    if hr >= 16 and ar >= 16:
        ctx.append("RELEGATION 6-POINTER: both desperate")
    if hr <= 6 and ar >= 14:
        ctx.append("TOP vs BOTTOM: upset trap possible")
    if ar <= 6 and hr >= 14:
        ctx.append("BOTTOM vs TOP: home underdog fire")
    if abs(hr - ar) <= 3 and hr <= 10:
        ctx.append("SIMILAR STRENGTH: draw risk higher")
    return ctx


def get_crs_top5(m):
    v2 = m.get("v2_odds_dict", {})
    if not v2:
        v2 = m
    crs_map = {"w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1",
               "s00":"0-0","s11":"1-1","s22":"2-2",
               "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3"}
    crs = []
    for k, sc in crs_map.items():
        ov = float(v2.get(k, 0) or 0)
        if ov > 1:
            crs.append((sc, ov, round(1/ov*100, 1)))
    crs.sort(key=lambda x: x[2], reverse=True)
    return crs[:5]


def parse_score(s):
    try:
        p = str(s).split("-")
        return int(p[0]), int(p[1])
    except:
        return None, None


# ====================================================================
# AI 1: GPT - Tactical & Probability Analyst
# Gets: baseface, odds, form, stats, CRS odds, league context
# ====================================================================
def build_gpt_prompt(m):
    h, a = m.get("home_team", "?"), m.get("away_team", "?")
    lg = m.get("league", "?")
    sp_h, sp_d, sp_a = m.get("sp_home", 0), m.get("sp_draw", 0), m.get("sp_away", 0)
    hs = m.get("home_stats", {})
    ast = m.get("away_stats", {})
    p = "[ROLE] Top tactical analyst. Expert in logic deduction and probability.\n"
    p += "[TASK] Analyze [%s] vs [%s] using chain-of-thought reasoning.\n" % (h, a)
    p += "[RULES] Independent thinking. No fabrication. Output exact score as JSON.\n\n"
    p += "[MATCH] %s | %s\n" % (lg, m.get("match_num", "?"))
    p += "[ODDS] H=%s D=%s A=%s | HC=%s\n" % (sp_h, sp_d, sp_a, m.get("give_ball", "?"))
    ctx = detect_league_context(m)
    if ctx:
        p += "[CONTEXT] %s\n" % " | ".join(ctx)
    baseface = str(m.get("baseface", "")).strip()
    if baseface:
        p += "\n[TACTICAL ANALYSIS]\n%s\n" % baseface[:500]
    p += "\n[HOME %s] Rank#%s\n" % (h, m.get("home_rank", "?"))
    p += "%sP %sW %sD %sL GF=%s GA=%s AvgGF=%s AvgGA=%s\n" % (
        hs.get("played","?"), hs.get("wins","?"), hs.get("draws","?"), hs.get("losses","?"),
        hs.get("goals_for","?"), hs.get("goals_against","?"),
        hs.get("avg_goals_for","?"), hs.get("avg_goals_against","?"))
    p += "Form: %s\n" % hs.get("form", "?")
    p += "\n[AWAY %s] Rank#%s\n" % (a, m.get("away_rank", "?"))
    p += "%sP %sW %sD %sL GF=%s GA=%s AvgGF=%s AvgGA=%s\n" % (
        ast.get("played","?"), ast.get("wins","?"), ast.get("draws","?"), ast.get("losses","?"),
        ast.get("goals_for","?"), ast.get("goals_against","?"),
        ast.get("avg_goals_for","?"), ast.get("avg_goals_against","?"))
    p += "Form: %s\n" % ast.get("form", "?")
    had = m.get("had_analyse", [])
    if had:
        p += "[OFFICIAL TIP] %s\n" % ",".join(str(x) for x in had)
    crs = get_crs_top5(m)
    if crs:
        p += "\n[CRS MARKET TOP5]\n"
        for sc, od, prob in crs:
            p += "  %s odds=%.2f prob=%.1f%%\n" % (sc, od, prob)
    p += "\n[THINK STEP BY STEP]\n"
    p += "1. Calculate implied prob from SP odds\n"
    p += "2. Use xG concept: home must attack, where are defensive gaps?\n"
    p += "3. Evaluate handicap depth vs true strength gap\n"
    p += "4. CRS TOP1-2 as anchor, adjust with tactical logic\n"
    p += "5. Give 3 most likely scores with probability distribution (sum=100%%)\n"
    p += "6. Pick the TOP1 score as your final answer\n\n"
    p += "[OUTPUT] Pure JSON only:\n"
    p += '{"ai_score":"2-1","analysis":"80 char tactical reasoning"}'
    return p


# ====================================================================
# AI 2: Grok - Contrarian Upset Hunter
# Gets: vote data, odds movement, public sentiment, trap signals
# ====================================================================
def build_grok_prompt(m):
    h, a = m.get("home_team", "?"), m.get("away_team", "?")
    lg = m.get("league", "?")
    sp_h, sp_d, sp_a = m.get("sp_home", 0), m.get("sp_draw", 0), m.get("sp_away", 0)
    intel = m.get("intelligence", {})
    p = "[ROLE] Contrarian data scout. You find market blind spots and upset traps.\n"
    p += "[TASK] Challenge consensus on [%s] vs [%s]. Find the upset angle.\n" % (h, a)
    p += "[STYLE] Direct, sharp, slightly provocative. Like a gambler who sees what others miss.\n"
    p += "[RULES] Independent. No fabrication. Output exact score as JSON.\n\n"
    p += "[MATCH] %s | %s\n" % (lg, m.get("match_num", "?"))
    p += "[ODDS] H=%s D=%s A=%s | HC=%s\n" % (sp_h, sp_d, sp_a, m.get("give_ball", "?"))
    vote = m.get("vote", {})
    if vote:
        p += "[PUBLIC VOTE] W=%s%% D=%s%% L=%s%%\n" % (vote.get("win","?"), vote.get("same","?"), vote.get("lose","?"))
    change = m.get("change", {})
    if change and isinstance(change, dict):
        p += "[ODDS MOVEMENT] win=%s draw=%s lose=%s\n" % (change.get("win","0"), change.get("same","0"), change.get("lose","0"))
    h_good = str(intel.get("home_good_news", "")).strip()
    h_bad = str(intel.get("home_bad_news", "")).strip()
    g_good = str(intel.get("guest_good_news", "")).strip()
    g_bad = str(intel.get("guest_bad_news", "")).strip()
    if h_good: p += "\n[HOME GOOD] %s\n" % h_good[:200]
    if h_bad: p += "[HOME BAD] %s\n" % h_bad[:200]
    if g_good: p += "[AWAY GOOD] %s\n" % g_good[:200]
    if g_bad: p += "[AWAY BAD] %s\n" % g_bad[:200]
    p += "\n[INJURIES]\n"
    p += "H: %s\n" % str(intel.get("h_inj", intel.get("home_injury", "?")))[:150]
    p += "A: %s\n" % str(intel.get("g_inj", intel.get("guest_injury", "?")))[:150]
    crs = get_crs_top5(m)
    if crs:
        p += "\n[CRS TOP5]\n"
        for sc, od, prob in crs:
            p += "  %s odds=%.2f prob=%.1f%%\n" % (sc, od, prob)
    p += "\n[CONTRARIAN ANALYSIS]\n"
    p += "1. Is the favorite overvalued? Public >55%% on one side + odds stable = TRAP?\n"
    p += "2. Key injury impact: is it a real blow or smoke screen?\n"
    p += "3. If underdog parks the bus, can favorite break through?\n"
    p += "4. Predict the most likely UPSET scenario (e.g. away win, 0-0, red card)\n"
    p += "5. Give your exact score prediction - be bold if data supports it\n\n"
    p += "[OUTPUT] Pure JSON only:\n"
    p += '{"ai_score":"1-2","analysis":"80 char contrarian reasoning"}'
    return p


# ====================================================================
# AI 3: Gemini - Multi-dimensional Data Integrator
# Gets: full stats, lineups, H2H, match points analysis
# ====================================================================
def build_gemini_prompt(m):
    h, a = m.get("home_team", "?"), m.get("away_team", "?")
    lg = m.get("league", "?")
    sp_h, sp_d, sp_a = m.get("sp_home", 0), m.get("sp_draw", 0), m.get("sp_away", 0)
    hs = m.get("home_stats", {})
    ast = m.get("away_stats", {})
    intel = m.get("intelligence", {})
    p = "[ROLE] Data integration specialist. Turn raw data into structured match verdict.\n"
    p += "[TASK] Score both teams on 5 dimensions, then predict [%s] vs [%s].\n" % (h, a)
    p += "[RULES] Independent. No fabrication. Output exact score as JSON.\n\n"
    p += "[MATCH] %s | %s\n" % (lg, m.get("match_num", "?"))
    p += "[ODDS] H=%s D=%s A=%s\n" % (sp_h, sp_d, sp_a)
    ctx = detect_league_context(m)
    if ctx:
        p += "[CONTEXT] %s\n" % " | ".join(ctx)
    p += "\n[HOME %s] Rank#%s\n" % (h, m.get("home_rank", "?"))
    p += "%sP %sW %sD %sL GF=%s GA=%s AvgGF=%s AvgGA=%s CS=%s\n" % (
        hs.get("played","?"), hs.get("wins","?"), hs.get("draws","?"), hs.get("losses","?"),
        hs.get("goals_for","?"), hs.get("goals_against","?"),
        hs.get("avg_goals_for","?"), hs.get("avg_goals_against","?"), hs.get("clean_sheets","?"))
    p += "Form: %s\n" % hs.get("form", "?")
    p += "\n[AWAY %s] Rank#%s\n" % (a, m.get("away_rank", "?"))
    p += "%sP %sW %sD %sL GF=%s GA=%s AvgGF=%s AvgGA=%s CS=%s\n" % (
        ast.get("played","?"), ast.get("wins","?"), ast.get("draws","?"), ast.get("losses","?"),
        ast.get("goals_for","?"), ast.get("goals_against","?"),
        ast.get("avg_goals_for","?"), ast.get("avg_goals_against","?"), ast.get("clean_sheets","?"))
    p += "Form: %s\n" % ast.get("form", "?")
    p += "\n[INJURIES] H: %s | A: %s\n" % (str(intel.get("h_inj","?"))[:100], str(intel.get("g_inj","?"))[:100])
    h2h = m.get("h2h", [])
    if h2h:
        p += "\n[H2H]\n"
        for x in h2h[:5]:
            p += "%s %s %s %s\n" % (x.get("date",""), x.get("home",""), x.get("score",""), x.get("away",""))
    points = str(m.get("match_points", m.get("expert_intro", ""))).strip()
    if points:
        p += "\n[EXPERT POINTS]\n%s\n" % points[:400]
    crs = get_crs_top5(m)
    if crs:
        p += "\n[CRS TOP5]\n"
        for sc, od, prob in crs:
            p += "  %s odds=%.2f prob=%.1f%%\n" % (sc, od, prob)
    p += "\n[SCORING DIMENSIONS] Rate each team 1-10:\n"
    p += "- Attack power (recent goals)\n"
    p += "- Defense stability (recent conceded + injuries)\n"
    p += "- Home/Away strength\n"
    p += "- Motivation (title/relegation/knockout)\n"
    p += "- Squad completeness (injuries impact)\n"
    p += "Then predict exact score based on your scoring.\n\n"
    p += "[OUTPUT] Pure JSON only:\n"
    p += '{"ai_score":"2-1","analysis":"80 char data-driven verdict"}'
    return p


# ====================================================================
# AI 4: Claude Phase-1 - Psychology & Narrative Analyst
# Gets: good/bad news, expert intro, emotional factors, H2H psychology
# ====================================================================
def build_claude_analyst_prompt(m):
    h, a = m.get("home_team", "?"), m.get("away_team", "?")
    lg = m.get("league", "?")
    sp_h, sp_d, sp_a = m.get("sp_home", 0), m.get("sp_draw", 0), m.get("sp_away", 0)
    intel = m.get("intelligence", {})
    p = "[ROLE] Football psychology expert. You see what data cannot show.\n"
    p += "[TASK] Analyze the human factors behind [%s] vs [%s].\n" % (h, a)
    p += "[RULES] Independent. No fabrication. Output exact score as JSON.\n\n"
    p += "[MATCH] %s | %s\n" % (lg, m.get("match_num", "?"))
    p += "[ODDS] H=%s D=%s A=%s\n" % (sp_h, sp_d, sp_a)
    ctx = detect_league_context(m)
    if ctx:
        p += "[CONTEXT] %s\n" % " | ".join(ctx)
    baseface = str(m.get("baseface", "")).strip()
    if baseface:
        p += "\n[SITUATION] %s\n" % baseface[:400]
    intro = str(m.get("expert_intro", "")).strip()
    if intro:
        p += "[EXPERT VIEW] %s\n" % intro[:250]
    h_good = str(intel.get("home_good_news", "")).strip()
    h_bad = str(intel.get("home_bad_news", "")).strip()
    g_good = str(intel.get("guest_good_news", "")).strip()
    g_bad = str(intel.get("guest_bad_news", "")).strip()
    if h_good: p += "\n[HOME POSITIVE] %s\n" % h_good[:250]
    if h_bad: p += "[HOME NEGATIVE] %s\n" % h_bad[:250]
    if g_good: p += "[AWAY POSITIVE] %s\n" % g_good[:250]
    if g_bad: p += "[AWAY NEGATIVE] %s\n" % g_bad[:250]
    h2h = m.get("h2h", [])
    if h2h:
        p += "\n[H2H HISTORY]\n"
        for x in h2h[:5]:
            p += "%s %s %s %s\n" % (x.get("date",""), x.get("home",""), x.get("score",""), x.get("away",""))
    p += "\n[PSYCHOLOGY ANALYSIS]\n"
    p += "1. Mental state: crisis team vs confident team - who handles pressure better?\n"
    p += "2. H2H psychology: historical dominance or first-time fear?\n"
    p += "3. Match flow: if home leads early, what happens? If away scores first?\n"
    p += "4. Motivation asymmetry: title chase vs mid-table vs relegation fight\n"
    p += "5. Based on psychology, predict the most likely exact score\n\n"
    p += "[OUTPUT] Pure JSON only:\n"
    p += '{"ai_score":"1-0","analysis":"80 char psychological verdict"}'
    return p


# ====================================================================
# AI 5: Claude Arbiter - Supreme Judge (Phase-2)
# Gets: ALL 4 AI outputs + market data + signals
# ====================================================================
def build_arbiter_prompt(m, gpt_res, grok_res, gemini_res, claude_p1_res, stats_pred):
    h, a = m.get("home_team", "?"), m.get("away_team", "?")
    sp_h, sp_d, sp_a = m.get("sp_home", 0), m.get("sp_draw", 0), m.get("sp_away", 0)
    p = "[ROLE] Supreme Sports Director. 4 expert reports on your desk. Make the FINAL call.\n"
    p += "[TASK] Final exact score for [%s] vs [%s].\n" % (h, a)
    p += "[CRITICAL] You must weigh all 4 experts and market data. Do NOT blindly copy anyone.\n\n"
    p += "[EXPERT REPORTS] (Grok=40%% GPT=35%% Gemini=15%% Claude-Analyst=10%%)\n"
    p += "Expert 1 GPT(35%%) Tactical: %s | %s\n" % (gpt_res.get("ai_score","?"), str(gpt_res.get("analysis","?"))[:120])
    p += "Expert 2 Grok(40%%) Contrarian: %s | %s\n" % (grok_res.get("ai_score","?"), str(grok_res.get("analysis","?"))[:120])
    p += "Expert 3 Gemini(15%%) Data: %s | %s\n" % (gemini_res.get("ai_score","?"), str(gemini_res.get("analysis","?"))[:120])
    p += "Expert 4 Claude(10%%) Psychology: %s | %s\n\n" % (claude_p1_res.get("ai_score","?"), str(claude_p1_res.get("analysis","?"))[:120])
    ctx = detect_league_context(m)
    if ctx:
        p += "[CONTEXT] %s\n" % " | ".join(ctx)
    crs = stats_pred.get("crs_analysis", {}).get("top_scores", [])[:5]
    if crs:
        p += "[MARKET CRS TOP5]\n"
        for s in crs:
            p += "  %s %.1f%%\n" % (s.get("score","?"), s.get("prob",0))
    p += "\n[ODDS] H=%s D=%s A=%s HC=%s\n" % (sp_h, sp_d, sp_a, m.get("give_ball","?"))
    smart = stats_pred.get("smart_signals", [])
    if smart:
        p += "[RISK SIGNALS] %s\n" % " | ".join(smart)
    baseface = str(m.get("baseface", "")).strip()
    if baseface:
        p += "[EXPERT] %s\n" % baseface[:200]
    p += "\n[VERDICT PROTOCOL]\n"
    p += "1. Find CONSENSUS: what do all/most experts agree on?\n"
    p += "2. Evaluate DISAGREEMENT: where do experts diverge? Why?\n"
    p += "3. WEIGHT: Grok(40%%)+GPT(35%%)=75%% combined, if they agree -> strong signal\n"
    p += "4. Context matters: knockout=tight, relegation=chaos, derby=unpredictable\n"
    p += "5. CRS market TOP1 = anchor point (unless all experts strongly disagree)\n"
    p += "6. Risk signals override everything\n"
    p += "7. Give ONE exact score. Explain why this beats alternatives in 100 chars.\n\n"
    p += "[OUTPUT] Pure JSON only:\n"
    p += '{"ai_score":"1-2","analysis":"100 char verdict with weight reasoning"}'
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
    if not url or not key:
        return {}
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
            {"role": "system", "content": "Pure JSON only. No markdown."},
            {"role": "user", "content": prompt}
        ], "temperature": 0.2, "max_tokens": 600}
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
    "https://api520.pro/v1",
    "https://www.api520.pro/v1",
    "https://api521.pro/v1",
    "https://www.api521.pro/v1",
    "https://api522.pro/v1",
    "https://www.api522.pro/v1",
    "https://69.63.213.33:666/v1",
]


def call_with_fallback(prompt, url_env, key_env, models_list):
    key = get_clean_env_key(key_env)
    primary_url = get_clean_env_url(url_env, globals().get(url_env, ""))
    if not key:
        return {}
    urls = [primary_url] + [u for u in FALLBACK_URLS if u and u != primary_url]
    for model_name in models_list:
        for url in urls:
            if not url: continue
            result = call_ai_model(prompt, url, key, model_name)
            if result and result.get("ai_score") and result["ai_score"] not in ["?", ""]:
                return result
            time.sleep(0.2)
    return {}


def call_gpt(prompt):
    return call_with_fallback(prompt, "GPT_API_URL", "GPT_API_KEY", [
        "\u718a\u732b-A-7-gpt-5.4",
        "\u718a\u732b-\u6309\u91cf-gpt-5.3-codex-\u6ee1\u8840",
        "\u718a\u732b-A-10-gpt-5.3-codex",
        "\u718a\u732b-A-1-gpt-5.2",
        "\u718a\u732b-A-5-gpt-5.2",
        "\u718a\u732b-A-8-deepseek-v3.2",
    ])

def call_grok(prompt):
    return call_with_fallback(prompt, "GROK_API_URL", "GROK_API_KEY", [
        "\u718a\u732b-A-7-grok-4.2-\u591a\u667a\u80fd\u4f53\u8ba8\u8bba",
        "\u718a\u732b-A-4-grok-4.2-fast",
    ])

def call_gemini(prompt):
    return call_with_fallback(prompt, "GEMINI_API_URL", "GEMINI_API_KEY", [
        "\u718a\u732b\u7279\u4f9bS-\u6309\u91cf-gemini-3-flash-preview",
        "\u718a\u732b\u7279\u4f9b-\u6309\u91cf-SSS-gemini-3.1-pro-preview",
        "\u718a\u732b-2-gemini-3.1-flash-lite-preview",
    ])

def call_claude(prompt):
    return call_with_fallback(prompt, "CLAUDE_API_URL", "CLAUDE_API_KEY", [
        "\u718a\u732b-\u6309\u91cf-\u9876\u7ea7\u7279\u4f9b-\u5b98max-claude-opus-4.6",
        "\u718a\u732b-\u6309\u91cf-\u9876\u7ea7\u7279\u4f9b-\u5b98max-claude-opus-4.6-thinking",
        "\u718a\u732b-\u6309\u91cf-\u7279\u4f9b\u9876\u7ea7-\u5b98\u65b9\u6b63\u5411\u6ee1\u8840-claude-opus-4.6",
        "\u718a\u732b-\u7279\u4f9b-\u6309\u91cf-Q-claude-opus-4.6",
        "\u718a\u732b-\u7279\u4f9b-\u6309\u91cf-Q-claude-opus-4.6-thinking",
        "\u718a\u732b-\u7279\u4f9b-\u6309\u91cf-Q-claude-sonnet-4.6",
        "\u718a\u732b-\u7279\u4ef7\u9006-19-claude-opus-4-6",
        "\u718a\u732b-\u7279\u4ef7\u9006-15-claude-sonnet-4-6",
    ])


def merge_all(gpt, grok, gemini, claude_p1, claude_arbiter, stats, match_obj):
    sp_h = float(match_obj.get("sp_home", 0) or 0)
    sp_d = float(match_obj.get("sp_draw", 0) or 0)
    sp_a = float(match_obj.get("sp_away", 0) or 0)
    sys_score = stats.get("predicted_score", "1-1")
    o25 = stats.get("over_2_5", 50)
    bt = stats.get("btts", 50)
    model_cons = stats.get("model_consensus", 0)
    total_mod = stats.get("total_models", 11)
    top_crs = stats.get("crs_analysis", {}).get("top_scores", [])[:5]
    smart = stats.get("smart_signals", [])
    extreme = stats.get("extreme_warning", "")

    if sp_h > 1 and sp_d > 1 and sp_a > 1:
        ih, id2, ia = 1/sp_h, 1/sp_d, 1/sp_a
        t = ih + id2 + ia
        odds_hp, odds_dp, odds_ap = ih/t*100, id2/t*100, ia/t*100
    else:
        odds_hp, odds_dp, odds_ap = 33, 33, 34

    candidates = {}
    ai_scores = []
    ai_w = {"grok": 0.40, "gpt": 0.35, "gemini": 0.15, "claude_p1": 0.10, "arbiter": 0.35}

    all_ai = [("gpt", gpt), ("grok", grok), ("gemini", gemini), ("claude_p1", claude_p1), ("arbiter", claude_arbiter)]
    for name, res in all_ai:
        score = res.get("ai_score") if isinstance(res, dict) else None
        if score and score not in ["-", "?", ""]:
            if score not in candidates:
                candidates[score] = {"weight": 0.0, "sources": []}
            candidates[score]["weight"] += ai_w.get(name, 0.15)
            candidates[score]["sources"].append(name)
            ai_scores.append(score)

    for s in top_crs[:3]:
        sc = s.get("score")
        if sc:
            if sc not in candidates:
                candidates[sc] = {"weight": 0.0, "sources": []}
            candidates[sc]["weight"] += 0.25
            candidates[sc]["sources"].append("market")

    has_warning = any("🚨" in str(sig) for sig in smart) or (extreme and extreme not in ["", "\u65e0"])
    market_top = top_crs[0].get("score") if top_crs else None

    for sc in candidates:
        if sc == market_top:
            candidates[sc]["weight"] += 0.25
        cnt = ai_scores.count(sc)
        if cnt >= 3:
            candidates[sc]["weight"] += 0.60
        elif cnt >= 2:
            candidates[sc]["weight"] += 0.35
        grok_sc = grok.get("ai_score") if isinstance(grok, dict) else None
        gpt_sc = gpt.get("ai_score") if isinstance(gpt, dict) else None
        if sc == grok_sc and sc == gpt_sc:
            candidates[sc]["weight"] += 0.30
        if has_warning and sc in [s.get("score") for s in top_crs]:
            candidates[sc]["weight"] += 0.15

    final_score = max(candidates.items(), key=lambda x: x[1]["weight"])[0] if candidates else sys_score

    ai_hp = ai_dp = ai_ap = 0.0
    ai_tw = 0.0
    for name, res in all_ai:
        score = res.get("ai_score") if isinstance(res, dict) else None
        if not score or score in ["-", "?", ""]: continue
        hg, ag = parse_score(score)
        if hg is None: continue
        w = ai_w.get(name, 0.15)
        ai_tw += w
        if hg > ag: ai_hp += w
        elif hg == ag: ai_dp += w
        else: ai_ap += w
    if ai_tw > 0:
        ai_hp = ai_hp / ai_tw * 100
        ai_dp = ai_dp / ai_tw * 100
        ai_ap = ai_ap / ai_tw * 100
    else:
        ai_hp, ai_dp, ai_ap = odds_hp, odds_dp, odds_ap

    fused_hp = ai_hp * 0.55 + odds_hp * 0.30 + stats.get("home_win_pct", 33) * 0.15
    fused_dp = ai_dp * 0.55 + odds_dp * 0.30 + stats.get("draw_pct", 33) * 0.15
    fused_ap = ai_ap * 0.55 + odds_ap * 0.30 + stats.get("away_win_pct", 33) * 0.15
    if fused_dp < 10:
        boost = 10 - fused_dp
        fused_dp = 10
        fused_hp -= boost * 0.6
        fused_ap -= boost * 0.4
    ft = fused_hp + fused_dp + fused_ap
    if ft > 0:
        fused_hp = round(fused_hp / ft * 100, 1)
        fused_dp = round(fused_dp / ft * 100, 1)
        fused_ap = round(100 - fused_hp - fused_dp, 1)

    unique_ai = len(set(ai_scores))
    agreement_bonus = 25 if unique_ai <= 1 and len(ai_scores) >= 3 else (15 if unique_ai <= 2 else 0)
    market_match = 18 if final_score == market_top else 0
    warning_penalty = -10 if has_warning else 0
    sys_cf = min(95, max(35, round(
        48 + agreement_bonus + market_match + warning_penalty + (max(fused_hp, fused_dp, fused_ap) - 33) * 0.3
    )))
    risk_str = "\u4f4e" if sys_cf >= 75 else ("\u4e2d" if sys_cf >= 55 else "\u9ad8")

    val_h = calculate_value_bet(fused_hp, sp_h)
    val_d = calculate_value_bet(fused_dp, sp_d)
    val_a = calculate_value_bet(fused_ap, sp_a)
    v_tags = []
    for k, v in zip(["\u4e3b\u80dc","\u5e73\u5c40","\u5ba2\u80dc"], [val_h, val_d, val_a]):
        if v and v.get("is_value"):
            v_tags.append("%s EV:+%s%% Kelly:%s%%" % (k, v["ev"], v["kelly"]))

    pcts = {"\u4e3b\u80dc": fused_hp, "\u5e73\u5c40": fused_dp, "\u5ba2\u80dc": fused_ap}
    result = max(pcts, key=pcts.get)

    seen = set()
    unique_smart = [s for s in smart if s not in seen and not seen.add(s)]

    return {
        "predicted_score": final_score,
        "home_win_pct": fused_hp, "draw_pct": fused_dp, "away_win_pct": fused_ap,
        "confidence": sys_cf, "result": result, "risk_level": risk_str,
        "over_under_2_5": "\u5927" if o25 > 55 else "\u5c0f",
        "both_score": "\u662f" if bt > 50 else "\u5426",
        "gpt_score": gpt.get("ai_score", "-"), "gpt_analysis": gpt.get("analysis", "N/A"),
        "grok_score": grok.get("ai_score", "-"), "grok_analysis": grok.get("analysis", "N/A"),
        "gemini_score": gemini.get("ai_score", "-"), "gemini_analysis": gemini.get("analysis", "N/A"),
        "claude_score": claude_arbiter.get("ai_score", "-"), "claude_analysis": claude_arbiter.get("analysis", "N/A"),
        "model_agreement": unique_ai <= 1 if ai_scores else False,
        "candidate_scores": {sc: round(c["weight"], 2) for sc, c in candidates.items()},
        "poisson": {**stats.get("poisson", {}), "home_expected_goals": stats.get("poisson", {}).get("home_xg", "?")},
        "refined_poisson": stats.get("refined_poisson", {}),
        "value_bets_summary": v_tags,
        "extreme_warning": extreme if extreme else "\u65e0",
        "smart_money_signal": " | ".join(unique_smart) if unique_smart else "\u6b63\u5e38",
        "smart_signals": unique_smart,
        "model_consensus": model_cons, "total_models": total_mod,
        "expected_total_goals": stats.get("expected_total_goals", 2.5),
        "top_scores": stats.get("refined_poisson", {}).get("top_scores", []),
        "elo": stats.get("elo", {}), "random_forest": stats.get("random_forest", {}),
        "gradient_boost": stats.get("gradient_boost", {}), "neural_net": stats.get("neural_net", {}),
        "logistic": stats.get("logistic", {}), "svm": stats.get("svm", {}), "knn": stats.get("knn", {}),
        "dixon_coles": stats.get("dixon_coles", {}), "bradley_terry": stats.get("bradley_terry", {}),
        "home_form": stats.get("home_form", {}), "away_form": stats.get("away_form", {}),
        "handicap_signal": stats.get("handicap_signal", ""),
        "odds_movement": stats.get("odds_movement", {}),
        "vote_analysis": stats.get("vote_analysis", {}),
        "h2h_blood": stats.get("h2h_blood", {}),
        "crs_analysis": stats.get("crs_analysis", {}),
        "ttg_analysis": stats.get("ttg_analysis", {}),
        "halftime": stats.get("halftime", {}),
        "over_2_5": o25, "btts": bt,
        "pace_rating": stats.get("pace_rating", ""),
        "kelly_home": stats.get("kelly_home", {}),
        "kelly_away": stats.get("kelly_away", {}),
        "odds": stats.get("odds", {}),
    }


def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})
        s = pr.get("confidence", 0) * 0.35
        mx = max(pr.get("home_win_pct", 33), pr.get("away_win_pct", 33), pr.get("draw_pct", 33))
        s += (mx - 33) * 0.25
        s += pr.get("model_consensus", 0) * 2.5
        if pr.get("risk_level") == "\u4f4e": s += 10
        elif pr.get("risk_level") == "\u9ad8": s -= 5
        if pr.get("model_agreement"): s += 15
        if pr.get("value_bets_summary"): s += 8
        p["recommend_score"] = round(s, 2)
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]


def extract_num(match_str):
    week_map = {"\u4e00":1000,"\u4e8c":2000,"\u4e09":3000,"\u56db":4000,
                "\u4e94":5000,"\u516d":6000,"\u65e5":7000,"\u5929":7000}
    base = next((v for k, v in week_map.items() if k in str(match_str)), 0)
    nums = re.findall(r"\d+", str(match_str))
    return base + int(nums[0]) if nums else 9999


def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 60)
    print("  ENGINE v7.0 | %d matches" % len(ms))
    print("  5-AI: GPT-Tactical Grok-Contrarian Gemini-Data")
    print("        Claude-Psychology Claude-Arbiter")
    print("=" * 60)
    res = []
    for i, m in enumerate(ms):
        h, a = m.get("home_team", "?"), m.get("away_team", "?")
        print("\n" + "-" * 50)
        print("  [%d/%d] %s %s vs %s" % (i+1, len(ms), m.get("league",""), h, a))
        print("-" * 50)
        sp = ensemble.predict(m, {})
        print("    Stats: H%.1f D%.1f A%.1f" % (sp["home_win_pct"], sp["draw_pct"], sp["away_win_pct"]))
        if use_ai:
            print("  Phase-1: 4 specialized experts...")
            gpt_res = call_gpt(build_gpt_prompt(m)); time.sleep(0.3)
            grok_res = call_grok(build_grok_prompt(m)); time.sleep(0.3)
            gemini_res = call_gemini(build_gemini_prompt(m)); time.sleep(0.3)
            claude_p1 = call_claude(build_claude_analyst_prompt(m)); time.sleep(0.3)
            print("  Phase-2: Claude Arbiter...")
            arbiter_prompt = build_arbiter_prompt(m, gpt_res or {}, grok_res or {}, gemini_res or {}, claude_p1 or {}, sp)
            claude_arb = call_claude(arbiter_prompt); time.sleep(0.3)
        else:
            blocked = {"ai_score": "-", "analysis": "blocked"}
            gpt_res = grok_res = gemini_res = claude_p1 = claude_arb = blocked
        mg = merge_all(gpt_res or {}, grok_res or {}, gemini_res or {}, claude_p1 or {}, claude_arb or {}, sp, m)
        print("  => %s (%s) %d%%" % (mg["result"], mg["predicted_score"], mg["confidence"]))
        print("     GPT:%s Grok:%s Gem:%s Claude:%s Arb:%s" % (
            mg["gpt_score"], mg["grok_score"], mg["gemini_score"],
            (claude_p1 or {}).get("ai_score","-"), mg["claude_score"]))
        res.append({**m, "prediction": mg})
    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res: r["is_recommended"] = r["id"] in t4ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    print("\n" + "=" * 60)
    print("  TOP4:")
    for i, t in enumerate(t4):
        pr = t.get("prediction", {})
        print("    %d. %s vs %s => %s (%s) %d%%" % (
            i+1, t.get("home_team"), t.get("away_team"),
            pr.get("result"), pr.get("predicted_score"), pr.get("confidence",0)))
    print("=" * 60)
    return res, t4