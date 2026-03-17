import json, requests, time, re, os
from config import *
from models import EnsemblePredictor

ensemble = EnsemblePredictor()

def calculate_value_bet(prob_pct, odds):
if not odds or odds <= 1.05:
return {“ev”: 0.0, “kelly”: 0.0, “is_value”: False}
prob = prob_pct / 100.0
ev = (prob * odds) - 1.0
b = odds - 1.0
q = 1.0 - prob
if b <= 0:
return {“ev”: round(ev * 100, 2), “kelly”: 0.0, “is_value”: False}
kelly = ((b * prob) - q) / b
return {“ev”: round(ev * 100, 2), “kelly”: round(max(0.0, kelly * 0.25) * 100, 2), “is_value”: ev > 0.05}

def detect_league_context(m):
hr = int(m.get(“home_rank”, 10) or 10)
ar = int(m.get(“away_rank”, 10) or 10)
lg = str(m.get(“league”, “”))
baseface = str(m.get(“baseface”, “”))
ctx = []
if hr <= 3 or ar <= 3:
ctx.append(“TITLE RACE: top-3 team involved, high motivation”)
if hr >= 16 or ar >= 16:
ctx.append(“RELEGATION BATTLE: bottom-5 team fighting for survival, expect desperate play”)
if hr <= 3 and ar <= 3:
ctx.append(“TITLE CLASH: both teams in top-3, expect tactical and tight game, low goals likely”)
if hr >= 16 and ar >= 16:
ctx.append(“RELEGATION SIX-POINTER: both teams desperate, chaotic game likely”)
if hr <= 6 and ar >= 14:
ctx.append(“TOP vs BOTTOM: strong favorite but upset trap possible if odds are shallow”)
if ar <= 6 and hr >= 14:
ctx.append(“BOTTOM vs TOP: home underdog with relegation fire, do not ignore home advantage”)
if abs(hr - ar) <= 3 and hr <= 10 and ar <= 10:
ctx.append(“MID-TABLE DERBY: similar strength, draw probability higher than normal”)
if “\u4fdd\u7ea7” in baseface or “\u964d\u7ea7” in baseface:
ctx.append(“RELEGATION CONTEXT detected in analysis”)
if “\u4e89\u51a0” in baseface or “\u7b2c\u4e00” in baseface:
ctx.append(“TITLE CONTEXT detected in analysis”)
return ctx

def build_scout_prompt(m):
h, a = m.get(“home_team”, “?”), m.get(“away_team”, “?”)
lg = m.get(“league”, “?”)
sp_h, sp_d, sp_a = m.get(“sp_home”, 0), m.get(“sp_draw”, 0), m.get(“sp_away”, 0)
intel = m.get(“intelligence”, {})
hs = m.get(“home_stats”, {})
ast = m.get(“away_stats”, {})
p = “”
p += “[SYSTEM] Football quant analyst. Independent verdict only.\n”
p += “[TASK] Predict exact score for [%s] vs [%s].\n” % (h, a)
p += “[CRITICAL RULES]\n”
p += “- You must think independently. Do NOT rely on any pre-computed model output.\n”
p += “- Use ONLY the raw match data below to form your own judgment.\n”
p += “- No fabrication. No player names. Pure JSON output. No Markdown.\n\n”
p += “[MATCH] %s | %s\n” % (lg, m.get(“match_num”, “?”))
p += “[ODDS] H=%s D=%s A=%s | HC=%s\n” % (sp_h, sp_d, sp_a, m.get(“give_ball”, “?”))
p += “[MOVEMENT] %s\n” % m.get(“odds_movement”, “?”)
league_ctx = detect_league_context(m)
if league_ctx:
p += “\n[LEAGUE CONTEXT]\n”
for c in league_ctx:
p += “- %s\n” % c
p += “\n[HOME %s] Rank#%s\n” % (h, m.get(“home_rank”, “?”))
p += “%sP %sW %sD %sL GF=%s GA=%s AvgGF=%s AvgGA=%s CS=%s\n” % (
hs.get(“played”,”?”), hs.get(“wins”,”?”), hs.get(“draws”,”?”), hs.get(“losses”,”?”),
hs.get(“goals_for”,”?”), hs.get(“goals_against”,”?”),
hs.get(“avg_goals_for”,”?”), hs.get(“avg_goals_against”,”?”), hs.get(“clean_sheets”,”?”))
p += “Form: %s\n” % hs.get(“form”, “?”)
p += “Injuries: %s\n” % intel.get(“h_inj”, “?”)
p += “\n[AWAY %s] Rank#%s\n” % (a, m.get(“away_rank”, “?”))
p += “%sP %sW %sD %sL GF=%s GA=%s AvgGF=%s AvgGA=%s CS=%s\n” % (
ast.get(“played”,”?”), ast.get(“wins”,”?”), ast.get(“draws”,”?”), ast.get(“losses”,”?”),
ast.get(“goals_for”,”?”), ast.get(“goals_against”,”?”),
ast.get(“avg_goals_for”,”?”), ast.get(“avg_goals_against”,”?”), ast.get(“clean_sheets”,”?”))
p += “Form: %s\n” % ast.get(“form”, “?”)
p += “Injuries: %s\n” % intel.get(“g_inj”, “?”)
h2h = m.get(“h2h”, [])
if h2h:
p += “\n[H2H]\n”
for x in h2h[:5]:
p += “%s %s %s %s\n” % (x.get(“date”,””), x.get(“home”,””), x.get(“score”,””), x.get(“away”,””))
baseface = str(m.get(“baseface”, “”)).strip()
if baseface:
p += “\n[EXPERT ANALYSIS]\n%s\n” % baseface[:350]
had = m.get(“had_analyse”, [])
if had:
p += “[OFFICIAL TIP] %s\n” % “,”.join(str(x) for x in had)
intro = str(m.get(“expert_intro”, “”)).strip()
if intro:
p += “[INTEL] %s\n” % intro[:250]
vote = m.get(“vote”, {})
if vote:
p += “[PUBLIC] W=%s%% D=%s%% L=%s%%\n” % (vote.get(“win”,”?”), vote.get(“same”,”?”), vote.get(“lose”,”?”))
v2 = m.get(“v2_odds_dict”, {})
if v2:
crs_map = {“w10”:“1-0”,“w20”:“2-0”,“w21”:“2-1”,“w30”:“3-0”,“w31”:“3-1”,
“s00”:“0-0”,“s11”:“1-1”,“s22”:“2-2”,
“l01”:“0-1”,“l02”:“0-2”,“l12”:“1-2”,“l03”:“0-3”,“l13”:“1-3”}
crs_probs = []
for k, sc in crs_map.items():
ov = float(v2.get(k, 0) or 0)
if ov > 1:
crs_probs.append((sc, ov, round(1/ov*100, 1)))
crs_probs.sort(key=lambda x: x[2], reverse=True)
if crs_probs:
p += “\n[CRS ODDS TOP5]\n”
for sc, od, prob in crs_probs[:5]:
p += “  %s odds=%.2f prob=%.1f%%\n” % (sc, od, prob)
p += “\n[THINK STEP BY STEP]\n”
p += “1. Calculate implied probability from SP odds\n”
p += “2. Compare with public vote - if public >55%% on one side but odds stable = trap\n”
p += “3. Evaluate home advantage (+0.3~0.5 goals in most leagues)\n”
p += “4. Injury impact: key player out = -0.3~0.5 expected goals\n”
p += “5. Form momentum: 3+ consecutive W/L matters\n”
p += “6. H2H patterns: some teams consistently beat others\n”
p += “7. League context: title/relegation matches have different dynamics\n”
p += “8. CRS odds TOP1-2 represent market consensus - use as anchor\n”
p += “9. Give exact score with 3 supporting reasons + 1 risk factor\n\n”
p += “[OUTPUT] Pure JSON:\n”
p += ‘{“ai_score”:“1-2”,“analysis”:“100 char analysis with reasons”}’
return p

def build_commander_prompt(m, gpt_res, grok_res, gemini_res, stats_pred):
h, a = m.get(“home_team”, “?”), m.get(“away_team”, “?”)
sp_h, sp_d, sp_a = m.get(“sp_home”, 0), m.get(“sp_draw”, 0), m.get(“sp_away”, 0)
p = “”
p += “[SYSTEM] Supreme Match Arbiter. Your verdict is FINAL and cannot be overturned.\n”
p += “[TASK] Final exact score for [%s] vs [%s].\n” % (h, a)
p += “[CRITICAL] You must make YOUR OWN judgment. Do NOT blindly follow any single AI.\n\n”
p += “[3-WAY AI RECON] (weighted: Grok 40%% GPT 35%% Gemini 25%%)\n”
p += “GPT (35%%): score=%s | %s\n” % (gpt_res.get(“ai_score”,“none”), str(gpt_res.get(“analysis”,“none”))[:120])
p += “Grok (40%%): score=%s | %s\n” % (grok_res.get(“ai_score”,“none”), str(grok_res.get(“analysis”,“none”))[:120])
p += “Gemini (25%%): score=%s | %s\n\n” % (gemini_res.get(“ai_score”,“none”), str(gemini_res.get(“analysis”,“none”))[:120])
league_ctx = detect_league_context(m)
if league_ctx:
p += “[LEAGUE CONTEXT]\n”
for c in league_ctx:
p += “- %s\n” % c
p += “\n”
crs = stats_pred.get(“crs_analysis”, {})
top_crs = crs.get(“top_scores”, [])[:5]
if top_crs:
p += “[MARKET CRS TOP5]\n”
for s in top_crs:
p += “  %s %.1f%% odds=%s\n” % (s.get(“score”,”?”), s.get(“prob”,0), s.get(“odds”,”?”))
p += “\n”
p += “[RAW ODDS] H=%s D=%s A=%s | HC=%s\n” % (sp_h, sp_d, sp_a, m.get(“give_ball”,”?”))
intel = m.get(“intelligence”, {})
p += “[INJURIES] H: %s | A: %s\n” % (str(intel.get(“h_inj”,”?”))[:80], str(intel.get(“g_inj”,”?”))[:80])
smart = stats_pred.get(“smart_signals”, [])
if smart:
p += “[RISK SIGNALS] %s\n” % “ | “.join(smart)
baseface = str(m.get(“baseface”, “”)).strip()
if baseface:
p += “[EXPERT] %s\n” % baseface[:200]
p += “\n[VERDICT PROTOCOL]\n”
p += “1. Grok weight=40%%, GPT weight=35%%, Gemini weight=25%%\n”
p += “2. If Grok+GPT agree (75%% combined) -> strong adopt\n”
p += “3. If all 3 AI agree -> lock immediately\n”
p += “4. CRS market TOP1 is the anchor point\n”
p += “5. If AI consensus contradicts market TOP1 -> trust AI but explain why\n”
p += “6. League context matters: title race = tight/low goals, relegation = chaotic\n”
p += “7. Risk signals override everything\n”
p += “8. Give ONE exact score. Explain in 100 chars why this beats alternatives.\n\n”
p += “[OUTPUT] Pure JSON:\n”
p += ‘{“ai_score”:“1-2”,“analysis”:“100 char verdict with weight reasoning”}’
return p

def extract_clean_json(text):
text = str(text or “”).strip()
fallback_score = “?”
fallback_analysis = “format error”
s_match = re.search(r’“ai_score”\s*:\s*”([^”]+)”’, text)
if s_match:
fallback_score = s_match.group(1)
a_match = re.search(r’“analysis”\s*:\s*”(.*?)”’, text, re.DOTALL)
if a_match:
fallback_analysis = a_match.group(1).replace(’”’, “’”).replace(”\n”, “ “).strip()[:150]
start, end = text.find(”{”), text.rfind(”}”)
if start != -1 and end != -1:
try:
return json.loads(text[start:end+1])
except Exception:
pass
cleaned = re.sub(r”```\w*”, “”, text).strip()
s2, e2 = cleaned.find(”{”), cleaned.rfind(”}”)
if s2 != -1 and e2 != -1:
try:
return json.loads(cleaned[s2:e2+1])
except Exception:
pass
if fallback_score != “?”:
return {“ai_score”: fallback_score, “analysis”: fallback_analysis}
return None

def get_clean_env_url(name, default=””):
v = os.environ.get(name, globals().get(name, default))
v = str(v).strip(” \t\n\r"’”)
match = re.search(r”(https?://[a-zA-Z0-9._:/-]+)”, v)
return match.group(1) if match else v

def get_clean_env_key(name):
return str(os.environ.get(name, globals().get(name, “”))).strip(” \t\n\r"’”)

def call_ai_model(prompt, url, key, model_name):
if not url or not key:
return {}
is_native_gemini = “generateContent” in url
if not is_native_gemini and “chat/completions” not in url:
url = url.rstrip(”/”) + “/chat/completions”
headers = {“Content-Type”: “application/json”}
if is_native_gemini:
headers[“x-goog-api-key”] = key
payload = {“contents”: [{“parts”: [{“text”: prompt}]}], “generationConfig”: {“temperature”: 0.2}}
else:
headers[“Authorization”] = “Bearer “ + key
payload = {
“model”: model_name,
“messages”: [
{“role”: “system”, “content”: “Pure JSON output only. No markdown.”},
{“role”: “user”, “content”: prompt}
],
“temperature”: 0.2,
“max_tokens”: 500
}
gw = url.split(”/v1”)[0] if “/v1” in url else url[:35]
print(”    > %s @ %s” % (model_name, gw))
try:
r = requests.post(url, headers=headers, json=payload, timeout=60)
if r.status_code == 200:
if is_native_gemini:
t = r.json()[“candidates”][0][“content”][“parts”][0][“text”].strip()
else:
t = r.json()[“choices”][0][“message”][“content”].strip()
parsed = extract_clean_json(t)
if parsed:
parsed[“analysis”] = str(parsed.get(“analysis”, “”)).replace(”`json", "").replace("`”, “”).strip()
print(”    OK %s: %s” % (model_name, parsed.get(“ai_score”, “?”)))
return parsed
else:
print(”    WARN %s: parse fail” % model_name)
else:
print(”    ERR %s: HTTP %d” % (model_name, r.status_code))
except Exception as e:
print(”    ERR %s: %s” % (model_name, str(e)[:50]))
return {}

FALLBACK_URLS = [
None,
“https://api520.pro/v1”,
“https://www.api520.pro/v1”,
“https://api521.pro/v1”,
“https://www.api521.pro/v1”,
“https://api522.pro/v1”,
“https://www.api522.pro/v1”,
“https://69.63.213.33:666/v1”,
]

def call_with_fallback(prompt, url_env, key_env, models_list):
key = get_clean_env_key(key_env)
primary_url = get_clean_env_url(url_env, globals().get(url_env, “”))
if not key:
print(”    X %s: no key” % key_env)
return {}
urls = [primary_url] + [u for u in FALLBACK_URLS if u and u != primary_url]
for model_name in models_list:
for url in urls:
if not url:
continue
result = call_ai_model(prompt, url, key, model_name)
if result and result.get(“ai_score”) and result[“ai_score”] not in [”?”, “”]:
return result
time.sleep(0.2)
print(”    WARN %s all URLs failed, next model…” % model_name)
print(”    X all models exhausted”)
return {}

def call_gpt(prompt):
return call_with_fallback(prompt, “GPT_API_URL”, “GPT_API_KEY”, [
“\u718a\u732b-A-7-gpt-5.4”,
“\u718a\u732b-\u6309\u91cf-gpt-5.3-codex-\u6ee1\u8840”,
“\u718a\u732b-A-10-gpt-5.3-codex”,
“\u718a\u732b-A-1-gpt-5.2”,
])

def call_grok(prompt):
return call_with_fallback(prompt, “GROK_API_URL”, “GROK_API_KEY”, [
“\u718a\u732b-A-7-grok-4.2-\u591a\u667a\u80fd\u4f53\u8ba8\u8bba”,
“\u718a\u732b-A-4-grok-4.2-fast”,
])

def call_gemini(prompt):
return call_with_fallback(prompt, “GEMINI_API_URL”, “GEMINI_API_KEY”, [
“\u718a\u732b\u7279\u4f9bS-\u6309\u91cf-gemini-3-flash-preview”,
“\u718a\u732b\u7279\u4f9b-\u6309\u91cf-SSS-gemini-3.1-pro-preview”,
“\u718a\u732b-2-gemini-3.1-flash-lite-preview”,
])

def call_claude(prompt):
return call_with_fallback(prompt, “CLAUDE_API_URL”, “CLAUDE_API_KEY”, [
“\u718a\u732b-\u6309\u91cf-\u9876\u7ea7\u7279\u4f9b-\u5b98max-claude-opus-4.6”,
“\u718a\u732b-\u6309\u91cf-\u9876\u7ea7\u7279\u4f9b-\u5b98max-claude-opus-4.6-thinking”,
“\u718a\u732b-\u6309\u91cf-\u7279\u4f9b\u9876\u7ea7-\u5b98\u65b9\u6b63\u5411\u6ee1\u8840-claude-opus-4.6”,
])

def merge_all(gpt, grok, gemini, claude, stats, match_obj):
sys_hp = stats.get(“home_win_pct”, 33.0)
sys_dp = stats.get(“draw_pct”, 33.0)
sys_ap = stats.get(“away_win_pct”, 33.0)
sys_cf = stats.get(“confidence”, 50)
sys_score = stats.get(“predicted_score”, “1-1”)
o25 = stats.get(“over_2_5”, 50)
bt = stats.get(“btts”, 50)
model_cons = stats.get(“model_consensus”, 0)
total_mod = stats.get(“total_models”, 11)
top_crs = stats.get(“crs_analysis”, {}).get(“top_scores”, [])[:5]
smart = stats.get(“smart_signals”, [])
extreme = stats.get(“extreme_warning”, “”)

```
candidates = {}
ai_scores = []
ai_weights = {"grok": 0.40, "gpt": 0.35, "gemini": 0.25, "claude": 0.30}

ai_sources = [("gpt", gpt), ("grok", grok), ("gemini", gemini), ("claude", claude)]
for name, res in ai_sources:
    score = res.get("ai_score") if isinstance(res, dict) else None
    if score and score not in ["-", "?", ""]:
        if score not in candidates:
            candidates[score] = {"weight": 0.0, "sources": []}
        w = ai_weights.get(name, 0.20)
        candidates[score]["weight"] += w
        candidates[score]["sources"].append(name)
        ai_scores.append(score)

for s in top_crs[:3]:
    sc = s.get("score")
    if sc:
        if sc not in candidates:
            candidates[sc] = {"weight": 0.0, "sources": []}
        candidates[sc]["weight"] += 0.28
        candidates[sc]["sources"].append("market")

has_warning = any("🚨" in str(sig) for sig in smart) or (extreme and extreme not in ["", "\u65e0"])
market_top = top_crs[0].get("score") if top_crs else None

for sc in candidates:
    if sc == market_top:
        candidates[sc]["weight"] += 0.30
    if ai_scores.count(sc) >= 2:
        candidates[sc]["weight"] += 0.45
    if ai_scores.count(sc) >= 3:
        candidates[sc]["weight"] += 0.35
    grok_score = grok.get("ai_score") if isinstance(grok, dict) else None
    gpt_score = gpt.get("ai_score") if isinstance(gpt, dict) else None
    if sc == grok_score and sc == gpt_score:
        candidates[sc]["weight"] += 0.25
    if has_warning and sc in [s.get("score") for s in top_crs]:
        candidates[sc]["weight"] += 0.15

if candidates:
    final_score = max(candidates.items(), key=lambda x: x[1]["weight"])[0]
else:
    final_score = sys_score

if ai_scores:
    home_wins = sum(1 for sc in ai_scores if sc.split("-")[0] > sc.split("-")[1])
    draws = sum(1 for sc in ai_scores if sc.split("-")[0] == sc.split("-")[1])
    away_wins = len(ai_scores) - home_wins - draws
    n = len(ai_scores)
    ai_hp = home_wins / n * 100
    ai_dp = draws / n * 100
    ai_ap = away_wins / n * 100
else:
    ai_hp, ai_dp, ai_ap = sys_hp, sys_dp, sys_ap

fused_hp = ai_hp * 0.85 + sys_hp * 0.15
fused_dp = ai_dp * 0.85 + sys_dp * 0.15
fused_ap = ai_ap * 0.85 + sys_ap * 0.15
ft = fused_hp + fused_dp + fused_ap
if ft > 0:
    fused_hp = fused_hp / ft * 100
    fused_dp = fused_dp / ft * 100
    fused_ap = 100 - fused_hp - fused_dp

unique_scores = len(set(ai_scores))
agreement_bonus = 25 if unique_scores <= 1 and len(ai_scores) >= 2 else (15 if unique_scores <= 2 else 0)
market_match = 20 if final_score == market_top else 0
warning_penalty = -10 if has_warning else 0
sys_cf = min(95, max(35, round(
    50 + agreement_bonus + market_match + warning_penalty + (max(fused_hp, fused_dp, fused_ap) - 33) * 0.4
)))

risk_str = "\u4f4e" if sys_cf >= 75 else ("\u4e2d" if sys_cf >= 55 else "\u9ad8")

val_h = calculate_value_bet(fused_hp, match_obj.get("sp_home", 0))
val_d = calculate_value_bet(fused_dp, match_obj.get("sp_draw", 0))
val_a = calculate_value_bet(fused_ap, match_obj.get("sp_away", 0))
v_tags = []
labels = ["\u4e3b\u80dc", "\u5e73\u5c40", "\u5ba2\u80dc"]
vals = [val_h, val_d, val_a]
for k, v in zip(labels, vals):
    if v and v.get("is_value"):
        v_tags.append("%s EV:+%s%% Kelly:%s%%" % (k, v["ev"], v["kelly"]))

pcts = {"\u4e3b\u80dc": fused_hp, "\u5e73\u5c40": fused_dp, "\u5ba2\u80dc": fused_ap}
result = max(pcts, key=pcts.get)

seen_signals = set()
unique_smart = []
for s in smart:
    if s not in seen_signals:
        seen_signals.add(s)
        unique_smart.append(s)

return {
    "predicted_score": final_score,
    "home_win_pct": round(fused_hp, 1),
    "draw_pct": round(fused_dp, 1),
    "away_win_pct": round(fused_ap, 1),
    "confidence": sys_cf,
    "result": result,
    "risk_level": risk_str,
    "over_under_2_5": "\u5927" if o25 > 55 else "\u5c0f",
    "both_score": "\u662f" if bt > 50 else "\u5426",
    "gpt_score": gpt.get("ai_score", "-"),
    "gpt_analysis": gpt.get("analysis", "N/A"),
    "grok_score": grok.get("ai_score", "-"),
    "grok_analysis": grok.get("analysis", "N/A"),
    "gemini_score": gemini.get("ai_score", "-"),
    "gemini_analysis": gemini.get("analysis", "N/A"),
    "claude_score": claude.get("ai_score", "-"),
    "claude_analysis": claude.get("analysis", "N/A"),
    "model_agreement": unique_scores <= 1 if ai_scores else False,
    "candidate_scores": {sc: round(c["weight"], 2) for sc, c in candidates.items()},
    "poisson": {**stats.get("poisson", {}), "home_expected_goals": stats.get("poisson", {}).get("home_xg", "?")},
    "refined_poisson": stats.get("refined_poisson", {}),
    "value_bets_summary": v_tags,
    "extreme_warning": extreme if extreme else "\u65e0",
    "smart_money_signal": " | ".join(unique_smart) if unique_smart else "\u6b63\u5e38",
    "smart_signals": unique_smart,
    "model_consensus": model_cons,
    "total_models": total_mod,
    "expected_total_goals": stats.get("expected_total_goals", 2.5),
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
    "over_2_5": o25,
    "btts": bt,
    "pace_rating": stats.get("pace_rating", ""),
    "kelly_home": stats.get("kelly_home", {}),
    "kelly_away": stats.get("kelly_away", {}),
    "odds": stats.get("odds", {}),
}
```

def select_top4(preds):
for p in preds:
pr = p.get(“prediction”, {})
s = pr.get(“confidence”, 0) * 0.35
mx = max(pr.get(“home_win_pct”, 33), pr.get(“away_win_pct”, 33), pr.get(“draw_pct”, 33))
s += (mx - 33) * 0.25
s += pr.get(“model_consensus”, 0) * 2.5
if pr.get(“risk_level”) == “\u4f4e”:
s += 10
elif pr.get(“risk_level”) == “\u9ad8”:
s -= 5
if pr.get(“model_agreement”):
s += 15
if pr.get(“value_bets_summary”):
s += 8
smart = pr.get(“smart_signals”, [])
if any(“🚨” in str(sig) for sig in smart):
s += 6
p[“recommend_score”] = round(s, 2)
preds.sort(key=lambda x: x.get(“recommend_score”, 0), reverse=True)
return preds[:4]

def extract_num(match_str):
week_map = {”\u4e00”: 1000, “\u4e8c”: 2000, “\u4e09”: 3000, “\u56db”: 4000,
“\u4e94”: 5000, “\u516d”: 6000, “\u65e5”: 7000, “\u5929”: 7000}
base = next((v for k, v in week_map.items() if k in str(match_str)), 0)
nums = re.findall(r”\d+”, str(match_str))
return base + int(nums[0]) if nums else 9999

def run_predictions(raw, use_ai=True):
ms = raw.get(“matches”, [])
print(”\n” + “=” * 60)
print(”  ENGINE v5.1 | %d matches” % len(ms))
print(”  AI 85%% | Grok40/GPT35/Gem25 | League context”)
print(”=” * 60)
res = []
for i, m in enumerate(ms):
h = m.get(“home_team”, “?”)
a = m.get(“away_team”, “?”)
print(”\n” + “-” * 50)
print(”  [%d/%d] %s %s vs %s” % (i+1, len(ms), m.get(“league”,””), h, a))
print(”-” * 50)
print(”  Phase-0: Stats…”)
sp = ensemble.predict(m, {})
print(”    H%.1f%% D%.1f%% A%.1f%% con:%d/%d” % (
sp[“home_win_pct”], sp[“draw_pct”], sp[“away_win_pct”],
sp.get(“model_consensus”,0), sp.get(“total_models”,11)))
smart = sp.get(“smart_signals”, [])
if smart:
print(”    %s” % “ | “.join(smart))
if use_ai:
print(”  Phase-1: AI recon…”)
scout = build_scout_prompt(m)
gpt_res = call_gpt(scout)
time.sleep(0.3)
grok_res = call_grok(scout)
time.sleep(0.3)
gemini_res = call_gemini(scout)
time.sleep(0.3)
print(”  Phase-2: Claude verdict…”)
cmd = build_commander_prompt(m, gpt_res or {}, grok_res or {}, gemini_res or {}, sp)
claude_res = call_claude(cmd)
time.sleep(0.3)
else:
blocked = {“ai_score”: “-”, “analysis”: “AI blocked”}
gpt_res = grok_res = gemini_res = claude_res = blocked
print(”  Phase-3: Merge…”)
mg = merge_all(gpt_res or {}, grok_res or {}, gemini_res or {}, claude_res or {}, sp, m)
print(”  => %s (%s) %d%%” % (mg[“result”], mg[“predicted_score”], mg[“confidence”]))
print(”     GPT:%s Grok:%s Gem:%s Claude:%s” % (
mg[“gpt_score”], mg[“grok_score”], mg[“gemini_score”], mg[“claude_score”]))
res.append({**m, “prediction”: mg})
t4 = select_top4(res)
t4ids = [t[“id”] for t in t4]
for r in res:
r[“is_recommended”] = r[“id”] in t4ids
res.sort(key=lambda x: extract_num(x.get(“match_num”, “”)))
print(”\n” + “=” * 60)
print(”  TOP4:”)
for i, t in enumerate(t4):
pr = t.get(“prediction”, {})
print(”    %d. %s vs %s => %s (%s) %d%%” % (
i+1, t.get(“home_team”), t.get(“away_team”),
pr.get(“result”), pr.get(“predicted_score”), pr.get(“confidence”,0)))
print(”=” * 60)
return res, t4