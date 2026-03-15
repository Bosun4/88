import json, requests, time, re, os
from config import *
from models import EnsemblePredictor

ensemble = EnsemblePredictor()

def calculate_value_bet(prob_pct, odds):
    if not odds or odds <= 1.05: return {"ev": 0.0, "kelly": 0.0, "is_value": False}
    prob = prob_pct / 100.0
    ev = (prob * odds) - 1.0
    b = odds - 1.0
    q = 1.0 - prob
    if b <= 0: return {"ev": round(ev * 100, 2), "kelly": 0.0, "is_value": False}
    kelly = ((b * prob) - q) / b
    return {"ev": round(ev * 100, 2), "kelly": round(max(0.0, kelly * 0.25) * 100, 2), "is_value": ev > 0.05}

def build_scout_prompt(m):
    h, a = m.get("home_team", "主队"), m.get("away_team", "客队")
    lg = m.get("league", "未知赛事")
    sp_h, sp_d, sp_a = m.get("sp_home", 0), m.get("sp_draw", 0), m.get("sp_away", 0)
    intel = m.get("intelligence", {})
    hs = m.get("home_stats", {})
    ast = m.get("away_stats", {})
    p = f"【系统协议】你是冷酷的足球量化风控官。任务：对【{h}】VS【{a}】进行独立胜负推演。\n"
    p += "【铁律】① 禁止虚构数据 ② 禁止提及无关球队球星 ③ 只返回纯JSON\n\n"
    p += f"【基本面】赛事：{lg} | 场次：{m.get('match_num', '?')}\n"
    p += f"SP赔率：主{sp_h} 平{sp_d} 客{sp_a} | 让球：{m.get('give_ball', '?')}\n"
    p += f"赔率变动：{m.get('odds_movement', '未知')}\n"
    p += f"\n【主队{h}】排名#{m.get('home_rank', '?')}\n"
    p += f"战绩：{hs.get('played', '?')}场{hs.get('wins', '?')}胜{hs.get('draws', '?')}平{hs.get('losses', '?')}负\n"
    p += f"进{hs.get('goals_for', '?')}球失{hs.get('goals_against', '?')}球 场均进{hs.get('avg_goals_for', '?')}失{hs.get('avg_goals_against', '?')}\n"
    p += f"近况：{hs.get('form', '?')} | 零封：{hs.get('clean_sheets', '?')}场\n"
    p += f"伤停：{intel.get('h_inj', '未知')}\n"
    p += f"\n【客队{a}】排名#{m.get('away_rank', '?')}\n"
    p += f"战绩：{ast.get('played', '?')}场{ast.get('wins', '?')}胜{ast.get('draws', '?')}平{ast.get('losses', '?')}负\n"
    p += f"进{ast.get('goals_for', '?')}球失{ast.get('goals_against', '?')}球 场均进{ast.get('avg_goals_for', '?')}失{ast.get('avg_goals_against', '?')}\n"
    p += f"近况：{ast.get('form', '?')} | 零封：{ast.get('clean_sheets', '?')}场\n"
    p += f"伤停：{intel.get('g_inj', '未知')}\n"
    h2h = m.get("h2h", [])
    if h2h:
        p += "\n【交锋记录】\n"
        for x in h2h[:5]:
            p += f"{x.get('date','')} {x.get('home','')} {x.get('score','')} {x.get('away','')}\n"
    baseface = str(m.get('baseface', '')).strip()
    if baseface:
        p += f"\n【专业基本面分析】\n{baseface[:300]}\n"
    had = m.get("had_analyse", [])
    if had:
        p += f"【官方推荐倾向】{','.join(str(x) for x in had)}\n"
    intro = str(m.get('expert_intro', '')).strip()
    if intro:
        p += f"\n【专家情报】{intro[:200]}\n"
    vote = m.get("vote", {})
    if vote:
        p += f"\n【民意】主胜{vote.get('win', '?')}% 平{vote.get('same', '?')}% 客胜{vote.get('lose', '?')}%\n"
    v2 = m.get("v2_odds_dict", {})
    if v2:
        crs_map = {"w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1",
                   "s00":"0-0","s11":"1-1","s22":"2-2",
                   "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3"}
        crs_probs = []
        for k, sc in crs_map.items():
            odds = float(v2.get(k, 0) or 0)
            if odds > 1:
                crs_probs.append((sc, odds, round(1/odds*100, 1)))
        crs_probs.sort(key=lambda x: x[2], reverse=True)
        if crs_probs:
            p += "\n【比分赔率热度TOP5】\n"
            for sc, od, prob in crs_probs[:5]:
                p += f"  {sc} 赔率{od} 隐含概率{prob}%\n"
    p += "\n【决策要求】\n"
    p += "1. 综合赔率、伤停、近况、交锋、基本面分析做出判断\n"
    p += "2. 重点关注比分赔率最热的比分，那是市场资金的共识\n"
    p += "3. 如果主胜大热但让球浅，考虑冷门可能\n"
    p += "4. 如果民意一边倒(>55%)，逆向思考是否有诱盘\n"
    p += "5. 给出你认为最可能的精确比分\n\n"
    p += "【输出】纯JSON，禁止Markdown：\n"
    p += '{"ai_score":"1-2","analysis":"80字冷血分析"}'
    return p

def build_commander_prompt(m, gpt_res, grok_res, gemini_res, stats_pred):
    h, a = m.get("home_team", "主队"), m.get("away_team", "客队")
    poi = stats_pred.get("poisson", {})
    ref = stats_pred.get("refined_poisson", {})
    sp_h, sp_d, sp_a = m.get("sp_home", 0), m.get("sp_draw", 0), m.get("sp_away", 0)
    p = f"【系统协议·最高权限】你是首席量化裁决官（中控）。你的判定为最终结论。\n"
    p += f"任务：对【{h}】vs【{a}】进行终局比分裁决。\n\n"
    p += "【三路前瞻情报】\n"
    p += f"GPT前瞻：比分[{gpt_res.get('ai_score', '无')}] 逻辑：{gpt_res.get('analysis', '无')}\n"
    p += f"Grok前瞻：比分[{grok_res.get('ai_score', '无')}] 逻辑：{grok_res.get('analysis', '无')}\n"
    p += f"Gemini前瞻：比分[{gemini_res.get('ai_score', '无')}] 逻辑：{gemini_res.get('analysis', '无')}\n\n"
    p += "【统计模型矩阵】\n"
    p += f"泊松比分：{poi.get('predicted_score', '?')} | 修正泊松：{ref.get('predicted_score', '?')}\n"
    p += f"融合概率：主{stats_pred.get('home_win_pct', 33):.1f}% 平{stats_pred.get('draw_pct', 33):.1f}% 客{stats_pred.get('away_win_pct', 33):.1f}%\n"
    p += f"模型共识：{stats_pred.get('model_consensus', 0)}/{stats_pred.get('total_models', 11)}\n"
    p += f"期望总球：{stats_pred.get('expected_total_goals', 2.5):.1f}\n"
    crs = stats_pred.get("crs_analysis", {})
    top_crs = crs.get("top_scores", [])[:5]
    if top_crs:
        p += "\n【比分赔率市场共识TOP5】\n"
        for s in top_crs:
            p += f"  {s.get('score', '?')} 概率{s.get('prob', 0):.1f}% 赔率{s.get('odds', '?')}\n"
    p += f"\n【原始赔率】SP：主{sp_h} 平{sp_d} 客{sp_a} | 让球：{m.get('give_ball', '?')}\n"
    intel = m.get("intelligence", {})
    p += f"【伤停】主队：{intel.get('h_inj', '?')[:80]}\n"
    p += f"      客队：{intel.get('g_inj', '?')[:80]}\n"
    smart = stats_pred.get("smart_signals", [])
    if smart:
        p += f"\n【风控信号】{'  |  '.join(smart)}\n"
    baseface = str(m.get('baseface', '')).strip()
    if baseface:
        p += f"\n【基本面】{baseface[:200]}\n"
    p += "\n【裁决指令】\n"
    p += "1. 交叉比对三份AI前瞻，如果2个以上AI预测同一比分，该比分可信度极高\n"
    p += "2. 比分赔率TOP1是市场资金最认可的比分，必须重点考虑\n"
    p += "3. 如果AI前瞻与比分赔率TOP1一致，大胆采纳\n"
    p += "4. 如果AI前瞻全部与比分赔率矛盾，以比分赔率为锚，微调即可\n"
    p += "5. 伤停严重的一方大概率丢球更多\n"
    p += "6. 风控信号（欧亚错位/资金流向）暗示机构态度，务必重视\n\n"
    p += "【输出】纯JSON，禁止Markdown：\n"
    p += '{"ai_score":"1-2","analysis":"100字终局裁决"}'
    return p

def extract_clean_json(text):
    text = str(text or "").strip()
    fallback_score, fallback_analysis = "未预测", "格式异常"
    s_match = re.search(r'"ai_score"\s*:\s*"([^"]+)"', text)
    if s_match: fallback_score = s_match.group(1)
    a_match = re.search(r'"analysis"\s*:\s*"(.*?)"', text, re.DOTALL)
    if a_match: fallback_analysis = a_match.group(1).replace('"', "'").replace('\n', ' ').strip()[:150]
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end != -1:
        try: return json.loads(text[start:end+1])
        except: pass
    cleaned = re.sub(r'```\w*', '', text).strip()
    s2, e2 = cleaned.find('{'), cleaned.rfind('}')
    if s2 != -1 and e2 != -1:
        try: return json.loads(cleaned[s2:e2+1])
        except: pass
    if fallback_score != "未预测":
        return {"ai_score": fallback_score, "analysis": fallback_analysis}
    return None

def get_clean_env_url(name, default=""):
    v = os.environ.get(name, globals().get(name, default))
    v = str(v).strip(' \t\n\r"\'')
    match = re.search(r'(https?://[a-zA-Z0-9._/-]+)', v)
    return match.group(1) if match else v

def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(' \t\n\r"\'')

def call_ai_model(prompt, url, key, model_name):
    if not url or not key:
        print(f"    ❌ {model_name}: 缺少配置")
        return {}
    is_native_gemini = "generateContent" in url
    if not is_native_gemini and "chat/completions" not in url:
        url = url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if is_native_gemini:
        headers["x-goog-api-key"] = key
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}
    else:
        headers["Authorization"] = f"Bearer {key}"
        payload = {"model": model_name, "messages": [
            {"role": "system", "content": "你是纯JSON输出机。严禁输出任何Markdown修饰符。"},
            {"role": "user", "content": prompt}
        ], "temperature": 0.2, "max_tokens": 500}
    gw = url.split('/v1')[0] if '/v1' in url else url[:35]
    print(f"    🤖 {model_name} → {gw}...")
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code == 200:
            if is_native_gemini:
                t = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                t = r.json()["choices"][0]["message"]["content"].strip()
            parsed = extract_clean_json(t)
            if parsed:
                parsed["analysis"] = str(parsed.get("analysis", "")).replace("```json", "").replace("```", "").strip()
                print(f"    ✅ {model_name}: {parsed.get('ai_score', '?')}")
                return parsed
            else:
                print(f"    ⚠️ {model_name}: JSON解析失败")
        else:
            print(f"    ❌ {model_name}: HTTP {r.status_code}")
    except Exception as e:
        print(f"    ⚠️ {model_name}: {str(e)[:50]}")
    return {}

def call_with_fallback(prompt, url_env, key_env, models_list):
    url = get_clean_env_url(url_env, globals().get(url_env, ""))
    key = get_clean_env_key(key_env)
    for i, model_name in enumerate(models_list):
        tag = "主力" if i == 0 else f"备用{i}"
        print(f"    [{tag}] 尝试 {model_name}...")
        result = call_ai_model(prompt, url, key, model_name)
        if result and result.get("ai_score") and result["ai_score"] not in ["未预测", "?", ""]:
            return result
        print(f"    [{tag}] {model_name} 失败或格式异常，切换下一个...")
        time.sleep(0.3)
    print(f"    ❌ 全部模型耗尽，返回空结果")
    return {}

def call_gpt(prompt):
    return call_with_fallback(prompt, "GPT_API_URL", "GPT_API_KEY", [
        "熊猫-A-7-gpt-5.4",
        "熊猫-按量-gpt-5.3-codex-满血",
        "熊猫-A-10-gpt-5.3-codex",
        "熊猫-A-1-gpt-5.2",
    ])

def call_grok(prompt):
    return call_with_fallback(prompt, "GROK_API_URL", "GROK_API_KEY", [
        "熊猫-A-7-grok-4.2-多智能体讨论",
        "熊猫-A-4-grok-4.2-fast",
    ])

def call_gemini(prompt):
    return call_with_fallback(prompt, "GEMINI_API_URL", "GEMINI_API_KEY", [
        "熊猫特供S-按量-gemini-3-flash-preview",
        "熊猫特供-按量-SSS-gemini-3.1-pro-preview",
        "熊猫-2-gemini-3.1-flash-lite-preview",
    ])

def call_claude(prompt):
    return call_with_fallback(prompt, "CLAUDE_API_URL", "CLAUDE_API_KEY", [
        "熊猫-按量-顶级特供-官max-claude-opus-4.6",
        "熊猫-按量-顶级特供-官max-claude-opus-4.6-thinking",
        "熊猫-按量-特供顶级-官方正向满血-claude-opus-4.6",
    ])

def merge_all(gpt, grok, gemini, claude, stats, match_obj):
    sys_hp = stats.get("home_win_pct", 33)
    sys_dp = stats.get("draw_pct", 33)
    sys_ap = stats.get("away_win_pct", 33)
    sys_cf = stats.get("confidence", 50)
    sys_score = stats.get("predicted_score", "1-1")
    pcts = {"主胜": sys_hp, "平局": sys_dp, "客胜": sys_ap}
    result = max(pcts, key=pcts.get)
    if claude.get("ai_score") and claude["ai_score"] not in ["-", "未预测", "?"]:
        sys_score = claude["ai_score"]
        sys_cf = min(sys_cf + 5, 95)
    else:
        ai_scores = [x.get("ai_score") for x in [gpt, grok, gemini]
                     if x.get("ai_score") and x["ai_score"] not in ["-", "未预测", "?"]]
        if ai_scores:
            from collections import Counter
            sc = Counter(ai_scores).most_common(1)[0]
            if sc[1] >= 2:
                sys_score = sc[0]
                sys_cf = min(sys_cf + 8, 95)
            else:
                sys_score = ai_scores[0]
    risk = "低" if sys_cf >= 70 else ("中" if sys_cf >= 50 else "高")
    val_h = calculate_value_bet(sys_hp, match_obj.get("sp_home", 0))
    val_d = calculate_value_bet(sys_dp, match_obj.get("sp_draw", 0))
    val_a = calculate_value_bet(sys_ap, match_obj.get("sp_away", 0))
    v_tags = []
    for k, v in zip(["主胜", "平局", "客胜"], [val_h, val_d, val_a]):
        if v and v.get("is_value"):
            v_tags.append(f"{k} EV:+{v['ev']}% 仓位:{v['kelly']}%")
    o25 = stats.get("over_2_5", 50)
    bt = stats.get("btts", 50)
    all_scores = [x.get("ai_score") for x in [gpt, grok, gemini, claude]
                  if x.get("ai_score") and x["ai_score"] not in ["-", "未预测", "?"]]
    model_agreement = len(set(all_scores)) <= 1 if all_scores else False
    return {
        "predicted_score": sys_score, "home_win_pct": sys_hp, "draw_pct": sys_dp, "away_win_pct": sys_ap,
        "confidence": sys_cf, "result": result, "risk_level": risk,
        "over_under_2_5": "大" if o25 > 55 else "小",
        "both_score": "是" if bt > 50 else "否",
        "gpt_score": gpt.get("ai_score", "-"), "gpt_analysis": gpt.get("analysis", "未响应"),
        "grok_score": grok.get("ai_score", "-"), "grok_analysis": grok.get("analysis", "未响应"),
        "gemini_score": gemini.get("ai_score", "-"), "gemini_analysis": gemini.get("analysis", "未响应"),
        "claude_score": claude.get("ai_score", "-"), "claude_analysis": claude.get("analysis", "未响应"),
        "model_agreement": model_agreement,
        "poisson": {**stats.get("poisson", {}),
                    "home_expected_goals": stats.get("poisson", {}).get("home_xg", "?"),
                    "away_expected_goals": stats.get("poisson", {}).get("away_xg", "?")},
        "refined_poisson": stats.get("refined_poisson", {}),
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
        "value_bets_summary": v_tags,
        "extreme_warning": stats.get("extreme_warning", "无"),
        "smart_money_signal": " | ".join(stats.get("smart_signals", [])) if stats.get("smart_signals") else "正常",
        "smart_signals": stats.get("smart_signals", []),
        "handicap_signal": stats.get("handicap_signal", ""),
        "odds_movement": stats.get("odds_movement", {}),
        "vote_analysis": stats.get("vote_analysis", {}),
        "h2h_blood": stats.get("h2h_blood", {}),
        "crs_analysis": stats.get("crs_analysis", {}),
        "ttg_analysis": stats.get("ttg_analysis", {}),
        "halftime": stats.get("halftime", {}),
        "over_2_5": o25, "btts": bt,
        "expected_total_goals": stats.get("expected_total_goals", 2.5),
        "pace_rating": stats.get("pace_rating", ""),
        "top_scores": stats.get("refined_poisson", {}).get("top_scores",
                      stats.get("poisson", {}).get("top_scores", [])),
        "model_consensus": stats.get("model_consensus", 0),
        "total_models": stats.get("total_models", 11),
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
        if pr.get("risk_level") == "低": s += 10
        elif pr.get("risk_level") == "高": s -= 5
        if pr.get("model_agreement"): s += 15
        if pr.get("value_bets_summary"): s += 8
        smart = pr.get("smart_signals", [])
        if any("🚨" in str(sig) for sig in smart): s += 6
        p["recommend_score"] = round(s, 2)
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]

def extract_num(match_str):
    week_map = {"一": 1000, "二": 2000, "三": 3000, "四": 4000, "五": 5000, "六": 6000, "日": 7000, "天": 7000}
    base = next((v for k, v in week_map.items() if k in str(match_str)), 0)
    nums = re.findall(r'\d+', str(match_str))
    return base + int(nums[0]) if nums else 9999

def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE ENGINE v4.2 | {len(ms)} matches")
    print(f"  11 Stats + 4 AI (GPT·Grok·Gemini·Claude中控)")
    print(f"  自动降级：GPT 4级 | Gemini 3级 | Grok 2级 | Claude 3级")
    print(f"{'='*60}")
    res = []
    for i, m in enumerate(ms):
        h, a = m.get("home_team", "?"), m.get("away_team", "?")
        print(f"\n{'─'*50}")
        print(f"  [{i+1}/{len(ms)}] {m.get('league', '')} {h} vs {a}")
        print(f"{'─'*50}")
        print("  Phase-0: Stats Matrix (11 models)...")
        sp = ensemble.predict(m, {})
        print(f"     融合: H{sp['home_win_pct']:.1f}% D{sp['draw_pct']:.1f}% A{sp['away_win_pct']:.1f}% 共识:{sp.get('model_consensus', 0)}/{sp.get('total_models', 11)}")
        smart = sp.get("smart_signals", [])
        if smart:
            print(f"     信号: {' | '.join(smart)}")
        if use_ai:
            print("  Phase-1: 三路独立前瞻...")
            scout_prompt = build_scout_prompt(m)
            gpt_res = call_gpt(scout_prompt)
            time.sleep(0.5)
            grok_res = call_grok(scout_prompt)
            time.sleep(0.5)
            gemini_res = call_gemini(scout_prompt)
            time.sleep(0.5)
            print("  Phase-2: Claude中控终裁...")
            commander_prompt = build_commander_prompt(m, gpt_res or {}, grok_res or {}, gemini_res or {}, sp)
            claude_res = call_claude(commander_prompt)
            time.sleep(0.5)
        else:
            blocked = {"ai_score": "-", "analysis": "AI调用已阻断"}
            gpt_res = grok_res = gemini_res = claude_res = blocked
        print("  Phase-3: 全维度融合...")
        mg = merge_all(gpt_res or {}, grok_res or {}, gemini_res or {}, claude_res or {}, sp, m)
        print(f"  => {mg['result']} ({mg['predicted_score']}) {mg['confidence']:.0f}%")
        print(f"     GPT:{mg['gpt_score']} Grok:{mg['grok_score']} Gem:{mg['gemini_score']} Claude:{mg['claude_score']}")
        res.append({**m, "prediction": mg})
    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res:
        r["is_recommended"] = r["id"] in t4ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    print(f"\n{'='*60}")
    print("  TOP4 推荐:")
    for i, t in enumerate(t4):
        pr = t.get("prediction", {})
        print(f"    {i+1}. {t.get('home_team')} vs {t.get('away_team')} => {pr.get('result')} ({pr.get('predicted_score')}) {pr.get('confidence'):.0f}%")
    print(f"{'='*60}")
    return res, t4
