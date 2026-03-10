import json, requests, time, re
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

# 提示词 1：发给两名独立顾问（GPT 和 Grok）
def build_independent_prompt(m):
    h, a, lg = m["home_team"], m["away_team"], m.get("league", "")
    intel = m.get("intelligence", {})
    hs, ast = m.get("home_stats", {}), m.get("away_stats", {})
    
    p = "你是一位独立思考的足球量化顾问。请基于以下情报进行推演，不要受任何系统概率干扰。\n\n"
    p += f"【赛事】{lg} | {h} vs {a}\n"
    p += f"【伤停利空】主队：{intel.get('h_inj')} | 客队：{intel.get('g_inj')}\n"
    p += f"【盘口异动】{m.get('handicap_info')} | 水位：{m.get('odds_movement')}\n"
    p += f"【专家研报】{m.get('expert_intro', '暂无')} | {m.get('base_face', '暂无')[:250]}\n"
    if hs: p += f"【近况】主队：{hs.get('played','?')}场{hs.get('wins','?')}胜 | 客队：{ast.get('played','?')}场{ast.get('wins','?')}胜\n"
    
    p += "\n给出你认为最合理的预测比分，并解析为何如此判断。严格返回JSON格式：\n"
    p += '{"ai_score":"1-1","analysis":"200字独立见解"}'
    return p

# 提示词 2：发给首席决策官（Gemini）
def build_synthesis_prompt(m, gpt_res, grok_res):
    h, a, lg = m["home_team"], m["away_team"], m.get("league", "")
    
    p = "你是本机构的『首席AI综合决策官(CIO)』。现在有一场比赛，请你综合手下两名顶级顾问(GPT与Grok)的独立意见，给出最终的官方裁决。\n\n"
    p += f"【赛事基本信息】{lg} | {h} vs {a}\n"
    p += f"【顾问1 (GPT) 意见】预测比分: {gpt_res.get('ai_score', '未知')} | 逻辑: {gpt_res.get('analysis', '无')}\n"
    p += f"【顾问2 (Grok) 意见】预测比分: {grok_res.get('ai_score', '未知')} | 逻辑: {grok_res.get('analysis', '无')}\n"
    
    p += "\n【你的任务】作为最终决策者，结合基本常识，对两位顾问的分歧进行裁判，或者总结他们的共识，并给出你认为最精准的『最终比分』。\n"
    p += "严格返回纯JSON格式，严禁Markdown修饰：\n"
    p += '{"ai_score":"1-1","analysis":"200字综合最终研判意见(必须提及对GPT和Grok意见的取舍)"}'
    return p

def call_ai_model(prompt, url, key, model_name, is_gpt_format=True, max_time=30):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    sys_msg = "你是顶级足球量化精算师，严格返回纯JSON格式，禁绝Markdown代码块。"
    print(f"    🤖 尝试请求 AI: {model_name}...")
    try:
        if is_gpt_format:
            payload = {"model": model_name, "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 1500}
        else:
            if "generateContent" in url: # 原生 Gemini API
                payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1500}}
            else: # 兼容代理 Gemini
                payload = {"model": model_name, "messages": [{"role": "user", "content": "系统指令：" + sys_msg + "\n\n" + prompt}], "temperature": 0.3, "max_tokens": 1500}
        
        r = requests.post(url, headers=headers, json=payload, timeout=max_time)
        if r.status_code == 200:
            t = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip() if "generateContent" in url else r.json()["choices"][0]["message"]["content"].strip()
            s = t.find("{"); e = t.rfind("}") + 1
            if s >= 0 and e > s: 
                return json.loads(t[s:e])
    except Exception as e:
        print(f"    ⚠️ {model_name} 异常或超时: {str(e)[:40]}")
    return {}

def call_gpt(prompt): return call_ai_model(prompt, GPT_API_URL, GPT_API_KEY, "gpt-5.4", True)
def call_grok(prompt): return call_ai_model(prompt, GEMINI_API_URL, GEMINI_API_KEY, "[次]grok-420-thinking", True) # Grok 走代理，格式通常兼容 GPT
def call_gemini(prompt): return call_ai_model(prompt, GEMINI_API_URL, GEMINI_API_KEY, "[次-流抗截]gemini-3.1-pro-preview-thinking", False, 45)

def merge_all(gpt, grok, gemini, stats, match_obj):
    sys_hp, sys_dp, sys_ap = stats.get("home_win_pct", 33), stats.get("draw_pct", 33), stats.get("away_win_pct", 33)
    sys_cf = stats.get("confidence", 50)
    sys_score = stats.get("predicted_score", "1-1")
    
    result = max({"主胜": sys_hp, "平局": sys_dp, "客胜": sys_ap}, key={"主胜": sys_hp, "平局": sys_dp, "客胜": sys_ap}.get)

    val_h = calculate_value_bet(sys_hp, match_obj.get("sp_home", 0))
    val_d = calculate_value_bet(sys_dp, match_obj.get("sp_draw", 0))
    val_a = calculate_value_bet(sys_ap, match_obj.get("sp_away", 0))
    v_tags = [f"{k} EV:+{v['ev']}% (仓位:{v['kelly']}%)" for k, v in zip(["主胜", "平局", "客胜"], [val_h, val_d, val_a]) if v and v.get("is_value")]
    
    return {
        "predicted_score": sys_score,
        "home_win_pct": sys_hp, "draw_pct": sys_dp, "away_win_pct": sys_ap,
        "confidence": sys_cf, "result": result,
        "risk_level": "低" if sys_cf >= 70 else ("中" if sys_cf >= 50 else "高"),
        
        "gpt_score": gpt.get("ai_score", "-"), "gpt_analysis": gpt.get("analysis", "未响应"),
        "grok_score": grok.get("ai_score", "-"), "grok_analysis": grok.get("analysis", "未响应"),
        "gemini_score": gemini.get("ai_score", "-"), "gemini_analysis": gemini.get("analysis", "未响应"),
        
        "smart_money_signal": stats.get("smart_money_signal", "正常"),
        "value_bets_summary": v_tags,
        
        "poisson": {**stats.get("poisson", {}), "home_expected_goals": stats.get("poisson", {}).get("home_xg", "?"), "away_expected_goals": stats.get("poisson", {}).get("away_xg", "?")},
        "refined_poisson": stats.get("refined_poisson", {}), 
        "elo": stats.get("elo", {}), "home_form": stats.get("home_form", {}), "away_form": stats.get("away_form", {}),
        "model_consensus": stats.get("model_consensus", 0), "total_models": stats.get("total_models", 4)
    }

def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})
        s = pr.get("confidence", 0) * 0.4
        s += (max(pr.get("home_win_pct", 33), pr.get("away_win_pct", 33)) - 33) * 0.3
        if pr.get("risk_level") == "低": s += 8
        elif pr.get("risk_level") == "高": s -= 5
        if pr.get("value_bets_summary"): s += 15 
        p["recommend_score"] = round(s, 2)
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]

def run_predictions(raw):
    ms = raw.get("matches", []); res = []
    for i, m in enumerate(ms):
        sp = ensemble.predict(m, {})
        
        # 1. 发送独立提示词给顾问
        ind_prompt = build_independent_prompt(m)
        gpt_res = call_gpt(ind_prompt)
        time.sleep(1)
        grok_res = call_grok(ind_prompt)
        time.sleep(1)
        
        # 2. 将顾问结果汇总给决策官
        syn_prompt = build_synthesis_prompt(m, gpt_res or {}, grok_res or {})
        gemini_res = call_gemini(syn_prompt)
        
        mg = merge_all(gpt_res or {}, grok_res or {}, gemini_res or {}, sp, m)
        res.append({**m, "prediction": mg})
        
    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res: r["is_recommended"] = r["id"] in t4ids
    
    def extract_num(match_str):
        nums = re.findall(r'\d+', match_str)
        return int(nums[0]) if nums else 9999
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    
    return res, t4
