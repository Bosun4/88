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

def build_independent_prompt(m):
    h, a, lg = m["home_team"], m["away_team"], m.get("league", "")
    intel = m.get("intelligence", {})
    
    p = "作为极其严谨的量化精算导师，请对以下情报数据进行极度深度的压力测试。我需要滴水不漏的缜密逻辑。\n\n"
    p += f"【比赛对阵】{lg} | {h} vs {a}\n"
    p += f"【阵容与隐患】主队：{intel.get('h_inj')} | 客队：{intel.get('g_inj')}\n"
    p += f"【盘口与动向】{m.get('handicap_info')} | 资金异动：{m.get('odds_movement')}\n"
    p += f"【外部观点】{m.get('expert_intro', '暂无')}\n"
    
    p += "\n【压测准则】\n"
    p += "1. 挑战表象：运用你对球队实力的深层认知，识破庄家通过盘口设置的诱导陷阱。\n"
    p += "2. 拒绝平庸：如果双方实力悬殊或防线残缺，请果断给出大比分穿盘预测（如0-3, 1-4）；如果是真正的防守战，给出合理的僵局比分。\n"
    p += "3. 必须严格返回纯JSON格式，严禁出现Markdown修饰或任何多余的解释文字。\n"
    p += '{"ai_score":"1-2","analysis":"200字极其严谨的逻辑复盘，揭露比赛的胜负核心点。"}'
    return p

def build_synthesis_prompt(m, gpt_res, claude_res):
    h, a, lg = m["home_team"], m["away_team"], m.get("league", "")
    p = "作为首席决策官兼严谨的量化导师，请对GPT和Claude的意见进行最高级别的逻辑压测。挑出它们的漏洞，给出绝对权威的最终裁决。\n\n"
    p += f"【赛事】{lg} | {h} vs {a}\n"
    p += f"【GPT 报告】比分: {gpt_res.get('ai_score', '未知')} | 逻辑: {gpt_res.get('analysis', '无')}\n"
    p += f"【Claude 报告】比分: {claude_res.get('ai_score', '未知')} | 逻辑: {claude_res.get('analysis', '无')}\n"
    p += "\n【终极裁决】冷酷地审视这两份报告，剔除感性部分，结合你的知识库，给出一锤定音的『预测比分』。\n"
    p += '必须严格返回纯JSON格式：\n{"ai_score":"1-2","analysis":"200字终极严谨裁决，指出你采纳或摒弃它们意见的根本逻辑。"}'
    return p

def call_ai_model(prompt, url, key, model_name, is_gpt_format=True):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    sys_msg = "你是严谨无情的量化精算师，只允许输出JSON，不准带```json代码块。"
    print(f"    🤖 等待 AI {model_name} 的独立思考 (600秒上限)...")
    try:
        if is_gpt_format:
            payload = {"model": model_name, "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}], "temperature": 0.3}
        else:
            if "generateContent" in url: 
                payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.3}}
            else: 
                payload = {"model": model_name, "messages": [{"role": "user", "content": "系统指令：" + sys_msg + "\n\n" + prompt}], "temperature": 0.3}
        
        r = requests.post(url, headers=headers, json=payload, timeout=600)
        if r.status_code == 200:
            t = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip() if "generateContent" in url else r.json()["choices"][0]["message"]["content"].strip()
            match = re.search(r'\{.*\}', t, re.DOTALL)
            if match:
                return json.loads(match.group(0))
    except Exception as e: print(f"    ⚠️ {model_name} 异常: {str(e)[:40]}")
    return {}

def call_gpt(prompt): return call_ai_model(prompt, GPT_API_URL, GPT_API_KEY, "gpt-5.4", True)

def call_claude(prompt): 
    try: claude_key = CLAUDE_API_KEY
    except NameError: claude_key = os.environ.get("CLAUDE_API_KEY", "")
    return call_ai_model(prompt, GPT_API_URL, claude_key, "claude-sonnet-4-6", True) 

def call_gemini(prompt): return call_ai_model(prompt, GEMINI_API_URL, GEMINI_API_KEY, "[次-流抗截]gemini-3.1-pro-preview-thinking", False)

def merge_all(gpt, claude, gemini, stats, match_obj):
    sys_hp, sys_dp, sys_ap = stats.get("home_win_pct", 33), stats.get("draw_pct", 33), stats.get("away_win_pct", 33)
    sys_cf = stats.get("confidence", 50)
    sys_score = stats.get("predicted_score", "1-1")
    
    result = max({"主胜": sys_hp, "平局": sys_dp, "客胜": sys_ap}, key={"主胜": sys_hp, "平局": sys_dp, "客胜": sys_ap}.get)

    val_h = calculate_value_bet(sys_hp, match_obj.get("sp_home", 0))
    val_d = calculate_value_bet(sys_dp, match_obj.get("sp_draw", 0))
    val_a = calculate_value_bet(sys_ap, match_obj.get("sp_away", 0))
    v_tags = [f"{k} EV:+{v['ev']}% (仓位:{v['kelly']}%)" for k, v in zip(["主胜", "平局", "客胜"], [val_h, val_d, val_a]) if v and v.get("is_value")]
    
    return {
        "predicted_score": sys_score, "home_win_pct": sys_hp, "draw_pct": sys_dp, "away_win_pct": sys_ap,
        "confidence": sys_cf, "result": result, "risk_level": "低" if sys_cf >= 70 else ("中" if sys_cf >= 50 else "高"),
        
        "gpt_score": gpt.get("ai_score", "-"), "gpt_analysis": gpt.get("analysis", "安全审查拦截/未响应"),
        "claude_score": claude.get("ai_score", "-"), "claude_analysis": claude.get("analysis", "安全审查拦截/未响应"),
        "gemini_score": gemini.get("ai_score", "-"), "gemini_analysis": gemini.get("analysis", "安全审查拦截/未响应"),
        
        "value_bets_summary": v_tags,
        "extreme_warning": stats.get("extreme_warning", "无"),
        "smart_money_signal": stats.get("smart_money_signal", "正常"),
        "poisson": {**stats.get("poisson", {}), "home_expected_goals": stats.get("poisson", {}).get("home_xg", "?"), "away_expected_goals": stats.get("poisson", {}).get("away_xg", "?")},
        "refined_poisson": stats.get("refined_poisson", {}), "elo": stats.get("elo", {}), 
        "home_form": stats.get("home_form", {}), "away_form": stats.get("away_form", {}),
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

# 🔥 核心修复：极其严谨的汉字星期权重映射！
def extract_num(match_str):
    week_map = {"一": 1000, "二": 2000, "三": 3000, "四": 4000, "五": 5000, "六": 6000, "日": 7000, "天": 7000}
    base_weight = 0
    for k, v in week_map.items():
        if k in match_str:
            base_weight = v
            break
    nums = re.findall(r'\d+', match_str)
    return base_weight + int(nums[0]) if nums else 9999

def run_predictions(raw):
    ms = raw.get("matches", []); res = []
    for i, m in enumerate(ms):
        sp = ensemble.predict(m, {})
        
        ind_prompt = build_independent_prompt(m)
        gpt_res = call_gpt(ind_prompt)
        time.sleep(1)
        claude_res = call_claude(ind_prompt)
        time.sleep(1)
        
        syn_prompt = build_synthesis_prompt(m, gpt_res or {}, claude_res or {})
        gemini_res = call_gemini(syn_prompt)
        
        mg = merge_all(gpt_res or {}, claude_res or {}, gemini_res or {}, sp, m)
        res.append({**m, "prediction": mg})
        
    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res: r["is_recommended"] = r["id"] in t4ids
    
    # 使用增强权重函数严谨排序
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    return res, t4
