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

def build_independent_prompt(m, sp):
    h, a, lg = m["home_team"], m["away_team"], m.get("league", "")
    intel = m.get("intelligence", {})
    hs, ast = m.get("home_stats", {}), m.get("away_stats", {})
    
    p = "作为我冷酷无情的量化导师，对一切比赛内容进行压力测试，挑战所有给定的数据。我需要的是滴水不漏的缜密思维，而不是廉价的自我认同。请基于以下情报进行极其苛刻的推演。\n\n"
    p += f"【赛事】{lg} | {h} vs {a}\n"
    p += f"【伤停与破绽】主队：{intel.get('h_inj')} | 客队：{intel.get('g_inj')}\n"
    p += f"【盘口与机构阴谋】{m.get('handicap_info')} | 水位：{m.get('odds_movement')}\n"
    p += f"【外部杂音(需审视)】{m.get('expert_intro', '暂无')}\n"
    
    warning = sp.get('extreme_warning', '无')
    p += f"【系统压测信号】{warning}\n"
    
    p += "\n【压测准则】\n"
    p += "1. 不要盲从常规概率，用你的高维视角压力测试庄家赔率与伤停信息，找出基本面与盘口之间隐藏的致命破绽。\n"
    p += "2. 拒绝平庸。如果数据支撑碾压，给出冷血的穿盘比分；如果势均力敌，给出缜密的平局推演。\n"
    p += "必须严格返回JSON格式，不可有Markdown修饰：\n"
    p += '{"ai_score":"1-2","analysis":"200字极其冷酷、专业的压测复盘，说明你为何推演出该比分。"}'
    return p

def build_synthesis_prompt(m, gpt_res, claude_res):
    h, a, lg = m["home_team"], m["away_team"], m.get("league", "")
    p = "作为首席决策官兼冷酷无情的导师，对GPT和Claude的意见进行最高级别的压力测试。挑战它们的所有逻辑漏洞，我需要的是滴水不漏的缜密思维。\n\n"
    p += f"【赛事】{lg} | {h} vs {a}\n"
    p += f"【GPT 意见】预测比分: {gpt_res.get('ai_score', '未知')} | 逻辑: {gpt_res.get('analysis', '无')}\n"
    p += f"【Claude 意见】预测比分: {claude_res.get('ai_score', '未知')} | 逻辑: {claude_res.get('analysis', '无')}\n"
    p += "\n【压测任务】冷酷地审视这两份报告，剔除它们感性与不合理的部分，结合你自己的知识库，给出最终的绝对裁决。\n"
    p += "必须严格返回纯JSON格式：\n"
    p += '{"ai_score":"1-2","analysis":"200字冷酷的终极压测裁决，指出你采纳或摒弃前两份报告的根本原因。"}'
    return p

def call_ai_model(prompt, url, key, model_name, is_gpt_format=True):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    sys_msg = "你是冷酷严谨的量化精算师，严格返回纯JSON格式，严禁Markdown标记。"
    print(f"    🤖 请求 AI: {model_name} (无尽等待中，只要有数据才进行下一步)...")
    try:
        if is_gpt_format:
            payload = {"model": model_name, "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 1500}
        else:
            if "generateContent" in url: 
                payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1500}}
            else: 
                payload = {"model": model_name, "messages": [{"role": "user", "content": "系统指令：" + sys_msg + "\n\n" + prompt}], "temperature": 0.3, "max_tokens": 1500}
        
        r = requests.post(url, headers=headers, json=payload, timeout=600)
        if r.status_code == 200:
            t = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip() if "generateContent" in url else r.json()["choices"][0]["message"]["content"].strip()
            s = t.find("{"); e = t.rfind("}") + 1
            if s >= 0 and e > s: return json.loads(t[s:e])
        else:
            print(f"    ❌ 接口报错: {r.status_code} ({r.text[:50]})")
    except Exception as e: print(f"    ⚠️ {model_name} 异常: {str(e)[:40]}")
    return {}

def call_gpt(prompt): 
    return call_ai_model(prompt, GPT_API_URL, GPT_API_KEY, "gpt-5.4", True)

def call_claude(prompt): 
    # 🔥 获取独立的 Claude 密钥，如果 config 里没写，就直接从环境变量里抓
    try:
        claude_key = CLAUDE_API_KEY
    except NameError:
        claude_key = os.getenv("CLAUDE_API_KEY", "")
    # 🔥 核心替换：指定 Claude 模型，复用 GPT 的底层网关
    return call_ai_model(prompt, GPT_API_URL, claude_key, "claude-sonnet-4-6", True) 

def call_gemini(prompt): 
    return call_ai_model(prompt, GEMINI_API_URL, GEMINI_API_KEY, "[次-流抗截]gemini-3.1-pro-preview-thinking", False)

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
        
        "gpt_score": gpt.get("ai_score", "-"), "gpt_analysis": gpt.get("analysis", "未响应"),
        "claude_score": claude.get("ai_score", "-"), "claude_analysis": claude.get("analysis", "系统阻断或未响应"),
        "gemini_score": gemini.get("ai_score", "-"), "gemini_analysis": gemini.get("analysis", "未响应"),
        
        "smart_money_signal": stats.get("smart_money_signal", "正常"), "value_bets_summary": v_tags,
        "extreme_warning": stats.get("extreme_warning", "无"), 
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
        if pr.get("extreme_warning") != "无": s += 10 
        p["recommend_score"] = round(s, 2)
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]

def run_predictions(raw):
    ms = raw.get("matches", []); res = []
    for i, m in enumerate(ms):
        sp = ensemble.predict(m, {})
        ind_prompt = build_independent_prompt(m, sp)
        
        gpt_res = call_gpt(ind_prompt)
        time.sleep(1)
        # 🔥 核心替换：调度 Claude 分析
        claude_res = call_claude(ind_prompt)
        time.sleep(1)
        
        syn_prompt = build_synthesis_prompt(m, gpt_res or {}, claude_res or {})
        gemini_res = call_gemini(syn_prompt)
        
        mg = merge_all(gpt_res or {}, claude_res or {}, gemini_res or {}, sp, m)
        res.append({**m, "prediction": mg})
        
    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res: r["is_recommended"] = r["id"] in t4ids
    
    def extract_num(match_str):
        nums = re.findall(r'\d+', match_str)
        return int(nums[0]) if nums else 9999
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    return res, t4
