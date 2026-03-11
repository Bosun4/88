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

# 🔥 核心修改：提示词仅提供【抓包情报】，让 AI 独立分析，不透漏本地系统概率
def build_independent_prompt(m):
    h, a, lg = m["home_team"], m["away_team"], m.get("league", "")
    intel = m.get("intelligence", {})
    hs, ast = m.get("home_stats", {}), m.get("away_stats", {})
    
    p = "作为我冷酷无情的量化导师，请你对以下抓取到的情报数据进行极度严苛的压力测试。我需要的是滴水不漏的缜密思维，不要给我廉价的附和或盲目的预测。\n\n"
    p += f"【比赛对阵】{lg} | {h} vs {a}\n"
    p += f"【伤停与破绽】主队：{intel.get('h_inj')} | 客队：{intel.get('g_inj')}\n"
    p += f"【盘口与机构阴谋】{m.get('handicap_info')} | 资金异动：{m.get('odds_movement')}\n"
    p += f"【外部声音(需你批判)】{m.get('expert_intro', '暂无')}\n"
    
    p += "\n【压测准则】\n"
    p += "1. 彻底挑战表象：运用你对球队、教练和伤病的深层认知，击碎庄家通过盘口设置的诱导陷阱。\n"
    p += "2. 拒绝平庸预测：如果双方实力悬殊或防线崩溃，给我残忍的穿盘比分（如1-4, 0-5）；如果是真正的绞肉机之战，给出合理的僵局比分。\n"
    p += "必须严格返回JSON格式，严禁出现Markdown修饰或多余废话：\n"
    p += '{"ai_score":"1-2","analysis":"200字极其冷酷的逻辑复盘，揭露比赛的胜负核心点。"}'
    return p

def build_synthesis_prompt(m, gpt_res, grok_res):
    h, a, lg = m["home_team"], m["away_team"], m.get("league", "")
    p = "作为冷酷的首席决策官兼导师，请对GPT和Grok的意见进行最高级别的压力测试。挑战它们的逻辑漏洞，给出你绝对权威的裁决。\n\n"
    p += f"【赛事】{lg} | {h} vs {a}\n"
    p += f"【GPT 报告】比分: {gpt_res.get('ai_score', '未知')} | 逻辑: {gpt_res.get('analysis', '无')}\n"
    p += f"【Grok 报告】比分: {grok_res.get('ai_score', '未知')} | 逻辑: {grok_res.get('analysis', '无')}\n"
    p += "\n【终极裁决】冷酷地审视这两份报告，剔除感性与荒谬的部分，给出你一锤定音的最终『预测比分』。\n"
    p += "必须严格返回纯JSON格式：\n"
    p += '{"ai_score":"1-2","analysis":"200字终极冷酷裁决，指出你采纳或摒弃它们意见的根本逻辑。"}'
    return p

# 🔥 解除时间枷锁：timeout=600，只要没断网，就让它思考到底！
def call_ai_model(prompt, url, key, model_name, is_gpt_format=True):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    sys_msg = "你是冷酷无情的量化精算师，严格返回纯JSON，不要任何Markdown修饰。"
    print(f"    🤖 等待 AI {model_name} 的独立深度思考 (无时间限制)...")
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
            s = t.find("{"); e = t.rfind("}") + 1
            if s >= 0 and e > s: return json.loads(t[s:e])
        else:
            print(f"    ❌ 接口报错: {r.status_code} ({r.text[:50]})")
    except Exception as e: print(f"    ⚠️ {model_name} 异常: {str(e)[:40]}")
    return {}

def call_gpt(prompt): return call_ai_model(prompt, GPT_API_URL, GPT_API_KEY, "gpt-5.4", True)
def call_grok(prompt): return call_ai_model(prompt, GPT_API_URL, GEMINI_API_KEY, "[次]grok-420-thinking", True) 
def call_gemini(prompt): return call_ai_model(prompt, GEMINI_API_URL, GEMINI_API_KEY, "[次-流抗截]gemini-3.1-pro-preview-thinking", False)

def merge_all(gpt, grok, gemini, stats, match_obj):
    # 🔥 绝对解耦：本地就是本地，坚决不和 AI 揉搓取平均！
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
        
        "gpt_score": gpt.get("ai_score", "-"), "gpt_analysis": gpt.get("analysis", "数据阻断或未响应"),
        "grok_score": grok.get("ai_score", "-"), "grok_analysis": grok.get("analysis", "数据阻断或未响应"),
        "gemini_score": gemini.get("ai_score", "-"), "gemini_analysis": gemini.get("analysis", "数据阻断或未响应"),
        
        "value_bets_summary": v_tags,
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

def run_predictions(raw):
    ms = raw.get("matches", []); res = []
    for i, m in enumerate(ms):
        # 本地走本地
        sp = ensemble.predict(m, {})
        # AI 走 AI 
        ind_prompt = build_independent_prompt(m)
        gpt_res = call_gpt(ind_prompt)
        time.sleep(1)
        grok_res = call_grok(ind_prompt)
        time.sleep(1)
        
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
