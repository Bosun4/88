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

# 🔥 核心修改 1：AI 提示词彻底“断奶”，不给它看任何本地算出的概率，让它独立思考！
def build_prompt(m):
    h, a, lg = m["home_team"], m["away_team"], m.get("league", "")
    intel = m.get("intelligence", {})
    hs, ast = m.get("home_stats", {}), m.get("away_stats", {})
    
    p = "你是一位独立思考的顶级足球投研专家。请仅基于以下【基本面数据】与【独家情报】，利用你自身的知识库进行推演。\n\n"
    p += f"【赛事对阵】{lg} | {h} vs {a}\n"
    p += f"【伤停与利空】主队：{intel.get('h_inj')} | 客队：{intel.get('g_inj')}\n"
    p += f"【盘口与异动】{m.get('handicap_info')} | 水位：{m.get('odds_movement')}\n"
    p += f"【专家与研报】{m.get('expert_intro', '暂无')} | {m.get('base_face', '暂无')[:250]}\n"
    
    if hs: p += f"【主队近况】{hs.get('played','?')}场{hs.get('wins','?')}胜{hs.get('draws','?')}平，进{hs.get('goals_for','?')}失{hs.get('goals_against','?')}\n"
    if ast: p += f"【客队近况】{ast.get('played','?')}场{ast.get('wins','?')}胜{ast.get('draws','?')}平，进{ast.get('goals_for','?')}失{ast.get('goals_against','?')}\n"
    
    p += "\n【你的任务】\n"
    p += "不要受外界干扰，利用你对球队实力的认知、伤停的致命程度以及庄家水位的暗示，给出你认为最合理的独立胜率和比分。\n"
    p += "严格返回纯JSON格式，严禁任何Markdown修饰或额外字符：\n"
    p += '{"ai_score":"1-1","ai_home_pct":40,"ai_draw_pct":30,"ai_away_pct":30,"analysis":"200字独立深度解析，必须说明伤停或庄家异动是如何影响你的判断的。"}'
    return p

def call_gpt(prompt):
    headers = {"Authorization": f"Bearer {GPT_API_KEY}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": "你是一位顶级足球量化精算师，必须严格返回纯JSON格式，禁止输出Markdown代码块。"}, {"role": "user", "content": prompt}]
    pool = ["gpt-5.4", "gpt-5.3-codex", "gpt-5.2", "gpt-5.1"]
    for model in pool:
        try:
            payload = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 1500}
            r = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=25)
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"]["content"].strip()
                s = t.find("{"); e = t.rfind("}") + 1
                if s >= 0 and e > s: return json.loads(t[s:e])
        except Exception: continue
    return None

def call_gemini(prompt):
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
    pool = ["[次-流抗截]gemini-3.1-pro-preview-thinking", "gemini-1.5-pro"]
    for model in pool:
        try:
            if "generateContent" in GEMINI_API_URL:
                payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1500}}
            else:
                messages = [{"role": "user", "content": "系统指令：你是一位顶级足球量化精算师，必须严格返回纯JSON格式，禁止输出Markdown。\n\n" + prompt}]
                payload = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 1500}

            r = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=40)
            if r.status_code == 200:
                t = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip() if "generateContent" in GEMINI_API_URL else r.json()["choices"][0]["message"]["content"].strip()
                s = t.find("{"); e = t.rfind("}") + 1
                if s >= 0 and e > s: return json.loads(t[s:e])
        except Exception: continue
    return None

def merge_all(gpt, gemini, stats, match_obj):
    # 🔥 核心修改 2：彻底阻断概率混合！主进度条和EV计算【完全由本地纯数学模型接管】
    sys_hp = stats.get("home_win_pct", 33)
    sys_dp = stats.get("draw_pct", 33)
    sys_ap = stats.get("away_win_pct", 33)
    sys_cf = stats.get("confidence", 50)
    sys_score = stats.get("predicted_score", "1-1")
    
    result = max({"主胜": sys_hp, "平局": sys_dp, "客胜": sys_ap}, key={"主胜": sys_hp, "平局": sys_dp, "客胜": sys_ap}.get)

    val_h = calculate_value_bet(sys_hp, match_obj.get("sp_home", 0))
    val_d = calculate_value_bet(sys_dp, match_obj.get("sp_draw", 0))
    val_a = calculate_value_bet(sys_ap, match_obj.get("sp_away", 0))
    v_tags = [f"{k} EV:+{v['ev']}% (仓位:{v['kelly']}%)" for k, v in zip(["主胜", "平局", "客胜"], [val_h, val_d, val_a]) if v and v.get("is_value")]
    
    # 提取 AI 的独立推演结果打包传给前端
    gpt_pred = gpt if isinstance(gpt, dict) else {}
    gemini_pred = gemini if isinstance(gemini, dict) else {}
    
    return {
        "predicted_score": sys_score,
        "home_win_pct": sys_hp, "draw_pct": sys_dp, "away_win_pct": sys_ap,
        "confidence": sys_cf, "result": result,
        "risk_level": "低" if sys_cf >= 70 else ("中" if sys_cf >= 50 else "高"),
        
        # AI 独立预测舱
        "gpt_score": gpt_pred.get("ai_score", "未预测"),
        "gpt_hp": gpt_pred.get("ai_home_pct", "-"),
        "gpt_dp": gpt_pred.get("ai_draw_pct", "-"),
        "gpt_ap": gpt_pred.get("ai_away_pct", "-"),
        "gpt_analysis": gpt_pred.get("analysis", "未响应"),
        
        "gemini_score": gemini_pred.get("ai_score", "未预测"),
        "gemini_hp": gemini_pred.get("ai_home_pct", "-"),
        "gemini_dp": gemini_pred.get("ai_draw_pct", "-"),
        "gemini_ap": gemini_pred.get("ai_away_pct", "-"),
        "gemini_analysis": gemini_pred.get("analysis", "未响应"),
        
        "smart_money_signal": stats.get("smart_money_signal", "正常"),
        "value_bets_summary": v_tags,
        "over_under_2_5": stats.get("over_2_5", "小"), "both_score": stats.get("btts", "否"),
        
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
        
        # 将赛事仅包含客观情报，直接发给 AI 独立预测
        prompt = build_prompt(m)
        gp = call_gpt(prompt)
        time.sleep(1)
        gm = call_gemini(prompt)
        
        mg = merge_all(gp, gm, sp, m)
        res.append({**m, "prediction": mg})
        
    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res: r["is_recommended"] = r["id"] in t4ids
    
    def extract_num(match_str):
        nums = re.findall(r'\d+', match_str)
        return int(nums[0]) if nums else 9999
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    
    return res, t4
