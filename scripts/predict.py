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

def build_prompt(m, sp, val_h, val_d, val_a):
    h, a, lg = m["home_team"], m["away_team"], m.get("league", "")
    intel = m.get("intelligence", {})
    ref_poi = sp.get("refined_poisson", {})
    
    p = "作为掌管亿万资金的顶级量化足球投研专家，请基于以下【量化矩阵数据】与【独家基本面情报】进行独立推演。\n\n"
    p += f"【赛事】{lg} | {h} vs {a}\n"
    p += f"【伤停与利空】主队：{intel.get('h_inj')} | 客队：{intel.get('g_inj')}\n"
    p += f"【盘口与风控】{m.get('handicap_info')} | 异动：{m.get('odds_movement')} | 系统预警：{sp.get('smart_money_signal')}\n"
            
    p += "\n【底层核心量化算力】\n"
    p += f"系统V2高危比分预警: 2-2({ref_poi.get('v2_details',{}).get('p22','0%')}) 1-1({ref_poi.get('v2_details',{}).get('p11','0%')})\n"
    p += f"底层融合胜率: 主{sp.get('home_win_pct',33):.1f}% 平{sp.get('draw_pct',33):.1f}% 客{sp.get('away_win_pct',33):.1f}%\n"
        
    p += f"\n【期望值(EV)与建议仓位】\n"
    p += f"主胜: EV={val_h['ev']}%, 仓位={val_h['kelly']}%\n平局: EV={val_d['ev']}%, 仓位={val_d['kelly']}%\n客胜: EV={val_a['ev']}%, 仓位={val_a['kelly']}%\n"
    
    p += "\n【你的核心任务】\n"
    p += "1. 深度纠偏：不要盲从底层量化胜率，请务必结合「伤停利空」和「水位异动（庄家真实意图）」来寻找冷门或诱盘。\n"
    p += "2. 独立比分推演：结合基本面与V2比分预警，给出你独立判断的最终比分（ai_score）。\n"
    p += "\n严格返回纯JSON格式，严禁任何Markdown修饰或额外解释字符：\n"
    p += '{"ai_score":"1-1","home_win_pct":45,"draw_pct":30,"away_win_pct":25,"confidence":75,"result":"平局","analysis":"200字深度解析，必须说明你给出该ai_score的底层逻辑，以及伤停/水位如何影响了你的判断。","key_factors":["核心因素1","核心因素2"]}'
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
    ai_preds = [x for x in [gpt, gemini] if isinstance(x, dict)]
    hp, dp, ap, cf = stats["home_win_pct"], stats["draw_pct"], stats["away_win_pct"], stats["confidence"]
    
    if ai_preds:
        hp = (sum(x.get("home_win_pct", 33) for x in ai_preds) / len(ai_preds)) * 0.45 + hp * 0.55 
        dp = (sum(x.get("draw_pct", 33) for x in ai_preds) / len(ai_preds)) * 0.45 + dp * 0.55
        ap = (sum(x.get("away_win_pct", 33) for x in ai_preds) / len(ai_preds)) * 0.45 + ap * 0.55
        cf = (sum(x.get("confidence", 50) for x in ai_preds) / len(ai_preds)) * 0.5 + cf * 0.5
        
    t = hp + dp + ap
    if t > 0: hp, dp, ap = round(hp/t*100, 1), round(dp/t*100, 1), round(100-hp-dp, 1)
    result = max({"主胜": hp, "平局": dp, "客胜": ap}, key={"主胜": hp, "平局": dp, "客胜": ap}.get)
    
    system_score = stats.get("predicted_score", "1-1")
    gpt_score = gpt.get("ai_score", "未预测") if isinstance(gpt, dict) else "未预测"
    gemini_score = gemini.get("ai_score", "未预测") if isinstance(gemini, dict) else "未预测"

    val_h = calculate_value_bet(hp, match_obj.get("sp_home", 0))
    val_d = calculate_value_bet(dp, match_obj.get("sp_draw", 0))
    val_a = calculate_value_bet(ap, match_obj.get("sp_away", 0))
    v_tags = [f"{k} EV:+{v['ev']}% (仓位:{v['kelly']}%)" for k, v in zip(["主胜", "平局", "客胜"], [val_h, val_d, val_a]) if v and v.get("is_value")]
    
    return {
        "predicted_score": system_score,
        "gpt_score": gpt_score,
        "gemini_score": gemini_score,
        "home_win_pct": hp, "draw_pct": dp, "away_win_pct": ap,
        "confidence": cf, "result": result,
        "risk_level": "低" if cf >= 70 else ("中" if cf >= 50 else "高"),
        "gpt_analysis": gpt.get("analysis", "未响应") if isinstance(gpt, dict) else "未响应",
        "gemini_analysis": gemini.get("analysis", "未响应") if isinstance(gemini, dict) else "未响应",
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
        v_h = calculate_value_bet(sp.get("home_win_pct",33), m.get("sp_home",0))
        v_d = calculate_value_bet(sp.get("draw_pct",33), m.get("sp_draw",0))
        v_a = calculate_value_bet(sp.get("away_win_pct",33), m.get("sp_away",0))
        
        prompt = build_prompt(m, sp, v_h, v_d, v_a)
        gp = call_gpt(prompt)
        time.sleep(1)
        gm = call_gemini(prompt)
        
        mg = merge_all(gp, gm, sp, m)
        res.append({**m, "prediction": mg})
        
    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res: r["is_recommended"] = r["id"] in t4ids
    
    # 🔥 核心修正：绝对按顺序排列 (从 周一001 开始排)
    def extract_num(match_str):
        nums = re.findall(r'\d+', match_str)
        return int(nums[0]) if nums else 9999
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    
    return res, t4
