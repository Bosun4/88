import json, requests, time
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
    hs, ast = m.get("home_stats", {}), m.get("away_stats", {})
    intel = m.get("intelligence", {})
    
    p = "你是顶级量化基金经理。请基于以下量化数据与独家情报给出终极预测。\n\n"
    p += f"【比赛对阵】{lg} | {h} vs {a}\n"
    p += f"【专家分析】{m.get('expert_intro', '暂无')}\n"
    p += f"【伤停与利空】主队：{intel.get('h_inj')} | 客队：{intel.get('g_inj')}\n"
    p += f"【盘口与异动】{m.get('handicap_info')} | 水位：{m.get('odds_movement')}\n"
    p += f"【风控系统预警】{sp.get('smart_money_signal', '无异常')}\n"
    
    p += "\n【4大核心量化算力】\n"
    p += f"融合胜率: 主{sp.get('home_win_pct',33):.1f}% 平{sp.get('draw_pct',33):.1f}% 客{sp.get('away_win_pct',33):.1f}% (共识度:{sp.get('model_consensus',0)}/4)\n"
    p += f"预期总进球: {sp.get('expected_total_goals',2.5):.1f}\n"
        
    p += f"\n【期望值(EV)与凯利仓位】\n"
    p += f"主胜: EV={val_h['ev']}%, 建议注码={val_h['kelly']}%\n"
    p += f"平局: EV={val_d['ev']}%, 建议注码={val_d['kelly']}%\n"
    p += f"客胜: EV={val_a['ev']}%, 建议注码={val_a['kelly']}%\n"
    
    p += "\n综合以上数据，给出最终预测。只返回纯JSON格式，严禁Markdown修饰：\n"
    p += '{"predicted_score":"2-1","home_win_pct":55,"draw_pct":25,"away_win_pct":20,"confidence":70,"result":"主胜","over_under_2_5":"大","both_score":"是","risk_level":"中","analysis":"结合伤停、水位及EV数据进行200字精辟解读","key_factors":["核心因素1","核心因素2"]}'
    return p

def call_model(prompt, url, key, model_pool, is_gpt=True):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    sys_msg = "你是一位管理千万级资金的足球量化精算师，必须严格返回纯JSON格式，禁止输出Markdown。"
    for model_name in model_pool:
        try:
            print(f"    🤖 尝试 AI: {model_name}...")
            payload = {"model": model_name, "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}] if is_gpt else [{"role": "user", "content": prompt}], "temperature": 0.3}
            r = requests.post(url, headers=headers, json=payload, timeout=25)
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"]["content"].strip() if not "generateContent" in url else r.json()["candidates"][0]["content"]["parts"][0]["text"]
                s = t.find("{"); e = t.rfind("}") + 1
                if s >= 0 and e > s: return json.loads(t[s:e])
            else: print(f"    ❌ {model_name} 报错: {r.status_code}")
        except Exception: continue
    return None

def call_gpt(p): return call_model(p, GPT_API_URL, GPT_API_KEY, ["gpt-5.4", "gpt-5.3", "gpt-5.2"], True)
def call_gemini(p): return call_model(p, GEMINI_API_URL, GEMINI_API_KEY, ["[次-流抗截]gemini-3.1-pro-preview-thinking", "gemini-1.5-pro"], False)

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
    score = next((x.get("predicted_score") for x in ai_preds if x.get("predicted_score")), stats.get("predicted_score", "1-1"))

    val_h = calculate_value_bet(hp, match_obj.get("sp_home", 0))
    val_d = calculate_value_bet(dp, match_obj.get("sp_draw", 0))
    val_a = calculate_value_bet(ap, match_obj.get("sp_away", 0))
    v_tags = [f"{k} EV:+{v['ev']}% (仓位:{v['kelly']}%)" for k, v in zip(["主胜", "平局", "客胜"], [val_h, val_d, val_a]) if v and v.get("is_value")]
    
    return {
        "predicted_score": score, "home_win_pct": hp, "draw_pct": dp, "away_win_pct": ap,
        "confidence": cf, "result": result,
        "over_under_2_5": "大" if stats.get("over_2_5", 50) > 55 else "小",
        "both_score": "是" if stats.get("btts", 50) > 50 else "否", 
        "risk_level": "低" if cf >= 70 else ("中" if cf >= 50 else "高"),
        "gpt_analysis": gpt.get("analysis", "未响应") if isinstance(gpt, dict) else "未响应",
        "gemini_analysis": gemini.get("analysis", "未响应") if isinstance(gemini, dict) else "未响应",
        "smart_money_signal": stats.get("smart_money_signal", "正常"),
        "value_bets_summary": v_tags
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
    ms = raw.get("matches", []); od = raw.get("odds", {})
    res = []
    for i, m in enumerate(ms):
        sp = ensemble.predict(m, od.get(f"{m['home_team']}_{m['away_team']}", {}))
        v_h = calculate_value_bet(sp.get("home_win_pct",33), m.get("sp_home",0))
        v_d = calculate_value_bet(sp.get("draw_pct",33), m.get("sp_draw",0))
        v_a = calculate_value_bet(sp.get("away_win_pct",33), m.get("sp_away",0))
        
        prompt = build_prompt(m, sp, v_h, v_d, v_a)
        gp = call_gpt(prompt)
        gm = call_gemini(prompt)
        
        mg = merge_all(gp, gm, sp, m)
        res.append({**m, "prediction": mg})
        
    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res: r["is_recommended"] = r["id"] in t4ids
    res.sort(key=lambda x: x.get("match_num", ""))
    return res, t4
