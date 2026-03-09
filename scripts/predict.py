import json, requests, time
from config import *
from models import EnsemblePredictor

ensemble = EnsemblePredictor()

# ==================== 量化资金管理引擎 ====================
def calculate_value_bet(prob_pct, odds):
    """计算 EV (期望值) 和 1/4 凯利建议注码"""
    if not odds or odds <= 1.05:
        return {"ev": 0.0, "kelly": 0.0, "is_value": False}
        
    prob = prob_pct / 100.0
    ev = (prob * odds) - 1.0 
    b = odds - 1.0
    q = 1.0 - prob
    if b <= 0: 
        return {"ev": round(ev * 100, 2), "kelly": 0.0, "is_value": False}
        
    kelly = ((b * prob) - q) / b
    safe_kelly = max(0.0, kelly * 0.25) * 100 
    return {"ev": round(ev * 100, 2), "kelly": round(safe_kelly, 2), "is_value": ev > 0.05}

# ==================== AI 提示词构建 (完美还原) ====================
def build_prompt(m, stats_pred, val_h, val_d, val_a):
    h = m["home_team"]; a = m["away_team"]; lg = m.get("league", "")
    hs = m.get("home_stats", {}); ast = m.get("away_stats", {}); h2h = m.get("h2h", [])
    sp = stats_pred
    
    # 【完整保留】提取所有底层模型数据
    poi = sp.get("poisson", {}); elo = sp.get("elo", {}); mc = sp.get("monte_carlo", {})
    rf = sp.get("random_forest", {}); gb = sp.get("gradient_boost", {}); nn = sp.get("neural_net", {})
    dc = sp.get("dixon_coles", {}); bay = sp.get("bayesian", {}); lr = sp.get("logistic", {})
    
    p = "你是顶级足球竞彩分析师与量化基金经理。以下是14大统计/ML模型的预测结果及资金期望值，请综合分析给出最终判断。\n\n"
    p += "【比赛】%s %s vs %s\n" % (lg, h, a)
    
    if hs: p += "【主队】%s场 %s胜%s平%s负 进%s失%s 均进%s均失%s 零封%s 近况:%s\n" % (hs.get("played","?"),hs.get("wins","?"),hs.get("draws","?"),hs.get("losses","?"),hs.get("goals_for","?"),hs.get("goals_against","?"),hs.get("avg_goals_for","?"),hs.get("avg_goals_against","?"),hs.get("clean_sheets","?"),hs.get("form","?"))
    if ast: p += "【客队】%s场 %s胜%s平%s负 进%s失%s 均进%s均失%s 零封%s 近况:%s\n" % (ast.get("played","?"),ast.get("wins","?"),ast.get("draws","?"),ast.get("losses","?"),ast.get("goals_for","?"),ast.get("goals_against","?"),ast.get("avg_goals_for","?"),ast.get("avg_goals_against","?"),ast.get("clean_sheets","?"),ast.get("form","?"))
    
    if h2h:
        p += "【交锋】\n"
        for x in h2h[:5]: p += "%s %s %s %s\n" % (x["date"], x["home"], x["score"], x["away"])
        
    # 【完美还原】14大模型预测汇总，原封不动地全部喂给 AI
    p += "\n【14大模型预测汇总】\n"
    p += "泊松分布: 主%.1f%% 平%.1f%% 客%.1f%% 比分%s xG:%.1f-%.1f\n" % (poi.get("home_win",33), poi.get("draw",33), poi.get("away_win",33), poi.get("predicted_score","?"), poi.get("home_xg",1.3), poi.get("away_xg",1.0))
    p += "Dixon-Coles: 主%.1f%% 平%.1f%% 客%.1f%%\n" % (dc.get("home_win",33), dc.get("draw",33), dc.get("away_win",33))
    p += "ELO评分: 主%.1f%% 平%.1f%% 客%.1f%% 差值:%.0f\n" % (elo.get("home_win",33), elo.get("draw",33), elo.get("away_win",33), elo.get("elo_diff",0))
    p += "蒙特卡洛(1万次): 主%.1f%% 平%.1f%% 客%.1f%% 均总球:%.1f\n" % (mc.get("home_win",33), mc.get("draw",33), mc.get("away_win",33), mc.get("avg_total_goals",2.5))
    p += "随机森林: 主%.1f%% 平%.1f%% 客%.1f%%\n" % (rf.get("home_win",33), rf.get("draw",33), rf.get("away_win",33))
    p += "梯度提升: 主%.1f%% 平%.1f%% 客%.1f%%\n" % (gb.get("home_win",33), gb.get("draw",33), gb.get("away_win",33))
    p += "神经网络: 主%.1f%% 平%.1f%% 客%.1f%%\n" % (nn.get("home_win",33), nn.get("draw",33), nn.get("away_win",33))
    p += "逻辑回归: 主%.1f%% 平%.1f%% 客%.1f%%\n" % (lr.get("home_win",33), lr.get("draw",33), lr.get("away_win",33))
    p += "贝叶斯: 主%.1f%% 平%.1f%% 客%.1f%%\n" % (bay.get("home_win",33), bay.get("draw",33), bay.get("away_win",33))
    p += "模型融合: 主%.1f%% 平%.1f%% 客%.1f%% 共识:%d/%d\n" % (sp.get("home_win_pct",33), sp.get("draw_pct",33), sp.get("away_win_pct",33), sp.get("model_consensus",0), sp.get("total_models",14))
    p += "大2.5球:%.1f%% 双方进球:%.1f%%\n" % (sp.get("over_2_5",50), sp.get("btts",50))
    p += "比赛节奏: %s (预期总进球: %.1f)\n" % (sp.get("pace_rating","中等"), sp.get("expected_total_goals",2.5))
    p += "机构逆向追踪: %s\n" % sp.get("smart_money_signal", "正常")
    
    if sp.get("odds"):
        od = sp["odds"]
        p += "赔率: 主%.2f 平%.2f 客%.2f 隐含:主%.1f%%平%.1f%%客%.1f%%\n" % (od.get("avg_home_odds",0), od.get("avg_draw_odds",0), od.get("avg_away_odds",0), od.get("implied_home",33), od.get("implied_draw",33), od.get("implied_away",33))
        
    p += f"\n【期望值(EV)与建议注码】\n"
    p += f"主胜: EV={val_h['ev']}%, 建议注码={val_h['kelly']}%\n"
    p += f"平局: EV={val_d['ev']}%, 建议注码={val_d['kelly']}%\n"
    p += f"客胜: EV={val_a['ev']}%, 建议注码={val_a['kelly']}%\n"
    
    p += "\n综合所有数据，给出最终预测。只返回JSON:\n"
    p += '{"predicted_score":"2-1","home_win_pct":55,"draw_pct":25,"away_win_pct":20,"confidence":70,"result":"主胜","over_under_2_5":"大","both_score":"是","risk_level":"中","analysis":"详细200字分析(需包含高阶模型及EV分析)","key_factors":["因素1","因素2","因素3"]}'
    return p

# ==================== AI 调用模块 ====================
def call_model(prompt, url, key, model_pool):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "你是一位管理千万级资金的足球量化精算师，必须严格返回纯JSON格式，禁止输出Markdown代码块修饰符。"},
        {"role": "user", "content": prompt}
    ]
    for model_name in model_pool:
        try:
            print(f"    🤖 尝试匹配模型: {model_name}...")
            payload = {"model": model_name, "messages": messages, "temperature": 0.3, "max_tokens": 1200}
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"]["content"].strip()
                print(f"    ✅ {model_name} 分析成功！")
                start = t.find("{"); end = t.rfind("}") + 1
                if start >= 0 and end > start: return json.loads(t[start:end])
            else: print(f"    ❌ {model_name} 状态码: {r.status_code}")
        except Exception as e:
            print(f"    ⚠️ {model_name} 异常: {str(e)[:40]}")
            continue
    return None

def call_gpt(p):
    print("  [GPT 链路启动]")
    gpt_pool = ["gpt-5.4", "gpt-5.3-codex", "gpt-5.2", "gpt-5.1"]
    return call_model(p, GPT_API_URL, GPT_API_KEY, gpt_pool)

def call_gemini(p):
    print("  [Gemini 链路启动]")
    gemini_pool = [GEMINI_MODEL] if 'GEMINI_MODEL' in globals() else ["gemini-1.5-pro", "gemini-pro"]
    return call_model(p, GEMINI_API_URL, GEMINI_API_KEY, gemini_pool)

# ==================== 核心融合中枢 ====================
def merge_all(gpt, gemini, stats):
    ai_preds = [x for x in [gpt, gemini] if x]
    
    if ai_preds:
        ai_h = sum(x.get("home_win_pct", 33) for x in ai_preds) / len(ai_preds)
        ai_d = sum(x.get("draw_pct", 33) for x in ai_preds) / len(ai_preds)
        ai_a = sum(x.get("away_win_pct", 33) for x in ai_preds) / len(ai_preds)
        ai_cf = sum(x.get("confidence", 50) for x in ai_preds) / len(ai_preds)
        hp = ai_h * 0.45 + stats["home_win_pct"] * 0.55 
        dp = ai_d * 0.45 + stats["draw_pct"] * 0.55
        ap = ai_a * 0.45 + stats["away_win_pct"] * 0.55
        cf = ai_cf * 0.5 + stats["confidence"] * 0.5
    else:
        hp = stats["home_win_pct"]; dp = stats["draw_pct"]; ap = stats["away_win_pct"]
        cf = stats["confidence"]
        
    t = hp + dp + ap
    if t > 0: hp = round(hp/t*100, 1); dp = round(dp/t*100, 1); ap = round(100-hp-dp, 1)
    cf = round(min(95, max(25, cf)), 1)
    
    pcts = {"主胜": hp, "平局": dp, "客胜": ap}; result = max(pcts, key=pcts.get)
    score = stats.get("predicted_score", "1-1")
    if ai_preds:
        ais = [x.get("predicted_score", "") for x in ai_preds if x.get("predicted_score")]
        if ais: score = ais[0]

    # 资金管理计算核心
    od = stats.get("odds", {})
    val_h = calculate_value_bet(hp, od.get("avg_home_odds", 0))
    val_d = calculate_value_bet(dp, od.get("avg_draw_odds", 0))
    val_a = calculate_value_bet(ap, od.get("avg_away_odds", 0))
    
    value_tags = []
    if val_h["is_value"]: value_tags.append(f"主胜 EV:+{val_h['ev']}% (仓位:{val_h['kelly']}%)")
    if val_d["is_value"]: value_tags.append(f"平局 EV:+{val_d['ev']}% (仓位:{val_d['kelly']}%)")
    if val_a["is_value"]: value_tags.append(f"客胜 EV:+{val_a['ev']}% (仓位:{val_a['kelly']}%)")
    
    risk = "低" if cf >= 70 else ("中" if cf >= 50 else "高")
    agree = (gpt.get("result", "") == gemini.get("result", "")) if gpt and gemini else True
    if agree and gpt and gemini: cf = min(cf + 3, 95)
    
    return {
        # --- 1. 核心结果区 ---
        "predicted_score": score, "home_win_pct": hp, "draw_pct": dp, "away_win_pct": ap,
        "confidence": cf, "result": result,
        "over_under_2_5": "大" if stats.get("over_2_5", 50) > 55 else "小",
        "both_score": "是" if stats.get("btts", 50) > 50 else "否", "risk_level": risk,
        
        # --- 2. AI 解析区 ---
        "gpt_analysis": gpt.get("analysis", "未响应") if gpt else "未响应",
        "gemini_analysis": gemini.get("analysis", "未响应") if gemini else "未响应",
        "analysis": "",
        "key_factors": list(set((gpt.get("key_factors", []) if gpt else []) + (gemini.get("key_factors", []) if gemini else [])))[:6],
        "gpt_score": gpt.get("predicted_score", "?") if gpt else "?",
        "gemini_score": gemini.get("predicted_score", "?") if gemini else "?",
        "model_agreement": agree,
        
        # --- 3. 所有底层模型区 (完美还原，一个不漏) ---
        "poisson": {**stats.get("poisson", {}), "home_expected_goals": stats.get("poisson", {}).get("home_xg", 0), "away_expected_goals": stats.get("poisson", {}).get("away_xg", 0)},
        "dixon_coles": stats.get("dixon_coles", {}),
        "elo": stats.get("elo", {}),
        "bradley_terry": stats.get("bradley_terry", {}),
        "monte_carlo": stats.get("monte_carlo", {}),
        "bayesian": stats.get("bayesian", {}),
        "random_forest": stats.get("random_forest", {}),
        "gradient_boost": stats.get("gradient_boost", {}),
        "neural_net": stats.get("neural_net", {}),
        "logistic": stats.get("logistic", {}),
        "home_form": stats.get("home_form", {}),
        "away_form": stats.get("away_form", {}),
        "odds_analysis": stats.get("odds", {}),
        "top_scores": stats.get("poisson", {}).get("top_scores", []) if "poisson" in stats else [],
        "over_2_5_pct": stats.get("over_2_5", 50),
        "btts_pct": stats.get("btts", 50),
        
        # --- 4. 融合汇总与高阶量化区 ---
        "model_consensus": stats.get("model_consensus", 0),
        "total_models": stats.get("total_models", 14),
        "value_bets_summary": value_tags, 
        "home_ev": val_h, "draw_ev": val_d, "away_ev": val_a,
        "smart_money_signal": stats.get("smart_money_signal", "正常"),
        "expected_total_goals": stats.get("expected_total_goals", 0),
        "pace_rating": stats.get("pace_rating", "中等")
    }

def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})
        s = pr.get("confidence", 0) * 0.4
        if pr.get("model_agreement"): s += 12
        mx = max(pr.get("home_win_pct", 33), pr.get("away_win_pct", 33))
        s += (mx - 33) * 0.3 + pr.get("model_consensus", 0) * 2
        
        if pr.get("risk_level") == "低": s += 8
        elif pr.get("risk_level") == "高": s -= 5
        
        if pr.get("value_bets_summary"): s += 15 
        p["recommend_score"] = round(s, 2)
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]

def run_predictions(raw):
    ms = raw.get("matches", []); od = raw.get("odds", {})
    print("\n=== 14 Stats + 2 AI = 16 Model Ensemble: %d matches ===" % len(ms))
    res = []
    for i, m in enumerate(ms):
        print("\n[%d/%d] %s vs %s" % (i+1, len(ms), m["home_team"], m["away_team"]))
        odds_key = f"{m['home_team']}_{m['away_team']}"
        match_odds = od.get(odds_key, {})
        
        # 1. 运行 14 大量化统计模型
        print("  1.Stats(14 models)...")
        sp = ensemble.predict(m, match_odds)
        
        # 完美还原你最初的华丽日志输出
        print("    Poisson:%s MC:%s RF:H%.0f%% GB:H%.0f%% NN:H%.0f%% Consensus:%d/%d" % (
            sp["poisson"]["predicted_score"], 
            sp["monte_carlo"].get("top_scores",[{}])[0].get("score","?") if "monte_carlo" in sp else "?",
            sp["random_forest"].get("home_win",33), 
            sp["gradient_boost"].get("home_win",33), 
            sp["neural_net"].get("home_win",33),
            sp.get("model_consensus",0), sp.get("total_models",14)))
            
        od_stats = sp.get("odds", {})
        v_h = calculate_value_bet(sp.get("home_win_pct",33), od_stats.get("avg_home_odds",0))
        v_d = calculate_value_bet(sp.get("draw_pct",33), od_stats.get("avg_draw_odds",0))
        v_a = calculate_value_bet(sp.get("away_win_pct",33), od_stats.get("avg_away_odds",0))
        
        # 2. AI 模型交叉验证
        print("  2.AI models...")
        prompt = build_prompt(m, sp, v_h, v_d, v_a)
        gp = call_gpt(prompt); time.sleep(1)
        gm = call_gemini(prompt)
        
        # 3. 最终融合输出
        print("  3.Final merge...")
        mg = merge_all(gp, gm, sp)
        print("  => %s (%s) %.1f%% [consensus:%d]" % (mg["result"], mg["predicted_score"], mg["confidence"], mg.get("model_consensus",0)))
        
        res.append({
            "match_id": m.get("id", i+1), "league": m.get("league", ""), "league_logo": m.get("league_logo", ""),
            "home_team": m["home_team"], "away_team": m["away_team"], "home_logo": m.get("home_logo", ""),
            "away_logo": m.get("away_logo", ""), "match_time": m.get("date", ""),
            "home_stats": m.get("home_stats", {}), "away_stats": m.get("away_stats", {}),
            "h2h": m.get("h2h", [])[:5], "prediction": mg, "match_num": m.get("match_num", "")
        })
        
    t4 = select_top4(res); t4ids = [t["match_id"] for t in t4]
    for r in res: r["is_recommended"] = r["match_id"] in t4ids
    return res, t4
