import json, requests, time
from config import *
from models import EnsemblePredictor

ensemble = EnsemblePredictor()

def calculate_value_bet(prob_pct, odds):
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

def build_prompt(m, stats_pred, val_h, val_d, val_a):
    h = m["home_team"]
    a = m["away_team"]
    lg = m.get("league", "")
    hs = m.get("home_stats", {})
    ast = m.get("away_stats", {})
    h2h = m.get("h2h", [])
    sp = stats_pred
    
    # 提取精简后的 11 核心模型
    poi = sp.get("poisson", {})
    bt = sp.get("bradley_terry", {})
    rf = sp.get("random_forest", {})
    nn = sp.get("neural_net", {})
    dc = sp.get("dixon_coles", {})
    bay = sp.get("bayesian", {})
    lr = sp.get("logistic", {})
    
    p = "你是顶级足球竞彩分析师与量化基金经理。以下是纯净版量化统计模型的预测结果、机构绝密情报及资金期望值，请综合分析给出最终判断。\n\n"
    p += f"【比赛】{lg} {h} vs {a}\n"
    
    if hs: p += f"【主队】{hs.get('played','?')}场 {hs.get('wins','?')}胜{hs.get('draws','?')}平{hs.get('losses','?')}负 进{hs.get('goals_for','?')}失{hs.get('goals_against','?')} 均进{hs.get('avg_goals_for','?')}均失{hs.get('avg_goals_against','?')} 近况:{hs.get('form','?')}\n"
    if ast: p += f"【客队】{ast.get('played','?')}场 {ast.get('wins','?')}胜{ast.get('draws','?')}平{ast.get('losses','?')}负 进{ast.get('goals_for','?')}失{ast.get('goals_against','?')} 均进{ast.get('avg_goals_for','?')}均失{ast.get('avg_goals_against','?')} 近况:{ast.get('form','?')}\n"
    
    if h2h:
        p += "【交锋】\n"
        for x in h2h[:5]: p += f"{x.get('date','')} {x.get('home','')} {x.get('score','')} {x.get('away','')}\n"
            
    # 🔥 核心升级：将刚才抓取到的绝密情报与伤停名单喂给 AI
    intel = m.get("intelligence", {})
    if intel:
        p += "\n【机构绝密情报与伤停名单】\n"
        if intel.get("home_injury"): p += f"主队伤停: {intel['home_injury']}\n"
        if intel.get("guest_injury"): p += f"客队伤停: {intel['guest_injury']}\n"
        if intel.get("home_bad_news"): p += f"主队利空: {intel['home_bad_news'][:120]}...\n"
        if intel.get("guest_bad_news"): p += f"客队利空: {intel['guest_bad_news'][:120]}...\n"
        if intel.get("match_points"): p += f"机构提要: {intel['match_points']}\n"

    p += "\n【核心模型预测汇总】\n"
    p += f"泊松分布: 主{poi.get('home_win',33):.1f}% 平{poi.get('draw',33):.1f}% 客{poi.get('away_win',33):.1f}% 比分{poi.get('predicted_score','?')} xG:{poi.get('home_xg',1.3):.1f}-{poi.get('away_xg',1.0):.1f}\n"
    p += f"Dixon-Coles: 主{dc.get('home_win',33):.1f}% 平{dc.get('draw',33):.1f}% 客{dc.get('away_win',33):.1f}%\n"
    p += f"BT硬实力模型: 主{bt.get('home_win',33):.1f}% 平{bt.get('draw',33):.1f}% 客{bt.get('away_win',33):.1f}%\n"
    p += f"随机森林: 主{rf.get('home_win',33):.1f}% 平{rf.get('draw',33):.1f}% 客{rf.get('away_win',33):.1f}%\n"
    p += f"神经网络: 主{nn.get('home_win',33):.1f}% 平{nn.get('draw',33):.1f}% 客{nn.get('away_win',33):.1f}%\n"
    p += f"逻辑回归: 主{lr.get('home_win',33):.1f}% 平{lr.get('draw',33):.1f}% 客{lr.get('away_win',33):.1f}%\n"
    p += f"贝叶斯: 主{bay.get('home_win',33):.1f}% 平{bay.get('draw',33):.1f}% 客{bay.get('away_win',33):.1f}%\n"
    p += f"模型融合: 主{sp.get('home_win_pct',33):.1f}% 平{sp.get('draw_pct',33):.1f}% 客{sp.get('away_win_pct',33):.1f}% 共识:{sp.get('model_consensus',0)}/{sp.get('total_models',11)}\n"
    p += f"大2.5球:{sp.get('over_2_5',50):.1f}% 双方进球:{sp.get('btts',50):.1f}%\n"
    p += f"机构追踪: {sp.get('smart_money_signal', '正常')} (预期总进球: {sp.get('expected_total_goals',2.5):.1f})\n"
        
    p += f"\n【期望值(EV)与建议注码】\n"
    p += f"主胜: EV={val_h['ev']}%, 建议注码={val_h['kelly']}%\n"
    p += f"平局: EV={val_d['ev']}%, 建议注码={val_d['kelly']}%\n"
    p += f"客胜: EV={val_a['ev']}%, 建议注码={val_a['kelly']}%\n"
    
    p += "\n综合所有数据与伤停情报，给出最终预测。请严格按以下JSON格式返回结果，不包含多余文字：\n"
    p += '{"predicted_score":"2-1","ai_independent_score":"2-1","home_win_pct":55,"draw_pct":25,"away_win_pct":20,"confidence":70,"result":"主胜","over_under_2_5":"大","both_score":"是","risk_level":"中","analysis":"200字深度逻辑分析(须结合伤停情报、盘口及底层模型)","key_factors":["伤停影响","数据支撑"]}'
    return p

def call_model(prompt, url, key, model_pool):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "你是一位管理千万级资金的足球量化精算师，必须严格返回纯JSON格式，禁止输出Markdown代码块修饰符。"},
        {"role": "user", "content": prompt}
    ]
    for model_name in model_pool:
        try:
            print(f"    🤖 尝试匹配 AI 模型: {model_name}...")
            payload = {"model": model_name, "messages": messages, "temperature": 0.3, "max_tokens": 1200}
            r = requests.post(url, headers=headers, json=payload, timeout=15)
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"]["content"].strip()
                print(f"    ✅ {model_name} 分析成功！")
                start = t.find("{"); end = t.rfind("}") + 1
                if start >= 0 and end > start: return json.loads(t[start:end])
            else: 
                print(f"    ❌ {model_name} 状态码: {r.status_code}")
        except Exception as e:
            print(f"    ⚠️ {model_name} 超时或异常, 切换备用...")
            continue
    return None

def call_gpt(p):
    print("  [GPT 链路启动]")
    return call_model(p, GPT_API_URL, GPT_API_KEY, ["gpt-5.4", "gpt-5.3-codex", "gpt-5.2", "gpt-5.1"])

def call_gemini(p):
    print("  [Gemini 链路启动]")
    gemini_pool = [GEMINI_MODEL] if 'GEMINI_MODEL' in globals() else ["gemini-1.5-pro", "gemini-pro"]
    return call_model(p, GEMINI_API_URL, GEMINI_API_KEY, gemini_pool)

def merge_all(gpt, gemini, stats):
    ai_preds = [x for x in [gpt, gemini] if x]
    hp, dp, ap, cf = stats["home_win_pct"], stats["draw_pct"], stats["away_win_pct"], stats["confidence"]
    
    if ai_preds:
        ai_h = sum(x.get("home_win_pct", 33) for x in ai_preds) / len(ai_preds)
        ai_d = sum(x.get("draw_pct", 33) for x in ai_preds) / len(ai_preds)
        ai_a = sum(x.get("away_win_pct", 33) for x in ai_preds) / len(ai_preds)
        hp = ai_h * 0.45 + hp * 0.55 
        dp = ai_d * 0.45 + dp * 0.55
        ap = ai_a * 0.45 + ap * 0.55
        cf = (sum(x.get("confidence", 50) for x in ai_preds) / len(ai_preds)) * 0.5 + cf * 0.5
        
    t = hp + dp + ap
    if t > 0: hp = round(hp/t*100, 1); dp = round(dp/t*100, 1); ap = round(100-hp-dp, 1)
    cf = round(min(95, max(25, cf)), 1)
    
    pcts = {"主胜": hp, "平局": dp, "客胜": ap}
    result = max(pcts, key=pcts.get)
    
    score = stats.get("predicted_score", "1-1")
    if ai_preds:
        score = next((x.get("predicted_score") for x in ai_preds if x.get("predicted_score")), score)

    od = stats.get("odds", {})
    val_h = calculate_value_bet(hp, od.get("avg_home_odds", 0))
    val_d = calculate_value_bet(dp, od.get("avg_draw_odds", 0))
    val_a = calculate_value_bet(ap, od.get("avg_away_odds", 0))
    
    v_tags = [f"{k} EV:+{v['ev']}% (仓位:{v['kelly']}%)" for k, v in zip(["主胜", "平局", "客胜"], [val_h, val_d, val_a]) if v["is_value"]]
    agree = (gpt.get("result", "") == gemini.get("result", "")) if gpt and gemini else True
    
    return {
        "predicted_score": score, "home_win_pct": hp, "draw_pct": dp, "away_win_pct": ap,
        "confidence": cf, "result": result,
        "over_under_2_5": "大" if stats.get("over_2_5", 50) > 55 else "小",
        "both_score": "是" if stats.get("btts", 50) > 50 else "否", 
        "risk_level": "低" if cf >= 70 else ("中" if cf >= 50 else "高"),
        
        "gpt_analysis": gpt.get("analysis", "未响应") if gpt else "未响应",
        "gemini_analysis": gemini.get("analysis", "未响应") if gemini else "未响应",
        
        # 将 AI 独立比分暴露给前端
        "gpt_score": gpt.get("ai_independent_score", "?") if gpt else "?",
        "gemini_score": gemini.get("ai_independent_score", "?") if gemini else "?",
        
        "key_factors": list(set((gpt.get("key_factors", []) if gpt else []) + (gemini.get("key_factors", []) if gemini else [])))[:6],
        "model_agreement": agree,
        
        "poisson": {**stats.get("poisson", {}), "home_expected_goals": stats.get("poisson", {}).get("home_xg", 0), "away_expected_goals": stats.get("poisson", {}).get("away_xg", 0)},
        "dixon_coles": stats.get("dixon_coles", {}),
        "bradley_terry": stats.get("bradley_terry", {}),
        "bayesian": stats.get("bayesian", {}),
        "random_forest": stats.get("random_forest", {}),
        "neural_net": stats.get("neural_net", {}),
        "logistic": stats.get("logistic", {}),
        
        # 兼容已被踢出的老模型（透传空壳，保证 index.html 完美显示“已停用”）
        "elo": stats.get("elo", {}),
        "monte_carlo": stats.get("monte_carlo", {}),
        "gradient_boost": stats.get("gradient_boost", {}),
        
        "home_form": stats.get("home_form", {}), "away_form": stats.get("away_form", {}),
        "model_consensus": stats.get("model_consensus", 0), "total_models": stats.get("total_models", 11),
        "value_bets_summary": v_tags, 
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
    ms = raw.get("matches", [])
    od = raw.get("odds", {})
    print(f"\n=== 11 核心精华模型 + 绝密情报 + 2 AI 验证: 共 {len(ms)} 场 ===")
    res = []
    
    for i, m in enumerate(ms):
        print(f"\n[{i+1}/{len(ms)}] {m['home_team']} vs {m['away_team']}")
        
        sp = ensemble.predict(m, od.get(f"{m['home_team']}_{m['away_team']}", {}))
        
        # 日志剔除了已删除的模型，替换为更稳健的 BT 评分
        print("    Poisson:%s BT:H%.0f%% RF:H%.0f%% NN:H%.0f%% Consensus:%d/%d" % (
            sp.get("poisson", {}).get("predicted_score", "?"), 
            sp.get("bradley_terry", {}).get("home_win",33),
            sp.get("random_forest", {}).get("home_win",33), 
            sp.get("neural_net", {}).get("home_win",33),
            sp.get("model_consensus",0), sp.get("total_models",11)))
            
        od_stats = sp.get("odds", {})
        v_h, v_d, v_a = [calculate_value_bet(sp.get(k,33), od_stats.get(o,0)) for k, o in [("home_win_pct", "avg_home_odds"), ("draw_pct", "avg_draw_odds"), ("away_win_pct", "avg_away_odds")]]
        
        prompt = build_prompt(m, sp, v_h, v_d, v_a)
        gp = call_gpt(prompt)
        gm = call_gemini(prompt)
        
        mg = merge_all(gp, gm, sp)
        print(f"  => 融合预测: {mg['result']} (EV: {mg['value_bets_summary']})")
        
        res.append({
            "match_id": m.get("id", i+1), "league": m.get("league", ""), "date": m.get("date", ""),
            "home_team": m["home_team"], "away_team": m["away_team"], "home_logo": m.get("home_logo", ""),
            "away_logo": m.get("away_logo", ""), "home_stats": m.get("home_stats", {}), "away_stats": m.get("away_stats", {}),
            "h2h": m.get("h2h", []), "prediction": mg, "match_num": m.get("match_num", "")
        })
        
    t4 = select_top4(res)
    t4ids = [t["match_id"] for t in t4]
    for r in res: 
        r["is_recommended"] = r["match_id"] in t4ids
    
    res.sort(key=lambda x: x.get("match_num", ""))
    return res, t4
