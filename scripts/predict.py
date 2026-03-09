import json, requests, time
from config import *
from models import EnsemblePredictor

ensemble = EnsemblePredictor()

# ==================== 量化资金管理引擎 ====================
def calculate_value_bet(prob_pct, odds):
    """
    计算 EV (期望值) 和 凯利建议注码 (采用稳健的 1/4 凯利策略)
    """
    if not odds or odds <= 1.05:
        return {"ev": 0.0, "kelly": 0.0, "is_value": False}
        
    prob = prob_pct / 100.0
    ev = (prob * odds) - 1.0 # 期望值公式
    
    b = odds - 1.0
    q = 1.0 - prob
    if b <= 0: 
        return {"ev": round(ev * 100, 2), "kelly": 0.0, "is_value": False}
        
    kelly = ((b * prob) - q) / b
    safe_kelly = max(0.0, kelly * 0.25) * 100 # 1/4 凯利，避免满仓爆仓风险
    
    return {
        "ev": round(ev * 100, 2), 
        "kelly": round(safe_kelly, 2), 
        "is_value": ev > 0.05 # EV 大于 5% 视为高价值盘口 (Value Bet)
    }

# ==================== AI 提示词与调用 ====================
def build_prompt(m, stats_pred, val_h, val_d, val_a):
    """构建提示词：加入凯利公式和期望值数据，让 AI 像基金经理一样思考"""
    h = m["home_team"]; a = m["away_team"]; lg = m.get("league", "")
    hs = m.get("home_stats", {}); ast = m.get("away_stats", {}); h2h = m.get("h2h", [])
    sp = stats_pred; poi = sp.get("poisson", {}); elo = sp.get("elo", {})
    
    p = "你是顶级体育量化基金经理。以下是多维模型预测与赔率期望值，请综合判断给出最终投资建议。\n\n"
    p += f"【赛事】{lg} {h} vs {a}\n"
    
    if hs: p += f"【主队】战绩:{hs.get('played','?')}场{hs.get('wins','?')}胜{hs.get('draws','?')}平{hs.get('losses','?')}负, 近况:{hs.get('form','?')}\n"
    if ast: p += f"【客队】战绩:{ast.get('played','?')}场{ast.get('wins','?')}胜{ast.get('draws','?')}平{ast.get('losses','?')}负, 近况:{ast.get('form','?')}\n"
    
    p += "\n【统计模型与资金数据】\n"
    p += f"- 模型共识看好度: {sp.get('model_consensus',0)}/10\n"
    p += f"- 期望值(EV)与建议注码(Kelly):\n"
    p += f"  主胜: 胜率{round(sp.get('home_win_pct',33),1)}%, 赔率{sp.get('odds',{}).get('avg_home_odds','-')}, EV={val_h['ev']}%, 建议注码={val_h['kelly']}%\n"
    p += f"  平局: 胜率{round(sp.get('draw_pct',33),1)}%, 赔率{sp.get('odds',{}).get('avg_draw_odds','-')}, EV={val_d['ev']}%, 建议注码={val_d['kelly']}%\n"
    p += f"  客胜: 胜率{round(sp.get('away_win_pct',33),1)}%, 赔率{sp.get('odds',{}).get('avg_away_odds','-')}, EV={val_a['ev']}%, 建议注码={val_a['kelly']}%\n"
    
    p += "\n请严格按以下JSON格式返回结果，不包含多余文字：\n"
    p += '{"predicted_score":"2-1","home_win_pct":55,"draw_pct":25,"away_win_pct":20,"confidence":70,"result":"主胜","over_under_2_5":"大","both_score":"是","risk_level":"中","analysis":"200字核心逻辑(须包含EV和凯利分析)","key_factors":["因素1","因素2"]}'
    return p

def call_model(prompt, url, key, unused_model=None):
    """自动轮询大模型并注入 System Instructions"""
    model_pool = ["gpt-5.4", "gpt-5.3-codex", "gpt-5.2", "gpt-5.1"]
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "你是一位管理千万级资金的足球量化精算师，必须严格返回纯JSON格式，禁止输出Markdown代码块修饰符。"},
        {"role": "user", "content": prompt}
    ]

    for model_name in model_pool:
        try:
            print(f"    🤖 正在尝试匹配模型: {model_name}...")
            payload = {"model": model_name, "messages": messages, "temperature": 0.3, "max_tokens": 1200}
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"]["content"].strip()
                print(f"    ✅ {model_name} 分析成功！")
                start = t.find("{"); end = t.rfind("}") + 1
                if start >= 0 and end > start: return json.loads(t[start:end])
            else:
                print(f"    ❌ {model_name} 状态码: {r.status_code}")
        except Exception as e:
            print(f"    ⚠️ {model_name} 异常: {str(e)[:40]}")
            continue
    return None

def call_gpt(p): return call_model(p, GPT_API_URL, GPT_API_KEY)
def call_gemini(p): return call_model(p, GEMINI_API_URL, GEMINI_API_KEY)

# ==================== 核心融合中枢 ====================
def merge_all(gpt, gemini, stats):
    ai_preds = [x for x in [gpt, gemini] if x]
    
    # 概率融合
    if ai_preds:
        ai_h = sum(x.get("home_win_pct", 33) for x in ai_preds) / len(ai_preds)
        ai_d = sum(x.get("draw_pct", 33) for x in ai_preds) / len(ai_preds)
        ai_a = sum(x.get("away_win_pct", 33) for x in ai_preds) / len(ai_preds)
        ai_cf = sum(x.get("confidence", 50) for x in ai_preds) / len(ai_preds)
        hp = ai_h * 0.4 + stats["home_win_pct"] * 0.6 # 量化数据权重大于AI
        dp = ai_d * 0.4 + stats["draw_pct"] * 0.6
        ap = ai_a * 0.4 + stats["away_win_pct"] * 0.6
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

    # --- 资金管理计算核心 ---
    od = stats.get("odds", {})
    val_h = calculate_value_bet(hp, od.get("avg_home_odds", 0))
    val_d = calculate_value_bet(dp, od.get("avg_draw_odds", 0))
    val_a = calculate_value_bet(ap, od.get("avg_away_odds", 0))
    
    # 标记高价值标签
    value_tags = []
    if val_h["is_value"]: value_tags.append(f"主胜 EV:+{val_h['ev']}% (仓位:{val_h['kelly']}%)")
    if val_d["is_value"]: value_tags.append(f"平局 EV:+{val_d['ev']}% (仓位:{val_d['kelly']}%)")
    if val_a["is_value"]: value_tags.append(f"客胜 EV:+{val_a['ev']}% (仓位:{val_a['kelly']}%)")
    
    risk = "低" if cf >= 70 else ("中" if cf >= 50 else "高")
    
    return {
        "predicted_score": score, "home_win_pct": hp, "draw_pct": dp, "away_win_pct": ap,
        "confidence": cf, "result": result,
        "over_under_2_5": "大" if stats.get("over_2_5", 50) > 55 else "小",
        "both_score": "是" if stats.get("btts", 50) > 50 else "否", "risk_level": risk,
        
        # 将价值盘口数据直接注入 JSON，供前端提取高亮
        "value_bets_summary": value_tags,
        "home_ev": val_h, "draw_ev": val_d, "away_ev": val_a,
        
        "gpt_analysis": gpt.get("analysis", "未响应") if gpt else "未响应",
        "gemini_analysis": gemini.get("analysis", "未响应") if gemini else "未响应",
        "key_factors": list(set((gpt.get("key_factors", []) if gpt else []) + (gemini.get("key_factors", []) if gemini else [])))[:6],
        "poisson": {**stats.get("poisson", {}), "home_expected_goals": stats.get("poisson", {}).get("home_xg", 0), "away_expected_goals": stats.get("poisson", {}).get("away_xg", 0)},
        "elo": stats.get("elo", {}),
        "model_consensus": stats.get("model_consensus", 0)
    }

def select_top4(preds):
    """排序推荐逻辑升级：优先推荐 EV > 0 的比赛"""
    for p in preds:
        pr = p.get("prediction", {})
        s = pr.get("confidence", 0) * 0.4
        mx = max(pr.get("home_win_pct", 33), pr.get("away_win_pct", 33))
        s += (mx - 33) * 0.3 + pr.get("model_consensus", 0) * 2
        
        # 奖励高期望价值的比赛
        if pr.get("value_bets_summary"):
            s += 15 # 有套利空间，优先级大幅提升
            
        p["recommend_score"] = round(s, 2)
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]

def run_predictions(raw):
    ms = raw.get("matches", []); od = raw.get("odds", {})
    print(f"\n=== Quant Engine: 处理 {len(ms)} 场比赛 ===")
    res = []
    for i, m in enumerate(ms):
        print(f"\n[{i+1}/{len(ms)}] {m['home_team']} vs {m['away_team']}")
        odds_key = f"{m['home_team']}_{m['away_team']}"
        match_odds = od.get(odds_key, {})
        
        sp = ensemble.predict(m, match_odds)
        
        # 提取赔率并预先计算EV，以便传给大模型分析
        od_stats = sp.get("odds", {})
        v_h = calculate_value_bet(sp.get("home_win_pct",33), od_stats.get("avg_home_odds",0))
        v_d = calculate_value_bet(sp.get("draw_pct",33), od_stats.get("avg_draw_odds",0))
        v_a = calculate_value_bet(sp.get("away_win_pct",33), od_stats.get("avg_away_odds",0))
        
        prompt = build_prompt(m, sp, v_h, v_d, v_a)
        gp = call_gpt(prompt); time.sleep(1)
        gm = call_gemini(prompt)
        
        mg = merge_all(gp, gm, sp)
        print(f"  => 预测: {mg['result']} (EV高亮: {mg['value_bets_summary']})")
        
        res.append({
            "match_id": m.get("id", i+1), "league": m.get("league", ""),
            "home_team": m["home_team"], "away_team": m["away_team"],
            "prediction": mg, "match_num": m.get("match_num", "")
        })
        
    t4 = select_top4(res); t4ids = [t["match_id"] for t in t4]
    for r in res: r["is_recommended"] = r["match_id"] in t4ids
    return res, t4
