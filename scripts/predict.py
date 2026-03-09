import json
import requests
import time
from config import *
from models import EnsemblePredictor

ensemble = EnsemblePredictor()

def build_prompt(m, stats_pred):
    """构建提示词，优化了数据展现格式以提升 AI 理解力"""
    h = m["home_team"]; a = m["away_team"]; lg = m.get("league", "")
    hs = m.get("home_stats", {}); ast = m.get("away_stats", {}); h2h = m.get("h2h", [])
    sp = stats_pred; poi = sp.get("poisson", {}); elo = sp.get("elo", {}); mc = sp.get("monte_carlo", {})
    rf = sp.get("random_forest", {}); gb = sp.get("gradient_boost", {}); nn = sp.get("neural_net", {})
    dc = sp.get("dixon_coles", {}); bay = sp.get("bayesian", {})
    
    p = "你是顶级足球竞彩分析师。以下是9个统计/ML模型的预测结果，请综合分析给出最终判断。\n\n"
    p += f"【比赛】{lg} {h} vs {a}\n"
    
    if hs: p += f"【主队】战绩:{hs.get('played','?')}场{hs.get('wins','?')}胜{hs.get('draws','?')}平{hs.get('losses','?')}负，均进/失:{hs.get('avg_goals_for','?')}/{hs.get('avg_goals_against','?')}, 近况:{hs.get('form','?')}\n"
    if ast: p += f"【客队】战绩:{ast.get('played','?')}场{ast.get('wins','?')}胜{ast.get('draws','?')}平{ast.get('losses','?')}负，均进/失:{ast.get('avg_goals_for','?')}/{ast.get('avg_goals_against','?')}, 近况:{ast.get('form','?')}\n"
    
    if h2h:
        p += "【交锋历史】\n"
        for x in h2h[:5]: p += f"- {x['date']} {x['home']} {x['score']} {x['away']}\n"
        
    p += "\n【模型数据汇总】\n"
    p += f"- 泊松分布: 主胜{poi.get('home_win',33)}% 平{poi.get('draw',33)}% 客胜{poi.get('away_win',33)}% | 预期比分: {poi.get('predicted_score','?')}\n"
    p += f"- Dixon-Coles: 主胜{dc.get('home_win',33)}% 平{dc.get('draw',33)}% 客胜{dc.get('away_win',33)}%\n"
    p += f"- ELO评分: 主胜{elo.get('home_win',33)}% 平{elo.get('draw',33)}% 客胜{elo.get('away_win',33)}% | 差值:{elo.get('elo_diff',0)}\n"
    p += f"- 蒙特卡洛: 主胜{mc.get('home_win',33)}% 平{mc.get('draw',33)}% 客胜{mc.get('away_win',33)}% | 均总球:{mc.get('avg_total_goals',2.5)}\n"
    p += f"- 机器学习(RF/GB/NN): 主胜均值{round((rf.get('home_win',33)+gb.get('home_win',33)+nn.get('home_win',33))/3,1)}%\n"
    p += f"- 综合共识: {sp.get('model_consensus',0)}/10 模型看好此方向\n"
    
    if sp.get("odds"):
        od = sp["odds"]
        p += f"【赔率参考】主{od.get('avg_home_odds',0)} 平{od.get('avg_draw_odds',0)} 客{od.get('avg_away_odds',0)}\n"
        
    p += "\n请严格按以下JSON格式返回，不要包含多余文字：\n"
    p += '{"predicted_score":"2-1","home_win_pct":55,"draw_pct":25,"away_win_pct":20,"confidence":70,"result":"主胜","over_under_2_5":"大","both_score":"是","risk_level":"中","analysis":"200字核心逻辑","key_factors":["因素1","因素2"]}'
    return p

def call_model(prompt, url, key, default_model):
    """
    针对 NAN 平台优化的自动匹配与 Instructions 注入逻辑
    """
    # 模型池：根据你截图中的列表，按优先级排序
    model_pool = ["gpt-5.4", "gpt-5.3-codex", "gpt-5.2", "gpt-5.1", default_model]
    
    # 核心：手动注入 System Instructions
    messages = [
        {"role": "system", "content": "你是一位拥有20年经验的顶级足球精算师。你擅长多维度数据融合分析，请严格以JSON格式输出你的预测结论。"},
        {"role": "user", "content": prompt}
    ]
    
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    for model_name in model_pool:
        try:
            print(f"    🤖 尝试模型: {model_name}...")
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if r.status_code == 200:
                resp = r.json()
                t = resp["choices"][0]["message"]["content"].strip()
                print(f"    ✅ {model_name} 成功返回数据")
                
                # JSON 提取增强逻辑
                start = t.find("{"); end = t.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(t[start:end])
            else:
                print(f"    ❌ {model_name} 失败 (Status: {r.status_code})")
                continue # 失败则尝试池中下一个模型
                
        except Exception as e:
            print(f"    ⚠️ {model_name} 发生异常: {str(e)[:50]}")
            continue
            
    return None

def call_gpt(p):
    print(f"    GPT 链路启动...")
    return call_model(p, GPT_API_URL, GPT_API_KEY, GPT_MODEL)

def call_gemini(p):
    print(f"    Gemini 链路启动...")
    # 这里根据你的需求，也可以改成 call_model 并传入 Gemini 的参数
    return call_model(p, GEMINI_API_URL, GEMINI_API_KEY, GEMINI_MODEL)

def merge_all(gpt, gemini, stats):
    """融合逻辑保持不变，确保结果稳定性"""
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
    score = stats["predicted_score"]
    
    if ai_preds:
        ais = [x.get("predicted_score", "") for x in ai_preds if x.get("predicted_score")]
        if ais: score = ais[0]
        if len(ais) == 2 and ais[0] == ais[1]: cf = min(cf + 5, 95)
        
    agree = (gpt.get("result", "") == gemini.get("result", "")) if (gpt and gemini) else True
    if agree and gpt and gemini: cf = min(cf + 3, 95)
    
    risk = "低" if cf >= 70 else ("中" if cf >= 50 else "高")
    
    return {
        "predicted_score": score, "home_win_pct": hp, "draw_pct": dp, "away_win_pct": ap,
        "confidence": cf, "result": result,
        "over_under_2_5": "大" if stats.get("over_2_5", 50) > 55 else "小",
        "both_score": "是" if stats.get("btts", 50) > 50 else "否", "risk_level": risk,
        "gpt_analysis": gpt.get("analysis", "未响应") if gpt else "未响应",
        "gemini_analysis": gemini.get("analysis", "未响应") if gemini else "未响应",
        "analysis": "", 
        "key_factors": list(set((gpt.get("key_factors", []) if gpt else []) + (gemini.get("key_factors", []) if gemini else [])))[:6],
        "model_agreement": agree,
        "poisson": stats.get("poisson", {}), "model_consensus": stats.get("model_consensus", 0)
    }

def select_top4(preds):
    """排序推荐逻辑"""
    for p in preds:
        pr = p.get("prediction", {}); s = pr.get("confidence", 0) * 0.4
        if pr.get("model_agreement"): s += 12
        mx = max(pr.get("home_win_pct", 33), pr.get("away_win_pct", 33), pr.get("draw_pct", 33))
        s += (mx - 33) * 0.3
        s += pr.get("model_consensus", 0) * 2
        if pr.get("risk_level") == "低": s += 8
        elif pr.get("risk_level") == "高": s -= 5
        p["recommend_score"] = round(s, 2)
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]

def run_predictions(raw):
    """主循环函数"""
    ms = raw.get("matches", []); od = raw.get("odds", {})
    print(f"\n=== 11 Model Ensemble System: 处理 {len(ms)} 场比赛 ===")
    res = []
    for i, m in enumerate(ms):
        print(f"\n[{i+1}/{len(ms)}] {m['home_team']} vs {m['away_team']}")
        odds_key = f"{m['home_team']}_{m['away_team']}"
        match_odds = od.get(odds_key, {})
        
        # 1. 运行统计模型
        sp = ensemble.predict(m, match_odds)
        
        # 2. 运行 AI 模型 (含自动匹配池)
        prompt = build_prompt(m, sp)
        gp = call_gpt(prompt)
        time.sleep(1) # 频率限制保护
        gm = call_gemini(prompt)
        
        # 3. 最终融合
        mg = merge_all(gp, gm, sp)
        print(f"  => 预测: {mg['result']} ({mg['predicted_score']}) 置信度: {mg['confidence']}%")
        
        res.append({
            "match_id": m.get("id", i+1), "league": m.get("league", ""),
            "home_team": m["home_team"], "away_team": m["away_team"],
            "prediction": mg, "match_num": m.get("match_num", "")
        })
        
    t4 = select_top4(res); t4ids = [t["match_id"] for t in t4]
    for r in res: r["is_recommended"] = r["match_id"] in t4ids
    return res, t4
