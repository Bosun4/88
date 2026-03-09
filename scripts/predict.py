import json, requests, time
from config import *
from models import EnsemblePredictor

ensemble = EnsemblePredictor()

# ==================== 1. 量化资金管理引擎 ====================
def calculate_value_bet(prob_pct, odds):
    if not odds or odds <= 1.05: return {"ev": 0.0, "kelly": 0.0, "is_value": False}
    prob = prob_pct / 100.0
    ev = (prob * odds) - 1.0 
    b = odds - 1.0
    q = 1.0 - prob
    if b <= 0: return {"ev": round(ev * 100, 2), "kelly": 0.0, "is_value": False}
    kelly = ((b * prob) - q) / b
    return {"ev": round(ev * 100, 2), "kelly": round(max(0.0, kelly * 0.25) * 100, 2), "is_value": ev > 0.05}

# ==================== 2. AI 提示词构建 ====================
def build_prompt(m, stats_pred, val_h, val_d, val_a):
    h, a, lg = m["home_team"], m["away_team"], m.get("league", "")
    hs, ast = m.get("home_stats", {}), m.get("away_stats", {})
    sp = stats_pred
    
    poi, rf, lr, dc = sp.get("poisson", {}), sp.get("random_forest", {}), sp.get("logistic", {}), sp.get("dixon_coles", {})
    intel = m.get("intelligence", {})
    
    p = "你是顶级量化基金经理。已为你剔除冗余噪音，请基于以下【纯净量化数据】与【独家情报】给出终极预测。\n\n"
    p += f"【比赛对阵】{lg} | {h} vs {a}\n"
    p += f"【专家分析】{m.get('expert_intro', '暂无')}\n"
    p += f"【基本面研报】{m.get('base_face', '暂无')[:250]}\n"
    p += f"【伤停与利空】主队：{intel.get('h_inj')} | 客队：{intel.get('g_inj')}\n"
    p += f"【盘口与异动】{m.get('handicap_info')} | 水位：{m.get('odds_movement')}\n"
    
    if hs: p += f"\n【主队近况】{hs.get('played','?')}场{hs.get('wins','?')}胜{hs.get('draws','?')}平\n"
    if ast: p += f"【客队近况】{ast.get('played','?')}场{ast.get('wins','?')}胜{ast.get('draws','?')}平\n"
            
    p += "\n【核心量化算力】\n"
    p += f"泊松分布: 主{poi.get('home_win',33):.1f}% 平{poi.get('draw',33):.1f}% 客{poi.get('away_win',33):.1f}%\n"
    p += f"Dixon-Coles: 主{dc.get('home_win',33):.1f}% 平{dc.get('draw',33):.1f}% 客{dc.get('away_win',33):.1f}%\n"
    p += f"随机森林: 主{rf.get('home_win',33):.1f}% 平{rf.get('draw',33):.1f}% 客{rf.get('away_win',33):.1f}%\n"
    p += f"逻辑回归: 主{lr.get('home_win',33):.1f}% 平{lr.get('draw',33):.1f}% 客{lr.get('away_win',33):.1f}%\n"
    p += f"融合胜率: 主{sp.get('home_win_pct',33):.1f}% 平{sp.get('draw_pct',33):.1f}% 客{sp.get('away_win_pct',33):.1f}%\n"
    p += f"风控追踪: {sp.get('smart_money_signal', '正常')} | 预期总球: {sp.get('expected_total_goals',2.5):.1f}\n"
        
    p += f"\n【期望值(EV)与凯利仓位】\n"
    p += f"主胜: EV={val_h['ev']}%, 建议注码={val_h['kelly']}%\n"
    p += f"平局: EV={val_d['ev']}%, 建议注码={val_d['kelly']}%\n"
    p += f"客胜: EV={val_a['ev']}%, 建议注码={val_a['kelly']}%\n"
    
    p += "\n综合以上数据，给出最终预测。只返回纯JSON格式，严禁Markdown修饰：\n"
    p += '{"predicted_score":"2-1","home_win_pct":55,"draw_pct":25,"away_win_pct":20,"confidence":70,"result":"主胜","over_under_2_5":"大","both_score":"是","risk_level":"中","analysis":"结合伤停、水位及EV数据进行200字精辟解读","key_factors":["核心因素1","核心因素2"]}'
    return p

# ==================== 3. 独立防崩 AI 调度中枢 ====================
def call_gpt(prompt):
    """GPT 专属链路：恢复 max_tokens，带上 system 指令，解决 400 Bad Request"""
    print("  [GPT 链路启动]")
    headers = {"Authorization": f"Bearer {GPT_API_KEY}", "Content-Type": "application/json"}
    
    messages = [
        {"role": "system", "content": "你是一位管理千万级资金的足球量化精算师，必须严格返回纯JSON格式，禁止输出Markdown代码块修饰符。"},
        {"role": "user", "content": prompt}
    ]
    
    # 恢复你截图里原本的轮询池
    pool = ["gpt-5.4", "gpt-5.3-codex", "gpt-5.2", "gpt-5.1"]
    
    for model in pool:
        try:
            print(f"    🤖 尝试匹配 GPT: {model}...")
            # 🔥 核心修复点：恢复 max_tokens，满足 Novai 代理的硬性验证要求！
            payload = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 1500}
            
            r = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=25)
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"]["content"].strip()
                s = t.find("{"); e = t.rfind("}") + 1
                if s >= 0 and e > s: 
                    print(f"    ✅ {model} 分析成功！")
                    return json.loads(t[s:e])
            else: 
                # 🔥 直接把后台拒绝的理由印在终端上
                print(f"    ❌ {model} 报错: {r.status_code} | 代理回执: {r.text[:80]}")
        except Exception as e:
            print(f"    ⚠️ {model} 异常: {str(e)[:40]}")
            continue
    return None

def call_gemini(prompt):
    """Gemini 专属链路：智能兼容代理，放宽超时时间，解决 403 与超时报错"""
    print("  [Gemini 链路启动]")
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
    
    # 精准匹配你在 Novai 后台带前缀的模型名
    pool = ["[次-流抗截]gemini-3.1-pro-preview-thinking", "gemini-1.5-pro"]
    
    for model in pool:
        try:
            print(f"    🤖 尝试匹配 Gemini: {model}...")
            
            if "generateContent" in GEMINI_API_URL:
                # 官方原生格式
                payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1500}}
            else:
                # 兼容代理格式 (移除 system 角色，将其合并入 user，防止代理报 403 权限错误)
                messages = [
                    {"role": "user", "content": "系统指令：你是一位管理千万级资金的足球量化精算师，必须严格返回纯JSON格式，禁止输出Markdown代码块。\n\n" + prompt}
                ]
                payload = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 1500}

            # 🔥 核心修复点：thinking 思考模型非常慢，必须放宽到 40 秒，否则直接报超时异常
            r = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=40)
            
            if r.status_code == 200:
                if "generateContent" in GEMINI_API_URL:
                    t = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                else:
                    t = r.json()["choices"][0]["message"]["content"].strip()
                    
                s = t.find("{"); e = t.rfind("}") + 1
                if s >= 0 and e > s: 
                    print(f"    ✅ {model} 分析成功！")
                    return json.loads(t[s:e])
            else:
                print(f"    ❌ {model} 报错: {r.status_code} | 代理回执: {r.text[:80]}")
        except Exception as e:
            print(f"    ⚠️ {model} 异常: {str(e)[:40]}")
            continue
    return None

# ==================== 4. 安全融合中枢 ====================
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
        "value_bets_summary": v_tags,
        
        # 传递给前端渲染模型面板
        "poisson": {**stats.get("poisson", {}), "home_expected_goals": stats.get("poisson", {}).get("home_xg", 0), "away_expected_goals": stats.get("poisson", {}).get("away_xg", 0)},
        "dixon_coles": stats.get("dixon_coles", {}), "random_forest": stats.get("random_forest", {}), "logistic": stats.get("logistic", {}),
        "model_consensus": stats.get("model_consensus", 0), "total_models": stats.get("total_models", 4)
    }

# ==================== 5. 排序与主调度 ====================
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
        time.sleep(1)
        gm = call_gemini(prompt)
        
        mg = merge_all(gp, gm, sp, m)
        res.append({**m, "prediction": mg})
        
    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res: r["is_recommended"] = r["id"] in t4ids
    res.sort(key=lambda x: x.get("match_num", ""))
    return res, t4
