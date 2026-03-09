"""
AI 预测与决策中枢 (经典架构重铸版):
1. [资金管理] 恢复 1/4 凯利公式与 EV 期望值精算。
2. [架构回退] 恢复经典的 merge_all 融合中枢与 select_top4 打分系统。
3. [通道分离] GPT(带system)与Gemini(无system)严格隔离，精准对齐代理商模型。
4. [情报注入] 将问彩的伤停、基本面、水位异动无缝对接给 AI。
"""
import json
import requests
import time
import itertools
from config import *
from models import EnsemblePredictor

ensemble = EnsemblePredictor()

# ==================== 1. 量化资金管理引擎 ====================
def calculate_value_bet(prob_pct, odds):
    """
    计算 EV (期望值) 和 凯利建议注码 (采用稳健的 1/4 凯利策略)
    """
    if not odds or odds <= 1.05:
        return {"ev": 0.0, "kelly": 0.0, "is_value": False}
        
    prob = prob_pct / 100.0
    ev = (prob * odds) - 1.0 
    
    b = odds - 1.0
    q = 1.0 - prob
    if b <= 0: 
        return {"ev": round(ev * 100, 2), "kelly": 0.0, "is_value": False}
        
    kelly = ((b * prob) - q) / b
    safe_kelly = max(0.0, kelly * 0.25) * 100 # 1/4 凯利
    
    return {
        "ev": round(ev * 100, 2), 
        "kelly": round(safe_kelly, 2), 
        "is_value": ev > 0.05 
    }

# ==================== 2. AI 提示词工厂 ====================
def build_prompt(m, stats_pred, val_h, val_d, val_a):
    """结合问彩情报与凯利资金数据的终极提示词"""
    intel = m.get("intelligence", {})
    sp = stats_pred
    
    p = "你是管理千万级资金的足球量化精算师。以下是基本面、机构情报与期望值，请综合判断。\n\n"
    p += f"【赛事对阵】{m.get('league', '')} | {m['home_team']} vs {m['away_team']}\n"
    
    p += "\n【机构研报与利空情报】\n"
    p += f"- 专家分析: {m.get('expert_intro', '暂无')}\n"
    p += f"- 官方研报: {m.get('base_face', '暂无')[:200]}\n"
    p += f"- 主队伤停/利空: {intel.get('h_inj', '无')} | {intel.get('h_bad', '无')}\n"
    p += f"- 客队伤停/利空: {intel.get('g_inj', '无')} | {intel.get('g_bad', '无')}\n"
    
    p += "\n【盘口水位与资金期望】\n"
    p += f"- 让球盘口: {m.get('handicap_info', '无')} | 水位异动: {m.get('odds_movement', '平稳')}\n"
    p += f"- 模型胜率: 主胜{sp.get('home_win_pct',33)}%, 平局{sp.get('draw_pct',33)}%, 客胜{sp.get('away_win_pct',33)}%\n"
    p += f"- 资金期望: 主胜EV={val_h['ev']}%(仓位{val_h['kelly']}%) | 客胜EV={val_a['ev']}%(仓位{val_a['kelly']}%)\n"
    
    p += "\n请严格按以下JSON格式返回结果，禁止输出多余的Markdown修饰符：\n"
    p += '{"predicted_score":"2-1","ai_independent_score":"2-1","home_win_pct":55,"draw_pct":25,"away_win_pct":20,"confidence":70,"result":"主胜","analysis":"200字核心逻辑(结合伤停与水位)","key_factors":["因素1","因素2"]}'
    return p

# ==================== 3. 独立 AI 调度引擎 ====================
def call_gpt(prompt):
    print("  [GPT 阵营启动]")
    headers = {"Authorization": f"Bearer {GPT_API_KEY}", "Content-Type": "application/json"}
    sys_msg = "你是一位管理千万级资金的足球量化精算师，必须严格返回纯JSON格式，禁止输出Markdown。"
    
    # 严格遵照指令：5.4 降级至 5.2
    pool = ["gpt-5.4", "gpt-5.3", "gpt-5.2"]
    
    for model in pool:
        try:
            print(f"    🤖 尝试匹配 GPT: {model}...")
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
            r = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=25)
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"]["content"].strip()
                s = t.find("{"); e = t.rfind("}") + 1
                if s >= 0 and e > s: return json.loads(t[s:e])
            else:
                print(f"    ❌ {model} 状态码: {r.status_code}")
        except Exception:
            print(f"    ⚠️ {model} 请求超时或异常")
            continue
    return None

def call_gemini(prompt):
    print("  [Gemini 阵营启动]")
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
    
    # 精准匹配代理商的定制模型名，且不带 system 角色防 403
    pool = ["[次-流抗截]gemini-3.1-pro-preview-thinking", "gemini-1.5-pro"]
    
    for model in pool:
        try:
            print(f"    🤖 尝试匹配 Gemini: {model}...")
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
            r = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"]["content"].strip()
                s = t.find("{"); e = t.rfind("}") + 1
                if s >= 0 and e > s: return json.loads(t[s:e])
            else:
                print(f"    ❌ {model} 状态码: {r.status_code}")
        except Exception:
            print(f"    ⚠️ {model} 请求超时或异常")
            continue
    return None

# ==================== 4. 核心融合中枢 (防崩兜底) ====================
def merge_all(gpt, gemini, stats, match_obj):
    """恢复经典的融合逻辑，完美容错 AI 宕机，并对齐前端所有 UI 字段"""
    # 物理级过滤无效返回值
    is_gpt_valid = isinstance(gpt, dict)
    is_gemini_valid = isinstance(gemini, dict)
    ai_preds = []
    if is_gpt_valid: ai_preds.append(gpt)
    if is_gemini_valid: ai_preds.append(gemini)
    
    # 概率融合：即便 AI 全挂了，stats 依然能顶住 (绝不崩溃)
    if ai_preds:
        ai_h = sum(x.get("home_win_pct", 33) for x in ai_preds) / len(ai_preds)
        ai_d = sum(x.get("draw_pct", 33) for x in ai_preds) / len(ai_preds)
        ai_a = sum(x.get("away_win_pct", 33) for x in ai_preds) / len(ai_preds)
        ai_cf = sum(x.get("confidence", 50) for x in ai_preds) / len(ai_preds)
        
        hp = ai_h * 0.4 + stats.get("home_win_pct", 33) * 0.6 
        dp = ai_d * 0.4 + stats.get("draw_pct", 33) * 0.6
        ap = ai_a * 0.4 + stats.get("away_win_pct", 33) * 0.6
        cf = ai_cf * 0.5 + stats.get("confidence", 50) * 0.5
    else:
        hp = stats.get("home_win_pct", 33)
        dp = stats.get("draw_pct", 33)
        ap = stats.get("away_win_pct", 33)
        cf = stats.get("confidence", 50)
        
    t = hp + dp + ap
    if t > 0: 
        hp = round(hp/t*100, 1); dp = round(dp/t*100, 1); ap = round(100-hp-dp, 1)
    cf = round(min(95, max(25, cf)), 1)
    
    pcts = {"主胜": hp, "平局": dp, "客胜": ap}
    result = max(pcts, key=pcts.get)
    
    # 获取比分
    score = stats.get("predicted_score", "1-1")
    if is_gemini_valid and gemini.get("predicted_score"):
        score = gemini.get("predicted_score")
    elif is_gpt_valid and gpt.get("predicted_score"):
        score = gpt.get("predicted_score")

    # 重新计算资金管理核心 (基于最新融合胜率)
    val_h = calculate_value_bet(hp, match_obj.get("sp_home", 0))
    val_d = calculate_value_bet(dp, match_obj.get("sp_draw", 0))
    val_a = calculate_value_bet(ap, match_obj.get("sp_away", 0))
    
    # 标记高价值标签
    value_tags = []
    if val_h["is_value"]: value_tags.append(f"主胜 EV:+{val_h['ev']}% (仓位:{val_h['kelly']}%)")
    if val_d["is_value"]: value_tags.append(f"平局 EV:+{val_d['ev']}% (仓位:{val_d['kelly']}%)")
    if val_a["is_value"]: value_tags.append(f"客胜 EV:+{val_a['ev']}% (仓位:{val_a['kelly']}%)")
    
    risk = "低" if cf >= 70 else ("中" if cf >= 50 else "高")
    
    # 安全提取 key_factors
    kf_gpt = gpt.get("key_factors", []) if is_gpt_valid else []
    kf_gemini = gemini.get("key_factors", []) if is_gemini_valid else []
    key_factors = list(set((kf_gpt if isinstance(kf_gpt, list) else []) + (kf_gemini if isinstance(kf_gemini, list) else [])))[:6]
    
    return {
        "predicted_score": score, "home_win_pct": hp, "draw_pct": dp, "away_win_pct": ap,
        "confidence": cf, "result": result,
        "over_under_2_5": "大" if stats.get("over_2_5", 50) > 55 else "小",
        "both_score": "是" if stats.get("btts", 50) > 50 else "否", 
        "risk_level": risk,
        "value_bets_summary": value_tags,
        "gpt_analysis": gpt.get("analysis", "未响应") if is_gpt_valid else "未响应",
        "gemini_analysis": gemini.get("analysis", "未响应") if is_gemini_valid else "未响应",
        "gpt_score": gpt.get("ai_independent_score", "?") if is_gpt_valid else "?",
        "gemini_score": gemini.get("ai_independent_score", "?") if is_gemini_valid else "?",
        "key_factors": key_factors,
        "model_consensus": stats.get("model_consensus", 0)
    }

# ==================== 5. 策略生成器 ====================
def select_top4(preds):
    """恢复经典的推荐排序逻辑：优先推荐 EV > 0 的比赛"""
    for p in preds:
        pr = p.get("prediction", {})
        s = pr.get("confidence", 0) * 0.4
        mx = max(pr.get("home_win_pct", 33), pr.get("away_win_pct", 33))
        s += (mx - 33) * 0.3 + pr.get("model_consensus", 0) * 2
        
        # 奖励高期望价值的比赛
        if pr.get("value_bets_summary"):
            s += 15 
            
        p["recommend_score"] = round(s, 2)
        
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    
    # 截取前 4 场作为核心推荐
    top_candidates = preds[:4]
    return top_candidates

def optimize_parlay(top4_matches):
    """基于筛选出的精选赛事，计算最佳 2串1 组合"""
    if len(top4_matches) < 2: 
        return None
        
    m1, m2 = top4_matches[0], top4_matches[1]
    p1, p2 = m1["prediction"], m2["prediction"]
    
    def get_sp(match_obj, result_str):
        if result_str == '主胜': return match_obj.get("sp_home", 1.0)
        if result_str == '客胜': return match_obj.get("sp_away", 1.0)
        return match_obj.get("sp_draw", 1.0)

    o1 = get_sp(m1, p1.get("result", ""))
    o2 = get_sp(m2, p2.get("result", ""))

    return {
        "combo": f"{m1.get('match_num', 'X')} ({p1.get('result', '')}) + {m2.get('match_num', 'Y')} ({p2.get('result', '')})",
        "combined_odds": round(o1 * o2, 2),
        "confidence": round((p1.get("confidence", 50) + p2.get("confidence", 50)) / 2, 1)
    }

# ==================== 6. 总调度中心 ====================
def run_predictions(raw):
    ms = raw.get("matches", [])
    print(f"\n=== Quant Engine: 处理 {len(ms)} 场比赛 ===")
    res = []
    
    for i, m in enumerate(ms):
        print(f"\n[{i+1}/{len(ms)}] {m.get('home_team', '未知主队')} vs {m.get('away_team', '未知客队')}")
        
        # 1. 底层 8 模型算力
        sp = ensemble.predict(m, {})
        
        # 2. 预先计算独立 EV 传给大模型参考
        v_h = calculate_value_bet(sp.get("home_win_pct",33), m.get("sp_home", 0))
        v_d = calculate_value_bet(sp.get("draw_pct",33), m.get("sp_draw", 0))
        v_a = calculate_value_bet(sp.get("away_win_pct",33), m.get("sp_away", 0))
        
        # 3. 组装终极提示词
        prompt = build_prompt(m, sp, v_h, v_d, v_a)
        
        # 4. 双路独立并防崩调用
        gp = call_gpt(prompt)
        time.sleep(1) # 增加延迟，防止并发过高被代理拦截
        gm = call_gemini(prompt)
        
        # 5. 核心数据融合 (绝不断链)
        mg = merge_all(gp, gm, sp, m)
        print(f"  => 预测: {mg['result']} (EV高亮: {mg['value_bets_summary']})")
        
        # 6. 数据回填
        m["prediction"] = mg
        m["match_id"] = m.get("id", i+1)
        res.append(m)
        
    # 7. 策略中心：打分排序并生成 2串1
    t4 = select_top4(res)
    t4ids = [t["match_id"] for t in t4]
    
    # 标记哪些比赛属于推荐位
    for r in res: 
        r["is_recommended"] = r["match_id"] in t4ids
        
    best_parlay = optimize_parlay(t4)
    
    return res, best_parlay
