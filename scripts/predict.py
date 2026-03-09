import json
import requests
import time
import itertools
from config import *
from models import EnsemblePredictor

ensemble = EnsemblePredictor()

# ==================== 1. 期望值计算 ====================
def calculate_value_bet(prob_pct, odds):
    """计算凯利公式与期望值"""
    if not odds or odds <= 1.05: 
        return {"ev": 0.0, "is_value": False}
        
    prob = prob_pct / 100.0
    ev = (prob * odds) - 1.0 
    
    return {
        "ev": round(ev * 100, 2), 
        "is_value": ev > 0.05
    }

# ==================== 2. AI 提示词工程 ====================
def build_prompt(m, sp):
    """注入全部专家文本与伤停数据"""
    intel = m.get("intelligence", {})
    
    p = f"你是顶级量化精算师。比赛：{m['home_team']} vs {m['away_team']}\n\n"
    p += f"【专家分析】{m.get('expert_intro', '暂无')}\n"
    p += f"【官方研报】{m.get('base_face', '暂无')[:350]}\n"
    p += f"【伤停利空】主：{intel.get('h_inj')} | 客：{intel.get('g_inj')}\n"
    p += f"【盘口水位】{m.get('handicap_info')} | 异动：{m.get('odds_movement')}\n"
    p += f"【量化模型胜率】主{sp['home_win_pct']}% 平{sp['draw_pct']}% 客{sp['away_win_pct']}%\n"
    
    p += "\n请结合伤停和水位异动给出独立裁决。严格返回JSON格式，不能包含任何多余字符！\n"
    p += "必须包含字段: predicted_score, ai_independent_score, analysis, confidence。"
    return p

# ==================== 3. 稳健型 AI 轮询 (绝不修改模型名) ====================
def call_model(prompt, url, key, model_pool):
    """支持多模型自动降级轮询"""
    headers = {
        "Authorization": f"Bearer {key}", 
        "Content-Type": "application/json"
    }
    
    for model_name in model_pool:
        try:
            print(f"    🤖 尝试匹配 AI: {model_name}...")
            payload = {
                "model": model_name, 
                "messages": [{"role": "user", "content": prompt}], 
                "temperature": 0.3
            }
            r = requests.post(url, headers=headers, json=payload, timeout=20)
            
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"]["content"]
                s = t.find("{")
                e = t.rfind("}") + 1
                if s >= 0 and e > s:
                    return json.loads(t[s:e])
            else:
                print(f"    ❌ {model_name} 接口返回报错: {r.status_code}")
        except Exception as err: 
            print(f"    ⚠️ {model_name} 请求超时或异常")
            continue
            
    return None

# ==================== 4. 2串1 优化引擎 (修复防崩版) ====================
def optimize_parlay(results):
    """2串1 笛卡尔积优化，彻底防御 TypeError"""
    
    # 物理级防崩：确保 m 是字典，且 m["prediction"] 也是字典
    valid_recs = []
    for m in results:
        if isinstance(m, dict):
            is_rec = m.get("is_recommended")
            pred_obj = m.get("prediction")
            if is_rec and isinstance(pred_obj, dict):
                valid_recs.append(m)
                
    if len(valid_recs) < 2: 
        return None
    
    # 按信心度排序
    valid_recs.sort(key=lambda x: x["prediction"].get("confidence", 0), reverse=True)
    m1 = valid_recs[0]
    m2 = valid_recs[1]
    
    p1 = m1["prediction"]
    p2 = m2["prediction"]
    
    def get_sp(match_obj, result_str):
        if result_str == '主胜': return match_obj.get("sp_home", 1.0)
        if result_str == '客胜': return match_obj.get("sp_away", 1.0)
        return match_obj.get("sp_draw", 1.0)

    o1 = get_sp(m1, p1.get("result", ""))
    o2 = get_sp(m2, p2.get("result", ""))

    return {
        "combo": f"{m1.get('match_num', 'X')} + {m2.get('match_num', 'Y')}",
        "combined_odds": round(o1 * o2, 2),
        "confidence": round((p1.get("confidence", 50) + p2.get("confidence", 50)) / 2, 1)
    }

# ==================== 5. 预测主调度 ====================
def run_predictions(raw):
    ms = raw.get("matches", [])
    res_list = []
    print(f"\n=== 启动 11 核心模型 + 双 AI 深度研判 (共 {len(ms)} 场) ===")
    
    # 严格遵从你的要求，一字不改
    gpt_pool = ["gpt-5.4", "gpt-5.3-codex", "gpt-5.2", "gpt-5.1"]
    gemini_pool = ["gemini-3.1-pro-preview-thinking", "gemini-1.5-pro"]
    
    for i, m in enumerate(ms):
        print(f"  [{i+1}/{len(ms)}] 执行深度量化: {m.get('home_team', '未知主队')}")
        
        sp = ensemble.predict(m, {})
        v_h = calculate_value_bet(sp["home_win_pct"], m.get("sp_home", 0))
        
        prompt = build_prompt(m, sp)
        
        # 分别调用两大阵营的 AI
        print("  [GPT 阵营]")
        gp = call_model(prompt, GPT_API_URL, GPT_API_KEY, gpt_pool)
        
        print("  [Gemini 阵营]")
        gm = call_model(prompt, GEMINI_API_URL, GEMINI_API_KEY, gemini_pool)
        
        ai_pool = []
        if gp: ai_pool.append(gp)
        if gm: ai_pool.append(gm)
        
        final_hp = sp["home_win_pct"]
        if len(ai_pool) > 0:
            ai_avg = sum(x.get("home_win_pct", 33) for x in ai_pool) / len(ai_pool)
            final_hp = round(sp["home_win_pct"] * 0.6 + ai_avg * 0.4, 1)

        mg = {
            "predicted_score": gm.get("predicted_score", sp["predicted_score"]) if gm else sp["predicted_score"],
            "home_win_pct": final_hp, 
            "draw_pct": sp["draw_pct"], 
            "away_win_pct": round(100 - final_hp - sp["draw_pct"], 1),
            "confidence": sp["confidence"], 
            "result": "主胜" if final_hp > 42 else "平局",
            "gemini_analysis": gm.get("analysis", "AI计算中，请稍后...") if gm else "未响应",
            "gpt_score": gp.get("ai_independent_score", "?") if gp else "?",
            "gemini_score": gm.get("ai_independent_score", "?") if gm else "?",
            "value_bets_summary": [f"主胜 EV+{v_h['ev']}%"] if v_h["is_value"] else []
        }
        
        m.update({
            "prediction": mg, 
            "is_recommended": sp["confidence"] > 75
        })
        res_list.append(m)
        
    best_parlay = optimize_parlay(res_list)
    return res_list, best_parlay
