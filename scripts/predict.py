import json
import requests
import time
import itertools
from config import *
from models import EnsemblePredictor

ensemble = EnsemblePredictor()

def calculate_value_bet(prob_pct, odds):
    if not odds or odds <= 1.05: return {"ev": 0.0, "is_value": False}
    ev = ((prob_pct / 100.0) * odds) - 1.0 
    return {"ev": round(ev * 100, 2), "is_value": ev > 0.05}

def build_prompt(m, sp):
    intel = m.get("intelligence", {})
    p = f"比赛对阵：{m['home_team']} vs {m['away_team']}\n\n"
    p += f"【专家分析】{m.get('expert_intro', '暂无')}\n"
    p += f"【官方研报】{m.get('base_face', '暂无')[:250]}\n"
    p += f"【伤停利空】主：{intel.get('h_inj')} | 客：{intel.get('g_inj')}\n"
    p += f"【盘口水位】{m.get('handicap_info')} | 异动：{m.get('odds_movement')}\n"
    p += f"【模型胜率】主{sp['home_win_pct']}% 平{sp['draw_pct']}% 客{sp['away_win_pct']}%\n"
    p += "\n请结合伤停和盘口给出独立裁决。必须包含字段: predicted_score, ai_independent_score, analysis, confidence。"
    return p

def call_model(prompt, url, key, model_pool):
    """🔥 修复：恢复标准 API Payload，防止报 400 错误"""
    headers = {
        "Authorization": f"Bearer {key}", 
        "Content-Type": "application/json"
    }
    
    for model_name in model_pool:
        try:
            print(f"    🤖 尝试匹配 AI: {model_name}...")
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a professional football data analyst. Output ONLY valid JSON without any markdown formatting."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1200
            }
            r = requests.post(url, headers=headers, json=payload, timeout=20)
            
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"]["content"]
                s = t.find("{"); e = t.rfind("}") + 1
                if s >= 0 and e > s:
                    return json.loads(t[s:e])
            else:
                print(f"    ❌ {model_name} 接口返回报错: {r.status_code}")
        except Exception as err: 
            print(f"    ⚠️ {model_name} 请求超时或异常")
            continue
    return None

def optimize_parlay(results):
    valid_recs = []
    for m in results:
        if isinstance(m, dict):
            is_rec = m.get("is_recommended")
            pred_obj = m.get("prediction")
            if is_rec and isinstance(pred_obj, dict):
                valid_recs.append(m)
                
    if len(valid_recs) < 2: return None
    
    valid_recs.sort(key=lambda x: x["prediction"].get("confidence", 0), reverse=True)
    m1, m2 = valid_recs[0], valid_recs[1]
    p1, p2 = m1["prediction"], m2["prediction"]
    
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

def run_predictions(raw):
    ms = raw.get("matches", []); res_list = []
    print(f"\n=== 启动 11 核心模型 + 双 AI 深度研判 (共 {len(ms)} 场) ===")
    
    # 你的模型池，一字未改！
    gpt_pool = ["gpt-5.4", "gpt-5.3-codex", "gpt-5.2", "gpt-5.1"]
    gemini_pool = ["gemini-3.1-pro-preview-thinking", "gemini-1.5-pro"]
    
    for i, m in enumerate(ms):
        print(f"  [{i+1}/{len(ms)}] 执行深度量化: {m.get('home_team', '未知主队')}")
        sp = ensemble.predict(m, {})
        v_h = calculate_value_bet(sp["home_win_pct"], m.get("sp_home", 0))
        
        prompt = build_prompt(m, sp)
        
        print("  [GPT 阵营]")
        gp = call_model(prompt, GPT_API_URL, GPT_API_KEY, gpt_pool)
        
        print("  [Gemini 阵营]")
        gm = call_model(prompt, GEMINI_API_URL, GEMINI_API_KEY, gemini_pool)
        
        ai_pool = [x for x in [gp, gm] if x]
        final_hp = sp["home_win_pct"]
        if ai_pool:
            final_hp = round(sp["home_win_pct"] * 0.6 + (sum(x.get("home_win_pct", 33) for x in ai_pool)/len(ai_pool)) * 0.4, 1)

        # 🔥 修复：救回你 UI 面板报错 undefined 的那些核心字段！
        mg = {
            "predicted_score": gm.get("predicted_score", sp["predicted_score"]) if gm else sp["predicted_score"],
            "home_win_pct": final_hp, 
            "draw_pct": sp["draw_pct"], 
            "away_win_pct": round(100 - final_hp - sp["draw_pct"], 1),
            "confidence": sp["confidence"], 
            "result": "主胜" if final_hp > 42 else "平局",
            "over_under_2_5": "大" if sp.get("over_2_5", 50) > 55 else "小",
            "both_score": "是" if sp.get("btts", 50) > 50 else "否",
            "risk_level": "低" if sp["confidence"] >= 70 else ("中" if sp["confidence"] >= 50 else "高"),
            "gpt_analysis": gp.get("analysis", "未响应") if gp else "未响应",
            "gemini_analysis": gm.get("analysis", "未响应") if gm else "未响应",
            "gpt_score": gp.get("ai_independent_score", "?") if gp else "?",
            "gemini_score": gm.get("ai_independent_score", "?") if gm else "?",
            "value_bets_summary": [f"主胜 EV+{v_h['ev']}%"] if v_h["is_value"] else []
        }
        
        m.update({"prediction": mg, "is_recommended": sp["confidence"] > 75})
        res_list.append(m)
        
    return res_list, optimize_parlay(res_list)
