"""
AI 预测与决策中枢 (精准代理模型版):
1. [模型对齐] 精准匹配代理商后台带前缀的模型名：[次-流抗截]gemini-3.1-pro-preview-thinking。
2. [指令修复] 保留 system 系统角色，防止 400 报错。
3. [强效防崩] 严密的 isinstance 校验，彻底屏蔽 NoneType 崩溃。
4. [全量字段] 保留前端必需的所有大小球、风险评估字段，拒绝 undefined。
"""
import json
import requests
import time
import itertools
from config import *
from models import EnsemblePredictor

# 初始化底层 11 核心量化模型
ensemble = EnsemblePredictor()

# ==================== 1. 期望值计算 ====================
def calculate_value_bet(prob_pct, odds):
    if not odds or odds <= 1.05: 
        return {"ev": 0.0, "is_value": False}
        
    ev = ((prob_pct / 100.0) * odds) - 1.0 
    return {"ev": round(ev * 100, 2), "is_value": ev > 0.05}

# ==================== 2. AI 提示词工程 ====================
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

# ==================== 3. 稳健型 AI 轮询调度 ====================
def call_model(prompt, url, key, model_pool):
    """带 system 指令的 API 调用，防止代理接口报 400 错误"""
    headers = {
        "Authorization": f"Bearer {key}", 
        "Content-Type": "application/json"
    }
    
    sys_msg = "你是顶级量化精算师。必须严格输出纯JSON格式，不能包含任何多余字符或Markdown标记！"
    
    for model_name in model_pool:
        try:
            print(f"    🤖 尝试匹配 AI: {model_name}...")
            
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
            # 放宽超时时间到 30 秒，给这种“次-流抗截”的高延迟代理多一点反应时间
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"]["content"]
                s = t.find("{")
                e = t.rfind("}") + 1
                if s >= 0 and e > s:
                    return json.loads(t[s:e])
            else:
                print(f"    ❌ {model_name} 接口返回报错: {r.status_code} ({r.text[:50]})")
        except Exception as err: 
            print(f"    ⚠️ {model_name} 请求超时或异常")
            continue
            
    return None

# ==================== 4. 2串1 优化引擎 ====================
def optimize_parlay(results):
    valid_recs = []
    
    for m in results:
        if isinstance(m, dict):
            is_rec = m.get("is_recommended")
            pred_obj = m.get("prediction")
            if is_rec and isinstance(pred_obj, dict):
                valid_recs.append(m)
                
    if len(valid_recs) < 2: 
        return None
    
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

# ==================== 5. 预测主调度中枢 ====================
def run_predictions(raw):
    ms = raw.get("matches", [])
    res_list = []
    print(f"\n=== 启动 11 核心模型 + 双 AI 深度研判 (共 {len(ms)} 场) ===")
    
    # 🔥 完全匹配你的 GPT 代理配置
    gpt_pool = ["gpt-5.4", "gpt-5.3", "gpt-5.2"]
    
    # 🔥 核心修复：精准匹配你截图里代理后台的真实名字
    gemini_pool = ["[次-流抗截]gemini-3.1-pro-preview-thinking", "gemini-1.5-pro"]
    
    for i, m in enumerate(ms):
        print(f"  [{i+1}/{len(ms)}] 执行深度量化: {m.get('home_team', '未知主队')}")
        
        sp = ensemble.predict(m, {})
        v_h = calculate_value_bet(sp["home_win_pct"], m.get("sp_home", 0))
        
        prompt = build_prompt(m, sp)
        
        print("  [GPT 阵营]")
        gp = call_model(prompt, GPT_API_URL, GPT_API_KEY, gpt_pool)
        
        print("  [Gemini 阵营]")
        gm = call_model(prompt, GEMINI_API_URL, GEMINI_API_KEY, gemini_pool)
        
        is_gp_valid = isinstance(gp, dict)
        is_gm_valid = isinstance(gm, dict)
        
        ai_pool = []
        if is_gp_valid: ai_pool.append(gp)
        if is_gm_valid: ai_pool.append(gm)
        
        final_hp = sp["home_win_pct"]
        if ai_pool:
            ai_avg = sum(x.get("home_win_pct", 33) for x in ai_pool) / len(ai_pool)
            final_hp = round(sp["home_win_pct"] * 0.6 + ai_avg * 0.4, 1)

        mg = {
            "predicted_score": gm.get("predicted_score", sp.get("predicted_score", "")) if is_gm_valid else sp.get("predicted_score", ""),
            "home_win_pct": final_hp, 
            "draw_pct": sp.get("draw_pct", 33), 
            "away_win_pct": round(100 - final_hp - sp.get("draw_pct", 33), 1),
            "confidence": sp.get("confidence", 50), 
            "result": "主胜" if final_hp > 42 else "平局",
            "over_under_2_5": "大" if sp.get("over_2_5", 50) > 55 else "小",  
            "both_score": "是" if sp.get("btts", 50) > 50 else "否",      
            "risk_level": "低" if sp.get("confidence", 50) >= 70 else ("中" if sp.get("confidence", 50) >= 50 else "高"),
            "gpt_analysis": gp.get("analysis", "未响应") if is_gp_valid else "未响应",
            "gemini_analysis": gm.get("analysis", "未响应") if is_gm_valid else "未响应",
            "gpt_score": gp.get("ai_independent_score", "?") if is_gp_valid else "?",
            "gemini_score": gm.get("ai_independent_score", "?") if is_gm_valid else "?",
            "value_bets_summary": [f"主胜 EV+{v_h['ev']}%"] if v_h.get("is_value") else []
        }
        
        m.update({
            "prediction": mg, 
            "is_recommended": sp.get("confidence", 0) > 75
        })
        res_list.append(m)
        
    best_parlay = optimize_parlay(res_list)
    return res_list, best_parlay
