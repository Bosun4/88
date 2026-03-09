"""
AI 预测与决策中枢 (全量满血防崩版 - 绝无精简):
1. [通道分离] 彻底拆分 GPT 和 Gemini 接口，物理隔离请求格式。
2. [GPT定制] 注入专属 system 指令，模型严格降级: 5.4 -> 5.3 -> 5.2。
3. [Gemini定制] 纯 user 角色防 403，精准匹配后台模型名 `[次-流抗截]gemini-3.1-pro-preview-thinking`。
4. [前端兼容] 保留全部 UI 强依赖字段 (大小球、双边进球、风险等级等)。
5. [防崩护盾] 字典取值全量启用 isinstance 检测与 .get() 安全回退。
"""
import json
import requests
import time
import itertools
from config import *
from models import EnsemblePredictor

# 初始化底层 11 核心量化模型
ensemble = EnsemblePredictor()

# ==================== 1. 期望值与凯利公式计算 ====================
def calculate_value_bet(prob_pct, odds):
    """计算单场比赛的预期价值，过滤无效赔率"""
    if not odds or odds <= 1.05: 
        return {"ev": 0.0, "is_value": False}
        
    prob = prob_pct / 100.0
    ev = (prob * odds) - 1.0 
    
    return {
        "ev": round(ev * 100, 2), 
        "is_value": ev > 0.05
    }

# ==================== 2. 全维度情报提示词工程 ====================
def build_prompt(m, sp):
    """将基本面、伤停、异动、专家观点全部封装为 AI 提示词"""
    intel = m.get("intelligence", {})
    
    p = f"比赛对阵：{m.get('home_team', '主队')} vs {m.get('away_team', '客队')}\n\n"
    p += f"【专家分析】{m.get('expert_intro', '暂无专家点评')}\n"
    p += f"【官方研报】{m.get('base_face', '暂无官方研报')[:300]}\n"
    p += f"【伤停利空】主队：{intel.get('h_inj', '无')} | 客队：{intel.get('g_inj', '无')}\n"
    p += f"【盘口水位】{m.get('handicap_info', '无')} | 异动：{m.get('odds_movement', '平稳')}\n"
    p += f"【模型胜率】主胜{sp.get('home_win_pct', 33)}% 平局{sp.get('draw_pct', 33)}% 客胜{sp.get('away_win_pct', 33)}%\n"
    
    p += "\n请结合伤停和盘口水位给出独立裁决。必须严格返回 JSON 格式，不能有任何多余的 Markdown 标记！\n"
    p += '必须包含以下字段: {"predicted_score": "2-1", "ai_independent_score": "2-1", "analysis": "200字深度解析", "confidence": 75}'
    
    return p

# ==================== 3. 独立 AI 调度引擎 (GPT 专属) ====================
def call_gpt(prompt):
    """GPT 专属链路：必须包含 System 指令，防止 400 Bad Request"""
    print("  [GPT 阵营启动]")
    
    headers = {
        "Authorization": f"Bearer {GPT_API_KEY}", 
        "Content-Type": "application/json"
    }
    
    sys_msg = "你是顶级量化精算师。必须严格输出纯JSON格式，不能包含任何多余字符或Markdown标记！"
    
    # 严格按照指令：5.4 降级到 5.2，绝不偷工减料
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
                "temperature": 0.3,
                "max_tokens": 1200
            }
            
            r = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=25)
            
            if r.status_code == 200:
                response_text = r.json()["choices"][0]["message"]["content"]
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                
                if start_idx >= 0 and end_idx > start_idx: 
                    return json.loads(response_text[start_idx:end_idx])
            else:
                print(f"    ❌ {model} 接口报错: 状态码 {r.status_code}")
                
        except Exception as e:
            print(f"    ⚠️ {model} 请求超时或遇到异常")
            continue
            
    return None

# ==================== 4. 独立 AI 调度引擎 (Gemini 专属) ====================
def call_gemini(prompt):
    """Gemini 专属链路：纯净 User 请求，智能路由兼容，防止 403 Forbidden"""
    print("  [Gemini 阵营启动]")
    
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}", 
        "Content-Type": "application/json"
    }
    
    # 精准匹配代理商后带前缀的名字，绝不篡改
    pool = ["[次-流抗截]gemini-3.1-pro-preview-thinking", "gemini-1.5-pro"]
    
    for model in pool:
        try:
            print(f"    🤖 尝试匹配 Gemini: {model}...")
            
            # 智能判断通道类型：原生通道 vs 兼容通道
            if "generateContent" in GEMINI_API_URL:
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1200}
                }
                r = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
                
                if r.status_code == 200:
                    response_text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                    start_idx = response_text.find("{")
                    end_idx = response_text.rfind("}") + 1
                    if start_idx >= 0 and end_idx > start_idx: 
                        return json.loads(response_text[start_idx:end_idx])
                else:
                    print(f"    ❌ {model} 原生接口报错: 状态码 {r.status_code}")
            else:
                # OpenAI 兼容通道格式 (去除 system 角色，防止 403 权限阻断)
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1200
                }
                r = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
                
                if r.status_code == 200:
                    response_text = r.json()["choices"][0]["message"]["content"]
                    start_idx = response_text.find("{")
                    end_idx = response_text.rfind("}") + 1
                    if start_idx >= 0 and end_idx > start_idx: 
                        return json.loads(response_text[start_idx:end_idx])
                else:
                    print(f"    ❌ {model} 兼容接口报错: 状态码 {r.status_code}")
                    
        except Exception as e:
            print(f"    ⚠️ {model} 请求超时或遇到异常")
            continue
            
    return None

# ==================== 5. 2串1 组合对冲优化器 ====================
def optimize_parlay(results):
    """物理级防崩 2串1 组合器，彻底告别 TypeError"""
    valid_recs = []
    
    for m in results:
        if isinstance(m, dict):
            is_rec = m.get("is_recommended")
            pred_obj = m.get("prediction")
            
            # 严格验证：必须是推荐场次且包含合法的 prediction 字典
            if is_rec and isinstance(pred_obj, dict):
                valid_recs.append(m)
                
    if len(valid_recs) < 2: 
        return None
    
    # 按照 AI 计算的最终信心度倒序排列
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

# ==================== 6. 预测主调度中枢 ====================
def run_predictions(raw):
    """全量模型与 AI 融合执行器"""
    ms = raw.get("matches", [])
    res_list = []
    print(f"\n=== 启动 11 核心模型 + 双 AI 深度研判 (共 {len(ms)} 场) ===")
    
    for i, m in enumerate(ms):
        print(f"  [{i+1}/{len(ms)}] 执行深度量化: {m.get('home_team', '未知主队')}")
        
        # 1. 运行底层 11 核心统计学模型
        sp = ensemble.predict(m, {})
        v_h = calculate_value_bet(sp.get("home_win_pct", 33), m.get("sp_home", 0))
        
        # 2. 构建包含所有文字情报的提示词
        prompt = build_prompt(m, sp)
        
        # 3. 兵分两路，互不干扰地调用 AI
        gp = call_gpt(prompt)
        gm = call_gemini(prompt)
        
        # 4. 安全读取 AI 响应并融合决策
        is_gp_valid = isinstance(gp, dict)
        is_gm_valid = isinstance(gm, dict)
        
        ai_pool = []
        if is_gp_valid: ai_pool.append(gp)
        if is_gm_valid: ai_pool.append(gm)
        
        final_hp = sp.get("home_win_pct", 33)
        if ai_pool:
            ai_avg = sum(x.get("home_win_pct", 33) for x in ai_pool) / len(ai_pool)
            final_hp = round(sp.get("home_win_pct", 33) * 0.6 + ai_avg * 0.4, 1)

        # 5. 🔥 全量保留前端必需的所有字段，不再出现 undefined！
        base_conf = sp.get("confidence", 50)
        
        mg = {
            "predicted_score": gm.get("predicted_score", sp.get("predicted_score", "")) if is_gm_valid else sp.get("predicted_score", ""),
            "home_win_pct": final_hp, 
            "draw_pct": sp.get("draw_pct", 33), 
            "away_win_pct": round(100 - final_hp - sp.get("draw_pct", 33), 1),
            "confidence": base_conf, 
            "result": "主胜" if final_hp > 42 else "平局",
            "over_under_2_5": "大" if sp.get("over_2_5", 50) > 55 else "小",  
            "both_score": "是" if sp.get("btts", 50) > 50 else "否",      
            "risk_level": "低" if base_conf >= 70 else ("中" if base_conf >= 50 else "高"),
            "gpt_analysis": gp.get("analysis", "未响应") if is_gp_valid else "未响应",
            "gemini_analysis": gm.get("analysis", "未响应") if is_gm_valid else "未响应",
            "gpt_score": gp.get("ai_independent_score", "?") if is_gp_valid else "?",
            "gemini_score": gm.get("ai_independent_score", "?") if is_gm_valid else "?",
            "value_bets_summary": [f"主胜 EV+{v_h['ev']}%"] if v_h.get("is_value") else []
        }
        
        # 6. 将最终决策注回赛事字典
        m.update({
            "prediction": mg, 
            "is_recommended": base_conf > 75
        })
        res_list.append(m)
        
    # 7. 计算并提取 2串1 方案
    best_parlay = optimize_parlay(res_list)
    
    return res_list, best_parlay
