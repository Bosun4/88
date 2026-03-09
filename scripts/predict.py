"""
AI 预测与决策中枢 v5.5:
1. [Model Preserving] 严格保留用户指定的 gpt-5.4, gemini-3.1 等尖端模型池。
2. [Intelligence Fusion] 将抓取到的专家点评、伤停利空、实时水位异动全量喂给 AI。
3. [Safety Logic] 引入 isinstance 强校验，彻底修复 'list indices must be integers' 的 TypeError。
4. [Value Metrics] 计算每一场比赛的期望值 (EV) 与建议仓位。
5. [Strategy] 2串1 笛卡尔积优化算法，筛选当日最高期望回报方案。
"""
import json
import requests
import time
import itertools
import math
from config import *
from models import EnsemblePredictor

# 初始化底层 11 核心量化模型
ensemble = EnsemblePredictor()

# ==================== 1. 资金管理引擎 (EV & Kelly) ====================
def calculate_value_bet(prob_pct, odds):
    """
    计算价值投注指标。
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
    # 采用四分之一凯利准则，保持博弈稳健性
    safe_kelly = max(0.0, kelly * 0.25) * 100
    
    return {
        "ev": round(ev * 100, 2), 
        "kelly": round(safe_kelly, 2), 
        "is_value": ev > 0.05 
    }

# ==================== 2. AI 提示词工厂 (全情报注入) ====================
def build_prompt(m, stats_pred, val_h, val_d, val_a):
    """
    构建高密度提示词。包含：专家分析、基本面、伤停情报、水位异动。
    """
    intel = m.get("intelligence", {})
    sp = stats_pred
    
    p = "你是顶级量化精算师。请综合【底层模型】、【伤停情报】、【专家研报】与【盘口异动】给出独立裁判。\n\n"
    p += f"【核心对阵】{m.get('league')} | {m['home_team']} vs {m['away_team']}\n"
    
    # 注入全量文本情报
    p += f"【专家深度分析】{m.get('expert_intro', '暂无')}\n"
    p += f"【官方研报摘要】{m.get('base_face', '暂无')[:300]}\n"
    
    # 注入伤停与利空
    p += "\n【伤停名单与利空情报】\n"
    p += f"主队利空: {intel.get('h_inj')} | {intel.get('h_bad')}\n"
    p += f"客队利空: {intel.get('g_inj')} | {intel.get('g_bad')}\n"
    
    # 注入实时水位与盘口
    p += "\n【庄家盘口与资金动向】\n"
    p += f"让球盘口: {m.get('handicap_info', '无')}\n"
    p += f"实时水位异动: {m.get('odds_movement', '平稳')} (请思考：结合基本面，这属于机构真实防范还是诱盘？)\n"

    # 注入数学模型算力
    p += "\n【底层量化模型算力输出】\n"
    p += f"泊松分布/随机森林综合胜率: 主{sp['home_win_pct']}% 平{sp['draw_pct']}% 客{sp['away_win_pct']}%\n"
    p += f"建议投注期望值: 主胜EV={val_h['ev']}% | 客胜EV={val_a['ev']}%\n"
    
    p += "\n请严格按照以下JSON格式返回，包含预测比分分析，禁止任何额外文字：\n"
    p += '{"predicted_score":"2-1","ai_independent_score":"2-1","home_win_pct":55,"draw_pct":25,"away_win_pct":20,"confidence":70,"result":"主胜","analysis":"200字深度逻辑分析(须结合伤停与盘口)"}'
    return p

# ==================== 3. 极速 AI 调度引擎 (严格保留原始模型) ====================
def call_model(prompt, url, key, model_pool):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for model_name in model_pool:
        try:
            print(f"    🤖 尝试匹配 AI 模型: {model_name}...")
            # 严格保留用户要求的模型型号
            payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
            r = requests.post(url, headers=headers, json=payload, timeout=20)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"].strip()
                s_idx = content.find("{")
                e_idx = content.rfind("}") + 1
                if s_idx >= 0 and e_idx > s_idx:
                    return json.loads(content[s_idx:e_idx])
            else:
                print(f"    ❌ {model_name} 状态码: {r.status_code}")
        except Exception:
            continue
    return None

def call_gpt(p):
    print("  [GPT 链路启动]")
    # 严格按照用户配置，不许修改模型名称
    pool = ["gpt-5.4", "gpt-5.3-codex", "gpt-5.2", "gpt-5.1"]
    return call_model(p, GPT_API_URL, GPT_API_KEY, pool)

def call_gemini(p):
    print("  [Gemini 链路启动]")
    # 严格按照用户配置，不许修改模型名称
    pool = ["gemini-3.1-pro-preview-thinking", "gemini-1.5-pro"]
    return call_model(p, GEMINI_API_URL, GEMINI_API_KEY, pool)

# ==================== 4. 🔥 2串1 联合对冲优化器 (修复 TypeError) ====================
def optimize_parlay(results):
    """
    笛卡尔积组合优化逻辑。
    修复核心：使用 isinstance(m, dict) 严防非预期对象导致崩溃。
    """
    # 1. 过滤出成功的预测结果
    valid_recs = []
    for m in results:
        if isinstance(m, dict) and m.get("is_recommended") and isinstance(m.get("prediction"), dict):
            valid_recs.append(m)
    
    if len(valid_recs) < 2:
        return None
    
    # 2. 按信心度排序，取最优 4 场进行两两组合尝试
    valid_recs.sort(key=lambda x: x["prediction"].get("confidence", 0), reverse=True)
    candidates = valid_recs[:4]
    
    all_combos = []
    for m1, m2 in itertools.combinations(candidates, 2):
        p1, p2 = m1["prediction"], m2["prediction"]
        
        # 提取目标 SP 值
        def fetch_sp(match, res_str):
            if res_str == "主胜": return match.get("sp_home", 1.0)
            if res_str == "客胜": return match.get("sp_away", 1.0)
            return match.get("sp_draw", 1.0)

        o1 = fetch_sp(m1, p1["result"])
        o2 = fetch_sp(m2, p2["result"])
        
        all_combos.append({
            "combo": f"{m1['match_num']} ({p1['result']}) + {m2['match_num']} ({p2['result']})",
            "combined_odds": round(o1 * o2, 2),
            "confidence": round((p1["confidence"] + p2["confidence"]) / 2, 1),
            "match_ids": [m1.get("match_id"), m2.get("match_id")]
        })
    
    # 3. 返回联合信心度最高的组合
    all_combos.sort(key=lambda x: x["confidence"], reverse=True)
    return all_combos[0] if all_combos else None

# ==================== 5. 主预测调度中枢 ====================
def run_predictions(raw):
    ms = raw.get("matches", [])
    res_list = []
    print(f"\n=== 执行全情报 AI 量化复核: 共 {len(ms)} 场 ===")
    
    for i, m in enumerate(ms):
        print(f"  [{i+1}/{len(ms)}] 分析场次: {m['home_team']} vs {m['away_team']}")
        
        # A. 底层 11 核心模型运算
        sp = ensemble.predict(m, {})
        
        # B. 实时价值计算
        v_h = calculate_value_bet(sp["home_win_pct"], m.get("sp_home"))
        v_d = calculate_value_bet(sp["draw_pct"], m.get("sp_draw"))
        v_a = calculate_value_bet(sp["away_win_pct"], m.get("sp_away"))
        
        # C. 调配 AI 进行深度研判
        prompt = build_prompt(m, sp, v_h, v_d, v_a)
        gp_res = call_gpt(prompt)
        gm_res = call_gemini(prompt)
        
        # D. 多方意见融合逻辑
        ai_pool = [x for x in [gp_res, gm_res] if x]
        final_hp = sp["home_win_pct"]
        if ai_pool:
            ai_avg_hp = sum(x.get("home_win_pct", 33) for x in ai_pool) / len(ai_pool)
            # AI 的情报修正权占 40%
            final_hp = round(sp["home_win_pct"] * 0.6 + ai_avg_hp * 0.4, 1)

        # E. 封装展现层数据
        mg = {
            "predicted_score": gm_res.get("predicted_score", sp["predicted_score"]) if gm_res else sp["predicted_score"],
            "home_win_pct": final_hp,
            "draw_pct": sp["draw_pct"],
            "away_win_pct": round(100 - final_hp - sp["draw_pct"], 1),
            "confidence": sp["confidence"],
            "result": "主胜" if final_hp > 42 else ("客胜" if (100 - final_hp - sp["draw_pct"]) > 40 else "平局"),
            "gemini_analysis": gm_res.get("analysis", "AI计算中...") if gm_res else "未响应",
            "gpt_score": gp_res.get("ai_independent_score", "?") if gp_res else "?",
            "gemini_score": gm_res.get("ai_independent_score", "?") if gm_res else "?",
            "value_bets_summary": [f"主胜 EV+{v_h['ev']}%"] if v_h["is_value"] else []
        }
        
        # 更新原始对象，标记强信心场次
        m.update({
            "prediction": mg,
            "is_recommended": sp["confidence"] > 78 or (v_h["is_value"] and final_hp > 55)
        })
        res_list.append(m)
        
    # F. 生成当日最佳 2串1 方案
    best_parlay = optimize_parlay(res_list)
    
    return res_list, best_parlay
