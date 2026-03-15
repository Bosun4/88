import json, requests, time, re, os
from config import *
from models import EnsemblePredictor

ensemble = EnsemblePredictor()

def calculate_value_bet(prob_pct, odds):
    if not odds or odds <= 1.05: return {"ev": 0.0, "kelly": 0.0, "is_value": False}
    prob = prob_pct / 100.0
    ev = (prob * odds) - 1.0 
    b = odds - 1.0
    q = 1.0 - prob
    if b <= 0: return {"ev": round(ev * 100, 2), "kelly": 0.0, "is_value": False}
    kelly = ((b * prob) - q) / b
    return {"ev": round(ev * 100, 2), "kelly": round(max(0.0, kelly * 0.25) * 100, 2), "is_value": ev > 0.05}

# ➡️ 3名“前线分析师”的专属提示词（独立作业，互不干扰）
def build_independent_prompt(m):
    h, a = m.get("home_team", "主队"), m.get("away_team", "客队")
    lg = m.get("league", "未知赛事")
    sp_h, sp_d, sp_a = m.get("sp_home", 0.0), m.get("sp_draw", 0.0), m.get("sp_away", 0.0)
    intel = m.get("intelligence", {})
    
    p = f"【系统硬指令】你是一名前线足球量化分析师。任务：对【{h}】VS【{a}】进行胜负推演。\n"
    p += f"【警报】绝对禁止提及与本场无关的球队或球星（如姆巴佩等）。\n\n"
    p += f"【基本面】赛事：{lg} | 初始SP：主胜{sp_h} 平局{sp_d} 客胜{sp_a} | 盘口与资金：{m.get('handicap_info')} | {m.get('odds_movement')}\n"
    p += f"【伤停】主队：{intel.get('h_inj')} | 客队：{intel.get('g_inj')}\n"
    
    intro = str(m.get('expert_intro', '')).strip()
    p += f"【情报】{intro[:150] if intro else '无'}\n\n"
    
    p += "【要求】评估伤停与诱盘意图，给出最冷血的比分推演。\n"
    p += "【格式铁律】必须且只能返回纯JSON对象，绝不允许有Markdown修饰符(如```json)！\n"
    p += '{"ai_score":"1-2","analysis":"不超过100字独立复盘，只围绕这两支队伍！"}'
    return p

# ➡️ 1名“总指挥官”的专属提示词（汇总三方报告，拍板决策）
def build_synthesis_prompt(m, gpt_res, grok_res, gemini_res):
    h, a = m.get("home_team", "主队"), m.get("away_team", "客队")
    
    p = f"【系统绝对指令】你是首席风控指挥官(CIO)。任务：对【{h}】vs【{a}】进行最终判定。\n"
    p += "你手下的三名顶尖分析师提交了以下报告：\n"
    p += f"- 分析师1(GPT)判定: 比分 [{gpt_res.get('ai_score', '无')}] | 逻辑 [{gpt_res.get('analysis', '无')}]\n"
    p += f"- 分析师2(Grok)判定: 比分 [{grok_res.get('ai_score', '无')}] | 逻辑 [{grok_res.get('analysis', '无')}]\n"
    p += f"- 分析师3(Gemini)判定: 比分 [{gemini_res.get('ai_score', '无')}] | 逻辑 [{gemini_res.get('analysis', '无')}]\n\n"
    
    p += "【终极要求】交叉比对这三份报告。如果有人出现常识错误或逻辑幻觉，直接摒弃。结合你自身的宗师级足球理解，给出最后的一锤定音。\n"
    p += "【格式铁律】必须只能返回纯JSON对象，绝不允许包含Markdown修饰符！\n"
    p += '{"ai_score":"1-2","analysis":"不超过100字的终局裁定，统领全局。"}'
    return p

def extract_clean_json(text):
    text = str(text or "").strip()
    fallback_score, fallback_analysis = "未预测", "格式混乱，已启用底线抽取。"
    
    s_match = re.search(r'"ai_score"\s*:\s*"([^"]+)"', text)
    if s_match: fallback_score = s_match.group(1)
    a_match = re.search(r'"analysis"\s*:\s*"(.*?)"', text, re.DOTALL)
    if a_match: fallback_analysis = a_match.group(1).replace('"', "'").replace('\n', ' ').strip()
    
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end != -1:
        try: return json.loads(text[start:end+1])
        except Exception: pass
            
    if fallback_score != "未预测":
        return {"ai_score": fallback_score, "analysis": fallback_analysis}
    return None

def get_clean_env_url(name, default="[https://www.api520.pro/v1](https://www.api520.pro/v1)"):
    v = os.environ.get(name, globals().get(name, default))
    v = str(v).strip(' \t\n\r"\'')
    match = re.search(r'(https?://[a-zA-Z0-9.-]+(?:/[^\s)\]]*)?)', v)
    if match: return match.group(1)
    return v

def get_clean_env_key(name):
    v = os.environ.get(name, globals().get(name, ""))
    return str(v).strip(' \t\n\r"\'')

def call_ai_model(prompt, url, key, model_name):
    if not url or not key:
        print(f"    ❌ 缺少 URL 或 Key，已跳过 ({model_name})")
        return {}
        
    is_native_gemini = "generateContent" in url
    if not is_native_gemini and "chat/completions" not in url:
        url = url.rstrip("/") + "/chat/completions"
        
    headers = {"Content-Type": "application/json"}
    
    if is_native_gemini:
        headers["x-goog-api-key"] = key
        payload = {"contents": [{"parts": [{"text": "你是无情的JSON输出机。绝对不准带Markdown。\n" + prompt}]}], "generationConfig": {"temperature": 0.2}}
    else:
        headers["Authorization"] = f"Bearer {key}"
        payload = {"model": model_name, "messages": [{"role": "system", "content": "严禁带Markdown。输出纯JSON。"}, {"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 500}
        
    print(f"    🤖 启动 {model_name}...")
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=600)
        if r.status_code == 200:
            t = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip() if is_native_gemini else r.json()["choices"][0]["message"]["content"].strip()
            parsed_data = extract_clean_json(t)
            if parsed_data:
                parsed_data["analysis"] = str(parsed_data.get("analysis", "")).replace("```json", "").replace("```", "").strip()
                return parsed_data
        else:
            print(f"    ❌ 报错 ({model_name[:15]}): {r.status_code}")
    except Exception as e: 
        print(f"    ⚠️ 异常 ({model_name[:15]}): {str(e)[:40]}")
    return {}

# ================= 4大模型绝对锁定 =================

def call_gpt(prompt): 
    return call_ai_model(prompt, get_clean_env_url("GPT_API_URL"), get_clean_env_key("GPT_API_KEY"), "熊猫-A-7-gpt-5.4")

def call_grok(prompt): 
    return call_ai_model(prompt, get_clean_env_url("GROK_API_URL"), get_clean_env_key("GROK_API_KEY"), "熊猫-A-6-grok-4.2-thinking")

def call_gemini(prompt): 
    return call_ai_model(prompt, get_clean_env_url("GEMINI_API_URL"), get_clean_env_key("GEMINI_API_KEY"), "熊猫特供S-按量-gemini-3-flash-preview")

def call_claude(prompt): 
    return call_ai_model(prompt, get_clean_env_url("CLAUDE_API_URL"), get_clean_env_key("CLAUDE_API_KEY"), "熊猫-按量-顶级特供-官max-claude-opus-4.6-thinking")

# =================================================

def merge_all(gpt, grok, gemini, claude, stats, match_obj):
    sys_hp, sys_dp, sys_ap = stats.get("home_win_pct", 33), stats.get("draw_pct", 33), stats.get("away_win_pct", 33)
    sys_cf, sys_score = stats.get("confidence", 50), stats.get("predicted_score", "1-1")
    result = max({"主胜": sys_hp, "平局": sys_dp, "客胜": sys_ap}, key={"主胜": sys_hp, "平局": sys_dp, "客胜": sys_ap}.get)

    val_h = calculate_value_bet(sys_hp, match_obj.get("sp_home", 0))
    val_d = calculate_value_bet(sys_dp, match_obj.get("sp_draw", 0))
    val_a = calculate_value_bet(sys_ap, match_obj.get("sp_away", 0))
    v_tags = [f"{k} EV:+{v['ev']}%" for k, v in zip(["主胜", "平局", "客胜"], [val_h, val_d, val_a]) if v and v.get("is_value")]
    
    return {
        "predicted_score": sys_score, "home_win_pct": sys_hp, "draw_pct": sys_dp, "away_win_pct": sys_ap,
        "confidence": sys_cf, "result": result, "risk_level": "低" if sys_cf >= 70 else ("中" if sys_cf >= 50 else "高"),
        
        "claude_score": claude.get("ai_score", "-"), "claude_analysis": claude.get("analysis", "总控裁决失败"),
        "gpt_score": gpt.get("ai_score", "-"), "gpt_analysis": gpt.get("analysis", "报告遗失"),
        "grok_score": grok.get("ai_score", "-"), "grok_analysis": grok.get("analysis", "报告遗失"),
        "gemini_score": gemini.get("ai_score", "-"), "gemini_analysis": gemini.get("analysis", "报告遗失"),
        
        "value_bets_summary": v_tags, "extreme_warning": stats.get("extreme_warning", "无"),
        "smart_money_signal": stats.get("smart_money_signal", "正常"),
        "poisson": stats.get("poisson", {}), "refined_poisson": stats.get("refined_poisson", {}), 
        "elo": stats.get("elo", {}), "home_form": stats.get("home_form", {}), "away_form": stats.get("away_form", {}),
        "model_consensus": stats.get("model_consensus", 0), "expected_total_goals": stats.get("expected_total_goals", 0),
        "over_2_5": stats.get("over_2_5", 50), "btts": stats.get("btts", 50)
    }

def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})
        s = pr.get("confidence", 0) * 0.4 + (max(pr.get("home_win_pct", 33), pr.get("away_win_pct", 33)) - 33) * 0.3
        s += 8 if pr.get("risk_level") == "低" else (-5 if pr.get("risk_level") == "高" else 0)
        s += 15 if pr.get("value_bets_summary") else 0
        p["recommend_score"] = round(s, 2)
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]

def extract_num(match_str):
    week_map = {"一": 1000, "二": 2000, "三": 3000, "四": 4000, "五": 5000, "六": 6000, "日": 7000, "天": 7000}
    base_weight = next((v for k, v in week_map.items() if k in match_str), 0)
    nums = re.findall(r'\d+', match_str)
    return base_weight + int(nums[0]) if nums else 9999

def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", []); res = []
    for i, m in enumerate(ms):
        sp = ensemble.predict(m, {})
        
        if use_ai:
            ind_prompt = build_independent_prompt(m)
            gpt_res = call_gpt(ind_prompt)
            grok_res = call_grok(ind_prompt)
            gemini_res = call_gemini(ind_prompt)  # 前线分析师3
            
            # 汇总给总指挥 Claude
            syn_prompt = build_synthesis_prompt(m, gpt_res or {}, grok_res or {}, gemini_res or {})
            claude_res = call_claude(syn_prompt)
        else:
            gpt_res, grok_res, gemini_res, claude_res = [{"ai_score": "-", "analysis": "历史已完场，免算"}] * 4
            
        mg = merge_all(gpt_res or {}, grok_res or {}, gemini_res or {}, claude_res or {}, sp, m)
        res.append({**m, "prediction": mg})
        
    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res: r["is_recommended"] = r["id"] in t4ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    return res, t4
