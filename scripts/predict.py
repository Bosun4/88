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

def build_independent_prompt(m):
    h = m.get("home_team", "主队")
    a = m.get("away_team", "客队")
    lg = m.get("league", "未知赛事")
    sp_h = m.get("sp_home", 0.0)
    sp_d = m.get("sp_draw", 0.0)
    sp_a = m.get("sp_away", 0.0)
    intel = m.get("intelligence", {})
    
    p = f"【系统绝对指令】你是一名冷酷的足球量化风控专家。任务：对【{h}】VS【{a}】进行胜负推演。\n"
    p += f"【警报】绝对禁止提及与本场无关的球队或球星（如姆巴佩等）。\n\n"
    p += f"【基本面】赛事：{lg} | 初始SP：主胜{sp_h} 平局{sp_d} 客胜{sp_a} | 盘口与资金：{m.get('handicap_info')} | {m.get('odds_movement')}\n"
    p += f"【伤停】主队：{intel.get('h_inj')} | 客队：{intel.get('g_inj')}\n"
    
    intro = str(m.get('expert_intro', '')).strip()
    p += f"【情报】{intro[:150] if intro else '无'}\n\n"
    
    p += "【要求】评估伤停与诱盘意图，给出最冷血的比分。\n"
    p += "【格式铁律】必须且只能返回纯JSON对象，绝不允许有Markdown修饰符(如```json)！\n"
    p += '{"ai_score":"1-2","analysis":"不超过100字复盘，只围绕这两支队伍！"}'
    return p

def build_synthesis_prompt(m, gpt_res, grok_res):
    h = m.get("home_team", "主队")
    a = m.get("away_team", "客队")
    
    p = f"【系统绝对指令】你是首席足球数据裁决官。任务：对【{h}】vs【{a}】进行终局判定。\n"
    p += f"GPT 前瞻: 比分 [{gpt_res.get('ai_score', '无')}] | 逻辑 [{gpt_res.get('analysis', '无')}]\n"
    p += f"Grok 前瞻: 比分 [{grok_res.get('ai_score', '无')}] | 逻辑 [{grok_res.get('analysis', '无')}]\n\n"
    
    p += "【要求】交叉比对两份前瞻，摒弃含有幻觉的数据，做出最无情的比分终裁。严禁提及未参与本场分析的AI名字。\n"
    p += "【格式铁律】必须只能返回纯JSON对象，绝不允许包含Markdown修饰符！\n"
    p += '{"ai_score":"1-2","analysis":"不超过100字的终局裁定。"}'
    return p

def extract_clean_json(text):
    text = str(text or "").strip()
    fallback_score = "未预测"
    fallback_analysis = "格式混乱，已启用底线抽取。"
    
    s_match = re.search(r'"ai_score"\s*:\s*"([^"]+)"', text)
    if s_match: fallback_score = s_match.group(1)
    a_match = re.search(r'"analysis"\s*:\s*"(.*?)"', text, re.DOTALL)
    if a_match: fallback_analysis = a_match.group(1).replace('"', "'").replace('\n', ' ').strip()
    
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try: return json.loads(text[start:end+1])
        except Exception: pass
            
    if fallback_score != "未预测":
        return {"ai_score": fallback_score, "analysis": fallback_analysis}
    return None

def get_env_var(name, default=""):
    v = os.environ.get(name)
    if v: return str(v).strip()
    try:
        v_cfg = globals().get(name)
        if v_cfg: return str(v_cfg).strip()
    except Exception: pass
    return default

def call_ai_model(prompt, url, key, model_name):
    if not url or not key:
        print(f"    ❌ 缺少 URL 或 Key 配置，已跳过调用 ({model_name})")
        return {}
        
    url = url.strip()
    key = key.strip()
    
    is_native_gemini = "generateContent" in url
    
    if not is_native_gemini and "chat/completions" not in url:
        url = url.rstrip("/") + "/chat/completions"
        
    headers = {"Content-Type": "application/json"}
    
    if is_native_gemini:
        headers["x-goog-api-key"] = key
        payload = {
            "contents": [{"parts": [{"text": "系统指令：你是无情的JSON输出机。绝对不准带Markdown。\n\n" + prompt}]}], 
            "generationConfig": {"temperature": 0.2}
        }
    else:
        headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": model_name, 
            "messages": [
                {"role": "system", "content": "你是无情的JSON输出机。绝对不准带任何Markdown修饰符。"}, 
                {"role": "user", "content": prompt}
            ], 
            "temperature": 0.2,
            "max_tokens": 500
        }
        
    print(f"    🤖 启动 {model_name} | 网关探测: {'OpenAI中转标准' if not is_native_gemini else '谷歌原生标准'}...")
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=600)
        if r.status_code == 200:
            if is_native_gemini:
                t = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                t = r.json()["choices"][0]["message"]["content"].strip()
                
            parsed_data = extract_clean_json(t)
            if parsed_data:
                raw_analysis = str(parsed_data.get("analysis") or "")
                parsed_data["analysis"] = raw_analysis.replace("```json", "").replace("```", "").strip()
                return parsed_data
            else:
                print("    ❌ 无法解析返回的格式。")
        else:
            print(f"    ❌ API 报错 ({model_name}): {r.status_code} - {r.text[:80]}")
    except Exception as e: 
        print(f"    ⚠️ 网络异常 ({model_name}): {str(e)[:40]}")
    return {}

def call_gpt(prompt): 
    url = get_env_var("GPT_API_URL")
    key = get_env_var("GPT_API_KEY")
    # 🔥 严格锁定为你指定的 gpt-5.3-codex
    return call_ai_model(prompt, url, key, "gpt-5.2-codex")

def call_grok(prompt): 
    url = get_env_var("GROK_API_URL", "[https://api.gemai.cc](https://api.gemai.cc/v1)")
    key = get_env_var("GROK_API_KEY")
    # 🔥 严格锁定为你指定的 grok-4.1-thinking
    return call_ai_model(prompt, url, key, "grok-4.1-thinking")

def call_gemini(prompt): 
    url = get_env_var("GEMINI_API_URL")
    key = get_env_var("GEMINI_API_KEY")
    return call_ai_model(prompt, url, key, "[次-流抗截]gemini-3.1-pro-preview-thinking")

def merge_all(gpt, grok, gemini, stats, match_obj):
    sys_hp, sys_dp, sys_ap = stats.get("home_win_pct", 33), stats.get("draw_pct", 33), stats.get("away_win_pct", 33)
    sys_cf = stats.get("confidence", 50)
    sys_score = stats.get("predicted_score", "1-1")
    
    result = max({"主胜": sys_hp, "平局": sys_dp, "客胜": sys_ap}, key={"主胜": sys_hp, "平局": sys_dp, "客胜": sys_ap}.get)

    val_h = calculate_value_bet(sys_hp, match_obj.get("sp_home", 0))
    val_d = calculate_value_bet(sys_dp, match_obj.get("sp_draw", 0))
    val_a = calculate_value_bet(sys_ap, match_obj.get("sp_away", 0))
    v_tags = [f"{k} EV:+{v['ev']}% (仓位:{v['kelly']}%)" for k, v in zip(["主胜", "平局", "客胜"], [val_h, val_d, val_a]) if v and v.get("is_value")]
    
    return {
        "predicted_score": sys_score, "home_win_pct": sys_hp, "draw_pct": sys_dp, "away_win_pct": sys_ap,
        "confidence": sys_cf, "result": result, "risk_level": "低" if sys_cf >= 70 else ("中" if sys_cf >= 50 else "高"),
        
        "gpt_score": gpt.get("ai_score", "-"), "gpt_analysis": gpt.get("analysis", "未响应或阻断"),
        "grok_score": grok.get("ai_score", "-"), "grok_analysis": grok.get("analysis", "未响应或阻断"),
        "gemini_score": gemini.get("ai_score", "-"), "gemini_analysis": gemini.get("analysis", "未响应或阻断"),
        
        "value_bets_summary": v_tags,
        "extreme_warning": stats.get("extreme_warning", "无"),
        "smart_money_signal": stats.get("smart_money_signal", "正常"),
        "poisson": {**stats.get("poisson", {}), "home_expected_goals": stats.get("poisson", {}).get("home_xg", "?"), "away_expected_goals": stats.get("poisson", {}).get("away_xg", "?")},
        "refined_poisson": stats.get("refined_poisson", {}), "elo": stats.get("elo", {}), 
        "home_form": stats.get("home_form", {}), "away_form": stats.get("away_form", {}),
        "model_consensus": stats.get("model_consensus", 0), "total_models": stats.get("total_models", 4),
        "expected_total_goals": stats.get("expected_total_goals", 0),
        "over_2_5": stats.get("over_2_5", 50),
        "btts": stats.get("btts", 50)
    }

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

def extract_num(match_str):
    week_map = {"一": 1000, "二": 2000, "三": 3000, "四": 4000, "五": 5000, "六": 6000, "日": 7000, "天": 7000}
    base_weight = 0
    for k, v in week_map.items():
        if k in match_str:
            base_weight = v
            break
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
            syn_prompt = build_synthesis_prompt(m, gpt_res or {}, grok_res or {})
            gemini_res = call_gemini(syn_prompt)
        else:
            gpt_res = {"ai_score": "-", "analysis": "历史完场免算。"}
            grok_res = {"ai_score": "-", "analysis": "历史完场免算。"}
            gemini_res = {"ai_score": "-", "analysis": "历史完场免算。"}
            
        mg = merge_all(gpt_res or {}, grok_res or {}, gemini_res or {}, sp, m)
        res.append({**m, "prediction": mg})
        
    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res: r["is_recommended"] = r["id"] in t4ids
    
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    return res, t4
