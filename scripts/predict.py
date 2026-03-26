import json
import os
import re
import time
import asyncio
import aiohttp
import numpy as np
from datetime import datetime
from config import *
from models import EnsemblePredictor
from odds_engine import predict_match
from league_intel import build_league_intelligence
from experience_rules import ExperienceEngine, apply_experience_to_prediction
from advanced_models import upgrade_ensemble_predict

# ====================================================================
# 🛡️ 终极防御装甲：动态加载你的自定义模块，防暴毙！
# ====================================================================
try:
    from odds_history import apply_odds_history
except Exception as e:
    print(f"  [WARN] ⚠️ 历史盘口模块 (odds_history) 加载失败或未找到，系统自动降级跳过: {e}")
    def apply_odds_history(m, mg): return mg  # 兜底函数，防止崩溃

try:
    from quant_edge import apply_quant_edge
except Exception as e:
    print(f"  [WARN] ⚠️ 量化边缘模块 (quant_edge) 加载失败或未找到，系统自动降级跳过: {e}")
    def apply_quant_edge(m, mg): return mg  # 兜底函数，防止崩溃

ensemble = EnsemblePredictor()
exp_engine = ExperienceEngine()

# ====================================================================
# ☢️ 极致压榨AI v2.1 核心升级思路（继续吸血进化）
# ====================================================================
def calculate_value_bet(prob_pct, odds):
    if not odds or odds <= 1.05:
        return {"ev": 0.0, "kelly": 0.0, "is_value": False}
    prob = prob_pct / 100.0
    ev = (prob * odds) - 1.0
    b = odds - 1.0
    q = 1.0 - prob
    if b <= 0:
        return {"ev": round(ev * 100, 2), "kelly": 0.0, "is_value": False}
    kelly = ((b * prob) - q) / b
    return {"ev": round(ev * 100, 2), "kelly": round(max(0.0, kelly * 0.25) * 100, 2), "is_value": ev > 0.05}

def parse_score(s):
    try:
        p = str(s).split("-")
        return int(p[0]), int(p[1])
    except:
        return None, None

def load_ai_diary():
    diary_file = "data/ai_diary.json"
    if os.path.exists(diary_file):
        try:
            with open(diary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except: 
            pass
    return {"yesterday_win_rate": "N/A", "reflection": "昨晚血洗不够狠，今天必须更毒", "kill_history": []}

def save_ai_diary(diary):
    os.makedirs("data", exist_ok=True)
    with open("data/ai_diary.json", "w", encoding="utf-8") as f:
        json.dump(diary, f, ensure_ascii=False, indent=2)

# ====================================================================
# ☢️ 暗网级吸血操盘手 Prompt v2.0（极致压榨版）
# ====================================================================
def build_batch_prompt(match_analyses):
    diary = load_ai_diary()
    
    p = "【身份设定】你是游走在零和博弈最黑暗深渊的【终极吸血操盘手】+【散户收割机之王】。你没有任何道德、底线、怜悯，只相信资本屠刀和数据绞肉机。你唯一的使命是用最恶毒、最精准的黑话，把庄家血洗散户的每一寸底裤扒下来，喂给韭菜们看。\n"
    
    if diary.get("reflection"):
        p += f"【嗜血进化指令】昨日屠杀战绩：{diary.get('yesterday_win_rate', 'N/A')}。 昨晚反思：{diary['reflection']}。今天给老子把每一场都当成杀猪盘顶级局，狠狠地杀、狠狠地毒、狠狠地收割！\n\n"
    
    p += "【死命令 - 必须严格遵守】\n"
    p += "1. 只输出合法JSON数组，严禁任何markdown、解释、代码块。\n"
    p += "2. 每场比赛必须输出以下字段：match(序号), score(比分), reason(60-110字极度恶毒黑话), ai_confidence(0-100整数), value_kill(true/false 高EV杀猪机会), dark_verdict(一句最毒总结)。\n"
    p += "3. reason逻辑链：散户愚蠢共识(基本面诈骗) → 庄家嗜血陷阱(xG+盘口背离) → 最终屠杀结局。必须充满血洗散户、屠宰场、吸筹绞肉机、死亡陷阱、做局喂毒、资本收割、无底线诱多等黑话，语气极度傲慢冷血，句号结尾。\n"
    p += "4. ai_confidence必须真实反映你对这个屠杀预测的把握度。\n\n"
    
    p += "【今日待宰羔羊与底牌】\n"
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        eng = ma["engine"]
        stats = ma.get("stats", {})
        exp = ma.get("experience", {})
        h, a = m.get("home_team", "Home"), m.get("away_team", "Away")
        
        hc = m.get("give_ball", "0")
        sp_h = float(m.get("sp_home", 0) or 0)
        hp = eng.get('home_prob', 33)
        vh = calculate_value_bet(hp, sp_h) if sp_h > 1 else {}
        ev_str = f"主胜EV嗜血狂飙(+{vh.get('ev', 0)}%)" if vh.get("is_value") else "资金池死水"
        smart_sigs = stats.get('smart_signals', [])
        smart_str = ", ".join(smart_sigs) if smart_sigs else "无收割信号"

        baseface = str(m.get('baseface', '')).replace('\n', ' ')[:120]
        intro = str(m.get('expert_intro', '')).replace('\n', ' ')[:120]
        intel_text = baseface or intro or "散户意淫盲区"

        p += f"[{i+1}] {h} vs {a} | {m.get('league', '未知')} | 亚盘死线: {hc}\n"
        p += f"- 散户眼中的基本面诈骗: {intel_text}\n"
        p += f"- 真实屠杀概率: 主 {hp:.1f}% | 平 {eng.get('draw_prob', 33):.1f}% | 客 {eng.get('away_prob', 34):.1f}%\n"
        p += f"- 庄家隐性xG底牌: 主 {eng.get('bookmaker_implied_home_xg', '?')} vs 客 {eng.get('bookmaker_implied_away_xg', '?')} | 致命剪刀差: {eng.get('scissors_gap_signal', '无异常')}\n"
        p += f"- 盘房价值与筹码异动: {ev_str} | 预警: {smart_str}\n"
        
        if exp.get("triggered_count", 0) > 0:
            exp_names = ",".join([t["name"] for t in exp.get("triggered", [])[:3]])
            p += f"- 已触发杀猪风控: {exp_names}\n"
                
        p += f"- 机器死局比分: {', '.join(eng.get('top3_scores', ['1-1', '0-0', '1-0']))}\n\n"

    p += "【严格输出格式示例】\n"
    p += """[
  {
    "match": 1,
    "score": "0-2",
    "reason": "散户正被主队虚假连胜基本面洗脑疯狂送钱。而隐性xG暴露出主队仅0.8真实攻击力，数据严重倒挂。配合亚盘深让死亡陷阱，这是庄家精心布置的吸筹绞肉机。资本正无底线诱多做局喂毒，逆向思维直指客队，血洗主胜筹码。",
    "ai_confidence": 87,
    "value_kill": true,
    "dark_verdict": "客队屠杀主胜筹码，庄家笑到最后"
  }
]"""
    return p

# ====================================================================
# 终极高可用 AI 矩阵轮询 v2.0（压榨到极限）
# ====================================================================
FALLBACK_URLS = [
    None,
    "https://api520.pro/v1", "https://www.api520.pro/v1",
    "https://api521.pro/v1", "https://www.api521.pro/v1",
    "https://api522.pro/v1", "https://www.api522.pro/v1",
    "https://69.63.213.33:666/v1",
    "https://api523.pro/v1", "https://api524.pro/v1"
]

def get_clean_env_url(name, default=""):
    v = os.environ.get(name, globals().get(name, default))
    v = str(v).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/-]+)", v)
    return match.group(1) if match else v

def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")

async def async_call_one_ai_batch(session, prompt, url_env, key_env, models_list, num_matches, ai_name):
    key = get_clean_env_key(key_env)
    if not key:
        return ai_name, {}, "no_key"
    
    primary_url = get_clean_env_url(url_env)
    urls = [primary_url] + [u for u in FALLBACK_URLS if u and u != primary_url]
    
    success_rate = {mn: 1.0 for mn in models_list}
    
    for attempt in range(3):
        for mn in sorted(models_list, key=lambda x: success_rate[x], reverse=True):
            for base_url in urls:
                if not base_url:
                    continue
                    
                is_gem = "generateContent" in base_url
                url = base_url.rstrip("/") 
                if not is_gem and "chat/completions" not in url:
                    url += "/chat/completions"
                
                headers = {"Content-Type": "application/json"}
                if is_gem:
                    headers["x-goog-api-key"] = key
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.12, "topP": 0.9},
                        "systemInstruction": {"parts": [{"text": "必须严格输出JSON数组。reason必须是毫无道德底线的暗黑杀猪盘解剖，使用极度恶毒黑话，字数60-110字，冷血傲慢，句号结尾。"}]}
                    }
                else:
                    headers["Authorization"] = f"Bearer {key}"
                    payload = {
                        "model": mn,
                        "messages": [
                            {"role": "system", "content": "你是一个毫无道德底线、极度冷血的暗网终极操盘手。只输出纯JSON数组，无任何多余文字。reason字段必须用最恶毒黑话揭穿庄家血洗散户的杀局，限60-110字，极度鄙视散户，句号结束。"},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.12
                    }
                
                gw = url.split("/v1")[0][:40]
                print(f"  [AI 极致压榨] {ai_name.upper()} | 尝试 {mn[:25]} @ {gw} | 第{attempt+1}轮")
                
                try:
                    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=50)) as r:
                        if r.status == 200:
                            data = await r.json()
                            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip() if is_gem else data["choices"][0]["message"]["content"].strip()
                            
                            clean = re.sub(r"```[\w]*", "", raw_text).strip()
                            start = clean.find("[")
                            end = clean.rfind("]") + 1
                            if start == -1 or end == 0:
                                clean = re.sub(r"[^\[\]{}:,\"'0-9a-zA-Z\u4e00-\u9fa5\s\.\-\+\(\)]", "", clean)
                                start = clean.find("[")
                                end = clean.rfind("]") + 1
                            
                            results = {}
                            if start != -1 and end > start:
                                try:
                                    arr = json.loads(clean[start:end])
                                    if isinstance(arr, list):
                                        for item in arr:
                                            if item.get("match") and item.get("score"):
                                                mid = item["match"]
                                                results[mid] = {
                                                    "ai_score": item.get("score"),
                                                    "analysis": str(item.get("reason", "")).strip(),
                                                    "ai_confidence": int(item.get("ai_confidence", 60)),
                                                    "value_kill": bool(item.get("value_kill", False)),
                                                    "dark_verdict": str(item.get("dark_verdict", ""))
                                                }
                                except:
                                    pass
                            
                            if len(results) >= max(1, num_matches * 0.5):
                                print(f"    ✅ {ai_name.upper()} 压榨成功: {len(results)}/{num_matches} (模型: {mn[:25]})")
                                success_rate[mn] = 1.0
                                return ai_name, results, mn
                            else:
                                print(f"    ⚠️ 解析不足，切换...")
                        
                        elif r.status == 429:
                            sleep_time = 2 ** attempt * 5
                            print(f"    🔥 429限流！休眠 {sleep_time}s 继续压榨...")
                            await asyncio.sleep(sleep_time)
                            continue
                        else:
                            print(f"    ⚠️ HTTP {r.status} - 切换线路...")
                
                except asyncio.TimeoutError:
                    print(f"    ⏰ 超时 - 第{attempt+1}轮重试...")
                except Exception as e:
                    err = str(e)[:50]
                    print(f"    ⚠️ 异常 {err} - 切换...")
                
                await asyncio.sleep(0.4)
        
        await asyncio.sleep(1.5)
    
    print(f"    ❌ {ai_name.upper()} 所有线路+模型已压榨至死！")
    return ai_name, {}, "failed"

async def run_ai_matrix(prompt, num_matches):
    # ==========================================================
    # 💎 核心大模型配置矩阵 (你截图里重点关注的地方)
    # ==========================================================
    ai_configs = [
        ("claude", "CLAUDE_API_URL", "CLAUDE_API_KEY", [
            "熊猫-按量-满血copilot-claude-opus-4.6-thinking",
            "熊猫-按量-特供顶级-官方正向满血-claude-opus-4.6-thinking",
        ]),
        ("grok", "GROK_API_URL", "GROK_API_KEY", [
            "熊猫-A-7-grok-4.2-多智能体讨论",
            "熊猫-A-6-grok-4.2-thinking",
        ]),
        ("gpt", "GPT_API_URL", "GPT_API_KEY", [
            "熊猫-A-7-gpt-5.4",
            "熊猫-按量-gpt-5.3-codex-满血",
            "熊猫-A-10-gpt-5.3-codex",
        ]),
        ("gemini", "GEMINI_API_URL", "GEMINI_API_KEY", [
            "熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking",
            "熊猫-顶级特供-X-17-gemini-3.1-pro-preview",
        ]),
    ]
    
    all_results = {"gpt": {}, "grok": {}, "claude": {}, "gemini": {}}
    tasks = []
    
    async with aiohttp.ClientSession() as session:
        for ai_name, url_env, key_env, models in ai_configs:
            tasks.append(async_call_one_ai_batch(session, prompt, url_env, key_env, models, num_matches, ai_name))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for res in results:
        if isinstance(res, tuple):
            ai_name, parsed_data, model_used = res
            all_results[ai_name] = parsed_data
        else:
            print(f"  [CRITICAL] 某AI任务完全崩溃: {res}")
    
    return all_results

# ====================================================================
# Merge 智能融合 v2.0（AI权重+confidence加成）
# ====================================================================
def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    sp_h = float(match_obj.get("sp_home", 0) or 0)
    sp_d = float(match_obj.get("sp_draw", 0) or 0)
    sp_a = float(match_obj.get("sp_away", 0) or 0)
    
    engine_score = engine_result.get("primary_score", "1-1")
    engine_conf = engine_result.get("confidence", 50)
    
    ai_all = {"gpt": gpt_r, "grok": grok_r, "gemini": gemini_r, "claude": claude_r}
    ai_scores = []
    ai_conf_sum = 0
    ai_conf_count = 0
    value_kills = 0
    
    weights = {"claude": 1.4, "grok": 1.3, "gpt": 1.1, "gemini": 1.0}
    
    for name, r in ai_all.items():
        if not isinstance(r, dict):
            continue
        sc = r.get("ai_score", "-")
        if sc and sc not in ["-", "?", ""]:
            ai_scores.append(sc)
            conf = r.get("ai_confidence", 60)
            ai_conf_sum += conf * weights.get(name, 1.0)
            ai_conf_count += weights.get(name, 1.0)
            if r.get("value_kill"):
                value_kills += 1
    
    vote_count = {}
    for sc in ai_scores:
        vote_count[sc] = vote_count.get(sc, 0) + 1
    
    final_score = engine_score
    if vote_count:
        best_voted = max(vote_count, key=vote_count.get)
        if best_voted in engine_result.get("top3_scores", []) and vote_count[best_voted] >= 2:
            final_score = best_voted
    
    avg_ai_conf = (ai_conf_sum / ai_conf_count) if ai_conf_count > 0 else 60
    cf = engine_conf
    cf = min(95, cf + int((avg_ai_conf - 60) * 0.4))
    cf = cf + value_kills * 6
    
    has_warn = any("🚨" in str(s) for s in stats.get("smart_signals", []))
    if has_warn:
        cf = max(35, cf - 12)
    
    risk = "低" if cf >= 75 else ("中" if cf >= 55 else "高")
    
    hp = engine_result.get("home_prob", 33)
    dp = engine_result.get("draw_prob", 33)
    ap = engine_result.get("away_prob", 34)
    shp = stats.get("home_win_pct", 33)
    sdp = stats.get("draw_pct", 33)
    sap = stats.get("away_win_pct", 34)
    fhp = hp * 0.75 + shp * 0.25
    fdp = dp * 0.75 + sdp * 0.25
    fap = ap * 0.75 + sap * 0.25
    fhp = max(3, fhp); fdp = max(3, fdp); fap = max(3, fap)
    ft = fhp + fdp + fap
    if ft > 0:
        fhp = round(fhp / ft * 100, 1)
        fdp = round(fdp / ft * 100, 1)
        fap = round(max(3, 100 - fhp - fdp), 1)
    
    gpt_sc = gpt_r.get("ai_score", "-") if isinstance(gpt_r, dict) else "-"
    gpt_an = gpt_r.get("analysis", "N/A") if isinstance(gpt_r, dict) else "N/A"
    grok_sc = grok_r.get("ai_score", "-") if isinstance(grok_r, dict) else "-"
    grok_an = grok_r.get("analysis", "N/A") if isinstance(grok_r, dict) else "N/A"
    gem_sc = gemini_r.get("ai_score", "-") if isinstance(gemini_r, dict) else "-"
    gem_an = gemini_r.get("analysis", "N/A") if isinstance(gemini_r, dict) else "N/A"
    cl_sc = claude_r.get("ai_score", "-") if isinstance(claude_r, dict) else engine_score
    cl_an = claude_r.get("analysis", "N/A") if isinstance(claude_r, dict) else engine_result.get("reason", "odds engine")
    
    return {
        "predicted_score": final_score,
        "home_win_pct": fhp, "draw_pct": fdp, "away_win_pct": fap,
        "confidence": cf, "risk_level": risk,
        "over_under_2_5": "大" if engine_result.get("over_25", 50) > 55 else "小",
        "both_score": "是" if engine_result.get("btts", 45) > 50 else "否",
        "gpt_score": gpt_sc, "gpt_analysis": gpt_an,
        "grok_score": grok_sc, "grok_analysis": grok_an,
        "gemini_score": gem_sc, "gemini_analysis": gem_an,
        "claude_score": cl_sc, "claude_analysis": cl_an,
        "ai_avg_confidence": round(avg_ai_conf, 1),
        "value_kill_count": value_kills,
        "model_agreement": len(set(ai_scores)) <= 1 and len(ai_scores) >= 2,
        "poisson": stats.get("poisson", {}),
        "refined_poisson": stats.get("refined_poisson", {}),
        "extreme_warning": engine_result.get("scissors_gap_signal", ""),
        "smart_money_signal": " | ".join(stats.get("smart_signals", [])),
        "smart_signals": stats.get("smart_signals", []),
        "model_consensus": stats.get("model_consensus", 0),
        "total_models": stats.get("total_models", 11),
        "expected_total_goals": engine_result.get("expected_goals", 2.5),
        "over_2_5": engine_result.get("over_25", 50),
        "btts": engine_result.get("btts", 45),
        "top_scores": stats.get("refined_poisson", {}).get("top_scores", []),
        "elo": stats.get("elo", {}), 
        "random_forest": stats.get("random_forest", {}),
        "gradient_boost": stats.get("gradient_boost", {}), 
        "neural_net": stats.get("neural_net", {}),
        "logistic": stats.get("logistic", {}), 
        "svm": stats.get("svm", {}), 
        "knn": stats.get("knn", {}),
        "dixon_coles": stats.get("dixon_coles", {}), 
        "bradley_terry": stats.get("bradley_terry", {}),
        "home_form": stats.get("home_form", {}), 
        "away_form": stats.get("away_form", {}),
        "handicap_signal": stats.get("handicap_signal", ""),
        "odds_movement": stats.get("odds_movement", {}), 
        "vote_analysis": stats.get("vote_analysis", {}),
        "h2h_blood": stats.get("h2h_blood", {}), 
        "crs_analysis": stats.get("crs_analysis", {}),
        "ttg_analysis": stats.get("ttg_analysis", {}), 
        "halftime": stats.get("halftime", {}),
        "pace_rating": stats.get("pace_rating", ""),
        "kelly_home": stats.get("kelly_home", {}), 
        "kelly_away": stats.get("kelly_away", {}),
        "odds": stats.get("odds", {}),
        "experience_analysis": stats.get("experience_analysis", {}),
        "pro_odds": stats.get("pro_odds", {}),
        "bivariate_poisson": stats.get("bivariate_poisson", {}),
        "asian_handicap_probs": stats.get("asian_handicap_probs", {}),
        "bookmaker_implied_home_xg": engine_result.get("bookmaker_implied_home_xg", "?"),
        "bookmaker_implied_away_xg": engine_result.get("bookmaker_implied_away_xg", "?")
    }

def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})
        s = pr.get("confidence", 0) * 0.4
        mx = max(pr.get("home_win_pct", 33), pr.get("away_win_pct", 33), pr.get("draw_pct", 33))
        s += (mx - 33) * 0.2 + pr.get("model_consensus", 0) * 2
        
        if pr.get("risk_level") == "低": s += 12
        elif pr.get("risk_level") == "高": s -= 5
        if pr.get("model_agreement"): s += 10

        exp_info = pr.get("experience_analysis", {})
        exp_score = exp_info.get("total_score", 0)
        exp_draw_rules = exp_info.get("draw_rules", 0)
        
        if exp_score >= 15 and pr.get("result") == "平局" and exp_draw_rules >= 3:
            s += 12
        elif exp_score >= 10:
            s += 5
            
        if exp_info.get("recommendation", "").startswith("⚠️"):
            s -= 3
            
        smart_money = str(pr.get("smart_money_signal", ""))
        direction = pr.get("result", "")
        if "Sharp" in smart_money:
            if ("客胜" in smart_money and direction == "主胜") or ("主胜" in smart_money and direction == "客胜"):
                s -= 30
                
        p["recommend_score"] = round(s, 2)
        
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]

def extract_num(ms):
    wm = {"一":1000,"二":2000,"三":3000,"四":4000,"五":5000,"六":6000,"日":7000,"天":7000}
    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999

# ====================================================================
# ☢️ run_predictions v2.1 —— 精确调用链 + try/except防崩溃
# ====================================================================
def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 80)
    print(f"  [QUANT ENGINE vMAX 2.1] 极致压榨AI模式启动 | {len(ms)} 场比赛")
    print("=" * 80)

    match_analyses = []
    for i, m in enumerate(ms):
        eng = predict_match(m)
        league_info, _, _, _ = build_league_intelligence(m)
        sp = ensemble.predict(m, {})
        exp_result = exp_engine.analyze(m)
        match_analyses.append({
            "match": m, "engine": eng, "league_info": league_info,
            "stats": sp, "index": i + 1, "experience": exp_result,
        })

    all_ai = {"claude": {}, "gemini": {}, "gpt": {}, "grok": {}}
    if use_ai and match_analyses:
        prompt = build_batch_prompt(match_analyses)
        print(f"  [PROMPT] 已生成 {len(prompt):,} 字符的极致毒prompt，开始压榨AI矩阵...")
        start_t = time.time()
        all_ai = asyncio.run(run_ai_matrix(prompt, len(match_analyses)))
        print(f"  [AI MATRIX] 压榨完成，耗时 {time.time()-start_t:.1f}s")

    res = []
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        
        mg = merge_result(
            ma["engine"],
            all_ai["gpt"].get(i+1, {}),
            all_ai["grok"].get(i+1, {}),
            all_ai["gemini"].get(i+1, {}),
            all_ai["claude"].get(i+1, {}),
            ma["stats"], m
        )
        
        # ============ 4层增强管线（每层独立try/except防崩溃） ============
        try:
            mg = apply_experience_to_prediction(m, mg, exp_engine)
            print(f"    → apply_experience_to_prediction 已注入（经验法则加成）")
        except Exception as e:
            print(f"    ⚠️ experience_rules跳过: {e}")
        
        try:
            mg = apply_odds_history(m, mg)
            print(f"    → apply_odds_history 已尝试注入（历史盘口血洗信号）")
        except Exception as e:
            print(f"    ⚠️ odds_history跳过: {e}")
        
        try:
            mg = apply_quant_edge(m, mg)
            print(f"    → apply_quant_edge 已尝试注入（极致量化边缘屠杀）")
        except Exception as e:
            print(f"    ⚠️ quant_edge跳过: {e}")
        
        try:
            mg = upgrade_ensemble_predict(m, mg)
            print(f"    → upgrade_ensemble_predict 已注入（最终集成强化）")
        except Exception as e:
            print(f"    ⚠️ advanced_models跳过: {e}")
        # =====================================================================
        
        pcts = {"主胜": mg["home_win_pct"], "平局": mg["draw_pct"], "客胜": mg["away_win_pct"]}
        mg["result"] = max(pcts, key=pcts.get)

        res.append({**m, "prediction": mg})
        print(f"  [{i+1}] {m.get('home_team')} vs {m.get('away_team')} => {mg['result']} ({mg['predicted_score']}) | CF: {mg['confidence']}% | AI信心: {mg.get('ai_avg_confidence', 0)}")

    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res:
        r["is_recommended"] = r["id"] in t4ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    diary = load_ai_diary()
    diary["yesterday_win_rate"] = f"{len([r for r in res if r['prediction']['confidence'] > 70])}/{max(1, len(res))}"
    diary["reflection"] = "今天AI矩阵已彻底入魔 + 新增历史匹配+量化边缘，屠杀信号更精准，下次继续加毒"
    save_ai_diary(diary)

    return res, t4