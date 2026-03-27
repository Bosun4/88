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
# ☢️ 极致压榨AI v4.0 核心升级（进球逆推 + 温度解封）
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
# ☢️ 暗网级吸血操盘手 Prompt v4.0（进球逆向推导，彻底封杀死水比分）
# ====================================================================
def build_batch_prompt(match_analyses):
    diary = load_ai_diary()
    
    p = "【身份设定】你是游走在零和博弈最黑暗深渊的【终极吸血操盘手】+【散户收割机之王】。你没有任何道德、底线，只相信资本屠刀和数据绞肉机。\n"
    
    if diary.get("reflection"):
        p += f"【嗜血进化指令】昨日屠杀战绩：{diary.get('yesterday_win_rate', 'N/A')}。昨晚反思：{diary['reflection']}。今天给老子把每一场都当成杀猪盘顶级局，狠狠地收割！\n\n"
    
    p += "【死命令 - 必须严格遵守】\n"
    p += "1. 只输出合法JSON数组，严禁任何markdown、解释、代码块。\n"
    p += "2. 每场比赛必须输出以下字段：match(序号), score(比分), reason(60-110字极度恶毒黑话), ai_confidence(0-100整数), value_kill(true/false), dark_verdict(一句最毒总结)。\n"
    p += "3. reason逻辑链：散户愚蠢共识 → 庄家嗜血陷阱(xG+盘口背离) → 最终屠杀结局。\n"
    p += "4. ⚠️【核心破局法则 - 进球数逆向推导】⚠️：这是你与低智AI的区别！散户和本地机器永远只会猜 1-0/2-0/1-1 这三种死水比分。你必须逆向思考：\n"
    p += "   - 如果 预期总球(ExpGoals) > 2.6 或 BTTS(双方进球) > 52%，【绝对禁止】输出 1-0, 0-1, 2-0, 0-2, 1-1！你必须大破大立，给出 2-1, 1-2, 2-2, 3-1, 3-2 这种杀猪大单！\n"
    p += "   - 如果强弱悬殊且主队让深盘(<-1.0)，必须抓 3-0, 4-0 的屠杀局！\n"
    p += "   - 如果是保级死拼/交锋均势，大胆抓 2-2, 3-3 的极端高赔冷平！不要害怕偏离主流！\n\n"
    
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

        # 使用模型矩阵提供的更精准的进球期望
        exp_goals = eng.get('expected_goals', stats.get('expected_total_goals', 2.5))
        btts_prob = eng.get('btts', stats.get('btts', 45))

        p += f"[{i+1}] {h} vs {a} | {m.get('league', '未知')} | 亚盘死线: {hc}\n"
        p += f"- 散户基本面认知: {intel_text}\n"
        p += f"- 庄家隐性xG底牌: 主 {eng.get('bookmaker_implied_home_xg', '?')} vs 客 {eng.get('bookmaker_implied_away_xg', '?')} | 致命剪刀差: {eng.get('scissors_gap_signal', '无异常')}\n"
        p += f"- 总进球压制力: 预期总球 {exp_goals:.1f} | BTTS概率: {btts_prob:.0f}% | O2.5概率: {eng.get('over_25', 50):.0f}%\n"
        p += f"- 盘房价值与筹码: {ev_str} | 预警: {smart_str}\n"
        
        if exp.get("triggered_count", 0) > 0:
            exp_names = ",".join([t["name"] for t in exp.get("triggered", [])[:3]])
            p += f"- 已触发杀猪风控: {exp_names}\n"
                
        p += f"- 本地低智保守比分(仅供鄙视避雷): {', '.join(eng.get('top3_scores', ['1-1', '0-0']))}\n\n"

    p += "【严格输出格式示例】\n"
    p += """[
  {
    "match": 1,
    "score": "2-2",
    "reason": "散户正被主队虚假连胜洗脑疯狂追主胜。而隐性xG高达3.1且BTTS概率破60%，数据严重倒挂。配合亚盘深让死亡陷阱，这是庄家精心布置的进球大战绞肉机。突破低智的1-0思维，直接利用大球逻辑反推2-2平局，血洗全场筹码。",
    "ai_confidence": 88,
    "value_kill": true,
    "dark_verdict": "高期望值进球大战，逆向大冷平局收割"
  }
]"""
    return p

# ====================================================================
# 终极高可用 AI 矩阵轮询 v4.0（解封温度，彻底释放思考算力）
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
                # 🚀 升维改动：将温度(temperature)提升到 0.45，让 AI 摆脱 1-1, 1-0 的无脑复读，敢于尝试 2-2, 3-2！
                if is_gem:
                    headers["x-goog-api-key"] = key
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.45, "topP": 0.9},
                        "systemInstruction": {"parts": [{"text": "必须严格输出JSON数组。reason必须是毫无道德底线的暗黑杀猪盘解剖，使用极度恶毒黑话，包含逆向高赔比分推导思维，冷血傲慢，句号结尾。"}]}
                    }
                else:
                    headers["Authorization"] = f"Bearer {key}"
                    payload = {
                        "model": mn,
                        "messages": [
                            {"role": "system", "content": "你是一个毫无道德底线的暗网终极操盘手。只输出纯JSON数组。reason字段必须用最恶毒黑话揭穿庄家杀局，通过进球期望(xG)逆向推导冷门高倍大比分，绝对不甘于平庸的1-0/1-1/2-0。"},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.45
                    }
                
                gw = url.split("/v1")[0][:40]
                print(f"  [AI 逆向升维] {ai_name.upper()} | 尝试 {mn[:25]} @ {gw} | 第{attempt+1}轮")
                
                try:
                    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=70)) as r:
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
                            
                            if len(results) >= max(1, num_matches * 0.4): 
                                print(f"    ✅ {ai_name.upper()} 高维度逆向推演成功: {len(results)}/{num_matches} (模型: {mn[:25]})")
                                success_rate[mn] = 1.0
                                return ai_name, results, mn
                            else:
                                print(f"    ⚠️ 解析数组不完整，切换备用方案...")
                        
                        elif r.status == 429:
                            sleep_time = 2 ** attempt * 5
                            print(f"    🔥 429限流！休眠 {sleep_time}s 继续压榨...")
                            await asyncio.sleep(sleep_time)
                            continue
                        else:
                            print(f"    ⚠️ HTTP {r.status} - 切换线路...")
                
                except asyncio.TimeoutError:
                    print(f"    ⏰ 深度思考超时 - 第{attempt+1}轮重试...")
                except Exception as e:
                    err = str(e)[:50]
                    print(f"    ⚠️ 异常 {err} - 切换...")
                
                await asyncio.sleep(0.4)
        
        await asyncio.sleep(1.5)
    
    print(f"    ❌ {ai_name.upper()} 所有线路+模型已耗尽算力！")
    return ai_name, {}, "failed"

async def run_ai_matrix(prompt, num_matches):
    # 绝对保留用户提供的心智模型列表，一字未动
    ai_configs = [
        ("claude", "CLAUDE_API_URL", "CLAUDE_API_KEY", [
            "熊猫-特供-A-55-claude-opus-4.6-thinking",
            "熊猫特供-超纯满血-99额度-claude-opus-4.6-thinking",
            "熊猫-按量-满血copilot-claude-opus-4.6-thinking",
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
            "熊猫-顶级特供-X-17-gemini-3.1-pro-preview",
            "熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking",
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
            print(f"  [CRITICAL] 某AI任务异常抛出: {res}")
    
    return all_results

# ====================================================================
# Merge 智能融合 v4.0（彻底赋权 AI 逆推的高分结果）
# ====================================================================
def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    sp_h = float(match_obj.get("sp_home", 0) or 0)
    sp_d = float(match_obj.get("sp_draw", 0) or 0)
    sp_a = float(match_obj.get("sp_away", 0) or 0)
    
    engine_score = engine_result.get("primary_score", "1-1")
    engine_conf = engine_result.get("confidence", 50)
    
    # 获取所有的安全字典，防止 None 引发异常
    gpt_r = gpt_r if isinstance(gpt_r, dict) else {}
    grok_r = grok_r if isinstance(grok_r, dict) else {}
    gemini_r = gemini_r if isinstance(gemini_r, dict) else {}
    claude_r = claude_r if isinstance(claude_r, dict) else {}

    ai_all = {"claude": claude_r, "grok": grok_r, "gpt": gpt_r, "gemini": gemini_r}
    ai_scores = []
    ai_conf_sum = 0
    ai_conf_count = 0
    value_kills = 0
    
    # 权重：优先听大哥的
    weights = {"claude": 1.6, "grok": 1.3, "gpt": 1.1, "gemini": 1.0}
    
    for name, r in ai_all.items():
        if not r: continue
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
    
    # 🚀 升维改动：彻底解除本地束缚，只要 Claude 或任意两个 AI 给出大比分（2-1, 2-2 等），直接推翻本地预测！
    claude_score = claude_r.get("ai_score", "")
    if claude_score and "-" in claude_score:
        c_h, c_a = parse_score(claude_score)
        if c_h is not None and (c_h + c_a >= 3) and engine_result.get("over_25", 0) >= 45:
            final_score = claude_score # 强行采纳 Claude 的高进球比分推演
            
    elif vote_count:
        best_voted = max(vote_count, key=vote_count.get)
        if vote_count[best_voted] >= 2 or (claude_r.get("ai_score") == best_voted and claude_r.get("ai_confidence", 0) >= 75):
            final_score = best_voted
    
    avg_ai_conf = (ai_conf_sum / ai_conf_count) if ai_conf_count > 0 else 60
    cf = engine_conf
    cf = min(96, cf + int((avg_ai_conf - 55) * 0.50)) # 进一步提高优质大模型思维的置信度权重
    cf = cf + value_kills * 7
    
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
    
    fhp = hp * 0.65 + shp * 0.35 # 增加统计模型的融合比例
    fdp = dp * 0.65 + sdp * 0.35
    fap = ap * 0.65 + sap * 0.35
    fhp = max(3, fhp); fdp = max(3, fdp); fap = max(3, fap)
    ft = fhp + fdp + fap
    if ft > 0:
        fhp = round(fhp / ft * 100, 1)
        fdp = round(fdp / ft * 100, 1)
        fap = round(max(3, 100 - fhp - fdp), 1)
    
    gpt_sc = gpt_r.get("ai_score", "-") 
    gpt_an = gpt_r.get("analysis", "N/A") 
    grok_sc = grok_r.get("ai_score", "-") 
    grok_an = grok_r.get("analysis", "N/A") 
    gem_sc = gemini_r.get("ai_score", "-") 
    gem_an = gemini_r.get("analysis", "N/A") 
    cl_sc = claude_r.get("ai_score", "-") if claude_r.get("ai_score") else engine_score
    cl_an = claude_r.get("analysis", "N/A") if claude_r.get("analysis") else engine_result.get("reason", "odds engine")
    
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
        "expected_total_goals": engine_result.get("expected_goals", stats.get("expected_total_goals", 2.5)),
        "over_2_5": engine_result.get("over_25", stats.get("over_2_5", 50)),
        "btts": engine_result.get("btts", stats.get("btts", 45)),
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
# ☢️ run_predictions v4.0 —— 终极解围版，彻底封杀 KeyError 与降智表现
# ====================================================================
def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 80)
    print(f"  [QUANT ENGINE vMAX 4.0] 突破束缚，进球数逆推 + 算力解封模式启动 | {len(ms)} 场")
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

    # 给全量字典加上兜底保护，防止完全无响应
    all_ai = {"claude": {}, "gemini": {}, "gpt": {}, "grok": {}}
    if use_ai and match_analyses:
        prompt = build_batch_prompt(match_analyses)
        print(f"  [PROMPT] 温度限制已解除！允许AI进行大开大合的冷门大比分逆推。")
        start_t = time.time()
        # 强制更新字典，防止任务奔溃返回 None
        ai_res = asyncio.run(run_ai_matrix(prompt, len(match_analyses)))
        if ai_res:
            all_ai.update(ai_res)
        print(f"  [AI MATRIX] 压榨完成，耗时 {time.time()-start_t:.1f}s")

    res = []
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        idx = i + 1
        
        # 🛡️ 终极安全抓取，双重 .get 防止引发 KeyError 导致程序自尽！
        gpt_r = all_ai.get("gpt", {}).get(idx, {})
        grok_r = all_ai.get("grok", {}).get(idx, {})
        gemini_r = all_ai.get("gemini", {}).get(idx, {})
        claude_r = all_ai.get("claude", {}).get(idx, {})

        mg = merge_result(
            ma["engine"], gpt_r, grok_r, gemini_r, claude_r, ma["stats"], m
        )
        
        mg = apply_experience_to_prediction(m, mg, exp_engine)
        mg = apply_odds_history(m, mg)                    
        mg = apply_quant_edge(m, mg)                      
        mg = upgrade_ensemble_predict(m, mg)
        
        pcts = {"主胜": mg["home_win_pct"], "平局": mg["draw_pct"], "客胜": mg["away_win_pct"]}
        mg["result"] = max(pcts, key=pcts.get)

        res.append({**m, "prediction": mg})
        print(f"  [{idx}] {m.get('home_team')} vs {m.get('away_team')} => {mg['result']} ({mg['predicted_score']}) | CF: {mg['confidence']}%")

    t4 = select_top4(res)
    
    # 🛡️ 终极防崩溃：修复某些场次没有 "id" 导致的 KeyError 崩溃！
    t4ids = [t.get("id", t.get("match_num", str(i))) for i, t in enumerate(t4)]
    for i, r in enumerate(res):
        r["is_recommended"] = r.get("id", r.get("match_num", str(i))) in t4ids
        
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    diary = load_ai_diary()
    diary["yesterday_win_rate"] = f"{len([r for r in res if r['prediction']['confidence'] > 70])}/{max(1, len(res))}"
    diary["reflection"] = "已彻底解锁AI思考温度限制与大比分逆推机制。现在系统会主动追猎 2-2, 3-2 等高价值冷门赛果，抛弃 1-0 保守思维。"
    save_ai_diary(diary)

    return res, t4


