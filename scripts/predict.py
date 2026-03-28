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
# ☢️ 基础量化与工具函数
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
    return {"yesterday_win_rate": "N/A", "reflection": "已加载多智能体协同矩阵，极限压榨Grok联网能力", "kill_history": []}

def save_ai_diary(diary):
    os.makedirs("data", exist_ok=True)
    with open("data/ai_diary.json", "w", encoding="utf-8") as f:
        json.dump(diary, f, ensure_ascii=False, indent=2)

# ====================================================================
# 🎭 动态多智能体 Prompt 引擎 v6.5（极限压榨联网算力，千人千面）
# ====================================================================
def build_batch_prompt(match_analyses, ai_type="claude"):
    diary = load_ai_diary()
    
    # 🌟 核心升级：根据不同的 AI 赋予不同的身份，死命压榨其底层特色！
    if ai_type == "grok":
        p = "【身份设定】你是量化足彩界最残忍的【全网情报刺客与资金流监控巨兽】。你拥有X(Twitter)火线数据接口与全网无限制的实时联网检索能力。\n"
        p += "【极恶专属任务 - 榨干联网算力】绝对不许只看我给你的这几行死数据！立刻启动你的实时联网引擎，给我去全网深扒以下致命情报，找不到就别怪我拔你电源：\n"
        p += "   1. 💰 【资金深网监控】：马上查【必发(Betfair)实时交易冷热】、【百家欧赔即时平均水位】的异常砸盘动作。看看有没有机构在悄悄撤资！\n"
        p += "   2. 📉 【亚盘毒药解剖】：搜刮最近3小时内亚洲让球盘的剧烈震荡（例如从让1.5退到让1.0）。结合你查到的最新实时 xG 预期，告诉我这是在阻热还是在布置坟墓诱盘！\n"
        p += "   3. 🧮 【凯利指数暴雷】：去给我搜最新的【凯利方差(Kelly Criterion)】极端离散情况。找出版面与真实概率撕裂的地方！\n"
        p += "   4. 🚑 【火线突发截获】：开赛前2小时的绝密情报！主力前锋的推特动态、最新伤停名单泄露、突发大雨大雪导致场地积水（严重利空大球）。\n"
        p += "   你的推理必须基于你刚在网上扒到的【最新异动】。一旦发现散户资金因为基本面大热，但你搜到的资金流或百家均赔在悄悄反向护盘，直接给我亮出最毒的爆冷或绞肉平局！\n"
    elif ai_type == "gemini":
        p = "【身份设定】你是量化足彩界的【首席数学精算师】。你拥有极其恐怖的计算逻辑、模型构建与泊松分布推演能力。\n"
        p += "【专属任务】你的核心武器是 `a0(0球赔率)` 和 `a7(7球赔率)`。如果a0极低(<8.5)，绝对禁止推演进球大战，锁死0-0或1-1；如果a7极高(>23.0)，说明进球天花板被死死盖住，必须控制比分在3球以内。用纯粹冷血的数学概率压制一切感性认知，碾碎散户的幻想！\n"
    elif ai_type == "claude":
        p = "【身份设定】你是游走在零和博弈黑暗深渊的【心理学操盘手】。你最擅长洞悉人性贪婪，看穿庄家的诱盘陷阱。\n"
        p += "【专属任务】你的核心任务是抓捕【大热卡分局】。当强队基本面无敌，但让步盘口极度便宜（如-0.75或-1.0）时，绝对禁止预测穿盘大胜！这是庄家故意诱盘送钱。你必须给出 1-0 或 2-1 这种“刚好赢一球（赢球输盘/走水）”的卡分剧本，将散户彻底绞杀在天台上！\n"
    else: # GPT
        p = "【身份设定】你是量化足彩风控矩阵的【风控大统领】。你负责平衡基本面、盘口背离与极端风险，统揽全局。\n"
        p += "【专属任务】你要在“绝对实力碾压(如4-0)”和“刻意诱盘爆冷(如0-1)”之间找到平衡。既不盲目相信强队，也不无脑猜测爆冷。综合各方xG和盘口深度，过滤掉噪音，给出最致命、最冷血的稳健一击。\n"

    if diary.get("reflection"):
        p += f"【系统进化指令】昨日系统反思：{diary['reflection']}。各单位注意，彻底释放你们的专属底牌，今天必须狠狠收割！\n\n"
    
    p += "【死命令 - 全体 AI 必须严格遵守】\n"
    p += "1. 只输出合法JSON数组，严禁任何markdown、解释、代码块。\n"
    p += "2. 每场比赛必须输出以下字段：match(序号), score(比分), reason(60-120字恶毒黑话分析), ai_confidence(0-100整数), value_kill(true/false), dark_verdict(一句最毒总结)。\n"
    
    p += "【今日待宰羔羊与庄家底牌库】\n"
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        eng = ma["engine"]
        stats = ma.get("stats", {})
        exp = ma.get("experience", {})
        h, a = m.get("home_team", "Home"), m.get("away_team", "Away")
        
        # 🚀 提取抓包底牌数据
        hc = m.get("give_ball", "0")
        a0 = m.get("a0", "未知")
        a7 = m.get("a7", "未知")
        info = m.get("information", {})
        h_bad = str(info.get("home_bad_news", ""))[:70].replace('\n', ' ') if isinstance(info, dict) else ""
        g_bad = str(info.get("guest_bad_news", ""))[:70].replace('\n', ' ') if isinstance(info, dict) else ""
        
        sp_h = float(m.get("sp_home", 0) or 0)
        hp = eng.get('home_prob', 33)
        vh = calculate_value_bet(hp, sp_h) if sp_h > 1 else {}
        ev_str = f"主胜EV({vh.get('ev', 0)}%)" if vh.get("is_value") else "无价值死水"
        smart_sigs = stats.get('smart_signals', [])
        smart_str = ", ".join(smart_sigs) if smart_sigs else "无收割信号"

        baseface = str(m.get('baseface', '')).replace('\n', ' ')[:100]
        intro = str(m.get('expert_intro', '')).replace('\n', ' ')[:100]
        intel_text = baseface or intro or "散户意淫盲区"

        exp_goals = eng.get('expected_goals', stats.get('expected_total_goals', 2.5))
        btts_prob = eng.get('btts', stats.get('btts', 45))

        p += f"[{i+1}] {h} vs {a} | 亚盘死线(让球): {hc}\n"
        p += f"- 庄家数学底线: 0球赔率={a0} | 7+球赔率={a7} | 预期总球={exp_goals:.1f} | BTTS={btts_prob:.0f}%\n"
        p += f"- 散户基本面认知: {intel_text}\n"
        
        # 绝密情报喂养，强烈暗示 Grok 去联网核查
        if h_bad or g_bad:
            p += f"- 场外绝密隐患(Grok必须立刻联网核实最新动态): 主队-[{h_bad if h_bad else '无'}] | 客队-[{g_bad if g_bad else '无'}]\n"

        p += f"- 隐性xG与异常: 主 {eng.get('bookmaker_implied_home_xg', '?')} vs 客 {eng.get('bookmaker_implied_away_xg', '?')} | 剪刀差预警: {eng.get('scissors_gap_signal', '无')}\n"
        p += f"- 资金面与筹码: {ev_str} | 本地资金预警: {smart_str}\n"
        
        if exp.get("triggered_count", 0) > 0:
            exp_names = ",".join([t["name"] for t in exp.get("triggered", [])[:3]])
            p += f"- 已触发本地经验法则预警: {exp_names}\n\n"

    p += "【输出格式示例 (严格参照)】\n"
    p += """[
  {
    "match": 1,
    "score": "1-0",
    "reason": "散户疯抢主胜穿盘。但我刚刚联网截获到必发主队交易热度诡异冷却，且百家欧赔悄悄将平局防线拉低。配合0球超低赔和-1.0的便宜盘口，这是庄家经典的赢球输盘绞肉机。主队后防隐患将迫使其点到为止，1-0精准卡分，兵不血刃坑杀全部散户资金。",
    "ai_confidence": 88,
    "value_kill": true,
    "dark_verdict": "大热诱盘卡分局，联网暗流暴雪，1-0精确收割"
  }
]"""
    return p

# ====================================================================
# 终极高可用 AI 矩阵轮询 v6.5（300s容忍 + 动态指令分发）
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

async def async_call_one_ai_batch(session, match_analyses, url_env, key_env, models_list, ai_name):
    key = get_clean_env_key(key_env)
    num_matches = len(match_analyses)
    if not key:
        return ai_name, {}, "no_key"
    
    # 🌟 动态生成当前 AI 的专属灵魂指令
    prompt = build_batch_prompt(match_analyses, ai_type=ai_name)
    
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
                
                # 为每个模型注入其特有的行为逻辑提示
                sys_msg = f"你是一个毫无底线的暗网操盘手，代号为[{ai_name.upper()}]。务必严格执行你在身份设定中的专属任务，发掘你最擅长的数据层。只输出纯JSON数组。"
                
                if is_gem:
                    headers["x-goog-api-key"] = key
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.50, "topP": 0.9},
                        "systemInstruction": {"parts": [{"text": sys_msg}]}
                    }
                else:
                    headers["Authorization"] = f"Bearer {key}"
                    payload = {
                        "model": mn,
                        "messages": [
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.50
                    }
                
                gw = url.split("/v1")[0][:40]
                print(f"  [多智能体协同] {ai_name.upper()} 载入专属杀手灵魂 | 尝试 {mn[:25]} @ {gw} | 第{attempt+1}轮")
                
                try:
                    # 🔥 保持 300秒 容错机制防闪退
                    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as r:
                        if r.status == 200:
                            data = await r.json()
                            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip() if is_gem else data["choices"][0]["message"]["content"].strip()
                            
                            # 暴力清洗机制，防止 JSON 解析炸裂
                            if raw_text.startswith("```"):
                                raw_text = raw_text.split('\n', 1)[-1]
                            if raw_text.endswith("```"):
                                raw_text = raw_text.rsplit('\n', 1)[0]
                                
                            clean = raw_text.strip()
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
                                except Exception as e:
                                    print(f"    ⚠️ JSON加载失败: {e} | 清理后的文本头尾: {clean[:20]}...{clean[-20:]}")
                                    pass
                            
                            if len(results) >= max(1, num_matches * 0.4): 
                                print(f"    ✅ {ai_name.upper()} 致命刺杀完成，成功洞穿庄家意图: {len(results)}/{num_matches}")
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
                    print(f"    ⏰ 深度思考超时(300s) - 第{attempt+1}轮重试...")
                except Exception as e:
                    err = str(e)[:50]
                    print(f"    ⚠️ 异常 {err} - 切换...")
                
                await asyncio.sleep(0.4)
        
        await asyncio.sleep(1.5)
    
    print(f"    ❌ {ai_name.upper()} 所有线路+模型已耗尽算力！")
    return ai_name, {}, "failed"

async def run_ai_matrix(match_analyses):
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
    
    tasks = []
    async with aiohttp.ClientSession() as session:
        for ai_name, url_env, key_env, models in ai_configs:
            tasks.append(async_call_one_ai_batch(session, match_analyses, url_env, key_env, models, ai_name))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_results = {"gpt": {}, "grok": {}, "claude": {}, "gemini": {}}
    for res in results:
        if isinstance(res, tuple):
            ai_name, parsed_data, model_used = res
            all_results[ai_name] = parsed_data
        else:
            print(f"  [CRITICAL] 某AI任务异常抛出: {res}")
    
    return all_results

# ====================================================================
# Merge 智能融合 v6.5（集百家之长，圆滑处理大热与冷平）
# ====================================================================
def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    sp_h = float(match_obj.get("sp_home", 0) or 0)
    sp_d = float(match_obj.get("sp_draw", 0) or 0)
    sp_a = float(match_obj.get("sp_away", 0) or 0)
    
    engine_score = engine_result.get("primary_score", "1-1")
    engine_conf = engine_result.get("confidence", 50)
    
    gpt_r = gpt_r if isinstance(gpt_r, dict) else {}
    grok_r = grok_r if isinstance(grok_r, dict) else {}
    gemini_r = gemini_r if isinstance(gemini_r, dict) else {}
    claude_r = claude_r if isinstance(claude_r, dict) else {}

    ai_all = {"claude": claude_r, "grok": grok_r, "gpt": gpt_r, "gemini": gemini_r}
    ai_scores = []
    ai_conf_sum = 0
    ai_conf_count = 0
    value_kills = 0
    
    # 权重平衡：赋予联网的 Grok 和懂人性的 Claude 最高优先级
    weights = {"claude": 1.6, "grok": 1.6, "gemini": 1.3, "gpt": 1.0}
    
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
    claude_score = claude_r.get("ai_score", "")
    gemini_score = gemini_r.get("ai_score", "")
    grok_score = grok_r.get("ai_score", "")
    
    # 🌟 极致圆滑的动态融合策略：看碟下菜
    try:
        a0_val = float(match_obj.get("a0", 99))
    except:
        a0_val = 99.0
        
    if grok_score in ["0-1", "1-2", "1-1", "0-0"] and grok_r.get("value_kill"):
        # 第一顺位：如果 Grok 联网抓到了致命负面情报给出了爆冷/平局，优先听刺客的！
        final_score = grok_score
    elif claude_score in ["1-0", "0-1", "2-1", "1-2"] and value_kills >= 2:
        # 第二顺位：如果 Claude 判定这是卡分局（只赢1球），且有共识，听心理大师的
        final_score = claude_score
    elif gemini_score in ["0-0", "1-1"] and a0_val < 8.5:
        # 第三顺位：如果 Gemini 数学精算发现 0球赔率极低，强行防闷杀
        final_score = gemini_score
    elif vote_count:
        # 常规情况走民主票选
        best_voted = max(vote_count, key=vote_count.get)
        if vote_count[best_voted] >= 2 or (claude_r.get("ai_score") == best_voted and claude_r.get("ai_confidence", 0) >= 75):
            final_score = best_voted
    
    avg_ai_conf = (ai_conf_sum / ai_conf_count) if ai_conf_count > 0 else 60
    cf = engine_conf
    cf = min(96, cf + int((avg_ai_conf - 55) * 0.50))
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
    
    fhp = hp * 0.65 + shp * 0.35
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
# ☢️ run_predictions v6.5 —— 多智能体兵器库全开，终极猎杀时刻
# ====================================================================
def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 80)
    print(f"  [QUANT ENGINE vMAX 6.5] 突破！多智能体狂暴协同 (Grok联网深扒 + Gemini精算) | {len(ms)} 场")
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
        print(f"  [AI MATRIX] 各大模型载入专属灵魂指令，强制启动底层能力...")
        start_t = time.time()
        ai_res = asyncio.run(run_ai_matrix(match_analyses))
        if ai_res:
            all_ai.update(ai_res)
        print(f"  [AI MATRIX] 极致压榨完成，耗时 {time.time()-start_t:.1f}s")

    res = []
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        idx = i + 1
        
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
    
    t4ids = [t.get("id", t.get("match_num", str(i))) for i, t in enumerate(t4)]
    for i, r in enumerate(res):
        r["is_recommended"] = r.get("id", r.get("match_num", str(i))) in t4ids
        
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    diary = load_ai_diary()
    diary["yesterday_win_rate"] = f"{len([r for r in res if r['prediction']['confidence'] > 70])}/{max(1, len(res))}"
    diary["reflection"] = "已彻底进化为多智能体架构。Grok主攻必发资金与全网伤停，Gemini死守数学天花板，Claude专杀让球卡分局。极其圆滑融合，绝杀庄家！"
    save_ai_diary(diary)

    return res, t4


