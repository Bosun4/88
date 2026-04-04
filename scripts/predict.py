```python
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
    def apply_odds_history(m, mg): return mg

try:
    from quant_edge import apply_quant_edge
except Exception as e:
    print(f"  [WARN] ⚠️ 量化边缘模块 (quant_edge) 加载失败或未找到，系统自动降级跳过: {e}")
    def apply_quant_edge(m, mg): return mg

try:
    from wencai_intel import apply_wencai_intel
except:
    def apply_wencai_intel(m, mg): return mg

ensemble = EnsemblePredictor()
exp_engine = ExperienceEngine()

# ====================================================================
# ☢️ 工具函数
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

# ====================================================================
# 🧊 冷门猎手引擎 (已剔除比分干预，仅负责提炼信号供警告)
# ====================================================================
class ColdDoorDetector:
    @staticmethod
    def detect(match, prediction):
        signals = []
        strength = 0
        steam = prediction.get("steam_move", {})
        if steam.get("steam") and "反向" in str(steam.get("signal", "")):
            signals.append("❄️ 反向Steam！庄家造热收割"); strength += 5
        vote = match.get("vote", {})
        try:
            vh = int(vote.get("win", 33)); va = int(vote.get("lose", 33))
            max_vote = max(vh, va)
            if max_vote >= 65: signals.append(f"❄️ 散户极端偏向{max_vote}%！冷门高危"); strength += 4
            elif max_vote >= 58: strength += 2
        except: pass
        info = match.get("intelligence", match.get("information", {}))
        if isinstance(info, dict):
            home_bad = str(info.get("home_bad_news", ""))
            away_bad = str(info.get("guest_bad_news", ""))
            hp = prediction.get("home_win_pct", 50)
            ap = prediction.get("away_win_pct", 50)
            if len(home_bad) > 80 and hp > 58: signals.append("❄️ 主队坏消息爆炸+散户狂热"); strength += 5
            if len(away_bad) > 80 and ap > 58: signals.append("❄️ 客队坏消息爆炸+散户狂热"); strength += 5
        sp_h = float(match.get("sp_home", match.get("win", 0)) or 0)
        if sp_h > 1:
            hp2 = prediction.get("home_win_pct", 50)
            implied_h = 100 / sp_h * 0.92
            if abs(implied_h - hp2) > 15 and hp2 > 58: signals.append(f"❄️ 赔率vs模型背离{abs(implied_h-hp2):.0f}%"); strength += 4
        for s in prediction.get("smart_signals", []):
            if "盘口太便宜" in str(s): signals.append("❄️ 盘口太便宜=庄家不看好"); strength += 3; break
        line = prediction.get("line_movement_anomaly", {})
        if line.get("has_anomaly") and "造热" in str(line.get("signal", "")): signals.append("❄️ 赔率变动造热=诱盘"); strength += 4
        is_cold = strength >= 7
        level = "顶级" if strength >= 12 else "高危" if strength >= 7 else "普通"
        return {"is_cold_door": is_cold, "strength": strength, "level": level, "signals": signals,
                "dark_verdict": f"❄️ {level}冷门！{len(signals)}条触发" if is_cold else ""}

# ====================================================================
# AI日记
# ====================================================================
def load_ai_diary():
    diary_file = "data/ai_diary.json"
    if os.path.exists(diary_file):
        try:
            with open(diary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return {"yesterday_win_rate": "N/A", "reflection": "持续进化中", "kill_history": []}

def save_ai_diary(diary):
    os.makedirs("data", exist_ok=True)
    with open("data/ai_diary.json", "w", encoding="utf-8") as f:
        json.dump(diary, f, ensure_ascii=False, indent=2)

# ====================================================================
# 🧠 极致压榨AI — 纯净数据环境 (切除所有预设比分引导)
# ====================================================================
def build_batch_prompt(match_analyses):
    diary = load_ai_diary()
    p = "【核心身份】你是全球顶尖量化足球预测引擎。你拥有绝对独立的判断力，不受任何人类主观意愿干扰。\n"
    p += "你面前是庄家真金白银的完整定价数据。你必须独立完成数学计算后，给出最具杀伤力的比分预测。\n\n"
    if diary.get("reflection"):
        p += f"【进化日记】{diary.get('yesterday_win_rate', 'N/A')} | {diary['reflection']}\n\n"
    p += "【独立推演铁律】\n"
    p += "1. 只输出合法JSON数组。字段：match(整数), score(如'2-1'或'4-0'等), reason(120-180字含具体推演数据), ai_confidence(0-100), value_kill(bool), dark_verdict(一句话总结)。\n"
    p += "2. ⚠️彻底抛弃所谓“必须从某几个比分里选”的枷锁！你必须独立推演：如果是绝对碾压的惨案就毫不犹豫给大比分；如果是极限防守的诱盘就给 0-0 闷杀。\n"
    p += "3. CRS赔率矩阵仅供你反推庄家意图，绝不是限制你的选项框架！\n"
    p += "4. reason必须包含硬核数据（如xG差距、泊松天花板、凯利异常）。语气极度专业、冷血、傲慢。\n\n"
    
    p += "【底层量化计算指令（沙盒内运算，仅输出最终结果）】\n"
    p += "STEP1: Shin去水位→真实胜平负概率。\n"
    p += "STEP2: 观察总进球a0-a7分布，确定庄家设定的进球天花板。\n"
    p += "STEP3: 亚盘+欧赔交叉验证（方向不一致=诱盘陷阱）。\n"
    p += "STEP4: 结合火线伤停情报与联赛特性，得出最终独立比分。\n\n"
    
    p += "【原始数据库】\n"

    for i, ma in enumerate(match_analyses):
        m = ma["match"]; eng = ma["engine"]; stats = ma.get("stats", {}); exp = ma.get("experience", {})
        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        league = m.get("league", m.get("cup", ""))
        hc = m.get("give_ball", "0")
        sp_h = float(m.get("sp_home", m.get("win", 0)) or 0)
        sp_d = float(m.get("sp_draw", m.get("same", 0)) or 0)
        sp_a = float(m.get("sp_away", m.get("lose", 0)) or 0)

        p += f"{'='*70}\n[{i+1}] {h} vs {a} | {league}\n{'='*70}\n"

        upset_leagues = {"英冠":32,"英甲":30,"英乙":28,"法乙":28,"荷乙":27,"德乙":26,"意乙":25}
        for lg_key, upset_pct in upset_leagues.items():
            if lg_key in str(league):
                p += f"⚠️ {lg_key}高风险警告：冷门率高达{upset_pct}%！\n"
                break

        odds_range = round(max(sp_h,sp_d,sp_a) - min(sp_h,sp_d,sp_a), 2) if sp_h > 1 else 0
        entropy_tag = " (三项极接近=高度均势)" if 0 < odds_range < 0.8 else ""

        p += f"欧赔三项: {sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f}{entropy_tag} | 亚盘死线: {hc}\n"

        if m.get("hhad_win"):
            p += f"让球欧赔: {m.get('hhad_win')}/{m.get('hhad_same')}/{m.get('hhad_lose')}\n"

        single = m.get("single", 0)
        if single == 1:
            p += f"📌 资金信号: 单关开放(庄家对结果把控度极高)\n"

        h_pos = m.get("home_position", ""); g_pos = m.get("guest_position", "")
        if h_pos or g_pos:
            p += f"排名落位: 主队{h_pos} vs 客队{g_pos}\n"

        a0=m.get("a0","");a1=m.get("a1","");a2=m.get("a2","");a3=m.get("a3","")
        a4=m.get("a4","");a5=m.get("a5","");a6=m.get("a6","");a7=m.get("a7","")
        if a0:
            p += f"庄家总进球赔率定价: 0球={a0}|1={a1}|2={a2}|3={a3}|4={a4}|5={a5}|6={a6}|7+={a7}\n"

        crs_lines = []
        crs_map = {"w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","w32":"3-2","w40":"4-0","w41":"4-1","w42":"4-2",
                   "s00":"0-0","s11":"1-1","s22":"2-2","s33":"3-3",
                   "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3","l23":"2-3","l04":"0-4","l14":"1-4"}
        crs_probs = []
        for key, score in crs_map.items():
            try:
                odds = float(m.get(key, 0) or 0)
                if odds > 1.0:
                    crs_lines.append(f"{score}={odds:.2f}")
                    crs_probs.append((score, odds, 1/odds))
            except: pass
        if crs_lines:
            p += f"机构重点防范比分(仅供参考): {' | '.join(crs_lines[:8])}\n"

        hf_lines = []
        for key, label in {"ss":"主/主","sp":"主/平","sf":"主/负","ps":"平/主","pp":"平/平","pf":"平/负","fs":"负/主","fp":"负/平","ff":"负/负"}.items():
            try:
                v = float(m.get(key, 0) or 0)
                if v > 1: hf_lines.append(f"{label}={v:.2f}")
            except: pass
        if hf_lines: p += f"半全场走势防范: {' | '.join(hf_lines)}\n"

        vote = m.get("vote", {})
        if vote:
            p += f"散户情绪面: 胜{vote.get('win','?')}% 平{vote.get('same','?')}% 负{vote.get('lose','?')}%"
            if vote.get("hhad_win"): p += f" | 让球情绪: 主{vote['hhad_win']}% 平{vote.get('hhad_same','?')}% 客{vote.get('hhad_lose','?')}%"
            p += "\n"

        change = m.get("change", {})
        if change and isinstance(change, dict):
            cw=change.get("win",0);cs=change.get("same",0);cl=change.get("lose",0)
            if cw or cs or cl: p += f"底层赔率异动: 胜{cw} 平{cs} 负{cl}\n"

        if eng.get('bookmaker_implied_home_xg'):
            p += f"纯量化战力预期(xG): 主队{eng['bookmaker_implied_home_xg']} vs 客队{eng.get('bookmaker_implied_away_xg')}\n"

        info = m.get("information", {})
        if isinstance(info, dict):
            for k,v in [("home_injury","主绝密伤停"),("guest_injury","客绝密伤停"),("home_good_news","主利好情报"),("guest_good_news","客利好情报"),("home_bad_news","主利空"),("guest_bad_news","客利空")]:
                if info.get(k): p += f"{v}: {str(info[k])[:300].replace(chr(10),' | ')}\n"

        hs=m.get("home_stats",{});ast2=m.get("away_stats",{})
        if hs.get("form"):
            p += f"主队实录: {hs.get('wins','?')}胜{hs.get('draws','?')}平{hs.get('losses','?')}负 {hs.get('form','?')} 进{hs.get('avg_goals_for','?')}/失{hs.get('avg_goals_against','?')}\n"
            p += f"客队实录: {ast2.get('wins','?')}胜{ast2.get('draws','?')}平{ast2.get('losses','?')}负 {ast2.get('form','?')} 进{ast2.get('avg_goals_for','?')}/失{ast2.get('avg_goals_against','?')}\n"

        analyse = str(m.get('analyse', '')).replace('\n', ' ')[:200]
        baseface = str(m.get('baseface', '')).replace('\n', ' ')[:200]
        intro = str(m.get('intro', m.get('expert_intro', ''))).replace('\n', ' ')[:150]
        if analyse and len(analyse) > 10: p += f"机构面分析: {analyse}\n"
        elif baseface and len(baseface) > 10: p += f"大众基本面: {baseface}\n"
        if intro and len(intro) > 10: p += f"赛事环境: {intro}\n"

        p += "\n"

    p += f"【输出{len(match_analyses)}场JSON数组，只输出数组不要任何其他文字！】\n"
    return p

# ====================================================================
# AI矩阵轮询 v3.0 (原封不动的时间防断与网络架构)
# ====================================================================
FALLBACK_URLS = [None, "https://api520.pro/v1", "https://api521.pro/v1", "https://api522.pro/v1", "https://69.63.213.33:666/v1"]

def get_clean_env_url(name, default=""):
    v = str(os.environ.get(name, globals().get(name, default))).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/-]+)", v)
    return match.group(1) if match else v

def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")

async def async_call_one_ai_batch(session, prompt, url_env, key_env, models_list, num_matches, ai_name):
    key = get_clean_env_key(key_env)
    if not key: return ai_name, {}, "no_key"
    primary_url = get_clean_env_url(url_env)
    backup = [u for u in FALLBACK_URLS if u and u != primary_url][:2]
    urls = [primary_url] + backup
    
    # 【严格执行】：时间设定一字未动
    timeout_map = {"claude": 1800, "grok": 300, "gpt": 240, "gemini": 360}
    timeout_sec = timeout_map.get(ai_name, 200)
    
    AI_PROFILES = {
        "claude": {
            "sys": "你是Claude-4.6-Thinking。你的核心优势是绝不盲从，彻底独立计算！抛弃所有预设比分，利用真实数据对欧赔、亚盘、进球天花板进行极深度的交叉验证。算出来是多少就是多少。只输出JSON数组。",
            "temp": 0.12
        },
        "grok": {
            "sys": "你是Grok-4.2-MultiAgent。立刻联网验证伤停、天气和必发异常。将动态情报与底层赔率结合。抛弃人类定势思维，独立输出最终结论。只输出JSON数组。",
            "temp": 0.22
        },
        "gpt": {
            "sys": "你是顶级量化博彩真神。彻底独立评估庄家意图，绝不迎合常规比分。只输出JSON数组。",
            "temp": 0.15
        },
        "gemini": {
            "sys": "你是Gemini-3.1-Pro。精通泊松极限建模，运用强大算力独立测算比分，只输出JSON数组。",
            "temp": 0.13
        },
    }
    
    best_results = {}; best_model = ""
    for mn in models_list:
        skip_model = False
        for base_url in urls:
            if not base_url or skip_model: continue
            is_gem = "generateContent" in base_url
            url = base_url.rstrip("/")
            if not is_gem and "chat/completions" not in url: url += "/chat/completions"
            headers = {"Content-Type": "application/json"}
            profile = AI_PROFILES.get(ai_name, AI_PROFILES["gpt"])
            if is_gem:
                headers["x-goog-api-key"] = key
                payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": profile["temp"]}, "systemInstruction": {"parts": [{"text": profile["sys"]}]}}
            else:
                headers["Authorization"] = f"Bearer {key}"
                base_payload = {"model": mn, "messages": [{"role": "system", "content": profile["sys"]}, {"role": "user", "content": prompt}]}
                if ai_name != "claude": base_payload["temperature"] = profile["temp"]
                payload = base_payload
            gw = url.split("/v1")[0][:35]
            print(f"  [⏳{timeout_sec}s] {ai_name.upper()} | {mn[:22]} @ {gw}")
            t0 = time.time()
            try:
                # 【严格执行】：ahttp请求配置一字未动
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec, connect=15)) as r:
                    elapsed = round(time.time() - t0, 1)
                    if r.status == 200:
                        data = await r.json()
                        raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip() if is_gem else data["choices"][0]["message"]["content"].strip()
                        clean = re.sub(r"```[\w]*", "", raw_text).strip()
                        start = clean.find("["); end = clean.rfind("]") + 1
                        if start == -1 or end == 0:
                            clean = re.sub(r"[^\[\]{}:,\"'0-9a-zA-Z\u4e00-\u9fa5\s\.\-\+\(\)]", "", clean)
                            start = clean.find("["); end = clean.rfind("]") + 1
                        results = {}
                        if start != -1 and end > start:
                            try:
                                arr = json.loads(clean[start:end])
                                if isinstance(arr, list):
                                    for item in arr:
                                        if item.get("match") and item.get("score"):
                                            try: mid = int(item["match"])
                                            except: mid = item["match"]
                                            results[mid] = {"ai_score": item.get("score"), "analysis": str(item.get("reason","")).strip()[:200], "ai_confidence": int(item.get("ai_confidence",60)), "value_kill": bool(item.get("value_kill",False)), "dark_verdict": str(item.get("dark_verdict",""))}
                            except: pass
                        if len(results) >= max(1, num_matches * 0.5):
                            print(f"    ✅ {ai_name.upper()} 成功: {len(results)}/{num_matches} | {elapsed}s ({mn[:20]})")
                            return ai_name, results, mn
                        if len(results) > len(best_results): best_results = results; best_model = mn; print(f"    ⚠️ 部分 {len(results)}/{num_matches} | {elapsed}s")
                        else: print(f"    ⚠️ 解析不足 {len(results)}条 | {elapsed}s")
                    elif r.status == 429: print(f"    🔥 429限流 | {elapsed}s"); await asyncio.sleep(3); continue
                    elif r.status >= 500: print(f"    💀 HTTP {r.status} | {elapsed}s → 跳模型"); skip_model = True; break
                    elif r.status == 400: print(f"    💀 HTTP 400 | {elapsed}s → 跳模型"); skip_model = True; break
                    else: print(f"    ⚠️ HTTP {r.status} | {elapsed}s")
            except asyncio.TimeoutError: elapsed = round(time.time()-t0,1); print(f"    ⏰ {elapsed}s超时 → 跳模型"); skip_model = True; break
            except Exception as e:
                elapsed = round(time.time()-t0,1); err = str(e)[:40]
                if "connect" in err.lower() or "resolve" in err.lower(): print(f"    ⚠️ 连接失败 {err} | {elapsed}s → 换URL")
                else: print(f"    ⚠️ {err} | {elapsed}s → 跳模型"); skip_model = True; break
            await asyncio.sleep(0.3)
        if len(best_results) >= max(1, num_matches * 0.4):
            print(f"    ✅ {ai_name.upper()} 采用: {len(best_results)}/{num_matches} ({best_model[:20]})"); return ai_name, best_results, best_model
    if best_results: print(f"    ⚠️ {ai_name.upper()} 勉强采用: {len(best_results)}条"); return ai_name, best_results, best_model
    print(f"    ❌ {ai_name.upper()} 全部失败"); return ai_name, {}, "failed"

async def run_ai_matrix(prompt, num_matches):
    ai_configs = [
        ("claude", "CLAUDE_API_URL", "CLAUDE_API_KEY", ["熊猫特供-超纯满血-99额度-claude-opus-4.6-thinking","熊猫-按量-特供顶级-官方正向满血-claude-opus-4.6-thinking"]),
        ("grok", "GROK_API_URL", "GROK_API_KEY", ["熊猫-A-7-grok-4.2-多智能体讨论","熊猫-A-6-grok-4.2-thinking"]),
        ("gpt", "GPT_API_URL", "GPT_API_KEY", ["熊猫-A-10-gpt-5.4","熊猫-按量-gpt-5.4","熊猫-A-10-gpt-5.3-codex"]),
        ("gemini", "GEMINI_API_URL", "GEMINI_API_KEY", ["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking","熊猫-顶级特供-X-17-gemini-3.1-pro-preview"]),
    ]
    all_results = {"gpt": {}, "grok": {}, "claude": {}, "gemini": {}}
    tasks = []
    async with aiohttp.ClientSession() as session:
        for ai_name, url_env, key_env, models in ai_configs:
            tasks.append(async_call_one_ai_batch(session, prompt, url_env, key_env, models, num_matches, ai_name))
        results = await asyncio.gather(*tasks, return_exceptions=True)
    for res in results:
        if isinstance(res, tuple): ai_name, parsed_data, model_used = res; all_results[ai_name] = parsed_data
        else: print(f"  [CRITICAL] AI任务崩溃: {res}")
    return all_results

# ====================================================================
# Merge v5.0 — 绝对尊重 AI 计算结果，切除本地方向强制扭转
# ====================================================================
def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    engine_score = engine_result.get("primary_score", "1-1")
    engine_conf = engine_result.get("confidence", 50)

    ai_all = {"claude": claude_r, "grok": grok_r, "gpt": gpt_r, "gemini": gemini_r}
    ai_scores = []
    ai_conf_sum = 0; ai_conf_count = 0; value_kills = 0
    weights = {"claude": 1.4, "grok": 1.3, "gpt": 1.1, "gemini": 1.0}

    for name, r in ai_all.items():
        if not isinstance(r, dict): continue
        sc = r.get("ai_score", "-")
        if sc and sc not in ["-", "?", ""]:
            ai_scores.append((sc, name))
            conf = r.get("ai_confidence", 60)
            ai_conf_sum += conf * weights.get(name, 1.0)
            ai_conf_count += weights.get(name, 1.0)
            if r.get("value_kill"): value_kills += 1

    # 🚀 AI 算力直达：完全抛弃旧版的 STEP1(强制改方向) 和 STEP2(强制读CRS)
    # 直接以 AI 矩阵的加权投票最高得分为最终比分！
    vote_count = {}
    weighted_vote = {}
    for sc, name in ai_scores:
        vote_count[sc] = vote_count.get(sc, 0) + 1
        weighted_vote[sc] = weighted_vote.get(sc, 0) + weights.get(name, 1.0)

    final_score = engine_score

    if weighted_vote:
        best_voted = max(weighted_vote, key=weighted_vote.get)
        best_count = vote_count.get(best_voted, 0)
        
        # 只要有两个 AI 模型达成共识，或者最核心的 Claude/Grok 独立推演极具信心
        if best_count >= 2:
            final_score = best_voted
        elif len(ai_scores) > 0:
            # 如果分歧极大，听权重最高的老大哥 (Claude / Grok)
            final_score = ai_scores[0][0]

    # ========== 信心计算 ==========
    avg_ai_conf = (ai_conf_sum / ai_conf_count) if ai_conf_count > 0 else 60
    cf = engine_conf
    cf = min(95, cf + int((avg_ai_conf - 60) * 0.4))
    cf = cf + value_kills * 6
    has_warn = any("🚨" in str(s) for s in stats.get("smart_signals", []))
    if has_warn: cf = max(35, cf - 12)
    risk = "低" if cf >= 75 else ("中" if cf >= 55 else "高")

    # ========== 保留原始客观概率计算 ==========
    hp = engine_result.get("home_prob", 33); dp = engine_result.get("draw_prob", 33); ap = engine_result.get("away_prob", 34)
    shp = stats.get("home_win_pct", 33); sdp = stats.get("draw_pct", 33); sap = stats.get("away_win_pct", 34)
    fhp = hp * 0.70 + shp * 0.30; fdp = dp * 0.70 + sdp * 0.30; fap = ap * 0.70 + sap * 0.30
    fhp = max(3, fhp); fdp = max(3, fdp); fap = max(3, fap)
    ft = fhp + fdp + fap
    if ft > 0: fhp = round(fhp/ft*100, 1); fdp = round(fdp/ft*100, 1); fap = round(max(3, 100-fhp-fdp), 1)

    gpt_sc = gpt_r.get("ai_score","-") if isinstance(gpt_r, dict) else "-"
    gpt_an = gpt_r.get("analysis","N/A") if isinstance(gpt_r, dict) else "N/A"
    grok_sc = grok_r.get("ai_score","-") if isinstance(grok_r, dict) else "-"
    grok_an = grok_r.get("analysis","N/A") if isinstance(grok_r, dict) else "N/A"
    gem_sc = gemini_r.get("ai_score","-") if isinstance(gemini_r, dict) else "-"
    gem_an = gemini_r.get("analysis","N/A") if isinstance(gemini_r, dict) else "N/A"
    cl_sc = claude_r.get("ai_score","-") if isinstance(claude_r, dict) else engine_score
    cl_an = claude_r.get("analysis","N/A") if isinstance(claude_r, dict) else engine_result.get("reason", "odds engine")

    pre_pred = {"home_win_pct": fhp, "draw_pct": fdp, "away_win_pct": fap, "steam_move": stats.get("steam_move", {}), "smart_signals": stats.get("smart_signals", []), "line_movement_anomaly": stats.get("line_movement_anomaly", {})}
    cold_door = ColdDoorDetector.detect(match_obj, pre_pred)
    sigs = list(stats.get("smart_signals", []))
    if cold_door["is_cold_door"]: sigs.extend(cold_door["signals"]); cf = max(30, cf - 5)
    
    # 注：移除 calibrate_realistic_score 强制降级拦截，100%相信AI推演的比分

    return {
        "predicted_score": final_score, "home_win_pct": fhp, "draw_pct": fdp, "away_win_pct": fap,
        "confidence": cf, "risk_level": risk,
        "over_under_2_5": "大" if engine_result.get("over_25", 50) > 55 else "小",
        "both_score": "是" if engine_result.get("btts", 45) > 50 else "否",
        "gpt_score": gpt_sc, "gpt_analysis": gpt_an, "grok_score": grok_sc, "grok_analysis": grok_an,
        "gemini_score": gem_sc, "gemini_analysis": gem_an, "claude_score": cl_sc, "claude_analysis": cl_an,
        "ai_avg_confidence": round(avg_ai_conf, 1), "value_kill_count": value_kills,
        "model_agreement": len(set(sc for sc,_ in ai_scores)) <= 1 and len(ai_scores) >= 2,
        "poisson": stats.get("poisson", {}), "refined_poisson": stats.get("refined_poisson", {}),
        "extreme_warning": engine_result.get("scissors_gap_signal", ""),
        "smart_money_signal": " | ".join(sigs), "smart_signals": sigs,
        "model_consensus": stats.get("model_consensus", 0), "total_models": stats.get("total_models", 11),
        "expected_total_goals": engine_result.get("expected_goals", 2.5),
        "over_2_5": engine_result.get("over_25", 50), "btts": engine_result.get("btts", 45),
        "top_scores": stats.get("refined_poisson", {}).get("top_scores", []),
        "elo": stats.get("elo", {}), "random_forest": stats.get("random_forest", {}),
        "gradient_boost": stats.get("gradient_boost", {}), "neural_net": stats.get("neural_net", {}),
        "logistic": stats.get("logistic", {}), "svm": stats.get("svm", {}), "knn": stats.get("knn", {}),
        "dixon_coles": stats.get("dixon_coles", {}), "bradley_terry": stats.get("bradley_terry", {}),
        "home_form": stats.get("home_form", {}), "away_form": stats.get("away_form", {}),
        "handicap_signal": stats.get("handicap_signal", ""), "odds_movement": stats.get("odds_movement", {}),
        "vote_analysis": stats.get("vote_analysis", {}), "h2h_blood": stats.get("h2h_blood", {}),
        "crs_analysis": stats.get("crs_analysis", {}), "ttg_analysis": stats.get("ttg_analysis", {}),
        "halftime": stats.get("halftime", {}), "pace_rating": stats.get("pace_rating", ""),
        "kelly_home": stats.get("kelly_home", {}), "kelly_away": stats.get("kelly_away", {}),
        "odds": stats.get("odds", {}), "experience_analysis": stats.get("experience_analysis", {}),
        "pro_odds": stats.get("pro_odds", {}), "bivariate_poisson": stats.get("bivariate_poisson", {}),
        "asian_handicap_probs": stats.get("asian_handicap_probs", {}),
        "bookmaker_implied_home_xg": engine_result.get("bookmaker_implied_home_xg", "?"),
        "bookmaker_implied_away_xg": engine_result.get("bookmaker_implied_away_xg", "?"),
        "cold_door": cold_door,
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
        if exp_score >= 15 and pr.get("result") == "平局" and exp_info.get("draw_rules", 0) >= 3: s += 12
        elif exp_score >= 10: s += 5
        if exp_info.get("recommendation", "").startswith("⚠️"): s -= 3
        smart_money = str(pr.get("smart_money_signal", ""))
        direction = pr.get("result", "")
        if "Sharp" in smart_money:
            if ("客胜" in smart_money and direction == "主胜") or ("主胜" in smart_money and direction == "客胜"): s -= 30
        cold = pr.get("cold_door", {})
        if cold.get("is_cold_door"): s -= 8
        p["recommend_score"] = round(s, 2)
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]

def extract_num(ms):
    wm = {"一":1000,"二":2000,"三":3000,"四":4000,"五":5000,"六":6000,"日":7000,"天":7000}
    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999

# ====================================================================
# run_predictions v5.0 — 算力彻底解放
# ====================================================================
def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 80)
    print(f"  [QUANT ENGINE vMAX 5.0] 彻底粉碎本地预设边界 | 100%接纳AI原生算力推演 | {len(ms)} 场")
    print("=" * 80)
    match_analyses = []
    for i, m in enumerate(ms):
        eng = predict_match(m)
        league_info, _, _, _ = build_league_intelligence(m)
        sp = ensemble.predict(m, {})
        exp_result = exp_engine.analyze(m)
        match_analyses.append({"match": m, "engine": eng, "league_info": league_info, "stats": sp, "index": i+1, "experience": exp_result})
    all_ai = {"claude": {}, "gemini": {}, "gpt": {}, "grok": {}}
    if use_ai and match_analyses:
        prompt = build_batch_prompt(match_analyses)
        print(f"  [PROMPT] 已切断所有比分诱导限制。纯净客观数据开始注射入底层...")
        start_t = time.time()
        all_ai = asyncio.run(run_ai_matrix(prompt, len(match_analyses)))
        print(f"  [AI MATRIX] 独立算力突破完成，耗时 {time.time()-start_t:.1f}s")
    res = []
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        mg = merge_result(ma["engine"], all_ai["gpt"].get(i+1,{}), all_ai["grok"].get(i+1,{}), all_ai["gemini"].get(i+1,{}), all_ai["claude"].get(i+1,{}), ma["stats"], m)
        try: mg = apply_experience_to_prediction(m, mg, exp_engine); print(f"    → apply_experience_to_prediction 已注入")
        except Exception as e: print(f"    ⚠️ experience跳过: {e}")
        try: mg = apply_odds_history(m, mg); print(f"    → apply_odds_history 已注入")
        except Exception as e: print(f"    ⚠️ odds_history跳过: {e}")
        try: mg = apply_quant_edge(m, mg); print(f"    → apply_quant_edge 已注入")
        except Exception as e: print(f"    ⚠️ quant_edge跳过: {e}")
        try: mg = apply_wencai_intel(m, mg); print(f"    → apply_wencai_intel 已注入")
        except Exception as e: print(f"    ⚠️ wencai_intel跳过: {e}")
        try: mg = upgrade_ensemble_predict(m, mg); print(f"    → upgrade_ensemble_predict 已注入")
        except Exception as e: print(f"    ⚠️ advanced_models跳过: {e}")
        
        score_str = mg.get("predicted_score", "1-1")
        # 🔥 根据 AI 生成的比分，反向判定比赛方向（主胜/平/负）
        try:
            sh, sa = map(int, score_str.split("-"))
            if sh > sa: mg["result"] = "主胜"
            elif sh < sa: mg["result"] = "客胜"
            else: mg["result"] = "平局"
        except:
            pcts = {"主胜": mg["home_win_pct"], "平局": mg["draw_pct"], "客胜": mg["away_win_pct"]}
            mg["result"] = max(pcts, key=pcts.get)
            
        res.append({**m, "prediction": mg})
        cold = mg.get("cold_door", {})
        cold_tag = f" [❄️{cold.get('level','')}冷门预警]" if cold.get("is_cold_door") else ""
        print(f"  [{i+1}] {m.get('home_team')} vs {m.get('away_team')} => {mg['result']} ({mg['predicted_score']}) | CF: {mg['confidence']}% | AI推演: {mg.get('ai_avg_confidence', 0)}{cold_tag}")
    
    t4 = select_top4(res)
    t4ids = [t.get("id") for t in t4]
    for r in res: r["is_recommended"] = r.get("id") in t4ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    diary = load_ai_diary()
    cold_count = len([r for r in res if r.get("prediction",{}).get("cold_door",{}).get("is_cold_door")])
    diary["yesterday_win_rate"] = f"{len([r for r in res if r['prediction']['confidence']>70])}/{max(1,len(res))}"
    diary["reflection"] = f"vMAX5.0 | {cold_count}冷门预警 | 彻底肃清本地干扰，100%采纳AI算力投票结果"
    save_ai_diary(diary)
    return res, t4


```
