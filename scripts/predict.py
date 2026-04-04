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
# 🧊 冷门猎手引擎 + 深度赔率映射
# ====================================================================
REALISTIC_MAP = {
    "ultra_low": ["0-0", "1-0", "0-1", "1-1", "2-0", "0-2"],
    "low": ["1-0", "0-1", "1-1", "2-0", "0-2", "2-1", "1-2", "0-0"],
    "medium": ["1-0", "0-1", "1-1", "2-0", "0-2", "2-1", "1-2", "2-2", "3-0", "0-3", "3-1", "1-3"],
    "high": ["2-1", "1-2", "3-1", "1-3", "2-2", "3-0", "0-3", "3-2", "2-3", "4-0", "0-4", "4-1", "1-4"]
}

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

def calibrate_realistic_score(current_score, sp_h, sp_d, sp_a, cold_door):
    if cold_door.get("is_cold_door") and cold_door.get("level") == "顶级":
        return current_score
    implied_total = (1/sp_h + 1/sp_d + 1/sp_a) * 100 if sp_h > 1 and sp_d > 1 and sp_a > 1 else 300
    if implied_total < 270: allowed = REALISTIC_MAP["ultra_low"]
    elif implied_total < 300: allowed = REALISTIC_MAP["low"]
    elif implied_total < 330: allowed = REALISTIC_MAP["medium"]
    else: allowed = REALISTIC_MAP["high"]
    if current_score in allowed: return current_score
    try:
        home, away = map(int, current_score.split("-"))
        if home + away <= 2: return "1-0" if home >= away else "0-1"
        elif home + away <= 3: return "2-1" if home >= away else "1-2"
        else: return "2-1" if home >= away else "1-2"
    except: return "1-1"

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
# 🧠 两阶段AI架构 v8.0 — 动态切片并发 + 极致算力压榨 + 104防断连修复
# ====================================================================

def build_phase1_prompt(match_analyses_chunk):
    """Phase1 Prompt: 极致压榨数学逻辑，单批次处理，防止注意力衰减"""
    diary = load_ai_diary()
    p = "【核心指令】你现在是顶级的华尔街体育量化对冲基金分析师。你将处理以下比赛数据。\n"
    p += "你必须独立使用概率论、泊松分布推演和赔率隐含期望进行深度计算。\n"
    if diary.get("reflection"):
        p += f"【系统进化法则】昨日复盘警告: {diary['reflection']}\n\n"

    p += "【严格输出格式约束】\n"
    p += "必须输出合法的 JSON 对象，禁止包含任何 Markdown 格式或额外说明文字。\n"
    p += '格式：{"matches": [{"match": 1, "top3": [{"score": "1-0", "prob": 18.2}], "reason": "在此处写下你的推导公式和分析逻辑(限100-150字)", "ai_confidence": 75}, ...]}\n\n'

    p += "【原始盘赔数据】\n"
    for ma in match_analyses_chunk:
        m = ma["match"]
        idx = ma["index"]
        h = m.get("home_team", "Home")
        a = m.get("away_team", "Away")
        sp_h = float(m.get("sp_home", 0) or 0)
        sp_d = float(m.get("sp_draw", 0) or 0)
        sp_a = float(m.get("sp_away", 0) or 0)

        p += f"{'='*40}\n[{idx}] {h} vs {a} | 联赛: {m.get('league', '')}\n"
        p += f"竞彩欧赔: {sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f} | 亚盘让球: {m.get('give_ball', '0')}\n"
        
        # 简化核心数据，降低 Token 噪声，聚焦核心概率
        crs_probs = []
        for key, score in {"w10":"1-0","w20":"2-0","w21":"2-1","s00":"0-0","s11":"1-1","s22":"2-2","l01":"0-1","l02":"0-2","l12":"1-2"}.items():
            odds = float(m.get(key, 0) or 0)
            if odds > 1: crs_probs.append((score, odds))
        if crs_probs:
            p += f"核心波胆(CRS)赔率: {' | '.join(f'{s}@{o:.1f}' for s,o in crs_probs)}\n"
            
        a0=m.get("a0","");a7=m.get("a7","")
        if a0 and a7:
            p += f"总进球信号: 0球赔率={a0} | 7+球赔率={a7}\n"
        
        info = m.get("information", {})
        if isinstance(info, dict) and (info.get("home_bad_news") or info.get("guest_bad_news")):
            p += f"绝密伤停: 主队坏消息({str(info.get('home_bad_news',''))[:80]}) | 客队坏消息({str(info.get('guest_bad_news',''))[:80]})\n"

    p += "\n【执行要求】请务必处理本批次提供的所有场次！只允许输出带有 matches 键的 JSON 对象！"
    return p

def build_phase2_prompt(match_analyses_chunk, p1_results):
    """Phase2 Prompt: Claude 作为主裁判进行共识仲裁"""
    p = "【核心指令】你是终极风控官。下方是三个独立初级 AI 给出的 TOP3 比分推演。\n"
    p += "你的任务：消除他们的幻觉，结合给出的欧赔和盘口，定夺每场的【唯一最终比分】。\n"
    p += "【裁决法则】1. 寻找共识；2. 共识比分若过于离谱(例如在当前赔率下不合逻辑)，强制降级为防守比分(1-1/1-0/0-1)。\n\n"
    p += "必须输出合法 JSON，格式：{\"matches\": [{\"match\": 1, \"score\": \"1-1\", \"reason\": \"裁决理由\", \"ai_confidence\": 85}]}\n\n"

    for ma in match_analyses_chunk:
        m = ma["match"]
        idx = ma["index"]
        sp_h = float(m.get("sp_home", 0) or 0)
        sp_d = float(m.get("sp_draw", 0) or 0)
        sp_a = float(m.get("sp_away", 0) or 0)
        p += f"[{idx}] {m.get('home_team')} vs {m.get('away_team')} | 欧赔: {sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f}\n"

        for ai_name in ["gpt", "grok", "gemini"]:
            ai_data = p1_results.get(ai_name, {}).get(idx, {})
            if ai_data and ai_data.get("top3"):
                scores_str = " ".join(f"{t.get('score')}({t.get('prob')}%)" for t in ai_data["top3"][:2])
                p += f"  {ai_name.upper()}: {scores_str} | 信心:{ai_data.get('ai_confidence')} | {str(ai_data.get('reason',''))[:40]}\n"
        p += "\n"
    return p

# ====================================================================
# AI调用引擎（强级 JSON 沙盒提取 + 超时管控）
# ====================================================================
FALLBACK_URLS = [None, "https://api520.pro/v1", "https://api521.pro/v1", "https://api522.pro/v1", "https://69.63.213.33:666/v1"]

def get_clean_env_url(name, default=""):
    v = str(os.environ.get(name, globals().get(name, default))).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/-]+)", v)
    return match.group(1) if match else v

def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")

async def async_call_one_ai_batch(session, prompt, url_env, key_env, models_list, num_matches, ai_name):
    """带有更强鲁棒性的单批次请求器"""
    key = get_clean_env_key(key_env)
    if not key: return ai_name, {}, "no_key"
    primary_url = get_clean_env_url(url_env)
    urls = [primary_url] + [u for u in FALLBACK_URLS if u and u != primary_url][:1]
    
    # 动态超时配置：针对单批次(Chunk)优化时长，防 Nginx 104 杀流
    timeout_map = {"claude": 180, "grok": 150, "gpt": 120, "gemini": 150} 
    timeout_sec = timeout_map.get(ai_name, 150)

    AI_PROFILES = {
        "claude": {"sys": "你是严格的JSON生成器，必须输出合法的JSON对象，决不允许附加任何分析和Markdown。", "temp": 0.1},
        "grok": {"sys": "你是Grok。对于高水平赛事务必利用联网搜索Oddsportal的Pinnacle赔率或球队突发伤停新闻。必须输出JSON对象。", "temp": 0.2},
        "gpt": {"sys": "你是量化推演引擎。使用泊松分布和贝叶斯定理推导概率。必须输出JSON对象。", "temp": 0.15},
        "gemini": {"sys": "你是概率学大师。必须输出合法的JSON对象。", "temp": 0.15},
    }

    best_results = {}; best_model = ""
    for mn in models_list:
        for base_url in urls:
            if not base_url: continue
            is_gem = "generateContent" in base_url
            url = base_url.rstrip("/")
            if not is_gem and "chat/completions" not in url: url += "/chat/completions"
            
            headers = {"Content-Type": "application/json"}
            profile = AI_PROFILES.get(ai_name, AI_PROFILES["gpt"])
            
            if is_gem:
                headers["x-goog-api-key"] = key
                payload = {
                    "contents": [{"parts":[{"text": prompt}]}],
                    "generationConfig": {"temperature": profile["temp"], "responseMimeType": "application/json"},
                    "systemInstruction": {"parts":[{"text": profile["sys"]}]}
                }
            else:
                headers["Authorization"] = f"Bearer {key}"
                payload = {
                    "model": mn,
                    "messages": [{"role": "system", "content": profile["sys"]}, {"role": "user", "content": prompt}],
                    "temperature": profile["temp"]
                }
                # ★ 核心修复：仅对支持 JSON mode 的 OpenAI/Grok 开启，跳过 Claude 防止 400 崩溃
                if ai_name in ["gpt", "grok"]: 
                    payload["response_format"] = {"type": "json_object"}

            gw = url.split("/v1")[0][:35]
            t0 = time.time()
            try:
                # 设定严格的 connect timeout，防止底层 Socket 死锁
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec, connect=10)) as r:
                    elapsed = round(time.time()-t0, 1)
                    if r.status == 200:
                        data = await r.json()
                        raw_text = data["candidates"][0]["content"]["parts"][0]["text"] if is_gem else data["choices"][0]["message"]["content"]
                        
                        # 安全 JSON 截取沙盒
                        start_idx = raw_text.find("{")
                        end_idx = raw_text.rfind("}") + 1
                        results = {}
                        
                        if start_idx != -1 and end_idx > start_idx:
                            try:
                                parsed_data = json.loads(raw_text[start_idx:end_idx])
                                arr = parsed_data.get("matches", parsed_data)
                                if isinstance(arr, list):
                                    for item in arr:
                                        mid = int(item.get("match", 0))
                                        if mid == 0: continue
                                        if item.get("top3"):
                                            results[mid] = {
                                                "top3": item["top3"],
                                                "ai_score": item["top3"][0].get("score","1-1") if isinstance(item["top3"], list) and len(item["top3"]) > 0 else "1-1",
                                                "reason": str(item.get("reason",""))[:150],
                                                "ai_confidence": int(item.get("ai_confidence", 60)),
                                            }
                                        elif item.get("score"):
                                            results[mid] = {
                                                "ai_score": item["score"],
                                                "analysis": str(item.get("reason",""))[:150],
                                                "ai_confidence": int(item.get("ai_confidence", 60)),
                                            }
                            except Exception: pass
                        
                        if len(results) >= max(1, num_matches * 0.5):
                            print(f"    ✅ {ai_name.upper()} 切片成功: {len(results)}/{num_matches}场 | {elapsed}s ({mn[:20]})")
                            return ai_name, results, mn
                        else:
                            if len(results) > len(best_results): best_results = results; best_model = mn
                    
                    elif r.status == 429: await asyncio.sleep(2); continue
                    else: break # 遇到 400 格式错或 500 级故障直接跳过当前 URL 换节点
            except Exception as e:
                pass # 忽略网络超时/断连，继续循环下一个 URL 或 Model

    if best_results:
        print(f"    ⚠️ {ai_name.upper()} 残缺降级采用: {len(best_results)}条")
    else:
        print(f"    ❌ {ai_name.upper()} 批次全军覆没")
    return ai_name, best_results, best_model

async def run_ai_matrix_two_phase(match_analyses):
    """
    两阶段 AI 架构：切片并发引擎 (Chunking)
    将大量比赛切分成每块 12 场，防止大模型注意力崩溃与网关 104 断连
    """
    print(f"  [AI 引擎] 总计 {len(match_analyses)} 场比赛，启动切片并发分发模式...")
    
    CHUNK_SIZE = 12
    p1_configs = [
        ("grok", "GROK_API_URL", "GROK_API_KEY", ["熊猫-A-7-grok-4.2-多智能体讨论", "熊猫-A-6-grok-4.2-thinking"]),
        ("gpt", "GPT_API_URL", "GPT_API_KEY", ["熊猫-A-10-gpt-5.4", "熊猫-按量-gpt-5.4"]),
        ("gemini", "GEMINI_API_URL", "GEMINI_API_KEY", ["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking"])
    ]
    
    p1_results_merged = {"gpt": {}, "grok": {}, "gemini": {}}
    
    async with aiohttp.ClientSession() as session:
        # Phase 1: 切片并发执行初级模型
        for i in range(0, len(match_analyses), CHUNK_SIZE):
            chunk = match_analyses[i:i + CHUNK_SIZE]
            p1_prompt = build_phase1_prompt(chunk)
            print(f"  ▶ [Phase 1] 正在派发批次 {i//CHUNK_SIZE + 1} (包含 {len(chunk)} 场)...")
            
            tasks = [async_call_one_ai_batch(session, p1_prompt, u, k, m, len(chunk), n) for n, u, k, m in p1_configs]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for res in chunk_results:
                if isinstance(res, tuple):
                    n, d, _ = res
                    p1_results_merged[n].update(d)
        
        # Phase 2: Claude 主裁判也必须切片执行，防止超时崩溃
        print(f"  ▶ [Phase 2] Claude 终审裁判介入...")
        claude_merged = {}
        claude_config = ("claude", "CLAUDE_API_URL", "CLAUDE_API_KEY", ["熊猫特供-超纯满血-99额度-claude-opus-4.6-thinking", "熊猫-按量-顶级特供-官max-claude-opus-4.6-thinking"])
        
        for i in range(0, len(match_analyses), CHUNK_SIZE):
            chunk = match_analyses[i:i + CHUNK_SIZE]
            p2_prompt = build_phase2_prompt(chunk, p1_results_merged)
            
            _, c_res, _ = await async_call_one_ai_batch(session, p2_prompt, claude_config[1], claude_config[2], claude_config[3], len(chunk), "claude")
            claude_merged.update(c_res)

    all_r = p1_results_merged.copy()
    # ★ 核心修复：修复原本由于 claude_rp 拼写错误导致的致命 NameError 崩溃
    all_r["claude"] = claude_merged 
    return all_r

# ====================================================================
# Merge v4.0 — 方向先行+CRS回检+AI投票校验
# ====================================================================
def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    sp_h = float(match_obj.get("sp_home", match_obj.get("win", 0)) or 0)
    sp_d = float(match_obj.get("sp_draw", match_obj.get("same", 0)) or 0)
    sp_a = float(match_obj.get("sp_away", match_obj.get("lose", 0)) or 0)
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

    # ========== STEP1: 方向投票（先定方向，再定比分）==========
    dir_vote = {"主胜": 0, "平局": 0, "客胜": 0}
    for sc, name in ai_scores:
        try:
            h, a = map(int, sc.split("-"))
            if h > a: dir_vote["主胜"] += weights.get(name, 1.0)
            elif h < a: dir_vote["客胜"] += weights.get(name, 1.0)
            else: dir_vote["平局"] += weights.get(name, 1.0)
        except: pass

    # 引擎方向也参与投票（权重1.0）
    try:
        eh, ea = map(int, engine_score.split("-"))
        if eh > ea: dir_vote["主胜"] += 1.0
        elif eh < ea: dir_vote["客胜"] += 1.0
        else: dir_vote["平局"] += 1.0
    except: pass

    odds_range = max(sp_h,sp_d,sp_a) - min(sp_h,sp_d,sp_a) if sp_h>1 and sp_d>1 and sp_a>1 else 99
    if odds_range < 0.8:
        dir_vote["平局"] += 2.0
    elif odds_range < 1.2:
        dir_vote["平局"] += 1.0

    if sp_d > 1:
        implied_draw = (1/sp_d) / ((1/sp_h if sp_h>1 else 0.4) + (1/sp_d) + (1/sp_a if sp_a>1 else 0.3))
        if implied_draw > 0.30:
            dir_vote["平局"] += 1.5

    best_direction = max(dir_vote, key=dir_vote.get)

    # ========== STEP2: CRS赔率辅助选比分 ==========
    crs_key_map = {"1-0":"w10","0-1":"l01","2-1":"w21","1-2":"l12","2-0":"w20","0-2":"l02",
                   "0-0":"s00","1-1":"s11","3-0":"w30","3-1":"w31","0-3":"l03","1-3":"l13",
                   "2-2":"s22","3-2":"w32","2-3":"l23"}
    
    def get_crs_odds(score):
        key = crs_key_map.get(score, "")
        try: return float(match_obj.get(key, 99) or 99)
        except: return 99.0

    direction_scores = {
        "主胜": ["1-0", "2-0", "2-1", "3-0", "3-1"],
        "平局": ["0-0", "1-1", "2-2"],
        "客胜": ["0-1", "0-2", "1-2", "0-3", "1-3"],
    }
    crs_best_in_dir = sorted(direction_scores[best_direction], key=lambda s: get_crs_odds(s))

    # ========== STEP3: AI比分投票 + CRS回检 ==========
    vote_count = {}
    weighted_vote = {}
    for sc, name in ai_scores:
        vote_count[sc] = vote_count.get(sc, 0) + 1
        weighted_vote[sc] = weighted_vote.get(sc, 0) + weights.get(name, 1.0)

    final_score = engine_score

    if weighted_vote:
        best_voted = max(weighted_vote, key=weighted_vote.get)
        best_count = vote_count.get(best_voted, 0)
        best_crs = get_crs_odds(best_voted)

        if best_count >= 3 and best_crs <= 10.0:
            final_score = best_voted
        elif best_count >= 2 and best_crs <= 8.0:
            final_score = best_voted
        elif best_count >= 2 and best_crs > 10.0:
            try:
                bh, ba = map(int, best_voted.split("-"))
                ai_dir = "主胜" if bh > ba else ("客胜" if bh < ba else "平局")
                final_score = sorted(direction_scores[ai_dir], key=lambda s: get_crs_odds(s))[0]
            except:
                final_score = crs_best_in_dir[0] if crs_best_in_dir else engine_score
        elif len(ai_scores) >= 2:
            final_score = crs_best_in_dir[0] if crs_best_in_dir else engine_score
        elif len(ai_scores) == 1 and ai_scores[0][1] in ["claude", "grok"]:
            r = ai_all.get(ai_scores[0][1], {})
            if isinstance(r, dict) and r.get("ai_confidence", 0) >= 80:
                sc_crs = get_crs_odds(ai_scores[0][0])
                if sc_crs <= 10.0:
                    final_score = ai_scores[0][0]
                else:
                    final_score = crs_best_in_dir[0] if crs_best_in_dir else engine_score

    # ========== STEP4: 0-0通道 ==========
    exp_analysis = stats.get("experience_analysis", {})
    zero_zero_boost = exp_analysis.get("zero_zero_boost", 0) if isinstance(exp_analysis, dict) else 0
    a0_val = float(match_obj.get("a0", 99) or 99)
    s00_val = float(match_obj.get("s00", 99) or 99)

    if zero_zero_boost >= 10:
        if any(sc == "0-0" for sc, _ in ai_scores):
            final_score = "0-0"
        elif a0_val < 8.0 and s00_val < 9.0:
            final_score = "0-0"
        elif final_score in ["1-0", "0-1", "1-1"] and zero_zero_boost >= 14:
            final_score = "0-0"
    elif zero_zero_boost >= 6 and a0_val < 8.5 and final_score == "1-1":
        final_score = "0-0"

    # ========== 信心计算 ==========
    avg_ai_conf = (ai_conf_sum / ai_conf_count) if ai_conf_count > 0 else 60
    cf = engine_conf
    cf = min(95, cf + int((avg_ai_conf - 60) * 0.4))
    cf = cf + value_kills * 6
    has_warn = any("🚨" in str(s) for s in stats.get("smart_signals", []))
    if has_warn: cf = max(35, cf - 12)
    risk = "低" if cf >= 75 else ("中" if cf >= 55 else "高")

    # ========== 概率融合 ==========
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
    final_score = calibrate_realistic_score(final_score, sp_h, sp_d, sp_a, cold_door)

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
# run_predictions v8.0 — 切片并发重构版
# ====================================================================
def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 80)
    print(f"  [QUANT ENGINE vMAX 8.0] 切片高并发模式 | {len(ms)} 场比赛")
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
        print(f"  [TWO-PHASE] 启动切片并发(Chunking)双阶段AI架构...")
        start_t = time.time()
        all_ai = asyncio.run(run_ai_matrix_two_phase(match_analyses))
        print(f"  [AI MATRIX] 数据榨取与仲裁完成，总耗时 {time.time()-start_t:.1f}s")
        
    res = []
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        # 安全读取 AI 回传的数据字典，防止下标越界
        gpt_data = all_ai.get("gpt", {}).get(i+1, {})
        grok_data = all_ai.get("grok", {}).get(i+1, {})
        gemini_data = all_ai.get("gemini", {}).get(i+1, {})
        claude_data = all_ai.get("claude", {}).get(i+1, {})
        
        mg = merge_result(ma["engine"], gpt_data, grok_data, gemini_data, claude_data, ma["stats"], m)
        
        try: mg = apply_experience_to_prediction(m, mg, exp_engine)
        except Exception: pass
        try: mg = apply_odds_history(m, mg)
        except Exception: pass
        try: mg = apply_quant_edge(m, mg)
        except Exception: pass
        try: mg = apply_wencai_intel(m, mg)
        except Exception: pass
        try: mg = upgrade_ensemble_predict(m, mg)
        except Exception: pass
        
        score_str = mg.get("predicted_score", "1-1")
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
        cold_tag = f" [❄️{cold.get('level','')}冷门]" if cold.get("is_cold_door") else ""
        print(f"  [{i+1}] {m.get('home_team')} vs {m.get('away_team')} => {mg['result']} ({mg['predicted_score']}) | CF: {mg['confidence']}% | AI信心: {mg.get('ai_avg_confidence', 0)}{cold_tag}")
        
    t4 = select_top4(res)
    t4ids = [t.get("id") for t in t4]
    for r in res: r["is_recommended"] = r.get("id") in t4ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    
    diary = load_ai_diary()
    cold_count = len([r for r in res if r.get("prediction",{}).get("cold_door",{}).get("is_cold_door")])
    diary["yesterday_win_rate"] = f"{len([r for r in res if r['prediction']['confidence']>70])}/{max(1,len(res))}"
    diary["reflection"] = f"vMAX 8.0 | {cold_count}冷门 | JSON切片加固 | AI并发榨取升级"
    save_ai_diary(diary)
    
    return res, t4
