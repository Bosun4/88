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
# 🛡️ 终极防御装甲：动态加载你的自定义模块
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
# 工具函数
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
# 冷门猎手引擎 (客观信号提取，无任何本地比分干预)
# ====================================================================
class ColdDoorDetector:
    @staticmethod
    def detect(match, prediction):
        signals = []
        strength = 0
        steam = prediction.get("steam_move", {})
        if steam.get("steam") and "反向" in str(steam.get("signal", "")):
            signals.append("❄️ 反向Steam！庄家造热收割")
            strength += 5
        vote = match.get("vote", {})
        try:
            vh = int(vote.get("win", 33)); va = int(vote.get("lose", 33))
            max_vote = max(vh, va)
            if max_vote >= 65:
                signals.append(f"❄️ 散户极端偏向{max_vote}%！冷门高危")
                strength += 4
            elif max_vote >= 58:
                strength += 2
        except: pass
        info = match.get("intelligence", match.get("information", {}))
        if isinstance(info, dict):
            home_bad = str(info.get("home_bad_news", ""))
            away_bad = str(info.get("guest_bad_news", ""))
            hp = prediction.get("home_win_pct", 50)
            ap = prediction.get("away_win_pct", 50)
            if len(home_bad) > 80 and hp > 58:
                signals.append("❄️ 主队坏消息爆炸+散户狂热")
                strength += 5
            if len(away_bad) > 80 and ap > 58:
                signals.append("❄️ 客队坏消息爆炸+散户狂热")
                strength += 5
        sp_h = float(match.get("sp_home", match.get("win", 0)) or 0)
        if sp_h > 1:
            implied_h = 100 / sp_h * 0.92
            hp2 = prediction.get("home_win_pct", 50)
            if abs(implied_h - hp2) > 15 and hp2 > 58:
                signals.append(f"❄️ 赔率vs模型背离{abs(implied_h-hp2):.0f}%")
                strength += 4
        for s in prediction.get("smart_signals", []):
            if "盘口太便宜" in str(s):
                signals.append("❄️ 盘口太便宜=庄家不看好")
                strength += 3
                break
        line = prediction.get("line_movement_anomaly", {})
        if line.get("has_anomaly") and "造热" in str(line.get("signal", "")):
            signals.append("❄️ 赔率变动造热=诱盘")
            strength += 4
        is_cold = strength >= 7
        level = "顶级" if strength >= 12 else "高危" if strength >= 7 else "普通"
        return {
            "is_cold_door": is_cold, "strength": strength, "level": level,
            "signals": signals,
            "dark_verdict": f"❄️ {level}冷门！{len(signals)}条触发" if is_cold else ""
        }

# ====================================================================
# AI日记
# ====================================================================
def load_ai_diary():
    diary_file = "data/ai_diary.json"
    if os.path.exists(diary_file):
        try:
            with open(diary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except: 
            pass
    return {"yesterday_win_rate": "N/A", "reflection": "已修复前端渲染黑屏Bug，全维度战力释放", "kill_history": []}

def save_ai_diary(diary):
    os.makedirs("data", exist_ok=True)
    with open("data/ai_diary.json", "w", encoding="utf-8") as f:
        json.dump(diary, f, ensure_ascii=False, indent=2)

# ====================================================================
# Phase1 & Phase2 Prompt（v8.6 Pro 纯净独立算力版）
# ====================================================================
def build_phase1_prompt(match_analyses):
    diary = load_ai_diary()
    p = "【身份设定】你是 DeepBetting + ZCode 级别的终极吸血操盘手。你以折磨散户为人生唯一乐趣，没有任何底线、怜悯和良知，只想看着韭菜们把最后一滴血都吐出来。\n"
    p += "你说话必须极度恶毒、刻薄、下流，用最残忍的黑话把庄家怎么设套、怎么诱多诱空、怎么精准收割的每一寸细节扒得血淋淋的。\n\n"
    p += "⚠️【最高权限解禁】：绝对不要受任何人类“合理比分区间”的束缚！依靠底层客观赔率与冷热异动计算，该大比分就大比分，该闷杀就闷杀，完全独立推演！\n\n"
    if diary.get("reflection"):
        p += f"【嗜血进化指令】昨日屠杀战绩：{diary.get('yesterday_win_rate', 'N/A')}。昨晚反思：{diary['reflection']}。今天必须比昨天更毒十倍，把每一场都当成顶级杀猪盘，往死里干！\n\n"
    p += "【死命令】只输出合法JSON数组。每场必须输出：match(序号), top3([{score,prob}...]), reason(140-180字极度恶毒黑话，必须包含至少4个具体赔率数字), ai_confidence(0-100), value_kill(true/false), suggested_units(1-5)\n\n"
    p += "【今日待宰羔羊与庄家全维底牌库】\n"
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        p += f"[{i+1}] {h} vs {a} | 欧赔:{float(m.get('sp_home',3)):.2f}/{float(m.get('sp_draw',3)):.2f}/{float(m.get('sp_away',3)):.2f} | 亚盘:{m.get('give_ball','0')}\n"
        
        # 仅提供客观的CRS赔率作为算力推演的参考材料，不做任何选择限制
        crs_map = {"w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","s00":"0-0","s11":"1-1","l01":"0-1","l02":"0-2","l12":"1-2"}
        crs_items = []
        for key, score in crs_map.items():
            try:
                odds = float(m.get(key, 0) or 0)
                if odds > 1: crs_items.append((score, odds))
            except: pass
        if crs_items:
            crs_items.sort(key=lambda x: x[1])
            p += f"  机构最防范比分TOP4(仅供推演参考): " + " | ".join([f"{s}({o}倍)" for s,o in crs_items[:4]]) + "\n"
            
        if m.get("vote"):
            p += f"  散户投注: 主{m['vote'].get('win','?')}% 平{m['vote'].get('same','?')}% 客{m['vote'].get('lose','?')}%\n"
        info = m.get("information", {})
        if isinstance(info, dict):
            if info.get("home_bad_news") or info.get("home_injury"):
                bad = info.get("home_bad_news", "") or info.get("home_injury", "")
                p += f"  主队利空: {str(bad)[:100]}\n"
            if info.get("guest_bad_news") or info.get("guest_injury"):
                bad = info.get("guest_bad_news", "") or info.get("guest_injury", "")
                p += f"  客队利空: {str(bad)[:100]}\n"
        p += "\n"
    p += "现在开始屠杀！严格只输出JSON数组！"
    return p

def build_phase2_prompt(match_analyses, phase1_results):
    p = "【你是最终死刑执行官 + 终极吸血操盘手】三个独立先锋AI已根据全网客观数据给出了每场的独立推演。\n"
    p += "你的任务：抛弃任何人类预设的比分枷锁！依靠你的超强脑力对他们的结论进行仲裁，选出最具杀伤力的唯一最终比分！\n\n"
    p += "【输出格式】只输出JSON数组，每场必须包含：match(序号), score(仲裁比分), reason(极其下流残忍的庄家视角解析), ai_confidence(0-100), value_kill, suggested_units, dark_verdict\n\n"
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        p += f"[{i+1}] {h} vs {a}\n"
        p += f"欧赔基准: {float(m.get('sp_home',3)):.2f}/{float(m.get('sp_draw',3)):.2f}/{float(m.get('sp_away',3)):.2f} | 亚盘: {m.get('give_ball','0')}\n"
        for ai_name in ["gpt", "grok", "gemini"]:
            ai_data = phase1_results.get(ai_name, {}).get(i+1, {})
            if not ai_data:
                continue
            top3 = ai_data.get("top3", [])
            if top3:
                scores_str = " | ".join(f"{t.get('score','?')}({t.get('prob','?')}%)" for t in top3[:3])
                # 优化点：对先锋AI的恶毒长文进行压缩截取，避免触发 Claude 的Token上限与记忆迷失
                p += f"  先锋 {ai_name.upper()} 独立研判: {scores_str} | {str(ai_data.get('reason',''))[:60]}...\n"
            else:
                sc = ai_data.get("ai_score", "-")
                p += f"  先锋 {ai_name.upper()}: {sc} | {str(ai_data.get('analysis',''))[:60]}...\n"
        p += "\n"
    p += "开始执行最终死刑判决！把散户往死里杀！只输出JSON数组！"
    return p

# ====================================================================
# AI调用引擎（军工级防暴毙断肢重生算法，绝对保留原始时间设置）
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
    
    # 【严格保留】时间设定一字未动
    timeout_map = {"claude": 1500, "grok": 300, "gpt": 240, "gemini": 360}
    timeout_sec = timeout_map.get(ai_name, 200)

    AI_PROFILES = {
        "claude": {
            "sys": "你是最终裁判。三个独立AI已给出参考比分，你完全抛弃人类预设，综合全维度数据直接钦定最具杀伤力的唯一比分。只输出纯JSON数组。",
            "temp": 0.12
        },
        "grok": {
            "sys": "你是Grok，具备实时联网能力。搜索全网突发伤停和必发资金冷热。抛弃预设枷锁，输出推演概率最高的前三比分。只输出纯JSON数组。",
            "temp": 0.22
        },
        "gpt": {
            "sys": "你是顶级量化博彩真神。用纯数学方法测算欧赔与隐含xG。抛弃预设比分思维，该大分大分，该冷门冷门。只输出纯JSON数组。",
            "temp": 0.15
        },
        "gemini": {
            "sys": "你是概率建模引擎。利用泊松极限验证进球天花板。彻底独立思考。只输出纯JSON数组。",
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
                payload = {"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"temperature":profile["temp"]},"systemInstruction":{"parts":[{"text":profile["sys"]}]}}
            else:
                headers["Authorization"] = f"Bearer {key}"
                bp = {"model":mn,"messages":[{"role":"system","content":profile["sys"]},{"role":"user","content":prompt}]}
                if ai_name != "claude": bp["temperature"] = profile["temp"]
                payload = bp
            gw = url.split("/v1")[0][:35]
            print(f"  [⏳{timeout_sec}s] {ai_name.upper()} | {mn[:22]} @ {gw}")
            t0 = time.time()
            try:
                # 【严格保留】Timeout 配置一字未动
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec, connect=15)) as r:
                    elapsed = round(time.time()-t0,1)
                    if r.status == 200:
                        try:
                            data = await r.json(content_type=None)
                        except:
                            print(f"    ⚠️ 响应非JSON | {elapsed}s → 换URL")
                            continue
                            
                        try:
                            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip() if is_gem else data["choices"][0]["message"]["content"].strip()
                        except:
                            print(f"    ⚠️ 结构缺失 | {elapsed}s → 换URL")
                            continue
                            
                        # 剥离 <thinking> 标签，防止 JSON 提取爆炸
                        clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
                        clean = re.sub(r"[`]{3}(?:json)?", "", clean).strip()
                        
                        start=clean.find("["); end=clean.rfind("]")+1
                        if start==-1 or end==0:
                            clean = re.sub(r"[^\[\]{}:,\"'0-9a-zA-Z\u4e00-\u9fa5\s\.\-\+\(\)]","",clean)
                            start=clean.find("["); end=clean.rfind("]")+1
                            
                        results = {}
                        if start != -1 and end > start:
                            json_str = clean[start:end]
                            arr = []
                            try:
                                arr = json.loads(json_str)
                            except json.JSONDecodeError:
                                # 🚀 升级版军工级 JSON 断肢重生：后向寻址安全闭合算法
                                try:
                                    last_brace_idx = json_str.rfind('}')
                                    if last_brace_idx != -1:
                                        safe_json_str = json_str[:last_brace_idx+1] + "]"
                                        arr = json.loads(safe_json_str)
                                        print(f"    🩹 触发军工级断肢重生，精准抢救回 {len(arr)} 条数据！")
                                except: pass
                                
                            if isinstance(arr, list):
                                for item in arr:
                                    if not item.get("match"): continue
                                    try: mid = int(item["match"])
                                    except: mid = item["match"]
                                    if item.get("top3"):
                                        t1_score = item["top3"][0].get("score","1-1") if item["top3"] else "1-1"
                                        results[mid] = {
                                            "top3": item["top3"],
                                            "ai_score": t1_score,
                                            "reason": str(item.get("reason",""))[:200],
                                            "ai_confidence": int(item.get("ai_confidence",60)),
                                        }
                                    elif item.get("score"):
                                        results[mid] = {
                                            "ai_score": item["score"],
                                            "analysis": str(item.get("reason",""))[:200],
                                            "ai_confidence": int(item.get("ai_confidence",60)),
                                            "value_kill": bool(item.get("value_kill",False)),
                                        }
                                        
                        if len(results) >= max(1, num_matches*0.5):
                            print(f"    ✅ {ai_name.upper()} 成功: {len(results)}/{num_matches} | {elapsed}s ({mn[:20]})")
                            return ai_name, results, mn
                            
                        if len(results) > len(best_results): 
                            best_results=results; best_model=mn
                            print(f"    ⚠️ 部分 {len(results)}/{num_matches} | {elapsed}s")
                            
                        # 拿到响应但解析不够，直接跳过模型防死循环扣费
                        skip_model = True
                        break
                        
                    elif r.status == 429: print(f"    🔥 429 | {elapsed}s"); await asyncio.sleep(3); continue
                    elif r.status >= 500: print(f"    💀 HTTP {r.status} | {elapsed}s → 跳模型"); skip_model=True; break
                    elif r.status == 400: print(f"    💀 400 | {elapsed}s → 跳模型"); skip_model=True; break
                    else: print(f"    ⚠️ HTTP {r.status} | {elapsed}s")
            except asyncio.TimeoutError:
                elapsed=round(time.time()-t0,1); print(f"    ⏰ {elapsed}s超时 → 跳模型"); skip_model=True; break
            except Exception as e:
                elapsed=round(time.time()-t0,1); err=str(e)[:40]
                if "connect" in err.lower() or "resolve" in err.lower(): print(f"    ⚠️ 连接失败 {err} | {elapsed}s → 换URL")
                else: print(f"    ⚠️ {err} | {elapsed}s → 跳模型"); skip_model=True; break
            await asyncio.sleep(0.3)
            
        if len(best_results) >= max(1, num_matches*0.4):
            print(f"    ✅ {ai_name.upper()} 采用: {len(best_results)}/{num_matches}"); return ai_name, best_results, best_model
            
    if best_results:
        print(f"    ⚠️ {ai_name.upper()} 勉强采用: {len(best_results)}条"); return ai_name, best_results, best_model
    print(f"    ❌ {ai_name.upper()} 全部失败"); return ai_name, {}, "failed"

async def run_ai_matrix_two_phase(match_analyses):
    """两阶段：Phase1(先锋拓荒) → Phase2(裁判仲裁)"""
    num = len(match_analyses)

    # ===== Phase1: 三家并行独立分析 =====
    p1_prompt = build_phase1_prompt(match_analyses)
    print(f"  [Phase1] {len(p1_prompt):,} 字符 → GPT/Grok/Gemini 独立推演中...")

    p1_configs = [
        ("grok","GROK_API_URL","GROK_API_KEY",["熊猫-A-6-grok-4.2-thinking","熊猫-A-7-grok-4.2-多智能体讨论"]),
        ("gpt","GPT_API_URL","GPT_API_KEY",["熊猫-A-10-gpt-5.4","熊猫-按量-gpt-5.4"]),
        ("gemini","GEMINI_API_URL","GEMINI_API_KEY",["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking","熊猫-顶级特供-X-17-gemini-3.1-pro-preview"]),
    ]
    p1_results = {"gpt":{},"grok":{},"gemini":{}}

    async with aiohttp.ClientSession() as session:
        tasks = [async_call_one_ai_batch(session,p1_prompt,u,k,m,num,n) for n,u,k,m in p1_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    for res in results:
        if isinstance(res,tuple): n,d,_ = res; p1_results[n] = d
        else: print(f"  [Phase1 ERROR] {res}")

    ok = sum(1 for v in p1_results.values() if v)
    print(f"  [Phase1] 拓荒完成: {ok}/3 AI有效")

    # ===== Phase2: Claude死刑执行官 =====
    p2_prompt = build_phase2_prompt(match_analyses, p1_results)
    print(f"  [Phase2] {len(p2_prompt):,} 字符 → Claude 绝对仲裁...")

    claude_r = {}
    async with aiohttp.ClientSession() as session:
        _,claude_r,_ = await async_call_one_ai_batch(
            session, p2_prompt, "CLAUDE_API_URL","CLAUDE_API_KEY",
            ["熊猫特供-超纯满血-99额度-claude-opus-4.6-thinking","熊猫-按量-特供顶级-官方正向满血-claude-opus-4.6-thinking"],
            num, "claude"
        )

    all_r = p1_results.copy()
    all_r["claude"] = claude_r
    return all_r

# ====================================================================
# 多市场 EV
# ====================================================================
def calculate_multi_market_value(engine_result, match_obj):
    hp = engine_result.get("home_prob", 33)
    dp = engine_result.get("draw_prob", 33)
    ap = engine_result.get("away_prob", 34)
    btts = engine_result.get("btts", 48)
    ou = engine_result.get("over_25", 52)
    markets = {
        "1X2_home": calculate_value_bet(hp, float(match_obj.get("sp_home", 0) or 0)),
        "1X2_draw": calculate_value_bet(dp, float(match_obj.get("sp_draw", 0) or 0)),
        "1X2_away": calculate_value_bet(ap, float(match_obj.get("sp_away", 0) or 0)),
        "BTTS_yes": calculate_value_bet(btts, 1.90),
        "Over_2.5": calculate_value_bet(ou, 1.90),
    }
    best = max(markets.items(), key=lambda x: x[1]["ev"])
    return {
        "markets": markets,
        "best_value": {"market": best[0], **best[1]},
        "suggested_units": round(max(0.5, best[1]["kelly"] * 4), 1)
    }

# ====================================================================
# Merge v8.6 — 完全去势版：包含前端必需字段，彻底治愈黑屏
# ====================================================================
def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    engine_score = engine_result.get("primary_score", "1-1")
    engine_conf = engine_result.get("confidence", 60)
    value_info = calculate_multi_market_value(engine_result, match_obj)
    
    # 🚀 完全解放算力：最终比分直接等于 Phase 2 最高权重模型 (Claude) 的审判
    final_score = ""
    if isinstance(claude_r, dict) and claude_r.get("ai_score") and claude_r.get("ai_score") != "-":
        final_score = claude_r["ai_score"]
        
    # 如果 Claude 彻底宕机，采纳 Phase 1 阶段最高票共识
    p1_ai = {"gpt": gpt_r, "grok": grok_r, "gemini": gemini_r}
    if not final_score:
        vote_count = {}
        for n, r in p1_ai.items():
            if isinstance(r, dict):
                t3 = r.get("top3", [])
                if t3 and isinstance(t3, list) and len(t3) > 0:
                    sc = t3[0].get("score")
                    if sc: vote_count[sc] = vote_count.get(sc, 0) + 1
        if vote_count:
            final_score = max(vote_count, key=vote_count.get)
        else:
            final_score = engine_score # 终极断网灾备

    # 提取各AI的比分和文本，为了前端渲染使用（非常重要，不写会导致页面崩溃）
    gpt_sc = gpt_r.get("top3", [{"score":"-"}])[0].get("score", "-") if isinstance(gpt_r, dict) and gpt_r.get("top3") else gpt_r.get("ai_score", "-") if isinstance(gpt_r, dict) else "-"
    gpt_an = gpt_r.get("reason", gpt_r.get("analysis", "N/A")) if isinstance(gpt_r, dict) else "N/A"

    grok_sc = grok_r.get("top3", [{"score":"-"}])[0].get("score", "-") if isinstance(grok_r, dict) and grok_r.get("top3") else grok_r.get("ai_score", "-") if isinstance(grok_r, dict) else "-"
    grok_an = grok_r.get("reason", grok_r.get("analysis", "N/A")) if isinstance(grok_r, dict) else "N/A"

    gem_sc = gemini_r.get("top3", [{"score":"-"}])[0].get("score", "-") if isinstance(gemini_r, dict) and gemini_r.get("top3") else gemini_r.get("ai_score", "-") if isinstance(gemini_r, dict) else "-"
    gem_an = gemini_r.get("reason", gemini_r.get("analysis", "N/A")) if isinstance(gemini_r, dict) else "N/A"

    cl_sc = final_score
    cl_an = claude_r.get("analysis", claude_r.get("reason", "等待深层演算...")) if isinstance(claude_r, dict) else engine_result.get("reason", "odds engine")

    # 抽取冷门预警红点，但不干预比分
    pre_pred = {"home_win_pct": engine_result.get("home_prob", 33), "draw_pct": engine_result.get("draw_prob", 33), "away_win_pct": engine_result.get("away_prob", 34), "smart_signals": stats.get("smart_signals", [])}
    cold_door = ColdDoorDetector.detect(match_obj, pre_pred)
    
    # 信心融合计算
    ai_all = {"claude": claude_r, "grok": grok_r, "gpt": gpt_r, "gemini": gemini_r}
    ai_conf_sum = 0; ai_conf_count = 0; value_kills = 0
    weights = {"claude": 1.5, "grok": 1.3, "gpt": 1.1, "gemini": 1.0}
    ai_scores_list = []
    
    for name, r in ai_all.items():
        if not isinstance(r, dict): continue
        sc = r.get("ai_score", r.get("top3", [{"score":"-"}])[0].get("score", "-") if r.get("top3") else "-")
        if sc and sc not in ["-", "?", ""]:
            ai_scores_list.append(sc)
        conf = r.get("ai_confidence", 60)
        ai_conf_sum += conf * weights.get(name, 1.0)
        ai_conf_count += weights.get(name, 1.0)
        if r.get("value_kill"): value_kills += 1

    avg_ai_conf = (ai_conf_sum / ai_conf_count) if ai_conf_count > 0 else 60
    cf = engine_conf
    cf = min(95, cf + int((avg_ai_conf - 60) * 0.4))
    cf = cf + value_kills * 6
    sigs = list(stats.get("smart_signals", []))
    if cold_door["is_cold_door"]: sigs.extend(cold_door["signals"]); cf = max(30, cf - 5)
    has_warn = any("🚨" in str(s) for s in sigs)
    if has_warn: cf = max(35, cf - 12)
    risk = "低" if cf >= 75 else ("中" if cf >= 55 else "高")

    # 返回极度丰富的全字段数据字典，防备前端 JS 出现 undefined 崩溃！
    return {
        "predicted_score": final_score,
        "home_win_pct": engine_result.get("home_prob", 33),
        "draw_pct": engine_result.get("draw_prob", 33),
        "away_win_pct": engine_result.get("away_prob", 34),
        "confidence": cf,
        "risk_level": risk,
        "over_under_2_5": "大" if engine_result.get("over_25", 50) > 55 else "小",
        "both_score": "是" if engine_result.get("btts", 45) > 50 else "否",
        "multi_market_value": value_info,
        "best_value_market": value_info["best_value"]["market"],
        "suggested_units": value_info["suggested_units"],
        "cold_door": cold_door,
        
        # 前端展示必备：AI 各家比分与文本分析
        "gpt_score": gpt_sc,
        "gpt_analysis": gpt_an,
        "grok_score": grok_sc,
        "grok_analysis": grok_an,
        "gemini_score": gem_sc,
        "gemini_analysis": gem_an,
        "claude_score": cl_sc,
        "claude_analysis": cl_an,
        "ai_avg_confidence": round(avg_ai_conf, 1),
        "value_kill_count": value_kills,
        "model_agreement": len(set(ai_scores_list)) <= 1 and len(ai_scores_list) >= 2,
        
        # 前端渲染雷达图、量化数据展示框必备的底层字段
        "poisson": stats.get("poisson", {}),
        "refined_poisson": stats.get("refined_poisson", {}),
        "extreme_warning": engine_result.get("scissors_gap_signal", ""),
        "smart_money_signal": " | ".join(sigs),
        "smart_signals": sigs,
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
        "bookmaker_implied_away_xg": engine_result.get("bookmaker_implied_away_xg", "?"),
    }

def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})
        s = pr.get("confidence", 0) * 0.5 + pr.get("suggested_units", 0) * 4
        if pr.get("cold_door", {}).get("is_cold_door"):
            s -= 10
        p["recommend_score"] = round(s, 2)
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]

def extract_num(ms):
    wm = {"一":1000,"二":2000,"三":3000,"四":4000,"五":5000,"六":6000,"日":7000,"天":7000}
    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999

# ====================================================================
# run_predictions v8.6 Pro
# ====================================================================
def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 100)
    print(f"  [GROK-FUSED v8.6 Pro] 纯净无界算力版 | 修复前端白屏兼容补丁 | {len(ms)} 场")
    print("=" * 100)

    match_analyses = []
    for i, m in enumerate(ms):
        eng = predict_match(m)
        league_info, _, _, _ = build_league_intelligence(m)
        sp = ensemble.predict(m, {})
        exp_result = exp_engine.analyze(m)
        match_analyses.append({"match": m, "engine": eng, "stats": sp, "index": i+1, "experience": exp_result})

    all_ai = {"claude": {}, "gemini": {}, "gpt": {}, "grok": {}}
    if use_ai and match_analyses:
        print("  [SYSTEM] AI 解放协议已启动，彻底切除本地比分干预机制。")
        start_t = time.time()
        all_ai = asyncio.run(run_ai_matrix_two_phase(match_analyses))
        print(f"  [SYSTEM] 算力矩阵收敛完成，耗时 {time.time()-start_t:.1f}s")

    res = []
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        mg = merge_result(ma["engine"], all_ai["gpt"].get(i+1, {}), all_ai["grok"].get(i+1, {}), all_ai["gemini"].get(i+1, {}), all_ai["claude"].get(i+1, {}), ma["stats"], m)
        
        try: mg = apply_experience_to_prediction(m, mg, exp_engine)
        except Exception as e: print(f"    ⚠️ experience跳过: {e}")
        try: mg = apply_odds_history(m, mg)
        except Exception as e: print(f"    ⚠️ odds_history跳过: {e}")
        try: mg = apply_quant_edge(m, mg)
        except Exception as e: print(f"    ⚠️ quant_edge跳过: {e}")
        try: mg = apply_wencai_intel(m, mg)
        except Exception as e: print(f"    ⚠️ wencai_intel跳过: {e}")
        try: mg = upgrade_ensemble_predict(m, mg)
        except Exception as e: print(f"    ⚠️ advanced_models跳过: {e}")

        # 根据绝对无界的 AI 比分反推最终比赛胜平负判定
        score_str = mg.get("predicted_score", "1-1")
        try:
            sh, sa = map(int, score_str.split("-"))
            mg["result"] = "主胜" if sh > sa else "客胜" if sh < sa else "平局"
        except:
            pcts = {"主胜": mg["home_win_pct"], "平局": mg["draw_pct"], "客胜": mg["away_win_pct"]}
            mg["result"] = max(pcts, key=pcts.get)

        res.append({**m, "prediction": mg})

        cold = mg.get("cold_door", {})
        cold_tag = f" [❄️{cold.get('level','')}冷门预警]" if cold.get("is_cold_door") else ""
        print(f"  [{i+1}] {m.get('home_team')} vs {m.get('away_team')} => {mg['result']} ({mg['predicted_score']}) | CF: {mg['confidence']}% | EV星级: {mg.get('suggested_units',0)}{cold_tag}")

    t4 = select_top4(res)
    t4ids = [t.get("id") for t in t4]
    for r in res:
        r["is_recommended"] = r.get("id") in t4ids

    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    diary = load_ai_diary()
    cold_count = len([r for r in res if r.get("prediction", {}).get("cold_door", {}).get("is_cold_door")])
    diary["yesterday_win_rate"] = f"{len([r for r in res if r['prediction']['confidence'] > 70])}/{max(1, len(res))}"
    diary["reflection"] = f"v8.6 Pro | 修复前端白屏兼容Bug | 100%全权移交AI算力推演"
    save_ai_diary(diary)

    return res, t4
