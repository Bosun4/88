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
# 冷门猎手引擎 (客观信号提取，无比分干预)
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
    return {"yesterday_win_rate": "N/A", "reflection": "彻底剥离本地比分枷锁，双阶段死刑判定系统就位", "kill_history": []}

def save_ai_diary(diary):
    os.makedirs("data", exist_ok=True)
    with open("data/ai_diary.json", "w", encoding="utf-8") as f:
        json.dump(diary, f, ensure_ascii=False, indent=2)

# ====================================================================
# Phase1 & Phase2 Prompt（v8.2 Pro 纯净独立算力版）
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
            if ai_data.get("top3"):
                scores_str = " | ".join(f"{t.get('score','?')}({t.get('prob','?')}%)" for t in ai_data.get("top3", [])[:3])
                p += f"  先锋 {ai_name.upper()} 独立研判: {scores_str}\n"
        p += "\n"
    p += "开始执行最终死刑判决！把散户往死里杀！只输出JSON数组！"
    return p

# ====================================================================
# AI调用引擎（完美融合JSON断肢重生与去思考标签技术，时间设置一字未动）
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
    if not key:
        return ai_name, {}, "no_key"
    primary_url = get_clean_env_url(url_env)
    backup = [u for u in FALLBACK_URLS if u and u != primary_url][:2]
    urls = [primary_url] + backup
    
    # 严格按照你指令保留的原版超时时间设定
    timeout_map = {"claude": 500, "grok": 300, "gpt": 300, "gemini": 250}
    timeout_sec = timeout_map.get(ai_name, 200)
    
    best_results = {}
    best_model = ""
    for mn in models_list:
        skip_model = False
        for base_url in urls:
            if not base_url or skip_model:
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
                    "generationConfig": {"temperature": 0.15},
                    "systemInstruction": {"parts": [{"text": "你是顶级量化足球分析师，完全抛弃人工预设框架，独立运算。只输出JSON数组。"}]}
                }
            else:
                headers["Authorization"] = f"Bearer {key}"
                payload = {
                    "model": mn,
                    "messages": [
                        {"role": "system", "content": "你是顶级吸血操盘手。抛弃所有固有常规比分思维，根据客观数据独立得出最残忍结论。只输出JSON数组。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.18
                }
            gw = url.split("/v1")[0][:35]
            print(f"  [⏳{timeout_sec}s] {ai_name.upper()} | {mn[:22]} @ {gw}")
            t0 = time.time()
            try:
                # 严格使用传入的 timeout 参数设置
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec, connect=15)) as r:
                    elapsed = round(time.time() - t0, 1)
                    if r.status == 200:
                        # 防暴毙机制：非JSON拦截
                        try:
                            data = await r.json(content_type=None)
                        except:
                            print(f"    ⚠️ 响应体非JSON | {elapsed}s → 换URL")
                            continue
                            
                        # 取出数据
                        try:
                            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip() if is_gem else data["choices"][0]["message"]["content"].strip()
                        except:
                            print(f"    ⚠️ API结构缺失 | {elapsed}s → 换URL")
                            continue
                            
                        # 🔥 核心防灾优化与 UI Bug 规避：剥离 <thinking> 标签，并安全提取 JSON
                        clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
                        # 使用正则类集匹配三个反引号，避免触发前端界面的 Markdown 误伤截断
                        clean = re.sub(r"[`]{3}(?:json)?", "", clean).strip()
                        
                        start = clean.find("[")
                        end = clean.rfind("]") + 1
                        if start == -1 or end == 0:
                            clean = re.sub(r"[^\[\]{}:,\"'0-9a-zA-Z\u4e00-\u9fa5\s\.\-\+\(\)]", "", clean)
                            start = clean.find("[")
                            end = clean.rfind("]") + 1
                            
                        results = {}
                        if start != -1 and end > start:
                            json_str = clean[start:end]
                            arr = []
                            try:
                                arr = json.loads(json_str)
                            except json.JSONDecodeError:
                                # 万字长文防截断抢救
                                try:
                                    if not json_str.endswith("]"): json_str += "]"
                                    if not json_str.endswith("}]"): json_str = json_str[:-1] + "}]"
                                    arr = json.loads(json_str)
                                    print(f"    🩹 启动断肢重生，成功抢救 {len(arr)} 条截断数据！")
                                except: pass

                            if isinstance(arr, list):
                                for item in arr:
                                    if isinstance(item, dict) and item.get("match"):
                                        try: mid = int(item["match"])
                                        except: mid = item["match"]
                                        # 兼容 Phase 1(top3) 和 Phase 2(score) 两种格式
                                        results[mid] = {
                                            "ai_score": item.get("score", "-"),
                                            "top3": item.get("top3", []),
                                            "analysis": str(item.get("reason", "")).strip()[:200],
                                            "ai_confidence": int(item.get("ai_confidence", 60)),
                                            "value_kill": bool(item.get("value_kill", False)),
                                            "suggested_units": float(item.get("suggested_units", 0)),
                                            "dark_verdict": str(item.get("dark_verdict", ""))
                                        }
                                        
                        if len(results) >= max(1, num_matches * 0.5):
                            print(f"    ✅ {ai_name.upper()} 成功: {len(results)}/{num_matches} | {elapsed}s")
                            return ai_name, results, mn
                        if len(results) > len(best_results):
                            best_results = results
                            best_model = mn
                            print(f"    ⚠️ 部分 {len(results)}/{num_matches} | {elapsed}s")
                            
                        # 防止已经拿到响应但解析失败导致无限重复扣费
                        skip_model = True
                        break
                        
                    elif r.status == 429: print(f"    🔥 429限流 | {elapsed}s"); await asyncio.sleep(3); continue
                    elif r.status >= 500: print(f"    💀 HTTP {r.status} | {elapsed}s → 跳模型"); skip_model = True; break
                    elif r.status == 400: print(f"    💀 HTTP 400 | {elapsed}s → 跳模型"); skip_model = True; break
                    else: print(f"    ⚠️ HTTP {r.status} | {elapsed}s")
            except asyncio.TimeoutError:
                print(f"    ⏰ {round(time.time()-t0,1)}s超时 → 跳模型")
                skip_model = True
                break
            except Exception as e:
                # 网络异常直接换 URL，不干掉模型
                print(f"    ⚠️ 异常 {str(e)[:40]} → 换URL")
                continue
            await asyncio.sleep(0.3)
    return ai_name, best_results, best_model

async def run_ai_matrix_two_phase(match_analyses):
    num = len(match_analyses)
    p1_prompt = build_phase1_prompt(match_analyses)
    p1_results = {"gpt": {}, "grok": {}, "gemini": {}}
    async with aiohttp.ClientSession() as session:
        tasks = [
            async_call_one_ai_batch(session, p1_prompt, "GPT_API_URL", "GPT_API_KEY", ["熊猫-按量-gpt-5.4","熊猫-A-10-gpt-5.3-codex"], num, "gpt"),
            async_call_one_ai_batch(session, p1_prompt, "GROK_API_URL", "GROK_API_KEY", ["熊猫-A-6-grok-4.2-thinking","熊猫-A-7-grok-4.2-多智能体讨论"], num, "grok"),
            async_call_one_ai_batch(session, p1_prompt, "GEMINI_API_URL", "GEMINI_API_KEY", ["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking","熊猫-顶级特供-X-17-gemini-3.1-pro-preview"], num, "gemini")
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, tuple):
                n, d, _ = res
                p1_results[n] = d
    p2_prompt = build_phase2_prompt(match_analyses, p1_results)
    claude_r = {}
    async with aiohttp.ClientSession() as session:
        _, claude_r, _ = await async_call_one_ai_batch(
            session, p2_prompt, "CLAUDE_API_URL", "CLAUDE_API_KEY",
            ["熊猫-按量-满血copilot-claude-opus-4.6-thinking", "熊猫-按量-特供顶级-官方正向满血-claude-opus-4.6-thinking"],
            num, "claude"
        )
    all_r = p1_results.copy()
    all_r["claude"] = claude_r
    return all_r

# ====================================================================
# 多市场 EV 与 彻底解放算力的 Merge
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

def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    # 本地 engine_score 降级为断网时的极端灾备，绝不干预正常逻辑
    engine_score = engine_result.get("primary_score", "1-1")
    engine_conf = engine_result.get("confidence", 60)
    value_info = calculate_multi_market_value(engine_result, match_obj)
    
    # 🚀 绝对算力觉醒：完全抛弃本地的强制拦截
    # 唯一裁决者是 Phase 2 阶段拥有最高权限的死刑执行官（Claude）
    final_score = ""
    if isinstance(claude_r, dict) and claude_r.get("ai_score") and claude_r.get("ai_score") != "-":
        final_score = claude_r["ai_score"]
    
    # 无网灾备：如果最强仲裁模型挂了，才降级使用 Phase 1 先锋AI的共识
    if not final_score:
        p1_ai = {"gpt": gpt_r, "grok": grok_r, "gemini": gemini_r}
        vote_count = {}
        for n, r in p1_ai.items():
            if isinstance(r, dict):
                # 从 top3 中取最可能的第一顺位比分
                t3 = r.get("top3", [])
                if t3 and isinstance(t3, list) and len(t3) > 0:
                    sc = t3[0].get("score")
                    if sc: vote_count[sc] = vote_count.get(sc, 0) + 1
        if vote_count:
            final_score = max(vote_count, key=vote_count.get)
        else:
            final_score = engine_score # 终极无网灾备

    # 冷门探测只作为前端红色标签预警，绝不干预最终比分
    pre_pred = {"home_win_pct": engine_result.get("home_prob", 33), "draw_pct": engine_result.get("draw_prob", 33), "away_win_pct": engine_result.get("away_prob", 34), "smart_signals": stats.get("smart_signals", [])}
    cold_door = ColdDoorDetector.detect(match_obj, pre_pred)

    return {
        "predicted_score": final_score,
        "home_win_pct": engine_result.get("home_prob", 33),
        "draw_pct": engine_result.get("draw_prob", 33),
        "away_win_pct": engine_result.get("away_prob", 34),
        "confidence": engine_conf,
        "risk_level": "低" if engine_conf >= 75 else ("中" if engine_conf >= 55 else "高"),
        "over_under_2_5": "大" if engine_result.get("over_25", 50) > 55 else "小",
        "both_score": "是" if engine_result.get("btts", 45) > 50 else "否",
        "multi_market_value": value_info,
        "best_value_market": value_info["best_value"]["market"],
        "suggested_units": value_info["suggested_units"],
        "cold_door": cold_door,
        "gpt_score": gpt_r.get("top3", [{"score":"-"}])[0].get("score", "-") if isinstance(gpt_r, dict) and gpt_r.get("top3") else "-",
        "grok_score": grok_r.get("top3", [{"score":"-"}])[0].get("score", "-") if isinstance(grok_r, dict) and grok_r.get("top3") else "-",
        "gemini_score": gemini_r.get("top3", [{"score":"-"}])[0].get("score", "-") if isinstance(gemini_r, dict) and gemini_r.get("top3") else "-",
        "claude_score": final_score,
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
# run_predictions v8.2 Pro - 纯净算力解放版
# ====================================================================
def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 100)
    print(f"  [GROK-FUSED v8.2 Pro] 纯净算力解放版 | 双阶段无干预推演 | {len(ms)} 场")
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
        print("  [TWO-PHASE] 启动双阶段纯净AI矩阵（剥离一切人工干预）...")
        start_t = time.time()
        all_ai = asyncio.run(run_ai_matrix_two_phase(match_analyses))
        print(f"  [AI MATRIX] 算力萃取完成，耗时 {time.time()-start_t:.1f}s")

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

        # 核心逻辑替换：根据绝对独立的 AI 仲裁比分反推胜平负，取代原版的本地概率模型
        score_str = mg.get("predicted_score", "1-1")
        try:
            sh, sa = map(int, score_str.split("-"))
            mg["result"] = "主胜" if sh > sa else "客胜" if sh < sa else "平局"
        except:
            pcts = {"主胜": mg["home_win_pct"], "平局": mg["draw_pct"], "客胜": mg["away_win_pct"]}
            mg["result"] = max(pcts, key=pcts.get)

        res.append({**m, "prediction": mg})

        cold = mg.get("cold_door", {})
        cold_tag = f" [❄️{cold.get('level','')}冷门]" if cold.get("is_cold_door") else ""
        print(f"  [{i+1}] {m.get('home_team')} vs {m.get('away_team')} => {mg['result']} ({mg['predicted_score']}) | CF: {mg['confidence']}% | Units: {mg.get('suggested_units',0)}{cold_tag}")

    t4 = select_top4(res)
    t4ids = [t.get("id") for t in t4]
    for r in res:
        r["is_recommended"] = r.get("id") in t4ids

    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    diary = load_ai_diary()
    cold_count = len([r for r in res if r.get("prediction", {}).get("cold_door", {}).get("is_cold_door")])
    diary["yesterday_win_rate"] = f"{len([r for r in res if r['prediction']['confidence'] > 70])}/{max(1, len(res))}"
    diary["reflection"] = f"v8.2 Pro | 彻底粉碎比分干预枷锁 | 防大输出截断搭载完毕 | 100%全权移交AI算力"
    save_ai_diary(diary)

    return res, t4
