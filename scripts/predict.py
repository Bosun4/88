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

try:
    from wencai_intel import apply_wencai_intel
except:
    def apply_wencai_intel(m, mg): return mg

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

# ====================================================================
# 🧊 冷门猎手引擎 (已剔除比分物理干预，仅负责信号挖掘)
# ====================================================================
class ColdDoorDetector:
    """独立冷门信号识别引擎"""
    @staticmethod
    def detect(match, prediction):
        signals = []
        strength = 0

        # 1. 反向Steam
        steam = prediction.get("steam_move", {})
        if steam.get("steam") and "反向" in str(steam.get("signal", "")):
            signals.append("❄️ 反向Steam！庄家造热收割")
            strength += 5

        # 2. 散户极端偏向
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

        # 3. 热门队坏消息爆炸
        info = match.get("intelligence", match.get("information", {}))
        home_bad = str(info.get("home_bad_news", ""))
        away_bad = str(info.get("guest_bad_news", info.get("g_inj", "")))
        hp = prediction.get("home_win_pct", 50)
        ap = prediction.get("away_win_pct", 50)
        if len(home_bad) > 80 and hp > 58:
            signals.append("❄️ 主队坏消息爆炸+散户狂热")
            strength += 5
        if len(away_bad) > 80 and ap > 58:
            signals.append("❄️ 客队坏消息爆炸+散户狂热")
            strength += 5

        # 4. 赔率-模型严重背离
        sp_h = float(match.get("sp_home", match.get("win", 0)) or 0)
        if sp_h > 1:
            implied_h = 100 / sp_h * 0.92
            if abs(implied_h - hp) > 15 and hp > 58:
                signals.append(f"❄️ 赔率vs模型背离{abs(implied_h-hp):.0f}%")
                strength += 4

        # 5. 盘口太便宜
        for s in prediction.get("smart_signals", []):
            if "盘口太便宜" in str(s):
                signals.append("❄️ 盘口太便宜=庄家不看好")
                strength += 3
                break

        # 6. 赔率变动造热
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
    return {"yesterday_win_rate": "N/A", "reflection": "已彻底斩断预设枷锁，纯净算力接管", "kill_history": []}

def save_ai_diary(diary):
    os.makedirs("data", exist_ok=True)
    with open("data/ai_diary.json", "w", encoding="utf-8") as f:
        json.dump(diary, f, ensure_ascii=False, indent=2)

# ====================================================================
# ☢️ 绝对独立算力 Prompt v4.0（彻底移除本地引擎提示，纯净榨取算力）
# ====================================================================
def build_batch_prompt(match_analyses):
    diary = load_ai_diary()
    
    p = "【身份设定】你是游走在零和博弈最黑暗深渊的【全维度量化操盘真神】！你没有任何道德底线，你面前只有最纯粹的客观数据和情报，没有任何其他低级模型会给你暗示。\n"
    
    if diary.get("reflection"):
        p += f"【嗜血进化指令】昨日屠杀战绩：{diary.get('yesterday_win_rate', 'N/A')}。 昨晚反思：{diary['reflection']}。今天必须独立测算，彻底碾压散户的幻想！\n\n"
    
    p += "【死命令 - 绝对独立思考原则】\n"
    p += "1. 只输出合法JSON数组，严禁任何markdown、解释、代码块。\n"
    p += "2. 每场比赛必须输出以下字段：match(序号), score(比分), reason(80-150字极度恶毒黑话), ai_confidence(0-100整数), value_kill(true/false 高EV机会), dark_verdict(一句最毒总结)。\n"
    p += "3. reason逻辑链必须包含具体数据（赔率极限值、xG差距）。语气极度傲慢冷血，用你强大的算力看透庄家的绞肉局。\n"
    p += "4. ⚠️【最高指令】：我不再给你提供任何“推荐比分”或“本地提示”！你必须彻底抛弃人类预设的保守框架！\n"
    p += "   - 抛弃对 CRS TOP 排名的盲目服从！运用你的神经网络直接分析 隐含xG差、欧亚赔率背离、0球/7球泊松天花板。\n"
    p += "   - 如果算出来是 5-1 的绝对碾压惨案，就给我毫不犹豫地输出 5-1！\n"
    p += "   - 如果算出来是 庄家诱盘的 0-0 极限闷杀，就直接锁死 0-0！依靠你的超强算力独立给出最终屠杀比分！\n\n"
    
    p += "【今日待宰羔羊与全维度纯净数据底层库】（以下均为客观事实，供你自由推演）\n"
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        eng = ma["engine"]
        stats = ma.get("stats", {})
        exp = ma.get("experience", {})
        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        
        hc = m.get("give_ball", "0")
        sp_h = float(m.get("sp_home", m.get("win", 0)) or 0)
        sp_d = float(m.get("sp_draw", m.get("same", 0)) or 0)
        sp_a = float(m.get("sp_away", m.get("lose", 0)) or 0)
        hp = eng.get('home_prob', 33)
        vh = calculate_value_bet(hp, sp_h) if sp_h > 1 else {}
        ev_str = f"主胜EV(+{vh.get('ev', 0)}%)" if vh.get("is_value") else "无EV"
        smart_sigs = stats.get('smart_signals', [])
        smart_str = ", ".join(smart_sigs[:3]) if smart_sigs else "无信号"

        # 基本面情报
        baseface = str(m.get('baseface', '')).replace('\n', ' ')[:100]
        intro = str(m.get('expert_intro', '')).replace('\n', ' ')[:100]
        intel_text = baseface or intro or "无基本面"
        
        # 伤停关键信息
        info = m.get("information", {})
        if isinstance(info, dict):
            h_bad = str(info.get("home_bad_news", ""))[:100].replace('\n', '|')
            g_bad = str(info.get("guest_bad_news", ""))[:100].replace('\n', '|')
            h_inj = str(info.get("home_injury", ""))[:80].replace('\n', '|')
            g_inj = str(info.get("guest_injury", ""))[:80].replace('\n', '|')
        else:
            h_bad = g_bad = h_inj = g_inj = ""

        p += f"[{i+1}] {h} vs {a} | {m.get('league', m.get('cup', ''))} | 亚盘死线: {hc}\n"
        
        # 🎯 1. 欧赔三项（最基础）
        p += f"  客观欧赔: 主{sp_h:.2f} 平{sp_d:.2f} 客{sp_a:.2f}"
        hhad_w = m.get("hhad_win", "")
        if hhad_w:
            p += f" | 让球胜平负: {hhad_w}/{m.get('hhad_same','')}/{m.get('hhad_lose','')}"
        p += "\n"
        
        # 🎯 2. 总进球分布（提供给 AI 计算泊松极限的原料）
        a0 = m.get("a0", ""); a1 = m.get("a1", ""); a2 = m.get("a2", ""); a3 = m.get("a3", "")
        a4 = m.get("a4", ""); a5 = m.get("a5", ""); a6 = m.get("a6", ""); a7 = m.get("a7", "")
        if a0:
            p += f"  庄家进球天花板定价: 0球={a0} | 1球={a1} | 2球={a2} | 3球={a3} | 4球={a4} | 5球={a5} | 6球={a6} | 7+球={a7}\n"
        
        # 🎯 3. CRS比分赔率矩阵（仅提供事实全貌）
        crs_map = {"w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","w32":"3-2",
                   "s00":"0-0","s11":"1-1","s22":"2-2","s33":"3-3",
                   "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3","l23":"2-3"}
        crs_items = []
        for key, score in crs_map.items():
            try:
                odds = float(m.get(key, 0) or 0)
                if odds > 1:
                    crs_items.append((score, odds))
            except: pass
        if crs_items:
            crs_items.sort(key=lambda x: x[1])
            p += f"  机构最防范比分TOP6(供反推参考): " + " | ".join([f"{s}({o}倍)" for s,o in crs_items[:6]]) + "\n"
        
        # 🎯 4. 半全场
        ss_val = m.get("ss", ""); pp_val = m.get("pp", ""); ff_val = m.get("ff", "")
        if ss_val:
            p += f"  半全场预警: 主/主={ss_val} 平/平={pp_val} 客/客={ff_val}\n"
        
        # 🎯 5. 隐性战力与基本面
        p += f"  庄家隐含真实战力xG: 主队{eng.get('bookmaker_implied_home_xg', '?')} vs 客队{eng.get('bookmaker_implied_away_xg', '?')}\n"
        p += f"  散户基本面认知: {intel_text}\n"
        
        # 🎯 6. 散户热度与情报
        vote = m.get("vote", {})
        if vote:
            p += f"  散户狂热度: 主胜{vote.get('win', '?')}% 平{vote.get('same', '?')}% 客胜{vote.get('lose', '?')}%\n"
        if h_bad or g_bad:
            p += f"  ⚠火线绝密伤停: 主队-[{h_inj[:60] if h_inj else h_bad[:60]}] | 客队-[{g_inj[:60] if g_inj else g_bad[:60]}]\n"
        
        # 🎯 7. 底层盘房信号
        p += f"  底层盘房异动: {ev_str} | {smart_str}\n\n"

    p += "【严格输出格式示例】\n"
    p += """[
  {
    "match": 1,
    "score": "1-1",
    "reason": "通过我的独立算力推演，虽然散户疯买主胜且主队名气大，但我发现0球赔率被诡异压低至7.0，亚盘退盘极其凶险。结合客队火线伤停诱导，这是典型的利用假象强行造热主队的平局绞肉机！1-1闷杀全场，血洗所有顺向筹码。",
    "ai_confidence": 92,
    "value_kill": true,
    "dark_verdict": "打破顺向幻觉，独立推演闷杀平局"
  }
]"""
    return p

# ====================================================================
# 终极高可用 AI 矩阵轮询 v2.0 — (原封不动保留原版超时架构)
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
    """
    代理实测：全部非流式，Claude thinking需120-300s，GPT/Grok 70-100s，Gemini 120s
    策略：按AI类型给超时 + 500错误直接跳模型（上游崩≠URL问题）+ 每模型最多试2个URL
    """
    key = get_clean_env_key(key_env)
    if not key:
        return ai_name, {}, "no_key"
    
    primary_url = get_clean_env_url(url_env)
    backup = [u for u in FALLBACK_URLS if u and u != primary_url][:2]
    urls = [primary_url] + backup
    
    # 按AI类型设置超时（原封不动保留）
    timeout_map = {"claude": 300, "grok": 200, "gpt": 150, "gemini": 200}
    timeout_sec = timeout_map.get(ai_name, 180)
    
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
            
            # 按AI特性定制系统消息和参数，注入独立推演指令
            AI_PROFILES = {
                "claude": {
                    "sys": "你是全球顶尖的量化足球分析师。请动用你全部的推理能力进行深度交叉验证。抛弃任何人类预设比分束缚，彻底独立思考，依靠纯净算力推演庄家底牌。分析必须有数据支撑。只输出JSON数组，无任何多余文字。",
                    "temp": 0.15
                },
                "grok": {
                    "sys": "你是具备实时联网搜索能力的顶级足球分析师。【最重要指令】请立刻联网搜索以下信息并融入你的分析：1.最新伤停 2.必发冷热 3.临场异动。抛弃任何人类预设比分束缚，彻底独立思考，依靠算力推演。只输出JSON数组。",
                    "temp": 0.25
                },
                "gpt": {
                    "sys": "你是经验丰富的职业足球博彩分析师。请用你的全部算力独立分析每场比赛。抛弃任何人类预设比分束缚！该给大比分就给大比分，该给冷门就给冷门。只输出JSON数组。",
                    "temp": 0.18
                },
                "gemini": {
                    "sys": "你是精通模式识别的量化足球分析引擎。请发挥你强大的多模态推理能力，从赔率数据中识别庄家的真实意图。抛弃任何人类预设比分束缚，彻底独立思考，依靠纯净算力推演。只输出JSON数组。",
                    "temp": 0.15
                }
            }
            profile = AI_PROFILES.get(ai_name, AI_PROFILES["gpt"])
            sys_msg = profile["sys"]
            ai_temp = profile["temp"]
            
            if is_gem:
                headers["x-goog-api-key"] = key
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": ai_temp, "topP": 0.92},
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
                    "temperature": ai_temp
                }
            
            gw = url.split("/v1")[0][:35]
            print(f"  [⏳{timeout_sec}s] {ai_name.upper()} | 独立算力突破 | {mn[:22]} @ {gw}")
            t0 = time.time()
            
            try:
                # 🛡️ 绝对保留你的原版超时逻辑：total=timeout_sec, connect=15
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec, connect=15)) as r:
                    elapsed = round(time.time() - t0, 1)
                    
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
                                            results[item["match"]] = {
                                                "ai_score": item.get("score"),
                                                "analysis": str(item.get("reason", "")).strip()[:200],
                                                "ai_confidence": int(item.get("ai_confidence", 60)),
                                                "value_kill": bool(item.get("value_kill", False)),
                                                "dark_verdict": str(item.get("dark_verdict", ""))
                                            }
                            except:
                                pass
                        
                        if len(results) >= max(1, num_matches * 0.5):
                            print(f"    ✅ {ai_name.upper()} 成功: {len(results)}/{num_matches} | {elapsed}s ({mn[:20]})")
                            return ai_name, results, mn
                        
                        if len(results) > len(best_results):
                            best_results = results
                            best_model = mn
                            print(f"    ⚠️ 部分 {len(results)}/{num_matches} | {elapsed}s")
                        else:
                            print(f"    ⚠️ 解析不足 {len(results)}条 | {elapsed}s")
                    
                    elif r.status == 429:
                        print(f"    🔥 429限流 | {elapsed}s")
                        await asyncio.sleep(3)
                        continue
                    
                    elif r.status >= 500:
                        # 500/502/503 = 上游模型崩溃，换URL没用，直接跳这个模型
                        print(f"    💀 HTTP {r.status} 上游崩溃 | {elapsed}s → 跳过此模型")
                        skip_model = True
                        break
                    
                    else:
                        print(f"    ⚠️ HTTP {r.status} | {elapsed}s")
            
            except asyncio.TimeoutError:
                elapsed = round(time.time() - t0, 1)
                # 超时 = 模型卡死，换URL大概率也卡，直接跳模型
                print(f"    ⏰ {elapsed}s超时 → 跳过此模型")
                skip_model = True
                break
            
            except Exception as e:
                elapsed = round(time.time() - t0, 1)
                err = str(e)[:40]
                # 连接错误才换URL，其他错误跳模型
                if "connect" in err.lower() or "resolve" in err.lower():
                    print(f"    ⚠️ 连接失败 {err} | {elapsed}s → 换URL")
                else:
                    print(f"    ⚠️ {err} | {elapsed}s → 跳过此模型")
                    skip_model = True
                    break
            
            await asyncio.sleep(0.3)
        
        # 当前模型结束，检查是否够用
        if len(best_results) >= max(1, num_matches * 0.4):
            print(f"    ✅ {ai_name.upper()} 采用: {len(best_results)}/{num_matches} ({best_model[:20]})")
            return ai_name, best_results, best_model
    
    if best_results:
        print(f"    ⚠️ {ai_name.upper()} 勉强采用: {len(best_results)}条")
        return ai_name, best_results, best_model
    
    print(f"    ❌ {ai_name.upper()} 全部失败")
    return ai_name, {}, "failed"

async def run_ai_matrix(prompt, num_matches):
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
# Merge 智能融合 v3.0（完全解放比分校验，尊重 AI 共识）
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
    
    # 🚀 切除枷锁：不再校验 in engine_result.get("top3_scores")。只要票数最高直接采纳！
    if vote_count:
        best_voted = max(vote_count, key=vote_count.get)
        if vote_count[best_voted] >= 2 or (isinstance(claude_r, dict) and claude_r.get("ai_score") == best_voted):
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

    # ========== 冷门信号识别 (提供预警信号，不再干预压低比分) ==========
    pre_pred = {
        "home_win_pct": fhp, "draw_pct": fdp, "away_win_pct": fap,
        "steam_move": stats.get("steam_move", {}),
        "smart_signals": stats.get("smart_signals", []),
        "line_movement_anomaly": stats.get("line_movement_anomaly", {}),
    }
    cold_door = ColdDoorDetector.detect(match_obj, pre_pred)
    
    sigs = list(stats.get("smart_signals", []))
    if cold_door["is_cold_door"]:
        sigs.extend(cold_door["signals"])
        cf = max(30, cf - 5)

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

        # v3.3新增：冷门比赛降级
        cold = pr.get("cold_door", {})
        if cold.get("is_cold_door"):
            s -= 8
                
        p["recommend_score"] = round(s, 2)
        
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]

def extract_num(ms):
    wm = {"一":1000,"二":2000,"三":3000,"四":4000,"五":5000,"六":6000,"日":7000,"天":7000}
    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999

# ====================================================================
# ☢️ run_predictions v3.3 — 彻底去势纯净推演版
# ====================================================================
def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 80)
    print(f"  [QUANT ENGINE vMAX 3.3] 彻底抛弃人工边界 | 全面解放AI大模型独立算力 | {len(ms)} 场比赛")
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
        print(f"  [PROMPT] 已拔除本地比分诱导词。将全维客观数据直接注入 AI...")
        start_t = time.time()
        all_ai = asyncio.run(run_ai_matrix(prompt, len(match_analyses)))
        print(f"  [AI MATRIX] 独立算力突破完成，耗时 {time.time()-start_t:.1f}s")

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
        
        # ============ 5层增强管线（每层独立try/except防崩溃） ============
        try:
            mg = apply_experience_to_prediction(m, mg, exp_engine)
        except: pass
        try:
            mg = apply_odds_history(m, mg)
        except: pass
        try:
            mg = apply_quant_edge(m, mg)
        except: pass
        try:
            mg = apply_wencai_intel(m, mg)
        except: pass
        try:
            mg = upgrade_ensemble_predict(m, mg)
        except: pass
        # =====================================================================
        
        pcts = {"主胜": mg["home_win_pct"], "平局": mg["draw_pct"], "客胜": mg["away_win_pct"]}
        mg["result"] = max(pcts, key=pcts.get)

        res.append({**m, "prediction": mg})
        
        cold = mg.get("cold_door", {})
        cold_tag = f" [❄️{cold.get('level','')}冷门预警]" if cold.get("is_cold_door") else ""
        print(f"  [{i+1}] {m.get('home_team')} vs {m.get('away_team')} => {mg['result']} ({mg['predicted_score']}) | CF: {mg['confidence']}% | AI独立确信度: {mg.get('ai_avg_confidence', 0)}{cold_tag}")

    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res:
        r["is_recommended"] = r["id"] in t4ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    diary = load_ai_diary()
    diary["yesterday_win_rate"] = f"{len([r for r in res if r['prediction']['confidence'] > 70])}/{max(1, len(res))}"
    cold_count = len([r for r in res if r.get("prediction", {}).get("cold_door", {}).get("is_cold_door")])
    diary["reflection"] = f"vMAX3.3冷门猎手 | {cold_count}场冷门信号 | 拆除本地比分枷锁，赋予AI完全自主测算权"
    save_ai_diary(diary)

    return res, t4


