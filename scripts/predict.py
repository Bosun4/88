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
# 冷门猎手引擎 (客观信号提取，彻底切除本地比分干预)
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
    return {"yesterday_win_rate": "N/A", "reflection": "已彻底清空本地比分阻拦，双阶段纯净算力接管", "kill_history": []}

def save_ai_diary(diary):
    os.makedirs("data", exist_ok=True)
    with open("data/ai_diary.json", "w", encoding="utf-8") as f:
        json.dump(diary, f, ensure_ascii=False, indent=2)

# ====================================================================
# 🧠 两阶段AI架构 — 零引擎暗示·纯量化解放版
# ====================================================================
def build_phase1_prompt(match_analyses):
    diary = load_ai_diary()
    p = "【身份】你是全球顶尖量化足球分析师。下面是庄家的原始定价数据。\n"
    p += "你必须运用泊松极限、xG剪刀差与欧亚背离，独立计算并给出每场比赛概率最高的3个候选比分。\n\n"
    if diary.get("reflection"):
        p += f"【进化】{diary.get('yesterday_win_rate','N/A')} | {diary['reflection']}\n\n"

    p += "【输出格式——必须严格遵守】\n"
    p += "只输出合法JSON数组，禁止任何其他文字。\n"
    p += "每场：match(整数), top3(数组含3个{score,prob}), reason(100-150字含3+赔率数字,极度专业恶毒), ai_confidence(0-100)。\n"
    p += '示例: {"match":1,"top3":[{"score":"4-1","prob":18.2},{"score":"3-0","prob":16.5},{"score":"1-1","prob":12.1}],"reason":"...","ai_confidence":75}\n\n'

    p += "【量化计算铁律——彻底独立思考】\n"
    p += "1. 彻底抛弃一切人工保守框架！该大屠杀就给 4-0，该闷杀就给 0-0！\n"
    p += "2. CRS 赔率仅作为你反推庄家底牌的参考数据，绝不是限制你选择比分的枷锁！\n"
    p += "3. 利用总进球赔率（a0-a7）推演庄家真实的进球天花板。\n"
    p += "4. 结合伤停情报、亚盘与欧赔的错配，敏锐捕捉冷门与陷阱。\n\n"

    p += "【原始数据底库】\n"
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        league = m.get("league", m.get("cup", ""))
        hc = m.get("give_ball", "0")
        sp_h = float(m.get("sp_home", m.get("win", 0)) or 0)
        sp_d = float(m.get("sp_draw", m.get("same", 0)) or 0)
        sp_a = float(m.get("sp_away", m.get("lose", 0)) or 0)

        p += f"{'='*60}\n[{i+1}] {h} vs {a} | {league}\n"

        for lg, rate in {"英冠":32,"英甲":30,"英乙":28,"法乙":28,"荷乙":27,"德乙":26,"意乙":25}.items():
            if lg in str(league):
                p += f"⚠️ {lg}历史冷门率{rate}%\n"; break

        odds_rng = round(max(sp_h,sp_d,sp_a)-min(sp_h,sp_d,sp_a),2) if sp_h>1 else 0
        tag = " ⚠️三项极接近=均势局" if 0<odds_rng<0.8 else ""
        p += f"欧赔: {sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f}{tag} | 亚盘死线: {hc}\n"

        if m.get("hhad_win"):
            p += f"让球胜平负: {m['hhad_win']}/{m.get('hhad_same','')}/{m.get('hhad_lose','')}\n"
        if m.get("single") == 1:
            p += f"📌 单关开放(庄家风控极严)\n"

        h_pos = m.get("home_position",""); g_pos = m.get("guest_position","")
        if h_pos or g_pos:
            p += f"排名: 主{h_pos} vs 客{g_pos}\n"

        a0=m.get("a0","");a1=m.get("a1","");a2=m.get("a2","");a3=m.get("a3","")
        a4=m.get("a4","");a5=m.get("a5","");a6=m.get("a6","");a7=m.get("a7","")
        if a0:
            p += f"庄家总进球底牌: 0球={a0}|1={a1}|2={a2}|3={a3}|4={a4}|5={a5}|6={a6}|7+={a7}\n"

        crs_map = {"w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","w32":"3-2","w40":"4-0","w41":"4-1","w42":"4-2",
                   "s00":"0-0","s11":"1-1","s22":"2-2","s33":"3-3",
                   "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3","l23":"2-3","l04":"0-4","l14":"1-4"}
        crs_lines=[]; crs_probs=[]
        for key,score in crs_map.items():
            try:
                odds=float(m.get(key,0)or 0)
                if odds>1: crs_lines.append(f"{score}={odds:.2f}"); crs_probs.append((score,odds))
            except: pass
        if crs_probs:
            crs_probs.sort(key=lambda x:x[1])
            p += f"机构重点防范比分参考: {' | '.join(f'{s}@{o:.2f}' for s,o in crs_probs[:6])}\n"

        hf_l=[]
        for k,lb in {"ss":"主/主","sp":"主/平","sf":"主/负","ps":"平/主","pp":"平/平","pf":"平/负","fs":"负/主","fp":"负/平","ff":"负/负"}.items():
            try:
                v=float(m.get(k,0)or 0)
                if v>1: hf_l.append(f"{lb}={v:.2f}")
            except: pass
        if hf_l: p += f"半全场: {' | '.join(hf_l)}\n"

        vote=m.get("vote",{})
        if vote:
            p += f"散户狂热度: 胜{vote.get('win','?')}% 平{vote.get('same','?')}% 负{vote.get('lose','?')}%"
            if vote.get("hhad_win"): p += f" | 让球方向: 主{vote['hhad_win']}% 平{vote.get('hhad_same','?')}% 客{vote.get('hhad_lose','?')}%"
            p += "\n"

        change=m.get("change",{})
        if change and isinstance(change,dict):
            cw=change.get("win",0);cs=change.get("same",0);cl=change.get("lose",0)
            if cw or cs or cl: p += f"赔率底层异动: 胜{cw} 平{cs} 负{cl}\n"

        info=m.get("information",{})
        if isinstance(info,dict):
            for k,v in [("home_injury","主绝密伤停"),("guest_injury","客绝密伤停"),("home_good_news","主利好"),("guest_good_news","客利好"),("home_bad_news","主利空"),("guest_bad_news","客利空")]:
                if info.get(k): p += f"{v}: {str(info[k])[:250].replace(chr(10),' | ')}\n"

        hs=m.get("home_stats",{}); ast2=m.get("away_stats",{})
        if hs.get("form"):
            p += f"实录: 主队 {hs.get('wins','?')}胜{hs.get('draws','?')}平{hs.get('losses','?')}负 进{hs.get('avg_goals_for','?')}/失{hs.get('avg_goals_against','?')} | 客队 {ast2.get('wins','?')}胜{ast2.get('draws','?')}平{ast2.get('losses','?')}负 进{ast2.get('avg_goals_for','?')}/失{ast2.get('avg_goals_against','?')}\n"

        for field in ['analyse','baseface','intro','expert_intro']:
            txt=str(m.get(field,'')).replace('\n',' ')[:200]
            if len(txt)>10: p += f"机构研判: {txt}\n"; break
        p += "\n"

    p += f"【输出{len(match_analyses)}场JSON数组，每场含top3。只输出纯数组！】\n"
    return p

def build_phase2_prompt(match_analyses, phase1_results):
    p = "【身份指令：最终死刑执行官】三个独立先锋AI已给出原始数据的候选比分。\n"
    p += "你的任务：抛弃任何保守思想，利用你极其强悍的仲裁算力，选出每场比赛的绝对唯一比分！\n\n"
    p += "【绝对自由仲裁原则】\n"
    p += "1. 如果3家AI的判断合理，你可以采纳；如果你认为存在更深层的陷阱，你完全可以推翻他们，给出你的惊天波胆！\n"
    p += "2. 不受CRS赔率上限束缚。如果是大逃杀，6-0也可以输出。\n"
    p += "3. reason中用极其残忍、刻薄的庄家视角黑话，戳穿散户的幻想。\n\n"
    p += "【输出格式】JSON数组：match(整数), score(唯一比分), reason(80-120字), ai_confidence(0-100)\n"
    p += "只输出JSON数组！禁止其他文字！\n\n"

    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        sp_h = float(m.get("sp_home", m.get("win", 0)) or 0)
        sp_d = float(m.get("sp_draw", m.get("same", 0)) or 0)
        sp_a = float(m.get("sp_away", m.get("lose", 0)) or 0)
        idx = i + 1

        p += f"{'='*60}\n[{idx}] {h} vs {a}\n"
        p += f"欧赔: {sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f} | 亚盘死线: {m.get('give_ball','0')}\n"

        for ai_name in ["gpt", "grok", "gemini"]:
            ai_data = phase1_results.get(ai_name, {}).get(idx, {})
            if not ai_data:
                continue
            top3 = ai_data.get("top3", [])
            if top3:
                scores_str = " | ".join(f"{t.get('score','?')}({t.get('prob','?')}%)" for t in top3[:3])
                p += f"  先锋 {ai_name.upper()} 独立研判: {scores_str} | {str(ai_data.get('reason',''))[:100]}\n"
            else:
                sc = ai_data.get("ai_score", "-")
                p += f"  先锋 {ai_name.upper()}: {sc} | {str(ai_data.get('analysis',''))[:100]}\n"
        p += "\n"

    p += f"【现在执行最终死刑判决！输出{len(match_analyses)}场纯JSON数组！】\n"
    return p

# ====================================================================
# AI调用引擎（防暴毙断肢重生，严格保留原始时间设置）
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
                            
                        # 🔥 修复：剔除深度思考标签，防止 JSON 提取爆炸
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
                                # 🔥 修复：长文本截断断肢重生抢救算法
                                try:
                                    if not json_str.endswith("]"): json_str += "]"
                                    if not json_str.endswith("}]"): json_str = json_str[:-1] + "}]"
                                    arr = json.loads(json_str)
                                    print(f"    🩹 触发断肢重生，成功抢救 {len(arr)} 条数据！")
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
                            
                        # 成功响应但解析不够，直接跳模型防死循环
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
                print(f"    ⚠️ {err} | {elapsed}s → 换URL")
                continue
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
    print(f"  [Phase2] {len(p2_prompt):,} 字符 → Claude 仲裁...")

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
# Merge v8.5 — 完全去势版：没有任何本地强制篡改，唯独尊崇 AI
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
    if not final_score:
        p1_ai = {"gpt": gpt_r, "grok": grok_r, "gemini": gemini_r}
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

    # 抽取冷门预警红点，但不干预比分
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
# run_predictions v8.5 Pro
# ====================================================================
def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 100)
    print(f"  [GROK-FUSED v8.5 Pro] 纯净无界算力版 | 双阶段纯粹推演 | {len(ms)} 场")
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
    diary["reflection"] = f"v8.5 Pro | 彻底粉碎本地设限 | 解决大文本截断Bug | 100%全权移交AI算力推演"
    save_ai_diary(diary)

    return res, t4
