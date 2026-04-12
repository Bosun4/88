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
# 🧊 冷门猎手引擎
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
# 🧠 两阶段AI架构 vMAX 8.0 — AI自主决策·赔率仅做参考
#
# 核心哲学变革：
#   旧版: 赔率=约束 → AI被CRS锁死在1-0/0-1/1-1
#   新版: 赔率=情报 → AI综合所有维度独立判断，CRS只是加减分
#
# Phase1: GPT/Grok/Gemini 独立深度分析 → TOP3候选比分+概率
# Phase2: Claude 裁判综合 → 加权评分选出最终比分（无否决制）
# ====================================================================

def build_phase1_prompt(match_analyses):
    """Phase1 Prompt: 纯数据·零暗示·让AI自由分析"""
    diary = load_ai_diary()

    p = "你是顶尖足球量化分析师。根据以下原始数据，独立分析每场比赛，给出概率最高的3个候选比分。\n\n"

    if diary.get("reflection"):
        p += f"【进化】{diary.get('yesterday_win_rate','N/A')} | {diary['reflection']}\n\n"

    p += "【输出格式】只输出合法JSON数组。每场：match(整数), top3([{score,prob},...]), reason(80字), ai_confidence(0-100)。只输出数组！\n\n"

    p += "【原始数据】\n"
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        league = m.get("league", m.get("cup", ""))
        hc = m.get("give_ball", "0")
        sp_h = float(m.get("sp_home", m.get("win", 0)) or 0)
        sp_d = float(m.get("sp_draw", m.get("same", 0)) or 0)
        sp_a = float(m.get("sp_away", m.get("lose", 0)) or 0)

        p += f"{'='*50}\n[{i+1}] {h} vs {a} | {league}\n"
        p += f"欧赔: {sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f} | 让球: {hc}\n"

        if sp_h > 1 and sp_d > 1 and sp_a > 1:
            margin = 1/sp_h + 1/sp_d + 1/sp_a
            p += f"Shin概率: 主{(1/sp_h)/margin*100:.1f}% 平{(1/sp_d)/margin*100:.1f}% 客{(1/sp_a)/margin*100:.1f}% | 返还率{1/margin*100:.1f}%\n"

        if m.get("hhad_win"):
            p += f"让球胜平负: {m['hhad_win']}/{m.get('hhad_same','')}/{m.get('hhad_lose','')}\n"
        if m.get("single") == 1:
            p += f"单关开放\n"
        h_pos = m.get("home_position",""); g_pos = m.get("guest_position","")
        if h_pos or g_pos:
            p += f"排名: 主{h_pos} vs 客{g_pos}\n"

        # 总进球
        a0=m.get("a0","");a1=m.get("a1","")
        if a0:
            a2=m.get("a2","");a3=m.get("a3","");a4=m.get("a4","");a5=m.get("a5","");a6=m.get("a6","");a7=m.get("a7","")
            p += f"总进球赔率: 0球={a0}|1={a1}|2={a2}|3={a3}|4={a4}|5={a5}|6={a6}|7+={a7}\n"
            try:
                gp=[(gi,1/float(v)) for gi,v in enumerate([a0,a1,a2,a3,a4,a5,a6,a7]) if float(v)>1]
                tp=sum(p2 for _,p2 in gp); eg=sum(g*(p2/tp) for g,p2 in gp)
                p += f"→ 期望进球λ={eg:.2f}\n"
            except: pass

        # CRS
        crs_map = {"w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","w32":"3-2","w40":"4-0","w41":"4-1","w42":"4-2",
                   "s00":"0-0","s11":"1-1","s22":"2-2","s33":"3-3",
                   "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3","l23":"2-3","l04":"0-4","l14":"1-4"}
        crs_lines=[]
        for key,score in crs_map.items():
            try:
                odds=float(m.get(key,0) or 0)
                if odds>1: crs_lines.append(f"{score}={odds:.1f}")
            except: pass
        if crs_lines:
            p += f"CRS: {' | '.join(crs_lines)}\n"

        # 半全场
        hf_l=[]
        for k,lb in {"ss":"主/主","sp":"主/平","sf":"主/负","ps":"平/主","pp":"平/平","pf":"平/负","fs":"负/主","fp":"负/平","ff":"负/负"}.items():
            try:
                v=float(m.get(k,0) or 0)
                if v>1: hf_l.append(f"{lb}={v:.2f}")
            except: pass
        if hf_l: p += f"半全场: {' | '.join(hf_l)}\n"

        # 散户
        vote=m.get("vote",{})
        if vote:
            p += f"散户: 胜{vote.get('win','?')}% 平{vote.get('same','?')}% 负{vote.get('lose','?')}%"
            if vote.get("hhad_win"): p += f" | 让球主{vote['hhad_win']}%平{vote.get('hhad_same','?')}%客{vote.get('hhad_lose','?')}%"
            p += "\n"

        # 赔率变动
        change=m.get("change",{})
        if change and isinstance(change,dict):
            cw=change.get("win",0);cs=change.get("same",0);cl=change.get("lose",0)
            if cw or cs or cl: p += f"赔率变动: 胜{cw} 平{cs} 负{cl}\n"

        # 伤停
        info=m.get("information",{})
        if isinstance(info,dict):
            for k,v in [("home_injury","主伤停"),("guest_injury","客伤停"),("home_bad_news","主利空"),("guest_bad_news","客利空")]:
                if info.get(k): p += f"{v}: {str(info[k])[:200].replace(chr(10),' ')}\n"

        # 状态
        hs=m.get("home_stats",{}); ast2=m.get("away_stats",{})
        if hs.get("form"):
            p += f"主队: {hs.get('wins','?')}胜{hs.get('draws','?')}平{hs.get('losses','?')}负 {hs.get('form','?')} 进{hs.get('avg_goals_for','?')}/失{hs.get('avg_goals_against','?')}\n"
            p += f"客队: {ast2.get('wins','?')}胜{ast2.get('draws','?')}平{ast2.get('losses','?')}负 {ast2.get('form','?')} 进{ast2.get('avg_goals_for','?')}/失{ast2.get('avg_goals_against','?')}\n"

        for field in ['analyse','baseface','intro','expert_intro']:
            txt=str(m.get(field,'')).replace('\n',' ')[:150]
            if len(txt)>10: p += f"分析: {txt}\n"; break
        p += "\n"

    p += f"【输出{len(match_analyses)}场JSON数组，只输出数组！】\n"
    return p


def build_phase2_prompt(match_analyses, phase1_results):
    """Phase2: Claude裁判——看数据做判断，不给框架"""
    p = "你是最终裁判。多个AI已独立分析每场比赛。你的任务：综合它们的分析，选出每场最终比分。\n"
    p += "如果多家AI一致，直接采用。如果分歧，用你自己的判断选最合理的。\n"
    p += "输出JSON数组：match(整数), score(比分), reason(50字), ai_confidence(0-100)。只输出数组！\n\n"

    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        sp_h = float(m.get("sp_home", m.get("win", 0)) or 0)
        sp_d = float(m.get("sp_draw", m.get("same", 0)) or 0)
        sp_a = float(m.get("sp_away", m.get("lose", 0)) or 0)
        idx = i + 1

        p += f"[{idx}] {h} vs {a} | 欧赔{sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f}\n"

        for ai_name in ["gpt", "grok", "gemini"]:
            ai_data = phase1_results.get(ai_name, {}).get(idx, {})
            if not ai_data: continue
            top3 = ai_data.get("top3", [])
            if top3:
                scores_str = " ".join(f"{t.get('score','?')}({t.get('prob','?')}%)" for t in top3[:3])
                p += f"  {ai_name.upper()}: {scores_str} | {str(ai_data.get('reason',''))[:80]}\n"
            else:
                sc = ai_data.get("ai_score", "-")
                p += f"  {ai_name.upper()}: {sc} | {str(ai_data.get('reason',ai_data.get('analysis','')))[:80]}\n"

    p += f"\n输出{len(match_analyses)}场JSON数组！\n"
    return p


# ====================================================================
# AI调用引擎（与原版相同，不动核心网络层）
# ====================================================================
FALLBACK_URLS = [None, "https://api520.pro/v1", "https://api521.pro/v1", "https://api522.pro/v1", "https://www.api522.pro/v1"]

def get_clean_env_url(name, default=""):
    v = str(os.environ.get(name, globals().get(name, default))).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/-]+)", v)
    return match.group(1) if match else v

def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")

async def async_call_one_ai_batch(session, prompt, url_env, key_env, models_list, num_matches, ai_name):
    """
    vMAX 8.0 — 连上就等·连不上就换
    
    逻辑极简：
      连接超时30秒：能不能连上服务器
      读取超时：无限（连上了就死等数据回来，因为钱已经花了）
      只有连接失败/502/504才换下一个模型
      连上了=这份钱花定了，等到底
    
    模型顺序按价格从低到高：按量→A系列→99额度
    一个AI只花一份钱，不会重复消耗
    """
    key = get_clean_env_key(key_env)
    if not key: return ai_name, {}, "no_key"
    primary_url = get_clean_env_url(url_env)
    backup = [u for u in FALLBACK_URLS if u and u != primary_url][:1]
    urls = [primary_url] + backup

    CONNECT_TIMEOUT = 20      # 连接超时20秒
    READ_TIMEOUT = 400        # 连上后等400秒（实际最慢350秒）

    AI_PROFILES = {
        "claude": {
            "sys": "你是最终裁判。综合多个AI的分析结果，选出每场最终比分。只输出JSON数组。",
            "temp": 0.15
        },
        "grok": {
            "sys": "你是Grok，有联网搜索能力。搜索Pinnacle赔率、Betfair交易量、球队伤停和最新动态，结合提供的数据分析每场比赛。输出每场TOP3候选比分。只输出JSON数组。",
            "temp": 0.22
        },
        "gpt": {
            "sys": "你是职业足球量化分析师。用数学方法分析赔率数据，计算每场比赛概率最高的3个比分。只输出JSON数组。",
            "temp": 0.18
        },
        "gemini": {
            "sys": "你是概率建模引擎。从赔率数据计算每个比分的真实概率，找出被市场错误定价的比分。输出每场TOP3候选比分。只输出JSON数组。",
            "temp": 0.15
        },
    }

    best_model = ""
    profile = AI_PROFILES.get(ai_name, AI_PROFILES["gpt"])

    for mn in models_list:
        connected = False  # 标记是否已经连上（=钱已花）

        for base_url in urls:
            if not base_url: continue

            is_gem = "generateContent" in base_url
            url = base_url.rstrip("/")
            if not is_gem and "chat/completions" not in url: url += "/chat/completions"
            headers = {"Content-Type": "application/json"}

            if is_gem:
                headers["x-goog-api-key"] = key
                payload = {"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"temperature":profile["temp"]},"systemInstruction":{"parts":[{"text":profile["sys"]}]}}
            else:
                headers["Authorization"] = f"Bearer {key}"
                bp = {"model":mn,"messages":[{"role":"system","content":profile["sys"]},{"role":"user","content":prompt}]}
                if ai_name != "claude": bp["temperature"] = profile["temp"]
                payload = bp

            gw = url.split("/v1")[0][:35]
            print(f"  [🔌连接中] {ai_name.upper()} | {mn[:22]} @ {gw}")
            t0 = time.time()

            try:
                timeout = aiohttp.ClientTimeout(
                    total=None,              # 总超时不限
                    connect=CONNECT_TIMEOUT,
                    sock_connect=CONNECT_TIMEOUT,
                    sock_read=READ_TIMEOUT,
                )
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                    elapsed_connect = round(time.time()-t0, 1)

                    # ===== 连接失败类：换URL或换模型 =====
                    if r.status in (502, 504):
                        print(f"    💀 HTTP {r.status} 网关超时 | {elapsed_connect}s → 换模型")
                        break

                    if r.status == 400:
                        print(f"    💀 400 模型不支持 | {elapsed_connect}s → 换模型")
                        break  # 这个模型不行，换下一个模型

                    if r.status == 429:
                        print(f"    🔥 429 限流 | {elapsed_connect}s → 换模型（同key换URL也429）")
                        break

                    if r.status != 200:
                        print(f"    ⚠️ HTTP {r.status} | {elapsed_connect}s → 换下一个")
                        continue

                    # ===== 200 = 连上了！钱已花，死等数据 =====
                    connected = True
                    print(f"    ✅ 已连上！{elapsed_connect}s | 等待模型思考返回数据...")

                    try:
                        data = await r.json(content_type=None)
                    except:
                        elapsed = round(time.time()-t0, 1)
                        print(f"    ⚠️ 响应非JSON | {elapsed}s → 换模型")
                        break  # 同模型同格式，换URL也一样

                    elapsed = round(time.time()-t0, 1)

                    # 提取token消耗（仅打印）
                    usage = data.get("usage", {})
                    req_tokens = usage.get("total_tokens", 0) or (
                        usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                    )
                    if not req_tokens:
                        um = data.get("usageMetadata", {})
                        req_tokens = um.get("totalTokenCount", 0)
                    if req_tokens:
                        print(f"    📊 消耗: {req_tokens:,} token | 耗时: {elapsed}s")

                    # 提取文本 — 极简版：先直接取，不要花哨
                    raw_text = ""
                    try:
                        if is_gem:
                            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        else:
                            # 第一步：直接从choices取，这是标准格式
                            if data.get("choices") and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                                msg = data["choices"][0].get("message", {})
                                if isinstance(msg, dict):
                                    # 遍历所有字段，找最长的包含[的文本
                                    best_with_bracket = ""
                                    best_any = ""
                                    for key in msg:
                                        val = msg[key]
                                        if not val or not isinstance(val, str): continue
                                        val = val.strip()
                                        if not val: continue
                                        # 优先：包含JSON数组标记的
                                        if "[" in val and "{" in val and len(val) > len(best_with_bracket):
                                            best_with_bracket = val
                                        # 备选：最长的文本
                                        if len(val) > len(best_any):
                                            best_any = val

                                    raw_text = best_with_bracket or best_any

                            # 第二步：output数组格式（某些代理）
                            if not raw_text and data.get("output") and isinstance(data["output"], list):
                                for out_item in data["output"]:
                                    if not isinstance(out_item, dict): continue
                                    if out_item.get("type") == "message":
                                        for ct in out_item.get("content", []):
                                            if isinstance(ct, dict) and ct.get("text"):
                                                txt = ct["text"].strip()
                                                if len(txt) > len(raw_text):
                                                    raw_text = txt
                                    elif isinstance(out_item.get("content"), str):
                                        txt = out_item["content"].strip()
                                        if len(txt) > len(raw_text):
                                            raw_text = txt

                            # 第三步：全部失败，dump整个response
                            if not raw_text:
                                full_str = json.dumps(data, ensure_ascii=False)
                                # 智能提取：在dump字符串里找 "match" 附近的JSON数组
                                # 找 [{"match" 模式
                                m_match = re.search(r'\[\s*\{\s*"match"', full_str)
                                if m_match:
                                    start_pos = m_match.start()
                                    # 从这里开始数括号找匹配的]
                                    depth = 0
                                    end_pos = start_pos
                                    for ci in range(start_pos, min(start_pos + 100000, len(full_str))):
                                        if full_str[ci] == '[': depth += 1
                                        elif full_str[ci] == ']': depth -= 1
                                        if depth == 0:
                                            end_pos = ci + 1
                                            break
                                    if end_pos > start_pos:
                                        extracted = full_str[start_pos:end_pos]
                                        # 处理JSON字符串转义 (\")
                                        if '\\"' in extracted:
                                            try:
                                                extracted = json.loads('"' + extracted + '"')
                                            except:
                                                extracted = extracted.replace('\\"', '"')
                                        raw_text = extracted
                                        print(f"    🔍 从dump中提取到JSON: {len(raw_text)}字")

                                if not raw_text:
                                    raw_text = full_str
                                    print(f"    ⚠️ 用整个response做raw_text: {len(raw_text)}字")

                    except Exception as ex:
                        print(f"    ⚠️ 文本提取异常: {str(ex)[:80]}")
                        try: raw_text = json.dumps(data, ensure_ascii=False)
                        except: pass

                    if not raw_text or len(raw_text) < 10:
                        print(f"    ⚠️ 模型返回空数据 | {elapsed}s | 换模型")
                        break

                    # 解析JSON — 多层清理
                    clean = raw_text
                    # 清理各种thinking标签格式
                    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", clean, flags=re.DOTALL|re.IGNORECASE)
                    clean = re.sub(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", "", clean, flags=re.DOTALL)
                    clean = re.sub(r"```[\w]*","",clean).strip()
                    start=clean.find("["); end=clean.rfind("]")+1
                    if start==-1 or end==0:
                        clean = re.sub(r"[^\[\]{}:,\"'0-9a-zA-Z\u4e00-\u9fa5\s\.\-\+\(\)]","",clean)
                        start=clean.find("["); end=clean.rfind("]")+1

                    results = {}
                    if start != -1 and end > start:
                        json_str = clean[start:end]
                        arr = []
                        try: arr = json.loads(json_str)
                        except json.JSONDecodeError:
                            try:
                                last_brace = json_str.rfind('}')
                                if last_brace != -1:
                                    arr = json.loads(json_str[:last_brace+1] + "]")
                                    print(f"    🩹 断肢重生: 抢救 {len(arr)} 条")
                            except: pass
                        if isinstance(arr, list):
                            for item in arr:
                                if not isinstance(item, dict) or not item.get("match"): continue
                                try: mid = int(item["match"])
                                except: mid = item["match"]
                                if item.get("top3"):
                                    t1_score = item["top3"][0].get("score","1-1").replace(" ","").strip() if item["top3"] else "1-1"
                                    results[mid] = {
                                        "top3": item["top3"],
                                        "ai_score": t1_score,
                                        "reason": str(item.get("reason",""))[:200],
                                        "ai_confidence": int(item.get("ai_confidence",60)),
                                    }
                                elif item.get("score"):
                                    results[mid] = {
                                        "ai_score": item["score"].replace(" ","").strip(),
                                        "analysis": str(item.get("reason",""))[:200],
                                        "ai_confidence": int(item.get("ai_confidence",60)),
                                        "value_kill": bool(item.get("value_kill",False)),
                                    }

                    if len(results) > 0:
                        print(f"    ✅ {ai_name.upper()} 完成: {len(results)}/{num_matches} | {elapsed}s ({mn[:20]})")
                        return ai_name, results, mn
                    else:
                        # 花了钱但解析0条 → break换下一个模型（Claude可从99额度→按量）
                        print(f"    ⚠️ 解析0条 | {elapsed}s | 换下一个模型")
                        break

            except aiohttp.ClientConnectorError as e:
                elapsed = round(time.time()-t0, 1)
                print(f"    🔌 连接失败 {str(e)[:30]} | {elapsed}s → 换URL")
                continue  # 没连上=没花钱，换URL

            except asyncio.TimeoutError:
                elapsed = round(time.time()-t0, 1)
                if not connected:
                    # 连接阶段超时=没花钱，换下一个
                    print(f"    🔌 {elapsed}s连接超时 → 换下一个")
                    continue
                else:
                    # 已连上但读取超时（极罕见，600秒还没返回）
                    print(f"    ⏰ 已连上但{elapsed}s仍无数据 | 钱已花")
                    return ai_name, {}, "read_timeout"

            except Exception as e:
                elapsed = round(time.time()-t0, 1)
                err = str(e)[:40]
                if not connected:
                    print(f"    ⚠️ {err} | {elapsed}s → 换下一个")
                    continue
                else:
                    print(f"    ⚠️ 已连上但异常: {err} | {elapsed}s | 钱已花")
                    return ai_name, {}, "error"

            await asyncio.sleep(0.2)

        # 如果这个模型连上过（connected=True），上面已经return了
        # 走到这里说明这个模型所有URL都连不上，试下一个模型

    # 所有模型都连不上
    print(f"    ❌ {ai_name.upper()} 所有模型均连接失败（未花钱）")
    return ai_name, {}, "all_connect_failed"


async def run_ai_matrix_two_phase(match_analyses):
    """两阶段：Phase1(GPT/Grok/Gemini并行)→ Phase2(Claude裁判)"""
    num = len(match_analyses)

    p1_prompt = build_phase1_prompt(match_analyses)
    print(f"  [Phase1] {len(p1_prompt):,} 字符 → GPT/Grok/Gemini 并行...")

    p1_configs = [
        ("grok","GROK_API_URL","GROK_API_KEY",["熊猫-A-6-grok-4.2-thinking"]),
        ("gpt","GPT_API_URL","GPT_API_KEY",["熊猫-按量-gpt-5.4"]),
        ("gemini","GEMINI_API_URL","GEMINI_API_KEY",["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking"]),
    ]
    p1_results = {"gpt":{},"grok":{},"gemini":{}}

    # 共享一个session复用TCP连接池，减少502/504
    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Phase1: 三家并行
        tasks = [async_call_one_ai_batch(session,p1_prompt,u,k,m,num,n) for n,u,k,m in p1_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res,tuple): n,d,_ = res; p1_results[n] = d
            else: print(f"  [Phase1 ERROR] {res}")

        ok = sum(1 for v in p1_results.values() if v)
        print(f"  [Phase1] 完成: {ok}/3 AI有数据")

        # Phase2: Claude裁判（复用同一个session）
        p2_prompt = build_phase2_prompt(match_analyses, p1_results)
        print(f"  [Phase2] {len(p2_prompt):,} 字符 → Claude 裁判...")

        claude_r = {}
        _,claude_r,_ = await async_call_one_ai_batch(
            session, p2_prompt, "CLAUDE_API_URL","CLAUDE_API_KEY",
            ["熊猫特供-超纯满血-99额度-claude-opus-4.6-thinking","熊猫-按量-特供顶级-官方正向满血-claude-opus-4.6-thinking"],
            num, "claude"
        )

    all_r = p1_results.copy()
    all_r["claude"] = claude_r
    return all_r


# ====================================================================
# Merge v6.0 — 三层决策链：方向→进球数→比分
#
# 核心哲学：
#   旧版: 4个AI各猜一个比分→投票→永远2-1/1-1
#   新版: 先用庄家信号定方向，再用进球数定范围，最后才选比分
#
# 第一层 [方向]: Sharp资金+赔率变动+散户反指+冷门预警→主胜/平/客胜
# 第二层 [进球数]: 总进球赔率a0-a7→期望λ→最可能进球数
# 第三层 [比分]: 在方向+进球数约束下，CRS+AI投票→最终比分
#
# 关键改进：
#   - Sharp资金流向客胜+散户66%押主胜 → 方向=客胜（不是主胜）
#   - 预期2.8球 → 3球范围 → 1-2/0-3/2-1（不是1-0/1-1）
#   - AI投票只在同方向+同进球数的比分里选，不再跨方向投票
# ====================================================================
def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    import math
    sp_h = float(match_obj.get("sp_home", match_obj.get("win", 0)) or 0)
    sp_d = float(match_obj.get("sp_draw", match_obj.get("same", 0)) or 0)
    sp_a = float(match_obj.get("sp_away", match_obj.get("lose", 0)) or 0)
    engine_score = engine_result.get("primary_score", "1-1")
    engine_conf = engine_result.get("confidence", 50)
    league = str(match_obj.get("league", match_obj.get("cup", "")))

    def clean_score(s):
        if not s or not isinstance(s, str): return ""
        s = s.strip().replace(" ", "").replace("：", "-").replace(":", "-")
        s = s.replace("\u2013", "-").replace("\u2014", "-")
        if re.match(r"^\d+-\d+$", s): return s
        return ""

    p1_ai = {"gpt": gpt_r, "grok": grok_r, "gemini": gemini_r}

    # ================================================================
    # 第一层：方向决策（主胜/平局/客胜）
    # 用加权投票，每个信号有方向和权重
    # ================================================================
    direction_scores = {"home": 0.0, "draw": 0.0, "away": 0.0}

    # --- 信号1: 欧赔隐含概率 [权重30] — 最客观的基准 ---
    if sp_h > 1 and sp_d > 1 and sp_a > 1:
        margin = 1/sp_h + 1/sp_d + 1/sp_a
        shin_h = (1/sp_h) / margin * 100
        shin_d = (1/sp_d) / margin * 100
        shin_a = (1/sp_a) / margin * 100
        direction_scores["home"] += shin_h / 100 * 30
        direction_scores["draw"] += shin_d / 100 * 30
        direction_scores["away"] += shin_a / 100 * 30
    else:
        direction_scores["home"] += 10; direction_scores["draw"] += 10; direction_scores["away"] += 10

    # --- 信号2: Sharp资金流向 [权重12] — 重要但不主导 ---
    smart_signals = stats.get("smart_signals", [])
    smart_str = " ".join(str(s) for s in smart_signals)
    sharp_detected = False

    if "Sharp" in smart_str or "sharp" in smart_str:
        sharp_detected = True
        if "客胜" in smart_str or "客队" in smart_str:
            direction_scores["away"] += 12
            print(f"    💰 Sharp→客胜 +12")
        elif "主胜" in smart_str or "主队" in smart_str:
            direction_scores["home"] += 12
            print(f"    💰 Sharp→主胜 +12")
        elif "平局" in smart_str or "平赔" in smart_str:
            direction_scores["draw"] += 12
            print(f"    💰 Sharp→平局 +12")
        else:
            direction_scores["home"] += 4; direction_scores["draw"] += 4; direction_scores["away"] += 4
    # Steam信号 [权重8]
    if "Steam" in smart_str:
        if "客胜Steam" in smart_str or "客胜反向" in smart_str:
            direction_scores["away"] += 8
        elif "主胜Steam" in smart_str or "主胜反向" in smart_str:
            direction_scores["home"] += 8
        elif "平局Steam" in smart_str:
            direction_scores["draw"] += 8

    # --- 信号3: 散户反指 [权重10] ---
    vote = match_obj.get("vote", {})
    try:
        vh = int(vote.get("win", 33) or 33)
        vd = int(vote.get("same", 33) or 33)
        va = int(vote.get("lose", 33) or 33)
        max_vote = max(vh, vd, va)
        if max_vote >= 58:
            if vh == max_vote:
                contrarian_weight = min(10, (vh - 50) * 0.6)
                direction_scores["away"] += contrarian_weight * 0.6
                direction_scores["draw"] += contrarian_weight * 0.4
            elif va == max_vote:
                contrarian_weight = min(10, (va - 50) * 0.6)
                direction_scores["home"] += contrarian_weight * 0.6
                direction_scores["draw"] += contrarian_weight * 0.4
        else:
            direction_scores["home"] += 3; direction_scores["draw"] += 3; direction_scores["away"] += 3
    except:
        direction_scores["home"] += 3; direction_scores["draw"] += 3; direction_scores["away"] += 3

    # --- 信号4: 冷门预警 [权重8] ---
    hp_eng = engine_result.get("home_prob", 33)
    ap_eng = engine_result.get("away_prob", 33)
    hot_side = "home" if hp_eng > ap_eng else "away"
    cold_signals_raw = [s for s in smart_signals if "❄️" in str(s) or "冷门" in str(s) or "大热" in str(s) or "造热" in str(s)]
    if cold_signals_raw:
        cold_weight = min(8, len(cold_signals_raw) * 2.5)
        if hot_side == "home":
            direction_scores["home"] -= cold_weight
            direction_scores["away"] += cold_weight * 0.5
            direction_scores["draw"] += cold_weight * 0.5
        else:
            direction_scores["away"] -= cold_weight
            direction_scores["home"] += cold_weight * 0.5
            direction_scores["draw"] += cold_weight * 0.5

    # --- 信号5: AI方向共识 [权重25] — AI分析结果应该主导 ---
    ai_directions = {"home": 0, "draw": 0, "away": 0}
    for name, r in {**p1_ai, "claude": claude_r}.items():
        if not isinstance(r, dict): continue
        sc = clean_score(r.get("ai_score", ""))
        if not sc:
            t3 = r.get("top3", [])
            if t3 and isinstance(t3, list) and len(t3) > 0:
                sc = clean_score(t3[0].get("score", ""))
        if sc:
            try:
                sh, sa = map(int, sc.split("-"))
                w = 1.5 if name == "claude" else 1.0
                if sh > sa: ai_directions["home"] += w
                elif sh < sa: ai_directions["away"] += w
                else: ai_directions["draw"] += w
            except: pass
    total_ai_votes = sum(ai_directions.values())
    if total_ai_votes > 0:
        for d in ["home", "draw", "away"]:
            direction_scores[d] += (ai_directions[d] / total_ai_votes) * 25

    # --- 信号6: 赔率变动方向 [权重7] ---
    change = match_obj.get("change", {})
    if change and isinstance(change, dict):
        try:
            cw = float(str(change.get("win", 0)).replace("+", "") or 0)
            cs = float(str(change.get("same", 0)).replace("+", "") or 0)
            cl = float(str(change.get("lose", 0)).replace("+", "") or 0)
            if cw < -0.05: direction_scores["home"] += 4
            if cs < -0.05: direction_scores["draw"] += 4
            if cl < -0.05: direction_scores["away"] += 4
            if cw > 0.05: direction_scores["home"] -= 2
            if cs > 0.05: direction_scores["draw"] -= 2
            if cl > 0.05: direction_scores["away"] -= 2
        except: pass

    # 归一化方向概率
    total_dir = sum(max(0.1, v) for v in direction_scores.values())
    dir_probs = {d: max(0.1, direction_scores[d]) / total_dir * 100 for d in direction_scores}
    final_direction = max(dir_probs, key=dir_probs.get)

    # 如果最高和次高差距<5%，标记为不确定
    sorted_dirs = sorted(dir_probs.items(), key=lambda x: x[1], reverse=True)
    dir_gap = sorted_dirs[0][1] - sorted_dirs[1][1]
    dir_confident = dir_gap > 5

    dir_labels = {"home": "主胜", "draw": "平局", "away": "客胜"}
    dir_icon = "✅" if dir_confident else "⚠️"
    print(f"    🎯 方向决策: 主{dir_probs['home']:.0f}% 平{dir_probs['draw']:.0f}% 客{dir_probs['away']:.0f}% → {dir_icon}{dir_labels[final_direction]}")

    # ================================================================
    # 第二层：进球数决策
    # ================================================================
    exp_goals = 2.5  # 默认值
    try:
        gp = []
        for gi, field in enumerate(["a0","a1","a2","a3","a4","a5","a6","a7"]):
            v = float(match_obj.get(field, 0) or 0)
            if v > 1: gp.append((gi, 1/v))
        if gp:
            tp = sum(p for _,p in gp)
            exp_goals = sum(g*(p/tp) for g,p in gp)
    except: pass

    # 最可能的进球数 = round(exp_goals)，但也考虑±1
    most_likely_goals = round(exp_goals)
    goal_range = [most_likely_goals]
    if most_likely_goals > 0: goal_range.append(most_likely_goals - 1)
    goal_range.append(most_likely_goals + 1)
    goal_range = sorted(set(g for g in goal_range if g >= 0))

    print(f"    ⚽ 进球决策: λ={exp_goals:.2f} → 最可能{most_likely_goals}球 范围{goal_range}")

    # ================================================================
    # 第三层：比分选择
    # 在方向+进球数范围内，对所有可能比分评分
    # ================================================================

    # 生成所有可能比分
    all_possible_scores = []
    for total_g in range(8):
        for h_g in range(total_g + 1):
            a_g = total_g - h_g
            all_possible_scores.append(f"{h_g}-{a_g}")

    # CRS赔率工具
    crs_key_map = {"1-0":"w10","0-1":"l01","2-1":"w21","1-2":"l12","2-0":"w20","0-2":"l02",
                   "0-0":"s00","1-1":"s11","3-0":"w30","3-1":"w31","0-3":"l03","1-3":"l13",
                   "2-2":"s22","3-2":"w32","2-3":"l23","4-0":"w40","4-1":"w41","0-4":"l04","1-4":"l14"}
    def get_crs_odds(score):
        key = crs_key_map.get(score, "")
        try: return float(match_obj.get(key, 99) or 99)
        except: return 99.0

    # CRS隐含概率
    crs_probs = {}
    crs_total = 0
    for score_str, key in crs_key_map.items():
        odds = get_crs_odds(score_str)
        if odds > 1 and odds < 200:
            crs_probs[score_str] = 1.0 / odds
            crs_total += 1.0 / odds
    if crs_total > 0:
        for k in crs_probs: crs_probs[k] = crs_probs[k] / crs_total * 100

    # 收集AI投票的比分
    ai_voted_scores = {}
    for name, r in {**p1_ai, "claude": claude_r}.items():
        if not isinstance(r, dict): continue
        sc = clean_score(r.get("ai_score", ""))
        if not sc:
            t3 = r.get("top3", [])
            if t3 and isinstance(t3, list) and len(t3) > 0:
                sc = clean_score(t3[0].get("score", ""))
        if sc:
            w = 1.5 if name == "claude" else (1.3 if name == "grok" else 1.0)
            ai_voted_scores[sc] = ai_voted_scores.get(sc, 0) + w
        # TOP2/TOP3
        t3 = r.get("top3", [])
        if isinstance(t3, list):
            for rank, t in enumerate(t3[1:3], 2):
                s2 = clean_score(t.get("score", ""))
                if s2:
                    w2 = (0.5 if name == "claude" else 0.3) if rank == 2 else 0.2
                    ai_voted_scores[s2] = ai_voted_scores.get(s2, 0) + w2

    # 对所有比分评分
    score_ratings = {}
    for score_str in all_possible_scores:
        try:
            h_g, a_g = map(int, score_str.split("-"))
        except: continue
        total_g = h_g + a_g

        s = 0.0

        # ① 方向参考 [15分] — 轻微nudge，不命令
        if final_direction == "home" and h_g > a_g:
            s += 15 * (dir_probs["home"] / 100)
        elif final_direction == "away" and h_g < a_g:
            s += 15 * (dir_probs["away"] / 100)
        elif final_direction == "draw" and h_g == a_g:
            s += 15 * (dir_probs["draw"] / 100)
        else:
            if h_g > a_g: s += 15 * (dir_probs["home"] / 100) * 0.5
            elif h_g < a_g: s += 15 * (dir_probs["away"] / 100) * 0.5
            else: s += 15 * (dir_probs["draw"] / 100) * 0.5

        # ② 进球数吻合 [20分] — 高斯衰减
        goal_diff = abs(total_g - exp_goals)
        s += round(20 * math.exp(-(goal_diff ** 2) / 1.5), 1)

        # ③ AI投票 [55分] — AI已经看过所有数据，它们的判断应该主导
        ai_vote = ai_voted_scores.get(score_str, 0)
        s += min(55, ai_vote * 12)

        # ⑤ 联赛风格 [10分]
        if any(lg in league for lg in ["德甲", "荷甲", "英超"]) and total_g >= 3: s += 10
        elif any(lg in league for lg in ["德甲", "荷甲"]) and total_g == 2: s += 5
        elif any(lg in league for lg in ["意甲", "法乙"]) and total_g <= 2: s += 10
        elif any(lg in league for lg in ["意甲"]) and h_g == a_g: s += 5  # 意甲平局加分
        elif any(lg in league for lg in ["英冠", "英甲"]) and 2 <= total_g <= 3: s += 6
        elif any(lg in league for lg in ["日职", "韩职", "澳超"]) and total_g >= 2: s += 5

        if s > 0:
            score_ratings[score_str] = round(s, 2)

    # 选出得分最高的比分
    if not score_ratings:
        final_score = engine_score
    else:
        ranked = sorted(score_ratings.items(), key=lambda x: x[1], reverse=True)
        final_score = ranked[0][0]

        # 打印TOP5
        print(f"    📊 比分评分: {' > '.join(f'{sc}({pts:.0f})' for sc, pts in ranked[:5])}")

    # ================================================================
    # 0-0特殊通道（仅极端信号）
    # ================================================================
    exp_analysis = stats.get("experience_analysis", {})
    zero_zero_boost = exp_analysis.get("zero_zero_boost", 0) if isinstance(exp_analysis, dict) else 0
    a0_val = float(match_obj.get("a0", 99) or 99)
    s00_val = float(match_obj.get("s00", 99) or 99)
    if zero_zero_boost >= 14 and a0_val < 7.5 and s00_val < 8.0:
        zero_rating = score_ratings.get("0-0", 0)
        top_rating = max(score_ratings.values()) if score_ratings else 0
        if zero_rating >= top_rating * 0.5:
            print(f"    🔒 0-0通道: boost={zero_zero_boost} → 采用0-0")
            final_score = "0-0"

    # ================================================================
    # 输出构建
    # ================================================================
    fhp = round(dir_probs["home"], 1)
    fdp = round(dir_probs["draw"], 1)
    fap = round(dir_probs["away"], 1)

    weights = {"claude": 1.4, "grok": 1.3, "gpt": 1.1, "gemini": 1.0}
    ai_conf_sum = 0; ai_conf_count = 0; value_kills = 0
    for name, r in {**p1_ai, "claude": claude_r}.items():
        if not isinstance(r, dict): continue
        conf = r.get("ai_confidence", 60)
        ai_conf_sum += conf * weights.get(name, 1.0)
        ai_conf_count += weights.get(name, 1.0)
        if r.get("value_kill"): value_kills += 1

    avg_ai_conf = (ai_conf_sum / ai_conf_count) if ai_conf_count > 0 else 60
    cf = engine_conf
    cf = min(95, cf + int((avg_ai_conf - 60) * 0.4))
    cf = cf + value_kills * 6
    # 方向不确定时降低信心
    if not dir_confident: cf = max(40, cf - 10)
    has_warn = any("🚨" in str(s) for s in smart_signals)
    if has_warn: cf = max(35, cf - 12)
    risk = "低" if cf >= 75 else ("中" if cf >= 55 else "高")

    gpt_sc = gpt_r.get("ai_score","-") if isinstance(gpt_r, dict) else "-"
    gpt_an = gpt_r.get("reason", gpt_r.get("analysis","N/A")) if isinstance(gpt_r, dict) else "N/A"
    grok_sc = grok_r.get("ai_score","-") if isinstance(grok_r, dict) else "-"
    grok_an = grok_r.get("reason", grok_r.get("analysis","N/A")) if isinstance(grok_r, dict) else "N/A"
    gem_sc = gemini_r.get("ai_score","-") if isinstance(gemini_r, dict) else "-"
    gem_an = gemini_r.get("reason", gemini_r.get("analysis","N/A")) if isinstance(gemini_r, dict) else "N/A"
    cl_sc = final_score
    cl_an = claude_r.get("reason", claude_r.get("analysis","N/A")) if isinstance(claude_r, dict) else "N/A"

    claude_score = clean_score(claude_r.get("ai_score", "")) if isinstance(claude_r, dict) else ""
    all_scores = []
    for name, r in p1_ai.items():
        if isinstance(r, dict):
            sc = r.get("ai_score", "-")
            if sc and sc not in ["-", "?"]: all_scores.append((sc, name))
    if claude_score: all_scores.append((claude_score, "claude"))

    pre_pred = {"home_win_pct": fhp, "draw_pct": fdp, "away_win_pct": fap, "steam_move": stats.get("steam_move", {}), "smart_signals": smart_signals, "line_movement_anomaly": stats.get("line_movement_anomaly", {})}
    cold_door = ColdDoorDetector.detect(match_obj, pre_pred)
    sigs = list(smart_signals)
    if cold_door["is_cold_door"]: sigs.extend(cold_door["signals"]); cf = max(30, cf - 5)

    return {
        "predicted_score": final_score, "home_win_pct": fhp, "draw_pct": fdp, "away_win_pct": fap,
        "confidence": cf, "risk_level": risk,
        "over_under_2_5": "大" if engine_result.get("over_25", 50) > 55 else "小",
        "both_score": "是" if engine_result.get("btts", 45) > 50 else "否",
        "gpt_score": gpt_sc, "gpt_analysis": gpt_an, "grok_score": grok_sc, "grok_analysis": grok_an,
        "gemini_score": gem_sc, "gemini_analysis": gem_an, "claude_score": cl_sc, "claude_analysis": cl_an,
        "ai_avg_confidence": round(avg_ai_conf, 1), "value_kill_count": value_kills,
        "model_agreement": len(set(sc for sc,_ in all_scores)) <= 1 and len(all_scores) >= 2,
        "poisson": stats.get("poisson", {}), "refined_poisson": stats.get("refined_poisson", {}),
        "extreme_warning": engine_result.get("scissors_gap_signal", ""),
        "smart_money_signal": " | ".join(sigs), "smart_signals": sigs,
        "model_consensus": stats.get("model_consensus", 0), "total_models": stats.get("total_models", 11),
        "expected_total_goals": exp_goals,
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
# run_predictions v3.5
# ====================================================================
def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 80)
    print(f"  [QUANT ENGINE vMAX 8.0] AI自主决策模式 | {len(ms)} 场比赛")
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
        print(f"  [TWO-PHASE] 启动两阶段AI架构...")
        start_t = time.time()
        all_ai = asyncio.run(run_ai_matrix_two_phase(match_analyses))
        print(f"  [AI MATRIX] 压榨完成，耗时 {time.time()-start_t:.1f}s")
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
    diary["reflection"] = f"vMAX8.0 | {cold_count}冷门 | AI自主决策·加权评分·无否决制"
    save_ai_diary(diary)
    return res, t4