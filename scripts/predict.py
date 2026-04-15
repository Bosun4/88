import json
import os
import re
import time
import asyncio
import aiohttp
import numpy as np
import math
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
    def apply_odds_history(m, mg): 
        return mg

try:
    from quant_edge import apply_quant_edge
except Exception as e:
    print(f"  [WARN] ⚠️ 量化边缘模块 (quant_edge) 加载失败或未找到，系统自动降级跳过: {e}")
    def apply_quant_edge(m, mg): 
        return mg

try:
    from wencai_intel import apply_wencai_intel
except:
    def apply_wencai_intel(m, mg): 
        return mg

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
    return {
        "ev": round(ev * 100, 2), 
        "kelly": round(max(0.0, kelly * 0.25) * 100, 2), 
        "is_value": ev > 0.05
    }

def parse_score(s):
    try:
        s = str(s).strip()
        s = s.replace(" ", "")
        s = s.replace("：", "-")
        s = s.replace(":", "-")
        s = s.replace("\u2013", "-")
        s = s.replace("\u2014", "-")
        p = s.split("-")
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
            signals.append("❄️ 反向Steam！庄家造热收割")
            strength += 5
            
        vote = match.get("vote", {})
        try:
            vh = int(vote.get("win", 33))
            va = int(vote.get("lose", 33))
            max_vote = max(vh, va)
            if max_vote >= 65: 
                signals.append(f"❄️ 散户极端偏向{max_vote}%！冷门高危")
                strength += 4
            elif max_vote >= 58: 
                strength += 2
        except: 
            pass
            
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
            hp2 = prediction.get("home_win_pct", 50)
            implied_h = 100 / sp_h * 0.92
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
        if strength >= 12:
            level = "顶级"
        elif strength >= 7:
            level = "高危"
        else:
            level = "普通"
            
        return {
            "is_cold_door": is_cold, 
            "strength": strength, 
            "level": level, 
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
    return {"yesterday_win_rate": "N/A", "reflection": "持续进化中", "kill_history": []}

def save_ai_diary(diary):
    os.makedirs("data", exist_ok=True)
    with open("data/ai_diary.json", "w", encoding="utf-8") as f:
        json.dump(diary, f, ensure_ascii=False, indent=2)


# ====================================================================
# 🧠 两阶段AI架构 vMAX 14.0 奇点版
# ====================================================================
def build_phase1_prompt(match_analyses):
    diary = load_ai_diary()
    p = "【身份】你是管理50亿美金体育基金的首席量化分析师。你的工作是结合亚盘与大小球交叉方程，找出被市场错误定价的真实比分。\n\n"
    if diary.get("reflection"):
        p += f"【进化日志】{diary.get('yesterday_win_rate','N/A')} | {diary['reflection']}\n\n"

    p += "【输出格式】只输出合法JSON数组。\n"
    p += "每场：match(整数), top3([{score,prob},...]), reason(100-150字含3+赔率数字), ai_confidence(0-100)。\n"
    p += 'top3中prob是该比分概率%。示例: {"match":1,"top3":[{"score":"2-1","prob":14.5},{"score":"3-1","prob":11.8},{"score":"1-2","prob":10.2}],"reason":"...","ai_confidence":75}\n\n'

    p += "【量化分析框架——按权重排序】\n"
    p += "① [权重35%] 庄家锚点方程(The Bookie Anchor)：结合大小球与亚盘让球解算出的真实比分坐标。\n"
    p += "② [权重25%] 联赛DNA+赛季阶段：大球联赛（如荷甲、英超、澳超）要果断给 3-1, 2-2；生死战或淘汰赛严禁给出 1-1。\n"
    p += "③ [权重20%] CRS波胆赔率矩阵→概率分布（寻找庄家刻意压低或抬高的长尾波胆异动）。\n"
    p += "④ [权重10%] 亚盘+欧赔交叉验证→裂角侦测。\n"
    p += "⑤ [权重10%] 散户情绪与资金流。\n\n"
    p += "【反常识思维】不要被 AI 的 1-1/1-0 保守幻觉控制！强队客场让球极深时，不要选 1-1；高期望进球局严禁零封！\n\n"

    p += "【原始数据与高维锚点】\n"
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
        p += f"欧赔: {sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f} | 让球: {hc}\n"

        if sp_h > 1 and sp_d > 1 and sp_a > 1:
            margin = 1/sp_h + 1/sp_d + 1/sp_a
            p += f"Shin概率: 主{(1/sp_h)/margin*100:.1f}% 平{(1/sp_d)/margin*100:.1f}% 客{(1/sp_a)/margin*100:.1f}% | 返还率{1/margin*100:.1f}%\n"

        if m.get("hhad_win"): p += f"让球胜平负: {m['hhad_win']}/{m.get('hhad_same','')}/{m.get('hhad_lose','')}\n"

        a0=m.get("a0",""); a1=m.get("a1","")
        eg = 2.5
        if a0:
            a2=m.get("a2",""); a3=m.get("a3",""); a4=m.get("a4",""); a5=m.get("a5",""); a6=m.get("a6",""); a7=m.get("a7","")
            p += f"总进球: 0球={a0}|1={a1}|2={a2}|3={a3}|4={a4}|5={a5}|6={a6}|7+={a7}\n"
            try:
                gp=[(gi,1/float(v)) for gi,v in enumerate([a0,a1,a2,a3,a4,a5,a6,a7]) if float(v)>1]
                tp=sum(p2 for _,p2 in gp)
                eg=sum(g*(p2/tp) for g,p2 in gp)
                p += f"→ 期望进球λ={eg:.2f}\n"
            except: pass

        # v14.0 注入庄家锚点
        try:
            handicap_val = float(str(hc).replace("+", ""))
            anchor_h_g = max(0.1, (eg - handicap_val) / 2.0)
            anchor_a_g = max(0.1, (eg + handicap_val) / 2.0)
            p += f"🔥 庄家绝对锚点坐标: 主队xG = {anchor_h_g:.2f}, 客队xG = {anchor_a_g:.2f} (这是大小球与让球交叉解算的终极答案，请围绕此坐标发散！)\n"
        except: pass

        crs_map = {"w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","w32":"3-2","w40":"4-0","w41":"4-1","w42":"4-2",
                   "s00":"0-0","s11":"1-1","s22":"2-2","s33":"3-3",
                   "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3","l23":"2-3","l04":"0-4","l14":"1-4"}
        crs_lines=[]
        for key,score in crs_map.items():
            try:
                odds=float(m.get(key,0) or 0)
                if odds>1: crs_lines.append(f"{score}={odds:.1f}")
            except: pass
        if crs_lines: p += f"CRS: {' | '.join(crs_lines)}\n"

        vote=m.get("vote",{})
        if vote:
            p += f"散户: 胜{vote.get('win','?')}% 平{vote.get('same','?')}% 负{vote.get('lose','?')}%"
            if max(int(vote.get('win',33)), int(vote.get('lose',33))) >= 60: p += " (大热预警)"
            p += "\n"

        change=m.get("change",{})
        if change and isinstance(change,dict):
            cw=change.get("win",0); cs=change.get("same",0); cl=change.get("lose",0)
            if cw or cs or cl: p += f"赔率变动: 胜{cw} 平{cs} 负{cl}\n"

        hs=m.get("home_stats",{}); ast2=m.get("away_stats",{})
        if hs.get("form"):
            p += f"主队: 近况{hs.get('form','?')} 进{hs.get('avg_goals_for','?')}/失{hs.get('avg_goals_against','?')}\n"
            p += f"客队: 近况{ast2.get('form','?')} 进{ast2.get('avg_goals_for','?')}/失{ast2.get('avg_goals_against','?')}\n"
        p += "\n"

    p += "\n【特别要求 - 核心指令】\n"
    p += "1. 如果庄家锚点坐标指向单方进球>1.5，强行放弃 1-0/0-1/1-1，必须给出带 2 或 3 的高赔比分。\n"
    p += "2. 严重警告：当锚点坐标双方进球相加>2.8时，即使主队极大优，也请果断给出 3-1, 2-2, 3-2！\n"
    p += f"【输出{len(match_analyses)}场JSON数组，只输出数组！】\n"
    return p

def build_phase2_prompt(match_analyses, phase1_results):
    p = "【你是最终裁判】三个独立AI已各自给出TOP3候选比分。你综合分析选出最终比分。\n\n"
    p += "【强引导原则】\n"
    p += "① 参照下方给出的【庄家锚点】，任何严重偏离锚点的选项直接否决！\n"
    p += "② 若其他AI给出 3-1/2-3/1-3 等高赔比分，且锚点支持大球，你必须跟随放大，禁止回撤防守 1-1！\n\n"
    p += "【输出格式】JSON数组：match(整数), score(最终比分), reason(80-120字含逻辑), ai_confidence(0-100)\n\n"

    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        league = m.get("league", m.get("cup", ""))
        hc = m.get("give_ball", "0")
        idx = i + 1

        p += f"{'='*50}\n[{idx}] {h} vs {a} | {league}\n"
        try:
            gp = []
            for gi, field in enumerate(["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]):
                v = float(m.get(field, 0) or 0)
                if v > 1: gp.append((gi, 1/v))
            if gp:
                tp = sum(p2 for _, p2 in gp)
                eg = sum(g * (p2/tp) for g, p2 in gp)
                handicap_val = float(str(hc).replace("+", ""))
                ahg = max(0.1, (eg - handicap_val) / 2.0)
                aag = max(0.1, (eg + handicap_val) / 2.0)
                p += f"⭐庄家绝对锚点坐标: 主{ahg:.2f} - 客{aag:.2f} (这是你裁决的唯一神圣坐标！)\n"
        except: pass

        for ai_name in ["gpt", "grok", "gemini"]:
            ai_data = phase1_results.get(ai_name, {}).get(idx, {})
            if not ai_data: continue
            top3 = ai_data.get("top3", [])
            if top3:
                scores_str = " | ".join(f"{t.get('score', '?')}({t.get('prob', '?')}%)" for t in top3[:3])
                p += f"  {ai_name.upper()}: {scores_str} | 信心{ai_data.get('ai_confidence', '?')}\n"
            else:
                sc = ai_data.get("ai_score", "-")
                p += f"  {ai_name.upper()}: {sc} | 信心{ai_data.get('ai_confidence', '?')}\n"
        p += "\n"

    p += f"【输出{len(match_analyses)}场JSON数组。大胆选出贴合锚点的高价值比分！只输出数组！】\n"
    return p

# ====================================================================
# AI调用引擎 (防暴毙，100%全量保留)
# ====================================================================
FALLBACK_URLS = [None, "https://api520.pro/v1", "https://api521.pro/v1", "https://api522.pro/v1", "https://www.api522.pro/v1"]

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
    backup = [u for u in FALLBACK_URLS if u and u != primary_url][:1]
    urls = [primary_url] + backup

    CONNECT_TIMEOUT = 20
    READ_TIMEOUT = 400

    AI_PROFILES = {
        "claude": {"sys": "你是最终裁判。必须严格计算锚点距离。只输出JSON数组。", "temp": 0.20},
        "grok": {"sys": "你是Grok，具备实时联网能力。结合亚盘交叉方程找出真实比分。只输出JSON数组。", "temp": 0.28},
        "gpt": {"sys": "你是激进派量化分析师。不要被表面低赔率误导。只输出JSON数组。", "temp": 0.25},
        "gemini": {"sys": "你是概率引擎。跟随庄家锚点输出，不要人为压向低赔。只输出JSON数组。", "temp": 0.20},
    }

    profile = AI_PROFILES.get(ai_name, AI_PROFILES["gpt"])

    for mn in models_list:
        connected = False
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
                    "contents":[{"parts":[{"text":prompt}]}],
                    "generationConfig":{"temperature":profile["temp"]},
                    "systemInstruction":{"parts":[{"text":profile["sys"]}]}
                }
            else:
                headers["Authorization"] = f"Bearer {key}"
                bp = {
                    "model": mn,
                    "messages": [
                        {"role": "system", "content": profile["sys"]},
                        {"role": "user", "content": prompt}
                    ]
                }
                if ai_name != "claude": 
                    bp["temperature"] = profile["temp"]
                payload = bp

            gw = url.split("/v1")[0][:35]
            print(f"  [🔌连接中] {ai_name.upper()} | {mn[:22]} @ {gw}")
            t0 = time.time()

            try:
                timeout = aiohttp.ClientTimeout(total=None, connect=CONNECT_TIMEOUT, sock_connect=CONNECT_TIMEOUT, sock_read=READ_TIMEOUT)
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                    elapsed_connect = round(time.time()-t0, 1)

                    if r.status in (502, 504, 400, 429):
                        print(f"    💀 HTTP {r.status} | {elapsed_connect}s → 换模型")
                        break
                    if r.status != 200: 
                        continue

                    connected = True
                    try: 
                        data = await r.json(content_type=None)
                    except: 
                        elapsed = round(time.time()-t0, 1)
                        print(f"    ⚠️ 响应非JSON | {elapsed}s → 换模型")
                        break

                    elapsed = round(time.time()-t0, 1)
                    usage = data.get("usage", {})
                    req_tokens = usage.get("total_tokens", 0) or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
                    if not req_tokens:
                        um = data.get("usageMetadata", {})
                        req_tokens = um.get("totalTokenCount", 0)
                    if req_tokens:
                        print(f"    📊 消耗: {req_tokens:,} token | 耗时: {elapsed}s")

                    raw_text = ""
                    try:
                        if is_gem: 
                            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        else:
                            if data.get("choices") and data["choices"]:
                                msg = data["choices"][0].get("message", {})
                                if isinstance(msg, dict):
                                    best_with_bracket = ""
                                    best_any = ""
                                    for key in msg:
                                        val = msg[key]
                                        if not val or not isinstance(val, str): continue
                                        val = val.strip()
                                        if not val: continue
                                        if "[" in val and "{" in val and len(val) > len(best_with_bracket):
                                            best_with_bracket = val
                                        if len(val) > len(best_any):
                                            best_any = val
                                    raw_text = best_with_bracket or best_any

                            if not raw_text and data.get("output") and isinstance(data["output"], list):
                                for out_item in data["output"]:
                                    if isinstance(out_item, dict) and out_item.get("type") == "message":
                                        for ct in out_item.get("content", []):
                                            if isinstance(ct, dict) and ct.get("text"):
                                                txt = ct["text"].strip()
                                                if len(txt) > len(raw_text):
                                                    raw_text = txt

                            if not raw_text:
                                full_str = json.dumps(data, ensure_ascii=False)
                                m_match = re.search(r'\[\s*\{\s*"match"', full_str)
                                if m_match:
                                    start_pos = m_match.start()
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
                                        if '\\"' in extracted:
                                            try: extracted = json.loads('"' + extracted + '"')
                                            except: extracted = extracted.replace('\\"', '"')
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

                    clean = raw_text
                    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", clean, flags=re.DOTALL|re.IGNORECASE)
                    clean = re.sub(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", "", clean, flags=re.DOTALL)
                    clean = re.sub(r"```[\w]*","",clean).strip()
                    start = clean.find("[")
                    end = clean.rfind("]")+1
                    if start == -1 or end == 0:
                        clean = re.sub(r"[^\[\]{}:,\"'0-9a-zA-Z\u4e00-\u9fa5\s\.\-\+\(\)]","",clean)
                        start = clean.find("[")
                        end = clean.rfind("]")+1

                    results = {}
                    if start != -1 and end > start:
                        try:
                            arr = json.loads(clean[start:end])
                            for item in arr:
                                if not item.get("match"): continue
                                try: mid = int(item["match"])
                                except: continue
                                if item.get("top3"):
                                    t1 = item["top3"][0].get("score","1-1").replace(" ","").strip() if item["top3"] else "1-1"
                                    results[mid] = {"top3": item["top3"], "ai_score": t1, "reason": str(item.get("reason",""))[:200], "ai_confidence": int(item.get("ai_confidence",60))}
                                elif item.get("score"):
                                    results[mid] = {"ai_score": item["score"].replace(" ","").strip(), "analysis": str(item.get("reason",""))[:200], "ai_confidence": int(item.get("ai_confidence",60)), "value_kill": bool(item.get("value_kill",False))}
                        except json.JSONDecodeError:
                            try:
                                last_brace = clean[start:end].rfind('}')
                                if last_brace != -1:
                                    arr = json.loads(clean[start:end][:last_brace+1] + "]")
                                    print(f"    🩹 断肢重生: 抢救 {len(arr)} 条")
                                    for item in arr:
                                        if not item.get("match"): continue
                                        try: mid = int(item["match"])
                                        except: continue
                                        if item.get("top3"):
                                            t1 = item["top3"][0].get("score","1-1").replace(" ","").strip() if item["top3"] else "1-1"
                                            results[mid] = {"top3": item["top3"], "ai_score": t1, "reason": str(item.get("reason",""))[:200], "ai_confidence": int(item.get("ai_confidence",60))}
                                        elif item.get("score"):
                                            results[mid] = {"ai_score": item["score"].replace(" ","").strip(), "analysis": str(item.get("reason",""))[:200], "ai_confidence": int(item.get("ai_confidence",60)), "value_kill": bool(item.get("value_kill",False))}
                            except: pass

                    if len(results) > 0:
                        print(f"    ✅ {ai_name.upper()} 完成: {len(results)}/{num_matches} | {elapsed}s ({mn[:20]})")
                        return ai_name, results, mn
                    else: break
            except Exception as e:
                if connected: return ai_name, {}, "error"
                continue
            await asyncio.sleep(0.2)
    return ai_name, {}, "failed"

async def run_ai_matrix_two_phase(match_analyses):
    num = len(match_analyses)
    p1_prompt = build_phase1_prompt(match_analyses)
    print(f"  [Phase1] {len(p1_prompt):,} 字符 → GPT/Grok/Gemini 并行...")

    p1_configs = [
        ("grok","GROK_API_URL","GROK_API_KEY",["熊猫-A-6-grok-4.2-thinking"]),
        ("gpt","GPT_API_URL","GPT_API_KEY",["熊猫-按量-gpt-5.4"]),
        ("gemini","GEMINI_API_URL","GEMINI_API_KEY",["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking"]),
    ]
    p1_results = {"gpt":{},"grok":{},"gemini":{}}

    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [async_call_one_ai_batch(session,p1_prompt,u,k,m,num,n) for n,u,k,m in p1_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res,tuple): 
                n, d, _ = res
                p1_results[n] = d

        p2_prompt = build_phase2_prompt(match_analyses, p1_results)
        print(f"  [Phase2] {len(p2_prompt):,} 字符 → Claude 裁判...")

        claude_r = {}
        _, claude_r, _ = await async_call_one_ai_batch(
            session, p2_prompt, "CLAUDE_API_URL","CLAUDE_API_KEY",
            ["熊猫特供-超纯满血-99额度-claude-opus-4.6-thinking","熊猫-按量-特供顶级-官方正向满血-claude-opus-4.6-thinking"], num, "claude"
        )
    all_r = p1_results.copy(); all_r["claude"] = claude_r
    return all_r


# ====================================================================
# 🌟 vMAX 14.0 奇点引擎：盘口链式锚点 + 动态参数 + 全局解耦
# ====================================================================
def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    import math
    league = str(match_obj.get("league", match_obj.get("cup", "")))
    sp_h = float(match_obj.get("sp_home", match_obj.get("win", 0)) or 0)
    sp_d = float(match_obj.get("sp_draw", match_obj.get("same", 0)) or 0)
    sp_a = float(match_obj.get("sp_away", match_obj.get("lose", 0)) or 0)
    
    # 【v14.0】提取亚盘让球
    try:
        hc_str = str(match_obj.get("give_ball", "0")).replace("+", "")
        handicap = float(hc_str)
    except:
        handicap = 0.0

    engine_conf = engine_result.get("confidence", 50)
    p1_ai = {"gpt": gpt_r, "grok": grok_r, "gemini": gemini_r}

    # ================================================================
    # 1. 第一层：基本面方向与聪明钱探测
    # ================================================================
    direction_scores = {"home": 0.0, "draw": 0.0, "away": 0.0}

    if sp_h > 1 and sp_d > 1 and sp_a > 1:
        margin = 1/sp_h + 1/sp_d + 1/sp_a
        shin_h = (1/sp_h) / margin * 100
        shin_d = (1/sp_d) / margin * 100
        shin_a = (1/sp_a) / margin * 100
        direction_scores["home"] += shin_h / 100 * 25
        direction_scores["draw"] += shin_d / 100 * 25
        direction_scores["away"] += shin_a / 100 * 25
    else:
        direction_scores["home"] += 8.3
        direction_scores["draw"] += 8.3
        direction_scores["away"] += 8.3

    smart_signals = stats.get("smart_signals", [])
    smart_str = " ".join(str(s) for s in smart_signals)
    sharp_detected = False

    if "Sharp" in smart_str or "sharp" in smart_str:
        sharp_detected = True
        if "客胜" in smart_str or "客队" in smart_str: direction_scores["away"] += 25
        elif "主胜" in smart_str or "主队" in smart_str: direction_scores["home"] += 25
        elif "平局" in smart_str or "平赔" in smart_str: direction_scores["draw"] += 25
        else: direction_scores["home"] += 8; direction_scores["draw"] += 8; direction_scores["away"] += 9
            
    if "Steam" in smart_str:
        if "客胜Steam" in smart_str or "客胜反向" in smart_str: direction_scores["away"] += 15
        elif "主胜Steam" in smart_str or "主胜反向" in smart_str: direction_scores["home"] += 15
        elif "平局Steam" in smart_str: direction_scores["draw"] += 15

    vote = match_obj.get("vote", {})
    try:
        vh = int(vote.get("win", 33) or 33)
        vd = int(vote.get("same", 33) or 33)
        va = int(vote.get("lose", 33) or 33)
        max_vote = max(vh, vd, va)
        if max_vote >= 55:
            if vh == max_vote:
                contrarian_weight = min(15, (vh - 50) * 0.75)
                direction_scores["away"] += contrarian_weight * 0.6
                direction_scores["draw"] += contrarian_weight * 0.4
            elif va == max_vote:
                contrarian_weight = min(15, (va - 50) * 0.75)
                direction_scores["home"] += contrarian_weight * 0.6
                direction_scores["draw"] += contrarian_weight * 0.4
    except: pass

    change = match_obj.get("change", {})
    if change and isinstance(change, dict):
        try:
            cw = float(str(change.get("win", 0)).replace("+", "") or 0)
            cs = float(str(change.get("same", 0)).replace("+", "") or 0)
            cl = float(str(change.get("lose", 0)).replace("+", "") or 0)
            if cw < -0.05: direction_scores["home"] += 5
            if cl < -0.05: direction_scores["away"] += 5
            if cw > 0.05: direction_scores["home"] -= 3
            if cl > 0.05: direction_scores["away"] -= 3
        except: pass

    # 凯利双爆·斩杀平局
    is_chaos_game = False
    if "凯利异常" in smart_str or "双极分布" in smart_str or ("分胜负" in smart_str):
        direction_scores["draw"] *= 0.6  
        if direction_scores["home"] > direction_scores["away"]: direction_scores["home"] *= 1.2
        else: direction_scores["away"] *= 1.2

    total_dir = sum(max(0.1, v) for v in direction_scores.values())
    dir_probs = {d: max(0.1, direction_scores[d]) / total_dir * 100 for d in direction_scores}


    # ================================================================
    # 2. 第二层：前置冷门探测与基础 xG 抓取
    # ================================================================
    pre_pred_for_cold = {
        "home_win_pct": dir_probs["home"], 
        "draw_pct": dir_probs["draw"], 
        "away_win_pct": dir_probs["away"],
        "steam_move": stats.get("steam_move", {}),
        "smart_signals": smart_signals,
        "line_movement_anomaly": stats.get("line_movement_anomaly", {})
    }
    cold_door_info = ColdDoorDetector.detect(match_obj, pre_pred_for_cold)

    # 提取引擎级别的精准期望进球 (λ)
    exp_goals = float(engine_result.get("expected_total_goals", stats.get("expected_total_goals", 0)))
    if exp_goals <= 0:
        try:
            gp = []
            for gi, field in enumerate(["a0","a1","a2","a3","a4","a5","a6","a7"]):
                v = float(match_obj.get(field, 0) or 0)
                if v > 1: gp.append((gi, 1/v))
            if gp:
                tp = sum(p for _,p in gp)
                exp_goals = sum(g*(p/tp) for g,p in gp)
        except: 
            exp_goals = 2.5

    # 【v14.0 核心突破：庄家绝对锚点坐标】 (AH-OU Interlock)
    anchor_h_g = max(0.1, (exp_goals - handicap) / 2.0)
    anchor_a_g = max(0.1, (exp_goals + handicap) / 2.0)
    print(f"    📐 盘口链式解算: 庄家绝对锚点比分锁定在 主 {anchor_h_g:.2f} - 客 {anchor_a_g:.2f}")

    home_xg_implied = float(engine_result.get("bookmaker_implied_home_xg", 1.3))
    away_xg_implied = float(engine_result.get("bookmaker_implied_away_xg", 0.9))
    
    # 大盘与欧赔的裂角侦测
    xg_gap = exp_goals - (home_xg_implied + away_xg_implied)
    
    home_xg = home_xg_implied
    away_xg = away_xg_implied

    a1_odds = float(match_obj.get("a1", 99) or 99)
    a3_odds = float(match_obj.get("a3", 99) or 99)
    
    league_xg_multiplier = {
        "英超": 1.15, "德甲": 1.20, "荷甲": 1.25, "荷乙": 1.28, 
        "挪超": 1.25, "瑞典超": 1.15, "澳超": 1.22, "美职": 1.18, "瑞士超": 1.20, "奥甲": 1.18,
        "意甲": 0.95, "意乙": 0.82, "法乙": 0.85, "西乙": 0.85, "阿甲": 0.88, "巴甲": 0.90,
        "默认": 1.0
    }
    multi = league_xg_multiplier.get("默认")
    for key in league_xg_multiplier:
        if key in league:
            multi = league_xg_multiplier[key]
            break
            
    if a3_odds > 1.0 and a1_odds > 1.0 and a3_odds < a1_odds * 0.95:
        if multi < 1.0:
            multi = 1.12 
            
    home_xg *= multi
    away_xg *= multi

    current_sum = home_xg + away_xg
    if current_sum > 0 and exp_goals > 0:
        home_xg = (home_xg / current_sum) * exp_goals
        away_xg = (away_xg / current_sum) * exp_goals

    # 解耦“聪明钱”与“散户大热”
    if cold_door_info["is_cold_door"]:
        market_hot_side = "home" if sp_h < sp_a else "away"
        try:
            vh = int(match_obj.get("vote", {}).get("win", 33))
            va = int(match_obj.get("vote", {}).get("lose", 33))
            if vh - va > 15: market_hot_side = "home"
            elif va - vh > 15: market_hot_side = "away"
        except: pass

        is_fake_cold = False
        if sharp_detected:
            if ("主胜" in smart_str or "主队" in smart_str) and market_hot_side == "home":
                is_fake_cold = True
            elif ("客胜" in smart_str or "客队" in smart_str) and market_hot_side == "away":
                is_fake_cold = True

        if not is_fake_cold:
            print(f"    💥 大热必死启动！市场真实热方为: {market_hot_side}。剥夺期望进球...")
            if market_hot_side == "home": 
                home_xg *= 0.60; away_xg *= 1.40
                dir_probs["home"] *= 0.4; dir_probs["away"] *= 2.0; dir_probs["draw"] *= 1.1
            else: 
                away_xg *= 0.60; home_xg *= 1.40
                dir_probs["away"] *= 0.4; dir_probs["home"] *= 2.0; dir_probs["draw"] *= 1.1
            
            total_dir2 = sum(dir_probs.values())
            dir_probs = {k: v/total_dir2*100 for k,v in dir_probs.items()}

    # 随机微扰
    home_xg += np.random.normal(0, 0.12); away_xg += np.random.normal(0, 0.11)
    home_xg = max(0.5, home_xg); away_xg = max(0.5, away_xg)

    final_direction = max(dir_probs, key=dir_probs.get)
    dir_gap = dir_probs[final_direction] - sorted(dir_probs.values(), reverse=True)[1]
    dir_confident = dir_gap > 5

    # ================================================================
    # 3. 第三层：动态双变量泊松分布 (v14.0 Dynamic Dixon-Coles)
    # ================================================================
    def dixon_coles_correction(hg, ag, score_str, lambda_total):
        if score_str in ["0-0", "1-0", "0-1", "1-1"]:
            if lambda_total <= 2.2: return 1.25
            elif lambda_total >= 3.0: return 0.90 # 大球局取消低比分保护
            return 1.12
        elif score_str in ["2-2", "3-1", "1-3", "3-2", "3-3", "2-3", "4-1", "1-4"]:
            if lambda_total >= 3.0: return 1.20 # 大球局反向奖励高比分
            return 0.88
        return 1.0

    poisson_scores = {}
    for h_g in range(6):
        p_h = math.exp(-home_xg) * (home_xg ** h_g) / math.factorial(h_g)
        for a_g in range(6):
            p_a = math.exp(-away_xg) * (away_xg ** a_g) / math.factorial(a_g)
            score_str = f"{h_g}-{a_g}"
            tau = dixon_coles_correction(home_xg, away_xg, score_str, exp_goals)
            prob = p_h * p_a * tau * 100
            poisson_scores[score_str] = round(prob, 2)

    # ================================================================
    # 4. 第四层：打分系统与风控一票否决权
    # ================================================================
    # 解放 AI 投票权重
    weights = {"claude": 0.85, "grok": 1.45, "gpt": 0.95, "gemini": 1.35}
    ai_voted_scores = {}
    
    for name, r in {**p1_ai, "claude": claude_r}.items():
        if not isinstance(r, dict): continue
        sc_raw = r.get("ai_score", "") or (r.get("top3", [{}])[0].get("score", "") if r.get("top3") else None)
        sc = parse_score(sc_raw)
        if sc and sc[0] is not None and sc[1] is not None:
            score_str = f"{sc[0]}-{sc[1]}"
            w = weights.get(name, 1.0)
            ai_voted_scores[score_str] = ai_voted_scores.get(score_str, 0) + w * 2.0 
        for t in r.get("top3", [])[1:3]:
            s2 = parse_score(t.get("score", ""))
            if s2 and s2[0] is not None and s2[1] is not None:
                w2 = weights.get(name, 1.0) * 0.5
                ai_voted_scores[f"{s2[0]}-{s2[1]}"] = ai_voted_scores.get(f"{s2[0]}-{s2[1]}", 0) + w2

    score_ratings = {}
    
    historical_penalty = {
        "1-0": -3, "0-1": -3, "0-0": -5, 
        "0-2": -1, "2-0": -1, "2-1": +1, "1-2": +1,
        "2-2": +6, "3-1": +8, "1-3": +8, 
        "3-0": +5, "0-3": +5, "3-2": +10, "2-3": +10, "4-1": +12
    }

    if max(dir_probs.values()) < 42.0:
        is_chaos_game = True

    is_away_favorite_trap = False
    if sp_a < 1.90 and sp_d < 4.0: 
        is_away_favorite_trap = True

    is_do_or_die = any(kw in smart_str for kw in ["保级", "死拼", "争冠", "生死", "淘汰", "晋级", "决", "杯"])

    for score_str, poisson_prob in poisson_scores.items():
        try: h_g, a_g = map(int, score_str.split("-"))
        except: continue
        total_g = h_g + a_g
        goal_margin = h_g - a_g
        s = 0.0
        
        # 泊松基准分 (砍半，降低独立模型的话语权)
        s += poisson_prob * 4.0 
        
        if final_direction == "home" and h_g > a_g: s += 20 * (dir_probs["home"] / 100)
        elif final_direction == "away" and h_g < a_g: s += 20 * (dir_probs["away"] / 100)
        elif final_direction == "draw" and h_g == a_g: s += 20 * (dir_probs["draw"] / 100)
        
        # 【v14.0 核心武器：庄家锚点高斯引力】
        anchor_dist = math.sqrt((h_g - anchor_h_g)**2 + (a_g - anchor_a_g)**2)
        s += 45 * math.exp(-(anchor_dist ** 2) / 0.8)

        # AI投票无封印注入
        s += ai_voted_scores.get(score_str, 0) * 12
        s += historical_penalty.get(score_str, 0)

        # 规则A：动态 1-1 惩罚
        if score_str == "1-1":
            if any(lg in league for lg in ["英超", "英冠", "荷甲", "荷乙", "德甲", "美职", "挪超", "澳超"]):
                s -= 8  
            elif any(lg in league for lg in ["西甲", "意甲", "法乙"]):
                s += 3  

        # 规则B：高 xG 强制防零封
        if exp_goals >= 3.0 and (h_g == 0 or a_g == 0):
            s -= 15
            if exp_goals >= 3.5: s -= 25 
            
        # 规则C：混沌局强制互捅
        if is_chaos_game:
            if h_g == 0 or a_g == 0: s -= 10 
            elif total_g >= 3 and h_g > 0 and a_g > 0: s += 12 

        # 【规则1：亚盘引力约束 (防走水)】
        if handicap <= -1.0: 
            if goal_margin == 1 and not cold_door_info["is_cold_door"]: s -= 10 
            elif goal_margin >= 2: s += 12 
        elif handicap >= 1.0: 
            if goal_margin == -1 and not cold_door_info["is_cold_door"]: s -= 10 
            elif goal_margin <= -2: s += 12 

        # 【规则2：生死战非对称抹平】
        if is_do_or_die and h_g == a_g: s -= 15 

        # 【规则3：大盘与欧赔裂角侦测器】
        if xg_gap > 0.4: 
            if total_g >= 3 and h_g > 0 and a_g > 0: s += 15 
        elif xg_gap < -0.4: 
            if total_g <= 1: s += 15 

        # 【规则4：经验风控一票否决(The Portsmouth Fix)】
        if "客队盘口太便宜" in smart_str or "长客陷阱" in smart_str or "诱盘" in smart_str:
            if h_g < a_g: 
                s -= 50 # 暴力扣除客胜 50 分，强拉主队

        if multi > 1.10 and total_g >= 3: s += 8
        if is_away_favorite_trap:
            if score_str == "1-1": s += 12 
            elif score_str == "1-0": s += 8  
        if exp_goals > 2.8 and abs(h_g - a_g) >= 3: s += 5 

        if s > 0: 
            score_ratings[score_str] = round(s, 2)

    ranked = sorted(score_ratings.items(), key=lambda x: x[1], reverse=True)
    final_score = ranked[0][0] if ranked else "1-1"
    
    print(f"    📊 终极引力打分: {' > '.join(f'{sc}({pts:.0f})' for sc, pts in ranked[:6])}")

    # ================================================================
    # 0-0特殊通道（仅极端信号保留）
    # ================================================================
    exp_analysis = stats.get("experience_analysis", {})
    zero_zero_boost = exp_analysis.get("zero_zero_boost", 0) if isinstance(exp_analysis, dict) else 0
    a0_val = float(match_obj.get("a0", 99) or 99)
    s00_val = float(match_obj.get("s00", 99) or 99)
    if zero_zero_boost >= 14 and a0_val < 7.5 and s00_val < 8.0:
        zero_rating = score_ratings.get("0-0", 0)
        top_rating = max(score_ratings.values()) if score_ratings else 0
        if zero_rating >= top_rating * 0.5:
            final_score = "0-0"

    # ================================================================
    # 输出构建
    # ================================================================
    fhp = round(dir_probs["home"], 1)
    fdp = round(dir_probs["draw"], 1)
    fap = round(dir_probs["away"], 1)

    ai_conf_sum = 0
    ai_conf_count = 0
    value_kills = 0
    
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
    
    if not dir_confident: cf = max(40, cf - 10)
    has_warn = any("🚨" in str(s) for s in smart_signals)
    if has_warn: cf = max(35, cf - 12)
        
    risk = "低" if cf >= 75 else ("中" if cf >= 55 else "高")

    # 【停止对 Claude 的“精神控制”】直接输出真实预测
    gpt_sc = gpt_r.get("ai_score","-") if isinstance(gpt_r, dict) else "-"
    gpt_an = gpt_r.get("reason", gpt_r.get("analysis","N/A")) if isinstance(gpt_r, dict) else "N/A"
    grok_sc = grok_r.get("ai_score","-") if isinstance(grok_r, dict) else "-"
    grok_an = grok_r.get("reason", grok_r.get("analysis","N/A")) if isinstance(grok_r, dict) else "N/A"
    gem_sc = gemini_r.get("ai_score","-") if isinstance(gemini_r, dict) else "-"
    gem_an = gemini_r.get("reason", gemini_r.get("analysis","N/A")) if isinstance(gemini_r, dict) else "N/A"
    
    # 释放 Claude，UI 上的字和比分将完全一致
    cl_sc = claude_r.get("ai_score", final_score) if isinstance(claude_r, dict) else final_score
    cl_an = claude_r.get("reason", claude_r.get("analysis","N/A")) if isinstance(claude_r, dict) else "N/A"

    sigs = list(smart_signals)
    if cold_door_info["is_cold_door"]: 
        sigs.extend(cold_door_info["signals"])
        cf = max(30, cf - 5)

    return {
        "predicted_score": final_score, 
        "home_win_pct": fhp, 
        "draw_pct": fdp, 
        "away_win_pct": fap,
        "confidence": cf, 
        "risk_level": risk,
        "over_under_2_5": "大" if engine_result.get("over_25", 50) > 55 else "小",
        "both_score": "是" if engine_result.get("btts", 45) > 50 else "否",
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
        "model_agreement": len(set([gpt_sc, grok_sc, gem_sc, final_score])) <= 2,
        "xG_home": round(home_xg, 2), 
        "xG_away": round(away_xg, 2),
        "league_multiplier": multi,
        "poisson": poisson_scores, 
        "refined_poisson": stats.get("refined_poisson", {}),
        "extreme_warning": engine_result.get("scissors_gap_signal", ""),
        "smart_money_signal": " | ".join(sigs), 
        "smart_signals": sigs,
        "model_consensus": stats.get("model_consensus", 0), 
        "total_models": stats.get("total_models", 11),
        "expected_total_goals": exp_goals,
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
        "cold_door": cold_door_info,
    }


def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})
        s = pr.get("confidence", 0) * 0.4
        mx = max(pr.get("home_win_pct", 33), pr.get("away_win_pct", 33), pr.get("draw_pct", 33))
        s += (mx - 33) * 0.2 + pr.get("model_consensus", 0) * 2
        if pr.get("risk_level") == "低": 
            s += 12
        elif pr.get("risk_level") == "高": 
            s -= 5
            
        if pr.get("model_agreement"): 
            s += 10
            
        exp_info = pr.get("experience_analysis", {})
        exp_score = exp_info.get("total_score", 0)
        if exp_score >= 15 and pr.get("result") == "平局" and exp_info.get("draw_rules", 0) >= 3: 
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
                
        cold = pr.get("cold_door", {})
        if cold.get("is_cold_door"): 
            s -= 8
            
        p["recommend_score"] = round(s, 2)
        
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]

def extract_num(ms):
    wm = {"一":1000, "二":2000, "三":3000, "四":4000, "五":5000, "六":6000, "日":7000, "天":7000}
    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999

# ====================================================================
# run_predictions vMAX 14.0
# ====================================================================
def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 80)
    print(f"  [QUANT ENGINE vMAX 14.0] 奇点版 | 庄家锚点高斯引力全面上线 | {len(ms)} 场比赛")
    print("=" * 80)
    
    match_analyses = []
    for i, m in enumerate(ms):
        eng = predict_match(m)
        league_info, _, _, _ = build_league_intelligence(m)
        sp = ensemble.predict(m, {})
        exp_result = exp_engine.analyze(m)
        match_analyses.append({
            "match": m, 
            "engine": eng, 
            "league_info": league_info, 
            "stats": sp, 
            "index": i+1, 
            "experience": exp_result
        })
        
    all_ai = {"claude": {}, "gemini": {}, "gpt": {}, "grok": {}}
    
    if use_ai and match_analyses:
        print(f"  [TWO-PHASE] 启动两阶段AI架构...")
        start_t = time.time()
        all_ai = asyncio.run(run_ai_matrix_two_phase(match_analyses))
        print(f"  [AI MATRIX] 压榨完成，耗时 {time.time()-start_t:.1f}s")
        
    res = []
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        mg = merge_result(
            ma["engine"], 
            all_ai["gpt"].get(i+1,{}), 
            all_ai["grok"].get(i+1,{}), 
            all_ai["gemini"].get(i+1,{}), 
            all_ai["claude"].get(i+1,{}), 
            ma["stats"], 
            m
        )
        
        try: mg = apply_experience_to_prediction(m, mg, exp_engine)
        except: pass
            
        try: mg = apply_odds_history(m, mg)
        except: pass
            
        try: mg = apply_quant_edge(m, mg)
        except: pass
            
        try: mg = apply_wencai_intel(m, mg)
        except: pass
            
        try: mg = upgrade_ensemble_predict(m, mg)
        except: pass
            
        score_str = mg.get("predicted_score", "1-1")
        try:
            sh, sa = map(int, score_str.split("-"))
            if sh > sa: 
                mg["result"] = "主胜"
            elif sh < sa: 
                mg["result"] = "客胜"
            else: 
                mg["result"] = "平局"
        except:
            pcts = {"主胜": mg["home_win_pct"], "平局": mg["draw_pct"], "客胜": mg["away_win_pct"]}
            mg["result"] = max(pcts, key=pcts.get)
            
        res.append({**m, "prediction": mg})
        
        cold = mg.get("cold_door", {})
        cold_tag = f" [❄️{cold.get('level','')}冷门]" if cold.get("is_cold_door") else ""
        print(f"  [{i+1}] {m.get('home_team')} vs {m.get('away_team')} => {mg['result']} ({mg['predicted_score']}) | CF: {mg['confidence']}% | AI信心: {mg.get('ai_avg_confidence', 0)}{cold_tag}")
        
    t4 = select_top4(res)
    t4ids = [t.get("id") for t in t4]
    
    for r in res: 
        r["is_recommended"] = r.get("id") in t4ids
        
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    
    diary = load_ai_diary()
    cold_count = len([r for r in res if r.get("prediction",{}).get("cold_door",{}).get("is_cold_door")])
    diary["yesterday_win_rate"] = f"{len([r for r in res if r['prediction']['confidence']>70])}/{max(1,len(res))}"
    diary["reflection"] = f"vMAX14.0 | {cold_count}冷门 | 庄家锚点高斯引力+战意降维+裂角侦测"
    save_ai_diary(diary)
    
    return res, t4

if __name__ == "__main__":
    print("✅ vMAX 14.0 奇点版本启动成功！盘口方程引力场已挂载。")
