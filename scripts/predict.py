import json
import os
import re
import time
import asyncio
import aiohttp
import logging
import numpy as np
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

# ====================================================================
# 🛡️ 终极防御装甲：日志模块平滑降级与动态依赖加载
# ====================================================================
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    print("  [WARN] ⚠️ 未检测到 structlog 库，自动降级为标准 logging 模块")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

try:
    from config import *
    from models import EnsemblePredictor
    from odds_engine import predict_match
    from league_intel import build_league_intelligence
    from experience_rules import ExperienceEngine, apply_experience_to_prediction
    from advanced_models import upgrade_ensemble_predict
except ImportError as e:
    logger.warning(f"基础核心模块导入异常: {e}")

try:
    from odds_history import apply_odds_history
except Exception as e:
    logger.warning("⚠️ 历史盘口模块 (odds_history) 加载失败，系统自动降级", exc_info=True)
    def apply_odds_history(m, mg): return mg

try:
    from quant_edge import apply_quant_edge
except Exception as e:
    logger.warning("⚠️ 量化边缘模块 (quant_edge) 加载失败，系统自动降级", exc_info=True)
    def apply_quant_edge(m, mg): return mg

try:
    from wencai_intel import apply_wencai_intel
except:
    def apply_wencai_intel(m, mg): return mg

try:
    ensemble = EnsemblePredictor()
    exp_engine = ExperienceEngine()
except:
    pass


# ====================================================================
# ☢️ 工具函数
# ====================================================================
def dixon_coles_tau(hg: int, ag: int, lambda_h: float, lambda_a: float, rho: float) -> float:
    """Dixon-Coles低比分修正 — 正经数学方法，保留"""
    if hg == 0 and ag == 0:
        return 1 - (lambda_h * lambda_a * rho)
    elif hg == 0 and ag == 1:
        return 1 + (lambda_h * rho)
    elif hg == 1 and ag == 0:
        return 1 + (lambda_a * rho)
    elif hg == 1 and ag == 1:
        return 1 - rho
    return 1.0

def calculate_dynamic_rho(league: str, total_goals_expected: float) -> float:
    """联赛相关的rho参数"""
    league_params = {
        "英超": {"base": 0.12, "slope": 0.02},
        "德甲": {"base": 0.15, "slope": 0.03},
        "意甲": {"base": 0.08, "slope": 0.02},
    }
    p = league_params.get(league[:2], {"base": 0.10, "slope": 0.02})
    return p["base"] + p["slope"] * (total_goals_expected - 1)

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
        "kelly": round(max(0.0, kelly * 0.5) * 100, 2),
        "is_value": ev > 0.05
    }

def parse_score(s):
    try:
        s = str(s).strip().replace(" ", "").replace("：", "-").replace(":", "-").replace("\u2013", "-").replace("\u2014", "-")
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

        smart_str = " ".join(str(s) for s in prediction.get("smart_signals", []))
        if "Sharp" in smart_str or "sharp" in smart_str:
            strength += 6
            signals.append("🔥 Sharp Money确认！")

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
            "sharp_confirmed": "Sharp" in smart_str,
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
# 🧠 单阶段AI架构 vMAX 15.0 — 纯数据prompt·零暗示·xG双泊松选分
# ====================================================================
def build_phase1_prompt(match_analyses):
    """纯数据，零暗示，让AI自由分析"""
    diary = load_ai_diary()
    p = "你是顶尖足球量化分析师。根据以下原始数据，独立分析每场比赛，给出概率最高的3个候选比分。\n\n"
    if diary.get("reflection"):
        p += f"【进化】胜率:{diary.get('yesterday_win_rate','N/A')} | {diary['reflection']}\n\n"

    p += "【输出格式】只输出合法JSON数组。每场：match(整数), top3([{score,prob},...]), reason(80字), ai_confidence(0-100)。只输出数组！\n\n"

    p += "【原始数据】\n"
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        eng = ma.get("engine", {})
        stats = ma.get("stats", {})
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
            p += f"Shin概率: 主{(1/sp_h)/margin*100:.1f}% 平{(1/sp_d)/margin*100:.1f}% 客{(1/sp_a)/margin*100:.1f}%\n"

        if m.get("hhad_win"):
            p += f"让球胜平负: {m['hhad_win']}/{m.get('hhad_same','')}/{m.get('hhad_lose','')}\n"

        hxg = eng.get('bookmaker_implied_home_xg', '?')
        axg = eng.get('bookmaker_implied_away_xg', '?')
        p += f"庄家隐含xG: 主{hxg} vs 客{axg}\n"

        a0=m.get("a0",""); a1=m.get("a1","")
        if a0:
            a2=m.get("a2",""); a3=m.get("a3",""); a4=m.get("a4",""); a5=m.get("a5",""); a6=m.get("a6",""); a7=m.get("a7","")
            p += f"总进球: 0球={a0}|1={a1}|2={a2}|3={a3}|4={a4}|5={a5}|6={a6}|7+={a7}\n"
            try:
                gp=[(gi,1/float(v)) for gi,v in enumerate([a0,a1,a2,a3,a4,a5,a6,a7]) if float(v)>1]
                tp=sum(p2 for _,p2 in gp)
                eg=sum(g*(p2/tp) for g,p2 in gp)
                p += f"→ 期望进球λ={eg:.2f}\n"
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

        hf_l=[]
        for k,lb in {"ss":"主/主","sp":"主/平","sf":"主/负","ps":"平/主","pp":"平/平","pf":"平/负","fs":"负/主","fp":"负/平","ff":"负/负"}.items():
            try:
                v=float(m.get(k,0) or 0)
                if v>1: hf_l.append(f"{lb}={v:.2f}")
            except: pass
        if hf_l: p += f"半全场: {' | '.join(hf_l)}\n"

        vote=m.get("vote",{})
        if vote:
            p += f"散户: 胜{vote.get('win','?')}% 平{vote.get('same','?')}% 负{vote.get('lose','?')}%\n"

        change=m.get("change",{})
        if change and isinstance(change,dict):
            cw=change.get("win",0); cs=change.get("same",0); cl=change.get("lose",0)
            if cw or cs or cl: p += f"赔率变动: 胜{cw} 平{cs} 负{cl}\n"

        info=m.get("information",{})
        if isinstance(info,dict):
            for k,v in [("home_injury","主伤停"),("guest_injury","客伤停"),("home_bad_news","主利空"),("guest_bad_news","客利空")]:
                if info.get(k): p += f"{v}: {str(info[k])[:200].replace(chr(10),' ')}\n"

        hs=m.get("home_stats",{}); ast2=m.get("away_stats",{})
        if hs.get("form"):
            p += f"主队: {hs.get('form','?')} 进{hs.get('avg_goals_for','?')}/失{hs.get('avg_goals_against','?')}\n"
            p += f"客队: {ast2.get('form','?')} 进{ast2.get('avg_goals_for','?')}/失{ast2.get('avg_goals_against','?')}\n"

        smart_sigs = stats.get('smart_signals', [])
        if smart_sigs:
            p += f"盘口信号: {', '.join(str(s) for s in smart_sigs[:4])}\n"

        for field in ['analyse','baseface','intro','expert_intro']:
            txt=str(m.get(field,'')).replace('\n',' ')[:150]
            if len(txt)>10: p += f"分析: {txt}\n"; break
        p += "\n"

    p += f"【输出{len(match_analyses)}场JSON数组，只输出数组！】\n"
    return p


# ====================================================================
# AI调用引擎
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
    if not key: return ai_name, {}, "no_key"

    primary_url = get_clean_env_url(url_env)
    backup = [u for u in FALLBACK_URLS if u and u != primary_url][:1]
    urls = [primary_url] + backup

    CONNECT_TIMEOUT = 20
    READ_TIMEOUT_MAP = {"claude": 350, "grok": 200, "gpt": 200, "gemini": 250}
    READ_TIMEOUT = READ_TIMEOUT_MAP.get(ai_name, 200)

    # 🔑 系统prompt零暗示，不告诉AI选什么不选什么
    AI_PROFILES = {
        "claude": {"sys": "你是量化足球分析师，根据数据分析每场比赛输出候选比分。只输出JSON数组。", "temp": 0.18},
        "grok": {"sys": "你是Grok，有联网搜索能力。搜索球队伤停、Pinnacle赔率、Betfair交易量，结合数据分析每场比赛，输出TOP3候选比分。只输出JSON数组。", "temp": 0.22},
        "gpt": {"sys": "你是职业足球量化分析师。用数学方法分析赔率数据，计算每场比赛概率最高的3个比分。只输出JSON数组。", "temp": 0.18},
        "gemini": {"sys": "你是概率建模引擎。从赔率数据计算每个比分的真实概率，输出TOP3候选比分。只输出JSON数组。", "temp": 0.15},
    }

    profile = AI_PROFILES.get(ai_name, AI_PROFILES["gpt"])

    for mn in models_list:
        connected = False
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
                bp = {"model": mn, "messages": [{"role": "system", "content": profile["sys"]}, {"role": "user", "content": prompt}]}
                if ai_name != "claude": bp["temperature"] = profile["temp"]
                payload = bp

            gw = url.split("/v1")[0][:35]
            print(f"  [🔌连接中] {ai_name.upper()} | {mn[:22]} @ {gw}")
            t0 = time.time()

            try:
                timeout = aiohttp.ClientTimeout(total=None, connect=CONNECT_TIMEOUT, sock_connect=CONNECT_TIMEOUT, sock_read=READ_TIMEOUT)
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                    elapsed_connect = round(time.time()-t0, 1)

                    if r.status in (502, 504):
                        print(f"    💀 HTTP {r.status} | {elapsed_connect}s → 换模型")
                        break
                    if r.status == 400:
                        print(f"    💀 400 不支持 | {elapsed_connect}s → 换模型")
                        break
                    if r.status == 429:
                        print(f"    🔥 429 限流 | {elapsed_connect}s → 换模型")
                        break
                    if r.status != 200:
                        print(f"    ⚠️ HTTP {r.status} | {elapsed_connect}s → 换URL")
                        continue

                    connected = True
                    print(f"    ✅ 已连上！{elapsed_connect}s | 等待数据...")

                    try:
                        data = await r.json(content_type=None)
                    except:
                        print(f"    ⚠️ 响应非JSON → 换模型")
                        break

                    elapsed = round(time.time()-t0, 1)
                    usage = data.get("usage", {})
                    req_tokens = usage.get("total_tokens", 0) or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
                    if not req_tokens:
                        um = data.get("usageMetadata", {})
                        req_tokens = um.get("totalTokenCount", 0)
                    if req_tokens:
                        print(f"    📊 {req_tokens:,} token | {elapsed}s")

                    raw_text = ""
                    try:
                        if is_gem:
                            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        else:
                            if data.get("choices") and data["choices"]:
                                msg = data["choices"][0].get("message", {})
                                if isinstance(msg, dict):
                                    best = ""
                                    for k in msg:
                                        v = msg[k]
                                        if isinstance(v, str) and v.strip():
                                            if "[" in v and "{" in v and len(v) > len(best):
                                                best = v.strip()
                                    if not best:
                                        for k in msg:
                                            v = msg[k]
                                            if isinstance(v, str) and len(v.strip()) > len(best):
                                                best = v.strip()
                                    raw_text = best

                            if not raw_text and data.get("output") and isinstance(data["output"], list):
                                for out_item in data["output"]:
                                    if isinstance(out_item, dict) and out_item.get("type") == "message":
                                        for ct in out_item.get("content", []):
                                            if isinstance(ct, dict) and ct.get("text"):
                                                t = ct["text"].strip()
                                                if len(t) > len(raw_text): raw_text = t

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
                    except:
                        pass

                    if not raw_text or len(raw_text) < 10:
                        print(f"    ⚠️ 空数据 → 换模型")
                        break

                    clean = raw_text
                    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", clean, flags=re.DOTALL|re.IGNORECASE)
                    clean = re.sub(r"```[\w]*", "", clean).strip()
                    start = clean.find("[")
                    end = clean.rfind("]") + 1

                    results = {}
                    if start != -1 and end > start:
                        try:
                            arr = json.loads(clean[start:end])
                        except json.JSONDecodeError:
                            try:
                                last_brace = clean[start:end].rfind('}')
                                arr = json.loads(clean[start:end][:last_brace+1] + "]") if last_brace != -1 else []
                            except:
                                arr = []

                        if isinstance(arr, list):
                            for item in arr:
                                if not isinstance(item, dict) or not item.get("match"): continue
                                try: mid = int(item["match"])
                                except: continue
                                if item.get("top3"):
                                    t1 = item["top3"][0].get("score", "1-1").replace(" ", "").strip() if item["top3"] else "1-1"
                                    results[mid] = {"top3": item["top3"], "ai_score": t1, "reason": str(item.get("reason", ""))[:200], "ai_confidence": int(item.get("ai_confidence", 60))}
                                elif item.get("score"):
                                    results[mid] = {"ai_score": item["score"].replace(" ", "").strip(), "reason": str(item.get("reason", ""))[:200], "ai_confidence": int(item.get("ai_confidence", 60))}

                    if len(results) > 0:
                        print(f"    ✅ {ai_name.upper()} 完成: {len(results)}/{num_matches} | {elapsed}s")
                        return ai_name, results, mn
                    else:
                        print(f"    ⚠️ 解析0条 → 换模型")
                        break

            except aiohttp.ClientConnectorError:
                print(f"    🔌 连接失败 → 换URL")
                continue
            except asyncio.TimeoutError:
                if not connected:
                    print(f"    🔌 连接超时 → 换URL")
                    continue
                else:
                    print(f"    ⏰ 读取超时 | 钱已花")
                    return ai_name, {}, "read_timeout"
            except Exception as e:
                if not connected:
                    print(f"    ⚠️ {str(e)[:40]} → 换URL")
                    continue
                else:
                    return ai_name, {}, "error"

            await asyncio.sleep(0.2)

    return ai_name, {}, "all_failed"


async def run_ai_matrix_two_phase(match_analyses):
    """单阶段：4个AI并行跑同一个prompt，Claude也参与投票"""
    num = len(match_analyses)
    prompt = build_phase1_prompt(match_analyses)
    print(f"  [单阶段] {len(prompt):,} 字符 → 4个AI并行...")

    ai_configs = [
        ("grok", "GROK_API_URL", "GROK_API_KEY", ["熊猫-A-6-grok-4.2-thinking"]),
        ("gpt", "GPT_API_URL", "GPT_API_KEY", ["熊猫-按量-gpt-5.4"]),
        ("gemini", "GEMINI_API_URL", "GEMINI_API_KEY", ["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking"]),
        ("claude", "CLAUDE_API_URL", "CLAUDE_API_KEY", ["熊猫特供-超纯满血-99额度-claude-opus-4.6-thinking", "熊猫-按量-特供顶级-官方正向满血-claude-opus-4.6-thinking"]),
    ]
    all_results = {"gpt": {}, "grok": {}, "gemini": {}, "claude": {}}

    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [async_call_one_ai_batch(session, prompt, u, k, m, num, n) for n, u, k, m in ai_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, tuple):
                all_results[res[0]] = res[1]
            else:
                print(f"  [ERROR] {res}")

    ok = sum(1 for v in all_results.values() if v)
    print(f"  [完成] {ok}/4 AI有数据")
    return all_results


# ====================================================================
# 🌟 Merge v15.0 — 方案B清爽评分公式
#
# 核心改动（对比v14.3）:
#   ✂️ 删除蒙特卡洛（和泊松重复）
#   ✂️ 删除进球吻合度（泊松自带进球分布）
#   ✂️ 删除方向一致性加分（方向层已调整xG）
#   ✂️ 删除handicap硬编码规则（让球1球扣1-0分）
#   ✂️ 删除双重冷门处罚（xG砍后不再砍方向概率）
#   ✂️ 删除Claude系统prompt"禁止1-1"的暗示
#   ✂️ 删除Phase1 prompt大比分指引
#   ✂️ 改两阶段为单阶段（省Claude裁判费用）
#   ✏️ 收紧is_do_or_die关键字（决→决赛/生死战/保级死拼/淘汰赛）
#   ✅ 保留Dixon-Coles低比分修正（正经数学）
#
# 评分公式:
#   泊松(60) + AI投票(30) + 联赛(10) = 100
# ====================================================================
def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    league = str(match_obj.get("league", match_obj.get("cup", "")))
    sp_h = float(match_obj.get("sp_home", match_obj.get("win", 0)) or 0)
    sp_d = float(match_obj.get("sp_draw", match_obj.get("same", 0)) or 0)
    sp_a = float(match_obj.get("sp_away", match_obj.get("lose", 0)) or 0)
    engine_conf = engine_result.get("confidence", 50)
    p1_ai = {"gpt": gpt_r, "grok": grok_r, "gemini": gemini_r}

    # ============ 第一层：方向概率（显示+信心分用） ============
    direction_scores = {"home": 0.0, "draw": 0.0, "away": 0.0}

    if sp_h > 1 and sp_d > 1 and sp_a > 1:
        margin = 1/sp_h + 1/sp_d + 1/sp_a
        direction_scores["home"] += (1/sp_h)/margin*30
        direction_scores["draw"] += (1/sp_d)/margin*30
        direction_scores["away"] += (1/sp_a)/margin*30
    else:
        direction_scores["home"] += 10; direction_scores["draw"] += 10; direction_scores["away"] += 10

    smart_signals = stats.get("smart_signals", [])
    smart_str = " ".join(str(s) for s in smart_signals)
    sharp_detected = "Sharp" in smart_str or "sharp" in smart_str

    if sharp_detected:
        if "客胜" in smart_str or "客队" in smart_str:
            direction_scores["away"] += 12
            print(f"    💰 Sharp→客胜 +12")
        elif "主胜" in smart_str or "主队" in smart_str:
            direction_scores["home"] += 12
            print(f"    💰 Sharp→主胜 +12")
        elif "平局" in smart_str or "平赔" in smart_str:
            direction_scores["draw"] += 12
            print(f"    💰 Sharp→平局 +12")

    ai_directions = {"home": 0, "draw": 0, "away": 0}
    for name, r in {**p1_ai, "claude": claude_r}.items():
        if not isinstance(r, dict): continue
        sc = parse_score(r.get("ai_score", ""))
        if not (sc and sc[0] is not None):
            t3 = r.get("top3", [])
            if t3 and isinstance(t3, list) and len(t3) > 0:
                sc = parse_score(t3[0].get("score", ""))
        if sc and sc[0] is not None:
            w = 1.5 if name == "claude" else 1.0
            if sc[0] > sc[1]: ai_directions["home"] += w
            elif sc[0] < sc[1]: ai_directions["away"] += w
            else: ai_directions["draw"] += w
    total_ai = sum(ai_directions.values())
    if total_ai > 0:
        for d in ["home", "draw", "away"]:
            direction_scores[d] += (ai_directions[d] / total_ai) * 25

    total_dir = sum(max(0.1, v) for v in direction_scores.values())
    dir_probs = {d: max(0.1, direction_scores[d]) / total_dir * 100 for d in direction_scores}
    final_direction = max(dir_probs, key=dir_probs.get)
    dir_gap = dir_probs[final_direction] - sorted(dir_probs.values(), reverse=True)[1]
    dir_confident = dir_gap > 5

    print(f"    🎯 方向: 主{dir_probs['home']:.0f}% 平{dir_probs['draw']:.0f}% 客{dir_probs['away']:.0f}%")

    # 冷门检测
    pre_pred = {
        "home_win_pct": dir_probs["home"], "draw_pct": dir_probs["draw"], "away_win_pct": dir_probs["away"],
        "steam_move": stats.get("steam_move", {}), "smart_signals": smart_signals,
        "line_movement_anomaly": stats.get("line_movement_anomaly", {})
    }
    cold_door = ColdDoorDetector.detect(match_obj, pre_pred)

    # ============ 第二层：期望进球 ============
    exp_goals = float(engine_result.get("expected_total_goals", stats.get("expected_total_goals", 0)) or 0)
    if exp_goals <= 0:
        try:
            gp = []
            for gi, field in enumerate(["a0","a1","a2","a3","a4","a5","a6","a7"]):
                v = float(match_obj.get(field, 0) or 0)
                if v > 1: gp.append((gi, 1/v))
            if gp:
                tp = sum(p for _, p in gp)
                exp_goals = sum(g*(p/tp) for g, p in gp)
        except:
            exp_goals = 2.5
    if exp_goals < 1.0:
        exp_goals = 2.5

    # ============ 第三层：xG调整（只调一次） ============
    home_xg = float(engine_result.get("bookmaker_implied_home_xg", 1.3) or 1.3)
    away_xg = float(engine_result.get("bookmaker_implied_away_xg", 0.9) or 0.9)

    xg_adj_log = []
    if sharp_detected:
        if "客胜" in smart_str or "客队" in smart_str:
            home_xg *= 0.85; away_xg *= 1.20
            xg_adj_log.append("Sharp客")
        elif "主胜" in smart_str or "主队" in smart_str:
            home_xg *= 1.15; away_xg *= 0.85
            xg_adj_log.append("Sharp主")
        elif "平局" in smart_str or "平赔" in smart_str:
            avg = (home_xg + away_xg) / 2
            home_xg = home_xg*0.7 + avg*0.3
            away_xg = away_xg*0.7 + avg*0.3
            xg_adj_log.append("Sharp平")

    # 冷门调整（只调xG，不再双重处罚方向概率）
    if cold_door["is_cold_door"] and not sharp_detected:
        hot_side = "home" if sp_h < sp_a else "away"
        if hot_side == "home":
            home_xg *= 0.75; away_xg *= 1.25
            xg_adj_log.append("冷主")
        else:
            away_xg *= 0.75; home_xg *= 1.25
            xg_adj_log.append("冷客")

    home_xg = max(0.3, min(4.0, home_xg))
    away_xg = max(0.2, min(3.5, away_xg))

    print(f"    ⚽ xG: 主{home_xg:.2f} 客{away_xg:.2f} (λ={home_xg+away_xg:.2f}) {' | '.join(xg_adj_log) if xg_adj_log else ''}")

    # ============ 第四层：双泊松 + Dixon-Coles ============
    rho = calculate_dynamic_rho(league, exp_goals)
    poisson_scores = {}
    for h_g in range(6):
        for a_g in range(6):
            p_h = math.exp(-home_xg) * (home_xg ** h_g) / math.factorial(h_g)
            p_a = math.exp(-away_xg) * (away_xg ** a_g) / math.factorial(a_g)
            tau = dixon_coles_tau(h_g, a_g, home_xg, away_xg, rho)
            prob = p_h * p_a * tau * 100
            poisson_scores[f"{h_g}-{a_g}"] = round(max(0, prob), 2)

    # ============ 第五层：清爽评分（方案B核心） ============
    ai_voted = {}
    for name, r in {**p1_ai, "claude": claude_r}.items():
        if not isinstance(r, dict): continue
        sc = parse_score(r.get("ai_score", ""))
        if not (sc and sc[0] is not None):
            t3 = r.get("top3", [])
            if t3 and isinstance(t3, list) and len(t3) > 0:
                sc = parse_score(t3[0].get("score", ""))
        if sc and sc[0] is not None:
            key = f"{sc[0]}-{sc[1]}"
            w = 1.5 if name == "claude" else (1.3 if name == "grok" else 1.0)
            ai_voted[key] = ai_voted.get(key, 0) + w
        t3 = r.get("top3", [])
        if isinstance(t3, list):
            for rank, t in enumerate(t3[1:3], 2):
                s2 = parse_score(t.get("score", ""))
                if s2 and s2[0] is not None:
                    key2 = f"{s2[0]}-{s2[1]}"
                    w2 = 0.4 if rank == 2 else 0.2
                    ai_voted[key2] = ai_voted.get(key2, 0) + w2

    # is_do_or_die收紧关键字
    is_do_or_die = any(kw in smart_str for kw in ["保级死拼", "生死战", "决赛", "淘汰赛"])

    score_ratings = {}
    for score_str, poisson_prob in poisson_scores.items():
        h_g, a_g = map(int, score_str.split("-"))
        total_g = h_g + a_g

        # ① 泊松 [60] — 最高概率约13% → 60分
        s = poisson_prob * 4.6

        # ② AI投票 [30]
        s += min(30, ai_voted.get(score_str, 0) * 7)

        # ③ 联赛风格 [10]
        if any(lg in league for lg in ["德甲", "荷甲", "英超", "澳超"]) and total_g >= 3: s += 10
        elif any(lg in league for lg in ["德甲", "荷甲"]) and total_g == 2: s += 5
        elif any(lg in league for lg in ["意甲", "法乙"]) and total_g <= 2: s += 10
        elif "意甲" in league and h_g == a_g: s += 5
        elif any(lg in league for lg in ["英冠", "英甲"]) and 2 <= total_g <= 3: s += 6
        elif any(lg in league for lg in ["日职", "韩职", "日乙"]) and total_g >= 2: s += 4

        # 生死战/决赛降低平局（已收紧关键字不会误触发）
        if is_do_or_die and h_g == a_g:
            s -= 8

        if s > 0:
            score_ratings[score_str] = round(s, 2)

    ranked = sorted(score_ratings.items(), key=lambda x: x[1], reverse=True)
    final_score = ranked[0][0] if ranked else "1-1"

    print(f"    📊 比分: {' > '.join(f'{sc}({pts:.0f}|P{poisson_scores.get(sc,0):.1f}%)' for sc, pts in ranked[:5])}")

    # ============ 第六层：输出 ============
    crs_map_rev = {"1-0": "w10", "2-0": "w20", "2-1": "w21", "3-0": "w30", "3-1": "w31", "3-2": "w32",
                   "0-0": "s00", "1-1": "s11", "2-2": "s22", "3-3": "s33",
                   "0-1": "l01", "0-2": "l02", "1-2": "l12", "0-3": "l03", "1-3": "l13", "2-3": "l23"}
    target_crs = crs_map_rev.get(final_score, "")
    final_odds = float(match_obj.get(target_crs, 0) or 0)
    final_prob = poisson_scores.get(final_score, 10)
    ev_data = calculate_value_bet(final_prob, final_odds)

    weights = {"claude": 1.4, "grok": 1.3, "gpt": 1.1, "gemini": 1.0}
    ai_conf_sum = 0; ai_conf_count = 0; value_kills = 0
    for name, r in {**p1_ai, "claude": claude_r}.items():
        if not isinstance(r, dict): continue
        conf = r.get("ai_confidence", 60)
        ai_conf_sum += conf * weights.get(name, 1.0)
        ai_conf_count += weights.get(name, 1.0)
        if r.get("value_kill"): value_kills += 1

    avg_ai_conf = (ai_conf_sum / ai_conf_count) if ai_conf_count > 0 else 60
    cf = min(95, engine_conf + int((avg_ai_conf - 60) * 0.4)) + value_kills * 6
    if not dir_confident: cf = max(40, cf - 10)
    if any("🚨" in str(s) for s in smart_signals): cf = max(35, cf - 12)
    risk = "低" if cf >= 75 else ("中" if cf >= 55 else "高")

    sigs = list(smart_signals)
    if cold_door["is_cold_door"]:
        sigs.extend(cold_door["signals"])
        cf = max(30, cf - 5)

    cl_raw = claude_r.get("ai_score", "") if isinstance(claude_r, dict) else ""
    cl_parsed = parse_score(cl_raw)
    cl_sc = cl_raw if cl_parsed[0] is not None else final_score

    return {
        "predicted_score": final_score,
        "home_win_pct": round(dir_probs["home"], 1),
        "draw_pct": round(dir_probs["draw"], 1),
        "away_win_pct": round(dir_probs["away"], 1),
        "confidence": cf,
        "risk_level": risk,
        "over_under_2_5": "大" if engine_result.get("over_25", 50) > 55 else "小",
        "both_score": "是" if engine_result.get("btts", 45) > 50 else "否",
        "gpt_score": gpt_r.get("ai_score", "-") if isinstance(gpt_r, dict) else "-",
        "gpt_analysis": gpt_r.get("reason", gpt_r.get("analysis", "N/A")) if isinstance(gpt_r, dict) else "N/A",
        "grok_score": grok_r.get("ai_score", "-") if isinstance(grok_r, dict) else "-",
        "grok_analysis": grok_r.get("reason", grok_r.get("analysis", "N/A")) if isinstance(grok_r, dict) else "N/A",
        "gemini_score": gemini_r.get("ai_score", "-") if isinstance(gemini_r, dict) else "-",
        "gemini_analysis": gemini_r.get("reason", gemini_r.get("analysis", "N/A")) if isinstance(gemini_r, dict) else "N/A",
        "claude_score": cl_sc,
        "claude_analysis": claude_r.get("reason", claude_r.get("analysis", "N/A")) if isinstance(claude_r, dict) else "N/A",
        "ai_avg_confidence": round(avg_ai_conf, 1),
        "value_kill_count": value_kills,
        "model_agreement": len(set([gpt_r.get("ai_score", "-") if isinstance(gpt_r, dict) else "-",
                                     grok_r.get("ai_score", "-") if isinstance(grok_r, dict) else "-",
                                     gemini_r.get("ai_score", "-") if isinstance(gemini_r, dict) else "-",
                                     final_score])) <= 2,
        "xG_home": round(home_xg, 2),
        "xG_away": round(away_xg, 2),
        "poisson": poisson_scores,
        "dynamic_rho": round(rho, 3),
        "suggested_kelly": ev_data["kelly"],
        "edge_vs_market": ev_data["ev"],
        "bivariate_poisson": poisson_scores,
        "refined_poisson": stats.get("refined_poisson", {}),
        "extreme_warning": engine_result.get("scissors_gap_signal", ""),
        "smart_money_signal": " | ".join(sigs),
        "smart_signals": sigs,
        "model_consensus": stats.get("model_consensus", 0),
        "total_models": stats.get("total_models", 11),
        "expected_total_goals": round(exp_goals, 2),
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
    wm = {"一":1000, "二":2000, "三":3000, "四":4000, "五":5000, "六":6000, "日":7000, "天":7000}
    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999


def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 80)
    print(f"  [QUANT ENGINE vMAX 15.0] 方案B·清爽评分公式·单阶段·xG双泊松+DC | {len(ms)} 场")
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
        print(f"  [单阶段] 启动4AI并行...")
        start_t = time.time()
        all_ai = asyncio.run(run_ai_matrix_two_phase(match_analyses))
        print(f"  [完成] 耗时 {time.time()-start_t:.1f}s")

    res = []
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        mg = merge_result(
            ma["engine"],
            all_ai["gpt"].get(i+1, {}),
            all_ai["grok"].get(i+1, {}),
            all_ai["gemini"].get(i+1, {}),
            all_ai["claude"].get(i+1, {}),
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
            if sh > sa: mg["result"] = "主胜"
            elif sh < sa: mg["result"] = "客胜"
            else: mg["result"] = "平局"
        except:
            pcts = {"主胜": mg["home_win_pct"], "平局": mg["draw_pct"], "客胜": mg["away_win_pct"]}
            mg["result"] = max(pcts, key=pcts.get)

        res.append({**m, "prediction": mg})

        cold = mg.get("cold_door", {})
        cold_tag = f" [❄️{cold.get('level','')}冷门]" if cold.get("is_cold_door") else ""
        print(f"  [{i+1}] {m.get('home_team')} vs {m.get('away_team')} => {mg['result']} ({mg['predicted_score']}) | CF: {mg['confidence']}% | EV: {mg.get('edge_vs_market',0)}%{cold_tag}")

    t4 = select_top4(res)
    t4ids = [t.get("id") for t in t4]
    for r in res: r["is_recommended"] = r.get("id") in t4ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    diary = load_ai_diary()
    cold_count = len([r for r in res if r.get("prediction", {}).get("cold_door", {}).get("is_cold_door")])
    diary["yesterday_win_rate"] = f"{len([r for r in res if r['prediction']['confidence']>70])}/{max(1,len(res))}"
    diary["reflection"] = f"vMAX15.0 | {cold_count}冷门 | 方案B·清爽评分·单阶段·Dixon-Coles"
    save_ai_diary(diary)

    return res, t4


if __name__ == "__main__":
    logger.info("vMAX 15.0 启动")
    print("✅ vMAX 15.0 (方案B) 已加载")