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
# 🧠 两阶段AI架构 v7.0 — 零引擎暗示·纯量化
# Phase1: GPT/Grok/Gemini 各自独立分析原始数据 → 每场给TOP3候选比分+概率
# Phase2: Claude 裁判综合三家结果 + CRS赔率校验 → 选出最终比分
# ====================================================================

def build_phase1_prompt(match_analyses):
    """Phase1 Prompt: 纯原始赔率数据，零引擎暗示，要求输出TOP3候选比分"""
    diary = load_ai_diary()
    p = "【身份】你是顶尖量化足球分析师。下面是庄家的原始定价数据。\n"
    p += "你必须用数学方法独立计算，给出每场比赛概率最高的3个候选比分。\n\n"
    if diary.get("reflection"):
        p += f"【进化】{diary.get('yesterday_win_rate','N/A')} | {diary['reflection']}\n\n"

    p += "【输出格式——必须严格遵守】\n"
    p += "只输出合法JSON数组，禁止任何其他文字。\n"
    p += "每场：match(整数), top3(数组含3个{score,prob}), reason(100-150字含3+赔率数字), ai_confidence(0-100)。\n"
    p += 'top3中prob是你计算的该比分概率百分比。\n'
    p += '示例: {"match":1,"top3":[{"score":"1-0","prob":18.2},{"score":"1-1","prob":16.5},{"score":"0-1","prob":12.1}],"reason":"...","ai_confidence":75}\n\n'

    p += "【量化计算方法——必须在思维链中执行】\n"
    p += "① Shin公式对欧赔三项去水位 → 真实胜平负概率\n"
    p += "② CRS全部比分赔率做1/odds → 归一化 → 每个比分的隐含概率 → 排出TOP3\n"
    p += "③ 总进球a0-a7做1/odds加权 → 庄家预期总进球λ → 验证TOP3进球数是否合理\n"
    p += "④ 亚盘让球方向 vs 欧赔方向交叉验证 → 不一致=有陷阱\n"
    p += "⑤ 半全场赔率推断节奏 → 验证比分合理性\n"
    p += "⑥ 伤停/状态 → 微调概率±10%\n\n"

    p += "【关键常识——违反=概率计算有误】\n"
    p += "- CRS赔率最低的比分 = 庄家真金白银认为最可能发生的，你的TOP1应该重点参考它\n"
    p += "- 让球≤1.0的比赛：1-0/0-1在统计上占18-22%，是最常见单一比分\n"
    p += "- 三项欧赔接近(最大-最小<0.8) = 均势局 → 平局概率极高\n"
    p += "- 0球赔率<9.0 = 庄家重防零球 → 必须考虑0-0\n"
    p += "- 不要被主场优势迷惑，英冠冷门率32%、法乙28%、荷乙27%\n\n"

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

        p += f"{'='*60}\n[{i+1}] {h} vs {a} | {league}\n"

        # 联赛冷门率提醒
        for lg, rate in {"英冠":32,"英甲":30,"英乙":28,"法乙":28,"荷乙":27,"德乙":26,"意乙":25}.items():
            if lg in str(league):
                p += f"⚠️ {lg}历史冷门率{rate}%\n"; break

        # 欧赔 + 离散度标记
        odds_rng = round(max(sp_h,sp_d,sp_a)-min(sp_h,sp_d,sp_a),2) if sp_h>1 else 0
        tag = " ⚠️三项极接近=高平局" if 0<odds_rng<0.8 else ""
        p += f"欧赔: {sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f}{tag} | 让球: {hc}\n"

        if m.get("hhad_win"):
            p += f"让球胜平负: {m['hhad_win']}/{m.get('hhad_same','')}/{m.get('hhad_lose','')}\n"
        if m.get("single") == 1:
            p += f"📌 单关开放\n"

        h_pos = m.get("home_position",""); g_pos = m.get("guest_position","")
        if h_pos or g_pos:
            p += f"排名: 主{h_pos} vs 客{g_pos}\n"

        # 总进球 + 预期进球计算
        a0=m.get("a0","");a1=m.get("a1","");a2=m.get("a2","");a3=m.get("a3","")
        a4=m.get("a4","");a5=m.get("a5","");a6=m.get("a6","");a7=m.get("a7","")
        if a0:
            p += f"总进球: 0球={a0}|1={a1}|2={a2}|3={a3}|4={a4}|5={a5}|6={a6}|7+={a7}\n"
            try:
                gp=[(gi,1/float(v)) for gi,v in enumerate([a0,a1,a2,a3,a4,a5,a6,a7]) if float(v)>1]
                tp=sum(p2 for _,p2 in gp); eg=sum(g*(p2/tp) for g,p2 in gp)
                ml=min(gp, key=lambda x:1/x[1])
                p += f"→ 庄家预期{eg:.1f}球 最可能{ml[0]}球({ml[1]/tp*100:.0f}%)\n"
            except: pass

        # CRS全比分 + TOP5
        crs_map = {"w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","w32":"3-2","w40":"4-0","w41":"4-1","w42":"4-2",
                   "s00":"0-0","s11":"1-1","s22":"2-2","s33":"3-3",
                   "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3","l23":"2-3","l04":"0-4","l14":"1-4"}
        crs_lines=[]; crs_probs=[]
        for key,score in crs_map.items():
            try:
                odds=float(m.get(key,0)or 0)
                if odds>1: crs_lines.append(f"{score}={odds:.2f}"); crs_probs.append((score,odds,1/odds))
            except: pass
        if crs_lines:
            p += f"CRS: {' | '.join(crs_lines)}\n"
            if crs_probs:
                crs_probs.sort(key=lambda x:x[1])
                tp2=sum(pr for _,_,pr in crs_probs)
                p += f"→ CRS TOP5: {' > '.join(f'{s}({pr/tp2*100:.1f}%)' for s,_,pr in crs_probs[:5])}\n"

        # 半全场
        hf_l=[]
        for k,lb in {"ss":"主/主","sp":"主/平","sf":"主/负","ps":"平/主","pp":"平/平","pf":"平/负","fs":"负/主","fp":"负/平","ff":"负/负"}.items():
            try:
                v=float(m.get(k,0)or 0)
                if v>1: hf_l.append(f"{lb}={v:.2f}")
            except: pass
        if hf_l: p += f"半全场: {' | '.join(hf_l)}\n"

        # 散户投注
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

        # 伤停情报
        info=m.get("information",{})
        if isinstance(info,dict):
            for k,v in [("home_injury","主伤停"),("guest_injury","客伤停"),("home_good_news","主利好"),("guest_good_news","客利好"),("home_bad_news","主利空"),("guest_bad_news","客利空")]:
                if info.get(k): p += f"{v}: {str(info[k])[:250].replace(chr(10),' | ')}\n"

        # 状态
        hs=m.get("home_stats",{}); ast2=m.get("away_stats",{})
        if hs.get("form"):
            p += f"主队: {hs.get('wins','?')}胜{hs.get('draws','?')}平{hs.get('losses','?')}负 {hs.get('form','?')} 进{hs.get('avg_goals_for','?')}/失{hs.get('avg_goals_against','?')}\n"
            p += f"客队: {ast2.get('wins','?')}胜{ast2.get('draws','?')}平{ast2.get('losses','?')}负 {ast2.get('form','?')} 进{ast2.get('avg_goals_for','?')}/失{ast2.get('avg_goals_against','?')}\n"

        # 基本面/分析
        for field in ['analyse','baseface','intro','expert_intro']:
            txt=str(m.get(field,'')).replace('\n',' ')[:200]
            if len(txt)>10: p += f"分析: {txt}\n"; break
        p += "\n"

    p += f"【输出{len(match_analyses)}场JSON数组，每场含top3。只输出数组！】\n"
    return p


def build_phase2_prompt(match_analyses, phase1_results):
    """Phase2 Prompt: Claude裁判综合三家AI的TOP3 + CRS赔率校验"""
    p = "【你是最终裁判】三个独立AI已各自给出每场TOP3候选比分和概率。\n"
    p += "你的任务：综合他们的分析，结合CRS赔率数据，选出每场最终比分。\n\n"
    p += "【决策规则——严格执行】\n"
    p += "1. 如果3家AI的TOP1相同 → 高信心采用\n"
    p += "2. 如果2家TOP1相同 → 检查CRS赔率≤10倍 → 采用\n"
    p += "3. 如果3家各不同 → 用CRS赔率最低的比分（=庄家认为最可能）作为锚点\n"
    p += "4. 你选的比分CRS赔率不应超过10倍（超过=概率太低不现实）\n"
    p += "5. 三项欧赔差值<0.8 = 均势 → 优先平局(1-1/0-0)\n"
    p += "6. 如果有AI引用了实时搜索结果（如Pinnacle赔率、Betfair交易量），给予额外权重\n\n"
    p += "【输出格式】JSON数组：match(整数), score(最终比分), reason(80-120字说明选择逻辑), ai_confidence(0-100)\n"
    p += "只输出JSON数组！\n\n"

    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        sp_h = float(m.get("sp_home", m.get("win", 0)) or 0)
        sp_d = float(m.get("sp_draw", m.get("same", 0)) or 0)
        sp_a = float(m.get("sp_away", m.get("lose", 0)) or 0)
        idx = i + 1

        p += f"{'='*60}\n[{idx}] {h} vs {a}\n"
        p += f"欧赔: {sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f} | 让球: {m.get('give_ball','0')}\n"

        # CRS TOP5
        crs_map = {"w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","s00":"0-0","s11":"1-1","s22":"2-2","l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3"}
        crs_probs=[]
        for key,score in crs_map.items():
            try:
                odds=float(m.get(key,0) or 0)
                if odds>1: crs_probs.append((score,odds))
            except: pass
        if crs_probs:
            crs_probs.sort(key=lambda x:x[1])
            p += f"CRS校验: {' > '.join(f'{s}@{o:.1f}' for s,o in crs_probs[:5])}\n"

        # 三家AI结果
        for ai_name in ["gpt", "grok", "gemini"]:
            ai_data = phase1_results.get(ai_name, {}).get(idx, {})
            if not ai_data:
                p += f"  {ai_name.upper()}: 无数据\n"; continue
            top3 = ai_data.get("top3", [])
            if top3:
                scores_str = " | ".join(f"{t.get('score','?')}({t.get('prob','?')}%)" for t in top3[:3])
                p += f"  {ai_name.upper()}: {scores_str} | 信心{ai_data.get('ai_confidence','?')} | {str(ai_data.get('reason',''))[:100]}\n"
            else:
                sc = ai_data.get("ai_score", "-")
                p += f"  {ai_name.upper()}: {sc} | 信心{ai_data.get('ai_confidence','?')} | {str(ai_data.get('analysis',''))[:100]}\n"
        p += "\n"

    p += f"【输出{len(match_analyses)}场JSON数组！只输出数组！】\n"
    return p


# ====================================================================
# AI调用引擎（处理top3和score两种格式）
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
    timeout_map = {"claude": 1500, "grok": 300, "gpt": 720, "gemini": 360}
    timeout_sec = timeout_map.get(ai_name, 200)

    AI_PROFILES = {
        "claude": {
            "sys": "你是最终裁判。三个独立AI已给出候选比分，你综合分析后选出每场最终比分。只输出JSON数组。",
            "temp": 0.12
        },
        "grok": {
            "sys": "你是Grok，具备实时联网搜索能力。这是你的核心优势。\n"
                   "【必须执行的搜索任务】\n"
                   "1. 搜索oddsportal.com拿Pinnacle赔率，与提供的竞彩赔率比较偏差（>5%=诱盘）\n"
                   "2. 搜索Betfair Exchange交易量和赔率\n"
                   "3. 搜索球队名+injury/lineup确认最新首发伤停\n"
                   "4. 搜索比赛城市天气+裁判历史数据\n"
                   "5. 搜索X平台球队最新动态\n"
                   "reason必须引用搜索到的具体事实(如Pinnacle主胜1.62，Betfair交易量占比68%)。\n"
                   "输出每场TOP3候选比分。只输出JSON数组。",
            "temp": 0.22
        },
        "gpt": {
            "sys": "你是20年实战职业博彩分析师。用纯数学方法计算每场TOP3候选比分：\n"
                   "1. CRS赔率1/odds归一化→概率矩阵→TOP3\n"
                   "2. 总进球a0-a7→预期进球数→验证TOP3合理性\n"
                   "3. 亚盘+欧赔交叉验证方向\n"
                   "4. 半全场推断节奏\n"
                   "该1-0就1-0，该0-0就0-0。reason含具体赔率。只输出JSON数组。",
            "temp": 0.15
        },
        "gemini": {
            "sys": "你是概率建模引擎。严格执行数学计算：\n"
                   "1. CRS全比分→概率矩阵→TOP3\n"
                   "2. 总进球→数学期望λ\n"
                   "3. 欧赔去水位→真实概率\n"
                   "4. 赔率异常检测(CRS vs 泊松偏差>50%=庄家操纵)\n"
                   "输出每场TOP3候选比分及概率。只输出JSON数组。",
            "temp": 0.13
        },
    }

    best_results = {}; best_model = ""
    struct_fail_count = 0  # 追踪结构缺失次数
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
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec, connect=15)) as r:
                    elapsed = round(time.time()-t0,1)
                    if r.status == 200:
                        try:
                            data = await r.json(content_type=None)
                        except:
                            print(f"    ⚠️ 响应非JSON | {elapsed}s → 换URL")
                            continue

                        # ★ 健壮response解析：兼容多种格式
                        raw_text = ""
                        try:
                            if is_gem:
                                raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                            else:
                                # 标准格式
                                msg = data.get("choices", [{}])[0].get("message", {})
                                raw_text = (msg.get("content") or "").strip()
                                # GPT-5.4 thinking格式：content可能为null，文本在reasoning_content
                                if not raw_text:
                                    raw_text = (msg.get("reasoning_content") or "").strip()
                                # 有些代理把content放在text字段
                                if not raw_text:
                                    raw_text = (msg.get("text") or "").strip()
                                # 兜底：整个response转字符串找JSON
                                if not raw_text:
                                    raw_text = json.dumps(data, ensure_ascii=False)
                        except:
                            pass

                        if not raw_text or len(raw_text) < 10:
                            struct_fail_count += 1
                            print(f"    ⚠️ 结构缺失(#{struct_fail_count}) | {elapsed}s → {'跳模型' if struct_fail_count >= 2 else '换URL'}")
                            if struct_fail_count >= 2:
                                skip_model = True; break
                            continue

                        # 清理thinking标签
                        clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", raw_text, flags=re.DOTALL|re.IGNORECASE)
                        clean = re.sub(r"```[\w]*","",clean).strip()
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
                                # 断肢重生：截断到最后一个完整对象
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
                        if len(results) > len(best_results): best_results=results; best_model=mn; print(f"    ⚠️ 部分 {len(results)}/{num_matches} | {elapsed}s")
                        else: print(f"    ⚠️ 解析不足 {len(results)}条 | {elapsed}s")
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
    """两阶段：Phase1(GPT/Grok/Gemini并行)→ Phase2(Claude裁判)"""
    num = len(match_analyses)

    # ===== Phase1: 三家并行独立分析 =====
    p1_prompt = build_phase1_prompt(match_analyses)
    print(f"  [Phase1] {len(p1_prompt):,} 字符 → GPT/Grok/Gemini 并行...")

    p1_configs = [
        ("grok","GROK_API_URL","GROK_API_KEY",["熊猫-A-7-grok-4.2-多智能体讨论","熊猫-A-6-grok-4.2-thinking"]),
        ("gpt","GPT_API_URL","GPT_API_KEY",["熊猫-A-10-gpt-5.4","熊猫-按量-gpt-5.4","熊猫-A-10-gpt-5.3-codex"]),
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
    print(f"  [Phase1] 完成: {ok}/3 AI有数据")

    # ===== Phase2: Claude裁判 =====
    p2_prompt = build_phase2_prompt(match_analyses, p1_results)
    print(f"  [Phase2] {len(p2_prompt):,} 字符 → Claude 裁判...")

    claude_r = {}
    async with aiohttp.ClientSession() as session:
        _,claude_r,_ = await async_call_one_ai_batch(
            session, p2_prompt, "CLAUDE_API_URL","CLAUDE_API_KEY",
            ["熊猫-特供-A-55-claude-opus-4.6-thinking","熊猫特供-超纯满血-99额度-claude-opus-4.6-thinking","熊猫-按量-特供顶级-官方正向满血-claude-opus-4.6-thinking"],
            num, "claude"
        )

    all_r = p1_results.copy()
    all_r["claude"] = claude_r
    return all_r

# ====================================================================
# Merge v4.0 — 方向先行+CRS回检+AI投票校验
# 核心修复：预测9主胜实际6主胜=主胜偏向；AI共识2-0实际≤1球=比分虚高
# ====================================================================
def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    sp_h = float(match_obj.get("sp_home", match_obj.get("win", 0)) or 0)
    sp_d = float(match_obj.get("sp_draw", match_obj.get("same", 0)) or 0)
    sp_a = float(match_obj.get("sp_away", match_obj.get("lose", 0)) or 0)
    engine_score = engine_result.get("primary_score", "1-1")
    engine_conf = engine_result.get("confidence", 50)

    # ========== 收集Phase1三家AI的比分 ==========
    p1_ai = {"gpt": gpt_r, "grok": grok_r, "gemini": gemini_r}
    p1_scores = []  # [(score, ai_name)]
    for name, r in p1_ai.items():
        if not isinstance(r, dict): continue
        sc = r.get("ai_score", "-")
        if not sc or sc in ["-", "?", ""]:
            # 尝试从top3取第一个
            t3 = r.get("top3", [])
            if t3 and isinstance(t3, list) and len(t3) > 0:
                sc = t3[0].get("score", "-")
        if sc and sc not in ["-", "?", ""]:
            p1_scores.append((sc, name))

    # Phase1投票统计
    p1_vote = {}
    for sc, _ in p1_scores:
        p1_vote[sc] = p1_vote.get(sc, 0) + 1
    p1_majority = max(p1_vote, key=p1_vote.get) if p1_vote else ""
    p1_majority_count = p1_vote.get(p1_majority, 0) if p1_majority else 0

    # Claude裁判比分
    claude_score = ""
    if isinstance(claude_r, dict):
        claude_score = claude_r.get("ai_score", "")
        if not claude_score or claude_score in ["-", "?"]:
            claude_score = ""

    # CRS工具函数
    crs_key_map = {"1-0":"w10","0-1":"l01","2-1":"w21","1-2":"l12","2-0":"w20","0-2":"l02",
                   "0-0":"s00","1-1":"s11","3-0":"w30","3-1":"w31","0-3":"l03","1-3":"l13",
                   "2-2":"s22","3-2":"w32","2-3":"l23"}
    def get_crs(score):
        key = crs_key_map.get(score, "")
        try: return float(match_obj.get(key, 99) or 99)
        except: return 99.0

    # 预期总进球（从a0-a7计算）
    exp_goals = engine_result.get("expected_goals", 2.3)
    try:
        gp = []
        for gi, field in enumerate(["a0","a1","a2","a3","a4","a5","a6","a7"]):
            v = float(match_obj.get(field, 0) or 0)
            if v > 1: gp.append((gi, 1/v))
        if gp:
            tp = sum(p for _,p in gp)
            exp_goals = sum(g*(p/tp) for g,p in gp)
    except: pass

    # ========== 核心决策：Claude输出 + 3道硬约束后校验 ==========
    final_score = claude_score if claude_score else engine_score

    # 硬约束1：Phase1三家全票一致 → 强制采用（Claude不能推翻3/3共识）
    if p1_majority_count >= 3 and get_crs(p1_majority) <= 12.0:
        if final_score != p1_majority:
            print(f"    🔒 硬约束1: 3/3共识{p1_majority} 覆盖Claude的{final_score}")
            final_score = p1_majority

    # 硬约束2：Phase1两家一致 + Claude不同 → 检查谁的CRS更低
    elif p1_majority_count >= 2 and claude_score and claude_score != p1_majority:
        crs_p1 = get_crs(p1_majority)
        crs_cl = get_crs(claude_score)
        if crs_p1 <= crs_cl and crs_p1 <= 10.0:
            print(f"    🔒 硬约束2: 2/3共识{p1_majority}(CRS@{crs_p1}) 覆盖Claude的{claude_score}(CRS@{crs_cl})")
            final_score = p1_majority
        elif crs_cl <= 10.0:
            final_score = claude_score  # Claude的CRS更低，允许
        else:
            final_score = p1_majority  # 都不好，用多数

    # 硬约束3：总进球数合理性校验
    try:
        fh, fa = map(int, final_score.split("-"))
        total = fh + fa
        # 如果预期进球≥2.3但比分总进球≤1 → 不合理
        if exp_goals >= 2.3 and total <= 1:
            # 找Phase1中进球数最接近预期的
            best_alt = final_score
            best_diff = abs(total - exp_goals)
            for sc, _ in p1_scores:
                try:
                    sh, sa = map(int, sc.split("-"))
                    diff = abs((sh+sa) - exp_goals)
                    if diff < best_diff and get_crs(sc) <= 12.0:
                        best_alt = sc; best_diff = diff
                except: pass
            if best_alt != final_score:
                print(f"    🔒 硬约束3: 预期{exp_goals:.1f}球但{final_score}仅{total}球 → 改为{best_alt}")
                final_score = best_alt
        # 如果预期进球≤1.5但比分总进球≥4 → 不合理
        elif exp_goals <= 1.5 and total >= 4:
            best_alt = final_score; best_diff = abs(total - exp_goals)
            for sc, _ in p1_scores:
                try:
                    sh, sa = map(int, sc.split("-"))
                    diff = abs((sh+sa) - exp_goals)
                    if diff < best_diff and get_crs(sc) <= 12.0:
                        best_alt = sc; best_diff = diff
                except: pass
            if best_alt != final_score:
                print(f"    🔒 硬约束3: 预期{exp_goals:.1f}球但{final_score}有{total}球 → 改为{best_alt}")
                final_score = best_alt
    except: pass

    # CRS最终检查：比分赔率>15倍=不现实，降级
    final_crs = get_crs(final_score)
    if final_crs > 15.0 and p1_majority and get_crs(p1_majority) <= 12.0:
        print(f"    🔒 CRS检查: {final_score}@{final_crs}倍太高 → 改为{p1_majority}")
        final_score = p1_majority

    # ========== 0-0通道 ==========
    exp_analysis = stats.get("experience_analysis", {})
    zero_zero_boost = exp_analysis.get("zero_zero_boost", 0) if isinstance(exp_analysis, dict) else 0
    a0_val = float(match_obj.get("a0", 99) or 99)
    s00_val = float(match_obj.get("s00", 99) or 99)

    if zero_zero_boost >= 10:
        if any(sc == "0-0" for sc, _ in p1_scores):
            final_score = "0-0"
        elif a0_val < 8.0 and s00_val < 9.0:
            final_score = "0-0"
        elif final_score in ["1-0", "0-1", "1-1"] and zero_zero_boost >= 14:
            final_score = "0-0"
    elif zero_zero_boost >= 6 and a0_val < 8.5 and final_score == "1-1":
        final_score = "0-0"

    # ========== 信心/概率/输出 ==========
    all_scores = [(sc, n) for sc, n in p1_scores]
    if claude_score: all_scores.append((claude_score, "claude"))
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
    has_warn = any("🚨" in str(s) for s in stats.get("smart_signals", []))
    if has_warn: cf = max(35, cf - 12)
    risk = "低" if cf >= 75 else ("中" if cf >= 55 else "高")

    hp = engine_result.get("home_prob", 33); dp = engine_result.get("draw_prob", 33); ap = engine_result.get("away_prob", 34)
    shp = stats.get("home_win_pct", 33); sdp = stats.get("draw_pct", 33); sap = stats.get("away_win_pct", 34)
    fhp = hp * 0.70 + shp * 0.30; fdp = dp * 0.70 + sdp * 0.30; fap = ap * 0.70 + sap * 0.30
    fhp = max(3, fhp); fdp = max(3, fdp); fap = max(3, fap)
    ft = fhp + fdp + fap
    if ft > 0: fhp = round(fhp/ft*100, 1); fdp = round(fdp/ft*100, 1); fap = round(max(3, 100-fhp-fdp), 1)

    gpt_sc = gpt_r.get("ai_score","-") if isinstance(gpt_r, dict) else "-"
    gpt_an = gpt_r.get("reason", gpt_r.get("analysis","N/A")) if isinstance(gpt_r, dict) else "N/A"
    grok_sc = grok_r.get("ai_score","-") if isinstance(grok_r, dict) else "-"
    grok_an = grok_r.get("reason", grok_r.get("analysis","N/A")) if isinstance(grok_r, dict) else "N/A"
    gem_sc = gemini_r.get("ai_score","-") if isinstance(gemini_r, dict) else "-"
    gem_an = gemini_r.get("reason", gemini_r.get("analysis","N/A")) if isinstance(gemini_r, dict) else "N/A"
    cl_sc = final_score  # 最终比分就是Claude显示的比分
    cl_an = claude_r.get("reason", claude_r.get("analysis","N/A")) if isinstance(claude_r, dict) else "N/A"

    pre_pred = {"home_win_pct": fhp, "draw_pct": fdp, "away_win_pct": fap, "steam_move": stats.get("steam_move", {}), "smart_signals": stats.get("smart_signals", []), "line_movement_anomaly": stats.get("line_movement_anomaly", {})}
    cold_door = ColdDoorDetector.detect(match_obj, pre_pred)
    sigs = list(stats.get("smart_signals", []))
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
# run_predictions v3.5
# ====================================================================
def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 80)
    print(f"  [QUANT ENGINE vMAX 7.0] 冷门猎手模式 | {len(ms)} 场比赛")
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
    diary["reflection"] = f"vMAX7.0 | {cold_count}冷门 | 5层增强 | AI投票解放+0-0通道"
    save_ai_diary(diary)
    return res, t4