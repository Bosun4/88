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
# ☢️ 极致压榨AI v2.1 核心升级思路（继续吸血进化）
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
# 🧊 冷门猎手引擎 + 深度赔率映射（v3.3新增，不影响原有逻辑）
# ====================================================================
REALISTIC_MAP = {
    "ultra_low": ["0-0", "1-0", "0-1", "1-1", "2-0", "0-2"],
    "low": ["1-0", "0-1", "1-1", "2-0", "0-2", "2-1", "1-2", "0-0"],
    "medium": ["1-0", "0-1", "1-1", "2-0", "0-2", "2-1", "1-2", "2-2", "3-0", "0-3", "3-1", "1-3"],
    "high": ["2-1", "1-2", "3-1", "1-3", "2-2", "3-0", "0-3", "3-2", "2-3", "4-0", "0-4", "4-1", "1-4"]
}

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

def calibrate_realistic_score(current_score, sp_h, sp_d, sp_a, cold_door):
    """赔率映射+冷门强度联合校准：防止输出不合理高比分"""
    if cold_door.get("is_cold_door") and cold_door.get("level") == "顶级":
        return current_score  # 顶级冷门允许原预测

    implied_total = (1/sp_h + 1/sp_d + 1/sp_a) * 100 if sp_h > 1 and sp_d > 1 and sp_a > 1 else 300
    if implied_total < 270: allowed = REALISTIC_MAP["ultra_low"]
    elif implied_total < 300: allowed = REALISTIC_MAP["low"]
    elif implied_total < 330: allowed = REALISTIC_MAP["medium"]
    else: allowed = REALISTIC_MAP["high"]

    if current_score in allowed:
        return current_score
    try:
        home, away = map(int, current_score.split("-"))
        if home + away <= 2: return "1-0" if home >= away else "0-1"
        elif home + away <= 3: return "2-1" if home >= away else "1-2"
        else: return "2-1" if home >= away else "1-2"
    except:
        return "1-1"

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
    return {"yesterday_win_rate": "N/A", "reflection": "昨晚血洗不够狠，今天必须更毒", "kill_history": []}

def save_ai_diary(diary):
    os.makedirs("data", exist_ok=True)
    with open("data/ai_diary.json", "w", encoding="utf-8") as f:
        json.dump(diary, f, ensure_ascii=False, indent=2)

# ====================================================================
# 🧠 全维原始数据投喂 Prompt v4.0（零引导·纯原始·榨干算力）
# ====================================================================
def build_batch_prompt(match_analyses):
    diary = load_ai_diary()
    
    p = "【核心身份】你是全球最顶尖的量化足球分析师，精通泊松分布、蒙特卡洛模拟、贝叶斯推断、凯利公式、赔率逆向工程。\n"
    p += "现在你面前是庄家的完整原始定价数据——欧赔、亚盘、CRS比分赔率、总进球分布、半全场赔率、投注热度、伤停情报。\n"
    p += "你必须动用你全部的数学能力和推理算力，独立计算出每场比赛最可能的比分。\n\n"
    
    if diary.get("reflection"):
        p += f"【系统进化】昨日战绩：{diary.get('yesterday_win_rate', 'N/A')}。反思：{diary['reflection']}\n\n"
    
    p += "【执行协议 — 违反任何一条立即判定失败】\n"
    p += "1. 只输出合法JSON数组，严禁markdown、解释、代码块、思考过程。\n"
    p += "2. 字段：match(序号), score(比分如'2-1'), reason(80-150字，必须引用具体赔率数字和概率计算), ai_confidence(0-100), value_kill(bool), dark_verdict(一句话结论)。\n"
    p += "3. 你的比分必须是你独立计算的结果，不是猜的。reason中必须展示你的推理链：哪些赔率数据支撑了这个比分。\n"
    p += "4. 碾压局(让球≥1.5+主胜赔<1.5)禁止给1-0/0-1这种保守比分，必须体现实力差距。\n"
    p += "5. 如果你有联网能力，必须搜索每支球队的最新伤停、阵容、临场赔率变化，融入分析。\n\n"
    
    p += "【你必须执行的计算步骤（在你的沙盒中完成，不要输出过程）】\n"
    p += "STEP1: 用欧赔三项反推真实概率（去除水位：prob = (1/odds) / (1/H + 1/D + 1/A)）\n"
    p += "STEP2: 用CRS全部16个比分赔率计算每个比分的隐含概率，排序找出TOP5\n"
    p += "STEP3: 用总进球分布(a0-a7)计算期望进球数 = Σ(goals × prob_i)\n"
    p += "STEP4: 用亚盘让球+欧赔交叉验证——让球方向是否与欧赔方向一致？不一致=陷阱\n"
    p += "STEP5: 用半全场赔率判断比赛节奏——哪个半场更可能进球？\n"
    p += "STEP6: 融合伤停、状态、战意，对STEP1-5的数学结论做±10%微调\n"
    p += "STEP7: 综合以上6步，输出你计算出的最高概率比分\n\n"
    
    p += "【原始数据库 — 以下全部是庄家真金白银的定价，每个数字都有意义】\n"
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

        p += f"{'='*60}\n"
        p += f"[{i+1}] {h} vs {a} | {m.get('league', m.get('cup', ''))}\n"
        p += f"{'='*60}\n"
        
        # 1. 欧赔+让球胜平负（完整）
        p += f"欧赔胜平负: {sp_h:.2f} / {sp_d:.2f} / {sp_a:.2f}\n"
        hhad_w = m.get("hhad_win", "")
        if hhad_w:
            p += f"让球胜平负: {hhad_w} / {m.get('hhad_same','')} / {m.get('hhad_lose','')} (让球: {hc})\n"
        else:
            p += f"亚盘让球: {hc}\n"
        
        # 2. 总进球分布（完整8档）
        a0 = m.get("a0", ""); a1 = m.get("a1", ""); a2 = m.get("a2", ""); a3 = m.get("a3", "")
        a4 = m.get("a4", ""); a5 = m.get("a5", ""); a6 = m.get("a6", ""); a7 = m.get("a7", "")
        if a0:
            p += f"总进球赔率: 0球={a0} | 1球={a1} | 2球={a2} | 3球={a3} | 4球={a4} | 5球={a5} | 6球={a6} | 7+球={a7}\n"
        
        # 3. CRS比分赔率（完整16+个比分，不做排序，让AI自己算）
        crs_lines = []
        crs_map_full = {
            "w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","w32":"3-2",
            "w40":"4-0","w41":"4-1","w42":"4-2","w50":"5-0","w51":"5-1","w52":"5-2",
            "s00":"0-0","s11":"1-1","s22":"2-2","s33":"3-3",
            "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3","l23":"2-3",
            "l04":"0-4","l14":"1-4","l24":"2-4","l05":"0-5","l15":"1-5","l25":"2-5"
        }
        for key, score in crs_map_full.items():
            try:
                odds = float(m.get(key, 0) or 0)
                if odds > 1:
                    crs_lines.append(f"{score}={odds}")
            except: pass
        if crs_lines:
            p += f"CRS比分赔率(全部): {' | '.join(crs_lines)}\n"
        
        # 4. 半全场赔率（完整9格）
        hf_map = {"ss":"主/主","sp":"主/平","sf":"主/负","ps":"平/主","pp":"平/平","pf":"平/负","fs":"负/主","fp":"负/平","ff":"负/负"}
        hf_lines = []
        for key, label in hf_map.items():
            try:
                odds = float(m.get(key, 0) or 0)
                if odds > 1:
                    hf_lines.append(f"{label}={odds}")
            except: pass
        if hf_lines:
            p += f"半全场赔率(全部): {' | '.join(hf_lines)}\n"
        
        # 5. 投注热度（完整）
        vote = m.get("vote", {})
        if vote:
            p += f"散户投注分布: 胜{vote.get('win','?')}% 平{vote.get('same','?')}% 负{vote.get('lose','?')}%"
            if vote.get("hhad_win"):
                p += f" | 让球: 主{vote['hhad_win']}% 平{vote.get('hhad_same','?')}% 客{vote.get('hhad_lose','?')}%"
            p += "\n"
        
        # 6. 庄家隐含xG
        ixh = eng.get('bookmaker_implied_home_xg', '?')
        ixa = eng.get('bookmaker_implied_away_xg', '?')
        if ixh != '?':
            p += f"庄家隐含xG: 主队{ixh} vs 客队{ixa}\n"
        
        # 7. 赔率变动
        change = m.get("change", {})
        if change and isinstance(change, dict):
            cw = change.get("win", 0); cs = change.get("same", 0); cl = change.get("lose", 0)
            if cw or cs or cl:
                p += f"赔率变动: 胜{'+' if cw>0 else ''}{cw} 平{'+' if cs>0 else ''}{cs} 负{'+' if cl>0 else ''}{cl}\n"
        
        # 8. 伤停情报（完整喂入，不截断）
        info = m.get("information", {})
        if isinstance(info, dict):
            h_good = str(info.get("home_good_news", ""))[:200].replace('\n', ' | ')
            g_good = str(info.get("guest_good_news", ""))[:200].replace('\n', ' | ')
            h_bad = str(info.get("home_bad_news", ""))[:200].replace('\n', ' | ')
            g_bad = str(info.get("guest_bad_news", ""))[:200].replace('\n', ' | ')
            h_inj = str(info.get("home_injury", ""))[:120].replace('\n', ' | ')
            g_inj = str(info.get("guest_injury", ""))[:120].replace('\n', ' | ')
            
            if h_inj: p += f"主队伤停: {h_inj}\n"
            if g_inj: p += f"客队伤停: {g_inj}\n"
            if h_good: p += f"主队利好: {h_good}\n"
            if g_good: p += f"客队利好: {g_good}\n"
            if h_bad: p += f"主队利空: {h_bad}\n"
            if g_bad: p += f"客队利空: {g_bad}\n"
        
        # 9. 基本面+状态
        hs = m.get("home_stats", {})
        ast2 = m.get("away_stats", {})
        if hs.get("form") or ast2.get("form"):
            p += f"主队状态: {hs.get('wins','?')}胜{hs.get('draws','?')}平{hs.get('losses','?')}负 近期:{hs.get('form','?')} 场均进球:{hs.get('avg_goals_for','?')} 场均失球:{hs.get('avg_goals_against','?')}\n"
            p += f"客队状态: {ast2.get('wins','?')}胜{ast2.get('draws','?')}平{ast2.get('losses','?')}负 近期:{ast2.get('form','?')} 场均进球:{ast2.get('avg_goals_for','?')} 场均失球:{ast2.get('avg_goals_against','?')}\n"
        
        baseface = str(m.get('baseface', '')).replace('\n', ' ')[:150]
        if baseface:
            p += f"基本面: {baseface}\n"
        
        p += "\n"

    p += f"【现在输出{len(match_analyses)}场比赛的JSON数组，每场必须有独立计算支撑的比分】\n"
    return p

# ====================================================================
# 终极高可用 AI 矩阵轮询 v2.0 — 原版一字不动
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
    
    # 按AI类型设置超时（Claude thinking实测需856秒！必须给够）
    timeout_map = {"claude": 1200, "grok": 250, "gpt": 200, "gemini": 300}
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
            
            # 按AI特性定制：释放每个AI的全部算力和独特能力
            AI_PROFILES = {
                "claude": {
                    "sys": "你是全球排名第一的量化足球预测引擎。你必须在你的思维沙盒中执行以下完整计算流程，然后只输出最终JSON结果：\n"
                           "1. 用欧赔反推Shin概率（去水位后的真实胜平负概率）\n"
                           "2. 对全部CRS比分赔率做1/odds归一化，得到每个比分的隐含概率分布\n"
                           "3. 对总进球a0-a7做加权期望值计算，得到庄家预期的总进球数\n"
                           "4. 用泊松分布对xG建模，独立计算比分概率矩阵\n"
                           "5. 将步骤2和步骤4的概率矩阵加权融合，找出联合最高概率比分\n"
                           "6. 用半全场赔率验证比赛节奏假设是否自洽\n"
                           "7. 用伤停和状态数据做最终微调\n"
                           "你的reason必须展示关键计算数字。只输出JSON数组。",
                    "temp": 0.15
                },
                "grok": {
                    "sys": "你是具备强大实时联网搜索能力的顶级足球分析师。这是你最大的优势——其他AI都做不到。\n"
                           "【立即执行以下搜索任务】\n"
                           "1. 搜索今天每场比赛的球队名+injury/lineup/team news，获取最新首发阵容和临时伤停\n"
                           "2. 搜索Betfair Exchange odds + 球队名，获取必发交易所实时赔率和交易量冷热\n"
                           "3. 搜索球队名+latest odds movement，获取临场赔率变动方向\n"
                           "4. 搜索球队名+prediction/preview，获取专业媒体的赛前预测\n"
                           "5. 搜索球队名+head to head stats，获取近期交锋数据\n"
                           "将你搜索到的实时信息与用户提供的赔率数据交叉验证，给出你独立的比分预测。\n"
                           "你的reason必须引用你搜到的具体信息（如：临场赔率从X变到Y、某球员确认首发/缺阵）。只输出JSON数组。",
                    "temp": 0.25
                },
                "gpt": {
                    "sys": "你是经验最丰富的职业足球博彩分析师，拥有20年实战经验。你必须用你的全部推理能力独立分析每场比赛。\n"
                           "你的分析方法：\n"
                           "1. 先看CRS比分赔率——找出庄家定价最低的3个比分，这是庄家真金白银给出的概率判断\n"
                           "2. 再看总进球分布——确定比赛的进球天花板在哪里\n"
                           "3. 用亚盘和欧赔交叉验证——如果方向不一致，说明有陷阱\n"
                           "4. 用半全场赔率判断比赛节奏和走势\n"
                           "5. 最后看伤停和基本面做微调\n"
                           "不要保守！如果数据明确指向大比分就给大比分，如果数据指向冷门就给冷门。你的价值在于敢于下判断。只输出JSON数组。",
                    "temp": 0.18
                },
                "gemini": {
                    "sys": "你是精通概率论和模式识别的量化分析引擎。你必须对提供的赔率数据执行完整的数学建模：\n"
                           "1. 将CRS全部比分赔率转换为概率分布（1/odds归一化），绘制概率矩阵\n"
                           "2. 将总进球赔率转换为进球数概率分布，计算数学期望\n"
                           "3. 用欧赔反推真实概率：true_prob = (1/odds) / sum(1/all_odds)\n"
                           "4. 检测赔率异常：如果某个比分的CRS隐含概率与泊松模型偏差>50%，标记为庄家刻意操纵\n"
                           "5. 如果你有联网能力，搜索每支球队的最新阵容和伤停\n"
                           "你的reason必须包含你计算出的关键概率数字。只输出JSON数组。",
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
                    "generationConfig": {"temperature": ai_temp},
                    "systemInstruction": {"parts": [{"text": sys_msg}]}
                }
            else:
                headers["Authorization"] = f"Bearer {key}"
                # Claude API不允许temperature+top_p同时存在，代理会自动注入top_p
                # 所以Claude不传temperature（thinking模型用默认值效果最好）
                # 其他AI正常传temperature
                base_payload = {
                    "model": mn,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": prompt}
                    ]
                }
                if ai_name != "claude":
                    base_payload["temperature"] = ai_temp
                payload = base_payload
            
            gw = url.split("/v1")[0][:35]
            print(f"  [⏳{timeout_sec}s] {ai_name.upper()} | {mn[:22]} @ {gw}")
            t0 = time.time()
            
            try:
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
                                            # 强制转int：Grok/Gemini可能返回"match":"1"(字符串)
                                            # 但run_predictions用int索引查找
                                            try:
                                                mid = int(item["match"])
                                            except:
                                                mid = item["match"]
                                            results[mid] = {
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
                    
                    elif r.status == 400:
                        # 400 = 参数冲突(如temperature+top_p)，换URL不会解决，跳模型
                        print(f"    💀 HTTP 400 参数冲突 | {elapsed}s → 跳过此模型")
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
            "熊猫特供-超纯满血-99额度-claude-opus-4.6-thinking",
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
# Merge 智能融合 v2.0 — 原版逻辑 + 冷门检测 + 比分校准
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
    if vote_count:
        best_voted = max(vote_count, key=vote_count.get)
        if best_voted in engine_result.get("top3_scores", []) and vote_count[best_voted] >= 2:
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

    # ========== v3.3新增：冷门信号识别 ==========
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

    # ========== v3.3新增：比分现实校准 ==========
    final_score = calibrate_realistic_score(final_score, sp_h, sp_d, sp_a, cold_door)

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
# ☢️ run_predictions v3.3 — 原版调用链 + 5层try/except + 冷门标签
# ====================================================================
def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 80)
    print(f"  [QUANT ENGINE vMAX 3.3] 冷门猎手模式 | {len(ms)} 场比赛")
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
        print(f"  [PROMPT] 已生成 {len(prompt):,} 字符的极致毒prompt，开始压榨AI矩阵...")
        start_t = time.time()
        all_ai = asyncio.run(run_ai_matrix(prompt, len(match_analyses)))
        print(f"  [AI MATRIX] 压榨完成，耗时 {time.time()-start_t:.1f}s")

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
            print(f"    → apply_experience_to_prediction 已注入（经验法则加成）")
        except Exception as e:
            print(f"    ⚠️ experience_rules跳过: {e}")
        
        try:
            mg = apply_odds_history(m, mg)
            print(f"    → apply_odds_history 已尝试注入（历史盘口血洗信号）")
        except Exception as e:
            print(f"    ⚠️ odds_history跳过: {e}")
        
        try:
            mg = apply_quant_edge(m, mg)
            print(f"    → apply_quant_edge 已尝试注入（极致量化边缘屠杀）")
        except Exception as e:
            print(f"    ⚠️ quant_edge跳过: {e}")

        try:
            mg = apply_wencai_intel(m, mg)
            print(f"    → apply_wencai_intel 已注入（文彩情报增强）")
        except Exception as e:
            print(f"    ⚠️ wencai_intel跳过: {e}")
        
        try:
            mg = upgrade_ensemble_predict(m, mg)
            print(f"    → upgrade_ensemble_predict 已注入（最终集成强化）")
        except Exception as e:
            print(f"    ⚠️ advanced_models跳过: {e}")
        # =====================================================================
        
        # result必须与比分一致（旧版用概率决定result，导致1-1标记为主胜的矛盾）
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
        
        # v3.3新增：冷门标签
        cold = mg.get("cold_door", {})
        cold_tag = f" [❄️{cold.get('level','')}冷门]" if cold.get("is_cold_door") else ""
        print(f"  [{i+1}] {m.get('home_team')} vs {m.get('away_team')} => {mg['result']} ({mg['predicted_score']}) | CF: {mg['confidence']}% | AI信心: {mg.get('ai_avg_confidence', 0)}{cold_tag}")

    t4 = select_top4(res)
    t4ids = [t["id"] for t in t4]
    for r in res:
        r["is_recommended"] = r["id"] in t4ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    diary = load_ai_diary()
    diary["yesterday_win_rate"] = f"{len([r for r in res if r['prediction']['confidence'] > 70])}/{max(1, len(res))}"
    cold_count = len([r for r in res if r.get("prediction", {}).get("cold_door", {}).get("is_cold_door")])
    diary["reflection"] = f"vMAX3.3冷门猎手 | {cold_count}场冷门信号 | 5层增强管线"
    save_ai_diary(diary)

    return res, t4