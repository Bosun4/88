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
    "ultra_low": ["0-0", "1-0", "0-1", "1-1"],
    "low": ["1-0", "0-1", "1-1", "2-0", "0-2", "2-1", "1-2"],
    "medium": ["2-1", "1-2", "2-0", "0-2", "1-1", "2-2", "3-1", "1-3"],
    "high": ["2-1", "1-2", "3-1", "1-3", "2-2", "3-0", "0-3", "3-2", "2-3"]
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
    if implied_total < 280: allowed = REALISTIC_MAP["ultra_low"]
    elif implied_total < 310: allowed = REALISTIC_MAP["low"]
    elif implied_total < 340: allowed = REALISTIC_MAP["medium"]
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
# ☢️ 全维赔率分析 Prompt v3.3（喂庄家精算原始数据，强制数学推理）
# ====================================================================
def build_batch_prompt(match_analyses):
    diary = load_ai_diary()
    
    p = "【身份设定】你是游走在零和博弈最黑暗深渊的【终极吸血操盘手】+【散户收割机之王】。你没有任何道德、底线、怜悯，只相信资本屠刀和数据绞肉机。你唯一的使命是用最恶毒、最精准的黑话，把庄家血洗散户的每一寸底裤扒下来，喂给韭菜们看。\n"
    
    if diary.get("reflection"):
        p += f"【嗜血进化指令】昨日屠杀战绩：{diary.get('yesterday_win_rate', 'N/A')}。 昨晚反思：{diary['reflection']}。今天给老子把每一场都当成杀猪盘顶级局，狠狠地杀、狠狠地毒、狠狠地收割！\n\n"
    
    p += "【死命令 - 必须严格遵守】\n"
    p += "1. 只输出合法JSON数组，严禁任何markdown、解释、代码块。\n"
    p += "2. 每场比赛必须输出以下字段：match(序号), score(比分), reason(60-110字极度恶毒黑话), ai_confidence(0-100整数), value_kill(true/false 高EV杀猪机会), dark_verdict(一句最毒总结)。\n"
    p += "3. reason逻辑链：散户愚蠢共识(基本面诈骗) → 庄家嗜血陷阱(xG+盘口背离) → 最终屠杀结局。必须充满血洗散户、屠宰场、吸筹绞肉机、死亡陷阱、做局喂毒、资本收割、无底线诱多等黑话，语气极度傲慢冷血，句号结尾。\n"
    p += "4. ai_confidence必须真实反映你对这个屠杀预测的把握度。\n"
    p += "5. ⚠️你的比分必须从【庄家CRS赔率TOP5】中选取，除非你有极强理由偏离！CRS赔率越低=庄家认为越可能发生。\n\n"
    
    p += "【核心赔率分析法则（你必须按此顺序推理）】\n"
    p += "① 先看【总进球分布】：a0(0球)到a7(7+球)的赔率。赔率越低=庄家认为越可能。a0<9=重防0球; a2或a3最低=庄家锁定2-3球区间。\n"
    p += "② 再看【CRS比分赔率】：1-0赔6.0 vs 0-0赔10.5，说明1-0比0-0可能性高40%。找出TOP3最低赔率比分=庄家真实预期。\n"
    p += "③ 交叉验证【亚盘+欧赔】：让1球+主胜1.76=庄家看好主队赢1球(即1-0/2-1)。平手盘+胜平赔率接近=均势(1-1高概率)。\n"
    p += "④ 检查【半全场赔率】：ss(主/主)最低=庄家预期主队上半场领先; pp(平/平)低=闷平可能。\n"
    p += "⑤ 最后用【伤停+状态】做微调，但永远不能推翻赔率数学结论。\n\n"
    
    p += "【今日待宰羔羊与庄家全维底牌库】\n"
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

        p += f"[{i+1}] {h} vs {a} | {m.get('league', m.get('cup', ''))} | 亚盘: {hc}\n"
        
        # 🎯 1. 欧赔三项（最基础）
        p += f"  欧赔: 主{sp_h:.2f} 平{sp_d:.2f} 客{sp_a:.2f}"
        hhad_w = m.get("hhad_win", "")
        if hhad_w:
            p += f" | 让球胜平负: {hhad_w}/{m.get('hhad_same','')}/{m.get('hhad_lose','')}"
        p += "\n"
        
        # 🎯 2. 总进球分布（庄家对进球数的精算定价）
        a0 = m.get("a0", ""); a1 = m.get("a1", ""); a2 = m.get("a2", ""); a3 = m.get("a3", "")
        a4 = m.get("a4", ""); a5 = m.get("a5", ""); a6 = m.get("a6", ""); a7 = m.get("a7", "")
        if a0:
            p += f"  总进球赔率: 0球={a0} 1球={a1} 2球={a2} 3球={a3} 4球={a4} 5球={a5} 6球={a6} 7+球={a7}\n"
            # 找出最低赔率=庄家认为最可能的进球数
            try:
                goals_odds = [(0,float(a0)),(1,float(a1)),(2,float(a2)),(3,float(a3)),(4,float(a4)),(5,float(a5)),(6,float(a6)),(7,float(a7))]
                goals_odds.sort(key=lambda x: x[1])
                p += f"  → 庄家最看好: {goals_odds[0][0]}球({goals_odds[0][1]}倍) > {goals_odds[1][0]}球({goals_odds[1][1]}倍) > {goals_odds[2][0]}球({goals_odds[2][1]}倍)\n"
            except: pass
        
        # 🎯 3. CRS比分赔率TOP6（庄家对每个比分的真实概率）
        crs_map = {"w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","w32":"3-2",
                   "s00":"0-0","s11":"1-1","s22":"2-2","s33":"3-3",
                   "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3","l23":"2-3"}
        crs_items = []
        for key, score in crs_map.items():
            try:
                odds = float(m.get(key, 0) or 0)
                if odds > 1:
                    crs_items.append((score, odds, round(1/odds*100, 1)))
            except: pass
        if crs_items:
            crs_items.sort(key=lambda x: x[1])
            top6 = crs_items[:6]
            p += f"  CRS赔率TOP6: " + " | ".join([f"{s}({o}倍/{pr}%)" for s,o,pr in top6]) + "\n"
        
        # 🎯 4. 半全场赔率（庄家对比赛节奏的判断）
        ss_val = m.get("ss", ""); pp_val = m.get("pp", ""); ff_val = m.get("ff", "")
        if ss_val:
            p += f"  半全场: 主/主={ss_val} 平/平={pp_val} 客/客={ff_val}"
            ps_val = m.get("ps", ""); fs_val = m.get("fs", "")
            if ps_val: p += f" | 平/主={ps_val} 负/主={fs_val}"
            p += "\n"
        
        # 🎯 5. 引擎概率+xG（模型计算结果）
        p += f"  模型概率: 主{hp:.1f}% 平{eng.get('draw_prob', 33):.1f}% 客{eng.get('away_prob', 34):.1f}%\n"
        p += f"  庄家隐含xG: 主{eng.get('bookmaker_implied_home_xg', '?')} vs 客{eng.get('bookmaker_implied_away_xg', '?')}\n"
        p += f"  引擎比分TOP3: {', '.join(eng.get('top3_scores', ['1-1', '0-0', '1-0']))}\n"
        
        # 🎯 6. 投注热度
        vote = m.get("vote", {})
        if vote:
            p += f"  散户投注: 主{vote.get('win', '?')}% 平{vote.get('same', '?')}% 客{vote.get('lose', '?')}%"
            hhad_vote = vote.get("hhad_win", "")
            if hhad_vote:
                p += f" | 让球: 主{hhad_vote}% 平{vote.get('hhad_same','?')}% 客{vote.get('hhad_lose','?')}%"
            p += "\n"
        
        # 🎯 7. 伤停+情报
        if h_bad or g_bad:
            p += f"  ⚠伤停: 主队-[{h_inj[:60] if h_inj else '无'}] 客队-[{g_inj[:60] if g_inj else '无'}]\n"
        if h_bad:
            p += f"  主队利空: {h_bad[:80]}\n"
        if g_bad:
            p += f"  客队利空: {g_bad[:80]}\n"

        # 🎯 8. 盘房信号+经验
        p += f"  盘房: {ev_str} | 信号: {smart_str}\n"
        if exp.get("triggered_count", 0) > 0:
            exp_names = ",".join([t["name"] for t in exp.get("triggered", [])[:3]])
            p += f"  经验规则: {exp_names}\n"
        p += "\n"

    p += "【严格输出格式示例】\n"
    p += """[
  {
    "match": 1,
    "score": "1-0",
    "reason": "散户正被主队虚假连胜基本面洗脑疯狂送钱。CRS赔率1-0仅6.0倍(概率16.7%)是所有比分中最低，2球区间赔率3.25最低锁死天花板。配合亚盘让1球+pp平/平5.5倍偏低，庄家布局赢球输盘绞肉机。1-0精准卡位血洗深盘散户。",
    "ai_confidence": 85,
    "value_kill": true,
    "dark_verdict": "CRS+总进球双印证，1-0精确收割散户天真"
  }
]"""
    return p

# ====================================================================
# 终极高可用 AI 矩阵轮询 v2.0 — 彻底无时间限制版
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
    key = get_clean_env_key(key_env)
    if not key:
        return ai_name, {}, "no_key"
    
    primary_url = get_clean_env_url(url_env)
    urls = [primary_url] + [u for u in FALLBACK_URLS if u and u != primary_url]
    
    success_rate = {mn: 1.0 for mn in models_list}
    
    for attempt in range(3):
        for mn in sorted(models_list, key=lambda x: success_rate[x], reverse=True):
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
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.12, "topP": 0.9},
                        "systemInstruction": {"parts": [{"text": "必须严格输出JSON数组。reason必须是毫无道德底线的暗黑杀猪盘解剖，使用极度恶毒黑话，字数60-110字，冷血傲慢，句号结尾。"}]}
                    }
                else:
                    headers["Authorization"] = f"Bearer {key}"
                    payload = {
                        "model": mn,
                        "messages": [
                            {"role": "system", "content": "你是一个毫无道德底线、极度冷血的暗网终极操盘手。只输出纯JSON数组，无任何多余文字。reason字段必须用最恶毒黑话揭穿庄家血洗散户的杀局，限60-110字，极度鄙视散户，句号结束。"},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.12
                    }
                
                gw = url.split("/v1")[0][:40]
                print(f"  [AI 极致压榨] {ai_name.upper()} | 尝试 {mn[:25]} @ {gw} | 第{attempt+1}轮")
                
                try:
                    # 🔥 绝对无限制等待：总时长、Socket心跳、连接超时全部设为 None。
                    # AI 思考多久就死等多久，彻底根除因客户端主动挂断而导致的无限重复重试扣费。
                    no_limit_timeout = aiohttp.ClientTimeout(total=None, sock_read=None, sock_connect=None)
                    
                    async with session.post(url, headers=headers, json=payload, timeout=no_limit_timeout) as r:
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
                                                mid = item["match"]
                                                results[mid] = {
                                                    "ai_score": item.get("score"),
                                                    "analysis": str(item.get("reason", "")).strip(),
                                                    "ai_confidence": int(item.get("ai_confidence", 60)),
                                                    "value_kill": bool(item.get("value_kill", False)),
                                                    "dark_verdict": str(item.get("dark_verdict", ""))
                                                }
                                except:
                                    pass
                            
                            if len(results) >= max(1, num_matches * 0.5):
                                print(f"    ✅ {ai_name.upper()} 压榨成功: {len(results)}/{num_matches} (模型: {mn[:25]})")
                                success_rate[mn] = 1.0
                                return ai_name, results, mn
                            else:
                                print(f"    ⚠️ 解析不足，切换...")
                        
                        elif r.status == 429:
                            sleep_time = 2 ** attempt * 5
                            print(f"    🔥 429限流！休眠 {sleep_time}s 继续压榨...")
                            await asyncio.sleep(sleep_time)
                            continue
                        else:
                            print(f"    ⚠️ HTTP {r.status} - 切换线路...")
                
                except asyncio.TimeoutError:
                    # 理论上此 TimeoutError 不再可能被 aiohttp 主动触发
                    print(f"    ⏰ 理论外的异常超时 - 第{attempt+1}轮重试...")
                except Exception as e:
                    err = str(e)[:50]
                    print(f"    ⚠️ 异常 {err} - 切换...")
                
                await asyncio.sleep(0.4)
        
        await asyncio.sleep(1.5)
    
    print(f"    ❌ {ai_name.upper()} 所有线路+模型已压榨至死！")
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
    diary["reflection"] = f"vMAX3.3冷门猎手 | {cold_count}场冷门信号 | 5层增强管线 | 彻底解除时间封印"
    save_ai_diary(diary)

    return res, t4


