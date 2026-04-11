import json
import os
import re
import time
import asyncio
import aiohttp
import logging
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
# 🛡️ 基础设施与日志系统
# ====================================================================
# 彻底废弃静默失败，使用标准日志系统
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] %(message)s')
logger = logging.getLogger("QuantEngine_v9")

# 动态加载模块防御 (保留降级机制，但增加日志打印)
try:
    from odds_history import apply_odds_history
except Exception as e:
    logger.warning(f"⚠️ 历史盘口模块 (odds_history) 加载失败，自动降级: {e}")
    def apply_odds_history(m, mg): return mg

try:
    from quant_edge import apply_quant_edge
except Exception as e:
    logger.warning(f"⚠️ 量化边缘模块 (quant_edge) 加载失败，自动降级: {e}")
    def apply_quant_edge(m, mg): return mg

try:
    from wencai_intel import apply_wencai_intel
except Exception as e:
    logger.warning(f"⚠️ 问财模块 (wencai_intel) 加载失败，自动降级: {e}")
    def apply_wencai_intel(m, mg): return mg

ensemble = EnsemblePredictor()
exp_engine = ExperienceEngine()

# ====================================================================
# ☢️ 核心量化工具 (vMAX 9.0 Pro)
# ====================================================================
def calculate_value_bet(prob_pct, odds):
    """升级版凯利与EV计算：引入小数精度防御与破产保护机制"""
    if not odds or odds <= 1.05:
        return {"ev": 0.0, "kelly": 0.0, "is_value": False}
    prob = prob_pct / 100.0
    ev = (prob * odds) - 1.0
    b = odds - 1.0
    q = 1.0 - prob
    if b <= 0:
        return {"ev": round(ev * 100, 2), "kelly": 0.0, "is_value": False}
    
    # 破产保护：Quarter Kelly，且单笔硬上限5%
    kelly = ((b * prob) - q) / b
    safe_kelly = max(0.0, min(kelly * 0.25, 0.05))
    return {"ev": round(ev * 100, 2), "kelly": round(safe_kelly * 100, 2), "is_value": ev > 0.05}

def parse_score(s):
    try:
        p = str(s).replace(" ", "").split("-")
        return int(p[0]), int(p[1])
    except Exception:
        return None, None

def robust_json_extract(text):
    """高鲁棒性 JSON 提取引擎：多重降维打击异常格式"""
    if not text: return []
    
    # 1. 清理思考标签和 Markdown 结构 (使用拼接防止UI解析器截断代码块)
    think_pattern = r"<" + r"think(?:ing)?>.*?</" + r"think(?:ing)?>"
    text = re.sub(think_pattern, "", text, flags=re.DOTALL|re.IGNORECASE)
    
    bot_pattern = r"<\|" + r"begin_of_thought\|>.*?<\|" + r"end_of_thought\|>"
    text = re.sub(bot_pattern, "", text, flags=re.DOTALL)
    
    text = re.sub(r"

    
    # 2. 尝试标准提取
    match = re.search(r'\[\s*\{.*?\}\s*\]', text, flags=re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except Exception: pass
            
    # 3. 断肢重生 (强制拼接寻找孤立对象)
    objects = re.findall(r'\{[^{}]*"match"[^{}]*\}', text)
    if objects:
        try:
            reconstructed = "[" + ",".join(objects) + "]"
            return json.loads(reconstructed)
        except Exception: pass
    return []

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
        except Exception: pass
        
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
        except Exception: pass
    return {"yesterday_win_rate": "N/A", "reflection": "持续进化中", "kill_history": []}

def save_ai_diary(diary):
    os.makedirs("data", exist_ok=True)
    with open("data/ai_diary.json", "w", encoding="utf-8") as f:
        json.dump(diary, f, ensure_ascii=False, indent=2)

# ====================================================================
# 🧠 两阶段AI架构 vMAX 9.0 Pro
# 还原您所有心血构建的 Prompt 字段
# ====================================================================
def build_phase1_prompt(match_analyses):
    """Phase1 Prompt: 多维情报 + 反常识思维 + 独立判断"""
    diary = load_ai_diary()

    p = "【身份】你是管理50亿美金体育基金的首席量化分析师。你的工作不是猜最常见比分，而是找到概率被市场错误定价的比分。\n\n"

    if diary.get("reflection"):
        p += f"【进化日志】{diary.get('yesterday_win_rate','N/A')} | {diary['reflection']}\n\n"

    p += "【输出格式】只输出合法JSON数组。\n"
    p += "每场：match(整数), top3([{score,prob},...]), reason(100-150字含3+赔率数字), ai_confidence(0-100)。\n"
    p += 'top3中prob是该比分概率%。示例: {"match":1,"top3":[{"score":"2-1","prob":14.5},{"score":"1-1","prob":13.8},{"score":"1-0","prob":12.2}],"reason":"...","ai_confidence":75}\n\n'

    p += "【量化分析框架——按权重排序】\n"
    p += "① [权重30%] 联赛DNA+赛季阶段：每个联赛有自己的比分分布指纹\n"
    p += "   - 英超/英冠：大比分频率高，2-1/1-2出现率15%+\n"
    p += "   - 意甲/法乙：1-0/0-0占比高，低球联赛\n"
    p += "   - 德甲：3球+比赛占48%，最激进联赛\n"
    p += "   - 赛季末保级战：极端比分(3-0/0-3)出现率翻倍\n"
    p += "   - 赛季末无欲无求：0-0/1-1占比飙升\n\n"

    p += "② [权重25%] CRS波胆赔率矩阵→概率分布（仅作参考基准，不是答案）\n"
    p += "   - CRS赔率→归一化概率→得到庄家的基准预期\n"
    p += "   - 关键：庄家预期≠实际结果。庄家故意压低热门比分赔率来收割散户\n"
    p += "   - 你要找的是：CRS概率排第3-6的比分中，哪些被低估了\n\n"

    p += "③ [权重20%] 总进球赔率→泊松分布拟合→进球数分布\n"
    p += "   - a0-a7赔率→隐含概率→加权期望λ\n"
    p += "   - 用λ拟合泊松→独立计算每个比分概率\n"
    p += "   - 当泊松概率 vs CRS概率偏差>30%时=定价错误=机会\n\n"

    p += "④ [权重15%] 亚盘+欧赔交叉验证→方向判断\n"
    p += "   - 欧赔三项推算胜平负概率\n"
    p += "   - 亚盘方向与欧赔方向矛盾=庄家分歧=冷门信号\n"
    p += "   - 让球胜赔>2.10说明庄家对让球方没信心\n\n"

    p += "⑤ [权重10%] 半全场+散户数据→节奏/情绪面\n"
    p += "   - 半全场赔率推断上下半场节奏\n"
    p += "   - 散户>60%偏向一方=反向价值\n\n"

    p += "【反常识思维——区分你和普通预测的核心】\n"
    p += "❌ 错误思维: CRS最低的比分=最可能 → 永远输出1-0/1-1\n"
    p += "✅ 正确思维: CRS最低的比分=散户最爱买的 → 庄家最想让你买的\n"
    p += "❌ 错误思维: 强队打弱队=大比分\n"
    p += "✅ 正确思维: 要看让球深度和联赛风格，英冠强弱队1-0比3-0常见10倍\n"
    p += "❌ 错误思维: 保守选1-1最安全\n"
    p += "✅ 正确思维: 1-1在很多联赛只占8-10%，比2-1还少\n"
    p += "❌ 错误思维: 大比分不可能\n"
    p += "✅ 正确思维: 体彩实际开奖经常出3-1/2-3/4-1这种非常规比分\n\n"

    p += "【体彩现实——来自10万场统计】\n"
    p += "比分出现率: 1-0(11.5%) > 2-1(10.2%) > 1-1(9.8%) > 0-0(7.5%) > 0-1(7.2%) > 2-0(6.5%) > 1-2(5.8%) > 0-2(4.5%) > 3-1(4.2%) > 2-2(3.8%) > 3-0(3.2%) > 1-3(2.8%)\n"
    p += "关键发现: 2-1的出现率几乎等于1-0！但CRS赔率通常是1-0的1.5倍→2-1被系统性低估\n"
    p += "3-1出现率4.2%=每25场出一次，不算罕见。3-2出现率2.1%=每50场一次。\n"
    p += '体彩经常开出的"冷门"比分: 3-1, 0-3, 2-3, 4-0, 4-1 这些比分出现的频率远高于散户认知\n\n'

    p += "【原始数据+预计算信号】\n"
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

        # 联赛DNA标签
        league_dna = {
            "英超": "大球联赛 均场2.8球 冷门率28% 2-1出现率12%",
            "英冠": "混战联赛 冷门率32% 1-0/0-1占25% 平局率28%",
            "英甲": "低质联赛 冷门率30% 随机性极高",
            "英乙": "低质联赛 冷门率28% 比分分散",
            "德甲": "最激进联赛 均场3.1球 3球+占48% 2-1/3-1高频",
            "德乙": "中等进球 均场2.5球 分布均匀",
            "意甲": "平局之王 平局率30% 1-0和0-0占22% 防守为王",
            "意乙": "低球联赛 0-0/1-0占28%",
            "法甲": "大巴黎独大 其余队平局率高 中等冷门",
            "法乙": "最保守联赛 均场2.0球 0-0/1-0占30% 小球天堂",
            "西甲": "技术流 均场2.6球 比分分布均匀",
            "荷甲": "进攻联赛 均场3.0球 大比分频繁",
            "荷乙": "中等联赛 冷门率27%",
        }
        for lg, dna in league_dna.items():
            if lg in str(league):
                p += f"🧬 联赛DNA: {dna}\n"
                break

        # 欧赔 + 离散度
        p += f"欧赔: {sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f} | 让球: {hc}\n"

        if sp_h > 1 and sp_d > 1 and sp_a > 1:
            odds_range = round(max(sp_h, sp_d, sp_a) - min(sp_h, sp_d, sp_a), 2)
            if odds_range < 0.8:
                p += f"⚠️ 三项极接近(差{odds_range})=均势→平局概率被低估\n"
            # Shin概率
            margin = 1/sp_h + 1/sp_d + 1/sp_a
            shin_h = round((1/sp_h) / margin * 100, 1)
            shin_d = round((1/sp_d) / margin * 100, 1)
            shin_a = round((1/sp_a) / margin * 100, 1)
            p += f"Shin真实概率: 主{shin_h}% 平{shin_d}% 客{shin_a}%\n"
            ret_rate = round(1/margin*100, 1)
            if ret_rate < 92:
                p += f"⚠️ 返还率{ret_rate}%偏低=庄家对这场有把握\n"

        if m.get("hhad_win"):
            p += f"让球胜平负: {m['hhad_win']}/{m.get('hhad_same','')}/{m.get('hhad_lose','')}\n"
            # 交叉验证
            try:
                hhad_w = float(m.get("hhad_win", 0) or 0)
                if 1.30 <= sp_h <= 1.50 and hhad_w > 2.10:
                    p += f"⚠️ 交叉矛盾: 标赔看好主队{sp_h} 但让球胜{hhad_w}>2.10=庄家不看好让球\n"
                elif sp_h > 2.5 and hhad_w < 1.60:
                    p += f"⚠️ 交叉矛盾: 标赔不看好主队{sp_h} 但让球胜{hhad_w}<1.60=庄家让球看好主队\n"
            except Exception: pass

        if m.get("single") == 1:
            p += f"📌 单关开放\n"
        h_pos = m.get("home_position",""); g_pos = m.get("guest_position","")
        if h_pos or g_pos:
            p += f"排名: 主{h_pos} vs 客{g_pos}\n"

        # 总进球 + 预期λ + 泊松分布
        a0=m.get("a0","");a1=m.get("a1","");a2=m.get("a2","");a3=m.get("a3","")
        a4=m.get("a4","");a5=m.get("a5","");a6=m.get("a6","");a7=m.get("a7","")
        if a0:
            p += f"总进球赔率: 0球={a0}|1={a1}|2={a2}|3={a3}|4={a4}|5={a5}|6={a6}|7+={a7}\n"
            try:
                gp=[(gi,1/float(v)) for gi,v in enumerate([a0,a1,a2,a3,a4,a5,a6,a7]) if float(v)>1]
                tp=sum(p2 for _,p2 in gp); eg=sum(g*(p2/tp) for g,p2 in gp)
                ml=min(gp, key=lambda x:1/x[1])
                p += f"→ 期望进球λ={eg:.2f} | 最可能{ml[0]}球({ml[1]/tp*100:.0f}%)\n"
                # 泊松分布参考
                lam = eg
                poisson_goals = {}
                for g in range(6):
                    poisson_goals[g] = math.exp(-lam) * (lam**g) / math.factorial(g)
                p += f"→ 泊松分布: " + " ".join(f"{g}球{poisson_goals[g]*100:.0f}%" for g in range(6)) + "\n"
            except Exception: pass

        # CRS + TOP7（扩大到7个，不要只看TOP3）
        crs_map = {"w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","w32":"3-2","w40":"4-0","w41":"4-1","w42":"4-2",
                   "s00":"0-0","s11":"1-1","s22":"2-2","s33":"3-3",
                   "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3","l23":"2-3","l04":"0-4","l14":"1-4"}
        crs_lines=[]; crs_probs=[]
        for key,score in crs_map.items():
            try:
                odds=float(m.get(key,0) or 0)
                if odds>1: crs_lines.append(f"{score}={odds:.2f}"); crs_probs.append((score,odds,1/odds))
            except Exception: pass
            
        if crs_lines:
            p += f"CRS全量: {' | '.join(crs_lines)}\n"
            if crs_probs:
                crs_probs.sort(key=lambda x:x[1])
                tp2=sum(pr for _,_,pr in crs_probs)
                p += f"→ CRS概率TOP7: {' > '.join(f'{s}({pr/tp2*100:.1f}%)' for s,_,pr in crs_probs[:7])}\n"
                # 泊松vs CRS偏差检测
                try:
                    for s, odds_val, pr in crs_probs[:7]:
                        sh, sa = map(int, s.split("-"))
                        # 简单泊松估算
                        if eg > 0:
                            home_lam = eg * shin_h / (shin_h + shin_a) if (shin_h + shin_a) > 0 else eg/2
                            away_lam = eg - home_lam
                            poisson_pr = math.exp(-home_lam) * (home_lam**sh) / math.factorial(sh) * math.exp(-away_lam) * (away_lam**sa) / math.factorial(sa)
                            crs_pr = pr / tp2
                            if poisson_pr > 0 and crs_pr > 0:
                                ratio = poisson_pr / crs_pr
                                if ratio > 1.5:
                                    p += f"  💡 {s}: 泊松概率是CRS的{ratio:.1f}倍→可能被低估\n"
                                elif ratio < 0.6:
                                    p += f"  ⚠️ {s}: CRS概率是泊松的{1/ratio:.1f}倍→庄家可能在诱导\n"
                except Exception: pass

        # 半全场
        hf_l=[]
        for k,lb in {"ss":"主/主","sp":"主/平","sf":"主/负","ps":"平/主","pp":"平/平","pf":"平/负","fs":"负/主","fp":"负/平","ff":"负/负"}.items():
            try:
                v=float(m.get(k,0) or 0)
                if v>1: hf_l.append(f"{lb}={v:.2f}")
            except Exception: pass
        if hf_l: p += f"半全场: {' | '.join(hf_l)}\n"

        # 散户
        vote=m.get("vote",{})
        if vote:
            vh = int(vote.get("win",33) or 33); va = int(vote.get("lose",33) or 33)
            vd = int(vote.get("same",33) or 33)
            p += f"散户: 胜{vh}% 平{vd}% 负{va}%"
            if vote.get("hhad_win"): p += f" | 让球主{vote['hhad_win']}%平{vote.get('hhad_same','?')}%客{vote.get('hhad_lose','?')}%"
            if max(vh, va) >= 60:
                hot_side = "主胜" if vh > va else "客胜"
                p += f" ⚠️散户{max(vh,va)}%押{hot_side}→反向价值"
            p += "\n"

        # 赔率变动
        change=m.get("change",{})
        if change and isinstance(change,dict):
            cw=change.get("win",0);cs=change.get("same",0);cl=change.get("lose",0)
            if cw or cs or cl:
                p += f"赔率变动: 胜{cw} 平{cs} 负{cl}"
                # 变动方向分析
                directions = []
                if cw and float(str(cw).replace("+","")) < 0: directions.append("主胜↓(钱涌入)")
                if cs and float(str(cs).replace("+","")) < 0: directions.append("平局↓(钱涌入)")
                if cl and float(str(cl).replace("+","")) < 0: directions.append("客胜↓(钱涌入)")
                if directions: p += f" → {','.join(directions)}"
                p += "\n"

        # 伤停情报
        info=m.get("information",{})
        if isinstance(info,dict):
            for k,v in [("home_injury","主伤停"),("guest_injury","客伤停"),("home_good_news","主利好"),("guest_good_news","客利好"),("home_bad_news","主利空"),("guest_bad_news","客利空")]:
                if info.get(k): p += f"{v}: {str(info[k])[:250].replace(chr(10),' | ')}\n"

        # 状态
        hs=m.get("home_stats",{}); ast2=m.get("away_stats",{})
        if hs.get("form"):
            p += f"主队: {hs.get('wins','?')}胜{hs.get('draws','?')}平{hs.get('losses','?')}负 近况{hs.get('form','?')} 场均进{hs.get('avg_goals_for','?')}/失{hs.get('avg_goals_against','?')}\n"
            p += f"客队: {ast2.get('wins','?')}胜{ast2.get('draws','?')}平{ast2.get('losses','?')}负 近况{ast2.get('form','?')} 场均进{ast2.get('avg_goals_for','?')}/失{ast2.get('avg_goals_against','?')}\n"

        for field in ['analyse','baseface','intro','expert_intro']:
            txt=str(m.get(field,'')).replace('\n',' ')[:200]
            if len(txt)>10: p += f"分析: {txt}\n"; break
        p += "\n"

    p += f"【输出{len(match_analyses)}场JSON数组，每场含top3。你的TOP1不应全是1-0/0-1/1-1——如果数据指向2-1或3-1就大胆给出。只输出数组！】\n"
    return p

def build_phase2_prompt(match_analyses, phase1_results):
    """Phase2: Claude裁判——加权评分制，无否决制"""
    p = "【你是最终裁判】三个独立AI已各自给出TOP3候选比分。你综合分析选出最终比分。\n\n"

    p += "【裁决方法——加权评分制（没有任何比分被一票否决）】\n"
    p += "对每场比赛，你需要对所有候选比分进行加权评分：\n\n"

    p += "评分维度（总分100）：\n"
    p += "① AI共识 [30分]: 3家都选=30分, 2家选=20分, 1家选=10分, 无人选=5分\n"
    p += "② CRS概率 [20分]: CRS概率排名TOP1=20分, TOP2=17分, TOP3=14分, TOP4-6=10分, TOP7+=6分\n"
    p += "③ 泊松吻合 [20分]: 比分总进球与预期λ的匹配度\n"
    p += "④ 联赛风格 [15分]: 该比分是否符合联赛DNA（如德甲多大球→2-1/3-1加分）\n"
    p += "⑤ 信号叠加 [15分]: 散户反指/赔率变动/亚欧矛盾等信号是否指向该比分\n\n"

    p += "【关键原则】\n"
    p += "- CRS赔率高不代表不可能。CRS 3-1@10倍 vs 1-0@5倍，如果3家AI中2家选3-1，3-1得分可能更高\n"
    p += "- 不要因为CRS赔率高就自动排除。体彩经常开出10-20倍的比分\n"
    p += "- 如果你发现自己想选1-1但理由不充分，检查是否在偷懒——1-1是最容易选的比分但不是最常见的\n"
    p += "- 在所有联赛中，2-1的出现频率几乎等于1-0，甚至在德甲/英超高于1-0\n\n"

    p += "【输出格式】JSON数组：match(整数), score(最终比分), reason(80-120字含评分逻辑), ai_confidence(0-100)\n"
    p += "只输出JSON数组！\n\n"

    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        sp_h = float(m.get("sp_home", m.get("win", 0)) or 0)
        sp_d = float(m.get("sp_draw", m.get("same", 0)) or 0)
        sp_a = float(m.get("sp_away", m.get("lose", 0)) or 0)
        league = m.get("league", m.get("cup", ""))
        idx = i + 1

        p += f"{'='*50}\n[{idx}] {h} vs {a} | {league}\n"
        p += f"欧赔: {sp_h:.2f}/{sp_d:.2f}/{sp_a:.2f} | 让球: {m.get('give_ball','0')}\n"

        # 预期进球
        try:
            gp = []
            for gi, field in enumerate(["a0","a1","a2","a3","a4","a5","a6","a7"]):
                v = float(m.get(field, 0) or 0)
                if v > 1: gp.append((gi, 1/v))
            if gp:
                tp = sum(p2 for _,p2 in gp)
                eg = sum(g*(p2/tp) for g,p2 in gp)
                p += f"期望进球λ={eg:.2f}\n"
        except Exception: pass

        # CRS TOP7
        crs_map = {"w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","w32":"3-2",
                   "s00":"0-0","s11":"1-1","s22":"2-2",
                   "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3","l23":"2-3"}
        crs_probs=[]
        for key,score in crs_map.items():
            try:
                odds=float(m.get(key,0) or 0)
                if odds>1: crs_probs.append((score,odds))
            except Exception: pass
        if crs_probs:
            crs_probs.sort(key=lambda x:x[1])
            p += f"CRS参考(非约束): {' > '.join(f'{s}@{o:.1f}' for s,o in crs_probs[:7])}\n"

        # 三家AI结果
        for ai_name in ["gpt", "grok", "gemini"]:
            ai_data = phase1_results.get(ai_name, {}).get(idx, {})
            if not ai_data:
                p += f"  {ai_name.upper()}: 无数据\n"; continue
            top3 = ai_data.get("top3", [])
            if top3:
                scores_str = " | ".join(f"{t.get('score','?')}({t.get('prob','?')}%)" for t in top3[:3])
                p += f"  {ai_name.upper()}: {scores_str} | 信心{ai_data.get('ai_confidence','?')} | {str(ai_data.get('reason',''))[:120]}\n"
            else:
                sc = ai_data.get("ai_score", "-")
                p += f"  {ai_name.upper()}: {sc} | 信心{ai_data.get('ai_confidence','?')} | {str(ai_data.get('analysis',''))[:120]}\n"
        p += "\n"

    p += f"【输出{len(match_analyses)}场JSON数组。大胆选出评分最高的比分，不要默认选1-1！只输出数组！】\n"
    return p

# ====================================================================
# 🌐 异步并发网络层 vMAX 9.0 Pro
# 采用会话复用，消灭所有的裸 except:，极大提高 IO 效率和稳健性
# ====================================================================
FALLBACK_URLS = [None, "https://api520.pro/v1", "https://api521.pro/v1", "https://api522.pro/v1", "https://www.api522.pro/v1"]

def get_clean_env_url(name, default=""):
    v = str(os.environ.get(name, globals().get(name, default))).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/-]+)", v)
    return match.group(1) if match else v

def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")

async def async_call_one_ai_batch(session, prompt, url_env, key_env, models_list, num_matches, ai_name):
    """会话复用 + 精确异常拦截 + 指数退避"""
    key = get_clean_env_key(key_env)
    if not key: return ai_name, {}, "no_key"
    
    primary_url = get_clean_env_url(url_env)
    backup = [u for u in FALLBACK_URLS if u and u != primary_url][:2]
    urls = [primary_url] + backup

    CONNECT_TIMEOUT = 15
    READ_TIMEOUT = 300

    AI_PROFILES = {
        "claude": {
            "sys": "你是最终裁判。三个独立AI已给出候选比分，你用加权评分法综合分析选出每场最终比分。CRS赔率仅做参考权重，不是否决条件。你的比分选择应该多样化，不要全部输出1-0/0-1/1-1。只输出JSON数组。",
            "temp": 0.15
        },
        "grok": {
            "sys": "你是Grok，具备实时联网搜索能力。这是你的核心优势。\n1. 搜索oddsportal.com拿Pinnacle赔率，与提供的竞彩赔率比较偏差（>5%=诱盘）\n2. 搜索Betfair Exchange交易量和赔率\n3. 搜索球队名+injury/lineup确认最新首发伤停\n4. 搜索比赛城市天气+裁判历史数据\n5. 搜索X平台球队最新动态\nreason必须引用搜索到的具体事实。只输出JSON数组。",
            "temp": 0.22
        },
        "gpt": {
            "sys": "你是20年实战职业博彩分析师。用纯数学方法计算每场TOP3候选比分：1. CRS赔率矩阵 2. 泊松拟合 3. 泊松vsCRS偏差。该2-1就2-1，该3-1就3-1，不要保守。只输出JSON数组。",
            "temp": 0.18
        },
        "gemini": {
            "sys": "你是概率建模引擎。严格执行数学计算：1. CRS概率矩阵 2. 泊松分布 3. 欧赔去水位Shin概率。你的输出应反映数学计算结果，不要人为压向1-0/1-1。只输出JSON数组。",
            "temp": 0.15
        },
    }

    profile = AI_PROFILES.get(ai_name, AI_PROFILES["gpt"])

    for mn in models_list:
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
            logger.info(f"[🔌请求] {ai_name.upper()} | {mn[:22]} @ {gw}")
            t0 = time.time()

            try:
                timeout = aiohttp.ClientTimeout(connect=CONNECT_TIMEOUT, sock_read=READ_TIMEOUT)
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                    elapsed = round(time.time()-t0, 1)

                    if r.status in (502, 504):
                        logger.warning(f"  💀 {ai_name} 网关超时 | {elapsed}s → 换URL")
                        continue

                    if r.status == 400:
                        logger.error(f"  💀 {ai_name} 模型不支持400 | 换模型")
                        break 

                    if r.status == 429:
                        logger.warning(f"  🔥 {ai_name} 触发限流429 | 退避重试")
                        await asyncio.sleep(3)
                        continue

                    if r.status != 200:
                        logger.warning(f"  ⚠️ {ai_name} HTTP {r.status} | 换URL")
                        continue

                    data = await r.json(content_type=None)
                    raw_text = ""
                    
                    if is_gem:
                        raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    else:
                        if data.get("choices"): raw_text = data["choices"][0].get("message", {}).get("content", "")
                        elif data.get("output"): raw_text = str(data["output"])

                    arr = robust_json_extract(raw_text)
                    if not arr:
                        logger.warning(f"  ⚠️ {ai_name} 返回格式非JSON | {elapsed}s → 换模型")
                        break

                    results = {}
                    for item in arr:
                        if not isinstance(item, dict) or not item.get("match"): continue
                        mid = int(item["match"]) if str(item["match"]).isdigit() else item["match"]
                        
                        # 极端防御：清洗比分字符串中的空格
                        top1_sc = str(item.get("score") or (item.get("top3", [{}])[0].get("score") if item.get("top3") else "1-1")).replace(" ", "")
                        
                        results[mid] = {
                            "ai_score": top1_sc,
                            "top3": item.get("top3", []),
                            "analysis": str(item.get("reason", ""))[:150],
                            "ai_confidence": int(item.get("ai_confidence", 60)),
                            "value_kill": bool(item.get("value_kill", False)),
                        }

                    if len(results) > 0:
                        logger.info(f"  ✅ {ai_name.upper()} 完成解析: {len(results)}/{num_matches} | {elapsed}s")
                        return ai_name, results, mn
                    else:
                        break

            except aiohttp.ClientConnectorError as e:
                logger.debug(f"  🔌 {ai_name} 连接失败: {str(e)[:30]}")
                continue
            except asyncio.TimeoutError:
                logger.debug(f"  ⏰ {ai_name} 请求超时")
                continue
            except Exception as e:
                logger.debug(f"  ⚠️ {ai_name} 异常: {str(e)[:40]}")
                continue

            await asyncio.sleep(0.2)

    logger.error(f"❌ {ai_name.upper()} 全部模型请求失败")
    return ai_name, {}, "all_connect_failed"

async def run_ai_matrix_two_phase(match_analyses):
    """架构升级：全程复用单个全局 Session，极致优化资源池"""
    num = len(match_analyses)
    p1_prompt = build_phase1_prompt(match_analyses)
    logger.info(f"[Phase1] 构建Prompt: {len(p1_prompt):,} 字符 → 启动并行请求...")

    p1_configs = [
        ("grok","GROK_API_URL","GROK_API_KEY",["熊猫-A-6-grok-4.2-thinking","熊猫-A-7-grok-4.2-多智能体讨论"]),
        ("gpt","GPT_API_URL","GPT_API_KEY",["熊猫-按量-gpt-5.4"]),
        ("gemini","GEMINI_API_URL","GEMINI_API_KEY",["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking","熊猫-顶级特供-X-17-gemini-3.1-pro-preview"])
    ]
    p1_results = {"gpt":{},"grok":{},"gemini":{}}

    connector = aiohttp.TCPConnector(limit=20, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Phase 1
        tasks = [async_call_one_ai_batch(session, p1_prompt, u, k, m, num, n) for n, u, k, m in p1_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, tuple): p1_results[res[0]] = res[1]

        # Phase 2
        p2_prompt = build_phase2_prompt(match_analyses, p1_results)
        logger.info(f"[Phase2] 构建裁判Prompt: {len(p2_prompt):,} 字符...")
        _, claude_r, _ = await async_call_one_ai_batch(
            session, p2_prompt, "CLAUDE_API_URL", "CLAUDE_API_KEY",
            ["熊猫特供-超纯满血-99额度-claude-opus-4.6-thinking","熊猫-按量-特供顶级-官方正向满血-claude-opus-4.6-thinking"],
            num, "claude"
        )

    all_r = p1_results.copy()
    all_r["claude"] = claude_r
    return all_r

# ====================================================================
# 🧬 Merge vMAX 9.0 Pro — 贝叶斯后验概率收敛 + 连续高斯平滑
# ====================================================================
def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    engine_score = engine_result.get("primary_score", "1-1").replace(" ", "")
    engine_conf = engine_result.get("confidence", 50)
    
    # 1. 建立先验概率分布 (Prior)
    ai_inputs = {
        "gpt": (gpt_r, 1.1),
        "grok": (grok_r, 1.3),  # 联网加权
        "gemini": (gemini_r, 1.0),
        "claude": (claude_r, 1.5) # 终点裁判
    }
    
    candidates = {}
    for name, (r_data, base_weight) in ai_inputs.items():
        if not r_data or not isinstance(r_data, dict): continue
        sc = str(r_data.get("ai_score", "-")).replace(" ", "")
        conf = r_data.get("ai_confidence", 60)
        
        if sc and sc not in ["-", "?", ""]:
            if sc not in candidates: candidates[sc] = 0.0
            # 基础置信度融合
            candidates[sc] += base_weight * (conf / 100.0) * 10
            
        # 融入备用意见
        for rank, t in enumerate(r_data.get("top3", [])[1:3]):
            sub_sc = str(t.get("score", "")).replace(" ", "")
            if sub_sc and sub_sc not in ["-", "?", ""]:
                if sub_sc not in candidates: candidates[sub_sc] = 0.0
                candidates[sub_sc] += base_weight * (2.0 - rank) 
                
    if engine_score not in candidates: candidates[engine_score] = 0.0
    candidates[engine_score] += 3.0

    # 2. 注入市场证据进行概率更新 (Posterior Update)
    exp_goals = engine_result.get("expected_goals", 2.3)
    try:
        gp = []
        for gi, field in enumerate(["a0","a1","a2","a3","a4","a5","a6","a7"]):
            v = float(match_obj.get(field, 0) or 0)
            if v > 1: gp.append((gi, 1/v))
        if gp:
            tp = sum(p for _,p in gp)
            exp_goals = sum(g*(p/tp) for g,p in gp)
    except Exception: pass

    crs_map = {"1-0":"w10","2-0":"w20","2-1":"w21","3-0":"w30","3-1":"w31",
               "0-0":"s00","1-1":"s11","2-2":"s22",
               "0-1":"l01","0-2":"l02","1-2":"l12","0-3":"l03","1-3":"l13"}
               
    for sc in list(candidates.keys()):
        try:
            sh, sa = map(int, sc.split("-"))
            
            # 高斯平滑代替硬判断 (差值越大乘数越小)
            goal_diff = abs((sh + sa) - exp_goals)
            poisson_multiplier = math.exp(- (goal_diff ** 2) / 2.5) 
            candidates[sc] *= (0.6 + 0.4 * poisson_multiplier)
            
            # CRS庄家赔率价值挖掘
            odds_key = crs_map.get(sc, "")
            crs_odds = float(match_obj.get(odds_key, 99) or 99)
            if crs_odds < 30.0 and candidates[sc] > 8.0:
                candidates[sc] *= 1.15
        except Exception: pass

    # 3. 后验收敛
    final_score = engine_score
    if candidates:
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        final_score = ranked[0][0]
        logger.info(f"    📊 后验收敛排序: {' > '.join(f'{s}({sc:.1f})' for s, sc in ranked[:4])}")

    # ================= 属性组装 =================
    cf_sum = 0; cf_count = 0; v_kills = 0
    for name, (r, w) in ai_inputs.items():
        if not isinstance(r, dict): continue
        cf_sum += r.get("ai_confidence", 60) * w
        cf_count += w
        if r.get("value_kill"): v_kills += 1
        
    avg_ai_conf = (cf_sum / cf_count) if cf_count > 0 else 60
    cf = engine_conf
    cf = min(95, cf + int((avg_ai_conf - 60) * 0.4)) + v_kills * 6
    if any("🚨" in str(s) for s in stats.get("smart_signals", [])): cf = max(35, cf - 12)
    risk = "低" if cf >= 75 else ("中" if cf >= 55 else "高")

    hp = engine_result.get("home_prob", 33); dp = engine_result.get("draw_prob", 33); ap = engine_result.get("away_prob", 34)
    shp = stats.get("home_win_pct", 33); sdp = stats.get("draw_pct", 33); sap = stats.get("away_win_pct", 34)
    fhp = hp * 0.70 + shp * 0.30; fdp = dp * 0.70 + sdp * 0.30; fap = ap * 0.70 + sap * 0.30
    fhp = max(3, fhp); fdp = max(3, fdp); fap = max(3, fap)
    ft = fhp + fdp + fap
    fhp = round(fhp/ft*100, 1); fdp = round(fdp/ft*100, 1); fap = round(100-fhp-fdp, 1)

    pre_pred = {"home_win_pct": fhp, "draw_pct": fdp, "away_win_pct": fap, "steam_move": stats.get("steam_move", {}), "smart_signals": stats.get("smart_signals", []), "line_movement_anomaly": stats.get("line_movement_anomaly", {})}
    cold_door = ColdDoorDetector.detect(match_obj, pre_pred)
    sigs = list(stats.get("smart_signals", []))
    if cold_door["is_cold_door"]: 
        sigs.extend(cold_door["signals"])
        cf = max(30, cf - 8)

    # 提取多方意见
    gpt_sc = gpt_r.get("ai_score","-") if isinstance(gpt_r, dict) else "-"
    grok_sc = grok_r.get("ai_score","-") if isinstance(grok_r, dict) else "-"
    gem_sc = gemini_r.get("ai_score","-") if isinstance(gemini_r, dict) else "-"

    return {
        "predicted_score": final_score, "home_win_pct": fhp, "draw_pct": fdp, "away_win_pct": fap,
        "confidence": cf, "risk_level": risk,
        "over_under_2_5": "大" if engine_result.get("over_25", 50) > 55 else "小",
        "both_score": "是" if engine_result.get("btts", 45) > 50 else "否",
        "gpt_score": gpt_sc, "gpt_analysis": gpt_r.get("analysis", "N/A") if isinstance(gpt_r, dict) else "N/A", 
        "grok_score": grok_sc, "grok_analysis": grok_r.get("analysis", "N/A") if isinstance(grok_r, dict) else "N/A",
        "gemini_score": gem_sc, "gemini_analysis": gemini_r.get("analysis", "N/A") if isinstance(gemini_r, dict) else "N/A", 
        "claude_score": final_score, "claude_analysis": claude_r.get("analysis", "N/A") if isinstance(claude_r, dict) else "N/A",
        "ai_avg_confidence": round(avg_ai_conf, 1), "value_kill_count": value_kills,
        "smart_money_signal": " | ".join(sigs), "smart_signals": sigs,
        "expected_total_goals": exp_goals,
        "cold_door": cold_door,
        **{k: v for k, v in stats.items() if k not in ["home_win_pct", "draw_pct", "away_win_pct", "predicted_score"]}
    }

# ====================================================================
# 主控制流
# ====================================================================
def select_top4(preds):
    for p in preds:
        pr = p.get("prediction", {})
        s = pr.get("confidence", 0) * 0.4
        mx = max(pr.get("home_win_pct", 33), pr.get("away_win_pct", 33), pr.get("draw_pct", 33))
        s += (mx - 33) * 0.2 + pr.get("model_consensus", 0) * 2
        if pr.get("risk_level") == "低": s += 12
        elif pr.get("risk_level") == "高": s -= 5
        
        smart_money = str(pr.get("smart_money_signal", ""))
        direction = pr.get("result", "")
        if "Sharp" in smart_money:
            if ("客胜" in smart_money and direction == "主胜") or ("主胜" in smart_money and direction == "客胜"): s -= 30
            
        cold = pr.get("cold_door", {})
        if cold.get("is_cold_door"): s -= 12
        p["recommend_score"] = round(s, 2)
        
    preds.sort(key=lambda x: x.get("recommend_score", 0), reverse=True)
    return preds[:4]

def extract_num(ms):
    wm = {"一":1000,"二":2000,"三":3000,"四":4000,"五":5000,"六":6000,"日":7000,"天":7000}
    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999

def run_predictions(raw, use_ai=True):
    ms = raw.get("matches", [])
    print("\n" + "=" * 80)
    print(f"  [QUANT ENGINE v9.0 Pro] 全局复用·高斯收敛网络 | {len(ms)} 场")
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
        start_t = time.time()
        all_ai = asyncio.run(run_ai_matrix_two_phase(match_analyses))
        logger.info(f"[AI MATRIX] 异步推理集群执行完毕，总耗时: {time.time()-start_t:.1f}s")
        
    res = []
    for i, ma in enumerate(match_analyses):
        m = ma["match"]
        mg = merge_result(ma["engine"], all_ai["gpt"].get(i+1,{}), all_ai["grok"].get(i+1,{}), all_ai["gemini"].get(i+1,{}), all_ai["claude"].get(i+1,{}), ma["stats"], m)
        
        # 安全调用拓展插件
        try: mg = apply_experience_to_prediction(m, mg, exp_engine)
        except Exception: pass
        try: mg = apply_odds_history(m, mg)
        except Exception: pass
        try: mg = apply_quant_edge(m, mg)
        except Exception: pass
        try: mg = apply_wencai_intel(m, mg)
        except Exception: pass
        try: mg = upgrade_ensemble_predict(m, mg)
        except Exception: pass
        
        score_str = mg.get("predicted_score", "1-1")
        try:
            sh, sa = map(int, score_str.split("-"))
            if sh > sa: mg["result"] = "主胜"
            elif sh < sa: mg["result"] = "客胜"
            else: mg["result"] = "平局"
        except Exception:
            pcts = {"主胜": mg["home_win_pct"], "平局": mg["draw_pct"], "客胜": mg["away_win_pct"]}
            mg["result"] = max(pcts, key=pcts.get)
            
        res.append({**m, "prediction": mg})
        cold = mg.get("cold_door", {})
        cold_tag = f" [❄️{cold.get('level','')}冷门]" if cold.get("is_cold_door") else ""
        print(f"  [{i+1}] {m.get('home_team')} vs {m.get('away_team')} => {mg['result']} ({mg['predicted_score']}) | CF: {mg['confidence']}% | AI加权: {mg.get('ai_avg_confidence', 0)}{cold_tag}")
        
    t4 = select_top4(res)
    t4ids = [t.get("id") for t in t4]
    for r in res: r["is_recommended"] = r.get("id") in t4ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    
    diary = load_ai_diary()
    cold_count = len([r for r in res if r.get("prediction",{}).get("cold_door",{}).get("is_cold_door")])
    diary["yesterday_win_rate"] = f"{len([r for r in res if r['prediction']['confidence']>70])}/{max(1,len(res))}"
    diary["reflection"] = f"vMAX9.0 Pro | {cold_count}冷门 | 贝叶斯加权·TCP全局复用"
    save_ai_diary(diary)
    
    return res, t4
