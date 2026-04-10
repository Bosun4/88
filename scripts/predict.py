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
            except: pass

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
                from math import exp, factorial
                lam = eg
                poisson_goals = {}
                for g in range(6):
                    poisson_goals[g] = exp(-lam) * (lam**g) / factorial(g)
                p += f"→ 泊松分布: " + " ".join(f"{g}球{poisson_goals[g]*100:.0f}%" for g in range(6)) + "\n"
            except: pass

        # CRS + TOP7（扩大到7个，不要只看TOP3）
        crs_map = {"w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","w32":"3-2","w40":"4-0","w41":"4-1","w42":"4-2",
                   "s00":"0-0","s11":"1-1","s22":"2-2","s33":"3-3",
                   "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3","l23":"2-3","l04":"0-4","l14":"1-4"}
        crs_lines=[]; crs_probs=[]
        for key,score in crs_map.items():
            try:
                odds=float(m.get(key,0) or 0)
                if odds>1: crs_lines.append(f"{score}={odds:.2f}"); crs_probs.append((score,odds,1/odds))
            except: pass
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
                            from math import exp, factorial
                            poisson_pr = exp(-home_lam) * (home_lam**sh) / factorial(sh) * exp(-away_lam) * (away_lam**sa) / factorial(sa)
                            crs_pr = pr / tp2
                            if poisson_pr > 0 and crs_pr > 0:
                                ratio = poisson_pr / crs_pr
                                if ratio > 1.5:
                                    p += f"  💡 {s}: 泊松概率是CRS的{ratio:.1f}倍→可能被低估\n"
                                elif ratio < 0.6:
                                    p += f"  ⚠️ {s}: CRS概率是泊松的{1/ratio:.1f}倍→庄家可能在诱导\n"
                except: pass

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
        except: pass

        # CRS TOP7
        crs_map = {"w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","w32":"3-2",
                   "s00":"0-0","s11":"1-1","s22":"2-2",
                   "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3","l23":"2-3"}
        crs_probs=[]
        for key,score in crs_map.items():
            try:
                odds=float(m.get(key,0) or 0)
                if odds>1: crs_probs.append((score,odds))
            except: pass
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
    backup = [u for u in FALLBACK_URLS if u and u != primary_url][:2]
    urls = [primary_url] + backup

    CONNECT_TIMEOUT = 30      # 连接超时30秒：连不上就换
    READ_TIMEOUT = 600        # 连上后最长等10分钟（实际上等于无限，模型一定会在这之内返回）

    AI_PROFILES = {
        "claude": {
            "sys": "你是最终裁判。三个独立AI已给出候选比分，你用加权评分法综合分析选出每场最终比分。CRS赔率仅做参考权重，不是否决条件。你的比分选择应该多样化，不要全部输出1-0/0-1/1-1。只输出JSON数组。",
            "temp": 0.15
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
                   "你的TOP3应该多样化——如果数据指向2-1就给2-1，不要因为保守全给1-0。\n"
                   "输出每场TOP3候选比分。只输出JSON数组。",
            "temp": 0.22
        },
        "gpt": {
            "sys": "你是20年实战职业博彩分析师。用纯数学方法计算每场TOP3候选比分：\n"
                   "1. CRS赔率→概率矩阵→参考基准（不是最终答案）\n"
                   "2. 总进球a0-a7→泊松拟合→独立概率分布\n"
                   "3. 当泊松概率vs CRS概率偏差>30%→该比分被市场错误定价\n"
                   "4. 亚盘+欧赔交叉验证方向\n"
                   "5. 半全场推断节奏\n"
                   "关键：2-1出现率几乎等于1-0(10% vs 11.5%)，不要忽视。3-1出现率4.2%不算罕见。\n"
                   "该2-1就2-1，该3-1就3-1，不要保守。只输出JSON数组。",
            "temp": 0.18
        },
        "gemini": {
            "sys": "你是概率建模引擎。严格执行数学计算：\n"
                   "1. CRS全比分→概率矩阵（这是庄家的定价，不是真实概率）\n"
                   "2. 总进球→泊松分布→独立计算每个比分概率\n"
                   "3. 对比泊松概率 vs CRS概率→找出被低估的比分\n"
                   "4. 欧赔去水位→Shin概率\n"
                   "5. 你的输出应反映数学计算结果，不要人为压向1-0/1-1\n"
                   "输出每场TOP3候选比分及概率。只输出JSON数组。",
            "temp": 0.15
        },
    }

    best_results = {}; best_model = ""
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
                    connect=CONNECT_TIMEOUT,  # 连接30秒
                    sock_connect=CONNECT_TIMEOUT,
                    sock_read=READ_TIMEOUT,   # 连上后等10分钟
                )
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                    elapsed_connect = round(time.time()-t0, 1)

                    # ===== 连接失败类：换URL或换模型 =====
                    if r.status in (502, 504):
                        print(f"    💀 HTTP {r.status} 网关超时 | {elapsed_connect}s → 换下一个")
                        continue

                    if r.status == 400:
                        print(f"    💀 400 模型不支持 | {elapsed_connect}s → 换模型")
                        break  # 这个模型不行，换下一个模型

                    if r.status == 429:
                        print(f"    🔥 429 限流 | {elapsed_connect}s → 换URL")
                        await asyncio.sleep(2)
                        continue

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
                        print(f"    ⚠️ 响应非JSON | {elapsed}s → 继续试下一个")
                        connected = False
                        continue

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

                    # 提取文本 — 兼容thinking模型的多种格式
                    raw_text = ""
                    try:
                        if is_gem:
                            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        else:
                            msg = data.get("choices", [{}])[0].get("message", {})
                            # thinking模型：content=最终答案, reasoning_content=思考过程
                            # 非thinking模型：content=全部内容
                            # 代理可能格式不同，全部尝试
                            content_text = (msg.get("content") or "").strip()
                            reasoning_text = (msg.get("reasoning_content") or "").strip()
                            text_field = (msg.get("text") or "").strip()

                            # 策略：谁里面有JSON数组就用谁
                            for candidate in [content_text, reasoning_text, text_field]:
                                if candidate and "[" in candidate and "]" in candidate:
                                    raw_text = candidate
                                    break

                            # 都没有JSON数组？拼接所有非空字段一起找
                            if not raw_text:
                                combined = " ".join(filter(None, [content_text, reasoning_text, text_field]))
                                if combined:
                                    raw_text = combined

                            # 最终兜底：整个response转字符串
                            if not raw_text or len(raw_text) < 10:
                                raw_text = json.dumps(data, ensure_ascii=False)
                    except: pass

                    if not raw_text or len(raw_text) < 10:
                        print(f"    ⚠️ 模型返回空数据 | {elapsed}s")
                        print(f"    🔍 原始响应: {json.dumps(data, ensure_ascii=False)[:500]}")
                        connected = False  # 标记为未成功，允许继续试
                        continue

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

                    if len(results) > 0:
                        print(f"    ✅ {ai_name.upper()} 完成: {len(results)}/{num_matches} | {elapsed}s ({mn[:20]})")
                        return ai_name, results, mn
                    else:
                        # 花了钱但解析出0条 → 打印原文帮助调试，继续试下一个模型
                        print(f"    ⚠️ 花了钱但解析0条 | {elapsed}s | 继续试下一个模型")
                        print(f"    🔍 raw_text前300字: {raw_text[:300]}")
                        print(f"    🔍 clean后前300字: {clean[:300]}")
                        connected = False  # 允许继续
                        continue

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
                    return ai_name, best_results, best_model or "read_timeout"

            except Exception as e:
                elapsed = round(time.time()-t0, 1)
                err = str(e)[:40]
                if not connected:
                    print(f"    ⚠️ {err} | {elapsed}s → 换下一个")
                    continue
                else:
                    print(f"    ⚠️ 已连上但异常: {err} | {elapsed}s | 钱已花")
                    return ai_name, best_results, best_model or "error"

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
        ("grok","GROK_API_URL","GROK_API_KEY",["熊猫-A-6-grok-4.2-thinking","熊猫-A-7-grok-4.2-多智能体讨论"]),
        ("gpt","GPT_API_URL","GPT_API_KEY",["熊猫-按量-gpt-5.4","熊猫-A-10-gpt-5.4"]),
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

    p2_prompt = build_phase2_prompt(match_analyses, p1_results)
    print(f"  [Phase2] {len(p2_prompt):,} 字符 → Claude 裁判...")

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
# Merge v5.0 — 加权评分制·无否决制·AI自主决策
#
# 核心变革：
#   旧版: CRS>10倍→否决, 3/3共识→强制, 预期进球→否决
#   新版: 所有因素都是加减分, 最终比分=得分最高的候选
#
# 评分维度:
#   1. Claude裁判权重 (35分) — Claude是最终裁判，权重最高
#   2. Phase1 AI共识 (25分) — 多家AI选同一比分加分
#   3. CRS概率排名 (15分) — CRS低=加分，但不否决
#   4. 预期进球吻合 (15分) — 比分总球vs λ的匹配度
#   5. 联赛风格匹配 (10分) — 联赛DNA加分
# ====================================================================
def merge_result(engine_result, gpt_r, grok_r, gemini_r, claude_r, stats, match_obj):
    sp_h = float(match_obj.get("sp_home", match_obj.get("win", 0)) or 0)
    sp_d = float(match_obj.get("sp_draw", match_obj.get("same", 0)) or 0)
    sp_a = float(match_obj.get("sp_away", match_obj.get("lose", 0)) or 0)
    engine_score = engine_result.get("primary_score", "1-1")
    engine_conf = engine_result.get("confidence", 50)
    league = str(match_obj.get("league", match_obj.get("cup", "")))

    # ========== 收集所有候选比分 ==========
    p1_ai = {"gpt": gpt_r, "grok": grok_r, "gemini": gemini_r}
    all_candidates = {}  # score -> {sources: [], total_score: 0}

    # 从Phase1三家AI收集
    for name, r in p1_ai.items():
        if not isinstance(r, dict): continue
        # TOP1
        sc = r.get("ai_score", "-")
        if not sc or sc in ["-", "?", ""]:
            t3 = r.get("top3", [])
            if t3 and isinstance(t3, list) and len(t3) > 0:
                sc = t3[0].get("score", "-")
        if sc and sc not in ["-", "?", ""]:
            if sc not in all_candidates:
                all_candidates[sc] = {"sources": [], "score": 0.0}
            all_candidates[sc]["sources"].append(name)
        # TOP2/TOP3也纳入候选（较低权重）
        t3 = r.get("top3", [])
        if isinstance(t3, list):
            for rank, t in enumerate(t3[1:3], 2):
                s2 = t.get("score", "")
                if s2 and s2 not in ["-", "?"]:
                    if s2 not in all_candidates:
                        all_candidates[s2] = {"sources": [], "score": 0.0}
                    # 标记为次选
                    all_candidates[s2]["sources"].append(f"{name}_top{rank}")

    # Claude裁判比分
    claude_score = ""
    if isinstance(claude_r, dict):
        claude_score = claude_r.get("ai_score", "")
        if not claude_score or claude_score in ["-", "?"]:
            claude_score = ""
    if claude_score:
        if claude_score not in all_candidates:
            all_candidates[claude_score] = {"sources": [], "score": 0.0}
        all_candidates[claude_score]["sources"].append("claude")

    # 引擎比分也纳入
    if engine_score and engine_score not in ["-", "?"]:
        if engine_score not in all_candidates:
            all_candidates[engine_score] = {"sources": [], "score": 0.0}
        all_candidates[engine_score]["sources"].append("engine")

    # ========== CRS工具 ==========
    crs_key_map = {"1-0":"w10","0-1":"l01","2-1":"w21","1-2":"l12","2-0":"w20","0-2":"l02",
                   "0-0":"s00","1-1":"s11","3-0":"w30","3-1":"w31","0-3":"l03","1-3":"l13",
                   "2-2":"s22","3-2":"w32","2-3":"l23","4-0":"w40","4-1":"w41","0-4":"l04","1-4":"l14"}
    def get_crs(score):
        key = crs_key_map.get(score, "")
        try: return float(match_obj.get(key, 99) or 99)
        except: return 99.0

    # CRS概率排名
    crs_all = []
    for score, key in crs_key_map.items():
        try:
            odds = float(match_obj.get(key, 0) or 0)
            if odds > 1: crs_all.append((score, odds))
        except: pass
    crs_all.sort(key=lambda x: x[1])
    crs_rank = {score: rank+1 for rank, (score, _) in enumerate(crs_all)}

    # 预期总进球
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

    # ========== 加权评分 ==========
    for score, info in all_candidates.items():
        s = 0.0
        sources = info["sources"]

        # ① Claude裁判权重 [35分]
        if "claude" in sources:
            claude_conf = claude_r.get("ai_confidence", 60) if isinstance(claude_r, dict) else 60
            s += 25 + (claude_conf - 50) * 0.2  # 25-35分范围

        # ② Phase1 AI共识 [25分]
        p1_top1_count = sum(1 for src in sources if src in ["gpt", "grok", "gemini"])
        p1_top2_count = sum(1 for src in sources if src.endswith("_top2"))
        p1_top3_count = sum(1 for src in sources if src.endswith("_top3"))
        s += p1_top1_count * 10  # 每个TOP1 = 10分
        s += p1_top2_count * 4   # 每个TOP2 = 4分
        s += p1_top3_count * 2   # 每个TOP3 = 2分

        # ③ CRS概率排名 [15分] — 排名越高分越高，但不否决
        rank = crs_rank.get(score, 20)
        if rank == 1: s += 15
        elif rank == 2: s += 13
        elif rank == 3: s += 11
        elif rank <= 5: s += 8
        elif rank <= 7: s += 5
        elif rank <= 10: s += 3
        else: s += 1  # 即使CRS排名很低也不是0分

        # ④ 预期进球吻合 [15分]
        try:
            sh, sa = map(int, score.split("-"))
            total = sh + sa
            goal_diff = abs(total - exp_goals)
            if goal_diff < 0.5: s += 15
            elif goal_diff < 1.0: s += 12
            elif goal_diff < 1.5: s += 8
            elif goal_diff < 2.0: s += 4
            elif goal_diff < 3.0: s += 1
            else: s += 0  # 偏差太大但不扣分
        except: pass

        # ⑤ 联赛风格匹配 [10分]
        try:
            sh, sa = map(int, score.split("-"))
            total = sh + sa

            # 德甲：大球联赛，2球+比分加分
            if any(lg in league for lg in ["德甲", "德乙", "荷甲"]):
                if total >= 3: s += 10
                elif total == 2: s += 7
                elif total == 1: s += 3
                elif total == 0: s += 1

            # 意甲/法乙：小球联赛，1-2球比分加分
            elif any(lg in league for lg in ["意甲", "意乙", "法乙"]):
                if total <= 1: s += 10
                elif total == 2: s += 8
                elif total == 3: s += 4
                else: s += 1

            # 英超/英冠：均衡偏大
            elif any(lg in league for lg in ["英超", "英冠", "英甲"]):
                if total == 2: s += 8
                elif total == 3: s += 10
                elif total == 1: s += 5
                elif total >= 4: s += 4
                else: s += 3

            # 其他联赛：中性
            else:
                if 1 <= total <= 3: s += 7
                elif total == 0: s += 5
                else: s += 3
        except: pass

        # 引擎也给一点分
        if "engine" in sources:
            s += 3

        info["score"] = round(s, 2)

    # ========== 选出得分最高的比分 ==========
    if not all_candidates:
        final_score = engine_score
    else:
        ranked = sorted(all_candidates.items(), key=lambda x: x[1]["score"], reverse=True)
        final_score = ranked[0][0]
        top_score_val = ranked[0][1]["score"]

        # 打印评分排名（调试用）
        print(f"    📊 比分评分: {' > '.join(f'{sc}({info['score']:.0f}分)' for sc, info in ranked[:5])}")

    # ========== 0-0特殊通道（保留但软化）==========
    exp_analysis = stats.get("experience_analysis", {})
    zero_zero_boost = exp_analysis.get("zero_zero_boost", 0) if isinstance(exp_analysis, dict) else 0
    a0_val = float(match_obj.get("a0", 99) or 99)
    s00_val = float(match_obj.get("s00", 99) or 99)

    # 只在极强信号时才覆盖
    if zero_zero_boost >= 14 and a0_val < 7.5 and s00_val < 8.0:
        if "0-0" in all_candidates:
            # 只有当0-0本身评分也不差时才覆盖
            zero_score = all_candidates.get("0-0", {}).get("score", 0)
            if zero_score >= top_score_val * 0.6:
                print(f"    🔒 0-0通道: boost={zero_zero_boost} a0={a0_val} → 采用0-0")
                final_score = "0-0"

    # ========== 信心/概率/输出 ==========
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
    cl_sc = final_score
    cl_an = claude_r.get("reason", claude_r.get("analysis","N/A")) if isinstance(claude_r, dict) else "N/A"

    # 计算所有AI的TOP1
    all_scores = []
    for name, r in p1_ai.items():
        if isinstance(r, dict):
            sc = r.get("ai_score", "-")
            if sc and sc not in ["-", "?"]: all_scores.append((sc, name))
    if claude_score: all_scores.append((claude_score, "claude"))

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