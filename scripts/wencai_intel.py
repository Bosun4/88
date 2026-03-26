#!/usr/bin/env python3
"""
wencai_intel.py v2.0 — 市面上最牛逼的文采吸血引擎
"""

import re
from collections import defaultdict

class WencaiBloodSucker:
    @staticmethod
    def analyze(match):
        edge = {
            "news_impact": 0,           # 新闻屠杀指数
            "public_fool_score": 0,     # 散户愚蠢度
            "lineup_edge": 0,           # 阵容星级
            "signals": [],
            "dark_verdict": "",
            "kill_level": "普通"        # 顶级/高危/普通
        }

        # 1. 散户投票愚蠢度（核心屠杀指标）
        vote = match.get("vote", {})
        vote_h = int(vote.get("win", 33))
        vote_d = int(vote.get("same", 33))
        vote_a = int(vote.get("lose", 34))
        model_h = match.get("prediction", {}).get("home_win_pct", 33) if isinstance(match.get("prediction"), dict) else 33

        divergence = abs(vote_h - model_h)
        edge["public_fool_score"] = min(15, divergence // 6)
        if edge["public_fool_score"] >= 8:
            edge["signals"].append(f"🚨 散户疯狂投票{ '主胜' if vote_h > 60 else '客胜' if vote_a > 60 else '平局'}（{vote_h}%），愚蠢度爆表！")

        # 2. 新闻屠杀指数 + 关键词情感打分
        info = match.get("information", {})
        good_h = len(str(info.get("home_good_news", "")))
        bad_h = len(str(info.get("home_bad_news", "")))
        good_a = len(str(info.get("guest_good_news", "")))
        bad_a = len(str(info.get("guest_bad_news", "")))
        neutral = str(info.get("neutral_news", ""))

        # 关键词加分
        bad_keywords = len(re.findall(r"伤|停|缺阵|危机|重病|输球率|防守短板", neutral + str(info.get("home_bad_news", "")) + str(info.get("guest_bad_news", ""))))
        news_score = (bad_h - good_h) * 1.2 + (good_a - bad_a) * 1.2 + bad_keywords * 3
        edge["news_impact"] = max(-12, min(12, int(news_score / 25)))

        if edge["news_impact"] <= -7:
            edge["signals"].append("🔪 主队坏消息+伤停爆炸，庄家已布好主胜陷阱")
            edge["kill_level"] = "顶级"
        elif edge["news_impact"] >= 7:
            edge["signals"].append("🩸 客队坏消息被隐藏，资本正在诱多主队")

        # 3. 阵容星级（核心球员识别）
        lineup_h = str(info.get("home_first_team", ""))
        lineup_a = str(info.get("guest_first_team", ""))
        stars_h = len(re.findall(r"居莱尔|恰尔汗奥卢|伊尔迪兹|巴雷拉|多纳鲁马|德米拉尔|居莱尔", lineup_h))
        stars_a = len(re.findall(r"斯坦丘|普莱斯|麦克奈尔|布尔克", lineup_a))
        edge["lineup_edge"] = (stars_h - stars_a) * 2.5

        if edge["lineup_edge"] >= 6:
            edge["signals"].append(f"⚡ 主队阵容核弹级碾压（{stars_h}大核心）")

        # 4. 最毒总结
        h = match.get("home", "主队")
        a = match.get("guest", "客队")
        if edge["news_impact"] <= -6 and vote_h > 65:
            edge["dark_verdict"] = f"庄家用{h}坏消息+散户狂热投票做局，准备血洗主胜筹码"
            edge["kill_level"] = "顶级"
        elif edge["news_impact"] >= 6 and vote_a > 55:
            edge["dark_verdict"] = f"{a}坏消息被完美隐藏，散户还在疯狂送钱，客队即将屠杀"
            edge["kill_level"] = "顶级"
        else:
            edge["dark_verdict"] = f"散户共识 vs 真实底牌严重背离，庄家已布好死亡陷阱"

        return edge


def apply_wencai_intel(match, prediction):
    """终极对接函数"""
    sucker = WencaiBloodSucker()
    edge = sucker.analyze(match)

    prediction["wencai_edge"] = edge
    prediction["news_impact"] = edge["news_impact"]
    prediction["public_fool_score"] = edge["public_fool_score"]
    prediction["kill_level"] = edge["kill_level"]

    # 合并信号 + 去重
    sigs = prediction.get("smart_signals", [])
    sigs.extend(edge["signals"])
    prediction["smart_signals"] = list(dict.fromkeys(sigs))

    # 概率微调（更激进）
    if edge["news_impact"] <= -5:
        prediction["home_win_pct"] = max(5, round(prediction.get("home_win_pct", 33) - 3.5, 1))
    elif edge["news_impact"] >= 5:
        prediction["away_win_pct"] = max(5, round(prediction.get("away_win_pct", 34) - 3.5, 1))

    pcts = {"主胜": prediction["home_win_pct"], "平局": prediction["draw_pct"], "客胜": prediction["away_win_pct"]}
    prediction["result"] = max(pcts, key=pcts.get)

    return prediction