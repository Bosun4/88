#!/usr/bin/env python3
"""
quant_edge.py v1.1 — 量化优势检测引擎（极速无MC版）
=========================================
已彻底移除最慢的MonteCarloSim，保留全部其他顶级算法
"""

import numpy as np
from collections import defaultdict


class SteamMoveDetector:
    """
    Steam Move检测器 (BettingIsCool方法论)
    当赔率在短时间内急剧下降=有Sharp资金大量涌入
    这是职业投注者公认的#1跟注信号
    """
    @staticmethod
    def detect(match, prediction):
        change = match.get("change", {})
        if not change: return {"steam": False, "signal": "", "direction": "", "strength": 0}
        try:
            wc = float(change.get("win", 0))
            lc = float(change.get("lose", 0))
            sc = float(change.get("same", 0))
        except:
            return {"steam": False, "signal": "", "direction": "", "strength": 0}

        vote = match.get("vote", {})
        vh = int(vote.get("win", 33) if vote else 33)
        va = int(vote.get("lose", 33) if vote else 33)

        if wc < -0.08:
            if vh < 50:
                return {"steam": True, "signal": f"🔥 主胜Steam! 赔率降{wc:.2f}但散户未跟(仅{vh}%)", "direction": "home", "strength": min(10, int(abs(wc) * 80))}
            elif vh >= 60:
                return {"steam": True, "signal": f"🚨 主胜反向Steam! 降水{wc:.2f}+热度{vh}%=庄家造热", "direction": "upset_away", "strength": min(10, int(abs(wc) * 60))}

        if lc < -0.08:
            if va < 50:
                return {"steam": True, "signal": f"🔥 客胜Steam! 赔率降{lc:.2f}但散户未跟(仅{va}%)", "direction": "away", "strength": min(10, int(abs(lc) * 80))}
            elif va >= 60:
                return {"steam": True, "signal": f"🚨 客胜反向Steam! 降水{lc:.2f}+热度{va}%=庄家造热", "direction": "upset_home", "strength": min(10, int(abs(lc) * 60))}

        if sc < -0.06 and wc >= 0 and lc >= 0:
            return {"steam": True, "signal": f"💰 平局Steam! 平赔独降{sc:.2f}+主客赔同升", "direction": "draw", "strength": 8}

        return {"steam": False, "signal": "", "direction": "", "strength": 0}


class ValueBetEngine:
    """
    价值投注引擎
    """
    @staticmethod
    def detect_all(prediction, sp_h, sp_d, sp_a, min_edge=0.05):
        results = []
        hp = prediction.get("home_win_pct", 33) / 100
        dp = prediction.get("draw_pct", 33) / 100
        ap = prediction.get("away_win_pct", 34) / 100

        for name, prob, odds in [("主胜", hp, sp_h), ("平局", dp, sp_d), ("客胜", ap, sp_a)]:
            if odds <= 1.0 or prob <= 0:
                continue
            ev = prob * odds - 1.0
            if ev > min_edge:
                b = odds - 1.0
                q = 1 - prob
                kelly_full = (b * prob - q) / b if b > 0 else 0
                kelly_quarter = max(0, kelly_full * 0.25)

                results.append({
                    "direction": name,
                    "prob": round(prob * 100, 1),
                    "odds": odds,
                    "ev": round(ev * 100, 1),
                    "kelly_pct": round(kelly_quarter * 100, 2),
                    "signal": f"💎 {name}价值注! EV+{ev*100:.1f}% Kelly{kelly_quarter*100:.1f}% @{odds}",
                    "grade": "A" if ev > 0.15 else ("B" if ev > 0.08 else "C"),
                })

        results.sort(key=lambda x: x["ev"], reverse=True)
        return results


class OddRangeFilter:
    RANGES = {
        "home": [
            (1.01, 1.30, 0.82, "超热门:胜率82%但赔率无利润"),
            (1.30, 1.60, 0.68, "强势主队:胜率68%最佳区间"),
            (1.60, 2.00, 0.55, "中等主队:需精选"),
            (2.00, 2.50, 0.42, "均势偏主:平局风险高"),
            (2.50, 4.00, 0.32, "弱势主队:冷门频出"),
            (4.00, 99.0, 0.20, "极端冷门主队"),
        ],
        "draw": [
            (2.50, 3.00, 0.32, "低赔平局:概率偏高"),
            (3.00, 3.50, 0.28, "标准平局区间"),
            (3.50, 4.00, 0.24, "平局偏少"),
            (4.00, 6.00, 0.18, "平局罕见"),
        ],
        "away": [
            (1.01, 1.60, 0.72, "超强客队"),
            (1.60, 2.50, 0.48, "强势客队"),
            (2.50, 4.00, 0.30, "正常客队"),
            (4.00, 8.00, 0.18, "弱势客队:FLB区间"),
            (8.00, 99.0, 0.08, "极端冷门客队"),
        ],
    }

    @classmethod
    def get_historical_rate(cls, odds, direction="home"):
        ranges = cls.RANGES.get(direction, [])
        for lo, hi, rate, desc in ranges:
            if lo <= odds < hi:
                return rate, desc
        return 0.33, "未知区间"

    @classmethod
    def analyze(cls, sp_h, sp_d, sp_a):
        h_rate, h_desc = cls.get_historical_rate(sp_h, "home")
        d_rate, d_desc = cls.get_historical_rate(sp_d, "draw")
        a_rate, a_desc = cls.get_historical_rate(sp_a, "away")
        return {
            "home": {"hist_rate": round(h_rate * 100, 1), "desc": h_desc},
            "draw": {"hist_rate": round(d_rate * 100, 1), "desc": d_desc},
            "away": {"hist_rate": round(a_rate * 100, 1), "desc": a_desc},
        }


class PairwiseStrength:
    @staticmethod
    def estimate(match):
        hs = match.get("home_stats", {})
        ast = match.get("away_stats", {})
        try:
            h_gf = float(hs.get("avg_goals_for", 1.3))
            h_ga = float(hs.get("avg_goals_against", 1.1))
            a_gf = float(ast.get("avg_goals_for", 1.1))
            a_ga = float(ast.get("avg_goals_against", 1.3))
        except:
            return {"strength_diff": 0, "signal": ""}

        h_eff = h_gf / max(0.3, h_ga)
        a_eff = a_gf / max(0.3, a_ga)
        diff = (h_eff * 1.05 - a_eff) / max(0.5, (h_eff + a_eff) / 2)

        if diff > 0.5:
            return {"strength_diff": round(diff, 2), "adj_h": 3, "adj_a": -2, "signal": f"⚡ 主队攻防效率碾压(差值{diff:.2f})"}
        elif diff < -0.5:
            return {"strength_diff": round(diff, 2), "adj_h": -2, "adj_a": 3, "signal": f"⚡ 客队攻防效率碾压(差值{abs(diff):.2f})"}

        return {"strength_diff": round(diff, 2), "adj_h": 0, "adj_a": 0, "signal": ""}


# ============================================================
#  统一对接函数（已彻底删除MonteCarloSim）
# ============================================================

_steam = SteamMoveDetector()
_value = ValueBetEngine()
_range = OddRangeFilter()
_pair = PairwiseStrength()

def apply_quant_edge(match, prediction):
    """极速版：已删除Monte Carlo，只保留Steam、ValueBet、赔率区间、配对实力"""
    sp_h = float(match.get("sp_home", 0) or 0)
    sp_d = float(match.get("sp_draw", 0) or 0)
    sp_a = float(match.get("sp_away", 0) or 0)
    sigs = prediction.get("smart_signals", [])

    # 1. Steam检测
    steam = _steam.detect(match, prediction)
    if steam["steam"]:
        sigs.append(steam["signal"])
        prediction["steam_move"] = steam

    # 2. 价值投注检测
    if sp_h > 1 and sp_d > 1 and sp_a > 1:
        vbs = _value.detect_all(prediction, sp_h, sp_d, sp_a)
        if vbs:
            prediction["value_bets"] = vbs
            for vb in vbs[:2]:
                sigs.append(vb["signal"])

    # 3. 赔率区间分析
    if sp_h > 1:
        ranges = _range.analyze(sp_h, sp_d, sp_a)
        prediction["odds_range_analysis"] = ranges

    # 4. 配对实力评估
    pair = _pair.estimate(match)
    if pair.get("signal"):
        sigs.append(pair["signal"])
        hp = prediction.get("home_win_pct", 33) + pair.get("adj_h", 0) * 0.2
        ap = prediction.get("away_win_pct", 34) + pair.get("adj_a", 0) * 0.2
        dp = 100 - hp - ap
        hp = max(5, hp); dp = max(5, dp); ap = max(5, ap)
        t = hp + dp + ap
        prediction["home_win_pct"] = round(hp / t * 100, 1)
        prediction["draw_pct"] = round(dp / t * 100, 1)
        prediction["away_win_pct"] = round(100 - prediction["home_win_pct"] - prediction["draw_pct"], 1)

    prediction["smart_signals"] = sigs
    pcts = {"主胜": prediction["home_win_pct"], "平局": prediction["draw_pct"], "客胜": prediction["away_win_pct"]}
    prediction["result"] = max(pcts, key=pcts.get)

    return prediction