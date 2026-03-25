#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advanced_models.py v2.0 — 适配vMAX的高级模型升级模块
"""

import numpy as np
import os
import json
from scipy.stats import poisson as pdist
from collections import Counter


class BivariatePoissonModel:
    def predict(self, home_xg, away_xg, correlation=0.12, max_goals=7):
        try: hxg = max(0.15, float(home_xg or 1.3)); axg = max(0.15, float(away_xg or 1.1))
        except: hxg, axg = 1.3, 1.1
        l1 = max(0.05, hxg - correlation); l2 = max(0.05, axg - correlation)
        l3 = max(0.0, min(correlation, min(hxg, axg) * 0.5))
        mg = max_goals + 1
        probs = np.zeros((mg, mg))
        for x in range(mg):
            for y in range(mg):
                for k in range(min(x, y) + 1):
                    probs[x, y] += pdist.pmf(x-k, l1) * pdist.pmf(y-k, l2) * pdist.pmf(k, l3)
        ps = probs.sum()
        if ps > 0: probs /= ps
        hw = dr = aw = bt = o25 = 0.0
        scores = []
        for x in range(mg):
            for y in range(mg):
                p = probs[x, y]
                if x > y: hw += p
                elif x == y: dr += p
                else: aw += p
                if x > 0 and y > 0: bt += p
                if x + y > 2: o25 += p
                scores.append({"score": f"{x}-{y}", "prob": round(p*100, 2)})
        scores.sort(key=lambda x: x["prob"], reverse=True)
        return {"home_win": round(hw*100,1), "draw": round(dr*100,1), "away_win": round(aw*100,1),
                "predicted_score": scores[0]["score"], "btts": round(bt*100,1), "over_2_5": round(o25*100,1),
                "correlation": round(l3,3), "top_scores": scores[:6]}


class ProOverroundRemoval:
    def calculate(self, sp_h, sp_d, sp_a):
        if min(sp_h, sp_d, sp_a) <= 1.05: return 0.33, 0.33, 0.34
        odds = np.array([sp_h, sp_d, sp_a])
        imp = 1.0 / odds
        margin = imp.sum() - 1.0
        z = margin / (1 + margin)
        shin = (imp - z * imp**2) / (1 - z); shin /= shin.sum()
        power = imp ** 1.05; power /= power.sum()
        mult = imp / imp.sum()
        final = shin * 0.50 + power * 0.25 + mult * 0.25; final /= final.sum()
        return round(float(final[0]),4), round(float(final[1]),4), round(float(final[2]),4)


class AsianHandicapConverter:
    @staticmethod
    def from_xg(hxg, axg, mg=8):
        probs = np.zeros((mg, mg))
        for i in range(mg):
            for j in range(mg):
                probs[i,j] = pdist.pmf(i, max(0.2,hxg)) * pdist.pmf(j, max(0.2,axg))
        return probs / probs.sum()

    @classmethod
    def ah(cls, probs, hc):
        mg = probs.shape[0]; hcover = push = acover = 0.0
        for i in range(mg):
            for j in range(mg):
                m = (i-j) - hc
                if m > 0: hcover += probs[i,j]
                elif m == 0: push += probs[i,j]
                else: acover += probs[i,j]
        t = hcover+push+acover
        if t > 0: hcover/=t; push/=t; acover/=t
        return [round(hcover,4), round(push,4), round(acover,4)]

    @classmethod
    def ou(cls, probs, line):
        mg = probs.shape[0]; over = under = push = 0.0
        for i in range(mg):
            for j in range(mg):
                total = i+j
                if total > line: over += probs[i,j]
                elif total == line: push += probs[i,j]
                else: under += probs[i,j]
        t = over+push+under
        if t > 0: over/=t; push/=t; under/=t
        return [round(over,4), round(push,4), round(under,4)]

    @classmethod
    def btts(cls, probs):
        mg = probs.shape[0]
        return round(sum(probs[i,j] for i in range(1,mg) for j in range(1,mg)), 4)

    @classmethod
    def full_analysis(cls, hxg, axg):
        p = cls.from_xg(hxg, axg)
        return {"ah_0": cls.ah(p,0), "ah_0.5": cls.ah(p,0.5), "ah_1.0": cls.ah(p,1.0),
                "ou_1.5": cls.ou(p,1.5), "ou_2.5": cls.ou(p,2.5), "ou_3.5": cls.ou(p,3.5), "btts": cls.btts(p)}


class ProbabilityCalibrator:
    def __init__(self, history_dir="data"):
        self.bias_h = self.bias_d = self.bias_a = 0.0
        self.calibrated = False
        try:
            hf = os.path.join(history_dir, "history.json")
            if os.path.exists(hf):
                with open(hf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                c = data.get("cumulative", {})
                if c.get("total", 0) >= 30:
                    rr = c.get("result_rate", 50)
                    if rr < 45: self.bias_h = -2.0; self.bias_d = 3.0; self.bias_a = -1.0
                    elif rr < 55: self.bias_h = -0.5; self.bias_d = 1.5; self.bias_a = -0.5
                    else: self.bias_h = 0.5; self.bias_d = -0.5; self.bias_a = 0.5
                    self.calibrated = True
        except: pass

    def calibrate(self, hp, dp, ap):
        if not self.calibrated: return hp, dp, ap
        mean = (hp + dp + ap) / 3; s = 0.92
        hc = max(5, hp*s + mean*(1-s) + self.bias_h)
        dc = max(5, dp*s + mean*(1-s) + self.bias_d)
        ac = max(5, ap*s + mean*(1-s) + self.bias_a)
        t = hc+dc+ac
        return round(hc/t*100,1), round(dc/t*100,1), round(100-hc/t*100-dc/t*100,1)


class CLVDetector:
    @staticmethod
    def detect(model_prob, odds, direction="home"):
        if not odds or odds <= 1.0: return {"has_clv": False, "clv_pct": 0, "edge": 0, "signal": ""}
        mp = model_prob / 100.0; fair_p = (1.0/odds) * 0.97
        clv = (mp - fair_p) / max(fair_p, 0.01) * 100; edge = (mp * odds - 1) * 100
        if clv > 5 and edge > 3:
            return {"has_clv": True, "clv_pct": round(clv,1), "edge": round(edge,1),
                    "signal": f"🔥 CLV+{clv:.1f}% Edge+{edge:.1f}% [{direction}]"}
        elif clv > 2:
            return {"has_clv": True, "clv_pct": round(clv,1), "edge": round(edge,1),
                    "signal": f"📈 CLV+{clv:.1f}% [{direction}]"}
        elif clv < -8:
            return {"has_clv": False, "clv_pct": round(clv,1), "edge": round(edge,1),
                    "signal": f"⚠️ 负CLV{clv:.1f}% [{direction}]"}
        return {"has_clv": False, "clv_pct": round(clv,1), "edge": round(edge,1), "signal": ""}


class MarkovScorePredictor:
    def predict(self, hxg, axg, n_sim=6000):
        try: hxg = max(0.2, float(hxg or 1.3)); axg = max(0.2, float(axg or 1.1))
        except: hxg, axg = 1.3, 1.1
        hr = hxg / 90; ar = axg / 90
        rng = np.random.RandomState(42)
        results = []
        for _ in range(n_sim):
            hg = ag = 0
            for minute in range(90):
                hb = 1.0 + max(0, (ag-hg))*0.08
                ab = 1.0 + max(0, (hg-ag))*0.08
                if minute > 75:
                    if hg > ag: hb *= 0.7; ab *= 1.15
                    elif ag > hg: ab *= 0.7; hb *= 1.15
                if rng.random() < hr * hb: hg += 1
                if rng.random() < ar * ab: ag += 1
            results.append(f"{hg}-{ag}")
        counts = Counter(results)
        top = [{"score": s, "prob": round(c/n_sim*100,1)} for s, c in counts.most_common(6)]
        hw = sum(1 for r in results if int(r.split("-")[0]) > int(r.split("-")[1])) / n_sim
        dr = sum(1 for r in results if int(r.split("-")[0]) == int(r.split("-")[1])) / n_sim
        return {"home_win": round(hw*100,1), "draw": round(dr*100,1), "away_win": round((1-hw-dr)*100,1),
                "predicted_score": top[0]["score"] if top else "1-0", "top_scores": top}


# 全局单例
_bvp = BivariatePoissonModel()
_ov = ProOverroundRemoval()
_cal = ProbabilityCalibrator()
_clv = CLVDetector()
_mk = MarkovScorePredictor()


def upgrade_ensemble_predict(match, prediction, odds_data=None):
    sp_h = float(match.get("sp_home",0) or 0)
    sp_d = float(match.get("sp_draw",0) or 0)
    sp_a = float(match.get("sp_away",0) or 0)

    hxg = prediction.get("bookmaker_implied_home_xg")
    axg = prediction.get("bookmaker_implied_away_xg")
    if not hxg or hxg == "?" or not axg or axg == "?":
        hs = match.get("home_stats", {}); ast = match.get("away_stats", {})
        try: hxg = float(hs.get("avg_goals_for", 1.3))
        except: hxg = 1.3
        try: axg = float(ast.get("avg_goals_for", 1.1))
        except: axg = 1.1
    else:
        hxg = float(hxg); axg = float(axg)

    # 1. BVP
    prediction["bivariate_poisson"] = _bvp.predict(hxg, axg)
    # 2. ProOdds
    if sp_h > 1 and sp_d > 1 and sp_a > 1:
        th, td, ta = _ov.calculate(sp_h, sp_d, sp_a)
        prediction["pro_odds"] = {"true_home": round(th*100,1), "true_draw": round(td*100,1), "true_away": round(ta*100,1)}
    # 3. 亚盘
    prediction["asian_handicap_probs"] = AsianHandicapConverter.full_analysis(hxg, axg)
    # 4. 概率校准
    hp = prediction.get("home_win_pct",33); dp = prediction.get("draw_pct",33); ap = prediction.get("away_win_pct",34)
    hp_c, dp_c, ap_c = _cal.calibrate(hp, dp, ap)
    # 5. BVP融合 (12%)
    bvp = prediction["bivariate_poisson"]
    hp_f = hp_c*0.88 + bvp["home_win"]*0.12
    dp_f = dp_c*0.88 + bvp["draw"]*0.12
    ap_f = ap_c*0.88 + bvp["away_win"]*0.12
    if "pro_odds" in prediction:
        po = prediction["pro_odds"]
        hp_f = hp_f*0.95 + po["true_home"]*0.05
        dp_f = dp_f*0.95 + po["true_draw"]*0.05
        ap_f = ap_f*0.95 + po["true_away"]*0.05
    t = hp_f+dp_f+ap_f
    if t > 0:
        prediction["home_win_pct"] = round(hp_f/t*100,1)
        prediction["draw_pct"] = round(dp_f/t*100,1)
        prediction["away_win_pct"] = round(100-prediction["home_win_pct"]-prediction["draw_pct"],1)
    # 6. CLV
    sigs = prediction.get("smart_signals", [])
    for d, o, n in [(prediction["home_win_pct"],sp_h,"主胜"),(prediction["draw_pct"],sp_d,"平局"),(prediction["away_win_pct"],sp_a,"客胜")]:
        c = _clv.detect(d, o, n)
        if c["signal"] and c["signal"] not in sigs: sigs.append(c["signal"])
    prediction["smart_signals"] = sigs
    # 7. 大小球/BTTS
    ahc = prediction["asian_handicap_probs"]
    o25_old = prediction.get("over_2_5",50)
    prediction["over_2_5"] = round(o25_old*0.55 + bvp["over_2_5"]*0.25 + ahc["ou_2.5"][0]*100*0.20, 1)
    btts_old = prediction.get("btts",50)
    prediction["btts"] = round(btts_old*0.55 + bvp["btts"]*0.25 + ahc["btts"]*100*0.20, 1)
    # 8. 更新方向
    pcts = {"主胜": prediction["home_win_pct"], "平局": prediction["draw_pct"], "客胜": prediction["away_win_pct"]}
    prediction["result"] = max(pcts, key=pcts.get)
    prediction["total_models"] = prediction.get("total_models",11) + 3
    return prediction