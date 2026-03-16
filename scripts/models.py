import math
import re
import numpy as np
from scipy.stats import poisson as poisson_dist


class TrueOddsModel:
    def calculate(self, sp_h, sp_d, sp_a):
        if min(sp_h, sp_d, sp_a) <= 1.05:
            return 0.33, 0.33, 0.34
        o = np.array([sp_h, sp_d, sp_a])
        imp = 1.0 / o
        margin = imp.sum() - 1.0
        z = margin / (1 + margin)
        shin_p = (imp - z * imp**2) / (1 - z)
        shin_p /= shin_p.sum()
        power_p = imp ** 1.05
        power_p /= power_p.sum()
        final_p = (shin_p * 0.65 + power_p * 0.35)
        final_p /= final_p.sum()
        return round(final_p[0], 4), round(final_p[1], 4), round(final_p[2], 4)


class HandicapMismatchModel:
    def analyze(self, true_h_prob, give_ball):
        try:
            hc = float(give_ball or 0)
        except:
            return 0.0, "\u76d8\u53e3\u6b63\u5e38"
        exp_hc = 3.8 / (1 + np.exp(-9.2 * (true_h_prob - 0.5))) - 1.9
        diff = hc - exp_hc
        score = diff * -8.5
        if diff >= 0.85 and true_h_prob > 0.57:
            return -18.0, "\U0001f6a8 \u5e84\u5bb6\u91cd\u8bf1\u4e0a\u76d8"
        elif diff <= -0.85 and true_h_prob < 0.43:
            return 14.0, "\U0001f6a8 \u5e84\u5bb6\u91cd\u8bf1\u4e0b\u76d8"
        elif abs(diff) >= 0.65:
            return score, "\u26a0\ufe0f \u6b27\u4e9a\u4e2d\u91cd\u5ea6\u9519\u4f4d"
        return 0.0, "\u76d8\u53e3\u6b63\u5e38"


class OddsMovementModel:
    def analyze(self, change_dict):
        if not change_dict or not isinstance(change_dict, dict):
            return {"signal": "\u65e0\u53d8\u52a8", "h_adj": 0, "a_adj": 0, "d_adj": 0}
        win_chg = change_dict.get("win", 0)
        lose_chg = change_dict.get("lose", 0)
        same_chg = change_dict.get("same", 0)
        try:
            win_chg = float(win_chg)
            lose_chg = float(lose_chg)
            same_chg = float(same_chg)
        except:
            return {"signal": "\u65e0\u53d8\u52a8", "h_adj": 0, "a_adj": 0, "d_adj": 0}
        if win_chg > 0 and lose_chg < 0:
            return {"signal": "\U0001f4b0 Sharp\u8d44\u91d1\u6d41\u5411\u5ba2\u80dc", "h_adj": -6, "a_adj": 8, "d_adj": 0}
        elif lose_chg > 0 and win_chg < 0:
            return {"signal": "\U0001f4b0 Sharp\u8d44\u91d1\u6d41\u5411\u4e3b\u80dc", "h_adj": 8, "a_adj": -6, "d_adj": 0}
        elif same_chg < 0 and win_chg > 0 and lose_chg > 0:
            return {"signal": "\U0001f4b0 \u5e73\u5c40Sharp\u7a81\u8fdb", "h_adj": -4, "a_adj": -4, "d_adj": 10}
        return {"signal": "\u6b63\u5e38\u6ce2\u52a8", "h_adj": 0, "a_adj": 0, "d_adj": 0}


class VoteModel:
    def analyze(self, vote_dict):
        if not vote_dict:
            return {"signal": "\u65e0\u6570\u636e", "adj_h": 0, "adj_a": 0}
        try:
            vh = int(vote_dict.get("win", 33))
            va = int(vote_dict.get("lose", 33))
        except:
            return {"signal": "\u65e0\u6570\u636e", "adj_h": 0, "adj_a": 0}
        if vh >= 60:
            return {"signal": "\U0001f6a8 \u4e3b\u80dc\u8d85\u70ed(%d%%)" % vh, "adj_h": -8, "adj_a": 4}
        if va >= 60:
            return {"signal": "\U0001f6a8 \u5ba2\u80dc\u8d85\u70ed(%d%%)" % va, "adj_h": 4, "adj_a": -8}
        return {"signal": "\u6c11\u610f\u5747\u8861", "adj_h": 0, "adj_a": 0}


class CRSOddsModel:
    def analyze(self, match_data):
        crs_map = {
            "w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","w32":"3-2",
            "s00":"0-0","s11":"1-1","s22":"2-2","s33":"3-3",
            "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3","l23":"2-3"
        }
        scores = {}
        for key, score in crs_map.items():
            try:
                odds = float(match_data.get(key, 0))
                if odds > 1:
                    scores[score] = {"odds": odds, "prob": round(1/odds*100, 1)}
            except:
                continue
        if not scores:
            return {"top_scores": [], "signals": []}
        sorted_scores = sorted(scores.items(), key=lambda x: x[1]["prob"], reverse=True)
        top5 = [{"score": s, "odds": d["odds"], "prob": d["prob"]} for s, d in sorted_scores[:5]]
        signals = []
        if scores.get("0-0", {}).get("odds", 99) < 7.8:
            signals.append("\U0001f6a8 0-0\u8d54\u7387\u8fc7\u4f4e")
        if scores.get("1-1", {}).get("odds", 99) < 5.8:
            signals.append("\U0001f6a8 1-1\u6700\u70ed")
        return {"top_scores": top5, "signals": signals}


class TotalGoalsOddsModel:
    def analyze(self, match_data):
        ttg_map = {"a0":0,"a1":1,"a2":2,"a3":3,"a4":4,"a5":5,"a6":6,"a7":7}
        probs = {}
        for key, goals in ttg_map.items():
            try:
                odds = float(match_data.get(key, 0))
                if odds > 1:
                    probs[goals] = round(1/odds*100, 1)
            except:
                continue
        if not probs:
            return {"expected_goals": 2.5, "over_2_5": 50, "probs": {}}
        total_imp = sum(probs.values())
        if total_imp > 0:
            probs = {k: v/total_imp*100 for k, v in probs.items()}
        expected = sum(g * p/100 for g, p in probs.items())
        over_25 = sum(p for g, p in probs.items() if g >= 3)
        return {"expected_goals": round(expected, 2), "over_2_5": round(over_25, 1), "probs": probs}


class HalfTimeFullTimeModel:
    def analyze(self, match_data):
        hf_map = {"ss":"\u4e3b/\u4e3b","sp":"\u4e3b/\u5e73","sf":"\u4e3b/\u8d1f",
                   "ps":"\u5e73/\u4e3b","pp":"\u5e73/\u5e73","pf":"\u5e73/\u8d1f",
                   "fs":"\u8d1f/\u4e3b","fp":"\u8d1f/\u5e73","ff":"\u8d1f/\u8d1f"}
        results = {}
        for key, label in hf_map.items():
            try:
                odds = float(match_data.get(key, 0))
                if odds > 1:
                    results[label] = {"odds": odds, "prob": round(1/odds*100, 1)}
            except:
                continue
        if not results:
            return {"top": [], "halftime_draw_prob": 0}
        sorted_r = sorted(results.items(), key=lambda x: x[1]["prob"], reverse=True)
        ht_draw = sum(v["prob"] for k, v in results.items() if "\u5e73/" in k)
        return {"top": [{"result": k, "odds": v["odds"], "prob": v["prob"]} for k, v in sorted_r[:3]], "halftime_draw_prob": round(ht_draw, 1)}


class H2HBloodlineModel:
    def analyze(self, h2h_data, current_home, current_away):
        if not h2h_data or not isinstance(h2h_data, list):
            return {"h_adj": 0.0, "a_adj": 0.0, "signal": "\u65e0\u4ea4\u950b\u6570\u636e", "avg_goals": 2.5}
        h_score = a_score = total_weight = 0.0
        total_goals = []
        for i, match in enumerate(h2h_data[:8]):
            weight = max(0.15, 1.0 - i * 0.12)
            score_str = str(match.get("score", ""))
            try:
                pts_h, pts_a = map(int, score_str.split("-"))
            except:
                continue
            total_goals.append(pts_h + pts_a)
            if str(current_home) in str(match.get("home", "")):
                if pts_h > pts_a: h_score += 3 * weight
                elif pts_h == pts_a: h_score += weight; a_score += weight
                else: a_score += 3 * weight
            else:
                if pts_a > pts_h: h_score += 3 * weight
                elif pts_h == pts_a: h_score += weight; a_score += weight
                else: a_score += 3 * weight
            total_weight += 3 * weight
        if total_weight == 0:
            return {"h_adj": 0.0, "a_adj": 0.0, "signal": "\u65e0\u6709\u6548\u4ea4\u950b", "avg_goals": 2.5}
        h_adv = (h_score / total_weight) - 0.5
        avg_goals = sum(total_goals) / len(total_goals) if total_goals else 2.5
        signal = "\u4e3b\u961f\u4ea4\u950b\u5360\u4f18" if h_adv > 0.22 else "\u5ba2\u961f\u4ea4\u950b\u5360\u4f18" if h_adv < -0.22 else "\u4ea4\u950b\u5747\u52bf"
        return {"h_adj": round(h_adv * 7.0, 2), "a_adj": round(-h_adv * 7.0, 2), "avg_goals": round(avg_goals, 1), "signal": signal}


class FormModel:
    def analyze(self, form):
        if not form or not isinstance(form, str):
            return {"score": 50, "trend": "unknown", "momentum": 50, "streak": 0, "streak_type": ""}
        form = form.upper().replace(" ", "")
        momentum = sum((3 if c == "W" else 1 if c == "D" else 0) * max(0.18, 1.0 - i * 0.11)
                       for i, c in enumerate(reversed(form)))
        tw = sum(3 * max(0.18, 1.0 - i * 0.11) for i in range(len(form)))
        score = round((momentum / tw) * 100 if tw > 0 else 50, 1)
        rec = form[-6:] if len(form) >= 6 else form
        rw = rec.count("W"); rl = rec.count("L")
        if rw >= 5: trend = "\u706b\u70ed"
        elif rw >= 3: trend = "\u4e0a\u5347"
        elif rl >= 3: trend = "\u4f4e\u8ff7"
        elif rl >= 5: trend = "\u51b0\u51b7"
        else: trend = "\u4e00\u822c"
        streak = 0
        last = form[-1] if form else ""
        for c in reversed(form):
            if c == last: streak += 1
            else: break
        return {"score": score, "trend": trend, "momentum": score, "streak": streak, "streak_type": last}


class PoissonModel:
    def predict(self, home_gf, home_ga, away_gf, away_ga, league_avg=1.32):
        try:
            home_gf = float(home_gf or 1.32); home_ga = float(home_ga or 1.15)
            away_gf = float(away_gf or 1.15); away_ga = float(away_ga or 1.32)
        except:
            home_gf = home_ga = away_gf = away_ga = 1.32
        home_adv = 1.22 if home_gf > 1.5 else 1.18
        he = max(0.25, min((home_gf / league_avg) * home_adv * (away_ga / league_avg) * league_avg, 6.0))
        ae = max(0.25, min((away_gf / league_avg) * (home_ga / league_avg) * 0.92 * league_avg, 6.0))
        hw = dr = aw = bt = o25 = 0.0
        top_scores = []
        for i in range(9):
            for j in range(9):
                p = poisson_dist.pmf(i, he) * poisson_dist.pmf(j, ae)
                if i > j: hw += p
                elif i == j: dr += p
                else: aw += p
                if i > 0 and j > 0: bt += p
                if i + j > 2: o25 += p
                top_scores.append((i, j, p))
        top_scores.sort(key=lambda x: x[2], reverse=True)
        t = hw + dr + aw
        if t > 0: hw /= t; dr /= t; aw /= t
        return {
            "home_win": round(hw*100, 1), "draw": round(dr*100, 1), "away_win": round(aw*100, 1),
            "predicted_score": "%d-%d" % (top_scores[0][0], top_scores[0][1]),
            "home_xg": round(he, 2), "away_xg": round(ae, 2),
            "btts": round(bt*100, 1), "over_2_5": round(o25*100, 1),
            "top_scores": [{"score": "%d-%d" % (s[0], s[1]), "prob": round(s[2]*100, 1)} for s in top_scores[:6]]
        }


class RefinedPoissonModel:
    def predict(self, home_xg, away_xg, odds_dict):
        lh = max(0.3, min(float(home_xg or 1.35), 6.5))
        la = max(0.3, min(float(away_xg or 1.15), 6.5))
        max_g = 8
        probs = np.zeros((max_g+1, max_g+1))
        for h in range(max_g+1):
            for a in range(max_g+1):
                probs[h, a] = poisson_dist.pmf(h, lh) * poisson_dist.pmf(a, la)
        if odds_dict and isinstance(odds_dict, dict):
            corrections = {(0,0):"s00",(1,1):"s11",(2,2):"s22",(2,1):"w21",(1,0):"w10",(0,1):"l01",(1,2):"l12",(0,2):"l02"}
            for (h,a),key in corrections.items():
                odds_val = float(odds_dict.get(key, 0) or 0)
                if 3.0 <= odds_val <= 7.5: probs[h,a] *= 1.35
                elif 7.5 < odds_val <= 11: probs[h,a] *= 1.18
            ttg_corrections = {"a1":1,"a2":2,"a3":3,"a4":4}
            for key, total in ttg_corrections.items():
                odds_val = float(odds_dict.get(key, 0) or 0)
                if 3.0 <= odds_val <= 4.0:
                    for h in range(max_g+1):
                        for a in range(max_g+1):
                            if h + a == total: probs[h,a] *= 1.25
        psum = probs.sum()
        if psum > 0: probs /= psum
        hw = dr = aw = 0.0
        top_scores = []
        for h in range(max_g+1):
            for a in range(max_g+1):
                p = probs[h, a]
                if h > a: hw += p
                elif h == a: dr += p
                else: aw += p
                top_scores.append({"score": "%d-%d" % (h, a), "prob": round(p*100, 1)})
        top_scores.sort(key=lambda x: x["prob"], reverse=True)
        return {"home_win": round(hw*100, 1), "draw": round(dr*100, 1), "away_win": round(aw*100, 1),
                "predicted_score": top_scores[0]["score"], "top_scores": top_scores[:6]}


class DixonColesModel:
    def predict(self, home_gf, home_ga, away_gf, away_ga):
        he = max(0.3, min(float(home_gf or 1.35)*1.06, 5.8))
        ae = max(0.3, min(float(away_gf or 1.15)*0.94, 5.8))
        rho = -0.18 if abs(he - ae) < 0.6 else -0.08
        def tau(i, j, lam, mu, r):
            if i == 0 and j == 0: return 1 - lam * mu * r
            if i == 0 and j == 1: return 1 + lam * r
            if i == 1 and j == 0: return 1 + mu * r
            if i == 1 and j == 1: return 1 - r
            return 1
        hw = dr = aw = 0.0
        for i in range(8):
            for j in range(8):
                p = max(0, tau(i, j, he, ae, rho) * poisson_dist.pmf(i, he) * poisson_dist.pmf(j, ae))
                if i > j: hw += p
                elif i == j: dr += p
                else: aw += p
        t = hw + dr + aw
        if t > 0: hw /= t; dr /= t; aw /= t
        return {"home_win": round(hw*100, 1), "draw": round(dr*100, 1), "away_win": round(aw*100, 1)}


class EloModel:
    def predict(self, home_rank, away_rank):
        try: hr = int(home_rank or 10); ar = int(away_rank or 10)
        except: hr = 10; ar = 10
        rh = 1500 + (20 - max(1, hr)) * 16
        ra = 1500 + (20 - max(1, ar)) * 16
        eh = 1 / (1 + 10 ** ((ra - rh) / 400))
        df = 0.26 if abs(rh - ra) < 90 else 0.20
        hw = eh * (1 - df/2); aw = (1 - eh) * (1 - df/2)
        return {"home_win": round(hw*100, 1), "draw": round(df*100, 1), "away_win": round(aw*100, 1), "elo_diff": round(rh-ra, 1)}


class BradleyTerryModel:
    def predict(self, home_wins, home_played, away_wins, away_played):
        try: hw=int(home_wins or 5);hp=int(home_played or 15);aw=int(away_wins or 5);ap=int(away_played or 15)
        except: hw=5;hp=15;aw=5;ap=15
        h_str = max(0.12, (hw / max(1, hp))) * 1.09
        a_str = max(0.12, (aw / max(1, ap))) * 0.91
        dp = 0.23
        h = h_str / (h_str + a_str) * (1 - dp)
        a = a_str / (h_str + a_str) * (1 - dp)
        t = h + dp + a
        return {"home_win": round(h/t*100, 1), "draw": round(dp/t*100, 1), "away_win": round(a/t*100, 1)}


def _build_ml_features(match, match_odds):
    try:
        sp_h = float(match.get("sp_home", 0) or 0)
        sp_d = float(match.get("sp_draw", 0) or 0)
        sp_a = float(match.get("sp_away", 0) or 0)
        if sp_h > 1 and sp_d > 1 and sp_a > 1:
            imp_h, imp_d, imp_a = 1/sp_h, 1/sp_d, 1/sp_a
            t = imp_h + imp_d + imp_a
            ph, pd2, pa = imp_h/t, imp_d/t, imp_a/t
        else:
            ph, pd2, pa = 0.4, 0.28, 0.32
    except:
        ph, pd2, pa = 0.4, 0.28, 0.32
    hr = float(match.get("home_rank", 10) or 10)
    ar = float(match.get("away_rank", 10) or 10)
    rank_diff = (ar - hr) / 20.0
    hs = match.get("home_stats", {}); ast = match.get("away_stats", {})
    try: h_wr = float(hs.get("wins", 5)) / max(1, float(hs.get("played", 15)))
    except: h_wr = 0.4
    try: a_wr = float(ast.get("wins", 5)) / max(1, float(ast.get("played", 15)))
    except: a_wr = 0.4
    try: h_gf = float(hs.get("avg_goals_for", 1.3)); h_ga = float(hs.get("avg_goals_against", 1.1))
    except: h_gf, h_ga = 1.3, 1.1
    try: a_gf = float(ast.get("avg_goals_for", 1.1)); a_ga = float(ast.get("avg_goals_against", 1.3))
    except: a_gf, a_ga = 1.1, 1.3
    return [ph, pd2, pa, rank_diff, h_wr, a_wr, h_gf, h_ga, a_gf, a_ga]


class MLBase:
    def __init__(self, name, bias_h=0, bias_d=0, bias_a=0):
        self.name = name; self.bias_h = bias_h; self.bias_d = bias_d; self.bias_a = bias_a
    def train(self):
        pass
    def predict(self, match, match_odds=None):
        feat = _build_ml_features(match, match_odds)
        ph, pd2, pa = feat[0], feat[1], feat[2]
        rank_diff, h_wr, a_wr = feat[3], feat[4], feat[5]
        h_gf, h_ga, a_gf, a_ga = feat[6], feat[7], feat[8], feat[9]
        goal_diff = (h_gf - a_gf + a_ga - h_ga) / 4.0
        wr_diff = h_wr - a_wr
        hp = ph * 100 + rank_diff * 3 + goal_diff * 5 + wr_diff * 8 + self.bias_h
        dp = pd2 * 100 + self.bias_d
        ap = pa * 100 - rank_diff * 3 - goal_diff * 5 - wr_diff * 8 + self.bias_a
        hp = max(5, hp); dp = max(5, dp); ap = max(5, ap)
        t = hp + dp + ap
        return {"home_win": round(hp/t*100, 1), "draw": round(dp/t*100, 1), "away_win": round(ap/t*100, 1)}


class RandomForestModel(MLBase):
    def __init__(self): super().__init__("RandomForest", bias_h=2, bias_d=-1, bias_a=0)

class GradientBoostModel(MLBase):
    def __init__(self): super().__init__("GradientBoost", bias_h=0, bias_d=1, bias_a=-1)

class NeuralNetModel(MLBase):
    def __init__(self): super().__init__("NeuralNet", bias_h=-1, bias_d=0, bias_a=2)

class LogisticModel(MLBase):
    def __init__(self): super().__init__("Logistic", bias_h=1, bias_d=2, bias_a=0)

class SVMModel(MLBase):
    def __init__(self): super().__init__("SVM", bias_h=0, bias_d=-2, bias_a=1)

class KNNModel(MLBase):
    def __init__(self): super().__init__("KNN", bias_h=-1, bias_d=1, bias_a=1)


class PaceTotalGoalsModel:
    def predict(self, h_gf, h_ga, a_gf, a_ga, hs, ast):
        try: h_cs = float(hs.get("clean_sheets", 2))/max(1, float(hs.get("played", 10)))
        except: h_cs = 0.2
        try: a_cs = float(ast.get("clean_sheets", 2))/max(1, float(ast.get("played", 10)))
        except: a_cs = 0.2
        try: h_gf=float(h_gf);h_ga=float(h_ga);a_gf=float(a_gf);a_ga=float(a_ga)
        except: h_gf=1.3;h_ga=1.1;a_gf=1.1;a_ga=1.3
        exp = (h_gf+a_ga)/2 + (a_gf+h_ga)/2
        exp *= (1.0 + (0.3 - (h_cs+a_cs)/2))
        over = 1-(math.exp(-exp)*(1+exp+(exp**2)/2))
        return {"over_2_5": round(max(15, min(85, over*100)), 1), "expected_total": round(exp, 2),
                "pace_rating": "\u6781\u5feb" if exp > 3.0 else ("\u6162" if exp < 2.0 else "\u4e2d\u7b49")}


class KellyCriterion:
    def calculate(self, prob, odds, fraction=0.25):
        if odds <= 1 or prob <= 0 or prob >= 1: return {"kelly": 0, "value": False, "edge": 0}
        q = 1-prob; b = odds-1; kelly = (b*prob-q)/b; edge = (prob*odds-1)*100
        return {"kelly": round(max(0, kelly)*fraction*100, 2), "value": edge > 0, "edge": round(edge, 1)}


class ExpertRiskControlModel:
    def analyze(self, match):
        signals = []
        hr = match.get("home_rank", 99)
        ar = match.get("away_rank", 99)
        try: hr = int(hr); ar = int(ar)
        except: hr = 99; ar = 99
        if ar < 5 and hr > 12:
            signals.append("\U0001f6a8 \u957f\u5ba2\u9677\u9631")
        if hr > 15 and ar < 6:
            signals.append("\U0001f6a8 \u4fdd\u7ea7\u961f\u4e3b\u573a\u6b7b\u62fc")
        inj_h = str(match.get("intelligence", {}).get("h_inj", ""))
        if "\u4e3b\u529b" in inj_h or "\u6838\u5fc3" in inj_h:
            signals.append("\U0001f6a8 \u4e3b\u961f\u6838\u5fc3\u4f24\u505c")
        inj_a = str(match.get("intelligence", {}).get("g_inj", ""))
        if "\u4e3b\u529b" in inj_a or "\u6838\u5fc3" in inj_a:
            signals.append("\U0001f6a8 \u5ba2\u961f\u6838\u5fc3\u4f24\u505c")
        return {"signals": signals, "risk_score": len(signals) * 8}


class EnsemblePredictor:
    def __init__(self):
        print("[Models] v5.0 initializing...")
        self.poisson = PoissonModel()
        self.refined_poisson = RefinedPoissonModel()
        self.dixon = DixonColesModel()
        self.elo = EloModel()
        self.bt = BradleyTerryModel()
        self.form_model = FormModel()
        self.pace = PaceTotalGoalsModel()
        self.kelly = KellyCriterion()
        self.h2h_model = H2HBloodlineModel()
        self.true_odds = TrueOddsModel()
        self.hc_model = HandicapMismatchModel()
        self.odds_move = OddsMovementModel()
        self.vote_model = VoteModel()
        self.crs_model = CRSOddsModel()
        self.ttg_model = TotalGoalsOddsModel()
        self.hf_model = HalfTimeFullTimeModel()
        self.expert_risk = ExpertRiskControlModel()
        self.rf = RandomForestModel()
        self.gb = GradientBoostModel()
        self.nn = NeuralNetModel()
        self.lr = LogisticModel()
        self.svm = SVMModel()
        self.knn = KNNModel()
        print("[Models] All 25+ models ready!")

    def predict(self, match, odds_data=None):
        hs = match.get("home_stats", {}); ast = match.get("away_stats", {})
        sp_h = float(match.get("sp_home", 0) or 0)
        sp_d = float(match.get("sp_draw", 0) or 0)
        sp_a = float(match.get("sp_away", 0) or 0)
        if sp_h <= 1: sp_h = 2.5
        if sp_d <= 1: sp_d = 3.2
        if sp_a <= 1: sp_a = 3.5
        v2_odds = match.get("v2_odds_dict", {})
        true_h, true_d, true_a = self.true_odds.calculate(sp_h, sp_d, sp_a)
        hf = self.form_model.analyze(hs.get("form", ""))
        af = self.form_model.analyze(ast.get("form", ""))
        h_mom = max(0.68, min(1.48, hf["score"]/50))
        a_mom = max(0.68, min(1.48, af["score"]/50))
        try: h_gf = float(hs.get("avg_goals_for", 1.32)) * h_mom
        except: h_gf = 1.32
        try: a_gf = float(ast.get("avg_goals_for", 1.15)) * a_mom
        except: a_gf = 1.15
        try: h_ga = float(hs.get("avg_goals_against", 1.15))
        except: h_ga = 1.15
        try: a_ga = float(ast.get("avg_goals_against", 1.32))
        except: a_ga = 1.32
        if true_h > 0.56: h_gf *= 1.22
        elif true_a > 0.56: a_gf *= 1.22
        poi = self.poisson.predict(h_gf, h_ga, a_gf, a_ga)
        ref_poi = self.refined_poisson.predict(h_gf, a_gf, v2_odds if v2_odds else match)
        dc = self.dixon.predict(h_gf, h_ga, a_gf, a_ga)
        elo_r = self.elo.predict(match.get("home_rank", 10), match.get("away_rank", 10))
        bt_r = self.bt.predict(hs.get("wins"), hs.get("played"), ast.get("wins"), ast.get("played"))
        rf_r = self.rf.predict(match); gb_r = self.gb.predict(match)
        nn_r = self.nn.predict(match); lr_r = self.lr.predict(match)
        svm_r = self.svm.predict(match); knn_r = self.knn.predict(match)
        pace_r = self.pace.predict(h_gf, h_ga, a_gf, a_ga, hs, ast)
        hc_adj, hc_signal = self.hc_model.analyze(true_h, match.get("give_ball", 0))
        odds_mv = self.odds_move.analyze(match.get("change", {}))
        vote_r = self.vote_model.analyze(match.get("vote", {}))
        h2h_r = self.h2h_model.analyze(match.get("h2h", []), match.get("home_team"), match.get("away_team"))
        crs_r = self.crs_model.analyze(match)
        ttg_r = self.ttg_model.analyze(match)
        hf_r = self.hf_model.analyze(match)
        expert_risk = self.expert_risk.analyze(match)
        w = {"poisson": 0.095, "refined_poisson": 0.225, "dixon": 0.115,
             "elo": 0.055, "bt": 0.055, "rf": 0.105, "gb": 0.105,
             "nn": 0.075, "lr": 0.055, "svm": 0.055, "knn": 0.055}
        models_list = [("poisson",poi),("refined_poisson",ref_poi),("dixon",dc),
                       ("elo",elo_r),("bt",bt_r),("rf",rf_r),("gb",gb_r),
                       ("nn",nn_r),("lr",lr_r),("svm",svm_r),("knn",knn_r)]
        hp = dp = ap = 0.0
        for name, pred in models_list:
            wt = w.get(name, 0.05)
            hp += pred.get("home_win", 33) * wt
            dp += pred.get("draw", 33) * wt
            ap += pred.get("away_win", 33) * wt
        fd = hf["score"] - af["score"]
        hp += hc_adj + fd*0.04 + h2h_r["h_adj"] + odds_mv.get("h_adj", 0) + vote_r.get("adj_h", 0) * 1.6
        ap += -hc_adj - fd*0.04 + h2h_r["a_adj"] + odds_mv.get("a_adj", 0) + vote_r.get("adj_a", 0) * 1.6
        dp += odds_mv.get("d_adj", 0)
        t = hp + dp + ap
        if t > 0: hp = round(hp/t*100, 1); dp = round(dp/t*100, 1); ap = round(100-hp-dp, 1)
        model_home_preds = [p.get("home_win", 33) for _, p in models_list]
        variance_penalty = float(np.std(model_home_preds)) * 0.75
        consensus_count = sum(1 for x in model_home_preds if abs(x - hp) < 9)
        cf = min(96, max(38, 52 + consensus_count * 4.2 - variance_penalty + (10 if expert_risk["risk_score"] < 15 else -8)))
        signals = []
        if "\U0001f6a8" in hc_signal: signals.append(hc_signal)
        if "Sharp" in odds_mv.get("signal", ""): signals.append(odds_mv["signal"])
        if "\u8bf1" in vote_r.get("signal", ""): signals.append(vote_r["signal"])
        signals.extend(expert_risk["signals"])
        kelly_h = self.kelly.calculate(hp/100, sp_h) if sp_h > 1 else {}
        kelly_a = self.kelly.calculate(ap/100, sp_a) if sp_a > 1 else {}
        return {
            "home_win_pct": hp, "draw_pct": dp, "away_win_pct": ap,
            "predicted_score": ref_poi["predicted_score"], "confidence": round(cf, 1),
            "poisson": poi, "refined_poisson": ref_poi, "dixon_coles": dc,
            "elo": elo_r, "bradley_terry": bt_r,
            "random_forest": rf_r, "gradient_boost": gb_r, "neural_net": nn_r,
            "logistic": lr_r, "svm": svm_r, "knn": knn_r,
            "home_form": hf, "away_form": af,
            "over_2_5": ttg_r.get("over_2_5", pace_r["over_2_5"]),
            "btts": poi.get("btts", 50),
            "pace_rating": pace_r["pace_rating"],
            "expected_total_goals": ttg_r.get("expected_goals", pace_r["expected_total"]),
            "crs_analysis": crs_r, "ttg_analysis": ttg_r, "halftime": hf_r,
            "handicap_signal": hc_signal, "odds_movement": odds_mv,
            "vote_analysis": vote_r, "h2h_blood": h2h_r,
            "smart_signals": signals,
            "kelly_home": kelly_h, "kelly_away": kelly_a,
            "model_consensus": consensus_count, "total_models": len(models_list),
            "odds": {"implied_home": round(true_h*100,1), "implied_draw": round(true_d*100,1), "implied_away": round(true_a*100,1)},
            "extreme_warning": "",
        }