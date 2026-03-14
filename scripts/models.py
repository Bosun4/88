import math
import random
import re
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    HAS_SK = True
except Exception:
    HAS_SK = False
    print("[WARN] sklearn not available")

# ============================================================
# 1. 赔率数学模型
# ============================================================

class TrueOddsModel:
    """欧赔真实概率还原"""
    def calculate(self, sp_h, sp_d, sp_a):
        if sp_h <= 1.0 or sp_d <= 1.0 or sp_a <= 1.0:
            return 0.33, 0.33, 0.34
        imp_h, imp_d, imp_a = 1.0/sp_h, 1.0/sp_d, 1.0/sp_a
        margin = imp_h + imp_d + imp_a
        return imp_h/margin, imp_d/margin, imp_a/margin

class HandicapMismatchModel:
    """欧亚盘口错位检测"""
    def analyze(self, true_h_prob, give_ball):
        try:
            hc = float(give_ball or 0)
        except:
            return 0.0, "盘口正常"
        if true_h_prob >= 0.80: exp_hc = -1.5
        elif true_h_prob >= 0.70: exp_hc = -1.0
        elif true_h_prob >= 0.60: exp_hc = -0.5
        elif true_h_prob >= 0.55: exp_hc = -0.25
        elif true_h_prob <= 0.20: exp_hc = 1.5
        elif true_h_prob <= 0.30: exp_hc = 1.0
        elif true_h_prob <= 0.40: exp_hc = 0.5
        else: exp_hc = 0.0
        diff = hc - exp_hc
        if diff >= 0.75 and true_h_prob > 0.55:
            return -12.0, "🚨 欧亚错位：让球畸浅，强烈诱上"
        elif diff <= -0.75 and true_h_prob < 0.45:
            return 8.0, "🚨 欧亚错位：逆势深盘，强力阻击"
        elif abs(diff) >= 0.5:
            return diff * -5.0, "⚠️ 盘口轻度偏离"
        return 0.0, "盘口正常"

class OddsMovementModel:
    """赔率变动方向分析(利用change字段)"""
    def analyze(self, change_dict):
        if not change_dict or not isinstance(change_dict, dict):
            return {"signal": "无变动", "adj": 0}
        win_chg = change_dict.get("win", 0)
        lose_chg = change_dict.get("lose", 0)
        same_chg = change_dict.get("same", 0)
        # 主胜赔率升+客胜赔率降 = 资金流向客胜
        if win_chg > 0 and lose_chg < 0:
            return {"signal": "💰 资金流向客胜", "h_adj": -3, "a_adj": 3}
        elif win_chg < 0 and lose_chg > 0:
            return {"signal": "💰 资金流向主胜", "h_adj": 3, "a_adj": -3}
        elif same_chg < 0 and win_chg > 0 and lose_chg > 0:
            return {"signal": "💰 资金流向平局", "h_adj": -2, "a_adj": -2, "d_adj": 4}
        return {"signal": "无明显资金流向", "h_adj": 0, "a_adj": 0}

class VoteModel:
    """民意投票反向指标"""
    def analyze(self, vote_dict):
        if not vote_dict:
            return {"signal": "无数据", "adj_h": 0, "adj_a": 0}
        try:
            vh = int(vote_dict.get("win", 33))
            vd = int(vote_dict.get("same", 33))
            va = int(vote_dict.get("lose", 33))
        except:
            return {"signal": "无数据", "adj_h": 0, "adj_a": 0}
        # 逆向思维：大热必死
        if vh >= 55:
            return {"signal": "⚠️ 主胜大热(%d%%)，逆向看冷" % vh, "adj_h": -4, "adj_a": 2}
        elif va >= 55:
            return {"signal": "⚠️ 客胜大热(%d%%)，逆向看冷" % va, "adj_h": 2, "adj_a": -4}
        return {"signal": "投票均衡", "adj_h": 0, "adj_a": 0}

class CRSOddsModel:
    """比分赔率深度分析"""
    def analyze(self, match_data):
        scores = {}
        crs_map = {
            "w10": "1-0", "w20": "2-0", "w21": "2-1", "w30": "3-0", "w31": "3-1", "w32": "3-2",
            "w40": "4-0", "w41": "4-1", "w42": "4-2",
            "s00": "0-0", "s11": "1-1", "s22": "2-2", "s33": "3-3",
            "l01": "0-1", "l02": "0-2", "l12": "1-2", "l03": "0-3", "l13": "1-3", "l23": "2-3",
        }
        for key, score in crs_map.items():
            try:
                odds = float(match_data.get(key, 0))
                if odds > 1:
                    scores[score] = {"odds": odds, "prob": round(1/odds*100, 1)}
            except:
                continue
        if not scores:
            return {"top_scores": [], "signal": "无数据"}
        sorted_scores = sorted(scores.items(), key=lambda x: x[1]["prob"], reverse=True)
        top5 = [{"score": s, "odds": d["odds"], "prob": d["prob"]} for s, d in sorted_scores[:5]]
        # 异常赔率检测
        signals = []
        if scores.get("0-0", {}).get("odds", 99) < 8:
            signals.append("0-0赔率偏低，闷平风险")
        if scores.get("1-1", {}).get("odds", 99) < 5.5:
            signals.append("1-1最热，双方有球闷平")
        return {"top_scores": top5, "signals": signals}

class TotalGoalsOddsModel:
    """总进球数赔率分析"""
    def analyze(self, match_data):
        ttg_map = {"a0": 0, "a1": 1, "a2": 2, "a3": 3, "a4": 4, "a5": 5, "a6": 6, "a7": 7}
        probs = {}
        for key, goals in ttg_map.items():
            try:
                odds = float(match_data.get(key, 0))
                if odds > 1:
                    probs[goals] = round(1/odds*100, 1)
            except:
                continue
        if not probs:
            return {"expected_goals": 2.5, "over_2_5": 50}
        total_imp = sum(probs.values())
        if total_imp > 0:
            probs = {k: v/total_imp*100 for k, v in probs.items()}
        expected = sum(g * p/100 for g, p in probs.items())
        over_25 = sum(p for g, p in probs.items() if g >= 3)
        most_likely = max(probs, key=probs.get)
        return {
            "expected_goals": round(expected, 2),
            "over_2_5": round(over_25, 1),
            "most_likely_total": most_likely,
            "probs": probs
        }

class HalfTimeFullTimeModel:
    """半全场赔率分析"""
    def analyze(self, match_data):
        hf_map = {
            "ss": "主/主", "sp": "主/平", "sf": "主/负",
            "ps": "平/主", "pp": "平/平", "pf": "平/负",
            "fs": "负/主", "fp": "负/平", "ff": "负/负"
        }
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
        ht_draw = sum(v["prob"] for k, v in results.items() if k.startswith("平/"))
        return {
            "top": [{"result": k, "odds": v["odds"], "prob": v["prob"]} for k, v in sorted_r[:3]],
            "halftime_draw_prob": round(ht_draw, 1)
        }

# ============================================================
# 2. 交锋与状态模型
# ============================================================

class H2HBloodlineModel:
    """交锋血统深度分析"""
    def analyze(self, h2h_data, current_home, current_away):
        if not h2h_data or not isinstance(h2h_data, list):
            return {"h_adj": 0.0, "a_adj": 0.0, "signal": "无交锋数据"}
        h_score, a_score, total_weight = 0.0, 0.0, 0.0
        total_goals = []
        for i, match in enumerate(h2h_data):
            weight = max(0.2, 1.0 - i * 0.15)
            score_str = str(match.get("score", ""))
            m_home = str(match.get("home", ""))
            try:
                pts_h, pts_a = map(int, score_str.split("-"))
            except:
                continue
            total_goals.append(pts_h + pts_a)
            if str(current_home) in m_home:
                if pts_h > pts_a: h_score += 3 * weight
                elif pts_h == pts_a: h_score += 1 * weight; a_score += 1 * weight
                else: a_score += 3 * weight
            else:
                if pts_a > pts_h: h_score += 3 * weight
                elif pts_h == pts_a: h_score += 1 * weight; a_score += 1 * weight
                else: a_score += 3 * weight
            total_weight += 3 * weight
        if total_weight == 0:
            return {"h_adj": 0.0, "a_adj": 0.0, "signal": "无有效交锋"}
        h_adv = (h_score / total_weight) - 0.5
        avg_goals = sum(total_goals) / len(total_goals) if total_goals else 2.5
        signal = ""
        if h_adv > 0.2: signal = "主队交锋占优"
        elif h_adv < -0.2: signal = "客队交锋占优"
        else: signal = "交锋均势"
        return {"h_adj": round(h_adv * 6.0, 2), "a_adj": round(-h_adv * 6.0, 2), "avg_goals": round(avg_goals, 1), "signal": signal}

class FormModel:
    """近期状态深度分析"""
    def analyze(self, form):
        if not form:
            return {"score": 50, "trend": "unknown", "momentum": 50}
        momentum = sum((3 if c == "W" else 1 if c == "D" else 0) * max(0.2, 1.0 - i * 0.12)
                       for i, c in enumerate(reversed(form)))
        tw = sum(3 * max(0.2, 1.0 - i * 0.12) for i in range(len(form)))
        score = round((momentum / tw) * 100 if tw > 0 else 50, 1)
        w = form.count("W"); d = form.count("D"); l = form.count("L")
        rec = form[-5:] if len(form) >= 5 else form
        rw = rec.count("W"); rl = rec.count("L")
        if rw >= 4: trend = "🔥火热"
        elif rw >= 3: trend = "上升"
        elif rl >= 3: trend = "低迷"
        elif rl >= 4: trend = "❄️冰冷"
        else: trend = "一般"
        streak = 0
        last = form[-1] if form else ""
        for c in reversed(form):
            if c == last: streak += 1
            else: break
        return {"score": score, "trend": trend, "wins": w, "draws": d, "losses": l,
                "recent": rec, "streak": streak, "streak_type": last, "momentum": score}

# ============================================================
# 3. 泊松族模型
# ============================================================

class PoissonModel:
    """经典泊松分布"""
    def predict(self, home_gf, home_ga, away_gf, away_ga, league_avg=1.35):
        try: home_gf=float(home_gf or 1.3);home_ga=float(home_ga or 1.1);away_gf=float(away_gf or 1.1);away_ga=float(away_ga or 1.3)
        except: home_gf=1.3;home_ga=1.1;away_gf=1.1;away_ga=1.3
        he = max(0.2, min((home_gf/league_avg)*1.10*(away_ga/league_avg)*league_avg, 5.5))
        ae = max(0.2, min((away_gf/league_avg)*(home_ga/league_avg)*0.90*league_avg, 5.5))
        def pmf(k, l): return (l**k)*math.exp(-l)/math.factorial(k)
        hw, dr, aw, bt, o25, scores = 0, 0, 0, 0, 0, []
        for i in range(8):
            for j in range(8):
                p = pmf(i, he)*pmf(j, ae)
                if i > j: hw += p
                elif i == j: dr += p
                else: aw += p
                if i > 0 and j > 0: bt += p
                if i + j > 2: o25 += p
                scores.append((i, j, p))
        scores.sort(key=lambda x: x[2], reverse=True)
        t = hw + dr + aw
        if t > 0: hw /= t; dr /= t; aw /= t
        return {"home_win": round(hw*100, 1), "draw": round(dr*100, 1), "away_win": round(aw*100, 1),
                "predicted_score": "%d-%d" % (scores[0][0], scores[0][1]),
                "home_xg": round(he, 2), "away_xg": round(ae, 2),
                "btts": round(bt/(hw+dr+aw)*100 if (hw+dr+aw) > 0 else 50, 1),
                "over_2_5": round(o25/(hw+dr+aw)*100 if (hw+dr+aw) > 0 else 50, 1),
                "top_scores": [{"score": "%d-%d"%(s[0],s[1]), "prob": round(s[2]*100, 1)} for s in scores[:5]]}

class RefinedPoissonModel:
    """赔率修正泊松(用比分赔率修正概率矩阵)"""
    def predict(self, home_xg, away_xg, odds_dict):
        try: lh = float(home_xg or 1.3); la = float(away_xg or 1.1)
        except: lh, la = 1.3, 1.1
        max_g = 7
        probs = np.zeros((max_g+1, max_g+1))
        def pmf(k, lam): return (lam**k)*math.exp(-lam)/math.factorial(k)
        for h in range(max_g+1):
            for a in range(max_g+1):
                probs[h, a] = pmf(h, lh)*pmf(a, la)
        # 用比分赔率修正
        if odds_dict and isinstance(odds_dict, dict):
            crs_corrections = {
                (1,1): "s11", (2,2): "s22", (0,0): "s00",
                (2,1): "w21", (1,0): "w10", (2,0): "w20",
                (0,1): "l01", (1,2): "l12", (0,2): "l02",
            }
            for (h, a), key in crs_corrections.items():
                odds_val = float(odds_dict.get(key, 0) or 0)
                if 3.0 <= odds_val <= 8.0:
                    probs[h, a] *= 1.3
                elif 8.0 < odds_val <= 12.0:
                    probs[h, a] *= 1.15
            # 总进球数修正
            ttg_corrections = {"a1": 1, "a2": 2, "a3": 3, "a4": 4}
            for key, total in ttg_corrections.items():
                odds_val = float(odds_dict.get(key, 0) or 0)
                if 3.0 <= odds_val <= 4.0:
                    for h in range(max_g+1):
                        for a in range(max_g+1):
                            if h + a == total: probs[h, a] *= 1.25
        psum = probs.sum()
        if psum > 0: probs /= psum
        hw, dr, aw, scores = 0, 0, 0, []
        for h in range(max_g+1):
            for a in range(max_g+1):
                p = probs[h, a]
                if h > a: hw += p
                elif h == a: dr += p
                else: aw += p
                scores.append({"score": "%d-%d" % (h, a), "prob": round(p*100, 1)})
        scores.sort(key=lambda x: x["prob"], reverse=True)
        return {"home_win": round(hw*100, 1), "draw": round(dr*100, 1), "away_win": round(aw*100, 1),
                "predicted_score": scores[0]["score"], "top_scores": scores[:5]}

class DixonColesModel:
    """Dixon-Coles低比分修正"""
    def predict(self, home_gf, home_ga, away_gf, away_ga):
        try: home_gf=float(home_gf or 1.3);away_gf=float(away_gf or 1.1)
        except: home_gf=1.3;away_gf=1.1
        he = max(0.2, min(home_gf*1.05, 5.0)); ae = max(0.2, min(away_gf*0.95, 5.0))
        rho = -0.15 if abs(he - ae) < 0.5 else -0.05
        def pmf(k, l): return (l**k)*math.exp(-l)/math.factorial(k)
        def tau(i, j, lam, mu, r):
            if i==0 and j==0: return 1-lam*mu*r
            elif i==0 and j==1: return 1+lam*r
            elif i==1 and j==0: return 1+mu*r
            elif i==1 and j==1: return 1-r
            return 1
        hw, dr, aw = 0, 0, 0
        for i in range(7):
            for j in range(7):
                p = max(0, tau(i,j,he,ae,rho)*pmf(i,he)*pmf(j,ae))
                if i > j: hw += p
                elif i == j: dr += p
                else: aw += p
        t = hw+dr+aw
        if t > 0: hw /= t; dr /= t; aw /= t
        return {"home_win": round(hw*100, 1), "draw": round(dr*100, 1), "away_win": round(aw*100, 1)}

# ============================================================
# 4. 评分模型
# ============================================================

class EloModel:
    """ELO评分(基于排名)"""
    def predict(self, home_rank, away_rank):
        try: hr = int(home_rank or 10); ar = int(away_rank or 10)
        except: hr = 10; ar = 10
        rh = 1500 + (20-max(1,hr))*15 + 50
        ra = 1500 + (20-max(1,ar))*15
        eh = 1/(1+10**((ra-rh)/400))
        df = 0.28 if abs(rh-ra) < 100 else 0.22
        hw = eh*(1-df/2); aw = (1-eh)*(1-df/2)
        return {"home_win": round(hw*100, 1), "draw": round(df*100, 1), "away_win": round(aw*100, 1), "elo_diff": round(rh-ra, 1)}

class BradleyTerryModel:
    """Bradley-Terry配对比较"""
    def predict(self, home_wins, home_played, away_wins, away_played):
        try:
            hw=int(home_wins or 5);hp=int(home_played or 15);aw=int(away_wins or 5);ap=int(away_played or 15)
        except: hw=5;hp=15;aw=5;ap=15
        h_str = max(0.1, (hw/max(1,hp)))*1.08
        a_str = max(0.1, (aw/max(1,ap)))*0.92
        dp = 0.24
        h = h_str/(h_str+a_str)*(1-dp); a = a_str/(h_str+a_str)*(1-dp)
        t = h+dp+a
        return {"home_win": round(h/t*100, 1), "draw": round(dp/t*100, 1), "away_win": round(a/t*100, 1)}

# ============================================================
# 5. ML模型
# ============================================================

def _generate_training_data(n=800):
    np.random.seed(42); X, y = [], []
    for _ in range(n):
        ph = np.random.uniform(0.25, 0.65)
        pd2 = np.random.uniform(0.15, 0.35)
        pa = max(0.05, 1.0-ph-pd2)
        total = ph+pd2+pa; ph/=total; pd2/=total; pa/=total
        X.append([ph, pd2, pa, ph/(ph+pa+0.001), pa/(ph+pa+0.001)])
        y.append(np.random.choice([0,1,2], p=[ph, pd2, pa]))
    return np.array(X), np.array(y)

def _build_ml_features(match, match_odds):
    try:
        sp_h = float(match.get("sp_home", 0))
        sp_d = float(match.get("sp_draw", 0))
        sp_a = float(match.get("sp_away", 0))
        if sp_h > 1 and sp_a > 1:
            ph, pd2, pa = 1/sp_h, 1/sp_d, 1/sp_a
            t = ph+pd2+pa
            return [ph/t, pd2/t, pa/t, (ph/t)/((ph+pa)/t+0.001), (pa/t)/((ph+pa)/t+0.001)]
    except: pass
    return [0.4, 0.28, 0.32, 0.55, 0.45]

class MLBase:
    def __init__(self, name):
        self.model = None; self.scaler = None; self.trained = False; self.name = name
    def train(self):
        if not HAS_SK: return
        try:
            X, y = _generate_training_data()
            self.scaler = StandardScaler(); X = self.scaler.fit_transform(X)
            self._init_model(); self.model.fit(X, y); self.trained = True
        except: pass
    def predict(self, match, match_odds=None):
        if not self.trained or not self.model:
            return {"home_win": 40, "draw": 30, "away_win": 30}
        try:
            feat = _build_ml_features(match, match_odds)
            proba = self.model.predict_proba(self.scaler.transform([feat]))[0]
            return {"home_win": round(proba[0]*100, 1), "draw": round(proba[1]*100, 1), "away_win": round(proba[2]*100, 1)}
        except:
            return {"home_win": 40, "draw": 30, "away_win": 30}

class RandomForestModel(MLBase):
    def __init__(self): super().__init__("RandomForest")
    def _init_model(self): self.model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)

class GradientBoostModel(MLBase):
    def __init__(self): super().__init__("GradientBoost")
    def _init_model(self): self.model = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)

class NeuralNetModel(MLBase):
    def __init__(self): super().__init__("NeuralNet")
    def _init_model(self): self.model = MLPClassifier(hidden_layer_sizes=(64,32,16), max_iter=500, random_state=42, early_stopping=True)

class LogisticModel(MLBase):
    def __init__(self): super().__init__("Logistic")
    def _init_model(self): self.model = LogisticRegression(C=0.5, max_iter=500, random_state=42)

class SVMModel(MLBase):
    def __init__(self): super().__init__("SVM")
    def _init_model(self): self.model = CalibratedClassifierCV(SVC(kernel="rbf", random_state=42), cv=3)

class KNNModel(MLBase):
    def __init__(self): super().__init__("KNN")
    def _init_model(self): self.model = KNeighborsClassifier(n_neighbors=7, weights="distance")

# ============================================================
# 6. 进球模型
# ============================================================

class PaceTotalGoalsModel:
    """节奏与总进球"""
    def predict(self, h_gf, h_ga, a_gf, a_ga, hs, ast):
        try:
            h_cs = float(hs.get("clean_sheets", 2))/max(1, float(hs.get("played", 10)))
            a_cs = float(ast.get("clean_sheets", 2))/max(1, float(ast.get("played", 10)))
        except: h_cs, a_cs = 0.2, 0.2
        try: h_gf=float(h_gf);h_ga=float(h_ga);a_gf=float(a_gf);a_ga=float(a_ga)
        except: h_gf=1.3;h_ga=1.1;a_gf=1.1;a_ga=1.3
        exp = (h_gf+a_ga)/2 + (a_gf+h_ga)/2
        exp *= (1.0 + (0.3 - (h_cs+a_cs)/2))
        over =​​​​​​​​​​​​​​​​
