import math
import random
import time
import io
import re
import numpy as np
import pandas as pd
import requests

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SK = True
except Exception:
    HAS_SK = False

class TrueOddsModel:
    def calculate(self, sp_h, sp_d, sp_a):
        if sp_h <= 1.0 or sp_d <= 1.0 or sp_a <= 1.0:
            return 0.33, 0.33, 0.34
        imp_h, imp_d, imp_a = 1.0 / sp_h, 1.0 / sp_d, 1.0 / sp_a
        margin = imp_h + imp_d + imp_a
        true_h = imp_h / margin
        true_d = imp_d / margin
        true_a = imp_a / margin
        return true_h, true_d, true_a

class HandicapMismatchModel:
    def analyze(self, true_h_prob, handicap_str):
        try:
            m = re.search(r'让(-?\d+(\.\d+)?)', str(handicap_str))
            if not m: return 0.0, "盘口正常"
            actual_hc = float(m.group(1))
            
            if true_h_prob >= 0.80: exp_hc = -1.5
            elif true_h_prob >= 0.70: exp_hc = -1.0
            elif true_h_prob >= 0.60: exp_hc = -0.5
            elif true_h_prob >= 0.55: exp_hc = -0.25
            elif true_h_prob <= 0.20: exp_hc = 1.5
            elif true_h_prob <= 0.30: exp_hc = 1.0
            elif true_h_prob <= 0.40: exp_hc = 0.5
            else: exp_hc = 0.0
            
            diff = actual_hc - exp_hc
            
            if diff >= 0.75 and true_h_prob > 0.55: 
                return -12.0, "🚨 欧亚错位：让球畸浅，强烈诱上"
            elif diff <= -0.75 and true_h_prob < 0.45:
                return 8.0, "🚨 欧亚错位：逆势深盘，强力阻击"
            return 0.0, "盘口正常"
        except:
            return 0.0, "盘口正常"

class H2HBloodlineModel:
    def analyze(self, h2h_data, current_home, current_away):
        if not h2h_data or not isinstance(h2h_data, list):
            return {"h_adj": 0.0, "a_adj": 0.0}
        
        h_score, a_score, total_weight = 0.0, 0.0, 0.0
        
        for i, match in enumerate(h2h_data):
            weight = max(0.2, 1.0 - i * 0.2) 
            score_str = match.get("score", "")
            m_home = str(match.get("home", ""))
            try: pts_h, pts_a = map(int, score_str.split("-"))
            except: continue
            
            if str(current_home) in m_home: 
                if pts_h > pts_a: h_score += 3 * weight
                elif pts_h == pts_a: h_score += 1 * weight; a_score += 1 * weight
                else: a_score += 3 * weight
            else:
                if pts_a > pts_h: h_score += 3 * weight
                elif pts_h == pts_a: h_score += 1 * weight; a_score += 1 * weight
                else: a_score += 3 * weight
            total_weight += 3 * weight
            
        if total_weight == 0: return {"h_adj": 0.0, "a_adj": 0.0}
        h_adv = (h_score / total_weight) - 0.5 
        return {"h_adj": h_adv * 6.0, "a_adj": -h_adv * 6.0} 

class RefinedPoissonModel:
    def predict(self, home_xg, away_xg, odds_dict):
        try: lh = float(home_xg or 1.3); la = float(away_xg or 1.1)
        except Exception: lh, la = 1.3, 1.1
        
        max_g = 8
        probs = np.zeros((max_g+1, max_g+1))
        def pmf(k, lam): return (lam**k) * math.exp(-lam) / math.factorial(k)
        
        for h in range(max_g + 1):
            for a in range(max_g + 1):
                probs[h, a] = pmf(h, lh) * pmf(a, la)
                
        if odds_dict and isinstance(odds_dict, dict):
            if 3.05 <= odds_dict.get("a2", 999) <= 3.10:
                for h in range(max_g+1):
                    for a in range(max_g+1):
                        if h+a == 2: probs[h, a] *= 1.40
            if 4.50 <= odds_dict.get("a4", 999) <= 5.00:
                for h in range(max_g+1):
                    for a in range(max_g+1):
                        if h+a == 4: probs[h, a] *= 1.42
            if 5.80 <= odds_dict.get("s11", 999) <= 6.10: probs[1, 1] *= 1.38
            if 10.00 <= odds_dict.get("s22", 999) <= 11.50: probs[2, 2] *= 1.45
            if 4.50 <= odds_dict.get("a1", 999) <= 5.00:
                for h in range(max_g+1):
                    for a in range(max_g+1):
                        if h+a == 1: probs[h, a] *= 1.35
            if 3.60 <= odds_dict.get("a3", 999) <= 3.90:
                for h in range(max_g+1):
                    for a in range(max_g+1):
                        if h+a == 3: probs[h, a] *= 1.40
            if 7.50 <= odds_dict.get("w21", 999) <= 8.50: probs[2, 1] *= 1.42

        probs /= probs.sum()
        hw, dr, aw, scores = 0.0, 0.0, 0.0, []
        for h in range(max_g+1):
            for a in range(max_g+1):
                p = probs[h, a]
                if h > a: hw += p
                elif h == a: dr += p
                else: aw += p
                scores.append({"score": f"{h}-{a}", "prob": round(p*100, 1)})
        
        scores.sort(key=lambda x: x["prob"], reverse=True)
        return {
            "home_win": round(hw*100, 1), "draw": round(dr*100, 1), "away_win": round(aw*100, 1),
            "predicted_score": scores[0]["score"], "top_scores": scores[:5],
            "v2_details": {"p22": f"{probs[2,2]*100:.1f}%", "p11": f"{probs[1,1]*100:.1f}%", "p21": f"{probs[2,1]*100:.1f}%", "p12": f"{probs[1,2]*100:.1f}%"}
        }

# 🔥 核心修复三：虚拟数据强制严格归一化，彻底铲除数学死机漏洞！
def fetch_real_historical_data():
    leagues, seasons = ['E0', 'SP1'], ['2324'] 
    dfs, headers = [], {"User-Agent": "Mozilla/5.0"}
    for season in seasons:
        for league in leagues:
            try:
                r = requests.get(f"[https://www.football-data.co.uk/mmz4281/](https://www.football-data.co.uk/mmz4281/){season}/{league}.csv", headers=headers, timeout=5)
                if r.status_code == 200: dfs.append(pd.read_csv(io.StringIO(r.text), on_bad_lines='skip'))
            except: continue
    if not dfs: return _fallback_training_data()
    full_df = pd.concat(dfs, ignore_index=True).dropna(subset=['FTR', 'B365H', 'B365D', 'B365A'])
    X, y = [], []
    for _, row in full_df.iterrows():
        try:
            prob_h, prob_d, prob_a = 1/float(row['B365H']), 1/float(row['B365D']), 1/float(row['B365A'])
            target = {'H': 0, 'D': 1, 'A': 2}.get(str(row['FTR']).upper())
            if target is not None:
                X.append([prob_h, prob_d, prob_a, prob_h/(prob_h+prob_a), prob_a/(prob_h+prob_a)]); y.append(target)
        except: continue
    if len(X) < 10: return _fallback_training_data()
    return np.array(X), np.array(y)

def _fallback_training_data(n=1000):
    np.random.seed(42); X, y = [], []
    for _ in range(n):
        ph = np.random.uniform(0.3, 0.6)
        pd = np.random.uniform(0.2, 0.3)
        pa = max(0.01, 1.0 - ph - pd)
        
        # 强制归一化处理，保证三个概率相加绝对等于 1.0
        total = ph + pd + pa
        ph, pd, pa = ph/total, pd/total, pa/total
        
        X.append([ph, pd, pa, ph/(ph+pa), pa/(ph+pa)])
        y.append(np.random.choice([0, 1, 2], p=[ph, pd, pa]))
    return np.array(X), np.array(y)

def _build_ml_features(match, match_odds):
    try:
        sp_h, sp_d, sp_a = float(match.get("sp_home", 0)), float(match.get("sp_draw", 0)), float(match.get("sp_away", 0))
        if sp_h > 1.0 and sp_a > 1.0:
            prob_h, prob_d, prob_a = 1/sp_h, 1/sp_d, 1/sp_a
            t = prob_h + prob_d + prob_a
            return [prob_h/t, prob_d/t, prob_a/t, (prob_h/t)/(prob_h/t+prob_a/t), (prob_a/t)/(prob_h/t+prob_a/t)]
        hw, hp = float(match.get("home_stats", {}).get("wins", 4)), float(match.get("home_stats", {}).get("played", 10))
        aw, ap = float(match.get("away_stats", {}).get("wins", 4)), float(match.get("away_stats", {}).get("played", 10))
        prob_h, prob_a = max(0.2, min(0.75, (hw/max(1, hp)) * 1.1)), max(0.2, min(0.75, (aw/max(1, ap)) * 0.9))
        prob_d = max(0.15, 1 - prob_h - prob_a)
        return [prob_h, prob_d, prob_a, prob_h/(prob_h+prob_a), prob_a/(prob_h+prob_a)]
    except Exception: return [0.45, 0.25, 0.30, 0.6, 0.4]

class MLPredictorBase:
    def __init__(self, name): self.model = None; self.scaler = None; self.trained = False; self.name = name
    def train(self):
        if not HAS_SK: return
        try:
            X, y = fetch_real_historical_data()
            self.scaler = StandardScaler(); X = self.scaler.fit_transform(X)
            self._init_model(); self.model.fit(X, y); self.trained = True
        except Exception: pass
    def predict(self, match, match_odds):
        if not self.trained or not self.model: return {"home_win": 40, "draw": 30, "away_win": 30}
        try:
            proba = self.model.predict_proba(self.scaler.transform([_build_ml_features(match, match_odds)]))[0]
            return {"home_win": round(proba[0]*100, 1), "draw": round(proba[1]*100, 1), "away_win": round(proba[2]*100, 1)}
        except: return {"home_win": 40, "draw": 30, "away_win": 30}

class RandomForestModel(MLPredictorBase):
    def __init__(self): super().__init__("RandomForest")
    def _init_model(self): self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

class LogisticModel(MLPredictorBase):
    def __init__(self): super().__init__("Logistic")
    def _init_model(self): self.model = LogisticRegression(C=0.5, max_iter=200, random_state=42)

class PaceTotalGoalsModel:
    def predict(self, h_gf, h_ga, a_gf, a_ga, hs, ast):
        try:
            h_cs = float(hs.get("clean_sheets", 2)) / max(1, float(hs.get("played", 10)))
            a_cs = float(ast.get("clean_sheets", 2)) / max(1, float(ast.get("played", 10)))
        except Exception: 
            h_cs, a_cs = 0.2, 0.2
        final_exp = (h_gf * a_ga + a_gf * h_ga) * (1.0 + (0.3 - ((h_cs + a_cs) / 2)))
        over_prob = 1.0 - (math.exp(-final_exp) * (1 + final_exp + (final_exp**2)/2))
        return {"over_2_5": max(15.0, min(85.0, over_prob * 100)), "expected_total": round(final_exp, 2), "pace_rating": "极快" if final_exp > 3.0 else ("慢" if final_exp < 2.0 else "中等")}

class FormModel:
    def analyze(self, form):
        if not form: return {"score": 50}
        momentum = sum((3 if c=="W" else 1 if c=="D" else 0) * max(0.2, 1.0 - i*0.15) for i, c in enumerate(reversed(form)))
        tw = sum(3 * max(0.2, 1.0 - i*0.15) for i in range(len(form)))
        return {"score": round((momentum / tw) * 100 if tw > 0 else 50, 1)}

class PoissonModel:
    def predict(self, home_gf, home_ga, away_gf, away_ga, league_avg=1.35):
        he = max(0.2, min((home_gf / league_avg) * 1.10 * (away_ga / league_avg) * league_avg, 5.5))
        ae = max(0.2, min((away_gf / league_avg) * (home_ga / league_avg) * 0.90 * league_avg, 5.5))
        def pmf(k, l): return (l**k) * math.exp(-l) / math.factorial(k)
        hw, dr, aw, bt, scores = 0, 0, 0, 0, []
        for i in range(8):
            for j in range(8):
                p = pmf(i, he) * pmf(j, ae)
                if i > j: hw += p; bt += p if i>0 and j>0 else 0
                elif i == j: dr += p; bt += p if i>0 and j>0 else 0
                else: aw += p; bt += p if i>0 and j>0 else 0
                scores.append((i, j, p))
        scores.sort(key=lambda x: x[2], reverse=True); t = hw + dr + aw
        if t > 0: hw /= t; dr /= t; aw /= t
        return {"home_win": round(hw*100, 1), "draw": round(dr*100, 1), "away_win": round(aw*100, 1), "predicted_score": "%d-%d" % (scores[0][0], scores[0][1]), "home_xg": round(he, 2), "away_xg": round(ae, 2), "btts": round(bt*100, 1)}

class DixonColesModel:
    def predict(self, home_gf, home_ga, away_gf, away_ga):
        he = max(0.2, min(home_gf * 1.05, 5.0)); ae = max(0.2, min(away_gf * 0.95, 5.0))
        rho = -0.15 if abs(he - ae) < 0.5 else -0.05
        def pmf(k, l): return (l**k) * math.exp(-l) / math.factorial(k)
        def tau(i, j, lam, mu, r):
            if i == 0 and j == 0: return 1 - lam * mu * r
            elif i == 0 and j == 1: return 1 + lam * r
            elif i == 1 and j == 0: return 1 + mu * r
            elif i == 1 and j == 1: return 1 - r
            return 1
        hw, dr, aw = 0, 0, 0
        for i in range(7):
            for j in range(7):
                p = max(0, tau(i, j, he, ae, rho) * pmf(i, he) * pmf(j, ae))
                if i > j: hw += p
                elif i == j: dr += p
                else: aw += p
        t = hw + dr + aw
        if t > 0: hw /= t; dr /= t; aw /= t
        return {"home_win": round(hw*100, 1), "draw": round(dr*100, 1), "away_win": round(aw*100, 1)}

class EloModel:
    def predict(self, home_rank, away_rank):
        rh = 1500 + (20 - max(1, home_rank)) * 15 + 50
        ra = 1500 + (20 - max(1, away_rank)) * 15
        eh = 1 / (1 + 10 ** ((ra - rh) / 400))
        df = 0.28 if abs(rh - ra) < 100 else 0.22
        hw = eh * (1 - df / 2); aw = (1 - eh) * (1 - df / 2)
        return {"home_win": round(hw*100, 1), "draw": round(df*100, 1), "away_win": round(aw*100, 1), "elo_diff": round(rh - ra, 1)}

class EnsemblePredictor:
    def __init__(self):
        self.poisson = PoissonModel()
        self.refined_poisson = RefinedPoissonModel()
        self.dixon = DixonColesModel()
        self.rf = RandomForestModel()
        self.lr = LogisticModel()
        self.pace_totals = PaceTotalGoalsModel()
        self.form = FormModel()
        self.elo = EloModel()
        
        self.true_odds = TrueOddsModel()
        self.hc_mismatch = HandicapMismatchModel()
        self.h2h_bloodline = H2HBloodlineModel()
        
        print("[Models] 本地冷酷数学模型矩阵加载完毕...")
        self.rf.train(); self.lr.train()

    def predict(self, match, odds_data=None):
        hs = match.get("home_stats", {}); ast = match.get("away_stats", {})
        h_gf_base = float(hs.get("avg_goals_for", 1.3)); h_ga_base = float(hs.get("avg_goals_against", 1.1))
        a_gf_base = float(ast.get("avg_goals_for", 1.1)); a_ga_base = float(ast.get("avg_goals_against", 1.3))
        
        sp_h = float(match.get("sp_home", 0))
        sp_d = float(match.get("sp_draw", 0))
        sp_a = float(match.get("sp_away", 0))
        v2_odds = match.get("v2_odds_dict", {})
        extreme_warning = "无"
        signals = []
        
        true_h, true_d, true_a = self.true_odds.calculate(sp_h, sp_d, sp_a)
        hc_adj, hc_signal = self.hc_mismatch.analyze(true_h, match.get("handicap_info", ""))
        if hc_signal != "盘口正常": signals.append(hc_signal)
            
        hf = self.form.analyze(hs.get("form", ""))
        af = self.form.analyze(ast.get("form", ""))
        h_mom = max(0.6, min(1.4, hf["score"] / 50.0))
        a_mom = max(0.6, min(1.4, af["score"] / 50.0))
        
        h_gf = h_gf_base * h_mom
        a_gf = a_gf_base * a_mom
        
        expected_total = 2.5 + (0.25 - true_d) * 8.0
        expected_total = max(1.5, expected_total)
        is_massacre = False
        if float(v2_odds.get("a5", 999)) < 7.5 or float(v2_odds.get("a6", 999)) < 13.0:
            expected_total *= 1.35
            is_massacre = True

        if true_a > 0.60 and true_h < 0.22: 
            a_gf, h_gf = expected_total * 0.85, expected_total * 0.15
            if is_massacre: 
                extreme_warning = "🔴 实力碾压预警 (客防大比分穿盘)"
                a_gf = max(a_gf, 3.8) 
        elif true_h > 0.60 and true_a < 0.22: 
            h_gf, a_gf = expected_total * 0.85, expected_total * 0.15
            if is_massacre: 
                extreme_warning = "🔴 实力碾压预警 (主防大比分穿盘)"
                h_gf = max(h_gf, 3.8)
        else:
            if true_h + true_a > 0:
                h_gf = expected_total * (true_h / (true_h + true_a))
                a_gf = expected_total * (true_a / (true_h + true_a))
                
        poi = self.poisson.predict(h_gf, h_ga_base, a_gf, a_ga_base)
        dc = self.dixon.predict(h_gf, h_ga_base, a_gf, a_ga_base)
        rf = self.rf.predict(match, odds_data)
        lr = self.lr.predict(match, odds_data)
        pace = self.pace_totals.predict(h_gf, h_ga_base, a_gf, a_ga_base, hs, ast) 
        ref_poi = self.refined_poisson.predict(h_gf, a_gf, v2_odds)
            
        w = {"poisson": 0.15, "refined_poisson": 0.25, "dixon": 0.20, "rf": 0.25, "lr": 0.15}
        models_list = [("poisson", poi), ("refined_poisson", ref_poi), ("dixon", dc), ("rf", rf), ("lr", lr)]
        
        hp, dp, ap = 0.0, 0.0, 0.0
        for name, pred in models_list:
            hp += pred.get("home_win", 33) * w[name]; dp += pred.get("draw", 33) * w[name]; ap += pred.get("away_win", 33) * w[name]
            
        h2h_adj_data = self.h2h_bloodline.analyze(match.get("h2h", []), match.get("home_team"), match.get("away_team"))
        
        hp += hc_adj + (hf["score"] - af["score"]) * 0.05 + h2h_adj_data["h_adj"]
        ap += -hc_adj - (hf["score"] - af["score"]) * 0.05 + h2h_adj_data["a_adj"]
        
        t = hp + dp + ap
        if t > 0: hp = round(hp/t*100, 1); dp = round(dp/t*100, 1); ap = round(100-hp-dp, 1)
        
        agree_h = sum(1 for _, p in models_list if p.get("home_win", 0) > max(p.get("draw", 0), p.get("away_win", 0)))
        agree_a = sum(1 for _, p in models_list if p.get("away_win", 0) > max(p.get("home_win", 0), p.get("draw", 0)))
        consensus = max(agree_h, agree_a)
        cf = min(95, max(30, 40 + consensus * 8 + (10 if max(hp, dp, ap) > 60 else 0)))
        
        return {
            "home_win_pct": hp, "draw_pct": dp, "away_win_pct": ap, 
            "predicted_score": ref_poi["predicted_score"], "confidence": cf, 
            "poisson": poi, "refined_poisson": ref_poi, "dixon_coles": dc, "random_forest": rf, "logistic": lr,
            "elo": self.elo.predict(match.get("home_rank", 10), match.get("away_rank", 10)), "home_form": hf, "away_form": af,
            "extreme_warning": extreme_warning, 
            "smart_money_signal": " | ".join(signals) if signals else "正常",
            "over_2_5": pace["over_2_5"], "btts": poi.get("btts", 50),
            "pace_rating": pace["pace_rating"],
            "expected_total_goals": pace["expected_total"],
            "model_consensus": consensus, "total_models": len(models_list)
        }
