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

def fetch_real_historical_data():
    leagues, seasons = ['E0', 'SP1', 'I1', 'D1', 'F1'], ['2324', '2223', '2122'] 
    dfs, headers = [], {"User-Agent": "Mozilla/5.0"}
    for season in seasons:
        for league in leagues:
            try:
                r = requests.get(f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv", headers=headers, timeout=10)
                if r.status_code == 200: dfs.append(pd.read_csv(io.StringIO(r.text), on_bad_lines='skip'))
            except Exception: continue
    if not dfs: return _fallback_training_data()
    full_df = pd.concat(dfs, ignore_index=True).dropna(subset=['FTR', 'B365H', 'B365D', 'B365A'])
    X, y = [], []
    for _, row in full_df.iterrows():
        try:
            prob_h, prob_d, prob_a = 1/float(row['B365H']), 1/float(row['B365D']), 1/float(row['B365A'])
            target = {'H': 0, 'D': 1, 'A': 2}.get(str(row['FTR']).upper())
            if target is not None:
                X.append([prob_h, prob_d, prob_a, prob_h/(prob_h+prob_a), prob_a/(prob_h+prob_a)]); y.append(target)
        except Exception: continue
    return np.array(X), np.array(y)

def _fallback_training_data(n=1000):
    np.random.seed(42); X, y = [], []
    for _ in range(n):
        ph, pd = np.random.uniform(0.2, 0.8), np.random.uniform(0.15, 0.35); pa = max(0.01, 1 - ph - pd)
        X.append([ph, pd, pa, ph/(ph+pa), pa/(ph+pa)]); y.append(np.random.choice([0, 1, 2], p=[ph, pd, pa]))
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
        X, y = fetch_real_historical_data()
        self.scaler = StandardScaler(); X = self.scaler.fit_transform(X)
        self._init_model(); self.model.fit(X, y); self.trained = True
    def predict(self, match, match_odds):
        if not self.trained or not self.model: return {"home_win": 40, "draw": 30, "away_win": 30}
        proba = self.model.predict_proba(self.scaler.transform([_build_ml_features(match, match_odds)]))[0]
        return {"home_win": round(proba[0]*100, 1), "draw": round(proba[1]*100, 1), "away_win": round(proba[2]*100, 1)}

class RandomForestModel(MLPredictorBase):
    def __init__(self): super().__init__("RandomForest")
    def _init_model(self): self.model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)

class LogisticModel(MLPredictorBase):
    def __init__(self): super().__init__("Logistic")
    def _init_model(self): self.model = LogisticRegression(C=0.5, max_iter=500, random_state=42)

class PublicSentimentModel:
    def analyze(self, votes, model_h, model_d, model_a):
        if not votes or not isinstance(votes, dict): return {"h_adj": 0, "d_adj": 0, "a_adj": 0, "signal": "情绪正常"}
        try:
            v_win, v_lose = float(votes.get("win", 33)), float(votes.get("lose", 33))
            h_adj, a_adj, signal = 0, 0, "情绪平稳"
            if v_win > 60 and (v_win - model_h) > 20: h_adj, signal = -5.0, "🔥主队大热必死预警"
            elif v_lose > 60 and (v_lose - model_a) > 20: a_adj, signal = -5.0, "🔥客队大热必死预警"
            return {"h_adj": h_adj, "d_adj": 0, "a_adj": a_adj, "signal": signal}
        except Exception: return {"h_adj": 0, "d_adj": 0, "a_adj": 0, "signal": "情绪正常"}

class HandicapMomentumModel:
    def analyze(self, handicap_str, odds_movement_str):
        h_adj, a_adj, signal = 0, 0, "水位平稳"
        give_ball = 0
        try:
            m = re.search(r'让(-?\d+)', str(handicap_str))
            if m: give_ball = int(m.group(1))
        except Exception: pass
        if give_ball < 0 and "主胜降水" in odds_movement_str: h_adj, signal = 3.0, "📈主队让球强挡"
        elif give_ball > 0 and "客胜降水" in odds_movement_str: a_adj, signal = 3.0, "📈客队让球强挡"
        elif give_ball < 0 and "主胜升水" in odds_movement_str: h_adj, signal = -2.0, "⚠️强让弱升水(诱导盘)"
        return {"h_adj": h_adj, "a_adj": a_adj, "signal": signal}

class InjuryPenaltyModel:
    def analyze(self, h_inj_text, g_inj_text):
        h_penalty, a_penalty = 0.0, 0.0
        for word in ["停赛", "红牌停赛", "赛季报销", "骨折", "韧带撕裂"]:
            if word in str(h_inj_text): h_penalty -= 2.5
            if word in str(g_inj_text): a_penalty -= 2.5
        h_penalty -= min(str(h_inj_text).count("受伤") * 0.5, 3.0)
        a_penalty -= min(str(g_inj_text).count("受伤") * 0.5, 3.0)
        return {"h_adj": h_penalty, "a_adj": a_penalty}

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
        self.public_sentiment = PublicSentimentModel()
        self.handicap_momentum = HandicapMomentumModel()
        self.injury_penalty = InjuryPenaltyModel()
        self.pace_totals = PaceTotalGoalsModel()
        self.form = FormModel()
        self.elo = EloModel()
        print("[Models] 暴力美学极限矩阵加载完毕...")
        self.rf.train(); self.lr.train()

    def predict(self, match, odds_data=None):
        hs = match.get("home_stats", {}); ast = match.get("away_stats", {})
        h_gf = float(hs.get("avg_goals_for", 1.3)); h_ga = float(hs.get("avg_goals_against", 1.1))
        a_gf = float(ast.get("avg_goals_for", 1.1)); a_ga = float(ast.get("avg_goals_against", 1.3))
        
        sp_h, sp_d, sp_a = float(match.get("sp_home", 0)), float(match.get("sp_draw", 0)), float(match.get("sp_away", 0))
        v2_odds = match.get("v2_odds_dict", {})
        extreme_warning = "无"
        
        # 🔥🔥🔥 核心升级：强队降维打击机制与惨案预警 🔥🔥🔥
        if sp_h > 1.0 and sp_a > 1.0 and sp_d > 1.0:
            prob_h, prob_d, prob_a = 1/sp_h, 1/sp_d, 1/sp_a
            margin = prob_h + prob_d + prob_a
            prob_h /= margin; prob_d /= margin; prob_a /= margin
            
            expected_total = 2.5 + (0.25 - prob_d) * 8.0
            expected_total = max(1.5, expected_total)
            
            # 检测庄家是否极度防范大球（a5, a6 赔率异常低）
            is_massacre = False
            if float(v2_odds.get("a5", 999)) < 7.5 or float(v2_odds.get("a6", 999)) < 13.0:
                expected_total *= 1.35  # 强行拉高总进球期望，解除 0-2 封印
                is_massacre = True

            # 强行实力纠偏，赋予碾压局 3.5 球以上的恐怖火力
            if prob_a > 0.60 and prob_h < 0.22: # 客队绝对碾压 (如拜仁打亚特兰大)
                a_gf = expected_total * 0.85
                h_gf = expected_total * 0.15
                if is_massacre: 
                    extreme_warning = "🩸 极限惨案预警 (客队穿盘)"
                    a_gf = max(a_gf, 3.8) # 给强队保底恐怖的 xG
            elif prob_h > 0.60 and prob_a < 0.22: # 主队绝对碾压
                h_gf = expected_total * 0.85
                a_gf = expected_total * 0.15
                if is_massacre: 
                    extreme_warning = "🩸 极限惨案预警 (主队穿盘)"
                    h_gf = max(h_gf, 3.8)
            elif is_massacre: # 势均力敌但庄家怕大球 (如马竞打热刺)
                extreme_warning = "🔥 极致对攻大战 (防穿盘)"
                h_gf = max(expected_total * (prob_h / (prob_h + prob_a)), 2.2)
                a_gf = max(expected_total * (prob_a / (prob_h + prob_a)), 2.2)
            else:
                h_gf = expected_total * (prob_h / (prob_h + prob_a))
                a_gf = expected_total * (prob_a / (prob_h + prob_a))
                
        poi = self.poisson.predict(h_gf, h_ga, a_gf, a_ga)
        dc = self.dixon.predict(h_gf, h_ga, a_gf, a_ga)
        rf = self.rf.predict(match, odds_data)
        lr = self.lr.predict(match, odds_data)
        pace = self.pace_totals.predict(h_gf, h_ga, a_gf, a_ga, hs, ast) 
        
        ref_poi = self.refined_poisson.predict(h_gf, a_gf, v2_odds)
            
        w = {"poisson": 0.15, "refined_poisson": 0.25, "dixon": 0.20, "rf": 0.25, "lr": 0.15}
        models_list = [("poisson", poi), ("refined_poisson", ref_poi), ("dixon", dc), ("rf", rf), ("lr", lr)]
        
        hp, dp, ap = 0.0, 0.0, 0.0
        for name, pred in models_list:
            hp += pred.get("home_win", 33) * w[name]; dp += pred.get("draw", 33) * w[name]; ap += pred.get("away_win", 33) * w[name]
            
        signals = []
        hf, af = self.form.analyze(hs.get("form", "")), self.form.analyze(ast.get("form", ""))
        hp += (hf["score"] - af["score"]) * 0.08; ap -= (hf["score"] - af["score"]) * 0.08
        
        sent_data = self.public_sentiment.analyze(match.get("votes", {}), hp, dp, ap)
        hp += sent_data["h_adj"]; ap += sent_data["a_adj"]
        if sent_data["signal"] != "情绪平稳": signals.append(sent_data["signal"])
            
        hcp_data = self.handicap_momentum.analyze(match.get("handicap_info", ""), match.get("odds_movement", ""))
        hp += hcp_data["h_adj"]; ap += hcp_data["a_adj"]
        if hcp_data["signal"] != "水位平稳": signals.append(hcp_data["signal"])
            
        intel = match.get("intelligence", {})
        inj_data = self.injury_penalty.analyze(intel.get("h_inj", ""), intel.get("g_inj", ""))
        hp += inj_data["h_adj"]; ap += inj_data["a_adj"]
        
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
            "smart_money_signal": " | ".join(signals) if signals else "盘口与情绪正常",
            "extreme_warning": extreme_warning, # 🔥 传给前端的惨案预警灯
            "over_2_5": pace["over_2_5"], "btts": poi.get("btts", 50),
            "pace_rating": pace["pace_rating"],
            "expected_total_goals": pace["expected_total"],
            "model_consensus": consensus, "total_models": len(models_list)
        }
