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
    print("[WARN] sklearn 未安装，将使用降级模型")

def fetch_real_historical_data():
    leagues = ['E0', 'SP1', 'I1', 'D1', 'F1']
    seasons = ['2324', '2223', '2122'] 
    dfs = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for season in seasons:
        for league in leagues:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
            try:
                r = requests.get(url, headers=headers, timeout=10)
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
                X.append([prob_h, prob_d, prob_a, prob_h/(prob_h+prob_a), prob_a/(prob_h+prob_a)])
                y.append(target)
        except Exception: continue
    return np.array(X), np.array(y)

def _fallback_training_data(n=1000):
    np.random.seed(42); X, y = [], []
    for _ in range(n):
        ph, pd = np.random.uniform(0.2, 0.8), np.random.uniform(0.15, 0.35)
        pa = max(0.01, 1 - ph - pd)
        X.append([ph, pd, pa, ph/(ph+pa), pa/(ph+pa)])
        y.append(np.random.choice([0, 1, 2], p=[ph, pd, pa]))
    return np.array(X), np.array(y)

def _build_ml_features(match, match_odds):
    try:
        if match_odds and "bookmakers" in match_odds:
            ho, do2, ao = [], [], []
            for bk in match_odds["bookmakers"]:
                h2h = bk.get("markets", {}).get("h2h", {})
                if "Home" in h2h: ho.append(h2h["Home"])
                if "Draw" in h2h: do2.append(h2h["Draw"])
                if "Away" in h2h: ao.append(h2h["Away"])
            if ho and do2 and ao:
                prob_h, prob_d, prob_a = 1/(sum(ho)/len(ho)), 1/(sum(do2)/len(do2)), 1/(sum(ao)/len(ao))
            else: raise Exception()
        else:
            hw, hp = float(match.get("home_stats", {}).get("wins", 4)), float(match.get("home_stats", {}).get("played", 10))
            aw, ap = float(match.get("away_stats", {}).get("wins", 4)), float(match.get("away_stats", {}).get("played", 10))
            prob_h = max(0.2, min(0.75, (hw/max(1, hp)) * 1.1))
            prob_a = max(0.2, min(0.75, (aw/max(1, ap)) * 0.9))
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
            if v_win > 60 and (v_win - model_h) > 20: h_adj, signal = -5.0, "🔥主队散户过热(防冷)"
            elif v_lose > 60 and (v_lose - model_a) > 20: a_adj, signal = -5.0, "🔥客队散户过热(防冷)"
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
        if give_ball < 0 and "主胜降水" in odds_movement_str: h_adj, signal = 3.0, "📈主队让球强挡(真实看好)"
        elif give_ball > 0 and "客胜降水" in odds_movement_str: a_adj, signal = 3.0, "📈客队让球强挡(真实看好)"
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

class SmartMoneyDetector:
    def analyze(self, model_h, model_a, match_odds):
        if not match_odds or not match_odds.get("bookmakers"): return {"home_rlm_adj": 0, "away_rlm_adj": 0, "signal": ""}
        ho, ao = [], []
        for bk in match_odds["bookmakers"]:
            h2h = bk.get("markets", {}).get("h2h", {})
            if "Home" in h2h: ho.append(h2h["Home"])
            if "Away" in h2h: ao.append(h2h["Away"])
        if not ho or not ao: return {"home_rlm_adj": 0, "away_rlm_adj": 0, "signal": ""}
        diff_h = (1/(sum(ho)/len(ho))*100) - model_h
        diff_a = (1/(sum(ao)/len(ao))*100) - model_a
        signal = "⚠️跨盘防主" if diff_h > 12 else ("⚠️跨盘防客" if diff_a > 12 else "")
        return {"home_rlm_adj": diff_h * 0.4, "away_rlm_adj": diff_a * 0.4, "signal": signal}

class PaceTotalGoalsModel:
    def predict(self, hs, ast):
        try:
            h_gf, h_ga = float(hs.get("avg_goals_for", 1.3)), float(hs.get("avg_goals_against", 1.1))
            a_gf, a_ga = float(ast.get("avg_goals_for", 1.1)), float(ast.get("avg_goals_against", 1.3))
            h_cs = float(hs.get("clean_sheets", 2)) / max(1, float(hs.get("played", 10)))
            a_cs = float(ast.get("clean_sheets", 2)) / max(1, float(ast.get("played", 10)))
        except Exception: return {"over_2_5": 50.0, "expected_total": 2.5, "pace_rating": "中等"}
        final_exp = (h_gf * a_ga + a_gf * h_ga) * (1.0 + (0.3 - ((h_cs + a_cs) / 2)))
        over_prob = 1.0 - (math.exp(-final_exp) * (1 + final_exp + (final_exp**2)/2))
        return {"over_2_5": max(15.0, min(85.0, over_prob * 100)), "expected_total": round(final_exp, 2), "pace_rating": "极快" if final_exp > 3.0 else ("慢" if final_exp < 2.0 else "中等")}

class FormModel:
    def analyze(self, form):
        if not form: return {"score": 50}
        momentum = sum((3 if c=="W" else 1 if c=="D" else 0) * max(0.2, 1.0 - i*0.15) for i, c in enumerate(reversed(form)))
        tw = sum(3 * max(0.2, 1.0 - i*0.15) for i in range(len(form)))
        return {"score": round((momentum / tw) * 100 if tw > 0 else 50, 1)}

class OddsAnalyzer:
    def analyze_market(self, bookmakers):
        if not bookmakers: return {}
        ho, do2, ao = [], [], []
        for bk in bookmakers:
            h2h = bk.get("markets", {}).get("h2h", {})
            if "Home" in h2h: ho.append(h2h["Home"])
            if "Draw" in h2h: do2.append(h2h["Draw"])
            if "Away" in h2h: ao.append(h2h["Away"])
        if not ho: return {}
        return {"avg_home_odds": round(sum(ho)/len(ho), 2), "avg_draw_odds": round(sum(do2)/len(do2), 2), "avg_away_odds": round(sum(ao)/len(ao), 2)}

class PoissonModel:
    def predict(self, home_gf, home_ga, away_gf, away_ga, league_avg=1.35):
        try: home_gf, home_ga = float(home_gf or 1.3), float(home_ga or 1.1)
        except Exception: home_gf, home_ga = 1.3, 1.1
        try: away_gf, away_ga = float(away_gf or 1.1), float(away_ga or 1.3)
        except Exception: away_gf, away_ga = 1.1, 1.3
        he = max(0.3, min((home_gf / league_avg) * 1.10 * (away_ga / league_avg) * league_avg, 4.5))
        ae = max(0.2, min((away_gf / league_avg) * (home_ga / league_avg) * 0.90 * league_avg, 4.0))
        def pmf(k, l): return (l**k) * math.exp(-l) / math.factorial(k)
        hw, dr, aw, bt, scores = 0, 0, 0, 0, []
        for i in range(7):
            for j in range(7):
                p = pmf(i, he) * pmf(j, ae)
                if i > j: hw += p
                elif i == j: dr += p
                else: aw += p
                if i > 0 and j > 0: bt += p
                scores.append((i, j, p))
        scores.sort(key=lambda x: x[2], reverse=True); t = hw + dr + aw
        if t > 0: hw /= t; dr /= t; aw /= t
        return {"home_win": round(hw*100, 1), "draw": round(dr*100, 1), "away_win": round(aw*100, 1), "predicted_score": "%d-%d" % (scores[0][0], scores[0][1]), "home_xg": round(he, 2), "away_xg": round(ae, 2), "btts": round(bt*100, 1)}

class DixonColesModel:
    def predict(self, home_gf, home_ga, away_gf, away_ga):
        try: home_gf, away_gf = float(home_gf or 1.3), float(away_gf or 1.1)
        except Exception: home_gf, away_gf = 1.3, 1.1
        he = max(0.3, min(home_gf * 1.05, 4.0)); ae = max(0.2, min(away_gf * 0.95, 3.5))
        rho = -0.15 if abs(he - ae) < 0.5 else -0.05
        def pmf(k, l): return (l**k) * math.exp(-l) / math.factorial(k)
        def tau(i, j, lam, mu, r):
            if i == 0 and j == 0: return 1 - lam * mu * r
            elif i == 0 and j == 1: return 1 + lam * r
            elif i == 1 and j == 0: return 1 + mu * r
            elif i == 1 and j == 1: return 1 - r
            return 1
        hw, dr, aw = 0, 0, 0
        for i in range(6):
            for j in range(6):
                p = max(0, tau(i, j, he, ae, rho) * pmf(i, he) * pmf(j, ae))
                if i > j: hw += p
                elif i == j: dr += p
                else: aw += p
        t = hw + dr + aw
        if t > 0: hw /= t; dr /= t; aw /= t
        return {"home_win": round(hw*100, 1), "draw": round(dr*100, 1), "away_win": round(aw*100, 1)}

class EnsemblePredictor:
    def __init__(self):
        self.poisson = PoissonModel()
        self.dixon = DixonColesModel()
        self.rf = RandomForestModel()
        self.lr = LogisticModel()
        self.public_sentiment = PublicSentimentModel()
        self.handicap_momentum = HandicapMomentumModel()
        self.injury_penalty = InjuryPenaltyModel()
        self.smart_money = SmartMoneyDetector()
        self.pace_totals = PaceTotalGoalsModel()
        self.form = FormModel()
        self.odds = OddsAnalyzer()
        print("[Models] 初始化量化引擎(4基础+5风控)就绪...")
        self.rf.train(); self.lr.train()

    def predict(self, match, odds_data=None):
        hs = match.get("home_stats", {}); ast = match.get("away_stats", {})
        poi = self.poisson.predict(hs.get("avg_goals_for"), hs.get("avg_goals_against"), ast.get("avg_goals_for"), ast.get("avg_goals_against"))
        dc = self.dixon.predict(hs.get("avg_goals_for"), hs.get("avg_goals_against"), ast.get("avg_goals_for"), ast.get("avg_goals_against"))
        rf = self.rf.predict(match, odds_data)
        lr = self.lr.predict(match, odds_data)
        pace = self.pace_totals.predict(hs, ast) 
        oa = self.odds.analyze_market(odds_data.get("bookmakers", [])) if odds_data else {}
            
        w = {"poisson": 0.35, "dixon": 0.25, "rf": 0.25, "lr": 0.15}
        models_list = [("poisson", poi), ("dixon", dc), ("rf", rf), ("lr", lr)]
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
        
        sm_data = self.smart_money.analyze(hp, ap, odds_data)
        hp += sm_data["home_rlm_adj"]; ap += sm_data["away_rlm_adj"]
        if sm_data["signal"]: signals.append(sm_data["signal"])
        
        t = hp + dp + ap
        if t > 0: hp = round(hp/t*100, 1); dp = round(dp/t*100, 1); ap = round(100-hp-dp, 1)
        
        agree_h = sum(1 for _, p in models_list if p.get("home_win", 0) > max(p.get("draw", 0), p.get("away_win", 0)))
        agree_a = sum(1 for _, p in models_list if p.get("away_win", 0) > max(p.get("home_win", 0), p.get("draw", 0)))
        consensus = max(agree_h, agree_a)
        cf = min(95, max(30, 40 + consensus * 8 + (10 if max(hp, dp, ap) > 60 else 0)))
        
        return {
            "home_win_pct": hp, "draw_pct": dp, "away_win_pct": ap, 
            "predicted_score": poi["predicted_score"], "confidence": cf, 
            "odds": oa, "smart_money_signal": " | ".join(signals) if signals else "盘口与情绪正常",
            "over_2_5": pace["over_2_5"], "btts": poi.get("btts", 50),
            "model_consensus": consensus, "total_models": 4
        }
