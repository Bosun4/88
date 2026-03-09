""" 主运行脚本： 1. 抓取数据 2. AI预测 3. 生成前端JSON 4. 更新 index.html """
import math
import random
import time
import io
import numpy as np
import pandas as pd
import requests
from collections import defaultdict

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SK = True
except:
    HAS_SK = False
    print("[WARN] sklearn 未安装，将使用降级模型")

# ==========================================
# 核心数据引擎: football-data.co.uk 真实历史数据
# ==========================================

def fetch_real_historical_data():
    """从 football-data.co.uk 实时拉取欧洲主流联赛真实历史数据作为训练集"""
    print("    [Data] 正在从 football-data.co.uk 下载真实历史比赛与赔率数据...")
    leagues = ['E0', 'SP1', 'I1', 'D1', 'F1']
    seasons = ['2324', '2223', '2122'] 
    
    dfs = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    for season in seasons:
        for league in leagues:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
            try:
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code == 200:
                    df = pd.read_csv(io.StringIO(r.text), on_bad_lines='skip')
                    dfs.append(df)
            except Exception:
                continue
                
    if not dfs:
        print("    [Data] ⚠️ 下载失败，降级使用模拟数据进行训练")
        return _fallback_training_data()

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.dropna(subset=['FTR', 'B365H', 'B365D', 'B365A'])
    print(f"    [Data] ✅ 成功加载 {len(full_df)} 场真实历史比赛数据！")
    
    X = []; y = []
    for _, row in full_df.iterrows():
        try:
            prob_h = 1 / float(row['B365H'])
            prob_d = 1 / float(row['B365D'])
            prob_a = 1 / float(row['B365A'])
            h_strength = prob_h / (prob_h + prob_a)
            a_strength = prob_a / (prob_h + prob_a)
            
            target_map = {'H': 0, 'D': 1, 'A': 2}
            target = target_map.get(str(row['FTR']).upper())
            
            if target is not None:
                features = [prob_h, prob_d, prob_a, h_strength, a_strength]
                X.append(features)
                y.append(target)
        except:
            continue
    return np.array(X), np.array(y)

def _fallback_training_data(n=1000):
    np.random.seed(42); X = []; y = []
    for _ in range(n):
        ph = np.random.uniform(0.2, 0.8); pd = np.random.uniform(0.15, 0.35); pa = max(0.01, 1 - ph - pd)
        X.append([ph, pd, pa, ph/(ph+pa), pa/(ph+pa)])
        y.append(np.random.choice([0, 1, 2], p=[ph, pd, pa]))
    return np.array(X), np.array(y)

def _build_ml_features(match, match_odds):
    try:
        if match_odds and "bookmakers" in match_odds:
            ho=[]; do2=[]; ao=[]
            for bk in match_odds["bookmakers"]:
                h2h = bk.get("markets", {}).get("h2h", {})
                if "Home" in h2h: ho.append(h2h["Home"])
                if "Draw" in h2h: do2.append(h2h["Draw"])
                if "Away" in h2h: ao.append(h2h["Away"])
            if ho and do2 and ao:
                ah = sum(ho)/len(ho); ad = sum(do2)/len(do2); aa = sum(ao)/len(ao)
                prob_h = 1 / ah; prob_d = 1 / ad; prob_a = 1 / aa
            else: raise Exception("Odds incomplete")
        else:
            hw = float(match.get("home_stats", {}).get("wins", 4)); hp = float(match.get("home_stats", {}).get("played", 10))
            aw = float(match.get("away_stats", {}).get("wins", 4)); ap = float(match.get("away_stats", {}).get("played", 10))
            prob_h = max(0.2, min(0.75, (hw/max(1, hp)) * 1.1))
            prob_a = max(0.2, min(0.75, (aw/max(1, ap)) * 0.9))
            prob_d = max(0.15, 1 - prob_h - prob_a)
            
        h_strength = prob_h / (prob_h + prob_a); a_strength = prob_a / (prob_h + prob_a)
        return [prob_h, prob_d, prob_a, h_strength, a_strength]
    except:
        return [0.45, 0.25, 0.30, 0.6, 0.4]

# ==========================================
# 1. 机器学习引擎 (已剔除易过拟合的 GB 模型)
# ==========================================
class MLPredictorBase:
    def __init__(self, name):
        self.model = None; self.scaler = None; self.trained = False; self.name = name
    def train(self):
        if not HAS_SK: return
        X, y = fetch_real_historical_data()
        self.scaler = StandardScaler(); X = self.scaler.fit_transform(X)
        self._init_model()
        self.model.fit(X, y); self.trained = True
    def predict(self, match, match_odds):
        if not self.trained: self.train()
        if not self.model: return {"home_win": 40, "draw": 30, "away_win": 30, "model": f"{self.name}-fallback"}
        feat = _build_ml_features(match, match_odds)
        X = self.scaler.transform([feat])
        proba = list(self.model.predict_proba(X)[0])
        
        # 防爆冷平滑器：防止机器学习给出极端高胜率
        if max(proba) > 0.75:
            p_max = max(proba)
            excess = p_max - 0.75
            proba = [0.75 if p == p_max else p + (excess / 2.0) for p in proba]

        return {"home_win": round(proba[0] * 100, 1), "draw": round(proba[1] * 100, 1), "away_win": round(proba[2] * 100, 1), "model": self.name}

class RandomForestModel(MLPredictorBase):
    def __init__(self): super().__init__("RandomForest")
    def _init_model(self): self.model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)

class NeuralNetModel(MLPredictorBase):
    def __init__(self): super().__init__("NeuralNet")
    def _init_model(self): self.model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=42)

class LogisticModel(MLPredictorBase):
    def __init__(self): super().__init__("Logistic")
    def _init_model(self): self.model = LogisticRegression(C=0.5, max_iter=500, random_state=42)

# ==========================================
# 2. 高阶量化模块 (逆向、大球)
# ==========================================
class SmartMoneyDetector:
    def analyze(self, model_home_prob, model_away_prob, match_odds):
        if not match_odds or not match_odds.get("bookmakers"):
            return {"home_rlm_adj": 0, "away_rlm_adj": 0, "signal": "无赔率数据"}
        ho = []; ao = []
        for bk in match_odds["bookmakers"]:
            h2h = bk.get("markets", {}).get("h2h", {})
            if "Home" in h2h: ho.append(h2h["Home"])
            if "Away" in h2h: ao.append(h2h["Away"])
        if not ho or not ao: return {"home_rlm_adj": 0, "away_rlm_adj": 0, "signal": "正常"}
        avg_h = sum(ho)/len(ho); avg_a = sum(ao)/len(ao)
        implied_h = (1 / avg_h) * 100; implied_a = (1 / avg_a) * 100
        diff_h = implied_h - model_home_prob; diff_a = implied_a - model_away_prob
        
        signal = "正常"
        if diff_h > 12: signal = "⚠️ 机构防范主队 (聪明钱流入)"
        elif diff_a > 12: signal = "⚠️ 机构防范客队 (大热必死预警)"
        elif diff_h < -12: signal = "🚨 庄家诱盘主队 (逆向看衰)"
        elif diff_a < -12: signal = "🚨 庄家诱盘客队 (逆向看衰)"
        return {"home_rlm_adj": diff_h * 0.4, "away_rlm_adj": diff_a * 0.4, "signal": signal}

class BivariatePoissonModel:
    def predict(self, home_gf, home_ga, away_gf, away_ga):
        try: home_gf = float(home_gf or 1.3); home_ga = float(home_ga or 1.1); away_gf = float(away_gf or 1.1); away_ga = float(away_ga or 1.3)
        except: home_gf = 1.3; home_ga = 1.1; away_gf = 1.1; away_ga = 1.3
        he = max(0.4, min(home_gf * 1.05, 3.5)); ae = max(0.3, min(away_gf * 0.95, 3.0))
        rho = 0.2 if abs(he - ae) < 0.4 else 0.05 
        hw = 0; dr = 0; aw = 0; scores = []
        def pmf(k, l): return (l**k) * math.exp(-l) / math.factorial(k)
        for i in range(5):
            for j in range(5):
                p = pmf(i, he) * pmf(j, ae)
                if i == j: 
                    if i == 0: p *= (1 + he*ae*rho)
                    elif i == 1: p *= (1 - rho)
                if i > j: hw += p
                elif i == j: dr += p
                else: aw += p
                scores.append((i, j, p))
        scores.sort(key=lambda x: x[2], reverse=True); t = hw + dr + aw
        if t > 0: hw /= t; dr /= t; aw /= t
        return {"home_win": round(hw*100, 1), "draw": round(dr*100, 1), "away_win": round(aw*100, 1), "predicted_score": "%d-%d" % (scores[0][0], scores[0][1])}

class PaceTotalGoalsModel:
    def predict(self, hs, ast):
        try:
            h_gf = float(hs.get("avg_goals_for", 1.3)); h_ga = float(hs.get("avg_goals_against", 1.1))
            a_gf = float(ast.get("avg_goals_for", 1.1)); a_ga = float(ast.get("avg_goals_against", 1.3))
            h_cs = float(hs.get("clean_sheets", 2)) / max(1, float(hs.get("played", 10)))
            a_cs = float(ast.get("clean_sheets", 2)) / max(1, float(ast.get("played", 10)))
        except: return {"over_2_5": 50.0, "expected_total": 2.5, "pace_rating": "中等"}
        exp_total = (h_gf * a_ga + a_gf * h_ga)
        pace_multiplier = 1.0 + (0.3 - ((h_cs + a_cs) / 2))
        final_exp = exp_total * pace_multiplier
        over_prob = 1.0 - (math.exp(-final_exp) * (1 + final_exp + (final_exp**2)/2))
        return {"over_2_5": max(15.0, min(85.0, over_prob * 100)), "expected_total": round(final_exp, 2), "pace_rating": "极快" if final_exp > 3.0 else ("慢" if final_exp < 2.0 else "中等")}

# ==========================================
# 3. 经典统计与概率引擎 (已剔除 ELO 和 MC)
# ==========================================
class PoissonModel:
    def predict(self, home_gf, home_ga, away_gf, away_ga, league_avg=1.35):
        try: home_gf = float(home_gf or 1.3); home_ga = float(home_ga or 1.1); away_gf = float(away_gf or 1.1); away_ga = float(away_ga or 1.3)
        except: home_gf = 1.3; home_ga = 1.1; away_gf = 1.1; away_ga = 1.3
        he = max(0.3, min((home_gf / league_avg) * 1.10 * (away_ga / league_avg) * league_avg, 4.5))
        ae = max(0.2, min((away_gf / league_avg) * (home_ga / league_avg) * 0.90 * league_avg, 4.0))
        def pmf(k, l): return (l**k) * math.exp(-l) / math.factorial(k)
        hw = 0; dr = 0; aw = 0; bt = 0; scores = []
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
        return {"home_win": round(hw*100, 1), "draw": round(dr*100, 1), "away_win": round(aw*100, 1), "predicted_score": "%d-%d" % (scores[0][0], scores[0][1]), "home_xg": round(he, 2), "away_xg": round(ae, 2), "btts": round(bt*100, 1), "top_scores": [{"score": "%d-%d" % (s[0], s[1]), "prob": round(s[2]*100, 1)} for s in scores[:5]]}

class DixonColesModel:
    def predict(self, home_gf, home_ga, away_gf, away_ga):
        try: home_gf = float(home_gf or 1.3); away_gf = float(away_gf or 1.1)
        except: home_gf = 1.3; away_gf = 1.1
        he = max(0.3, min(home_gf * 1.05, 4.0)); ae = max(0.2, min(away_gf * 0.95, 3.5))
        rho = -0.15 if abs(he - ae) < 0.5 else -0.05
        def pmf(k, l): return (l**k) * math.exp(-l) / math.factorial(k)
        def tau(i, j, lam, mu, r):
            if i == 0 and j == 0: return 1 - lam * mu * r
            elif i == 0 and j == 1: return 1 + lam * r
            elif i == 1 and j == 0: return 1 + mu * r
            elif i == 1 and j == 1: return 1 - r
            return 1
        hw = 0; dr = 0; aw = 0
        for i in range(6):
            for j in range(6):
                p = max(0, tau(i, j, he, ae, rho) * pmf(i, he) * pmf(j, ae))
                if i > j: hw += p
                elif i == j: dr += p
                else: aw += p
        t = hw + dr + aw
        if t > 0: hw /= t; dr /= t; aw /= t
        return {"home_win": round(hw*100, 1), "draw": round(dr*100, 1), "away_win": round(aw*100, 1)}

class BradleyTerryModel:
    def predict(self, home_wins, home_total, away_wins, away_total):
        try: hw = int(home_wins or 0); ht = max(1, int(home_total or 1)); aw = int(away_wins or 0); at = max(1, int(away_total or 1))
        except: hw = 5; ht = 15; aw = 5; at = 15
        hp = max(0.1, hw / ht); ap = max(0.1, aw / at)
        h_str = hp / (hp + ap) * 1.10; a_str = ap / (hp + ap) * 0.90
        dr = 0.25; h = h_str * (1 - dr); a = a_str * (1 - dr); t = h + dr + a
        return {"home_win": round(h/t*100, 1), "draw": round(dr/t*100, 1), "away_win": round(a/t*100, 1)}

class BayesianModel:
    def predict(self, hw, hd, hl, aw, ad, al):
        try: hw=int(hw or 5); hd=int(hd or 3); hl=int(hl or 3); aw=int(aw or 5); ad=int(ad or 3); al=int(al or 3)
        except: hw=5; hd=3; hl=3; aw=5; ad=3; al=3
        ph = 1.2 + hw * 1.0 + al * 0.6; pd = 1.0 + hd * 0.8 + ad * 0.8; pa = 0.8 + aw * 1.0 + hl * 0.6; t = ph + pd + pa
        return {"home_win": round(ph/t*100, 1), "draw": round(pd/t*100, 1), "away_win": round(pa/t*100, 1)}

class FormModel:
    def analyze(self, form):
        if not form: return {"score": 50, "trend": "unknown"}
        momentum = sum((3 if c=="W" else 1 if c=="D" else 0) * max(0.2, 1.0 - i*0.15) for i, c in enumerate(reversed(form)))
        tw = sum(3 * max(0.2, 1.0 - i*0.15) for i in range(len(form)))
        return {"score": round((momentum / tw) * 100 if tw > 0 else 50, 1)}

class OddsAnalyzer:
    def analyze_market(self, bookmakers):
        if not bookmakers: return {}
        ho = []; do2 = []; ao = []
        for bk in bookmakers:
            h2h = bk.get("markets", {}).get("h2h", {})
            if "Home" in h2h: ho.append(h2h["Home"])
            if "Draw" in h2h: do2.append(h2h["Draw"])
            if "Away" in h2h: ao.append(h2h["Away"])
        if not ho: return {}
        ah = sum(ho)/len(ho); ad = sum(do2)/len(do2); aa = sum(ao)/len(ao); mg = 1/ah + 1/ad + 1/aa - 1
        return {"avg_home_odds": round(ah, 2), "avg_draw_odds": round(ad, 2), "avg_away_odds": round(aa, 2)}

# ==========================================
# 4. 终极融合中枢 (纯净 11 模型矩阵)
# ==========================================
class EnsemblePredictor:
    def __init__(self):
        self.poisson = PoissonModel()
        self.dixon = DixonColesModel()
        self.bt = BradleyTerryModel()
        self.bayes = BayesianModel()
        self.form = FormModel()
        self.odds = OddsAnalyzer()
        
        self.bivariate = BivariatePoissonModel()
        self.smart_money = SmartMoneyDetector()
        self.pace_totals = PaceTotalGoalsModel()
        
        self.rf = RandomForestModel()
        self.nn = NeuralNetModel()
        self.lr = LogisticModel()
        
        print("[Models] 正在初始化量化分析引擎(11核心模型纯净版)...")
        self.rf.train(); self.nn.train(); self.lr.train()
        print("[Models] 所有模型就绪！")

    def predict(self, match, odds_data=None):
        hs = match.get("home_stats", {}); ast = match.get("away_stats", {}); h2h = match.get("h2h", [])
        
        poi = self.poisson.predict(hs.get("avg_goals_for"), hs.get("avg_goals_against"), ast.get("avg_goals_for"), ast.get("avg_goals_against"))
        biv = self.bivariate.predict(hs.get("avg_goals_for"), hs.get("avg_goals_against"), ast.get("avg_goals_for"), ast.get("avg_goals_against"))
        dc = self.dixon.predict(hs.get("avg_goals_for"), hs.get("avg_goals_against"), ast.get("avg_goals_for"), ast.get("avg_goals_against"))
        bt = self.bt.predict(hs.get("wins"), hs.get("played"), ast.get("wins"), ast.get("played"))
        bay = self.bayes.predict(hs.get("wins"), hs.get("draws"), hs.get("losses"), ast.get("wins"), ast.get("draws"), ast.get("losses"))
        hf = self.form.analyze(hs.get("form", "")); af = self.form.analyze(ast.get("form", ""))
        pace = self.pace_totals.predict(hs, ast) 
        
        rf = self.rf.predict(match, odds_data)
        nn = self.nn.predict(match, odds_data)
        lr = self.lr.predict(match, odds_data)
        
        oa = {}
        if odds_data and odds_data.get("bookmakers"):
            oa = self.odds.analyze_market(odds_data["bookmakers"])
            
        w = {
            "poisson": 0.12, "bivariate": 0.10, "dixon": 0.10, 
            "bt": 0.08, "bayes": 0.06, 
            "rf": 0.24, "nn": 0.16, "lr": 0.14
        }
        
        models_list = [
            ("poisson", poi), ("bivariate", biv), ("dixon", dc), 
            ("bt", bt), ("bayes", bay), 
            ("rf", rf), ("nn", nn), ("lr", lr)
        ]
        
        hp = 0; dp = 0; ap = 0
        for name, pred in models_list:
            wt = w.get(name, 0)
            hp += pred.get("home_win", 33) * wt
            dp += pred.get("draw", 33) * wt
            ap += pred.get("away_win", 33) * wt
            
        fd = hf["score"] - af["score"]
        hp += (fd * 0.08); ap -= (fd * 0.08)
        
        sm_data = self.smart_money.analyze(hp, ap, odds_data)
        hp += sm_data["home_rlm_adj"]; ap += sm_data["away_rlm_adj"]
        
        t = hp + dp + ap
        if t > 0: hp = round(hp/t*100, 1); dp = round(dp/t*100, 1); ap = round(100-hp-dp, 1)
        
        agree_h = sum(1 for _, p in models_list if p.get("home_win", 0) > max(p.get("draw", 0), p.get("away_win", 0)))
        agree_a = sum(1 for _, p in models_list if p.get("away_win", 0) > max(p.get("home_win", 0), p.get("draw", 0)))
        consensus = max(agree_h, agree_a)
        
        cf = min(95, max(30, 30 + consensus * 5 + (12 if max(hp, dp, ap) > 60 else 6)))
        
        return {
            "home_win_pct": hp, "draw_pct": dp, "away_win_pct": ap, 
            "predicted_score": poi["predicted_score"], "confidence": cf, 
            
            "poisson": poi, "dixon_coles": dc, "bradley_terry": bt,
            "bayesian": bay, "random_forest": rf, "neural_net": nn, "logistic": lr,
            "odds": oa, "home_form": hf, "away_form": af,
            
            # 兼容空壳，防前端报错
            "elo": {"home_win": "-", "draw": "-", "away_win": "-", "elo_diff": "已停用"},
            "monte_carlo": {"top_scores": [{"score": "已停用", "prob": "-"}]},
            "gradient_boost": {"home_win": "-", "draw": "-", "away_win": "-"},
            
            "smart_money_signal": sm_data["signal"], 
            "over_2_5": pace["over_2_5"], "btts": poi.get("btts", 50),
            "pace_rating": pace["pace_rating"],
            "expected_total_goals": pace["expected_total"],
            "model_consensus": consensus, "total_models": len(models_list)
        }
