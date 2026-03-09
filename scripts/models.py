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
except ImportError:
    HAS_SK = False
    print("[WARN] sklearn 未安装，将使用降级统计模型")

# ==================== 1. 历史数据与特征工程 ====================
def fetch_real_historical_data():
    """从权威数据库抓取五大联赛真实历史赔率用于模型训练"""
    print("    [Data] 正在从 football-data.co.uk 下载历史训练数据...")
    leagues = ['E0', 'SP1', 'I1', 'D1', 'F1']
    seasons = ['2324', '2223', '2122'] 
    dfs = []
    headers = {"User-Agent": "Mozilla/5.0"}
    
    for season in seasons:
        for league in leagues:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
            try:
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code == 200:
                    df = pd.read_csv(io.StringIO(r.text), on_bad_lines='skip')
                    dfs.append(df)
            except Exception as e:
                continue
                
    if not dfs:
        return _fallback_training_data()
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.dropna(subset=['FTR', 'B365H', 'B365D', 'B365A'])
    
    X = []
    y = []
    for _, row in full_df.iterrows():
        try:
            prob_h = 1 / float(row['B365H'])
            prob_d = 1 / float(row['B365D'])
            prob_a = 1 / float(row['B365A'])
            h_str = prob_h / (prob_h + prob_a)
            a_str = prob_a / (prob_h + prob_a)
            
            target_map = {'H': 0, 'D': 1, 'A': 2}
            target = target_map.get(str(row['FTR']).upper())
            
            if target is not None:
                X.append([prob_h, prob_d, prob_a, h_str, a_str])
                y.append(target)
        except Exception:
            continue
            
    return np.array(X), np.array(y)

def _fallback_training_data(n=1000):
    """当网络阻断时，生成符合博彩分布的兜底数据"""
    np.random.seed(42)
    X = []
    y = []
    for _ in range(n):
        ph = np.random.uniform(0.2, 0.8)
        pd = np.random.uniform(0.15, 0.35)
        pa = max(0.01, 1 - ph - pd)
        X.append([ph, pd, pa, ph/(ph+pa), pa/(ph+pa)])
        y.append(np.random.choice([0, 1, 2], p=[ph, pd, pa]))
    return np.array(X), np.array(y)

def _build_ml_features(match, match_odds):
    """提取单场比赛的机器学习特征"""
    try:
        if match_odds and "bookmakers" in match_odds:
            ho = []
            do2 = []
            ao = []
            for bk in match_odds["bookmakers"]:
                h2h = bk.get("markets", {}).get("h2h", {})
                if "Home" in h2h: ho.append(h2h["Home"])
                if "Draw" in h2h: do2.append(h2h["Draw"])
                if "Away" in h2h: ao.append(h2h["Away"])
                
            if ho and do2 and ao:
                prob_h = 1 / (sum(ho) / len(ho))
                prob_d = 1 / (sum(do2) / len(do2))
                prob_a = 1 / (sum(ao) / len(ao))
            else:
                raise Exception("赔率数据不全")
        else:
            # 降级：使用基本面换算赔率特征
            hw = float(match.get("home_stats", {}).get("wins", 4))
            hp = float(match.get("home_stats", {}).get("played", 10))
            aw = float(match.get("away_stats", {}).get("wins", 4))
            ap = float(match.get("away_stats", {}).get("played", 10))
            
            prob_h = max(0.2, min(0.75, (hw / max(1, hp)) * 1.1))
            prob_a = max(0.2, min(0.75, (aw / max(1, ap)) * 0.9))
            prob_d = max(0.15, 1 - prob_h - prob_a)
            
        return [prob_h, prob_d, prob_a, prob_h/(prob_h+prob_a), prob_a/(prob_h+prob_a)]
    except Exception:
        return [0.45, 0.25, 0.30, 0.6, 0.4]

# ==================== 2. 机器学习模型基类 ====================
class MLPredictorBase:
    def __init__(self, name):
        self.model = None
        self.scaler = None
        self.trained = False
        self.name = name
        
    def train(self):
        if not HAS_SK: return
        X, y = fetch_real_historical_data()
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        self._init_model()
        self.model.fit(X, y)
        self.trained = True
        
    def predict(self, match, match_odds):
        if not self.trained or not self.model: 
            return {"home_win": 40, "draw": 30, "away_win": 30}
            
        X = self.scaler.transform([_build_ml_features(match, match_odds)])
        proba = list(self.model.predict_proba(X)[0])
        
        # 极值平滑处理，防止模型过度自信
        if max(proba) > 0.75: 
            proba = [0.75 if p == max(proba) else p + (max(proba)-0.75)/2.0 for p in proba]
            
        return {
            "home_win": round(proba[0]*100, 1), 
            "draw": round(proba[1]*100, 1), 
            "away_win": round(proba[2]*100, 1)
        }

class RandomForestModel(MLPredictorBase):
    def __init__(self): super().__init__("RandomForest")
    def _init_model(self): self.model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)

class NeuralNetModel(MLPredictorBase):
    def __init__(self): super().__init__("NeuralNet")
    def _init_model(self): self.model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=42)

class LogisticModel(MLPredictorBase):
    def __init__(self): super().__init__("Logistic")
    def _init_model(self): self.model = LogisticRegression(C=0.5, max_iter=500, random_state=42)

# ==================== 3. 统计学与风控模型 ====================
class SmartMoneyDetector:
    def analyze(self, model_h, model_a, match_odds):
        if not match_odds or not match_odds.get("bookmakers"): 
            return {"home_rlm_adj": 0, "away_rlm_adj": 0, "signal": "正常"}
            
        ho = []
        ao = []
        for bk in match_odds["bookmakers"]:
            h2h = bk.get("markets", {}).get("h2h", {})
            if "Home" in h2h: ho.append(h2h["Home"])
            if "Away" in h2h: ao.append(h2h["Away"])
            
        if not ho or not ao: 
            return {"home_rlm_adj": 0, "away_rlm_adj": 0, "signal": "正常"}
            
        diff_h = (1 / (sum(ho)/len(ho)) * 100) - model_h
        diff_a = (1 / (sum(ao)/len(ao)) * 100) - model_a
        
        signal = "正常"
        if diff_h > 12: signal = "⚠️ 机构防范主队"
        elif diff_a > 12: signal = "⚠️ 机构防范客队"
        elif diff_h < -12: signal = "🚨 庄家诱盘主队"
        elif diff_a < -12: signal = "🚨 庄家诱盘客队"
        
        return {"home_rlm_adj": diff_h * 0.4, "away_rlm_adj": diff_a * 0.4, "signal": signal}

class BivariatePoissonModel:
    def predict(self, home_gf, home_ga, away_gf, away_ga):
        he = max(0.4, min(float(home_gf or 1.3) * 1.05, 3.5))
        ae = max(0.3, min(float(away_gf or 1.1) * 0.95, 3.0))
        rho = 0.2 if abs(he - ae) < 0.4 else 0.05 
        
        hw = 0.0
        dr = 0.0
        aw = 0.0
        scores = []
        
        def pmf(k, l): 
            return (l**k) * math.exp(-l) / math.factorial(k)
            
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
                
        scores.sort(key=lambda x: x[2], reverse=True)
        t = hw + dr + aw
        return {
            "home_win": round(hw/t*100, 1), 
            "draw": round(dr/t*100, 1), 
            "away_win": round(aw/t*100, 1), 
            "predicted_score": f"{scores[0][0]}-{scores[0][1]}"
        }

class PaceTotalGoalsModel:
    def predict(self, hs, ast):
        exp_total = float(hs.get("avg_goals_for", 1.3)) * float(ast.get("avg_goals_against", 1.3)) + \
                    float(ast.get("avg_goals_for", 1.1)) * float(hs.get("avg_goals_against", 1.1))
        
        h_cs = float(hs.get("clean_sheets", 2)) / max(1, float(hs.get("played", 10)))
        a_cs = float(ast.get("clean_sheets", 2)) / max(1, float(ast.get("played", 10)))
        
        final_exp = exp_total * (1.0 + (0.3 - ((h_cs + a_cs) / 2)))
        over_prob = 1.0 - (math.exp(-final_exp) * (1 + final_exp + (final_exp**2)/2))
        
        rating = "中等"
        if final_exp > 3.0: rating = "极快"
        elif final_exp < 2.0: rating = "慢"
        
        return {
            "over_2_5": max(15.0, min(85.0, over_prob * 100)), 
            "expected_total": round(final_exp, 2), 
            "pace_rating": rating
        }

class PoissonModel:
    def predict(self, home_gf, home_ga, away_gf, away_ga, league_avg=1.35):
        he = max(0.3, min((float(home_gf or 1.3) / league_avg) * 1.10 * (float(away_ga or 1.3) / league_avg) * league_avg, 4.5))
        ae = max(0.2, min((float(away_gf or 1.1) / league_avg) * (float(home_ga or 1.1) / league_avg) * 0.90 * league_avg, 4.0))
        
        def pmf(k, l): 
            return (l**k) * math.exp(-l) / math.factorial(k)
            
        hw = 0.0
        dr = 0.0
        aw = 0.0
        bt = 0.0
        scores = []
        
        for i in range(7):
            for j in range(7):
                p = pmf(i, he) * pmf(j, ae)
                if i > j: hw += p
                elif i == j: dr += p
                else: aw += p
                if i > 0 and j > 0: bt += p
                scores.append((i, j, p))
                
        scores.sort(key=lambda x: x[2], reverse=True)
        t = hw + dr + aw
        return {
            "home_win": round(hw/t*100, 1), 
            "draw": round(dr/t*100, 1), 
            "away_win": round(aw/t*100, 1), 
            "predicted_score": f"{scores[0][0]}-{scores[0][1]}", 
            "btts": round(bt*100, 1)
        }

class DixonColesModel:
    def predict(self, home_gf, home_ga, away_gf, away_ga):
        he = max(0.3, min(float(home_gf or 1.3) * 1.05, 4.0))
        ae = max(0.2, min(float(away_gf or 1.1) * 0.95, 3.5))
        rho = -0.15 if abs(he - ae) < 0.5 else -0.05
        
        def pmf(k, l): 
            return (l**k) * math.exp(-l) / math.factorial(k)
            
        def tau(i, j, l, m, r): 
            if i==0 and j==0: return 1 - l*m*r
            if i==0 and j==1: return 1 + l*r
            if i==1 and j==0: return 1 + m*r
            if i==1 and j==1: return 1 - r
            return 1
            
        hw = 0.0
        dr = 0.0
        aw = 0.0
        for i in range(6):
            for j in range(6):
                p = max(0, tau(i, j, he, ae, rho) * pmf(i, he) * pmf(j, ae))
                if i > j: hw += p
                elif i == j: dr += p
                else: aw += p
        t = hw + dr + aw
        return {"home_win": round(hw/t*100, 1), "draw": round(dr/t*100, 1), "away_win": round(aw/t*100, 1)}

class BradleyTerryModel:
    def predict(self, hw, ht, aw, at):
        hp = max(0.1, int(hw or 5) / max(1, int(ht or 15)))
        ap = max(0.1, int(aw or 5) / max(1, int(at or 15)))
        h_str = hp / (hp + ap) * 1.10
        a_str = ap / (hp + ap) * 0.90
        dr = 0.25
        h = h_str * (1 - dr)
        a = a_str * (1 - dr)
        t = h + dr + a
        return {"home_win": round(h/t*100, 1), "draw": round(dr/t*100, 1), "away_win": round(a/t*100, 1)}

class BayesianModel:
    def predict(self, hw, hd, hl, aw, ad, al):
        ph = 1.2 + int(hw or 5)*1.0 + int(al or 3)*0.6
        pd = 1.0 + int(hd or 3)*0.8 + int(ad or 3)*0.8
        pa = 0.8 + int(aw or 5)*1.0 + int(hl or 3)*0.6
        t = ph + pd + pa
        return {"home_win": round(ph/t*100, 1), "draw": round(pd/t*100, 1), "away_win": round(pa/t*100, 1)}

# ==================== 4. 预测系统总枢纽 ====================
class EnsemblePredictor:
    def __init__(self):
        self.poisson = PoissonModel()
        self.dixon = DixonColesModel()
        self.bt = BradleyTerryModel()
        self.bayes = BayesianModel()
        self.bivariate = BivariatePoissonModel()
        self.smart_money = SmartMoneyDetector()
        self.pace_totals = PaceTotalGoalsModel()
        self.rf = RandomForestModel()
        self.nn = NeuralNetModel()
        self.lr = LogisticModel()
        
        print("[Models] 正在初始化量化分析引擎(11核心模型矩阵)...")
        self.rf.train()
        self.nn.train()
        self.lr.train()

    def predict(self, match, odds_data=None):
        hs = match.get("home_stats", {})
        ast = match.get("away_stats", {})
        
        poi = self.poisson.predict(hs.get("avg_goals_for"), hs.get("avg_goals_against"), ast.get("avg_goals_for"), ast.get("avg_goals_against"))
        biv = self.bivariate.predict(hs.get("avg_goals_for"), hs.get("avg_goals_against"), ast.get("avg_goals_for"), ast.get("avg_goals_against"))
        dc = self.dixon.predict(hs.get("avg_goals_for"), hs.get("avg_goals_against"), ast.get("avg_goals_for"), ast.get("avg_goals_against"))
        bt = self.bt.predict(hs.get("wins"), hs.get("played"), ast.get("wins"), ast.get("played"))
        bay = self.bayes.predict(hs.get("wins"), hs.get("draws"), hs.get("losses"), ast.get("wins"), ast.get("draws"), ast.get("losses"))
        pace = self.pace_totals.predict(hs, ast) 
        rf = self.rf.predict(match, odds_data)
        nn = self.nn.predict(match, odds_data)
        lr = self.lr.predict(match, odds_data)
        
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
        
        hp = 0.0
        dp = 0.0
        ap = 0.0
        
        for name, pred in models_list:
            hp += pred.get("home_win", 33) * w[name]
            dp += pred.get("draw", 33) * w[name]
            ap += pred.get("away_win", 33) * w[name]
            
        sm_data = self.smart_money.analyze(hp, ap, odds_data)
        hp += sm_data["home_rlm_adj"]
        ap += sm_data["away_rlm_adj"]
        
        t = hp + dp + ap
        if t > 0: 
            hp = round(hp/t*100, 1)
            dp = round(dp/t*100, 1)
            ap = round(100-hp-dp, 1)
        
        agree_h = sum(1 for _, p in models_list if p.get("home_win", 0) > max(p.get("draw", 0), p.get("away_win", 0)))
        agree_a = sum(1 for _, p in models_list if p.get("away_win", 0) > max(p.get("home_win", 0), p.get("draw", 0)))
        
        base_conf = 30 + max(agree_h, agree_a) * 5
        if max(hp, dp, ap) > 60:
            base_conf += 12
        else:
            base_conf += 6
            
        return {
            "home_win_pct": hp, 
            "draw_pct": dp, 
            "away_win_pct": ap, 
            "predicted_score": poi["predicted_score"], 
            "confidence": min(95, max(30, base_conf)),
            "poisson": poi, 
            "random_forest": rf, 
            "neural_net": nn,
            "elo": {"elo_diff": "已停用"}, 
            "monte_carlo": {"top_scores": [{"score": "已停用", "prob": "-"}]}, 
            "gradient_boost": {"home_win": "-"},
            "smart_money_signal": sm_data["signal"], 
            "over_2_5": pace["over_2_5"], 
            "btts": poi.get("btts", 50),
            "model_consensus": max(agree_h, agree_a), 
            "total_models": len(models_list)
        }
