import math
import random
import time
import io
import numpy as np
import pandas as pd
import requests
from collections import defaultdict

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
        print("    [Data] ⚠️ 下载真实数据失败或超时，降级使用模拟数据进行训练")
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
# 机器学习预测基类与子类
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
        proba = self.model.predict_proba(X)[0]
        return {"home_win": round(proba[0] * 100, 1), "draw": round(proba[1] * 100, 1), "away_win": round(proba[2] * 100, 1), "model": self.name}

class RandomForestModel(MLPredictorBase):
    def __init__(self): super().__init__("RandomForest")
    def _init_model(self): self.model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)

class GradientBoostModel(MLPredictorBase):
    def __init__(self): super().__init__("GradientBoost")
    def _init_model(self): self.model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)

class NeuralNetModel(MLPredictorBase):
    def __init__(self): super().__init__("NeuralNet")
    def _init_model(self): self.model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=42)

class LogisticModel(MLPredictorBase):
    def __init__(self): super().__init__("Logistic")
    def _init_model(self): self.model = LogisticRegression(C=0.5, max_iter=500, random_state=42)

# ==========================================
# 高阶量化模块：逆向追踪、防爆冷泊松、大球节奏
# ==========================================

class SmartMoneyDetector:
    """逆向分析与聪明钱追踪模型 (RLM)"""
    def analyze(self, model_home_prob, model_away_prob, match_odds):
        if not match_odds or not match_odds.get("bookmakers"):
            return {"home_rlm_adj": 0, "away_rlm_adj": 0, "signal": "无赔率数据"}
            
        ho = []; ao = []
        for bk in match_odds["bookmakers"]:
            h2h = bk.get("markets", {}).get("h2h", {})
            if "Home" in h2h: ho.append(h2h["Home"])
            if "Away" in h2h: ao.append(h2h["Away"])
            
        if not ho or not ao:
            return {"home_rlm_adj": 0, "away_rlm_adj": 0, "signal": "正常"}
            
        avg_h = sum(ho)/len(ho); avg_a = sum(ao)/len(ao)
        implied_h = (1 / avg_h) * 100; implied_a = (1 / avg_a) * 100
        
        diff_h = implied_h - model_home_prob
        diff_a = implied_a - model_away_prob
        
        signal = "正常"
        if diff_h > 12: signal = "⚠️ 机构防范主队 (聪明钱流入)"
        elif diff_a > 12: signal = "⚠️ 机构防范客队 (大热必死预警)"
        elif diff_h < -12: signal = "🚨 庄家诱盘主队 (逆向看衰)"
        elif diff_a < -12: signal = "🚨 庄家诱盘客队 (逆向看衰)"
        
        return {
            "home_rlm_adj": diff_h * 0.4, 
            "away_rlm_adj": diff_a * 0.4,
            "signal": signal,
            "implied_h": round(implied_h, 1),
            "implied_a": round(implied_a, 1)
        }

class BivariatePoissonModel:
    """双变量泊松防爆冷模型 (强化低比分平局)"""
    def predict(self, home_gf, home_ga, away_gf, away_ga):
        try:
            home_gf = float(home_gf or 1.3); home_ga = float(home_ga or 1.1)
            away_gf = float(away_gf or 1.1); away_ga = float(away_ga or 1.3)
        except:
            home_gf = 1.3; home_ga = 1.1; away_gf = 1.1; away_ga = 1.3
            
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
                
        scores.sort(key=lambda x: x[2], reverse=True)
        t = hw + dr + aw
        if t > 0: hw /= t; dr /= t; aw /= t
        
        return {
            "home_win": round(hw*100, 1), "draw": round(dr*100, 1), "away_win": round(aw*100, 1),
            "predicted_score": "%d-%d" % (scores[0][0], scores[0][1]),
            "rho_factor": rho
        }

class PaceTotalGoalsModel:
    """独立的大小球节奏预测模型"""
    def predict(self, hs, ast):
        try:
            h_gf = float(hs.get("avg_goals_for", 1.3)); h_ga = float(hs.get("avg_goals_against", 1.1))
            a_gf = float(ast.get("avg_goals_for", 1.1)); a_ga = float(ast.get("avg_goals_against", 1.3))
            h_cs = float(hs.get("clean_sheets", 2)) / max(1, float(hs.get("played", 10)))
            a_cs = float(ast.get("clean_sheets", 2)) / max(1, float(ast.get("played", 10)))
        except:
            return {"over_2_5": 50.0, "expected_total": 2.5, "pace_rating": "中等"}

        exp_total = (h_gf * a_ga + a_gf * h_ga)
        defense_factor = (h_cs + a_cs) / 2
        pace_multiplier = 1.0 + (0.3 - defense_factor) 
        
        final_exp = exp_total * pace_multiplier
        over_prob = 1.0 - (math.exp(-final_exp) * (1 + final_exp + (final_exp**2)/2))
        over_pct = max(15.0, min(85.0, over_prob * 100))
        
        pace = "极快(互捅局)" if final_exp > 3.0 else ("慢(防守肉搏)" if final_exp < 2.0 else "中等")
        return {"over_2_5": round(over_pct, 1), "expected_total": round(final_exp, 2), "pace_rating": pace}

# ==========================================
# 传统统计与概率模型 (保持不变)
# ==========================================

class PoissonModel:
    def predict(self, home_gf, home_ga, away_gf, away_ga, league_avg=1.35):
        try:
            home_gf = float(home_gf or 1.3); home_ga = float(home_ga or 1.1)
            away_gf = float(away_gf or 1.1); away_ga = float(away_ga or 1.3)
        except:
            home_gf = 1.3; home_ga = 1.1; away_gf = 1.1; away_ga = 1.3
            
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
                
        scores.sort(key=lambda x: x[2], reverse=True)
        t = hw + dr + aw
        if t > 0: hw /= t; dr /= t; aw /= t
        
        return {
            "home_win": round(hw * 100, 1), "draw": round(dr * 100, 1), "away_win": round(aw * 100, 1), 
            "predicted_score": "%d-%d" % (scores[0][0], scores[0][1]),
            "home_xg": round(he, 2), "away_xg": round(ae, 2),
            "btts": round(bt * 100, 1), 
            "top_scores": [{"score": "%d-%d" % (s[0], s[1]), "prob": round(s[2] * 100, 1)} for s in scores[:5]]
        }

class EloModel:
    def __init__(self):
        self.ratings = defaultdict(lambda: 1500); self.k = 30
    def update(self, h, a, hg, ag):
        rh = self.ratings[h]; ra = self.ratings[a]
        eh = 1 / (1 + 10 ** ((ra - rh) / 400)); ea = 1 - eh
        sh = 1 if hg > ag else (0.5 if hg == ag else 0); sa = 1 - sh
        mov = math.log(abs(hg - ag) + 2) if hg != ag else 1.0
        self.ratings[h] = rh + self.k * mov * (sh - eh)
        self.ratings[a] = ra + self.k * mov * (sa - ea)
    def predict(self, h, a):
        rh = self.ratings[h] + 60; ra = self.ratings[a]
        eh = 1 / (1 + 10 ** ((ra - rh) / 400))
        df = 0.28 if abs(rh - ra) < 100 else 0.22
        hw = eh * (1 - df / 2); aw = (1 - eh) * (1 - df / 2); dr = df
        return {"home_win": round(hw * 100, 1), "draw": round(dr * 100, 1), "away_win": round(aw * 100, 1), "elo_diff": round(rh - ra, 1)}
    def load_h2h(self, records):
        for r in reversed(records):
            try:
                p = r["score"].split("-")
                self.update(r["home"], r["away"], int(p[0]), int(p[1]))
            except: pass

class MonteCarloModel:
    def simulate(self, home_gf, home_ga, away_gf, away_ga, n=10000):
        try: home_gf = float(home_gf or 1.3); away_gf = float(away_gf or 1.1)
        except: home_gf = 1.3; away_gf = 1.1
        he = max(0.3, min(home_gf * 1.05, 4.0)); ae = max(0.2, min(away_gf * 0.95, 3.5))
        np.random.seed(int(time.time() % 1000))
        hg = np.random.poisson(he, n); ag = np.random.poisson(ae, n)
        hw = np.sum(hg > ag) / n; dr = np.sum(hg == ag) / n; aw = np.sum(hg < ag) / n
        from collections import Counter
        sc = Counter(zip(hg.tolist(), ag.tolist())).most_common(5)
        return {"home_win": round(hw*100, 1), "draw": round(dr*100, 1), "away_win": round(aw*100, 1), "top_scores": [{"score": f"{s[0][0]}-{s[0][1]}", "prob": round(s[1]/n*100, 1)} for s in sc]}

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
        ah = sum(ho)/len(ho); ad = sum(do2)/len(do2); aa = sum(ao)/len(ao)
        mg = 1/ah + 1/ad + 1/aa - 1
        return {"avg_home_odds": round(ah, 2), "avg_draw_odds": round(ad, 2), "avg_away_odds": round(aa, 2), "implied_home": round(1/ah/(1+mg)*100, 1), "implied_draw": round(1/ad/(1+mg)*100, 1), "implied_away": round(100-(1/ah/(1+mg)*100)-(1/ad/(1+mg)*100), 1)}

# ==========================================
# 终极融合中枢
# ==========================================

class EnsemblePredictor:
    def __init__(self):
        self.poisson = PoissonModel()
        self.bivariate = BivariatePoissonModel() # 防爆冷
        self.elo = EloModel()
        self.mc = MonteCarloModel()
        self.form = FormModel()
        self.odds = OddsAnalyzer()
        self.smart_money = SmartMoneyDetector()  # 逆向分析
        self.pace_totals = PaceTotalGoalsModel() # 大球节奏
        
        self.rf = RandomForestModel()
        self.gb = GradientBoostModel()
        self.nn = NeuralNetModel()
        
        print("[Models] 正在初始化量化分析引擎(含防爆冷与逆向追踪模块)...")
        self.rf.train(); self.gb.train(); self.nn.train()
        print("[Models] 所有模型就绪！")

    def predict(self, match, odds_data=None):
        hs = match.get("home_stats", {}); ast = match.get("away_stats", {}); h2h = match.get("h2h", [])
        home = match["home_team"]; away = match["away_team"]
        
        poi = self.poisson.predict(hs.get("avg_goals_for"), hs.get("avg_goals_against"), ast.get("avg_goals_for"), ast.get("avg_goals_against"))
        biv = self.bivariate.predict(hs.get("avg_goals_for"), hs.get("avg_goals_against"), ast.get("avg_goals_for"), ast.get("avg_goals_against"))
        if h2h: self.elo.load_h2h(h2h)
        elo = self.elo.predict(home, away)
        mc = self.mc.simulate(hs.get("avg_goals_for"), hs.get("avg_goals_against"), ast.get("avg_goals_for"), ast.get("avg_goals_against"))
        hf = self.form.analyze(hs.get("form", "")); af = self.form.analyze(ast.get("form", ""))
        
        pace = self.pace_totals.predict(hs, ast) # 独立大球
        
        rf = self.rf.predict(match, odds_data)
        gb = self.gb.predict(match, odds_data)
        nn = self.nn.predict(match, odds_data)
        
        oa = {}
        if odds_data and odds_data.get("bookmakers"):
            oa = self.odds.analyze_market(odds_data["bookmakers"])
            
        w = {"poisson": 0.15, "bivariate": 0.15, "elo": 0.20, "mc": 0.10, "rf": 0.15, "gb": 0.15, "nn": 0.10}
        models = [("poisson", poi), ("bivariate", biv), ("elo", elo), ("mc", mc), ("rf", rf), ("gb", gb), ("nn", nn)]
        
        hp = 0; dp = 0; ap = 0
        for name, pred in models:
            wt = w.get(name, 0)
            hp += pred.get("home_win", 33) * wt
            dp += pred.get("draw", 33) * wt
            ap += pred.get("away_win", 33) * wt
            
        fd = hf["score"] - af["score"]
        hp += (fd * 0.08); ap -= (fd * 0.08)
        
        # 🚨 逆向追踪：聪明钱预警权重修正
        sm_data = self.smart_money.analyze(hp, ap, odds_data)
        hp += sm_data["home_rlm_adj"]
        ap += sm_data["away_rlm_adj"]
        
        t = hp + dp + ap
        if t > 0: hp = round(hp/t*100, 1); dp = round(dp/t*100, 1); ap = round(100-hp-dp, 1)
        
        agree_h = sum(1 for _, p in models if p.get("home_win", 0) > max(p.get("draw", 0), p.get("away_win", 0)))
        agree_a = sum(1 for _, p in models if p.get("away_win", 0) > max(p.get("home_win", 0), p.get("draw", 0)))
        consensus = max(agree_h, agree_a)
        
        cf = min(95, max(30, 30 + consensus * 5 + (12 if max(hp, dp, ap) > 60 else 6)))
        
        return {
            "home_win_pct": hp, "draw_pct": dp, "away_win_pct": ap, 
            "predicted_score": poi["predicted_score"], "confidence": cf, 
            "poisson": poi, "elo": elo, "monte_carlo": mc, "odds": oa,
            "smart_money_signal": sm_data["signal"], 
            "over_2_5": pace["over_2_5"], "btts": poi.get("btts", 50),
            "pace_rating": pace["pace_rating"],
            "expected_total_goals": pace["expected_total"],
            "model_consensus": consensus, "total_models": len(models)
        }
