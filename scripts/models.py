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
    """
    从 football-data.co.uk 实时拉取欧洲主流联赛真实历史数据
    作为机器学习模型的高质量量化训练集
    """
    print("    [Data] 正在从 football-data.co.uk 下载真实历史比赛与赔率数据...")
    
    # E0:英超, SP1:西甲, I1:意甲, D1:德甲, F1:法甲
    leagues = ['E0', 'SP1', 'I1', 'D1', 'F1']
    seasons = ['2324', '2223', '2122'] # 拉取近几个完结赛季的数据保证稳定
    
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
    
    # 清洗：只保留包含全场赛果(FTR)和 B365 赔率的有效行
    full_df = full_df.dropna(subset=['FTR', 'B365H', 'B365D', 'B365A'])
    print(f"    [Data] ✅ 成功加载 {len(full_df)} 场真实历史比赛数据！")
    
    X = []; y = []
    for _, row in full_df.iterrows():
        try:
            # 提取庄家隐含概率 (特征)
            prob_h = 1 / float(row['B365H'])
            prob_d = 1 / float(row['B365D'])
            prob_a = 1 / float(row['B365A'])
            
            # 计算衍生特征：主客队相对强弱指数
            h_strength = prob_h / (prob_h + prob_a)
            a_strength = prob_a / (prob_h + prob_a)
            
            # 提取目标值 FTR (0:主胜, 1:平局, 2:客胜)
            target_map = {'H': 0, 'D': 1, 'A': 2}
            target = target_map.get(str(row['FTR']).upper())
            
            if target is not None:
                # 构建训练特征向量 (必须与预测时的特征维度对齐)
                features = [prob_h, prob_d, prob_a, h_strength, a_strength]
                X.append(features)
                y.append(target)
        except:
            continue
            
    return np.array(X), np.array(y)

def _fallback_training_data(n=1000):
    """防断网备用：基于逻辑的模拟数据"""
    np.random.seed(42); X = []; y = []
    for _ in range(n):
        ph = np.random.uniform(0.2, 0.8)
        pd = np.random.uniform(0.15, 0.35)
        pa = max(0.01, 1 - ph - pd)
        X.append([ph, pd, pa, ph/(ph+pa), pa/(ph+pa)])
        # 根据概率生成赛果
        res = np.random.choice([0, 1, 2], p=[ph, pd, pa])
        y.append(res)
    return np.array(X), np.array(y)

def _build_ml_features(match, match_odds):
    """
    在预测阶段，提取比赛数据转化为 ML 特征向量
    必须与训练时的特征保持一致 [prob_h, prob_d, prob_a, h_strength, a_strength]
    """
    try:
        # 优先使用真实提取到的平均赔率
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
            else:
                raise Exception("Odds incomplete")
        else:
            # 降级：从统计战绩中估算胜率
            hw = float(match.get("home_stats", {}).get("wins", 4))
            hp = float(match.get("home_stats", {}).get("played", 10))
            aw = float(match.get("away_stats", {}).get("wins", 4))
            ap = float(match.get("away_stats", {}).get("played", 10))
            
            prob_h = max(0.2, min(0.75, (hw/max(1, hp)) * 1.1))
            prob_a = max(0.2, min(0.75, (aw/max(1, ap)) * 0.9))
            prob_d = max(0.15, 1 - prob_h - prob_a)
            
        h_strength = prob_h / (prob_h + prob_a)
        a_strength = prob_a / (prob_h + prob_a)
        
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
        
        res = {
            "home_win": round(proba[0] * 100, 1), 
            "draw": round(proba[1] * 100, 1), 
            "away_win": round(proba[2] * 100, 1), 
            "model": self.name
        }
        return res

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
# 传统统计与概率模型 (保持原样)
# ==========================================

class PoissonModel:
    def predict(self, home_gf, home_ga, away_gf, away_ga, league_avg=1.35):
        try:
            home_gf = float(home_gf or 1.3); home_ga = float(home_ga or 1.1)
            away_gf = float(away_gf or 1.1); away_ga = float(away_ga or 1.3)
        except:
            home_gf = 1.3; home_ga = 1.1; away_gf = 1.1; away_ga = 1.3
            
        ha = (home_gf / league_avg) * 1.10
        hd = (home_ga / league_avg) * 0.90
        aa = (away_gf / league_avg)
        ad = (away_ga / league_avg)
        
        he = max(0.3, min(ha * ad * league_avg, 4.5))
        ae = max(0.2, min(aa * hd * league_avg, 4.0))
        
        def pmf(k, l): return (l**k) * math.exp(-l) / math.factorial(k)
        
        hw = 0; dr = 0; aw = 0; o25 = 0; bt = 0; scores = []
        for i in range(7):
            for j in range(7):
                p = pmf(i, he) * pmf(j, ae)
                if i > j: hw += p
                elif i == j: dr += p
                else: aw += p
                if i + j > 2: o25 += p
                if i > 0 and j > 0: bt += p
                scores.append((i, j, p))
                
        scores.sort(key=lambda x: x[2], reverse=True)
        t = hw + dr + aw
        if t > 0: hw /= t; dr /= t; aw /= t
        
        return {
            "home_win": round(hw * 100, 1), "draw": round(dr * 100, 1), "away_win": round(aw * 100, 1), 
            "predicted_score": "%d-%d" % (scores[0][0], scores[0][1]),
            "home_xg": round(he, 2), "away_xg": round(ae, 2),
            "over_2_5": round(o25 * 100, 1), "btts": round(bt * 100, 1), 
            "top_scores": [{"score": "%d-%d" % (s[0], s[1]), "prob": round(s[2] * 100, 1)} for s in scores[:5]]
        }

class DixonColesModel:
    def predict(self, home_gf, home_ga, away_gf, away_ga):
        try:
            home_gf = float(home_gf or 1.3); away_gf = float(away_gf or 1.1)
        except:
            home_gf = 1.3; away_gf = 1.1
        he = max(0.3, min(home_gf * 1.05, 4.0)); ae = max(0.2, min(away_gf * 0.95, 3.5))
        rho = -0.15 if abs(he - ae) < 0.5 else -0.05
        
        def pmf(k, l): return (l**k) * math.exp(-l) / math.factorial(k)
        def tau(i, j, lam, mu, r):
            if i == 0 and j == 0: return 1 - lam * mu * r
            elif i == 0 and j == 1: return 1 + lam * r
            elif i == 1 and j == 0: return 1 + mu * r
            elif i == 1 and j == 1: return 1 - r
            return 1
            
        hw = 0; dr = 0; aw = 0; scores = []
        for i in range(6):
            for j in range(6):
                p = max(0, tau(i, j, he, ae, rho) * pmf(i, he) * pmf(j, ae))
                if i > j: hw += p
                elif i == j: dr += p
                else: aw += p
                scores.append((i, j, p))
                
        scores.sort(key=lambda x: x[2], reverse=True)
        t = hw + dr + aw
        if t > 0: hw /= t; dr /= t; aw /= t
        return {"home_win": round(hw * 100, 1), "draw": round(dr * 100, 1), "away_win": round(aw * 100, 1), "predicted_score": "%d-%d" % (scores[0][0], scores[0][1])}

class EloModel:
    def __init__(self):
        self.ratings = defaultdict(lambda: 1500)
        self.k = 30
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

class BradleyTerryModel:
    def predict(self, home_wins, home_total, away_wins, away_total):
        try:
            hw = int(home_wins or 0); ht = max(1, int(home_total or 1))
            aw = int(away_wins or 0); at = max(1, int(away_total or 1))
        except:
            hw = 5; ht = 15; aw = 5; at = 15
        hp = max(0.1, hw / ht); ap = max(0.1, aw / at)
        h_str = hp / (hp + ap) * 1.10; a_str = ap / (hp + ap) * 0.90
        dr = 0.25; h = h_str * (1 - dr); a = a_str * (1 - dr)
        t = h + dr + a
        return {"home_win": round(h/t*100, 1), "draw": round(dr/t*100, 1), "away_win": round(a/t*100, 1)}

class MonteCarloModel:
    def simulate(self, home_gf, home_ga, away_gf, away_ga, n=10000):
        try:
            home_gf = float(home_gf or 1.3); away_gf = float(away_gf or 1.1)
        except:
            home_gf = 1.3; away_gf = 1.1
        he = max(0.3, min(home_gf * 1.05, 4.0)); ae = max(0.2, min(away_gf * 0.95, 3.5))
        np.random.seed(int(time.time() % 1000))
        hg = np.random.poisson(he, n); ag = np.random.poisson(ae, n)
        hw = np.sum(hg > ag) / n; dr = np.sum(hg == ag) / n; aw = np.sum(hg < ag) / n
        o25 = np.sum((hg + ag) > 2) / n; bt = np.sum((hg > 0) & (ag > 0)) / n
        from collections import Counter
        sc = Counter(zip(hg.tolist(), ag.tolist())).most_common(5)
        return {"home_win": round(hw*100, 1), "draw": round(dr*100, 1), "away_win": round(aw*100, 1), "avg_total_goals": round(np.mean(hg + ag), 2), "top_scores": [{"score": f"{s[0][0]}-{s[0][1]}", "prob": round(s[1]/n*100, 1)} for s in sc]}

class BayesianModel:
    def predict(self, hw, hd, hl, aw, ad, al):
        try:
            hw=int(hw or 5); hd=int(hd or 3); hl=int(hl or 3)
            aw=int(aw or 5); ad=int(ad or 3); al=int(al or 3)
        except:
            hw=5; hd=3; hl=3; aw=5; ad=3; al=3
        ph = 1.2 + hw * 1.0 + al * 0.6
        pd = 1.0 + hd * 0.8 + ad * 0.8
        pa = 0.8 + aw * 1.0 + hl * 0.6
        t = ph + pd + pa
        return {"home_win": round(ph/t*100, 1), "draw": round(pd/t*100, 1), "away_win": round(pa/t*100, 1)}

class FormModel:
    def analyze(self, form):
        if not form: return {"score": 50, "trend": "unknown"}
        w = form.count("W"); d = form.count("D"); l = form.count("L"); t = w + d + l
        if t == 0: return {"score": 50, "trend": "unknown"}
        momentum = sum((3 if c=="W" else 1 if c=="D" else 0) * max(0.2, 1.0 - i*0.15) for i, c in enumerate(reversed(form)))
        tw = sum(3 * max(0.2, 1.0 - i*0.15) for i in range(len(form)))
        score = (momentum / tw) * 100 if tw > 0 else 50
        rec = form[-5:] if len(form) >= 5 else form
        rw = rec.count("W"); rl = rec.count("L")
        trend = "fire" if rw >= 4 else "hot" if rw >= 3 and rl == 0 else "ice" if rl >= 4 else "cold" if rl >= 3 and rw == 0 else "good" if rw > rl else "poor" if rl > rw else "mixed"
        return {"score": round(score, 1), "trend": trend}

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
        hp = round(1/ah/(1+mg)*100, 1); dp = round(1/ad/(1+mg)*100, 1); ap = round(100-hp-dp, 1)
        return {"avg_home_odds": round(ah, 2), "avg_draw_odds": round(ad, 2), "avg_away_odds": round(aa, 2), "implied_home": hp, "implied_draw": dp, "implied_away": ap}

# ==========================================
# 综合预测聚合器
# ==========================================

class EnsemblePredictor:
    def __init__(self):
        self.poisson = PoissonModel()
        self.dixon = DixonColesModel()
        self.elo = EloModel()
        self.bt = BradleyTerryModel()
        self.mc = MonteCarloModel()
        self.bayes = BayesianModel()
        self.form = FormModel()
        self.odds = OddsAnalyzer()
        self.rf = RandomForestModel()
        self.gb = GradientBoostModel()
        self.nn = NeuralNetModel()
        self.lr = LogisticModel()
        
        print("[Models] 正在初始化量化分析引擎并加载 CSV 历史数据...")
        self.rf.train(); self.gb.train(); self.nn.train(); self.lr.train()
        print("[Models] 所有模型就绪！")

    def predict(self, match, odds_data=None):
        hs = match.get("home_stats", {}); ast = match.get("away_stats", {}); h2h = match.get("h2h", [])
        home = match["home_team"]; away = match["away_team"]
        
        poi = self.poisson.predict(hs.get("avg_goals_for"), hs.get("avg_goals_against"), ast.get("avg_goals_for"), ast.get("avg_goals_against"))
        dc = self.dixon.predict(hs.get("avg_goals_for"), hs.get("avg_goals_against"), ast.get("avg_goals_for"), ast.get("avg_goals_against"))
        if h2h: self.elo.load_h2h(h2h)
        elo = self.elo.predict(home, away)
        bt = self.bt.predict(hs.get("wins"), hs.get("played"), ast.get("wins"), ast.get("played"))
        mc = self.mc.simulate(hs.get("avg_goals_for"), hs.get("avg_goals_against"), ast.get("avg_goals_for"), ast.get("avg_goals_against"))
        bay = self.bayes.predict(hs.get("wins"), hs.get("draws"), hs.get("losses"), ast.get("wins"), ast.get("draws"), ast.get("losses"))
        hf = self.form.analyze(hs.get("form", "")); af = self.form.analyze(ast.get("form", ""))
        
        rf = self.rf.predict(match, odds_data)
        gb = self.gb.predict(match, odds_data)
        nn = self.nn.predict(match, odds_data)
        lr = self.lr.predict(match, odds_data)
        
        oa = {}
        if odds_data and odds_data.get("bookmakers"): oa = self.odds.analyze_market(odds_data["bookmakers"])
        
        w = {"poisson": 0.16, "dixon": 0.12, "elo": 0.10, "bt": 0.04, "mc": 0.10, "bayes": 0.04, "rf": 0.13, "gb": 0.13, "nn": 0.10, "lr": 0.08}
        models = [("poisson", poi), ("dixon", dc), ("elo", elo), ("bt", bt), ("mc", mc), ("bayes", bay), ("rf", rf), ("gb", gb), ("nn", nn), ("lr", lr)]
        
        hp = 0; dp = 0; ap = 0
        for name, pred in models:
            wt = w.get(name, 0.05)
            hp += pred.get("home_win", 33) * wt
            dp += pred.get("draw", 33) * wt
            ap += pred.get("away_win", 33) * wt
            
        fd = hf["score"] - af["score"]
        hp += (fd * 0.08); ap -= (fd * 0.08)
        
        if oa:
            hp = hp * 0.85 + oa.get("implied_home", 33) * 0.15
            dp = dp * 0.85 + oa.get("implied_draw", 33) * 0.15
            ap = ap * 0.85 + oa.get("implied_away", 33) * 0.15
            
        t = hp + dp + ap
        if t > 0: hp = round(hp/t*100, 1); dp = round(dp/t*100, 1); ap = round(100-hp-dp, 1)
        
        agree_h = sum(1 for _, p in models if p.get("home_win", 0) > max(p.get("draw", 0), p.get("away_win", 0)))
        agree_a = sum(1 for _, p in models if p.get("away_win", 0) > max(p.get("home_win", 0), p.get("draw", 0)))
        consensus = max(agree_h, agree_a)
        
        cf = min(95, max(30, 30 + consensus * 5 + (12 if max(hp, dp, ap) > 60 else 6)))
        
        return {
            "home_win_pct": hp, "draw_pct": dp, "away_win_pct": ap, 
            "predicted_score": poi["predicted_score"], "confidence": cf, 
            "poisson": poi, "dixon_coles": dc, "elo": elo, "bradley_terry": bt, 
            "monte_carlo": mc, "bayesian": bay, "random_forest": rf, 
            "gradient_boost": gb, "neural_net": nn, "logistic": lr, 
            "home_form": hf, "away_form": af, "odds": oa, 
            "over_2_5": poi.get("over_2_5", 50), "btts": poi.get("btts", 50), 
            "model_consensus": consensus, "total_models": len(models)
        }
