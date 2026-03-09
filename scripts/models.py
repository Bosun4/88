import math, random
import numpy as np
from collections import defaultdict

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SK = True
except:
    HAS_SK = False
    print("[WARN] sklearn not available, using fallback models")

class PoissonModel:
    """泊松分布预测比分概率"""
    def predict(self, home_gf, home_ga, away_gf, away_ga, league_avg=1.35):
        try:
            home_gf = float(home_gf or 1.3)
            home_ga = float(home_ga or 1.1)
            away_gf = float(away_gf or 1.1)
            away_ga = float(away_ga or 1.3)
        except:
            home_gf = 1.3; home_ga = 1.1; away_gf = 1.1; away_ga = 1.3
            
        ha = home_gf / league_avg; aa = away_gf / league_avg
        hd = home_ga / league_avg; ad = away_ga / league_avg
        he = ha * ad * league_avg * 1.05; ae = aa * hd * league_avg * 0.95
        he = max(0.3, min(he, 4.5)); ae = max(0.2, min(ae, 4.0))
        
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
            "home_win": round(hw * 100, 1), 
            "draw": round(dr * 100, 1), 
            "away_win": round(aw * 100, 1), 
            "predicted_score": "%d-%d" % (scores[0][0], scores[0][1]),
            "home_xg": round(he, 2), 
            "away_xg": round(ae, 2),
            "home_expected_goals": round(he, 2), # 修复网页前端显示问号的问题
            "away_expected_goals": round(ae, 2), # 修复网页前端显示问号的问题
            "over_2_5": round(o25 * 100, 1), 
            "btts": round(bt * 100, 1), 
            "top_scores": [{"score": "%d-%d" % (s[0], s[1]), "prob": round(s[2] * 100, 1)} for s in scores[:5]]
        }

class DixonColesModel:
    """Dixon-Coles改进泊松模型(低比分修正)"""
    def predict(self, home_gf, home_ga, away_gf, away_ga):
        try:
            home_gf = float(home_gf or 1.3); home_ga = float(home_ga or 1.1)
            away_gf = float(away_gf or 1.1); away_ga = float(away_ga or 1.3)
        except:
            home_gf = 1.3; home_ga = 1.1; away_gf = 1.1; away_ga = 1.3
        he = home_gf * 1.05; ae = away_gf * 0.95
        he = max(0.3, min(he, 4.5)); ae = max(0.2, min(ae, 4.0))
        rho = -0.13
        
        def pmf(k, l): return (l**k) * math.exp(-l) / math.factorial(k)
        def tau(i, j, lam, mu, r):
            if i == 0 and j == 0: return 1 - lam * mu * r
            elif i == 0 and j == 1: return 1 + lam * r
            elif i == 1 and j == 0: return 1 + mu * r
            elif i == 1 and j == 1: return 1 - r
            return 1
            
        hw = 0; dr = 0; aw = 0; scores = []
        for i in range(7):
            for j in range(7):
                p = tau(i, j, he, ae, rho) * pmf(i, he) * pmf(j, ae)
                p = max(0, p)
                if i > j: hw += p
                elif i == j: dr += p
                else: aw += p
                scores.append((i, j, p))
                
        scores.sort(key=lambda x: x[2], reverse=True)
        t = hw + dr + aw
        if t > 0: hw /= t; dr /= t; aw /= t
        
        return {"home_win": round(hw * 100, 1), "draw": round(dr * 100, 1), "away_win": round(aw * 100, 1), "predicted_score": "%d-%d" % (scores[0][0], scores[0][1]), "model": "Dixon-Coles"}

class EloModel:
    """ELO评分系统"""
    def __init__(self):
        self.ratings = defaultdict(lambda: 1500)
        self.k = 32
        
    def update(self, h, a, hg, ag):
        rh = self.ratings[h]; ra = self.ratings[a]
        eh = 1 / (1 + 10 ** ((ra - rh) / 400)); ea = 1 - eh
        sh = 1 if hg > ag else (0.5 if hg == ag else 0); sa = 1 - sh
        self.ratings[h] = rh + self.k * (sh - eh); self.ratings[a] = ra + self.k * (sa - ea)
        
    def predict(self, h, a):
        rh = self.ratings[h] + 65; ra = self.ratings[a]
        eh = 1 / (1 + 10 ** ((ra - rh) / 400))
        df = 0.26; hw = eh * (1 - df / 2); aw = (1 - eh) * (1 - df / 2); dr = df
        return {"home_win": round(hw * 100, 1), "draw": round(dr * 100, 1), "away_win": round(aw * 100, 1), "home_elo": round(rh, 1), "away_elo": round(ra, 1), "elo_diff": round(rh - ra, 1)}
        
    def load_h2h(self, records):
        for r in reversed(records):
            try:
                p = r["score"].split("-")
                self.update(r["home"], r["away"], int(p[0]), int(p[1]))
            except: pass

class BradleyTerryModel:
    """Bradley-Terry配对比较模型"""
    def predict(self, home_wins, home_total, away_wins, away_total):
        try:
            hw = int(home_wins or 0); ht = int(home_total or 1)
            aw = int(away_wins or 0); at = int(away_total or 1)
        except:
            hw = 5; ht = 15; aw = 5; at = 15
        hp = max(0.1, (hw / ht if ht else 0.33)); ap = max(0.1, (aw / at if at else 0.33))
        home_strength = hp / (hp + ap) * 1.08
        away_strength = ap / (hp + ap) * 0.92
        draw_prob = 0.24
        h = home_strength * (1 - draw_prob); a = away_strength * (1 - draw_prob)
        t = h + draw_prob + a
        return {"home_win": round(h / t * 100, 1), "draw": round(draw_prob / t * 100, 1), "away_win": round(a / t * 100, 1), "home_strength": round(home_strength, 3), "away_strength": round(away_strength, 3), "model": "Bradley-Terry"}

class MonteCarloModel:
    """蒙特卡洛模拟(10000次)"""
    def simulate(self, home_gf, home_ga, away_gf, away_ga, n=10000):
        try:
            home_gf = float(home_gf or 1.3); home_ga = float(home_ga or 1.1)
            away_gf = float(away_gf or 1.1); away_ga = float(away_ga or 1.3)
        except:
            home_gf = 1.3; home_ga = 1.1; away_gf = 1.1; away_ga = 1.3
        he = home_gf * 1.05; ae = away_gf * 0.95
        he = max(0.3, min(he, 4.0)); ae = max(0.2, min(ae, 3.5))
        np.random.seed(42)
        hg = np.random.poisson(he, n); ag = np.random.poisson(ae, n)
        hw = np.sum(hg > ag) / n; dr = np.sum(hg == ag) / n; aw = np.sum(hg < ag) / n
        o25 = np.sum((hg + ag) > 2) / n; bt = np.sum((hg > 0) & (ag > 0)) / n
        avg_hg = round(np.mean(hg), 2); avg_ag = round(np.mean(ag), 2)
        avg_total = round(np.mean(hg + ag), 2)
        from collections import Counter
        sc = Counter(zip(hg.tolist(), ag.tolist())).most_common(5)
        return {"home_win": round(hw * 100, 1), "draw": round(dr * 100, 1), "away_win": round(aw * 100, 1), "simulations": n, "avg_home_goals": avg_hg, "avg_away_goals": avg_ag, "avg_total_goals": avg_total, "over_2_5": round(o25 * 100, 1), "btts": round(bt * 100, 1), "top_scores": [{"score": "%d-%d" % (s[0][0], s[0][1]), "prob": round(s[1] / n * 100, 1)} for s in sc], "model": "MonteCarlo"}

class BayesianModel:
    """贝叶斯推断模型"""
    def predict(self, home_wins, home_draws, home_losses, away_wins, away_draws, away_losses):
        try:
            hw = int(home_wins or 5); hd = int(home_draws or 3); hl = int(home_losses or 3)
            aw = int(away_wins or 5); ad = int(away_draws or 3); al = int(away_losses or 3)
        except:
            hw = 5; hd = 3; hl = 3; aw = 5; ad = 3; al = 3
        prior_h = 1; prior_d = 1; prior_a = 1
        post_h = prior_h + hw * 1.1 + al * 0.5
        post_d = prior_d + hd * 0.8 + ad * 0.8
        post_a = prior_a + aw * 0.9 + hl * 0.4
        t = post_h + post_d + post_a
        return {"home_win": round(post_h / t * 100, 1), "draw": round(post_d / t * 100, 1), "away_win": round(post_a / t * 100, 1), "posterior_h": round(post_h, 2), "posterior_d": round(post_d, 2), "posterior_a": round(post_a, 2), "model": "Bayesian"}

class FormModel:
    """近期状态深度分析"""
    def analyze(self, form):
        if not form: return {"score": 50, "trend": "unknown", "wins": 0, "draws": 0, "losses": 0, "momentum": 0, "recent": ""}
        w = form.count("W"); d = form.count("D"); l = form.count("L"); t = w + d + l
        if t == 0: return {"score": 50, "trend": "unknown", "wins": 0, "draws": 0, "losses": 0, "momentum": 0, "recent": ""}
        score = (w * 3 + d) / (t * 3) * 100
        rec = form[-5:] if len(form) >= 5 else form
        rw = rec.count("W"); rl = rec.count("L")
        momentum = 0
        for i, c in enumerate(form):
            weight = (i + 1) / len(form)
            if c == "W": momentum += 3 * weight
            elif c == "D": momentum += 1 * weight
        momentum = momentum / (t * 3) * 100
        if rw >= 4: trend = "fire"
        elif rw >= 3: trend = "hot"
        elif rl >= 3: trend = "cold"
        elif rl >= 4: trend = "ice"
        elif rw > rl: trend = "good"
        elif rl > rw: trend = "poor"
        else: trend = "mixed"
        streak = 0; last = form[-1] if form else ""
        for c in reversed(form):
            if c == last: streak += 1
            else: break
        return {"score": round(score, 1), "trend": trend, "wins": w, "draws": d, "losses": l, "momentum": round(momentum, 1), "recent": rec, "streak": streak, "streak_type": last}

class KellyCriterion:
    """凯利公式"""
    def calculate(self, prob, odds, fraction=0.25):
        if odds <= 1 or prob <= 0 or prob >= 1: return {"kelly": 0, "value": False, "edge": 0}
        q = 1 - prob; b = odds - 1; kelly = (b * prob - q) / b; edge = (prob * odds - 1) * 100
        return {"kelly": round(max(0, kelly) * fraction * 100, 2), "value": edge > 0, "edge": round(edge, 1)}

class OddsAnalyzer:
    """赔率深度分析"""
    def analyze_market(self, bookmakers):
        if not bookmakers: return {}
        ho = []; do2 = []; ao = []
        for bk in bookmakers:
            h2h = bk.get("markets", {}).get("h2h", {})
            if "Home" in h2h: ho.append(h2h["Home"])
            if "Draw" in h2h: do2.append(h2h["Draw"])
            if "Away" in h2h: ao.append(h2h["Away"])
        if not ho: return {}
        ah = sum(ho) / len(ho); ad = sum(do2) / len(do2) if do2 else 3.3; aa = sum(ao) / len(ao) if ao else 3.0
        mg = 1 / ah + 1 / ad + 1 / aa - 1
        hp = round(1 / ah / (1 + mg) * 100, 1) if mg > -1 else 33
        dp = round(1 / ad / (1 + mg) * 100, 1) if mg > -1 else 33
        ap = round(100 - hp - dp, 1)
        return {"avg_home_odds": round(ah, 2), "avg_draw_odds": round(ad, 2), "avg_away_odds": round(aa, 2), "implied_home": hp, "implied_draw": dp, "implied_away": ap, "margin": round(mg * 100, 1), "consensus": "home" if hp > dp and hp > ap else ("away" if ap > hp and ap > dp else "draw"), "bookmaker_count": len(ho)}

def _build_features(hs, ast):
    """从球队统计构建特征向量"""
    def s(v, d=0):
        try: return float(v)
        except: return d
    return [
        s(hs.get("wins"), 5), s(hs.get("draws"), 3), s(hs.get("losses"), 3),
        s(hs.get("goals_for"), 15), s(hs.get("goals_against"), 12),
        s(hs.get("avg_goals_for"), 1.3), s(hs.get("avg_goals_against"), 1.1),
        s(hs.get("clean_sheets"), 3),
        len([c for c in hs.get("form", "") if c == "W"]),
        len([c for c in hs.get("form", "") if c == "L"]),
        s(ast.get("wins"), 5), s(ast.get("draws"), 3), s(ast.get("losses"), 3),
        s(ast.get("goals_for"), 15), s(ast.get("goals_against"), 12),
        s(ast.get("avg_goals_for"), 1.3), s(ast.get("avg_goals_against"), 1.1),
        s(ast.get("clean_sheets"), 3),
        len([c for c in ast.get("form", "") if c == "W"]),
        len([c for c in ast.get("form", "") if c == "L"]),
    ]

def _generate_training_data(n=500):
    """生成模拟训练数据"""
    np.random.seed(42); X = []; y = []
    for _ in range(n):
        hw = np.random.randint(2, 18); hd = np.random.randint(1, 8); hl = np.random.randint(1, 12)
        hgf = int(hw * 1.8 + hd * 0.8 + np.random.randint(0, 5)); hga = int(hl * 1.5 + hd * 0.7 + np.random.randint(0, 4))
        ht = hw + hd + hl; hagf = round(hgf / ht, 2) if ht else 1.3; haga = round(hga / ht, 2) if ht else 1.1
        hcs = np.random.randint(1, ht // 2 + 1); hfw = np.random.randint(0, min(5, hw + 1)); hfl = np.random.randint(0, min(5, hl + 1))
        aw = np.random.randint(2, 18); ad = np.random.randint(1, 8); al = np.random.randint(1, 12)
        agf = int(aw * 1.8 + ad * 0.8 + np.random.randint(0, 5)); aga = int(al * 1.5 + ad * 0.7 + np.random.randint(0, 4))
        at = aw + ad + al; aagf = round(agf / at, 2) if at else 1.1; aaga = round(aga / at, 2) if at else 1.3
        acs = np.random.randint(1, at // 2 + 1); afw = np.random.randint(0, min(5, aw + 1)); afl = np.random.randint(0, min(5, al + 1))
        feat = [hw, hd, hl, hgf, hga, hagf, haga, hcs, hfw, hfl, aw, ad, al, agf, aga, aagf, aaga, acs, afw, afl]
        X.append(feat)
        home_s = hw * 3 + hd + hcs * 0.5 + hfw * 2 - hfl * 2 + hgf * 0.3 - hga * 0.2
        away_s = aw * 3 + ad + acs * 0.5 + afw * 2 - afl * 2 + agf * 0.3 - aga * 0.2
        diff = home_s - away_s + np.random.normal(0, 8) + 5
        if diff > 6: y.append(0)
        elif diff > -3: y.append(1)
        else: y.append(2)
    return np.array(X), np.array(y)

class RandomForestModel:
    """随机森林分类器"""
    def __init__(self):
        self.model = None; self.scaler = None; self.trained = False
    def train(self):
        if not HAS_SK: return
        X, y = _generate_training_data(800)
        self.scaler = StandardScaler(); X = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
        self.model.fit(X, y); self.trained = True
    def predict(self, hs, ast):
        if not self.trained: self.train()
        if not self.model: return {"home_win": 40, "draw": 30, "away_win": 30, "model": "RF-fallback"}
        feat = _build_features(hs, ast)
        X = self.scaler.transform([feat])
        proba = self.model.predict_proba(X)[0]
        imp = self.model.feature_importances_
        top_feat = sorted(enumerate(imp), key=lambda x: x[1], reverse=True)[:5]
        names = ["主胜场", "主平场", "主负场", "主进球", "主失球", "主均进", "主均失", "主零封", "主近胜", "主近负", "客胜场", "客平场", "客负场", "客进球", "客失球", "客均进", "客均失", "客零封", "客近胜", "客近负"]
        return {"home_win": round(proba[0] * 100, 1), "draw": round(proba[1] * 100, 1), "away_win": round(proba[2] * 100, 1), "top_features": [(names[i] if i < len(names) else "f%d" % i, round(v * 100, 1)) for i, v in top_feat], "model": "RandomForest"}

class GradientBoostModel:
    """梯度提升分类器(XGBoost风格)"""
    def __init__(self): self.model = None; self.scaler = None; self.trained = False
    def train(self):
        if not HAS_SK: return
        X, y = _generate_training_data(800)
        self.scaler = StandardScaler(); X = self.scaler.fit_transform(X)
        self.model = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
        self.model.fit(X, y); self.trained = True
    def predict(self, hs, ast):
        if not self.trained: self.train()
        if not self.model: return {"home_win": 40, "draw": 30, "away_win": 30, "model": "GB-fallback"}
        feat = _build_features(hs, ast)
        X = self.scaler.transform([feat])
        proba = self.model.predict_proba(X)[0]
        return {"home_win": round(proba[0] * 100, 1), "draw": round(proba[1] * 100, 1), "away_win": round(proba[2] * 100, 1), "model": "GradientBoost"}

class NeuralNetModel:
    """多层感知器神经网络"""
    def __init__(self): self.model = None; self.scaler = None; self.trained = False
    def train(self):
        if not HAS_SK: return
        X, y = _generate_training_data(800)
        self.scaler = StandardScaler(); X = self.scaler.fit_transform(X)
        self.model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation="relu", max_iter=500, random_state=42, early_stopping=True)
        self.model.fit(X, y); self.trained = True
    def predict(self, hs, ast):
        if not self.trained: self.train()
        if not self.model: return {"home_win": 40, "draw": 30, "away_win": 30, "model": "NN-fallback"}
        feat = _build_features(hs, ast)
        X = self.scaler.transform([feat])
        proba = self.model.predict_proba(X)[0]
        return {"home_win": round(proba[0] * 100, 1), "draw": round(proba[1] * 100, 1), "away_win": round(proba[2] * 100, 1), "model": "NeuralNet"}

class LogisticModel:
    """逻辑回归模型"""
    def __init__(self): self.model = None; self.scaler = None; self.trained = False
    def train(self):
        if not HAS_SK: return
        X, y = _generate_training_data(800)
        self.scaler = StandardScaler(); X = self.scaler.fit_transform(X)
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X, y); self.trained = True
    def predict(self, hs, ast):
        if not self.trained: self.train()
        if not self.model: return {"home_win": 40, "draw": 30, "away_win": 30, "model": "LR-fallback"}
        feat = _build_features(hs, ast)
        X = self.scaler.transform([feat])
        proba = self.model.predict_proba(X)[0]
        return {"home_win": round(proba[0] * 100, 1), "draw": round(proba[1] * 100, 1), "away_win": round(proba[2] * 100, 1), "model": "Logistic"}

class EnsemblePredictor:
    """全模型融合预测器(9个模型)"""
    def __init__(self):
        self.poisson = PoissonModel()
        self.dixon = DixonColesModel()
        self.elo = EloModel()
        self.bt = BradleyTerryModel()
        self.mc = MonteCarloModel()
        self.bayes = BayesianModel()
        self.form = FormModel()
        self.kelly = KellyCriterion()
        self.odds = OddsAnalyzer()
        self.rf = RandomForestModel()
        self.gb = GradientBoostModel()
        self.nn = NeuralNetModel()
        self.lr = LogisticModel()
        print("[Models] Training ML models...")
        self.rf.train(); self.gb.train(); self.nn.train(); self.lr.train()
        print("[Models] All ready!")

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
        rf = self.rf.predict(hs, ast); gb = self.gb.predict(hs, ast); nn = self.nn.predict(hs, ast); lr = self.lr.predict(hs, ast)
        oa = {}
        if odds_data and odds_data.get("bookmakers"): oa = self.odds.analyze_market(odds_data["bookmakers"])
        
        w = {"poisson": 0.18, "dixon": 0.10, "elo": 0.10, "bt": 0.05, "mc": 0.12, "bayes": 0.05, "rf": 0.12, "gb": 0.10, "nn": 0.08, "lr": 0.05, "form": 0.05}
        models = [("poisson", poi), ("dixon", dc), ("elo", elo), ("bt", bt), ("mc", mc), ("bayes", bay), ("rf", rf), ("gb", gb), ("nn", nn), ("lr", lr)]
        hp = 0; dp = 0; ap = 0
        
        for name, pred in models:
            wt = w.get(name, 0.05)
            hp += pred.get("home_win", 33) * wt
            dp += pred.get("draw", 33) * wt
            ap += pred.get("away_win", 33) * wt
            
        fd = hf["score"] - af["score"]; hp += (50 + fd * 0.15) * w["form"]; dp += 25 * w["form"]; ap += (50 - fd * 0.15) * w["form"]
        if oa:
            hp += oa.get("implied_home", 33) * 0.05; dp += oa.get("implied_draw", 33) * 0.05; ap += oa.get("implied_away", 33) * 0.05
            
        t = hp + dp + ap
        if t > 0: hp = round(hp / t * 100, 1); dp = round(dp / t * 100, 1); ap = round(100 - hp - dp, 1)
        agree = sum(1 for _, p in models if p.get("home_win", 0) > p.get("draw", 0) and p.get("home_win", 0) > p.get("away_win", 0))
        agree_a = sum(1 for _, p in models if p.get("away_win", 0) > p.get("home_win", 0))
        consensus = max(agree, agree_a)
        cf = 40 + consensus * 5
        mx = max(hp, dp, ap)
        if mx > 60: cf += 8
        elif mx > 50: cf += 4
        if hf["trend"] in ["fire", "hot"] and fd > 15: cf += 5
        cf = min(92, max(30, cf))
        
        return {
            "home_win_pct": hp, "draw_pct": dp, "away_win_pct": ap, 
            "predicted_score": poi["predicted_score"], "confidence": cf, 
            "poisson": poi, "dixon_coles": dc, "elo": elo, "bradley_terry": bt, 
            "monte_carlo": mc, "bayesian": bay, "random_forest": rf, 
            "gradient_boost": gb, "neural_net": nn, "logistic": lr, 
            "home_form": hf, "away_form": af, "odds": oa, 
            "over_2_5": poi["over_2_5"], "btts": poi["btts"], 
            "top_scores": mc.get("top_scores", poi.get("top_scores", [])), 
            "model_weights": w, "model_consensus": consensus, "total_models": len(models)
        }
