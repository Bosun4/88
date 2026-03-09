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

# ==================== 1. 纯统计学与概率模型 ====================

class PoissonModel:
    """泊松分布预测 (带动态主场优势修正)"""
    def predict(self, home_gf, home_ga, away_gf, away_ga, league_avg=1.35):
        try:
            home_gf = float(home_gf or 1.3); home_ga = float(home_ga or 1.1)
            away_gf = float(away_gf or 1.1); away_ga = float(away_ga or 1.3)
        except:
            home_gf = 1.3; home_ga = 1.1; away_gf = 1.1; away_ga = 1.3
            
        # 引入真实世界的主场优势系数 (Home Advantage)
        home_adv_att = 1.12; home_adv_def = 0.90
        
        ha = (home_gf / league_avg) * home_adv_att
        hd = (home_ga / league_avg) * home_adv_def
        aa = (away_gf / league_avg)
        ad = (away_ga / league_avg)
        
        he = ha * ad * league_avg
        ae = aa * hd * league_avg
        
        # 限制极值，防止泊松爆炸
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
            "home_win": round(hw * 100, 1), "draw": round(dr * 100, 1), "away_win": round(aw * 100, 1), 
            "predicted_score": "%d-%d" % (scores[0][0], scores[0][1]),
            "home_xg": round(he, 2), "away_xg": round(ae, 2),
            "home_expected_goals": round(he, 2), "away_expected_goals": round(ae, 2),
            "over_2_5": round(o25 * 100, 1), "btts": round(bt * 100, 1), 
            "top_scores": [{"score": "%d-%d" % (s[0], s[1]), "prob": round(s[2] * 100, 1)} for s in scores[:5]]
        }

class DixonColesModel:
    """Dixon-Coles 模型 (强化低比分平局捕捉)"""
    def predict(self, home_gf, home_ga, away_gf, away_ga):
        try:
            home_gf = float(home_gf or 1.3); home_ga = float(home_ga or 1.1)
            away_gf = float(away_gf or 1.1); away_ga = float(away_ga or 1.3)
        except:
            home_gf = 1.3; home_ga = 1.1; away_gf = 1.1; away_ga = 1.3
            
        he = home_gf * 1.08; ae = away_gf * 0.92
        he = max(0.3, min(he, 4.0)); ae = max(0.2, min(ae, 3.5))
        
        # 动态 Rho 系数：强强对话平局概率更高，强弱对话平局概率低
        diff = abs(he - ae)
        rho = -0.15 if diff < 0.5 else -0.05
        
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
    """升级版 ELO 评分 (含净胜球 MOV 权重)"""
    def __init__(self):
        self.ratings = defaultdict(lambda: 1500)
        self.k = 30
        
    def update(self, h, a, hg, ag):
        rh = self.ratings[h]; ra = self.ratings[a]
        eh = 1 / (1 + 10 ** ((ra - rh) / 400)); ea = 1 - eh
        sh = 1 if hg > ag else (0.5 if hg == ag else 0); sa = 1 - sh
        # 净胜球对数乘数 (Margin of Victory)
        mov_multiplier = math.log(abs(hg - ag) + 2) if hg != ag else 1.0
        
        self.ratings[h] = rh + self.k * mov_multiplier * (sh - eh)
        self.ratings[a] = ra + self.k * mov_multiplier * (sa - ea)
        
    def predict(self, h, a):
        rh = self.ratings[h] + 60; ra = self.ratings[a] # 主场优势降低到60分更趋于合理
        eh = 1 / (1 + 10 ** ((ra - rh) / 400))
        df = 0.28 if abs(rh - ra) < 100 else 0.22 # 势均力敌平局概率高
        hw = eh * (1 - df / 2); aw = (1 - eh) * (1 - df / 2); dr = df
        return {"home_win": round(hw * 100, 1), "draw": round(dr * 100, 1), "away_win": round(aw * 100, 1), "home_elo": round(rh, 1), "away_elo": round(ra, 1), "elo_diff": round(rh - ra, 1)}
        
    def load_h2h(self, records):
        for r in reversed(records):
            try:
                p = r["score"].split("-")
                self.update(r["home"], r["away"], int(p[0]), int(p[1]))
            except: pass

class BradleyTerryModel:
    def predict(self, home_wins, home_total, away_wins, away_total):
        try:
            hw = int(home_wins or 0); ht = int(home_total or 1)
            aw = int(away_wins or 0); at = int(away_total or 1)
        except:
            hw = 5; ht = 15; aw = 5; at = 15
        hp = max(0.1, (hw / max(1, ht))); ap = max(0.1, (aw / max(1, at)))
        home_strength = hp / (hp + ap) * 1.10
        away_strength = ap / (hp + ap) * 0.90
        draw_prob = 0.25
        h = home_strength * (1 - draw_prob); a = away_strength * (1 - draw_prob)
        t = h + draw_prob + a
        return {"home_win": round(h / t * 100, 1), "draw": round(draw_prob / t * 100, 1), "away_win": round(a / t * 100, 1), "home_strength": round(home_strength, 3), "away_strength": round(away_strength, 3), "model": "Bradley-Terry"}

class MonteCarloModel:
    def simulate(self, home_gf, home_ga, away_gf, away_ga, n=10000):
        try:
            home_gf = float(home_gf or 1.3); home_ga = float(home_ga or 1.1)
            away_gf = float(away_gf or 1.1); away_ga = float(away_ga or 1.3)
        except:
            home_gf = 1.3; home_ga = 1.1; away_gf = 1.1; away_ga = 1.3
        he = home_gf * 1.05; ae = away_gf * 0.95
        he = max(0.3, min(he, 4.0)); ae = max(0.2, min(ae, 3.5))
        np.random.seed(int(time.time() % 1000)) # 增加随机性
        hg = np.random.poisson(he, n); ag = np.random.poisson(ae, n)
        hw = np.sum(hg > ag) / n; dr = np.sum(hg == ag) / n; aw = np.sum(hg < ag) / n
        o25 = np.sum((hg + ag) > 2) / n; bt = np.sum((hg > 0) & (ag > 0)) / n
        from collections import Counter
        sc = Counter(zip(hg.tolist(), ag.tolist())).most_common(5)
        return {"home_win": round(hw * 100, 1), "draw": round(dr * 100, 1), "away_win": round(aw * 100, 1), "simulations": n, "avg_home_goals": round(np.mean(hg), 2), "avg_away_goals": round(np.mean(ag), 2), "avg_total_goals": round(np.mean(hg + ag), 2), "over_2_5": round(o25 * 100, 1), "btts": round(bt * 100, 1), "top_scores": [{"score": "%d-%d" % (s[0][0], s[0][1]), "prob": round(s[1] / n * 100, 1)} for s in sc], "model": "MonteCarlo"}

class BayesianModel:
    def predict(self, home_wins, home_draws, home_losses, away_wins, away_draws, away_losses):
        try:
            hw = int(home_wins or 5); hd = int(home_draws or 3); hl = int(home_losses or 3)
            aw = int(away_wins or 5); ad = int(away_draws or 3); al = int(away_losses or 3)
        except:
            hw = 5; hd = 3; hl = 3; aw = 5; ad = 3; al = 3
        # 使用 Dirichlet 先验分布
        post_h = 1.2 + hw * 1.0 + al * 0.6
        post_d = 1.0 + hd * 0.8 + ad * 0.8
        post_a = 0.8 + aw * 1.0 + hl * 0.6
        t = post_h + post_d + post_a
        return {"home_win": round(post_h / t * 100, 1), "draw": round(post_d / t * 100, 1), "away_win": round(post_a / t * 100, 1), "model": "Bayesian"}

class FormModel:
    """带时间衰减的高阶状态分析模型"""
    def analyze(self, form):
        if not form: return {"score": 50, "trend": "unknown", "wins": 0, "draws": 0, "losses": 0, "momentum": 0, "recent": ""}
        w = form.count("W"); d = form.count("D"); l = form.count("L"); t = w + d + l
        if t == 0: return {"score": 50, "trend": "unknown", "wins": 0, "draws": 0, "losses": 0, "momentum": 0, "recent": ""}
        
        # 指数衰减动量：越近的比赛权重越大 (最近一场权重 1.0，最远一场 0.2)
        momentum = 0; total_weight = 0
        for i, c in enumerate(reversed(form)):
            weight = max(0.2, 1.0 - i * 0.15)
            if c == "W": momentum += 3 * weight
            elif c == "D": momentum += 1 * weight
            total_weight += 3 * weight
            
        score = (momentum / total_weight) * 100 if total_weight > 0 else 50
        rec = form[-5:] if len(form) >= 5 else form
        rw = rec.count("W"); rl = rec.count("L")
        
        if rw >= 4: trend = "fire"
        elif rw >= 3 and rl == 0: trend = "hot"
        elif rl >= 4: trend = "ice"
        elif rl >= 3 and rw == 0: trend = "cold"
        elif rw > rl: trend = "good"
        elif rl > rw: trend = "poor"
        else: trend = "mixed"
        
        streak = 0; last = form[-1] if form else ""
        for c in reversed(form):
            if c == last: streak += 1
            else: break
        return {"score": round(score, 1), "trend": trend, "wins": w, "draws": d, "losses": l, "momentum": round(momentum, 1), "recent": rec, "streak": streak, "streak_type": last}

# ==================== 2. 核心机器学习系统 (重构升级) ====================
import time

def _build_features(hs, ast):
    """提取高阶特征：PPG, 净胜球率, 场均失球等"""
    def s(v, d=0):
        try: return float(v)
        except: return d
    
    ht = max(1, s(hs.get("played"), 10))
    at = max(1, s(ast.get("played"), 10))
    
    h_win_pct = s(hs.get("wins"), 3) / ht
    h_ppg = (s(hs.get("wins"), 3) * 3 + s(hs.get("draws"), 3)) / ht
    h_gd_per_game = (s(hs.get("goals_for"), 10) - s(hs.get("goals_against"), 10)) / ht
    h_clean_pct = s(hs.get("clean_sheets"), 2) / ht
    h_form_score = FormModel().analyze(hs.get("form", ""))["score"] / 100

    a_win_pct = s(ast.get("wins"), 3) / at
    a_ppg = (s(ast.get("wins"), 3) * 3 + s(ast.get("draws"), 3)) / at
    a_gd_per_game = (s(ast.get("goals_for"), 10) - s(ast.get("goals_against"), 10)) / at
    a_clean_pct = s(ast.get("clean_sheets"), 2) / at
    a_form_score = FormModel().analyze(ast.get("form", ""))["score"] / 100

    return [
        h_win_pct, h_ppg, h_gd_per_game, s(hs.get("avg_goals_for"), 1.3), s(hs.get("avg_goals_against"), 1.1), h_clean_pct, h_form_score,
        a_win_pct, a_ppg, a_gd_per_game, s(ast.get("avg_goals_for"), 1.1), s(ast.get("avg_goals_against"), 1.3), a_clean_pct, a_form_score,
        (h_ppg - a_ppg) # 特征交叉：实力差
    ]

def _generate_realistic_training_data(n=1500):
    """真正的足球环境概率模拟器"""
    np.random.seed(42); X = []; y = []
    for _ in range(n):
        # 模拟真实世界球队实力分布
        h_strength = np.random.uniform(0.6, 2.2)
        a_strength = np.random.uniform(0.6, 2.2)
        
        # 转化为进失球预期 (主场优势 1.2)
        h_gf = h_strength * 1.3
        h_ga = (1 / h_strength) * 1.1
        a_gf = a_strength * 1.1
        a_ga = (1 / a_strength) * 1.3
        
        # 伪造表现特征
        h_ppg = min(3.0, h_strength * 1.2 + np.random.normal(0, 0.2))
        h_gd = (h_gf - h_ga) + np.random.normal(0, 0.3)
        a_ppg = min(3.0, a_strength * 1.1 + np.random.normal(0, 0.2))
        a_gd = (a_gf - a_ga) + np.random.normal(0, 0.3)
        
        feat = [
            h_strength/2.5, h_ppg, h_gd, h_gf, h_ga, np.random.uniform(0.1, 0.5), np.random.uniform(0.2, 0.8),
            a_strength/2.5, a_ppg, a_gd, a_gf, a_ga, np.random.uniform(0.1, 0.5), np.random.uniform(0.2, 0.8),
            (h_ppg - a_ppg)
        ]
        X.append(feat)
        
        # 依据真实实力推导赛果
        h_goals = np.random.poisson(h_gf * 1.2 / max(0.5, a_ga))
        a_goals = np.random.poisson(a_gf * 0.9 / max(0.5, h_ga))
        if h_goals > a_goals: y.append(0) # 主胜
        elif h_goals == a_goals: y.append(1) # 平局
        else: y.append(2) # 客胜
        
    return np.array(X), np.array(y)

class MLPredictorBase:
    """ML 基类，复用训练逻辑"""
    def __init__(self, name):
        self.model = None; self.scaler = None; self.trained = False; self.name = name
    def train(self):
        if not HAS_SK: return
        X, y = _generate_realistic_training_data(1200)
        self.scaler = StandardScaler(); X = self.scaler.fit_transform(X)
        self._init_model()
        self.model.fit(X, y); self.trained = True
    def predict(self, hs, ast):
        if not self.trained: self.train()
        if not self.model: return {"home_win": 40, "draw": 30, "away_win": 30, "model": f"{self.name}-fallback"}
        feat = _build_features(hs, ast)
        X = self.scaler.transform([feat])
        proba = self.model.predict_proba(X)[0]
        res = {"home_win": round(proba[0] * 100, 1), "draw": round(proba[1] * 100, 1), "away_win": round(proba[2] * 100, 1), "model": self.name}
        if hasattr(self.model, "feature_importances_"):
            names = ["主胜率","主PPG","主场均净胜","主均进","主均失","主零封率","主状态","客胜率","客PPG","客场均净胜","客均进","客均失","客零封率","客状态","强弱分化度"]
            imp = self.model.feature_importances_
            res["top_features"] = [(names[i], round(v * 100, 1)) for i, v in sorted(enumerate(imp), key=lambda x: x[1], reverse=True)[:5]]
        return res

class RandomForestModel(MLPredictorBase):
    def __init__(self): super().__init__("RandomForest")
    def _init_model(self): self.model = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=10, random_state=42)

class GradientBoostModel(MLPredictorBase):
    def __init__(self): super().__init__("GradientBoost")
    def _init_model(self): self.model = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)

class NeuralNetModel(MLPredictorBase):
    def __init__(self): super().__init__("NeuralNet")
    def _init_model(self): self.model = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", max_iter=800, alpha=0.01, random_state=42)

class LogisticModel(MLPredictorBase):
    def __init__(self): super().__init__("Logistic")
    def _init_model(self): self.model = LogisticRegression(C=0.5, max_iter=500, random_state=42)


# ==================== 3. 赔率及凯利 (杂项) ====================
class KellyCriterion:
    def calculate(self, prob, odds, fraction=0.25):
        if odds <= 1 or prob <= 0 or prob >= 1: return {"kelly": 0, "value": False, "edge": 0}
        q = 1 - prob; b = odds - 1; kelly = (b * prob - q) / b; edge = (prob * odds - 1) * 100
        return {"kelly": round(max(0, kelly) * fraction * 100, 2), "value": edge > 0, "edge": round(edge, 1)}

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
        ah = sum(ho) / len(ho); ad = sum(do2) / len(do2) if do2 else 3.3; aa = sum(ao) / len(ao) if ao else 3.0
        mg = 1 / ah + 1 / ad + 1 / aa - 1
        hp = round(1 / ah / (1 + mg) * 100, 1) if mg > -1 else 33
        dp = round(1 / ad / (1 + mg) * 100, 1) if mg > -1 else 33
        ap = round(100 - hp - dp, 1)
        return {"avg_home_odds": round(ah, 2), "avg_draw_odds": round(ad, 2), "avg_away_odds": round(aa, 2), "implied_home": hp, "implied_draw": dp, "implied_away": ap, "margin": round(mg * 100, 1), "consensus": "home" if hp > dp and hp > ap else ("away" if ap > hp and ap > dp else "draw")}


# ==================== 4. 融合中枢 (Dynamic Ensemble) ====================

class EnsemblePredictor:
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
        print("[Models] Training ML models with Advanced Features...")
        self.rf.train(); self.gb.train(); self.nn.train(); self.lr.train()
        print("[Models] ML Engine Ready!")

    def predict(self, match, odds_data=None):
        hs = match.get("home_stats", {}); ast = match.get("away_stats", {}); h2h = match.get("h2h", [])
        home = match["home_team"]; away = match["away_team"]
        
        # 批量获取预测
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
        
        # 智能动态权重
        # 默认权重
        w = {"poisson": 0.16, "dixon": 0.12, "elo": 0.10, "bt": 0.04, "mc": 0.10, "bayes": 0.04, "rf": 0.13, "gb": 0.13, "nn": 0.10, "lr": 0.08}
        
        # 若有赔率介入，切分权重给市场赔率 (最聪明的数据)
        if oa:
            scale = 0.85
            for k in w: w[k] *= scale
            w["odds"] = 0.15
            
        models = [("poisson", poi), ("dixon", dc), ("elo", elo), ("bt", bt), ("mc", mc), ("bayes", bay), ("rf", rf), ("gb", gb), ("nn", nn), ("lr", lr)]
        hp = 0; dp = 0; ap = 0
        
        # 加权融合
        for name, pred in models:
            wt = w.get(name, 0.05)
            hp += pred.get("home_win", 33) * wt
            dp += pred.get("draw", 33) * wt
            ap += pred.get("away_win", 33) * wt
            
        # 引入近期状态偏置 (Form Bias)
        fd = hf["score"] - af["score"]
        hp += (fd * 0.08)  # 状态好，胜率微调上升
        ap -= (fd * 0.08)
        
        if "odds" in w:
            hp += oa.get("implied_home", 33) * w["odds"]
            dp += oa.get("implied_draw", 33) * w["odds"]
            ap += oa.get("implied_away", 33) * w["odds"]
            
        t = hp + dp + ap
        if t > 0: hp = round(hp / t * 100, 1); dp = round(dp / t * 100, 1); ap = round(100 - hp - dp, 1)
        
        # 共识度计算 (极度分化 vs 高度一致)
        agree_h = sum(1 for _, p in models if p.get("home_win", 0) > max(p.get("draw", 0), p.get("away_win", 0)))
        agree_a = sum(1 for _, p in models if p.get("away_win", 0) > max(p.get("home_win", 0), p.get("draw", 0)))
        consensus = max(agree_h, agree_a)
        
        cf = 30 + consensus * 5
        mx = max(hp, dp, ap)
        if mx > 60: cf += 12
        elif mx > 50: cf += 6
        if consensus >= 8: cf += 10 # 强共识奖励
        cf = min(95, max(30, cf))
        
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
