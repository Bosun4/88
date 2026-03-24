import numpy as np
import math
from scipy.optimize import minimize
from scipy.stats import poisson as pdist

class BookmakerXGSolver:
    """顶级量化核心：牛顿-拉弗森逼近法反解庄家真实预期进球 (Implied xG)"""
    def __init__(self):
        pass

    def solve_implied_xg(self, true_home_prob, true_draw_prob, true_away_prob):
        """通过无水真实胜平负概率，逆向求解庄家预设的的主客队 xG"""
        # 目标函数：寻找一组 (lh, la) 使得泊松分布推算出的胜平负概率最接近庄家给出的真实概率
        def loss(vars):
            lh, la = vars[0], vars[1]
            if lh <= 0.1 or la <= 0.1: 
                return 1e6 # 边界惩罚
                
            hw, dr, aw = 0.0, 0.0, 0.0
            # 截断泊松分布计算到7球足够覆盖99.9%的情况
            for i in range(8):
                for j in range(8):
                    p = pdist.pmf(i, lh) * pdist.pmf(j, la)
                    if i > j: hw += p
                    elif i == j: dr += p
                    else: aw += p
                    
            # 最小二乘法损失函数
            return (hw - true_home_prob)**2 + (dr - true_draw_prob)**2 + (aw - true_away_prob)**2

        # 初始猜测：通常联赛主队1.3，客队1.1
        initial_guess = [1.3, 1.1]
        bounds = [(0.2, 4.0), (0.2, 4.0)]
        
        try:
            res = minimize(loss, initial_guess, bounds=bounds, method='L-BFGS-B')
            implied_lh, implied_la = res.x[0], res.x[1]
            return round(implied_lh, 2), round(implied_la, 2)
        except:
            return 1.3, 1.1

def predict_match(match_data):
    """
    Odds Engine 主入口
    整合反向求解器，提取“剪刀差洼地”
    """
    sp_h = float(match_data.get("sp_home", 0) or 0)
    sp_d = float(match_data.get("sp_draw", 0) or 0)
    sp_a = float(match_data.get("sp_away", 0) or 0)
    
    # 1. 计算去水真实概率 (正规体彩返还率偏低，采用平方去水近似)
    if sp_h > 1 and sp_d > 1 and sp_a > 1:
        imp = 1.0 / np.array([sp_h, sp_d, sp_a])
        margin = imp.sum() - 1.0
        z = margin / (1 + margin)
        shin = (imp - z * imp**2) / (1 - z)
        shin /= shin.sum()
        ph, pd, pa = shin[0], shin[1], shin[2]
    else:
        ph, pd, pa = 0.4, 0.28, 0.32

    # 2. 调用核心反推求解器
    solver = BookmakerXGSolver()
    implied_hxg, implied_axg = solver.solve_implied_xg(ph, pd, pa)
    
    # 3. 计算常规泊松概率生成候选比分
    hw = dr = aw = bt = o25 = 0.0
    scores = []
    for i in range(8):
        for j in range(8):
            p = pdist.pmf(i, implied_hxg) * pdist.pmf(j, implied_axg)
            if i > j: hw += p
            elif i == j: dr += p
            else: aw += p
            if i > 0 and j > 0: bt += p
            if i + j > 2: o25 += p
            scores.append((i, j, p))
            
    scores.sort(key=lambda x: x[2], reverse=True)
    t = hw + dr + aw
    if t > 0: hw /= t; dr /= t; aw /= t
    
    top3 = ["%d-%d" % (s[0], s[1]) for s in scores[:3]]
    primary_score = top3[0]
    
    # 4. 剪刀差探测 (Scissors Gap)
    hs = match_data.get("home_stats", {})
    ast = match_data.get("away_stats", {})
    try: 
        real_hgf = float(hs.get("avg_goals_for", 1.3))
        real_agf = float(ast.get("avg_goals_for", 1.1))
    except: 
        real_hgf, real_agf = 1.3, 1.1

    gap_h = real_hgf - implied_hxg
    gap_a = real_agf - implied_axg
    
    gap_signal = ""
    if gap_h > 0.4 and ph < 0.5:
        gap_signal = "🚨 真实主攻击力大幅碾压庄家预期 (价值洼地)"
    elif gap_h < -0.4 and ph > 0.6:
        gap_signal = "🚨 庄家强开深盘虚诱主胜 (防冷警告)"

    return {
        "primary_score": primary_score,
        "top3_scores": top3,
        "home_prob": round(hw * 100, 1),
        "draw_prob": round(dr * 100, 1),
        "away_prob": round(aw * 100, 1),
        "expected_goals": round(implied_hxg + implied_axg, 2),
        "bookmaker_implied_home_xg": implied_hxg,
        "bookmaker_implied_away_xg": implied_axg,
        "scissors_gap_signal": gap_signal,
        "over_25": round(o25 * 100, 1),
        "btts": round(bt * 100, 1),
        "confidence": 60 + int(hw * 20 if hw > aw else aw * 20),
        "direction": "主胜" if hw > aw else "客胜",
        "direction_confidence": f"{max(hw, aw)*100:.1f}%",
        "reason": gap_signal if gap_signal else "赔率反推正常"
    }

def build_ai_context(engine_result):
    """提取给 AI 专用的精简上下文"""
    return (
        f"Bookmaker Implied xG: Home {engine_result['bookmaker_implied_home_xg']} vs Away {engine_result['bookmaker_implied_away_xg']}. "
        f"Gap Signal: {engine_result['scissors_gap_signal']}. "
        f"Expected Total Goals: {engine_result['expected_goals']}."
    )
