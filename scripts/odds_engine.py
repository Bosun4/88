import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson as pdist

class BookmakerXGSolver:
    """终极武器：逆向破解庄家预设的期望进球 xG"""
    def solve_implied_xg(self, ph, pd, pa):
        def loss(vars):
            lh, la = vars[0], vars[1]
            if lh <= 0.1 or la <= 0.1: return 1e6
            hw = dr = aw = 0.0
            # 截断泊松分布计算到8球，足够覆盖99.9%的足球赛事
            for i in range(9):
                for j in range(9):
                    p = pdist.pmf(i, lh) * pdist.pmf(j, la)
                    if i > j: hw += p
                    elif i == j: dr += p
                    else: aw += p
            return (hw - ph)**2 + (dr - pd)**2 + (aw - pa)**2

        # 初始猜测与边界
        initial_guess = [1.35, 1.15]
        bounds = [(0.2, 4.5), (0.2, 4.5)]
        
        try:
            res = minimize(loss, initial_guess, bounds=bounds, method='L-BFGS-B', options={'ftol': 1e-7})
            return round(res.x[0], 3), round(res.x[1], 3)
        except Exception as e:
            return 1.35, 1.15

def predict_match(match_data):
    """
    Odds Engine 主入口
    核心逻辑：反推机构隐藏意图，严谨推演比分矩阵
    """
    sp_h = float(match_data.get("sp_home", 0) or 0)
    sp_d = float(match_data.get("sp_draw", 0) or 0)
    sp_a = float(match_data.get("sp_away", 0) or 0)
    
    # 1. 深度去水，计算市场隐含的绝对概率
    if sp_h > 1 and sp_d > 1 and sp_a > 1:
        imp = 1.0 / np.array([sp_h, sp_d, sp_a])
        margin = imp.sum() - 1.0
        z = margin / (1 + margin)
        shin = (imp - z * imp**2) / (1 - z)
        shin /= shin.sum()
        ph, pd, pa = shin[0], shin[1], shin[2]
    else:
        ph, pd, pa = 0.40, 0.28, 0.32

    # 2. 启动核心反推求解器，撕开盘口伪装
    solver = BookmakerXGSolver()
    i_hxg, i_axg = solver.solve_implied_xg(ph, pd, pa)
    
    # 3. 🚨 真正的泊松比分矩阵推演 (绝对不再偷工减料！)
    hw = dr = aw = bt = o25 = 0.0
    scores = []
    for i in range(9):
        for j in range(9):
            p = pdist.pmf(i, i_hxg) * pdist.pmf(j, i_axg)
            if i > j: hw += p
            elif i == j: dr += p
            else: aw += p
            if i > 0 and j > 0: bt += p
            if i + j > 2: o25 += p
            scores.append((i, j, p))
            
    # 按发生概率从大到小严格排序
    scores.sort(key=lambda x: x[2], reverse=True)
    
    # 归一化胜负平概率
    t = hw + dr + aw
    if t > 0: hw /= t; dr /= t; aw /= t
    
    # 提取真实概率最高的前 3 个比分
    top3 = [f"{s[0]}-{s[1]}" for s in scores[:3]]
    primary_score = top3[0]
    
    # 4. 剪刀差探测 (Scissors Gap) - 寻找价值偏离点
    hs = match_data.get("home_stats", {})
    try: 
        real_hxg = float(hs.get("avg_goals_for", 1.3))
    except: 
        real_hxg = 1.3

    gap_h = real_hxg - i_hxg
    gap_sig = ""
    if gap_h > 0.45 and ph < 0.48:
        gap_sig = "🚨 真实攻击力大幅碾压机构预期 (深水价值)"
    elif gap_h < -0.45 and ph > 0.60:
        gap_sig = "🚨 机构强开深盘虚构主队战力 (诱盘警告)"

    return {
        "primary_score": primary_score,
        "top3_scores": top3,
        "home_prob": round(hw * 100, 1),
        "draw_prob": round(dr * 100, 1),
        "away_prob": round(aw * 100, 1),
        "expected_goals": round(i_hxg + i_axg, 2),
        "bookmaker_implied_home_xg": i_hxg,
        "bookmaker_implied_away_xg": i_axg,
        "scissors_gap_signal": gap_sig,
        "over_25": round(o25 * 100, 1),
        "btts": round(bt * 100, 1),
        "confidence": 60 + int(hw * 20 if hw > aw else aw * 20),
        "direction": "主胜" if hw > aw else "客胜",
        "direction_confidence": f"{max(hw, aw)*100:.1f}%",
        "reason": gap_sig if gap_sig else "赔率底牌反推正常"
    }

def build_ai_context(engine_result):
    """提取给 AI 专用的精简上下文"""
    return (
        f"Bookmaker Implied xG: Home {engine_result['bookmaker_implied_home_xg']} vs Away {engine_result['bookmaker_implied_away_xg']}. "
        f"Gap Signal: {engine_result['scissors_gap_signal']}. "
        f"Expected Total Goals: {engine_result['expected_goals']}."
    )


