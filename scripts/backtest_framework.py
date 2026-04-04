#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
足球预测回测框架 v2.0（专为 vMAX 7.0 设计 | 数学逻辑彻底修复版）
修复: 蒙特卡洛置信区间坍塌Bug、夏普比率跨领域错用Bug、空比分TypeError崩溃
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm

# ====================== 原有模块导入 ======================
# 注意：请确保这些模块在你的项目根目录下
from main import run_predictions
from predict import KellyCriterion

# 实例化 Kelly 计算器
kelly_calc = KellyCriterion()

def calculate_value_bet(prob_pct, odds):
    """提取的辅助函数，适配你的 KellyCriterion 返回格式"""
    prob = prob_pct / 100.0
    return kelly_calc.calculate(prob, odds, fraction=1.0) # 这里取全凯利，下面再打折


# ====================== 回测核心函数 ======================
def run_backtest(
    historical_json_path: str,          # 历史比赛JSON路径
    output_dir: str = "backtest_results",
    confidence_threshold: float = 55.0, # 只投注置信度 >= 此值
    kelly_fraction: float = 0.25,       # Kelly仓位折扣（1/4 Kelly）
    use_ai: bool = True,                # 是否开启多模态AI推理
    min_odds: float = 1.60,             # 最低赔率过滤
    top4_only: bool = True              # 只回测推荐场次
):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载历史数据
    if not os.path.exists(historical_json_path):
        print(f"❌ 找不到历史文件: {historical_json_path}")
        return None, None, ""
        
    with open(historical_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    matches = raw.get("matches", [])
    print(f"🚀 开始回测 {len(matches)} 场历史比赛...")

    results = []
    equity = 1000.0          # 初始本金 1000 单位
    equity_curve = [equity]
    bet_history = []

    for m in tqdm(matches, desc="回测进度"):
        match_id = m.get("id", m.get("match_num", "未知"))
        home, away = m.get("home_team"), m.get("away_team")
        # ★ 强制字符串转换防御 KeyError/TypeError
        actual_score = str(m.get("actual_score", "0-0")) 
        actual_result = m.get("actual_result")        
        sp_h = float(m.get("sp_home", 0) or 0)
        sp_d = float(m.get("sp_draw", 0) or 0)
        sp_a = float(m.get("sp_away", 0) or 0)

        if not actual_score or actual_score == "0-0" and not actual_result:
            continue

        # 运行完整预测引擎
        _, top4 = run_predictions({"matches": [m]}, use_ai=use_ai)
        pred = top4[0]["prediction"] if top4 and top4_only else \
               next((x["prediction"] for x in [{"prediction": m.get("prediction", {})}] if m.get("prediction")), None)

        if not pred:
            continue

        pred_score = str(pred.get("predicted_score", "0-0"))
        conf = pred.get("confidence", 0)
        result_pred = pred.get("result")
        fused_hp = pred.get("home_win_pct", 33)
        fused_dp = pred.get("draw_pct", 33)
        fused_ap = pred.get("away_win_pct", 33)
        risk = pred.get("risk_level")

        # ====================== 投注决策 ======================
        stake = 0.0
        profit = 0.0
        ev = 0.0

        if conf >= confidence_threshold:
            val_h = calculate_value_bet(fused_hp, sp_h)
            val_d = calculate_value_bet(fused_dp, sp_d)
            val_a = calculate_value_bet(fused_ap, sp_a)

            best_val = max(
                (val_h, "home", sp_h, fused_hp),
                (val_d, "draw", sp_d, fused_dp),
                (val_a, "away", sp_a, fused_ap),
                key=lambda x: x[0].get("ev", 0) if x[0] else 0
            )

            if best_val[0] and best_val[0].get("is_value") and best_val[2] >= min_odds:
                bet_type = best_val[1]
                odds = best_val[2]
                prob = best_val[3] / 100.0
                ev = best_val[0]["ev"]

                # Quarter Kelly 动态仓位管理
                kelly = max(0.0, best_val[0]["kelly"] / 100 * kelly_fraction)
                stake = equity * kelly
                if stake < 10: stake = 0  

                if stake > 0:
                    # 结算利润
                    if (bet_type == "home" and actual_result == "home") or \
                       (bet_type == "draw" and actual_result == "draw") or \
                       (bet_type == "away" and actual_result == "away"):
                        profit = stake * (odds - 1)
                    else:
                        profit = -stake

                    equity += profit
                    bet_history.append({
                        "match": f"{home} vs {away}",
                        "date": m.get("match_num", ""),
                        "bet_type": bet_type,
                        "odds": odds,
                        "stake": round(stake, 2),
                        "profit": round(profit, 2),
                        "new_equity": round(equity, 2),
                        "pred_score": pred_score,
                        "actual_score": actual_score,
                        "confidence": conf,
                        "ev": round(ev, 2)
                    })

        # ====================== 记录每场比赛指标 ======================
        exact_hit = (pred_score == actual_score)
        win_hit = (result_pred == actual_result) if result_pred else False
        total_goals_pred = sum(int(x) for x in pred_score.split("-")) if "-" in pred_score else 2
        total_goals_actual = sum(int(x) for x in actual_score.split("-")) if "-" in actual_score else 2
        over_hit = (total_goals_actual > 2.5) == (pred.get("over_under_2_5") == "大")

        results.append({
            "match_id": match_id,
            "home": home,
            "away": away,
            "pred_score": pred_score,
            "actual_score": actual_score,
            "pred_result": result_pred,
            "actual_result": actual_result,
            "confidence": round(conf, 1),
            "risk": risk,
            "exact_hit": exact_hit,
            "win_hit": win_hit,
            "over_hit": over_hit,
            "btts_hit": (min(total_goals_actual,1) and min(total_goals_pred,1)) == (pred.get("both_score") == "是"),
            "equity_after": round(equity, 2),
            "stake": round(stake, 2),
            "profit": round(profit, 2),
            "ev": round(ev, 2)
        })

        equity_curve.append(equity)

    # ====================== 生成报告与统计 ======================
    df = pd.DataFrame(results)
    df_bet = pd.DataFrame(bet_history)

    total_matches = len(df)
    exact_acc = df["exact_hit"].mean() * 100 if total_matches > 0 else 0
    win_acc = df["win_hit"].mean() * 100 if total_matches > 0 else 0
    over_acc = df["over_hit"].mean() * 100 if total_matches > 0 else 0
    btts_acc = df["btts_hit"].mean() * 100 if total_matches > 0 else 0

    # ★ 彻底修复的统计指标逻辑 ★
    if len(df_bet) > 0:
        total_stake = df_bet["stake"].sum()
        total_profit = df_bet["profit"].sum()
        roi = (total_profit / total_stake * 100) if total_stake > 0 else 0
        yield_ = roi  # 足球Yield等于ROI
        win_rate = (df_bet["profit"] > 0).mean() * 100
        
        # 最大回撤计算
        cum_max = pd.Series(equity_curve).cummax()
        max_drawdown = ((cum_max - pd.Series(equity_curve)) / cum_max).max() * 100
        
        # 修正版单注夏普比率 (Per-Bet Sharpe)
        if total_stake > 0 and df_bet["profit"].std() > 0:
            bet_returns = df_bet["profit"] / df_bet["stake"]
            sharpe = bet_returns.mean() / bet_returns.std()
        else:
            sharpe = 0.0
    else:
        roi = yield_ = win_rate = max_drawdown = sharpe = 0
        total_stake = total_profit = 0

    # ★ 彻底修复的 Monte Carlo 模拟 (NumPy 矩阵极速版) ★
    if len(df_bet) > 30:
        mc_returns = []
        profits_array = df_bet["profit"].values
        stakes_array = df_bet["stake"].values
        num_bets = len(df_bet)

        for _ in range(1000):
            # 有放回随机抽样
            sample_indices = np.random.choice(num_bets, size=num_bets, replace=True)
            sampled_profit = profits_array[sample_indices].sum()
            sampled_stake = stakes_array[sample_indices].sum()
            
            if sampled_stake > 0:
                mc_returns.append((sampled_profit / sampled_stake) * 100)
            else:
                mc_returns.append(0)
                
        mc_mean = np.mean(mc_returns)
        mc_95_low = np.percentile(mc_returns, 5)
        mc_95_high = np.percentile(mc_returns, 95)
    else:
        mc_mean = mc_95_low = mc_95_high = 0

    # ====================== 输出报告 ======================
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║                 vMAX 足球量化回测报告（{datetime.now().strftime('%Y-%m-%d')}）                
║══════════════════════════════════════════════════════════════║
║ 总场次          : {total_matches} 场
║ 精确比分命中率  : {exact_acc:.2f}%   
║ 胜平负命中率    : {win_acc:.2f}%
║ 大小球命中率    : {over_acc:.2f}%
║ BTTS命中率      : {btts_acc:.2f}%
║ 投注次数        : {len(df_bet)} 次
║ 总投注          : {total_stake:.0f} 单位
║ 总盈利          : {total_profit:.0f} 单位
║ ROI（回报率）   : {roi:.2f}%
║ Yield（场均收益）: {yield_:.2f}%
║ 投注胜率        : {win_rate:.2f}%
║ 最大回撤        : {max_drawdown:.2f}%
║ 夏普比率        : {sharpe:.4f}  ← (真实 Per-Bet Sharpe)
║ Monte Carlo 95% 置信区间 : [{mc_95_low:.1f}% ~ {mc_95_high:.1f}%]
╚══════════════════════════════════════════════════════════════╝
"""
    print(report)

    # 导出结果
    df.to_csv(f"{output_dir}/full_results.csv", index=False, encoding="utf-8-sig")
    if len(df_bet) > 0:
        df_bet.to_csv(f"{output_dir}/bet_history.csv", index=False, encoding="utf-8-sig")
    
    # 绘制真实资金曲线
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="Equity Curve", color="#00ff00", linewidth=2)
    plt.title("Quarter Kelly 资金复利曲线 (初始 1000 单位)")
    plt.xlabel("投注场次")
    plt.ylabel("资金 (单位)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{output_dir}/equity_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    if len(df) > 0:
        print("\n分置信度表现：")
        for thresh in [50, 60, 70, 80]:
            sub = df[df["confidence"] >= thresh]
            if len(sub) > 5:
                print(f"  ≥{thresh}% → 胜率 {sub['win_hit'].mean()*100:.1f}% | 场次 {len(sub)}")

    print(f"\n✅ 回测引擎执行完毕！报告及资金曲线已保存至 {output_dir}/ 目录。")
    return df, df_bet, report

if __name__ == "__main__":
    # 直接运行即可进行测试
    run_backtest(
        historical_json_path="historical_matches.json", 
        confidence_threshold=58,      
        kelly_fraction=0.25,
        use_ai=True,                  
        top4_only=True
    )
