# A/B Compare — Original vs Current Main vs v18.1 Matrix Shadow

## 1. Executive Summary

本报告比较三套结果：
- A：原始昨天预测
- B：当前 main d7575c6 post-merge blind
- C：v18.1 matrix-only shadow，use_ai=False

结论必须保持克制：v18.1 matrix-only 并没有提升方向命中率，也没有提升比分命中率；它主要在总进球区间上相对更好。因此，v18.1 不应直接回滚或替换当前 main。若进入 PR #3，只能以 shadow matrix layer 的方式接入，用于旁路诊断和风险提示。

## 2. Metrics

| 版本 | 方向命中率 | 比分命中率 | 总进球区间命中率 | BTTS 命中率 |
|---|---:|---:|---:|---:|
| 原始昨天预测 | 37.5% | 0.0% | 25.0% | 37.5% |
| 当前 main d7575c6 | 25.0% | 0.0% | 37.5% | 50.0% |
| v18.1 matrix-only use_ai=False | 25.0% | 0.0% | 50.0% | 37.5% |

## 3. Interpretation

### 3.1 Direction Layer
v18.1 matrix-only 的方向命中率为 25.0%，与当前 main 相同，低于原始昨天预测的 37.5%。因此不能证明 v18.1 的本地矩阵单独能提升方向判断。
这说明方向问题的根因仍在上游：
- 1X2 概率解释不足
- 让球与 1X2 冲突处理不足
- 客胜/平局风险的校准不足
- AI 或矩阵都可能在弱主胜场景下继续偏向主队
- 缺少历史分桶校准

### 3.2 Exact Score Layer
三者比分命中率均为 0.0%。这说明当前样本中 exact score 的提升没有证据。不能把 v18.1 历史高命中归因于当前 blind 样本中的纯矩阵能力。

### 3.3 Goal Band Layer
v18.1 matrix-only 的总进球区间命中率达到 50.0%，优于当前 main 的 37.5% 和原始预测的 25.0%。这说明统一比分矩阵和 TTG/IPF 结构对总进球形态可能有价值。

### 3.4 BTTS Layer
当前 main 的 BTTS 命中率为 50.0%，高于 v18.1 matrix-only 的 37.5%。说明 v18.1 不能整体替换 main。main 的 BTTS 相关风控字段仍有保留价值。

## 4. Match-Level Focus: Celta vs Levante
- 实际赛果：塞尔塔 2-3 莱万特
- v18.1 matrix-only：2-1 主胜
- 当前 main：2-1 主胜，但 risk_score_candidates 包含 1-2 / 2-2 / 2-3

结论：
v18.1 matrix-only 没有解决该场方向和比分问题；当前 main 的 PR #2 至少把 2-3 纳入风险候选，但没有改变最终主判断。
因此，PR #3 不应让 v18.1 接管 final decision，而应只输出 matrix shadow fields。

## 5. What v18.1 Still Contributes
v18.1 最有价值的不是直接预测结果，而是以下结构：
1. `fair_probs_from_1x2`
2. `fair_probs_from_ttg`
3. `crs_implied_probabilities`
4. `analyze_crs_matrix`
5. `build_unified_score_matrix`
6. IPF 同时贴合 1X2 和 TTG
7. matrix-derived `goal_probs`
8. matrix-derived `top_scores`
9. matrix-derived lambda_h / lambda_a
10. trap residual 限幅思路

## 6. What Must Not Be Migrated Directly
1. 不直接替换 `scripts/predict.py`
2. 不直接让 v18.1 接管 final_direction
3. 不直接让 v18.1 接管 predicted_score
4. 不直接迁移所有 T1-T16 residual，避免和当前 main 风险字段重复计数
5. 不使用 v18.1 中过期 AI 模型名
6. 不使用“shin”命名误导，应改为 `power_fair_probs` 或 `shin_approx`
7. 不照搬联赛硬偏置
8. 不照搬“其他比分均分”作为最终概率，只可作为近似 shadow

## 7. PR #3 Recommendation
建议进入 PR #3，但范围必须收窄为：
`feat: add unified score matrix shadow layer`

PR #3 只输出 shadow fields，不改变 final decision。
建议新增字段：
- `matrix_direction_probs`
- `matrix_top_scores`
- `matrix_goal_probs`
- `matrix_lambda_home`
- `matrix_lambda_away`
- `matrix_shape_verdict`
- `matrix_recommended_score`
- `matrix_recommended_direction`
- `matrix_disagreement_flags`

触发风险标记：
- `matrix_vs_ai_direction_conflict`
- `matrix_high_goal_tail_conflict`
- `matrix_away_tail_warning`
- `matrix_draw_risk_warning`
- `matrix_low_confidence_warning`

## 8. Acceptance Criteria for PR #3
PR #3 必须满足：
1. 不改变 `predicted_score`
2. 不改变 `final_direction`
3. 不改变 `confidence`
4. 不改变 OpenClaw/Hermes 配置
5. 只增加 matrix shadow 输出字段
6. 新增测试验证 shadow fields 存在
7. 新增测试验证 final decision 不被 matrix 改写
8. 新增测试验证 v18.1 旧函数可独立运行
9. 新增报告说明 shadow layer 仅用于诊断

## 9. Conclusion
v18.1 matrix-only 没有证明其方向预测能力优于当前 main，因此不建议回滚或直接迁移为主决策层。
但 v18.1 的统一比分矩阵在总进球区间上有潜在价值，并且能提供独立于 AI 叙事的结构化旁路信号。因此建议进入 PR #3，采用 shadow matrix layer 方式接入当前 main。