# PR #3: unified score matrix shadow layer

## 1. 为什么不回滚 v18.1

本 PR 不回滚 v18.1，也不把 v18.1 作为主预测引擎。当前研究结论显示，v18.1 matrix-only 在方向与比分上不适合直接替换 main：方向命中率约 25.0%，比分命中率 0.0%，BTTS 约 37.5%。这些指标不足以支撑它接管 `final_direction`、`predicted_score` 或推荐置信度。

同时，v18.1 的统一比分矩阵对总进球区间有诊断价值：总进球区间命中率约 50.0%。因此更合理的处理方式是保留 main 的 AI-native 决策链，仅把矩阵作为旁路诊断信号接入。

## 2. 为什么只做 shadow layer

本 PR 的目标是观测矩阵信号，而不是改变线上决策。shadow layer 只输出独立诊断字段，用于后续离线评估：

- 不修改 `final_direction`
- 不修改 `predicted_score`
- 不修改 `confidence`
- 不修改 `result` / `display_direction`
- 不修改 `home_win_pct` / `draw_pct` / `away_win_pct`
- 不参与 Top4 排序、推荐门控或主决策路径

这样可以在不引入回归风险的情况下收集统一比分矩阵的方向概率、比分候选、总进球分布、lambda 与分歧告警。

## 3. 迁移了哪些函数

本 PR 没有整段复制 v18.1，只新增最小必要 shadow 版本函数：

- `fair_probs_from_1x2_shadow`
  - 从 1X2 赔率恢复去水公平概率，输出百分比。
- `fair_probs_from_ttg_shadow`
  - 从 `a0`-`a7` 总进球赔率恢复总进球分布。
- `crs_implied_probabilities_shadow`
  - 从正确比分赔率恢复 CRS implied probabilities。
- `build_unified_score_matrix_shadow`
  - 用 Poisson base + CRS 融合 + 1X2/TTG IPF 约束构建统一比分矩阵。
- `attach_matrix_shadow_fields`
  - 把矩阵诊断字段挂到 prediction dict 上，并显式保护主决策字段不被改写。

同时新增了若干仅供 shadow 函数使用的小型内部 helper，例如 `_normalize_prob_dict_shadow`、`_poisson_pmf_shadow`、`_matrix_moments_shadow`、`_matrix_shape_verdict_shadow`。

## 4. 没有迁移哪些函数

明确没有迁移：

- v18.1 的 AI 调用逻辑
- v18.1 的 Claude/GPT/Grok/Gemini 编排逻辑
- 完整 T1-T16 陷阱矩阵作为主决策层
- trap residual 对主方向/比分/置信度的调整
- EV / Kelly 接管逻辑
- 任何球队、日期、赛果硬编码逻辑
- 任何 OpenClaw / Hermes 配置修改

## 5. 新增字段说明

新增输出字段如下：

- `matrix_direction_probs`
  - shadow matrix 推导的 home/draw/away 概率，单位为百分比。
- `matrix_top_scores`
  - shadow matrix 排名前列的比分候选与概率。
- `matrix_goal_probs`
  - shadow matrix 的总进球分布，`0` 到 `7`，其中 `7` 表示 7+ bucket。
- `matrix_lambda_home`
  - shadow matrix 使用/推导的主队 lambda。
- `matrix_lambda_away`
  - shadow matrix 使用/推导的客队 lambda。
- `matrix_shape_verdict`
  - 矩阵形态诊断，如 `balanced`、`grinder`、`shootout`、`lopsided_h`、`lopsided_a`、`normal`、`unknown`。
- `matrix_recommended_score`
  - shadow matrix 的最高概率比分候选，仅供诊断。
- `matrix_recommended_direction`
  - `matrix_recommended_score` 对应方向，仅供诊断。
- `matrix_disagreement_flags`
  - 矩阵与主预测的分歧/风险标记。
- `matrix_shadow_error`
  - shadow 计算失败时记录错误；成功时为空字符串。

`matrix_disagreement_flags` 至少包含：

- `matrix_vs_final_direction_conflict`
- `matrix_high_goal_tail_conflict`
- `matrix_away_tail_warning`
- `matrix_draw_risk_warning`
- `matrix_low_confidence_warning`

## 6. 不接管 final decision 的证明

接入点在最终 `adapt_ai_to_frontend` 组装完 prediction dict 后调用 `attach_matrix_shadow_fields(pred, match_obj)`。

`attach_matrix_shadow_fields` 进入时会保存以下受保护字段：

- `predicted_score`
- `final_direction`
- `confidence`
- `result`
- `display_direction`
- `home_win_pct`
- `draw_pct`
- `away_win_pct`

shadow 字段写入完成或失败后，函数会把上述字段恢复为调用前的原值。因此即使矩阵推荐方向或比分与 AI-native 主预测不同，也只会体现在 `matrix_*` 诊断字段和 `matrix_disagreement_flags` 中，不会改变主决策输出。

如果 shadow 计算失败，只写入：

- `matrix_shadow_error`

主流程照常返回原 prediction。

## 7. 测试结果

本地环境没有安装 pytest：

```text
python3 -m pytest tests/test_review_20260512_regressions.py::test_shadow_1x2_fair_probs_sum_to_100 -q
/usr/bin/python3: No module named pytest
```

已执行等价的 Python 直接测试函数 runner，覆盖 `tests/test_review_20260512_regressions.py` 中全部测试函数：

```text
PASS test_async_ai_call_marks_invalid_json_as_parse_failed
PASS test_build_evidence_packet_runtime_function_has_no_prediction_leakage
PASS test_goal_band_high_score_tail_protection
PASS test_json_output_parser_handles_fenced_json_and_rejects_empty
PASS test_matrix_goal_probs_sum_to_100_and_top_scores_nonempty
PASS test_matrix_shadow_fields_exist_and_do_not_override_core_decision_fields
PASS test_matrix_shadow_source_has_no_team_date_or_result_hardcoding
PASS test_odds_overround_removal_normalizes_probabilities
PASS test_predicted_score_and_final_direction_are_closed_by_score
PASS test_shadow_1x2_fair_probs_sum_to_100
PASS test_tail_risk_protection_for_weak_home_favorite
```

并执行语法校验：

```text
python3 -m py_compile scripts/predict.py tests/test_review_20260512_regressions.py
# passed
```

新增测试覆盖：

1. matrix shadow 字段存在。
2. matrix shadow 不篡改 `predicted_score`。
3. matrix shadow 不篡改 `final_direction`。
4. matrix shadow 不篡改 `confidence`。
5. 1X2 shadow 去水概率总和约等于 100。
6. `matrix_goal_probs` 总和约等于 100。
7. `matrix_top_scores` 返回非空候选。
8. shadow 函数源码无球队/日期/赛果硬编码。

## 8. 已知风险

- shadow matrix 仍基于赔率反推与 Poisson/IPF 约束，不代表真实赛果概率。
- 正确比分赔率覆盖不足时，CRS 融合权重会降低，输出更依赖 1X2 与总进球约束。
- `matrix_recommended_score` 是诊断候选，不应直接用于投注或替代 AI-native 最终比分。
- `matrix_disagreement_flags` 当前是规则化告警，阈值需要继续用盲测数据校准。
- 若输入缺失 1X2、a0-a7 或 CRS 字段，shadow 输出可能退化，但主流程不会失败。

## 9. 下一步如何评估 matrix shadow 是否值得升级为风控层

建议至少用 2-4 周盲测数据离线评估 shadow 字段：

1. 记录每场 `matrix_direction_probs`、`matrix_top_scores`、`matrix_goal_probs` 与最终赛果。
2. 评估 `matrix_goal_probs` 的总进球 bucket 命中率，尤其 0-2、3、4+ 的区间能力。
3. 统计 `matrix_vs_final_direction_conflict=true` 时，main 与 matrix 哪一方更接近赛果。
4. 统计 `matrix_high_goal_tail_conflict`、`matrix_away_tail_warning`、`matrix_draw_risk_warning` 对错判场次的提前预警能力。
5. 如果 shadow 在风险告警上稳定提升召回率，下一阶段可考虑升级为“风控提示层”，但仍不直接接管 `final_direction` / `predicted_score`。
6. 只有当连续盲测证明 matrix 对特定风险类型有稳定增益，才考虑把它纳入 recommendation downgrade，而不是纳入主方向或比分裁决。
