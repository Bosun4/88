# live_predictions_9999_github_actions 最新审计报告

审计对象：`reports/live_predictions/live_predictions_9999_github_actions.json`
生成时间：2026-05-14 00:09:13
输出版本：MAX-v1.1
范围：today_only
审计限制：未修改代码、未 push、未创建 PR、未重新运行预测、未调用 AI API 重新预测、未打印任何 key/token。

补充说明：用户要求读取的 `/root/.hermes/memories/projects/vmax88/PROJECT.md` 与 `/root/.hermes/memories/projects/vmax88/model_notes/next_research.md` 在当前环境不存在；已用文件系统检索确认 `/root/.hermes/memories` 下也未找到对应文件。因此本报告基于实际可读取的预测 JSON、`scripts/predict.py`、`tests/test_review_20260512_regressions.py`、`reports/pr3_unified_score_matrix_shadow.md` 完成。

## 1. 本次预测是否真实生产还是 mock/smoke

结论：本次是 GitHub Actions 产出的真实 AI-native 预测结果，不是 mock/smoke。

证据：

- 预测文件内每场 `ai_call_status` 均显示：
  - `gpt.phase1.status = ok`，model = `gpt-5.4`
  - `grok.phase1.status = ok`，model = `grok-4.3`
  - `grok.critic.status = ok`
  - `gemini.final.status = ok`，model = `gemini-3.1-pro-preview-thinking-high`
  - `gpt.consistency.status = ok`
- 每场 `mock_mode = false`。
- 未出现 `mock_ok`、`AI_MOCK_MODE_no_real_web`、`smoke`、`unit-test` 等 mock/smoke 输出标记。
- 每场 `decision_source = ai_native:gemini:final`，不是本地 mock 规则。
- 注意：`gpt.critic` 每场为 `parse_failed`，但这只表示 GPT critic 阶段解析失败；Grok critic 与 Gemini final 成功，最终预测不是 mock。

## 2. 总场数

- 总场数：15 场
- `top4` 场数：1 场
- `recommend_gate_pass = true`：1 场，即周四005 皇马 vs 奥维耶多
- 其余 14 场均未通过推荐 gate，多数因 PR #2 / v20.3 风控字段下调为 C 级或风险较高。

## 3. 字段完整性

结论：核心预测字段、PR #2 风险字段、PR #3 matrix shadow 字段在 15 场中均完整存在；未发现字段缺失或类型错误。

已核对核心字段：

- `predicted_score`
- `final_direction`
- `result`
- `display_direction`
- `score_implied_direction`
- `home_win_pct` / `draw_pct` / `away_win_pct`
- `confidence`
- `recommendation`
- `recommendation_tier`
- `recommend_gate_pass`
- `risk_level`
- `risk_score_candidates`
- `tail_risk_flags`
- `confidence_downgrade_reason`
- `validation_warnings`
- `decision_source`
- `engine_version`

已核对 PR #3 matrix 字段：

- `matrix_direction_probs`
- `matrix_top_scores`
- `matrix_goal_probs`
- `matrix_lambda_home`
- `matrix_lambda_away`
- `matrix_shape_verdict`
- `matrix_recommended_score`
- `matrix_recommended_direction`
- `matrix_disagreement_flags`
- `matrix_shadow_error`

类型检查结果：

- `risk_score_candidates`：15/15 为 list
- `tail_risk_flags`：15/15 为 list
- `matrix_top_scores`：15/15 为 list
- `matrix_disagreement_flags`：15/15 为 dict
- `matrix_shadow_error`：15/15 为空字符串

## 4. PR #1 是否生效

结论：PR #1 已真实生效。

从代码与最新输出看，PR #1 的关键目标是让 `predicted_score`、`final_direction`、`result` 闭环一致，并避免 evidence packet 泄漏旧预测字段。

审计结果：

- 15/15 场 `predicted_score` 隐含方向与 `final_direction` 一致。
- 15/15 场 `final_direction` 与中文 `result` 一致。
- 未发现 `predicted_score / final_direction / result` 自相矛盾。
- `tests/test_review_20260512_regressions.py` 中 `test_predicted_score_and_final_direction_are_closed_by_score` 覆盖了方向修复：当 AI 返回 `final_direction=home` 但比分为 `1-2` 时，本地协议会修正为 `away` 并写入 warning。
- `test_build_evidence_packet_runtime_function_has_no_prediction_leakage` 覆盖 evidence compiler 不把旧 `prediction` / `predicted_score` 放入 prompt evidence。
- `scripts/predict.py` 的 `build_evidence_packet` 返回 `evidence_compiler_version = v20.3.0_sharp_cluster_full`，且说明只编译事实，不输出 `predicted_score`，不做本地足球裁决。

## 5. PR #2 风险候选字段是否生效

结论：PR #2 已真实生效，而且在最新预测中影响明显。

生效证据：

- 15/15 场均有 `risk_score_candidates`。
- 15/15 场均有 `tail_risk_flags`。
- 15/15 场均有 `confidence_downgrade_reason`。
- 至少 3 场触发明确的 `weak_home_favorite_btts_tail_protection_applied`：
  - 周三003 比利亚雷 vs 塞维利亚
  - 周三010 赫塔费 vs 马洛卡
  - 周四004 赫罗纳 vs 皇家社会
- 弱主胜尾部保护成功把 `1-2`、`2-2`、`2-3` 等风险比分保留到 `risk_score_candidates`，并在 `tail_risk_flags` 中写入：
  - `weak_home_favorite_btts_tail`
  - `away_win_not_negligible`
  - `protect_1_2_2_2_2_3_tail`
- 最新输出中多数场次 `recommendation_tier = C`，反映风险/组件降级已经进入前端输出对象，而不是只停留在 prompt schema。

## 6. PR #3 matrix shadow 字段是否生效

结论：PR #3 已真实生效，并且没有接管最终预测。

生效证据：

- 15/15 场均有完整 `matrix_*` 字段。
- 15/15 场 `matrix_shadow_error = ""`。
- 15/15 场 `matrix_top_scores` 非空。
- 15/15 场 `matrix_direction_probs` 有 home/draw/away。
- 15/15 场 `matrix_goal_probs` 有 0-7 bucket。
- 3 场出现 `matrix_vs_final_direction_conflict = true`，说明 shadow 层确实独立计算，而不是复制最终预测。
- `reports/pr3_unified_score_matrix_shadow.md` 明确约束 matrix shadow 不修改：
  - `final_direction`
  - `predicted_score`
  - `confidence`
  - `result` / `display_direction`
  - `home_win_pct` / `draw_pct` / `away_win_pct`
- `scripts/predict.py` 的 `attach_matrix_shadow_fields` 会保存并恢复上述 protected 字段，因此 matrix 只作为诊断层。

## 7. 是否存在 predicted_score / final_direction / result 不一致

结论：不存在。

逐场核对结果：15/15 一致。

- `predicted_score = 2-1` 的场次均为 `final_direction = home` / `result = 主胜`
- `predicted_score = 1-1` 的场次均为 `final_direction = draw` / `result = 平局`
- `predicted_score = 0-1 / 0-2 / 1-2` 的场次均为 `final_direction = away` / `result = 客胜`
- `predicted_score = 3-0` 的场次均为 `final_direction = home` / `result = 主胜`

## 8. 是否存在 matrix_shadow_error

结论：不存在。

- 15/15 场 `matrix_shadow_error` 为空字符串。
- 未发现 shadow 计算失败或异常降级。

## 9. HIGH_RISK / MATRIX_CONFLICT / TAIL_RISK 场次清单

### HIGH_RISK

1. 周三004 西甲 西班牙人 vs 毕尔巴鄂
2. 周三005 法甲 布雷斯特 vs 斯特拉斯
3. 周三008 法甲 朗斯 vs 巴黎圣曼
4. 周三009 西甲 阿拉维斯 vs 巴萨
5. 周三011 美职 辛辛那提 vs 迈国际
6. 周四004 西甲 赫罗纳 vs 皇家社会

### MATRIX_CONFLICT

以下场次 `matrix_recommended_direction` 与 `final_direction` 冲突：

1. 周三004 西甲 西班牙人 vs 毕尔巴鄂
   - final：客胜 0-1
   - matrix：平局 1-1
   - flags：`matrix_vs_final_direction_conflict`, `matrix_draw_risk_warning`

2. 周三006 英超 曼城 vs 水晶宫
   - final：主胜 3-0
   - matrix：平局 2-2
   - flags：`matrix_vs_final_direction_conflict`, `matrix_away_tail_warning`, `matrix_draw_risk_warning`

3. 周三010 西甲 赫塔费 vs 马洛卡
   - final：主胜 1-0
   - matrix：平局 1-1
   - flags：`matrix_vs_final_direction_conflict`, `matrix_away_tail_warning`, `matrix_draw_risk_warning`

### TAIL_RISK

15/15 场均存在尾部风险候选或降级原因，因此都应在 UI 上展示风险入口：

1. 周三003 比利亚雷 vs 塞维利亚
2. 周三004 西班牙人 vs 毕尔巴鄂
3. 周三005 布雷斯特 vs 斯特拉斯
4. 周三006 曼城 vs 水晶宫
5. 周三007 拉齐奥 vs 国际米兰
6. 周三008 朗斯 vs 巴黎圣曼
7. 周三009 阿拉维斯 vs 巴萨
8. 周三010 赫塔费 vs 马洛卡
9. 周三011 辛辛那提 vs 迈国际
10. 周三012 西雅图 vs 圣何塞
11. 周四001 巴伦西亚 vs 巴列卡诺
12. 周四002 达曼协定 vs 吉达联合
13. 周四003 胡巴卡德 vs 拉斯决心
14. 周四004 赫罗纳 vs 皇家社会
15. 周四005 皇马 vs 奥维耶多

## 10. 重点可疑场次

### A. 周三006 曼城 vs 水晶宫

- final：主胜 3-0，confidence 75，risk_level low
- matrix：2-2 / draw
- matrix probabilities：home 33.38 / draw 33.04 / away 33.58
- matrix 4+ goal tail：55.53%
- active flags：`matrix_vs_final_direction_conflict`, `matrix_away_tail_warning`, `matrix_draw_risk_warning`

审计判断：这是最需要前端提示的场次。最终预测很强势，但 matrix 完全不支持单边主胜，且高比分尾部极高。即使 matrix 不接管 final，也必须让用户看到“shadow 与 final 强冲突”。

### B. 周三010 赫塔费 vs 马洛卡

- final：主胜 1-0，confidence 66
- matrix：1-1 / draw
- matrix probabilities：home 42.23 / draw 31.96 / away 25.80
- risk candidates：1-1、0-0、2-0、0-1、1-2、2-2、2-3
- active flags：`matrix_vs_final_direction_conflict`, `matrix_away_tail_warning`, `matrix_draw_risk_warning`

审计判断：PR #2 与 PR #3 同时提示风险。final 是一球小胜，但 draw/away 尾部很明显，应在 UI 上标为高注意，而不是普通主胜卡片。

### C. 周三004 西班牙人 vs 毕尔巴鄂

- final：客胜 0-1，confidence 55，risk_level high
- matrix：1-1 / draw
- matrix draw prob：31.14
- active flags：`matrix_vs_final_direction_conflict`, `matrix_draw_risk_warning`

审计判断：客胜优势很弱，1-1 是 matrix top score。前端应突出“final 客胜，但 matrix 首选平局”。

### D. 周三005 布雷斯特 vs 斯特拉斯

- final：平局 1-1，confidence 45，risk_level high
- matrix：1-1 / draw，但 matrix away prob 35.70 高于 draw 30.14
- active flags：`matrix_away_tail_warning`, `matrix_low_confidence_warning`

审计判断：虽然 matrix top score 与 final 一致，但方向概率显示 away tail 强，低信心标记应被显著提示。

### E. 周四004 赫罗纳 vs 皇家社会

- final：主胜 2-1，confidence 55，risk_level high
- matrix：2-1 / home，但 top2 `1-1` 概率 10.767 与 `2-1` 概率 10.801 几乎持平
- PR #2 弱主胜尾部触发，包含 1-2、2-2、2-3

审计判断：matrix 不冲突但极接近，PR #2 的尾部风险是关键。前端不应只显示“主胜 2-1”。

## 11. 前端 UI 当前可能的问题

如果当前 UI 仍以旧字段为主，可能存在以下问题：

1. 只展示 `predicted_score`、`result`、`confidence`，用户看不到 PR #2 的 `risk_score_candidates` 与 `tail_risk_flags`。
2. 只展示 final 方向，不展示 matrix shadow，会漏掉周三004、周三006、周三010 这类 matrix 冲突。
3. 若直接把 `matrix_recommended_score` 与 `predicted_score` 平级显示，用户可能误以为 matrix 是最终预测。
4. 若把 `matrix_top_scores` 全量展开，会让卡片过长；若完全不展示，又浪费 PR #3 的诊断价值。
5. 若直接展示 `matrix_disagreement_flags` 的英文 boolean，会降低可读性。
6. `confidence_downgrade_reason` 很长，直接放在卡片主体会挤压主信息；完全隐藏则用户看不到为什么 C 级/低信心。
7. `risk_score_candidates` 与 `top_score_candidates` / `matrix_top_scores` 容易混淆：一个是 final 风险候选，一个是 AI top 候选，一个是 matrix shadow 候选，UI 必须明确分区。
8. 当前只有 1 场 gate pass；如果 UI 仍按 top4 固定显示 4 场，可能会误导用户以为缺失或排序异常。应显示“本轮仅 1 场通过推荐 gate”。

## 12. 前端 UI 优化建议

### 12.1 卡片顶部如何显示 final 与 matrix

建议卡片顶部采用“双层但不平权”的结构：

- 主标题区域只放最终预测：
  - `FINAL：主胜 2-1`
  - `confidence 60`
  - `tier C`
  - `risk_level medium/high`
- 旁路诊断区域放 matrix：
  - `Matrix shadow：2-1 / 主胜`
  - 小字固定说明：`诊断信号，不是最终预测`
- 当 `matrix_vs_final_direction_conflict = true` 时，在顶部显示醒目但非覆盖式标签：
  - `Matrix disagrees`
  - `Final 客胜 vs Matrix 平局`
- 不要把 matrix score 直接放在 final score 旁边写成 `2-1 / 1-1`，否则用户会误解成双预测。

### 12.2 风险标签如何显示

建议把风险标签分三类：

1. AI 风险：来自 `risk_level`、`recommendation.risk_tags`
2. Tail 风险：来自 `tail_risk_flags`
3. Matrix 风险：来自 `matrix_disagreement_flags`

显示形式：

- `HIGH_RISK`：红色标签
- `MATRIX_CONFLICT`：橙色标签
- `TAIL_RISK`：黄色标签
- `LOW_CONFIDENCE`：灰/紫标签

示例：

- 周三006 曼城 vs 水晶宫：`MATRIX_CONFLICT`、`DRAW_RISK`、`AWAY_TAIL`、`HIGH_GOAL_TAIL`
- 周三010 赫塔费 vs 马洛卡：`MATRIX_CONFLICT`、`DRAW_RISK`、`WEAK_HOME_TAIL`

### 12.3 matrix_top_scores 如何折叠展示

建议默认只展示 Top 3：

- `Matrix Top 3：2-1 12.56% / 1-1 11.37% / 1-0 9.24%`

其余折叠到“展开 matrix 候选”中：

- 默认：Top 3
- 展开：Top 10 或 Top 20
- 每一项显示：`score`、`prob`、方向颜色
- 如果 `matrix_recommended_score != predicted_score`，把 matrix top1 加橙色边框，但仍标注“shadow”。

### 12.4 risk_score_candidates 如何和 predicted_score 并列展示

建议在 final score 下方显示“风险比分候选”，但不要覆盖 final：

- 主预测：`Final score：2-1`
- 风险候选：`Risk scores：1-1、1-2、2-2、2-3`

每个候选应显示 `risk_type` tooltip：

- `1-1`：draw_live / low_score_btts
- `1-2`：away_fightback / high_btts_tail
- `2-2`：high_btts_tail
- `2-3`：weak_home_favorite_btts_tail

视觉层级：

- final score 用主色、最大字号
- risk score 用小 chip，不要使用和 final 相同的主色
- 如果 risk candidate 与 matrix top score 重合，可加小标记 `also in matrix`

### 12.5 confidence_downgrade_reason 如何提示

建议不要把长文本直接铺在卡片中。推荐：

- 在 confidence 旁显示一个 info icon：`confidence 60 ⓘ`
- hover/click 展示 tooltip 或 drawer：
  - `下调原因：主胜概率不足52%且客胜概率大于23%，1-1赔率全场最低，存在较强的平局与客队反打尾部风险。`
- 如果 `confidence_downgrade_reason` 非空，confidence 区域加虚线下划线或小警示图标。
- 如果是 PR #2 自动保护触发，可额外显示：`Protocol tail protection applied`。

### 12.6 matrix_disagreement_flags 如何提示

不要直接展示原始 boolean。建议映射为中文解释：

- `matrix_vs_final_direction_conflict`：`Matrix 与最终方向冲突`
- `matrix_high_goal_tail_conflict`：`Matrix 高进球尾部与 final 小球冲突`
- `matrix_away_tail_warning`：`Matrix 客队尾部风险偏高`
- `matrix_draw_risk_warning`：`Matrix 平局风险偏高`
- `matrix_low_confidence_warning`：`Matrix 低置信/头部不集中`

显示规则：

- 只有 true 的 flag 展示。
- 若无 true flag，显示 `Matrix 未发现显著分歧` 即可，不必展示空对象。
- 若 `matrix_shadow_error` 非空，展示 `Matrix unavailable`，并隐藏 matrix top scores，避免空数据误导。

### 12.7 不要让用户误以为 matrix 是最终预测

必须在 UI 文案中固定加说明：

- `Matrix shadow is diagnostic only.`
- 中文：`矩阵为旁路诊断，不接管最终预测。`
- final 区域命名为 `Final / AI-native`
- matrix 区域命名为 `Matrix shadow / 诊断`
- 不要用 `推荐比分` 这个词描述 `matrix_recommended_score`；建议改为 `matrix_top1_score` 或前端显示为 `Matrix Top1`。

## 13. 是否建议开 PR #4

建议开 PR #4。

理由：后端字段已经完整生效，但如果前端没有明确展示新增风险字段，用户仍会看到一个过度简化的“比分 + 方向 + 信心”卡片，无法理解为什么多数场次被降级为 C，也无法看到 matrix shadow 对周三004、周三006、周三010 的强冲突提示。

PR #4 应是前端展示层 PR，不应改预测逻辑。

## 14. PR #4 范围建议

建议 PR #4 范围限定为 UI / presentation / copy，不动后端预测逻辑：

1. 在比赛卡片顶部新增 Final vs Matrix shadow 双层展示。
2. 新增风险标签组件：`HIGH_RISK`、`MATRIX_CONFLICT`、`TAIL_RISK`、`LOW_CONFIDENCE`。
3. 新增 `risk_score_candidates` 展示区，与 `predicted_score` 并列但弱化。
4. 新增 `confidence_downgrade_reason` tooltip/drawer。
5. 新增 `matrix_top_scores` 折叠展示，默认 Top 3，展开 Top 10/20。
6. 新增 `matrix_disagreement_flags` 中文映射与提示。
7. 当 `matrix_shadow_error` 非空时，显示 matrix 计算失败状态。
8. 推荐页/top4 页增加说明：`本轮仅展示通过推荐 gate 的场次；matrix 不参与推荐 gate。`
9. 增加前端单元测试或快照测试：
   - matrix conflict 场次
   - matrix 无冲突场次
   - risk candidates 多条场次
   - confidence downgrade 非空场次
   - matrix_shadow_error 非空 fallback 场次

## 15. 不建议修改的部分

不建议在 PR #4 中修改：

1. 不改 `scripts/predict.py` 的预测、归一化、风控、matrix shadow 逻辑。
2. 不改 `final_direction`、`predicted_score`、`confidence` 的生成规则。
3. 不让 matrix 接管 final decision。
4. 不把 `matrix_recommended_score` 用作最终推荐比分。
5. 不调整 PR #2 的弱主胜尾部保护阈值。
6. 不调整 recommendation gate / Top4 排序逻辑。
7. 不重新跑预测，不调用 AI API 重新预测。
8. 不改 GitHub Actions secrets / token / provider 配置。
9. 不删除 `matrix_*` 字段，即使前端暂时只展示部分字段。
10. 不把英文 flag 直接暴露给普通用户，应在前端做中文映射。

## 附：逐场摘要

| # | 场次 | Final | Matrix shadow | 风险 | active matrix flags |
|---|---|---|---|---|---|
| 1 | 周三003 比利亚雷 vs 塞维利亚 | 主胜 2-1 / C / conf 60 | 2-1 主胜 | medium | 无 |
| 2 | 周三004 西班牙人 vs 毕尔巴鄂 | 客胜 0-1 / C / conf 55 | 1-1 平局 | high | direction_conflict, draw_risk |
| 3 | 周三005 布雷斯特 vs 斯特拉斯 | 平局 1-1 / C / conf 45 | 1-1 平局 | high | away_tail, low_confidence |
| 4 | 周三006 曼城 vs 水晶宫 | 主胜 3-0 / C / conf 75 | 2-2 平局 | low | direction_conflict, away_tail, draw_risk |
| 5 | 周三007 拉齐奥 vs 国际米兰 | 客胜 0-2 / C / conf 65 | 1-2 客胜 | medium | 无 |
| 6 | 周三008 朗斯 vs 巴黎圣曼 | 客胜 1-2 / C / conf 60 | 1-2 客胜 | high | 无 |
| 7 | 周三009 阿拉维斯 vs 巴萨 | 客胜 1-2 / C / conf 62 | 1-2 客胜 | high | 无 |
| 8 | 周三010 赫塔费 vs 马洛卡 | 主胜 1-0 / C / conf 66 | 1-1 平局 | medium | direction_conflict, away_tail, draw_risk |
| 9 | 周三011 辛辛那提 vs 迈国际 | 客胜 1-2 / C / conf 60 | 1-2 客胜 | high | 无 |
| 10 | 周三012 西雅图 vs 圣何塞 | 主胜 2-1 / C / conf 65 | 2-1 主胜 | medium | 无 |
| 11 | 周四001 巴伦西亚 vs 巴列卡诺 | 平局 1-1 / C / conf 60 | 1-1 平局 | medium | 无 |
| 12 | 周四002 达曼协定 vs 吉达联合 | 客胜 1-2 / C / conf 65 | 1-2 客胜 | medium | 无 |
| 13 | 周四003 胡巴卡德 vs 拉斯决心 | 主胜 3-0 / C / conf 75 | 2-0 主胜 | medium | 无 |
| 14 | 周四004 赫罗纳 vs 皇家社会 | 主胜 2-1 / C / conf 55 | 2-1 主胜 | high | 无 |
| 15 | 周四005 皇马 vs 奥维耶多 | 主胜 3-0 / B / conf 75 | 2-0 主胜 | medium | 无 |
