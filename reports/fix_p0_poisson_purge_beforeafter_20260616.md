# P0 泊松去污 — 前后对比报告 (v20.7)
日期：2026-06-16 | 分支：fix/p0-poisson-purge-deadcode-20260616 | 基线：origin/main 0c7038e
备份 tag：backup/pre-poisson-purge-20260616102946 → 0c7038e
状态：**已完成改码+全回归，未 commit、未 push、未碰 main（待用户批准）**

## 1. 目标
切除复活的泊松影子矩阵（matrix_shadow），消除"机械数理锚反向污染盘感"，比分/方向回归纯 AI 读盘。参照 6/8 PR#27-30 手法。

## 2. 改动清单（git diff: 6 files, +35 / -504）

### scripts/predict.py（净删 ~290 行）
**A. 删除 10 个泊松 shadow 函数（AST 精确删除，无误伤）：**
- `_normalize_prob_dict_shadow` / `fair_probs_from_1x2_shadow` / `fair_probs_from_ttg_shadow`
- `crs_implied_probabilities_shadow` / `_poisson_pmf_shadow` / `_matrix_moments_shadow`
- `_matrix_shape_verdict_shadow` / `_direction_probs_from_score_probs_shadow`
- `build_unified_score_matrix_shadow` / `attach_matrix_shadow_fields`

**B. 切断 4 处下游消费点：**
| 位置 | 改前 | 改后 |
|---|---|---|
| 主流程 attach 调用 | `attach_matrix_shadow_fields(pred, match_obj)` | 删除（注释标记） |
| 方向概率融合 | `draw/away/home_prob = max(probs, matrix_direction_probs)` | `= probs`（仅 AI 读盘） |
| high_tail | `sum(matrix_goal_probs ≥4球)` | `= 0.0`（行为等价退化，不引新源） |
| 2-1 硬闸 | 4 个 `matrix_flags`/`matrix_draw`/`matrix_away` NO_BET 判定 | 删除（仅留 AI 概率+显式候选尾部驱动） |

**C. 版本号一致性：**
- `v20.6.0_shadow_pre_injected*` → `v20.7.0_poisson_purged_pure_board_reading`（compiler / evidence_compiler_version）

### tests/（净删 ~270 行）— 删除/改写 10 个泊松测试
- **删整文件**：`test_public_football_data_backtest.py`（唯一测试是泊松矩阵回测 smoke）
- **删测试**：test_review_20260512_regressions 4 个（matrix_shadow_fields/shadow_1x2_fair_probs/matrix_goal_probs/matrix_shadow_source）+ _matrix_shadow_fixture/_ai_row_fixture
- **删测试**：test_two_one_hard_gate 2 个（matrix_draw_away_warning / benign_matrix_distribution）+ _benign_matrix_match helper
- **删测试**：test_upset_sandbox 1 个（alaves_barca 用 shadow 函数）
- **改写期望值**：test_prematch_factor_gate（draw_defense：纯 AI draw=29 不触发 gate，由 high_draw_league_cap 拦截）、test_selection_layer（rotation_risk：selection_layer 由纯 AI 信号定为"小注"）— 这 2 个测试的旧期望是泊松污染后的行为，新值反映纯 AI 读盘。

## 3. 验证（实跑）
- ✅ AST 语法 OK；`import predict` OK
- ✅ pytest 全回归：**114 passed**（原 122，-8 为删除的纯泊松测试）
- ✅ 真实 12 场 slate：12/12 跑通无异常，方向/比分/分级正常产出，主决策流程未破坏
- ✅ 残留扫描：predict.py 无泊松活引用（仅剩 `poisson:{}`/`refined_poisson:{}` 空 dict 输出占位 + 去污注释）
- ✅ LEAGUE_DNA_PROFILES / 世界杯 DNA / 版本常量完好无误伤

## 4. 过程坦白（军规如实记录）
1. **首次按行号盲删误伤**：泊松函数与 LEAGUE_DNA_PROFILES 字典在源码交错，按行号删误删了世界杯 DNA 字典 → 立即从备份恢复，改用 AST 按函数名精确删除。
2. **死码清理任务证伪——无死码可清**：审计曾把 14 个"重复函数"定为死码 bug。深查发现这是**有意的"基线+包装层"架构**：第一份是基线实现，经 `_BASE_*_V2021 = fn` 别名保存（predict.py:2982-2995），第二份是增强包装层，内部调 `_BASE_*` 基线（3730/4566/5221）；工具函数在 3700-3709 又显式赋回基线版。删第一份会触发 NameError（已被真实 import 测试当场抓住并恢复）。**结论：14 处全部必需，非死码，不删。** 原审计"P1 重复函数死码"判定纠正为误判。

## 5. 行为变化说明（去污的预期副作用）
泊松删除后，少数边缘 gate 触发细节改变（如某些 2-1 硬闸不再被泊松 matrix_flags 触发、selection_layer 在无泊松抬升下更接近 AI 原始信号）。主决策（方向/比分/tier/gate_pass）在真实 12 场上无破坏。这正是去污目标：让判断只由 AI 读盘驱动，不被机械数理锚反向污染。

## 6. 待用户决策
- [ ] 批准 commit（本分支）
- [ ] 批准 push（仅推新分支 fix/p0-poisson-purge-deadcode-20260616，不碰 main）
- 后续（另行批准）：赛果数据源恢复后做复盘 → 读盘/prompt 升级 + 世界杯轮次感知
