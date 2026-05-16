# v18.1 legacy 与当前 main 静态比较分析

分析对象：

- `legacy/v18_1/predict_v18_1.py`
- `scripts/predict.py`
- `reports/v18_1_matrix_research/static_scan_summary.txt`

约束：本次只做静态比较；未修改主代码，未运行预测，未 commit，未 push。

说明：`static_scan_summary.txt` 当前为空文件，因此本报告以两个 Python 源文件的静态扫描和人工阅读为依据。

---

## 1. 执行摘要

结论：建议进入 Step 3 shadow runner，但不建议回滚 v18.1，也不建议替换 `scripts/predict.py`。

v18.1 的核心价值不是某个 prompt 或某个 AI 模型，而是一条本地可解释的概率骨架：

1. 从 1X2 赔率恢复公平方向概率；
2. 从 TTG a0-a7 恢复总进球分布；
3. 从 CRS 正确比分赔率恢复比分几何形状；
4. 以 Poisson/CRS 融合生成统一比分矩阵 `P(H=h,A=a)`；
5. 用 IPF 同时贴合 1X2 与 TTG 约束；
6. 将 T1-T16 陷阱信号作为 residual 作用到矩阵；
7. 将 AI 作为有限幅 residual，而不是让 AI 单独裁决；
8. 从同一个矩阵派生方向、比分、总球、EV。

当前 main，即 `scripts/predict.py`，已经是更现代的 AI-native / Web-aware / score-cluster / sharp-facts 架构。它强调本地只编译 evidence，不做足球裁决，并把最终方向和比分交给 Gemini final referee。这个设计降低了本地规则硬判的风险，也增强了 prompt 协议和来源审计；但它缺少 v18.1 那种可独立导出的本地概率矩阵。因此 main 当前更像“证据编译器 + AI 裁判”，而 v18.1 更像“赔率量化矩阵 + AI residual”。

PR #3 最佳范围应是 shadow matrix layer：旁路生成并输出矩阵派生字段，供审计、回测、前端对比、AI prompt 引用或推荐降级使用，但不夺取 current main 的最终裁判权。

建议 PR #3 输出字段：

- `matrix_direction_probs`
- `matrix_top_scores`
- `matrix_goal_probs`
- `matrix_lambda_home`
- `matrix_lambda_away`
- `matrix_shape_verdict`
- `matrix_disagreement_flags`
- `matrix_recommended_score`
- `matrix_recommended_direction`

---

## 2. v18.1 核心优势

### 2.1 `fair_probs_from_1x2`

位置：`legacy/v18_1/predict_v18_1.py:448-546`

作用：从主/平/客欧赔或竞彩 HAD 赔率反推公平概率。

结构特点：

- 支持 `multiplicative` 普通归一化；
- 支持 `additive` 平均扣水；
- 默认使用 `power` 幂校准去水；
- 提供名为 `shin` 的近似路径，失败时 fallback 到 `power`；
- 输出包括：
  - `fair_probs`
  - `raw_implied`
  - `overround`
  - `shin_z`

优势：方向概率有可解释市场先验，而不是完全依赖 AI 输出的 `direction_probs`。

风险：`shin` 命名不严谨。代码中的 `detect_all_traps` 还把 `fair_1x2` 兼容命名为 `shin`，实际已不是严格 Shin 模型。这一点不能原样迁移为“Shin 概率”。

### 2.2 `fair_probs_from_ttg`

位置：`legacy/v18_1/predict_v18_1.py:549-577`

作用：从 a0-a7 总进球赔率恢复 0-7+ 总进球公平分布。

结构特点：

- 逐项读取 `a0` 到 `a7`；
- 使用赔率倒数作为 raw implied；
- 默认使用 power 去水；
- 返回 0~1 概率分布。

优势：总进球不是 prompt 里的文字判断，而是一个可被矩阵强制贴合的边际分布。

### 2.3 `crs_implied_probabilities`

位置：`legacy/v18_1/predict_v18_1.py:1580-1625`

作用：从 CRS 正确比分赔率恢复比分概率。

结构特点：

- 读取常规 CRS 比分表 `CRS_FULL_MAP`；
- 读取 `crs_win`、`crs_same`、`crs_lose` 三类“其他比分”；
- 将“其他比分”的 implied probability 均分给预设 score set；
- 输出：
  - `probs`
  - `margin`
  - `coverage`

优势：CRS 信息进入可计算矩阵，而不是只作为 prompt 证据。

风险：其他比分均分会扭曲尾部。真实尾部通常不是均匀分布，例如 4-3、5-1、6-0 的概率不应相同。PR #3 如果迁移，必须把 other allocation 标注为近似，或改为 Poisson-tail / geometric-tail 分配。

### 2.4 `analyze_crs_matrix`

位置：`legacy/v18_1/predict_v18_1.py:1762-1793`

作用：对 CRS implied probabilities 做矩阵形状分析。

结构特点：

- 调用 `compute_statistical_moments` 得到：
  - `lambda_h`
  - `lambda_a`
  - `lambda_total`
  - 方差、协方差、相关、偏度；
- 调用 `classify_shape` 输出：
  - `shootout`
  - `grinder`
  - `lopsided_h`
  - `lopsided_a`
  - `balanced`
  - `normal`
- 聚合 CRS 派生方向概率；
- 输出 CRS top scores。

优势：可以检测“比分赔率形状”，例如低比分磨局、互射局、单边碾压、均势等。这种几何信息对比分选择比单一最低赔率更稳定。

### 2.5 `build_unified_score_matrix`

位置：`legacy/v18_1/predict_v18_1.py:1874-1997`

这是 v18.1 的核心。

结构链路：

1. 读取 1X2 赔率并生成 `target_dir`；
2. 读取 TTG 并生成总进球边际分布；
3. 估计基础 `lambda_h/lambda_a`；
4. 构建 Poisson base grid：`P(h,a)=Pois(lambda_h,h)*Pois(lambda_a,a)`；
5. 如果 CRS 覆盖足够，把 CRS grid 与 Poisson grid 做几何融合；
6. 进行 12 轮 IPF：
   - 先贴合 1X2 方向边际；
   - 再贴合 TTG 总进球边际；
7. 应用 trap residual；
8. 从矩阵导出：
   - `direction_probs`
   - `goal_probs`
   - `top_scores`
   - `lambda_h`
   - `lambda_a`

优势：方向、总球、比分来自同一个 `P(H=h,A=a)`，天然减少“方向说主胜、比分写 1-1、总球说 4+”这类结构性冲突。

### 2.6 `apply_trap_residual_to_matrix`

位置：`legacy/v18_1/predict_v18_1.py:1840-1871`

作用：把陷阱检测结果作为 residual 作用在矩阵上。

结构特点：

- `direction_adjust` 按比分方向乘上指数因子；
- 单个方向修正被限制在 `exp(-0.65)` 到 `exp(0.65)`；
- `score_multipliers` 被限制在 `0.15` 到 `2.2`；
- `boost_scores` 固定乘 `1.28`；
- 最后重新归一化。

优势：陷阱规则不会直接覆盖最终结果，而是以有限幅方式改变概率分布。这比“触发 Txx 就改方向”更稳健。

### 2.7 `decision_lock_chain`

位置：`legacy/v18_1/predict_v18_1.py:2199-2399`

作用：用统一矩阵锁定方向、比分和证据。

结构特点：

- 解析 AI top score 与 top3；
- 生成 AI 方向票和比分票；
- 构建 unified score matrix；
- AI 只做 residual：
  - 对方向 logits 的 delta 限制在 `[-0.18, +0.18]`；
  - 回灌矩阵方向 ratio 限制在 `[0.65, 1.55]`；
- 从矩阵 posterior 选择最终方向；
- 根据 `determine_goal_range` 选择总球区间；
- 调用 `select_score_from_matrix` 选择比分；
- 保证比分方向与最终方向一致。

优势：AI 不再是独立重复计数证据，而是有限幅 residual。这个结构可降低多模型相关性导致的“伪共识”。

### 2.8 `select_score_from_matrix`

位置：`legacy/v18_1/predict_v18_1.py:2012-2057`

作用：在最终方向和总进球范围内，从矩阵选出最优比分。

结构特点：

- 只筛选与方向一致的比分；
- 只筛选落在 goal range 内的比分；
- AI top3 票只作为小幅加权，最高加成被限制；
- 没有候选时使用方向 fallback。

优势：比分选择受同一矩阵、方向和总球约束，不容易出现字段错配。

### 2.9 `determine_goal_range`

位置：`legacy/v18_1/predict_v18_1.py:2118-2192`

作用：根据 CRS moments、TTG 高球赔率、让球、xG、联赛 profile 决定候选比分总球区间。

优势：比分不是全空间 argmax，而是在合理进球带内选择。

风险：包含 `LEAGUE_LOW_GOALS` / `LEAGUE_HIGH_GOALS` 硬偏置，只用字符串命中对 `lt_avg` 加减 0.2。这可以作为 shadow diagnostic，但不能直接当强规则迁移。

### 2.10 `merge_result` / `run_predictions`

位置：

- `merge_result`: `legacy/v18_1/predict_v18_1.py:3184-3554`
- `_enforce_consistency`: `legacy/v18_1/predict_v18_1.py:3636-3681`
- `run_predictions`: `legacy/v18_1/predict_v18_1.py:3688-3896`

结构特点：

- `merge_result` 汇总 engine、AI、stats、match；
- 先计算 exp_goals，再检测 traps，再分析 CRS，再进入 decision lock chain；
- 计算比分市场赔率与模型概率之间的 EV：
  - `score_model_prob`
  - `score_market_odds`
  - `score_market_implied_pct`
  - `edge_vs_market`
  - `suggested_kelly`
- 输出大量矩阵和市场解释字段；
- `_enforce_consistency` 只做字段闭环：根据 `predicted_score` 修 `result/display_direction/final_direction/predicted_label`，没有生成或伪造概率。

优势：EV 是“模型概率 vs 市场赔率”的概念，而不是“AI 置信度 vs 赔率”。这是 v18.1 值得保留的重要量化结构。

---

## 3. 当前 main 缺失结构

当前 main：`scripts/predict.py`，静态扫描显示其核心是 v20.3 AI-native evidence compiler。关键结构如下：

- `run_predictions`: `scripts/predict.py:2410-2454`
- v20.3 增强层起点：`scripts/predict.py:2458`
- `compile_1x2_facts`: `scripts/predict.py:2652-2666`
- `compile_ttg_facts`: `scripts/predict.py:2709-2731`
- `compile_crs_rows`: `scripts/predict.py:2744-2769`
- `compile_score_cluster_diagnostics`: `scripts/predict.py:2799-2850`
- `compile_sharp_money_facts`: `scripts/predict.py:2870-2941`
- `build_evidence_packet_v203`: `scripts/predict.py:3032-3091`
- 当前覆盖版 `build_evidence_packet`: `scripts/predict.py:3192-3209`
- Gemini final schema / prompt：`scripts/predict.py:3212-3420`
- consistency judge prompt：`scripts/predict.py:3438-3451`

### 3.1 是否缺少统一比分矩阵 `P(H=h,A=a)`

判断：缺少。

main 有 CRS rows、score clusters、adjacent audit table，但没有构建全比分网格概率矩阵，也没有返回每个 `(h,a)` 的本地概率。`compile_score_cluster_diagnostics` 是赔率簇事实表，不是概率矩阵。

### 3.2 是否缺少 IPF 贴合 1X2 与 TTG

判断：缺少。

main 的 `compile_1x2_facts` 使用 `_devig_3way` 做简单归一化；`compile_ttg_facts` 做总进球赔率事实和 band pressure。它们只进入 evidence，不被统一矩阵同时贴合。没有 v18.1 `build_unified_score_matrix` 中 12 轮 IPF 那样的约束校准。

### 3.3 是否缺少 CRS 几何矩阵

判断：部分缺少。

main 有 CRS rows 与 score-cluster diagnostics，包括：

- lowest scores overall；
- lowest scores by direction；
- cluster ranking；
- adjacent score audit table；
- CRS movement summary。

但这些是事实表和 prompt 审计数据，不是完整 CRS implied probability matrix，也没有 `lambda_h/lambda_a/variance/corr/skew/shape_verdict` 那类矩阵矩统计。

### 3.4 是否缺少 trap residual 层

判断：缺少 v18.1 形式的 residual 层。

main 有 `sharp_money_facts_v203`、tail risk protection、recommendation downgrade，但它明确“不改足球方向/比分，只做 Evidence 编译、协议校验、推荐风险展示”。没有把 T1-T16 或 sharp/cluster 信号转成矩阵乘子。

### 3.5 是否缺少 AI residual 限幅

判断：缺少本地概率意义上的 AI residual 限幅。

main 通过 schema、critic、Gemini final、consistency judge 管理 AI 输出；但最终方向/比分仍是 AI 裁判输出。没有 v18.1 中 `delta clamp [-0.18,+0.18]`、方向 ratio clamp `[0.65,1.55]` 的本地数学限幅。

### 3.6 是否缺少 `matrix_direction_probs`

判断：缺少。

main schema 有 AI 输出的 `direction_probs`，但没有本地 shadow matrix 派生的 `matrix_direction_probs`。

### 3.7 是否缺少 `matrix_top_scores`

判断：缺少。

main 有 CRS `lowest_scores_overall`、AI `top3`、score_cluster audit，但没有本地概率矩阵 top scores。

### 3.8 是否缺少 `matrix_goal_probs`

判断：缺少。

main 有 TTG facts 与 `goal_band` schema，但没有从统一矩阵导出的总进球概率分布。

### 3.9 是否缺少 EV 模型概率 vs 市场赔率

判断：基本缺少。

v18.1 在 `merge_result` 中用 `model_prob_pct` 与 `final_odds` 计算 independent EV / Kelly。当前 main 更强调推荐等级、组件分、风险标签；没有稳定的本地 `score_model_prob` 来源，因此不宜直接计算真正 EV。AI 的 `bet_confidence` 不应替代模型概率。

---

## 4. 可迁移模块 Top10

### P0-1：统一比分矩阵 summary，不迁移最终裁决权

迁移 `build_unified_score_matrix` 的思想，但以 shadow 方式输出，不改最终预测。

输出：

- `matrix_direction_probs`
- `matrix_goal_probs`
- `matrix_top_scores`
- `matrix_lambda_home`
- `matrix_lambda_away`

理由：这是 v18.1 最核心的量化骨架。

### P0-2：1X2 公平概率恢复

迁移 `fair_probs_from_1x2` 的 power / multiplicative 去水能力。

注意：不要把 power fair 叫 Shin；字段应命名为 `fair_1x2_power`、`fair_1x2_method`。

### P0-3：TTG 公平总球分布

迁移 `fair_probs_from_ttg`，用于 shadow matrix 的总球边际约束。

输出可包括：

- `matrix_goal_probs`
- `ttg_fair_probs`
- `ttg_available`

### P0-4：IPF 贴合 1X2 + TTG

迁移 v18.1 中同时贴合方向边际和总球边际的约束思路。

建议：先作为 pure function，输入 odds facts，输出 shadow summary，不进入 main 裁决。

### P0-5：CRS moments 与 shape verdict

迁移 `compute_statistical_moments` + `classify_shape`，但修正 CRS other 分配。

输出：

- `matrix_shape_verdict`
- `matrix_lambda_home`
- `matrix_lambda_away`
- `matrix_shape_anomalies`

### P1-6：CRS/Poisson 几何融合

迁移 Poisson base + CRS geometric blend，但要给 CRS coverage 与 data quality 加硬门槛。

### P1-7：AI residual 限幅思想

不要让 shadow layer 覆盖 AI；只输出 disagreement flags：

- AI 方向 vs matrix 方向冲突；
- AI top score 不在 matrix top N；
- AI goal band 与 matrix goal mode 冲突。

### P1-8：trap residual 的有限幅乘子

可以把 current main 的 sharp/cluster/tail facts 转成 shadow residual，但第一版建议只做诊断，不直接调矩阵。第二阶段再引入可配置乘子。

### P1-9：EV 所需字段

迁移 EV 概念，但前提是 `matrix_recommended_score` 有本地概率。

输出可先为：

- `matrix_score_model_prob`
- `matrix_score_market_odds`
- `matrix_score_market_implied_pct`
- `matrix_edge_vs_market`

第一版可以不展示 Kelly 或只标注 experimental。

### P2-10：goal range 作为解释层

迁移 `determine_goal_range` 的“总球带约束”概念，但不要直接迁移联赛硬偏置。

---

## 5. 不可直接迁移模块

### 5.1 Shin 近似命名不严谨

v18.1 `fair_probs_from_1x2(method="shin")` 有 Shin-like 近似；但实际主流程默认 `power`。`detect_all_traps` 又把 `fair_1x2` 兼容字段命名为 `shin`。这会导致报告和前端误以为是严格 Shin probability。

建议：

- PR #3 禁用或重命名 `shin` 字段；
- 输出 `fair_1x2_method=power`；
- 如保留 Shin-like 算法，命名为 `shin_like_experimental`。

### 5.2 CRS 其他比分均分会扭曲尾部

v18.1 把 `crs_win/crs_same/crs_lose` 均分到其他比分集合。这个做法简单，但对尾部不合理。

风险：

- 高比分尾部概率被抬得过平；
- 4-3、5-2、6-0 等概率可能被错误接近；
- 可能误导 `lambda_total`、skew、shape verdict。

建议：

- PR #3 第一版可不展开 others，只记录 other mass；
- 或用 Poisson tail 按 base probability 分配；
- 所有 other-derived 概率标注 `approx_tail_allocation=true`。

### 5.3 T1-T16 仍有重复计数风险

v18.1 有互斥规则：T14 排除 T1，T2/T3 二选一，T13 排除 T6，T4/T5 与 T8 二选一。但 T1-T16 本质仍可能基于同一赔率结构重复触发，例如：

- 1X2 fair gap；
- handicap depth；
- CRS/TTG 低赔；
- public heat；
- sharp/steam 文本信号。

这些信号不一定独立，直接叠加 severity 和 direction_adjust 仍会重复计数。

建议：PR #3 不直接迁移 T1-T16 加权。第一版只输出 `matrix_disagreement_flags`，不把 trap 当强残差。

### 5.4 `determine_goal_range` 存在联赛硬偏置

v18.1 对 `LEAGUE_LOW_GOALS` 和 `LEAGUE_HIGH_GOALS` 字符串命中后直接调整 `lt_avg`。这会把历史 league stereotype 写进单场预测。

建议：

- 只作为解释型 profile；
- 不得直接改 final matrix；
- 若使用，必须可配置、可回测、可关闭。

### 5.5 AI 调用模型名可能过期

v18.1 静态模型名包括：

- `熊猫-A-5-grok-4.2-fast-200w上下文`
- `gpt-5.5`
- `熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking`
- `熊猫-69-满血openrouter-claude-opus-4.7-上下文1000k`

这些依赖特定网关和历史命名，不应迁移。PR #3 不应触碰 AI 调用链。

### 5.6 大 prompt 成本问题

v18.1 的 `build_v18_prompt` 把大量规则、市场数据、AI 初审结构写入单个大 prompt。当前 main 也已有大 schema 和多阶段 prompt。继续把 v18.1 prompt 合并进 main 会增加：

- token 成本；
- 注意力稀释；
- JSON 失真概率；
- 多模型调用失败面。

建议：PR #3 迁移本地 shadow summary，而不是迁移 v18.1 大 prompt。

### 5.7 `_enforce_consistency` 只做字段闭环，不伪造概率

v18.1 `_enforce_consistency` 根据 `predicted_score` 修正 `result/display_direction/final_direction/predicted_label`。它没有重算概率，也没有伪造 `home_win_pct/draw_pct/away_win_pct`。

迁移建议：

- 保留 current main consistency judge “只修字段，不做足球判断”的原则；
- 不要用 consistency repair 生成新概率；
- shadow matrix 字段应独立计算，不能由 AI 输出反推。

---

## 6. 与 PR #1 / PR #2 的关系

基于当前 reports 和 main 静态结构判断：

- PR #1 更偏工程加固：AI JSON 解析、parse_failed、协议字段、tail risk schema 等；
- PR #2 更偏 evidence / sharp / score-cluster 增强：v20.3 引入 `market_microstructure_v203`、`score_cluster_diagnostics_v203`、`sharp_money_facts_v203`、相邻比分审计、recommendation components；
- PR #1 / PR #2 没有恢复 v18.1 的本地概率骨架。

因此 PR #3 不应重做 PR #1 / PR #2 的 prompt 和 evidence 工作，而应补上“shadow matrix probability layer”。

推荐定位：

- PR #1：协议与工程可靠性；
- PR #2：市场证据编译与 AI 审计；
- PR #3：v18.1 矩阵旁路，只输出本地概率摘要，不夺权。

---

## 7. 对方向命中率低的解释

从量化结构角度看，当前 main 方向命中率低可能来自以下原因：

1. 缺少统一概率先验。
   当前 final_direction 主要来自 AI referee。即使 evidence 很丰富，AI 也可能在方向、比分、总球之间做非概率化折中。v18.1 则先有 1X2 fair + TTG + CRS 约束出的方向 posterior。

2. 多模型意见不是独立样本。
   GPT/Grok/Gemini 读取同一 evidence 和相似 prompt 后产生的共识高度相关。把它当多数投票会高估置信度。v18.1 将 AI 作为 residual，并限制影响幅度。

3. CRS/TTG 只作为文本事实时，无法强制一致。
   main 的 score-cluster 和 TTG facts 很强，但它们进入 prompt 后是否被正确使用取决于 AI。v18.1 的 IPF 会强制矩阵边际贴合这些市场分布。

4. 缺少本地 disagreement gate。
   如果 AI 选 2-1，但矩阵方向概率偏低、总球 mode 不支持、CRS top score 不支持，当前 main 主要靠 prompt 让 AI 自审。shadow matrix 可以显式输出冲突 flags。

5. 推荐等级与模型概率混用风险。
   main 的 `bet_confidence` 和 recommendation components 是 AI 评分，不等于校准后的市场概率。方向命中率低时，缺少可校准概率会让推荐门控难以稳定。

因此，v18.1 历史表现可能更好，不是因为“玄学规则”更准，而是因为它用同一个概率矩阵消除了多源证据之间的自由漂移，并限制了 AI residual 的过度影响。

---

## 8. PR #3 shadow matrix layer 设计建议

### 8.1 设计原则

PR #3 不回滚 v18.1，不替换 `scripts/predict.py`，不改 current main 的 AI-native 主流程。

只新增旁路 shadow layer：

- 不覆盖 `predicted_score`；
- 不覆盖 `final_direction`；
- 不修改 Gemini final referee 输出；
- 不改推荐 tier 的第一版逻辑；
- 只在 prediction/evidence 中附加矩阵摘要和冲突标记。

### 8.2 推荐输出字段

字段建议：

```json
{
  "matrix_direction_probs": {"home": 0.0, "draw": 0.0, "away": 0.0},
  "matrix_top_scores": [{"score": "1-1", "prob": 0.0}],
  "matrix_goal_probs": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7+": 0.0},
  "matrix_lambda_home": 0.0,
  "matrix_lambda_away": 0.0,
  "matrix_shape_verdict": "normal/grinder/shootout/balanced/lopsided_h/lopsided_a/unknown",
  "matrix_disagreement_flags": [],
  "matrix_recommended_score": "",
  "matrix_recommended_direction": "home/draw/away/unknown"
}
```

### 8.3 Shadow runner 输入

只使用 raw match odds / facts：

- 1X2：`sp_home/sp_draw/sp_away` 或 `win/same/lose`；
- TTG：`a0-a7`；
- CRS：`w10...l25`，可选 `crs_win/crs_same/crs_lose`；
- 可选 engine xG：如果 current main 已有稳定字段；
- 不读 AI final 作为矩阵输入，避免后验污染。

### 8.4 Shadow runner 输出位置

两种低风险方案：

1. 在 `build_evidence_packet` 中加入 `shadow_matrix_v18_1`，供 AI 读取；
2. 在 `adapt_ai_to_frontend` 后附加 `prediction.shadow_matrix`，供前端和回测读取。

建议第一版选择方案 2：先不影响 prompt，避免改变线上行为。回测稳定后，再考虑把 summary 纳入 prompt。

### 8.5 `matrix_disagreement_flags` 规则

建议第一版只做 flags，不做裁决：

- `ai_direction_differs_from_matrix_top_direction`
- `ai_score_not_in_matrix_top10`
- `ai_goal_band_differs_from_matrix_goal_mode`
- `matrix_low_confidence_direction_gap_lt_5`
- `crs_coverage_low_matrix_poisson_dominant`
- `ttg_missing_goal_probs_less_reliable`
- `other_score_tail_allocation_approx`

### 8.6 迁移顺序

P0：pure functions + unit tests。

- `fair_probs_from_1x2`
- `fair_probs_from_ttg`
- Poisson grid
- IPF fit
- summary extraction

P1：CRS moments / shape verdict。

- CRS implied probs
- moments
- shape verdict
- top scores

P2：residual / EV。

- trap residual diagnostics only；
- EV fields experimental；
- 不直接进入推荐门控。

---

## 9. 风险和测试要求

### 9.1 单元测试

必须覆盖：

1. 1X2 fair probs sum 接近 100%；
2. TTG fair probs sum 接近 1；
3. matrix probabilities sum 接近 1；
4. matrix direction probs sum 接近 100%；
5. matrix goal probs sum 接近 100%；
6. IPF 后方向边际接近 1X2 fair；
7. IPF 后总球边际接近 TTG fair；
8. 0-1 不被误判为无效比分；
9. `predicted_score` 与 `matrix_recommended_score` 字段互不覆盖；
10. 无 CRS 时 fallback 到 Poisson + 1X2/TTG 仍可运行。

### 9.2 回归测试

必须确认：

- `scripts/predict.py` 原有输出字段不减少；
- AI-native 调用链不变；
- Gemini final referee 不被 bypass；
- no-AI / AI failure 路径不因 shadow matrix 崩溃；
- 前端读取旧字段不受影响。

### 9.3 数据质量测试

必须覆盖：

- 缺 1X2；
- 缺 TTG；
- 缺 CRS；
- CRS 覆盖低；
- odds 为 0、空字符串、`N/A`；
- `a7` 代表 7+ tail；
- `crs_win/same/lose` 存在但常规比分不足。

### 9.4 校准与回测要求

不要用单日结果判断是否提高命中率。建议至少记录：

- AI final direction vs matrix direction；
- AI score vs matrix top scores；
- matrix direction top1 hit；
- AI direction top1 hit；
- disagreement cases hit rate；
- matrix goal band hit；
- matrix top3 score hit；
- EV 正负与实际赔率结果关系。

### 9.5 成本和 prompt 风险

第一版不要把完整矩阵塞进 prompt。最多给 compact summary：

- top 5 scores；
- direction probs；
- goal mode；
- shape verdict；
- disagreement flags。

避免把 9x9 或 11x11 全矩阵写入 prompt。

---

## 10. 结论：是否建议进入 Step 3 shadow runner

建议进入 Step 3 shadow runner。

理由：

1. v18.1 的核心优势是可解释的统一比分概率矩阵，而不是旧 AI 模型或旧 prompt；
2. 当前 main 的 evidence/compiler/prompt 架构更现代，但缺少本地概率骨架；
3. 直接回滚 v18.1 会带回旧模型名、命名不严谨、CRS other 均分、T1-T16 重复计数、联赛硬偏置和大 prompt 成本；
4. shadow layer 可以低风险恢复最有价值的量化结构，同时保留 current main 的 AI-native 裁判和 v20.3 evidence compiler。

PR #3 推荐范围：

P0：实现并输出 shadow matrix summary：

- `matrix_direction_probs`
- `matrix_top_scores`
- `matrix_goal_probs`
- `matrix_lambda_home`
- `matrix_lambda_away`
- `matrix_shape_verdict`
- `matrix_disagreement_flags`
- `matrix_recommended_score`
- `matrix_recommended_direction`

P1：加入 CRS moments / shape verdict / coverage quality。

P2：加入实验性 EV 和 residual diagnostics，但不进入最终裁决。

最终判断：可以进入 Step 3 shadow runner；但 Step 3 必须是旁路、可关闭、可回测、零主流程夺权。